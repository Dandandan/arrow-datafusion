// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! A [`LocalFileSystem`] wrapper that performs IO on a per-core IO thread
//! instead of dispatching to tokio's blocking thread pool.
//!
//! When partition threads are pinned to specific CPU cores, using
//! `spawn_blocking` dispatches IO to a random thread on a potentially
//! different core, losing cache locality. This module provides a dedicated
//! IO thread per core that stays pinned to the same core as the requesting
//! tokio worker, so file reads share the same L2/L3 cache.

use std::fmt::{Debug, Display, Formatter};
use std::fs::{File, Metadata};
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::stream::BoxStream;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{
    Attributes, CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions,
    PutPayload, PutResult, RenameOptions, Result,
};
use tokio::sync::oneshot;

/// Global flag: when set, `runtime_env_builder()` returns a builder with
/// [`SameThreadLocalFileSystem`] instead of the default [`LocalFileSystem`].
static USE_SAME_THREAD_IO: AtomicBool = AtomicBool::new(false);

/// Enable same-thread IO for all subsequently created object store registries.
pub fn enable_same_thread_io() {
    USE_SAME_THREAD_IO.store(true, Ordering::Relaxed);
}

/// Returns true if same-thread IO is enabled.
pub fn is_same_thread_io_enabled() -> bool {
    USE_SAME_THREAD_IO.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Per-core IO thread pool
// ---------------------------------------------------------------------------

type IoWork = Box<dyn FnOnce() + Send>;

/// A pool of IO threads, one per core, each pinned to its respective core.
/// Work is dispatched round-robin based on the calling thread's ID so that
/// a given tokio worker always talks to the same IO thread (the one sharing
/// its core).
struct IoThreadPool {
    senders: Vec<std::sync::mpsc::Sender<IoWork>>,
}

impl IoThreadPool {
    /// Spawn one IO thread per core ID, pinned to that core.
    fn new(core_ids: &[core_affinity::CoreId]) -> Self {
        let mut senders = Vec::with_capacity(core_ids.len());

        for &core_id in core_ids {
            let (tx, rx) = std::sync::mpsc::channel::<IoWork>();
            std::thread::Builder::new()
                .name(format!("datafusion-io-{}", core_id.id))
                .spawn(move || {
                    // Pin this IO thread to the same core
                    core_affinity::set_for_current(core_id);
                    // Process work items until the channel closes
                    while let Ok(work) = rx.recv() {
                        work();
                    }
                })
                .expect("failed to spawn IO thread");

            senders.push(tx);
        }

        Self { senders }
    }

    /// Submit work to the IO thread for the current core.
    /// Uses thread ID to consistently map a tokio worker to the same IO thread.
    fn submit<F, T>(&self, f: F) -> oneshot::Receiver<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        // Use the current thread's ID to pick an IO thread.
        // This gives a stable mapping: the same tokio worker always
        // goes to the same IO thread (sharing its pinned core).
        let thread_id = thread_id_hash();
        let idx = thread_id % self.senders.len();

        let work: IoWork = Box::new(move || {
            let result = f();
            let _ = tx.send(result);
        });

        // If the IO thread has shut down, this will fail — but that
        // only happens during process exit.
        let _ = self.senders[idx].send(work);
        rx
    }
}

/// Fast, stable hash of the current thread's ID.
fn thread_id_hash() -> usize {
    // ThreadId doesn't expose the underlying integer, but its Debug
    // output is stable within a process. Using as_u64 via transmute
    // is not stable. Instead use a thread-local counter assigned at
    // first access, which is cheaper than hashing.
    thread_local! {
        static IDX: usize = next_thread_index();
    }
    IDX.with(|idx| *idx)
}

fn next_thread_index() -> usize {
    use std::sync::atomic::AtomicUsize;
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Global IO thread pool, initialized once.
static IO_POOL: std::sync::OnceLock<IoThreadPool> = std::sync::OnceLock::new();

/// Initialize the global IO thread pool with one thread per core.
/// Must be called before any `SameThreadLocalFileSystem` operations.
/// The provided core IDs should match those used for the tokio workers.
pub fn init_io_thread_pool(core_ids: &[core_affinity::CoreId]) {
    IO_POOL
        .set(IoThreadPool::new(core_ids))
        .ok()
        .expect("IO thread pool already initialized");
}

fn io_pool() -> &'static IoThreadPool {
    IO_POOL
        .get()
        .expect("IO thread pool not initialized — call init_io_thread_pool first")
}

// ---------------------------------------------------------------------------
// ObjectStore implementation
// ---------------------------------------------------------------------------

/// A [`LocalFileSystem`] wrapper that dispatches read IO to a dedicated
/// per-core IO thread, rather than tokio's blocking thread pool.
///
/// Each tokio worker (pinned to core N) sends IO work to IO thread N
/// (also pinned to core N), preserving L2/L3 cache locality while
/// keeping the tokio worker free to run other async tasks.
#[derive(Debug)]
pub struct SameThreadLocalFileSystem {
    inner: LocalFileSystem,
}

impl SameThreadLocalFileSystem {
    pub fn new() -> Self {
        Self {
            inner: LocalFileSystem::new(),
        }
    }
}

impl Display for SameThreadLocalFileSystem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "SameThreadLocalFileSystem({})", self.inner)
    }
}

// ---------------------------------------------------------------------------
// Sync file helpers (replicated from object_store::local since they're
// pub(crate) there)
// ---------------------------------------------------------------------------

fn open_file(path: &PathBuf) -> Result<(File, Metadata)> {
    match File::open(path).and_then(|f| Ok((f.metadata()?, f))) {
        Err(e) => Err(match e.kind() {
            ErrorKind::NotFound => object_store::Error::NotFound {
                path: path.to_string_lossy().to_string(),
                source: e.into(),
            },
            _ => object_store::Error::Generic {
                store: "SameThreadLocalFileSystem",
                source: Box::new(e),
            },
        }),
        Ok((metadata, file)) => {
            if metadata.is_dir() {
                Err(object_store::Error::NotFound {
                    path: path.to_string_lossy().to_string(),
                    source: std::io::Error::new(ErrorKind::NotFound, "is directory")
                        .into(),
                })
            } else {
                Ok((file, metadata))
            }
        }
    }
}

fn convert_metadata(metadata: &Metadata, location: Path) -> ObjectMeta {
    let last_modified: DateTime<Utc> = metadata
        .modified()
        .expect("Modified file time should be supported on this platform")
        .into();

    #[cfg(unix)]
    let inode = std::os::unix::fs::MetadataExt::ino(metadata);
    #[cfg(not(unix))]
    let inode = 0u64;

    let size = metadata.len();
    let mtime = metadata
        .modified()
        .ok()
        .and_then(|mtime| mtime.duration_since(SystemTime::UNIX_EPOCH).ok())
        .unwrap_or_default()
        .as_micros();

    let e_tag = format!("{inode:x}-{mtime:x}-{size:x}");

    ObjectMeta {
        location,
        last_modified,
        size,
        e_tag: Some(e_tag),
        version: None,
    }
}

fn read_range(file: &mut File, path: &PathBuf, range: Range<u64>) -> Result<Bytes> {
    let file_len = file
        .metadata()
        .map_err(|e| object_store::Error::Generic {
            store: "SameThreadLocalFileSystem",
            source: Box::new(e),
        })?
        .len();

    if range.start >= file_len {
        return Err(object_store::Error::Generic {
            store: "SameThreadLocalFileSystem",
            source: format!(
                "Range start {} exceeds file length {} for {}",
                range.start,
                file_len,
                path.display()
            )
            .into(),
        });
    }

    let to_read = range.end.min(file_len) - range.start;

    file.seek(SeekFrom::Start(range.start))
        .map_err(|e| object_store::Error::Generic {
            store: "SameThreadLocalFileSystem",
            source: Box::new(e),
        })?;

    let mut buf = Vec::with_capacity(to_read as usize);
    let read = file
        .take(to_read)
        .read_to_end(&mut buf)
        .map_err(|e| object_store::Error::Generic {
            store: "SameThreadLocalFileSystem",
            source: Box::new(e),
        })? as u64;

    if read != to_read {
        return Err(object_store::Error::Generic {
            store: "SameThreadLocalFileSystem",
            source: format!(
                "Out of range for {}: expected {} bytes, got {}",
                path.display(),
                to_read,
                read
            )
            .into(),
        });
    }

    Ok(buf.into())
}

// ---------------------------------------------------------------------------
// ObjectStore trait
// ---------------------------------------------------------------------------

#[async_trait]
impl ObjectStore for SameThreadLocalFileSystem {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> Result<GetResult> {
        let location = location.clone();
        let path = self.inner.path_to_filesystem(&location)?;

        let rx = io_pool().submit(move || {
            let (file, metadata) = open_file(&path)?;
            let meta = convert_metadata(&metadata, location);
            options.check_preconditions(&meta)?;

            let range = match options.range {
                Some(r) => r.as_range(meta.size).map_err(|e| {
                    object_store::Error::Generic {
                        store: "SameThreadLocalFileSystem",
                        source: Box::new(e),
                    }
                })?,
                None => 0..meta.size,
            };

            Ok(GetResult {
                payload: GetResultPayload::File(file, path),
                attributes: Attributes::default(),
                range,
                meta,
            })
        });

        rx.await.map_err(|_| object_store::Error::Generic {
            store: "SameThreadLocalFileSystem",
            source: "IO thread shut down".into(),
        })?
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> Result<Vec<Bytes>> {
        let path = self.inner.path_to_filesystem(location)?;
        let ranges = ranges.to_vec();

        let rx = io_pool().submit(move || {
            let (mut file, _) = open_file(&path)?;
            ranges
                .into_iter()
                .map(|r| read_range(&mut file, &path, r))
                .collect()
        });

        rx.await.map_err(|_| object_store::Error::Generic {
            store: "SameThreadLocalFileSystem",
            source: "IO thread shut down".into(),
        })?
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, Result<Path>>,
    ) -> BoxStream<'static, Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&Path>,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, Result<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&Path>,
    ) -> Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> Result<()> {
        self.inner.copy_opts(from, to, options).await
    }

    async fn rename_opts(
        &self,
        from: &Path,
        to: &Path,
        options: RenameOptions,
    ) -> Result<()> {
        self.inner.rename_opts(from, to, options).await
    }
}
