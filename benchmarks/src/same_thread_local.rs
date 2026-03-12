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

//! A [`LocalFileSystem`] wrapper that performs IO on the calling thread
//! instead of dispatching to tokio's blocking thread pool.
//!
//! When partition threads are pinned to specific CPU cores, using
//! `spawn_blocking` dispatches IO to a random thread on a potentially
//! different core, losing cache locality. This wrapper uses
//! [`tokio::task::block_in_place`] to run file IO directly on the
//! calling tokio worker thread, keeping the work on the same pinned core.

use std::fmt::{Debug, Display, Formatter};
use std::fs::{File, Metadata};
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

/// Global flag: when set, `object_store_registry()` returns a registry
/// with [`SameThreadLocalFileSystem`] instead of the default [`LocalFileSystem`].
static USE_SAME_THREAD_IO: AtomicBool = AtomicBool::new(false);

/// Enable same-thread IO for all subsequently created object store registries.
pub fn enable_same_thread_io() {
    USE_SAME_THREAD_IO.store(true, Ordering::Relaxed);
}

/// Returns true if same-thread IO is enabled.
pub fn is_same_thread_io_enabled() -> bool {
    USE_SAME_THREAD_IO.load(Ordering::Relaxed)
}

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

/// A [`LocalFileSystem`] wrapper that performs blocking IO on the calling
/// tokio worker thread via [`tokio::task::block_in_place`], rather than
/// dispatching to the blocking thread pool.
///
/// This ensures that when worker threads are pinned to CPU cores, the IO
/// stays on the same core as the partition that requested it, preserving
/// cache locality.
#[derive(Debug)]
pub struct SameThreadLocalFileSystem {
    inner: LocalFileSystem,
}

impl SameThreadLocalFileSystem {
    /// Create a new `SameThreadLocalFileSystem` with no path prefix.
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

/// Open a file and return it with its metadata.
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

#[async_trait]
impl ObjectStore for SameThreadLocalFileSystem {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> Result<PutResult> {
        // Writes are not on the hot read path; delegate to inner.
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
        tokio::task::block_in_place(move || {
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
        })
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> Result<Vec<Bytes>> {
        let path = self.inner.path_to_filesystem(location)?;
        let ranges = ranges.to_vec();
        tokio::task::block_in_place(move || {
            let (mut file, _) = open_file(&path)?;
            ranges
                .into_iter()
                .map(|r| read_range(&mut file, &path, r))
                .collect()
        })
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
