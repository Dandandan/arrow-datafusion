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

//! A [`LocalFileSystem`] wrapper that caches open file descriptors and uses
//! [`maybe_spawn_blocking`] for blocking I/O on the read path.

use std::fmt::{Debug, Display, Formatter};
use std::fs::{File, Metadata};
use std::io::ErrorKind;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

#[cfg(target_family = "unix")]
use std::os::unix::fs::FileExt;
#[cfg(target_family = "windows")]
use std::os::windows::fs::FileExt;

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use futures::stream::BoxStream;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{
    Attributes, CopyOptions, GetOptions, GetResult, GetResultPayload, ListResult,
    MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOptions, PutOptions,
    PutPayload, PutResult, RenameOptions, Result,
};

/// A [`LocalFileSystem`] wrapper that caches open file descriptors so each file
/// is opened at most once for the lifetime of this store.
///
/// On the read path (`get_opts`, `get_ranges`) the cached handle is reused via
/// positional reads (`pread` / `read_at`) which are thread-safe and do not
/// require seeking.
///
/// Non-read operations (put, delete, list, copy, rename) are delegated to the
/// inner [`LocalFileSystem`].
pub struct BlockInPlaceLocalFileSystem {
    inner: LocalFileSystem,
    /// Cache of open file descriptors, keyed by absolute filesystem path.
    file_cache: DashMap<PathBuf, Arc<File>>,
}

impl Debug for BlockInPlaceLocalFileSystem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockInPlaceLocalFileSystem")
            .field("inner", &self.inner)
            .field("cached_files", &self.file_cache.len())
            .finish()
    }
}

impl BlockInPlaceLocalFileSystem {
    /// Create new filesystem storage with no prefix
    pub fn new() -> Self {
        Self {
            inner: LocalFileSystem::new(),
            file_cache: DashMap::new(),
        }
    }

    /// Create new filesystem storage with `prefix` applied to all paths
    pub fn new_with_prefix(prefix: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self {
            inner: LocalFileSystem::new_with_prefix(prefix)?,
            file_cache: DashMap::new(),
        })
    }

    /// Get a cached file handle or open and cache a new one.
    fn get_or_open_file(&self, path: &std::path::Path) -> Result<Arc<File>> {
        if let Some(entry) = self.file_cache.get(path) {
            return Ok(Arc::clone(entry.value()));
        }
        let file = open_file(path)?;
        let file = Arc::new(file);
        self.file_cache
            .insert(path.to_path_buf(), Arc::clone(&file));
        Ok(file)
    }
}

impl Default for BlockInPlaceLocalFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for BlockInPlaceLocalFileSystem {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlockInPlaceLocalFileSystem")
    }
}

#[async_trait]
impl ObjectStore for BlockInPlaceLocalFileSystem {
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
        let path = self.inner.path_to_filesystem(location)?;
        let file = self.get_or_open_file(&path)?;
        let location = location.clone();
        maybe_spawn_blocking(move || {
            let metadata = file_metadata(&file, &path)?;
            let meta = convert_metadata(metadata, location);
            options.check_preconditions(&meta)?;

            let range = match options.range {
                Some(r) => {
                    r.as_range(meta.size)
                        .map_err(|e| object_store::Error::Generic {
                            store: "LocalFileSystem",
                            source: Box::new(e),
                        })?
                }
                None => 0..meta.size,
            };

            // Dup the fd so the caller can take ownership without
            // invalidating the cached handle.
            let owned_file = file.try_clone().map_err(|e| {
                object_store::Error::Generic {
                    store: "LocalFileSystem",
                    source: Box::new(IoError {
                        source: e,
                        path: path.clone(),
                    }),
                }
            })?;

            Ok(GetResult {
                payload: GetResultPayload::File(owned_file, path),
                attributes: Attributes::default(),
                range,
                meta,
            })
        })
        .await
    }

    async fn get_ranges(
        &self,
        location: &Path,
        ranges: &[Range<u64>],
    ) -> Result<Vec<Bytes>> {
        let path = self.inner.path_to_filesystem(location)?;
        let file = self.get_or_open_file(&path)?;
        let ranges = ranges.to_vec();
        maybe_spawn_blocking(move || {
            ranges
                .into_iter()
                .map(|r| read_range(&file, &path, r))
                .collect()
        })
        .await
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

// --- Helper functions (mirroring object_store::local internals) ---

/// If running in a tokio context, dispatches `f` via `spawn_blocking`.
/// Otherwise runs `f` directly on the current thread.
async fn maybe_spawn_blocking<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    match tokio::runtime::Handle::try_current() {
        Ok(runtime) => runtime.spawn_blocking(f).await?,
        Err(_) => f(),
    }
}

fn open_file(path: &std::path::Path) -> Result<File> {
    File::open(path).map_err(|e| map_open_error(e, path).into())
}

fn file_metadata(file: &File, path: &std::path::Path) -> Result<Metadata> {
    let metadata = file.metadata().map_err(|e| map_open_error(e, path))?;
    if metadata.is_dir() {
        Err(map_not_found(
            path,
            std::io::Error::new(ErrorKind::NotFound, "is directory"),
        )
        .into())
    } else {
        Ok(metadata)
    }
}

fn map_open_error(
    source: std::io::Error,
    path: &std::path::Path,
) -> object_store::Error {
    match source.kind() {
        ErrorKind::NotFound => map_not_found(path, source),
        _ => object_store::Error::Generic {
            store: "LocalFileSystem",
            source: Box::new(IoError {
                source,
                path: path.to_path_buf(),
            }),
        },
    }
}

fn map_not_found(
    path: &std::path::Path,
    source: std::io::Error,
) -> object_store::Error {
    object_store::Error::NotFound {
        path: path.to_string_lossy().to_string(),
        source: source.into(),
    }
}

fn convert_metadata(metadata: Metadata, location: Path) -> ObjectMeta {
    let last_modified: DateTime<Utc> = metadata
        .modified()
        .expect("Modified file time should be supported on this platform")
        .into();

    ObjectMeta {
        location,
        last_modified,
        size: metadata.len(),
        e_tag: Some(get_etag(&metadata)),
        version: None,
    }
}

fn get_etag(metadata: &Metadata) -> String {
    let inode = get_inode(metadata);
    let size = metadata.len();
    let mtime = metadata
        .modified()
        .ok()
        .and_then(|mtime| mtime.duration_since(SystemTime::UNIX_EPOCH).ok())
        .unwrap_or_default()
        .as_micros();

    format!("{inode:x}-{mtime:x}-{size:x}")
}

#[cfg(unix)]
fn get_inode(metadata: &Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::ino(metadata)
}

#[cfg(not(unix))]
fn get_inode(_metadata: &Metadata) -> u64 {
    0
}

/// Read a byte range from a file, using positional reads (`pread` / `read_at`)
/// which do not require seeking and are safe to call concurrently on a shared
/// file descriptor.
#[cfg(any(target_family = "unix", target_family = "windows"))]
pub(crate) fn read_range(
    file: &File,
    path: &std::path::Path,
    range: Range<u64>,
) -> Result<Bytes> {
    let requested = range.end - range.start;
    let mut buf = vec![0u8; requested as usize];

    let mut buf_slice = &mut buf[..];
    let mut offset = range.start;

    while !buf_slice.is_empty() {
        #[cfg(target_family = "unix")]
        let read_result = file.read_at(buf_slice, offset);

        #[cfg(target_family = "windows")]
        let read_result = file.seek_read(buf_slice, offset);

        match read_result {
            Ok(0) => break,
            Ok(n) => {
                let tmp = buf_slice;
                buf_slice = &mut tmp[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(source) => {
                return Err(object_store::Error::Generic {
                    store: "LocalFileSystem",
                    source: Box::new(IoError {
                        source,
                        path: path.to_path_buf(),
                    }),
                });
            }
        }
    }

    if !buf_slice.is_empty() {
        let metadata = file_metadata(file, path)?;
        let file_len = metadata.len();

        if range.start >= file_len {
            return Err(object_store::Error::Generic {
                store: "LocalFileSystem",
                source: format!(
                    "Range start {} exceeds file length {}",
                    range.start, file_len
                )
                .into(),
            });
        }

        return Err(object_store::Error::Generic {
            store: "LocalFileSystem",
            source: format!(
                "Out of range for {}: expected {} bytes, got {}",
                path.display(),
                range.end.min(file_len) - range.start,
                offset - range.start,
            )
            .into(),
        });
    }

    Ok(buf.into())
}

/// Fallback for platforms without positional read support.
#[cfg(all(not(target_family = "unix"), not(target_family = "windows")))]
pub(crate) fn read_range(
    file: &File,
    path: &std::path::Path,
    range: Range<u64>,
) -> Result<Bytes> {
    use std::io::{Read, Seek, SeekFrom};

    // Need a mutable handle for seek; clone the fd.
    let mut file = file.try_clone().map_err(|source| {
        object_store::Error::Generic {
            store: "LocalFileSystem",
            source: Box::new(IoError {
                source,
                path: path.to_path_buf(),
            }),
        }
    })?;

    let requested = range.end - range.start;
    let mut buf = Vec::with_capacity(requested as usize);

    file.seek(SeekFrom::Start(range.start)).map_err(|source| {
        object_store::Error::Generic {
            store: "LocalFileSystem",
            source: Box::new(IoError {
                source,
                path: path.to_path_buf(),
            }),
        }
    })?;

    let read = (&mut file)
        .take(requested)
        .read_to_end(&mut buf)
        .map_err(|source| object_store::Error::Generic {
            store: "LocalFileSystem",
            source: Box::new(IoError {
                source,
                path: path.to_path_buf(),
            }),
        })? as u64;

    if read != requested {
        return Err(object_store::Error::Generic {
            store: "LocalFileSystem",
            source: format!(
                "Out of range for {}: expected {} bytes, got {}",
                path.display(),
                requested,
                read,
            )
            .into(),
        });
    }

    Ok(buf.into())
}

/// Wrapper to make io::Error displayable with path context
#[derive(Debug)]
struct IoError {
    source: std::io::Error,
    path: PathBuf,
}

impl Display for IoError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.path.display(), self.source)
    }
}

impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
