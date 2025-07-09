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

use crate::sorts::cursor::{ArrayValues, CursorArray, RowValues};
use crate::SendableRecordBatchStream;
use crate::{PhysicalExpr, PhysicalSortExpr};
use arrow::array::Array;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, Rows, SortField};
use datafusion_common::{internal_datafusion_err, Result};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_physical_expr_common::sort_expr::LexOrdering;
use futures::stream::StreamExt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::task::{ready, Context, Poll};
use tokio::sync::mpsc::{channel, Receiver};

/// A [`Stream`](futures::Stream) that has multiple partitions that can
/// be polled separately but not concurrently
///
/// Used by sort preserving merge to decouple the cursor merging logic from
/// the source of the cursors, the intention being to allow preserving
/// any row encoding performed for intermediate sorts
pub trait PartitionedStream: std::fmt::Debug + Send {
    type Output;

    /// Returns the number of partitions
    fn partitions(&self) -> usize;

    fn poll_next(
        &mut self,
        cx: &mut Context<'_>,
        stream_idx: usize,
    ) -> Poll<Option<Self::Output>>;
}

const ROW_CURSOR_STREAM_BUFFER_SIZE: usize = 2;

/// A new type wrapper around a set of fused [`SendableRecordBatchStream`]
/// that implements debug, and skips over empty [`RecordBatch`]
struct FusedStreams(Vec<Receiver<Option<Result<RecordBatch>>>>);

impl std::fmt::Debug for FusedStreams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FusedStreams")
            .field("num_streams", &self.0.len())
            .finish()
    }
}

impl FusedStreams {
    fn poll_next(
        &mut self,
        cx: &mut Context<'_>,
        stream_idx: usize,
    ) -> Poll<Option<Result<RecordBatch>>> {
        match self.0[stream_idx].poll_recv(cx) {
            Poll::Ready(Some(Some(r))) => Poll::Ready(Some(r)),
            Poll::Ready(Some(None)) => Poll::Ready(None), // Stream ended
            Poll::Ready(None) => Poll::Ready(None), // Channel closed
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T: CursorArray> Drop for FieldCursorStream<T> {
    fn drop(&mut self) {
        for handle in &self.task_handles {
            handle.abort();
        }
    }
}

impl Drop for RowCursorStream {
    fn drop(&mut self) {
        for handle in &self.task_handles {
            handle.abort();
        }
    }
}

/// A pair of `Arc<Rows>` that can be reused
#[derive(Debug)]
struct ReusableRows {
    // inner[stream_idx] holds a two Arcs:
    // at start of a new poll
    // .0 is the rows from the previous poll (at start),
    // .1 is the one that is being written to
    // at end of a poll, .0 will be swapped with .1,
    inner: Vec<[Option<Arc<Rows>>; 2]>,
}

impl ReusableRows {
    // return a Rows for writing,
    // does not clone if the existing rows can be reused
    fn take_next(&mut self, stream_idx: usize) -> Result<Rows> {
        Arc::try_unwrap(self.inner[stream_idx][1].take().unwrap()).map_err(|_| {
            internal_datafusion_err!(
                "Rows from RowCursorStream is still in use by consumer"
            )
        })
    }
    // save the Rows
    fn save(&mut self, stream_idx: usize, rows: Arc<Rows>) {
        self.inner[stream_idx][1] = Some(Arc::clone(&rows));
        // swap the curent with the previous one, so that the next poll can reuse the Rows from the previous poll
        let [a, b] = &mut self.inner[stream_idx];
        std::mem::swap(a, b);
    }
}

/// A [`PartitionedStream`] that wraps a set of [`SendableRecordBatchStream`]
/// and computes [`RowValues`] based on the provided [`PhysicalSortExpr`]
/// Note: the stream returns an error if the consumer buffers more than one RowValues (i.e. holds on to two RowValues
/// from the same partition at the same time).
// Tokios MPSC channel is used to send data from spawned tasks to the RowCursorStream.
// When RowCursorStream is dropped, the receivers are dropped. The spawned tasks will detect this
// when they try to send data, and they will terminate. However, if a spawned task is idle (e.g.
// waiting for its input stream to produce a batch), it might not terminate immediately.
// To ensure prompt termination and resource cleanup, especially for tests like `test_drop_cancel`,
// we explicitly store and abort the JoinHandles of these tasks in the Drop impl.
#[derive(Debug)]
pub struct RowCursorStream {
    /// Converter to convert output of physical expressions
    converter: RowConverter,
    /// The physical expressions to sort by
    column_expressions: Vec<Arc<dyn PhysicalExpr>>,
    /// Input streams (receivers for data from spawned tasks)
    streams: FusedStreams,
    /// Tracks the memory used by `converter`
    reservation: MemoryReservation,
    /// Allocated rows for each partition, we keep two to allow for buffering one
    /// in the consumer of the stream
    rows: ReusableRows,
    /// JoinHandles for the tokio tasks polling input streams
    task_handles: Vec<tokio::task::JoinHandle<()>>,
}

impl RowCursorStream {
    pub fn try_new(
        schema: &Schema,
        expressions: &LexOrdering,
        streams: Vec<SendableRecordBatchStream>,
        reservation: MemoryReservation,
    ) -> Result<Self> {
        let sort_fields = expressions
            .iter()
            .map(|expr| {
                let data_type = expr.expr.data_type(schema)?;
                Ok(SortField::new_with_options(data_type, expr.options))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut receivers = Vec::with_capacity(streams.len());
        let mut task_handles = Vec::with_capacity(streams.len());
        for stream in streams {
            let (tx, rx) = channel(ROW_CURSOR_STREAM_BUFFER_SIZE);
            let mut fused_stream = stream.fuse();
            let handle = tokio::spawn(async move {
                loop {
                    match fused_stream.next().await {
                        Some(Ok(batch)) if batch.num_rows() == 0 => continue,
                        Some(Ok(batch)) => {
                            if tx.send(Some(Ok(batch))).await.is_err() {
                                // Receiver dropped, exiting
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            if tx.send(Some(Err(e))).await.is_err() {
                                // Receiver dropped, exiting
                                break;
                            }
                        }
                        None => {
                            // Stream finished
                            if tx.send(None).await.is_err() {
                                // Receiver dropped, exiting
                            }
                            break;
                        }
                    }
                }
            });
            receivers.push(rx);
            task_handles.push(handle);
        }

        let converter = RowConverter::new(sort_fields)?;
        let mut rows = Vec::with_capacity(receivers.len());
        for _ in &receivers {
            // Initialize each stream with an empty Rows
            rows.push([
                Some(Arc::new(converter.empty_rows(0, 0))),
                Some(Arc::new(converter.empty_rows(0, 0))),
            ]);
        }
        Ok(Self {
            converter,
            reservation,
            column_expressions: expressions.iter().map(|x| Arc::clone(&x.expr)).collect(),
            streams: FusedStreams(receivers),
            rows: ReusableRows { inner: rows },
            task_handles,
        })
    }

    fn convert_batch(
        &mut self,
        batch: &RecordBatch,
        stream_idx: usize,
    ) -> Result<RowValues> {
        let cols = self
            .column_expressions
            .iter()
            .map(|expr| expr.evaluate(batch)?.into_array(batch.num_rows()))
            .collect::<Result<Vec<_>>>()?;

        // At this point, ownership should of this Rows should be unique
        let mut rows = self.rows.take_next(stream_idx)?;

        rows.clear();

        self.converter.append(&mut rows, &cols)?;
        self.reservation.try_resize(self.converter.size())?;

        let rows = Arc::new(rows);

        self.rows.save(stream_idx, Arc::clone(&rows));

        // track the memory in the newly created Rows.
        let mut rows_reservation = self.reservation.new_empty();
        rows_reservation.try_grow(rows.size())?;
        Ok(RowValues::new(rows, rows_reservation))
    }
}

impl PartitionedStream for RowCursorStream {
    type Output = Result<(RowValues, RecordBatch)>;

    fn partitions(&self) -> usize {
        self.streams.0.len()
    }

    fn poll_next(
        &mut self,
        cx: &mut Context<'_>,
        stream_idx: usize,
    ) -> Poll<Option<Self::Output>> {
        match ready!(self.streams.poll_next(cx, stream_idx)) {
            Some(Ok(batch)) => {
                let cursor_result = self.convert_batch(&batch, stream_idx);
                Poll::Ready(Some(cursor_result.map(|cursor| (cursor, batch))))
            }
            Some(Err(e)) => Poll::Ready(Some(Err(e))),
            None => Poll::Ready(None),
        }
    }
}

const FIELD_CURSOR_STREAM_BUFFER_SIZE: usize = 2;

/// Specialized stream for sorts on single primitive columns
pub struct FieldCursorStream<T: CursorArray> {
    /// The physical expressions to sort by
    sort: PhysicalSortExpr,
    /// Input streams (receivers for data from spawned tasks)
    streams: FusedStreams,
    /// Create new reservations for each array
    reservation: MemoryReservation,
    phantom: PhantomData<fn(T) -> T>,
    /// JoinHandles for the tokio tasks polling input streams
    task_handles: Vec<tokio::task::JoinHandle<()>>,
}

impl<T: CursorArray> std::fmt::Debug for FieldCursorStream<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimitiveCursorStream")
            .field("num_streams", &self.streams)
            .finish()
    }
}

impl<T: CursorArray> FieldCursorStream<T> {
    pub fn new(
        sort: PhysicalSortExpr,
        streams: Vec<SendableRecordBatchStream>,
        reservation: MemoryReservation,
    ) -> Self {
        let mut receivers = Vec::with_capacity(streams.len());
        let mut task_handles = Vec::with_capacity(streams.len());
        for stream in streams {
            let (tx, rx) = channel(FIELD_CURSOR_STREAM_BUFFER_SIZE);
            let mut fused_stream = stream.fuse();
            let handle = tokio::spawn(async move {
                loop {
                    match fused_stream.next().await {
                        Some(Ok(batch)) if batch.num_rows() == 0 => continue,
                        Some(Ok(batch)) => {
                            if tx.send(Some(Ok(batch))).await.is_err() {
                                // Receiver dropped, exiting
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            if tx.send(Some(Err(e))).await.is_err() {
                                // Receiver dropped, exiting
                                break;
                            }
                        }
                        None => {
                            // Stream finished
                            if tx.send(None).await.is_err() {
                                // Receiver dropped, exiting
                            }
                            break;
                        }
                    }
                }
            });
            receivers.push(rx);
            task_handles.push(handle);
        }

        Self {
            sort,
            streams: FusedStreams(receivers),
            reservation,
            phantom: Default::default(),
            task_handles,
        }
    }

    fn convert_batch(&mut self, batch: &RecordBatch) -> Result<ArrayValues<T::Values>> {
        let value = self.sort.expr.evaluate(batch)?;
        let array = value.into_array(batch.num_rows())?;
        let size_in_mem = array.get_buffer_memory_size();
        let array = array.as_any().downcast_ref::<T>().expect("field values");
        let mut array_reservation = self.reservation.new_empty();
        array_reservation.try_grow(size_in_mem)?;
        Ok(ArrayValues::new(
            self.sort.options,
            array,
            array_reservation,
        ))
    }
}

impl<T: CursorArray> PartitionedStream for FieldCursorStream<T> {
    type Output = Result<(ArrayValues<T::Values>, RecordBatch)>;

    fn partitions(&self) -> usize {
        self.streams.0.len()
    }

    fn poll_next(
        &mut self,
        cx: &mut Context<'_>,
        stream_idx: usize,
    ) -> Poll<Option<Self::Output>> {
        match ready!(self.streams.poll_next(cx, stream_idx)) {
            Some(Ok(batch)) => {
                let cursor_result = self.convert_batch(&batch);
                Poll::Ready(Some(cursor_result.map(|cursor| (cursor, batch))))
            }
            Some(Err(e)) => Poll::Ready(Some(Err(e))),
            None => Poll::Ready(None),
        }
    }
}
