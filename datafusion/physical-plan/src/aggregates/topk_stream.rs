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

//! A memory-conscious aggregation implementation that limits group buckets to a fixed number

use crate::aggregates::group_values::GroupByMetrics;
use crate::aggregates::topk::priority_map::{InsertAction, PriorityMap};
#[cfg(debug_assertions)]
use crate::aggregates::topk_types_supported;
use crate::aggregates::{
    AggregateExec, AggregateInputMode, AggregateMode, PhysicalGroupBy,
    aggregate_expressions, evaluate_group_by, evaluate_many,
};
use crate::metrics::BaselineMetrics;
use crate::{RecordBatchStream, SendableRecordBatchStream};
use arrow::array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow::compute;
use arrow::datatypes::SchemaRef;
use arrow::util::pretty::print_batches;
use datafusion_common::Result;
use datafusion_common::internal_datafusion_err;
use datafusion_execution::TaskContext;
use datafusion_expr::Accumulator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use datafusion_physical_expr_common::metrics::RecordOutput;
use futures::stream::{Stream, StreamExt};
use log::{Level, trace};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// The operating mode of the TopK aggregate stream.
enum TopKMode {
    /// No aggregate expressions; ordering is by the group key.
    Distinct,
    /// A single MIN or MAX aggregate determines ordering.
    MinMax,
    /// Ordering is by the group key, but there are aggregate expressions
    /// tracked via per-slot accumulators.
    GroupKeyOrdered,
}

pub struct GroupedTopKAggregateStream {
    partition: usize,
    row_count: usize,
    started: bool,
    schema: SchemaRef,
    input: SendableRecordBatchStream,
    baseline_metrics: BaselineMetrics,
    group_by_metrics: GroupByMetrics,
    aggregate_arguments: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    group_by: Arc<PhysicalGroupBy>,
    priority_map: PriorityMap,
    mode: TopKMode,
    /// Aggregate mode (Partial, Final, etc.).
    agg_mode: AggregateMode,
    /// Aggregate function expressions.
    aggr_expr: Arc<[Arc<AggregateFunctionExpr>]>,
    /// Per-slot accumulators, indexed by PriorityMap's `map_idx`.
    /// Only used in [`TopKMode::GroupKeyOrdered`].
    slot_accumulators: Vec<Option<Vec<Box<dyn Accumulator>>>>,
}

impl GroupedTopKAggregateStream {
    pub fn new(
        aggr: &AggregateExec,
        context: &Arc<TaskContext>,
        partition: usize,
        limit: usize,
    ) -> Result<Self> {
        let agg_schema = Arc::clone(&aggr.schema);
        let group_by = Arc::clone(&aggr.group_by);
        let input = aggr.input.execute(partition, Arc::clone(context))?;
        let baseline_metrics = BaselineMetrics::new(&aggr.metrics, partition);
        let group_by_metrics = GroupByMetrics::new(&aggr.metrics, partition);
        let aggregate_arguments =
            aggregate_expressions(&aggr.aggr_expr, &aggr.mode, group_by.expr.len())?;

        let (expr, _) = &aggr.group_expr().expr()[0];
        let kt = expr.data_type(&aggr.input().schema())?;

        let (mode, vt, desc) = if let Some((val_field, desc)) = aggr.get_minmax_desc() {
            (TopKMode::MinMax, val_field.data_type().clone(), desc)
        } else if aggr.aggr_expr().is_empty() {
            let desc = aggr
                .limit_options()
                .and_then(|config| config.descending)
                .ok_or_else(|| {
                    internal_datafusion_err!(
                        "Ordering direction required for DISTINCT with limit"
                    )
                })?;
            (TopKMode::Distinct, kt.clone(), desc)
        } else {
            let desc = aggr
                .limit_options()
                .and_then(|config| config.descending)
                .ok_or_else(|| {
                    internal_datafusion_err!(
                        "Ordering direction required for ORDER BY group key with limit"
                    )
                })?;
            (TopKMode::GroupKeyOrdered, kt.clone(), desc)
        };

        #[cfg(debug_assertions)]
        {
            debug_assert!(
                topk_types_supported(&kt, &vt),
                "TopK type validation should have been performed by optimizer. \
                 Found unsupported types: key={kt:?}, value={vt:?}"
            );
        }

        let priority_map = PriorityMap::new(kt, vt, limit, desc)?;

        Ok(GroupedTopKAggregateStream {
            partition,
            started: false,
            row_count: 0,
            schema: agg_schema,
            input,
            baseline_metrics,
            group_by_metrics,
            aggregate_arguments,
            group_by,
            priority_map,
            mode,
            agg_mode: aggr.mode,
            aggr_expr: Arc::clone(&aggr.aggr_expr),
            slot_accumulators: Vec::new(),
        })
    }
}

impl RecordBatchStream for GroupedTopKAggregateStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// Update accumulators at `slot` with the given `row_indices` from `arg_arrays`.
/// Uses `update_batch` for raw input, `merge_batch` for partial state.
fn flush_rows_to_slot(
    slot_accumulators: &mut [Option<Vec<Box<dyn Accumulator>>>],
    slot: usize,
    row_indices: &[u32],
    arg_arrays: &[Vec<ArrayRef>],
    raw_input: bool,
) -> Result<()> {
    if row_indices.is_empty() {
        return Ok(());
    }
    let indices = UInt32Array::from(row_indices.to_vec());
    if let Some(accumulators) = &mut slot_accumulators[slot] {
        for (agg_idx, acc) in accumulators.iter_mut().enumerate() {
            let args: Vec<ArrayRef> = arg_arrays[agg_idx]
                .iter()
                .map(|arr| compute::take(arr.as_ref(), &indices, None))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if raw_input {
                acc.update_batch(&args)?;
            } else {
                acc.merge_batch(&args)?;
            }
        }
    }
    Ok(())
}

impl GroupedTopKAggregateStream {
    /// Intern path for Distinct and MinMax modes (no accumulators needed).
    fn intern(&mut self, ids: &ArrayRef, vals: &ArrayRef) -> Result<()> {
        let _timer = self.group_by_metrics.time_calculating_group_ids.timer();

        let len = ids.len();
        self.priority_map
            .set_batch(Arc::clone(ids), Arc::clone(vals));

        let has_nulls = vals.null_count() > 0;
        for row_idx in 0..len {
            if has_nulls && vals.is_null(row_idx) {
                continue;
            }
            self.priority_map.insert(row_idx)?;
        }
        Ok(())
    }

    /// Intern path for GroupKeyOrdered mode: maintains accumulators alongside PriorityMap.
    fn intern_group_key_ordered(
        &mut self,
        group_by_values: &ArrayRef,
        batch: &RecordBatch,
    ) -> Result<()> {
        let len = group_by_values.len();
        self.priority_map
            .set_batch(Arc::clone(group_by_values), Arc::clone(group_by_values));

        let has_nulls = group_by_values.null_count() > 0;

        // Pre-evaluate all aggregate argument expressions
        let all_arg_arrays: Vec<Vec<ArrayRef>> = self
            .aggregate_arguments
            .iter()
            .map(|args| {
                args.iter()
                    .map(|expr| {
                        expr.evaluate(batch)
                            .and_then(|v| v.into_array(batch.num_rows()))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let raw_input = self.agg_mode.input_mode() == AggregateInputMode::Raw;
        let mut rows_per_slot: HashMap<usize, Vec<u32>> = HashMap::new();

        for row_idx in 0..len {
            if has_nulls && group_by_values.is_null(row_idx) {
                continue;
            }
            let action = self.priority_map.insert_with_action(row_idx)?;
            match action {
                InsertAction::Skipped => continue,
                InsertAction::Added(map_idx) => {
                    if map_idx >= self.slot_accumulators.len() {
                        self.slot_accumulators.resize_with(map_idx + 1, || None);
                    }
                    let accs: Vec<Box<dyn Accumulator>> = self
                        .aggr_expr
                        .iter()
                        .map(|e| e.create_accumulator())
                        .collect::<Result<_>>()?;
                    self.slot_accumulators[map_idx] = Some(accs);
                    rows_per_slot
                        .entry(map_idx)
                        .or_default()
                        .push(row_idx as u32);
                }
                InsertAction::Replaced(map_idx) => {
                    // Flush pending rows for the EVICTED group before resetting
                    if let Some(old_rows) = rows_per_slot.remove(&map_idx) {
                        flush_rows_to_slot(
                            &mut self.slot_accumulators,
                            map_idx,
                            &old_rows,
                            &all_arg_arrays,
                            raw_input,
                        )?;
                    }
                    if map_idx >= self.slot_accumulators.len() {
                        self.slot_accumulators.resize_with(map_idx + 1, || None);
                    }
                    let accs: Vec<Box<dyn Accumulator>> = self
                        .aggr_expr
                        .iter()
                        .map(|e| e.create_accumulator())
                        .collect::<Result<_>>()?;
                    self.slot_accumulators[map_idx] = Some(accs);
                    rows_per_slot
                        .entry(map_idx)
                        .or_default()
                        .push(row_idx as u32);
                }
                InsertAction::Updated(map_idx) => {
                    rows_per_slot
                        .entry(map_idx)
                        .or_default()
                        .push(row_idx as u32);
                }
            }
        }

        // Flush all remaining rows
        for (map_idx, rows) in &rows_per_slot {
            flush_rows_to_slot(
                &mut self.slot_accumulators,
                *map_idx,
                rows,
                &all_arg_arrays,
                raw_input,
            )?;
        }

        Ok(())
    }

    /// Emit accumulator results for GroupKeyOrdered mode.
    fn emit_group_key_ordered(&mut self) -> Result<RecordBatch> {
        let (mut cols, map_idxs) = self.priority_map.emit_with_indices()?;
        // Only keep the group key column (drop the duplicate ordering column)
        cols.truncate(1);

        let num_aggs = self.aggr_expr.len();
        let produces_state = self.agg_mode.output_mode()
            == crate::aggregates::AggregateOutputMode::Partial;

        if produces_state {
            // Emit accumulator state columns
            for agg_idx in 0..num_aggs {
                let mut all_states: Vec<Vec<datafusion_common::ScalarValue>> = Vec::new();
                for &mi in &map_idxs {
                    if let Some(accumulators) = &mut self.slot_accumulators[mi] {
                        let state: Vec<datafusion_common::ScalarValue> =
                            accumulators[agg_idx].state()?;
                        if all_states.is_empty() {
                            all_states = (0..state.len())
                                .map(|_| {
                                    Vec::<datafusion_common::ScalarValue>::with_capacity(
                                        map_idxs.len(),
                                    )
                                })
                                .collect();
                        }
                        for (col_idx, val) in state.into_iter().enumerate() {
                            all_states[col_idx].push(val);
                        }
                    }
                }
                for state_col in all_states {
                    cols.push(datafusion_common::ScalarValue::iter_to_array(state_col)?);
                }
            }
        } else {
            // Emit evaluated values
            for agg_idx in 0..num_aggs {
                let mut values: Vec<datafusion_common::ScalarValue> =
                    Vec::with_capacity(map_idxs.len());
                for &mi in &map_idxs {
                    let val = if let Some(accs) = &mut self.slot_accumulators[mi] {
                        accs[agg_idx].evaluate()?
                    } else {
                        datafusion_common::ScalarValue::Null
                    };
                    values.push(val);
                }
                cols.push(datafusion_common::ScalarValue::iter_to_array(values)?);
            }
        }

        Ok(RecordBatch::try_new(Arc::clone(&self.schema), cols)?)
    }
}

impl Stream for GroupedTopKAggregateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let elapsed_compute = self.baseline_metrics.elapsed_compute().clone();
        let emitting_time = self.group_by_metrics.emitting_time.clone();
        while let Poll::Ready(res) = self.input.poll_next_unpin(cx) {
            let _timer = elapsed_compute.timer();
            match res {
                Some(Ok(batch)) => {
                    self.started = true;
                    trace!(
                        "partition {} has {} rows and got batch with {} rows",
                        self.partition,
                        self.row_count,
                        batch.num_rows()
                    );
                    if log::log_enabled!(Level::Trace) && batch.num_rows() < 20 {
                        print_batches(std::slice::from_ref(&batch))?;
                    }
                    self.row_count += batch.num_rows();
                    let group_by_values = evaluate_group_by(&self.group_by, &batch)?;
                    assert_eq!(
                        group_by_values.len(),
                        1,
                        "Exactly 1 group value required"
                    );
                    assert_eq!(
                        group_by_values[0].len(),
                        1,
                        "Exactly 1 group value required"
                    );
                    let group_by_values = Arc::clone(&group_by_values[0][0]);

                    match self.mode {
                        TopKMode::GroupKeyOrdered => {
                            (*self).intern_group_key_ordered(&group_by_values, &batch)?;
                        }
                        TopKMode::Distinct => {
                            (*self).intern(&group_by_values, &group_by_values)?;
                        }
                        TopKMode::MinMax => {
                            let input_values = {
                                let _timer = self
                                    .group_by_metrics
                                    .aggregate_arguments_time
                                    .timer();
                                let input_values =
                                    evaluate_many(&self.aggregate_arguments, &batch)?;
                                assert_eq!(
                                    input_values.len(),
                                    1,
                                    "Exactly 1 input required"
                                );
                                assert_eq!(
                                    input_values[0].len(),
                                    1,
                                    "Exactly 1 input required"
                                );
                                Arc::clone(&input_values[0][0])
                            };
                            (*self).intern(&group_by_values, &input_values)?;
                        }
                    }
                }
                None => {
                    if self.priority_map.is_empty() {
                        trace!("partition {} emit None", self.partition);
                        return Poll::Ready(None);
                    }
                    let batch = {
                        let _timer = emitting_time.timer();
                        match self.mode {
                            TopKMode::GroupKeyOrdered => self.emit_group_key_ordered()?,
                            TopKMode::Distinct => {
                                let mut cols = self.priority_map.emit()?;
                                cols.truncate(1);
                                RecordBatch::try_new(Arc::clone(&self.schema), cols)?
                            }
                            TopKMode::MinMax => {
                                let cols = self.priority_map.emit()?;
                                RecordBatch::try_new(Arc::clone(&self.schema), cols)?
                            }
                        }
                    };
                    let batch = batch.record_output(&self.baseline_metrics);
                    trace!(
                        "partition {} emit batch with {} rows",
                        self.partition,
                        batch.num_rows()
                    );
                    if log::log_enabled!(Level::Trace) {
                        print_batches(std::slice::from_ref(&batch))?;
                    }
                    return Poll::Ready(Some(Ok(batch)));
                }
                Some(Err(e)) => {
                    return Poll::Ready(Some(Err(e)));
                }
            }
        }
        Poll::Pending
    }
}
