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

//! Fused partial aggregation + hash repartition operator.
//!
//! [`AggregateRepartitionExec`] combines a partial aggregation with a hash
//! repartition into a single operator, avoiding intermediate materialization
//! and redundant hashing.

use std::any::Any;
use std::fmt::Formatter;
use std::sync::Arc;

use crate::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use crate::execution_plan::{CardinalityEffect, EmissionType};
use crate::metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use crate::repartition::BatchPartitioner;
use crate::stream::RecordBatchStreamAdapter;
use crate::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream,
};

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, internal_err};
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion_physical_expr_common::physical_expr::fmt_sql;

use futures::StreamExt;
use log::debug;
use parking_lot::Mutex;

/// Fused partial aggregation + hash repartition operator.
///
/// This operator performs partial aggregation and hash repartitioning in a
/// single step, eliminating the intermediate materialization between
/// a `Partial AggregateExec` and a `Hash RepartitionExec`.
///
/// It has N input partitions and M output partitions. Internally, for each
/// input partition it runs a `GroupedHashAggregateStream` (via a temporary
/// `AggregateExec` in Partial mode), then hash-partitions the emitted batches
/// by the group keys and sends them to the appropriate output channel.
#[derive(Debug)]
pub struct AggregateRepartitionExec {
    /// Group by expressions for partial aggregation
    group_by: Arc<PhysicalGroupBy>,
    /// Aggregate expressions
    aggr_expr: Arc<[Arc<AggregateFunctionExpr>]>,
    /// FILTER (WHERE clause) expression for each aggregate expression
    filter_expr: Arc<[Option<Arc<dyn PhysicalExpr>>]>,
    /// Input plan
    input: Arc<dyn ExecutionPlan>,
    /// Schema after partial aggregation is applied
    schema: SchemaRef,
    /// Input schema before any aggregation
    input_schema: SchemaRef,
    /// Number of output partitions for hash repartitioning
    num_output_partitions: usize,
    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Cached plan properties
    cache: Arc<PlanProperties>,
    /// Shared state: lazily initialized channels + spawned tasks
    state: Arc<Mutex<FusedState>>,
}

impl Clone for AggregateRepartitionExec {
    fn clone(&self) -> Self {
        Self {
            group_by: Arc::clone(&self.group_by),
            aggr_expr: Arc::clone(&self.aggr_expr),
            filter_expr: Arc::clone(&self.filter_expr),
            input: Arc::clone(&self.input),
            schema: Arc::clone(&self.schema),
            input_schema: Arc::clone(&self.input_schema),
            num_output_partitions: self.num_output_partitions,
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::clone(&self.cache),
            state: Arc::new(Mutex::new(FusedState::NotInitialized)),
        }
    }
}

/// Internal state for managing channels and spawned tasks.
#[derive(Debug)]
enum FusedState {
    /// Not yet initialized.
    NotInitialized,
    /// Channels created and tasks spawned.
    Initialized {
        /// One receiver per output partition. Set to `None` once taken.
        receivers: Vec<Option<tokio::sync::mpsc::Receiver<Result<RecordBatch>>>>,
        /// Keeps spawned tasks alive; dropping aborts them.
        _abort_helper: Vec<SpawnedTask<()>>,
    },
}

impl AggregateRepartitionExec {
    /// Create a new `AggregateRepartitionExec`.
    pub fn try_new(
        group_by: Arc<PhysicalGroupBy>,
        aggr_expr: Vec<Arc<AggregateFunctionExpr>>,
        filter_expr: Vec<Option<Arc<dyn PhysicalExpr>>>,
        input: Arc<dyn ExecutionPlan>,
        input_schema: SchemaRef,
        num_output_partitions: usize,
    ) -> Result<Self> {
        let schema = crate::aggregates::create_schema(
            &input.schema(),
            &group_by,
            &aggr_expr,
            AggregateMode::Partial,
        )?;
        let schema = Arc::new(schema);

        let group_key_columns =
            Self::group_key_column_exprs(group_by.num_group_exprs(), &schema);
        let output_partitioning =
            Partitioning::Hash(group_key_columns, num_output_partitions);

        let eq_properties = EquivalenceProperties::new(Arc::clone(&schema));

        let cache = PlanProperties::new(
            eq_properties,
            output_partitioning,
            EmissionType::Final,
            input.boundedness(),
        );

        Ok(Self {
            group_by,
            aggr_expr: aggr_expr.into(),
            filter_expr: filter_expr.into(),
            input,
            schema,
            input_schema,
            num_output_partitions,
            metrics: ExecutionPlanMetricsSet::new(),
            cache: Arc::new(cache),
            state: Arc::new(Mutex::new(FusedState::NotInitialized)),
        })
    }

    /// Build Column expressions for the group keys in the output schema.
    fn group_key_column_exprs(
        num_group_keys: usize,
        schema: &SchemaRef,
    ) -> Vec<Arc<dyn PhysicalExpr>> {
        (0..num_group_keys)
            .map(|i| {
                let field = schema.field(i);
                Arc::new(Column::new(field.name(), i)) as _
            })
            .collect()
    }

    /// Grouping expressions
    pub fn group_expr(&self) -> &PhysicalGroupBy {
        &self.group_by
    }

    /// Aggregate expressions
    pub fn aggr_expr(&self) -> &[Arc<AggregateFunctionExpr>] {
        &self.aggr_expr
    }

    /// FILTER (WHERE clause) expression for each aggregate expression
    pub fn filter_expr(&self) -> &[Option<Arc<dyn PhysicalExpr>>] {
        &self.filter_expr
    }

    /// Input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Input schema before any aggregation
    pub fn input_schema(&self) -> &SchemaRef {
        &self.input_schema
    }

    /// Number of output partitions
    pub fn num_output_partitions(&self) -> usize {
        self.num_output_partitions
    }

    /// Initialize channels and spawn input tasks if not already done.
    /// Returns the receiver for the given output partition.
    fn initialize_and_take_receiver(
        &self,
        partition: usize,
        context: &Arc<TaskContext>,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<RecordBatch>>> {
        let mut state = self.state.lock();

        if matches!(*state, FusedState::NotInitialized) {
            let num_input_partitions = self.input.output_partitioning().partition_count();
            let num_output_partitions = self.num_output_partitions;
            let channel_size = 2;

            // Create M channels (one per output partition), collect senders
            // per_input_senders[input_idx][output_idx] = Sender
            let mut receivers: Vec<
                Option<tokio::sync::mpsc::Receiver<Result<RecordBatch>>>,
            > = Vec::with_capacity(num_output_partitions);
            let mut senders: Vec<tokio::sync::mpsc::Sender<Result<RecordBatch>>> =
                Vec::with_capacity(num_output_partitions);

            for _ in 0..num_output_partitions {
                let (tx, rx) = tokio::sync::mpsc::channel(channel_size);
                receivers.push(Some(rx));
                senders.push(tx);
            }

            // Build per-input sender arrays by cloning the output senders
            let per_input_senders: Vec<Vec<_>> = (0..num_input_partitions)
                .map(|_| senders.to_vec())
                .collect();
            // Drop the original senders so only per_input_senders hold refs
            drop(senders);

            // Spawn N input tasks
            let mut abort_helper = Vec::with_capacity(num_input_partitions);

            for (input_idx, senders) in per_input_senders.into_iter().enumerate() {
                // Create a temporary AggregateExec in Partial mode for this input partition
                let partial_agg = AggregateExec::try_new(
                    AggregateMode::Partial,
                    Arc::clone(&self.group_by),
                    self.aggr_expr.to_vec(),
                    self.filter_expr.to_vec(),
                    Arc::clone(&self.input),
                    Arc::clone(&self.input_schema),
                )?;

                let ctx = Arc::clone(context);
                let schema = Arc::clone(&self.schema);
                let num_group_keys = self.group_by.num_group_exprs();
                let metrics = self.metrics.clone();
                let repartition_time = MetricBuilder::new(&metrics)
                    .subset_time("repartition_time", input_idx);

                let task = SpawnedTask::spawn(async move {
                    let result = run_fused_input_task(
                        partial_agg,
                        input_idx,
                        ctx,
                        schema,
                        num_group_keys,
                        senders,
                        repartition_time,
                    )
                    .await;

                    if let Err(e) = result {
                        debug!(
                            "AggregateRepartitionExec input task {input_idx} failed: {e}"
                        );
                    }
                });

                abort_helper.push(task);
            }

            *state = FusedState::Initialized {
                receivers,
                _abort_helper: abort_helper,
            };
        }

        // Take the receiver for the requested partition
        match &mut *state {
            FusedState::Initialized { receivers, .. } => {
                receivers
                    .get_mut(partition)
                    .and_then(|r| r.take())
                    .ok_or_else(|| {
                        datafusion_common::exec_datafusion_err!(
                            "AggregateRepartitionExec: partition {partition} already consumed or out of range"
                        )
                    })
            }
            FusedState::NotInitialized => {
                internal_err!("FusedState should be initialized at this point")
            }
        }
    }
}

/// Run one input partition's fused aggregate + repartition task.
async fn run_fused_input_task(
    partial_agg: AggregateExec,
    input_partition: usize,
    context: Arc<TaskContext>,
    schema: SchemaRef,
    num_group_keys: usize,
    senders: Vec<tokio::sync::mpsc::Sender<Result<RecordBatch>>>,
    repartition_time: crate::metrics::Time,
) -> Result<()> {
    let num_output_partitions = senders.len();

    // Execute the partial aggregate for this input partition
    let mut stream = partial_agg.execute(input_partition, context)?;

    // Build column expressions for the group keys (Column(0)..Column(K-1))
    let group_key_exprs: Vec<Arc<dyn PhysicalExpr>> = (0..num_group_keys)
        .map(|i| {
            let field = schema.field(i);
            Arc::new(Column::new(field.name(), i)) as Arc<dyn PhysicalExpr>
        })
        .collect();

    let mut partitioner = BatchPartitioner::new_hash_partitioner(
        group_key_exprs,
        num_output_partitions,
        repartition_time,
    );

    while let Some(result) = stream.next().await {
        let batch = result?;

        // Collect partitioned batches synchronously, then send asynchronously
        let mut partitioned_batches: Vec<(usize, RecordBatch)> = Vec::new();
        partitioner.partition(batch, |partition_idx, sub_batch| {
            partitioned_batches.push((partition_idx, sub_batch));
            Ok(())
        })?;

        for (output_partition, sub_batch) in partitioned_batches {
            // If send fails, the output stream was dropped — stop gracefully
            if senders[output_partition].send(Ok(sub_batch)).await.is_err() {
                debug!(
                    "AggregateRepartitionExec: output partition {output_partition} dropped, \
                     stopping input task {input_partition}"
                );
                return Ok(());
            }
        }
    }

    // Senders are dropped here, which closes the channels and signals completion
    Ok(())
}

impl DisplayAs for AggregateRepartitionExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        let input_partition_count = self.input.output_partitioning().partition_count();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let g: Vec<String> = self
                    .group_by
                    .expr()
                    .iter()
                    .map(|(e, alias)| {
                        let expr_str = fmt_sql(e.as_ref()).to_string();
                        if expr_str != *alias {
                            format!("{expr_str} AS {alias}")
                        } else {
                            expr_str
                        }
                    })
                    .collect();
                let a: Vec<String> = self
                    .aggr_expr
                    .iter()
                    .map(|agg| agg.human_display().to_string())
                    .collect();
                write!(
                    f,
                    "AggregateRepartitionExec: mode=Partial, partitions={input_partition_count}->{}",
                    self.num_output_partitions,
                )?;
                if !g.is_empty() {
                    write!(f, ", group_by=[{}]", g.join(", "))?;
                }
                if !a.is_empty() {
                    write!(f, ", aggr=[{}]", a.join(", "))?;
                }
                Ok(())
            }
            DisplayFormatType::TreeRender => {
                let g: Vec<String> = self
                    .group_by
                    .expr()
                    .iter()
                    .map(|(e, alias)| {
                        let expr_str = fmt_sql(e.as_ref()).to_string();
                        if expr_str != *alias {
                            format!("{expr_str} AS {alias}")
                        } else {
                            expr_str
                        }
                    })
                    .collect();
                let a: Vec<String> = self
                    .aggr_expr
                    .iter()
                    .map(|agg| agg.human_display().to_string())
                    .collect();
                writeln!(f, "mode=Partial")?;
                writeln!(
                    f,
                    "partitions={input_partition_count}->{}",
                    self.num_output_partitions
                )?;
                if !g.is_empty() {
                    writeln!(f, "group_by=[{}]", g.join(", "))?;
                }
                if !a.is_empty() {
                    writeln!(f, "aggr=[{}]", a.join(", "))?;
                }
                Ok(())
            }
        }
    }
}

impl ExecutionPlan for AggregateRepartitionExec {
    fn name(&self) -> &'static str {
        "AggregateRepartitionExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn required_input_distribution(&self) -> Vec<crate::Distribution> {
        vec![crate::Distribution::UnspecifiedDistribution]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&dyn PhysicalExpr) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let mut tnr = TreeNodeRecursion::Continue;
        for expr in self.group_by.input_exprs() {
            tnr = tnr.visit_sibling(|| f(expr.as_ref()))?;
        }
        for aggr in self.aggr_expr.iter() {
            for expr in aggr.expressions() {
                tnr = tnr.visit_sibling(|| f(expr.as_ref()))?;
            }
        }
        for filter in self.filter_expr.iter().flatten() {
            tnr = tnr.visit_sibling(|| f(filter.as_ref()))?;
        }
        Ok(tnr)
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(AggregateRepartitionExec::try_new(
            Arc::clone(&self.group_by),
            self.aggr_expr.to_vec(),
            self.filter_expr.to_vec(),
            children.swap_remove(0),
            Arc::clone(&self.input_schema),
            self.num_output_partitions,
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let rx = self.initialize_and_take_receiver(partition, &context)?;
        let schema = Arc::clone(&self.schema);

        // Track the abort helper to keep tasks alive
        let state = Arc::clone(&self.state);

        let stream =
            futures::stream::unfold((rx, state), move |(mut rx, state)| async move {
                let _ = &state; // prevent drop of state (and thus abort of tasks)
                rx.recv().await.map(|batch| (batch, (rx, state)))
            });

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::LowerEqual
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregates::PhysicalGroupBy;
    use crate::test::exec::MockExec;
    use arrow::array::{Float64Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_execution::TaskContext;
    use futures::TryStreamExt;

    #[tokio::test]
    async fn test_basic_aggregate_repartition() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Float64, false),
        ]));

        let batch1 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 1, 3])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            ],
        )?;

        let batch2 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![2, 3, 1, 2])),
                Arc::new(Float64Array::from(vec![50.0, 60.0, 70.0, 80.0])),
            ],
        )?;

        let input = Arc::new(MockExec::new(
            vec![Ok(batch1), Ok(batch2)],
            Arc::clone(&schema),
        ));

        // Group by column "a" with no aggregate functions
        let group_by = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("a", 0)) as _,
            "a".to_string(),
        )]);

        let exec = AggregateRepartitionExec::try_new(
            Arc::new(group_by),
            vec![],
            vec![],
            input,
            Arc::clone(&schema),
            2,
        )?;

        assert_eq!(exec.properties().partitioning.partition_count(), 2);
        assert_eq!(exec.name(), "AggregateRepartitionExec");
        assert_eq!(exec.children().len(), 1);

        let ctx = Arc::new(TaskContext::default());
        let mut total_rows = 0;

        for partition in 0..2 {
            let stream = exec.execute(partition, Arc::clone(&ctx))?;
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            for batch in &batches {
                total_rows += batch.num_rows();
            }
        }

        // 3 distinct keys distributed across 2 output partitions
        assert!(total_rows > 0, "Should have produced some output rows");

        Ok(())
    }
}
