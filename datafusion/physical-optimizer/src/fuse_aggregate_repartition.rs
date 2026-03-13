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

//! [`FuseAggregateRepartition`] optimizer rule fuses adjacent
//! `Partial AggregateExec` + `Hash RepartitionExec` into a single
//! [`AggregateRepartitionExec`].

use std::sync::Arc;

use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_physical_plan::aggregate_repartition::AggregateRepartitionExec;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::{ExecutionPlan, Partitioning};

use crate::PhysicalOptimizerRule;

/// Optimizer rule that fuses a `Partial AggregateExec` followed by a
/// `Hash RepartitionExec` into a single [`AggregateRepartitionExec`].
///
/// This eliminates the intermediate materialization between partial
/// aggregation and hash repartitioning, avoiding redundant hashing
/// and channel overhead.
///
/// # Pattern
///
/// ```text
/// RepartitionExec (Hash, preserve_order=false)
///   └── AggregateExec (mode=Partial)
/// ```
///
/// becomes:
///
/// ```text
/// AggregateRepartitionExec
///   └── Input (from the original AggregateExec)
/// ```
#[derive(Default, Debug)]
pub struct FuseAggregateRepartition {}

impl FuseAggregateRepartition {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl PhysicalOptimizerRule for FuseAggregateRepartition {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !config.optimizer.enable_fuse_aggregate_repartition {
            return Ok(plan);
        }
        plan.transform_up(|plan| {
            // Check if the plan is a RepartitionExec with Hash partitioning
            let Some(repart) = plan.as_any().downcast_ref::<RepartitionExec>() else {
                return Ok(Transformed::no(plan));
            };

            // Only fuse non-order-preserving repartitions
            if repart.preserve_order() {
                return Ok(Transformed::no(plan));
            }

            // Must be Hash partitioning
            let Partitioning::Hash(hash_exprs, num_output_partitions) =
                repart.partitioning().clone()
            else {
                return Ok(Transformed::no(plan));
            };

            // Input must be a Partial AggregateExec
            let Some(partial_agg) =
                repart.input().as_any().downcast_ref::<AggregateExec>()
            else {
                return Ok(Transformed::no(plan));
            };

            if *partial_agg.mode() != AggregateMode::Partial {
                return Ok(Transformed::no(plan));
            }

            // Don't fuse if there are grouping sets (complex case)
            if partial_agg.group_expr().has_grouping_set() {
                return Ok(Transformed::no(plan));
            }

            // Verify that the hash expressions match the group-by output columns.
            // The group keys in the partial aggregate output are Column(0)..Column(K-1).
            let num_group_keys = partial_agg.group_expr().num_group_exprs();
            if hash_exprs.len() != num_group_keys {
                return Ok(Transformed::no(plan));
            }

            // Check that hash expressions are Column references to the group key positions
            for (i, expr) in hash_exprs.iter().enumerate() {
                if let Some(col) =
                    expr.as_any()
                        .downcast_ref::<datafusion_physical_expr::expressions::Column>()
                {
                    if col.index() != i {
                        return Ok(Transformed::no(plan));
                    }
                } else {
                    return Ok(Transformed::no(plan));
                }
            }

            // All checks passed — create the fused operator
            let fused = AggregateRepartitionExec::try_new(
                Arc::clone(&partial_agg.group_expr().clone().into()),
                partial_agg.aggr_expr().to_vec(),
                partial_agg.filter_expr().to_vec(),
                Arc::clone(partial_agg.input()),
                partial_agg.input_schema(),
                num_output_partitions,
            )?;

            Ok(Transformed::yes(Arc::new(fused) as Arc<dyn ExecutionPlan>))
        })
        .data()
    }

    fn name(&self) -> &str {
        "FuseAggregateRepartition"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_plan::aggregates::PhysicalGroupBy;
    use datafusion_physical_plan::empty::EmptyExec;

    fn make_test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Float64, false),
        ]))
    }

    #[test]
    fn test_fuse_partial_agg_with_hash_repartition() -> Result<()> {
        let schema = make_test_schema();
        let input = Arc::new(EmptyExec::new(Arc::clone(&schema)));

        let group_by = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("a", 0)) as _,
            "a".to_string(),
        )]);

        let partial_agg = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            group_by,
            vec![],
            vec![],
            input,
            schema,
        )?);

        let hash_exprs: Vec<Arc<dyn PhysicalExpr>> = vec![Arc::new(Column::new("a", 0))];
        let repartition = Arc::new(RepartitionExec::try_new(
            partial_agg,
            Partitioning::Hash(hash_exprs, 4),
        )?);

        let mut config = ConfigOptions::default();
        config.optimizer.enable_fuse_aggregate_repartition = true;
        let rule = FuseAggregateRepartition::new();
        let result = rule.optimize(repartition, &config)?;

        assert!(
            result
                .as_any()
                .downcast_ref::<AggregateRepartitionExec>()
                .is_some(),
            "Expected AggregateRepartitionExec, got: {}",
            result.name()
        );

        let fused = result
            .as_any()
            .downcast_ref::<AggregateRepartitionExec>()
            .unwrap();
        assert_eq!(fused.num_output_partitions(), 4);
        assert_eq!(fused.properties().partitioning.partition_count(), 4);

        Ok(())
    }

    #[test]
    fn test_no_fuse_final_agg() -> Result<()> {
        let schema = make_test_schema();
        let input = Arc::new(EmptyExec::new(Arc::clone(&schema)));

        let group_by = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("a", 0)) as _,
            "a".to_string(),
        )]);

        let final_agg = Arc::new(AggregateExec::try_new(
            AggregateMode::FinalPartitioned,
            group_by,
            vec![],
            vec![],
            input,
            schema,
        )?);

        let hash_exprs: Vec<Arc<dyn PhysicalExpr>> = vec![Arc::new(Column::new("a", 0))];
        let repartition = Arc::new(RepartitionExec::try_new(
            final_agg,
            Partitioning::Hash(hash_exprs, 4),
        )?);

        let mut config = ConfigOptions::default();
        config.optimizer.enable_fuse_aggregate_repartition = true;
        let rule = FuseAggregateRepartition::new();
        let result = rule.optimize(repartition, &config)?;

        assert!(
            result.as_any().downcast_ref::<RepartitionExec>().is_some(),
            "Expected RepartitionExec to remain, got: {}",
            result.name()
        );

        Ok(())
    }

    #[test]
    fn test_no_fuse_round_robin() -> Result<()> {
        let schema = make_test_schema();
        let input = Arc::new(EmptyExec::new(Arc::clone(&schema)));

        let group_by = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("a", 0)) as _,
            "a".to_string(),
        )]);

        let partial_agg = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            group_by,
            vec![],
            vec![],
            input,
            schema,
        )?);

        let repartition = Arc::new(RepartitionExec::try_new(
            partial_agg,
            Partitioning::RoundRobinBatch(4),
        )?);

        let mut config = ConfigOptions::default();
        config.optimizer.enable_fuse_aggregate_repartition = true;
        let rule = FuseAggregateRepartition::new();
        let result = rule.optimize(repartition, &config)?;

        assert!(
            result.as_any().downcast_ref::<RepartitionExec>().is_some(),
            "Expected RepartitionExec to remain"
        );

        Ok(())
    }
}
