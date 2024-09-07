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

//! [`FilterNullJoinKeys`] adds filters to join inputs when input isn't nullable

use crate::optimizer::ApplyOrder;
use crate::push_down_filter::on_lr_is_preserved;
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::Transformed;
use datafusion_common::Result;
use datafusion_expr::utils::conjunction;
use datafusion_expr::{logical_plan::Filter, Expr, ExprSchemable, LogicalPlan};
use std::sync::Arc;

/// The FilterNullJoinKeys rule will identify  joins with equi-join conditions
/// where the join key is nullable and then insert an `IsNotNull` filter on the nullable side since null values
/// can never match.
#[derive(Default)]
pub struct FilterNullJoinKeys {}

impl OptimizerRule for FilterNullJoinKeys {
    fn supports_rewrite(&self) -> bool {
        true
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if !config.options().optimizer.filter_null_join_keys {
            return Ok(Transformed::no(plan));
        }
        match plan {
            LogicalPlan::Join(mut join) if !join.on.is_empty() && !join.null_equals_null => {
                let (left_preserved, right_preserved) =
                    on_lr_is_preserved(join.join_type);

                let left_schema = join.left.schema();
                let right_schema = join.right.schema();

                let mut left_filters = vec![];
                let mut right_filters = vec![];

                for (l, r) in &join.on {
                    if left_preserved && l.nullable(left_schema)? {
                        left_filters.push(l.clone());
                    }

                    if right_preserved && r.nullable(right_schema)? {
                        right_filters.push(r.clone());
                    }
                }

                if !left_filters.is_empty() {
                    let predicate = create_not_null_predicate(left_filters);
                    join.left = Arc::new(LogicalPlan::Filter(Filter::try_new(
                        predicate, join.left,
                    )?));
                }
                if !right_filters.is_empty() {
                    let predicate = create_not_null_predicate(right_filters);
                    join.right = Arc::new(LogicalPlan::Filter(Filter::try_new(
                        predicate, join.right,
                    )?));
                }
                Ok(Transformed::yes(LogicalPlan::Join(join)))
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
    fn name(&self) -> &str {
        "filter_null_join_keys"
    }
}

fn create_not_null_predicate(filters: Vec<Expr>) -> Expr {
    let not_null_exprs: Vec<Expr> = filters
        .into_iter()
        .map(|c| Expr::IsNotNull(Box::new(c)))
        .collect();

    // directly unwrap since it should always have a value
    conjunction(not_null_exprs).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::assert_optimized_plan_eq;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::Column;
    use datafusion_expr::logical_plan::table_scan;
    use datafusion_expr::{col, lit, JoinType, LogicalPlanBuilder};

    fn assert_optimized_plan_equal(plan: LogicalPlan, expected: &str) -> Result<()> {
        assert_optimized_plan_eq(Arc::new(FilterNullJoinKeys {}), plan, expected)
    }

    #[test]
    fn left_nullable() -> Result<()> {
        let (t1, t2) = test_tables()?;
        let plan = build_plan(t1, t2, "t1.optional_id", "t2.id", JoinType::Inner)?;
        let expected = "Inner Join: t1.optional_id = t2.id\
        \n  Filter: t1.optional_id IS NOT NULL\
        \n    TableScan: t1\
        \n  TableScan: t2";
        assert_optimized_plan_equal(plan, expected)
    }

    #[test]
    fn left_nullable_left_join() -> Result<()> {
        let (t1, t2) = test_tables()?;
        let plan = build_plan(t1, t2, "t1.optional_id", "t2.id", JoinType::Left)?;
        let expected = "Left Join: t1.optional_id = t2.id\
        \n  TableScan: t1\
        \n  TableScan: t2";
        assert_optimized_plan_equal(plan, expected)
    }

    #[test]
    fn left_nullable_left_join_reordered() -> Result<()> {
        let (t_left, t_right) = test_tables()?;
        // Note: order of tables is reversed
        let plan =
            build_plan(t_right, t_left, "t2.id", "t1.optional_id", JoinType::Left)?;
        let expected = "Left Join: t2.id = t1.optional_id\
        \n  TableScan: t2\
        \n  Filter: t1.optional_id IS NOT NULL\
        \n    TableScan: t1";
        assert_optimized_plan_equal(plan, expected)
    }

    #[test]
    fn left_nullable_on_condition_reversed() -> Result<()> {
        let (t1, t2) = test_tables()?;
        let plan = build_plan(t1, t2, "t2.id", "t1.optional_id", JoinType::Inner)?;
        let expected = "Inner Join: t1.optional_id = t2.id\
        \n  Filter: t1.optional_id IS NOT NULL\
        \n    TableScan: t1\
        \n  TableScan: t2";
        assert_optimized_plan_equal(plan, expected)
    }

    #[test]
    fn nested_join_multiple_filter_expr() -> Result<()> {
        let (t1, t2) = test_tables()?;
        let plan = build_plan(t1, t2, "t1.optional_id", "t2.id", JoinType::Inner)?;
        let schema = Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("t1_id", DataType::UInt32, true),
            Field::new("t2_id", DataType::UInt32, true),
        ]);
        let t3 = table_scan(Some("t3"), &schema, None)?.build()?;
        let plan = LogicalPlanBuilder::from(t3)
            .join(
                plan,
                JoinType::Inner,
                (
                    vec![
                        Column::from_qualified_name("t3.t1_id"),
                        Column::from_qualified_name("t3.t2_id"),
                    ],
                    vec![
                        Column::from_qualified_name("t1.id"),
                        Column::from_qualified_name("t2.id"),
                    ],
                ),
                None,
            )?
            .build()?;
        let expected = "Inner Join: t3.t1_id = t1.id, t3.t2_id = t2.id\
        \n  Filter: t3.t1_id IS NOT NULL AND t3.t2_id IS NOT NULL\
        \n    TableScan: t3\
        \n  Inner Join: t1.optional_id = t2.id\
        \n    Filter: t1.optional_id IS NOT NULL\
        \n      TableScan: t1\
        \n    TableScan: t2";
        assert_optimized_plan_equal(plan, expected)
    }

    #[test]
    fn left_nullable_expr_key() -> Result<()> {
        let (t1, t2) = test_tables()?;
        let plan = LogicalPlanBuilder::from(t1)
            .join_with_expr_keys(
                t2,
                JoinType::Inner,
                (
                    vec![col("t1.optional_id") + lit(1u32)],
                    vec![col("t2.id") + lit(1u32)],
                ),
                None,
            )?
            .build()?;
        let expected = "Inner Join: t1.optional_id + UInt32(1) = t2.id + UInt32(1)\
        \n  Filter: t1.optional_id + UInt32(1) IS NOT NULL\
        \n    TableScan: t1\
        \n  TableScan: t2";
        assert_optimized_plan_equal(plan, expected)
    }

    #[test]
    fn right_nullable_expr_key() -> Result<()> {
        let (t1, t2) = test_tables()?;
        let plan = LogicalPlanBuilder::from(t1)
            .join_with_expr_keys(
                t2,
                JoinType::Inner,
                (
                    vec![col("t1.id") + lit(1u32)],
                    vec![col("t2.optional_id") + lit(1u32)],
                ),
                None,
            )?
            .build()?;
        let expected = "Inner Join: t1.id + UInt32(1) = t2.optional_id + UInt32(1)\
        \n  TableScan: t1\
        \n  Filter: t2.optional_id + UInt32(1) IS NOT NULL\
        \n    TableScan: t2";
        assert_optimized_plan_equal(plan, expected)
    }

    #[test]
    fn both_side_nullable_expr_key() -> Result<()> {
        let (t1, t2) = test_tables()?;
        let plan = LogicalPlanBuilder::from(t1)
            .join_with_expr_keys(
                t2,
                JoinType::Inner,
                (
                    vec![col("t1.optional_id") + lit(1u32)],
                    vec![col("t2.optional_id") + lit(1u32)],
                ),
                None,
            )?
            .build()?;
        let expected =
            "Inner Join: t1.optional_id + UInt32(1) = t2.optional_id + UInt32(1)\
        \n  Filter: t1.optional_id + UInt32(1) IS NOT NULL\
        \n    TableScan: t1\
        \n  Filter: t2.optional_id + UInt32(1) IS NOT NULL\
        \n    TableScan: t2";
        assert_optimized_plan_equal(plan, expected)
    }

    fn build_plan(
        left_table: LogicalPlan,
        right_table: LogicalPlan,
        left_key: &str,
        right_key: &str,
        join_type: JoinType,
    ) -> Result<LogicalPlan> {
        LogicalPlanBuilder::from(left_table)
            .join(
                right_table,
                join_type,
                (
                    vec![Column::from_qualified_name(left_key)],
                    vec![Column::from_qualified_name(right_key)],
                ),
                None,
            )?
            .build()
    }

    fn test_tables() -> Result<(LogicalPlan, LogicalPlan)> {
        let schema = Schema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("optional_id", DataType::UInt32, true),
        ]);
        let t1 = table_scan(Some("t1"), &schema, None)?.build()?;
        let t2 = table_scan(Some("t2"), &schema, None)?.build()?;
        Ok((t1, t2))
    }
}
