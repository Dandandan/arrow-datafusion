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

//! Optimizer rule to rewrite AVG(col) to SUM(col) / COUNT(col)
use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion_common::{Column, DataFusionError, Result, TableReference};
use datafusion_expr::{
    expr::{AggregateFunction, AggregateFunctionParams},
    logical_plan::{Aggregate, LogicalPlan, Projection},
    Expr, Operator,
};
use datafusion_expr::{ExprFunctionExt, ExprSchemable};
use datafusion_functions_aggregate::{count::count_udaf, sum::sum_udaf};

use crate::optimizer::ApplyOrder;
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::Transformed;
use datafusion_expr::expr::BinaryExpr;

/// Optimizer rule to rewrite AVG(col) to SUM(col) / COUNT(col)
#[derive(Default, Debug)]
pub struct RewriteAvgToSum;

impl RewriteAvgToSum {
    /// Creates a new RewriteAvgToSum optimizer rule.
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for RewriteAvgToSum {
    fn name(&self) -> &str {
        "rewrite_avg_to_sum"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>, DataFusionError> {
        match plan {
            LogicalPlan::Aggregate(Aggregate {
                input,
                aggr_expr,
                schema,
                group_expr,
                ..
            }) => {
                let mut avg_parts = vec![];
                let mut has_avg = false;
                for agg_expr in &aggr_expr {
                    let mut unaliased_agg_expr = agg_expr;
                    while let Expr::Alias(alias) = unaliased_agg_expr {
                        unaliased_agg_expr = &alias.expr;
                    }

                    if let Expr::AggregateFunction(agg_func) = unaliased_agg_expr {
                        if agg_func.func.name() == "avg" {
                            if !agg_func.params.order_by.is_empty() {
                                return Ok(Transformed::no(LogicalPlan::Aggregate(
                                    Aggregate::try_new(input, group_expr, aggr_expr)?,
                                )));
                            }
                            has_avg = true;
                            let sum = Expr::AggregateFunction(AggregateFunction {
                                func: sum_udaf(),
                                params: AggregateFunctionParams {
                                    args: vec![agg_func.params.args[0].clone()],
                                    distinct: agg_func.params.distinct,
                                    filter: agg_func.params.filter.clone(),
                                    order_by: vec![],
                                    null_treatment: None,
                                },
                            });

                            let count = Expr::AggregateFunction(AggregateFunction {
                                func: count_udaf(),
                                params: AggregateFunctionParams {
                                    args: vec![agg_func.params.args[0].clone()],
                                    distinct: agg_func.params.distinct,
                                    filter: agg_func.params.filter.clone(),
                                    order_by: vec![],
                                    null_treatment: None,
                                },
                            });

                            avg_parts.push(Some((sum, count)));
                        } else {
                            avg_parts.push(None);
                        }
                    } else {
                        avg_parts.push(None);
                    }
                }

                if !has_avg {
                    return Ok(Transformed::no(LogicalPlan::Aggregate(
                        Aggregate::try_new(input, group_expr, aggr_expr)?,
                    )));
                }

                let mut new_aggr_expr = vec![];
                let mut aggr_map: HashMap<Expr, usize> = HashMap::new();
                let mut aggr_indices = vec![];

                for (i, agg_expr) in aggr_expr.iter().enumerate() {
                    if let Some((sum_expr, count_expr)) = &avg_parts[i] {
                        let sum_idx =
                            *aggr_map.entry(sum_expr.clone()).or_insert_with(|| {
                                new_aggr_expr.push(sum_expr.clone());
                                new_aggr_expr.len() - 1
                            });
                        let count_idx =
                            *aggr_map.entry(count_expr.clone()).or_insert_with(|| {
                                new_aggr_expr.push(count_expr.clone());
                                new_aggr_expr.len() - 1
                            });
                        aggr_indices.push(Ok((sum_idx, count_idx)));
                    } else {
                        let idx =
                            *aggr_map.entry(agg_expr.clone()).or_insert_with(|| {
                                new_aggr_expr.push(agg_expr.clone());
                                new_aggr_expr.len() - 1
                            });
                        aggr_indices.push(Err(idx));
                    }
                }

                let new_agg = LogicalPlan::Aggregate(Aggregate::try_new(
                    input.clone(),
                    group_expr.clone(),
                    new_aggr_expr,
                )?);
                let new_schema = new_agg.schema();

                let group_expr_len = group_expr.len();
                let mut proj_exprs = group_expr
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let (qualifier, field) = new_schema.qualified_field(i);
                        Ok(Expr::Column(Column::new(
                            qualifier.cloned(),
                            field.name().clone(),
                        )))
                    })
                    .collect::<Result<Vec<_>>>()?;

                for (i, indices) in aggr_indices.iter().enumerate() {
                    match indices {
                        Ok((sum_idx, count_idx)) => {
                            let (_, sum_field) =
                                new_schema.qualified_field(group_expr_len + sum_idx);
                            let (_, count_field) =
                                new_schema.qualified_field(group_expr_len + count_idx);

                            let sum_expr = Expr::Column(Column::new(
                                None::<TableReference>,
                                sum_field.name().clone(),
                            ))
                            .cast_to(&DataType::Float64, new_agg.schema())?;
                            let count_expr = Expr::Column(Column::new(
                                None::<TableReference>,
                                count_field.name().clone(),
                            ))
                            .cast_to(&DataType::Float64, new_agg.schema())?;
                            let (_, original_field) =
                                schema.qualified_field(group_expr_len + i);
                            proj_exprs.push(
                                Expr::BinaryExpr(BinaryExpr::new(
                                    Box::new(sum_expr),
                                    Operator::Divide,
                                    Box::new(count_expr),
                                ))
                                .alias(original_field.name().clone()),
                            );
                        }
                        Err(idx) => {
                            let (qualifier, field) =
                                new_schema.qualified_field(group_expr_len + idx);
                            proj_exprs.push(Expr::Column(Column::new(
                                qualifier.cloned(),
                                field.name().clone(),
                            )));
                        }
                    }
                }

                let proj = Projection::try_new(proj_exprs, Arc::new(new_agg))?;
                Ok(Transformed::yes(LogicalPlan::Projection(proj)))
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::OptimizerContext;
    use crate::test::*;
    use datafusion_expr::{
        col, lit,
        logical_plan::{builder::LogicalPlanBuilder, table_scan},
        ExprFunctionExt,
    };
    use datafusion_functions_aggregate::expr_fn::{avg, count, sum};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::Result;

    fn test_table_scan_with_nulls() -> Result<LogicalPlan> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
        ]);
        table_scan(Some("test"), &schema, None)?.build()
    }

    #[test]
    fn test_single_avg() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![avg(col("b"))])?
            .build()?;
        let rule = RewriteAvgToSum::new();
        let optimizer_config = OptimizerContext::new();
        let optimized_plan = rule.rewrite(plan, &optimizer_config)?.data;
        let expected = "Projection: test.a, CAST(SUM(test.b) AS Float64) / CAST(COUNT(test.b) AS Float64) AS AVG(test.b)\
            \n  Aggregate: groupBy=[[test.a]], aggr=[[SUM(test.b), COUNT(test.b)]]\
            \n    TableScan: test [a:UInt32, b:UInt32, c:UInt32]";
        assert_eq!(expected, format!("{:?}", optimized_plan));
        Ok(())
    }

    #[test]
    fn test_multiple_avg() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![avg(col("b")), avg(col("c"))])?
            .build()?;
        let rule = RewriteAvgToSum::new();
        let optimizer_config = OptimizerContext::new();
        let optimized_plan = rule.rewrite(plan, &optimizer_config)?.data;
        let expected = "Projection: test.a, CAST(SUM(test.b) AS Float64) / CAST(COUNT(test.b) AS Float64) AS AVG(test.b), CAST(SUM(test.c) AS Float64) / CAST(COUNT(test.c) AS Float64) AS AVG(test.c)\
            \n  Aggregate: groupBy=[[test.a]], aggr=[[SUM(test.b), COUNT(test.b), SUM(test.c), COUNT(test.c)]]\
            \n    TableScan: test [a:UInt32, b:UInt32, c:UInt32]";
        assert_eq!(expected, format!("{:?}", optimized_plan));
        Ok(())
    }

    #[test]
    fn test_multiple_avg_same_col() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![avg(col("b")), avg(col("b"))])?
            .build()?;
        let rule = RewriteAvgToSum::new();
        let optimizer_config = OptimizerContext::new();
        let optimized_plan = rule.rewrite(plan, &optimizer_config)?.data;
        let expected = "Projection: test.a, CAST(SUM(test.b) AS Float64) / CAST(COUNT(test.b) AS Float64) AS AVG(test.b), CAST(SUM(test.b) AS Float64) / CAST(COUNT(test.b) AS Float64) AS AVG(test.b)\
            \n  Aggregate: groupBy=[[test.a]], aggr=[[SUM(test.b), COUNT(test.b)]]\
            \n    TableScan: test [a:UInt32, b:UInt32, c:UInt32]";
        assert_eq!(expected, format!("{:?}", optimized_plan));
        Ok(())
    }

    #[test]
    fn test_avg_with_nulls() -> Result<()> {
        let table_scan = test_table_scan_with_nulls()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![avg(col("b"))])?
            .build()?;
        let rule = RewriteAvgToSum::new();
        let optimizer_config = OptimizerContext::new();
        let optimized_plan = rule.rewrite(plan, &optimizer_config)?.data;
        let expected = "Projection: test.a, CAST(SUM(test.b) AS Float64) / CAST(COUNT(test.b) AS Float64) AS AVG(test.b)\
            \n  Aggregate: groupBy=[[test.a]], aggr=[[SUM(test.b), COUNT(test.b)]]\
            \n    TableScan: test [a:Int32, b:Int32;N]";
        assert_eq!(expected, format!("{:?}", optimized_plan));
        Ok(())
    }

    #[test]
    fn test_avg_distinct() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![avg(col("b")).distinct().build()?])?
            .build()?;
        let rule = RewriteAvgToSum::new();
        let optimizer_config = OptimizerContext::new();
        let optimized_plan = rule.rewrite(plan, &optimizer_config)?.data;
        let expected = "Projection: test.a, CAST(SUM(DISTINCT test.b) AS Float64) / CAST(COUNT(DISTINCT test.b) AS Float64) AS AVG(DISTINCT test.b)\
            \n  Aggregate: groupBy=[[test.a]], aggr=[[SUM(DISTINCT test.b), COUNT(DISTINCT test.b)]]\
            \n    TableScan: test [a:UInt32, b:UInt32, c:UInt32]";
        assert_eq!(expected, format!("{:?}", optimized_plan));
        Ok(())
    }

    #[test]
    fn test_avg_with_filter() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![avg(col("b")).filter(col("c").gt(lit(1))).build()?],
            )?
            .build()?;
        let rule = RewriteAvgToSum::new();
        let optimizer_config = OptimizerContext::new();
        let optimized_plan = rule.rewrite(plan, &optimizer_config)?.data;
        let expected = "Projection: test.a, CAST(SUM(test.b) FILTER (WHERE test.c > Int32(1)) AS Float64) / CAST(COUNT(test.b) FILTER (WHERE test.c > Int32(1)) AS Float64) AS AVG(test.b) FILTER (WHERE test.c > Int32(1))\
            \n  Aggregate: groupBy=[[test.a]], aggr=[[SUM(test.b) FILTER (WHERE test.c > Int32(1)), COUNT(test.b) FILTER (WHERE test.c > Int32(1))]]\
            \n    TableScan: test [a:UInt32, b:UInt32, c:UInt32]";
        assert_eq!(expected, format!("{:?}", optimized_plan));
        Ok(())
    }

    #[test]
    fn test_avg_with_alias() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(vec![col("a")], vec![avg(col("b")).alias("avg_b")])?
            .build()?;
        let rule = RewriteAvgToSum::new();
        let optimizer_config = OptimizerContext::new();
        let optimized_plan = rule.rewrite(plan, &optimizer_config)?.data;
        let expected = "Projection: test.a, CAST(SUM(test.b) AS Float64) / CAST(COUNT(test.b) AS Float64) AS avg_b\
            \n  Aggregate: groupBy=[[test.a]], aggr=[[SUM(test.b), COUNT(test.b)]]\
            \n    TableScan: test [a:UInt32, b:UInt32, c:UInt32]";
        assert_eq!(expected, format!("{:?}", optimized_plan));
        Ok(())
    }

    #[test]
    fn test_no_avg() -> Result<()> {
        let table_scan = test_table_scan()?;
        let plan = LogicalPlanBuilder::from(table_scan)
            .aggregate(
                vec![col("a")],
                vec![sum(col("b")), count(lit(1u8))],
            )?
            .build()?;
        let rule = RewriteAvgToSum::new();
        let optimizer_config = OptimizerContext::new();
        let optimized_plan = rule.rewrite(plan.clone(), &optimizer_config)?;
        assert!(!optimized_plan.transformed);
        Ok(())
    }
}
