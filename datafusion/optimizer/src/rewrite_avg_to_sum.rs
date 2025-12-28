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
use std::sync::Arc;

use datafusion_common::{Column, DataFusionError, Result, TableReference};
use datafusion_expr::{
    expr::{AggregateFunction, AggregateFunctionParams},
    logical_plan::{Aggregate, LogicalPlan, Projection},
    Expr, Operator,
};
use datafusion_expr::ExprFunctionExt;

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
                let mut avg_indices = vec![];
                for (i, expr) in aggr_expr.iter().enumerate() {
                    if let Expr::AggregateFunction(AggregateFunction { func, .. }) = expr {
                        if func.name() == "avg" {
                            avg_indices.push(i);
                        }
                    }
                }

                if avg_indices.is_empty() {
                    return Ok(Transformed::no(LogicalPlan::Aggregate(
                        Aggregate::try_new(input, group_expr, aggr_expr)?,
                    )));
                }

                let mut new_aggr_expr = aggr_expr.clone();
                let mut count_indices = vec![];
                for i in &avg_indices {
                    if let Expr::AggregateFunction(AggregateFunction {
                        params:
                            AggregateFunctionParams {
                                args,
                                distinct,
                                filter,
                                order_by,
                                null_treatment,
                            },
                        ..
                    }) = &aggr_expr[*i]
                    {
                        let sum = if *distinct {
                            datafusion_functions_aggregate::expr_fn::sum(
                                args[0].clone(),
                            )
                            .distinct()
                            .build()?
                        } else {
                            datafusion_functions_aggregate::expr_fn::sum(
                                args[0].clone(),
                            )
                        };
                        new_aggr_expr[*i] = sum;

                        let count_expr = if *distinct {
                            datafusion_functions_aggregate::expr_fn::count(
                                args[0].clone(),
                            )
                            .distinct()
                            .build()?
                        } else {
                            datafusion_functions_aggregate::expr_fn::count(
                                args[0].clone(),
                            )
                        };

                        let mut found = false;
                        for (j, expr) in new_aggr_expr.iter().enumerate() {
                            if *expr == count_expr {
                                count_indices.push(j);
                                found = true;
                                break;
                            }
                        }

                        if !found {
                            new_aggr_expr.push(count_expr);
                            count_indices.push(new_aggr_expr.len() - 1);
                        }
                    }
                }

                let new_agg = LogicalPlan::Aggregate(Aggregate::try_new(
                    input.clone(),
                    group_expr.clone(),
                    new_aggr_expr.clone(),
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

                let mut avg_count = 0;
                for i in 0..aggr_expr.len() {
                    if avg_indices.contains(&i) {
                        let (_, sum_field) =
                            new_schema.qualified_field(group_expr_len + i);
                        let (_, count_field) = new_schema
                            .qualified_field(group_expr_len + count_indices[avg_count]);

                        let sum_expr = Expr::Column(Column::new(
                            None::<TableReference>,
                            sum_field.name().clone(),
                        ));
                        let count_expr = Expr::Column(Column::new(
                            None::<TableReference>,
                            count_field.name().clone(),
                        ));
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
                        avg_count += 1;
                    } else {
                        let (qualifier, field) =
                            new_schema.qualified_field(group_expr_len + i);
                        proj_exprs.push(Expr::Column(Column::new(
                            qualifier.cloned(),
                            field.name().clone(),
                        )));
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
        let expected = "Projection: test.a, SUM(test.b) / COUNT(test.b) AS AVG(test.b)\
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
        let expected = "Projection: test.a, SUM(test.b) / COUNT(test.b) AS AVG(test.b), SUM(test.c) / COUNT(test.c) AS AVG(test.c)\
            \n  Aggregate: groupBy=[[test.a]], aggr=[[SUM(test.b), SUM(test.c), COUNT(test.b), COUNT(test.c)]]\
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
        let expected = "Projection: test.a, SUM(test.b) / COUNT(test.b) AS AVG(test.b)\
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
        let expected = "Projection: test.a, SUM(DISTINCT test.b) / COUNT(DISTINCT test.b) AS AVG(DISTINCT test.b)\
            \n  Aggregate: groupBy=[[test.a]], aggr=[[SUM(DISTINCT test.b), COUNT(DISTINCT test.b)]]\
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
