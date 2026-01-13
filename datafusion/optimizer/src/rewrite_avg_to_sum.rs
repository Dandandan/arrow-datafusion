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

use datafusion_common::{
    arrow::datatypes::DataType, Column, DataFusionError, Result, TableReference,
};
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
                            let (_, original_field) =
                                schema.qualified_field(group_expr_len + i);
                            let target_type = original_field.data_type();

                            let sum_expr = Expr::Column(Column::new(
                                None::<TableReference>,
                                sum_field.name().clone(),
                            ));
                            let count_expr = Expr::Column(Column::new(
                                None::<TableReference>,
                                count_field.name().clone(),
                            ));

                            let count_expr = if *target_type == DataType::Float64 {
                                count_expr.cast_to(
                                    &DataType::Float64,
                                    new_agg.schema(),
                                )?
                            } else {
                                count_expr
                            };

                            let div_expr = Expr::BinaryExpr(BinaryExpr::new(
                                Box::new(sum_expr),
                                Operator::Divide,
                                Box::new(count_expr),
                            ));

                            let final_expr = if &div_expr
                                .get_type(new_agg.schema())?
                                != target_type
                            {
                                div_expr.cast_to(target_type, new_agg.schema())?
                            } else {
                                div_expr
                            };

                            proj_exprs.push(
                                final_expr.alias(original_field.name().clone()),
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
