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

//! TopK late materialization via join.
//!
//! For wide tables (e.g. 100+ columns), queries like:
//! ```sql
//! SELECT * FROM hits WHERE url LIKE '%google%' ORDER BY event_time LIMIT 10
//! ```
//! read ALL columns through TopK even though only `url` (filter) and
//! `event_time` (sort) are needed to determine the top-10 rows.
//!
//! This rule transforms the plan so a lightweight first pass reads only
//! filter+sort columns, then a HashJoin with the full scan fetches the
//! remaining columns only for the matching rows. The existing dynamic
//! filter pushdown mechanism pushes the TopK threshold through the join
//! into the right-side scan, skipping parquet row groups outside the
//! threshold range.

use std::collections::HashSet;
use std::sync::Arc;

use crate::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{
    Transformed, TransformedResult, TreeNode, TreeNodeRecursion,
};
use datafusion_common::{JoinType, NullEquality, Result};
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::projection::project_ordering;
use datafusion_physical_expr::utils::{collect_columns, reassign_expr_columns};
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::ExecutionPlan;

/// Minimum ratio of total columns to required columns to trigger the
/// optimization. If the table has fewer than this multiple of the
/// columns needed for sort+filter, the rewrite is unlikely to help.
const MIN_COLUMN_RATIO: usize = 2;

/// Minimum total columns in the table for the optimization to fire.
const MIN_TOTAL_COLUMNS: usize = 8;

/// Optimizer rule that rewrites TopK over wide file scans into a
/// two-pass plan: a lightweight inner TopK (reading only sort+filter
/// columns) joined back to the full scan.
#[derive(Debug)]
pub struct TopKJoin {}

impl TopKJoin {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for TopKJoin {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysicalOptimizerRule for TopKJoin {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !config.optimizer.enable_topk_join {
            return Ok(plan);
        }
        plan.transform_down(|plan| {
            Ok(
                if let Some(transformed) = try_transform_topk_join(&plan)? {
                    // Use Jump to skip recursing into children of the new plan,
                    // preventing infinite re-application on the inner SortExec.
                    Transformed::new(transformed, true, TreeNodeRecursion::Jump)
                } else {
                    Transformed::no(plan)
                },
            )
        })
        .data()
    }

    fn name(&self) -> &str {
        "TopKJoin"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Intermediate struct holding the components of a TopK pattern we
/// detected in the plan tree.
struct TopKPattern {
    /// The SortExec with fetch (the TopK node)
    sort: SortExec,
    /// The FilterExec below the sort
    filter: FilterExec,
    /// The DataSourceExec (file-backed scan)
    data_source: DataSourceExec,
}

/// Try to detect and transform a TopK-over-wide-scan pattern.
fn try_transform_topk_join(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(pattern) = detect_topk_pattern(plan) else {
        return Ok(None);
    };

    let file_scan_config = pattern
        .data_source
        .data_source()
        .as_any()
        .downcast_ref::<FileScanConfig>()
        .expect("detect_topk_pattern guarantees FileScanConfig");

    // Check heuristic: is the table wide enough to benefit?
    let table_schema = file_scan_config
        .file_source()
        .table_schema()
        .table_schema();
    let total_columns = table_schema.fields().len();

    // Collect columns needed for sort + filter
    let mut required_col_names: HashSet<String> = HashSet::new();
    for expr in pattern.sort.expr().iter() {
        for col in collect_columns(&expr.expr) {
            required_col_names.insert(col.name().to_string());
        }
    }
    for col in collect_columns(pattern.filter.predicate()) {
        required_col_names.insert(col.name().to_string());
    }

    let required_count = required_col_names.len();

    if total_columns < MIN_TOTAL_COLUMNS
        || total_columns < required_count * MIN_COLUMN_RATIO
    {
        return Ok(None);
    }

    // Map required column names to indices in the table schema.
    // Iterating by index produces sorted indices naturally.
    let required_indices: Vec<usize> = table_schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| required_col_names.contains(field.name()))
        .map(|(idx, _)| idx)
        .collect();

    if required_indices.len() != required_count {
        return Ok(None);
    }

    // Build the lightweight scan with only the required columns
    let builder: FileScanConfigBuilder = file_scan_config.clone().into();
    let Ok(builder) = builder.with_projection_indices(Some(required_indices.to_vec()))
    else {
        return Ok(None);
    };
    let lightweight_scan = DataSourceExec::new(Arc::new(builder.build()));
    let lightweight_schema = lightweight_scan.schema();

    // Remap filter predicate for lightweight schema
    let Ok(remapped_predicate) = reassign_expr_columns(
        Arc::clone(pattern.filter.predicate()),
        &lightweight_schema,
    ) else {
        return Ok(None);
    };

    // Build inner filter
    let inner_filter =
        FilterExec::try_new(remapped_predicate, Arc::new(lightweight_scan))?;

    // Remap sort expressions for lightweight schema
    let Some(remapped_ordering) =
        project_ordering(pattern.sort.expr(), &lightweight_schema)
    else {
        return Ok(None);
    };

    // Build inner SortExec with fetch (the lightweight TopK)
    let inner_sort = SortExec::new(remapped_ordering, Arc::new(inner_filter))
        .with_fetch(pattern.sort.fetch());

    // Build the right side: full scan with filter re-applied
    let full_scan: Arc<dyn ExecutionPlan> =
        Arc::new(DataSourceExec::new(Arc::new(file_scan_config.clone())));
    let right_filter =
        FilterExec::try_new(Arc::clone(pattern.filter.predicate()), full_scan)?;

    // Build join keys: sort columns (left lightweight schema -> right full schema)
    let sort_cols: Vec<Column> = pattern
        .sort
        .expr()
        .iter()
        .flat_map(|e| collect_columns(&e.expr))
        .collect();
    let mut join_on: Vec<(PhysicalExprRef, PhysicalExprRef)> = Vec::new();
    for col in &sort_cols {
        let Some((left_idx, _)) = lightweight_schema.column_with_name(col.name())
        else {
            return Ok(None);
        };
        let left_col: PhysicalExprRef = Arc::new(Column::new(
            lightweight_schema.field(left_idx).name(),
            left_idx,
        ));
        let right_col: PhysicalExprRef =
            Arc::new(Column::new(col.name(), col.index()));
        join_on.push((left_col, right_col));
    }

    // Build HashJoinExec: inner TopK (left) join full filtered scan (right)
    let hash_join = HashJoinExec::try_new(
        Arc::new(inner_sort),
        Arc::new(right_filter),
        join_on,
        None, // no join filter
        &JoinType::Inner,
        None, // no projection
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
        false, // not null-aware
    )?;

    // Build projection to select only right-side columns
    // Join schema is [lightweight_cols..., full_cols...]
    let left_col_count = lightweight_schema.fields().len();
    let full_schema = pattern.data_source.schema();
    let mut proj_exprs: Vec<(PhysicalExprRef, String)> = Vec::new();
    for (i, field) in full_schema.fields().iter().enumerate() {
        let join_idx = left_col_count + i;
        let col: PhysicalExprRef = Arc::new(Column::new(
            hash_join.schema().field(join_idx).name(),
            join_idx,
        ));
        proj_exprs.push((col, field.name().clone()));
    }
    let projection = ProjectionExec::try_new(proj_exprs, Arc::new(hash_join))?;

    // Build outer SortExec with fetch for correctness (handles duplicates)
    let outer_sort = SortExec::new(pattern.sort.expr().clone(), Arc::new(projection))
        .with_fetch(pattern.sort.fetch());

    Ok(Some(Arc::new(outer_sort)))
}

/// Walk down through single-child "pass-through" nodes (RepartitionExec,
/// CoalescePartitionsExec, CoalesceBatchesExec, etc.) to find the first
/// FilterExec or DataSourceExec.
fn walk_down_single_child(plan: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let mut current = Arc::clone(plan);
    loop {
        let children = current.children();
        if children.len() != 1 {
            return current;
        }
        if current.as_any().downcast_ref::<FilterExec>().is_some()
            || current.as_any().downcast_ref::<DataSourceExec>().is_some()
        {
            return current;
        }
        current = Arc::clone(children[0]);
    }
}

/// Detect the pattern: SortExec(fetch) -> ... -> FilterExec -> ... -> DataSourceExec
fn detect_topk_pattern(plan: &Arc<dyn ExecutionPlan>) -> Option<TopKPattern> {
    let sort = plan.as_any().downcast_ref::<SortExec>()?;
    sort.fetch()?;

    // Walk down through pass-through nodes to find FilterExec
    let filter_candidate = walk_down_single_child(sort.input());
    let filter = filter_candidate
        .as_any()
        .downcast_ref::<FilterExec>()?;

    // Walk down from filter's child to find DataSourceExec
    let ds_candidate = walk_down_single_child(filter.input());
    let data_source = ds_candidate
        .as_any()
        .downcast_ref::<DataSourceExec>()?;

    // Must be file-backed
    data_source
        .data_source()
        .as_any()
        .downcast_ref::<FileScanConfig>()?;

    Some(TopKPattern {
        sort: sort.clone(),
        filter: filter.clone(),
        data_source: data_source.clone(),
    })
}
