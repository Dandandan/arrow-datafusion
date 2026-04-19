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

//! [`InListPruningExpr`]: native IN-list pruning over min/max statistics.
//!
//! For a predicate `col IN (v1, v2, ..., vN)`, expanding into an OR of
//! equalities blows up the pruning expression tree to `O(N)` nodes. This
//! expression evaluates the IN-list check directly as a single node: for
//! each (deduped, sorted) list value it computes `(min <= v) AND (v <= max)`
//! with vectorized arrow kernels across all containers, OR-reducing until
//! every container is known-true (then short-circuits).
//!
//! - `col IN (list)`: keep container iff `∃ v ∈ list : min ≤ v ≤ max`.
//! - `col NOT IN (list)`: keep container unless `min == max` and that single
//!   value is in the list — the only case where `min`/`max` stats let us
//!   prove the predicate is false for every row.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{Array, BooleanArray};
use arrow::buffer::BooleanBuffer;
use arrow::compute::kernels::boolean::{and_kleene, not, or_kleene};
use arrow::compute::kernels::cmp::{eq, lt_eq};
use arrow::datatypes::{DataType, FieldRef, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::ColumnarValue;

/// Pruning-time `col IN (list)` over a column's min/max statistics.
///
/// Returns a `BooleanArray` with one entry per container (row group): `true`
/// if the list may intersect `[min, max]`, `false` if it definitely does not,
/// `null` if either bound is unknown.
#[derive(Debug, Clone)]
pub(crate) struct InListPruningExpr {
    /// Evaluates to the per-container `min` statistics array.
    min_expr: Arc<dyn PhysicalExpr>,
    /// Evaluates to the per-container `max` statistics array.
    max_expr: Arc<dyn PhysicalExpr>,
    /// IN-list values, sorted ascending with duplicates removed. All values
    /// are pairwise comparable (constructor rejects lists where `partial_cmp`
    /// would return `None` between any pair).
    sorted_values: Vec<ScalarValue>,
    /// `true` for `NOT IN`: container is kept unless `min == max ∈ list`.
    negated: bool,
    /// Source column name, for `Display` / explain output only.
    column_name: String,
}

impl InListPruningExpr {
    /// Build a new `InListPruningExpr`.
    ///
    /// Returns `None` if `values` is empty or contains pairwise-incomparable
    /// elements (e.g. mixed timestamp units where `partial_cmp` is `None`).
    pub(crate) fn try_new(
        min_expr: Arc<dyn PhysicalExpr>,
        max_expr: Arc<dyn PhysicalExpr>,
        values: Vec<ScalarValue>,
        negated: bool,
        column_name: impl Into<String>,
    ) -> Option<Self> {
        if values.is_empty() {
            return None;
        }

        let mut sorted_values = values;
        let mut incomparable = false;
        sorted_values.sort_by(|a, b| {
            a.partial_cmp(b).unwrap_or_else(|| {
                incomparable = true;
                Ordering::Equal
            })
        });
        if incomparable {
            return None;
        }
        sorted_values.dedup();

        Some(Self {
            min_expr,
            max_expr,
            sorted_values,
            negated,
            column_name: column_name.into(),
        })
    }
}

impl PartialEq for InListPruningExpr {
    fn eq(&self, other: &Self) -> bool {
        self.min_expr.eq(&other.min_expr)
            && self.max_expr.eq(&other.max_expr)
            && self.sorted_values == other.sorted_values
            && self.negated == other.negated
    }
}

impl Eq for InListPruningExpr {}

impl Hash for InListPruningExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.min_expr.hash(state);
        self.max_expr.hash(state);
        self.sorted_values.hash(state);
        self.negated.hash(state);
    }
}

impl fmt::Display for InListPruningExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let preview = self
            .sorted_values
            .iter()
            .take(5)
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let rest = self.sorted_values.len().saturating_sub(5);
        let name = if self.negated {
            "NotInListPruning"
        } else {
            "InListPruning"
        };
        if rest > 0 {
            write!(
                f,
                "{name}({}, {}, [{}, +{rest} more])",
                self.min_expr, self.max_expr, preview
            )
        } else {
            write!(
                f,
                "{name}({}, {}, [{preview}])",
                self.min_expr, self.max_expr
            )
        }
    }
}

impl PhysicalExpr for InListPruningExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::Boolean)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(true)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let min_array = self
            .min_expr
            .evaluate(batch)?
            .into_array(batch.num_rows())?;
        let max_array = self
            .max_expr
            .evaluate(batch)?
            .into_array(batch.num_rows())?;

        if min_array.len() != max_array.len() {
            return internal_err!(
                "InListPruningExpr: min/max arrays differ in length ({} vs {})",
                min_array.len(),
                max_array.len()
            );
        }
        let num_rows = min_array.len();

        let result = if self.negated {
            evaluate_not_in(&self.sorted_values, &min_array, &max_array, num_rows)?
        } else {
            evaluate_in(&self.sorted_values, &min_array, &max_array, num_rows)?
        };
        Ok(ColumnarValue::Array(Arc::new(result)))
    }

    fn return_field(&self, _input_schema: &Schema) -> Result<FieldRef> {
        Ok(Arc::new(arrow::datatypes::Field::new(
            format!("{}_in_list_pruning", self.column_name),
            DataType::Boolean,
            true,
        )))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.min_expr, &self.max_expr]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if children.len() != 2 {
            return internal_err!(
                "InListPruningExpr expects 2 children, got {}",
                children.len()
            );
        }
        Ok(Arc::new(Self {
            min_expr: Arc::clone(&children[0]),
            max_expr: Arc::clone(&children[1]),
            sorted_values: self.sorted_values.clone(),
            negated: self.negated,
            column_name: self.column_name.clone(),
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

/// `col IN (list)` over stats: keep iff `∃ v ∈ list : min ≤ v ≤ max`.
fn evaluate_in(
    sorted_values: &[ScalarValue],
    min_array: &dyn Array,
    max_array: &dyn Array,
    num_rows: usize,
) -> Result<BooleanArray> {
    let mut iter = sorted_values.iter();
    let mut found: BooleanArray = match iter.next() {
        Some(first) => hit_array(first, min_array, max_array)?,
        None => BooleanArray::new(BooleanBuffer::new_unset(num_rows), None),
    };
    for value in iter {
        if found.null_count() == 0 && found.true_count() == num_rows {
            break;
        }
        let hit = hit_array(value, min_array, max_array)?;
        found = or_kleene(&found, &hit)?;
    }
    Ok(found)
}

/// `col NOT IN (list)` over stats: only prunable case is
/// `min == max ∈ list`, where every row in the container has a value that's
/// excluded. Returns `NOT(min == max AND min ∈ list)`; null min/max propagate
/// as null (treated as "keep").
fn evaluate_not_in(
    sorted_values: &[ScalarValue],
    min_array: &dyn Array,
    max_array: &dyn Array,
    num_rows: usize,
) -> Result<BooleanArray> {
    let min_eq_max = eq(&min_array, &max_array)?;

    // `min ∈ list`: OR of (min == v) over all list values.
    let mut iter = sorted_values.iter();
    let mut min_in_list: BooleanArray = match iter.next() {
        Some(first) => eq(&min_array, &first.to_scalar()?)?,
        None => BooleanArray::new(BooleanBuffer::new_unset(num_rows), None),
    };
    for value in iter {
        if min_in_list.null_count() == 0 && min_in_list.true_count() == num_rows {
            break;
        }
        let hit = eq(&min_array, &value.to_scalar()?)?;
        min_in_list = or_kleene(&min_in_list, &hit)?;
    }

    let prune = and_kleene(&min_eq_max, &min_in_list)?;
    Ok(not(&prune)?)
}

/// Compute `(min <= v) AND (v <= max)` across all containers using vectorized
/// kernels. Returns a `BooleanArray` of length `min_array.len()`.
fn hit_array(
    value: &ScalarValue,
    min_array: &dyn Array,
    max_array: &dyn Array,
) -> Result<BooleanArray> {
    let scalar = value.to_scalar()?;
    let min_le_v = lt_eq(&min_array, &scalar)?;
    let v_le_max = lt_eq(&scalar, &max_array)?;
    Ok(and_kleene(&min_le_v, &v_le_max)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, Int32Array};
    use arrow::datatypes::Field;
    use datafusion_physical_expr::expressions as phys_expr;

    fn col_int32(name: &str, idx: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(phys_expr::Column::new(name, idx))
    }

    #[test]
    fn intersects_returns_true_when_value_in_range() {
        let expr = InListPruningExpr::try_new(
            col_int32("c_min", 0),
            col_int32("c_max", 1),
            vec![
                ScalarValue::Int32(Some(5)),
                ScalarValue::Int32(Some(50)),
                ScalarValue::Int32(Some(500)),
            ],
            false,
            "c",
        )
        .unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("c_min", DataType::Int32, true),
            Field::new("c_max", DataType::Int32, true),
        ]));
        // Row 0: [0, 10]  → contains 5 → true
        // Row 1: [20, 40] → no overlap → false
        // Row 2: null min → null
        let mins = Int32Array::from(vec![Some(0), Some(20), None]);
        let maxs = Int32Array::from(vec![Some(10), Some(40), Some(100)]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(mins), Arc::new(maxs)]).unwrap();

        let out = expr.evaluate(&batch).unwrap();
        let arr = match out {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let bools = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(bools.value(0));
        assert!(!bools.value(1));
        assert!(bools.is_null(2));
    }

    #[test]
    fn try_new_rejects_empty_list() {
        assert!(
            InListPruningExpr::try_new(
                col_int32("c_min", 0),
                col_int32("c_max", 1),
                vec![],
                false,
                "c",
            )
            .is_none()
        );
    }

    #[test]
    fn try_new_rejects_incomparable_values() {
        // Mixed Int32 and Utf8 — partial_cmp returns None
        let values = vec![
            ScalarValue::Int32(Some(1)),
            ScalarValue::Utf8(Some("foo".into())),
        ];
        assert!(
            InListPruningExpr::try_new(
                col_int32("c_min", 0),
                col_int32("c_max", 1),
                values,
                false,
                "c",
            )
            .is_none()
        );
    }

    #[test]
    fn intersects_false_when_list_spans_over_range() {
        // List = [5, 100] — range [20, 40] contains neither
        let expr = InListPruningExpr::try_new(
            col_int32("c_min", 0),
            col_int32("c_max", 1),
            vec![ScalarValue::Int32(Some(5)), ScalarValue::Int32(Some(100))],
            false,
            "c",
        )
        .unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("c_min", DataType::Int32, true),
            Field::new("c_max", DataType::Int32, true),
        ]));
        let mins = Int32Array::from(vec![Some(20)]);
        let maxs = Int32Array::from(vec![Some(40)]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(mins), Arc::new(maxs)]).unwrap();

        let out = expr.evaluate(&batch).unwrap();
        let arr = match out {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let bools = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!bools.value(0));
    }

    #[test]
    fn not_in_prunes_only_when_range_is_single_value_in_list() {
        // `col NOT IN (5, 100)` pruned iff min == max ∈ {5, 100}.
        let expr = InListPruningExpr::try_new(
            col_int32("c_min", 0),
            col_int32("c_max", 1),
            vec![ScalarValue::Int32(Some(5)), ScalarValue::Int32(Some(100))],
            true,
            "c",
        )
        .unwrap();

        let schema = Arc::new(Schema::new(vec![
            Field::new("c_min", DataType::Int32, true),
            Field::new("c_max", DataType::Int32, true),
        ]));
        // Row 0: [5, 5]      → min=max=5, 5 ∈ list → prune (false)
        // Row 1: [100, 100]  → min=max=100, 100 ∈ list → prune (false)
        // Row 2: [7, 7]      → min=max=7, 7 ∉ list → keep (true)
        // Row 3: [5, 100]    → min ≠ max → keep (true)
        // Row 4: null min    → keep (null)
        let mins = Int32Array::from(vec![Some(5), Some(100), Some(7), Some(5), None]);
        let maxs =
            Int32Array::from(vec![Some(5), Some(100), Some(7), Some(100), Some(5)]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(mins), Arc::new(maxs)]).unwrap();

        let out = expr.evaluate(&batch).unwrap();
        let arr = match out {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let bools = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!bools.value(0));
        assert!(!bools.value(1));
        assert!(bools.value(2));
        assert!(bools.value(3));
        assert!(bools.is_null(4));
    }
}
