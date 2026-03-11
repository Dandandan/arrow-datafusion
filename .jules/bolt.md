## 2025-05-15 - Fast paths for join utilities
**Learning:** Join utilities like `equal_rows_arr` and `apply_join_filter_to_indices` often process batches where all rows match. In these cases, calling `compute::filter` is redundant and expensive as it allocates and copies the indices.
**Action:** Always add early returns for "all-true" mask cases in hot path filtering logic to avoid unnecessary copies.
