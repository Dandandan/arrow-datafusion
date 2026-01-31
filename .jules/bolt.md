# Bolt's Journal - Critical Learnings

## 2025-05-15 - [HashJoin Build Side Optimization]
**Learning:** DataFusion's join operators (HashJoin, NestedLoopJoin, etc.) often perform redundant work during the build phase, such as concatenating batches even when there is only one batch, and re-evaluating join key expressions multiple times (once for each batch during streaming, and again on the concatenated batch).
**Action:** Always look for single-batch fast paths in operators that buffer input. Reuse evaluated results across different stages of the build phase (e.g., PHJ check and hash map building).
