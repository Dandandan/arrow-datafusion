## 2025-05-15 - [Optimize single-batch build side in hash join]
**Learning:** In `HashJoinExec`, build-side batches were being evaluated for join keys individually to build the hash map, then concatenated and re-evaluated on the merged batch. This resulted in redundant CPU work.
**Action:** Implement a fast path for single-batch build side to evaluate expressions once and reuse the results, bypassing `concat_batches` entirely.
