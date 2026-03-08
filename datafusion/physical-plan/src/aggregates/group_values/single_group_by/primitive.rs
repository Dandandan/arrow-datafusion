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

use crate::aggregates::group_values::GroupValues;
use ahash::RandomState;
use arrow::array::types::{IntervalDayTime, IntervalMonthDayNano};
use arrow::array::{
    Array, ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, NullBufferBuilder, PrimitiveArray,
    cast::AsArray,
};
use arrow::datatypes::{DataType, i256};
use datafusion_common::Result;
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::EmitTo;
use half::f16;
use hashbrown::hash_table::HashTable;
use std::mem::size_of;
use std::sync::Arc;

/// A trait to allow hashing of floating point numbers
pub(crate) trait HashValue {
    fn hash(&self, state: &RandomState) -> u64;
}

macro_rules! hash_integer {
    ($($t:ty),+) => {
        $(impl HashValue for $t {
            #[cfg(not(feature = "force_hash_collisions"))]
            fn hash(&self, state: &RandomState) -> u64 {
                state.hash_one(self)
            }

            #[cfg(feature = "force_hash_collisions")]
            fn hash(&self, _state: &RandomState) -> u64 {
                0
            }
        })+
    };
}
hash_integer!(i8, i16, i32, i64, i128, i256);
hash_integer!(u8, u16, u32, u64);
hash_integer!(IntervalDayTime, IntervalMonthDayNano);

macro_rules! hash_float {
    ($($t:ty),+) => {
        $(impl HashValue for $t {
            #[cfg(not(feature = "force_hash_collisions"))]
            fn hash(&self, state: &RandomState) -> u64 {
                state.hash_one(self.to_bits())
            }

            #[cfg(feature = "force_hash_collisions")]
            fn hash(&self, _state: &RandomState) -> u64 {
                0
            }
        })+
    };
}

hash_float!(f16, f32, f64);

/// A [`GroupValues`] storing a single column of primitive values
///
/// This specialization is significantly faster than using the more general
/// purpose `Row`s format
pub struct GroupValuesPrimitive<T: ArrowPrimitiveType> {
    /// The data type of the output array
    data_type: DataType,
    /// Stores the `(group_index, hash)` based on the hash of its value
    ///
    /// We also store `hash` is for reducing cost of rehashing. Such cost
    /// is obvious in high cardinality group by situation.
    /// More details can see:
    /// <https://github.com/apache/datafusion/issues/15961>
    map: HashTable<(usize, u64)>,
    /// The group index of the null value if any
    null_group: Option<usize>,
    /// The values for each group index
    values: Vec<T::Native>,
    /// The random state used to generate hashes
    random_state: RandomState,
    /// Reused buffer to store hashes
    hashes_buffer: Vec<u64>,
    /// Reused buffer for sorted indices during intern
    sorted_indices: Vec<u32>,
    /// Reused buffer for gathered hashes in sorted order
    gathered_hashes: Vec<u64>,
    /// Reused buffer for counting sort bucket offsets
    bucket_offsets: Vec<u32>,
}

impl<T: ArrowPrimitiveType> GroupValuesPrimitive<T> {
    pub fn new(data_type: DataType) -> Self {
        assert!(PrimitiveArray::<T>::is_compatible(&data_type));
        Self {
            data_type,
            map: HashTable::with_capacity(128),
            values: Vec::with_capacity(128),
            null_group: None,
            random_state: crate::aggregates::AGGREGATION_HASH_SEED,
            hashes_buffer: Vec::new(),
            sorted_indices: Vec::new(),
            gathered_hashes: Vec::new(),
            bucket_offsets: Vec::new(),
        }
    }
}

impl<T: ArrowPrimitiveType> GroupValues for GroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);
        let arr = cols[0].as_primitive::<T>();
        let n_rows = arr.len();

        groups.clear();
        groups.resize(n_rows, 0);

        // Compute all hashes upfront
        let raw_values = arr.values();
        let state = &self.random_state;
        let hashes_buffer = &mut self.hashes_buffer;
        hashes_buffer.clear();
        hashes_buffer.reserve(n_rows);
        for i in 0..n_rows {
            hashes_buffer.push(raw_values[i].hash(state));
        }

        // Reserve capacity upfront so the table won't rehash during inserts,
        // keeping the bucket layout stable for our sort order.
        self.map.reserve(n_rows, |&(_, h)| h);

        let sorted_indices = &mut self.sorted_indices;
        sorted_indices.clear();

        let gathered_hashes = &mut self.gathered_hashes;
        gathered_hashes.clear();

        // Only apply counting sort when the hash table is large enough to
        // benefit from improved cache locality (i.e. doesn't fit in L2 cache).
        let num_buckets = self.map.capacity().next_power_of_two();
        // Each bucket holds a control byte + entry: (usize, u64) = 16 bytes + 1
        let table_bytes = num_buckets * (size_of::<(usize, u64)>() + 1);
        const L2_CACHE_BYTES: usize = 256 * 1024;

        if table_bytes > L2_CACHE_BYTES {
            let bucket_mask = num_buckets - 1;

            let sort_bits = 8u32;
            let num_partitions = 1usize << sort_bits;
            let bucket_bits = num_buckets.trailing_zeros();
            let shift = bucket_bits.saturating_sub(sort_bits);

            // Phase 1: Count elements per partition
            let bucket_offsets = &mut self.bucket_offsets;
            bucket_offsets.clear();
            bucket_offsets.resize(num_partitions, 0);

            for &hash in hashes_buffer.iter() {
                let partition = ((hash as usize) & bucket_mask) >> shift;
                bucket_offsets[partition] += 1;
            }

            // Phase 2: Prefix sum to get starting offsets
            let mut sum = 0u32;
            for count in bucket_offsets.iter_mut() {
                let c = *count;
                *count = sum;
                sum += c;
            }

            // Phase 3: Scatter indices and gather hashes into sorted order
            sorted_indices.resize(n_rows, 0);
            gathered_hashes.resize(n_rows, 0);

            for (i, &hash) in hashes_buffer.iter().enumerate() {
                let partition = ((hash as usize) & bucket_mask) >> shift;
                let pos = bucket_offsets[partition] as usize;
                sorted_indices[pos] = i as u32;
                gathered_hashes[pos] = hash;
                bucket_offsets[partition] += 1;
            }
        } else {
            // Small table fits in L2 cache; skip sorting overhead
            sorted_indices.extend(0..n_rows as u32);
            gathered_hashes.extend_from_slice(hashes_buffer);
        }

        // Process in sorted hash order for cache locality
        let mut prev_hash: u64 = 0;
        let mut prev_group: usize = 0;
        let mut has_prev = false;

        for (sorted_pos, &idx) in sorted_indices.iter().enumerate() {
            let row = idx as usize;

            // Handle nulls
            if arr.is_null(row) {
                let group_id = *self.null_group.get_or_insert_with(|| {
                    let group_id = self.values.len();
                    self.values.push(Default::default());
                    group_id
                });
                groups[row] = group_id;
                continue;
            }

            let key = raw_values[row];
            let target_hash = gathered_hashes[sorted_pos];

            // Fast path: same hash as previous row, check equality directly
            if has_prev && target_hash == prev_hash {
                if unsafe { self.values.get_unchecked(prev_group).is_eq(key) } {
                    groups[row] = prev_group;
                    continue;
                }
            }

            let entry = self.map.find_mut(target_hash, |&(g, h)| unsafe {
                target_hash == h && self.values.get_unchecked(g).is_eq(key)
            });

            let group_idx = match entry {
                Some(&mut (g, _)) => g,
                None => {
                    let g = self.values.len();
                    self.map
                        .insert_unique(target_hash, (g, target_hash), |&(_, h)| h);
                    self.values.push(key);
                    g
                }
            };

            groups[row] = group_idx;
            prev_hash = target_hash;
            prev_group = group_idx;
            has_prev = true;
        }

        Ok(())
    }

    fn size(&self) -> usize {
        self.map.capacity() * size_of::<(usize, u64)>()
            + self.values.allocated_size()
            + self.hashes_buffer.allocated_size()
            + self.sorted_indices.allocated_size()
            + self.gathered_hashes.allocated_size()
            + self.bucket_offsets.allocated_size()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        fn build_primitive<T: ArrowPrimitiveType>(
            values: Vec<T::Native>,
            null_idx: Option<usize>,
        ) -> PrimitiveArray<T> {
            let nulls = null_idx.map(|null_idx| {
                let mut buffer = NullBufferBuilder::new(values.len());
                buffer.append_n_non_nulls(null_idx);
                buffer.append_null();
                buffer.append_n_non_nulls(values.len() - null_idx - 1);
                // NOTE: The inner builder must be constructed as there is at least one null
                buffer.finish().unwrap()
            });
            PrimitiveArray::<T>::new(values.into(), nulls)
        }

        let array: PrimitiveArray<T> = match emit_to {
            EmitTo::All => {
                self.map.clear();
                build_primitive(std::mem::take(&mut self.values), self.null_group.take())
            }
            EmitTo::First(n) => {
                self.map.retain(|entry| {
                    // Decrement group index by n
                    let group_idx = entry.0;
                    match group_idx.checked_sub(n) {
                        // Group index was >= n, shift value down
                        Some(sub) => {
                            entry.0 = sub;
                            true
                        }
                        // Group index was < n, so remove from table
                        None => false,
                    }
                });
                let null_group = match &mut self.null_group {
                    Some(v) if *v >= n => {
                        *v -= n;
                        None
                    }
                    Some(_) => self.null_group.take(),
                    None => None,
                };
                let mut split = self.values.split_off(n);
                std::mem::swap(&mut self.values, &mut split);
                build_primitive(split, null_group)
            }
        };

        Ok(vec![Arc::new(array.with_data_type(self.data_type.clone()))])
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.values.clear();
        self.values.shrink_to(num_rows);
        self.map.clear();
        self.map.shrink_to(num_rows, |_| 0); // hasher does not matter since the map is cleared
        self.hashes_buffer.clear();
        self.hashes_buffer.shrink_to(num_rows);
        self.sorted_indices.clear();
        self.sorted_indices.shrink_to(num_rows);
        self.gathered_hashes.clear();
        self.gathered_hashes.shrink_to(num_rows);
    }
}
