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
    ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, NullBufferBuilder, PrimitiveArray,
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
    /// Whether this is in streaming mode (must preserve input order for group indices)
    streaming: bool,
    /// Reused buffer for sorted indices during intern
    sorted_indices: Vec<u32>,
    /// Reused buffer for hashes during intern
    hashes_buffer: Vec<u64>,
}

impl<T: ArrowPrimitiveType> GroupValuesPrimitive<T> {
    pub fn new(data_type: DataType, streaming: bool) -> Self {
        assert!(PrimitiveArray::<T>::is_compatible(&data_type));
        Self {
            data_type,
            map: HashTable::with_capacity(128),
            values: Vec::with_capacity(128),
            null_group: None,
            random_state: crate::aggregates::AGGREGATION_HASH_SEED,
            streaming,
            sorted_indices: Vec::new(),
            hashes_buffer: Vec::new(),
        }
    }
}

impl<T: ArrowPrimitiveType> GroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    /// Sequential (non-sorting) intern for streaming mode.
    /// Preserves input order for group index assignment.
    fn intern_sequential(
        &mut self,
        cols: &[ArrayRef],
        groups: &mut Vec<usize>,
    ) -> Result<()> {
        groups.clear();
        for v in cols[0].as_primitive::<T>() {
            let group_id = match v {
                None => *self.null_group.get_or_insert_with(|| {
                    let group_id = self.values.len();
                    self.values.push(Default::default());
                    group_id
                }),
                Some(key) => {
                    let state = &self.random_state;
                    let hash = key.hash(state);
                    let insert = self.map.entry(
                        hash,
                        |&(g, h)| unsafe {
                            hash == h && self.values.get_unchecked(g).is_eq(key)
                        },
                        |&(_, h)| h,
                    );

                    match insert {
                        hashbrown::hash_table::Entry::Occupied(o) => o.get().0,
                        hashbrown::hash_table::Entry::Vacant(v) => {
                            let g = self.values.len();
                            v.insert((g, hash));
                            self.values.push(key);
                            g
                        }
                    }
                }
            };
            groups.push(group_id)
        }
        Ok(())
    }
}

impl<T: ArrowPrimitiveType> GroupValues for GroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);

        // In streaming mode, must preserve input order for group index
        // assignment (required by GroupOrdering).
        if self.streaming {
            return self.intern_sequential(cols, groups);
        }

        let n_rows = cols[0].len();
        groups.clear();
        groups.resize(n_rows, 0);

        let array = cols[0].as_primitive::<T>();
        let state = &self.random_state;

        // Compute hashes for all rows and handle nulls
        let hashes = &mut self.hashes_buffer;
        hashes.clear();
        hashes.resize(n_rows, 0);

        let mut sorted_indices = std::mem::take(&mut self.sorted_indices);
        sorted_indices.clear();
        sorted_indices.reserve(n_rows);

        for (i, v) in array.iter().enumerate() {
            match v {
                None => {
                    let group_id = *self.null_group.get_or_insert_with(|| {
                        let group_id = self.values.len();
                        self.values.push(Default::default());
                        group_id
                    });
                    groups[i] = group_id;
                }
                Some(key) => {
                    hashes[i] = key.hash(state);
                    sorted_indices.push(i as u32);
                }
            }
        }

        // Ensure capacity upfront so the table won't rehash during inserts,
        // keeping the bucket layout stable for our sort order.
        self.map.reserve(sorted_indices.len(), |&(_, h)| h);

        // Sort by (bucket_index, hash) for optimal cache locality and
        // duplicate detection. Rotating left by bucket_bits moves the
        // bucket-determining low bits to the top of the u64, making them
        // the primary sort key, while keeping same-hash entries adjacent.
        //
        // capacity() returns element capacity (buckets * 7/8), not bucket
        // count. next_power_of_two() recovers the actual bucket count.
        let bucket_bits =
            self.map.capacity().next_power_of_two().trailing_zeros();
        sorted_indices
            .sort_unstable_by_key(|&i| hashes[i as usize].rotate_left(bucket_bits));

        // Process in sorted bucket order
        let mut prev_hash: u64 = 0;
        let mut prev_group: usize = 0;
        let mut has_prev = false;

        for &idx in &sorted_indices {
            let row = idx as usize;
            let hash = hashes[row];
            let key = array.value(row);

            // Fast path: same hash as previous row, check value equality directly
            if has_prev && hash == prev_hash {
                if unsafe { self.values.get_unchecked(prev_group).is_eq(key) } {
                    groups[row] = prev_group;
                    continue;
                }
            }

            // Normal path: probe hash table
            let insert = self.map.entry(
                hash,
                |&(g, h)| unsafe { hash == h && self.values.get_unchecked(g).is_eq(key) },
                |&(_, h)| h,
            );

            let group_id = match insert {
                hashbrown::hash_table::Entry::Occupied(o) => o.get().0,
                hashbrown::hash_table::Entry::Vacant(v) => {
                    let g = self.values.len();
                    v.insert((g, hash));
                    self.values.push(key);
                    g
                }
            };

            groups[row] = group_id;
            prev_hash = hash;
            prev_group = group_id;
            has_prev = true;
        }

        self.sorted_indices = sorted_indices;
        Ok(())
    }

    fn size(&self) -> usize {
        self.map.capacity() * size_of::<(usize, u64)>() + self.values.allocated_size()
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
    }
}
