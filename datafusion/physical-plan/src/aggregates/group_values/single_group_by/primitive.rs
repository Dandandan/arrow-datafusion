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
    Array, ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, NullBufferBuilder,
    PrimitiveArray, cast::AsArray,
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
        }
    }
}

/// Batch size for batched lookups in intern
const INTERN_BATCH_SIZE: usize = 8;

impl<T: ArrowPrimitiveType> GroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    /// Look up a single value in the hash table, inserting if not found.
    #[inline(always)]
    fn find_or_insert(&mut self, key: T::Native, hash: u64) -> usize {
        let found = self.map.find(hash, |&(g, h)| unsafe {
            hash == h && self.values.get_unchecked(g).is_eq(key)
        });
        match found {
            Some(&(g, _)) => g,
            None => {
                let g = self.values.len();
                self.map.insert_unique(hash, (g, hash), |&(_, h)| h);
                self.values.push(key);
                g
            }
        }
    }

    /// Process a batch of INTERN_BATCH_SIZE non-null values, deduplicating
    /// within the batch and batching hash table probes to hide memory latency.
    #[inline(always)]
    fn intern_batch(
        &mut self,
        chunk: &[T::Native; INTERN_BATCH_SIZE],
        groups: &mut Vec<usize>,
    ) {
        // Step 1: Deduplicate within the batch by comparing values.
        // dedup[i] == i means the value is unique among chunk[0..=i].
        // dedup[i] == j (j < i) means chunk[i] == chunk[j].
        let mut dedup = [0usize; INTERN_BATCH_SIZE];
        for i in 0..INTERN_BATCH_SIZE {
            dedup[i] = i;
            for j in 0..i {
                if chunk[j].is_eq(chunk[i]) {
                    dedup[i] = j;
                    break;
                }
            }
        }

        // Step 2: Hash only unique values
        let mut hashes = [0u64; INTERN_BATCH_SIZE];
        for i in 0..INTERN_BATCH_SIZE {
            if dedup[i] == i {
                hashes[i] = chunk[i].hash(&self.random_state);
            }
        }

        // Step 3: Create iter_hash iterators for all unique values.
        // Creating the iterator loads the control byte group from the hash
        // table, triggering cache line loads at up to 8 different positions.
        // By the time we advance the iterators, the data is likely warm.
        let iters = std::array::from_fn::<_, INTERN_BATCH_SIZE, _>(|i| {
            if dedup[i] == i {
                Some(self.map.iter_hash(hashes[i]))
            } else {
                None
            }
        });

        // Step 4: Advance iterators to find matching entries.
        let mut found = [None::<usize>; INTERN_BATCH_SIZE];
        for (i, iter) in iters.into_iter().enumerate() {
            if let Some(mut iter) = iter {
                found[i] = iter
                    .find(|&(g, h)| unsafe {
                        hashes[i] == *h
                            && self.values.get_unchecked(*g).is_eq(chunk[i])
                    })
                    .map(|&(g, _)| g);
            }
        }

        // Step 4: Insert missing values and collect group IDs
        let mut group_ids = [0usize; INTERN_BATCH_SIZE];
        for i in 0..INTERN_BATCH_SIZE {
            if dedup[i] != i {
                group_ids[i] = group_ids[dedup[i]];
            } else if let Some(g) = found[i] {
                group_ids[i] = g;
            } else {
                let g = self.values.len();
                self.map
                    .insert_unique(hashes[i], (g, hashes[i]), |&(_, h)| h);
                self.values.push(chunk[i]);
                group_ids[i] = g;
            }
        }

        groups.extend_from_slice(&group_ids);
    }
}

impl<T: ArrowPrimitiveType> GroupValues for GroupValuesPrimitive<T>
where
    T::Native: HashValue,
{
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), 1);
        groups.clear();

        let arr = cols[0].as_primitive::<T>();
        let len = arr.len();
        groups.reserve(len);

        if arr.null_count() == 0 {
            // Fast path: no nulls, use batched lookups
            let values = arr.values();
            for chunk in values.chunks_exact(INTERN_BATCH_SIZE) {
                let chunk: &[T::Native; INTERN_BATCH_SIZE] =
                    chunk.try_into().unwrap();
                self.intern_batch(chunk, groups);
            }
            for &key in values.chunks_exact(INTERN_BATCH_SIZE).remainder() {
                let hash = key.hash(&self.random_state);
                let group_id = self.find_or_insert(key, hash);
                groups.push(group_id);
            }
        } else {
            // Path with nulls: process one at a time
            for v in arr.iter() {
                let group_id = match v {
                    None => *self.null_group.get_or_insert_with(|| {
                        let group_id = self.values.len();
                        self.values.push(Default::default());
                        group_id
                    }),
                    Some(key) => {
                        let hash = key.hash(&self.random_state);
                        self.find_or_insert(key, hash)
                    }
                };
                groups.push(group_id);
            }
        }
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
