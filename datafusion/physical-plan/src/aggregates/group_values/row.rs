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
use arrow::array::{Array, ArrayRef, ListArray, StructArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::row::{RowConverter, Rows, SortField};
use datafusion_common::Result;
use datafusion_common::hash_utils::create_hashes;
use datafusion_execution::memory_pool::proxy::{HashTableAllocExt, VecAllocExt};
use datafusion_expr::EmitTo;
use hashbrown::hash_table::HashTable;
use log::debug;
use std::mem::size_of;
use std::sync::Arc;

/// A [`GroupValues`] making use of [`Rows`]
///
/// This is a general implementation of [`GroupValues`] that works for any
/// combination of data types and number of columns, including nested types such as
/// structs and lists.
///
/// It uses the arrow-rs [`Rows`] to store the group values, which is a row-wise
/// representation.
pub struct GroupValuesRows {
    /// The output schema
    schema: SchemaRef,

    /// Converter for the group values
    row_converter: RowConverter,

    /// Logically maps group values to a group_index in
    /// [`Self::group_values`] and in each accumulator
    ///
    /// Uses the raw API of hashbrown to avoid actually storing the
    /// keys (group values) in the table
    ///
    /// keys: u64 hashes of the GroupValue
    /// values: (hash, group_index)
    map: HashTable<(u64, usize)>,

    /// The size of `map` in bytes
    map_size: usize,

    /// The actual group by values, stored in arrow [`Row`] format.
    /// `group_values[i]` holds the group value for group_index `i`.
    ///
    /// The row format is used to compare group keys quickly and store
    /// them efficiently in memory. Quick comparison is especially
    /// important for multi-column group keys.
    ///
    /// [`Row`]: arrow::row::Row
    group_values: Option<Rows>,

    /// reused buffer to store hashes
    hashes_buffer: Vec<u64>,

    /// reused buffer to store rows
    rows_buffer: Rows,

    /// Reused buffer for sorted indices during intern
    sorted_indices: Vec<u32>,

    /// Reused buffer for gathered hashes in sorted order
    gathered_hashes: Vec<u64>,

    /// Reused buffer for counting sort bucket offsets
    bucket_offsets: Vec<u32>,

    /// Whether this is in streaming mode (must preserve input order)
    streaming: bool,

    /// Random state for creating hashes
    random_state: RandomState,
}

impl GroupValuesRows {
    pub fn try_new(schema: SchemaRef, streaming: bool) -> Result<Self> {
        // Print a debugging message, so it is clear when the (slower) fallback
        // GroupValuesRows is used.
        debug!("Creating GroupValuesRows for schema: {schema}");
        let row_converter = RowConverter::new(
            schema
                .fields()
                .iter()
                .map(|f| SortField::new(f.data_type().clone()))
                .collect(),
        )?;

        let map = HashTable::with_capacity(0);

        let starting_rows_capacity = 1000;

        let starting_data_capacity = 64 * starting_rows_capacity;
        let rows_buffer =
            row_converter.empty_rows(starting_rows_capacity, starting_data_capacity);
        Ok(Self {
            schema,
            row_converter,
            map,
            map_size: 0,
            group_values: None,
            hashes_buffer: Default::default(),
            rows_buffer,
            sorted_indices: Vec::new(),
            gathered_hashes: Vec::new(),
            bucket_offsets: Vec::new(),
            streaming,
            random_state: crate::aggregates::AGGREGATION_HASH_SEED,
        })
    }
}

impl GroupValues for GroupValuesRows {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        // Convert the group keys into the row format
        let group_rows = &mut self.rows_buffer;
        group_rows.clear();
        self.row_converter.append(group_rows, cols)?;
        let n_rows = group_rows.num_rows();

        let mut group_values = match self.group_values.take() {
            Some(group_values) => group_values,
            None => self.row_converter.empty_rows(0, 0),
        };

        // tracks to which group each of the input rows belongs
        groups.clear();
        groups.resize(n_rows, 0);

        // 1.1 Calculate the group keys for the group values
        let batch_hashes = &mut self.hashes_buffer;
        batch_hashes.clear();
        batch_hashes.resize(n_rows, 0);
        create_hashes(cols, &self.random_state, batch_hashes)?;

        // Sort indices by hash bucket for cache locality and duplicate detection
        // (only when not streaming, to preserve group index assignment order)
        let sorted_indices = &mut self.sorted_indices;
        sorted_indices.clear();

        let gathered_hashes = &mut self.gathered_hashes;
        gathered_hashes.clear();

        if !self.streaming {
            // Ensure capacity upfront so the table won't rehash during inserts,
            // keeping the bucket layout stable for our sort order.
            self.map.reserve(n_rows, |(hash, _)| *hash);

            // Only apply counting sort when the hash table is large enough to
            // benefit from improved cache locality (i.e. doesn't fit in L2 cache).
            let num_buckets = self.map.capacity().next_power_of_two();
            // Each bucket holds a control byte + entry: (u64, usize) = 16 bytes + 1
            let table_bytes = num_buckets * (size_of::<(u64, usize)>() + 1);
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

                for &hash in batch_hashes.iter() {
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

                for (i, &hash) in batch_hashes.iter().enumerate() {
                    let partition = ((hash as usize) & bucket_mask) >> shift;
                    let pos = bucket_offsets[partition] as usize;
                    sorted_indices[pos] = i as u32;
                    gathered_hashes[pos] = hash;
                    bucket_offsets[partition] += 1;
                }
            } else {
                // Small table fits in L2 cache; skip sorting overhead
                sorted_indices.extend(0..n_rows as u32);
                gathered_hashes.extend_from_slice(batch_hashes);
            }
        } else {
            sorted_indices.extend(0..n_rows as u32);
            gathered_hashes.extend_from_slice(batch_hashes);
        }

        // Process in (potentially sorted) hash order
        let mut prev_hash: u64 = 0;
        let mut prev_group: usize = 0;
        let mut has_prev = false;

        for (sorted_pos, &idx) in sorted_indices.iter().enumerate() {
            let row = idx as usize;
            let target_hash = gathered_hashes[sorted_pos];

            // Fast path: same hash as previous row, check row equality directly
            if has_prev && target_hash == prev_hash {
                if group_rows.row(row) == group_values.row(prev_group) {
                    groups[row] = prev_group;
                    continue;
                }
            }

            let entry = self.map.find_mut(target_hash, |(exist_hash, group_idx)| {
                target_hash == *exist_hash
                    && group_rows.row(row) == group_values.row(*group_idx)
            });

            let group_idx = match entry {
                // Existing group_index for this group value
                Some((_hash, group_idx)) => *group_idx,
                //  1.2 Need to create new entry for the group
                None => {
                    // Add new entry to aggr_state and save newly created index
                    let group_idx = group_values.num_rows();
                    group_values.push(group_rows.row(row));

                    // for hasher function, use precomputed hash value
                    self.map.insert_accounted(
                        (target_hash, group_idx),
                        |(hash, _group_index)| *hash,
                        &mut self.map_size,
                    );
                    group_idx
                }
            };
            groups[row] = group_idx;
            prev_hash = target_hash;
            prev_group = group_idx;
            has_prev = true;
        }

        self.group_values = Some(group_values);

        Ok(())
    }

    fn size(&self) -> usize {
        let group_values_size = self.group_values.as_ref().map(|v| v.size()).unwrap_or(0);
        self.row_converter.size()
            + group_values_size
            + self.map_size
            + self.rows_buffer.size()
            + self.hashes_buffer.allocated_size()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        self.group_values
            .as_ref()
            .map(|group_values| group_values.num_rows())
            .unwrap_or(0)
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let mut group_values = self
            .group_values
            .take()
            .expect("Can not emit from empty rows");

        let mut output = match emit_to {
            EmitTo::All => {
                let output = self.row_converter.convert_rows(&group_values)?;
                group_values.clear();
                self.map.clear();
                output
            }
            EmitTo::First(n) => {
                let groups_rows = group_values.iter().take(n);
                let output = self.row_converter.convert_rows(groups_rows)?;
                // Clear out first n group keys by copying them to a new Rows.
                // TODO file some ticket in arrow-rs to make this more efficient?
                let mut new_group_values = self.row_converter.empty_rows(0, 0);
                for row in group_values.iter().skip(n) {
                    new_group_values.push(row);
                }
                std::mem::swap(&mut new_group_values, &mut group_values);

                self.map.retain(|(_exists_hash, group_idx)| {
                    // Decrement group index by n
                    match group_idx.checked_sub(n) {
                        // Group index was >= n, shift value down
                        Some(sub) => {
                            *group_idx = sub;
                            true
                        }
                        // Group index was < n, so remove from table
                        None => false,
                    }
                });
                output
            }
        };

        // TODO: Materialize dictionaries in group keys
        // https://github.com/apache/datafusion/issues/7647
        for (field, array) in self.schema.fields.iter().zip(&mut output) {
            let expected = field.data_type();
            *array = dictionary_encode_if_necessary(array, expected)?;
        }

        self.group_values = Some(group_values);
        Ok(output)
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.group_values = self.group_values.take().map(|mut rows| {
            rows.clear();
            rows
        });
        self.map.clear();
        self.map.shrink_to(num_rows, |_| 0); // hasher does not matter since the map is cleared
        self.map_size = self.map.capacity() * size_of::<(u64, usize)>();
        self.hashes_buffer.clear();
        self.hashes_buffer.shrink_to(num_rows);
        self.sorted_indices.clear();
        self.sorted_indices.shrink_to(num_rows);
    }
}

fn dictionary_encode_if_necessary(
    array: &ArrayRef,
    expected: &DataType,
) -> Result<ArrayRef> {
    match (expected, array.data_type()) {
        (DataType::Struct(expected_fields), _) => {
            let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
            let arrays = expected_fields
                .iter()
                .zip(struct_array.columns())
                .map(|(expected_field, column)| {
                    dictionary_encode_if_necessary(column, expected_field.data_type())
                })
                .collect::<Result<Vec<_>>>()?;

            Ok(Arc::new(StructArray::try_new(
                expected_fields.clone(),
                arrays,
                struct_array.nulls().cloned(),
            )?))
        }
        (DataType::List(expected_field), &DataType::List(_)) => {
            let list = array.as_any().downcast_ref::<ListArray>().unwrap();

            Ok(Arc::new(ListArray::try_new(
                Arc::<arrow::datatypes::Field>::clone(expected_field),
                list.offsets().clone(),
                dictionary_encode_if_necessary(
                    list.values(),
                    expected_field.data_type(),
                )?,
                list.nulls().cloned(),
            )?))
        }
        (DataType::Dictionary(_, _), _) => Ok(cast(array.as_ref(), expected)?),
        (_, _) => Ok(Arc::<dyn Array>::clone(array)),
    }
}
