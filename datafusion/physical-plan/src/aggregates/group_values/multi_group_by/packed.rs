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

//! [`GroupValuesPacked`] for multi-column GROUP BY with small integer/boolean columns
//! packed into a single u128 key.
//!
//! When all group-by columns are small fixed-width types (integers, booleans, dates, etc.)
//! whose combined bit-width fits within 128 bits, we pack all columns into a single
//! `u128` key. This eliminates per-column hashing, per-column equality comparison,
//! and trait-object dispatch — reducing the hot path to a single hash + single comparison.

use crate::aggregates::group_values::GroupValues;
use ahash::RandomState;
use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, NullBufferBuilder};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Date32Type, Date64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, SchemaRef, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
    Time64NanosecondType, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
use datafusion_common::Result;
use datafusion_execution::memory_pool::proxy::VecAllocExt;
use datafusion_expr::EmitTo;
use hashbrown::hash_table::HashTable;
use std::mem::size_of;
use std::sync::Arc;

/// Value bit width of each column type when packed (excluding the null flag).
fn value_bit_width(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::Boolean => Some(1),
        DataType::Int8 | DataType::UInt8 => Some(8),
        DataType::Int16 | DataType::UInt16 => Some(16),
        DataType::Int32 | DataType::UInt32 | DataType::Date32 | DataType::Time32(_) => {
            Some(32)
        }
        DataType::Int64
        | DataType::UInt64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, _) => Some(64),
        DataType::Decimal128(_, _) => None, // 128 bits is too wide to pack with others
        _ => None,
    }
}

/// Returns true if the schema can be packed into a single u128 key.
/// Non-nullable columns save 1 bit each by not needing a null flag.
pub fn can_pack_schema(schema: &SchemaRef) -> bool {
    if schema.fields().len() < 2 {
        return false;
    }

    let total_bits: Option<usize> = schema
        .fields()
        .iter()
        .map(|f| {
            value_bit_width(f.data_type())
                .map(|bits| bits + if f.is_nullable() { 1 } else { 0 })
        })
        .sum();

    matches!(total_bits, Some(bits) if bits <= 128)
}

/// Description of how a single column is packed into the u128 key.
#[derive(Debug, Clone)]
struct PackedColumn {
    /// Bit offset within the u128 where this column's value starts
    bit_offset: usize,
    /// Number of value bits (excluding the null flag)
    value_bits: usize,
    /// Whether this column is nullable (if false, no null flag bit is allocated)
    nullable: bool,
    /// The data type for this column
    data_type: DataType,
}

impl PackedColumn {
    /// Bit offset of the null flag (1 bit right after value bits).
    /// Only valid when `nullable` is true.
    fn null_bit_offset(&self) -> usize {
        debug_assert!(self.nullable);
        self.bit_offset + self.value_bits
    }
}

/// A [`GroupValues`] that packs multiple small integer/boolean columns into a single
/// `u128` key for fast hashing and comparison.
pub struct GroupValuesPacked {
    /// Layout for each column
    columns: Vec<PackedColumn>,
    /// Hash table mapping hash -> (group_index, hash)
    map: HashTable<(usize, u64)>,
    /// The packed values for each group
    values: Vec<u128>,
    /// Reusable buffer for packed keys within a batch
    packed_buffer: Vec<u128>,
    /// Random state for hashing
    random_state: RandomState,
}

impl GroupValuesPacked {
    pub fn try_new(schema: &SchemaRef) -> Result<Self> {
        let mut columns = Vec::with_capacity(schema.fields().len());
        let mut bit_offset = 0;

        for field in schema.fields() {
            let value_bits = match field.data_type() {
                DataType::Boolean => 1,
                DataType::Int8 | DataType::UInt8 => 8,
                DataType::Int16 | DataType::UInt16 => 16,
                DataType::Int32
                | DataType::UInt32
                | DataType::Date32
                | DataType::Time32(_) => 32,
                DataType::Int64
                | DataType::UInt64
                | DataType::Date64
                | DataType::Time64(_)
                | DataType::Timestamp(_, _) => 64,
                other => {
                    return Err(datafusion_common::DataFusionError::Internal(format!(
                        "Unsupported type for packed group by: {other}"
                    )));
                }
            };

            let nullable = field.is_nullable();
            columns.push(PackedColumn {
                bit_offset,
                value_bits,
                nullable,
                data_type: field.data_type().clone(),
            });

            bit_offset += value_bits + if nullable { 1 } else { 0 };
        }

        debug_assert!(bit_offset <= 128);

        Ok(Self {
            columns,
            map: HashTable::with_capacity(128),
            values: Vec::with_capacity(128),
            packed_buffer: Vec::new(),
            random_state: crate::aggregates::AGGREGATION_HASH_SEED,
        })
    }

    /// Pack all rows from a batch of columns into the packed_buffer.
    ///
    /// Type dispatch happens once per column (at the batch level), not per row.
    fn pack_batch(&mut self, cols: &[ArrayRef], n_rows: usize) {
        self.packed_buffer.clear();
        self.packed_buffer.resize(n_rows, 0u128);

        for (col_desc, array) in self.columns.iter().zip(cols.iter()) {
            match &col_desc.data_type {
                DataType::Boolean => {
                    let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                    pack_boolean(arr, col_desc, &mut self.packed_buffer);
                }
                DataType::Int8 => {
                    pack_primitive::<Int8Type>(array, col_desc, &mut self.packed_buffer)
                }
                DataType::Int16 => {
                    pack_primitive::<Int16Type>(array, col_desc, &mut self.packed_buffer)
                }
                DataType::Int32 => {
                    pack_primitive::<Int32Type>(array, col_desc, &mut self.packed_buffer)
                }
                DataType::Int64 => {
                    pack_primitive::<Int64Type>(array, col_desc, &mut self.packed_buffer)
                }
                DataType::UInt8 => {
                    pack_primitive::<UInt8Type>(array, col_desc, &mut self.packed_buffer)
                }
                DataType::UInt16 => {
                    pack_primitive::<UInt16Type>(array, col_desc, &mut self.packed_buffer)
                }
                DataType::UInt32 => {
                    pack_primitive::<UInt32Type>(array, col_desc, &mut self.packed_buffer)
                }
                DataType::UInt64 => {
                    pack_primitive::<UInt64Type>(array, col_desc, &mut self.packed_buffer)
                }
                DataType::Date32 => {
                    pack_primitive::<Date32Type>(array, col_desc, &mut self.packed_buffer)
                }
                DataType::Date64 => {
                    pack_primitive::<Date64Type>(array, col_desc, &mut self.packed_buffer)
                }
                DataType::Time32(TimeUnit::Second) => pack_primitive::<Time32SecondType>(
                    array,
                    col_desc,
                    &mut self.packed_buffer,
                ),
                DataType::Time32(TimeUnit::Millisecond) => {
                    pack_primitive::<Time32MillisecondType>(
                        array,
                        col_desc,
                        &mut self.packed_buffer,
                    )
                }
                DataType::Time64(TimeUnit::Microsecond) => {
                    pack_primitive::<Time64MicrosecondType>(
                        array,
                        col_desc,
                        &mut self.packed_buffer,
                    )
                }
                DataType::Time64(TimeUnit::Nanosecond) => {
                    pack_primitive::<Time64NanosecondType>(
                        array,
                        col_desc,
                        &mut self.packed_buffer,
                    )
                }
                DataType::Timestamp(TimeUnit::Second, _) => {
                    pack_primitive::<TimestampSecondType>(
                        array,
                        col_desc,
                        &mut self.packed_buffer,
                    )
                }
                DataType::Timestamp(TimeUnit::Millisecond, _) => {
                    pack_primitive::<TimestampMillisecondType>(
                        array,
                        col_desc,
                        &mut self.packed_buffer,
                    )
                }
                DataType::Timestamp(TimeUnit::Microsecond, _) => {
                    pack_primitive::<TimestampMicrosecondType>(
                        array,
                        col_desc,
                        &mut self.packed_buffer,
                    )
                }
                DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                    pack_primitive::<TimestampNanosecondType>(
                        array,
                        col_desc,
                        &mut self.packed_buffer,
                    )
                }
                _ => unreachable!(),
            }
        }
    }

    /// Unpack a u128 key back into individual column arrays for a range of group values.
    fn unpack_values(&self, values: &[u128]) -> Vec<ArrayRef> {
        self.columns
            .iter()
            .map(|col_desc| unpack_column(col_desc, values))
            .collect()
    }
}

/// Bit-preserving cast to u64 for packing into the u128 key.
/// Signed types are cast via their unsigned counterpart to preserve bit patterns.
trait PackAsU64 {
    fn as_pack_u64(self) -> u64;
}

macro_rules! impl_pack_as_u64_unsigned {
    ($($t:ty),+) => {
        $(impl PackAsU64 for $t {
            #[inline(always)]
            fn as_pack_u64(self) -> u64 { self as u64 }
        })+
    };
}

macro_rules! impl_pack_as_u64_signed {
    ($signed:ty, $unsigned:ty) => {
        impl PackAsU64 for $signed {
            #[inline(always)]
            fn as_pack_u64(self) -> u64 {
                (self as $unsigned) as u64
            }
        }
    };
}

impl_pack_as_u64_unsigned!(u8, u16, u32, u64);
impl_pack_as_u64_signed!(i8, u8);
impl_pack_as_u64_signed!(i16, u16);
impl_pack_as_u64_signed!(i32, u32);
impl_pack_as_u64_signed!(i64, u64);

/// Pack a primitive column into the packed buffer using batch-level dispatch.
/// The inner loop is a tight typed loop with no branching on data type.
fn pack_primitive<T: ArrowPrimitiveType>(
    array: &ArrayRef,
    col_desc: &PackedColumn,
    packed: &mut [u128],
) where
    T::Native: PackAsU64,
{
    let arr = array.as_primitive::<T>();
    let bit_offset = col_desc.bit_offset;

    if col_desc.nullable && arr.null_count() > 0 {
        let null_bit_offset = col_desc.null_bit_offset();
        let null_flag = 1u128 << null_bit_offset;
        for (i, p) in packed.iter_mut().enumerate() {
            if arr.is_null(i) {
                *p |= null_flag;
            } else {
                // SAFETY: i is within bounds and not null
                let val: u64 = unsafe { arr.value_unchecked(i) }.as_pack_u64();
                *p |= (val as u128) << bit_offset;
            }
        }
    } else {
        // Fast path: no nulls to check
        for (i, p) in packed.iter_mut().enumerate() {
            // SAFETY: i is within bounds
            let val: u64 = unsafe { arr.value_unchecked(i) }.as_pack_u64();
            *p |= (val as u128) << bit_offset;
        }
    }
}

/// Pack a boolean column into the packed buffer.
fn pack_boolean(arr: &BooleanArray, col_desc: &PackedColumn, packed: &mut [u128]) {
    let bit_offset = col_desc.bit_offset;

    if col_desc.nullable && arr.null_count() > 0 {
        let null_bit_offset = col_desc.null_bit_offset();
        let null_flag = 1u128 << null_bit_offset;
        for (i, p) in packed.iter_mut().enumerate() {
            if arr.is_null(i) {
                *p |= null_flag;
            } else {
                *p |= (arr.value(i) as u128) << bit_offset;
            }
        }
    } else {
        for (i, p) in packed.iter_mut().enumerate() {
            *p |= (arr.value(i) as u128) << bit_offset;
        }
    }
}

/// Unpack a single column from packed u128 values back into an Arrow array.
fn unpack_column(col_desc: &PackedColumn, values: &[u128]) -> ArrayRef {
    let nullable = col_desc.nullable;

    macro_rules! unpack_primitive {
        ($arrow_type:ty, $native_type:ty) => {{
            let mask = if col_desc.value_bits == 64 {
                u64::MAX
            } else {
                (1u64 << col_desc.value_bits) - 1
            };
            if nullable {
                let null_bit = col_desc.null_bit_offset();
                let mut nulls = NullBufferBuilder::new(values.len());
                let vals: Vec<$native_type> = values
                    .iter()
                    .map(|packed| {
                        if (*packed >> null_bit) & 1 == 1 {
                            nulls.append_null();
                            <$native_type>::default()
                        } else {
                            nulls.append_non_null();
                            ((*packed >> col_desc.bit_offset) as u64 & mask)
                                as $native_type
                        }
                    })
                    .collect();
                let array = arrow::array::PrimitiveArray::<$arrow_type>::new(
                    vals.into(),
                    nulls.finish(),
                );
                Arc::new(array.with_data_type(col_desc.data_type.clone())) as ArrayRef
            } else {
                let vals: Vec<$native_type> = values
                    .iter()
                    .map(|packed| {
                        ((*packed >> col_desc.bit_offset) as u64 & mask) as $native_type
                    })
                    .collect();
                let array =
                    arrow::array::PrimitiveArray::<$arrow_type>::new(vals.into(), None);
                Arc::new(array.with_data_type(col_desc.data_type.clone())) as ArrayRef
            }
        }};
    }

    match &col_desc.data_type {
        DataType::Boolean => {
            if nullable {
                let null_bit = col_desc.null_bit_offset();
                let mut nulls = NullBufferBuilder::new(values.len());
                let bools: arrow::buffer::BooleanBuffer =
                    arrow::buffer::BooleanBuffer::collect_bool(values.len(), |i| {
                        let packed = values[i];
                        if (packed >> null_bit) & 1 == 1 {
                            nulls.append_null();
                            false
                        } else {
                            nulls.append_non_null();
                            ((packed >> col_desc.bit_offset) & 1) == 1
                        }
                    });
                Arc::new(BooleanArray::new(bools, nulls.finish()))
            } else {
                let bools: arrow::buffer::BooleanBuffer =
                    arrow::buffer::BooleanBuffer::collect_bool(values.len(), |i| {
                        (values[i] >> col_desc.bit_offset) & 1 == 1
                    });
                Arc::new(BooleanArray::new(bools, None))
            }
        }
        DataType::Int8 => unpack_primitive!(Int8Type, i8),
        DataType::Int16 => unpack_primitive!(Int16Type, i16),
        DataType::Int32 => unpack_primitive!(Int32Type, i32),
        DataType::Int64 => unpack_primitive!(Int64Type, i64),
        DataType::UInt8 => unpack_primitive!(UInt8Type, u8),
        DataType::UInt16 => unpack_primitive!(UInt16Type, u16),
        DataType::UInt32 => unpack_primitive!(UInt32Type, u32),
        DataType::UInt64 => unpack_primitive!(UInt64Type, u64),
        DataType::Date32 => unpack_primitive!(Date32Type, i32),
        DataType::Date64 => unpack_primitive!(Date64Type, i64),
        DataType::Time32(TimeUnit::Second) => {
            unpack_primitive!(Time32SecondType, i32)
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            unpack_primitive!(Time32MillisecondType, i32)
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            unpack_primitive!(Time64MicrosecondType, i64)
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            unpack_primitive!(Time64NanosecondType, i64)
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            unpack_primitive!(TimestampSecondType, i64)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            unpack_primitive!(TimestampMillisecondType, i64)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            unpack_primitive!(TimestampMicrosecondType, i64)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            unpack_primitive!(TimestampNanosecondType, i64)
        }
        _ => unreachable!(
            "Unsupported type in packed group by: {:?}",
            col_desc.data_type
        ),
    }
}

impl GroupValues for GroupValuesPacked {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), self.columns.len());
        let n_rows = cols[0].len();
        groups.clear();

        // Pack all rows — type dispatch happens once per column, not per row
        self.pack_batch(cols, n_rows);

        for row in 0..n_rows {
            let packed = self.packed_buffer[row];
            let hash = self.random_state.hash_one(packed);

            let insert = self.map.entry(
                hash,
                |&(g, h)| unsafe { hash == h && *self.values.get_unchecked(g) == packed },
                |&(_, h)| h,
            );

            let group_id = match insert {
                hashbrown::hash_table::Entry::Occupied(o) => o.get().0,
                hashbrown::hash_table::Entry::Vacant(v) => {
                    let g = self.values.len();
                    v.insert((g, hash));
                    self.values.push(packed);
                    g
                }
            };
            groups.push(group_id);
        }

        Ok(())
    }

    fn size(&self) -> usize {
        self.map.capacity() * size_of::<(usize, u64)>()
            + self.values.allocated_size()
            + self.packed_buffer.allocated_size()
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        match emit_to {
            EmitTo::All => {
                let result = self.unpack_values(&self.values);
                self.map.clear();
                self.values.clear();
                Ok(result)
            }
            EmitTo::First(n) => {
                let first_n: Vec<u128> = self.values[..n].to_vec();
                let result = self.unpack_values(&first_n);

                // Shift remaining values
                let mut remaining = self.values.split_off(n);
                std::mem::swap(&mut self.values, &mut remaining);

                // Update map indices
                self.map.retain(|entry| match entry.0.checked_sub(n) {
                    Some(sub) => {
                        entry.0 = sub;
                        true
                    }
                    None => false,
                });

                Ok(result)
            }
        }
    }

    fn clear_shrink(&mut self, num_rows: usize) {
        self.values.clear();
        self.values.shrink_to(num_rows);
        self.packed_buffer.clear();
        self.packed_buffer.shrink_to(num_rows);
        self.map.clear();
        self.map.shrink_to(num_rows, |_| 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int8Array, Int32Array, UInt16Array};
    use arrow::datatypes::{Field, Schema};

    #[test]
    fn test_can_pack_schema() {
        // Two Int32 nullable columns = 2 * 33 = 66 bits, fits in 128
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]));
        assert!(can_pack_schema(&schema));

        // Single column - not eligible (handled by single-column specialization)
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        assert!(!can_pack_schema(&schema));

        // Two nullable Int64 columns = 2 * 65 = 130 bits, doesn't fit
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
        ]));
        assert!(!can_pack_schema(&schema));

        // Two non-nullable Int64 columns = 2 * 64 = 128 bits, fits!
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        assert!(can_pack_schema(&schema));

        // One nullable + one non-nullable Int64 = 65 + 64 = 129 bits, doesn't fit
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, false),
        ]));
        assert!(!can_pack_schema(&schema));

        // Int64 + Int32 nullable = 65 + 33 = 98 bits, fits
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int32, true),
        ]));
        assert!(can_pack_schema(&schema));

        // String column - not supported
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]));
        assert!(!can_pack_schema(&schema));
    }

    #[test]
    fn test_two_non_nullable_int64() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        assert!(can_pack_schema(&schema));
        let mut gv = GroupValuesPacked::try_new(&schema)?;

        let col_a: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![
            100, 200, 100, 300, 200,
        ]));
        let col_b: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![
            1000, 2000, 1000, 3000, 2000,
        ]));

        let mut groups = Vec::new();
        gv.intern(&[col_a, col_b], &mut groups)?;

        assert_eq!(groups, vec![0, 1, 0, 2, 1]);
        assert_eq!(gv.len(), 3);

        let emitted = gv.emit(EmitTo::All)?;
        assert_eq!(emitted.len(), 2);

        let a = emitted[0].as_primitive::<Int64Type>();
        let b = emitted[1].as_primitive::<Int64Type>();
        assert_eq!(a.values().as_ref(), &[100i64, 200, 300]);
        assert_eq!(b.values().as_ref(), &[1000i64, 2000, 3000]);
        // Non-nullable columns should produce no null buffer
        assert!(a.nulls().is_none());
        assert!(b.nulls().is_none());

        Ok(())
    }

    #[test]
    fn test_basic_intern_and_emit() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int8, true),
        ]));
        let mut gv = GroupValuesPacked::try_new(&schema)?;

        let col_a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 1, 3, 2]));
        let col_b: ArrayRef = Arc::new(Int8Array::from(vec![10, 20, 10, 30, 20]));

        let mut groups = Vec::new();
        gv.intern(&[col_a, col_b], &mut groups)?;

        // (1,10), (2,20), (1,10), (3,30), (2,20)
        // Groups: 0, 1, 0, 2, 1
        assert_eq!(groups, vec![0, 1, 0, 2, 1]);
        assert_eq!(gv.len(), 3);

        let emitted = gv.emit(EmitTo::All)?;
        assert_eq!(emitted.len(), 2);

        let a = emitted[0].as_primitive::<Int32Type>();
        let b = emitted[1].as_primitive::<Int8Type>();
        assert_eq!(a.values().as_ref(), &[1, 2, 3]);
        assert_eq!(b.values().as_ref(), &[10, 20, 30]);

        Ok(())
    }

    #[test]
    fn test_with_nulls() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::UInt16, true),
        ]));
        let mut gv = GroupValuesPacked::try_new(&schema)?;

        let col_a: ArrayRef =
            Arc::new(Int32Array::from(vec![Some(1), None, Some(1), None]));
        let col_b: ArrayRef =
            Arc::new(UInt16Array::from(vec![Some(10), Some(20), None, Some(20)]));

        let mut groups = Vec::new();
        gv.intern(&[col_a, col_b], &mut groups)?;

        // (1, 10), (NULL, 20), (1, NULL), (NULL, 20)
        // All different except last two NULLs with 20
        assert_eq!(groups, vec![0, 1, 2, 1]);
        assert_eq!(gv.len(), 3);

        let emitted = gv.emit(EmitTo::All)?;
        let a = emitted[0].as_primitive::<Int32Type>();
        let b = emitted[1].as_primitive::<UInt16Type>();

        assert_eq!(a.value(0), 1);
        assert!(a.is_null(1));
        assert_eq!(a.value(2), 1);

        assert_eq!(b.value(0), 10);
        assert_eq!(b.value(1), 20);
        assert!(b.is_null(2));

        Ok(())
    }

    #[test]
    fn test_emit_first_n() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int8, true),
        ]));
        let mut gv = GroupValuesPacked::try_new(&schema)?;

        let col_a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let col_b: ArrayRef = Arc::new(Int8Array::from(vec![10, 20, 30]));

        let mut groups = Vec::new();
        gv.intern(&[col_a, col_b], &mut groups)?;
        assert_eq!(groups, vec![0, 1, 2]);

        // Emit first 2
        let emitted = gv.emit(EmitTo::First(2))?;
        let a = emitted[0].as_primitive::<Int32Type>();
        assert_eq!(a.values().as_ref(), &[1, 2]);

        assert_eq!(gv.len(), 1);

        Ok(())
    }
}
