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
//! packed into a single u64 or u128 key.
//!
//! When all group-by columns are small fixed-width types (integers, booleans, dates, etc.)
//! whose combined bit-width fits within 64 or 128 bits, we pack all columns into a single
//! key. This eliminates per-column hashing, per-column equality comparison,
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
use std::hash::Hash;
use std::mem::size_of;
use std::sync::Arc;

// ============================================================================
// PackedKey trait — abstracts over u64 and u128 as packed key storage
// ============================================================================

/// A packed key type that can store multiple column values in a single integer.
pub trait PackedKey:
    Copy + Default + Eq + Hash + Send + Sync + std::fmt::Debug + 'static
{
    fn from_u64(val: u64, shift: usize) -> Self;
    fn or(self, other: Self) -> Self;
    fn shr_to_u64(self, shift: usize) -> u64;
    fn bit(self, pos: usize) -> bool;
    fn set_bit(self, pos: usize) -> Self;
}

impl PackedKey for u64 {
    #[inline(always)]
    fn from_u64(val: u64, shift: usize) -> Self {
        val << shift
    }
    #[inline(always)]
    fn or(self, other: Self) -> Self {
        self | other
    }
    #[inline(always)]
    fn shr_to_u64(self, shift: usize) -> u64 {
        self >> shift
    }
    #[inline(always)]
    fn bit(self, pos: usize) -> bool {
        (self >> pos) & 1 == 1
    }
    #[inline(always)]
    fn set_bit(self, pos: usize) -> Self {
        self | (1u64 << pos)
    }
}

impl PackedKey for u128 {
    #[inline(always)]
    fn from_u64(val: u64, shift: usize) -> Self {
        (val as u128) << shift
    }
    #[inline(always)]
    fn or(self, other: Self) -> Self {
        self | other
    }
    #[inline(always)]
    fn shr_to_u64(self, shift: usize) -> u64 {
        (self >> shift) as u64
    }
    #[inline(always)]
    fn bit(self, pos: usize) -> bool {
        (self >> pos) & 1 == 1
    }
    #[inline(always)]
    fn set_bit(self, pos: usize) -> Self {
        self | (1u128 << pos)
    }
}

// ============================================================================
// Schema analysis
// ============================================================================

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
        _ => None,
    }
}

/// Compute the total packed bit width for a schema.
/// Returns None if any column type is unsupported.
fn total_packed_bits(schema: &SchemaRef) -> Option<usize> {
    schema
        .fields()
        .iter()
        .map(|f| {
            value_bit_width(f.data_type())
                .map(|bits| bits + if f.is_nullable() { 1 } else { 0 })
        })
        .sum()
}

/// The packed width tier for a schema, if eligible.
/// Returns None if the schema has < 2 columns or doesn't fit.
pub enum PackedWidth {
    U64,
    U128,
}

/// Returns the packed width for the schema, or None if not eligible.
pub fn packed_width(schema: &SchemaRef) -> Option<PackedWidth> {
    if schema.fields().len() < 2 {
        return None;
    }
    let bits = total_packed_bits(schema)?;
    if bits <= 64 {
        Some(PackedWidth::U64)
    } else if bits <= 128 {
        Some(PackedWidth::U128)
    } else {
        None
    }
}

// ============================================================================
// Column layout
// ============================================================================

/// Description of how a single column is packed into the key.
#[derive(Debug, Clone)]
struct PackedColumn {
    bit_offset: usize,
    value_bits: usize,
    nullable: bool,
    data_type: DataType,
}

impl PackedColumn {
    fn null_bit_offset(&self) -> usize {
        debug_assert!(self.nullable);
        self.bit_offset + self.value_bits
    }
}

fn build_column_layout(schema: &SchemaRef) -> Result<Vec<PackedColumn>> {
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

    Ok(columns)
}

// ============================================================================
// Bit-preserving cast to u64
// ============================================================================

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

// ============================================================================
// Batch-level packing (generic over K)
// ============================================================================

/// Pack a primitive column into the packed buffer. Type dispatch happens once
/// at the batch level; the inner loop is a tight monomorphized loop.
fn pack_primitive<T: ArrowPrimitiveType, K: PackedKey>(
    array: &ArrayRef,
    col_desc: &PackedColumn,
    packed: &mut [K],
) where
    T::Native: PackAsU64,
{
    let arr = array.as_primitive::<T>();
    let bit_offset = col_desc.bit_offset;

    if col_desc.nullable && arr.null_count() > 0 {
        let null_bit_offset = col_desc.null_bit_offset();
        for (i, p) in packed.iter_mut().enumerate() {
            if arr.is_null(i) {
                *p = p.set_bit(null_bit_offset);
            } else {
                let val: u64 = unsafe { arr.value_unchecked(i) }.as_pack_u64();
                *p = p.or(K::from_u64(val, bit_offset));
            }
        }
    } else {
        for (i, p) in packed.iter_mut().enumerate() {
            let val: u64 = unsafe { arr.value_unchecked(i) }.as_pack_u64();
            *p = p.or(K::from_u64(val, bit_offset));
        }
    }
}

fn pack_boolean<K: PackedKey>(
    arr: &BooleanArray,
    col_desc: &PackedColumn,
    packed: &mut [K],
) {
    let bit_offset = col_desc.bit_offset;

    if col_desc.nullable && arr.null_count() > 0 {
        let null_bit_offset = col_desc.null_bit_offset();
        for (i, p) in packed.iter_mut().enumerate() {
            if arr.is_null(i) {
                *p = p.set_bit(null_bit_offset);
            } else {
                *p = p.or(K::from_u64(arr.value(i) as u64, bit_offset));
            }
        }
    } else {
        for (i, p) in packed.iter_mut().enumerate() {
            *p = p.or(K::from_u64(arr.value(i) as u64, bit_offset));
        }
    }
}

/// Dispatch packing for all columns in a batch into the packed buffer.
fn pack_batch_columns<K: PackedKey>(
    columns: &[PackedColumn],
    cols: &[ArrayRef],
    packed: &mut [K],
) {
    for (col_desc, array) in columns.iter().zip(cols.iter()) {
        match &col_desc.data_type {
            DataType::Boolean => {
                let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                pack_boolean(arr, col_desc, packed);
            }
            DataType::Int8 => pack_primitive::<Int8Type, K>(array, col_desc, packed),
            DataType::Int16 => pack_primitive::<Int16Type, K>(array, col_desc, packed),
            DataType::Int32 => pack_primitive::<Int32Type, K>(array, col_desc, packed),
            DataType::Int64 => pack_primitive::<Int64Type, K>(array, col_desc, packed),
            DataType::UInt8 => pack_primitive::<UInt8Type, K>(array, col_desc, packed),
            DataType::UInt16 => pack_primitive::<UInt16Type, K>(array, col_desc, packed),
            DataType::UInt32 => pack_primitive::<UInt32Type, K>(array, col_desc, packed),
            DataType::UInt64 => pack_primitive::<UInt64Type, K>(array, col_desc, packed),
            DataType::Date32 => pack_primitive::<Date32Type, K>(array, col_desc, packed),
            DataType::Date64 => pack_primitive::<Date64Type, K>(array, col_desc, packed),
            DataType::Time32(TimeUnit::Second) => {
                pack_primitive::<Time32SecondType, K>(array, col_desc, packed)
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                pack_primitive::<Time32MillisecondType, K>(array, col_desc, packed)
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                pack_primitive::<Time64MicrosecondType, K>(array, col_desc, packed)
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                pack_primitive::<Time64NanosecondType, K>(array, col_desc, packed)
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                pack_primitive::<TimestampSecondType, K>(array, col_desc, packed)
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                pack_primitive::<TimestampMillisecondType, K>(array, col_desc, packed)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                pack_primitive::<TimestampMicrosecondType, K>(array, col_desc, packed)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                pack_primitive::<TimestampNanosecondType, K>(array, col_desc, packed)
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// Unpacking (generic over K)
// ============================================================================

fn unpack_column<K: PackedKey>(col_desc: &PackedColumn, values: &[K]) -> ArrayRef {
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
                        if packed.bit(null_bit) {
                            nulls.append_null();
                            <$native_type>::default()
                        } else {
                            nulls.append_non_null();
                            (packed.shr_to_u64(col_desc.bit_offset) & mask)
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
                        (packed.shr_to_u64(col_desc.bit_offset) & mask) as $native_type
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
                let bools =
                    arrow::buffer::BooleanBuffer::collect_bool(values.len(), |i| {
                        if values[i].bit(null_bit) {
                            nulls.append_null();
                            false
                        } else {
                            nulls.append_non_null();
                            values[i].bit(col_desc.bit_offset)
                        }
                    });
                Arc::new(BooleanArray::new(bools, nulls.finish()))
            } else {
                let bools =
                    arrow::buffer::BooleanBuffer::collect_bool(values.len(), |i| {
                        values[i].bit(col_desc.bit_offset)
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

// ============================================================================
// GroupValuesPacked<K> — the main struct, generic over key width
// ============================================================================

/// A [`GroupValues`] that packs multiple small integer/boolean columns into a
/// single `K` key (u64 or u128) for fast hashing and comparison.
pub struct GroupValuesPacked<K: PackedKey> {
    columns: Vec<PackedColumn>,
    map: HashTable<(usize, u64)>,
    values: Vec<K>,
    packed_buffer: Vec<K>,
    random_state: RandomState,
}

impl<K: PackedKey> GroupValuesPacked<K> {
    pub fn try_new(schema: &SchemaRef) -> Result<Self> {
        let columns = build_column_layout(schema)?;
        Ok(Self {
            columns,
            map: HashTable::with_capacity(128),
            values: Vec::with_capacity(128),
            packed_buffer: Vec::new(),
            random_state: crate::aggregates::AGGREGATION_HASH_SEED,
        })
    }

    fn pack_batch(&mut self, cols: &[ArrayRef], n_rows: usize) {
        self.packed_buffer.clear();
        self.packed_buffer.resize(n_rows, K::default());
        pack_batch_columns(&self.columns, cols, &mut self.packed_buffer);
    }

    fn unpack_values(&self, values: &[K]) -> Vec<ArrayRef> {
        self.columns
            .iter()
            .map(|col_desc| unpack_column(col_desc, values))
            .collect()
    }
}

impl<K: PackedKey> GroupValues for GroupValuesPacked<K> {
    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        assert_eq!(cols.len(), self.columns.len());
        let n_rows = cols[0].len();
        groups.clear();

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
                let first_n: Vec<K> = self.values[..n].to_vec();
                let result = self.unpack_values(&first_n);

                let mut remaining = self.values.split_off(n);
                std::mem::swap(&mut self.values, &mut remaining);

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
    fn test_packed_width() {
        // Two Int32 nullable = 2*33 = 66 → u128
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]));
        assert!(matches!(packed_width(&schema), Some(PackedWidth::U128)));

        // Two Int16 nullable = 2*17 = 34 → u64
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int16, true),
            Field::new("b", DataType::Int16, true),
        ]));
        assert!(matches!(packed_width(&schema), Some(PackedWidth::U64)));

        // Two non-nullable Int32 = 2*32 = 64 → u64
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        assert!(matches!(packed_width(&schema), Some(PackedWidth::U64)));

        // Two nullable Int32 = 2*33 = 66 → u128
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]));
        assert!(matches!(packed_width(&schema), Some(PackedWidth::U128)));

        // Two non-nullable Int64 = 2*64 = 128 → u128
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        assert!(matches!(packed_width(&schema), Some(PackedWidth::U128)));

        // Two nullable Int64 = 2*65 = 130 → None
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
        ]));
        assert!(packed_width(&schema).is_none());

        // Single column → None
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        assert!(packed_width(&schema).is_none());

        // String → None
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]));
        assert!(packed_width(&schema).is_none());
    }

    #[test]
    fn test_u64_basic() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int16, true),
            Field::new("b", DataType::Int8, true),
        ]));
        let mut gv = GroupValuesPacked::<u64>::try_new(&schema)?;

        let col_a: ArrayRef =
            Arc::new(arrow::array::Int16Array::from(vec![1, 2, 1, 3, 2]));
        let col_b: ArrayRef = Arc::new(Int8Array::from(vec![10, 20, 10, 30, 20]));

        let mut groups = Vec::new();
        gv.intern(&[col_a, col_b], &mut groups)?;

        assert_eq!(groups, vec![0, 1, 0, 2, 1]);
        assert_eq!(gv.len(), 3);

        let emitted = gv.emit(EmitTo::All)?;
        let a = emitted[0].as_primitive::<Int16Type>();
        let b = emitted[1].as_primitive::<Int8Type>();
        assert_eq!(a.values().as_ref(), &[1i16, 2, 3]);
        assert_eq!(b.values().as_ref(), &[10i8, 20, 30]);

        Ok(())
    }

    #[test]
    fn test_u128_basic() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int8, true),
        ]));
        let mut gv = GroupValuesPacked::<u128>::try_new(&schema)?;

        let col_a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 1, 3, 2]));
        let col_b: ArrayRef = Arc::new(Int8Array::from(vec![10, 20, 10, 30, 20]));

        let mut groups = Vec::new();
        gv.intern(&[col_a, col_b], &mut groups)?;

        assert_eq!(groups, vec![0, 1, 0, 2, 1]);
        assert_eq!(gv.len(), 3);

        let emitted = gv.emit(EmitTo::All)?;
        let a = emitted[0].as_primitive::<Int32Type>();
        let b = emitted[1].as_primitive::<Int8Type>();
        assert_eq!(a.values().as_ref(), &[1, 2, 3]);
        assert_eq!(b.values().as_ref(), &[10i8, 20, 30]);

        Ok(())
    }

    #[test]
    fn test_u64_non_nullable_two_int32() -> Result<()> {
        // 2 * 32 = 64 bits, fits in u64
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        assert!(matches!(packed_width(&schema), Some(PackedWidth::U64)));
        let mut gv = GroupValuesPacked::<u64>::try_new(&schema)?;

        let col_a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 1, 3, 2]));
        let col_b: ArrayRef = Arc::new(Int32Array::from(vec![100, 200, 100, 300, 200]));

        let mut groups = Vec::new();
        gv.intern(&[col_a, col_b], &mut groups)?;
        assert_eq!(groups, vec![0, 1, 0, 2, 1]);

        let emitted = gv.emit(EmitTo::All)?;
        let a = emitted[0].as_primitive::<Int32Type>();
        let b = emitted[1].as_primitive::<Int32Type>();
        assert_eq!(a.values().as_ref(), &[1, 2, 3]);
        assert_eq!(b.values().as_ref(), &[100, 200, 300]);
        assert!(a.nulls().is_none());
        assert!(b.nulls().is_none());

        Ok(())
    }

    #[test]
    fn test_u128_two_non_nullable_int64() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        assert!(matches!(packed_width(&schema), Some(PackedWidth::U128)));
        let mut gv = GroupValuesPacked::<u128>::try_new(&schema)?;

        let col_a: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![
            100, 200, 100, 300, 200,
        ]));
        let col_b: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![
            1000, 2000, 1000, 3000, 2000,
        ]));

        let mut groups = Vec::new();
        gv.intern(&[col_a, col_b], &mut groups)?;
        assert_eq!(groups, vec![0, 1, 0, 2, 1]);

        let emitted = gv.emit(EmitTo::All)?;
        let a = emitted[0].as_primitive::<Int64Type>();
        let b = emitted[1].as_primitive::<Int64Type>();
        assert_eq!(a.values().as_ref(), &[100i64, 200, 300]);
        assert_eq!(b.values().as_ref(), &[1000i64, 2000, 3000]);
        assert!(a.nulls().is_none());
        assert!(b.nulls().is_none());

        Ok(())
    }

    #[test]
    fn test_with_nulls() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::UInt16, true),
        ]));
        let mut gv = GroupValuesPacked::<u128>::try_new(&schema)?;

        let col_a: ArrayRef =
            Arc::new(Int32Array::from(vec![Some(1), None, Some(1), None]));
        let col_b: ArrayRef =
            Arc::new(UInt16Array::from(vec![Some(10), Some(20), None, Some(20)]));

        let mut groups = Vec::new();
        gv.intern(&[col_a, col_b], &mut groups)?;
        assert_eq!(groups, vec![0, 1, 2, 1]);

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
        let mut gv = GroupValuesPacked::<u128>::try_new(&schema)?;

        let col_a: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let col_b: ArrayRef = Arc::new(Int8Array::from(vec![10, 20, 30]));

        let mut groups = Vec::new();
        gv.intern(&[col_a, col_b], &mut groups)?;
        assert_eq!(groups, vec![0, 1, 2]);

        let emitted = gv.emit(EmitTo::First(2))?;
        let a = emitted[0].as_primitive::<Int32Type>();
        assert_eq!(a.values().as_ref(), &[1, 2]);
        assert_eq!(gv.len(), 1);

        Ok(())
    }
}
