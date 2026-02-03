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

use crate::expressions::case::literal_lookup_table::WhenLiteralIndexMap;
use arrow::array::{
    Array, ArrayRef, ArrowNativeTypeOp, ArrowPrimitiveType, AsArray, PrimitiveArray,
};
use arrow::datatypes::{DataType, IntervalDayTime, IntervalMonthDayNano, i256};
use datafusion_common::{HashMap, ScalarValue, internal_err};
use half::f16;
use std::fmt::Debug;
use std::hash::Hash;

#[derive(Clone)]
pub(super) struct PrimitiveIndexMap<T>
where
    T: ArrowPrimitiveType,
    T::Native: ToHashableKey,
{
    data_type: DataType,
    /// Literal value to map index
    ///
    /// If searching this map becomes a bottleneck consider using linear map implementations for small hashmaps
    map: HashMap<<T::Native as ToHashableKey>::HashableKey, u32>,

    /// Optional direct mapping for dense integer keys
    /// (min_value, lookup_vector) where lookup_vector[value - min_value] = branch_index
    dense_map: Option<(i128, Vec<u32>)>,
}

impl<T> Debug for PrimitiveIndexMap<T>
where
    T: ArrowPrimitiveType,
    T::Native: ToHashableKey,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimitiveIndexMap")
            .field("map", &self.map)
            .field("dense_map", &self.dense_map)
            .finish()
    }
}

impl<T> PrimitiveIndexMap<T>
where
    T: ArrowPrimitiveType,
    T::Native: ToHashableKey,
{
    /// Try creating a new lookup table from the given literals and else index.
    /// The index of each literal in the vector is used as the mapped value in the lookup table.
    ///
    /// `literals` are guaranteed to be unique and non-nullable
    pub(super) fn try_new(
        unique_non_null_literals: Vec<ScalarValue>,
    ) -> datafusion_common::Result<Self> {
        let input = ScalarValue::iter_to_array(unique_non_null_literals)?;

        // Literals are guaranteed to not contain nulls
        if input.null_count() > 0 {
            return internal_err!("Literal values for WHEN clauses cannot contain nulls");
        }

        let primitive_array = input.as_primitive::<T>();
        let primitive_values = primitive_array.values();

        // Optional dense mapping for small integer ranges
        let mut min_val = i128::MAX;
        let mut max_val = i128::MIN;
        let mut all_integers = true;

        for &val in primitive_values {
            if let Some(i) = val.to_i128() {
                if i < min_val {
                    min_val = i;
                }
                if i > max_val {
                    max_val = i;
                }
            } else {
                all_integers = false;
                break;
            }
        }

        let mut map: HashMap<_, _> = HashMap::new();
        let mut dense_map = None;
        if all_integers && !primitive_values.is_empty() {
            let range = max_val - min_val;
            // 2048 is an arbitrary limit for the dense map size
            if range >= 0 && range < 2048 {
                let mut v = vec![u32::MAX; (range as usize) + 1];
                for (branch_idx, &val) in primitive_values.iter().enumerate() {
                    let idx = (val.to_i128().unwrap() - min_val) as usize;
                    v[idx] = branch_idx as u32;
                }
                dense_map = Some((min_val, v));
            }
        } else {
            map = primitive_values
                .iter()
                .enumerate()
                .map(|(branch_idx, &val)| (val.into_hashable_key(), branch_idx as u32))
                .collect();
        }

        Ok(Self {
            map,
            data_type: input.data_type().clone(),
            dense_map,
        })
    }

    fn map_primitive_array_to_when_indices(
        &self,
        array: &PrimitiveArray<T>,
        else_index: u32,
    ) -> datafusion_common::Result<Vec<u32>> {
        if let Some((min_val, dense_map)) = &self.dense_map {
            if array.null_count() == 0 {
                return Ok(array
                    .values()
                    .iter()
                    .map(|value| {
                        let v = value.to_i128().unwrap();
                        let idx = v.checked_sub(*min_val).unwrap();
                        if idx >= 0 && (idx as usize) < dense_map.len() {
                            let branch_idx = dense_map[idx as usize];
                            if branch_idx != u32::MAX {
                                return branch_idx;
                            }
                        }
                        else_index
                    })
                    .collect());
            }
            return Ok(array
                .iter()
                .map(|value| {
                    value
                        .and_then(|v| v.to_i128())
                        .and_then(|v| {
                            let idx = v.checked_sub(*min_val)?;
                            if idx >= 0 && (idx as usize) < dense_map.len() {
                                let branch_idx = dense_map[idx as usize];
                                if branch_idx != u32::MAX {
                                    return Some(branch_idx);
                                }
                            }
                            None
                        })
                        .unwrap_or(else_index)
                })
                .collect());
        }

        let indices = array
            .into_iter()
            .map(|value| match value {
                Some(value) => self
                    .map
                    .get(&value.into_hashable_key())
                    .copied()
                    .unwrap_or(else_index),

                None => else_index,
            })
            .collect::<Vec<u32>>();

        Ok(indices)
    }
}

impl<T> WhenLiteralIndexMap for PrimitiveIndexMap<T>
where
    T: ArrowPrimitiveType,
    T::Native: ToHashableKey,
{
    fn map_to_when_indices(
        &self,
        array: &ArrayRef,
        else_index: u32,
    ) -> datafusion_common::Result<Vec<u32>> {
        match array.data_type() {
            dt if dt == &self.data_type => {
                let primitive_array = array.as_primitive::<T>();

                self.map_primitive_array_to_when_indices(primitive_array, else_index)
            }
            // We support dictionary primitive array as we create the lookup table in `CaseWhen` expression
            // creation when we don't know the schema, so we may receive dictionary encoded primitive arrays at execution time.
            DataType::Dictionary(_, value_type)
                if value_type.as_ref() == &self.data_type =>
            {
                // Cast here to simplify the implementation.
                let converted = arrow::compute::cast(array.as_ref(), &self.data_type)?;
                self.map_primitive_array_to_when_indices(
                    converted.as_primitive::<T>(),
                    else_index,
                )
            }
            _ => internal_err!(
                "PrimitiveIndexMap expected array of type {:?} but got {:?}",
                self.data_type,
                array.data_type()
            ),
        }
    }
}

// TODO - We need to port it to arrow so that it can be reused in other places

/// Trait that help convert a value to a key that is hashable and equatable
/// This is needed as some types like f16/f32/f64 do not implement Hash/Eq directly
pub(super) trait ToHashableKey: ArrowNativeTypeOp {
    /// The type that is hashable and equatable
    /// It must be an Arrow native type but it NOT GUARANTEED to be the same as Self
    /// this is just a helper trait so you can reuse the same code for all arrow native types
    type HashableKey: Hash + Eq + Debug + Clone + Copy + Send + Sync;

    /// Converts self to a hashable key
    /// the result of this value can be used as the key in hash maps/sets
    fn into_hashable_key(self) -> Self::HashableKey;

    /// Convert to i128 if it's an integer type, else None
    fn to_i128(self) -> Option<i128>;
}

macro_rules! impl_to_hashable_key {
    (@single_already_hashable | $t:ty, $to_i128:expr) => {
        impl ToHashableKey for $t {
            type HashableKey = $t;

            #[inline]
            fn into_hashable_key(self) -> Self::HashableKey {
                self
            }

            #[inline]
            fn to_i128(self) -> Option<i128> {
                $to_i128(self)
            }
        }
    };
    (@already_hashable | $($t:ty, $to_i128:expr),+ $(,)?) => {
        $(
            impl_to_hashable_key!(@single_already_hashable | $t, $to_i128);
        )+
    };
    (@float | $t:ty => $hashable:ty) => {
        impl ToHashableKey for $t {
            type HashableKey = $hashable;

            #[inline]
            fn into_hashable_key(self) -> Self::HashableKey {
                self.to_bits()
            }

            #[inline]
            fn to_i128(self) -> Option<i128> {
                None
            }
        }
    };
}

impl_to_hashable_key!(
    @already_hashable |
    i8, |x: i8| Some(x as i128),
    i16, |x: i16| Some(x as i128),
    i32, |x: i32| Some(x as i128),
    i64, |x: i64| Some(x as i128),
    i128, |x: i128| Some(x),
    i256, |_x: i256| None,
    u8, |x: u8| Some(x as i128),
    u16, |x: u16| Some(x as i128),
    u32, |x: u32| Some(x as i128),
    u64, |x: u64| Some(x as i128),
    IntervalDayTime, |_x: IntervalDayTime| None,
    IntervalMonthDayNano, |_x: IntervalMonthDayNano| None
);
impl_to_hashable_key!(@float | f16 => u16);
impl_to_hashable_key!(@float | f32 => u32);
impl_to_hashable_key!(@float | f64 => u64);

#[cfg(test)]
mod tests {
    use super::ToHashableKey;
    use arrow::array::downcast_primitive;

    // This test ensure that all arrow primitive types implement ToHashableKey
    // otherwise the code will not compile
    #[test]
    fn should_implement_to_hashable_key_for_all_primitives() {
        #[derive(Debug, Default)]
        struct ExampleSet<T>
        where
            T: arrow::datatypes::ArrowPrimitiveType,
            T::Native: ToHashableKey,
        {
            _map: std::collections::HashSet<<T::Native as ToHashableKey>::HashableKey>,
        }

        macro_rules! create_matching_set {
            ($t:ty) => {{
                let _lookup_table = ExampleSet::<$t> {
                    _map: Default::default(),
                };

                return;
            }};
        }

        let data_type = arrow::datatypes::DataType::Float16;

        downcast_primitive! {
            data_type => (create_matching_set),
            _ => panic!("not implemented for {data_type}"),
        }
    }
}
