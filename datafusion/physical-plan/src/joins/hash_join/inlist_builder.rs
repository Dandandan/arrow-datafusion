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

//! Utilities for building InList expressions from hash join build side data

use std::sync::Arc;

use arrow::array::{ArrayRef, StructArray};
use arrow::datatypes::{Field, FieldRef, Fields};
use arrow_schema::DataType;
use datafusion_common::Result;

pub(super) fn build_struct_fields(data_types: &[DataType]) -> Result<Fields> {
    data_types
        .iter()
        .enumerate()
        .map(|(i, dt)| Ok(Field::new(format!("c{i}"), dt.clone(), true)))
        .collect()
}

/// Builds InList values from per-batch join key column arrays.
///
/// `values_per_batch` is indexed as `values_per_batch[batch_idx][key_idx]`.
///
/// If there is a single key column, this concatenates the per-batch arrays into a single flat array.
/// If there are multiple key columns, this builds per-batch StructArrays and concatenates them.
///
/// Note that this function does not deduplicate values - deduplication will happen later
/// when building an InList expression from this array via `InListExpr::try_new_from_array`.
pub(super) fn build_struct_inlist_values(
    values_per_batch: &[Vec<ArrayRef>],
) -> Result<Option<ArrayRef>> {
    if values_per_batch.is_empty() || values_per_batch[0].is_empty() {
        return Ok(None);
    }

    let num_keys = values_per_batch[0].len();

    if num_keys == 1 {
        // Single column: concat per-batch arrays directly
        let arrays: Vec<&dyn arrow::array::Array> = values_per_batch
            .iter()
            .map(|keys| keys[0].as_ref())
            .collect();
        let concatenated = arrow::compute::concat(&arrays)?;
        Ok(Some(concatenated))
    } else {
        // Multi-column: build per-batch StructArrays (zero-copy wrap), then concat
        let data_types: Vec<DataType> = values_per_batch[0]
            .iter()
            .map(|arr| arr.data_type().clone())
            .collect();
        let fields = build_struct_fields(&data_types)?;

        let struct_arrays: Vec<ArrayRef> = values_per_batch
            .iter()
            .map(|keys| {
                let arrays_with_fields: Vec<(FieldRef, ArrayRef)> =
                    fields.iter().cloned().zip(keys.iter().cloned()).collect();
                Arc::new(StructArray::from(arrays_with_fields)) as ArrayRef
            })
            .collect();

        let refs: Vec<&dyn arrow::array::Array> =
            struct_arrays.iter().map(|a| a.as_ref()).collect();
        let concatenated = arrow::compute::concat(&refs)?;
        Ok(Some(concatenated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        DictionaryArray, Int8Array, Int32Array, StringArray, StringDictionaryBuilder,
    };
    use arrow_schema::DataType;
    use std::sync::Arc;

    #[test]
    fn test_build_single_column_inlist_array() {
        let array = Arc::new(Int32Array::from(vec![1, 2, 3, 2, 1])) as ArrayRef;
        // Single batch with single key column
        let batches = vec![vec![Arc::clone(&array)]];
        let result = build_struct_inlist_values(&batches).unwrap().unwrap();

        assert!(array.eq(&result));
    }

    #[test]
    fn test_build_single_column_multi_batch() {
        let array1 = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;
        let array2 = Arc::new(Int32Array::from(vec![2, 1])) as ArrayRef;
        // Two batches with single key column
        let batches = vec![vec![Arc::clone(&array1)], vec![Arc::clone(&array2)]];
        let result = build_struct_inlist_values(&batches).unwrap().unwrap();

        let expected = Arc::new(Int32Array::from(vec![1, 2, 3, 2, 1])) as ArrayRef;
        assert!(expected.eq(&result));
    }

    #[test]
    fn test_build_multi_column_inlist() {
        let array1 = Arc::new(Int32Array::from(vec![1, 2, 3, 2, 1])) as ArrayRef;
        let array2 =
            Arc::new(StringArray::from(vec!["a", "b", "c", "b", "a"])) as ArrayRef;

        // Single batch with two key columns
        let batches = vec![vec![array1, array2]];
        let result = build_struct_inlist_values(&batches).unwrap().unwrap();

        assert_eq!(
            *result.data_type(),
            DataType::Struct(
                build_struct_fields(&[DataType::Int32, DataType::Utf8]).unwrap()
            )
        );
    }

    #[test]
    fn test_build_multi_column_inlist_with_dictionary() {
        let mut builder = StringDictionaryBuilder::<arrow::datatypes::Int8Type>::new();
        builder.append_value("foo");
        builder.append_value("foo");
        builder.append_value("foo");
        let dict_array = Arc::new(builder.finish()) as ArrayRef;

        let int_array = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;

        // Single batch with two key columns
        let batches = vec![vec![dict_array, int_array]];
        let result = build_struct_inlist_values(&batches).unwrap().unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(
            *result.data_type(),
            DataType::Struct(
                build_struct_fields(&[
                    DataType::Dictionary(
                        Box::new(DataType::Int8),
                        Box::new(DataType::Utf8)
                    ),
                    DataType::Int32
                ])
                .unwrap()
            )
        );
    }

    #[test]
    fn test_build_single_column_dictionary_inlist() {
        let keys = Int8Array::from(vec![0i8, 0, 0]);
        let values = Arc::new(StringArray::from(vec!["foo"]));
        let dict_array = Arc::new(DictionaryArray::new(keys, values)) as ArrayRef;

        // Single batch with single key column
        let batches = vec![vec![Arc::clone(&dict_array)]];
        let result = build_struct_inlist_values(&batches).unwrap().unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result.data_type(), dict_array.data_type());
    }
}
