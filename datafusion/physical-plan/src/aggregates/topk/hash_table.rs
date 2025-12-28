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

//! A wrapper around `hashbrown::HashTable` that allows entries to be tracked by index

use crate::aggregates::group_values::HashValue;
use crate::aggregates::topk::heap::Comparable;
use ahash::RandomState;
use arrow::array::types::{IntervalDayTime, IntervalMonthDayNano};
use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, LargeStringArray, PrimitiveArray, StringArray,
    StringViewArray, builder::PrimitiveBuilder, cast::AsArray, downcast_primitive,
};
use arrow::datatypes::{DataType, i256};
use datafusion_common::Result;
use datafusion_common::exec_datafusion_err;
use half::f16;
use hashbrown::hash_table::{Entry, HashTable};
use std::fmt::Debug;
use std::sync::Arc;

/// A "type alias" for Keys which are stored in our map
pub trait KeyType: Clone + Comparable + Debug {}

impl<T> KeyType for T where T: Clone + Comparable + Debug {}

/// An entry in our hash table that:
/// 1. contains the key (ID)
/// 2. contains the value (heap_idx - an index into the corresponding heap)
#[derive(Clone)]
pub struct HashTableItem<ID: KeyType> {
    pub id: ID,
    pub heap_idx: usize,
}

/// A custom wrapper around `hashbrown::HashTable` that:
/// 1. limits the number of entries to the top K
/// 2. Allocates a capacity greater than top K to maintain a low-fill factor and prevent resizing
/// 3. Tracks indexes to allow corresponding heap to refer to entries by index vs hash
/// 4. Catches resize events to allow the corresponding heap to update it's indexes
struct TopKHashTable<ID: KeyType> {
    map: HashTable<HashTableItem<ID>>,
    limit: usize,
}

/// An interface to hide the generic type signature of TopKHashTable behind arrow arrays
pub trait ArrowHashTable {
    fn set_batch(&mut self, ids: ArrayRef);
    fn len(&self) -> usize;
    fn update_heap_idx(&mut self, mapper: &[(usize, usize)]);
    fn heap_idx_at(&self, map_idx: usize) -> usize;
    fn take_all(&mut self, indexes: Vec<usize>) -> ArrayRef;
    fn find_or_insert(
        &mut self,
        row_idx: usize,
        replace_idx: usize,
        map: &mut Vec<(usize, usize)>,
    ) -> (usize, bool);
}

// An implementation of ArrowHashTable for String keys
pub struct StringHashTable {
    owned: ArrayRef,
    map: TopKHashTable<Option<String>>,
    rnd: RandomState,
    data_type: DataType,
}

// An implementation of ArrowHashTable for any `ArrowPrimitiveType` key
struct PrimitiveHashTable<VAL: ArrowPrimitiveType>
where
    Option<<VAL as ArrowPrimitiveType>::Native>: Comparable,
{
    owned: ArrayRef,
    map: TopKHashTable<Option<VAL::Native>>,
    rnd: RandomState,
    kt: DataType,
}

impl StringHashTable {
    pub fn new(limit: usize, data_type: DataType) -> Self {
        let vals: Vec<&str> = Vec::new();
        let owned: ArrayRef = match data_type {
            DataType::Utf8 => Arc::new(StringArray::from(vals)),
            DataType::Utf8View => Arc::new(StringViewArray::from(vals)),
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vals)),
            _ => panic!("Unsupported data type"),
        };

        Self {
            owned,
            map: TopKHashTable::new(limit, limit * 10),
            rnd: RandomState::default(),
            data_type,
        }
    }
}

impl ArrowHashTable for StringHashTable {
    fn set_batch(&mut self, ids: ArrayRef) {
        self.owned = ids;
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn update_heap_idx(&mut self, mapper: &[(usize, usize)]) {
        self.map.update_heap_idx(mapper);
    }

    fn heap_idx_at(&self, map_idx: usize) -> usize {
        self.map.heap_idx_at(map_idx)
    }

    fn take_all(&mut self, indexes: Vec<usize>) -> ArrayRef {
        let ids = self.map.take_all(indexes);
        match self.data_type {
            DataType::Utf8 => Arc::new(StringArray::from(ids)),
            DataType::LargeUtf8 => Arc::new(LargeStringArray::from(ids)),
            DataType::Utf8View => Arc::new(StringViewArray::from(ids)),
            _ => unreachable!(),
        }
    }

    fn find_or_insert(
        &mut self,
        row_idx: usize,
        replace_idx: usize,
        mapper: &mut Vec<(usize, usize)>,
    ) -> (usize, bool) {
        let id = match self.data_type {
            DataType::Utf8 => {
                let ids = self
                    .owned
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("Expected StringArray for DataType::Utf8");
                if ids.is_null(row_idx) {
                    None
                } else {
                    Some(ids.value(row_idx))
                }
            }
            DataType::LargeUtf8 => {
                let ids = self
                    .owned
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("Expected LargeStringArray for DataType::LargeUtf8");
                if ids.is_null(row_idx) {
                    None
                } else {
                    Some(ids.value(row_idx))
                }
            }
            DataType::Utf8View => {
                let ids = self
                    .owned
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .expect("Expected StringViewArray for DataType::Utf8View");
                if ids.is_null(row_idx) {
                    None
                } else {
                    Some(ids.value(row_idx))
                }
            }
            _ => panic!("Unsupported data type"),
        };

        let hash = self.rnd.hash_one(id);
        let hasher = |mi: &HashTableItem<Option<String>>| self.rnd.hash_one(&mi.id);
        let entry = self.map.map.entry(
            hash,
            |mi| id == mi.id.as_ref().map(|id| id.as_str()),
            hasher,
        );

        match entry {
            Entry::Occupied(entry) => (entry.into_mut().heap_idx, false),
            Entry::Vacant(_entry) => {
                // we're full and this is a better value, so remove the worst
                let heap_idx = self.map.remove_if_full(replace_idx);

                // add the new group
                let id = id.map(|id| id.to_string());
                let map_idx = self.map.insert(hash, id, heap_idx, mapper, &self.rnd);
                (map_idx, true)
            }
        }
    }
}

impl<VAL: ArrowPrimitiveType> PrimitiveHashTable<VAL>
where
    Option<<VAL as ArrowPrimitiveType>::Native>: Comparable,
    Option<<VAL as ArrowPrimitiveType>::Native>: HashValue,
{
    pub fn new(limit: usize, kt: DataType) -> Self {
        let owned = Arc::new(
            PrimitiveArray::<VAL>::builder(0)
                .with_data_type(kt.clone())
                .finish(),
        );
        Self {
            owned,
            map: TopKHashTable::new(limit, limit * 10),
            rnd: RandomState::default(),
            kt,
        }
    }
}

impl<VAL: ArrowPrimitiveType> ArrowHashTable for PrimitiveHashTable<VAL>
where
    Option<<VAL as ArrowPrimitiveType>::Native>: Comparable,
    Option<<VAL as ArrowPrimitiveType>::Native>: HashValue,
{
    fn set_batch(&mut self, ids: ArrayRef) {
        self.owned = ids;
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn update_heap_idx(&mut self, mapper: &[(usize, usize)]) {
        self.map.update_heap_idx(mapper);
    }

    fn heap_idx_at(&self, map_idx: usize) -> usize {
        self.map.heap_idx_at(map_idx)
    }

    fn take_all(&mut self, indexes: Vec<usize>) -> ArrayRef {
        let ids = self.map.take_all(indexes);
        let mut builder: PrimitiveBuilder<VAL> =
            PrimitiveArray::builder(ids.len()).with_data_type(self.kt.clone());
        for id in ids.into_iter() {
            match id {
                None => builder.append_null(),
                Some(id) => builder.append_value(id),
            }
        }
        let ids = builder.finish();
        Arc::new(ids)
    }

    fn find_or_insert(
        &mut self,
        row_idx: usize,
        replace_idx: usize,
        mapper: &mut Vec<(usize, usize)>,
    ) -> (usize, bool) {
        let ids = self.owned.as_primitive::<VAL>();
        let id: Option<VAL::Native> = if ids.is_null(row_idx) {
            None
        } else {
            Some(ids.value(row_idx))
        };

        let hash: u64 = id.hash(&self.rnd);
        let hasher = |mi: &HashTableItem<Option<VAL::Native>>| mi.id.hash(&self.rnd);
        let entry = self.map.map.entry(hash, |mi| id == mi.id, hasher);

        match entry {
            Entry::Occupied(entry) => (entry.into_mut().heap_idx, false),
            Entry::Vacant(_entry) => {
                // we're full and this is a better value, so remove the worst
                let heap_idx = self.map.remove_if_full(replace_idx);

                // add the new group
                let map_idx = self.map.insert(hash, id, heap_idx, mapper, &self.rnd);
                (map_idx, true)
            }
        }
    }
}

impl<ID: KeyType> TopKHashTable<ID>
where
    ID: HashValue,
{
    pub fn new(limit: usize, capacity: usize) -> Self {
        Self {
            map: HashTable::with_capacity(capacity),
            limit,
        }
    }

    pub fn heap_idx_at(&self, map_idx: usize) -> usize {
        // Since we can't get at the buckets directly, we have to iterate
        // to find the one we're looking for
        for item in self.map.iter() {
            if item.heap_idx == map_idx {
                return item.heap_idx;
            }
        }
        unreachable!()
    }

    pub fn remove_if_full(&mut self, replace_idx: usize) -> usize {
        if self.map.len() >= self.limit {
            let mut removed = false;
            self.map.retain(|mi| {
                if !removed && mi.heap_idx == replace_idx {
                    removed = true;
                    false
                } else {
                    true
                }
            });
            0 // if full, always replace top node
        } else {
            self.map.len() // if we're not full, always append to end
        }
    }

    fn update_heap_idx(&mut self, mapper: &[(usize, usize)]) {
        for (m, h) in mapper {
            for item in self.map.iter_mut() {
                if item.heap_idx == *m {
                    item.heap_idx = *h;
                }
            }
        }
    }

    pub fn insert(
        &mut self,
        hash: u64,
        id: ID,
        heap_idx: usize,
        mapper: &mut Vec<(usize, usize)>,
        rnd: &RandomState,
    ) -> usize {
        let mi = HashTableItem::new(id, heap_idx);
        let hasher = |item: &HashTableItem<ID>| item.id.hash(rnd);
        let old_cap = self.map.capacity();
        let final_heap_idx = {
            let entry = self.map.insert_unique(hash, mi, hasher);
            entry.into_mut().heap_idx
        };

        let new_cap = self.map.capacity();
        if old_cap != new_cap {
            for item in self.map.iter() {
                let heap_idx = item.heap_idx;
                mapper.push((heap_idx, heap_idx));
            }
        }
        final_heap_idx
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn take_all(&mut self, idxs: Vec<usize>) -> Vec<ID> {
        let idx_set: std::collections::HashSet<usize> = idxs.iter().cloned().collect();
        let mut taken = std::collections::HashMap::with_capacity(idx_set.len());

        self.map.retain(|item| {
            if idx_set.contains(&item.heap_idx) {
                taken.insert(item.heap_idx, item.id.clone());
                false
            } else {
                true
            }
        });

        let mut result = Vec::with_capacity(taken.len());
        for idx in idxs {
            if let Some(id) = taken.remove(&idx) {
                result.push(id);
            }
        }
        result
    }
}

impl<ID: KeyType> HashTableItem<ID> {
    pub fn new(id: ID, heap_idx: usize) -> Self {
        Self { id, heap_idx }
    }
}

impl HashValue for Option<String> {
    fn hash(&self, state: &RandomState) -> u64 {
        state.hash_one(self)
    }
}

macro_rules! hash_float {
    ($($t:ty),+) => {
        $(impl HashValue for Option<$t> {
            fn hash(&self, state: &RandomState) -> u64 {
                self.map(|me| me.hash(state)).unwrap_or(0)
            }
        })+
    };
}

macro_rules! has_integer {
    ($($t:ty),+) => {
        $(impl HashValue for Option<$t> {
            fn hash(&self, state: &RandomState) -> u64 {
                self.map(|me| me.hash(state)).unwrap_or(0)
            }
        })+
    };
}

has_integer!(i8, i16, i32, i64, i128, i256);
has_integer!(u8, u16, u32, u64);
has_integer!(IntervalDayTime, IntervalMonthDayNano);
hash_float!(f16, f32, f64);

pub fn new_hash_table(
    limit: usize,
    kt: DataType,
) -> Result<Box<dyn ArrowHashTable + Send>> {
    macro_rules! downcast_helper {
        ($kt:ty, $d:ident) => {
            return Ok(Box::new(PrimitiveHashTable::<$kt>::new(limit, kt)))
        };
    }

    downcast_primitive! {
        kt => (downcast_helper, kt),
        DataType::Utf8 => return Ok(Box::new(StringHashTable::new(limit, DataType::Utf8))),
        DataType::LargeUtf8 => return Ok(Box::new(StringHashTable::new(limit, DataType::LargeUtf8))),
        DataType::Utf8View => return Ok(Box::new(StringHashTable::new(limit, DataType::Utf8View))),
        _ => {}
    }

    Err(exec_datafusion_err!(
        "Can't create HashTable for type: {kt:?}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::TimestampMillisecondArray;
    use arrow_schema::TimeUnit;
    use std::collections::BTreeMap;

    #[test]
    fn should_emit_correct_type() -> Result<()> {
        let ids =
            TimestampMillisecondArray::from(vec![1000]).with_timezone("UTC".to_string());
        let dt = DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()));
        let mut ht = new_hash_table(1, dt.clone())?;
        ht.set_batch(Arc::new(ids));
        let mut mapper = vec![];
        ht.find_or_insert(0, 0, &mut mapper);
        let ids = ht.take_all(vec![0]);
        assert_eq!(ids.data_type(), &dt);

        Ok(())
    }

    #[test]
    fn should_resize_properly() -> Result<()> {
        let mut heap_to_map = BTreeMap::<usize, usize>::new();
        let mut map = TopKHashTable::<Option<String>>::new(5, 3);
        let rnd = RandomState::new();
        for (heap_idx, id) in vec!["1", "2", "3", "4", "5"].into_iter().enumerate() {
            let mut mapper = vec![];
            let hash = heap_idx as u64;
            let map_idx =
                map.insert(hash, Some(id.to_string()), heap_idx, &mut mapper, &rnd);
            let _ = heap_to_map.insert(heap_idx, map_idx);
            if heap_idx == 3 {
                let mut mapper_sorted = mapper.clone();
                mapper_sorted.sort();
                assert_eq!(
                    mapper_sorted,
                    vec![(0, 0), (1, 1), (2, 2), (3, 3)],
                    "Pass {heap_idx} resized incorrectly!"
                );
                for (heap_idx, map_idx) in mapper {
                    let _ = heap_to_map.insert(heap_idx, map_idx);
                }
            } else {
                assert_eq!(mapper, vec![], "Pass {heap_idx} should not have resized!");
            }
        }

        let (_heap_idxs, map_idxs): (Vec<_>, Vec<_>) = heap_to_map.into_iter().unzip();
        let ids = map.take_all(map_idxs);
        assert_eq!(
            format!("{ids:?}"),
            r#"[Some("1"), Some("2"), Some("3"), Some("4"), Some("5")]"#
        );
        assert_eq!(map.len(), 0, "Map should have been cleared!");

        Ok(())
    }
}
