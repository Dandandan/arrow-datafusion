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

//! Hash map for join operations using contiguous storage.
//!
//! Packed value encoding (stored in the hash table per key):
//! - **Inline** (single match): high bit set, remaining bits = row index
//! - **Group** (multiple matches): high bit clear, remaining bits = group ID
//!   → `group_offsets[id]..group_offsets[id+1]` indexes into `flat_indices`
//!
//! Build phase: first insert per key is stored inline. Subsequent inserts for
//! the same key go into an overflow buffer. `flatten()` promotes inline entries
//! with overflow to groups and builds contiguous `flat_indices` in one sequential pass.

use std::fmt::{self, Debug};
use std::ops::Sub;

use arrow::array::BooleanArray;
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::ArrowNativeType;
use hashbrown::HashTable;
use hashbrown::hash_table::Entry::{Occupied, Vacant};

use crate::joins::MapOffset;
use crate::joins::chain::traverse_chain;

const INLINE_BIT_U32: u32 = 1 << 31;
const INLINE_BIT_U64: u64 = 1 << 63;

pub trait JoinHashMapType: Send + Sync {
    fn extend_zero(&mut self, len: usize);

    fn update_from_iter<'a>(
        &mut self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + Send + 'a>,
        deleted_offset: usize,
    );

    fn get_matched_indices<'a>(
        &self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + 'a>,
        deleted_offset: Option<usize>,
    ) -> (Vec<u32>, Vec<u64>);

    fn get_matched_indices_with_limit_offset(
        &self,
        hash_values: &[u64],
        limit: usize,
        offset: MapOffset,
        input_indices: &mut Vec<u32>,
        match_indices: &mut Vec<u64>,
    ) -> Option<MapOffset>;

    fn contain_hashes(&self, hash_values: &[u64]) -> BooleanArray;
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;

    /// Flatten overflow into contiguous storage. Call after all inserts, before probing.
    fn flatten(&mut self);
}

// --- InlineBit: packed value encoding ---

trait InlineBit: Copy + PartialEq {
    fn is_inline(self) -> bool;
    fn inline_value(self) -> u64;
    fn make_inline(row: u64) -> Self;
    fn make_group_id(id: u32) -> Self;
    fn group_id(self) -> u32;
}

impl InlineBit for u32 {
    #[inline(always)]
    fn is_inline(self) -> bool {
        self & INLINE_BIT_U32 != 0
    }
    #[inline(always)]
    fn inline_value(self) -> u64 {
        (self & !INLINE_BIT_U32) as u64
    }
    #[inline(always)]
    fn make_inline(row: u64) -> u32 {
        INLINE_BIT_U32 | row as u32
    }
    #[inline(always)]
    fn make_group_id(id: u32) -> u32 {
        id
    }
    #[inline(always)]
    fn group_id(self) -> u32 {
        self
    }
}

impl InlineBit for u64 {
    #[inline(always)]
    fn is_inline(self) -> bool {
        self & INLINE_BIT_U64 != 0
    }
    #[inline(always)]
    fn inline_value(self) -> u64 {
        self & !INLINE_BIT_U64
    }
    #[inline(always)]
    fn make_inline(row: u64) -> u64 {
        INLINE_BIT_U64 | row
    }
    #[inline(always)]
    fn make_group_id(id: u32) -> u64 {
        id as u64
    }
    #[inline(always)]
    fn group_id(self) -> u32 {
        self as u32
    }
}

// --- Generic JoinHashMap ---

/// Hash map for join build/probe using contiguous storage.
/// `T` is the packed map value type (u32 or u64).
/// `F` is the flat index type (u32 or u64), must impl `Into<u64>` for output.
pub(crate) struct JoinHashMap<T: InlineBit, F: Copy + Default + Into<u64>> {
    map: HashTable<(u64, T)>,
    flat_indices: Vec<F>,
    group_offsets: Vec<u32>,
    overflow: Vec<(u32, F)>,
    num_groups: u32,
}

pub type JoinHashMapU32 = JoinHashMap<u32, u32>;
pub type JoinHashMapU64 = JoinHashMap<u64, u64>;

impl<T: InlineBit, F: Copy + Default + Into<u64>> JoinHashMap<T, F> {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: HashTable::with_capacity(cap),
            flat_indices: Vec::new(),
            group_offsets: Vec::new(),
            overflow: Vec::new(),
            num_groups: 0,
        }
    }
}

impl<T: InlineBit, F: Copy + Default + Into<u64>> Debug for JoinHashMap<T, F> {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        Ok(())
    }
}

// --- Build ---

fn build_insert<T: InlineBit, F: Copy + Default + Into<u64>>(
    map: &mut HashTable<(u64, T)>,
    num_groups: &mut u32,
    overflow: &mut Vec<(u32, F)>,
    row: usize,
    hash_value: u64,
) where
    F: From<u32>,
{
    let entry = map.entry(hash_value, |&(h, _)| hash_value == h, |&(h, _)| h);
    match entry {
        Occupied(mut occ) => {
            let (_, packed) = occ.get_mut();
            if packed.is_inline() {
                let old_row = packed.inline_value();
                let gid = *num_groups;
                *num_groups += 1;
                *packed = T::make_group_id(gid);
                overflow.push((gid, F::from(old_row as u32)));
            }
            overflow.push((packed.group_id(), F::from(row as u32)));
        }
        Vacant(vac) => {
            vac.insert((hash_value, T::make_inline(row as u64)));
        }
    }
}

// --- Flatten ---

fn flatten_overflow<F: Copy + Default + Into<u64>>(
    num_groups: u32,
    overflow: &mut Vec<(u32, F)>,
    flat_indices: &mut Vec<F>,
    group_offsets: &mut Vec<u32>,
) {
    if overflow.is_empty() {
        return;
    }

    let ng = num_groups as usize;

    // Count entries per group directly into group_offsets
    group_offsets.clear();
    group_offsets.resize(ng + 1, 0);
    for &(gid, _) in overflow.iter() {
        group_offsets[gid as usize + 1] += 1;
    }
    // Prefix sum
    for i in 1..=ng {
        group_offsets[i] += group_offsets[i - 1];
    }

    // Place entries in reverse order (LIFO) to match linked-list traversal order.
    let total = group_offsets[ng] as usize;
    flat_indices.clear();
    flat_indices.resize(total, F::default());
    // Cursors start at end of each group and decrement
    let mut cursors = group_offsets[1..=ng].to_vec();
    for &(gid, row) in overflow.iter() {
        cursors[gid as usize] -= 1;
        flat_indices[cursors[gid as usize] as usize] = row;
    }

    overflow.clear();
}

// --- Probe ---

/// Emit matches for a single packed entry. Returns `Some(offset)` if limit reached.
#[inline(always)]
fn emit_packed<T: InlineBit, F: Copy + Into<u64>>(
    packed: T,
    row_idx: usize,
    start_pos: usize,
    flat_indices: &[F],
    group_offsets: &[u32],
    remaining: &mut usize,
    input_indices: &mut Vec<u32>,
    match_indices: &mut Vec<u64>,
) -> Option<MapOffset> {
    if packed.is_inline() {
        if *remaining == 0 {
            return Some((row_idx, None));
        }
        match_indices.push(packed.inline_value());
        input_indices.push(row_idx as u32);
        *remaining -= 1;
    } else {
        let gid = packed.group_id() as usize;
        let start = if start_pos > 0 {
            start_pos
        } else {
            group_offsets[gid] as usize
        };
        let end = group_offsets[gid + 1] as usize;
        for pos in start..end {
            if *remaining == 0 {
                return Some((row_idx, Some(pos as u64 + 1)));
            }
            match_indices.push(flat_indices[pos].into());
            input_indices.push(row_idx as u32);
            *remaining -= 1;
        }
    }
    None
}

/// Probe the flattened hash map with batched finds (4 at a time).
///
/// Offset convention: `Some(0)` = done with this probe idx.
/// For resume within a group: `Some(pos + 1)` where pos is the flat_indices position.
fn probe_flat<T: InlineBit, F: Copy + Default + Into<u64>>(
    map: &HashTable<(u64, T)>,
    flat_indices: &[F],
    group_offsets: &[u32],
    hash_values: &[u64],
    limit: usize,
    offset: MapOffset,
    input_indices: &mut Vec<u32>,
    match_indices: &mut Vec<u64>,
) -> Option<MapOffset> {
    input_indices.clear();
    match_indices.clear();
    let mut remaining = limit;

    let to_skip = match offset {
        (idx, None) => idx,
        (idx, Some(0)) => idx + 1,
        (idx, Some(pos_plus_one)) => {
            if let Some((_, packed)) =
                map.find(hash_values[idx], |(h, _)| hash_values[idx] == *h)
            {
                let resume = (pos_plus_one - 1) as usize;
                if let Some(off) = emit_packed(
                    *packed,
                    idx,
                    resume,
                    flat_indices,
                    group_offsets,
                    &mut remaining,
                    input_indices,
                    match_indices,
                ) {
                    return Some(off);
                }
            }
            idx + 1
        }
    };

    let remaining_slice = &hash_values[to_skip..];
    let chunks = remaining_slice.chunks_exact(4);
    let remainder = chunks.remainder();

    for (chunk_idx, chunk) in chunks.enumerate() {
        let base = to_skip + chunk_idx * 4;
        let r0 = map.find(chunk[0], |(h, _)| chunk[0] == *h);
        let r1 = map.find(chunk[1], |(h, _)| chunk[1] == *h);
        let r2 = map.find(chunk[2], |(h, _)| chunk[2] == *h);
        let r3 = map.find(chunk[3], |(h, _)| chunk[3] == *h);

        for (j, r) in [r0, r1, r2, r3].into_iter().enumerate() {
            if let Some((_, packed)) = r {
                if let Some(off) = emit_packed(
                    *packed,
                    base + j,
                    0,
                    flat_indices,
                    group_offsets,
                    &mut remaining,
                    input_indices,
                    match_indices,
                ) {
                    return Some(off);
                }
            }
        }
    }

    let remainder_start = to_skip + remaining_slice.len() - remainder.len();
    for (i, &hash) in remainder.iter().enumerate() {
        if let Some((_, packed)) = map.find(hash, |(h, _)| hash == *h) {
            if let Some(off) = emit_packed(
                *packed,
                remainder_start + i,
                0,
                flat_indices,
                group_offsets,
                &mut remaining,
                input_indices,
                match_indices,
            ) {
                return Some(off);
            }
        }
    }
    None
}

// --- JoinHashMapType impl ---

impl<T, F> JoinHashMapType for JoinHashMap<T, F>
where
    T: InlineBit + Send + Sync,
    F: Copy + Default + Into<u64> + From<u32> + Send + Sync,
{
    fn extend_zero(&mut self, _: usize) {}

    fn update_from_iter<'a>(
        &mut self,
        iter: Box<dyn Iterator<Item = (usize, &'a u64)> + Send + 'a>,
        _deleted_offset: usize,
    ) {
        for (row, hash) in iter {
            build_insert(
                &mut self.map,
                &mut self.num_groups,
                &mut self.overflow,
                row,
                *hash,
            );
        }
    }

    fn get_matched_indices<'a>(
        &self,
        _iter: Box<dyn Iterator<Item = (usize, &'a u64)> + 'a>,
        _deleted_offset: Option<usize>,
    ) -> (Vec<u32>, Vec<u64>) {
        unimplemented!(
            "JoinHashMap does not support get_matched_indices; use get_matched_indices_with_limit_offset"
        )
    }

    fn get_matched_indices_with_limit_offset(
        &self,
        hash_values: &[u64],
        limit: usize,
        offset: MapOffset,
        input_indices: &mut Vec<u32>,
        match_indices: &mut Vec<u64>,
    ) -> Option<MapOffset> {
        probe_flat(
            &self.map,
            &self.flat_indices,
            &self.group_offsets,
            hash_values,
            limit,
            offset,
            input_indices,
            match_indices,
        )
    }

    fn contain_hashes(&self, hash_values: &[u64]) -> BooleanArray {
        contain_hashes(&self.map, hash_values)
    }

    fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn flatten(&mut self) {
        flatten_overflow(
            self.num_groups,
            &mut self.overflow,
            &mut self.flat_indices,
            &mut self.group_offsets,
        );
    }
}

// --- Legacy free functions for PruningJoinHashMap (streaming join) ---

pub fn update_from_iter<'a, T>(
    map: &mut HashTable<(u64, T)>,
    next: &mut [T],
    iter: Box<dyn Iterator<Item = (usize, &'a u64)> + Send + 'a>,
    deleted_offset: usize,
) where
    T: Copy + TryFrom<usize> + PartialOrd,
    <T as TryFrom<usize>>::Error: Debug,
{
    for (row, &hash_value) in iter {
        let entry = map.entry(
            hash_value,
            |&(hash, _)| hash_value == hash,
            |&(hash, _)| hash,
        );
        match entry {
            Occupied(mut occ) => {
                let (_, index) = occ.get_mut();
                let prev_index = *index;
                *index = T::try_from(row + 1).unwrap();
                next[row - deleted_offset] = prev_index;
            }
            Vacant(vac) => {
                vac.insert((hash_value, T::try_from(row + 1).unwrap()));
            }
        }
    }
}

pub fn get_matched_indices<'a, T>(
    map: &HashTable<(u64, T)>,
    next: &[T],
    iter: Box<dyn Iterator<Item = (usize, &'a u64)> + 'a>,
    deleted_offset: Option<usize>,
) -> (Vec<u32>, Vec<u64>)
where
    T: Copy + TryFrom<usize> + PartialOrd + Into<u64> + Sub<Output = T>,
    <T as TryFrom<usize>>::Error: Debug,
{
    let mut input_indices = vec![];
    let mut match_indices = vec![];
    let zero = T::try_from(0).unwrap();
    let one = T::try_from(1).unwrap();

    for (row_idx, hash_value) in iter {
        if let Some((_, index)) = map.find(*hash_value, |(hash, _)| *hash_value == *hash)
        {
            let mut i = *index - one;
            loop {
                let match_row_idx = if let Some(offset) = deleted_offset {
                    let offset = T::try_from(offset).unwrap();
                    if i < offset {
                        break;
                    }
                    i - offset
                } else {
                    i
                };
                match_indices.push(match_row_idx.into());
                input_indices.push(row_idx as u32);
                let next_chain = next[match_row_idx.into() as usize];
                if next_chain == zero {
                    break;
                }
                i = next_chain - one;
            }
        }
    }
    (input_indices, match_indices)
}

pub fn get_matched_indices_with_limit_offset<T>(
    map: &HashTable<(u64, T)>,
    next_chain: &[T],
    hash_values: &[u64],
    limit: usize,
    offset: MapOffset,
    input_indices: &mut Vec<u32>,
    match_indices: &mut Vec<u64>,
) -> Option<MapOffset>
where
    T: Copy + TryFrom<usize> + PartialOrd + Into<u64> + Sub<Output = T>,
    <T as TryFrom<usize>>::Error: Debug,
    T: ArrowNativeType,
{
    input_indices.clear();
    match_indices.clear();

    let mut remaining_output = limit;

    let to_skip = match offset {
        (idx, None) => idx,
        (idx, Some(0)) => idx + 1,
        (idx, Some(next_idx)) => {
            let next_idx: T = T::usize_as(next_idx as usize);
            let is_last = idx == hash_values.len() - 1;
            if let Some(next_offset) = traverse_chain(
                next_chain,
                idx,
                next_idx,
                &mut remaining_output,
                input_indices,
                match_indices,
                is_last,
            ) {
                return Some(next_offset);
            }
            idx + 1
        }
    };

    for (i, &hash) in hash_values[to_skip..].iter().enumerate() {
        let row_idx = to_skip + i;
        if let Some((_, idx)) = map.find(hash, |(h, _)| hash == *h) {
            let idx: T = *idx;
            let is_last = row_idx == hash_values.len() - 1;
            if let Some(next_offset) = traverse_chain(
                next_chain,
                row_idx,
                idx,
                &mut remaining_output,
                input_indices,
                match_indices,
                is_last,
            ) {
                return Some(next_offset);
            }
        }
    }
    None
}

pub fn contain_hashes<T>(map: &HashTable<(u64, T)>, hash_values: &[u64]) -> BooleanArray {
    let buffer = BooleanBuffer::collect_bool(hash_values.len(), |i| {
        let hash = hash_values[i];
        map.find(hash, |(h, _)| hash == *h).is_some()
    });
    BooleanArray::new(buffer, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contain_hashes() {
        let mut hash_map = JoinHashMapU32::with_capacity(10);
        hash_map.update_from_iter(Box::new([10u64, 20u64, 30u64].iter().enumerate()), 0);
        let probe_hashes = vec![10, 11, 20, 21, 30, 31];
        let array = hash_map.contain_hashes(&probe_hashes);
        assert_eq!(array.len(), probe_hashes.len());
        for (i, &hash) in probe_hashes.iter().enumerate() {
            if matches!(hash, 10 | 20 | 30) {
                assert!(array.value(i), "Hash {hash} should exist");
            } else {
                assert!(!array.value(i), "Hash {hash} should NOT exist");
            }
        }
    }

    #[test]
    fn test_unique() {
        let mut m = JoinHashMapU32::with_capacity(3);
        m.update_from_iter(Box::new([10u64, 20u64, 30u64].iter().enumerate()), 0);
        m.flatten();
        let mut inp = vec![];
        let mut mat = vec![];
        let r = m.get_matched_indices_with_limit_offset(
            &[10, 20, 30, 99],
            100,
            (0, None),
            &mut inp,
            &mut mat,
        );
        assert!(r.is_none());
        assert_eq!(mat.len(), 3);
    }

    #[test]
    fn test_with_dups() {
        let mut m = JoinHashMapU32::with_capacity(4);
        m.update_from_iter(Box::new([10u64, 10u64, 10u64, 20u64].iter().enumerate()), 0);
        m.flatten();
        let mut inp = vec![];
        let mut mat = vec![];
        let r = m.get_matched_indices_with_limit_offset(
            &[10, 20],
            100,
            (0, None),
            &mut inp,
            &mut mat,
        );
        assert!(r.is_none());
        assert_eq!(mat.len(), 4);
        let mut h10: Vec<u64> = inp
            .iter()
            .zip(mat.iter())
            .filter(|&(&i, _)| i == 0)
            .map(|(_, &m)| m)
            .collect();
        h10.sort();
        assert_eq!(h10, vec![0, 1, 2]);
    }

    #[test]
    fn test_with_limit() {
        let mut m = JoinHashMapU32::with_capacity(4);
        m.update_from_iter(Box::new([10u64, 10u64, 10u64, 20u64].iter().enumerate()), 0);
        m.flatten();
        let mut inp = vec![];
        let mut mat = vec![];
        let r = m.get_matched_indices_with_limit_offset(
            &[10, 20],
            2,
            (0, None),
            &mut inp,
            &mut mat,
        );
        assert_eq!(mat.len(), 2);
        assert!(r.is_some());
        let r = m.get_matched_indices_with_limit_offset(
            &[10, 20],
            100,
            r.unwrap(),
            &mut inp,
            &mut mat,
        );
        assert!(r.is_none());
        assert_eq!(mat.len(), 2);
    }
}
