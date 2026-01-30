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

//! AtomicBitSet for thread-safe bitwise operations.

use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow::buffer::{BooleanBuffer, Buffer};

/// A thread-safe bitset that allows multiple threads to set bits concurrently
/// without lock contention.
#[derive(Debug)]
pub struct AtomicBitSet {
    data: Arc<[AtomicUsize]>,
    len: usize,
}

impl AtomicBitSet {
    /// Create a new AtomicBitSet with the given number of bits, all initialized to false.
    pub fn new(len: usize) -> Self {
        let word_size = size_of::<usize>() * 8;
        let num_words = len.div_ceil(word_size);
        let mut data = Vec::with_capacity(num_words);
        for _ in 0..num_words {
            data.push(AtomicUsize::new(0));
        }
        Self {
            data: data.into(),
            len,
        }
    }

    /// Set the bit at the given index to true.
    pub fn set_bit(&self, idx: usize, value: bool) {
        if !value {
            // Currently we only support setting to true atomically.
            // If we need to support setting to false, we can use fetch_and.
            return;
        }
        let word_size = size_of::<usize>() * 8;
        let word_idx = idx / word_size;
        let bit_idx = idx % word_size;
        let mask = 1 << bit_idx;
        self.data[word_idx].fetch_or(mask, Ordering::Relaxed);
    }

    /// Get the bit at the given index.
    pub fn get_bit(&self, idx: usize) -> bool {
        let word_size = size_of::<usize>() * 8;
        let word_idx = idx / word_size;
        let bit_idx = idx % word_size;
        let mask = 1 << bit_idx;
        (self.data[word_idx].load(Ordering::Relaxed) & mask) != 0
    }

    /// Returns the number of bits in the bitset.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the bitset is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Convert the AtomicBitSet to a BooleanBuffer.
    pub fn to_boolean_buffer(&self) -> BooleanBuffer {
        let mut bytes = Vec::with_capacity(self.data.len() * size_of::<usize>());
        for word in self.data.as_ref() {
            let val = word.load(Ordering::Relaxed);
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes.truncate(self.len.div_ceil(8));
        BooleanBuffer::new(Buffer::from(bytes), 0, self.len)
    }
}
