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

//! A batch of data that is indexed by a set of indices.

use std::sync::Arc;
use arrow::array::{PrimitiveArray, UInt32Type};
use arrow::record_batch::RecordBatch;

/// A batch of data that is indexed by a set of indices.
/// This is used to avoid copying data when repartitioning.
#[derive(Debug, Clone)]
pub struct IndexedBatch {
    /// The batch of data.
    pub batch: Arc<RecordBatch>,
    /// The indices into the batch.
    pub indices: PrimitiveArray<UInt32Type>,
}
