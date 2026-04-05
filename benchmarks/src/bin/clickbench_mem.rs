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

//! Measures peak memory usage for representative ClickBench query patterns.
//!
//! Generates synthetic data matching the ClickBench schema to avoid requiring
//! the 14GB hits.parquet download. Reports per-query peak memory and
//! top memory consumers.

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use core::num::NonZeroUsize;
use datafusion::error::Result;
use datafusion::execution::memory_pool::{
    MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation, TrackConsumersPool,
};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use datafusion_common::instant::Instant;

const NUM_ROWS: usize = 5_000_000; // 5M rows
const BATCH_SIZE: usize = 8192;

/// Wrapper around a MemoryPool that tracks the global peak reservation.
#[derive(Debug)]
struct PeakTrackingPool<I: MemoryPool> {
    inner: I,
    peak: AtomicUsize,
}

impl<I: MemoryPool> PeakTrackingPool<I> {
    fn new(inner: I) -> Self {
        Self {
            inner,
            peak: AtomicUsize::new(0),
        }
    }

    fn peak_reserved(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    fn update_peak(&self) {
        let current = self.inner.reserved();
        self.peak.fetch_max(current, Ordering::Relaxed);
    }
}

impl<I: MemoryPool> MemoryPool for PeakTrackingPool<I> {
    fn register(&self, consumer: &MemoryConsumer) {
        self.inner.register(consumer);
    }

    fn unregister(&self, consumer: &MemoryConsumer) {
        self.inner.unregister(consumer);
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
        self.update_peak();
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        self.inner.try_grow(reservation, additional)?;
        self.update_peak();
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
}

fn generate_random_strings(
    rng: &mut StdRng,
    n: usize,
    cardinality: usize,
    max_len: usize,
) -> StringArray {
    let dict: Vec<String> = (0..cardinality)
        .map(|_| {
            let len = rng.random_range(0..=max_len);
            (0..len)
                .map(|_| rng.random_range(b'a'..=b'z') as char)
                .collect()
        })
        .collect();
    let values: Vec<String> = (0..n)
        .map(|_| dict[rng.random_range(0..cardinality)].clone())
        .collect();
    StringArray::from(values)
}

fn generate_data() -> Result<Vec<RecordBatch>> {
    println!("Generating {NUM_ROWS} rows of synthetic ClickBench data...");
    let mut rng = StdRng::seed_from_u64(42);

    let schema = Arc::new(Schema::new(vec![
        Field::new("WatchID", DataType::Int64, false),
        Field::new("JavaEnable", DataType::Int16, false),
        Field::new("Title", DataType::Utf8, false),
        Field::new("EventTime", DataType::Int64, false),
        Field::new("EventDate", DataType::UInt16, false),
        Field::new("CounterID", DataType::Int32, false),
        Field::new("ClientIP", DataType::Int32, false),
        Field::new("RegionID", DataType::Int32, false),
        Field::new("UserID", DataType::Int64, false),
        Field::new("OS", DataType::Int16, false),
        Field::new("UserAgent", DataType::Int16, false),
        Field::new("URL", DataType::Utf8, false),
        Field::new("Referer", DataType::Utf8, false),
        Field::new("IsRefresh", DataType::Int16, false),
        Field::new("ResolutionWidth", DataType::Int16, false),
        Field::new("ResolutionHeight", DataType::Int16, false),
        Field::new("SearchPhrase", DataType::Utf8, false),
        Field::new("SearchEngineID", DataType::Int16, false),
        Field::new("AdvEngineID", DataType::Int16, false),
        Field::new("MobilePhone", DataType::Int16, false),
        Field::new("MobilePhoneModel", DataType::Utf8, false),
        Field::new("IsMobile", DataType::Int16, false),
    ]));

    let mut batches = Vec::new();
    let mut rows_left = NUM_ROWS;

    while rows_left > 0 {
        let n = rows_left.min(BATCH_SIZE);

        let watch_id: Int64Array = (0..n).map(|_| rng.random::<i64>()).collect();
        let user_id: Int64Array =
            (0..n).map(|_| rng.random_range(0i64..1_000_000)).collect();
        let event_time: Int64Array = (0..n)
            .map(|_| rng.random_range(1372636800i64..1375315200))
            .collect();
        let event_date: UInt16Array =
            (0..n).map(|_| rng.random_range(15887u16..15917)).collect();
        let counter_id: Int32Array =
            (0..n).map(|_| rng.random_range(0i32..10000)).collect();
        let client_ip: Int32Array = (0..n).map(|_| rng.random::<i32>()).collect();
        let region_id: Int32Array =
            (0..n).map(|_| rng.random_range(0i32..30000)).collect();
        let java_enable: Int16Array = (0..n).map(|_| rng.random_range(0i16..2)).collect();
        let os: Int16Array = (0..n).map(|_| rng.random_range(0i16..100)).collect();
        let user_agent: Int16Array =
            (0..n).map(|_| rng.random_range(0i16..100)).collect();
        let is_refresh: Int16Array = (0..n).map(|_| rng.random_range(0i16..2)).collect();
        let resolution_width: Int16Array =
            (0..n).map(|_| rng.random_range(0i16..2000)).collect();
        let resolution_height: Int16Array =
            (0..n).map(|_| rng.random_range(0i16..2000)).collect();
        let search_engine_id: Int16Array =
            (0..n).map(|_| rng.random_range(0i16..50)).collect();
        let adv_engine_id: Int16Array =
            (0..n).map(|_| rng.random_range(0i16..30)).collect();
        let mobile_phone: Int16Array =
            (0..n).map(|_| rng.random_range(0i16..200)).collect();
        let is_mobile: Int16Array = (0..n).map(|_| rng.random_range(0i16..2)).collect();

        let title = generate_random_strings(&mut rng, n, 50000, 80);
        let url = generate_random_strings(&mut rng, n, 500000, 120);
        let referer = generate_random_strings(&mut rng, n, 200000, 100);
        let search_phrase = {
            let dict: Vec<String> = (0..100000)
                .map(|i| {
                    if i % 2 == 0 {
                        String::new()
                    } else {
                        let len = rng.random_range(3..80);
                        (0..len)
                            .map(|_| rng.random_range(b'a'..=b'z') as char)
                            .collect()
                    }
                })
                .collect();
            let values: Vec<String> = (0..n)
                .map(|_| dict[rng.random_range(0..dict.len())].clone())
                .collect();
            StringArray::from(values)
        };
        let mobile_phone_model = generate_random_strings(&mut rng, n, 500, 20);

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(watch_id),
                Arc::new(java_enable),
                Arc::new(title),
                Arc::new(event_time),
                Arc::new(event_date),
                Arc::new(counter_id),
                Arc::new(client_ip),
                Arc::new(region_id),
                Arc::new(user_id),
                Arc::new(os),
                Arc::new(user_agent),
                Arc::new(url),
                Arc::new(referer),
                Arc::new(is_refresh),
                Arc::new(resolution_width),
                Arc::new(resolution_height),
                Arc::new(search_phrase),
                Arc::new(search_engine_id),
                Arc::new(adv_engine_id),
                Arc::new(mobile_phone),
                Arc::new(mobile_phone_model),
                Arc::new(is_mobile),
            ],
        )?;

        batches.push(batch);
        rows_left -= n;
    }

    println!("Generated {} batches", batches.len());
    Ok(batches)
}

/// Representative ClickBench query patterns, ordered by expected memory impact
fn get_queries() -> Vec<(&'static str, &'static str)> {
    vec![
        // --- Low memory: simple scans/aggregations ---
        ("Q0: COUNT(*)", "SELECT COUNT(*) FROM hits"),
        (
            "Q2: Simple aggs",
            r#"SELECT SUM("AdvEngineID"), COUNT(*), AVG("ResolutionWidth") FROM hits"#,
        ),
        // --- Medium memory: low-cardinality GROUP BY ---
        (
            "Q7: Low-card GROUP BY",
            r#"SELECT "AdvEngineID", COUNT(*) FROM hits WHERE "AdvEngineID" <> 0 GROUP BY "AdvEngineID" ORDER BY COUNT(*) DESC"#,
        ),
        // --- Medium memory: GROUP BY with COUNT DISTINCT ---
        (
            "Q8: RegionID + COUNT DISTINCT UserID",
            r#"SELECT "RegionID", COUNT(DISTINCT "UserID") AS u FROM hits GROUP BY "RegionID" ORDER BY u DESC LIMIT 10"#,
        ),
        // --- High memory: COUNT DISTINCT on high-cardinality ---
        (
            "Q4: COUNT DISTINCT UserID",
            r#"SELECT COUNT(DISTINCT "UserID") FROM hits"#,
        ),
        (
            "Q5: COUNT DISTINCT SearchPhrase",
            r#"SELECT COUNT(DISTINCT "SearchPhrase") FROM hits"#,
        ),
        // --- High memory: GROUP BY high-cardinality string ---
        (
            "Q12: GROUP BY SearchPhrase",
            r#"SELECT "SearchPhrase", COUNT(*) AS c FROM hits WHERE "SearchPhrase" <> '' GROUP BY "SearchPhrase" ORDER BY c DESC LIMIT 10"#,
        ),
        (
            "Q15: GROUP BY UserID",
            r#"SELECT "UserID", COUNT(*) FROM hits GROUP BY "UserID" ORDER BY COUNT(*) DESC LIMIT 10"#,
        ),
        // --- Very high memory: GROUP BY multiple high-cardinality columns ---
        (
            "Q16: GROUP BY UserID, SearchPhrase",
            r#"SELECT "UserID", "SearchPhrase", COUNT(*) FROM hits GROUP BY "UserID", "SearchPhrase" ORDER BY COUNT(*) DESC LIMIT 10"#,
        ),
        (
            "Q31: GROUP BY WatchID, ClientIP (filtered)",
            r#"SELECT "WatchID", "ClientIP", COUNT(*) AS c, SUM("IsRefresh"), AVG("ResolutionWidth") FROM hits WHERE "SearchPhrase" <> '' GROUP BY "WatchID", "ClientIP" ORDER BY c DESC LIMIT 10"#,
        ),
        (
            "Q32: GROUP BY WatchID, ClientIP (full)",
            r#"SELECT "WatchID", "ClientIP", COUNT(*) AS c, SUM("IsRefresh"), AVG("ResolutionWidth") FROM hits GROUP BY "WatchID", "ClientIP" ORDER BY c DESC LIMIT 10"#,
        ),
        // --- Very high memory: GROUP BY URL (long strings) ---
        (
            "Q33: GROUP BY URL",
            r#"SELECT "URL", COUNT(*) AS c FROM hits GROUP BY "URL" ORDER BY c DESC LIMIT 10"#,
        ),
        // --- Sort-heavy queries ---
        (
            "Q25: ORDER BY SearchPhrase",
            r#"SELECT "SearchPhrase" FROM hits WHERE "SearchPhrase" <> '' ORDER BY "SearchPhrase" LIMIT 10"#,
        ),
        // --- Extended: multiple COUNT DISTINCT ---
        (
            "EQ0: Multiple COUNT DISTINCT",
            r#"SELECT COUNT(DISTINCT "SearchPhrase"), COUNT(DISTINCT "MobilePhone"), COUNT(DISTINCT "MobilePhoneModel") FROM hits"#,
        ),
        // --- Extended: high-card GROUP BY with FIRST_VALUE ---
        (
            "EQ9: FIRST_VALUE(URL) GROUP BY UserID",
            r#"SELECT MAX(len) FROM (SELECT LENGTH(FIRST_VALUE("URL" ORDER BY "EventTime")) as len FROM hits GROUP BY "UserID")"#,
        ),
        // --- Extended: high-card GROUP BY primitive ---
        (
            "EQ7: GROUP BY WatchID (nearly unique)",
            r#"SELECT "WatchID", MIN("ResolutionWidth") as wmin, MAX("ResolutionWidth") as wmax, SUM("IsRefresh") as srefresh FROM hits GROUP BY "WatchID" ORDER BY "WatchID" DESC LIMIT 10"#,
        ),
    ]
}

#[tokio::main]
async fn main() -> Result<()> {
    let batches = generate_data()?;

    println!(
        "\n{:<50} {:>12} {:>12} {:>12}  Top Consumers",
        "Query", "Peak Mem", "Duration", "Rows"
    );
    println!("{}", "=".repeat(130));

    for (name, sql) in get_queries() {
        // Create a fresh context with peak-tracking + consumer-tracking pool
        let peak_pool = Arc::new(PeakTrackingPool::new(TrackConsumersPool::new(
            datafusion::execution::memory_pool::FairSpillPool::new(
                8 * 1024 * 1024 * 1024,
            ), // 8GB limit
            NonZeroUsize::new(10).unwrap(),
        )));

        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(peak_pool.clone())
            .build_arc()?;

        let config = SessionConfig::new()
            .with_target_partitions(4)
            .with_batch_size(BATCH_SIZE);

        let ctx = SessionContext::new_with_config_rt(config, runtime);

        let schema = batches[0].schema();
        let provider =
            datafusion::datasource::MemTable::try_new(schema, vec![batches.clone()])?;
        ctx.register_table("hits", Arc::new(provider))?;

        let start = Instant::now();
        let result = ctx.sql(sql).await;
        match result {
            Ok(df) => match df.collect().await {
                Ok(results) => {
                    let elapsed = start.elapsed();
                    let row_count: usize = results.iter().map(|b| b.num_rows()).sum();
                    let peak_mem = peak_pool.peak_reserved();

                    // Get top consumers info from the inner TrackConsumersPool
                    let top_report = peak_pool.inner.report_top(3);

                    println!(
                        "{:<50} {:>9.1} MB {:>9.1}ms {:>10}  {}",
                        name,
                        peak_mem as f64 / 1024.0 / 1024.0,
                        elapsed.as_secs_f64() * 1000.0,
                        row_count,
                        top_report.replace('\n', " | "),
                    );
                }
                Err(e) => {
                    println!("{name:<50} ERROR: {e}");
                }
            },
            Err(e) => {
                println!("{name:<50} ERROR: {e}");
            }
        }
    }

    println!(
        "\nData: {NUM_ROWS} rows, 4 partitions, synthetic ClickBench-like distribution."
    );
    Ok(())
}
