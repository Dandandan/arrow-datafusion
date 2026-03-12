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

//! DataFusion benchmark runner
use datafusion::error::Result;

use clap::{Parser, Subcommand};

#[cfg(all(feature = "snmalloc", feature = "mimalloc"))]
compile_error!(
    "feature \"snmalloc\" and feature \"mimalloc\" cannot be enabled at the same time"
);

#[cfg(feature = "snmalloc")]
#[global_allocator]
static ALLOC: snmalloc_rs::SnMalloc = snmalloc_rs::SnMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

use datafusion_benchmarks::{
    cancellation, clickbench, h2o, hj, imdb, nlj, smj, sort_tpch, tpcds, tpch,
};

#[derive(Debug, Parser)]
#[command(about = "benchmark command")]
struct Cli {
    /// Pin each tokio worker thread to a distinct CPU core for more stable benchmarks
    #[arg(long = "pin-threads")]
    pin_threads: bool,

    #[command(subcommand)]
    command: Options,
}

#[derive(Debug, Subcommand)]
enum Options {
    Cancellation(cancellation::RunOpt),
    Clickbench(clickbench::RunOpt),
    H2o(h2o::RunOpt),
    HJ(hj::RunOpt),
    Imdb(imdb::RunOpt),
    Nlj(nlj::RunOpt),
    Smj(smj::RunOpt),
    SortTpch(sort_tpch::RunOpt),
    Tpch(tpch::RunOpt),
    Tpcds(tpcds::RunOpt),
}

async fn run_command(cli: Cli) -> Result<()> {
    match cli.command {
        Options::Cancellation(opt) => opt.run().await,
        Options::Clickbench(opt) => opt.run().await,
        Options::H2o(opt) => opt.run().await,
        Options::HJ(opt) => opt.run().await,
        Options::Imdb(opt) => Box::pin(opt.run()).await,
        Options::Nlj(opt) => opt.run().await,
        Options::Smj(opt) => opt.run().await,
        Options::SortTpch(opt) => opt.run().await,
        Options::Tpch(opt) => Box::pin(opt.run()).await,
        Options::Tpcds(opt) => Box::pin(opt.run()).await,
    }
}

// Main benchmark runner entrypoint
pub fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();
    let pin_threads = cli.pin_threads;

    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();

    if pin_threads {
        let core_ids = core_affinity::get_core_ids().expect("failed to get core IDs");

        // Spawn a dedicated IO thread per core, pinned to the same core
        // as the corresponding tokio worker.
        datafusion_benchmarks::same_thread_local::init_io_thread_pool(&core_ids);
        datafusion_benchmarks::same_thread_local::enable_same_thread_io();

        let core_ids =
            std::sync::Arc::new(std::sync::Mutex::new(core_ids.into_iter().cycle()));
        builder.on_thread_start(move || {
            let core_id = core_ids.lock().unwrap().next().unwrap();
            core_affinity::set_for_current(core_id);
        });

        eprintln!("Thread pinning enabled (with per-core IO threads)");
    }

    let runtime = builder.build().expect("failed to build tokio runtime");
    runtime.block_on(run_command(cli))
}
