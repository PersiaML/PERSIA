#![allow(clippy::needless_return)]

#[macro_use]
extern crate shadow_rs;

use anyhow::Result;
use persia_embedding_config::PersiaGlobalConfig;
use persia_embedding_holder::PersiaEmbeddingHolder;
use persia_embedding_sharded_server::hashmap_sharded_service::{
    HashMapShardedService, HashMapShardedServiceInner,
};
use persia_full_amount_manager::FullAmountManager;
use persia_incremental_update_manager::PerisaIncrementalUpdateManager;
use persia_model_manager::PersiaPersistenceManager;

use std::{path::PathBuf, sync::Arc};
use structopt::StructOpt;

#[derive(Debug, StructOpt, Clone)]
#[structopt()]
struct Cli {
    #[structopt(long)]
    port: u16,
    #[structopt(long)]
    shard_idx: usize,
    #[structopt(long)]
    num_shards: usize,
    #[structopt(long)]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install().unwrap();
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("LOG_LEVEL"))
        .init();
    shadow!(build);
    eprintln!("project_name: {}", build::PROJECT_NAME);
    eprintln!("is_debug: {}", shadow_rs::is_debug());
    eprintln!("version: {}", build::version());
    eprintln!("tag: {}", build::TAG);
    eprintln!("commit_hash: {}", build::COMMIT_HASH);
    eprintln!("commit_date: {}", build::COMMIT_DATE);
    eprintln!("build_os: {}", build::BUILD_OS);
    eprintln!("rust_version: {}", build::RUST_VERSION);
    eprintln!("build_time: {}", build::BUILD_TIME);
    let args: Cli = Cli::from_args();

    PersiaGlobalConfig::set(
        args.config,
        args.shard_idx,
        args.num_shards,
        String::from("emb_server"),
    )?;

    let global_config = PersiaGlobalConfig::get()?;
    let embedding_holder = PersiaEmbeddingHolder::get()?;
    let full_amount_manager = FullAmountManager::get()?;
    let inc_update_manager = PerisaIncrementalUpdateManager::get()?;
    let model_persistence_manager = PersiaPersistenceManager::get()?;

    let inner = Arc::new(HashMapShardedServiceInner {
        embedding: embedding_holder,
        optimizer: persia_futures::async_lock::RwLock::new(None),
        hyperparameter_config: persia_futures::async_lock::RwLock::new(None),
        hyperparameter_configured: persia_futures::async_lock::Mutex::new(false),
        global_config,
        inc_update_manager,
        model_persistence_manager,
        full_amount_manager,
        shard_idx: args.shard_idx,
    });

    let service = HashMapShardedService::new(inner).await;

    let server = hyper::Server::bind(&([0, 0, 0, 0], args.port).into())
        // .http2_only(true)
        // .http2_adaptive_window(true)
        .tcp_nodelay(true)
        .serve(hyper::service::make_service_fn(|_| {
            let service = service.clone();
            async move { Ok::<_, hyper::Error>(service) }
        }));

    server.await?;
    Ok(())
}
