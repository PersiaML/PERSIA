#![allow(clippy::needless_return)]

#[macro_use]
extern crate shadow_rs;

use anyhow::Result;
use persia_embedding_config::{
    EmbeddingConfig, PerisaIntent, PersiaCommonConfig, PersiaGlobalConfig, PersiaReplicaInfo,
    PersiaShardedServerConfig,
};
use persia_embedding_holder::PersiaEmbeddingHolder;
use persia_embedding_sharded_server::hashmap_sharded_service::{
    EmbeddingServerNatsStub, EmbeddingServerNatsStubResponder, HashMapShardedService,
    HashMapShardedServiceInner,
};
use persia_full_amount_manager::FullAmountManager;
use persia_incremental_update_manager::PerisaIncrementalUpdateManager;
use persia_model_manager::PersiaPersistenceManager;
use persia_nats_client::NatsClient;

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
    global_config: PathBuf,
    #[structopt(long)]
    embedding_config: PathBuf,
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

    PersiaGlobalConfig::set_configures(
        &args.global_config,
        args.port,
        args.shard_idx,
        args.num_shards,
    )?;

    EmbeddingConfig::set(&args.embedding_config)?;

    let embedding_config = EmbeddingConfig::get()?;
    let common_config = PersiaCommonConfig::get()?;
    let server_config = PersiaShardedServerConfig::get()?;
    let embedding_holder = PersiaEmbeddingHolder::get()?;
    let full_amount_manager = FullAmountManager::get()?;
    let inc_update_manager = PerisaIncrementalUpdateManager::get()?;
    let model_persistence_manager = PersiaPersistenceManager::get()?;
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    let inner = Arc::new(HashMapShardedServiceInner::new(
        embedding_holder,
        server_config,
        common_config,
        embedding_config,
        inc_update_manager,
        model_persistence_manager,
        full_amount_manager,
        args.shard_idx,
    ));

    let service = HashMapShardedService {
        inner: inner.clone(),
        shutdown_channel: Arc::new(persia_futures::async_lock::RwLock::new(Some(tx))),
    };

    let server = hyper::Server::bind(&([0, 0, 0, 0], args.port).into())
        .tcp_nodelay(true)
        .serve(hyper::service::make_service_fn(|_| {
            let service = service.clone();
            async move { Ok::<_, hyper::Error>(service) }
        }));

    let intent = inner.get_intent()?;
    let _responder = match intent {
        PerisaIntent::Infer(_) => None,
        _ => {
            let nats_stub = EmbeddingServerNatsStub {
                inner: inner.clone(),
            };
            let repilca_info = PersiaReplicaInfo::get()?.as_ref().clone();
            let responder = EmbeddingServerNatsStubResponder {
                inner: nats_stub,
                nats_client: NatsClient::new(repilca_info),
            };
            responder
                .spawn_subscriptions()
                .expect("failed to spawn nats subscriptions");
            Some(responder)
        }
    };

    match intent {
        PerisaIntent::Infer(ref conf) => {
            let sparse_ckpt = conf.embedding_checkpoint.clone();
            inner.load(sparse_ckpt).await?;
        }
        _ => {}
    }
    let graceful = server.with_graceful_shutdown(async {
        rx.await.ok();
    });

    if let Err(err) = graceful.await {
        tracing::error!("embedding server exited with error: {:?}!", err);
    } else {
        tracing::info!("embedding server exited successfully");
    }
    Ok(())
}
