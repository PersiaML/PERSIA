#[macro_use]
extern crate shadow_rs;
use hashbrown::HashMap;
use persia_embedding_config::{
    EmbeddingConfig, PerisaIntent, PersiaCommonConfig, PersiaGlobalConfig, PersiaMiddlewareConfig,
    PersiaReplicaInfo,
};
use persia_embedding_sharded_server::hashmap_sharded_service::EmbeddingServerNatsStubPublisher;

use persia_embedding_sharded_server::sharded_middleware_service::{
    AllShardsClient, MiddlewareNatsStub, MiddlewareNatsStubResponder, ShardedMiddlewareServer,
    ShardedMiddlewareServerInner,
};
use persia_nats_client::NatsClient;
use std::path::PathBuf;
use std::sync::Arc;
use structopt::StructOpt;

#[derive(Debug, StructOpt, Clone)]
#[structopt()]
struct Cli {
    #[structopt(long)]
    port: u16,
    #[structopt(long)]
    embedding_config: PathBuf,
    #[structopt(long)]
    global_config: PathBuf,
    #[structopt(long)]
    replica_index: usize,
    #[structopt(long)]
    replica_size: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
        args.replica_index,
        args.replica_size,
    )?;

    EmbeddingConfig::set(&args.embedding_config)?;

    let intent = &PersiaCommonConfig::get()?.intent;
    let all_shards_client = match intent {
        PerisaIntent::Infer(ref conf) => {
            let servers = conf.servers.clone();
            AllShardsClient::with_addrs(servers)
        }
        _ => {
            let replica_info = PersiaReplicaInfo::get()?.as_ref().clone();
            let nats_publisher = EmbeddingServerNatsStubPublisher {
                nats_client: NatsClient::new(replica_info),
            };

            AllShardsClient::with_nats(nats_publisher)
        }
    };

    let num_shards = all_shards_client.num_shards() as u64;
    let middleware_config = PersiaMiddlewareConfig::get()?;
    let embedding_config = EmbeddingConfig::get()?;

    let inner = Arc::new(ShardedMiddlewareServerInner {
        all_shards_client,
        num_shards,
        forward_id: std::sync::atomic::AtomicU64::new(rand::random()),
        forward_id_buffer: persia_futures::async_lock::RwLock::new(HashMap::with_capacity(10000)),
        post_forward_buffer: persia_futures::async_lock::RwLock::new(HashMap::with_capacity(10000)),
        cannot_forward_batched_time: crossbeam::atomic::AtomicCell::new(
            std::time::SystemTime::now(),
        ),
        embedding_config,
        staleness: Default::default(),
        middleware_config,
    });

    let _responder = match intent {
        PerisaIntent::Infer(_) => None,
        _ => {
            let nats_stub = MiddlewareNatsStub {
                inner: inner.clone(),
            };

            let repilca_info = PersiaReplicaInfo::get()?.as_ref().clone();
            let responder = MiddlewareNatsStubResponder {
                inner: nats_stub,
                nats_client: NatsClient::new(repilca_info),
            };

            responder
                .spawn_subscriptions()
                .expect("failed to spawn nats subscriptions");
            tracing::info!("middleware responder started");

            Some(responder)
        }
    };

    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let service = ShardedMiddlewareServer {
        inner: inner,
        shutdown_channel: Arc::new(persia_futures::async_lock::RwLock::new(Some(tx))),
    };

    let server = hyper::server::Server::bind(&([0, 0, 0, 0], args.port).into())
        .tcp_nodelay(true)
        .serve(hyper::service::make_service_fn(|_| {
            let service = service.clone();
            async { Ok::<_, hyper::Error>(service) }
        }));

    tracing::info!("middleware rpc server started");

    let graceful = server.with_graceful_shutdown(async {
        rx.await.ok();
    });

    if let Err(err) = graceful.await {
        tracing::error!("middleware exited with error: {:?}!", err);
    } else {
        tracing::info!("middleware exited successfully");
    }

    Ok(())
}
