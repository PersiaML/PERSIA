#[macro_use]
extern crate shadow_rs;
use hashbrown::HashMap;
use persia_embedding_config::PersiaGlobalConfig;
use persia_embedding_sharded_server::hashmap_sharded_service::HashMapShardedServiceClient;
use persia_embedding_sharded_server::middleware_config_parser::{
    convert_middleware_config, MiddlewareConfig,
};
use persia_embedding_sharded_server::sharded_middleware_service::{
    AllShardsClient, ShardedMiddlewareServer, ShardedMiddlewareServerInner,
};
use std::path::PathBuf;
use std::sync::Arc;
use structopt::StructOpt;

#[derive(Debug, StructOpt, Clone)]
#[structopt()]
struct Cli {
    #[structopt(long)]
    servers: Vec<String>,
    #[structopt(long)]
    port: u16,
    #[structopt(long)]
    config: PathBuf,
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

    PersiaGlobalConfig::set(
        args.global_config,
        args.replica_index,
        args.replica_size,
        String::from("middleware"),
    )?;

    let num_shards = args.servers.len() as u64;

    let mut clients = Vec::new();
    for server_addr in args.servers {
        let rpc_client = persia_rpc::RpcClient::new(server_addr.as_str()).unwrap();
        let client = HashMapShardedServiceClient::new(rpc_client);
        clients.push(Arc::new(client));
    }

    clients.sort_by_key(|c| {
        while persia_futures::smol::block_on(c.shard_idx(&())).is_err() {
            persia_futures::smol::block_on(persia_futures::tokio::time::sleep(
                persia_futures::tokio::time::Duration::from_secs(1),
            ));
        }
        persia_futures::smol::block_on(c.shard_idx(&())).expect("cannot get server shard idx");
    });

    for (i, c) in clients.iter().enumerate() {
        while c.shard_idx(&()).await.is_err() {
            assert_eq!(i, c.shard_idx(&()).await?, "shard idx mismatch");
        }
    }

    let all_shards_client = AllShardsClient::new(clients);

    while !args.config.is_file() {
        tracing::warn!("waiting for middleware config yaml file...");
    }
    let mut config: MiddlewareConfig =
        serde_yaml::from_reader(std::fs::File::open(args.config).expect("cannot read config file"))
            .expect("cannot parse config file");

    let feature2group = convert_middleware_config(&mut config);

    tracing::info!("middleware config: {:#?}", config);

    let global_config = PersiaGlobalConfig::get()?;
    let guard = global_config.read();

    let service = ShardedMiddlewareServer {
        inner: Arc::new(ShardedMiddlewareServerInner {
            all_shards_client,
            num_shards,
            forward_id: std::sync::atomic::AtomicU64::new(rand::random()),
            forward_id_buffer: persia_futures::async_lock::RwLock::new(HashMap::with_capacity(
                10000,
            )),
            post_forward_buffer: persia_futures::async_lock::RwLock::new(HashMap::with_capacity(
                10000,
            )),
            cannot_forward_batched_time: crossbeam::atomic::AtomicCell::new(
                std::time::SystemTime::now(),
            ),
            config,
            staleness: Default::default(),
            feature2group,
            forward_buffer_size: guard.middleware_config.forward_buffer_size,
        }),
    };

    let server = hyper::server::Server::bind(&([0, 0, 0, 0], args.port).into())
        // .http2_only(true)
        // .http2_adaptive_window(true)
        .tcp_nodelay(true)
        .serve(hyper::service::make_service_fn(|_| {
            let service = service.clone();
            async { Ok::<_, hyper::Error>(service) }
        }));

    server.await?;

    Ok(())
}
