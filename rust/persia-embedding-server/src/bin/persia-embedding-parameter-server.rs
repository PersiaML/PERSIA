#![allow(clippy::needless_return)]

#[macro_use]
extern crate shadow_rs;

use std::{path::PathBuf, sync::Arc};

use persia_libs::{anyhow::Result, color_eyre, hyper, tracing, tracing_subscriber};
use structopt::StructOpt;

use persia_common::utils::start_deadlock_detection_thread;
use persia_embedding_config::{
    EmbeddingConfig, EmbeddingParameterServerConfig, PerisaJobType, PersiaCommonConfig,
    PersiaGlobalConfig,
};
use persia_embedding_holder::PersiaEmbeddingHolder;
use persia_embedding_server::embedding_parameter_service::{
    EmbeddingParameterNatsService, EmbeddingParameterNatsServiceResponder,
    EmbeddingParameterService, EmbeddingParameterServiceInner,
};
use persia_incremental_update_manager::PerisaIncrementalUpdateManager;
use persia_model_manager::EmbeddingModelManager;

#[derive(Debug, StructOpt, Clone)]
#[structopt()]
struct Cli {
    #[structopt(long)]
    port: u16,
    #[structopt(long)]
    replica_index: usize,
    #[structopt(long)]
    replica_size: usize,
    #[structopt(long, env = "PERSIA_GLOBAL_CONFIG")]
    global_config: PathBuf,
    #[structopt(long, env = "PERSIA_EMBEDDING_CONFIG")]
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

    start_deadlock_detection_thread();

    PersiaGlobalConfig::set_configures(
        &args.global_config,
        args.port,
        args.replica_index,
        args.replica_size,
    )?;

    EmbeddingConfig::set(&args.embedding_config)?;

    let embedding_config = EmbeddingConfig::get()?;
    let common_config = PersiaCommonConfig::get()?;
    let server_config = EmbeddingParameterServerConfig::get()?;
    let embedding_holder = PersiaEmbeddingHolder::get()?;
    let inc_update_manager = PerisaIncrementalUpdateManager::get()?;
    let embedding_model_manager = EmbeddingModelManager::get()?;
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();

    let inner = Arc::new(EmbeddingParameterServiceInner::new(
        embedding_holder,
        server_config,
        common_config,
        embedding_config,
        inc_update_manager,
        embedding_model_manager,
        args.replica_index,
    ));

    let service = EmbeddingParameterService {
        inner: inner.clone(),
        shutdown_channel: Arc::new(persia_libs::async_lock::RwLock::new(Some(tx))),
    };

    let server = hyper::Server::bind(&([0, 0, 0, 0], args.port).into())
        .tcp_nodelay(true)
        .serve(hyper::service::make_service_fn(|_| {
            let service = service.clone();
            async move { Ok::<_, hyper::Error>(service) }
        }));

    let job_type = &inner.get_job_type()?;
    let _responder = match job_type {
        PerisaJobType::Infer => None,
        _ => {
            let nats_service = EmbeddingParameterNatsService {
                inner: inner.clone(),
            };
            let responder = EmbeddingParameterNatsServiceResponder::new(nats_service).await;
            Some(responder)
        }
    };

    match job_type {
        PerisaJobType::Infer => {
            let common_config = PersiaCommonConfig::get()?;
            let embedding_cpk = common_config.infer_config.embedding_checkpoint.clone();
            inner.load(embedding_cpk).await?;
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
