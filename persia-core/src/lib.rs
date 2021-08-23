#![allow(clippy::needless_return)]

#[macro_use]
extern crate shadow_rs;

#[cfg(feature = "cuda")]
mod backward;
#[cfg(feature = "cuda")]
mod cuda;
mod data;
#[cfg(feature = "cuda")]
mod forward;
mod nats;
mod optim;
mod utils;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use hashbrown::HashMap;
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use persia_embedding_config::{
    BoundedUniformInitialization, InitializationMethod, PersiaSparseModelHyperparameters,
};
use persia_embedding_sharded_server::sharded_middleware_service::{
    ShardedMiddlewareError, ShardedMiddlewareServerClient,
};
use persia_metrics::{Histogram, IntCounter, PersiaMetricsManager, PersiaMetricsManagerError};
use persia_model_manager::PersiaPersistenceStatus;

static METRICS_HOLDER: once_cell::sync::OnceCell<MetricsHolder> = once_cell::sync::OnceCell::new();
static RPC_CLIENT: OnceCell<Arc<PersiaRpcClient>> = OnceCell::new();

struct MetricsHolder {
    pub forward_client_to_gpu_time_cost: Histogram,
    pub forward_client_time_cost: Histogram,
    pub forward_error: IntCounter,
    pub backward_client_time_cost: Histogram,
    pub long_get_train_batch_time_cost: Histogram,
    pub long_update_gradient_batched_time_cost: Histogram,
}

impl MetricsHolder {
    pub fn get() -> Result<&'static Self, PersiaMetricsManagerError> {
        METRICS_HOLDER.get_or_try_init(|| {
            let m = PersiaMetricsManager::get()?;
            let holder = Self {
                forward_client_to_gpu_time_cost: m
                    .create_histogram("forward_client_to_gpu_time_cost", "ATT")?,
                forward_client_time_cost: m.create_histogram("forward_client_time_cost", "ATT")?,
                forward_error: m.create_counter("forward_error", "ATT")?,
                backward_client_time_cost: m
                    .create_histogram("backward_client_time_cost", "ATT")?,
                long_get_train_batch_time_cost: m
                    .create_histogram("long_get_train_batch_time_cost", "ATT")?,
                long_update_gradient_batched_time_cost: m
                    .create_histogram("long_update_gradient_batched_time_cost", "ATT")?,
            };
            Ok(holder)
        })
    }
}

struct PersiaRpcClient {
    pub clients: RwLock<HashMap<String, Arc<ShardedMiddlewareServerClient>>>,
    pub middleware_addrs: RwLock<Vec<String>>,
    pub runtime: Arc<persia_futures::tokio::runtime::Runtime>,
}

impl PersiaRpcClient {
    pub fn get_instance() -> Arc<PersiaRpcClient> {
        match RPC_CLIENT.get() {
            Some(val) => val.clone(),
            None => panic!("init the persia rpc client first"),
        }
    }

    fn new(worker_size: usize) -> Arc<PersiaRpcClient> {
        RPC_CLIENT
            .get_or_init(|| {
                let runtime = Arc::new(
                    persia_futures::tokio::runtime::Builder::new_multi_thread()
                        .enable_all()
                        .worker_threads(worker_size)
                        .build()
                        .unwrap(),
                );

                Arc::new(Self {
                    clients: RwLock::new(HashMap::new()),
                    middleware_addrs: RwLock::new(vec![]),
                    runtime: runtime,
                })
            })
            .clone()
    }

    fn get_random_client_with_addr(&self) -> (String, Arc<ShardedMiddlewareServerClient>) {
        let middleware_addrs = self.middleware_addrs.read();
        let addr = middleware_addrs[rand::random::<usize>() % middleware_addrs.len()].as_str();
        let client = self.get_client_by_addr(addr);
        (addr.to_string(), client)
    }

    fn get_random_client(&self) -> Arc<ShardedMiddlewareServerClient> {
        return self.get_random_client_with_addr().1;
    }

    fn get_client_by_addr(&self, middleware_addr: &str) -> Arc<ShardedMiddlewareServerClient> {
        if self.clients.read().contains_key(middleware_addr) {
            self.clients.read().get(middleware_addr).unwrap().clone()
        } else {
            let _guard = self.runtime.enter();
            let rpc_client = persia_rpc::RpcClient::new(middleware_addr).unwrap();
            let client = Arc::new(ShardedMiddlewareServerClient::new(rpc_client));

            self.clients
                .write()
                .insert(middleware_addr.to_string(), client.clone());

            self.middleware_addrs
                .write()
                .push(middleware_addr.to_string());
            tracing::info!("created client for middleware {}", middleware_addr);
            client
        }
    }

    pub fn submit_configuration(&self, config: PersiaSparseModelHyperparameters) -> Result<()> {
        tracing::info!("configuring sharded servers: {:#?}", config);
        while self.configure_sharded_servers(&config).is_err() {
            tracing::warn!(
                "configure sharded servers failed, server might be start later ,retrying..."
            );
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
        Ok(())
    }

    fn configure_sharded_servers(
        &self,
        config: &PersiaSparseModelHyperparameters,
    ) -> Result<(), ShardedMiddlewareError> {
        let handler = self.runtime.clone();
        let _guard = handler.enter();
        let result = handler.block_on(persia_futures::tokio::time::timeout(
            Duration::from_secs(10),
            self.clients
                .read()
                .iter()
                .next()
                .expect("clients not initialized")
                .1
                .configure_sharded_servers(&config),
        ));

        if let Ok(Ok(_)) = result {
            Ok(())
        } else {
            Err(ShardedMiddlewareError::RpcError(format!("{:?}", result)))
        }
    }

    fn dump(&self, dst_dir: String) -> Result<(), ShardedMiddlewareError> {
        let handler = self.runtime.clone();
        let _guard = handler.enter();
        handler
            .block_on(
                self.clients
                    .read()
                    .iter()
                    .next()
                    .expect("clients not initialized")
                    .1
                    .dump(&dst_dir),
            )
            .unwrap_or_else(|e| Err(ShardedMiddlewareError::RpcError(format!("{:?}", e))))
    }

    fn load(&self, dst_dir: String) -> Result<(), ShardedMiddlewareError> {
        let handler = self.runtime.clone();
        let _guard = handler.enter();
        handler
            .block_on(
                self.clients
                    .read()
                    .iter()
                    .next()
                    .expect("clients not initialized")
                    .1
                    .load(&dst_dir),
            )
            .unwrap_or_else(|e| Err(ShardedMiddlewareError::RpcError(format!("{:?}", e))))
    }

    fn wait_for_serving(&self) -> PyResult<()> {
        let handler = self.runtime.clone();
        let _guard = handler.enter();
        let client = self
            .clients
            .read()
            .iter()
            .next()
            .expect("clients not initialized")
            .1
            .clone();

        loop {
            if let Ok(ready) = handler.block_on(client.ready_for_serving(&())) {
                if ready {
                    return Ok(());
                }
                std::thread::sleep(Duration::from_secs(5));
                let status: Vec<PersiaPersistenceStatus> =
                    handler.block_on(client.model_manager_status(&())).unwrap();

                match self.process_status(status) {
                    Ok(_) => {}
                    Err(err_msg) => {
                        return Err(PyRuntimeError::new_err(err_msg));
                    }
                }
            } else {
                tracing::warn!("failed to get sparse model status, retry later");
            }
        }
    }

    pub fn shutdown(&self) -> PyResult<()> {
        let handler = self.runtime.clone();
        let _guard = handler.enter();

        let client = self.get_random_client();

        match handler.block_on(client.shutdown_server(&())) {
            Ok(response) => match response {
                Ok(_) => {
                    let clients = self.clients.read();
                    let mut futs = clients
                        .iter()
                        .map(|client| handler.block_on(client.1.shutdown(&())));

                    let middleware_shutdown_status = futs.all(|x| x.is_ok());
                    if middleware_shutdown_status {
                        Ok(())
                    } else {
                        Err(PyRuntimeError::new_err("shutdown middleware failed"))
                    }
                }
                Err(err) => {
                    tracing::error!("shutdown server failed, Rpc error: {:?}", err);
                    Err(PyRuntimeError::new_err(err.to_string()))
                }
            },
            Err(err) => {
                tracing::error!("shutdown server failed, Rpc error: {:?}", err);
                Err(PyRuntimeError::new_err(err.to_string()))
            }
        }
    }

    fn wait_for_emb_dumping(&self) -> PyResult<()> {
        let handler = self.runtime.clone();
        let _guard = handler.enter();
        let client = self
            .clients
            .read()
            .iter()
            .next()
            .expect("clients not initialized")
            .1
            .clone();

        loop {
            std::thread::sleep(Duration::from_secs(5));
            let status: Result<Vec<PersiaPersistenceStatus>, _> =
                handler.block_on(client.model_manager_status(&()));
            if let Ok(status) = status {
                if status.iter().any(|s| match s {
                    PersiaPersistenceStatus::Loading(_) => true,
                    _ => false,
                }) {
                    let err_msg = String::from("emb status is loading but waiting for dump.");
                    return Err(PyRuntimeError::new_err(err_msg));
                }
                let num_total = status.len();
                match self.process_status(status) {
                    Ok(num_compeleted) => {
                        if num_compeleted == num_total {
                            return Ok(());
                        }
                    }
                    Err(err_msg) => {
                        return Err(PyRuntimeError::new_err(err_msg));
                    }
                }
            } else {
                tracing::warn!("failed to get sparse model status, retry later");
            }
        }
    }

    fn process_status(&self, status: Vec<PersiaPersistenceStatus>) -> Result<usize, String> {
        let mut num_compeleted: usize = 0;
        let mut errors = Vec::new();
        status
            .into_iter()
            .enumerate()
            .for_each(|(shard_idx, s)| match s {
                PersiaPersistenceStatus::Failed(e) => {
                    let err_msg = format!("emb dump FAILED for shard {}, due to {}.", shard_idx, e);
                    errors.push(err_msg);
                }
                PersiaPersistenceStatus::Loading(p) => {
                    tracing::info!(
                        "loading emb for shard {}, pregress: {:?}%",
                        shard_idx,
                        p * 100.0
                    );
                }
                PersiaPersistenceStatus::Idle => {
                    num_compeleted = num_compeleted + 1;
                }
                PersiaPersistenceStatus::Dumping(p) => {
                    tracing::info!(
                        "dumping emb for shard {}, pregress: {:?}%",
                        shard_idx,
                        p * 100.0
                    );
                }
            });
        if errors.len() > 0 {
            Err(errors.join(", "))
        } else {
            Ok(num_compeleted)
        }
    }
}

#[pymethods]
impl PyPersiaRpcClient {
    #[new]
    pub fn new(worker_size: usize) -> Self {
        PyPersiaRpcClient {
            inner: PersiaRpcClient::new(worker_size),
        }
    }

    pub fn add_rpc_client(&mut self, addr: &str) -> PyResult<()> {
        let _ = self.inner.get_client_by_addr(addr);
        Ok(())
    }

    pub fn set_configuration(
        &mut self,
        initialize_lower: f32,
        initialize_upper: f32,
        admit_probability: f32,
        enable_weight_bound: bool,
        weight_bound: f32,
    ) -> PyResult<()> {
        assert!(
            (0. <= admit_probability) && (admit_probability <= 1.),
            "admit probability should be within 0 ~ 1"
        );
        let config = PersiaSparseModelHyperparameters {
            initialization_method: InitializationMethod::BoundedUniform(
                BoundedUniformInitialization {
                    lower: initialize_lower,
                    upper: initialize_upper,
                },
            ),
            admit_probability,
            weight_bound,
            enable_weight_bound,
        };

        self.inner
            .submit_configuration(config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    pub fn shutdown_service(&self) -> PyResult<()> {
        self.inner.shutdown()
    }

    pub fn wait_for_load_embedding(&self) -> PyResult<()> {
        self.inner.wait_for_serving()
    }

    pub fn wait_for_dump_embedding(&self) -> PyResult<()> {
        self.inner.wait_for_emb_dumping()
    }

    pub fn dump_embedding(&self, dst_dir: String) -> PyResult<()> {
        self.inner
            .dump(dst_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
    }

    pub fn load_embedding(&self, dst_dir: String) -> PyResult<()> {
        self.inner
            .load(dst_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{:?}", e)))
    }
}

#[pyclass]
pub struct PyPersiaRpcClient {
    inner: Arc<PersiaRpcClient>,
}

#[pyfunction]
pub fn is_cuda_feature_available() -> bool {
    if cfg!(feature = "cuda") {
        true
    } else {
        false
    }
}

#[pymodule]
fn persia_core(py: Python, m: &PyModule) -> PyResult<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("LOG_LEVEL"))
        .init();
    color_eyre::install().unwrap();

    if std::env::var("HTTP_PROXY").is_ok() || std::env::var("http_proxy").is_ok() {
        tracing::warn!("http_proxy environment is set, this is generally not what we want, please double check");
    }

    m.add_class::<PyPersiaRpcClient>()?;

    data::init_module(m, py)?;
    utils::init_module(m, py)?;
    optim::init_module(m, py)?;
    nats::init_module(m, py)?;
    m.add_function(wrap_pyfunction!(is_cuda_feature_available, m)?)?;

    #[cfg(feature = "cuda")]
    {
        forward::init_module(m, py)?;
        backward::init_module(m, py)?;
    }

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

    Ok(())
}
