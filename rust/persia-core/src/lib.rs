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
mod metrics;
mod nats;
mod optim;
mod rpc;
mod utils;

use crate::data::PyPersiaBatchData;
#[cfg(feature = "cuda")]
use crate::forward::{forward_directly, PythonTrainBatch};
use crate::optim::PyOptimizerBase;
use crate::rpc::PersiaRpcClient;

use std::sync::Arc;

use persia_libs::{
    anyhow::Result,
    color_eyre,
    once_cell::sync::OnceCell,
    parking_lot::RwLock,
    thiserror,
    tokio::{self, runtime::Runtime},
    tracing, tracing_subscriber,
};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;

use persia_common::PersiaBatchData;
use persia_embedding_config::{PersiaGlobalConfigError, PersiaReplicaInfo};
use persia_embedding_server::middleware_service::MiddlewareServerError;
use persia_speedy::Readable;

#[derive(thiserror::Error, Debug)]
pub enum PersiaError {
    #[error("Persia context NOT initialized")]
    NotInitializedError,
    #[error("enter persia context multiple times")]
    MultipleContextError,
    #[error("shutdown server failed: {0}")]
    ShutdownError(String),
    #[error("server dump/load status error: {0}")]
    ServerStatusError(String),
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("server side error: {0}")]
    ServerSideError(#[from] MiddlewareServerError),
    #[error("rpc error: {0}")]
    RpcError(#[from] persia_rpc::PersiaRpcError),
    #[error("nats error: {0}")]
    NatsError(#[from] persia_nats_client::NatsError),
    #[error("send sparse data to middleware server multi times")]
    MultipleSendError,
    #[error("sparse data is null, please call batch.add_sparse first")]
    NullSparseDataError,
    #[error("batch id is null, please call send_sparse_to_middleware first")]
    NullBatchIdError,
    #[error("sparse optimizer not set yet")]
    NullOptimizerError,
    #[error("data send failed")]
    SendDataError,
    #[error("nats publisher not initialized")]
    NatsNotInitializedError,
    #[error("LeaderDiscoveryService not initialized")]
    LeaderDiscoveryServiceNotInitializedError,
    #[error("leader addr input wrong")]
    LeaderAddrInputError,
}

impl PersiaError {
    pub fn to_py_runtime_err(&self) -> PyErr {
        PyRuntimeError::new_err(format!("{:?}", self))
    }
}

static PERSIA_COMMON_CONTEXT: OnceCell<Arc<PersiaCommonContext>> = OnceCell::new();

struct PersiaCommonContext {
    pub rpc_client: Arc<PersiaRpcClient>,
    pub nats_publisher: Arc<RwLock<Option<nats::PersiaBatchFlowNatsServicePublisherWrapper>>>,
    pub leader_discovery_service: Arc<RwLock<Option<nats::LeaderDiscoveryNatsServiceWrapper>>>,
    pub async_runtime: Arc<Runtime>,
}

impl PersiaCommonContext {
    pub fn get() -> Arc<Self> {
        PERSIA_COMMON_CONTEXT
            .get()
            .expect("not in persia context")
            .clone()
    }

    pub fn new(
        num_coroutines_worker: usize,
        replica_index: usize,
        replica_size: usize,
    ) -> Result<Arc<Self>, PersiaError> {
        if let Some(instance) = PERSIA_COMMON_CONTEXT.get() {
            return Ok(instance.clone());
        }
        let _ = PersiaReplicaInfo::set(replica_size, replica_index);
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(num_coroutines_worker)
                .build()
                .unwrap(),
        );

        let rpc_client = Arc::new(PersiaRpcClient::new(runtime.clone()));

        let common_context = Self {
            rpc_client,
            nats_publisher: Arc::new(RwLock::new(None)),
            leader_discovery_service: Arc::new(RwLock::new(None)),
            async_runtime: runtime,
        };

        let instance = Arc::new(common_context);
        let result = PERSIA_COMMON_CONTEXT.set(instance.clone());
        if result.is_err() {
            tracing::warn!("calling init persia common context multiple times");
        }

        Ok(instance)
    }

    pub fn register_optimizer(&self, opt: &PyOptimizerBase) -> Result<(), PersiaError> {
        self.nats_publisher
            .read()
            .as_ref()
            .ok_or_else(|| PersiaError::NatsNotInitializedError)?
            .register_optimizer(opt)
    }

    pub fn init_rpc_client_with_addr(&self, addr: String) -> Result<(), PersiaError> {
        let _ = self.rpc_client.get_client_by_addr(&addr);
        Ok(())
    }
}

#[pyclass]
pub struct PyPersiaCommonContext {
    inner: Arc<PersiaCommonContext>,
}

#[pymethods]
impl PyPersiaCommonContext {
    #[new]
    pub fn new(
        num_coroutines_worker: usize,
        replica_index: usize,
        replica_size: usize,
    ) -> PyResult<Self> {
        let inner = PersiaCommonContext::new(num_coroutines_worker, replica_index, replica_size)
            .map_err(|e| e.to_py_runtime_err())?;
        Ok(Self { inner })
    }

    pub fn init_nats_publisher(&self, world_size: Option<usize>) -> PyResult<()> {
        let instance = nats::PersiaBatchFlowNatsServicePublisherWrapper::new(
            world_size,
            self.inner.async_runtime.clone(),
        );
        let mut nats_publisher = self.inner.nats_publisher.write();
        *nats_publisher = Some(instance);
        Ok(())
    }

    pub fn init_leader_discovery_service(&self, leader_addr: Option<String>) -> PyResult<()> {
        let replica_info = PersiaReplicaInfo::get().expect("not in persia context");
        if replica_info.is_leader() == leader_addr.is_none() {
            return Err(PersiaError::LeaderAddrInputError.to_py_runtime_err());
        }
        let instance = nats::LeaderDiscoveryNatsServiceWrapper::new(
            leader_addr,
            self.inner.async_runtime.clone(),
        );
        let mut leader_discovery_service = self.inner.leader_discovery_service.write();
        *leader_discovery_service = Some(instance);
        Ok(())
    }

    pub fn get_leader_addr(&self) -> PyResult<String> {
        let leader_addr = self
            .inner
            .leader_discovery_service
            .read()
            .as_ref()
            .ok_or_else(|| PersiaError::LeaderDiscoveryServiceNotInitializedError)
            .map_err(|e| e.to_py_runtime_err())?
            .get_leader_addr();
        Ok(leader_addr)
    }

    pub fn init_rpc_client_with_addr(&self, middleware_addr: String) -> PyResult<()> {
        self.inner
            .init_rpc_client_with_addr(middleware_addr)
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn wait_servers_ready(&self) -> PyResult<String> {
        self.inner
            .nats_publisher
            .read()
            .as_ref()
            .ok_or_else(|| PersiaError::NatsNotInitializedError)
            .map_err(|e| e.to_py_runtime_err())?
            .wait_servers_ready()
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn get_embedding_size(&self) -> PyResult<Vec<usize>> {
        self.inner
            .rpc_client
            .get_embedding_size()
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn clear_embeddings(&self) -> PyResult<()> {
        self.inner
            .rpc_client
            .clear_embeddings()
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn dump(&self, dst_dir: String) -> PyResult<()> {
        self.inner
            .rpc_client
            .dump(dst_dir)
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn load(&self, src_dir: String) -> PyResult<()> {
        self.inner
            .rpc_client
            .load(src_dir)
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn wait_for_serving(&self) -> PyResult<()> {
        self.inner
            .rpc_client
            .wait_for_serving()
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn wait_for_emb_dumping(&self) -> PyResult<()> {
        self.inner
            .rpc_client
            .wait_for_emb_dumping()
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn shutdown_servers(&self) -> PyResult<()> {
        self.inner
            .rpc_client
            .shutdown()
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn send_sparse_to_middleware(
        &self,
        batch: &mut PyPersiaBatchData,
        block: bool,
    ) -> PyResult<()> {
        self.inner
            .nats_publisher
            .read()
            .as_ref()
            .ok_or_else(|| PersiaError::NatsNotInitializedError)
            .map_err(|e| e.to_py_runtime_err())?
            .send_sparse_to_middleware(batch, block)
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn send_dense_to_trainer(&self, batch: &PyPersiaBatchData, block: bool) -> PyResult<()> {
        self.inner
            .nats_publisher
            .read()
            .as_ref()
            .ok_or_else(|| PersiaError::NatsNotInitializedError)
            .map_err(|e| e.to_py_runtime_err())?
            .send_dense_to_trainer(batch, block)
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn configure_embedding_servers(
        &self,
        initialize_lower: f32,
        initialize_upper: f32,
        admit_probability: f32,
        enable_weight_bound: bool,
        weight_bound: f32,
    ) -> PyResult<()> {
        self.inner
            .nats_publisher
            .read()
            .as_ref()
            .ok_or_else(|| PersiaError::NatsNotInitializedError)
            .map_err(|e| e.to_py_runtime_err())?
            .configure_embedding_servers(
                initialize_lower,
                initialize_upper,
                admit_probability,
                enable_weight_bound,
                weight_bound,
            )
            .map_err(|e| e.to_py_runtime_err())
    }

    #[cfg(feature = "cuda")]
    pub fn get_embedding_from_data(
        &self,
        batch: &mut PyPersiaBatchData,
        device_id: i32,
    ) -> PyResult<PythonTrainBatch> {
        let batch = std::mem::replace(&mut batch.inner, PersiaBatchData::default());
        forward_directly(batch, device_id)
    }

    #[cfg(feature = "cuda")]
    pub fn get_embedding_from_bytes(
        &self,
        batch: &PyBytes,
        device_id: i32,
    ) -> PyResult<PythonTrainBatch> {
        let batch: PersiaBatchData = PersiaBatchData::read_from_buffer(batch.as_bytes()).unwrap();
        forward_directly(batch, device_id)
    }
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

    m.add_class::<PyPersiaCommonContext>()?;

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
