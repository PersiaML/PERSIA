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

use anyhow::Result;
use once_cell::sync::OnceCell;
use std::sync::Arc;

use persia_futures::tokio::runtime::Runtime;
use persia_speedy::Readable;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;
use thiserror::Error;

use persia_embedding_config::{PersiaGlobalConfigError, PersiaReplicaInfo};
use persia_embedding_datatypes::PersiaBatchData;
use persia_embedding_sharded_server::sharded_middleware_service::ShardedMiddlewareError;

#[derive(Error, Debug)]
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
    ServerSideError(#[from] ShardedMiddlewareError),
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
}

impl PersiaError {
    pub fn to_py_runtime_err(&self) -> PyErr {
        PyRuntimeError::new_err(format!("{:?}", self))
    }
}

static PERSIA_COMMON_CONTEXT: OnceCell<Arc<PersiaCommonContext>> = OnceCell::new();

struct PersiaCommonContext {
    pub rpc_client: Arc<PersiaRpcClient>,
    pub nats_publisher: Arc<nats::PersiaBatchFlowNatsStubPublisherWrapper>,
    pub async_runtime: Arc<Runtime>,
}

impl PersiaCommonContext {
    pub fn get() -> Arc<Self> {
        PERSIA_COMMON_CONTEXT
            .get()
            .expect("not in persia context")
            .clone()
    }

    pub fn init(
        num_coroutines_worker: usize,
        replica_index: usize,
        replica_size: usize,
        world_size: Option<usize>,
    ) -> Result<Arc<Self>, PersiaError> {
        if let Some(instance) = PERSIA_COMMON_CONTEXT.get() {
            return Ok(instance.clone());
        }
        let runtime = Arc::new(
            persia_futures::tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(num_coroutines_worker)
                .build()
                .unwrap(),
        );

        let rpc_client = Arc::new(PersiaRpcClient::new(runtime.clone()));

        let _ = PersiaReplicaInfo::set(replica_size, replica_index);
        let nats_publisher = Arc::new(nats::PersiaBatchFlowNatsStubPublisherWrapper::new(
            world_size,
            runtime.clone(),
        ));

        let common_context = Self {
            rpc_client,
            nats_publisher,
            async_runtime: runtime,
        };

        let addr = common_context.wait_servers_ready()?;
        common_context.init_rpc_client_with_addr(addr)?;

        let instance = Arc::new(common_context);
        let result = PERSIA_COMMON_CONTEXT.set(instance.clone());
        if result.is_err() {
            tracing::warn!("calling init persia common context multiple times");
        }

        Ok(instance)
    }

    pub fn register_optimizer(&self, opt: &PyOptimizerBase) -> Result<(), PersiaError> {
        self.nats_publisher.register_optimizer(opt)
    }

    pub fn wait_servers_ready(&self) -> Result<String, PersiaError> {
        let addr = self.nats_publisher.wait_servers_ready()?;
        Ok(addr)
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
        world_size: Option<usize>,
    ) -> PyResult<Self> {
        let inner = PersiaCommonContext::init(
            num_coroutines_worker,
            replica_index,
            replica_size,
            world_size,
        )
        .map_err(|e| e.to_py_runtime_err())?;
        Ok(Self { inner })
    }

    pub fn get_embedding_size(&self) -> PyResult<Vec<usize>> {
        self.inner
            .rpc_client
            .get_embedding_size()
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
            .send_sparse_to_middleware(batch, block)
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn send_dense_to_trainer(&self, batch: &PyPersiaBatchData, block: bool) -> PyResult<()> {
        self.inner
            .nats_publisher
            .send_dense_to_trainer(batch, block)
            .map_err(|e| e.to_py_runtime_err())
    }

    pub fn configure_sharded_servers(
        &self,
        initialize_lower: f32,
        initialize_upper: f32,
        admit_probability: f32,
        enable_weight_bound: bool,
        weight_bound: f32,
    ) -> PyResult<()> {
        self.inner
            .nats_publisher
            .configure_sharded_servers(
                initialize_lower,
                initialize_upper,
                admit_probability,
                enable_weight_bound,
                weight_bound,
            )
            .map_err(|e| e.to_py_runtime_err())
    }

    #[cfg(feature = "cuda")]
    pub fn forward_directly_from_data(
        &self,
        batch: &mut PyPersiaBatchData,
        device_id: i32,
    ) -> PyResult<PythonTrainBatch> {
        let batch = std::mem::replace(&mut batch.inner, PersiaBatchData::default());
        forward_directly(batch, device_id)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_directly_from_bytes(
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
