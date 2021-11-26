#![allow(clippy::needless_return)]

#[macro_use]
extern crate shadow_rs;

mod backward;
mod data;
mod dlpack;
mod forward;
mod metrics;
mod nats;
mod optim;
mod rpc;
mod tensor;
mod utils;

#[cfg(feature = "cuda")]
mod cuda;

use crate::data::{PersiaBatch, PersiaBatchImpl};
use crate::forward::{forward_directly, PersiaTrainingBatch};
use crate::optim::OptimizerBase;
use crate::rpc::PersiaRpcClient;

use std::path::PathBuf;
use std::sync::Arc;

use numpy::PyArray1;
use persia_libs::{
    anyhow::Result,
    color_eyre,
    once_cell::sync::OnceCell,
    parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard},
    thiserror,
    tokio::{self, runtime::Runtime},
    tracing, tracing_subscriber,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;

use persia_common::utils::start_deadlock_detection_thread;
use persia_embedding_config::{PersiaGlobalConfigError, PersiaReplicaInfo};
use persia_embedding_holder::emb_entry::HashMapEmbeddingEntry;
use persia_embedding_server::embedding_worker_service::EmbeddingWorkerError;
use persia_speedy::Readable;
use persia_storage::{PersiaPath, PersiaPathImpl};

#[derive(thiserror::Error, Debug)]
pub enum PersiaError {
    #[error("Persia context NOT initialized")]
    NotInitializedError,
    #[error("Enter persia context multiple times")]
    MultipleContextError,
    #[error("Shutdown server failed: {0}")]
    ShutdownError(String),
    #[error("Server dump/load status error: {0}")]
    ServerStatusError(String),
    #[error("Global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("Server side error: {0}")]
    ServerSideError(#[from] EmbeddingWorkerError),
    #[error("Rpc error: {0}")]
    RpcError(#[from] persia_rpc::PersiaRpcError),
    #[error("Nats error: {0}")]
    NatsError(#[from] persia_nats_client::NatsError),
    #[error("Send ID type features to embedding worker multiple times")]
    MultipleSendError,
    #[error("Id type features is null, please call batch.add_id_type_features before sending PersiaBatch")]
    NullIDTypeFeaturesError,
    #[error("Batch id is null, please call send_id_type_features_to_embedding_worker first")]
    NullBatchIdError,
    #[error("Embedding optimizer not set yet")]
    NullOptimizerError,
    #[error("Data send failed")]
    SendDataError,
    #[error("Nats publisher not initialized")]
    NatsNotInitializedError,
    #[error("MasterDiscoveryService not initialized")]
    MasterDiscoveryServiceNotInitializedError,
    #[error("Master service empty")]
    MasterServiceEmpty,
    #[error("Storage visit error {0}")]
    StorageVisitError(String),
    #[error("Master discovery error: {0}")]
    MasterDiscoveryError(#[from] nats::master_discovery_service::Error),
    #[error("Dataflow error: {0}")]
    PersiaBatchFlowError(#[from] nats::persia_dataflow_service::Error),
    #[error("Load emb error")]
    LoadEmbError,
}

impl From<PersiaError> for PyErr {
    fn from(e: PersiaError) -> Self {
        PyRuntimeError::new_err(e.to_string())
    }
}

static PERSIA_COMMON_CONTEXT: OnceCell<Arc<PersiaCommonContextImpl>> = OnceCell::new();

struct PersiaCommonContextImpl {
    pub rpc_client: Arc<PersiaRpcClient>,
    pub nats_publisher: RwLock<Option<nats::PersiaDataFlowComponent>>,
    pub master_discovery_service: RwLock<Option<nats::MasterDiscoveryComponent>>,
    pub async_runtime: Arc<Runtime>,
    pub device_id: Arc<Option<i32>>,
}

impl PersiaCommonContextImpl {
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
        device_id: Option<i32>,
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

        let rpc_client = Arc::new(PersiaRpcClient::new());

        #[cfg(feature = "cuda")]
        {
            if let Some(device_id) = device_id.as_ref() {
                {
                    use crate::cuda::set_device;

                    set_device(*device_id);
                }
            }
        }

        let common_context = Self {
            rpc_client,
            nats_publisher: RwLock::new(None),
            master_discovery_service: RwLock::new(None),
            async_runtime: runtime,
            device_id: Arc::new(device_id),
        };

        let instance = Arc::new(common_context);
        let result = PERSIA_COMMON_CONTEXT.set(instance.clone());
        if result.is_err() {
            tracing::warn!("calling init persia common context multiple times");
        }

        Ok(instance)
    }

    pub fn register_optimizer(&self, optimizer: &OptimizerBase) -> Result<(), PersiaError> {
        self.async_runtime.block_on(
            self.get_nats_publish_service()?
                .register_optimizer(optimizer),
        )
    }

    fn get_nats_publish_service(
        &self,
    ) -> Result<MappedRwLockReadGuard<nats::PersiaDataFlowComponent>, PersiaError> {
        let guard = self.nats_publisher.read();
        if guard.as_ref().is_none() {
            return Err(PersiaError::NatsNotInitializedError);
        }
        let result = RwLockReadGuard::map(guard, |x| x.as_ref().unwrap());
        Ok(result)
    }

    pub fn init_rpc_client_with_addr(&self, addr: String) -> Result<(), PersiaError> {
        let _ = self.rpc_client.get_client_by_addr(&addr);
        Ok(())
    }
}

#[pyclass]
pub struct PersiaCommonContext {
    inner: Arc<PersiaCommonContextImpl>,
}

#[pymethods]
impl PersiaCommonContext {
    #[new]
    pub fn new(
        num_coroutines_worker: usize,
        replica_index: usize,
        replica_size: usize,
        device_id: Option<i32>,
    ) -> PyResult<Self> {
        let inner = PersiaCommonContextImpl::new(
            num_coroutines_worker,
            replica_index,
            replica_size,
            device_id,
        )
        .map_err(|e| PyErr::from(e))?;
        Ok(Self { inner })
    }

    pub fn init_nats_publisher(&mut self, world_size: Option<usize>) -> PyResult<()> {
        if self.inner.nats_publisher.read().is_some() {
            return Ok(());
        }
        let nats_publisher = self
            .inner
            .async_runtime
            .block_on(nats::PersiaDataFlowComponent::new_initialized(world_size))
            .map_err(|e| PyErr::from(e))?;

        *self.inner.nats_publisher.write() = Some(nats_publisher);
        Ok(())
    }

    pub fn init_master_discovery_service(&mut self, master_addr: Option<String>) -> PyResult<()> {
        if self.inner.master_discovery_service.read().is_some() {
            return Ok(());
        }
        let replica_info = PersiaReplicaInfo::get().expect("not in persia context");
        if replica_info.is_master() && master_addr.is_none() {
            return Err(PyErr::from(PersiaError::MasterServiceEmpty));
        }

        let master_discovery_service = self
            .inner
            .async_runtime
            .block_on(nats::MasterDiscoveryComponent::new(master_addr));

        *self.inner.master_discovery_service.write() = Some(master_discovery_service);
        Ok(())
    }

    #[getter]
    pub fn master_addr(&self) -> PyResult<String> {
        self.inner
            .async_runtime
            .block_on(
                self.inner
                    .master_discovery_service
                    .read()
                    .as_ref()
                    .ok_or_else(|| PersiaError::MasterDiscoveryServiceNotInitializedError)
                    .map_err(|e| PyErr::from(e))?
                    .get_master_addr(),
            )
            .map_err(|e| e.into())
    }

    pub fn get_embedding_worker_addr_list(&self) -> PyResult<Vec<String>> {
        self.inner
            .async_runtime
            .block_on(
                self.inner
                    .get_nats_publish_service()?
                    .get_embedding_worker_addr_list(),
            )
            .map_err(|e| e.into())
    }

    pub fn init_rpc_client_with_addr(&self, embedding_worker_addr: String) -> PyResult<()> {
        self.inner
            .init_rpc_client_with_addr(embedding_worker_addr)
            .map_err(|e| e.into())
    }

    pub fn wait_servers_ready(&self) -> PyResult<String> {
        self.inner
            .async_runtime
            .block_on(self.inner.get_nats_publish_service()?.wait_servers_ready())
            .map_err(|e| e.into())
    }

    pub fn get_embedding_size(&self) -> PyResult<Vec<usize>> {
        self.inner
            .async_runtime
            .block_on(self.inner.rpc_client.get_embedding_size())
            .map_err(|e| e.into())
    }

    pub fn clear_embeddings(&self) -> PyResult<()> {
        self.inner
            .async_runtime
            .block_on(self.inner.rpc_client.clear_embeddings())
            .map_err(|e| e.into())
    }

    pub fn dump(&self, dst_dir: String) -> PyResult<()> {
        self.inner
            .async_runtime
            .block_on(self.inner.rpc_client.dump(dst_dir))
            .map_err(|e| e.into())
    }

    pub fn load(&self, src_dir: String) -> PyResult<()> {
        self.inner
            .async_runtime
            .block_on(self.inner.rpc_client.load(src_dir))
            .map_err(|e| e.into())
    }

    pub fn wait_for_serving(&self) -> PyResult<()> {
        self.inner
            .async_runtime
            .block_on(self.inner.rpc_client.wait_for_serving())
            .map_err(|e| e.into())
    }

    pub fn wait_for_emb_loading(&self) -> PyResult<()> {
        self.inner
            .async_runtime
            .block_on(self.inner.rpc_client.wait_for_emb_loading())
            .map_err(|e| e.into())
    }

    pub fn wait_for_emb_dumping(&self) -> PyResult<()> {
        self.inner
            .async_runtime
            .block_on(self.inner.rpc_client.wait_for_emb_dumping())
            .map_err(|e| e.into())
    }

    pub fn shutdown_servers(&self) -> PyResult<()> {
        self.inner
            .async_runtime
            .block_on(self.inner.rpc_client.shutdown())
            .map_err(|e| e.into())
    }

    pub fn send_id_type_features_to_embedding_worker(
        &self,
        batch: &mut PersiaBatch,
    ) -> PyResult<()> {
        self.inner
            .async_runtime
            .block_on(
                self.inner
                    .get_nats_publish_service()?
                    .send_id_type_features_to_embedding_worker(batch),
            )
            .map_err(|e| e.into())
    }

    pub fn send_non_id_type_features_to_nn_worker(&self, batch: &PersiaBatch) -> PyResult<()> {
        self.inner
            .async_runtime
            .block_on(
                self.inner
                    .get_nats_publish_service()?
                    .send_non_id_type_features_to_nn_worker(batch),
            )
            .map_err(|e| e.into())
    }

    pub fn configure_embedding_parameter_servers(
        &self,
        initialize_lower: f32,
        initialize_upper: f32,
        admit_probability: f32,
        enable_weight_bound: bool,
        weight_bound: f32,
    ) -> PyResult<()> {
        self.inner
            .async_runtime
            .block_on(
                self.inner
                    .get_nats_publish_service()?
                    .configure_embedding_parameter_servers(
                        initialize_lower,
                        initialize_upper,
                        admit_probability,
                        enable_weight_bound,
                        weight_bound,
                    ),
            )
            .map_err(|e| e.into())
    }

    pub fn get_embedding_from_data(
        &self,
        batch: &mut PersiaBatch,
        device_id: Option<i32>,
    ) -> PyResult<PersiaTrainingBatch> {
        let batch = std::mem::take(&mut batch.inner);
        forward_directly(batch, device_id)
    }

    pub fn get_embedding_from_bytes(
        &self,
        batch: &PyBytes,
        device_id: Option<i32>,
    ) -> PyResult<PersiaTrainingBatch> {
        let batch: PersiaBatchImpl = PersiaBatchImpl::read_from_buffer(batch.as_bytes()).unwrap();
        forward_directly(batch, device_id)
    }

    pub fn read_from_file<'a>(&self, file_path: String, py: Python<'a>) -> PyResult<&'a PyBytes> {
        let file_path = PersiaPath::from_string(file_path);
        file_path
            .read_to_end()
            .map_err(|e| PyErr::from(PersiaError::StorageVisitError(e.to_string())))
            .map(|content| PyBytes::new(py, content.as_slice()))
    }

    pub fn dump_to_file(
        &self,
        content: &PyBytes,
        file_dir: String,
        file_name: String,
    ) -> PyResult<()> {
        let file_dir = PathBuf::from(file_dir);
        let file_name = PathBuf::from(file_name);
        let file_path = PersiaPath::from_vec(vec![&file_dir, &file_name]);
        let content = content.as_bytes().to_vec();
        file_path
            .write_all(content)
            .map_err(|e| PersiaError::StorageVisitError(e.to_string()).into())
    }

    // Currently only used for debug
    pub fn set_embedding(
        &self,
        embeddings: Vec<(u64, &PyArray1<f32>, &PyArray1<f32>)>,
    ) -> PyResult<()> {
        let entries: Vec<HashMapEmbeddingEntry> = embeddings
            .iter()
            .map(|(sign, emb, opt)| {
                let emb = emb.to_vec().expect("convert ndarray to vec failed");
                let opt = opt.to_vec().expect("convert ndarray to vec failed");
                HashMapEmbeddingEntry::from_emb_and_opt(emb, opt.as_slice(), *sign)
            })
            .collect();
        self.inner
            .async_runtime
            .block_on(self.inner.rpc_client.set_embedding(entries))
            .map_err(|e| e.into())
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

    m.add_class::<PersiaCommonContext>()?;

    forward::init_module(m, py)?;
    backward::init_module(m, py)?;
    data::init_module(m, py)?;
    utils::init_module(m, py)?;
    optim::init_module(m, py)?;
    nats::init_module(m, py)?;

    m.add_function(wrap_pyfunction!(is_cuda_feature_available, m)?)?;

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

    start_deadlock_detection_thread();

    Ok(())
}
