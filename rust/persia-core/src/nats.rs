use crate::data::{EmbeddingTensor, PyPersiaBatchData};
use crate::optim::PyOptimizerBase;
use crate::utils::PyPersiaBatchDataSender;
use crate::{PersiaCommonContext, PersiaError};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use persia_libs::{async_lock::RwLock, backoff, once_cell::sync::OnceCell, tracing};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use persia_common::SparseBatchRemoteReference;
use persia_embedding_config::PersiaReplicaInfo;
use persia_embedding_config::{
    BoundedUniformInitialization, InitializationMethod, PersiaSparseModelHyperparameters,
};
use persia_embedding_server::embedding_worker_service::EmbeddingWorkerNatsServicePublisher;
use persia_nats_client::NatsError;

pub mod master_discovery_service {
    use persia_embedding_config::PersiaReplicaInfo;
    use persia_libs::{async_lock::RwLock, thiserror, tokio, tracing};
    use persia_nats_client::{NatsClient, NatsError};
    use persia_speedy::{Readable, Writable};
    use std::sync::Arc;

    #[derive(thiserror::Error, Debug, Readable, Writable)]
    pub enum Error {
        #[error("addr not set error")]
        AddrNotSet,
    }

    #[derive(Clone)]
    pub struct MasterDiscoveryService {
        pub master_addr: Arc<RwLock<Option<String>>>,
    }

    #[persia_nats_marcos::service]
    impl MasterDiscoveryService {
        pub async fn get_master_addr(&self, _placeholder: ()) -> Result<String, Error> {
            self.master_addr
                .read()
                .await
                .clone()
                .ok_or_else(|| Error::AddrNotSet)
        }
    }
}

pub struct MasterDiscoveryComponent {
    publisher: master_discovery_service::MasterDiscoveryServicePublisher,
    _responder: master_discovery_service::MasterDiscoveryServiceResponder,
    master_addr: Option<String>,
}

impl MasterDiscoveryComponent {
    pub async fn new(master_addr: Option<String>) -> Self {
        let service = master_discovery_service::MasterDiscoveryService {
            master_addr: Arc::new(RwLock::new(master_addr.clone())),
        };

        let instance = Self {
            publisher: master_discovery_service::MasterDiscoveryServicePublisher::new().await,
            _responder: master_discovery_service::MasterDiscoveryServiceResponder::new(service)
                .await,
            master_addr,
        };
        instance
    }

    pub async fn get_master_addr(&self) -> Result<String, PersiaError> {
        if let Some(master_addr) = &self.master_addr {
            Ok(master_addr.clone())
        } else {
            let backoff = backoff::ExponentialBackoff::default();

            let master_addr = backoff::future::retry(backoff, || async {
                let master_addr = self
                    .publisher
                    .publish_get_master_addr(&(), Some(0))
                    .await
                    .map_err(|e| PersiaError::from(e))?
                    .map_err(|e| PersiaError::from(e));
                if master_addr.is_err() {
                    tracing::warn!(
                        "failed to get master addr due to {:?}, retrying...",
                        master_addr
                    );
                }
                Ok(master_addr?)
            })
            .await?;

            Ok(master_addr)
        }
    }
}

pub mod persia_dataflow_service {
    use crate::data::PersiaBatchData;
    use persia_embedding_config::PersiaReplicaInfo;
    use persia_libs::{flume, thiserror, tokio, tracing};

    use persia_nats_client::{NatsClient, NatsError};
    use persia_speedy::{Readable, Writable};

    #[derive(thiserror::Error, Debug, Readable, Writable)]
    pub enum Error {
        #[error("trainer buffer full error")]
        TrainerBufferFullError,
    }

    #[derive(Clone)]
    pub struct DataflowService {
        pub output_channel: flume::Sender<PersiaBatchData>,
        pub world_size: usize,
    }

    #[persia_nats_marcos::service]
    impl DataflowService {
        pub async fn batch(&self, batch: PersiaBatchData) -> Result<(), Error> {
            tracing::debug!("nats get batch");
            let result = self
                .output_channel
                .send_timeout(batch, std::time::Duration::from_millis(500));
            if !result.is_ok() {
                Err(Error::TrainerBufferFullError)
            } else {
                Ok(())
            }
        }

        pub async fn get_world_size(&self, _placeholder: ()) -> usize {
            self.world_size
        }
    }
}

static RESPONDER: OnceCell<Arc<persia_dataflow_service::DataflowServiceResponder>> =
    OnceCell::new();

pub struct PersiaDataFlowComponent {
    embedding_worker_publish_service: EmbeddingWorkerNatsServicePublisher,
    num_embedding_workers: usize,
    cur_embedding_worker_id: AtomicUsize,
    cur_batch_id: AtomicUsize,
    replica_info: Arc<PersiaReplicaInfo>,
    dataflow_publish_service: persia_dataflow_service::DataflowServicePublisher,
    world_size: usize,
}

impl PersiaDataFlowComponent {
    pub async fn new_initialized(world_size: Option<usize>) -> Result<Self, PersiaError> {
        let dataflow_publish_service =
            persia_dataflow_service::DataflowServicePublisher::new().await;

        let world_size = match world_size {
            Some(w) => Ok(w),
            None => {
                let backoff = backoff::ExponentialBackoff::default();
                backoff::future::retry(backoff, || async {
                    let result: Result<usize, NatsError> = dataflow_publish_service
                        .publish_get_world_size(&(), None)
                        .await;
                    if result.is_err() {
                        tracing::warn!(
                            "failed to get world size via nats due to {:?}, retrying...",
                            result
                        );
                    }
                    Ok(result?)
                })
                .await
            }
        }?;

        tracing::info!("Get world_size {}", world_size);

        let preforward_sparse_publish_service = EmbeddingWorkerNatsServicePublisher::new().await;

        let backoff = backoff::ExponentialBackoff::default();
        let num_embedding_workers = backoff::future::retry(backoff, || async {
            let result: Result<usize, PersiaError> = preforward_sparse_publish_service
                .publish_get_replica_size(&(), None)
                .await
                .map_err(|e| PersiaError::from(e))?
                .map_err(|e| PersiaError::from(e));
            if result.is_err() {
                tracing::warn!(
                    "failed to get embedding worker replica size via nats due to {:?}, retrying...",
                    result
                );
            }
            Ok(result?)
        })
        .await?;

        let replica_info = PersiaReplicaInfo::get().expect("NOT in persia context");

        Ok(Self {
            dataflow_publish_service,
            embedding_worker_publish_service: preforward_sparse_publish_service,
            num_embedding_workers,
            cur_embedding_worker_id: AtomicUsize::new(0),
            world_size,
            cur_batch_id: AtomicUsize::new(0),
            replica_info,
        })
    }

    pub async fn get_embedding_worker_addr_list(&self) -> Result<Vec<String>, PersiaError> {
        // TODO: auto update embedding worker addr list to avoid the bad embedding worker addr
        let embedding_worker_replica_size = self.num_embedding_workers;

        let mut embedding_worker_addr_list = Vec::new();
        for embedding_worker_idx in 0..embedding_worker_replica_size {
            let backoff = backoff::ExponentialBackoff::default();
            let embedding_worker_addr = backoff::future::retry(backoff, || async {
                let embedding_worker_addr = self
                    .embedding_worker_publish_service
                    .publish_get_address(&(), Some(embedding_worker_idx))
                    .await
                    .map_err(|e| PersiaError::from(e))?
                    .map_err(|e| PersiaError::from(e));
                if embedding_worker_addr.is_err() {
                    tracing::warn!(
                        "failed to get addr for embedding worker {} due to {:?}",
                        embedding_worker_idx,
                        embedding_worker_addr
                    );
                }
                Ok(embedding_worker_addr?)
            })
            .await?;

            embedding_worker_addr_list.push(embedding_worker_addr.to_string())
        }
        Ok(embedding_worker_addr_list)
    }

    pub async fn send_sparse_to_embedding_worker(
        &self,
        batch: &mut PyPersiaBatchData,
    ) -> Result<(), PersiaError> {
        let start = std::time::Instant::now();
        match &mut batch.inner.sparse_data {
            EmbeddingTensor::SparseBatchRemoteReference(_) => {
                tracing::error!("sparse data has already sent to embedding worker, you are calling send_sparse_to_embedding_worker muti times");
                return Err(PersiaError::MultipleSendError);
            }
            EmbeddingTensor::SparseBatch(sparse_batch) => {
                let replica_index = self.replica_info.replica_index;
                sparse_batch.batcher_idx = Some(replica_index);

                let cur_embedding_worker_id =
                    self.cur_embedding_worker_id.fetch_add(1, Ordering::AcqRel);

                let backoff = backoff::ExponentialBackoff::default();
                let sparse_ref: SparseBatchRemoteReference =
                    backoff::future::retry(backoff, || async {
                        let sparse_ref = self
                            .embedding_worker_publish_service
                            .publish_forward_batched(
                                sparse_batch,
                                Some(cur_embedding_worker_id % self.num_embedding_workers),
                            )
                            .await
                            .map_err(|e| PersiaError::from(e))?
                            .map_err(|e| PersiaError::from(e));
                        if sparse_ref.is_err() {
                            tracing::warn!(
                                "failed to send ids to embedding worker due to {:?}, retrying...",
                                sparse_ref
                            );
                        }
                        Ok(sparse_ref?)
                    })
                    .await?;

                batch.inner.sparse_data = EmbeddingTensor::SparseBatchRemoteReference(sparse_ref);
                let local_batch_id = self.cur_batch_id.fetch_add(1, Ordering::AcqRel);
                let batch_id = local_batch_id * self.replica_info.replica_size
                    + self.replica_info.replica_index;
                batch.inner.batch_id = Some(batch_id);
                tracing::debug!(
                    "send_sparse_to_embedding_worker time cost {} ms",
                    start.elapsed().as_millis()
                );
                return Ok(());
            }
            EmbeddingTensor::Null => {
                tracing::warn!("sparse data is null, please call batch.add_sparse first");
                return Err(PersiaError::NullSparseDataError);
            }
        }
    }

    pub async fn send_dense_to_trainer(
        &self,
        batch: &PyPersiaBatchData,
    ) -> Result<(), PersiaError> {
        let start = std::time::Instant::now();
        if batch.inner.batch_id.is_none() {
            tracing::warn!("batch id is null, please call send_sparse_to_embedding_worker first");
            return Err(PersiaError::NullBatchIdError);
        }
        let rank_id = batch.inner.batch_id.unwrap() % self.world_size;

        let backoff = backoff::ExponentialBackoff::default();
        backoff::future::retry(backoff, || async {
            let resp = self
                .dataflow_publish_service
                .publish_batch(&batch.inner, Some(rank_id))
                .await
                .map_err(|e| PersiaError::from(e))?
                .map_err(|e| PersiaError::from(e));
            if resp.is_err() {
                tracing::warn!(
                    "failed to send data to trainer due to {:?}, retrying...",
                    resp
                );
            }
            Ok(resp?)
        })
        .await?;

        tracing::debug!(
            "send_dense_to_trainer {} time cost {} ms",
            rank_id,
            start.elapsed().as_millis()
        );
        Ok(())
    }

    pub async fn configure_embedding_parameter_servers(
        &self,
        initialize_lower: f32,
        initialize_upper: f32,
        admit_probability: f32,
        enable_weight_bound: bool,
        weight_bound: f32,
    ) -> Result<(), PersiaError> {
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

        self.embedding_worker_publish_service
            .publish_configure_embedding_parameter_servers(&config, None)
            .await??;

        Ok(())
    }

    pub async fn register_optimizer(&self, opt: &PyOptimizerBase) -> Result<(), PersiaError> {
        let optimizer = opt.get_inner();
        if optimizer.is_none() {
            return Err(PersiaError::NullOptimizerError);
        }
        let optimizer = optimizer.unwrap();

        self.embedding_worker_publish_service
            .publish_register_optimizer(&optimizer, None)
            .await??;

        Ok(())
    }

    pub async fn wait_servers_ready(&self) -> Result<String, PersiaError> {
        let addr = self
            .embedding_worker_publish_service
            .publish_get_address(&(), None)
            .await??;
        Ok(addr)
    }
}

#[pyfunction]
pub fn init_responder(world_size: usize, channel: &PyPersiaBatchDataSender) -> PyResult<()> {
    let common_context = PersiaCommonContext::get();
    RESPONDER.get_or_init(|| {
        let nats_service = persia_dataflow_service::DataflowService {
            output_channel: channel.inner.clone(),
            world_size,
        };
        Arc::new(common_context.async_runtime.block_on(
            persia_dataflow_service::DataflowServiceResponder::new(nats_service),
        ))
    });

    Ok(())
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "nats")?;
    module.add_function(wrap_pyfunction!(init_responder, module)?)?;
    super_module.add_submodule(module)?;
    Ok(())
}
