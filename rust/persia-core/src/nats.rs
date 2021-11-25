use crate::data::{EmbeddingTensor, PersiaBatch};
use crate::optim::OptimizerBase;
use crate::utils::PersiaBatchDataSender;
use crate::{PersiaCommonContextImpl, PersiaError};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use persia_libs::{async_lock::RwLock, backoff, once_cell::sync::OnceCell, tracing};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use persia_common::IDTypeFeatureRemoteRef;
use persia_embedding_config::PersiaReplicaInfo;
use persia_embedding_config::{
    BoundedUniformInitialization, InitializationMethod, PersiaEmbeddingModelHyperparameters,
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
            let mut backoff = backoff::ExponentialBackoff::default();
            backoff.max_interval = std::time::Duration::from_millis(500);

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
    use crate::data::PersiaBatchImpl;
    use persia_embedding_config::PersiaReplicaInfo;
    use persia_libs::{flume, thiserror, tokio, tracing};

    use persia_nats_client::{NatsClient, NatsError};
    use persia_speedy::{Readable, Writable};

    #[derive(thiserror::Error, Debug, Readable, Writable)]
    pub enum Error {
        #[error("nn worker buffer full error")]
        NNWorkerBufferFullError,
    }

    #[derive(Clone)]
    pub struct DataflowService {
        pub output_channel: flume::Sender<PersiaBatchImpl>,
        pub world_size: usize,
    }

    #[persia_nats_marcos::service]
    impl DataflowService {
        pub async fn batch(&self, batch: PersiaBatchImpl) -> Result<(), Error> {
            tracing::debug!("nats get batch");
            let result = self
                .output_channel
                .send_timeout(batch, std::time::Duration::from_millis(500));
            if !result.is_ok() {
                Err(Error::NNWorkerBufferFullError)
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
                let mut backoff = backoff::ExponentialBackoff::default();
                backoff.max_interval = std::time::Duration::from_millis(500);

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

        let embedding_worker_publish_service = EmbeddingWorkerNatsServicePublisher::new().await;

        let mut backoff = backoff::ExponentialBackoff::default();
        backoff.max_interval = std::time::Duration::from_millis(500);

        let num_embedding_workers = backoff::future::retry(backoff, || async {
            let result: Result<usize, PersiaError> = embedding_worker_publish_service
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
            embedding_worker_publish_service,
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
            let mut backoff = backoff::ExponentialBackoff::default();
            backoff.max_interval = std::time::Duration::from_millis(500);

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

    pub async fn send_id_type_features_to_embedding_worker(
        &self,
        batch: &mut PersiaBatch,
    ) -> Result<(), PersiaError> {
        let start = std::time::Instant::now();
        match &mut batch.inner.id_type_features {
            EmbeddingTensor::IDTypeFeatureRemoteRef(_) => {
                tracing::error!("id type features has already sent to embedding worker, you are calling send_id_type_features_to_embedding_worker multiple times");
                return Err(PersiaError::MultipleSendError);
            }
            EmbeddingTensor::IDTypeFeature(id_type_features) => {
                let replica_index = self.replica_info.replica_index;
                id_type_features.batcher_idx = Some(replica_index);

                let cur_embedding_worker_id =
                    self.cur_embedding_worker_id.fetch_add(1, Ordering::AcqRel);

                let mut backoff = backoff::ExponentialBackoff::default();
                backoff.max_interval = std::time::Duration::from_millis(500);

                let id_type_feature_ref: IDTypeFeatureRemoteRef =
                    backoff::future::retry(backoff, || async {
                        let id_type_features_ref = self
                            .embedding_worker_publish_service
                            .publish_forward_batched(
                                id_type_features,
                                Some(cur_embedding_worker_id % self.num_embedding_workers),
                            )
                            .await
                            .map_err(|e| PersiaError::from(e))?
                            .map_err(|e| PersiaError::from(e));

                        if id_type_features_ref.is_err() {
                            tracing::warn!(
                                "failed to send id type features to embedding worker due to {:?}, retrying...",
                                id_type_features_ref
                            );
                        }

                        Ok(id_type_features_ref?)
                    })
                    .await?;

                batch.inner.id_type_features =
                    EmbeddingTensor::IDTypeFeatureRemoteRef(id_type_feature_ref);
                let local_batch_id = self.cur_batch_id.fetch_add(1, Ordering::AcqRel);
                let batch_id = local_batch_id * self.replica_info.replica_size
                    + self.replica_info.replica_index;
                batch.inner.batch_id = Some(batch_id);
                tracing::debug!(
                    "send_id_type_features_to_embedding_worker time cost {} ms",
                    start.elapsed().as_millis()
                );
                return Ok(());
            }
            EmbeddingTensor::Null => {
                tracing::warn!(
                    "id type features is null, please call batch.add_id_type_features first"
                );
                return Err(PersiaError::NullIDTypeFeaturesError);
            }
        }
    }

    pub async fn send_non_id_type_features_to_nn_worker(
        &self,
        batch: &PersiaBatch,
    ) -> Result<(), PersiaError> {
        let start = std::time::Instant::now();
        if batch.inner.batch_id.is_none() {
            tracing::warn!(
                "batch id is null, please call send_id_type_features_to_embedding_worker first"
            );
            return Err(PersiaError::NullBatchIdError);
        }
        let rank_id = batch.inner.batch_id.unwrap() % self.world_size;

        let mut backoff = backoff::ExponentialBackoff::default();
        backoff.max_interval = std::time::Duration::from_millis(500);

        backoff::future::retry(backoff, || async {
            let resp = self
                .dataflow_publish_service
                .publish_batch(&batch.inner, Some(rank_id))
                .await
                .map_err(|e| PersiaError::from(e))?
                .map_err(|e| PersiaError::from(e));
            if resp.is_err() {
                tracing::warn!(
                    "failed to send non_id_type_features to nn worker due to {:?}, retrying...",
                    resp
                );
            }
            Ok(resp?)
        })
        .await?;

        tracing::debug!(
            "send_non_id_type_features_to_nn_worker {} time cost {} ms",
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
        let config = PersiaEmbeddingModelHyperparameters {
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

    pub async fn register_optimizer(&self, opt: &OptimizerBase) -> Result<(), PersiaError> {
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
pub fn init_responder(world_size: usize, channel: &PersiaBatchDataSender) -> PyResult<()> {
    let common_context = PersiaCommonContextImpl::get();
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
