use crate::data::PyPersiaBatchData;
use crate::optim::PyOptimizerBase;
use crate::utils::PyPersiaBatchDataSender;
use crate::{PersiaCommonContext, PersiaError};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use persia_libs::{async_lock::RwLock, once_cell::sync::OnceCell, tracing};

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use persia_common::{EmbeddingTensor, SparseBatchRemoteReference};
use persia_embedding_config::PersiaReplicaInfo;
use persia_embedding_config::{
    BoundedUniformInitialization, InitializationMethod, PersiaSparseModelHyperparameters,
};
use persia_embedding_server::middleware_service::MiddlewareNatsServicePublisher;

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
    pub struct Service {
        pub master_addr: Arc<RwLock<Option<String>>>,
    }

    #[persia_nats_marcos::service]
    impl Service {
        pub async fn get_master_addr(&self, _placeholder: ()) -> Result<String, Error> {
            self.master_addr
                .read()
                .await
                .clone()
                .ok_or_else(|| Error::AddrNotSet)
        }
    }
}

pub struct MasterDiscoveryNatsServiceWrapper {
    publisher: master_discovery_service::ServicePublisher,
    _responder: master_discovery_service::ServiceResponder,
    master_addr: Option<String>,
}

impl MasterDiscoveryNatsServiceWrapper {
    pub async fn new(master_addr: Option<String>) -> Self {
        let service = master_discovery_service::Service {
            master_addr: Arc::new(RwLock::new(master_addr.clone())),
        };

        let instance = Self {
            publisher: master_discovery_service::ServicePublisher::new().await,
            _responder: master_discovery_service::ServiceResponder::new(service).await,
            master_addr,
        };
        instance
    }

    pub async fn get_master_addr(&self) -> Result<String, PersiaError> {
        if let Some(master_addr) = &self.master_addr {
            Ok(master_addr.clone())
        } else {
            let master_addr = self
                .publisher
                .publish_get_master_addr(&(), Some(0))
                .await??;
            Ok(master_addr)
        }
    }
}

pub mod persia_dataflow_service {
    use persia_common::PersiaBatchData;
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
    pub struct Service {
        pub output_channel: flume::Sender<PersiaBatchData>,
        pub world_size: usize,
    }

    #[persia_nats_marcos::service]
    impl Service {
        pub async fn batch(&self, batch: PersiaBatchData) -> Result<(), Error> {
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

static RESPONDER: OnceCell<Arc<persia_dataflow_service::ServiceResponder>> = OnceCell::new();

pub struct PersiaBatchFlowNatsServicePublisherWrapper {
    middleware_publish_service: MiddlewareNatsServicePublisher,
    num_middlewares: usize,
    cur_middleware_id: AtomicUsize,
    cur_batch_id: AtomicUsize,
    replica_info: Arc<PersiaReplicaInfo>,
    dataflow_publish_service: persia_dataflow_service::ServicePublisher,
    world_size: usize,
}

impl PersiaBatchFlowNatsServicePublisherWrapper {
    pub async fn new(world_size: Option<usize>) -> Result<Self, PersiaError> {
        let dataflow_publish_service = persia_dataflow_service::ServicePublisher::new().await;

        let world_size = match world_size {
            Some(w) => Ok(w),
            None => {
                dataflow_publish_service
                    .publish_get_world_size(&(), None)
                    .await
            }
        }?;

        let preforward_sparse_publish_service = MiddlewareNatsServicePublisher::new().await;
        let num_middlewares: usize = preforward_sparse_publish_service
            .publish_get_replica_size(&(), None)
            .await??;

        let replica_info = PersiaReplicaInfo::get().expect("NOT in persia context");

        Ok(Self {
            dataflow_publish_service,
            middleware_publish_service: preforward_sparse_publish_service,
            num_middlewares,
            cur_middleware_id: AtomicUsize::new(0),
            world_size,
            cur_batch_id: AtomicUsize::new(0),
            replica_info,
        })
    }

    pub async fn get_middleware_addr_list(&self) -> Result<Vec<String>, PersiaError> {
        // TODO: auto update middleware addr list to avoid the bad middleware addr
        let middleware_replica_size = self
            .middleware_publish_service
            .publish_get_replica_size(&(), None)
            .await??;

        let mut middleware_addr_list = Vec::new();
        for middleware_idx in 0..middleware_replica_size {
            let middleware_addr = self
                .middleware_publish_service
                .publish_get_address(&(), Some(middleware_idx))
                .await??;
            middleware_addr_list.push(middleware_addr.to_string())
        }
        Ok(middleware_addr_list)
    }

    pub async fn send_sparse_to_middleware(
        &self,
        batch: &mut PyPersiaBatchData,
    ) -> Result<(), PersiaError> {
        let start = std::time::Instant::now();
        match &mut batch.inner.sparse_data {
            EmbeddingTensor::SparseBatchRemoteReference(_) => {
                tracing::error!("sparse data has already sent to middleware, you are calling send_sparse_to_middleware muti times");
                return Err(PersiaError::MultipleSendError);
            }
            EmbeddingTensor::SparseBatch(sparse_batch) => {
                let replica_index = self.replica_info.replica_index;
                sparse_batch.batcher_idx = Some(replica_index);

                let cur_middleware_id = self.cur_middleware_id.fetch_add(1, Ordering::AcqRel);
                let sparse_ref: SparseBatchRemoteReference = self
                    .middleware_publish_service
                    .publish_forward_batched(
                        sparse_batch,
                        Some(cur_middleware_id % self.num_middlewares),
                    )
                    .await??;

                batch.inner.sparse_data = EmbeddingTensor::SparseBatchRemoteReference(sparse_ref);
                let local_batch_id = self.cur_batch_id.fetch_add(1, Ordering::AcqRel);
                let batch_id = local_batch_id * self.replica_info.replica_size
                    + self.replica_info.replica_index;
                batch.inner.batch_id = Some(batch_id);
                tracing::debug!(
                    "send_sparse_to_middleware time cost {} ms",
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
            tracing::warn!("batch id is null, please call send_sparse_to_middleware first");
            return Err(PersiaError::NullBatchIdError);
        }
        let rank_id = batch.inner.batch_id.unwrap() % self.world_size;
        self.dataflow_publish_service
            .publish_batch(&batch.inner, Some(rank_id))
            .await??;

        tracing::debug!(
            "send_dense_to_trainer {} time cost {} ms",
            rank_id,
            start.elapsed().as_millis()
        );
        Ok(())
    }

    pub async fn configure_embedding_servers(
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

        self.middleware_publish_service
            .publish_configure_embedding_servers(&config, None)
            .await??;

        Ok(())
    }

    pub async fn register_optimizer(&self, opt: &PyOptimizerBase) -> Result<(), PersiaError> {
        let optimizer = opt.get_inner();
        if optimizer.is_none() {
            return Err(PersiaError::NullOptimizerError);
        }
        let optimizer = optimizer.unwrap();

        self.middleware_publish_service
            .publish_register_optimizer(&optimizer, None)
            .await??;

        Ok(())
    }

    pub async fn wait_servers_ready(&self) -> Result<String, PersiaError> {
        let addr = self
            .middleware_publish_service
            .publish_get_address(&(), None)
            .await??;
        Ok(addr)
    }
}

#[pyfunction]
pub fn init_responder(world_size: usize, channel: &PyPersiaBatchDataSender) -> PyResult<()> {
    let common_context = PersiaCommonContext::get();
    RESPONDER.get_or_init(|| {
        let nats_service = persia_dataflow_service::Service {
            output_channel: channel.inner.clone(),
            world_size,
        };
        Arc::new(
            common_context
                .async_runtime
                .block_on(persia_dataflow_service::ServiceResponder::new(nats_service)),
        )
    });

    Ok(())
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "nats")?;
    module.add_function(wrap_pyfunction!(init_responder, module)?)?;
    super_module.add_submodule(module)?;
    Ok(())
}
