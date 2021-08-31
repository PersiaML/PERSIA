use crate::data::PyPersiaBatchData;
use crate::optim::PyOptimizerBase;
use crate::utils::PyPersiaBatchDataSender;
use crate::{PersiaCommonContext, PersiaError};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use retry::{delay::Fixed, retry};

use persia_embedding_config::PersiaReplicaInfo;
use persia_embedding_config::{
    BoundedUniformInitialization, InitializationMethod, PersiaSparseModelHyperparameters,
};
use persia_embedding_datatypes::{EmbeddingTensor, PersiaBatchData, PreForwardStub};
use persia_embedding_sharded_server::sharded_middleware_service::{
    MiddlewareNatsStubPublisher, ShardedMiddlewareError,
};
use persia_futures::tokio::runtime::Runtime;
use persia_futures::{flume, tokio};
use persia_nats_client::{NatsClient, NatsError};
use persia_speedy::Writable;

#[derive(Clone)]
pub struct PersiaBatchFlowNatsStub {
    pub output_channel: flume::Sender<PersiaBatchData>,
    pub world_size: usize,
}

#[persia_nats_marcos::stub]
impl PersiaBatchFlowNatsStub {
    pub async fn batch(&self, batch: PersiaBatchData) -> bool {
        let result = self.output_channel.try_send(batch);
        result.is_ok()
    }

    pub async fn get_world_size(&self, _placeholder: ()) -> usize {
        self.world_size
    }
}

static RESPONDER: OnceCell<Arc<PersiaBatchFlowNatsStubResponder>> =
    once_cell::sync::OnceCell::new();

pub struct PersiaBatchFlowNatsStubPublisherWrapper {
    to_middleware: MiddlewareNatsStubPublisher,
    num_middlewares: usize,
    cur_middleware_id: AtomicUsize,
    cur_batch_id: AtomicUsize,
    replica_info: Arc<PersiaReplicaInfo>,
    to_trainer: PersiaBatchFlowNatsStubPublisher,
    world_size: usize,
    async_runtime: Arc<Runtime>,
}

impl PersiaBatchFlowNatsStubPublisherWrapper {
    pub fn new(world_size: Option<usize>, async_runtime: Arc<Runtime>) -> Self {
        let to_trainer = PersiaBatchFlowNatsStubPublisher::new();
        let world_size = world_size.unwrap_or_else(|| {
            retry(Fixed::from_millis(5000), || {
                let resp = async_runtime.block_on(to_trainer.publish_get_world_size(&(), None));
                if resp.is_err() {
                    tracing::warn!("failed to get world size of trainer, due to {:?}", resp);
                }
                resp
            })
            .expect("failed to get world_size of trainer")
        });

        let to_middleware = MiddlewareNatsStubPublisher::new();
        let num_middlewares = retry(Fixed::from_millis(5000), || {
            let resp: Result<usize, _> =
                async_runtime.block_on(to_middleware.publish_get_replica_size(&(), None))?;
            if resp.is_err() {
                tracing::warn!(
                    "failed to get world replica of middleware, due to {:?}",
                    resp
                );
            }
            resp
        });
        let num_middlewares = num_middlewares.expect("failed to get replica size of middleware");

        let replica_info = PersiaReplicaInfo::get().expect("NOT in persia context");

        Self {
            to_trainer,
            to_middleware,
            num_middlewares,
            cur_middleware_id: AtomicUsize::new(0),
            world_size,
            cur_batch_id: AtomicUsize::new(0),
            replica_info,
            async_runtime,
        }
    }

    pub fn send_sparse_to_middleware(
        &self,
        batch: &mut PyPersiaBatchData,
        block: bool,
    ) -> Result<(), PersiaError> {
        let start = std::time::Instant::now();
        match &mut batch.inner.sparse_data {
            EmbeddingTensor::PreForwardStub(_) => {
                tracing::error!("sparse data has already sent to middleware, you are calling sparse_to_middleware muti times");
                return Err(PersiaError::MultipleSendError);
            }
            EmbeddingTensor::SparseBatch(sparse_batch) => {
                let replica_index = self.replica_info.replica_index;
                sparse_batch.batcher_idx = Some(replica_index);
                let op = || {
                    let cur_middleware_id = self.cur_middleware_id.fetch_add(1, Ordering::AcqRel);
                    let _gurad = self.async_runtime.enter();
                    let result: Result<PreForwardStub, ShardedMiddlewareError> = self
                        .async_runtime
                        .block_on(self.to_middleware.publish_forward_batched(
                            sparse_batch,
                            Some(cur_middleware_id % self.num_middlewares),
                        ))?;
                    if result.is_err() {
                        tracing::warn!(
                            "fail to send sparse data to middleware due to {:?}",
                            result
                        );
                    }
                    result
                };
                let resp = if block {
                    retry(Fixed::from_millis(1000), op)
                } else {
                    retry(Fixed::from_millis(1000).take(1), op)
                };

                let result = resp.map_err(|_| PersiaError::SendDataError)?;

                batch.inner.sparse_data = EmbeddingTensor::PreForwardStub(result);
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

    pub fn send_dense_to_trainer(
        &self,
        batch: &PyPersiaBatchData,
        block: bool,
    ) -> Result<(), PersiaError> {
        let start = std::time::Instant::now();
        if batch.inner.batch_id.is_none() {
            tracing::warn!("batch id is null, please call send_sparse_to_middleware first");
            return Err(PersiaError::NullBatchIdError);
        }
        let rank_id = batch.inner.batch_id.unwrap() % self.world_size;
        let op = || {
            let _gurad = self.async_runtime.enter();
            let result: Result<bool, _> = self
                .async_runtime
                .block_on(self.to_trainer.publish_batch(&batch.inner, Some(rank_id)));
            if result.is_ok() && result.unwrap() {
                Ok(())
            } else {
                tracing::warn!("failed to send dense to trainer {}, retrying...", rank_id);
                Err(())
            }
        };
        let result = if block {
            retry(Fixed::from_millis(1000), op)
        } else {
            retry(Fixed::from_millis(1000).take(1), op)
        };

        result.map_err(|_| PersiaError::SendDataError)?;

        tracing::debug!(
            "send_dense_to_trainer {} time cost {} ms",
            rank_id,
            start.elapsed().as_millis()
        );
        Ok(())
    }

    pub fn configure_sharded_servers(
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

        let _gurad = self.async_runtime.enter();
        self.async_runtime.block_on(
            self.to_middleware
                .publish_configure_sharded_servers(&config, None),
        )??;

        Ok(())
    }

    pub fn register_optimizer(&self, opt: &PyOptimizerBase) -> Result<(), PersiaError> {
        let optimizer = opt.get_inner();
        if optimizer.is_none() {
            return Err(PersiaError::NullOptimizerError);
        }
        let optimizer = optimizer.unwrap();
        let _gurad = self.async_runtime.enter();
        self.async_runtime.block_on(
            self.to_middleware
                .publish_register_optimizer(&optimizer, None),
        )??;

        Ok(())
    }

    pub fn wait_servers_ready(&self) -> Result<String, PersiaError> {
        let addr: Result<String, _> = retry(Fixed::from_millis(5000), || {
            let resp = self
                .async_runtime
                .block_on(self.to_middleware.publish_get_address(&(), None))?;
            if resp.is_err() {
                tracing::warn!("waiting for servers ready...")
            }
            resp
        });
        let addr = addr.expect("failed to wait server ready");
        Ok(addr)
    }
}

#[pyfunction]
pub fn init_responder(world_size: usize, channel: &PyPersiaBatchDataSender) -> PyResult<()> {
    let common_context = PersiaCommonContext::get();
    RESPONDER.get_or_init(|| {
        let nats_stub = PersiaBatchFlowNatsStub {
            output_channel: channel.inner.clone(),
            world_size,
        };
        let _guard = common_context.async_runtime.enter();
        Arc::new(PersiaBatchFlowNatsStubResponder::new(nats_stub))
    });

    Ok(())
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "nats")?;
    module.add_function(wrap_pyfunction!(init_responder, module)?)?;
    super_module.add_submodule(module)?;
    Ok(())
}
