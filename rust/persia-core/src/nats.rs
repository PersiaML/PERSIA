use crate::data::PyPersiaBatchData;
use crate::optim::PyOptimizerBase;
use crate::utils::{PyPersiaBatchDataSender, PyPersiaReplicaInfo};
use crate::PersiaRpcClient;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use once_cell::sync::OnceCell;
use pyo3::exceptions::PyRuntimeError;
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
use persia_futures::{flume, smol::block_on, tokio};
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

static RESPONDER: OnceCell<(Arc<PersiaBatchFlowNatsStubResponder>, Arc<Runtime>)> =
    once_cell::sync::OnceCell::new();

#[pyclass]
pub struct PyPersiaBatchFlowNatsStubPublisher {
    to_middleware: MiddlewareNatsStubPublisher,
    num_middlewares: usize,
    cur_middleware_id: AtomicUsize,
    runtime: Runtime,
    cur_batch_id: AtomicUsize,
    replica_info: PersiaReplicaInfo,
    to_trainer: PersiaBatchFlowNatsStubPublisher,
    world_size: usize,
}

#[pymethods]
impl PyPersiaBatchFlowNatsStubPublisher {
    #[new]
    fn new(replica_info: &PyPersiaReplicaInfo, world_size: Option<usize>) -> Self {
        let replica_info = replica_info.get_replica_info();
        let nats_client = NatsClient::new(replica_info.clone());

        let to_trainer = PersiaBatchFlowNatsStubPublisher {
            nats_client: nats_client.clone(),
        };

        let world_size = if world_size.is_none() {
            let world_size: Result<usize, _> = retry(Fixed::from_millis(5000), || {
                let resp = block_on(to_trainer.publish_get_world_size(&(), None));
                if resp.is_err() {
                    tracing::warn!("failed to get world size of trainer, due to {:?}", resp);
                }
                resp
            });
            world_size.expect("failed to get world_size of trainer")
        } else {
            world_size.unwrap()
        };

        let to_middleware = MiddlewareNatsStubPublisher { nats_client };
        let num_middlewares = retry(Fixed::from_millis(5000), || {
            let resp: Result<usize, _> =
                block_on(to_middleware.publish_get_replica_size(&(), None))?;
            if resp.is_err() {
                tracing::warn!(
                    "failed to get world replica of middleware, due to {:?}",
                    resp
                );
            }
            resp
        });
        let num_middlewares = num_middlewares.expect("failed to get replica size of middleware");

        Self {
            to_trainer,
            to_middleware,
            num_middlewares,
            cur_middleware_id: AtomicUsize::new(0),
            runtime: persia_futures::tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(5)
                .build()
                .unwrap(),
            world_size,
            cur_batch_id: AtomicUsize::new(0),
            replica_info,
        }
    }

    fn send_sparse_to_middleware(
        &self,
        batch: &mut PyPersiaBatchData,
        block: bool,
    ) -> PyResult<()> {
        let start = std::time::Instant::now();
        match &mut batch.inner.sparse_data {
            EmbeddingTensor::PreForwardStub(_) => {
                tracing::error!("sparse data has already sent to middleware, you are calling sparse_to_middleware muti times");
                return Err(PyRuntimeError::new_err(String::from(
                    "send batch to middleware muti times",
                )));
            }
            EmbeddingTensor::SparseBatch(sparse_batch) => {
                let replica_index = self.replica_info.replica_index;
                sparse_batch.batcher_idx = Some(replica_index);
                let op = || {
                    let cur_middleware_id = self.cur_middleware_id.fetch_add(1, Ordering::AcqRel);
                    let _gurad = self.runtime.enter();
                    let result: Result<PreForwardStub, ShardedMiddlewareError> =
                        block_on(self.to_middleware.publish_forward_batched(
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
                let resp: Result<PreForwardStub, _> = if block {
                    retry(Fixed::from_millis(1000), op)
                } else {
                    retry(Fixed::from_millis(1000).take(1), op)
                };
                match resp {
                    Ok(result) => {
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
                    Err(e) => {
                        let err_msg = format!("{:?}", e);
                        return Err(PyRuntimeError::new_err(err_msg));
                    }
                }
            }
            EmbeddingTensor::Null => {
                tracing::warn!("sparse data is null, please call batch.add_sparse first");
                return Err(PyRuntimeError::new_err(String::from(
                    "send_sparse_to_middleware before add_sparse",
                )));
            }
        }
    }

    fn send_dense_to_trainer(&self, batch: &PyPersiaBatchData, block: bool) -> PyResult<()> {
        let start = std::time::Instant::now();
        if batch.inner.batch_id.is_none() {
            tracing::warn!("batch id is null, please call send_sparse_to_middleware first");
            return Err(PyRuntimeError::new_err(String::from(
                "send_dense_to_trainer before send_sparse_to_middleware",
            )));
        }
        let rank_id = batch.inner.batch_id.unwrap() % self.world_size;
        let op = || {
            let _gurad = self.runtime.enter();
            let result: Result<bool, _> =
                block_on(self.to_trainer.publish_batch(&batch.inner, Some(rank_id)));
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

        if result.is_err() {
            let err_msg = String::from("failed to send dense to trainer");
            return Err(PyRuntimeError::new_err(err_msg));
        }
        tracing::debug!(
            "send_dense_to_trainer {} time cost {} ms",
            rank_id,
            start.elapsed().as_millis()
        );
        Ok(())
    }

    fn configure_sharded_servers(
        &self,
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

        let _gurad = self.runtime.enter();
        let result = block_on(
            self.to_middleware
                .publish_configure_sharded_servers(&config, None),
        );
        if result.is_err() {
            let err_msg = format!(
                "failed config shard servers due to nats error {:?}",
                result.unwrap_err()
            );
            return Err(PyRuntimeError::new_err(err_msg));
        }
        let result = result.unwrap();
        if result.is_err() {
            let err_msg = format!(
                "failed config shard servers due to persia server side error {:?}",
                result.unwrap_err()
            );
            return Err(PyRuntimeError::new_err(err_msg));
        }
        Ok(())
    }

    fn register_optimizer(&self, opt: &PyOptimizerBase) -> PyResult<()> {
        let optimizer = opt.get_inner();
        if optimizer.is_none() {
            let err_msg = String::from("sparse optimizer not set yet");
            return Err(PyRuntimeError::new_err(err_msg));
        }
        let optimizer = optimizer.unwrap();
        let _gurad = self.runtime.enter();
        let result = block_on(
            self.to_middleware
                .publish_register_optimizer(&optimizer, None),
        );
        if result.is_err() {
            let err_msg = format!(
                "failed register optimizer due to nats error {:?}",
                result.unwrap_err()
            );
            return Err(PyRuntimeError::new_err(err_msg));
        }
        let result = result.unwrap();
        if result.is_err() {
            let err_msg = format!(
                "failed register optimizer due to persia server side error {:?}",
                result.unwrap_err()
            );
            return Err(PyRuntimeError::new_err(err_msg));
        }

        Ok(())
    }

    fn wait_servers_ready(&self) -> PyResult<()> {
        let addr: Result<String, _> = retry(Fixed::from_millis(5000), || {
            let resp = block_on(self.to_middleware.publish_get_address(&(), None))?;
            if resp.is_err() {
                tracing::warn!("waiting for servers ready...")
            }
            resp
        });
        let rpc_client = PersiaRpcClient::get_instance();
        let _ = rpc_client.get_client_by_addr(addr.unwrap().as_str());
        Ok(())
    }
}

#[pyfunction]
pub fn init_responder(
    replica_info: &PyPersiaReplicaInfo,
    channel: &PyPersiaBatchDataSender,
) -> PyResult<()> {
    let replica_info = replica_info.get_replica_info();

    RESPONDER.get_or_init(|| {
        let nats_stub = PersiaBatchFlowNatsStub {
            output_channel: channel.inner.clone(),
            world_size: replica_info.replica_size,
        };
        let nats_responder = Arc::new(PersiaBatchFlowNatsStubResponder {
            inner: nats_stub,
            nats_client: NatsClient::new(replica_info.clone()),
        });

        let runtime = Arc::new(
            persia_futures::tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(4)
                .build()
                .unwrap(),
        );

        let _guard = runtime.enter();
        nats_responder
            .spawn_subscriptions()
            .expect("failed to spawn nats subscriptions");

        (nats_responder, runtime)
    });

    Ok(())
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "nats")?;
    module.add_class::<PyPersiaBatchFlowNatsStubPublisher>()?;
    module.add_function(wrap_pyfunction!(init_responder, module)?)?;
    super_module.add_submodule(module)?;
    Ok(())
}
