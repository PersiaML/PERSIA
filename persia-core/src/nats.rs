use crate::data::PyPersiaBatchData;
use crate::optim::PyOptimizerBase;
use crate::utils::{PyPersiaBatchDataChannel, PyPersiaReplicaInfo};
use crate::PersiaRpcClient;

use persia_embedding_config::PersiaReplicaInfo;
use persia_embedding_config::{
    BoundedUniformInitialization, InitializationMethod, PersiaSparseModelHyperparameters,
};
use persia_embedding_datatypes::{EmbeddingTensor, PersiaBatchData};
use persia_embedding_sharded_server::sharded_middleware_service::{
    MiddlewareNatsStubPublisher, ShardedMiddlewareError,
};
use persia_futures::tokio::runtime::Runtime;
use persia_futures::{flume, smol::block_on, tokio};
use persia_nats_client::{NatsClient, NatsError};
use persia_speedy::Writable;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use retry::{delay::Fixed, retry};
#[derive(Clone)]
pub struct PersiaBatchFlowNatsStub {
    pub nats_channel_s: flume::Sender<PersiaBatchData>,
    pub world_size: usize,
}

#[persia_nats_marcos::stub]
impl PersiaBatchFlowNatsStub {
    pub async fn batch(&self, batch: PersiaBatchData) -> bool {
        let result = self.nats_channel_s.try_send(batch);
        result.is_ok()
    }

    pub async fn get_world_size(&self, _placeholder: ()) -> usize {
        self.world_size
    }
}

#[pyclass]
pub struct PyPersiaBatchFlowNatsStubPublisher {
    to_trainer: PersiaBatchFlowNatsStubPublisher,
    to_middleware: MiddlewareNatsStubPublisher,
    runtime: Runtime,
    world_size: usize,
    cur_batch_id: AtomicUsize,
    replica_info: PersiaReplicaInfo,
}

#[pymethods]
impl PyPersiaBatchFlowNatsStubPublisher {
    #[new]
    fn new(replica_info: &PyPersiaReplicaInfo) -> Self {
        let replica_info = replica_info.get_replica_info();
        let nats_client = NatsClient::new(replica_info.clone());
        let to_trainer = PersiaBatchFlowNatsStubPublisher {
            nats_client: nats_client.clone(),
        };
        let world_size: Result<usize, _> = retry(Fixed::from_millis(5000), || {
            let resp = block_on(to_trainer.publish_get_world_size(&(), None));
            if resp.is_err() {
                tracing::warn!("failed to get world size of trainer, due to {:?}", resp);
            }
            resp
        });
        let world_size = world_size.expect("failed to get world_size of trainer");

        Self {
            to_trainer,
            to_middleware: MiddlewareNatsStubPublisher { nats_client },
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
        let sparse_data = &batch.inner.sparse_data;
        match sparse_data {
            EmbeddingTensor::ID(_) => {
                tracing::error!("sparse data has already sent to middleware, you are calling sparse_to_middleware muti times");
                return Err(PyRuntimeError::new_err(String::from(
                    "send batch to middleware muti times",
                )));
            }
            EmbeddingTensor::SparseBatch(sparse_batch) => {
                let op = || {
                    let _gurad = self.runtime.enter();
                    let result: Result<(String, u64), ShardedMiddlewareError> = block_on(
                        self.to_middleware
                            .publish_forward_batched(sparse_batch, None),
                    )?;
                    result
                };
                let resp: Result<(String, u64), _> = if block {
                    retry(Fixed::from_millis(1000), op)
                } else {
                    retry(Fixed::from_millis(1000).take(1), op)
                };
                match resp {
                    Ok(result) => {
                        batch.inner.sparse_data = EmbeddingTensor::ID(result);
                        let local_batch_id = self.cur_batch_id.fetch_add(1, Ordering::AcqRel);
                        let batch_id = local_batch_id * self.replica_info.replica_size
                            + self.replica_info.replica_index;
                        batch.inner.batch_id = Some(batch_id);
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

#[pyclass]
pub struct PyPersiaBatchFlowNatsStubResponder {
    _inner: PersiaBatchFlowNatsStubResponder,
    _runtime: Arc<Runtime>,
}

#[pymethods]
impl PyPersiaBatchFlowNatsStubResponder {
    #[new]
    fn new(repilca_info: &PyPersiaReplicaInfo, forward_input: &PyPersiaBatchDataChannel) -> Self {
        let replica_info = repilca_info.get_replica_info();
        let nats_stub = PersiaBatchFlowNatsStub {
            nats_channel_s: forward_input.sender.clone(),
            world_size: replica_info.replica_size,
        };
        let nats_responder = PersiaBatchFlowNatsStubResponder {
            inner: nats_stub,
            nats_client: NatsClient::new(replica_info.clone()),
        };

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

        Self {
            _inner: nats_responder,
            _runtime: runtime,
        }
    }
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "nats")?;
    module.add_class::<PyPersiaBatchFlowNatsStubResponder>()?;
    module.add_class::<PyPersiaBatchFlowNatsStubPublisher>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
