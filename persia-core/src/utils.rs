use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::data::PyPersiaBatchData;

use persia_embedding_config::PersiaReplicaInfo;
use persia_embedding_datatypes::PersiaBatchData;
use persia_futures::{flume, tokio::runtime::Runtime};
use persia_message_queue::{PersiaMessageQueueClient, PersiaMessageQueueServer};

#[pyclass]
pub struct PyPersiaMessageQueueClient {
    pub inner: PersiaMessageQueueClient,
    pub runtime: Runtime,
}

#[pymethods]
impl PyPersiaMessageQueueClient {
    #[new]
    fn new(server_addr: &str) -> Self {
        let runtime = persia_futures::tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(5)
            .build()
            .unwrap();

        let _guard = runtime.enter();

        Self {
            inner: PersiaMessageQueueClient::new(server_addr),
            runtime,
        }
    }

    fn put(&self, data: Vec<u8>) {
        let _gurad = self.runtime.enter();
        self.runtime.block_on(self.inner.send(data)).unwrap();
    }

    fn get<'a>(&self, _py: Python<'a>) -> &'a PyBytes {
        let _gurad = self.runtime.enter();
        let bytes = self.runtime.block_on(self.inner.recv());
        PyBytes::new(_py, bytes.unwrap().as_slice())
    }
}

#[pyclass]
pub struct PyPersiaMessageQueueServer {
    inner: PersiaMessageQueueServer,
    runtime: Runtime, // thread unsafe
}

#[pymethods]
impl PyPersiaMessageQueueServer {
    #[new]
    fn new(port: u16, cap: usize) -> Self {
        let runtime = persia_futures::tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(5)
            .build()
            .unwrap();

        let _guard = runtime.enter();

        Self {
            inner: PersiaMessageQueueServer::new(port, cap),
            runtime,
        }
    }

    fn put(&self, data: Vec<u8>) {
        let _gurad = self.runtime.enter();
        self.runtime.block_on(self.inner.send(data))
    }

    fn get<'a>(&self, _py: Python<'a>) -> &'a PyBytes {
        let _gurad = self.runtime.enter();
        let bytes = self.runtime.block_on(self.inner.recv());
        PyBytes::new(_py, bytes.as_slice())
    }
}

#[pyclass]
pub struct PyPersiaReplicaInfo {
    inner: PersiaReplicaInfo,
}

impl PyPersiaReplicaInfo {
    pub fn get_replica_info(&self) -> PersiaReplicaInfo {
        self.inner.clone()
    }
}

#[pymethods]
impl PyPersiaReplicaInfo {
    #[new]
    fn new(replica_size: usize, replica_index: usize) -> Self {
        Self {
            inner: PersiaReplicaInfo {
                replica_index,
                replica_size,
            },
        }
    }

    pub fn replica_index(&self) -> usize {
        self.inner.replica_index
    }

    pub fn replica_size(&self) -> usize {
        self.inner.replica_size
    }
}

#[pyclass]
pub struct PyPersiaBatchDataSender {
    pub inner: flume::Sender<PersiaBatchData>,
}

#[pymethods]
impl PyPersiaBatchDataSender {
    pub fn send(&self, batch_data: &mut PyPersiaBatchData, py: Python) -> PyResult<()> {
        let batch_data = std::mem::take(&mut batch_data.inner);
        py.allow_threads(move || {
            self.inner.send(batch_data).unwrap();
            Ok(())
        })
    }
}

#[pyclass]
pub struct PyPersiaBatchDataReceiver {
    pub inner: flume::Receiver<PersiaBatchData>,
}
#[pyclass]
pub struct PyPersiaBatchDataChannel {
    pub sender: flume::Sender<PersiaBatchData>,
    pub receiver: flume::Receiver<PersiaBatchData>,
}

#[pymethods]
impl PyPersiaBatchDataChannel {
    #[new]
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = flume::bounded(capacity);
        Self { sender, receiver }
    }

    pub fn get_sender(&self) -> PyPersiaBatchDataSender {
        PyPersiaBatchDataSender {
            inner: self.sender.clone(),
        }
    }

    pub fn get_receiver(&self) -> PyPersiaBatchDataReceiver {
        PyPersiaBatchDataReceiver {
            inner: self.receiver.clone(),
        }
    }
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "utils")?;
    module.add_class::<PyPersiaMessageQueueClient>()?;
    module.add_class::<PyPersiaMessageQueueServer>()?;
    module.add_class::<PyPersiaReplicaInfo>()?;
    module.add_class::<PyPersiaBatchDataChannel>()?;
    module.add_class::<PyPersiaBatchDataReceiver>()?;
    module.add_class::<PyPersiaBatchDataSender>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
