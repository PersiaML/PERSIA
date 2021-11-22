use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::data::{PersiaBatch, PersiaBatchImpl};

use persia_common::message_queue::{PersiaMessageQueueClientImpl, PersiaMessageQueueServerImpl};
use persia_libs::{flume, tokio::runtime::Runtime};

#[pyclass]
pub struct PersiaMessageQueueClient {
    pub inner: PersiaMessageQueueClientImpl,
    pub runtime: Runtime,
}

#[pymethods]
impl PersiaMessageQueueClient {
    #[new]
    fn new(server_addr: &str) -> Self {
        let runtime = persia_libs::tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(5)
            .build()
            .unwrap();

        let _guard = runtime.enter();

        Self {
            inner: PersiaMessageQueueClientImpl::new(server_addr),
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
pub struct PersiaMessageQueueServer {
    inner: PersiaMessageQueueServerImpl,
    runtime: Runtime, // thread unsafe
}

#[pymethods]
impl PersiaMessageQueueServer {
    #[new]
    fn new(port: u16, cap: usize) -> Self {
        let runtime = persia_libs::tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(5)
            .build()
            .unwrap();

        let _guard = runtime.enter();

        Self {
            inner: PersiaMessageQueueServerImpl::new(port, cap),
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
pub struct PersiaBatchDataSender {
    pub inner: flume::Sender<PersiaBatchImpl>,
}

#[pymethods]
impl PersiaBatchDataSender {
    pub fn send(&self, batch_data: &mut PersiaBatch, py: Python) -> PyResult<()> {
        let batch_data = std::mem::take(&mut batch_data.inner);
        py.allow_threads(move || {
            self.inner.send(batch_data).unwrap();
            Ok(())
        })
    }
}

#[pyclass]
pub struct PersiaBatchDataReceiver {
    pub inner: flume::Receiver<PersiaBatchImpl>,
}
#[pyclass]
pub struct PersiaBatchDataChannel {
    pub sender: flume::Sender<PersiaBatchImpl>,
    pub receiver: flume::Receiver<PersiaBatchImpl>,
}

#[pymethods]
impl PersiaBatchDataChannel {
    #[new]
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = flume::bounded(capacity);
        Self { sender, receiver }
    }

    pub fn get_sender(&self) -> PersiaBatchDataSender {
        PersiaBatchDataSender {
            inner: self.sender.clone(),
        }
    }

    pub fn get_receiver(&self) -> PersiaBatchDataReceiver {
        PersiaBatchDataReceiver {
            inner: self.receiver.clone(),
        }
    }
}

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "utils")?;
    module.add_class::<PersiaMessageQueueClient>()?;
    module.add_class::<PersiaMessageQueueServer>()?;
    module.add_class::<PersiaBatchDataChannel>()?;
    module.add_class::<PersiaBatchDataReceiver>()?;
    module.add_class::<PersiaBatchDataSender>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
