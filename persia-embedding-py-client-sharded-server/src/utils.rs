use pyo3::prelude::*;
use pyo3::types::PyBytes;

use persia_futures::tokio::runtime::Runtime;
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

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "utils")?;
    module.add_class::<PyPersiaMessageQueueClient>()?;
    module.add_class::<PyPersiaMessageQueueServer>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
