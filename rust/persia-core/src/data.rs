use paste::paste;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use persia_libs::numpy::{PyArray1, PyArray2};

use persia_common::{
    tensor::{CPUStorage, Storage, Tensor},
    EmbeddingTensor, PersiaBatchData, SparseBatch,
};
use persia_speedy::Writable;

#[pyclass]
pub struct PyPersiaBatchData {
    pub inner: PersiaBatchData,
}

#[pymethods]
impl PyPersiaBatchData {
    #[new]
    pub fn new() -> Self {
        PyPersiaBatchData {
            inner: PersiaBatchData::default(),
        }
    }

    pub fn add_dense(&mut self, data: Vec<&PyArray2<f32>>) {
        data.iter().for_each(|x| {
            self.inner.dense_data.push(Tensor {
                storage: Storage::CPU(CPUStorage::from_f32(
                    x.to_vec().expect("convert ndarray to vec failed"),
                )),
                shape: x.shape().to_vec(),
                name: None,
            });
        });
    }

    pub fn add_sparse(&mut self, sparse_data: Vec<(String, Vec<&PyArray1<u64>>)>) {
        self.inner.sparse_data = EmbeddingTensor::SparseBatch(SparseBatch::from(sparse_data));
    }

    pub fn add_target(&mut self, target_data: &PyArray2<f32>) {
        self.inner.target_data.push(Tensor {
            storage: Storage::CPU(CPUStorage::from_f32(
                target_data.to_vec().expect("convert ndarray to vec failed"),
            )),
            shape: target_data.shape().to_vec(),
            name: None,
        });
    }

    pub fn add_meta(&mut self, data: &PyBytes) {
        self.inner.meta_data = Some(data.as_bytes().to_vec());
    }

    pub fn to_bytes<'a>(&mut self, _py: Python<'a>) -> &'a PyBytes {
        PyBytes::new(_py, self.inner.write_to_vec().unwrap().as_slice())
    }

    pub fn batch_id(&self) -> usize {
        self.inner
            .batch_id
            .expect("please call forward_id before get batch_id")
    }
}

macro_rules! add_dense_func2batch_data {
    ($(($typ:ty, $attr:ident)),*) => {
        paste! {
            #[pymethods]
            impl PyPersiaBatchData {
                    $(
                        pub fn [<add_dense_ $typ:lower>](&mut self, data: &PyArray2<$typ>) {
                            self.inner.dense_data.push(Tensor {
                                storage: Storage::CPU(CPUStorage::[<from_ $typ:lower>] (data.to_vec().expect("convert ndarray to vec failed"))),
                                shape: data.shape().to_vec(),
                                name: None,
                            });
                        }
                    )*
            }
        }
    }
}

add_dense_func2batch_data!((f32, F32), (f64, F64), (i32, I32), (i64, I64));

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "data")?;
    module.add_class::<PyPersiaBatchData>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
