use std::cmp::Ordering;

use crate::tensor::{CPUStorage, Storage, Tensor};

use paste::paste;
use persia_libs::numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use persia_common::{FeatureBatch, SparseBatch, SparseBatchRemoteReference};
use persia_speedy::{Readable, Writable};

#[derive(Readable, Writable, Debug)]
pub enum EmbeddingTensor {
    Null,
    SparseBatch(SparseBatch),
    SparseBatchRemoteReference(SparseBatchRemoteReference),
}

impl EmbeddingTensor {
    pub fn get_remote_ref_info(&self) -> (&str, u64) {
        match &self {
            EmbeddingTensor::SparseBatchRemoteReference(sparse_ref) => {
                (&sparse_ref.middleware_addr, sparse_ref.ref_id)
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Readable, Writable, Debug)]
pub struct PersiaBatchData {
    pub dense_data: Vec<Tensor>,
    pub sparse_data: EmbeddingTensor,
    pub target_data: Vec<Tensor>,
    pub meta_data: Option<Vec<u8>>,
    pub batch_id: Option<usize>,
}

impl Default for PersiaBatchData {
    fn default() -> Self {
        PersiaBatchData {
            dense_data: Vec::new(),
            sparse_data: EmbeddingTensor::Null,
            target_data: Vec::new(),
            meta_data: None,
            batch_id: None,
        }
    }
}

impl PartialEq for PersiaBatchData {
    fn eq(&self, other: &Self) -> bool {
        self.batch_id.unwrap_or(usize::MIN) == other.batch_id.unwrap_or(usize::MIN)
    }
}

impl Eq for PersiaBatchData {}

impl PartialOrd for PersiaBatchData {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PersiaBatchData {
    fn cmp(&self, other: &Self) -> Ordering {
        self.batch_id
            .unwrap_or(usize::MIN)
            .cmp(&other.batch_id.unwrap_or(usize::MIN))
            .reverse()
    }
}

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
            self.inner.dense_data.push(Tensor::new(
                Storage::CPU(CPUStorage::from_f32(
                    x.to_vec().expect("convert ndarray to vec failed"),
                )),
                x.shape().to_vec(),
                None,
                None,
            ));
        });
    }

    pub fn add_sparse(
        &mut self,
        sparse_data: Vec<(String, Vec<&PyArray1<u64>>)>,
        requires_grad: Option<bool>,
    ) {
        self.inner.sparse_data = EmbeddingTensor::SparseBatch(SparseBatch {
            requires_grad: requires_grad.unwrap_or(true),
            batches: sparse_data
                .into_iter()
                .map(|(feature_name, batch)| {
                    let indices = batch
                        .iter()
                        .map(|x| {
                            x.readonly()
                                .as_slice()
                                .expect("cannot read np array")
                                .to_vec()
                        })
                        .collect();
                    FeatureBatch::new(feature_name, indices)
                })
                .collect(),
            ..SparseBatch::default()
        });
    }

    pub fn add_target(&mut self, target_data: &PyArray2<f32>) {
        self.inner.target_data.push(Tensor::new(
            Storage::CPU(CPUStorage::from_f32(
                target_data.to_vec().expect("convert ndarray to vec failed"),
            )),
            target_data.shape().to_vec(),
            None,
            None,
        ));
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
                            self.inner.dense_data.push(Tensor::new(
                                Storage::CPU(CPUStorage::[<from_ $typ:lower>] (data.to_vec().expect("convert ndarray to vec failed"))),
                                data.shape().to_vec(),
                                None,
                                None
                            ));
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
