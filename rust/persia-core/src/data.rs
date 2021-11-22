use std::cmp::Ordering;

use crate::tensor::{CPUStorage, Storage, TensorImpl};

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
pub struct PersiaBatchImpl {
    pub not_id_type_features: Vec<TensorImpl>,
    pub id_type_features: EmbeddingTensor,
    pub labels: Vec<TensorImpl>,
    pub meta_data: Option<Vec<u8>>,
    pub batch_id: Option<usize>,
}

impl Default for PersiaBatchImpl {
    fn default() -> Self {
        PersiaBatchImpl {
            not_id_type_features: Vec::new(),
            id_type_features: EmbeddingTensor::Null,
            labels: Vec::new(),
            meta_data: None,
            batch_id: None,
        }
    }
}

impl PartialEq for PersiaBatchImpl {
    fn eq(&self, other: &Self) -> bool {
        self.batch_id.unwrap_or(usize::MIN) == other.batch_id.unwrap_or(usize::MIN)
    }
}

impl Eq for PersiaBatchImpl {}

impl PartialOrd for PersiaBatchImpl {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PersiaBatchImpl {
    fn cmp(&self, other: &Self) -> Ordering {
        self.batch_id
            .unwrap_or(usize::MIN)
            .cmp(&other.batch_id.unwrap_or(usize::MIN))
            .reverse()
    }
}

#[pyclass]
pub struct PersiaBatch {
    pub inner: PersiaBatchImpl,
}

#[pymethods]
impl PersiaBatch {
    #[new]
    pub fn new() -> Self {
        PersiaBatch {
            inner: PersiaBatchImpl::default(),
        }
    }

    pub fn add_not_id_type_feature(&mut self, not_id_type_feature: Vec<&PyArray2<f32>>) {
        not_id_type_feature.iter().for_each(|x| {
            self.inner.not_id_type_features.push(TensorImpl::new(
                Storage::CPU(CPUStorage::from_f32(
                    x.to_vec().expect("convert ndarray to vec failed"),
                )),
                x.shape().to_vec(),
                None,
                None,
            ));
        });
    }

    pub fn add_id_type_features(
        &mut self,
        id_type_features: Vec<(String, Vec<&PyArray1<u64>>)>,
        requires_grad: Option<bool>,
    ) {
        self.inner.id_type_features = EmbeddingTensor::SparseBatch(SparseBatch {
            requires_grad: requires_grad.unwrap_or(true),
            batches: id_type_features
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

    pub fn add_label(&mut self, label_data: &PyArray2<f32>) {
        self.inner.labels.push(TensorImpl::new(
            Storage::CPU(CPUStorage::from_f32(
                label_data.to_vec().expect("convert ndarray to vec failed"),
            )),
            label_data.shape().to_vec(),
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
            impl PersiaBatch {
                    $(
                        pub fn [<add_not_id_type_features_ $typ:lower>](&mut self, data: &PyArray2<$typ>) {
                            self.inner.not_id_type_features.push(TensorImpl::new(
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
    module.add_class::<PersiaBatch>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
