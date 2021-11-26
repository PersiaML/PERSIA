use std::cmp::Ordering;

use crate::tensor::{CPUStorage, Storage, TensorImpl};

use numpy::{DataType, PyArray, PyArray1, PyArrayDescr, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::AsPyPointer;

use persia_common::{FeatureBatch, IDTypeFeatureBatch, IDTypeFeatureRemoteRef};
use persia_speedy::{Readable, Writable};

#[derive(Readable, Writable, Debug)]
pub enum EmbeddingTensor {
    Null,
    IDTypeFeature(IDTypeFeatureBatch),
    IDTypeFeatureRemoteRef(IDTypeFeatureRemoteRef),
}

impl EmbeddingTensor {
    pub fn get_remote_ref_info(&self) -> (&str, u64) {
        match &self {
            EmbeddingTensor::IDTypeFeatureRemoteRef(id_type_feature_ref) => (
                &id_type_feature_ref.embedding_worker_addr,
                id_type_feature_ref.ref_id,
            ),
            _ => unreachable!(),
        }
    }
}

#[derive(Readable, Writable, Debug)]
pub struct PersiaBatchImpl {
    pub non_id_type_features: Vec<TensorImpl>,
    pub id_type_features: EmbeddingTensor,
    pub labels: Vec<TensorImpl>,
    pub meta_data: Option<Vec<u8>>,
    pub batch_id: Option<usize>,
}

impl Default for PersiaBatchImpl {
    fn default() -> Self {
        PersiaBatchImpl {
            non_id_type_features: Vec::new(),
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

/// This macro generate the TensorImpl by pyobject and dtype.
macro_rules! gen_tensor_impl_by_datatype {
    ($ptr:expr, $dtype:expr, $py:expr, $name:expr, $(($typ:ty, $np_dtype:ident, $attr:ident)),*) => {
        match $dtype {
            $(
                DataType::$np_dtype => {
                    let py_array: &PyArrayDyn<$typ> = PyArray::from_borrowed_ptr($py, $ptr);
                    TensorImpl::new(
                        Storage::CPU(CPUStorage::$attr(
                            py_array.to_vec().expect("convert ndarray to vec failed"),
                        )),
                        py_array.shape().to_vec(),
                        $name,
                        None,
                    )
                }
            )*
            _ => panic!("Unsupport datatype of ndarray"),
        }

    }
}

/// convert pyarray object to persia tensor
/// this function will panic when meet some datatype numpy::Datatype can't support
fn convert_pyarray_object_to_tensor_impl(
    pyarray_object: &PyAny,
    dtype: &PyAny,
    name: Option<String>,
    python: Python,
) -> TensorImpl {
    let dtype: &PyArrayDescr = dtype.downcast().expect(format!(
        "PersiaBatch datatype parse error {}, check PersiaBatch datatype support list to prevent datatype parse error.",
        name.clone().unwrap_or("unknow_data".to_string())
    ).as_str());
    let datatype = dtype.get_datatype().unwrap();

    unsafe {
        let pyarray_ptr = AsPyPointer::as_ptr(pyarray_object);
        gen_tensor_impl_by_datatype!(
            pyarray_ptr,
            datatype,
            python,
            name,
            (bool, Bool, BOOL),
            (f32, Float32, F32),
            (f64, Float64, F64),
            (i8, Int8, I8),
            (i16, Int16, I16),
            (i32, Int32, I32),
            (i64, Int64, I64),
            (u8, Uint8, U8),
            (u16, Uint16, U16),
            (u32, Uint32, U32),
            (u64, Uint64, U64)
        )
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

    pub fn add_non_id_type_feature(
        &mut self,
        pyarray_object: &PyAny,
        dtype: &PyAny,
        name: Option<String>,
        py: Python,
    ) {
        let tensor_impl = convert_pyarray_object_to_tensor_impl(pyarray_object, dtype, name, py);
        self.inner.non_id_type_features.push(tensor_impl);
    }

    pub fn add_label(
        &mut self,
        py_object: &PyAny,
        dtype: &PyAny,
        name: Option<String>,
        py: Python,
    ) {
        let tensor_impl = convert_pyarray_object_to_tensor_impl(py_object, dtype, name, py);
        self.inner.labels.push(tensor_impl);
    }

    pub fn add_id_type_features(
        &mut self,
        id_type_features: Vec<(String, Vec<&PyArray1<u64>>)>,
        requires_grad: Option<bool>,
    ) {
        self.inner.id_type_features = EmbeddingTensor::IDTypeFeature(IDTypeFeatureBatch {
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
            ..IDTypeFeatureBatch::default()
        });
    }

    pub fn add_meta(&mut self, data: Option<&PyBytes>) {
        self.inner.meta_data = data.map(|x| x.as_bytes().to_vec());
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

pub fn init_module(super_module: &PyModule, py: Python) -> PyResult<()> {
    let module = PyModule::new(py, "data")?;
    module.add_class::<PersiaBatch>()?;
    super_module.add_submodule(module)?;
    Ok(())
}
