#![allow(clippy::needless_return)]

use half::prelude::*;
use hashbrown::HashMap;
use itertools::Itertools;
use ndarray::Array2;
use ndarray_rand::rand_distr::{Gamma, Normal, Poisson, Uniform};
use ndarray_rand::RandomExt;
use numpy::PyArray1;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::u64;

use persia_embedding_config::InitializationMethod;
use persia_speedy::{Readable, Writable};

pub mod optim;

#[derive(Serialize, Deserialize, Readable, Writable, Clone, Debug)]
pub struct HashMapEmbeddingEntry {
    inner: Vec<f32>, // TODO option1: consider using smallvec and slab allocator, and reference that smallvec with &[f32] here to avoid const generics
                     // TODO option2: consider wrap BufferPool (see crates.io) or modify sharded slab to allocate &[f32] here
                     // TODO option3: consider using a object pool of &[f32] with predefined length and all these &[f32] comes from a large continuous Vec. When the object pool is exhausted, create a new large continuous Vec and split it to &[f32]s and add them to the object pool
                     // TODO option4: allocate slices and put them in the slice_arena (see crates.io), then put the slice in the arena into a reusable object pool for consumption
                     // TODO option5: allocate slices in bumpalo_herd allocator with alloc_slice_fill_default, and unsafely converts it to Vec, then put the Vec in a reusable object pool for consumption. In this case we can actually put the whole entry in the pool
}

impl HashMapEmbeddingEntry {
    pub fn new(
        initialization_method: &InitializationMethod,
        dim: usize,
        require_space: usize,
        seed: u64,
    ) -> Self {
        let emb = {
            let mut rng = rand::prelude::SmallRng::seed_from_u64(seed);
            match initialization_method {
                InitializationMethod::BoundedUniform(x) => {
                    ndarray::Array1::random_using((dim,), Uniform::new(x.lower, x.upper), &mut rng)
                }
                InitializationMethod::BoundedGamma(x) => ndarray::Array1::random_using(
                    (dim,),
                    Gamma::new(x.shape, x.scale).unwrap(),
                    &mut rng,
                ),
                InitializationMethod::BoundedPoisson(x) => {
                    ndarray::Array1::random_using((dim,), Poisson::new(x.lambda).unwrap(), &mut rng)
                }
                InitializationMethod::BoundedNormal(x) => ndarray::Array1::random_using(
                    (dim,),
                    Normal::new(x.mean, x.standard_deviation).unwrap(),
                    &mut rng,
                ),
                _ => panic!(
                    "unsupported initialization method for hashmap impl: {:?}",
                    initialization_method
                ),
            }
        };

        let mut emb = emb.into_raw_vec();
        if require_space > 0 {
            emb.extend(vec![0f32; require_space]);
        }
        Self { inner: emb }
    }

    pub fn new_empty(len: usize) -> Self {
        Self {
            inner: vec![0f32; len],
        }
    }

    pub fn from_emb_and_opt(mut emb: Vec<f32>, opt: &[f32]) -> Self {
        assert_eq!(emb.len(), opt.len());
        emb.extend_from_slice(opt);
        Self { inner: emb }
    }

    pub fn from_other(&mut self, other: Self) -> bool {
        self.inner = other.inner;
        true
    }

    pub fn copy_from_other(&mut self, other: &Self) -> bool {
        if self.dim_infer() != other.dim_infer() {
            return false;
        }
        for (dst, src) in self.inner.iter_mut().zip(other.inner.iter()) {
            *dst = *src;
        }
        return true;
    }

    pub fn as_mut_emb_entry_slice(&mut self) -> &mut [f32] {
        self.inner.as_mut_slice()
    }

    pub fn as_emb_entry_slice(&self) -> &[f32] {
        self.inner.as_slice()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn dim(&self) -> usize {
        self.inner.len() / 2
    }

    pub fn emb(&self) -> &[f32] {
        &self.inner[..self.dim()]
    }

    pub fn emb_mut(&mut self) -> &mut [f32] {
        let dim = self.dim();
        &mut self.inner[..dim]
    }

    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }

    pub fn opt(&self) -> &[f32] {
        &self.inner[self.dim()..]
    }

    pub fn opt_mut(&mut self) -> &mut [f32] {
        let dim = self.dim();
        &mut self.inner[dim..]
    }

    pub fn emb_and_opt_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        let dim = self.dim();
        self.inner.split_at_mut(dim)
    }

    pub fn emb_infer(&self) -> &[f32] {
        &self.inner.as_slice()
    }

    pub fn dim_infer(&self) -> usize {
        self.inner.len()
    }

    pub fn from_emb_infer(emb: Vec<f32>) -> Self {
        Self { inner: emb }
    }
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct SingleSignInFeatureBatch {
    pub sign: u64,
    pub in_which_batch_samples: Vec<(u16, u16)>,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct FeatureBatch {
    pub feature_name: String,
    pub index_batch: Vec<SingleSignInFeatureBatch>,
    /// how many signs in each sample of the batch
    pub sample_num_signs: Vec<u32>,
    pub hashed2index_batch_idx: HashMap<u64, usize>, // hashed2index_batch_idx is the mapping from emb_id to raw_embedding_result idx. Place each emb result in correct idx and processed the raw emb gradient update
    pub batch_size: u16,
}

impl FeatureBatch {
    pub fn new(feature_name: String, batch: Vec<Vec<u64>>) -> Self {
        let batch_size = batch.len();
        let mut sample_num_signs = Vec::with_capacity(batch_size);
        if batch_size > u16::MAX as usize {
            panic!("batch size cannot be larger than {}", u16::MAX);
        }

        let mut m: HashMap<u64, Vec<(u16, u16)>> = HashMap::default();
        batch
            .into_iter()
            .enumerate()
            .for_each(|(batch_idx, indices)| {
                sample_num_signs.push(indices.len() as u32);
                indices.into_iter().enumerate().for_each(|(col_idx, id)| {
                    m.entry(id)
                        .or_default()
                        .push((batch_idx as u16, col_idx as u16));
                })
            });
        let mut hashed2index_batch_idx: HashMap<u64, usize> = HashMap::default();
        m.iter().enumerate().for_each(|(idx, (id, _))| {
            hashed2index_batch_idx.insert(id.clone(), idx);
        });
        Self {
            feature_name,
            index_batch: m
                .into_iter()
                .map(|x| SingleSignInFeatureBatch {
                    sign: x.0,
                    in_which_batch_samples: x.1,
                })
                .collect_vec(),
            sample_num_signs,
            hashed2index_batch_idx: hashed2index_batch_idx,
            batch_size: batch_size as u16,
        }
    }
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub struct FeatureRawEmbeddingBatch {
    pub feature_name: String,
    pub embeddings: ndarray::Array2<half::f16>,
    pub index: Vec<usize>,
    pub sample_id_num: Vec<usize>,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub struct FeatureSumEmbeddingBatch {
    pub feature_name: String,
    pub embeddings: ndarray::Array2<half::f16>,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub enum FeatureEmbeddingBatch {
    RawEmbedding(FeatureRawEmbeddingBatch),
    SumEmbedding(FeatureSumEmbeddingBatch),
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub struct EmbeddingBatch {
    pub batches: Vec<FeatureEmbeddingBatch>,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
pub struct SparseBatch {
    pub batches: Vec<FeatureBatch>,
    #[serde(skip)]
    pub enter_forward_id_buffer_time: Option<std::time::SystemTime>,
    #[serde(skip)]
    pub enter_post_forward_buffer_time: Option<std::time::SystemTime>,
    #[serde(skip)]
    pub batcher_idx: Option<usize>,
}

impl From<Vec<(String, Vec<&PyArray1<u64>>)>> for SparseBatch {
    fn from(batches: Vec<(String, Vec<&PyArray1<u64>>)>) -> Self {
        SparseBatch {
            batches: batches
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
            enter_forward_id_buffer_time: None,
            enter_post_forward_buffer_time: None,
            batcher_idx: None,
        }
    }
}

#[derive(Readable, Writable, Debug)]
pub enum BaseTensor {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl BaseTensor {
    pub fn type_size(&self) -> usize {
        match &self {
            BaseTensor::F32(_) => std::mem::size_of::<f32>(),
            BaseTensor::F64(_) => std::mem::size_of::<f64>(),
            BaseTensor::I32(_) => std::mem::size_of::<i32>(),
            BaseTensor::I64(_) => std::mem::size_of::<i64>(),
        }
    }
}

#[derive(Readable, Writable, Debug, Clone)]
pub struct PreForwardStub {
    pub middleware_addr: String,
    pub forward_id: u64,
    pub batcher_idx: usize,
}

impl Default for PreForwardStub {
    fn default() -> Self {
        Self {
            middleware_addr: String::from(""),
            forward_id: 0,
            batcher_idx: 0,
        }
    }
}

#[derive(Readable, Writable, Debug)]
pub struct DenseTensor {
    pub data: BaseTensor,
    pub shape: Vec<usize>,
}

#[derive(Readable, Writable, Debug)]
pub struct SparseTensor {
    pub data: BaseTensor,
    pub offset: Vec<u64>,
}
#[derive(Readable, Writable, Debug)]
pub enum Tensor {
    Dense(DenseTensor),
    Sparse(SparseTensor),
}

#[derive(Readable, Writable, Debug)]
pub enum EmbeddingTensor {
    Null,
    PreForwardStub(PreForwardStub),
    SparseBatch(SparseBatch),
}

impl EmbeddingTensor {
    pub fn to_forward_id(&self) -> (&str, u64) {
        match &self {
            EmbeddingTensor::PreForwardStub(stub) => (&stub.middleware_addr, stub.forward_id),
            EmbeddingTensor::SparseBatch(_) => ("", 0u64),
            _ => panic!("forward id not found on embedding tensor"),
        }
    }
}
#[derive(Readable, Writable, Debug)]
pub struct PersiaBatchData {
    pub dense_data: Vec<DenseTensor>,
    pub sparse_data: EmbeddingTensor,
    pub target_data: Vec<DenseTensor>,
    pub map_data: HashMap<String, Tensor>,
    pub meta_data: Option<Vec<u8>>,
    pub batch_id: Option<usize>,
}

impl Default for PersiaBatchData {
    fn default() -> Self {
        PersiaBatchData {
            dense_data: Vec::new(),
            sparse_data: EmbeddingTensor::Null,
            target_data: Vec::new(),
            map_data: HashMap::new(),
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

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub enum Gradients {
    F16(ndarray::Array2<half::f16>),
    F32(ndarray::Array2<f32>),
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub struct FeatureEmbeddingGradientBatch {
    pub feature_name: String,
    pub gradients: Gradients,
    /// true gradient = gradients / scale_factor
    pub scale_factor: f32,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub struct SkippedGradientBatch {
    pub feature_name: String,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub enum SkippableFeatureEmbeddingGradientBatch {
    GradientBatch(FeatureEmbeddingGradientBatch),
    Skipped(SkippedGradientBatch),
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
pub struct EmbeddingGradientBatch {
    pub gradients: Vec<SkippableFeatureEmbeddingGradientBatch>,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Default)]
pub struct EmbeddingMeta {
    pub embedding_name: String,
    pub embedding_dim: usize,
    pub batch_size: usize,
}

#[derive(Readable, Writable, Debug, Default)]
pub struct PersiaBatchedEmbeddingsResponse {
    pub emb_metas: Vec<EmbeddingMeta>,
    pub gateway_server: String,
    pub err_message: String,
    pub data: Vec<f16>,
}

#[derive(Debug, Default)]
pub struct PersiaBatchedEmbeddings {
    pub response: PersiaBatchedEmbeddingsResponse,
    pub addressing: HashMap<String, (usize, usize)>,
    pub full_precision_data: Vec<f32>,
}

impl PersiaBatchedEmbeddings {
    pub fn from_ptr(ptr: *const PersiaBatchedEmbeddings) -> &'static Self {
        unsafe { &*ptr }
    }
}

#[derive(Deserialize, Serialize, Debug, Readable, Writable, Default)]
pub struct IndicesMeta {
    pub embedding_name: String,
    pub indices: Vec<u64>,
    pub indices_offset: Vec<u64>,
}

#[derive(Deserialize, Serialize, Debug, Readable, Writable, Default)]
pub struct PersiaBatchedIndicesRequest {
    pub inner: Vec<IndicesMeta>,
}

#[derive(Debug, Default)]
pub struct PersiaBatchedIndices {
    pub request: PersiaBatchedIndicesRequest,
    pub serialized: Vec<u8>,
}

impl PersiaBatchedIndices {
    pub fn from_ptr_mut(ptr: *mut PersiaBatchedIndices) -> &'static mut Self {
        unsafe { &mut *ptr }
    }

    pub fn from_ptr(ptr: *const PersiaBatchedIndices) -> &'static Self {
        unsafe { &*ptr }
    }
}

#[derive(Debug)]
pub enum TensorDtype {
    F16,
    F32,
}

pub fn ndarray_f32_to_f16(input: &ndarray::Array2<f32>) -> ndarray::Array2<f16> {
    let s = input.as_slice().unwrap();
    let f16v = Vec::from_f32_slice(s);
    unsafe { ndarray::Array2::from_shape_vec_unchecked((input.shape()[0], input.shape()[1]), f16v) }
}

pub fn ndarray_f16_to_f32(input: &ndarray::Array2<f16>) -> ndarray::Array2<f32> {
    let f32v = input
        .as_slice()
        .unwrap()
        .to_f32_vec()
        .into_iter()
        .map(|x| {
            if x == f32::INFINITY {
                half::f16::MAX.to_f32()
            } else if x == f32::NEG_INFINITY {
                half::f16::MIN.to_f32()
            } else {
                x
            }
        })
        .collect_vec();
    unsafe { ndarray::Array2::from_shape_vec_unchecked((input.shape()[0], input.shape()[1]), f32v) }
}

#[derive(Serialize, Deserialize, Readable, Writable, Debug)]
pub struct PersiaBatchRecordedShardedServer {
    pub dense: Vec<PersiaDenseTensor<f32>>,
    pub target: Vec<PersiaDenseTensor<f32>>,
    pub uids: Vec<u64>,
    pub pids: Vec<u64>,
    pub num_samples: usize,
    pub middleware_server_addr: String,
    pub forward_id: u64,
    pub timestamps: Vec<i64>,
    pub metadata: bytes::Bytes,
}

#[derive(Default, Serialize, Deserialize, Readable, Writable, Debug, Clone)]
pub struct PersiaDenseTensor<T> {
    pub name: String,
    pub dim: usize,
    pub content: Vec<T>,
}

impl<T> PersiaDenseTensor<T> {
    pub fn add_sample(&mut self, mut sample: Vec<T>) {
        assert_eq!(sample.len(), self.dim);
        self.content.append(&mut sample);
    }

    pub fn empty_like(&self) -> Self {
        Self {
            name: self.name.clone(),
            dim: self.dim,
            content: vec![],
        }
    }

    pub fn get(&self, sample_idx: usize) -> &[T] {
        &self.content[sample_idx * self.dim..(sample_idx + 1) * self.dim]
    }
}

impl<T> std::convert::TryInto<ndarray::Array2<T>> for PersiaDenseTensor<T> {
    type Error = ndarray::ShapeError;

    fn try_into(self) -> Result<Array2<T>, Self::Error> {
        ndarray::Array2::<T>::from_shape_vec(
            (self.content.len() / self.dim, self.dim),
            self.content,
        )
    }
}

impl<T> IntoIterator for PersiaDenseTensor<T> {
    type Item = Vec<T>;
    type IntoIter = PersiaDenseTensorSampleIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        assert_eq!(self.content.len() % self.dim, 0);
        let mut chunks = self
            .content
            .into_iter()
            .chunks(self.dim)
            .into_iter()
            .map(|chunk| chunk.collect_vec())
            .collect_vec();
        chunks.reverse();
        PersiaDenseTensorSampleIterator { chunks }
    }
}

pub struct PersiaDenseTensorSampleIterator<T> {
    chunks: Vec<Vec<T>>,
}

impl<T> Iterator for PersiaDenseTensorSampleIterator<T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.pop()
    }
}
