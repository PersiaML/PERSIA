#![allow(clippy::needless_return)]

pub mod grad;
pub mod message_queue;
pub mod optim;
pub mod tensor;
pub mod utils;

use std::cmp::Ordering;
use std::u64;

use tensor::{DenseTensor, Tensor};

use persia_libs::{
    half,
    half::prelude::*,
    hashbrown::HashMap,
    itertools::Itertools,
    ndarray::{Array1, Array2},
    ndarray_rand::rand_distr::{Gamma, Normal, Poisson, Uniform},
    ndarray_rand::RandomExt,
    numpy::PyArray1,
    rand::prelude::SmallRng,
    rand::SeedableRng,
    serde::{self, Deserialize, Serialize},
};

use persia_embedding_config::InitializationMethod;
use persia_speedy::{Readable, Writable};

#[derive(Serialize, Deserialize, Readable, Writable, Clone, Debug)]
#[serde(crate = "self::serde")]
pub struct HashMapEmbeddingEntry {
    inner: Vec<f32>, // TODO option1: consider using smallvec and slab allocator, and reference that smallvec with &[f32] here to avoid const generics
    // TODO option2: consider wrap BufferPool (see crates.io) or modify sharded slab to allocate &[f32] here
    // TODO option3: consider using a object pool of &[f32] with predefined length and all these &[f32] comes from a large continuous Vec. When the object pool is exhausted, create a new large continuous Vec and split it to &[f32]s and add them to the object pool
    // TODO option4: allocate slices and put them in the slice_arena (see crates.io), then put the slice in the arena into a reusable object pool for consumption
    // TODO option5: allocate slices in bumpalo_herd allocator with alloc_slice_fill_default, and unsafely converts it to Vec, then put the Vec in a reusable object pool for consumption. In this case we can actually put the whole entry in the pool
    embedding_dim: usize,
}

impl HashMapEmbeddingEntry {
    pub fn new(
        initialization_method: &InitializationMethod,
        dim: usize,
        require_space: usize,
        seed: u64,
    ) -> Self {
        let emb = {
            let mut rng = SmallRng::seed_from_u64(seed);
            match initialization_method {
                InitializationMethod::BoundedUniform(x) => {
                    Array1::random_using((dim,), Uniform::new(x.lower, x.upper), &mut rng)
                }
                InitializationMethod::BoundedGamma(x) => {
                    Array1::random_using((dim,), Gamma::new(x.shape, x.scale).unwrap(), &mut rng)
                }
                InitializationMethod::BoundedPoisson(x) => {
                    Array1::random_using((dim,), Poisson::new(x.lambda).unwrap(), &mut rng)
                }
                InitializationMethod::BoundedNormal(x) => Array1::random_using(
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

        let mut inner = emb.into_raw_vec();
        if require_space > 0 {
            inner.resize(inner.len() + require_space, 0.0_f32);
        }
        Self {
            inner,
            embedding_dim: dim,
        }
    }

    pub fn new_empty(dim: usize, require_space: usize) -> Self {
        Self {
            inner: vec![0f32; dim + require_space],
            embedding_dim: dim,
        }
    }

    pub fn from_emb(emb: Vec<f32>) -> Self {
        let embedding_dim = emb.len();
        Self {
            inner: emb,
            embedding_dim,
        }
    }

    pub fn from_emb_and_opt(emb: Vec<f32>, opt: &[f32]) -> Self {
        let embedding_dim = emb.len();
        let mut inner = emb;
        inner.extend_from_slice(opt);
        Self {
            inner,
            embedding_dim,
        }
    }

    pub fn copy_from_other(&mut self, other: &Self) -> bool {
        if self.embedding_dim() != other.embedding_dim() {
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

    pub fn inner_size(&self) -> usize {
        self.inner.len()
    }

    pub fn dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn emb(&self) -> &[f32] {
        &self.inner[..self.embedding_dim()]
    }

    pub fn emb_mut(&mut self) -> &mut [f32] {
        let dim = self.embedding_dim();
        &mut self.inner[..dim]
    }

    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }

    pub fn opt(&self) -> &[f32] {
        &self.inner[self.embedding_dim()..]
    }

    pub fn opt_mut(&mut self) -> &mut [f32] {
        let dim = self.embedding_dim();
        &mut self.inner[dim..]
    }

    pub fn emb_and_opt_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        let dim = self.embedding_dim();
        self.inner.split_at_mut(dim)
    }
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
#[serde(crate = "self::serde")]
pub struct SingleSignInFeatureBatch {
    pub sign: u64,
    pub in_which_batch_samples: Vec<(u16, u16)>,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
#[serde(crate = "self::serde")]
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
#[serde(crate = "self::serde")]
pub struct FeatureRawEmbeddingBatch {
    pub feature_name: String,
    pub embeddings: Array2<half::f16>,
    pub index: Vec<usize>,
    pub sample_id_num: Vec<usize>,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
#[serde(crate = "self::serde")]
pub struct FeatureSumEmbeddingBatch {
    pub feature_name: String,
    pub embeddings: Array2<half::f16>,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
#[serde(crate = "self::serde")]
pub enum FeatureEmbeddingBatch {
    RawEmbedding(FeatureRawEmbeddingBatch),
    SumEmbedding(FeatureSumEmbeddingBatch),
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug)]
#[serde(crate = "self::serde")]
pub struct EmbeddingBatch {
    pub batches: Vec<FeatureEmbeddingBatch>,
    pub backward_ref_id: Option<u64>,
}

#[derive(Deserialize, Serialize, Readable, Writable, Debug, Clone)]
#[serde(crate = "self::serde")]
pub struct SparseBatch {
    pub requires_grad: bool,
    pub batches: Vec<FeatureBatch>,
    #[serde(skip)]
    pub enter_forward_id_buffer_time: Option<std::time::SystemTime>,
    #[serde(skip)]
    pub enter_post_forward_buffer_time: Option<std::time::SystemTime>,
    #[serde(skip)]
    pub batcher_idx: Option<usize>,
}

impl SparseBatch {
    pub fn new(batches: Vec<(String, Vec<&PyArray1<u64>>)>, requires_grad: Option<bool>) -> Self {
        SparseBatch {
            requires_grad: requires_grad.unwrap_or(true),
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

#[derive(Readable, Writable, Debug, Clone)]
pub struct SparseBatchRemoteReference {
    pub middleware_addr: String,
    pub ref_id: u64,
    pub batcher_idx: usize,
}

impl Default for SparseBatchRemoteReference {
    fn default() -> Self {
        Self {
            middleware_addr: String::from(""),
            ref_id: 0,
            batcher_idx: 0,
        }
    }
}

#[derive(Readable, Writable, Debug)]
pub enum EmbeddingTensor {
    Null,
    SparseBatch(SparseBatch),
    SparseBatchRemoteReference(SparseBatchRemoteReference),
}

impl EmbeddingTensor {
    pub fn to_forward_id(&self) -> (&str, u64) {
        match &self {
            EmbeddingTensor::SparseBatchRemoteReference(sparse_ref) => {
                (&sparse_ref.middleware_addr, sparse_ref.ref_id)
            }
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

pub fn ndarray_f32_to_f16(input: &Array2<f32>) -> Array2<f16> {
    let s = input.as_slice().unwrap();
    let f16v = Vec::from_f32_slice(s);
    unsafe { Array2::from_shape_vec_unchecked((input.shape()[0], input.shape()[1]), f16v) }
}

pub fn ndarray_f16_to_f32(input: &Array2<f16>) -> Array2<f32> {
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
    unsafe { Array2::from_shape_vec_unchecked((input.shape()[0], input.shape()[1]), f32v) }
}
