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
    ndarray::Array2,
    numpy::PyArray1,
    serde::{self, Deserialize, Serialize},
};

use persia_speedy::{Readable, Writable};

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
