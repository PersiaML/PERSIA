use crate::embedding_service::{
    EmbeddingServerError, EmbeddingServerNatsServicePublisher, EmbeddingServiceClient,
};

use std::ops::MulAssign;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use persia_libs::{
    async_lock::RwLock,
    bytes, futures,
    hashbrown::HashMap,
    hyper,
    itertools::Itertools,
    lz4, ndarray, once_cell,
    retry::{delay::Fixed, retry},
    smol::block_on,
    thiserror, tokio, tracing,
};
use snafu::ResultExt;

use persia_common::{
    grad::{EmbeddingGradientBatch, Gradients, SkippableFeatureEmbeddingGradientBatch},
    ndarray_f16_to_f32, ndarray_f32_to_f16,
    optim::OptimizerConfig,
    EmbeddingBatch, FeatureEmbeddingBatch, FeatureRawEmbeddingBatch, FeatureSumEmbeddingBatch,
    HashMapEmbeddingEntry, SingleSignInFeatureBatch, SparseBatch, SparseBatchRemoteReference,
};
use persia_embedding_config::{
    EmbeddingConfig, InstanceInfo, PersiaGlobalConfigError, PersiaMiddlewareConfig,
    PersiaReplicaInfo, PersiaSparseModelHyperparameters, SlotConfig,
};
use persia_metrics::{
    Gauge, GaugeVec, Histogram, IntCounterVec, PersiaMetricsManager, PersiaMetricsManagerError,
};
use persia_model_manager::PersiaPersistenceStatus;
use persia_nats_client::{NatsClient, NatsError};
use persia_speedy::{Readable, Writable};

static METRICS_HOLDER: once_cell::sync::OnceCell<MetricsHolder> = once_cell::sync::OnceCell::new();

struct MetricsHolder {
    pub batch_unique_indices_rate: GaugeVec,
    pub num_pending_batches: Gauge,
    pub staleness: Gauge,
    pub nan_count: IntCounterVec,
    pub nan_grad_skipped: IntCounterVec,
    pub lookup_create_requests_time_cost: Histogram,
    pub lookup_rpc_time_cost: Histogram,
    pub update_gradient_time_cost: Histogram,
    pub summation_time_cost: Histogram,
    pub lookup_batched_time_cost: Histogram,
}

impl MetricsHolder {
    pub fn get() -> Result<&'static Self, PersiaMetricsManagerError> {
        METRICS_HOLDER.get_or_try_init(|| {
            let m = PersiaMetricsManager::get()?;
            let holder = Self {
                batch_unique_indices_rate: m
                    .create_gauge_vec("batch_unique_indices_rate", "ATT")?,
                num_pending_batches: m.create_gauge("num_pending_batches", "ATT")?,
                staleness: m.create_gauge("staleness", "ATT")?,
                nan_count: m.create_counter_vec(
                    "nan_count",
                    "nan count of gradient pushed to emb server",
                )?,
                nan_grad_skipped: m.create_counter_vec(
                    "nan_grad_skipped",
                    "nan count of gradient filtered by gpu node",
                )?,
                lookup_create_requests_time_cost: m
                    .create_histogram("lookup_create_requests_time_cost", "ATT")?,
                lookup_rpc_time_cost: m.create_histogram("lookup_rpc_time_cost", "ATT")?,
                update_gradient_time_cost: m
                    .create_histogram("update_gradient_time_cost", "ATT")?,
                summation_time_cost: m.create_histogram("summation_time_cost", "ATT")?,
                lookup_batched_time_cost: m.create_histogram("lookup_batched_time_cost", "ATT")?,
            };
            Ok(holder)
        })
    }
}

#[derive(thiserror::Error, Debug, Readable, Writable)]
pub enum MiddlewareServerError {
    #[error("rpc error")]
    RpcError(String),
    #[error("embedding server error: {0}")]
    EmbeddingServerError(#[from] EmbeddingServerError),
    #[error("forward id not found")]
    ForwardIdNotFound(u64),
    #[error("forward failed")]
    ForwardFailed(String),
    #[error("backward failed")]
    BackwardFailed(String),
    #[error("connection failed")]
    ConnectionError,
    #[error("lookup server ip address error")]
    LookupIpAddressError,
    #[error("server replica idx not match error")]
    ReplicaIdxNotMatchError,
    #[error("gradient contains nan")]
    NanGradient,
    #[error("nats error: {0}")]
    NatsError(#[from] NatsError),
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("forward buffer full")]
    ForwardBufferFull,
    #[error("data src idx not set")]
    DataSrcIdxNotSet,
}

pub struct AllEmbeddingServerClient {
    pub clients: RwLock<Vec<Arc<EmbeddingServiceClient>>>,
    pub nats_publisher: Option<EmbeddingServerNatsServicePublisher>,
    pub dst_replica_size: usize,
}

impl AllEmbeddingServerClient {
    pub fn with_nats(nats_publisher: EmbeddingServerNatsServicePublisher) -> Self {
        tracing::info!("trying to get replica info of embedding servers");
        let dst_replica_info: Result<PersiaReplicaInfo, _> =
            retry(Fixed::from_millis(5000), || {
                let resp = block_on(AllEmbeddingServerClient::get_dst_replica_info(
                    &nats_publisher,
                ));
                if resp.is_err() {
                    tracing::warn!(
                        "failed to get replica info of embedding servers, due to {:?}, retrying",
                        resp
                    );
                } else {
                    tracing::info!(
                        "succeed to get replica info of embedding servers, {:?}",
                        resp
                    );
                }
                resp
            });
        let dst_replica_info =
            dst_replica_info.expect("failed to get replica info of embedding server");

        let instance = Self {
            clients: RwLock::new(Vec::with_capacity(dst_replica_info.replica_size)),
            nats_publisher: Some(nats_publisher),
            dst_replica_size: dst_replica_info.replica_size,
        };

        let servers = instance
            .get_all_addresses()
            .expect("failed to get ips of servers");

        instance
            .update_rpc_clients(servers)
            .expect("failed to init rpc client for embedding server");

        instance
    }

    pub fn with_addrs(servers: Vec<String>) -> Self {
        tracing::info!(
            "AllEmbeddingServerClient::with_addrs, servers are {:?}",
            servers
        );
        let instance = Self {
            clients: RwLock::new(Vec::with_capacity(servers.len())),
            nats_publisher: None,
            dst_replica_size: servers.len(),
        };

        instance
            .update_rpc_clients(servers)
            .expect("failed to init rpc client for embedding server");

        instance
    }

    pub async fn ready_for_serving(&self) -> bool {
        let clients = block_on(self.clients.read());
        let futs = clients.iter().map(|client| async move {
            let resp = client.ready_for_serving(&()).await;
            if let Ok(x) = resp {
                if x {
                    return Ok(());
                }
            }
            return Err(());
        });

        futures::future::try_join_all(futs).await.is_ok()
    }

    pub async fn model_manager_status(&self) -> Vec<PersiaPersistenceStatus> {
        let clients = block_on(self.clients.read());
        let futs = clients
            .iter()
            .map(|client| async move { client.model_manager_status(&()).await });

        let status: Vec<_> = futures::future::try_join_all(futs).await.unwrap_or(vec![
                PersiaPersistenceStatus::Failed(String::from(
                    "failed to get status"
                ));
                self.clients.read().await.len()
            ]);

        return status;
    }

    pub fn replica_size(&self) -> usize {
        self.dst_replica_size
    }

    pub async fn get_client_by_index(&self, client_index: usize) -> Arc<EmbeddingServiceClient> {
        let clients = self.clients.read().await;
        clients.get(client_index).unwrap().clone()
    }

    pub async fn get_dst_replica_info(
        nats_publisher: &EmbeddingServerNatsServicePublisher,
    ) -> Result<PersiaReplicaInfo, MiddlewareServerError> {
        let dst_replica_info = nats_publisher.publish_get_replica_info(&(), None).await??;
        Ok(dst_replica_info)
    }

    pub async fn get_address(&self, replica_index: usize) -> Result<String, MiddlewareServerError> {
        let addr = self
            .nats_publisher
            .as_ref()
            .expect("nats_publisher is None, you are using infer mode")
            .publish_get_address(&(), Some(replica_index))
            .await??;
        Ok(addr)
    }

    pub fn get_all_addresses(&self) -> Result<Vec<String>, MiddlewareServerError> {
        let servers: Vec<_> = (0..self.dst_replica_size)
            .map(|replica_index| {
                tracing::info!("trying to get ip address of server {}", replica_index);
                retry(Fixed::from_millis(1000), || {
                    let addr = block_on(self.get_address(replica_index));
                    if addr.is_err() {
                        tracing::warn!(
                            "failed to get address of server {}, due to {:?}, retrying",
                            replica_index,
                            addr
                        );
                    } else {
                        tracing::info!(
                            "succeed to get address of embedding server {}, {:?}",
                            replica_index,
                            addr
                        );
                    }
                    addr
                })
            })
            .collect();

        if servers.iter().any(|x| x.is_err()) {
            return Err(MiddlewareServerError::LookupIpAddressError);
        }
        let servers = servers.into_iter().map(|x| x.unwrap()).collect();
        Ok(servers)
    }

    pub fn update_rpc_clients(&self, servers: Vec<String>) -> Result<(), MiddlewareServerError> {
        let mut clients = block_on(self.clients.write());
        clients.clear();

        for server_addr in servers {
            let rpc_client = persia_rpc::RpcClient::new(server_addr.as_str()).unwrap();
            let client = EmbeddingServiceClient::new(rpc_client);
            clients.push(Arc::new(client));
        }

        clients.sort_by_key(|c| {
            let replica_index = retry(Fixed::from_millis(100), || block_on(c.replica_index(&())));
            let replica_index = replica_index.expect("failed to call replica_index via rpc");
            replica_index
        });

        for (i, c) in clients.iter().enumerate() {
            match block_on(c.replica_index(&())) {
                Ok(idx) => {
                    if idx != i {
                        tracing::error!("replica index wrong");
                        return Err(MiddlewareServerError::ReplicaIdxNotMatchError);
                    }
                }
                Err(e) => {
                    tracing::error!("failed to call replica_index due to {:?}", e);
                    return Err(MiddlewareServerError::ConnectionError);
                }
            }
        }

        Ok(())
    }
}

#[inline]
pub fn sign_to_shard_modulo(sign: u64, replica_size: u64) -> u64 {
    let sign = farmhash::hash64(&sign.to_le_bytes());
    sign % replica_size
}

#[inline]
pub fn indices_to_hashstack_indices(indices: &mut SparseBatch, config: &EmbeddingConfig) -> () {
    for feature_batch in indices.batches.iter_mut() {
        let slot_conf = config
            .slots_config
            .get(&feature_batch.feature_name)
            .expect("slot not found");

        if slot_conf.hash_stack_config.hash_stack_rounds > 0 {
            let mut hash_stack_indices: Vec<HashMap<u64, Vec<(u16, u16)>>> =
                vec![HashMap::new(); slot_conf.hash_stack_config.hash_stack_rounds];
            let mut hashed2index_batch_idx: HashMap<u64, usize> = HashMap::with_capacity(
                feature_batch.index_batch.len() * slot_conf.hash_stack_config.hash_stack_rounds,
            );
            feature_batch.index_batch.iter().enumerate().for_each(
                |(distinct_tensor_idx, single_sign)| {
                    let mut hashed_sign = single_sign.sign;
                    for (round, map) in hash_stack_indices.iter_mut().enumerate() {
                        hashed_sign = farmhash::hash64(&hashed_sign.to_le_bytes());
                        let hashed_sign_bucket = hashed_sign.clone()
                            % slot_conf.hash_stack_config.embedding_size as u64
                            + (round * slot_conf.hash_stack_config.embedding_size) as u64;
                        // TODO: to avoid hash conflict, try replace hashed2index_batch_idx to key2list
                        hashed2index_batch_idx.insert(hashed_sign_bucket, distinct_tensor_idx);
                        map.entry(hashed_sign_bucket)
                            .or_insert_with(|| {
                                Vec::with_capacity(single_sign.in_which_batch_samples.len())
                            })
                            .extend_from_slice(single_sign.in_which_batch_samples.as_slice());
                    }
                },
            );
            let mut hashed_index_batch: Vec<SingleSignInFeatureBatch> = Vec::with_capacity(
                hash_stack_indices.first().unwrap().len()
                    * slot_conf.hash_stack_config.hash_stack_rounds,
            );
            for map in hash_stack_indices.into_iter() {
                for (k, v) in map {
                    hashed_index_batch.push(SingleSignInFeatureBatch {
                        sign: k,
                        in_which_batch_samples: v,
                    });
                }
            }
            feature_batch.index_batch = hashed_index_batch;
            feature_batch.hashed2index_batch_idx = hashed2index_batch_idx;
            feature_batch.sample_num_signs = feature_batch
                .sample_num_signs
                .iter()
                .map(|x| x * slot_conf.hash_stack_config.hash_stack_rounds as u32)
                .collect();
        }
    }
}

#[inline]
pub fn indices_add_prefix(indices: &mut SparseBatch, config: &EmbeddingConfig) -> () {
    let feature_spacing = if config.feature_index_prefix_bit > 0 {
        (1u64 << (u64::BITS - config.feature_index_prefix_bit as u32)) - 1
    } else {
        u64::MAX
    };
    for feature_batch in indices.batches.iter_mut() {
        let slot_conf = config
            .slots_config
            .get(&feature_batch.feature_name)
            .expect("slot not found");
        if slot_conf.index_prefix > 0 {
            for single_sign in feature_batch.index_batch.iter_mut() {
                single_sign.sign %= feature_spacing;
                single_sign.sign += slot_conf.index_prefix;
            }
            let mut index_prefix_mapping: HashMap<u64, usize> =
                HashMap::with_capacity(feature_batch.hashed2index_batch_idx.len());

            feature_batch
                .hashed2index_batch_idx
                .iter()
                .for_each(|(id, batch_idx)| {
                    index_prefix_mapping
                        .insert(id % feature_spacing + slot_conf.index_prefix, *batch_idx);
                });
            feature_batch.hashed2index_batch_idx = index_prefix_mapping;
        }
    }
}

#[derive(Clone)]
pub struct SignWithConfig {
    sign: u64,
    sign_idx: usize,
    feature_idx: usize,
    dim: usize,
}

impl SignWithConfig {
    pub fn get_sign(&self) -> u64 {
        self.sign
    }
    pub fn get_dim(&self) -> usize {
        self.dim
    }
}

pub fn lookup_batched_all_slots_preprocess(
    indices: &mut SparseBatch,
    config: &EmbeddingConfig,
    replica_size: u64,
) -> Vec<Vec<SignWithConfig>> {
    #[inline]
    fn indices_to_sharded_indices(
        indices: &SparseBatch,
        config: &EmbeddingConfig,
        replica_size: u64,
    ) -> Vec<Vec<SignWithConfig>> {
        // TODO: optimization point: duplicate sign may exists in lookup result, split
        // Vec<Vec<SignWithConfig>> into Vec<Vec<id>>,
        let mut results = vec![Vec::new(); replica_size as usize];
        for (feature_idx, feature_batch) in indices.batches.iter().enumerate() {
            let slot_conf = config
                .slots_config
                .get(&feature_batch.feature_name)
                .expect("slot not found");
            for (sign_idx, single_sign) in feature_batch.index_batch.iter().enumerate() {
                let replica_index = sign_to_shard_modulo(single_sign.sign, replica_size);
                unsafe {
                    results
                        .get_unchecked_mut(replica_index as usize)
                        .push(SignWithConfig {
                            sign: single_sign.sign,
                            sign_idx,
                            feature_idx,
                            dim: slot_conf.dim,
                        });
                }
            }
        }
        results
    }

    indices_to_hashstack_indices(indices, config);
    indices_add_prefix(indices, config);
    indices_to_sharded_indices(&indices, config, replica_size)
}

pub fn lookup_batched_all_slots_postprocess<'a>(
    indices: &SparseBatch,
    forwarded_groups: Vec<(Vec<f32>, Vec<SignWithConfig>)>,
    config: &'a EmbeddingConfig,
) -> Vec<FeatureEmbeddingBatch> {
    struct LookupResultWithSlotConfig<'a> {
        result: ndarray::Array2<f32>,
        config: &'a SlotConfig,
        sign2idx: HashMap<u64, usize>,
    }

    let mut results: Vec<LookupResultWithSlotConfig<'a>> = indices
        .batches
        .iter()
        .map(|x| {
            let slot_conf = config
                .slots_config
                .get(x.feature_name.as_str())
                .expect("slot not found");
            let (feature_len, sign2idx) = if slot_conf.embedding_summation {
                (x.batch_size as usize, HashMap::new())
            } else {
                let distinct_id_size = if slot_conf.hash_stack_config.hash_stack_rounds > 0 {
                    x.index_batch.len() / slot_conf.hash_stack_config.hash_stack_rounds
                } else {
                    x.index_batch.len()
                };
                (distinct_id_size + 1, x.hashed2index_batch_idx.clone())
            };
            LookupResultWithSlotConfig {
                result: ndarray::Array2::<f32>::zeros((feature_len, slot_conf.dim)),
                config: slot_conf,
                sign2idx,
            }
        })
        .collect_vec();

    for group in forwarded_groups {
        let (embeddings, signs) = group;
        let lookup_raw_results = {
            let mut results = Vec::with_capacity(signs.len());
            let mut embeddings_slice = embeddings.as_slice();
            for sign in &signs {
                let (l, r) = embeddings_slice.split_at(sign.dim);
                embeddings_slice = r;
                results.push(l);
            }
            assert_eq!(
                embeddings_slice.len(),
                0,
                "embeddings lookup results do not match dimension"
            );
            results
        };
        for (emb, single_sign) in lookup_raw_results.iter().zip(signs) {
            let feature_idx = single_sign.feature_idx;
            let result = unsafe { results.get_unchecked_mut(feature_idx) };
            if !result.config.embedding_summation {
                let mut row = result
                    .result
                    .row_mut(result.sign2idx.get(&single_sign.sign).unwrap() + 1);
                let row = row.as_slice_mut().unwrap();
                row.clone_from_slice(emb);
            } else {
                let sign_idx = single_sign.sign_idx;
                let single_sign = unsafe {
                    indices
                        .batches
                        .get_unchecked(feature_idx)
                        .index_batch
                        .get_unchecked(sign_idx)
                };
                for (batch_id, _) in &single_sign.in_which_batch_samples {
                    let mut row = result.result.row_mut(*batch_id as usize);
                    let row = row.as_slice_mut().unwrap();
                    unsafe {
                        persia_simd::add_assign_avx2(row, emb);
                    }
                }
            }
        }
    }

    let batches = results
        .into_iter()
        .zip(indices.batches.iter())
        .map(|(mut x, indices)| {
            if x.config.embedding_summation {
                if x.config.sqrt_scaling {
                    let sample_num_ids = ndarray::Array2::from_shape_vec(
                        (indices.sample_num_signs.len(), 1),
                        indices.sample_num_signs.clone(),
                    )
                    .unwrap();
                    x.result.mul_assign(
                        &sample_num_ids.mapv(|x| (std::cmp::max(x, 1) as f32).sqrt().recip()),
                    );
                }
                FeatureEmbeddingBatch::SumEmbedding(FeatureSumEmbeddingBatch {
                    feature_name: indices.feature_name.clone(),
                    embeddings: ndarray_f32_to_f16(&x.result),
                })
            } else {
                if x.config.sqrt_scaling && x.config.hash_stack_config.hash_stack_rounds > 1 {
                    x.result.mul_assign(
                        (x.config.hash_stack_config.hash_stack_rounds as f32)
                            .sqrt()
                            .recip(),
                    )
                }
                // transform distinct_id tensor to origin batch format
                let mut index: Vec<usize> =
                    vec![0; indices.batch_size as usize * x.config.sample_fixed_size];
                let mut sample_id_num: Vec<usize> = vec![0; indices.batch_size as usize];
                let mut transform_id_set =
                    std::collections::HashSet::with_capacity(indices.index_batch.len());
                let id2idx = &x.sign2idx;

                indices.index_batch.iter().for_each(|item| {
                    let distinct_tensor_idx = id2idx.get(&item.sign).unwrap();
                    if !transform_id_set.contains(distinct_tensor_idx) {
                        transform_id_set.insert(distinct_tensor_idx);
                        for (batch_idx, col_idx) in &item.in_which_batch_samples {
                            let batch_idx = *batch_idx as usize;
                            let col_idx = *col_idx as usize;
                            if sample_id_num[batch_idx] < x.config.sample_fixed_size
                                && col_idx < x.config.sample_fixed_size
                            {
                                index[batch_idx * x.config.sample_fixed_size + col_idx] =
                                    distinct_tensor_idx + 1;
                                sample_id_num[batch_idx] += 1;
                            }
                        }
                    }
                });
                FeatureEmbeddingBatch::RawEmbedding(FeatureRawEmbeddingBatch {
                    feature_name: indices.feature_name.clone(),
                    embeddings: ndarray_f32_to_f16(&x.result),
                    index,
                    sample_id_num,
                })
            }
        })
        .collect();

    batches
}

#[repr(align(64))] // cache line optimization
pub struct MiddlewareServerInner {
    pub all_embedding_server_client: AllEmbeddingServerClient,
    pub replica_size: u64,
    pub forward_id: AtomicU64,
    pub cannot_forward_batched_time: crossbeam::atomic::AtomicCell<SystemTime>, // TODO: use parking_lot::RwLock to replace
    pub forward_id_buffer:
        persia_libs::async_lock::RwLock<HashMap<usize, HashMap<u64, SparseBatch>>>,
    pub post_forward_buffer: persia_libs::async_lock::RwLock<HashMap<u64, SparseBatch>>,
    pub staleness: AtomicUsize,
    pub embedding_config: Arc<EmbeddingConfig>,
    pub middleware_config: Arc<PersiaMiddlewareConfig>,
}

impl MiddlewareServerInner {
    fn get_id(&self) -> u64 {
        self.forward_id.fetch_add(1, Ordering::AcqRel)
    }

    async fn forward_batched(
        &self,
        indices: SparseBatch,
        batcher_idx: usize,
    ) -> Result<u64, MiddlewareServerError> {
        let id = self.get_id();

        if let Ok(m) = MetricsHolder::get() {
            for batch in indices.batches.iter() {
                let num_ids_this_batch: usize = tokio::task::block_in_place(|| {
                    batch.sample_num_signs.iter().sum::<u32>() as usize
                });
                let num_unique_ids: usize = batch.index_batch.len();
                let batch_unique_indices_rate = num_unique_ids as f32 / num_ids_this_batch as f32;

                m.batch_unique_indices_rate
                    .with_label_values(&[batch.feature_name.as_str()])
                    .set(batch_unique_indices_rate.into());

                m.num_pending_batches
                    .set(self.forward_id_buffer.read().await.len() as f64);
                m.staleness
                    .set(self.staleness.load(Ordering::Acquire) as f64);
            }
        }

        {
            let mut indices = indices;
            indices.enter_forward_id_buffer_time = Some(SystemTime::now());

            let mut forward_id_buffer = self.forward_id_buffer.write().await;
            let sub_buffer = forward_id_buffer.get_mut(&batcher_idx);
            match sub_buffer {
                Some(b) => {
                    b.insert(id, indices);
                }
                None => {
                    let mut new_sub =
                        HashMap::with_capacity(self.middleware_config.forward_buffer_size);
                    new_sub.insert(id, indices);
                    forward_id_buffer.insert(batcher_idx, new_sub);
                }
            }
        }
        Ok(id)
    }

    fn get_slot_conf(&self, slot_name: &str) -> &SlotConfig {
        self.embedding_config
            .slots_config
            .get(slot_name)
            .expect("slot not found")
    }

    pub async fn update_all_batched_gradients(
        &self,
        gradients: &EmbeddingGradientBatch,
        indices: &SparseBatch,
    ) -> Result<(), MiddlewareServerError> {
        let start_time = std::time::Instant::now();

        let indices_kv: HashMap<_, _> = indices
            .batches
            .iter()
            .map(|batch| (batch.feature_name.as_str(), batch))
            .collect();

        let mut sharded_gradients = vec![vec![]; self.all_embedding_server_client.replica_size()];
        let mut sharded_gradient_signs =
            vec![vec![]; self.all_embedding_server_client.replica_size()];

        for gradient in &gradients.gradients {
            match gradient {
                SkippableFeatureEmbeddingGradientBatch::GradientBatch(feature_gradient) => {
                    let feature_batch = indices_kv
                        .get(&feature_gradient.feature_name.as_str())
                        .unwrap();
                    let slot_conf = self.get_slot_conf(feature_batch.feature_name.as_str());
                    let raw_gradients = &feature_gradient.gradients;

                    if tokio::task::block_in_place(|| match &raw_gradients {
                        Gradients::F16(f16_gradients) => {
                            f16_gradients.as_slice().unwrap().iter().any(|x| x.is_nan())
                        }
                        Gradients::F32(f32_gradients) => {
                            f32_gradients.as_slice().unwrap().iter().any(|x| x.is_nan())
                        }
                    }) {
                        tracing::warn!("nan found in gradient update, skipping");
                        if let Ok(m) = MetricsHolder::get() {
                            m.nan_count
                                .with_label_values(&[feature_batch.feature_name.as_str()])
                                .inc();
                        }
                        continue;
                    }
                    let mut f32_gradients = tokio::task::block_in_place(|| match raw_gradients {
                        Gradients::F16(gradients_array) => ndarray_f16_to_f32(&gradients_array),
                        Gradients::F32(gradients_array) => gradients_array.clone(), // FIXME: replace with empty ndarray
                    });
                    if (feature_gradient.scale_factor - 1.0).abs() > f32::EPSILON {
                        let scale = feature_gradient.scale_factor.recip();
                        assert!(scale.is_finite(), "scale on gradient must be finite");
                        tokio::task::block_in_place(|| f32_gradients.mul_assign(scale));
                    }

                    if slot_conf.sqrt_scaling {
                        tokio::task::block_in_place(|| {
                            if slot_conf.embedding_summation {
                                let sample_num_ids = ndarray::Array2::from_shape_vec(
                                    (feature_batch.sample_num_signs.len(), 1),
                                    feature_batch.sample_num_signs.clone(),
                                )
                                .unwrap();
                                f32_gradients.mul_assign(
                                    &sample_num_ids.mapv(|x| (x as f32).sqrt().recip()),
                                );
                            } else {
                                if slot_conf.hash_stack_config.hash_stack_rounds > 0 {
                                    f32_gradients.mul_assign(
                                        (slot_conf.hash_stack_config.hash_stack_rounds as f32)
                                            .sqrt()
                                            .recip(),
                                    );
                                }
                            }
                        });
                    }

                    tokio::task::block_in_place(|| {
                        let mut sign_gradients = ndarray::Array2::<f32>::zeros((
                            feature_batch.index_batch.len(),
                            slot_conf.dim,
                        ));
                        let hashed2index_batch_idx = &feature_batch.hashed2index_batch_idx;
                        for (row, single_sign) in feature_batch.index_batch.iter().enumerate() {
                            let mut sign_grad = sign_gradients.row_mut(row);
                            let sign_grad = sign_grad.as_slice_mut().unwrap();

                            if !slot_conf.embedding_summation {
                                let batch_idx =
                                    hashed2index_batch_idx.get(&single_sign.sign).unwrap();
                                unsafe {
                                    persia_simd::add_assign_avx2(
                                        sign_grad,
                                        f32_gradients.row(*batch_idx as usize).as_slice().unwrap(),
                                    );
                                }
                            } else {
                                single_sign.in_which_batch_samples.iter().for_each(
                                    |(batch_id, _)| {
                                        let row_grad = f32_gradients.row(*batch_id as usize);
                                        unsafe {
                                            persia_simd::add_assign_avx2(
                                                sign_grad,
                                                row_grad.as_slice().unwrap(),
                                            );
                                        }
                                    },
                                );
                            }
                        }
                        for (grad, sign) in sign_gradients
                            .axis_iter(ndarray::Axis(0))
                            .zip(feature_batch.index_batch.iter())
                        {
                            let replica_index = sign_to_shard_modulo(sign.sign, self.replica_size);
                            sharded_gradients[replica_index as usize]
                                .extend_from_slice(grad.as_slice().unwrap());
                            sharded_gradient_signs[replica_index as usize].push(sign.sign);
                        }
                    });
                }
                SkippableFeatureEmbeddingGradientBatch::Skipped(skipped) => {
                    if let Ok(m) = MetricsHolder::get() {
                        m.nan_grad_skipped
                            .with_label_values(&[skipped.feature_name.as_str()])
                            .inc();
                    }
                    continue;
                }
            }
        }

        let futs = sharded_gradients
            .into_iter()
            .zip(sharded_gradient_signs)
            .enumerate()
            .map(|(replica_index, (grads, signs))| {
                let client = block_on(
                    self.all_embedding_server_client
                        .get_client_by_index(replica_index),
                );
                async move {
                    let start_time = Instant::now();
                    client
                        .update_gradient_mixed(&(signs, grads))
                        .await
                        .map_err(|e| MiddlewareServerError::RpcError(format!("{:?}", e)))??;
                    let result = Ok::<_, MiddlewareServerError>(());
                    tracing::debug!(
                        "update gradient middleware time cost {:?}",
                        start_time.elapsed()
                    );
                    result
                }
            });

        let _updated_gradient_groups: Vec<_> = futures::future::try_join_all(futs).await?;

        tracing::debug!(
            "update gradients all slots time cost {:?}",
            start_time.elapsed()
        );

        if let Ok(m) = MetricsHolder::get() {
            m.update_gradient_time_cost
                .observe(start_time.elapsed().as_secs_f64());
        }

        Ok(())
    }

    pub async fn lookup_batched_all_slots(
        &self,
        indices: &mut SparseBatch,
        requires_grad: bool,
    ) -> Result<Vec<FeatureEmbeddingBatch>, MiddlewareServerError> {
        let start_time_all = std::time::Instant::now();
        let start_time = std::time::Instant::now();

        let all_shards_ids = tokio::task::block_in_place(|| {
            lookup_batched_all_slots_preprocess(indices, &self.embedding_config, self.replica_size)
        });

        let futs = all_shards_ids
            .into_iter()
            .enumerate()
            .map(|(replica_index, shard_indices)| {
                let req = tokio::task::block_in_place(|| {
                    (
                        shard_indices.iter().map(|x| (x.sign, x.dim)).collect(),
                        requires_grad,
                    )
                });
                let client = block_on(
                    self.all_embedding_server_client
                        .get_client_by_index(replica_index),
                );
                async move {
                    let lookup_results: Vec<f32> = client
                        .lookup_mixed(&req)
                        .await
                        .map_err(|e| MiddlewareServerError::RpcError(format!("{:?}", e)))??;
                    Ok::<_, MiddlewareServerError>((lookup_results, shard_indices))
                }
            });

        tracing::debug!(
            "create sharded requests time cost {:?}",
            start_time.elapsed()
        );
        if let Ok(m) = MetricsHolder::get() {
            m.lookup_create_requests_time_cost
                .observe(start_time.elapsed().as_secs_f64());
        }
        let start_time = std::time::Instant::now();

        let forwarded_groups: Vec<_> = futures::future::try_join_all(futs).await?;

        tracing::debug!("rpc time cost {:?}", start_time.elapsed());
        if let Ok(m) = MetricsHolder::get() {
            m.lookup_rpc_time_cost
                .observe(start_time.elapsed().as_secs_f64());
        }

        let start_time = std::time::Instant::now();

        let batches = tokio::task::block_in_place(|| {
            lookup_batched_all_slots_postprocess(indices, forwarded_groups, &self.embedding_config)
        });

        tracing::debug!("summation time cost {:?}", start_time.elapsed());
        if let Ok(m) = MetricsHolder::get() {
            m.summation_time_cost
                .observe(start_time.elapsed().as_secs_f64());
            m.lookup_batched_time_cost
                .observe(start_time_all.elapsed().as_secs_f64());
        }

        return Ok(batches);
    }

    pub async fn ready_for_serving(&self) -> bool {
        let result = self.all_embedding_server_client.ready_for_serving().await;
        tracing::info!("middleware server ready for serving: {}", result);
        result
    }

    pub async fn model_manager_status(&self) -> Vec<PersiaPersistenceStatus> {
        let result = self
            .all_embedding_server_client
            .model_manager_status()
            .await;
        tracing::info!("embedding server dumping model: {:?}", result);
        result
    }

    pub async fn set_embedding(
        &self,
        req: Vec<(u64, HashMapEmbeddingEntry)>,
    ) -> Result<(), MiddlewareServerError> {
        let replica_size = self.replica_size;
        let futs: Vec<_> = tokio::task::block_in_place(|| {
            let grouped_entries = req
                .into_iter()
                .sorted_by_key(|(k, _)| sign_to_shard_modulo(*k, replica_size))
                .group_by(|(k, _)| sign_to_shard_modulo(*k, replica_size));

            grouped_entries
                .into_iter()
                .map(|(replica_index, requests)| {
                    let group = requests.into_iter().collect_vec();
                    let client = block_on(
                        self.all_embedding_server_client
                            .get_client_by_index(replica_index as usize),
                    );
                    async move {
                        client
                            .set_embedding(&group)
                            .await
                            .map_err(|e| MiddlewareServerError::RpcError(format!("{:?}", e)))??;
                        Ok::<_, MiddlewareServerError>(())
                    }
                })
                .collect()
        });
        futures::future::try_join_all(futs).await.map(|_| ())
    }

    pub async fn can_forward_batched(&self, batcher_idx: usize) -> bool {
        let result = match self.forward_id_buffer.read().await.get(&batcher_idx) {
            Some(buffer) => buffer.len() < self.middleware_config.forward_buffer_size,
            None => true,
        };
        let t = self.cannot_forward_batched_time.load();
        if !result {
            let current_time = SystemTime::now();
            if current_time.duration_since(t).unwrap() > Duration::from_secs(60) {
                let mut forward_id_buffer = self.forward_id_buffer.write().await;
                let sub_buffer = forward_id_buffer.get_mut(&batcher_idx).unwrap();
                tokio::task::block_in_place(|| {
                    let old_keys = sub_buffer
                        .iter()
                        .filter_map(|(k, v)| {
                            if current_time
                                .duration_since(v.enter_forward_id_buffer_time.unwrap())
                                .unwrap()
                                > Duration::from_secs(
                                    self.middleware_config.buffered_data_expired_sec as u64,
                                )
                            {
                                Some(*k)
                            } else {
                                None
                            }
                        })
                        .collect_vec();
                    for k in old_keys {
                        sub_buffer.remove(&k);
                    }
                    self.cannot_forward_batched_time.store(SystemTime::now());
                });
            }
        } else {
            self.cannot_forward_batched_time.store(SystemTime::now());
        }
        result
    }

    pub async fn forward_batch_id(
        &self,
        req: (SparseBatchRemoteReference, bool),
    ) -> Result<EmbeddingBatch, MiddlewareServerError> {
        let (sparse_ref, requires_grad) = req;
        let ref_id = sparse_ref.ref_id;

        let inner = self.clone();
        let mut indices = {
            let mut forward_id_buffer = inner.forward_id_buffer.write().await;
            let sub_buffer = forward_id_buffer
                .get_mut(&sparse_ref.batcher_idx)
                .ok_or_else(|| MiddlewareServerError::ForwardIdNotFound(ref_id))?;
            sub_buffer
                .remove(&ref_id)
                .ok_or_else(|| MiddlewareServerError::ForwardIdNotFound(ref_id))?
        };

        tracing::debug!("received forward_batch_id request");
        self.staleness.fetch_add(1, Ordering::AcqRel);
        let result = inner
            .lookup_batched_all_slots(&mut indices, requires_grad)
            .await;

        if result.is_err() {
            self.staleness.fetch_sub(1, Ordering::AcqRel);
        }
        let result = result?;

        if requires_grad {
            indices.enter_post_forward_buffer_time = Some(SystemTime::now());
            inner
                .post_forward_buffer
                .write()
                .await
                .insert(ref_id, indices);
            tracing::debug!("indices inserted into post forward buffer");
        }

        return Ok(EmbeddingBatch {
            batches: result,
            backward_ref_id: Some(ref_id),
        });
    }

    pub async fn forward_batched_direct(
        &self,
        indices: SparseBatch,
    ) -> Result<EmbeddingBatch, MiddlewareServerError> {
        let mut indices = indices;

        let requires_grad = indices.requires_grad.clone();
        let result = self
            .lookup_batched_all_slots(&mut indices, requires_grad)
            .await?;

        let backward_ref_id = if requires_grad {
            let backward_ref_id = self.get_id();
            let inner = self.clone();

            indices.enter_post_forward_buffer_time = Some(SystemTime::now());
            inner
                .post_forward_buffer
                .write()
                .await
                .insert(backward_ref_id, indices);
            tracing::debug!("indices inserted into post forward buffer");
            Some(backward_ref_id)
        } else {
            None
        };

        Ok(EmbeddingBatch {
            batches: result,
            backward_ref_id,
        })
    }

    pub async fn update_gradient_batched(
        &self,
        req: (u64, EmbeddingGradientBatch),
    ) -> Result<(), MiddlewareServerError> {
        let (backward_ref_id, gradients) = req;
        let indices = self
            .post_forward_buffer
            .write()
            .await
            .remove(&backward_ref_id)
            .ok_or_else(|| MiddlewareServerError::ForwardIdNotFound(backward_ref_id))?;

        let inner = self.clone();
        inner
            .update_all_batched_gradients(&gradients, &indices)
            .await?;

        self.staleness.fetch_sub(1, Ordering::AcqRel);

        Ok(())
    }

    pub async fn dump(&self, req: String) -> Result<(), MiddlewareServerError> {
        let inner = self.clone();
        let futs = (0..inner.all_embedding_server_client.replica_size()).map(|client_idx| {
            let req = req.clone();
            async move {
                let client = inner
                    .all_embedding_server_client
                    .get_client_by_index(client_idx)
                    .await;
                client
                    .dump(&req)
                    .await
                    .map_err(|e| MiddlewareServerError::RpcError(e.to_string()))??;
                Ok(())
            }
        });
        futures::future::try_join_all(futs).await.map(|_| ())
    }

    pub async fn load(&self, req: String) -> Result<(), MiddlewareServerError> {
        let inner = self.clone();
        let futs = (0..inner.all_embedding_server_client.replica_size()).map(|client_idx| {
            let req = req.clone();
            async move {
                let client = inner
                    .all_embedding_server_client
                    .get_client_by_index(client_idx)
                    .await;
                client
                    .load(&req)
                    .await
                    .map_err(|e| MiddlewareServerError::RpcError(e.to_string()))??;
                Ok(())
            }
        });
        let result = futures::future::try_join_all(futs).await.map(|_| ());
        result
    }

    pub async fn configure_embedding_servers(
        &self,
        req: PersiaSparseModelHyperparameters,
    ) -> Result<(), MiddlewareServerError> {
        let inner = self.clone();
        let req = req;
        let futs = (0..inner.all_embedding_server_client.replica_size()).map(|client_idx| {
            let req = req.clone();
            async move {
                let client = inner
                    .all_embedding_server_client
                    .get_client_by_index(client_idx)
                    .await;
                client
                    .configure(&req)
                    .await
                    .map_err(|e| MiddlewareServerError::RpcError(e.to_string()))??;
                Ok(())
            }
        });
        let result = futures::future::try_join_all(futs).await.map(|_| ());
        tracing::info!("embedding servers configured: {:?}", result);
        result
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), MiddlewareServerError> {
        let inner = self.clone();
        let futs = (0..inner.all_embedding_server_client.replica_size()).map(|client_idx| {
            let optimizer = optimizer.clone();
            async move {
                let client = inner
                    .all_embedding_server_client
                    .get_client_by_index(client_idx)
                    .await;
                client
                    .register_optimizer(&optimizer)
                    .await
                    .map_err(|e| MiddlewareServerError::RpcError(e.to_string()))??;
                Ok(())
            }
        });
        tracing::info!("register optimizer: {:?}", &optimizer);
        futures::future::try_join_all(futs).await.map(|_| ())
    }

    pub async fn get_address(&self) -> Result<String, MiddlewareServerError> {
        let instance_info = InstanceInfo::get()?;
        let address = format!("{}:{}", instance_info.ip_address, instance_info.port);
        Ok(address)
    }

    pub async fn get_replica_size(&self) -> Result<usize, MiddlewareServerError> {
        let repilca_info = PersiaReplicaInfo::get()?;
        Ok(repilca_info.replica_size)
    }

    pub async fn error_handle(
        &self,
        err: &MiddlewareServerError,
    ) -> Result<(), MiddlewareServerError> {
        match err {
            MiddlewareServerError::RpcError(_) => {
                let servers = self.all_embedding_server_client.get_all_addresses()?;
                self.all_embedding_server_client.update_rpc_clients(servers)
            }
            _ => Ok(()),
        }
    }

    pub async fn get_embedding_size(&self) -> Result<Vec<usize>, MiddlewareServerError> {
        let clients = self.all_embedding_server_client.clients.read().await;
        let futs = clients.iter().map(|client| {
            let client = client.clone();
            async move {
                let result = client
                    .get_embedding_size(&())
                    .await
                    .map_err(|e| MiddlewareServerError::RpcError(format!("{:?}", e)))??;
                Ok(result)
            }
        });

        let result = futures::future::try_join_all(futs).await;
        result
    }

    pub async fn clear_embeddings(&self) -> Result<(), MiddlewareServerError> {
        let clients = self.all_embedding_server_client.clients.read().await;
        let futs = clients.iter().map(|client| {
            let client = client.clone();
            async move {
                client
                    .clear_embeddings(&())
                    .await
                    .map_err(|e| MiddlewareServerError::RpcError(format!("{:?}", e)))??;
                Ok(())
            }
        });
        futures::future::try_join_all(futs).await.map(|_| ())
    }
}

#[derive(Clone)]
pub struct MiddlewareServer {
    pub inner: Arc<MiddlewareServerInner>,
    pub shutdown_channel:
        Arc<persia_libs::async_lock::RwLock<Option<tokio::sync::oneshot::Sender<()>>>>,
}

#[persia_rpc_macro::service]
impl MiddlewareServer {
    pub async fn ready_for_serving(&self, _req: ()) -> bool {
        self.inner.ready_for_serving().await
    }

    pub async fn model_manager_status(&self, _req: ()) -> Vec<PersiaPersistenceStatus> {
        self.inner.model_manager_status().await
    }

    pub async fn set_embedding(
        &self,
        req: Vec<(u64, HashMapEmbeddingEntry)>,
    ) -> Result<(), MiddlewareServerError> {
        self.inner.set_embedding(req).await
    }

    pub async fn get_embedding_size(&self, _req: ()) -> Result<Vec<usize>, MiddlewareServerError> {
        self.inner.get_embedding_size().await
    }

    pub async fn clear_embeddings(&self, _req: ()) -> Result<(), MiddlewareServerError> {
        self.inner.clear_embeddings().await
    }

    pub async fn shutdown_server(&self, _req: ()) -> Result<(), EmbeddingServerError> {
        let clients = self.inner.all_embedding_server_client.clients.read().await;

        let futs = clients
            .iter()
            .map(|client| async move { client.shutdown(&()).await });

        let result = futures::future::try_join_all(futs).await;

        if result.is_ok() {
            Ok(())
        } else {
            Err(EmbeddingServerError::ShutdownError)
        }
    }

    pub async fn shutdown(&self, _req: ()) -> Result<(), MiddlewareServerError> {
        let mut shutdown_channel = self.shutdown_channel.write().await;
        let shutdown_channel = shutdown_channel.take();

        match shutdown_channel {
            Some(sender) => {
                sender.send(()).unwrap();
                Ok(())
            }
            None => {
                tracing::debug!("shutdown channel already been taken, wait server shutdown.");
                Ok(())
            }
        }
    }

    pub async fn forward_batch_id(
        &self,
        req: (SparseBatchRemoteReference, bool),
    ) -> Result<EmbeddingBatch, MiddlewareServerError> {
        let resp = self.inner.forward_batch_id(req).await;
        if resp.is_err() {
            self.inner.error_handle(resp.as_ref().unwrap_err()).await?
        }
        resp
    }

    pub async fn forward_batched_direct(
        &self,
        indices: SparseBatch,
    ) -> Result<EmbeddingBatch, MiddlewareServerError> {
        self.inner.forward_batched_direct(indices).await
    }

    pub async fn update_gradient_batched(
        &self,
        req: (u64, EmbeddingGradientBatch),
    ) -> Result<(), MiddlewareServerError> {
        let resp = self.inner.update_gradient_batched(req).await;
        if resp.is_err() {
            self.inner.error_handle(resp.as_ref().unwrap_err()).await?
        }
        resp
    }

    pub async fn dump(&self, req: String) -> Result<(), MiddlewareServerError> {
        self.inner.dump(req).await
    }

    pub async fn load(&self, req: String) -> Result<(), MiddlewareServerError> {
        self.inner.load(req).await
    }

    pub async fn configure_embedding_servers(
        &self,
        req: PersiaSparseModelHyperparameters,
    ) -> Result<(), MiddlewareServerError> {
        self.inner.configure_embedding_servers(req).await
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), MiddlewareServerError> {
        self.inner.register_optimizer(optimizer).await
    }
}

#[derive(Clone)]
pub struct MiddlewareNatsService {
    pub inner: Arc<MiddlewareServerInner>,
}

#[persia_nats_marcos::service]
impl MiddlewareNatsService {
    pub async fn ready_for_serving(&self, _req: ()) -> bool {
        self.inner.ready_for_serving().await
    }

    pub async fn model_manager_status(&self, _req: ()) -> Vec<PersiaPersistenceStatus> {
        self.inner.model_manager_status().await
    }

    pub async fn can_forward_batched(&self, batcher_idx: usize) -> bool {
        self.inner.can_forward_batched(batcher_idx).await
    }

    pub async fn forward_batched(
        &self,
        indices: SparseBatch,
    ) -> Result<SparseBatchRemoteReference, MiddlewareServerError> {
        let batcher_idx = indices
            .batcher_idx
            .ok_or_else(|| MiddlewareServerError::DataSrcIdxNotSet)?;
        if !self.inner.can_forward_batched(batcher_idx).await {
            return Err(MiddlewareServerError::ForwardBufferFull);
        }
        let ref_id = self.inner.forward_batched(indices, batcher_idx).await?;
        let middleware_addr = self.inner.get_address().await?;
        let sparse_ref = SparseBatchRemoteReference {
            middleware_addr,
            ref_id,
            batcher_idx,
        };
        Ok(sparse_ref)
    }

    pub async fn dump(&self, req: String) -> Result<(), MiddlewareServerError> {
        self.inner.dump(req).await
    }

    pub async fn load(&self, req: String) -> Result<(), MiddlewareServerError> {
        self.inner.load(req).await
    }

    pub async fn configure_embedding_servers(
        &self,
        req: PersiaSparseModelHyperparameters,
    ) -> Result<(), MiddlewareServerError> {
        self.inner.configure_embedding_servers(req).await
    }

    pub async fn register_optimizer(
        &self,
        optimizer: OptimizerConfig,
    ) -> Result<(), MiddlewareServerError> {
        self.inner.register_optimizer(optimizer).await
    }

    pub async fn get_address(&self, _req: ()) -> Result<String, MiddlewareServerError> {
        self.inner.get_address().await
    }

    pub async fn get_replica_size(&self, _req: ()) -> Result<usize, MiddlewareServerError> {
        self.inner.get_replica_size().await
    }
}

#[cfg(test)]
mod lookup_batched_all_slots_preprocess_tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use persia_common::FeatureBatch;
    use persia_libs::serde_yaml;

    #[test]
    fn test_indices_to_hashstack_indices() {
        let config = "feature_index_prefix_bit: 12\nslots_config:\n  Test:\n    dim: 32\n    hash_stack_config:\n      hash_stack_rounds: 2\n      embedding_size: 10\nfeature_groups: {}\n";

        let config: EmbeddingConfig = serde_yaml::from_str(config).expect("failed to parse config");

        let raw_batch: Vec<Vec<u64>> = vec![vec![12, 23, 34], vec![56, 78, 90], vec![12, 56]];
        let feature_name = "Test".to_string();
        let feature_batch = FeatureBatch::new(feature_name.clone(), raw_batch);
        let mut sparse_batch = SparseBatch {
            batches: vec![feature_batch],
            enter_forward_id_buffer_time: None,
            enter_post_forward_buffer_time: None,
            batcher_idx: None,
        };
        indices_to_hashstack_indices(&mut sparse_batch, &config);
        let hashstack_feature_batch = sparse_batch.batches.first().unwrap();

        let target_raw_batch: Vec<Vec<u64>> = vec![
            vec![2, 18, 5, 10, 0, 11],
            vec![6, 17, 7, 12, 8, 16],
            vec![2, 18, 6, 17],
        ];
        let target_feature_batch = FeatureBatch::new(feature_name, target_raw_batch);

        for single_sign in hashstack_feature_batch.index_batch.iter() {
            for target_single_sign in target_feature_batch.index_batch.iter() {
                if single_sign.sign == target_single_sign.sign {
                    let mut result = single_sign.in_which_batch_samples.clone();
                    let mut target = target_single_sign.in_which_batch_samples.clone();
                    assert_eq!(result.len(), target.len());
                    result.sort();
                    target.sort();
                    let matching = result
                        .iter()
                        .zip(&target)
                        .filter(|&((r, _), (t, _))| r == t)
                        .count();
                    assert_eq!(matching, target.len());
                }
            }
        }
    }

    #[test]
    fn test_indices_add_prefix() {
        let config = "feature_index_prefix_bit: 12\nslots_config:\n  feature1:\n    dim: 64\n    index_prefix: 450359962737049600\n";

        let config: EmbeddingConfig = serde_yaml::from_str(config).expect("failed to parse config");

        let mut raw_batch: Vec<Vec<u64>> = vec![
            vec![12, 23, 34],
            vec![56, 78, 90],
            vec![16000000000000000, 56],
        ];
        let feature_name = "feature1".to_string();
        let feature_batch = FeatureBatch::new(feature_name.clone(), raw_batch.clone());
        let mut sparse_batch = SparseBatch {
            batches: vec![feature_batch],
            enter_forward_id_buffer_time: None,
            enter_post_forward_buffer_time: None,
            batcher_idx: None,
        };
        indices_add_prefix(&mut sparse_batch, &config);
        let result_feature_batch = sparse_batch.batches.first().unwrap();

        result_feature_batch.index_batch.iter().for_each(|x| {
            x.in_which_batch_samples
                .iter()
                .for_each(|(batch_idx, col_idx)| {
                    raw_batch[*batch_idx as usize][*col_idx as usize] = x.sign;
                })
        });

        let target_raw_batch: Vec<Vec<u64>> = vec![
            vec![450359962737049612, 450359962737049623, 450359962737049634],
            vec![450359962737049656, 450359962737049678, 450359962737049690],
            vec![452849163854938115, 450359962737049656],
        ];

        raw_batch
            .iter()
            .zip(target_raw_batch.iter())
            .for_each(|(result, target)| {
                result.iter().zip(target.iter()).for_each(|(r, t)| {
                    assert_eq!(r, t);
                })
            });
    }
}
