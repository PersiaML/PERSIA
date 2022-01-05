pub mod array_linked_list;
pub mod emb_entry;
pub mod eviction_map;
pub mod sharded;

use eviction_map::EvictionMap;
use persia_common::optim::Optimizable;
use persia_embedding_config::{
    EmbeddinHyperparameters, EmbeddingConfig, EmbeddingParameterServerConfig,
    PersiaGlobalConfigError, PersiaReplicaInfo,
};
use persia_libs::{once_cell, parking_lot::RwLock, thiserror};
use persia_speedy::{Readable, Writable};
use sharded::Sharded;
use std::sync::Arc;

#[derive(Clone, Readable, Writable, thiserror::Error, Debug)]
pub enum EmbeddingShardedMapError {
    #[error("set embedding map mutiple times error")]
    EmbeddingShardedMapMutiSetError,
    #[error("embedding map not set error")]
    NotSetError,
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("id not fonud")]
    IdNotFound,
}

static EMBEDDING_SHARDED_MAP: once_cell::sync::OnceCell<EmbeddingShardedMap> =
    once_cell::sync::OnceCell::new();

#[derive(Clone)]
pub struct EmbeddingShardedMap {
    inner: Arc<Sharded<EvictionMap, u64>>,
}

impl EmbeddingShardedMap {
    pub fn empty(num_shards: usize) {
        let shard_inner = (0..num_shards).map(|_| {
            RwLock::new(EvictionMap::default())
        }).collect();
        let empty = Arc::new(Sharded {
            inner: shard_inner,
            phantom: std::marker::PhantomData::default(),
        });
        let _ = EMBEDDING_SHARDED_MAP.set(EmbeddingShardedMap {inner: empty});
    }

    pub fn set(
        optimizer: Arc<Box<dyn Optimizable + Send + Sync>>,
        hyperparameters: EmbeddinHyperparameters,
    ) -> Result<(), EmbeddingShardedMapError> {
        let map = EmbeddingShardedMap::new(optimizer, hyperparameters)?;
        let _ = EMBEDDING_SHARDED_MAP.set(map);
        Ok(())
    }

    pub fn get() -> Result<Self, EmbeddingShardedMapError> {
        EMBEDDING_SHARDED_MAP
            .get()
            .cloned()
            .ok_or(EmbeddingShardedMapError::NotSetError)
    }

    fn new(
        optimizer: Arc<Box<dyn Optimizable + Send + Sync>>,
        hyperparameters: EmbeddinHyperparameters,
    ) -> Result<Self, EmbeddingShardedMapError> {
        let embedding_parameter_server_config = EmbeddingParameterServerConfig::get()?;
        let embedding_config = EmbeddingConfig::get()?;
        let replica_info = PersiaReplicaInfo::get()?;

        let bucket_size = embedding_parameter_server_config.num_hashmap_internal_shards;
        let handles: Vec<std::thread::JoinHandle<_>> = (0..bucket_size)
            .map(|shard_idx| {
                let embedding_config = embedding_config.clone();
                let embedding_parameter_server_config = embedding_parameter_server_config.clone();
                let hyperparameters = hyperparameters.clone();
                let replica_info = replica_info.clone();
                let shard_idx = shard_idx.clone();
                let optimizer = optimizer.clone();

                std::thread::spawn(move || {
                    EvictionMap::new(
                        embedding_config.as_ref(),
                        embedding_parameter_server_config.as_ref(),
                        replica_info.as_ref(),
                        optimizer,
                        hyperparameters,
                        shard_idx,
                    )
                })
            })
            .collect();

        let maps: Vec<_> = handles
            .into_iter()
            .map(|h| RwLock::new(h.join().expect("failed to create map")))
            .collect();

        let sharded = Sharded {
            inner: maps,
            phantom: std::marker::PhantomData::default(),
        };

        Ok(Self {
            inner: Arc::new(sharded),
        })
    }

    pub fn num_total_signs(&self) -> usize {
        self.inner
            .inner
            .iter()
            .map(|x| x.read().len())
            .sum::<usize>()
    }

    pub fn clear(&self) {
        self.inner.inner.iter().for_each(|x| x.write().clear());
    }

    pub fn shard(&self, key: &u64) -> &RwLock<EvictionMap> {
        self.inner.shard(key)
    }

    pub fn get_shard_by_index(&self, index: usize) -> &RwLock<EvictionMap> {
        self.inner.get_shard_by_index(index)
    }

    pub fn num_internal_shards(&self) -> usize {
        self.inner.inner.len()
    }
}
