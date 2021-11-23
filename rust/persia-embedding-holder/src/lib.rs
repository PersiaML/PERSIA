pub mod array_linked_list;
pub mod emb_entry;
pub mod eviction_map;
pub mod sharded;

use std::sync::Arc;

use persia_libs::{once_cell, parking_lot::RwLock, thiserror};

use emb_entry::HashMapEmbeddingEntry;
use eviction_map::EvictionMap;
use persia_embedding_config::{EmbeddingParameterServerConfig, PersiaGlobalConfigError};
use persia_speedy::{Readable, Writable};
use sharded::Sharded;

#[derive(Clone, Readable, Writable, thiserror::Error, Debug)]
pub enum PersiaEmbeddingHolderError {
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("id not fonud")]
    IdNotFound,
}

static PERSIA_EMBEDDING_HOLDER: once_cell::sync::OnceCell<PersiaEmbeddingHolder> =
    once_cell::sync::OnceCell::new();

#[derive(Clone)]
pub struct PersiaEmbeddingHolder {
    inner: Arc<Sharded<EvictionMap<u64, HashMapEmbeddingEntry>, u64>>,
}

impl PersiaEmbeddingHolder {
    pub fn get() -> Result<PersiaEmbeddingHolder, PersiaEmbeddingHolderError> {
        let singleton = PERSIA_EMBEDDING_HOLDER.get_or_try_init(|| {
            let config = EmbeddingParameterServerConfig::get()?;

            let bucket_size = config.num_hashmap_internal_shards;
            let cpapacity_per_bucket = config.capacity / bucket_size;

            let handles: Vec<std::thread::JoinHandle<_>> = (0..bucket_size)
                .map(|_| {
                    std::thread::spawn(move || {
                        EvictionMap::with_capacity(cpapacity_per_bucket as usize)
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
            Ok(PersiaEmbeddingHolder {
                inner: Arc::new(sharded),
            })
        });
        match singleton {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(e),
        }
    }

    pub fn num_total_signs(&self) -> usize {
        self.inner
            .inner
            .iter()
            .map(|x| x.read().len())
            .sum::<usize>()
    }

    pub fn num_internal_shards(&self) -> usize {
        self.inner.inner.len()
    }

    pub fn capacity(&self) -> usize {
        self.inner
            .inner
            .iter()
            .map(|x| x.read().capacity())
            .sum::<usize>()
    }

    pub fn clear(&self) {
        self.inner.inner.iter().for_each(|x| x.write().clear());
    }

    pub fn shard(&self, key: &u64) -> &RwLock<EvictionMap<u64, HashMapEmbeddingEntry>> {
        self.inner.shard(key)
    }

    pub fn get_shard_by_index(
        &self,
        index: usize,
    ) -> &RwLock<EvictionMap<u64, HashMapEmbeddingEntry>> {
        self.inner.get_shard_by_index(index)
    }
}
