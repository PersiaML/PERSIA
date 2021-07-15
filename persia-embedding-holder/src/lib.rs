use persia_embedding_config::{PersiaGlobalConfigError, PersiaShardedServerConfig};
use persia_embedding_datatypes::HashMapEmbeddingEntry;
use persia_eviction_map::PersiaEvictionMap;
use persia_speedy::{Readable, Writable};

use parking_lot::RwLock;
use std::sync::Arc;
use thiserror::Error;

#[derive(Readable, Writable, Error, Debug)]
pub enum PersiaEmbeddingHolderError {
    #[error("global config error: {0}")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
}

static PERSIA_EMBEDDING_HOLDER: once_cell::sync::OnceCell<PersiaEmbeddingHolder> =
    once_cell::sync::OnceCell::new();

#[derive(Clone)]
pub struct PersiaEmbeddingHolder {
    pub inner: Arc<PersiaEvictionMap<u64, Arc<RwLock<HashMapEmbeddingEntry>>>>,
}

impl PersiaEmbeddingHolder {
    pub fn get() -> Result<PersiaEmbeddingHolder, PersiaEmbeddingHolderError> {
        let singleton = PERSIA_EMBEDDING_HOLDER.get_or_try_init(|| {
            let config = PersiaShardedServerConfig::get()?;
            let eviction_map: PersiaEvictionMap<u64, Arc<RwLock<HashMapEmbeddingEntry>>> =
                PersiaEvictionMap::new(config.capacity, config.num_hashmap_internal_shards);
            Ok(PersiaEmbeddingHolder {
                inner: Arc::new(eviction_map),
            })
        });
        match singleton {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(e),
        }
    }

    pub fn num_total_signs(&self) -> usize {
        self.inner.len()
    }
}
