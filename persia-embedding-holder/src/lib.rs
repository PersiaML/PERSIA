use persia_eviction_map::PersiaEvictionMap;
use persia_embedding_datatypes::HashMapEmbeddingEntry;
use persia_embedding_config::{PersiaGlobalConfig, PersiaGlobalConfigError};

use std::sync::Arc;
use thiserror::Error;
use parking_lot::RwLock;

#[derive(Error, Debug)]
pub enum PersiaEmbeddingHolderError {
    #[error("global config error")]
    GlobalConfigError(PersiaGlobalConfigError),
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
            if let Ok(global_config) = PersiaGlobalConfig::get() {
                let guard = global_config.read();
                let eviction_map: PersiaEvictionMap<u64, Arc<RwLock<HashMapEmbeddingEntry>>> = PersiaEvictionMap::new(
                    guard.sharded_server_config.capacity,
                    guard.sharded_server_config.num_hashmap_internal_shards,
                );
                Ok(PersiaEmbeddingHolder { inner: Arc::new(eviction_map) })
            }
            else {
                Err(PersiaEmbeddingHolderError::GlobalConfigError(PersiaGlobalConfigError::NotReadyError))
            }

        });
        match singleton {
            Ok(s) => {
                Ok(s.clone())
            }
            Err(e) => {
                Err(e)
            }
        }
    }

    pub fn num_total_signs(&self) -> usize {
        self.inner.len()
    }
}