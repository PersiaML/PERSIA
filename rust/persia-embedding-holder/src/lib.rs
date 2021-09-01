use std::collections::LinkedList;
use std::{sync::Arc, sync::Weak};

use persia_libs::{
    hashbrown::HashMap,
    once_cell,
    parking_lot::{Mutex, RwLock},
    thiserror,
};

use persia_common::HashMapEmbeddingEntry;
use persia_embedding_config::{PersiaEmbeddingServerConfig, PersiaGlobalConfigError};
use persia_eviction_map::PersiaEvictionMap;
use persia_speedy::{Readable, Writable};

#[derive(Readable, Writable, thiserror::Error, Debug)]
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
    inner: Arc<PersiaEvictionMap<u64, Arc<RwLock<HashMapEmbeddingEntry>>>>,
    _recycle_pool: Arc<RecyclePool>,
}

impl PersiaEmbeddingHolder {
    pub fn get() -> Result<PersiaEmbeddingHolder, PersiaEmbeddingHolderError> {
        let singleton = PERSIA_EMBEDDING_HOLDER.get_or_try_init(|| {
            let config = PersiaEmbeddingServerConfig::get()?;
            let eviction_map: PersiaEvictionMap<u64, Arc<RwLock<HashMapEmbeddingEntry>>> =
                PersiaEvictionMap::new(config.capacity, config.num_hashmap_internal_shards);
            Ok(PersiaEmbeddingHolder {
                inner: Arc::new(eviction_map),
                _recycle_pool: Arc::new(RecyclePool::new(config.embedding_recycle_pool_capacity)),
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

    pub fn insert(
        &self,
        key: u64,
        value: Arc<RwLock<HashMapEmbeddingEntry>>,
    ) -> Option<Weak<RwLock<HashMapEmbeddingEntry>>> {
        let (_old_val, evcited) = self.inner.insert(key, value);
        if let Some(entry) = evcited {
            let evcited = Arc::downgrade(&entry);
            Some(evcited)
        } else {
            None
        }
    }

    pub fn get_value(&self, key: &u64) -> Option<Weak<RwLock<HashMapEmbeddingEntry>>> {
        match self.inner.get(key) {
            Some(value) => Some(Arc::downgrade(&value)),
            None => None,
        }
    }

    pub fn get_value_refresh(&self, key: &u64) -> Option<Weak<RwLock<HashMapEmbeddingEntry>>> {
        match self.inner.get_refresh(key) {
            Some(value) => Some(Arc::downgrade(&value)),
            None => None,
        }
    }

    pub fn clear(&self) {
        self.inner.clear();
    }
}

pub struct RecyclePool {
    pub inner: Arc<RwLock<HashMap<usize, Mutex<LinkedList<Arc<RwLock<HashMapEmbeddingEntry>>>>>>>,
    pub capacity: usize,
}

impl RecyclePool {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            capacity,
        }
    }

    pub fn restore(&self, entry: Arc<RwLock<HashMapEmbeddingEntry>>) {
        let dim = { entry.read().inner_size() };
        let entry_list_exist = {
            let map = self.inner.read();
            map.get(&dim).is_some()
        };
        if !entry_list_exist {
            let mut map = self.inner.write();
            let list = map.get(&dim);
            if list.is_none() {
                let empty_list = Mutex::new(LinkedList::new());
                map.insert(dim, empty_list);
            }
        }
        let map = self.inner.read();
        let mut list = map.get(&dim).unwrap().lock();
        if list.len() < self.capacity {
            list.push_back(entry);
        }
    }

    pub fn take(&self, inner_size: usize) -> Option<Arc<RwLock<HashMapEmbeddingEntry>>> {
        let map = self.inner.read();
        let list = map.get(&inner_size);
        if list.is_none() {
            return None;
        }
        let mut list = list.unwrap().lock();
        list.pop_back()
    }
}
