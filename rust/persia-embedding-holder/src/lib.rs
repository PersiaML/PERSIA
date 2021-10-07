use hashlink::LinkedHashMap;
use indexlist::IndexList;
use slotmap::{DefaultKey, HopSlotMap};
use std::collections::LinkedList;
use std::{sync::Arc, sync::Weak};

use persia_libs::{
    half,
    hashbrown::HashMap,
    once_cell,
    parking_lot::{Mutex, RwLock},
    thiserror, tracing,
};

use persia_common::HashMapEmbeddingEntry;
use persia_embedding_config::{PersiaEmbeddingServerConfig, PersiaGlobalConfigError};
use persia_eviction_map::{PersiaEvictionMap, Sharded};
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

pub struct PersiaHashLink {
    hashmap: HashMap<u64, DefaultKey>,
    slotmap: HopSlotMap<DefaultKey, RwLock<HashMapEmbeddingEntry>>,
    indexlist: IndexList<u64>,
    capacity: usize,
}

impl PersiaHashLink {
    pub fn new(capacity: usize) -> Self {
        Self {
            hashmap: HashMap::with_capacity(capacity),
            slotmap: HopSlotMap::with_capacity(capacity),
            indexlist: IndexList::with_capacity(capacity),
            capacity,
        }
    }

    pub fn insert(
        &mut self,
        key: u64,
        value: RwLock<HashMapEmbeddingEntry>,
    ) -> Option<RwLock<HashMapEmbeddingEntry>> {
        match self.hashmap.get(&key) {
            Some(old_k) => {
                if let Some(old_v) = self.slotmap.get_mut(*old_k) {
                    *old_v = value;
                }
                if let Some(old_index) = self.indexlist.index_of(&key) {
                    self.indexlist.remove(old_index);
                    self.indexlist.push_back(key);
                }
            }
            None => {
                let slot_key = self.slotmap.insert(value);
                self.hashmap.insert(key.clone(), slot_key);
                self.indexlist.push_back(key);
            }
        }
        if self.hashmap.len() > self.capacity {
            if let Some(evicted) = self.indexlist.pop_front() {
                if let Some(slot_key) = self.hashmap.remove(&evicted) {
                    let evicted_v = self.slotmap.remove(slot_key);
                    return evicted_v;
                }
            }
        }
        return None;
    }

    pub fn get_value(&self, key: &u64) -> Option<&RwLock<HashMapEmbeddingEntry>> {
        match self.hashmap.get(key) {
            Some(slot_k) => {
                let val = self.slotmap.get(*slot_k);
                return val;
            }
            None => {
                return None;
            }
        }
    }

    pub fn get_value_refresh(&mut self, key: &u64) -> Option<&RwLock<HashMapEmbeddingEntry>> {
        match self.hashmap.get(key) {
            Some(slot_k) => {
                let val = self.slotmap.get(*slot_k);
                if let Some(index) = self.indexlist.index_of(key) {
                    self.indexlist.remove(index);
                    self.indexlist.push_back(key.clone());
                }
                return val;
            }
            None => {
                return None;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.slotmap.len()
    }
}

#[derive(Clone)]
pub struct PersiaEmbeddingHolder {
    pub inner: Arc<Sharded<PersiaHashLink, u64>>,
    _recycle_pool: Arc<RecyclePool>,
}

impl PersiaEmbeddingHolder {
    pub fn get() -> Result<PersiaEmbeddingHolder, PersiaEmbeddingHolderError> {
        let singleton = PERSIA_EMBEDDING_HOLDER.get_or_try_init(|| {
            let config = PersiaEmbeddingServerConfig::get()?;
            let initial_emb_size = std::env::var("INITIAL_EMB_SIZE")
                .unwrap_or(String::from("0"))
                .parse::<u64>()
                .expect("INITIAL_EMB_SIZE not a number");
            let eviction_map: Sharded<PersiaHashLink, u64> = if initial_emb_size == 0 {
                Sharded {
                    inner: vec![RwLock::new(PersiaHashLink::new(
                        config.capacity / config.num_hashmap_internal_shards,
                    ))],
                    phantom: std::marker::PhantomData::default(),
                }
            } else {
                let initial_emb_dim = std::env::var("INITIAL_EMB_DIM")
                    .unwrap_or(String::from("0"))
                    .parse::<usize>()
                    .expect("INITIAL_EMB_DIM not a number");
                let bucket_size = config.num_hashmap_internal_shards as u64;
                let num_ids_per_bucket = initial_emb_size / bucket_size;

                tracing::info!("start to generate embedding");

                let handles: Vec<std::thread::JoinHandle<_>> = (0..bucket_size)
                    .map(|_| {
                        let num_ids_per_bucket = num_ids_per_bucket.clone();
                        std::thread::spawn(move || {
                            let float_embs = vec![0.01_f32; initial_emb_dim];
                            // let inner =
                            //     half::vec::HalfFloatVecExt::from_f32_slice(float_embs.as_slice());
                            let inner = [half::f16::from_f32(0.01_f32); 32];
                            let entry = HashMapEmbeddingEntry {
                                inner,
                                embedding_dim: initial_emb_dim,
                            };
                            let mut map = PersiaHashLink::new(num_ids_per_bucket as usize);
                            // (0..num_ids_per_bucket).for_each(|id| {
                            //     if id % 100000 == 0 {
                            //         let progress =
                            //             id as f32 / num_ids_per_bucket as f32 * 100.0_f32;
                            //         tracing::info!("generating embedding, progress {}%", progress);
                            //     }
                            //     map.insert(id, RwLock::new(entry.clone()));
                            // });
                            map
                        })
                    })
                    .collect();

                let maps: Vec<_> = handles
                    .into_iter()
                    .map(|h| RwLock::new(h.join().expect("failed to gen map")))
                    .collect();

                panic!("exit");

                Sharded {
                    inner: maps,
                    phantom: std::marker::PhantomData::default(),
                }
            };
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
        self.inner
            .inner
            .iter()
            .map(|x| x.read().len())
            .sum::<usize>()
    }

    // pub fn insert(
    //     &self,
    //     key: u64,
    //     value: RwLock<HashMapEmbeddingEntry>,
    // ) -> Option<RwLock<HashMapEmbeddingEntry>> {
    //     self.inner.shard(&key).write().insert(key, value)
    // }

    // pub fn get_value(&self, key: &u64) -> Option<&RwLock<HashMapEmbeddingEntry>> {
    //     self.inner.shard(key).read().get_value(key)
    // }

    // pub fn get_value_refresh(&self, key: &u64) -> Option<&RwLock<HashMapEmbeddingEntry>> {
    //     self.inner.shard(key).write().get_value_refresh(key)
    // }

    pub fn clear(&self) {
        tracing::warn!("clear do nothing for now");
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
