use std::{
    cmp::max,
    sync::{Arc, Weak},
};

use hashbrown::HashMap;
use parking_lot::RwLock;
use thiserror::Error;

use persia_embedding_config::{PersiaGlobalConfigError, PersiaShardedServerConfig};
use persia_embedding_datatypes::HashMapEmbeddingEntry;
use persia_eviction_map::Sharded;
use persia_speedy::{Readable, Writable};

#[derive(Readable, Writable, Error, Debug, Clone)]
pub enum PersiaFullAmountManagerError {
    #[error("full amount manager not ready error")]
    NotReadyError,
    #[error("global config error")]
    PersiaGlobalConfigError(#[from] PersiaGlobalConfigError),
    #[error("failed to commit weakptrs or evicted ids, please try a bigger buffer size for full anmount manager")]
    CommitError,
}

static FULL_AMOUNT_MANAGER: once_cell::sync::OnceCell<Arc<FullAmountManager>> =
    once_cell::sync::OnceCell::new();

// this sturct keep weak ptrs for all embedding entry
pub struct FullAmountManager {
    weak_map: Sharded<HashMap<u64, Weak<RwLock<HashMapEmbeddingEntry>>>, u64>,
    weak_ptr_channel: persia_futures::ChannelPair<Vec<(u64, Weak<RwLock<HashMapEmbeddingEntry>>)>>,
    evicted_ids_channel: persia_futures::ChannelPair<Vec<u64>>,
    _handles: Arc<parking_lot::Mutex<Vec<std::thread::JoinHandle<()>>>>,
}

impl FullAmountManager {
    pub fn get() -> Result<Arc<Self>, PersiaFullAmountManagerError> {
        let singleton = FULL_AMOUNT_MANAGER.get_or_try_init(|| {
            let config = PersiaShardedServerConfig::get()?;
            let handles = Arc::new(parking_lot::Mutex::new(Vec::new()));
            let guard = config.read();
            let full_amount_manager = Self::new(
                guard.capacity,
                guard.num_hashmap_internal_shards,
                guard.full_amount_manager_buffer_size,
                handles.clone(),
            );
            let singleton = Arc::new(full_amount_manager);

            let num_collect_threads = max(1, guard.num_hashmap_internal_shards / 32);
            for _ in 0..num_collect_threads {
                let handle = std::thread::spawn({
                    let singleton = singleton.clone();
                    move || {
                        singleton.collect_weak_ptrs();
                    }
                });
                let mut handles_guard = handles.lock();
                handles_guard.push(handle);
            }
            for _ in 0..num_collect_threads {
                let handle = std::thread::spawn({
                    let singleton = singleton.clone();
                    move || {
                        singleton.remove_evicted_ids();
                    }
                });
                let mut handles_guard = handles.lock();
                handles_guard.push(handle);
            }

            Ok(singleton)
        });

        match singleton {
            Ok(s) => Ok(s.clone()),
            Err(e) => Err(e),
        }
    }

    fn new(
        capacity: usize,
        bucket_size: usize,
        buffer_size: usize,
        handles: Arc<parking_lot::Mutex<Vec<std::thread::JoinHandle<()>>>>,
    ) -> Self {
        Self {
            weak_map: Sharded {
                inner: vec![HashMap::with_capacity(capacity / bucket_size); bucket_size]
                    .into_iter()
                    .map(RwLock::new)
                    .collect(),
                phantom: std::marker::PhantomData::default(),
            },
            weak_ptr_channel: persia_futures::ChannelPair::new(buffer_size),
            evicted_ids_channel: persia_futures::ChannelPair::new(buffer_size),
            _handles: handles,
        }
    }

    pub fn try_commit_weak_ptrs(
        &self,
        weak_ptrs: Vec<(u64, Weak<RwLock<HashMapEmbeddingEntry>>)>,
    ) -> Result<(), PersiaFullAmountManagerError> {
        if let Ok(_) = self.weak_ptr_channel.sender.try_send(weak_ptrs) {
            Ok(())
        } else {
            Err(PersiaFullAmountManagerError::CommitError)
        }
    }

    pub fn commit_weak_ptrs(
        &self,
        weak_ptrs: Vec<(u64, Weak<RwLock<HashMapEmbeddingEntry>>)>,
    ) -> Result<(), PersiaFullAmountManagerError> {
        if let Ok(_) = self.weak_ptr_channel.sender.send(weak_ptrs) {
            Ok(())
        } else {
            Err(PersiaFullAmountManagerError::CommitError)
        }
    }

    pub fn try_commit_evicted_ids(
        &self,
        evicted_ids: Vec<u64>,
    ) -> Result<(), PersiaFullAmountManagerError> {
        if let Ok(_) = self.evicted_ids_channel.sender.try_send(evicted_ids) {
            Ok(())
        } else {
            Err(PersiaFullAmountManagerError::CommitError)
        }
    }

    fn remove_evicted_ids(&self) -> () {
        self.evicted_ids_channel
            .receiver
            .iter()
            .for_each(|evicted_ids| {
                evicted_ids.into_iter().for_each(|evicted| {
                    let mut guard = self.weak_map.shard(&evicted).write();
                    if let Some(v) = guard.get(&evicted) {
                        if v.upgrade().is_none() {
                            guard.remove(&evicted);
                        }
                    }
                });
            });
    }

    fn collect_weak_ptrs(&self) -> () {
        self.weak_ptr_channel.receiver.iter().for_each(|weak_ptrs| {
            weak_ptrs.iter().for_each(|(k, v)| {
                self.weak_map.shard(k).write().insert(k.clone(), v.clone());
            });
        });
    }

    pub fn len(&self) -> usize {
        self.weak_map
            .inner
            .iter()
            .map(|x| x.read().len())
            .sum::<usize>()
    }

    pub fn keys_values(&self) -> Vec<(u64, Arc<RwLock<HashMapEmbeddingEntry>>)> {
        let mut result = Vec::with_capacity(self.len());
        self.weak_map.inner.iter().for_each(|x| {
            let mut guard = x.write();
            guard.retain(|k, v| {
                let ptr_opt = v.upgrade();
                match ptr_opt {
                    Some(value) => {
                        result.push((k.clone(), value.clone()));
                        true
                    }
                    None => false,
                }
            });
        });
        result
    }
}
