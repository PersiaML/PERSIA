use std::hash::{Hash, Hasher};

use hashlink::linked_hash_map::RawEntryMut;
use hashlink::LinkedHashMap;
use persia_libs::{async_lock, hashbrown::HashMap, parking_lot};

#[derive(Debug)]
pub struct Sharded<T, K> {
    pub inner: Vec<parking_lot::RwLock<T>>,
    pub phantom: std::marker::PhantomData<K>,
}

#[inline]
pub fn get_index<K>(key: &K, count: usize) -> usize
where
    K: Hash + Eq + Clone,
{
    let mut s = ahash::AHasher::default();
    key.hash(&mut s);
    (s.finish() as usize % count) as usize
}

impl<T, K> Sharded<T, K> {
    #[inline]
    pub fn shard(&self, key: &K) -> &parking_lot::RwLock<T>
    where
        K: Hash + Eq + Clone,
    {
        unsafe { self.inner.get_unchecked(get_index(key, self.inner.len())) }
    }
}

#[derive(Debug)]
pub struct ShardedAsync<T, K> {
    pub inner: Vec<async_lock::RwLock<T>>,
    pub phantom: std::marker::PhantomData<K>,
}

impl<T, K> ShardedAsync<T, K> {
    #[inline]
    pub fn shard(&self, key: &K) -> &async_lock::RwLock<T>
    where
        K: Hash + Eq + Clone,
    {
        unsafe { self.inner.get_unchecked(get_index(key, self.inner.len())) }
    }
}

pub type ShardedMap<K, V> = Sharded<HashMap<K, V>, K>;

impl<K, V> ShardedMap<K, V> {
    pub fn len(&self) -> usize {
        self.inner.iter().map(|x| x.read().len()).sum::<usize>()
    }
}

pub type ShardedAsyncMap<K, V> = ShardedAsync<HashMap<K, V>, K>;

impl<K, V> ShardedAsyncMap<K, V> {
    pub async fn len(&self) -> usize {
        let mut total = 0;
        for x in self.inner.iter() {
            total += x.read().await.len();
        }
        total
    }
}

/// NOTE THAT THIS MAP WILL CLONE VALUE DURING GET
pub struct PersiaEvictionMap<K: Hash + Eq + Clone, V: Clone> {
    pub inner: Sharded<LinkedHashMap<K, V>, K>,
    pub capacity: usize,
    pub capacity_per_bucket: usize,
}

impl<K, V> PersiaEvictionMap<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(capacity: usize, bucket_size: usize) -> Self {
        let cap_with_buffer = (capacity as f32 * 1.1) as usize;
        Self {
            inner: Sharded {
                inner: vec![
                    LinkedHashMap::with_capacity(cap_with_buffer / bucket_size);
                    bucket_size
                ]
                .into_iter()
                .map(parking_lot::RwLock::new)
                .collect(),
                phantom: std::marker::PhantomData::default(),
            },
            capacity,
            capacity_per_bucket: capacity / bucket_size,
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.shard(key).read().get(key).cloned()
    }

    pub fn get_refresh(&self, key: &K) -> Option<V> {
        let mut guard = self.inner.shard(key).write();
        let mut entry = guard.raw_entry_mut().from_key(key);
        match entry {
            RawEntryMut::Occupied(ref mut x) => {
                x.to_back();
                Some(x.get_key_value().1).cloned()
            }
            RawEntryMut::Vacant(_) => None,
        }
    }

    pub fn insert(&self, key: K, value: V) -> (Option<V>, Option<V>) {
        let mut guard = self.inner.shard(&key).write();
        let old_val = guard.insert(key, value);
        let evcited = {
            if guard.len() > self.capacity_per_bucket {
                let evcited = guard.pop_front();
                match evcited {
                    Some((_ek, ev)) => Some(ev),
                    None => None,
                }
            } else {
                None
            }
        };
        (old_val, evcited)
    }

    pub fn clear(&self) {
        self.inner.inner.iter().for_each(|x| x.write().clear());
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner
            .inner
            .iter()
            .map(|x| x.read().len())
            .sum::<usize>()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod eviction_map_tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use persia_common::HashMapEmbeddingEntry;
    use persia_embedding_config::InitializationMethod;
    use std::sync::Arc;

    type ArcEntry = Arc<parking_lot::RwLock<HashMapEmbeddingEntry>>;

    #[test]
    fn test_evict() {
        let map: PersiaEvictionMap<u64, ArcEntry> = PersiaEvictionMap::new(5, 1);

        let initialization = InitializationMethod::default();

        for i in 0..5 {
            let entry = HashMapEmbeddingEntry::new(&initialization, 8, 16, i);
            map.insert(i, Arc::new(parking_lot::RwLock::new(entry)));
        }

        assert_eq!(map.len(), 5);

        for i in 5..10 {
            let entry = HashMapEmbeddingEntry::new(&initialization, 8, 16, i);
            map.insert(i, Arc::new(parking_lot::RwLock::new(entry)));
        }

        assert_eq!(map.len(), 5);
        assert_eq!(map.get_refresh(&4).is_none(), true);
        assert_eq!(map.get_refresh(&5).is_some(), true);

        let entry = HashMapEmbeddingEntry::new(&initialization, 8, 16, 10);
        map.insert(10, Arc::new(parking_lot::RwLock::new(entry)));

        assert_eq!(map.len(), 5);
        assert_eq!(map.get_refresh(&6).is_none(), true);
        assert_eq!(map.get_refresh(&5).is_some(), true);
    }
}
