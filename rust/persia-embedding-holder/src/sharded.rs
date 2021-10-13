use persia_libs::{async_lock, hashbrown::HashMap, parking_lot};
use std::hash::{Hash, Hasher};

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

    #[inline]
    pub fn get_shard_by_index(&self, idx: usize) -> &parking_lot::RwLock<T>
    where
        K: Hash + Eq + Clone,
    {
        unsafe { self.inner.get_unchecked(idx) }
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
