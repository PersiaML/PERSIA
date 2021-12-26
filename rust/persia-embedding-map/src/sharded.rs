use persia_libs::parking_lot;
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
