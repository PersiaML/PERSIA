#![feature(test)]
extern crate test;

use hashlink::linked_hash_map::RawEntryMut;
use lru_cache::LruCache;
use persia_sharded::Sharded;
use rand_distr::{Distribution, Normal};
use random_fast_rng::Random;
use std::collections::HashSet;
use std::hash::Hash;
use std::marker::{Send, Sync};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use thread_local::ThreadLocal;

trait Backend<V> {
    fn b_get(&self, key: u64) -> Option<V>;
    fn b_put(&self, key: u64, value: V);
}

pub trait RingAddress<K> {
    fn set_ring_address(&self, new_address: usize) -> ();

    fn get_ring_address(&self) -> Option<usize>;

    fn get_key(&self) -> K;
}

#[derive(Clone, Debug)]
pub struct EmptyEmbeddingEntry {
    sign: u64,
    ring_address: Option<usize>,
}

impl EmptyEmbeddingEntry {
    pub fn new(sign: u64) -> Self {
        Self {
            sign,
            ring_address: None,
        }
    }

    fn set_ring_address(&mut self, new_address: usize) -> () {
        self.ring_address = Some(new_address);
    }

    fn get_ring_address(&self) -> Option<usize> {
        self.ring_address
    }

    fn get_sign(&self) -> u64 {
        self.sign
    }
}

type ArcEntry = Arc<parking_lot::RwLock<EmptyEmbeddingEntry>>;

impl RingAddress<u64> for ArcEntry {
    fn set_ring_address(&self, new_address: usize) -> () {
        let mut guard = self.write();
        guard.set_ring_address(new_address);
    }

    fn get_ring_address(&self) -> Option<usize> {
        let guard = self.read();
        guard.get_ring_address()
    }

    fn get_key(&self) -> u64 {
        let guard = self.read();
        guard.get_sign()
    }
}

#[derive(Clone)]
pub struct PersiaRingPointer {
    pub inner: usize,
    pub capacity: usize,
}

impl PersiaRingPointer {
    pub fn new(capacity: usize) -> Self {
        Self { inner: 0, capacity }
    }

    fn get_next(&self) -> usize {
        if self.inner == self.capacity - 1 {
            0
        } else {
            self.inner + 1
        }
    }

    fn _get_prev(&self) -> usize {
        if self.inner == 0 {
            self.capacity - 1
        } else {
            self.inner - 1
        }
    }

    fn next(&mut self) -> usize {
        self.inner = self.get_next();
        self.inner
    }
}

#[derive(Clone)]
pub struct PersiaRingBuffer<K, V> {
    pub inner: Vec<Option<V>>,
    tail: PersiaRingPointer,
    phantom: std::marker::PhantomData<K>,
}

impl<K, V> PersiaRingBuffer<K, V>
where
    K: Clone,
    V: Clone + RingAddress<K>,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: vec![Option::None; capacity],
            tail: PersiaRingPointer::new(capacity),
            phantom: std::marker::PhantomData::default(),
        }
    }

    pub fn put(&mut self, value: V) -> Option<V> {
        let cur = self.tail.next();
        let evicted = self.inner.get(cur).unwrap().clone();
        value.set_ring_address(cur);
        *self.inner.get_mut(cur).unwrap() = Some(value);

        evicted
    }

    pub fn update_order(&mut self, newer: usize) -> () {
        if newer == self.tail.inner {
            return;
        }

        let older = if newer == self.inner.len() - 1 {
            0
        } else {
            newer + 1
        };
        self.inner.swap(newer, older);
        let new = self.inner.get_mut(newer).unwrap();
        match new {
            Some(v) => {
                v.set_ring_address(newer);
            }
            None => {
                tracing::error!("update order with a None value, it is a bug");
            }
        }
        let old = self.inner.get_mut(older).unwrap();
        match old {
            Some(v) => {
                v.set_ring_address(older);
            }
            None => {
                tracing::error!("update order with a None value, it is a bug");
            }
        }
    }

    pub fn len(&self) -> usize {
        let next = self.inner.get(self.tail.get_next()).unwrap();
        match next {
            Some(_) => self.inner.len(),
            None => self.tail.inner,
        }
    }
}

// linkedhashmap + hashset
pub struct PersiaEvictionMapMarkI<K: Sync + Send, V> {
    pub inner: Sharded<hashlink::LinkedHashMap<K, V>, K>,
    pub capacity: usize,
    total_size: AtomicUsize,
    local_set: ThreadLocal<parking_lot::Mutex<HashSet<K>>>,
    global_set: parking_lot::RwLock<HashSet<K>>,
}

impl<K, V> PersiaEvictionMapMarkI<K, V>
where
    K: Hash + Eq + Clone + Send + Sync,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Sharded {
                inner: vec![hashlink::LinkedHashMap::with_capacity(capacity / 128); 128]
                    .into_iter()
                    .map(|x| parking_lot::RwLock::new(x))
                    .collect(),
                phantom: std::marker::PhantomData::default(),
            },
            capacity,
            total_size: AtomicUsize::new(0),
            local_set: ThreadLocal::new(),
            global_set: parking_lot::RwLock::new(HashSet::new()),
        }
    }

    #[inline]
    fn should_update_order(&self) -> bool {
        random_fast_rng::local_rng().get_u8() < 16
    }

    #[inline]
    fn should_start_eviction(&self) -> bool {
        random_fast_rng::local_rng().get_u16() < 2 && self.len() > self.capacity
    }

    pub fn get(&self, key: &K) -> Option<V> {
        if self.should_update_order() {
            let mut guard = self.inner.shard(key).write();
            let mut entry = guard.raw_entry_mut().from_key(key);
            match entry {
                RawEntryMut::Occupied(ref mut x) => {
                    x.to_back();
                    Some(x.get_key_value().1).cloned()
                }
                RawEntryMut::Vacant(_) => None,
            }
        } else {
            self.inner.shard(key).read().get(key).cloned()
        }
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let result = {
            let mut guard = self.inner.shard(&key).write();
            let result = guard.insert(key.clone(), value);

            if result.is_none() {
                self.total_size.fetch_add(1, Ordering::AcqRel);
            }

            if self.should_start_eviction() {
                let num_evicted_id = guard
                    .len()
                    .saturating_sub(self.capacity / self.inner.inner.len());
                for _ in 0..num_evicted_id {
                    guard.pop_front();
                }
                self.total_size.fetch_sub(num_evicted_id, Ordering::AcqRel);
            }
            result
        };

        let tls_set = self
            .local_set
            .get_or(|| parking_lot::Mutex::new(HashSet::new()));
        let mut local = tls_set.lock();
        if let Some(mut global) = self.global_set.try_write() {
            global.insert(key.clone());
            for k in local.drain() {
                global.insert(k);
            }
        } else {
            local.insert(key.clone());
        }

        result
    }

    pub fn len(&self) -> usize {
        self.total_size.load(Ordering::Acquire)
    }

    pub fn keys_values(&self) -> Vec<(K, V)> {
        let mut global = self.global_set.write();
        let mut evictied = Vec::with_capacity(global.len());
        let mut kv = Vec::with_capacity(global.len());
        for key in global.iter() {
            let value = self.inner.shard(key).read().get(key).cloned();
            match value {
                Some(v) => {
                    kv.push((key.clone(), v.clone()));
                }
                None => {
                    evictied.push(key.clone());
                }
            }
        }
        for e in evictied {
            global.remove(&e);
        }
        kv
    }
}

pub enum EvictOp<K> {
    Insert(K),
    UpdateOrder(K),
}

// hashmap + linkedhashset + channel
pub struct PersiaEvictionMapMarkII<K, V> {
    pub inner: Sharded<griddle::HashMap<K, V>, K>,
    pub evictor: Sharded<hashlink::LinkedHashSet<K>, K>,
    pub capacity: usize,
    evict_channel: persia_futures::ChannelPair<EvictOp<K>>,
    total_size: AtomicUsize,
}

impl<K, V> PersiaEvictionMapMarkII<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Sharded {
                inner: vec![griddle::HashMap::with_capacity(capacity / 128); 128]
                    .into_iter()
                    .map(parking_lot::RwLock::new)
                    .collect(),
                phantom: std::marker::PhantomData::default(),
            },
            evictor: Sharded {
                inner: vec![hashlink::LinkedHashSet::with_capacity(capacity / 128); 128]
                    .into_iter()
                    .map(parking_lot::RwLock::new)
                    .collect(),
                phantom: std::marker::PhantomData::default(),
            },
            capacity,
            evict_channel: persia_futures::ChannelPair::new(capacity / 10),
            total_size: AtomicUsize::new(0),
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        if self
            .evict_channel
            .sender
            .try_send(EvictOp::UpdateOrder(key.clone()))
            .is_err()
        {
            // add perflog
        }
        self.inner.shard(key).read().get(key).cloned()
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        if self
            .evict_channel
            .sender
            .try_send(EvictOp::Insert(key.clone()))
            .is_err()
        {
            // add perflog
        }
        let mut guard = self.inner.shard(&key).write();
        let prev = guard.insert(key, value);
        if prev.is_none() {
            self.total_size.fetch_add(1, Ordering::AcqRel);
        }
        prev
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.total_size.load(Ordering::Acquire)
    }

    pub fn evict(&self) -> () {
        // add perflog
        while let Ok(op) = self.evict_channel.receiver.try_recv() {
            match op {
                EvictOp::Insert(key) => {
                    let mut guard = self.evictor.shard(&key).write();
                    guard.insert(key);
                }
                EvictOp::UpdateOrder(key) => {
                    let mut guard = self.evictor.shard(&key).write();
                    if guard.contains(&key) {
                        guard.to_back(&key);
                    }
                }
            }
        }
        if self.len() > self.capacity {
            tracing::info!(
                "currnt len is {}, while capacity is {}, start to evict...",
                self.len(),
                self.capacity
            );
            let mut evicted = Vec::new();
            for shard in self.evictor.inner.iter() {
                let num_evicted_id = { shard.read().len() } - self.capacity / 128;
                for _ in 0..num_evicted_id {
                    let mut guard = shard.write();
                    if let Some(e) = guard.pop_front() {
                        evicted.push(e);
                    }
                }
            }
            self.total_size.fetch_sub(evicted.len(), Ordering::AcqRel);
            for e in evicted {
                let mut guard = self.inner.shard(&e).write();
                guard.remove(&e);
            }
        }
    }

    pub fn keys_values(&self) -> Vec<(K, V)> {
        let mut kv = Vec::with_capacity(self.len());
        for shard in self.evictor.inner.iter() {
            let guard = shard.read();
            for key in guard.iter() {
                if let Some(value) = self.inner.shard(&key).read().get(&key) {
                    kv.push((key.clone(), value.clone()));
                }
            }
        }
        kv
    }
}

// hashmap + linkedhashset do best effort
pub struct PersiaEvictionMapMarkIII<K, V> {
    pub inner: Sharded<griddle::HashMap<K, V>, K>,
    pub evictor: Sharded<hashlink::LinkedHashSet<K>, K>,
    pub capacity: usize,
    total_size: AtomicUsize,
}

impl<K, V> PersiaEvictionMapMarkIII<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Sharded {
                inner: vec![griddle::HashMap::with_capacity(capacity / 128); 128]
                    .into_iter()
                    .map(parking_lot::RwLock::new)
                    .collect(),
                phantom: std::marker::PhantomData::default(),
            },
            evictor: Sharded {
                inner: vec![hashlink::LinkedHashSet::with_capacity(capacity / 128); 128]
                    .into_iter()
                    .map(parking_lot::RwLock::new)
                    .collect(),
                phantom: std::marker::PhantomData::default(),
            },
            capacity,
            total_size: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn should_update_order(&self) -> bool {
        random_fast_rng::local_rng().get_u8() < 16
    }

    #[inline]
    fn should_start_eviction(&self) -> bool {
        random_fast_rng::local_rng().get_u16() < 2 && self.len() > self.capacity
    }

    pub fn get(&self, key: &K) -> Option<V> {
        if self.should_update_order() {
            if let Some(mut guard) = self.evictor.shard(key).try_write() {
                if guard.contains(&key) {
                    guard.to_back(&key);
                }
            }
        }
        self.inner.shard(key).read().get(key).cloned()
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        if let Some(mut guard) = self.evictor.shard(&key).try_write() {
            guard.insert(key.clone());
        }
        let mut evicted = Vec::new();
        if self.should_start_eviction() {
            if let Some(mut guard) = self.evictor.shard(&key).try_write() {
                let num_evicted_id = guard.len() - self.capacity / self.inner.inner.len();
                for _ in 0..num_evicted_id {
                    if let Some(e) = guard.pop_front() {
                        evicted.push(e);
                    }
                }
            }
        }

        self.total_size.fetch_sub(evicted.len(), Ordering::AcqRel);

        let result = {
            let mut guard = self.inner.shard(&key).write();
            for e in evicted {
                guard.remove(&e);
            }
            let prev = guard.insert(key, value);
            if prev.is_none() {
                self.total_size.fetch_add(1, Ordering::AcqRel);
            }
            prev
        };
        result
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.total_size.load(Ordering::Acquire)
    }
}

pub struct PersiaEvictionMapMarkIV<K, V> {
    pub inner: Sharded<griddle::HashMap<K, V>, K>,
    pub evictor: Sharded<PersiaRingBuffer<K, V>, K>,
    pub capacity: usize,
}

impl<K, V> PersiaEvictionMapMarkIV<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone + RingAddress<K>,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Sharded {
                inner: vec![griddle::HashMap::with_capacity(capacity / 128); 128]
                    .into_iter()
                    .map(parking_lot::RwLock::new)
                    .collect(),
                phantom: std::marker::PhantomData::default(),
            },
            evictor: Sharded {
                inner: vec![PersiaRingBuffer::new(capacity / 128); 128]
                    .into_iter()
                    .map(parking_lot::RwLock::new)
                    .collect(),
                phantom: std::marker::PhantomData::default(),
            },
            capacity,
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        let res = self.inner.shard(key).read().get(key).cloned();
        match res {
            Some(ref v) => {
                let mut guard = self.evictor.shard(key).write();
                let addr = v.get_ring_address().unwrap();
                guard.update_order(addr);
            }
            None => {}
        }
        res
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let evicted = {
            let mut guard = self.evictor.shard(&key).write();
            guard.put(value.clone())
        };
        let result = {
            let mut guard = self.inner.shard(&key).write();
            match evicted {
                Some(e) => {
                    // NOTE: for now, eviction map is used for sharded embedding server.
                    //       fn `get` will be called before fn `insert` to check if a
                    //       key is alreay in map. But in some cases, say, thread A and
                    //       thread B, call fn `get` almost simultaneously for a same key,
                    //       both two thread get `Option::None` returned. So they would
                    //       call fn `insert` respectively, Causing two value in evictor
                    //       with a same key. During one of the evictor evicting, it will
                    //       remove the corresponding value in `inner` hashmap, which
                    //       lifetime controled by the other evictor. Following code could
                    //       avoid this happening, but with a poor performance. As far as
                    //       we known, above-mentioned situation will not occur very
                    //       frequently, for now, we just remove the key.

                    //       let evicting_key = e.get_key();
                    //       let evicting_val = guard.get(&evicting_key);
                    //       match evicting_val {
                    //           Some(v) => {
                    //               if e.get_ring_address() == v.get_ring_address() {
                    //                   guard.remove(&evicting_key);
                    //                   self.total_size.fetch_sub(1, Ordering::AcqRel);
                    //               }
                    //           }
                    //           None => {}
                    //       }

                    guard.remove(&e.get_key());
                }
                None => {}
            }
            guard.insert(key, value)
        };
        result
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

pub struct PersiaEvictionMapMarkV<K: Hash + Eq + Clone, V: Clone> {
    pub inner: Sharded<LruCache<K, V>, K>,
}

impl<K, V> PersiaEvictionMapMarkV<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Sharded {
                inner: vec![LruCache::new(capacity / 128); 128]
                    .into_iter()
                    .map(parking_lot::RwLock::new)
                    .collect(),
                phantom: std::marker::PhantomData::default(),
            },
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        self.inner.shard(key).write().get_mut(key).cloned()
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        self.inner.shard(&key).write().insert(key, value)
    }

    pub fn keys_values(&self) -> Vec<(K, V)> {
        Vec::new()
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

impl Backend<ArcEntry> for Arc<PersiaEvictionMapMarkI<u64, ArcEntry>> {
    fn b_get(&self, key: u64) -> Option<ArcEntry> {
        self.get(&key)
    }

    fn b_put(&self, key: u64, value: ArcEntry) {
        self.insert(key, value);
    }
}

impl Backend<ArcEntry> for Arc<PersiaEvictionMapMarkII<u64, ArcEntry>> {
    fn b_get(&self, key: u64) -> Option<ArcEntry> {
        self.get(&key)
    }

    fn b_put(&self, key: u64, value: ArcEntry) {
        self.insert(key, value);
    }
}

impl Backend<ArcEntry> for Arc<PersiaEvictionMapMarkIII<u64, ArcEntry>> {
    fn b_get(&self, key: u64) -> Option<ArcEntry> {
        self.get(&key)
    }

    fn b_put(&self, key: u64, value: ArcEntry) {
        self.insert(key, value);
    }
}

impl Backend<ArcEntry> for Arc<PersiaEvictionMapMarkIV<u64, ArcEntry>> {
    fn b_get(&self, key: u64) -> Option<ArcEntry> {
        self.get(&key)
    }

    fn b_put(&self, key: u64, value: ArcEntry) {
        self.insert(key, value);
    }
}

impl Backend<ArcEntry> for Arc<PersiaEvictionMapMarkV<u64, ArcEntry>> {
    fn b_get(&self, key: u64) -> Option<ArcEntry> {
        self.get(&key)
    }

    fn b_put(&self, key: u64, value: ArcEntry) {
        self.insert(key, value);
    }
}

impl Backend<ArcEntry> for Arc<Sharded<hashbrown::HashMap<u64, ArcEntry>, u64>> {
    fn b_get(&self, key: u64) -> Option<ArcEntry> {
        let guard = self.shard(&key).read();
        let value = guard.get(&key);
        match value {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }

    fn b_put(&self, key: u64, value: ArcEntry) {
        self.shard(&key).write().insert(key, value);
    }
}

impl Backend<ArcEntry> for Arc<Sharded<griddle::HashMap<u64, ArcEntry>, u64>> {
    fn b_get(&self, key: u64) -> Option<ArcEntry> {
        let guard = self.shard(&key).read();
        let value = guard.get(&key);
        match value {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }

    fn b_put(&self, key: u64, value: ArcEntry) {
        self.shard(&key).write().insert(key, value);
    }
}

impl Backend<ArcEntry> for Arc<Sharded<hashlink::LinkedHashMap<u64, ArcEntry>, u64>> {
    fn b_get(&self, key: u64) -> Option<ArcEntry> {
        let guard = self.shard(&key).read();
        let value = guard.get(&key);
        match value {
            Some(v) => Some(v.clone()),
            None => None,
        }
    }

    fn b_put(&self, key: u64, value: ArcEntry) {
        self.shard(&key).write().insert(key, value);
    }
}

fn run_with_normal<B: Backend<ArcEntry>>(
    backend: B,
    end: std::time::Instant,
    write: bool,
) -> (bool, usize) {
    let mut ops = 0;
    let normal = Normal::new(50000000.0, 100.0).unwrap();
    while std::time::Instant::now() < end {
        let id = normal.sample(&mut rand::thread_rng()) as u64;
        if write {
            let val = Arc::new(parking_lot::RwLock::new(EmptyEmbeddingEntry::new(
                id.clone(),
            )));
            backend.b_put(id, val);
        } else {
            backend.b_get(id);
        }
        ops += 1;
    }

    (write, ops)
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("LOG_LEVEL"))
        .init();

    // some parameters for test
    let num_readers: usize = 30;
    let num_writers: usize = 30;
    let dur = std::time::Duration::from_secs(60);
    let dur_in_ns = dur.as_secs() * 1_000_000_000_u64 + u64::from(dur.subsec_nanos());
    let dur_in_s = dur_in_ns as f64 / 1_000_000_000_f64;

    let stat = |op, results: Vec<(_, usize)>| {
        for (i, res) in results.into_iter().enumerate() {
            tracing::info!(
                "execute {:8.0} ops/s in {} thread {}",
                res.1 as f64 / dur_in_s as f64,
                op,
                i
            )
        }
    };

    tracing::info!("start to bench hashbrown::HashMap");
    let mut join = Vec::with_capacity(num_readers + num_writers);
    let map = Sharded {
        inner: vec![hashbrown::HashMap::with_capacity(50_000_000 / 128); 128]
            .into_iter()
            .map(|x| parking_lot::RwLock::new(x))
            .collect(),
        phantom: std::marker::PhantomData::default(),
    };
    let map = Arc::new(map);
    let start = std::time::Instant::now();
    let end = start + dur;
    join.extend((0..num_readers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, false))
    }));
    join.extend((0..num_writers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, true))
    }));
    let (wres, rres): (Vec<_>, _) = join
        .drain(..)
        .map(|jh| jh.join().unwrap())
        .partition(|&(write, _)| write);
    stat("write", wres);
    stat("read", rres);
    tracing::info!("bench hashbrown::HashMap complete\r");

    tracing::info!("start to bench griddle::HashMap");
    let mut join = Vec::with_capacity(num_readers + num_writers);
    let map = Sharded {
        inner: vec![griddle::HashMap::with_capacity(50_000_000 / 128); 128]
            .into_iter()
            .map(|x| parking_lot::RwLock::new(x))
            .collect(),
        phantom: std::marker::PhantomData::default(),
    };
    let map = Arc::new(map);
    let start = std::time::Instant::now();
    let end = start + dur;
    join.extend((0..num_readers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, false))
    }));
    join.extend((0..num_writers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, true))
    }));
    let (wres, rres): (Vec<_>, _) = join
        .drain(..)
        .map(|jh| jh.join().unwrap())
        .partition(|&(write, _)| write);
    stat("write", wres);
    stat("read", rres);
    tracing::info!("bench griddle::HashMap complete\r");

    tracing::info!("start to bench hashlink::LinkedHashMap");
    let mut join = Vec::with_capacity(num_readers + num_writers);
    let map = Sharded {
        inner: vec![hashlink::LinkedHashMap::with_capacity(50_000_000 / 128); 128]
            .into_iter()
            .map(|x| parking_lot::RwLock::new(x))
            .collect(),
        phantom: std::marker::PhantomData::default(),
    };
    let map = Arc::new(map);
    let start = std::time::Instant::now();
    let end = start + dur;
    join.extend((0..num_readers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, false))
    }));
    join.extend((0..num_writers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, true))
    }));
    let (wres, rres): (Vec<_>, _) = join
        .drain(..)
        .map(|jh| jh.join().unwrap())
        .partition(|&(write, _)| write);
    stat("write", wres);
    stat("read", rres);
    tracing::info!("bench hashlink::LinkedHashMap complete\r");

    tracing::info!("start to run with normal distribution\r");
    tracing::info!("start to bench PersiaEvictionMapMarkI");
    let mut join = Vec::with_capacity(num_readers + num_writers);
    let map: PersiaEvictionMapMarkI<u64, ArcEntry> = PersiaEvictionMapMarkI::new(50_000_000);
    let map = Arc::new(map);
    let start = std::time::Instant::now();
    let end = start + dur;
    join.extend((0..num_readers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, false))
    }));
    join.extend((0..num_writers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, true))
    }));
    let (wres, rres): (Vec<_>, _) = join
        .drain(..)
        .map(|jh| jh.join().unwrap())
        .partition(|&(write, _)| write);
    stat("write", wres);
    stat("read", rres);
    tracing::info!("bench PersiaEvictionMapMarkI complete\r");

    tracing::info!("start to bench PersiaEvictionMapMarkII");
    let mut join = Vec::with_capacity(num_readers + num_writers);
    let map: PersiaEvictionMapMarkII<u64, ArcEntry> = PersiaEvictionMapMarkII::new(50_000_000);
    let map = Arc::new(map);
    let start = std::time::Instant::now();
    let end = start + dur;
    join.extend((0..num_readers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, false))
    }));
    join.extend((0..num_writers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, true))
    }));
    let background = {
        let map = map.clone();
        thread::spawn(move || {
            while std::time::Instant::now() < end {
                std::thread::sleep(std::time::Duration::from_secs(10));
                map.evict();
            }
        })
    };
    let (wres, rres): (Vec<_>, _) = join
        .drain(..)
        .map(|jh| jh.join().unwrap())
        .partition(|&(write, _)| write);
    stat("write", wres);
    stat("read", rres);
    let _ = background.join();
    tracing::info!("bench PersiaEvictionMapMarkII complete\r");

    tracing::info!("start to bench PersiaEvictionMapMarkIII");
    let mut join = Vec::with_capacity(num_readers + num_writers);
    let map: PersiaEvictionMapMarkIII<u64, ArcEntry> = PersiaEvictionMapMarkIII::new(50_000_000);
    let map = Arc::new(map);
    let start = std::time::Instant::now();
    let end = start + dur;
    join.extend((0..num_readers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, false))
    }));
    join.extend((0..num_writers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, true))
    }));
    let (wres, rres): (Vec<_>, _) = join
        .drain(..)
        .map(|jh| jh.join().unwrap())
        .partition(|&(write, _)| write);
    stat("write", wres);
    stat("read", rres);
    tracing::info!("bench PersiaEvictionMapMarkIII complete\r");

    tracing::info!("start to bench PersiaEvictionMapMarkIV");
    let mut join = Vec::with_capacity(num_readers + num_writers);
    let map: PersiaEvictionMapMarkIV<u64, ArcEntry> = PersiaEvictionMapMarkIV::new(50_000_000);
    let map = Arc::new(map);
    let start = std::time::Instant::now();
    let end = start + dur;
    join.extend((0..num_readers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, false))
    }));
    join.extend((0..num_writers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, true))
    }));
    let (wres, rres): (Vec<_>, _) = join
        .drain(..)
        .map(|jh| jh.join().unwrap())
        .partition(|&(write, _)| write);
    stat("write", wres);
    stat("read", rres);
    tracing::info!("bench PersiaEvictionMapMarkIV complete\r");

    tracing::info!("start to bench PersiaEvictionMapMarkV");
    let mut join = Vec::with_capacity(num_readers + num_writers);
    let map: PersiaEvictionMapMarkV<u64, ArcEntry> = PersiaEvictionMapMarkV::new(50_000_000);
    let map = Arc::new(map);
    let start = std::time::Instant::now();
    let end = start + dur;
    join.extend((0..num_readers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, false))
    }));
    join.extend((0..num_writers).map(|_| {
        let map = map.clone();
        thread::spawn(move || run_with_normal(map, end, true))
    }));
    let (wres, rres): (Vec<_>, _) = join
        .drain(..)
        .map(|jh| jh.join().unwrap())
        .partition(|&(write, _)| write);
    stat("write", wres);
    stat("read", rres);
    tracing::info!("bench PersiaEvictionMapMarkV complete\r");
}
