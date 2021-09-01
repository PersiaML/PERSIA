use std::sync::atomic::Ordering;

use persia_common::utils::ChannelPair;
use persia_libs::tracing;

pub trait Allocatable {
    fn new(size: usize) -> Self;
    fn size(&self) -> usize;
}

pub struct Pool<T: Allocatable> {
    /// each entry represents an allocation queue of 2**n bytes block
    sub_pools: [SubPool<T>; 30],
}

impl<T: Allocatable> Default for Pool<T> {
    fn default() -> Self {
        Pool::new()
    }
}

impl<T> Pool<T>
where
    T: Allocatable,
{
    pub fn new() -> Self {
        Self {
            sub_pools: arr_macro::arr![SubPool::new(); 30],
        }
    }

    fn get_pool_location(size: usize) -> usize {
        if size == 0 {
            0
        } else {
            (size.next_power_of_two().trailing_zeros() + 1) as usize
        }
    }

    pub fn allocate(&self, size: usize) -> T {
        let pool = &self.sub_pools[Self::get_pool_location(size)];
        pool.allocate(size.next_power_of_two())
    }

    pub fn recycle(&self, item: T) {
        let pool = &self.sub_pools[Self::get_pool_location(item.size())];
        pool.recycle(item);
    }
}

pub struct SubPool<T: Allocatable> {
    channel: ChannelPair<T>,
    num_allocated: std::sync::atomic::AtomicU32,
}

impl<T> Default for SubPool<T>
where
    T: Allocatable,
{
    fn default() -> Self {
        let channel = ChannelPair::new_unbounded();
        Self {
            channel,
            num_allocated: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

impl<T> SubPool<T>
where
    T: Allocatable,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allocate(&self, size: usize) -> T {
        if let Ok(allocated) = self.channel.receiver.try_recv() {
            allocated
        } else {
            tracing::debug!(
                message = "no available resource in pool, creating a new one",
                type_info = tracing::field::debug(std::any::type_name::<T>()),
                num_resources = self.num_allocated.load(Ordering::Acquire)
            );
            let allocated = T::new(size);
            self.num_allocated.fetch_add(1, Ordering::AcqRel);
            allocated
        }
    }

    pub fn recycle(&self, item: T) {
        self.channel.sender.send(item).unwrap();
    }
}
