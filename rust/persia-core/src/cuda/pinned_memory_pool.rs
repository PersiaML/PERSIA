use super::resource_pool::{Allocatable, Pool};

use cuda_runtime_sys as cuda;
use persia_libs::once_cell;

pub static PINNED_MEMORY_POOL: once_cell::sync::Lazy<Pool<PinnedMemoryPtr>> =
    once_cell::sync::Lazy::new(|| return Pool::new());

#[derive(Debug)]
pub struct PinnedMemoryPtr {
    pub inner: *mut std::os::raw::c_void,
    pub num_bytes: usize,
}

impl Drop for PinnedMemoryPtr {
    fn drop(&mut self) {
        PINNED_MEMORY_POOL.recycle(PinnedMemoryPtr {
            inner: self.inner,
            num_bytes: self.num_bytes,
        });
    }
}

unsafe impl Send for PinnedMemoryPtr {}

impl Allocatable for PinnedMemoryPtr {
    fn new(size: usize) -> Self {
        let mut data_ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
        let result =
            unsafe { cuda::cudaMallocHost(&mut data_ptr as *mut *mut std::os::raw::c_void, size) };
        assert_eq!(result, cuda::cudaError::cudaSuccess);
        return PinnedMemoryPtr {
            inner: data_ptr,
            num_bytes: size,
        };
    }

    fn size(&self) -> usize {
        self.num_bytes
    }
}

impl PinnedMemoryPtr {
    pub fn as_slice<T>(&self, num_elements: usize) -> &[T] {
        assert!(
            num_elements * std::mem::size_of::<T>() <= self.num_bytes,
            "num_elements {}, num_bytes {}",
            num_elements,
            self.num_bytes
        );
        unsafe { std::slice::from_raw_parts(self.inner as *const T, num_elements) }
    }
}
