use cuda_runtime_sys as cuda;
use resource_pool::{Allocatable, Pool};

pub static CUDA_DEVICE_MEMORY_POOL: once_cell::sync::Lazy<Pool<CudaMallocPtr>> =
    once_cell::sync::Lazy::new(|| return Pool::new());

/// We are not going to implement Drop trait for it, since we cannot recycle here, we need to recycle after the training process call free.
#[derive(Debug)]
pub struct CudaMallocPtr {
    pub inner: *mut std::os::raw::c_void,
    pub num_bytes: usize,
}

impl Drop for CudaMallocPtr {
    fn drop(&mut self) {
        tracing::debug!("cuda pinned memory recycled, size {}", self.num_bytes);
        CUDA_DEVICE_MEMORY_POOL.recycle(CudaMallocPtr {
            inner: self.inner,
            num_bytes: self.num_bytes,
        });
    }
}

unsafe impl Send for CudaMallocPtr {}

impl Allocatable for CudaMallocPtr {
    fn new(size: usize) -> Self {
        let mut data_ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
        let result =
            unsafe { cuda::cudaMalloc(&mut data_ptr as *mut *mut std::os::raw::c_void, size) };
        assert_eq!(result, cuda::cudaError::cudaSuccess);
        tracing::debug!("allocating cuda pinned memory, size {}", size);
        return CudaMallocPtr {
            inner: data_ptr,
            num_bytes: size,
        };
    }

    fn size(&self) -> usize {
        self.num_bytes
    }
}
