use super::cuda_stream_pool::CudaStreamPtr;
use super::resource_pool::{Allocatable, Pool};

use cuda_runtime_sys as cuda;
use persia_libs::once_cell;

pub static CUDA_EVENT_POOL: once_cell::sync::Lazy<Pool<CudaEventPtr>> =
    once_cell::sync::Lazy::new(|| return Pool::new());

#[derive(Debug)]
#[must_use = "d2h or h2d memcpy should synchronize manually"]
pub struct CudaEventPtr {
    pub inner: cuda::cudaEvent_t,
}

unsafe impl Send for CudaEventPtr {}

impl CudaEventPtr {
    pub fn record(&self, stream: CudaStreamPtr) {
        let result = unsafe { cuda::cudaEventRecord(self.inner, stream.inner) };
        assert_eq!(result, cuda::cudaError::cudaSuccess);
    }

    pub fn synchronize(&self) {
        let result = unsafe { cuda::cudaEventSynchronize(self.inner) };
        assert_eq!(result, cuda::cudaError::cudaSuccess);
    }
}

impl Allocatable for CudaEventPtr {
    fn new(_size: usize) -> Self {
        let mut event = std::ptr::null_mut();
        let result = unsafe { cuda::cudaEventCreate(&mut event as *mut cuda::cudaEvent_t) };
        assert_eq!(result, cuda::cudaError::cudaSuccess);

        return CudaEventPtr { inner: event };
    }

    fn size(&self) -> usize {
        0
    }
}

impl Drop for CudaEventPtr {
    fn drop(&mut self) {
        CUDA_EVENT_POOL.recycle(CudaEventPtr { inner: self.inner });
    }
}
