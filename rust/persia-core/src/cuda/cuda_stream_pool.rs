use super::resource_pool::{Allocatable, Pool};

use cuda_runtime_sys as cuda;
use persia_libs::once_cell;

pub static CUDA_STREAM_POOL: once_cell::sync::Lazy<Pool<CudaStreamPtr>> =
    once_cell::sync::Lazy::new(|| return Pool::new());

#[derive(Debug)]
pub struct CudaStreamPtr {
    pub inner: cuda::cudaStream_t,
}

unsafe impl Send for CudaStreamPtr {}

impl Allocatable for CudaStreamPtr {
    fn new(_size: usize) -> Self {
        let mut stream = std::ptr::null_mut();
        let result = unsafe {
            cuda::cudaStreamCreateWithFlags(
                &mut stream as *mut cuda::cudaStream_t,
                cuda::cudaStreamNonBlocking,
            )
        };
        assert_eq!(result, cuda::cudaError::cudaSuccess);
        return CudaStreamPtr { inner: stream };
    }

    fn size(&self) -> usize {
        0
    }
}

impl Drop for CudaStreamPtr {
    fn drop(&mut self) {
        CUDA_STREAM_POOL.recycle(CudaStreamPtr { inner: self.inner });
    }
}
