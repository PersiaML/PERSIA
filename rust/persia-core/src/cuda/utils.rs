use crate::cuda::cuda_event_pool::{CudaEventPtr, CUDA_EVENT_POOL};
use crate::cuda::cuda_stream_pool::CUDA_STREAM_POOL;

use cuda_runtime_sys as cuda;
use persia_libs::anyhow::Result;

pub fn cuda_d2h(
    num_bytes: usize,
    data_ptr: *mut std::os::raw::c_void,
    host_ptr: *mut std::os::raw::c_void,
) -> Result<CudaEventPtr> {
    unsafe {
        let stream = CUDA_STREAM_POOL.allocate(0);
        let result = cuda::cudaMemcpyAsync(
            host_ptr,
            data_ptr,
            num_bytes,
            cuda::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            stream.inner,
        );
        assert_eq!(result, cuda::cudaError::cudaSuccess);
        let event = CUDA_EVENT_POOL.allocate(0);
        event.record(stream);
        Ok(event)
    }
}
