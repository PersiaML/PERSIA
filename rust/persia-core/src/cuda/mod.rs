use cuda_runtime_sys as cuda;

pub mod cuda_event_pool;
pub mod cuda_memory_pool;
pub mod cuda_stream_pool;
pub mod pinned_memory_pool;
pub mod resource_pool;
pub mod utils;

use cuda_event_pool::{CudaEventPtr, CUDA_EVENT_POOL};
use cuda_memory_pool::{CudaMallocPtr, CUDA_DEVICE_MEMORY_POOL};
use cuda_stream_pool::CUDA_STREAM_POOL;
use pinned_memory_pool::{PinnedMemoryPtr, PINNED_MEMORY_POOL};

use persia_libs::{anyhow::Result, tracing};
use persia_speedy::{Readable, Writable};

use crate::tensor::{CPUStorage, DTypeImpl};

pub fn set_device(card_index: i32) {
    let result = unsafe { cuda::cudaSetDevice(card_index) };
    assert_eq!(result, cuda::cudaError::cudaSuccess);
}

#[derive(Debug, Readable, Writable)]
pub struct GPUStorage {
    #[speedy(skip)]
    pub ptr: CudaMallocPtr,
    pub shape: Vec<usize>,
    #[speedy(skip)]
    pub event: CudaEventPtr,
    #[speedy(skip)]
    pub host_ptr: PinnedMemoryPtr,
    pub dtype: DTypeImpl,
    pub is_ready: bool,
}

impl GPUStorage {
    pub fn new(storage: CPUStorage, shape: Vec<usize>) -> Result<Self> {
        unsafe {
            let stream = CUDA_STREAM_POOL.allocate(0);
            let mut storage = storage;
            let dtype = storage.get_dtype();
            let byte_count = shape.iter().product::<usize>() * dtype.get_type_size();

            tracing::debug!("tensor shape is: {:?}, bytes: {:?}", &shape, &byte_count);

            let host_ptr = storage.get_raw_ptr();
            let pinned_host_ptr = PINNED_MEMORY_POOL.allocate(byte_count);
            std::ptr::copy_nonoverlapping(host_ptr, pinned_host_ptr.inner, byte_count);

            let data_ptr = CUDA_DEVICE_MEMORY_POOL.allocate(byte_count);
            let result = cuda::cudaMemcpyAsync(
                data_ptr.inner,
                pinned_host_ptr.inner,
                byte_count,
                cuda::cudaMemcpyKind::cudaMemcpyHostToDevice,
                stream.inner,
            );
            assert_eq!(
                result,
                cuda::cudaError::cudaSuccess,
                "data_ptr {:?}, pinned_host_ptr: {:?}, byte_count: {:?}, stream: {:?}",
                data_ptr,
                pinned_host_ptr,
                byte_count,
                stream
            );
            let event = CUDA_EVENT_POOL.allocate(0);
            event.record(stream);

            Ok(GPUStorage {
                ptr: data_ptr,
                shape: shape.clone(),
                host_ptr: pinned_host_ptr,
                event,
                dtype,
                is_ready: false,
            })
        }
    }

    pub fn sync_event(&mut self) {
        if !self.is_ready {
            self.event.synchronize();
            self.is_ready = true;
        }
    }

    pub fn get_raw_ptr(&mut self) -> *mut std::os::raw::c_void {
        self.sync_event();
        self.ptr.inner
    }

    pub fn get_dtype(&self) -> DTypeImpl {
        self.dtype.clone()
    }
}
