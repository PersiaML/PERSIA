use cuda_runtime_sys as cuda;

pub mod cuda_event_pool;
pub mod cuda_memory_pool;
pub mod cuda_stream_pool;
pub mod pinned_memory_pool;
pub mod utils;

use cuda_event_pool::CudaEventPtr;
use cuda_memory_pool::CudaMallocPtr;
use pinned_memory_pool::PinnedMemoryPtr;

use persia_embedding_datatypes::TensorDtype;

pub fn set_device(card_index: i32) {
    let result = unsafe { cuda::cudaSetDevice(card_index) };
    assert_eq!(result, cuda::cudaError::cudaSuccess);
}

#[derive(Debug)]
pub enum AsyncEmbeddingOnCuda {
    Raw(AsyncRawEmbeddingOnCuda),
    Sum(AsyncTensorOnCuda),
}

#[derive(Debug)]
pub struct AsyncRawEmbeddingOnCuda {
    pub tensor: AsyncTensorOnCuda,
    pub index: AsyncTensorOnCuda, // store the origin Vec<Sample> mapping to tensos
    pub non_empty_index: AsyncTensorOnCuda,
    pub samples_id_num: Vec<usize>,
}

#[derive(Debug)]
pub struct AsyncTensorOnCuda {
    pub name: String,
    pub ptr: CudaMallocPtr,
    pub shape: [usize; 2],
    pub event: CudaEventPtr,
    pub host_ptr: PinnedMemoryPtr,
    pub dtype: TensorDtype,
}

impl AsyncTensorOnCuda {
    pub fn wait_ready(&self) {
        self.event.synchronize();
    }
}
