#include "cuda_runtime.h"
#include <ATen/Functions.h>
#include <bits/stdint-uintn.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <chrono>
#include <pybind11/pybind11.h>
#include <thread>
#include <torch/extension.h>
#include <vector>
#include <nccl.h>
#include "ThreadPool.h"
#include "base64.h"
#include "dbg.h"

#define CUDACHECK(cmd)                                                    \
    do                                                                    \
    {                                                                     \
        cudaError_t e = cmd;                                              \
        if (e != cudaSuccess)                                             \
        {                                                                 \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                                \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define NCCLCHECK(cmd)                                                    \
    do                                                                    \
    {                                                                     \
        ncclResult_t r = cmd;                                             \
        if (r != ncclSuccess)                                             \
        {                                                                 \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
                   ncclGetErrorString(r));                                \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

static ncclComm_t NCCL_COMM;
static int RANK;
static int N_RANKS;
static int DEVICE_ID;
static float *DUMMY_BUF;
ThreadPool THREAD_POOL(4);

torch::Tensor pointer_to_tensor_f32(uint64_t data_ptr, std::vector<int64_t> shape,
                                    bool requires_grad)
{
    auto options = at::TensorOptions()
                       .device(at::kCUDA)
                       .dtype(at::kFloat)
                       .requires_grad(requires_grad);
    return torch::from_blob(reinterpret_cast<void *>(data_ptr),
                            at::IntList(shape), options);
}

std::vector<torch::Tensor> pointers_to_tensors_f32(
    const std::vector<std::tuple<uint64_t, std::vector<int64_t>>> &entries,
    bool requires_grad)
{
    std::vector<torch::Tensor> outputs;
    uint64_t data_ptr;
    std::vector<int64_t> shape;
    for (auto entry : entries)
    {
        std::tie(data_ptr, shape) = entry;
        outputs.push_back(pointer_to_tensor_f32(data_ptr, shape, requires_grad));
    }
    return outputs;
}

torch::Tensor pointer_to_tensor_f16(uint64_t data_ptr,
                                    std::vector<int64_t> shape,
                                    bool requires_grad)
{
    auto options =
        at::TensorOptions().device(at::kCUDA).dtype(at::kHalf).requires_grad(
            requires_grad);
    return torch::from_blob(reinterpret_cast<void *>(data_ptr),
                            at::IntList(shape), options);
}

std::vector<torch::Tensor> pointers_to_tensors_f16(
    const std::vector<std::tuple<uint64_t, std::vector<int64_t>>> &entries,
    bool requires_grad)
{
    std::vector<torch::Tensor> outputs;
    uint64_t data_ptr;
    std::vector<int64_t> shape;
    for (auto entry : entries)
    {
        std::tie(data_ptr, shape) = entry;
        outputs.push_back(pointer_to_tensor_f16(data_ptr, shape, requires_grad));
    }
    return outputs;
}

torch::Tensor pointer_to_tensor_long(uint64_t data_ptr, std::vector<int64_t> shape)
{
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong).requires_grad(false);
    return torch::from_blob(reinterpret_cast<void *>(data_ptr), at::IntList(shape), options);
}

std::vector<torch::Tensor> pointers_to_tensors_long(
    const std::vector<std::tuple<uint64_t, std::vector<int64_t>>> &entries)
{
    std::vector<torch::Tensor> outputs;
    uint64_t data_ptr;
    std::vector<int64_t> shape;
    for (auto entry : entries)
    {
        std::tie(data_ptr, shape) = entry;
        outputs.push_back(pointer_to_tensor_long(data_ptr, shape));
    }
    return outputs;
}

torch::Tensor pointer_to_tensor_i32(uint64_t data_ptr, std::vector<int64_t> shape)
{
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kInt).requires_grad(false);
    return torch::from_blob(reinterpret_cast<void *>(data_ptr), at::IntList(shape), options);
}

std::vector<torch::Tensor> pointers_to_tensors_i32(
    const std::vector<std::tuple<uint64_t, std::vector<int64_t>>> &entries)
{
    std::vector<torch::Tensor> outputs;
    uint64_t data_ptr;
    std::vector<int64_t> shape;
    for (auto entry : entries)
    {
        std::tie(data_ptr, shape) = entry;
        outputs.push_back(pointer_to_tensor_i32(data_ptr, shape));
    }
    return outputs;
}

std::string create_nccl_unique_id_base64()
{
    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    return base64_encode(reinterpret_cast<const unsigned char *>(id.internal), NCCL_UNIQUE_ID_BYTES);
}

void init_distributed(std::string nccl_unique_id_base64, int rank, int nranks, int device_id)
{
    RANK = rank;
    N_RANKS = nranks;
    DEVICE_ID = device_id;
    CUDACHECK(cudaSetDevice(DEVICE_ID));
    auto nccl_unique_id = b64decode(nccl_unique_id_base64.data(), nccl_unique_id_base64.length());
    ncclUniqueId nccl_unique_id_container;
    std::copy(std::begin(nccl_unique_id), std::end(nccl_unique_id), std::begin(nccl_unique_id_container.internal));
    // dbg(base64_encode(reinterpret_cast<const unsigned char *>(nccl_unique_id_container.internal), NCCL_UNIQUE_ID_BYTES));
    NCCLCHECK(ncclCommInitRank(&NCCL_COMM, N_RANKS, nccl_unique_id_container, RANK));
    CUDACHECK(cudaMalloc(&DUMMY_BUF, sizeof(float)));
}

void barrier()
{
    at::DeviceGuard guard(at::Device(at::kCUDA, DEVICE_ID));
    CUDACHECK(cudaSetDevice(DEVICE_ID));
    auto stream = at::cuda::getDefaultCUDAStream(DEVICE_ID);

    // cudaStream_t* stream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
    // CUDACHECK(cudaStreamCreate(stream));

    NCCLCHECK(ncclAllReduce(DUMMY_BUF, DUMMY_BUF, 1, ncclFloat32, ncclSum, NCCL_COMM, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
}

void async_allreduce_func(void *data_ptr, int64_t num_elements, int interval)
{
    at::DeviceGuard guard(at::Device(at::kCUDA, DEVICE_ID));
    CUDACHECK(cudaSetDevice(DEVICE_ID));
    auto input_tensor = torch::from_blob(data_ptr, {num_elements}, at::kCUDA);
    auto copy1 = torch::zeros_like(input_tensor, at::kCUDA);
    auto copy2 = torch::zeros_like(copy1, at::kCUDA);

    // cudaStream_t stream;
    // CUDACHECK(cudaStreamCreate(&stream));
    auto stream = at::cuda::getStreamFromPool();

    // initial broadcast model
    NCCLCHECK(ncclBroadcast(input_tensor.data_ptr(), input_tensor.data_ptr(), num_elements, ncclFloat32, 0, NCCL_COMM,
                            stream));

    int num_averages = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (true)
    {
        at::cuda::CUDAStreamGuard guard(stream);
        copy1.copy_(input_tensor);
        // std::cout << " rank " << my_rank << " copy2 " << copy2 << std::endl;
        NCCLCHECK(ncclAllReduce(copy1.data_ptr(), copy2.data_ptr(), num_elements, ncclFloat32, ncclSum, NCCL_COMM,
                                stream));
        // MPICHECK(MPI_Allreduce(copy1.data_ptr(), copy2.data_ptr(), num_elements,
        // MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD)); std::cout << " rank " << my_rank <<
        // " copy2 after allreduce " << copy2 << std::endl;
        copy2.div_((float)N_RANKS);
        // std::cout << " rank " << my_rank << " copy2 after average " << copy2 <<
        // std::endl;
        copy1.sub_(copy2);
        // std::cout << " rank " << my_rank << " copy1 diff " << copy1 << std::endl;
        input_tensor.sub_(copy1);
        // std::cout << " rank " << my_rank << " input tensor " << input_tensor <<
        // std::endl; std::cout << " num_averages " << num_averages << std::endl;
        num_averages += 1;
        if (num_averages % 100 == 0)
        {
            std::cout
                << " num sync per second "
                << double(100000000000) /
                       double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::high_resolution_clock::now() - start)
                                  .count())
                << std::endl;
            start = std::chrono::high_resolution_clock::now();
            num_averages = 0;
        }
        CUDACHECK(cudaStreamSynchronize(stream));
        std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    }
}

void start_synchronization(torch::Tensor input_tensor, int interval)
{
    // std::cout << input_tensor << std::endl;
    THREAD_POOL.enqueue(async_allreduce_func, input_tensor.data_ptr(), input_tensor.numel(), interval);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ptr_to_tensor_f32", &pointer_to_tensor_f32, "pointer to tensor");
    m.def("ptrs_to_tensors_f32", &pointers_to_tensors_f32, "pointers to tensors");
    m.def("ptr_to_tensor_f16", &pointer_to_tensor_f16, "pointer to tensor");
    m.def("ptrs_to_tensors_f16", &pointers_to_tensors_f16, "pointers to tensors");
    m.def("ptr_to_tensor_i32", &pointer_to_tensor_i32, "pointer to tensor");
    m.def("ptrs_to_tensors_i32", &pointers_to_tensors_i32, "pointers to tensors");
    m.def("ptr_to_tensor_long", &pointer_to_tensor_long, "pointer to tensor");
    m.def("ptrs_to_tensors_long", &pointers_to_tensors_long, "pointers to tensors");
    m.def("start_synchronization", &start_synchronization, "start tensor synchronization");
    m.def("create_nccl_unique_id_base64", &create_nccl_unique_id_base64,
          "create nccl unique id for distributed computing");
    m.def("init_distributed", &init_distributed, "initialize distributed training mode");
    m.def("barrier", &barrier, "sync barrier");
}
