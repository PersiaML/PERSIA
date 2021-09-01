#include <vector>

#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>

#include <torch/extension.h>
#include <ATen/Functions.h>
#include <pybind11/pybind11.h>


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
    const std::vector<std::tuple<uint64_t, std::vector<int64_t> > > &entries,
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
    const std::vector<std::tuple<uint64_t, std::vector<int64_t> > > &entries,
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
    const std::vector<std::tuple<uint64_t, std::vector<int64_t> > > &entries)
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
    const std::vector<std::tuple<uint64_t, std::vector<int64_t> > > &entries)
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
}
