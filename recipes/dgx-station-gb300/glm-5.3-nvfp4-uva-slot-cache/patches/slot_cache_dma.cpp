// Host-only CUDA runtime extension: no SM copy kernels.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <unordered_set>

namespace py = pybind11;

static void check(cudaError_t error) {
    TORCH_CHECK(error == cudaSuccess, "slot-cache DMA: ", cudaGetErrorString(error));
}

struct Bank {
    torch::Tensor source, destination;  // retain storage for this plan's lifetime
    const char* src;
    char* dst;
    size_t row_bytes;
    cudaMemcpyKind kind;
    cudaPointerAttributes attributes;
};

class RowCopier {
    std::vector<Bank> banks;
    int device;
    int64_t experts, slots;

public:
    RowCopier(std::vector<torch::Tensor> sources, std::vector<torch::Tensor> destinations) {
        TORCH_CHECK(!sources.empty() && sources.size() == destinations.size(), "bank count mismatch");
        TORCH_CHECK(destinations[0].is_cuda(), "destination must be CUDA memory");
        device = destinations[0].get_device();
        const c10::cuda::CUDAGuard guard(device);
        TORCH_CHECK(sources[0].dim() >= 1 && destinations[0].dim() >= 1, "banks need a row dimension");
        experts = sources[0].size(0);
        slots = destinations[0].size(0);
        TORCH_CHECK(experts > 0 && slots > 0, "empty banks are unsupported");
        for (size_t i = 0; i < sources.size(); ++i) {
            auto src = sources[i], dst = destinations[i];
            TORCH_CHECK(src.dim() >= 1 && dst.dim() == src.dim(), "bank rank mismatch");
            TORCH_CHECK(src.is_contiguous() && dst.is_contiguous(), "DMA banks must be contiguous; no implicit clones");
            TORCH_CHECK(src.size(0) == experts && dst.size(0) == slots, "bank row count mismatch");
            TORCH_CHECK(src.scalar_type() == dst.scalar_type(), "bank dtype mismatch");
            TORCH_CHECK(dst.is_cuda() && dst.get_device() == device, "destination device mismatch");
            for (int64_t d = 1; d < src.dim(); ++d)
                TORCH_CHECK(src.size(d) == dst.size(d), "bank row shape mismatch");
            const size_t bytes = src.numel() / experts * src.element_size();
            TORCH_CHECK(bytes > 0, "empty expert rows are unsupported");
            cudaPointerAttributes sa{}, da{};
            check(cudaPointerGetAttributes(&sa, src.data_ptr()));
            check(cudaPointerGetAttributes(&da, dst.data_ptr()));
            TORCH_CHECK(da.type == cudaMemoryTypeDevice && da.device == device,
                        "destination must be device allocation on selected GPU");
            const char* pointer = nullptr;
            cudaMemcpyKind kind;
            if (sa.type == cudaMemoryTypeHost) {
                // Use CUDA's host alias even if Torch describes a mapped UVA view as CUDA.
                TORCH_CHECK(sa.hostPointer != nullptr, "pinned source has no host alias");
                pointer = static_cast<const char*>(sa.hostPointer);
                kind = cudaMemcpyHostToDevice;
            } else if (sa.type == cudaMemoryTypeDevice) {
                TORCH_CHECK(sa.device == device, "peer-device sources are unsupported");
                pointer = static_cast<const char*>(src.data_ptr());
                kind = cudaMemcpyDeviceToDevice;
            } else {
                TORCH_CHECK(false, "unsupported source memory type ", int(sa.type),
                            "; require CUDA-registered pinned host or device memory, not managed/pageable memory");
            }
            auto dst_ptr = static_cast<char*>(dst.data_ptr());
            const auto sb = reinterpret_cast<uintptr_t>(src.data_ptr());
            const auto db = reinterpret_cast<uintptr_t>(dst_ptr);
            TORCH_CHECK(sb + bytes * experts <= db || db + bytes * slots <= sb, "overlapping banks");
            banks.push_back({src, dst, pointer, dst_ptr, bytes, kind, sa});
        }
    }

    py::list describe() const {
        py::list result;
        for (const auto& b : banks) {
            py::dict info;
            info["source_type"] = b.attributes.type == cudaMemoryTypeHost ? "host" : "device";
            info["source_device"] = b.attributes.device;
            info["has_host_alias"] = b.attributes.hostPointer != nullptr;
            info["has_device_alias"] = b.attributes.devicePointer != nullptr;
            info["copy_kind"] = b.kind == cudaMemcpyHostToDevice ? "H2D" : "D2D";
            info["row_bytes"] = b.row_bytes;
            result.append(info);
        }
        return result;
    }

    void enqueue(std::vector<int64_t> src_ids, std::vector<int64_t> dst_ids, uintptr_t stream_ptr) {
        TORCH_CHECK(src_ids.size() == dst_ids.size(), "copy index count mismatch");
        // Validate the entire batch before submitting any copy.
        std::unordered_set<int64_t> used;
        for (size_t i = 0; i < src_ids.size(); ++i) {
            TORCH_CHECK(src_ids[i] >= 0 && src_ids[i] < experts, "expert index out of range");
            TORCH_CHECK(dst_ids[i] >= 0 && dst_ids[i] < slots, "slot index out of range");
            TORCH_CHECK(used.insert(dst_ids[i]).second, "duplicate destination slot");
        }
        const c10::cuda::CUDAGuard guard(device);
        auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        unsigned int flags = 0;
        check(cudaStreamGetFlags(stream, &flags));
        TORCH_CHECK(flags & cudaStreamNonBlocking, "DMA requires a nonblocking stream");
        cudaStreamCaptureStatus status;
        check(cudaStreamIsCapturing(stream, &status));
        TORCH_CHECK(status == cudaStreamCaptureStatusNone, "DMA backend requires eager execution");
        for (size_t i = 0; i < src_ids.size(); ++i)
            for (const auto& b : banks)
                check(cudaMemcpyAsync(b.dst + dst_ids[i] * b.row_bytes,
                                      b.src + src_ids[i] * b.row_bytes,
                                      b.row_bytes, b.kind, stream));
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<RowCopier>(m, "RowCopier")
        .def(py::init<std::vector<torch::Tensor>, std::vector<torch::Tensor>>())
        .def("describe", &RowCopier::describe)
        .def("enqueue", &RowCopier::enqueue);
}
