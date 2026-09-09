/**
 * CUDA Gaussian Splatting - Dispatch Layer
 *
 * This file provides the host-side dispatch and public API for the CUDA splatting backend:
 *
 * 1. Template instantiations for all DIM × InputDType combinations
 * 2. Input validation
 * 3. Forward/backward dispatch (templated on InputDType for FP32/FP16 unification)
 * 4. Public API: forward(), forward_fp16(), backward(), backward_fp16()
 *
 * The kernels themselves are in kernels_core.cuh.
 * Launch wrappers are in kernel_launchers.cuh (already templated on InputDType).
 *
 * See README.md and OPTIMIZATION_REPORT.md for algorithm details.
 */

#include "cuda_splatting.h"
#include "utils.cuh"
#include "kernel_launchers.cuh"

#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <type_traits>

// =============================================================================
// DISPATCH HELPERS
// =============================================================================

/**
 * Dispatch a body for each supported dimension (2-8).
 * Inside the body, `D` is available as a constexpr int for template parameters.
 */
#define DIM_DISPATCH(dim, ...) \
    switch (dim) { \
        case 2: { constexpr int D = 2; __VA_ARGS__; } break; \
        case 3: { constexpr int D = 3; __VA_ARGS__; } break; \
        case 4: { constexpr int D = 4; __VA_ARGS__; } break; \
        case 5: { constexpr int D = 5; __VA_ARGS__; } break; \
        case 6: { constexpr int D = 6; __VA_ARGS__; } break; \
        case 7: { constexpr int D = 7; __VA_ARGS__; } break; \
        case 8: { constexpr int D = 8; __VA_ARGS__; } break; \
        default: TORCH_CHECK(false, "Unsupported dimension: ", dim); \
    }

/**
 * Extract a typed data pointer from a torch::Tensor.
 * Maps InputDType to the correct data_ptr<>() call:
 * - float: data_ptr<float>()
 * - __half: reinterpret_cast from data_ptr<at::Half>() (binary compatible)
 */
template <typename InputDType>
inline const InputDType* get_data_ptr(const torch::Tensor& t);

template <>
inline const float* get_data_ptr<float>(const torch::Tensor& t) {
    return t.data_ptr<float>();
}

template <>
inline const __half* get_data_ptr<__half>(const torch::Tensor& t) {
    return reinterpret_cast<const __half*>(t.data_ptr<at::Half>());
}

// =============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS (FP32 + FP16)
// =============================================================================
// Each macro instantiates both float and __half variants for dimensions 2-8.

// Splat-centric forward kernel instantiations
#define INSTANTIATE_SPLAT_FWD(D) \
    template void launch_rasterize_forward_splat_centric<D, float>(const float*, const float*, const float*, int, const int*, float, float, float*, cudaStream_t); \
    template void launch_rasterize_forward_splat_centric<D, __half>(const __half*, const __half*, const __half*, int, const int*, float, float, float*, cudaStream_t);

INSTANTIATE_SPLAT_FWD(2)
INSTANTIATE_SPLAT_FWD(3)
INSTANTIATE_SPLAT_FWD(4)
INSTANTIATE_SPLAT_FWD(5)
INSTANTIATE_SPLAT_FWD(6)
INSTANTIATE_SPLAT_FWD(7)
INSTANTIATE_SPLAT_FWD(8)

#undef INSTANTIATE_SPLAT_FWD

// Splat-centric backward kernel instantiations (no BATCH_SIZE dependency)
#define INSTANTIATE_SPLAT_BWD(D) \
    template void launch_rasterize_backward_splat_centric<D, float>(const float*, const float*, const float*, const float*, int, const int*, float, float, float*, float*, float*, float*, cudaStream_t); \
    template void launch_rasterize_backward_splat_centric<D, __half>(const float*, const __half*, const __half*, const __half*, int, const int*, float, float, float*, float*, float*, float*, cudaStream_t);

INSTANTIATE_SPLAT_BWD(2)
INSTANTIATE_SPLAT_BWD(3)
INSTANTIATE_SPLAT_BWD(4)
INSTANTIATE_SPLAT_BWD(5)
INSTANTIATE_SPLAT_BWD(6)
INSTANTIATE_SPLAT_BWD(7)
INSTANTIATE_SPLAT_BWD(8)

#undef INSTANTIATE_SPLAT_BWD

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

void validate_inputs(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    torch::ScalarType expected_dtype
) {
    // Check device
    TORCH_CHECK(centers.is_cuda(), "centers must be on CUDA device");
    TORCH_CHECK(conic.is_cuda(), "conic must be on CUDA device");
    TORCH_CHECK(amps.is_cuda(), "amps must be on CUDA device");

    // Check dimensions
    int dim = (int)shape.size();
    TORCH_CHECK(dim >= MIN_DIM && dim <= MAX_SUPPORTED_DIM,
        "Dimension must be between ", MIN_DIM, " and ", MAX_SUPPORTED_DIM, ", got ", dim);

    int N = (int)centers.size(0);
    TORCH_CHECK(centers.size(1) == dim, "centers must have shape (N, ", dim, ")");

    int expected_conic_size = dim * (dim + 1) / 2;
    TORCH_CHECK(conic.size(1) == expected_conic_size,
        "conic must have shape (N, ", expected_conic_size, ")");

    TORCH_CHECK(amps.size(0) == N, "amps must have shape (", N, ",)");

    // Check dtypes
    const char* dtype_name = (expected_dtype == torch::kFloat16) ? "float16" : "float32";
    TORCH_CHECK(centers.dtype() == expected_dtype, "centers must be ", dtype_name);
    TORCH_CHECK(conic.dtype() == expected_dtype, "conic must be ", dtype_name);
    TORCH_CHECK(amps.dtype() == expected_dtype, "amps must be ", dtype_name);

    // Check contiguous
    TORCH_CHECK(centers.is_contiguous(), "centers must be contiguous");
    TORCH_CHECK(conic.is_contiguous(), "conic must be contiguous");
    TORCH_CHECK(amps.is_contiguous(), "amps must be contiguous");
}

// =============================================================================
// UNIFIED FORWARD DISPATCHER (templated on InputDType)
// =============================================================================

/**
 * Forward dispatcher - handles both FP32 and FP16 via InputDType template.
 *
 * Splat-centric pipeline: single kernel (1 block per splat, atomicAdd to output).
 *
 * FP32: InputDType=float, pointers via data_ptr<float>()
 * FP16: InputDType=__half, pointers via reinterpret_cast from data_ptr<at::Half>()
 *       Inputs loaded as FP16 from global memory, converted to FP32 in shared memory.
 *       Output is always FP32.
 */
template <typename InputDType>
void dispatch_forward_impl(
    int dim,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    torch::Tensor& output,
    ForwardState& state
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int N = (int)centers.size(0);
    auto device = centers.device();

    // Cache shape tensor on device for backward pass reuse
    state.shape_tensor = torch::tensor(std::vector<int>(shape.begin(), shape.end()),
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Extract typed data pointers
    const InputDType* centers_ptr = get_data_ptr<InputDType>(centers);
    const InputDType* conic_ptr = get_data_ptr<InputDType>(conic);

    // NOTE: output is ALREADY zero-initialized by forward_impl
    // (either torch::zeros for first call, or pre-zeroed buffer from backward)

    // Single kernel launch: each block processes one splat, atomicAdd to output
    DIM_DISPATCH(dim,
        launch_rasterize_forward_splat_centric<D, InputDType>(
            centers_ptr, conic_ptr,
            get_data_ptr<InputDType>(amps),
            N,
            state.shape_tensor.data_ptr<int>(),
            truncate, intensity_floor,
            output.data_ptr<float>(),
            stream)
    );

    CUDA_CHECK_LAST();
}

// =============================================================================
// UNIFIED BACKWARD DISPATCHER (templated on InputDType)
// =============================================================================

/**
 * Backward dispatcher - handles both FP32 and FP16 via InputDType template.
 *
 * grad_output is always FP32 regardless of InputDType.
 * Gradients (d_centers, d_conic, d_amps) are always FP32.
 */
template <typename InputDType>
void dispatch_backward_impl(
    int dim,
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    torch::Tensor& d_centers,
    torch::Tensor& d_conic,
    torch::Tensor& d_amps,
    const torch::Tensor& shape_tensor_cached,
    const torch::Tensor& output_to_zero
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int N = (int)centers.size(0);
    auto device = centers.device();

    // Use cached shape tensor or create new one
    auto shape_tensor = shape_tensor_cached.defined() ? shape_tensor_cached :
        torch::tensor(std::vector<int>(shape.begin(), shape.end()),
            torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Extract typed data pointers
    const InputDType* centers_ptr = get_data_ptr<InputDType>(centers);
    const InputDType* conic_ptr = get_data_ptr<InputDType>(conic);
    const InputDType* amps_ptr = get_data_ptr<InputDType>(amps);

    // Splat-centric backward: each block processes ONE splat, iterating over
    // all voxels in its AABB. Handles all splats uniformly.
    DIM_DISPATCH(dim,
        launch_rasterize_backward_splat_centric<D, InputDType>(
            grad_output.data_ptr<float>(),
            centers_ptr, conic_ptr, amps_ptr,
            N,
            shape_tensor.data_ptr<int>(),
            truncate, intensity_floor,
            d_centers.data_ptr<float>(),
            d_conic.data_ptr<float>(),
            d_amps.data_ptr<float>(),
            output_to_zero.defined() ? output_to_zero.data_ptr<float>() : nullptr,
            stream)
    );

    CUDA_CHECK_LAST();
}

// =============================================================================
// PUBLIC API - Forward/Backward (unified implementation + thin wrappers)
// =============================================================================

/**
 * Unified forward implementation, templated on InputDType.
 * Validates inputs, allocates output, dispatches to CUDA kernels.
 * Output is always FP32 regardless of InputDType.
 */
template <typename InputDType>
static std::tuple<torch::Tensor, torch::Tensor>
forward_impl(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    const torch::Tensor& output_buffer
) {
    constexpr torch::ScalarType expected_dtype =
        std::is_same_v<InputDType, __half> ? torch::kFloat16 : torch::kFloat32;

    validate_inputs(centers, conic, amps, shape, expected_dtype);

    const c10::cuda::CUDAGuard device_guard(centers.device());

    int dim = (int)shape.size();
    auto device = centers.device();

    // Compute output size
    int64_t num_pixels = 1;
    for (int64_t s : shape) {
        num_pixels *= s;
    }

    // OPTIMIZATION: Reuse pre-zeroed output buffer if provided (from backward zeroing).
    torch::Tensor output;
    if (output_buffer.defined() && output_buffer.numel() == num_pixels &&
        output_buffer.dtype() == torch::kFloat32 && output_buffer.is_cuda()) {
        output = output_buffer;
    } else {
        output = torch::zeros({num_pixels}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    }

    // Run forward pass
    ForwardState state;
    dispatch_forward_impl<InputDType>(dim, centers, conic, amps, shape,
                                      truncate, intensity_floor, output, state);

    return std::make_tuple(output, state.shape_tensor);
}

/**
 * Unified backward implementation, templated on InputDType.
 * Gradients are always FP32 regardless of InputDType.
 */
template <typename InputDType>
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
backward_impl(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    const torch::Tensor& shape_tensor_cached,
    const torch::Tensor& output_to_zero
) {
    constexpr torch::ScalarType expected_dtype =
        std::is_same_v<InputDType, __half> ? torch::kFloat16 : torch::kFloat32;

    validate_inputs(centers, conic, amps, shape, expected_dtype);

    const c10::cuda::CUDAGuard device_guard(centers.device());

    int dim = (int)shape.size();
    int N = (int)centers.size(0);
    int conic_sz = dim * (dim + 1) / 2;
    auto device = centers.device();

    // Allocate gradient buffers (always FP32)
    auto d_centers = torch::zeros({N, dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto d_conic = torch::zeros({N, conic_sz}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto d_amps = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Run backward pass
    dispatch_backward_impl<InputDType>(dim, grad_output, centers, conic, amps,
                                       shape, truncate, intensity_floor,
                                       d_centers, d_conic, d_amps,
                                       shape_tensor_cached, output_to_zero);

    return std::make_tuple(d_centers, d_conic, d_amps);
}

// Non-templated wrappers for binary compatibility (called by bindings.cpp)

std::tuple<torch::Tensor, torch::Tensor>
forward(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    const torch::Tensor& output_buffer
) {
    return forward_impl<float>(centers, conic, amps,
                               shape, truncate, intensity_floor, output_buffer);
}

std::tuple<torch::Tensor, torch::Tensor>
forward_fp16(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    const torch::Tensor& output_buffer
) {
    return forward_impl<__half>(centers, conic, amps,
                                shape, truncate, intensity_floor, output_buffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    bool use_fp16,
    const torch::Tensor& shape_tensor_cached,
    const torch::Tensor& output_to_zero
) {
    if (use_fp16) {
        return backward_impl<__half>(grad_output, centers, conic, amps,
                                     shape, truncate, intensity_floor,
                                     shape_tensor_cached, output_to_zero);
    }
    return backward_impl<float>(grad_output, centers, conic, amps,
                                shape, truncate, intensity_floor,
                                shape_tensor_cached, output_to_zero);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
backward_fp16(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    const torch::Tensor& shape_tensor_cached,
    const torch::Tensor& output_to_zero
) {
    return backward_impl<__half>(grad_output, centers, conic, amps,
                                 shape, truncate, intensity_floor,
                                 shape_tensor_cached, output_to_zero);
}
