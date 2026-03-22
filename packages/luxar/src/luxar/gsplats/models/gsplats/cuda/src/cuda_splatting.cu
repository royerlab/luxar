/**
 * CUDA Gaussian Splatting - Dispatch Layer
 *
 * This file provides the host-side dispatch and public API for the CUDA splatting backend:
 *
 * 1. Template instantiations for all DIM × InputDType × BATCH_SIZE combinations
 * 2. Input validation
 * 3. Forward/backward dispatch (templated on InputDType for FP32/FP16 unification)
 * 4. Public API: forward(), forward_fp16(), backward(), backward_fp16()
 *
 * The kernels themselves are in kernels_core.cuh and kernels_global.cuh.
 * Launch wrappers are in kernel_launchers.cuh (already templated on InputDType).
 *
 * See SPECIFICATIONS.md for detailed algorithm descriptions.
 */

#include "cuda_splatting.h"
#include "utils.cuh"
#include "kernel_launchers.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
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
// Each macro instantiates both float and __half variants.
// Preprocess and bin kernels don't use BATCH_SIZE (they run once per splat).
// Forward and backward kernels are instantiated for batch sizes 32, 128, 256.

// Preprocess kernel (no BATCH_SIZE dependency)
#define INSTANTIATE_PREPROCESS(D) \
    template void launch_preprocess<D, float>(const float*, const float*, const float*, \
        int, const int*, const int*, int, float, float, int*, bool*, int*, int*, int64_t, int*, cudaStream_t); \
    template void launch_preprocess<D, __half>(const __half*, const __half*, const __half*, \
        int, const int*, const int*, int, float, float, int*, bool*, int*, int*, int64_t, int*, cudaStream_t);

INSTANTIATE_PREPROCESS(2)
INSTANTIATE_PREPROCESS(3)
INSTANTIATE_PREPROCESS(4)
INSTANTIATE_PREPROCESS(5)
INSTANTIATE_PREPROCESS(6)
INSTANTIATE_PREPROCESS(7)
INSTANTIATE_PREPROCESS(8)

#undef INSTANTIATE_PREPROCESS

// Binning kernel (no BATCH_SIZE or InputDType dependency - uses cached int AABBs)
#define INSTANTIATE_BIN(D) \
    template void launch_bin<D>(const int*, const int*, const bool*, int, const int*, \
        const int64_t*, int*, int*, int64_t, cudaStream_t);

INSTANTIATE_BIN(2)
INSTANTIATE_BIN(3)
INSTANTIATE_BIN(4)
INSTANTIATE_BIN(5)
INSTANTIATE_BIN(6)
INSTANTIATE_BIN(7)
INSTANTIATE_BIN(8)

#undef INSTANTIATE_BIN

// Forward and backward kernels with BATCH_SIZE template parameter.
// Instantiate only valid batch size / dimension combinations that fit in shared memory.
// Shared memory limit: 48KB (0xc000) on most GPUs.
//
// Strategy: Use batch=256 for low dims, batch=128 for medium, batch=32 for high dims
// See shared memory budget analysis in SPECIFICATIONS.md.
#define INSTANTIATE_RASTERIZE(D, BATCH) \
    template void launch_rasterize_forward<D, BATCH, float>(const float*, const float*, const float*, \
        int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t); \
    template void launch_rasterize_backward<D, BATCH, float>(const float*, const float*, const float*, const float*, \
        int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t); \
    template void launch_rasterize_forward<D, BATCH, __half>(const __half*, const __half*, const __half*, \
        int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t); \
    template void launch_rasterize_backward<D, BATCH, __half>(const float*, const __half*, const __half*, const __half*, \
        int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);

// Batch size 32 (all dimensions - minimum safe batch size)
INSTANTIATE_RASTERIZE(2, 32)
INSTANTIATE_RASTERIZE(3, 32)
INSTANTIATE_RASTERIZE(4, 32)
INSTANTIATE_RASTERIZE(5, 32)
INSTANTIATE_RASTERIZE(6, 32)
INSTANTIATE_RASTERIZE(7, 32)
INSTANTIATE_RASTERIZE(8, 32)

// Batch size 128 (dimensions 2-6 - fits in 48KB shared memory)
INSTANTIATE_RASTERIZE(2, 128)
INSTANTIATE_RASTERIZE(3, 128)
INSTANTIATE_RASTERIZE(4, 128)
INSTANTIATE_RASTERIZE(5, 128)
INSTANTIATE_RASTERIZE(6, 128)
// DIM=7,8 with BATCH=128 exceed 48KB shared memory, skipped

// Batch size 256 (dimensions 2-4 - fits in 48KB shared memory)
INSTANTIATE_RASTERIZE(2, 256)
INSTANTIATE_RASTERIZE(3, 256)
INSTANTIATE_RASTERIZE(4, 256)
// DIM>=5 with BATCH=256 exceed 48KB shared memory, skipped

#undef INSTANTIATE_RASTERIZE

// Global splat kernel instantiations (no BATCH_SIZE dependency)
#define INSTANTIATE_GLOBAL(D) \
    template void launch_rasterize_global_forward<D, float>(const float*, const float*, const float*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t); \
    template void launch_rasterize_global_backward<D, float>(const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, float*, float*, int64_t, cudaStream_t); \
    template void launch_rasterize_global_forward<D, __half>(const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t); \
    template void launch_rasterize_global_backward<D, __half>(const float*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, float*, float*, int64_t, cudaStream_t);

INSTANTIATE_GLOBAL(2)
INSTANTIATE_GLOBAL(3)
INSTANTIATE_GLOBAL(4)
INSTANTIATE_GLOBAL(5)
INSTANTIATE_GLOBAL(6)
INSTANTIATE_GLOBAL(7)
INSTANTIATE_GLOBAL(8)

#undef INSTANTIATE_GLOBAL

// Splat-centric forward kernel instantiations (no BATCH_SIZE dependency)
#define INSTANTIATE_SPLAT_FWD(D) \
    template void launch_rasterize_forward_splat_centric<D, float>(const float*, const float*, const float*, int, const int*, float, float, float*, bool*, int*, int64_t, int, int*, const int*, cudaStream_t); \
    template void launch_rasterize_forward_splat_centric<D, __half>(const __half*, const __half*, const __half*, int, const int*, float, float, float*, bool*, int*, int64_t, int, int*, const int*, cudaStream_t);

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
    template void launch_rasterize_backward_splat_centric<D, float>(const float*, const float*, const float*, const float*, int, const int*, float, float, float*, float*, float*, cudaStream_t); \
    template void launch_rasterize_backward_splat_centric<D, __half>(const float*, const __half*, const __half*, const __half*, int, const int*, float, float, float*, float*, float*, cudaStream_t);

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

std::vector<int> compute_tile_dims(
    const std::vector<int64_t>& shape,
    int tile_size
) {
    std::vector<int> tile_dims(shape.size());
    for (size_t d = 0; d < shape.size(); d++) {
        tile_dims[d] = (int)((shape[d] + tile_size - 1) / tile_size);
    }
    return tile_dims;
}

int64_t compute_num_tiles(const std::vector<int>& tile_dims) {
    int64_t num_tiles = 1;
    for (int td : tile_dims) {
        num_tiles *= td;
    }
    return num_tiles;
}

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
 * Pipeline: preprocess → prefix_sum → bin → rasterize_forward → global_forward
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
    const torch::Tensor& L_row_norms,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size,
    torch::Tensor& output,
    BinningState& state
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int N = (int)centers.size(0);
    auto tile_dims = compute_tile_dims(shape, tile_size);
    int64_t num_tiles = compute_num_tiles(tile_dims);

    TORCH_CHECK(num_tiles <= MAX_TILES,
        "Too many tiles (", num_tiles, "). Maximum is ", MAX_TILES,
        ". Increase tile_size or reduce volume size.");

    auto device = centers.device();

    // OPTIMIZATION: Splat-centric forward — eliminates the ENTIRE tile binning pipeline.
    // Instead of: preprocess → prefix_sum → bin → rasterize_fwd → global_fwd (5 kernels + sync)
    // Now: output.zero_() → 1 splat-centric kernel (each block = 1 splat, atomicAdd to output)
    //
    // This eliminates: ~13 tensor allocations, 4 kernel launches, CUB prefix sum,
    // cudaStreamSynchronize, shared memory loading/barriers, tile binning entirely.

    // Only keep shape_tensor for backward pass compatibility
    state.shape_tensor = torch::tensor(std::vector<int>(shape.begin(), shape.end()),
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto tile_dims_vec = compute_tile_dims(shape, tile_size);
    state.tile_dims_tensor = torch::tensor(tile_dims_vec,
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.tile_size = tile_size;
    state.num_tiles = num_tiles;

    // BinningState populated for API/diagnostic compatibility (not used by splat-centric kernels)
    state.tile_counts = torch::zeros({num_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.tile_offsets = torch::zeros({num_tiles}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    state.tile_content = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.total_pairs = 0;

    // Global splat detection (lightweight: just flags + counter in the splat-centric kernel)
    state.global_splat_flags = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(device));
    auto global_count_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Extract typed data pointers
    const InputDType* centers_ptr = get_data_ptr<InputDType>(centers);
    const InputDType* conic_ptr = get_data_ptr<InputDType>(conic);

    // Zero-initialize output (splat-centric kernel uses atomicAdd)
    output.zero_();

    // Single kernel launch: each block processes one splat
    // Also computes global splat flags as a side effect (thread 0 per block)
    DIM_DISPATCH(dim,
        launch_rasterize_forward_splat_centric<D, InputDType>(
            centers_ptr, conic_ptr,
            get_data_ptr<InputDType>(amps),
            N,
            state.shape_tensor.data_ptr<int>(),
            truncate, intensity_floor,
            output.data_ptr<float>(),
            state.global_splat_flags.data_ptr<bool>(),
            global_count_tensor.data_ptr<int>(),
            num_tiles,
            tile_size,
            state.tile_counts.data_ptr<int>(),
            state.tile_dims_tensor.data_ptr<int>(),
            stream)
    );

    CUDA_CHECK_LAST();

    // Read global splat count (async memcpy + sync only if count > 0)
    int h_global_count = 0;
    cudaMemcpyAsync(&h_global_count, global_count_tensor.data_ptr<int>(),
        sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    state.num_global_splats = h_global_count;
    if (h_global_count > 0) {
        auto global_indices = torch::nonzero(state.global_splat_flags);
        state.global_splat_ids = global_indices.squeeze(1).to(torch::kInt32).contiguous();
        state.num_global_splats = (int)global_indices.size(0);
    } else {
        state.global_splat_ids = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }
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
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size,
    torch::Tensor& d_centers,
    torch::Tensor& d_conic,
    torch::Tensor& d_amps,
    const torch::Tensor& shape_tensor_cached,
    const torch::Tensor& tile_dims_tensor_cached
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int N = (int)centers.size(0);
    auto tile_dims = compute_tile_dims(shape, tile_size);
    int64_t num_tiles = compute_num_tiles(tile_dims);

    auto device = centers.device();

    // Copy shape and tile_dims to device (use cached if available)
    auto shape_tensor = shape_tensor_cached.defined() ? shape_tensor_cached :
        torch::tensor(std::vector<int>(shape.begin(), shape.end()),
            torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto tile_dims_tensor = tile_dims_tensor_cached.defined() ? tile_dims_tensor_cached :
        torch::tensor(tile_dims, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Extract typed data pointers
    const InputDType* centers_ptr = get_data_ptr<InputDType>(centers);
    const InputDType* conic_ptr = get_data_ptr<InputDType>(conic);
    const InputDType* amps_ptr = get_data_ptr<InputDType>(amps);

    // OPTIMIZATION: Splat-centric backward — each block processes ONE splat,
    // iterating over all voxels in its AABB. This replaces BOTH the tile-centric
    // backward AND global splat backward kernels.
    // Benefits: no tile binning dependency, no global atomics, no shared memory
    // gradient accumulators, handles all splats (including global) uniformly.
    // Note: gradient buffers are NOT pre-zeroed — the kernel writes directly
    // (each block owns its splat exclusively, no concurrent writes).
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
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor>
forward_impl(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& L_row_norms,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size
) {
    constexpr torch::ScalarType expected_dtype =
        std::is_same_v<InputDType, __half> ? torch::kFloat16 : torch::kFloat32;

    validate_inputs(centers, conic, amps, shape, expected_dtype);

    // Validate L_row_norms
    int dim = (int)shape.size();
    int N = (int)centers.size(0);
    TORCH_CHECK(L_row_norms.is_cuda(), "L_row_norms must be on CUDA device");
    TORCH_CHECK(L_row_norms.is_contiguous(), "L_row_norms must be contiguous");
    TORCH_CHECK(L_row_norms.size(0) == N && L_row_norms.size(1) == dim,
        "L_row_norms must have shape (", N, ", ", dim, ")");
    TORCH_CHECK(L_row_norms.dtype() == expected_dtype, "L_row_norms dtype mismatch");

    // Validate batch_size
    TORCH_CHECK(batch_size == 32 || batch_size == 128 || batch_size == 256,
        "batch_size must be 32, 128, or 256, got ", batch_size);

    auto device = centers.device();

    // Compute output size
    int64_t num_pixels = 1;
    for (int64_t s : shape) {
        num_pixels *= s;
    }

    // Allocate output (always FP32)
    auto output = torch::zeros({num_pixels}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Run forward pass
    BinningState state;
    dispatch_forward_impl<InputDType>(dim, centers, conic, amps, L_row_norms, shape,
                                      truncate, intensity_floor, tile_size, batch_size, output, state);

    return std::make_tuple(
        output,
        state.tile_counts,
        state.tile_offsets,
        state.tile_content,
        state.global_splat_ids,
        state.shape_tensor,
        state.tile_dims_tensor
    );
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
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size,
    const torch::Tensor& shape_tensor_cached,
    const torch::Tensor& tile_dims_tensor_cached
) {
    constexpr torch::ScalarType expected_dtype =
        std::is_same_v<InputDType, __half> ? torch::kFloat16 : torch::kFloat32;

    validate_inputs(centers, conic, amps, shape, expected_dtype);

    // Validate batch_size
    TORCH_CHECK(batch_size == 32 || batch_size == 128 || batch_size == 256,
        "batch_size must be 32, 128, or 256, got ", batch_size);

    int dim = (int)shape.size();
    int N = (int)centers.size(0);
    int conic_size = dim * (dim + 1) / 2;
    auto device = centers.device();

    // Allocate gradient buffers (always FP32)
    auto d_centers = torch::zeros({N, dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto d_conic = torch::zeros({N, conic_size}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto d_amps = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Run backward pass
    dispatch_backward_impl<InputDType>(dim, grad_output, centers, conic, amps,
                                       tile_offsets, tile_counts, tile_content, global_splat_ids,
                                       shape, truncate, intensity_floor, tile_size, batch_size,
                                       d_centers, d_conic, d_amps,
                                       shape_tensor_cached, tile_dims_tensor_cached);

    return std::make_tuple(d_centers, d_conic, d_amps);
}

// Non-templated wrappers for binary compatibility (called by bindings.cpp)

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& L_row_norms,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size
) {
    return forward_impl<float>(centers, conic, amps, L_row_norms,
                               shape, truncate, intensity_floor, tile_size, batch_size);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_fp16(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& L_row_norms,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size
) {
    return forward_impl<__half>(centers, conic, amps, L_row_norms,
                                shape, truncate, intensity_floor, tile_size, batch_size);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size,
    const torch::Tensor& shape_tensor_cached,
    const torch::Tensor& tile_dims_tensor_cached
) {
    return backward_impl<float>(grad_output, centers, conic, amps,
                                tile_offsets, tile_counts, tile_content, global_splat_ids,
                                shape, truncate, intensity_floor, tile_size, batch_size,
                                shape_tensor_cached, tile_dims_tensor_cached);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
backward_fp16(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size,
    const torch::Tensor& shape_tensor_cached,
    const torch::Tensor& tile_dims_tensor_cached
) {
    return backward_impl<__half>(grad_output, centers, conic, amps,
                                 tile_offsets, tile_counts, tile_content, global_splat_ids,
                                 shape, truncate, intensity_floor, tile_size, batch_size,
                                 shape_tensor_cached, tile_dims_tensor_cached);
}
