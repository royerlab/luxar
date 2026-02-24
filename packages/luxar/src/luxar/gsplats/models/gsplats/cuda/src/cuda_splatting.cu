/**
 * CUDA Gaussian Splatting Kernels
 *
 * This file implements the core CUDA kernels for volumetric Gaussian splatting:
 *
 * 1. Preprocess: Compute AABBs and count splats per tile
 * 2. Binning: Assign splats to tiles using prefix sum + atomic writes
 * 3. Forward Rasterization: Render splats to pixels (tile-parallel)
 * 4. Backward Rasterization: Compute gradients with warp reduction
 *
 * Key optimization techniques:
 * - Tile-based binning (no depth sorting needed for additive blending)
 * - Shared memory batch loading (BalanceGS pattern)
 * - Warp-level gradient reduction (DISTWAR pattern)
 * - Template specialization for 2D/3D fast paths
 *
 * See SPECIFICATIONS.md for detailed algorithm descriptions.
 *
 * File organization:
 * - kernels_core.cuh: Core kernel implementations (preprocess, bin, forward, backward)
 * - kernels_global.cuh: Global splat kernel implementations
 * - kernel_launchers.cuh: Launch wrapper functions
 * - This file: Template instantiations, utility functions, dispatchers, public API
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

// =============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS (FP32)
// =============================================================================
// Preprocess and bin kernels don't use BATCH_SIZE (they run once per splat).
// Forward and backward kernels are instantiated for batch sizes 32, 128, 256.

// Preprocess kernel (no BATCH_SIZE dependency)
// Signature: (centers, amps, sharpness, L_row_norms, N, shape, tile_dims, tile_size,
//             truncate, intensity_floor, tile_counts, global_flags, aabb_lo, aabb_hi, num_tiles, global_count, stream)
#define INSTANTIATE_PREPROCESS(D) \
    template void launch_preprocess<D, float>(const float*, const float*, const float*, const float*, \
        int, const int*, const int*, int, float, float, int*, bool*, int*, int*, int64_t, int*, cudaStream_t);

INSTANTIATE_PREPROCESS(2)
INSTANTIATE_PREPROCESS(3)
INSTANTIATE_PREPROCESS(4)
INSTANTIATE_PREPROCESS(5)
INSTANTIATE_PREPROCESS(6)
INSTANTIATE_PREPROCESS(7)
INSTANTIATE_PREPROCESS(8)

#undef INSTANTIATE_PREPROCESS

// Binning kernel (no BATCH_SIZE or InputDType dependency - uses cached AABBs)
// Signature: (aabb_lo, aabb_hi, global_flags, N, tile_dims, tile_offsets, tile_write_heads, tile_content, num_tiles, stream)
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

// Forward and backward kernels with BATCH_SIZE template parameter
// Instantiate only valid batch size / dimension combinations that fit in shared memory.
// Shared memory limit: 48KB (0xc000) on most GPUs
//
// Per-splat shared memory (backward kernel, float32):
// - centers: center_stride * 4 (center_stride = ((d+3)//4)*4 for bank alignment)
// - conic: d*(d+1)/2 * 4
// - amps, sharpness, truncate_sq: 3 * 4
// - splat_id: 4 (int)
//
// Approximate usage per batch:
// - DIM=2, BATCH=256: ~8KB (OK)
// - DIM=3, BATCH=256: ~14KB (OK)
// - DIM=4, BATCH=256: ~22KB (OK)
// - DIM=4, BATCH=128: ~11KB (OK)
// - DIM=5, BATCH=256: ~51KB (EXCEEDS!)
// - DIM=5, BATCH=128: ~25KB (OK)
// - DIM=6, BATCH=128: ~35KB (OK)
// - DIM=7, BATCH=128: ~47KB (BORDERLINE)
// - DIM=8, BATCH=128: ~52KB (EXCEEDS!)
//
// Strategy: Use batch=256 for low dims, batch=128 for medium, batch=32 for high dims

#define INSTANTIATE_RASTERIZE(D, BATCH) \
    template void launch_rasterize_forward<D, BATCH, float>(const float*, const float*, const float*, const float*, \
        int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t); \
    template void launch_rasterize_backward<D, BATCH, float>(const float*, const float*, const float*, const float*, const float*, \
        int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);

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

// Global splat kernel instantiations (with intensity_floor parameter)
template void launch_rasterize_global_forward<2, float>(const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<2, float>(const float*, const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_forward<3, float>(const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<3, float>(const float*, const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_forward<4, float>(const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<4, float>(const float*, const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_forward<5, float>(const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<5, float>(const float*, const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_forward<6, float>(const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<6, float>(const float*, const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_forward<7, float>(const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<7, float>(const float*, const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_forward<8, float>(const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<8, float>(const float*, const float*, const float*, const float*, const float*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);

// =============================================================================
// FP16 (__half) EXPLICIT TEMPLATE INSTANTIATIONS
// =============================================================================
// These instantiate the FP16 versions of all kernels for Phase 2 mixed precision:
// - Inputs are loaded as FP16 from global memory
// - Computation uses FP32 in shared memory and registers
// - Outputs and gradients remain FP32

// Preprocess kernel FP16 (no BATCH_SIZE dependency)
// Signature: (centers, amps, sharpness, L_row_norms, N, shape, tile_dims, tile_size,
//             truncate, intensity_floor, tile_counts, global_flags, aabb_lo, aabb_hi, num_tiles, global_count, stream)
#define INSTANTIATE_PREPROCESS_FP16(D) \
    template void launch_preprocess<D, __half>(const __half*, const __half*, const __half*, const __half*, \
        int, const int*, const int*, int, float, float, int*, bool*, int*, int*, int64_t, int*, cudaStream_t);

INSTANTIATE_PREPROCESS_FP16(2)
INSTANTIATE_PREPROCESS_FP16(3)
INSTANTIATE_PREPROCESS_FP16(4)
INSTANTIATE_PREPROCESS_FP16(5)
INSTANTIATE_PREPROCESS_FP16(6)
INSTANTIATE_PREPROCESS_FP16(7)
INSTANTIATE_PREPROCESS_FP16(8)

#undef INSTANTIATE_PREPROCESS_FP16

// Note: launch_bin<D> is NOT templated on InputDType (uses cached int AABBs),
// so it only needs the FP32 instantiation above. No FP16 instantiation needed.

// Forward and backward kernels with BATCH_SIZE template parameter
// Same shared memory constraints as FP32 (FP16 inputs are converted to FP32 in shared memory)
#define INSTANTIATE_RASTERIZE_FP16(D, BATCH) \
    template void launch_rasterize_forward<D, BATCH, __half>(const __half*, const __half*, const __half*, const __half*, \
        int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t); \
    template void launch_rasterize_backward<D, BATCH, __half>(const float*, const __half*, const __half*, const __half*, const __half*, \
        int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);

// Batch size 32 (all dimensions)
INSTANTIATE_RASTERIZE_FP16(2, 32)
INSTANTIATE_RASTERIZE_FP16(3, 32)
INSTANTIATE_RASTERIZE_FP16(4, 32)
INSTANTIATE_RASTERIZE_FP16(5, 32)
INSTANTIATE_RASTERIZE_FP16(6, 32)
INSTANTIATE_RASTERIZE_FP16(7, 32)
INSTANTIATE_RASTERIZE_FP16(8, 32)

// Batch size 128 (dimensions 2-6)
INSTANTIATE_RASTERIZE_FP16(2, 128)
INSTANTIATE_RASTERIZE_FP16(3, 128)
INSTANTIATE_RASTERIZE_FP16(4, 128)
INSTANTIATE_RASTERIZE_FP16(5, 128)
INSTANTIATE_RASTERIZE_FP16(6, 128)
// DIM=7,8 with BATCH=128 exceed 48KB shared memory, skipped

// Batch size 256 (dimensions 2-4)
INSTANTIATE_RASTERIZE_FP16(2, 256)
INSTANTIATE_RASTERIZE_FP16(3, 256)
INSTANTIATE_RASTERIZE_FP16(4, 256)
// DIM>=5 with BATCH=256 exceed 48KB shared memory, skipped

#undef INSTANTIATE_RASTERIZE_FP16

// Global splat kernels (no BATCH_SIZE dependency)
#define INSTANTIATE_GLOBAL_FP16(D) \
    template void launch_rasterize_global_forward<D, __half>(const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t); \
    template void launch_rasterize_global_backward<D, __half>(const float*, const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);

INSTANTIATE_GLOBAL_FP16(2)
INSTANTIATE_GLOBAL_FP16(3)
INSTANTIATE_GLOBAL_FP16(4)
INSTANTIATE_GLOBAL_FP16(5)
INSTANTIATE_GLOBAL_FP16(6)
INSTANTIATE_GLOBAL_FP16(7)
INSTANTIATE_GLOBAL_FP16(8)

#undef INSTANTIATE_GLOBAL_FP16

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
    const torch::Tensor& sharpness,
    const std::vector<int64_t>& shape
) {
    // Check device
    TORCH_CHECK(centers.is_cuda(), "centers must be on CUDA device");
    TORCH_CHECK(conic.is_cuda(), "conic must be on CUDA device");
    TORCH_CHECK(amps.is_cuda(), "amps must be on CUDA device");
    TORCH_CHECK(sharpness.is_cuda(), "sharpness must be on CUDA device");

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
    TORCH_CHECK(sharpness.size(0) == N, "sharpness must have shape (", N, ",)");

    // Check dtypes
    TORCH_CHECK(centers.dtype() == torch::kFloat32, "centers must be float32");
    TORCH_CHECK(conic.dtype() == torch::kFloat32, "conic must be float32");
    TORCH_CHECK(amps.dtype() == torch::kFloat32, "amps must be float32");
    TORCH_CHECK(sharpness.dtype() == torch::kFloat32, "sharpness must be float32");

    // Check contiguous
    TORCH_CHECK(centers.is_contiguous(), "centers must be contiguous");
    TORCH_CHECK(conic.is_contiguous(), "conic must be contiguous");
    TORCH_CHECK(amps.is_contiguous(), "amps must be contiguous");
    TORCH_CHECK(sharpness.is_contiguous(), "sharpness must be contiguous");
}

// =============================================================================
// DISPATCHER FUNCTIONS
// =============================================================================

void dispatch_forward(
    int dim,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
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

    // Allocate binning state
    state.tile_counts = torch::zeros({num_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.global_splat_flags = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(device));
    state.num_tiles = num_tiles;

    // Allocate AABB cache (written by preprocess, read by bin)
    state.aabb_lo = torch::empty({N, dim}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.aabb_hi = torch::empty({N, dim}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Atomic counter for global splats (avoids expensive torch::nonzero when 0)
    auto global_count_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Copy shape and tile_dims to device
    // OPTIMIZATION: Store in BinningState for potential reuse in backward pass
    state.shape_tensor = torch::tensor(std::vector<int>(shape.begin(), shape.end()),
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.tile_dims_tensor = torch::tensor(tile_dims,
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.tile_size = tile_size;

    // Use local references for readability
    auto& shape_tensor = state.shape_tensor;
    auto& tile_dims_tensor = state.tile_dims_tensor;

    // Launch preprocess kernel
    #define LAUNCH_PREPROCESS(D) \
        launch_preprocess<D, float>( \
            centers.data_ptr<float>(), \
            amps.data_ptr<float>(), \
            sharpness.data_ptr<float>(), \
            L_row_norms.data_ptr<float>(), \
            N, \
            shape_tensor.data_ptr<int>(), \
            tile_dims_tensor.data_ptr<int>(), \
            tile_size, truncate, intensity_floor, \
            state.tile_counts.data_ptr<int>(), \
            state.global_splat_flags.data_ptr<bool>(), \
            state.aabb_lo.data_ptr<int>(), \
            state.aabb_hi.data_ptr<int>(), \
            num_tiles, \
            global_count_tensor.data_ptr<int>(), \
            stream)

    switch (dim) {
        case 2: LAUNCH_PREPROCESS(2); break;
        case 3: LAUNCH_PREPROCESS(3); break;
        case 4: LAUNCH_PREPROCESS(4); break;
        case 5: LAUNCH_PREPROCESS(5); break;
        case 6: LAUNCH_PREPROCESS(6); break;
        case 7: LAUNCH_PREPROCESS(7); break;
        case 8: LAUNCH_PREPROCESS(8); break;
        default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
    }
    #undef LAUNCH_PREPROCESS

    CUDA_CHECK_LAST();

    // Compute prefix sum for tile offsets
    state.tile_offsets = torch::empty({num_tiles}, torch::TensorOptions().dtype(torch::kInt64).device(device));

    // Query CUB temp storage size
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, temp_bytes,
        state.tile_counts.data_ptr<int>(),
        state.tile_offsets.data_ptr<int64_t>(),
        (int)num_tiles, stream
    );

    state.scan_temp_storage = torch::empty({(int64_t)temp_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
    state.scan_temp_bytes = temp_bytes;

    // Run prefix sum
    cub::DeviceScan::ExclusiveSum(
        state.scan_temp_storage.data_ptr<uint8_t>(), temp_bytes,
        state.tile_counts.data_ptr<int>(),
        state.tile_offsets.data_ptr<int64_t>(),
        (int)num_tiles, stream
    );

    CUDA_CHECK_LAST();

    // Compute total pairs from last offset + last count
    int64_t last_offset = 0;
    int last_count = 0;
    cudaMemcpyAsync(&last_offset, state.tile_offsets.data_ptr<int64_t>() + num_tiles - 1,
        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_count, state.tile_counts.data_ptr<int>() + num_tiles - 1,
        sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    state.total_pairs = last_offset + last_count;

    // Allocate tile content
    state.tile_content = torch::empty({state.total_pairs},
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.tile_write_heads = torch::zeros({num_tiles},
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Launch binning kernel
    #define LAUNCH_BIN(D) \
        launch_bin<D>( \
            state.aabb_lo.data_ptr<int>(), \
            state.aabb_hi.data_ptr<int>(), \
            state.global_splat_flags.data_ptr<bool>(), \
            N, \
            tile_dims_tensor.data_ptr<int>(), \
            state.tile_offsets.data_ptr<int64_t>(), \
            state.tile_write_heads.data_ptr<int>(), \
            state.tile_content.data_ptr<int>(), \
            num_tiles, stream)

    switch (dim) {
        case 2: LAUNCH_BIN(2); break;
        case 3: LAUNCH_BIN(3); break;
        case 4: LAUNCH_BIN(4); break;
        case 5: LAUNCH_BIN(5); break;
        case 6: LAUNCH_BIN(6); break;
        case 7: LAUNCH_BIN(7); break;
        case 8: LAUNCH_BIN(8); break;
        default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
    }
    #undef LAUNCH_BIN

    CUDA_CHECK_LAST();

    // Initialize output to zero
    output.zero_();

    // Launch rasterization kernel
    // OPTIMIZATION: Pass host tile_dims for 3D grid launch (2D/3D volumes)
    // Dispatch based on batch_size and dimension
    // Note: Not all batch_size/dim combinations are instantiated due to shared memory limits
    //       - Batch 256: dims 2-4 only
    //       - Batch 128: dims 2-6 only
    //       - Batch 32: all dims
    #define LAUNCH_RASTER(D, BATCH) \
        launch_rasterize_forward<D, BATCH, float>( \
            centers.data_ptr<float>(), \
            conic.data_ptr<float>(), \
            amps.data_ptr<float>(), \
            sharpness.data_ptr<float>(), \
            N, \
            shape_tensor.data_ptr<int>(), \
            tile_dims_tensor.data_ptr<int>(), \
            tile_size, truncate, intensity_floor, \
            state.tile_offsets.data_ptr<int64_t>(), \
            state.tile_counts.data_ptr<int>(), \
            state.tile_content.data_ptr<int>(), \
            output.data_ptr<float>(), \
            num_tiles, tile_dims, stream)

    // Clamp batch_size based on dimension due to shared memory constraints
    int effective_batch = batch_size;
    if (dim >= 7 && effective_batch > 32) effective_batch = 32;
    else if (dim >= 5 && effective_batch > 128) effective_batch = 128;

    #define DISPATCH_RASTER_BY_DIM_32 \
        switch (dim) { \
            case 2: LAUNCH_RASTER(2, 32); break; \
            case 3: LAUNCH_RASTER(3, 32); break; \
            case 4: LAUNCH_RASTER(4, 32); break; \
            case 5: LAUNCH_RASTER(5, 32); break; \
            case 6: LAUNCH_RASTER(6, 32); break; \
            case 7: LAUNCH_RASTER(7, 32); break; \
            case 8: LAUNCH_RASTER(8, 32); break; \
            default: TORCH_CHECK(false, "Unsupported dimension: ", dim); \
        }

    #define DISPATCH_RASTER_BY_DIM_128 \
        switch (dim) { \
            case 2: LAUNCH_RASTER(2, 128); break; \
            case 3: LAUNCH_RASTER(3, 128); break; \
            case 4: LAUNCH_RASTER(4, 128); break; \
            case 5: LAUNCH_RASTER(5, 128); break; \
            case 6: LAUNCH_RASTER(6, 128); break; \
            default: TORCH_CHECK(false, "Unsupported dimension for batch_size 128: ", dim); \
        }

    #define DISPATCH_RASTER_BY_DIM_256 \
        switch (dim) { \
            case 2: LAUNCH_RASTER(2, 256); break; \
            case 3: LAUNCH_RASTER(3, 256); break; \
            case 4: LAUNCH_RASTER(4, 256); break; \
            default: TORCH_CHECK(false, "Unsupported dimension for batch_size 256: ", dim); \
        }

    switch (effective_batch) {
        case 32: DISPATCH_RASTER_BY_DIM_32; break;
        case 128: DISPATCH_RASTER_BY_DIM_128; break;
        case 256: DISPATCH_RASTER_BY_DIM_256; break;
        default: TORCH_CHECK(false, "Unsupported batch_size: ", effective_batch, ". Must be 32, 128, or 256.");
    }
    #undef DISPATCH_RASTER_BY_DIM_32
    #undef DISPATCH_RASTER_BY_DIM_128
    #undef DISPATCH_RASTER_BY_DIM_256
    #undef LAUNCH_RASTER

    CUDA_CHECK_LAST();

    // ==========================================================================
    // GLOBAL SPLAT HANDLING
    // ==========================================================================
    // Extract global splat IDs and process them with dedicated kernel.
    // Global splats are those that touch too many tiles (>10% AND >1024 tiles).

    // OPTIMIZATION: Check atomic counter before expensive torch::nonzero
    // This avoids a GPU kernel launch + sync when there are 0 global splats (common case)
    int h_global_count = 0;
    cudaMemcpyAsync(&h_global_count, global_count_tensor.data_ptr<int>(),
        sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    state.num_global_splats = h_global_count;

    if (h_global_count > 0) {
        // Only run torch::nonzero when we know there are global splats
        auto global_indices = torch::nonzero(state.global_splat_flags);
        // Flatten to 1D tensor of int32 indices
        state.global_splat_ids = global_indices.squeeze(1).to(torch::kInt32).contiguous();
        // Use actual count from nonzero (authoritative) rather than atomic counter
        state.num_global_splats = (int)global_indices.size(0);

        // Compute number of pixels
        int64_t num_pixels = 1;
        for (int d = 0; d < dim; d++) {
            num_pixels *= shape[d];
        }

        // Launch global splat forward kernel
        #define LAUNCH_GLOBAL_FWD(D) \
            launch_rasterize_global_forward<D, float>( \
                centers.data_ptr<float>(), \
                conic.data_ptr<float>(), \
                amps.data_ptr<float>(), \
                sharpness.data_ptr<float>(), \
                state.global_splat_ids.data_ptr<int>(), \
                state.num_global_splats, \
                shape_tensor.data_ptr<int>(), \
                truncate, \
                intensity_floor, \
                output.data_ptr<float>(), \
                num_pixels, stream)

        switch (dim) {
            case 2: LAUNCH_GLOBAL_FWD(2); break;
            case 3: LAUNCH_GLOBAL_FWD(3); break;
            case 4: LAUNCH_GLOBAL_FWD(4); break;
            case 5: LAUNCH_GLOBAL_FWD(5); break;
            case 6: LAUNCH_GLOBAL_FWD(6); break;
            case 7: LAUNCH_GLOBAL_FWD(7); break;
            case 8: LAUNCH_GLOBAL_FWD(8); break;
            default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
        }
        #undef LAUNCH_GLOBAL_FWD

        CUDA_CHECK_LAST();
    } else {
        // No global splats - create empty tensor
        state.global_splat_ids = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }
}

void dispatch_backward(
    int dim,
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,  // Global splat IDs from forward pass
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size,
    torch::Tensor& d_centers,
    torch::Tensor& d_conic,
    torch::Tensor& d_amps,
    torch::Tensor& d_sharpness
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int N = (int)centers.size(0);
    auto tile_dims = compute_tile_dims(shape, tile_size);
    int64_t num_tiles = compute_num_tiles(tile_dims);

    auto device = centers.device();

    // Copy shape and tile_dims to device
    auto shape_tensor = torch::tensor(std::vector<int>(shape.begin(), shape.end()),
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto tile_dims_tensor = torch::tensor(tile_dims,
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Zero gradient buffers
    d_centers.zero_();
    d_conic.zero_();
    d_amps.zero_();
    d_sharpness.zero_();

    // Launch backward kernel for tile-based splats
    // OPTIMIZATION: Pass host tile_dims for 3D grid launch (2D/3D volumes)
    // Dispatch based on batch_size and dimension
    // Note: Not all batch_size/dim combinations are instantiated due to shared memory limits
    #define LAUNCH_BACKWARD(D, BATCH) \
        launch_rasterize_backward<D, BATCH, float>( \
            grad_output.data_ptr<float>(), \
            centers.data_ptr<float>(), \
            conic.data_ptr<float>(), \
            amps.data_ptr<float>(), \
            sharpness.data_ptr<float>(), \
            N, \
            shape_tensor.data_ptr<int>(), \
            tile_dims_tensor.data_ptr<int>(), \
            tile_size, truncate, intensity_floor, \
            tile_offsets.data_ptr<int64_t>(), \
            tile_counts.data_ptr<int>(), \
            tile_content.data_ptr<int>(), \
            d_centers.data_ptr<float>(), \
            d_conic.data_ptr<float>(), \
            d_amps.data_ptr<float>(), \
            d_sharpness.data_ptr<float>(), \
            num_tiles, tile_dims, stream)

    // Clamp batch_size based on dimension due to shared memory constraints
    int effective_batch = batch_size;
    if (dim >= 7 && effective_batch > 32) effective_batch = 32;
    else if (dim >= 5 && effective_batch > 128) effective_batch = 128;

    #define DISPATCH_BACKWARD_BY_DIM_32 \
        switch (dim) { \
            case 2: LAUNCH_BACKWARD(2, 32); break; \
            case 3: LAUNCH_BACKWARD(3, 32); break; \
            case 4: LAUNCH_BACKWARD(4, 32); break; \
            case 5: LAUNCH_BACKWARD(5, 32); break; \
            case 6: LAUNCH_BACKWARD(6, 32); break; \
            case 7: LAUNCH_BACKWARD(7, 32); break; \
            case 8: LAUNCH_BACKWARD(8, 32); break; \
            default: TORCH_CHECK(false, "Unsupported dimension: ", dim); \
        }

    #define DISPATCH_BACKWARD_BY_DIM_128 \
        switch (dim) { \
            case 2: LAUNCH_BACKWARD(2, 128); break; \
            case 3: LAUNCH_BACKWARD(3, 128); break; \
            case 4: LAUNCH_BACKWARD(4, 128); break; \
            case 5: LAUNCH_BACKWARD(5, 128); break; \
            case 6: LAUNCH_BACKWARD(6, 128); break; \
            default: TORCH_CHECK(false, "Unsupported dimension for batch_size 128: ", dim); \
        }

    #define DISPATCH_BACKWARD_BY_DIM_256 \
        switch (dim) { \
            case 2: LAUNCH_BACKWARD(2, 256); break; \
            case 3: LAUNCH_BACKWARD(3, 256); break; \
            case 4: LAUNCH_BACKWARD(4, 256); break; \
            default: TORCH_CHECK(false, "Unsupported dimension for batch_size 256: ", dim); \
        }

    switch (effective_batch) {
        case 32: DISPATCH_BACKWARD_BY_DIM_32; break;
        case 128: DISPATCH_BACKWARD_BY_DIM_128; break;
        case 256: DISPATCH_BACKWARD_BY_DIM_256; break;
        default: TORCH_CHECK(false, "Unsupported batch_size: ", effective_batch, ". Must be 32, 128, or 256.");
    }
    #undef DISPATCH_BACKWARD_BY_DIM_32
    #undef DISPATCH_BACKWARD_BY_DIM_128
    #undef DISPATCH_BACKWARD_BY_DIM_256
    #undef LAUNCH_BACKWARD

    CUDA_CHECK_LAST();

    // ==========================================================================
    // GLOBAL SPLAT BACKWARD PASS
    // ==========================================================================
    int n_global_splats = (int)global_splat_ids.size(0);
    if (n_global_splats > 0) {
        // Compute number of pixels
        int64_t num_pixels = 1;
        for (int d = 0; d < dim; d++) {
            num_pixels *= shape[d];
        }

        // Launch global splat backward kernel
        #define LAUNCH_GLOBAL_BWD(D) \
            launch_rasterize_global_backward<D, float>( \
                grad_output.data_ptr<float>(), \
                centers.data_ptr<float>(), \
                conic.data_ptr<float>(), \
                amps.data_ptr<float>(), \
                sharpness.data_ptr<float>(), \
                global_splat_ids.data_ptr<int>(), \
                n_global_splats, \
                shape_tensor.data_ptr<int>(), \
                truncate, \
                intensity_floor, \
                d_centers.data_ptr<float>(), \
                d_conic.data_ptr<float>(), \
                d_amps.data_ptr<float>(), \
                d_sharpness.data_ptr<float>(), \
                num_pixels, stream)

        switch (dim) {
            case 2: LAUNCH_GLOBAL_BWD(2); break;
            case 3: LAUNCH_GLOBAL_BWD(3); break;
            case 4: LAUNCH_GLOBAL_BWD(4); break;
            case 5: LAUNCH_GLOBAL_BWD(5); break;
            case 6: LAUNCH_GLOBAL_BWD(6); break;
            case 7: LAUNCH_GLOBAL_BWD(7); break;
            case 8: LAUNCH_GLOBAL_BWD(8); break;
            default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
        }
        #undef LAUNCH_GLOBAL_BWD

        CUDA_CHECK_LAST();
    }
}

// =============================================================================
// PUBLIC API
// =============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const torch::Tensor& L_row_norms,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size
) {
    validate_inputs(centers, conic, amps, sharpness, shape);

    // Validate L_row_norms
    int dim = (int)shape.size();
    int N = (int)centers.size(0);
    TORCH_CHECK(L_row_norms.is_cuda(), "L_row_norms must be on CUDA device");
    TORCH_CHECK(L_row_norms.is_contiguous(), "L_row_norms must be contiguous");
    TORCH_CHECK(L_row_norms.size(0) == N && L_row_norms.size(1) == dim,
        "L_row_norms must have shape (", N, ", ", dim, ")");
    TORCH_CHECK(L_row_norms.dtype() == torch::kFloat32, "L_row_norms must be float32");

    // Validate batch_size
    TORCH_CHECK(batch_size == 32 || batch_size == 128 || batch_size == 256,
        "batch_size must be 32, 128, or 256, got ", batch_size);

    auto device = centers.device();

    // Compute output size
    int64_t num_pixels = 1;
    for (int64_t s : shape) {
        num_pixels *= s;
    }

    // Allocate output
    auto output = torch::zeros({num_pixels}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Run forward pass
    BinningState state;
    dispatch_forward(dim, centers, conic, amps, sharpness, L_row_norms, shape,
                    truncate, intensity_floor, tile_size, batch_size, output, state);

    // Return output + binning state + global splat IDs
    return std::make_tuple(
        output,
        state.tile_counts,
        state.tile_offsets,
        state.tile_content,
        state.global_splat_ids  // New: global splat IDs for backward pass
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,  // New: global splat IDs from forward
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size
) {
    validate_inputs(centers, conic, amps, sharpness, shape);

    // Validate batch_size
    TORCH_CHECK(batch_size == 32 || batch_size == 128 || batch_size == 256,
        "batch_size must be 32, 128, or 256, got ", batch_size);

    int dim = (int)shape.size();
    int N = (int)centers.size(0);
    int conic_size = dim * (dim + 1) / 2;
    auto device = centers.device();

    // Allocate gradient buffers
    auto d_centers = torch::zeros({N, dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto d_conic = torch::zeros({N, conic_size}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto d_amps = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto d_sharpness = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Run backward pass (includes global splat handling)
    dispatch_backward(dim, grad_output, centers, conic, amps, sharpness,
                     tile_offsets, tile_counts, tile_content, global_splat_ids,
                     shape, truncate, intensity_floor, tile_size, batch_size,
                     d_centers, d_conic, d_amps, d_sharpness);

    return std::make_tuple(d_centers, d_conic, d_amps, d_sharpness);
}

// =============================================================================
// FP16 (HALF PRECISION) SUPPORT
// =============================================================================

/**
 * Validate FP16 input tensors.
 * Same checks as FP32 but expects kFloat16 dtype.
 */
void validate_inputs_fp16(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const std::vector<int64_t>& shape
) {
    // Check device
    TORCH_CHECK(centers.is_cuda(), "centers must be on CUDA device");
    TORCH_CHECK(conic.is_cuda(), "conic must be on CUDA device");
    TORCH_CHECK(amps.is_cuda(), "amps must be on CUDA device");
    TORCH_CHECK(sharpness.is_cuda(), "sharpness must be on CUDA device");

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
    TORCH_CHECK(sharpness.size(0) == N, "sharpness must have shape (", N, ",)");

    // Check dtypes - expect FP16
    TORCH_CHECK(centers.dtype() == torch::kFloat16, "centers must be float16 for FP16 mode");
    TORCH_CHECK(conic.dtype() == torch::kFloat16, "conic must be float16 for FP16 mode");
    TORCH_CHECK(amps.dtype() == torch::kFloat16, "amps must be float16 for FP16 mode");
    TORCH_CHECK(sharpness.dtype() == torch::kFloat16, "sharpness must be float16 for FP16 mode");

    // Check contiguous
    TORCH_CHECK(centers.is_contiguous(), "centers must be contiguous");
    TORCH_CHECK(conic.is_contiguous(), "conic must be contiguous");
    TORCH_CHECK(amps.is_contiguous(), "amps must be contiguous");
    TORCH_CHECK(sharpness.is_contiguous(), "sharpness must be contiguous");
}

/**
 * FP16 Forward dispatcher (Phase 2 - True FP16 kernels).
 *
 * Mixed precision implementation: loads FP16 directly from global memory,
 * converts to FP32 during shared memory load, computes in FP32, outputs FP32.
 *
 * This provides true 2x memory bandwidth improvement vs Phase 1 which converted
 * FP16->FP32 at the API boundary (before kernel launch).
 *
 * Key difference from FP32 dispatch:
 * - Uses .data_ptr<at::Half>() and casts to __half*
 * - Calls launch_*<D, __half>() instead of launch_*<D>()
 * - Kernels use DTypeTraits<__half>::load() to convert during shared mem load
 */
void dispatch_forward_fp16(
    int dim,
    const torch::Tensor& centers_fp16,
    const torch::Tensor& conic_fp16,
    const torch::Tensor& amps_fp16,
    const torch::Tensor& sharpness_fp16,
    const torch::Tensor& L_row_norms_fp16,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size,
    torch::Tensor& output,
    BinningState& state
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int N = (int)centers_fp16.size(0);
    auto tile_dims = compute_tile_dims(shape, tile_size);
    int64_t num_tiles = compute_num_tiles(tile_dims);

    TORCH_CHECK(num_tiles <= MAX_TILES,
        "Too many tiles (", num_tiles, "). Maximum is ", MAX_TILES,
        ". Increase tile_size or reduce volume size.");

    auto device = centers_fp16.device();

    // Allocate binning state
    state.tile_counts = torch::zeros({num_tiles}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.global_splat_flags = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(device));
    state.num_tiles = num_tiles;

    // Allocate AABB cache (written by preprocess, read by bin)
    state.aabb_lo = torch::empty({N, dim}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.aabb_hi = torch::empty({N, dim}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Atomic counter for global splats (avoids expensive torch::nonzero when 0)
    auto global_count_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Copy shape and tile_dims to device
    state.shape_tensor = torch::tensor(std::vector<int>(shape.begin(), shape.end()),
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.tile_dims_tensor = torch::tensor(tile_dims,
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.tile_size = tile_size;

    auto& shape_tensor = state.shape_tensor;
    auto& tile_dims_tensor = state.tile_dims_tensor;

    // Get FP16 data pointers (cast at::Half* to __half* - they are binary compatible)
    const __half* centers_ptr = reinterpret_cast<const __half*>(centers_fp16.data_ptr<at::Half>());
    const __half* conic_ptr = reinterpret_cast<const __half*>(conic_fp16.data_ptr<at::Half>());
    const __half* amps_ptr = reinterpret_cast<const __half*>(amps_fp16.data_ptr<at::Half>());
    const __half* sharpness_ptr = reinterpret_cast<const __half*>(sharpness_fp16.data_ptr<at::Half>());
    const __half* L_row_norms_ptr = reinterpret_cast<const __half*>(L_row_norms_fp16.data_ptr<at::Half>());

    // Launch preprocess kernel with FP16 inputs
    #define LAUNCH_PREPROCESS_FP16(D) \
        launch_preprocess<D, __half>( \
            centers_ptr, amps_ptr, sharpness_ptr, L_row_norms_ptr, \
            N, \
            shape_tensor.data_ptr<int>(), \
            tile_dims_tensor.data_ptr<int>(), \
            tile_size, truncate, intensity_floor, \
            state.tile_counts.data_ptr<int>(), \
            state.global_splat_flags.data_ptr<bool>(), \
            state.aabb_lo.data_ptr<int>(), \
            state.aabb_hi.data_ptr<int>(), \
            num_tiles, \
            global_count_tensor.data_ptr<int>(), \
            stream)

    switch (dim) {
        case 2: LAUNCH_PREPROCESS_FP16(2); break;
        case 3: LAUNCH_PREPROCESS_FP16(3); break;
        case 4: LAUNCH_PREPROCESS_FP16(4); break;
        case 5: LAUNCH_PREPROCESS_FP16(5); break;
        case 6: LAUNCH_PREPROCESS_FP16(6); break;
        case 7: LAUNCH_PREPROCESS_FP16(7); break;
        case 8: LAUNCH_PREPROCESS_FP16(8); break;
        default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
    }
    #undef LAUNCH_PREPROCESS_FP16

    CUDA_CHECK_LAST();

    // Compute prefix sum for tile offsets
    state.tile_offsets = torch::empty({num_tiles}, torch::TensorOptions().dtype(torch::kInt64).device(device));

    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, temp_bytes,
        state.tile_counts.data_ptr<int>(),
        state.tile_offsets.data_ptr<int64_t>(),
        (int)num_tiles, stream
    );

    state.scan_temp_storage = torch::empty({(int64_t)temp_bytes},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
    state.scan_temp_bytes = temp_bytes;

    cub::DeviceScan::ExclusiveSum(
        state.scan_temp_storage.data_ptr<uint8_t>(), temp_bytes,
        state.tile_counts.data_ptr<int>(),
        state.tile_offsets.data_ptr<int64_t>(),
        (int)num_tiles, stream
    );

    CUDA_CHECK_LAST();

    // Compute total pairs
    int64_t last_offset = 0;
    int last_count = 0;
    cudaMemcpyAsync(&last_offset, state.tile_offsets.data_ptr<int64_t>() + num_tiles - 1,
        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_count, state.tile_counts.data_ptr<int>() + num_tiles - 1,
        sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    state.total_pairs = last_offset + last_count;

    // Allocate tile content
    state.tile_content = torch::empty({state.total_pairs},
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    state.tile_write_heads = torch::zeros({num_tiles},
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Launch binning kernel with FP16 inputs
    #define LAUNCH_BIN_FP16(D) \
        launch_bin<D>( \
            state.aabb_lo.data_ptr<int>(), \
            state.aabb_hi.data_ptr<int>(), \
            state.global_splat_flags.data_ptr<bool>(), \
            N, \
            tile_dims_tensor.data_ptr<int>(), \
            state.tile_offsets.data_ptr<int64_t>(), \
            state.tile_write_heads.data_ptr<int>(), \
            state.tile_content.data_ptr<int>(), \
            num_tiles, stream)

    switch (dim) {
        case 2: LAUNCH_BIN_FP16(2); break;
        case 3: LAUNCH_BIN_FP16(3); break;
        case 4: LAUNCH_BIN_FP16(4); break;
        case 5: LAUNCH_BIN_FP16(5); break;
        case 6: LAUNCH_BIN_FP16(6); break;
        case 7: LAUNCH_BIN_FP16(7); break;
        case 8: LAUNCH_BIN_FP16(8); break;
        default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
    }
    #undef LAUNCH_BIN_FP16

    CUDA_CHECK_LAST();

    // Initialize output to zero
    output.zero_();

    // Launch rasterization kernel with FP16 inputs
    // Dispatch based on batch_size and dimension
    // Note: batch_size must be pre-validated by Python layer based on GPU capabilities
    #define LAUNCH_RASTER_FP16(D, BATCH) \
        launch_rasterize_forward<D, BATCH, __half>( \
            centers_ptr, conic_ptr, amps_ptr, sharpness_ptr, \
            N, \
            shape_tensor.data_ptr<int>(), \
            tile_dims_tensor.data_ptr<int>(), \
            tile_size, truncate, intensity_floor, \
            state.tile_offsets.data_ptr<int64_t>(), \
            state.tile_counts.data_ptr<int>(), \
            state.tile_content.data_ptr<int>(), \
            output.data_ptr<float>(), \
            num_tiles, tile_dims, stream)

    // Clamp batch_size based on dimension (shared memory constraints for 48KB GPUs)
    // Future GPUs with more shared memory can use larger batches - Python layer selects
    int effective_batch = batch_size;
    if (dim >= 7 && effective_batch > 32) effective_batch = 32;
    else if (dim >= 5 && effective_batch > 128) effective_batch = 128;

    #define DISPATCH_RASTER_FP16_BY_DIM_32 \
        switch (dim) { \
            case 2: LAUNCH_RASTER_FP16(2, 32); break; \
            case 3: LAUNCH_RASTER_FP16(3, 32); break; \
            case 4: LAUNCH_RASTER_FP16(4, 32); break; \
            case 5: LAUNCH_RASTER_FP16(5, 32); break; \
            case 6: LAUNCH_RASTER_FP16(6, 32); break; \
            case 7: LAUNCH_RASTER_FP16(7, 32); break; \
            case 8: LAUNCH_RASTER_FP16(8, 32); break; \
            default: TORCH_CHECK(false, "Unsupported dimension: ", dim); \
        }

    #define DISPATCH_RASTER_FP16_BY_DIM_128 \
        switch (dim) { \
            case 2: LAUNCH_RASTER_FP16(2, 128); break; \
            case 3: LAUNCH_RASTER_FP16(3, 128); break; \
            case 4: LAUNCH_RASTER_FP16(4, 128); break; \
            case 5: LAUNCH_RASTER_FP16(5, 128); break; \
            case 6: LAUNCH_RASTER_FP16(6, 128); break; \
            default: TORCH_CHECK(false, "Unsupported dimension for batch_size 128: ", dim); \
        }

    #define DISPATCH_RASTER_FP16_BY_DIM_256 \
        switch (dim) { \
            case 2: LAUNCH_RASTER_FP16(2, 256); break; \
            case 3: LAUNCH_RASTER_FP16(3, 256); break; \
            case 4: LAUNCH_RASTER_FP16(4, 256); break; \
            default: TORCH_CHECK(false, "Unsupported dimension for batch_size 256: ", dim); \
        }

    switch (effective_batch) {
        case 32: DISPATCH_RASTER_FP16_BY_DIM_32; break;
        case 128: DISPATCH_RASTER_FP16_BY_DIM_128; break;
        case 256: DISPATCH_RASTER_FP16_BY_DIM_256; break;
        default: TORCH_CHECK(false, "Unsupported batch_size: ", effective_batch, ". Must be 32, 128, or 256.");
    }
    #undef DISPATCH_RASTER_FP16_BY_DIM_32
    #undef DISPATCH_RASTER_FP16_BY_DIM_128
    #undef DISPATCH_RASTER_FP16_BY_DIM_256
    #undef LAUNCH_RASTER_FP16

    CUDA_CHECK_LAST();

    // Handle global splats - check atomic counter first to avoid expensive torch::nonzero
    int h_global_count = 0;
    cudaMemcpyAsync(&h_global_count, global_count_tensor.data_ptr<int>(),
        sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    state.num_global_splats = h_global_count;

    if (h_global_count > 0) {
        auto global_indices = torch::nonzero(state.global_splat_flags);
        state.global_splat_ids = global_indices.squeeze(1).to(torch::kInt32).contiguous();
        // Use actual count from nonzero (authoritative) rather than atomic counter
        state.num_global_splats = (int)global_indices.size(0);

        int64_t num_pixels = 1;
        for (int d = 0; d < dim; d++) {
            num_pixels *= shape[d];
        }

        #define LAUNCH_GLOBAL_FWD_FP16(D) \
            launch_rasterize_global_forward<D, __half>( \
                centers_ptr, conic_ptr, amps_ptr, sharpness_ptr, \
                state.global_splat_ids.data_ptr<int>(), \
                state.num_global_splats, \
                shape_tensor.data_ptr<int>(), \
                truncate, intensity_floor, \
                output.data_ptr<float>(), \
                num_pixels, stream)

        switch (dim) {
            case 2: LAUNCH_GLOBAL_FWD_FP16(2); break;
            case 3: LAUNCH_GLOBAL_FWD_FP16(3); break;
            case 4: LAUNCH_GLOBAL_FWD_FP16(4); break;
            case 5: LAUNCH_GLOBAL_FWD_FP16(5); break;
            case 6: LAUNCH_GLOBAL_FWD_FP16(6); break;
            case 7: LAUNCH_GLOBAL_FWD_FP16(7); break;
            case 8: LAUNCH_GLOBAL_FWD_FP16(8); break;
            default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
        }
        #undef LAUNCH_GLOBAL_FWD_FP16

        CUDA_CHECK_LAST();
    } else {
        state.global_splat_ids = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    }
}

/**
 * FP16 Backward dispatcher (Phase 2 - True FP16 kernels).
 *
 * Mixed precision implementation: loads FP16 directly from global memory,
 * converts to FP32 during shared memory load, computes in FP32.
 * Gradients are always FP32 for numerical stability.
 *
 * This provides true 2x memory bandwidth improvement for inputs.
 */
void dispatch_backward_fp16(
    int dim,
    const torch::Tensor& grad_output,
    const torch::Tensor& centers_fp16,
    const torch::Tensor& conic_fp16,
    const torch::Tensor& amps_fp16,
    const torch::Tensor& sharpness_fp16,
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
    torch::Tensor& d_sharpness
) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int N = (int)centers_fp16.size(0);
    auto tile_dims = compute_tile_dims(shape, tile_size);
    int64_t num_tiles = compute_num_tiles(tile_dims);

    auto device = centers_fp16.device();

    // Copy shape and tile_dims to device
    auto shape_tensor = torch::tensor(std::vector<int>(shape.begin(), shape.end()),
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto tile_dims_tensor = torch::tensor(tile_dims,
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Zero gradient buffers
    d_centers.zero_();
    d_conic.zero_();
    d_amps.zero_();
    d_sharpness.zero_();

    // Get FP16 data pointers (cast at::Half* to __half* - binary compatible)
    const __half* centers_ptr = reinterpret_cast<const __half*>(centers_fp16.data_ptr<at::Half>());
    const __half* conic_ptr = reinterpret_cast<const __half*>(conic_fp16.data_ptr<at::Half>());
    const __half* amps_ptr = reinterpret_cast<const __half*>(amps_fp16.data_ptr<at::Half>());
    const __half* sharpness_ptr = reinterpret_cast<const __half*>(sharpness_fp16.data_ptr<at::Half>());

    // Launch backward kernel for tile-based splats with FP16 inputs
    // Dispatch based on batch_size and dimension
    #define LAUNCH_BACKWARD_FP16(D, BATCH) \
        launch_rasterize_backward<D, BATCH, __half>( \
            grad_output.data_ptr<float>(), \
            centers_ptr, conic_ptr, amps_ptr, sharpness_ptr, \
            N, \
            shape_tensor.data_ptr<int>(), \
            tile_dims_tensor.data_ptr<int>(), \
            tile_size, truncate, intensity_floor, \
            tile_offsets.data_ptr<int64_t>(), \
            tile_counts.data_ptr<int>(), \
            tile_content.data_ptr<int>(), \
            d_centers.data_ptr<float>(), \
            d_conic.data_ptr<float>(), \
            d_amps.data_ptr<float>(), \
            d_sharpness.data_ptr<float>(), \
            num_tiles, tile_dims, stream)

    // Clamp batch_size based on dimension (shared memory constraints for 48KB GPUs)
    int effective_batch = batch_size;
    if (dim >= 7 && effective_batch > 32) effective_batch = 32;
    else if (dim >= 5 && effective_batch > 128) effective_batch = 128;

    #define DISPATCH_BACKWARD_FP16_BY_DIM_32 \
        switch (dim) { \
            case 2: LAUNCH_BACKWARD_FP16(2, 32); break; \
            case 3: LAUNCH_BACKWARD_FP16(3, 32); break; \
            case 4: LAUNCH_BACKWARD_FP16(4, 32); break; \
            case 5: LAUNCH_BACKWARD_FP16(5, 32); break; \
            case 6: LAUNCH_BACKWARD_FP16(6, 32); break; \
            case 7: LAUNCH_BACKWARD_FP16(7, 32); break; \
            case 8: LAUNCH_BACKWARD_FP16(8, 32); break; \
            default: TORCH_CHECK(false, "Unsupported dimension: ", dim); \
        }

    #define DISPATCH_BACKWARD_FP16_BY_DIM_128 \
        switch (dim) { \
            case 2: LAUNCH_BACKWARD_FP16(2, 128); break; \
            case 3: LAUNCH_BACKWARD_FP16(3, 128); break; \
            case 4: LAUNCH_BACKWARD_FP16(4, 128); break; \
            case 5: LAUNCH_BACKWARD_FP16(5, 128); break; \
            case 6: LAUNCH_BACKWARD_FP16(6, 128); break; \
            default: TORCH_CHECK(false, "Unsupported dimension for batch_size 128: ", dim); \
        }

    #define DISPATCH_BACKWARD_FP16_BY_DIM_256 \
        switch (dim) { \
            case 2: LAUNCH_BACKWARD_FP16(2, 256); break; \
            case 3: LAUNCH_BACKWARD_FP16(3, 256); break; \
            case 4: LAUNCH_BACKWARD_FP16(4, 256); break; \
            default: TORCH_CHECK(false, "Unsupported dimension for batch_size 256: ", dim); \
        }

    switch (effective_batch) {
        case 32: DISPATCH_BACKWARD_FP16_BY_DIM_32; break;
        case 128: DISPATCH_BACKWARD_FP16_BY_DIM_128; break;
        case 256: DISPATCH_BACKWARD_FP16_BY_DIM_256; break;
        default: TORCH_CHECK(false, "Unsupported batch_size: ", effective_batch, ". Must be 32, 128, or 256.");
    }
    #undef DISPATCH_BACKWARD_FP16_BY_DIM_32
    #undef DISPATCH_BACKWARD_FP16_BY_DIM_128
    #undef DISPATCH_BACKWARD_FP16_BY_DIM_256
    #undef LAUNCH_BACKWARD_FP16

    CUDA_CHECK_LAST();

    // Global splat backward pass
    int n_global_splats = (int)global_splat_ids.size(0);
    if (n_global_splats > 0) {
        int64_t num_pixels = 1;
        for (int d = 0; d < dim; d++) {
            num_pixels *= shape[d];
        }

        #define LAUNCH_GLOBAL_BWD_FP16(D) \
            launch_rasterize_global_backward<D, __half>( \
                grad_output.data_ptr<float>(), \
                centers_ptr, conic_ptr, amps_ptr, sharpness_ptr, \
                global_splat_ids.data_ptr<int>(), \
                n_global_splats, \
                shape_tensor.data_ptr<int>(), \
                truncate, intensity_floor, \
                d_centers.data_ptr<float>(), \
                d_conic.data_ptr<float>(), \
                d_amps.data_ptr<float>(), \
                d_sharpness.data_ptr<float>(), \
                num_pixels, stream)

        switch (dim) {
            case 2: LAUNCH_GLOBAL_BWD_FP16(2); break;
            case 3: LAUNCH_GLOBAL_BWD_FP16(3); break;
            case 4: LAUNCH_GLOBAL_BWD_FP16(4); break;
            case 5: LAUNCH_GLOBAL_BWD_FP16(5); break;
            case 6: LAUNCH_GLOBAL_BWD_FP16(6); break;
            case 7: LAUNCH_GLOBAL_BWD_FP16(7); break;
            case 8: LAUNCH_GLOBAL_BWD_FP16(8); break;
            default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
        }
        #undef LAUNCH_GLOBAL_BWD_FP16

        CUDA_CHECK_LAST();
    }
}

// =============================================================================
// FP16 PUBLIC API
// =============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_fp16(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const torch::Tensor& L_row_norms,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size
) {
    validate_inputs_fp16(centers, conic, amps, sharpness, shape);

    // Validate L_row_norms
    int dim = (int)shape.size();
    int N = (int)centers.size(0);
    TORCH_CHECK(L_row_norms.is_cuda(), "L_row_norms must be on CUDA device");
    TORCH_CHECK(L_row_norms.is_contiguous(), "L_row_norms must be contiguous");
    TORCH_CHECK(L_row_norms.size(0) == N && L_row_norms.size(1) == dim,
        "L_row_norms must have shape (", N, ", ", dim, ")");
    TORCH_CHECK(L_row_norms.dtype() == torch::kFloat16, "L_row_norms must be float16 for FP16 mode");

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

    // Run forward pass with FP16 inputs
    BinningState state;
    dispatch_forward_fp16(dim, centers, conic, amps, sharpness, L_row_norms, shape,
                         truncate, intensity_floor, tile_size, batch_size, output, state);

    // Return output + binning state + global splat IDs
    return std::make_tuple(
        output,
        state.tile_counts,
        state.tile_offsets,
        state.tile_content,
        state.global_splat_ids
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_fp16(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size
) {
    validate_inputs_fp16(centers, conic, amps, sharpness, shape);

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
    auto d_sharpness = torch::zeros({N}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Run backward pass with FP16 inputs
    dispatch_backward_fp16(dim, grad_output, centers, conic, amps, sharpness,
                          tile_offsets, tile_counts, tile_content, global_splat_ids,
                          shape, truncate, intensity_floor, tile_size, batch_size,
                          d_centers, d_conic, d_amps, d_sharpness);

    return std::make_tuple(d_centers, d_conic, d_amps, d_sharpness);
}
