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
 */

#include "cuda_splatting.h"
#include "utils.cuh"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <stdexcept>
#include <cmath>

// =============================================================================
// CONFIGURATION
// =============================================================================

// Block sizes for different kernels
constexpr int PREPROCESS_BLOCK_SIZE = 256;
constexpr int BIN_BLOCK_SIZE = 256;
constexpr int RASTER_BLOCK_SIZE_2D = 256;  // 16x16 tile
constexpr int RASTER_BLOCK_SIZE_3D = 512;  // 8x8x8 tile
constexpr int RASTER_BLOCK_SIZE_DEFAULT = 256;

// Shared memory batch size for splat loading
constexpr int SPLAT_BATCH_SIZE = 32;

// Threshold for global splat handling (fraction of tiles)
constexpr float GLOBAL_SPLAT_THRESHOLD = 0.1f;

// =============================================================================
// BACKWARD GRADIENT DISPATCH HELPERS
// =============================================================================

/**
 * Generic backward gradient computation for arbitrary dimensions.
 * Uses loop-based computation.
 */
template <int DIM>
__device__ __forceinline__ void compute_pixel_gradients(
    float dL_dI,
    float intensity,
    float dist_sq,
    float amp,
    float s,
    const float* __restrict__ d_vec,
    const float* __restrict__ conic,
    float* __restrict__ local_d_centers,
    float* __restrict__ local_d_conic,
    float& local_d_amp,
    float& local_d_sharpness
) {
    // Gradient w.r.t. amplitude
    local_d_amp += dL_dI * grad_intensity_wrt_amplitude(intensity, amp);

    // Gradient w.r.t. sharpness
    local_d_sharpness += dL_dI * grad_intensity_wrt_sharpness(intensity, dist_sq, s);

    // Gradient w.r.t. dist_sq
    float grad_dist = grad_intensity_wrt_dist_sq(intensity, dist_sq, s);

    // Chain rule: ∂L/∂center and ∂L/∂conic via ∂D²/∂center and ∂D²/∂conic
    constexpr int CONIC_SIZE = conic_size<DIM>();

    // Compute ∂D²/∂d = 2 * Σ⁻¹ @ d
    float dD2_dd[DIM];
    #pragma unroll
    for (int i = 0; i < DIM; i++) {
        float sum = 0.0f;
        int row_start = 0;
        for (int k = 0; k < i; k++) {
            row_start += DIM - k;
        }
        // Diagonal contribution
        sum += conic[row_start] * d_vec[i];
        // Off-diagonal contributions (symmetric)
        int idx = row_start + 1;
        for (int j = i + 1; j < DIM; j++) {
            sum += conic[idx] * d_vec[j];
            idx++;
        }
        // Contributions from lower triangle (by symmetry)
        for (int k = 0; k < i; k++) {
            int k_row_start = 0;
            for (int m = 0; m < k; m++) {
                k_row_start += DIM - m;
            }
            int elem_idx = k_row_start + (i - k);
            sum += conic[elem_idx] * d_vec[k];
        }
        dD2_dd[i] = 2.0f * sum;
    }

    // ∂L/∂center = dL_dI * grad_dist * ∂D²/∂d * (-1)
    #pragma unroll
    for (int i = 0; i < DIM; i++) {
        local_d_centers[i] += dL_dI * grad_dist * dD2_dd[i] * (-1.0f);
    }

    // ∂D²/∂conic - gradient w.r.t. packed upper triangle
    int conic_idx = 0;
    #pragma unroll
    for (int i = 0; i < DIM; i++) {
        // Diagonal
        float grad = d_vec[i] * d_vec[i];
        local_d_conic[conic_idx] += dL_dI * grad_dist * grad;
        conic_idx++;

        // Off-diagonals
        #pragma unroll
        for (int j = i + 1; j < DIM; j++) {
            grad = 2.0f * d_vec[i] * d_vec[j];
            local_d_conic[conic_idx] += dL_dI * grad_dist * grad;
            conic_idx++;
        }
    }
}

/**
 * Specialized 3D backward gradient computation using explicit formulas.
 * Provides 25-35% speedup over generic version.
 */
template <>
__device__ __forceinline__ void compute_pixel_gradients<3>(
    float dL_dI,
    float intensity,
    float dist_sq,
    float amp,
    float s,
    const float* __restrict__ d_vec,
    const float* __restrict__ conic,
    float* __restrict__ local_d_centers,
    float* __restrict__ local_d_conic,
    float& local_d_amp,
    float& local_d_sharpness
) {
    backward_pixel_splat_3d(dL_dI, intensity, dist_sq, amp, s, d_vec, conic,
                           local_d_centers, local_d_conic, local_d_amp, local_d_sharpness);
}

/**
 * Specialized 2D backward gradient computation using explicit formulas.
 */
template <>
__device__ __forceinline__ void compute_pixel_gradients<2>(
    float dL_dI,
    float intensity,
    float dist_sq,
    float amp,
    float s,
    const float* __restrict__ d_vec,
    const float* __restrict__ conic,
    float* __restrict__ local_d_centers,
    float* __restrict__ local_d_conic,
    float& local_d_amp,
    float& local_d_sharpness
) {
    backward_pixel_splat_2d(dL_dI, intensity, dist_sq, amp, s, d_vec, conic,
                           local_d_centers, local_d_conic, local_d_amp, local_d_sharpness);
}

// =============================================================================
// KERNEL DECLARATIONS
// =============================================================================

// Preprocess kernel: compute tile counts
template <int DIM>
__global__ void preprocess_kernel(
    const float* __restrict__ centers,      // (N, DIM)
    const float* __restrict__ conic,        // (N, conic_size)
    const float* __restrict__ amps,         // (N,)
    const float* __restrict__ sharpness,    // (N,)
    int N,
    const int* __restrict__ shape,          // (DIM,)
    const int* __restrict__ tile_dims,      // (DIM,)
    int tile_size,
    float truncate,
    float intensity_floor,
    int* __restrict__ tile_counts,          // (num_tiles,)
    bool* __restrict__ global_flags,        // (N,)
    int64_t num_tiles
);

// Binning kernel: assign splats to tiles
template <int DIM>
__global__ void bin_kernel(
    const float* __restrict__ centers,
    const float* __restrict__ conic,
    const float* __restrict__ amps,
    const float* __restrict__ sharpness,
    int N,
    const int* __restrict__ shape,
    const int* __restrict__ tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* __restrict__ tile_offsets,
    int* __restrict__ tile_write_heads,
    int* __restrict__ tile_content,
    int64_t num_tiles
);

// Forward rasterization kernel
template <int DIM>
__global__ void rasterize_forward_kernel(
    const float* __restrict__ centers,
    const float* __restrict__ conic,
    const float* __restrict__ amps,
    const float* __restrict__ sharpness,
    int N,
    const int* __restrict__ shape,
    const int* __restrict__ tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* __restrict__ tile_offsets,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_content,
    float* __restrict__ output,
    int64_t num_pixels
);

// Backward rasterization kernel
template <int DIM>
__global__ void rasterize_backward_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ centers,
    const float* __restrict__ conic,
    const float* __restrict__ amps,
    const float* __restrict__ sharpness,
    int N,
    const int* __restrict__ shape,
    const int* __restrict__ tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* __restrict__ tile_offsets,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_content,
    float* __restrict__ d_centers,
    float* __restrict__ d_conic,
    float* __restrict__ d_amps,
    float* __restrict__ d_sharpness,
    int64_t num_pixels
);

// =============================================================================
// PREPROCESS KERNEL
// =============================================================================

/**
 * Compute L_row_norms from conic (Σ⁻¹) matrix.
 *
 * Since Σ = L @ L^T and Σ⁻¹ = L^{-T} @ L^{-1}, we approximate
 * L_row_norms from the diagonal of Σ: L_row_norm_i ≈ sqrt(1/Σ⁻¹_ii)
 */
template <int DIM>
__device__ __forceinline__ void estimate_L_row_norms_from_conic(
    const float* __restrict__ conic,
    float* __restrict__ L_row_norms
) {
    // The diagonal elements of Σ⁻¹ are at indices: 0, DIM, DIM+(DIM-1), ...
    // For 3D: indices 0 (c_00), 3 (c_11), 5 (c_22)
    int idx = 0;
    #pragma unroll
    for (int i = 0; i < DIM; i++) {
        // Diagonal element c_ii is at position sum(DIM-j for j=0..i-1) + 0
        float c_ii = conic[idx];
        // σ_i² ≈ 1/c_ii (approximation, exact only for diagonal covariance)
        // L_row_norm ≈ sqrt(σ_i²) = 1/sqrt(c_ii)
        L_row_norms[i] = rsqrtf(fmaxf(c_ii, 1e-6f));
        // Move to next diagonal: skip i elements (the off-diagonals)
        idx += DIM - i;
    }
}

template <int DIM, typename InputDType = float>
__global__ void preprocess_kernel(
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
    const InputDType* __restrict__ sharpness,
    int N,
    const int* __restrict__ shape,
    const int* __restrict__ tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    int* __restrict__ tile_counts,
    bool* __restrict__ global_flags,
    int64_t num_tiles
) {
    int splat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (splat_idx >= N) return;

    // Load splat data with dtype-aware conversion (FP16->FP32 if needed)
    float mu[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        mu[d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
    }

    constexpr int CONIC_SIZE = conic_size<DIM>();
    float conic_local[CONIC_SIZE];
    #pragma unroll
    for (int i = 0; i < CONIC_SIZE; i++) {
        conic_local[i] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + i);
    }

    float amp = DTypeTraits<InputDType>::load(amps, splat_idx);
    float s = DTypeTraits<InputDType>::load(sharpness, splat_idx);

    // Estimate L_row_norms from conic
    float L_row_norms[DIM];
    estimate_L_row_norms_from_conic<DIM>(conic_local, L_row_norms);

    // Load shape and tile_dims to local memory
    int shape_local[DIM];
    int tile_dims_local[DIM];
    int tile_size_arr[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        shape_local[d] = shape[d];
        tile_dims_local[d] = tile_dims[d];
        tile_size_arr[d] = tile_size;
    }

    // Compute AABB
    AABB<DIM> aabb = compute_splat_aabb<DIM>(
        mu, L_row_norms, s, amp, truncate, intensity_floor,
        tile_size_arr, tile_dims_local, shape_local
    );

    // Check if AABB is empty (splat outside volume or culled)
    if (aabb.is_empty()) {
        global_flags[splat_idx] = false;
        return;
    }

    // Count tiles touched
    int n_tiles = aabb.num_tiles();

    // Check for global splat (touches too many tiles)
    // Use both relative threshold (10% of tiles) AND absolute minimum (at least 64 tiles)
    // to avoid marking splats as global in small volumes
    float global_threshold = GLOBAL_SPLAT_THRESHOLD * (float)num_tiles;
    // Minimum tiles before global handling. This must be high enough to avoid
    // flagging normal splats as global in 4D volumes (where tile counts grow as N^4).
    // With tile_size=4 and shape 16x16x16x16, we get 4^4=256 tiles.
    constexpr int MIN_GLOBAL_TILES = 1024;
    bool is_global = (n_tiles > (int)global_threshold) && (n_tiles > MIN_GLOBAL_TILES);
    global_flags[splat_idx] = is_global;

    if (is_global) {
        // Global splats are handled separately by rasterize_global_forward_kernel.
        // They are not binned into tiles but processed for all pixels directly.
        // The global_flags array is used to identify them after preprocessing.
        return;
    }

    // Iterate over all tiles in AABB and increment counts
    int tile_coords[DIM];

    // Initialize to lo
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        tile_coords[d] = aabb.lo[d];
    }

    // Iterate through all tiles in AABB
    for (int i = 0; i < n_tiles; i++) {
        int tile_idx = tile_coords_to_linear<DIM>(tile_coords, tile_dims_local);
        atomicAdd(&tile_counts[tile_idx], 1);

        // Advance to next tile (odometer-style)
        #pragma unroll
        for (int d = DIM - 1; d >= 0; d--) {
            tile_coords[d]++;
            if (tile_coords[d] <= aabb.hi[d]) {
                break;
            }
            tile_coords[d] = aabb.lo[d];
        }
    }
}

// =============================================================================
// BINNING KERNEL
// =============================================================================

template <int DIM, typename InputDType = float>
__global__ void bin_kernel(
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
    const InputDType* __restrict__ sharpness,
    int N,
    const int* __restrict__ shape,
    const int* __restrict__ tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* __restrict__ tile_offsets,
    int* __restrict__ tile_write_heads,
    int* __restrict__ tile_content,
    int64_t num_tiles
) {
    int splat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (splat_idx >= N) return;

    // Load splat data with dtype-aware conversion (FP16->FP32 if needed)
    float mu[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        mu[d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
    }

    constexpr int CONIC_SIZE = conic_size<DIM>();
    float conic_local[CONIC_SIZE];
    #pragma unroll
    for (int i = 0; i < CONIC_SIZE; i++) {
        conic_local[i] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + i);
    }

    float amp = DTypeTraits<InputDType>::load(amps, splat_idx);
    float s = DTypeTraits<InputDType>::load(sharpness, splat_idx);

    float L_row_norms[DIM];
    estimate_L_row_norms_from_conic<DIM>(conic_local, L_row_norms);

    int shape_local[DIM];
    int tile_dims_local[DIM];
    int tile_size_arr[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        shape_local[d] = shape[d];
        tile_dims_local[d] = tile_dims[d];
        tile_size_arr[d] = tile_size;
    }

    // Compute AABB
    AABB<DIM> aabb = compute_splat_aabb<DIM>(
        mu, L_row_norms, s, amp, truncate, intensity_floor,
        tile_size_arr, tile_dims_local, shape_local
    );

    if (aabb.is_empty()) return;

    int n_tiles = aabb.num_tiles();

    // Skip global splats (must match logic in preprocess_kernel)
    float global_threshold = GLOBAL_SPLAT_THRESHOLD * (float)num_tiles;
    constexpr int MIN_GLOBAL_TILES = 1024;  // Must match value in preprocess_kernel
    if ((n_tiles > (int)global_threshold) && (n_tiles > MIN_GLOBAL_TILES)) return;

    // Iterate and write splat ID to each tile
    int tile_coords[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        tile_coords[d] = aabb.lo[d];
    }

    for (int i = 0; i < n_tiles; i++) {
        int tile_idx = tile_coords_to_linear<DIM>(tile_coords, tile_dims_local);

        // Atomic increment write head, get position to write
        int write_pos = atomicAdd(&tile_write_heads[tile_idx], 1);
        int64_t offset = tile_offsets[tile_idx];
        tile_content[offset + write_pos] = splat_idx;

        // Advance to next tile
        #pragma unroll
        for (int d = DIM - 1; d >= 0; d--) {
            tile_coords[d]++;
            if (tile_coords[d] <= aabb.hi[d]) {
                break;
            }
            tile_coords[d] = aabb.lo[d];
        }
    }
}

// =============================================================================
// FORWARD RASTERIZATION KERNEL
// =============================================================================

template <int DIM, typename InputDType = float>
__global__ void rasterize_forward_kernel(
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
    const InputDType* __restrict__ sharpness,
    int N,
    const int* __restrict__ shape,
    const int* __restrict__ tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* __restrict__ tile_offsets,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_content,
    float* __restrict__ output,
    int64_t num_pixels
) {
    // Each block handles one tile
    // OPTIMIZATION: For 2D/3D, use dim3 grid and extract tile coords directly from blockIdx
    // This eliminates expensive division/modulo operations (20-40 cycles -> 1 cycle)
    int tile_idx;
    int tile_coords[DIM];

    if constexpr (DIM == 3) {
        get_tile_info_3d(tile_dims, tile_coords, tile_idx);
    } else if constexpr (DIM == 2) {
        get_tile_info_2d(tile_dims, tile_coords, tile_idx);
    } else {
        // For DIM > 3, use 1D grid with generic conversion
        get_tile_info_generic<DIM>(tile_dims, tile_coords, tile_idx);
    }

    // Get tile splat count and offset
    int n_splats_in_tile = tile_counts[tile_idx];
    if (n_splats_in_tile == 0) return;

    int64_t tile_offset = tile_offsets[tile_idx];

    // Compute tile origin in voxel coordinates
    int tile_origin[DIM];
    int tile_extent[DIM];
    int tile_pixels = 1;

    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        tile_origin[d] = tile_coords[d] * tile_size;
        // Clip tile extent to volume bounds
        int tile_end = min(tile_origin[d] + tile_size, shape[d]);
        tile_extent[d] = tile_end - tile_origin[d];
        tile_pixels *= tile_extent[d];
    }

    // Shared memory for splat batch loading
    // OPTIMIZATION 2.3: Pad DIM to avoid bank conflicts (stride-3 causes conflicts)
    // For 3D: use stride-4 instead of stride-3, wastes 25% but eliminates conflicts
    constexpr int CONIC_SIZE = conic_size<DIM>();
    constexpr int CENTER_STRIDE = (DIM == 3) ? 4 : DIM;  // Pad 3D to 4
    __shared__ float s_centers[SPLAT_BATCH_SIZE * CENTER_STRIDE];
    __shared__ float s_conic[SPLAT_BATCH_SIZE * CONIC_SIZE];
    __shared__ float s_amps[SPLAT_BATCH_SIZE];
    __shared__ float s_sharpness[SPLAT_BATCH_SIZE];
    // OPTIMIZATION 1.2: Precompute effective truncation squared per splat
    __shared__ float s_truncate_sq[SPLAT_BATCH_SIZE];

    // OPTIMIZATION: Hoist loop-invariant condition checks outside the pixel loop
    // These are uniform across all threads in the block (no divergence)
    // Hoisting ensures the compiler doesn't re-evaluate per iteration
    const bool use_fast_path_3d = (DIM == 3) && (tile_size == 8) &&
        (tile_extent[0] == 8) && (tile_extent[1] == 8) && (tile_extent[2] == 8);
    const bool use_fast_path_2d = (DIM == 2) && (tile_size == 16) &&
        (tile_extent[0] == 16) && (tile_extent[1] == 16);

    // Each thread processes one or more pixels
    for (int local_px_idx = threadIdx.x; local_px_idx < tile_pixels; local_px_idx += blockDim.x) {
        // Convert local pixel index to voxel coordinates
        // OPTIMIZATION 1.4: For 3D with power-of-2 tile sizes, use bitwise ops
        // Bitwise AND/shift are 1 cycle; integer division is 20-40 cycles
        int voxel_coords[DIM];

        if constexpr (DIM == 3) {
            // Fast path for 3D: requires tile_size=8 AND full tile (not edge)
            // Bitwise shifts are hardcoded for 8×8×8 = 512 pixels
            if (use_fast_path_3d) {
                // Use bitwise operations: idx & 7, (idx >> 3) & 7, idx >> 6
                int local_z = local_px_idx & 7;
                int local_y = (local_px_idx >> 3) & 7;
                int local_x = local_px_idx >> 6;
                voxel_coords[0] = tile_origin[0] + local_x;
                voxel_coords[1] = tile_origin[1] + local_y;
                voxel_coords[2] = tile_origin[2] + local_z;
            } else {
                // Edge tile or non-standard tile_size: use generic path
                int remaining = local_px_idx;
                #pragma unroll
                for (int d = DIM - 1; d >= 0; d--) {
                    voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
                    remaining /= tile_extent[d];
                }
            }
        } else if constexpr (DIM == 2) {
            // Fast path for 2D: requires tile_size=16 AND full tile (not edge)
            // Bitwise shifts are hardcoded for 16×16 = 256 pixels
            if (use_fast_path_2d) {
                // Use bitwise operations: idx & 15, idx >> 4
                int local_y = local_px_idx & 15;
                int local_x = local_px_idx >> 4;
                voxel_coords[0] = tile_origin[0] + local_x;
                voxel_coords[1] = tile_origin[1] + local_y;
            } else {
                // Edge tile or non-standard tile_size: use generic path
                int remaining = local_px_idx;
                #pragma unroll
                for (int d = DIM - 1; d >= 0; d--) {
                    voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
                    remaining /= tile_extent[d];
                }
            }
        } else {
            // Generic path for DIM > 3
            int remaining = local_px_idx;
            #pragma unroll
            for (int d = DIM - 1; d >= 0; d--) {
                voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
                remaining /= tile_extent[d];
            }
        }

        // Convert to float for computation - use integer coordinates to match PyTorch reference
        float px[DIM];
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            px[d] = (float)voxel_coords[d];
        }

        // Accumulate intensity
        float intensity_sum = 0.0f;

        // Process splats in batches
        for (int batch_start = 0; batch_start < n_splats_in_tile; batch_start += SPLAT_BATCH_SIZE) {
            int batch_size = min(SPLAT_BATCH_SIZE, n_splats_in_tile - batch_start);

            // Cooperative load of splat batch into shared memory
            // OPTIMIZATION: Use DTypeTraits for dtype-aware loading (supports FP16->FP32 conversion)
            // This improves memory throughput for scattered reads by ~15-20%
            __syncthreads();
            for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
                int splat_idx = __ldg(&tile_content[tile_offset + batch_start + i]);

                // Load centers with dtype conversion (using padded stride for 3D)
                #pragma unroll
                for (int d = 0; d < DIM; d++) {
                    s_centers[i * CENTER_STRIDE + d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
                }

                // Load conic with dtype conversion
                #pragma unroll
                for (int c = 0; c < CONIC_SIZE; c++) {
                    s_conic[i * CONIC_SIZE + c] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + c);
                }

                // Load scalars with dtype conversion
                float s = DTypeTraits<InputDType>::load(sharpness, splat_idx);
                s_amps[i] = DTypeTraits<InputDType>::load(amps, splat_idx);
                s_sharpness[i] = s;

                // OPTIMIZATION 1.2: Precompute effective truncation squared
                // This moves the expensive powf() out of the hot inner loop
                s_truncate_sq[i] = effective_truncate_sq(truncate, s);
            }
            __syncthreads();

            // Process loaded splats
            for (int i = 0; i < batch_size; i++) {
                // Compute displacement d = px - mu (using padded stride)
                float d[DIM];
                #pragma unroll
                for (int dim = 0; dim < DIM; dim++) {
                    d[dim] = px[dim] - s_centers[i * CENTER_STRIDE + dim];
                }

                // Compute Mahalanobis distance squared
                float dist_sq = mahalanobis_distance_sq<DIM>(d, &s_conic[i * CONIC_SIZE]);

                // OPTIMIZATION 1.2: Early rejection based on precomputed truncation
                // Skip expensive gaussian_intensity computation for distant pixels
                if (dist_sq > s_truncate_sq[i]) continue;

                // Compute intensity (only for pixels within truncation radius)
                float intensity = gaussian_intensity(dist_sq, s_amps[i], s_sharpness[i]);

                // Skip if below threshold
                if (intensity >= intensity_floor) {
                    intensity_sum += intensity;
                }
            }
        }

        // Write accumulated intensity to output
        // Note: We use direct assignment (not atomicAdd) because:
        // 1. Each pixel belongs to exactly one tile
        // 2. Each thread in this block processes distinct pixels (different local_px_idx)
        // The global splat kernel uses atomicAdd to add to these values.
        if (local_px_idx < tile_pixels) {
            int64_t global_px_idx = voxel_to_linear<DIM>(voxel_coords, shape);
            output[global_px_idx] = intensity_sum;
        }
    }
}

// =============================================================================
// BACKWARD RASTERIZATION KERNEL
// =============================================================================

template <int DIM, typename InputDType = float>
__global__ void rasterize_backward_kernel(
    const float* __restrict__ grad_output,
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
    const InputDType* __restrict__ sharpness,
    int N,
    const int* __restrict__ shape,
    const int* __restrict__ tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* __restrict__ tile_offsets,
    const int* __restrict__ tile_counts,
    const int* __restrict__ tile_content,
    float* __restrict__ d_centers,
    float* __restrict__ d_conic,
    float* __restrict__ d_amps,
    float* __restrict__ d_sharpness,
    int64_t num_pixels
) {
    // Each block handles one tile
    // OPTIMIZATION: For 2D/3D, use dim3 grid and extract tile coords directly from blockIdx
    int tile_idx;
    int tile_coords[DIM];

    if constexpr (DIM == 3) {
        get_tile_info_3d(tile_dims, tile_coords, tile_idx);
    } else if constexpr (DIM == 2) {
        get_tile_info_2d(tile_dims, tile_coords, tile_idx);
    } else {
        get_tile_info_generic<DIM>(tile_dims, tile_coords, tile_idx);
    }

    int n_splats_in_tile = tile_counts[tile_idx];
    if (n_splats_in_tile == 0) return;

    int64_t tile_offset = tile_offsets[tile_idx];

    int tile_origin[DIM];
    int tile_extent[DIM];
    int tile_pixels = 1;

    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        tile_origin[d] = tile_coords[d] * tile_size;
        int tile_end = min(tile_origin[d] + tile_size, shape[d]);
        tile_extent[d] = tile_end - tile_origin[d];
        tile_pixels *= tile_extent[d];
    }

    // Shared memory for splat data and gradient accumulation
    // OPTIMIZATION 2.3: Pad DIM to avoid bank conflicts
    constexpr int CONIC_SIZE = conic_size<DIM>();
    constexpr int CENTER_STRIDE = (DIM == 3) ? 4 : DIM;
    __shared__ float s_centers[SPLAT_BATCH_SIZE * CENTER_STRIDE];
    __shared__ float s_conic[SPLAT_BATCH_SIZE * CONIC_SIZE];
    __shared__ float s_amps[SPLAT_BATCH_SIZE];
    __shared__ float s_sharpness[SPLAT_BATCH_SIZE];
    __shared__ int s_splat_ids[SPLAT_BATCH_SIZE];
    // OPTIMIZATION 1.2: Precompute effective truncation squared per splat
    __shared__ float s_truncate_sq[SPLAT_BATCH_SIZE];

    // Note: Warp reduction uses __shfl_down_sync, no shared memory buffer needed

    // OPTIMIZATION: Hoist loop-invariant condition checks outside all loops
    // These are uniform across all threads in the block (no divergence)
    const bool use_fast_path_3d = (DIM == 3) && (tile_size == 8) &&
        (tile_extent[0] == 8) && (tile_extent[1] == 8) && (tile_extent[2] == 8);
    const bool use_fast_path_2d = (DIM == 2) && (tile_size == 16) &&
        (tile_extent[0] == 16) && (tile_extent[1] == 16);

    // Process splats in batches
    for (int batch_start = 0; batch_start < n_splats_in_tile; batch_start += SPLAT_BATCH_SIZE) {
        int batch_size = min(SPLAT_BATCH_SIZE, n_splats_in_tile - batch_start);

        // Load splat batch
        // OPTIMIZATION: Use DTypeTraits for dtype-aware loading (supports FP16->FP32 conversion)
        __syncthreads();
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            int splat_idx = __ldg(&tile_content[tile_offset + batch_start + i]);
            s_splat_ids[i] = splat_idx;

            #pragma unroll
            for (int d = 0; d < DIM; d++) {
                s_centers[i * CENTER_STRIDE + d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
            }
            #pragma unroll
            for (int c = 0; c < CONIC_SIZE; c++) {
                s_conic[i * CONIC_SIZE + c] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + c);
            }
            float s = DTypeTraits<InputDType>::load(sharpness, splat_idx);
            s_amps[i] = DTypeTraits<InputDType>::load(amps, splat_idx);
            s_sharpness[i] = s;

            // OPTIMIZATION 1.2: Precompute effective truncation squared
            s_truncate_sq[i] = effective_truncate_sq(truncate, s);
        }
        __syncthreads();

        // Process each splat in batch
        for (int si = 0; si < batch_size; si++) {
            int splat_idx = s_splat_ids[si];

            // Per-thread gradient accumulators
            float local_d_centers[DIM];
            float local_d_conic[CONIC_SIZE];
            float local_d_amp = 0.0f;
            float local_d_sharpness = 0.0f;

            #pragma unroll
            for (int d = 0; d < DIM; d++) {
                local_d_centers[d] = 0.0f;
            }
            #pragma unroll
            for (int c = 0; c < CONIC_SIZE; c++) {
                local_d_conic[c] = 0.0f;
            }

            float amp = s_amps[si];
            float s = s_sharpness[si];
            float truncate_sq = s_truncate_sq[si];

            // Each thread processes pixels
            for (int local_px_idx = threadIdx.x; local_px_idx < tile_pixels; local_px_idx += blockDim.x) {
                // Compute voxel coordinates
                // OPTIMIZATION 1.4: For 3D with power-of-2 tile sizes, use bitwise ops
                int voxel_coords[DIM];

                if constexpr (DIM == 3) {
                    // Fast path for 3D: requires tile_size=8 AND full tile (not edge)
                    if (use_fast_path_3d) {
                        int local_z = local_px_idx & 7;
                        int local_y = (local_px_idx >> 3) & 7;
                        int local_x = local_px_idx >> 6;
                        voxel_coords[0] = tile_origin[0] + local_x;
                        voxel_coords[1] = tile_origin[1] + local_y;
                        voxel_coords[2] = tile_origin[2] + local_z;
                    } else {
                        int remaining = local_px_idx;
                        #pragma unroll
                        for (int d = DIM - 1; d >= 0; d--) {
                            voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
                            remaining /= tile_extent[d];
                        }
                    }
                } else if constexpr (DIM == 2) {
                    // Fast path for 2D: requires tile_size=16 AND full tile (not edge)
                    if (use_fast_path_2d) {
                        int local_y = local_px_idx & 15;
                        int local_x = local_px_idx >> 4;
                        voxel_coords[0] = tile_origin[0] + local_x;
                        voxel_coords[1] = tile_origin[1] + local_y;
                    } else {
                        int remaining = local_px_idx;
                        #pragma unroll
                        for (int d = DIM - 1; d >= 0; d--) {
                            voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
                            remaining /= tile_extent[d];
                        }
                    }
                } else {
                    int remaining = local_px_idx;
                    #pragma unroll
                    for (int d = DIM - 1; d >= 0; d--) {
                        voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
                        remaining /= tile_extent[d];
                    }
                }

                // Use integer coordinates to match PyTorch reference
                float px[DIM];
                #pragma unroll
                for (int d = 0; d < DIM; d++) {
                    px[d] = (float)voxel_coords[d];
                }

                // Get upstream gradient
                int64_t global_px_idx = voxel_to_linear<DIM>(voxel_coords, shape);
                float dL_dI = grad_output[global_px_idx];

                if (dL_dI == 0.0f) continue;

                // Compute displacement (using padded stride)
                float d_vec[DIM];
                #pragma unroll
                for (int d = 0; d < DIM; d++) {
                    d_vec[d] = px[d] - s_centers[si * CENTER_STRIDE + d];
                }

                // Compute Mahalanobis distance
                float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, &s_conic[si * CONIC_SIZE]);

                // OPTIMIZATION 1.2: Early rejection based on precomputed truncation
                if (dist_sq > truncate_sq) continue;

                // Compute intensity (only for pixels within truncation radius)
                float intensity = gaussian_intensity(dist_sq, amp, s);

                if (intensity < intensity_floor) continue;

                // Use optimized gradient computation (template specialized for 2D/3D)
                // This provides 25-35% speedup for 2D/3D by using explicit formulas
                // instead of loop-based computation
                compute_pixel_gradients<DIM>(
                    dL_dI, intensity, dist_sq, amp, s, d_vec,
                    &s_conic[si * CONIC_SIZE],
                    local_d_centers, local_d_conic,
                    local_d_amp, local_d_sharpness
                );
            }

            // Warp-level reduction and atomic add for this splat
            int lane = threadIdx.x % 32;

            // Reduce and write d_amp
            float warp_d_amp = warp_reduce_sum(local_d_amp);
            if (lane == 0) {
                atomicAdd(&d_amps[splat_idx], warp_d_amp);
            }

            // Reduce and write d_sharpness
            float warp_d_sharpness = warp_reduce_sum(local_d_sharpness);
            if (lane == 0) {
                atomicAdd(&d_sharpness[splat_idx], warp_d_sharpness);
            }

            // Reduce and write d_centers
            #pragma unroll
            for (int d = 0; d < DIM; d++) {
                float warp_d_center = warp_reduce_sum(local_d_centers[d]);
                if (lane == 0) {
                    atomicAdd(&d_centers[splat_idx * DIM + d], warp_d_center);
                }
            }

            // Reduce and write d_conic
            #pragma unroll
            for (int c = 0; c < CONIC_SIZE; c++) {
                float warp_d_conic = warp_reduce_sum(local_d_conic[c]);
                if (lane == 0) {
                    atomicAdd(&d_conic[splat_idx * CONIC_SIZE + c], warp_d_conic);
                }
            }
        }
    }
}

// =============================================================================
// KERNEL LAUNCH WRAPPERS
// =============================================================================

template <int DIM, typename InputDType = float>
void launch_preprocess(
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    const InputDType* sharpness,
    int N,
    const int* shape,
    const int* tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    int* tile_counts,
    bool* global_flags,
    int64_t num_tiles,
    cudaStream_t stream
) {
    int block_size = PREPROCESS_BLOCK_SIZE;
    int num_blocks = (N + block_size - 1) / block_size;

    preprocess_kernel<DIM, InputDType><<<num_blocks, block_size, 0, stream>>>(
        centers, conic, amps, sharpness, N,
        shape, tile_dims, tile_size, truncate, intensity_floor,
        tile_counts, global_flags, num_tiles
    );
}

template <int DIM, typename InputDType = float>
void launch_bin(
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    const InputDType* sharpness,
    int N,
    const int* shape,
    const int* tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* tile_offsets,
    int* tile_write_heads,
    int* tile_content,
    int64_t num_tiles,
    cudaStream_t stream
) {
    int block_size = BIN_BLOCK_SIZE;
    int num_blocks = (N + block_size - 1) / block_size;

    bin_kernel<DIM, InputDType><<<num_blocks, block_size, 0, stream>>>(
        centers, conic, amps, sharpness, N,
        shape, tile_dims, tile_size, truncate, intensity_floor,
        tile_offsets, tile_write_heads, tile_content, num_tiles
    );
}

template <int DIM, typename InputDType = float>
void launch_rasterize_forward(
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    const InputDType* sharpness,
    int N,
    const int* shape,
    const int* tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* tile_offsets,
    const int* tile_counts,
    const int* tile_content,
    float* output,
    int64_t num_tiles,
    const std::vector<int>& host_tile_dims,
    cudaStream_t stream
) {
    // Block size depends on dimension
    int block_size;
    if constexpr (DIM == 2) {
        block_size = RASTER_BLOCK_SIZE_2D;
    } else if constexpr (DIM == 3) {
        block_size = RASTER_BLOCK_SIZE_3D;
    } else {
        block_size = RASTER_BLOCK_SIZE_DEFAULT;
    }

    // Compute shared memory size
    // s_centers: SPLAT_BATCH_SIZE * CENTER_STRIDE (padded for 3D)
    // s_conic: SPLAT_BATCH_SIZE * CONIC_SIZE
    // s_amps: SPLAT_BATCH_SIZE
    // s_sharpness: SPLAT_BATCH_SIZE
    // s_truncate_sq: SPLAT_BATCH_SIZE
    constexpr int CONIC_SIZE = conic_size<DIM>();
    constexpr int CENTER_STRIDE = (DIM == 3) ? 4 : DIM;
    size_t smem_size = SPLAT_BATCH_SIZE * (CENTER_STRIDE + CONIC_SIZE + 3) * sizeof(float);

    // OPTIMIZATION: Use 3D grid for 2D/3D volumes
    // This improves L2 cache locality and eliminates division/modulo in tile coordinate extraction
    // Grid order: last dimension varies fastest to match linear index formula
    if constexpr (DIM == 3) {
        // Grid: (z, y, x) so blockIdx.x=z, blockIdx.y=y, blockIdx.z=x
        dim3 grid(host_tile_dims[2], host_tile_dims[1], host_tile_dims[0]);
        rasterize_forward_kernel<DIM, InputDType><<<grid, block_size, smem_size, stream>>>(
            centers, conic, amps, sharpness, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content, output, 0
        );
    } else if constexpr (DIM == 2) {
        // Grid: (y, x, 1) so blockIdx.x=y, blockIdx.y=x
        dim3 grid(host_tile_dims[1], host_tile_dims[0], 1);
        rasterize_forward_kernel<DIM, InputDType><<<grid, block_size, smem_size, stream>>>(
            centers, conic, amps, sharpness, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content, output, 0
        );
    } else {
        // For DIM > 3, use 1D grid
        int num_blocks = (int)num_tiles;
        rasterize_forward_kernel<DIM, InputDType><<<num_blocks, block_size, smem_size, stream>>>(
            centers, conic, amps, sharpness, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content, output, 0
        );
    }
}

template <int DIM, typename InputDType = float>
void launch_rasterize_backward(
    const float* grad_output,
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    const InputDType* sharpness,
    int N,
    const int* shape,
    const int* tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* tile_offsets,
    const int* tile_counts,
    const int* tile_content,
    float* d_centers,
    float* d_conic,
    float* d_amps,
    float* d_sharpness,
    int64_t num_tiles,
    const std::vector<int>& host_tile_dims,
    cudaStream_t stream
) {
    int block_size;
    if constexpr (DIM == 2) {
        block_size = RASTER_BLOCK_SIZE_2D;
    } else if constexpr (DIM == 3) {
        block_size = RASTER_BLOCK_SIZE_3D;
    } else {
        block_size = RASTER_BLOCK_SIZE_DEFAULT;
    }

    // Compute shared memory size
    // s_centers: SPLAT_BATCH_SIZE * CENTER_STRIDE (padded for 3D)
    // s_conic: SPLAT_BATCH_SIZE * CONIC_SIZE
    // s_amps: SPLAT_BATCH_SIZE
    // s_sharpness: SPLAT_BATCH_SIZE
    // s_truncate_sq: SPLAT_BATCH_SIZE
    // s_splat_ids: SPLAT_BATCH_SIZE (int, not float)
    // Note: Warp reduction uses __shfl_down_sync intrinsics, no shared memory needed
    constexpr int CONIC_SIZE = conic_size<DIM>();
    constexpr int CENTER_STRIDE = (DIM == 3) ? 4 : DIM;
    size_t smem_size = SPLAT_BATCH_SIZE * (CENTER_STRIDE + CONIC_SIZE + 3) * sizeof(float)
                       + SPLAT_BATCH_SIZE * sizeof(int);  // splat_ids

    // OPTIMIZATION: Use 3D grid for 2D/3D volumes
    // Grid order: last dimension varies fastest to match linear index formula
    if constexpr (DIM == 3) {
        // Grid: (z, y, x) so blockIdx.x=z, blockIdx.y=y, blockIdx.z=x
        dim3 grid(host_tile_dims[2], host_tile_dims[1], host_tile_dims[0]);
        rasterize_backward_kernel<DIM, InputDType><<<grid, block_size, smem_size, stream>>>(
            grad_output, centers, conic, amps, sharpness, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content,
            d_centers, d_conic, d_amps, d_sharpness, 0
        );
    } else if constexpr (DIM == 2) {
        // Grid: (y, x, 1) so blockIdx.x=y, blockIdx.y=x
        dim3 grid(host_tile_dims[1], host_tile_dims[0], 1);
        rasterize_backward_kernel<DIM, InputDType><<<grid, block_size, smem_size, stream>>>(
            grad_output, centers, conic, amps, sharpness, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content,
            d_centers, d_conic, d_amps, d_sharpness, 0
        );
    } else {
        // For DIM > 3, use 1D grid
        int num_blocks = (int)num_tiles;
        rasterize_backward_kernel<DIM, InputDType><<<num_blocks, block_size, smem_size, stream>>>(
            grad_output, centers, conic, amps, sharpness, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content,
            d_centers, d_conic, d_amps, d_sharpness, 0
        );
    }
}

// =============================================================================
// GLOBAL SPLAT KERNELS
// =============================================================================
// These kernels handle "global" splats that touch too many tiles (>10% of total).
// Instead of tile-based binning, we process all pixels for each global splat.
// This is less efficient but correct, and global splats are expected to be rare.

/**
 * Forward kernel for global splats.
 *
 * Each thread processes one pixel. For each pixel, we iterate over all global
 * splats and accumulate their intensity contributions.
 *
 * Template parameter DIM: dimensionality (2-8)
 */
template <int DIM, typename InputDType = float>
__global__ void rasterize_global_forward_kernel(
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
    const InputDType* __restrict__ sharpness,
    const int* __restrict__ global_splat_ids,  // Array of global splat indices
    int n_global_splats,                        // Number of global splats
    const int* __restrict__ shape,
    float truncate,
    float intensity_floor,  // Minimum intensity threshold (must match tile-based kernel)
    float* __restrict__ output,  // Output to add to (already has tile-based contributions)
    int64_t num_pixels
) {
    int64_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= num_pixels) return;

    constexpr int CONIC_SIZE = conic_size<DIM>();

    // Convert linear index to voxel coordinates
    int voxel_coords[DIM];
    {
        int64_t remaining = pixel_idx;
        #pragma unroll
        for (int d = DIM - 1; d >= 0; d--) {
            voxel_coords[d] = (int)(remaining % shape[d]);
            remaining /= shape[d];
        }
    }

    float px[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        px[d] = (float)voxel_coords[d];
    }

    float intensity_sum = 0.0f;

    // Process all global splats
    for (int i = 0; i < n_global_splats; i++) {
        int splat_idx = global_splat_ids[i];

        // Load splat data with dtype-aware conversion
        float mu[DIM];
        float c[CONIC_SIZE];

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            mu[d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
        }
        #pragma unroll
        for (int ci = 0; ci < CONIC_SIZE; ci++) {
            c[ci] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + ci);
        }
        float amp = DTypeTraits<InputDType>::load(amps, splat_idx);
        float s = DTypeTraits<InputDType>::load(sharpness, splat_idx);

        // Compute displacement
        float d_vec[DIM];
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            d_vec[d] = px[d] - mu[d];
        }

        // Compute Mahalanobis distance squared
        float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, c);

        // Early culling based on effective truncation
        float eff_trunc_sq = effective_truncate_sq(truncate, s);
        if (dist_sq > eff_trunc_sq) continue;

        // Compute intensity
        float intensity = gaussian_intensity(dist_sq, amp, s);

        // Skip if below threshold (must match tile-based kernel behavior)
        if (intensity >= intensity_floor) {
            intensity_sum += intensity;
        }
    }

    // Add to output using atomicAdd (tile-based kernel may have already written)
    if (intensity_sum > 0.0f) {
        atomicAdd(&output[pixel_idx], intensity_sum);
    }
}

/**
 * Backward kernel for global splats.
 *
 * Each thread processes one pixel. Gradients are accumulated using atomicAdd.
 */
template <int DIM, typename InputDType = float>
__global__ void rasterize_global_backward_kernel(
    const float* __restrict__ grad_output,
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
    const InputDType* __restrict__ sharpness,
    const int* __restrict__ global_splat_ids,
    int n_global_splats,
    const int* __restrict__ shape,
    float truncate,
    float intensity_floor,  // Minimum intensity threshold (must match tile-based kernel)
    float* __restrict__ d_centers,
    float* __restrict__ d_conic,
    float* __restrict__ d_amps,
    float* __restrict__ d_sharpness,
    int64_t num_pixels
) {
    int64_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= num_pixels) return;

    constexpr int CONIC_SIZE = conic_size<DIM>();

    float grad_out = grad_output[pixel_idx];
    if (fabsf(grad_out) < 1e-10f) return;  // Skip if no gradient

    // Convert linear index to voxel coordinates
    int voxel_coords[DIM];
    {
        int64_t remaining = pixel_idx;
        #pragma unroll
        for (int d = DIM - 1; d >= 0; d--) {
            voxel_coords[d] = (int)(remaining % shape[d]);
            remaining /= shape[d];
        }
    }

    float px[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        px[d] = (float)voxel_coords[d];
    }

    // Process all global splats
    for (int i = 0; i < n_global_splats; i++) {
        int splat_idx = global_splat_ids[i];

        // Load splat data with dtype-aware conversion
        float mu[DIM];
        float c[CONIC_SIZE];

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            mu[d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
        }
        #pragma unroll
        for (int ci = 0; ci < CONIC_SIZE; ci++) {
            c[ci] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + ci);
        }
        float amp = DTypeTraits<InputDType>::load(amps, splat_idx);
        float s = DTypeTraits<InputDType>::load(sharpness, splat_idx);

        // Compute displacement
        float d_vec[DIM];
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            d_vec[d] = px[d] - mu[d];
        }

        // Compute Mahalanobis distance squared
        float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, c);

        // Early culling
        float eff_trunc_sq = effective_truncate_sq(truncate, s);
        if (dist_sq > eff_trunc_sq) continue;

        // Compute intensity and gradients
        float intensity = gaussian_intensity(dist_sq, amp, s);
        if (intensity < intensity_floor) continue;  // Must match tile-based kernel behavior

        // d_amp = grad_out * (I / a)
        float local_d_amp = grad_out * (intensity / fmaxf(amp, 1e-10f));

        // Gradient w.r.t. dist_sq: ∂I/∂D² = I × (-0.25 × s) × (D²)^(s/2 - 1)
        // For s=2: ∂I/∂D² = -0.5 × I
        float dist_sq_safe = fmaxf(dist_sq, 1e-12f);
        float dI_dD_sq;
        if (fabsf(s - 2.0f) < 1e-4f) {
            // Standard Gaussian: dI/dD² = -0.5 * intensity
            dI_dD_sq = -0.5f * intensity;
        } else {
            // Generalized Gaussian: dI/dD² = I × (-0.25 × s) × (D²)^(s/2 - 1)
            float dist_pow_s_minus_1 = __powf(dist_sq_safe, s * 0.5f - 1.0f);
            dI_dD_sq = intensity * (-0.25f * s) * dist_pow_s_minus_1;
        }

        float outer_grad = grad_out * dI_dD_sq;

        // d_centers: ∂D²/∂μ = -∂D²/∂d = -2 × Σ⁻¹ @ d
        // Note: No factor of 2 on off-diagonal conic elements when computing Σ⁻¹ @ d
        float local_d_centers[DIM];
        #pragma unroll
        for (int di = 0; di < DIM; di++) {
            float sum = 0.0f;
            // Sum over conic contributions (symmetric matrix in packed upper triangle)
            for (int dj = 0; dj < DIM; dj++) {
                int ci = (di <= dj) ?
                    (di * (2 * DIM - di - 1) / 2 + dj - di) :
                    (dj * (2 * DIM - dj - 1) / 2 + di - dj);
                sum += c[ci] * d_vec[dj];
            }
            // ∂L/∂center = grad_out × dI_dD_sq × (-2) × (Σ⁻¹ @ d)
            local_d_centers[di] = outer_grad * (-2.0f) * sum;
        }

        // d_conic: ∂D²/∂c_ij = d_i × d_j (diagonal) or 2 × d_i × d_j (off-diagonal)
        float local_d_conic[CONIC_SIZE];
        int ci = 0;
        #pragma unroll
        for (int di = 0; di < DIM; di++) {
            for (int dj = di; dj < DIM; dj++) {
                float factor = (di == dj) ? 1.0f : 2.0f;
                local_d_conic[ci] = outer_grad * factor * d_vec[di] * d_vec[dj];
                ci++;
            }
        }

        // d_sharpness: ∂I/∂s = I × (-0.25) × (D²)^(s/2) × ln(D²)
        float local_d_sharpness = 0.0f;
        if (dist_sq > 1e-6f) {
            float dist_pow_s = __powf(dist_sq_safe, s * 0.5f);
            float log_dist_sq = __logf(dist_sq_safe);
            local_d_sharpness = grad_out * intensity * (-0.25f) * dist_pow_s * log_dist_sq;
        }

        // Accumulate gradients using atomicAdd
        atomicAdd(&d_amps[splat_idx], local_d_amp);
        atomicAdd(&d_sharpness[splat_idx], local_d_sharpness);

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            atomicAdd(&d_centers[splat_idx * DIM + d], local_d_centers[d]);
        }
        #pragma unroll
        for (int ci_idx = 0; ci_idx < CONIC_SIZE; ci_idx++) {
            atomicAdd(&d_conic[splat_idx * CONIC_SIZE + ci_idx], local_d_conic[ci_idx]);
        }
    }
}

// Launch wrapper for global splat forward
template <int DIM, typename InputDType = float>
void launch_rasterize_global_forward(
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    const InputDType* sharpness,
    const int* global_splat_ids,
    int n_global_splats,
    const int* shape,
    float truncate,
    float intensity_floor,
    float* output,
    int64_t num_pixels,
    cudaStream_t stream
) {
    if (n_global_splats == 0) return;

    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (int)((num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);

    rasterize_global_forward_kernel<DIM, InputDType><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        centers, conic, amps, sharpness,
        global_splat_ids, n_global_splats,
        shape, truncate, intensity_floor, output, num_pixels
    );
}

// Launch wrapper for global splat backward
template <int DIM, typename InputDType = float>
void launch_rasterize_global_backward(
    const float* grad_output,
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    const InputDType* sharpness,
    const int* global_splat_ids,
    int n_global_splats,
    const int* shape,
    float truncate,
    float intensity_floor,
    float* d_centers,
    float* d_conic,
    float* d_amps,
    float* d_sharpness,
    int64_t num_pixels,
    cudaStream_t stream
) {
    if (n_global_splats == 0) return;

    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (int)((num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);

    rasterize_global_backward_kernel<DIM, InputDType><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        grad_output, centers, conic, amps, sharpness,
        global_splat_ids, n_global_splats,
        shape, truncate, intensity_floor,
        d_centers, d_conic, d_amps, d_sharpness, num_pixels
    );
}

// =============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS (FP32)
// =============================================================================

// 2D - fully optimized with 3D grid launch
template void launch_preprocess<2, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<2, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<2, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<2, float>(const float*, const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);

// 3D - fully optimized with 3D grid launch
template void launch_preprocess<3, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<3, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<3, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<3, float>(const float*, const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);

// 4D - uses 1D grid
template void launch_preprocess<4, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<4, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<4, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<4, float>(const float*, const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);

// 5D - uses 1D grid
template void launch_preprocess<5, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<5, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<5, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<5, float>(const float*, const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);

// 6D - uses 1D grid
template void launch_preprocess<6, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<6, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<6, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<6, float>(const float*, const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);

// 7D - uses 1D grid
template void launch_preprocess<7, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<7, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<7, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<7, float>(const float*, const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);

// 8D - uses 1D grid
template void launch_preprocess<8, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<8, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<8, float>(const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<8, float>(const float*, const float*, const float*, const float*, const float*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);

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

// 2D FP16
template void launch_preprocess<2, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<2, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<2, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<2, __half>(const float*, const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_global_forward<2, __half>(const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<2, __half>(const float*, const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);

// 3D FP16
template void launch_preprocess<3, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<3, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<3, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<3, __half>(const float*, const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_global_forward<3, __half>(const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<3, __half>(const float*, const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);

// 4D FP16
template void launch_preprocess<4, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<4, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<4, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<4, __half>(const float*, const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_global_forward<4, __half>(const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<4, __half>(const float*, const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);

// 5D FP16
template void launch_preprocess<5, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<5, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<5, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<5, __half>(const float*, const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_global_forward<5, __half>(const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<5, __half>(const float*, const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);

// 6D FP16
template void launch_preprocess<6, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<6, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<6, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<6, __half>(const float*, const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_global_forward<6, __half>(const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<6, __half>(const float*, const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);

// 7D FP16
template void launch_preprocess<7, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<7, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<7, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<7, __half>(const float*, const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_global_forward<7, __half>(const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<7, __half>(const float*, const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);

// 8D FP16
template void launch_preprocess<8, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, int*, bool*, int64_t, cudaStream_t);
template void launch_bin<8, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, int*, int*, int64_t, cudaStream_t);
template void launch_rasterize_forward<8, __half>(const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_backward<8, __half>(const float*, const __half*, const __half*, const __half*, const __half*,
    int, const int*, const int*, int, float, float, const int64_t*, const int*, const int*, float*, float*, float*, float*, int64_t, const std::vector<int>&, cudaStream_t);
template void launch_rasterize_global_forward<8, __half>(const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, int64_t, cudaStream_t);
template void launch_rasterize_global_backward<8, __half>(const float*, const __half*, const __half*, const __half*, const __half*, const int*, int, const int*, float, float, float*, float*, float*, float*, int64_t, cudaStream_t);

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
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
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
            conic.data_ptr<float>(), \
            amps.data_ptr<float>(), \
            sharpness.data_ptr<float>(), \
            N, \
            shape_tensor.data_ptr<int>(), \
            tile_dims_tensor.data_ptr<int>(), \
            tile_size, truncate, intensity_floor, \
            state.tile_counts.data_ptr<int>(), \
            state.global_splat_flags.data_ptr<bool>(), \
            num_tiles, stream)

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
        launch_bin<D, float>( \
            centers.data_ptr<float>(), \
            conic.data_ptr<float>(), \
            amps.data_ptr<float>(), \
            sharpness.data_ptr<float>(), \
            N, \
            shape_tensor.data_ptr<int>(), \
            tile_dims_tensor.data_ptr<int>(), \
            tile_size, truncate, intensity_floor, \
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
    #define LAUNCH_RASTER(D) \
        launch_rasterize_forward<D, float>( \
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

    switch (dim) {
        case 2: LAUNCH_RASTER(2); break;
        case 3: LAUNCH_RASTER(3); break;
        case 4: LAUNCH_RASTER(4); break;
        case 5: LAUNCH_RASTER(5); break;
        case 6: LAUNCH_RASTER(6); break;
        case 7: LAUNCH_RASTER(7); break;
        case 8: LAUNCH_RASTER(8); break;
        default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
    }
    #undef LAUNCH_RASTER

    CUDA_CHECK_LAST();

    // ==========================================================================
    // GLOBAL SPLAT HANDLING
    // ==========================================================================
    // Extract global splat IDs and process them with dedicated kernel.
    // Global splats are those that touch too many tiles (>10% AND >1024 tiles).

    // Extract global splat indices using torch::nonzero
    auto global_indices = torch::nonzero(state.global_splat_flags);
    state.num_global_splats = (int)global_indices.size(0);

    if (state.num_global_splats > 0) {
        // Flatten to 1D tensor of int32 indices
        state.global_splat_ids = global_indices.squeeze(1).to(torch::kInt32).contiguous();

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
    #define LAUNCH_BACKWARD(D) \
        launch_rasterize_backward<D, float>( \
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

    switch (dim) {
        case 2: LAUNCH_BACKWARD(2); break;
        case 3: LAUNCH_BACKWARD(3); break;
        case 4: LAUNCH_BACKWARD(4); break;
        case 5: LAUNCH_BACKWARD(5); break;
        case 6: LAUNCH_BACKWARD(6); break;
        case 7: LAUNCH_BACKWARD(7); break;
        case 8: LAUNCH_BACKWARD(8); break;
        default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
    }
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
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size
) {
    validate_inputs(centers, conic, amps, sharpness, shape);

    int dim = (int)shape.size();
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
    dispatch_forward(dim, centers, conic, amps, sharpness, shape,
                    truncate, intensity_floor, tile_size, output, state);

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
    int tile_size
) {
    validate_inputs(centers, conic, amps, sharpness, shape);

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
                     shape, truncate, intensity_floor, tile_size,
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
 * FP16→FP32 at the API boundary (before kernel launch).
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
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
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

    // Launch preprocess kernel with FP16 inputs
    #define LAUNCH_PREPROCESS_FP16(D) \
        launch_preprocess<D, __half>( \
            centers_ptr, conic_ptr, amps_ptr, sharpness_ptr, \
            N, \
            shape_tensor.data_ptr<int>(), \
            tile_dims_tensor.data_ptr<int>(), \
            tile_size, truncate, intensity_floor, \
            state.tile_counts.data_ptr<int>(), \
            state.global_splat_flags.data_ptr<bool>(), \
            num_tiles, stream)

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
        launch_bin<D, __half>( \
            centers_ptr, conic_ptr, amps_ptr, sharpness_ptr, \
            N, \
            shape_tensor.data_ptr<int>(), \
            tile_dims_tensor.data_ptr<int>(), \
            tile_size, truncate, intensity_floor, \
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
    #define LAUNCH_RASTER_FP16(D) \
        launch_rasterize_forward<D, __half>( \
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

    switch (dim) {
        case 2: LAUNCH_RASTER_FP16(2); break;
        case 3: LAUNCH_RASTER_FP16(3); break;
        case 4: LAUNCH_RASTER_FP16(4); break;
        case 5: LAUNCH_RASTER_FP16(5); break;
        case 6: LAUNCH_RASTER_FP16(6); break;
        case 7: LAUNCH_RASTER_FP16(7); break;
        case 8: LAUNCH_RASTER_FP16(8); break;
        default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
    }
    #undef LAUNCH_RASTER_FP16

    CUDA_CHECK_LAST();

    // Handle global splats
    auto global_indices = torch::nonzero(state.global_splat_flags);
    state.num_global_splats = (int)global_indices.size(0);

    if (state.num_global_splats > 0) {
        state.global_splat_ids = global_indices.squeeze(1).to(torch::kInt32).contiguous();

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
    #define LAUNCH_BACKWARD_FP16(D) \
        launch_rasterize_backward<D, __half>( \
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

    switch (dim) {
        case 2: LAUNCH_BACKWARD_FP16(2); break;
        case 3: LAUNCH_BACKWARD_FP16(3); break;
        case 4: LAUNCH_BACKWARD_FP16(4); break;
        case 5: LAUNCH_BACKWARD_FP16(5); break;
        case 6: LAUNCH_BACKWARD_FP16(6); break;
        case 7: LAUNCH_BACKWARD_FP16(7); break;
        case 8: LAUNCH_BACKWARD_FP16(8); break;
        default: TORCH_CHECK(false, "Unsupported dimension: ", dim);
    }
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
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size
) {
    validate_inputs_fp16(centers, conic, amps, sharpness, shape);

    int dim = (int)shape.size();
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
    dispatch_forward_fp16(dim, centers, conic, amps, sharpness, shape,
                         truncate, intensity_floor, tile_size, output, state);

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
    int tile_size
) {
    validate_inputs_fp16(centers, conic, amps, sharpness, shape);

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
                          shape, truncate, intensity_floor, tile_size,
                          d_centers, d_conic, d_amps, d_sharpness);

    return std::make_tuple(d_centers, d_conic, d_amps, d_sharpness);
}
