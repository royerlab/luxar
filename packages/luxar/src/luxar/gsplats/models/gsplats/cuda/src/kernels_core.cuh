/**
 * CUDA Core Kernel Implementations
 *
 * This header contains the core kernel implementations for volumetric Gaussian splatting:
 * - Backward gradient dispatch helpers (compute_pixel_gradients)
 * - Preprocess kernel (AABB computation and tile counting)
 * - Binning kernel (splat-to-tile assignment)
 * - Forward rasterization kernel (tile-parallel rendering)
 * - Backward rasterization kernel (gradient computation with warp reduction)
 *
 * These kernels are templated on DIM for compile-time optimization of 2D/3D cases.
 */

#ifndef CUDA_SPLATTING_KERNELS_CORE_CUH
#define CUDA_SPLATTING_KERNELS_CORE_CUH

#include "utils.cuh"

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

    // Pre-compute the common factor (CSE: used DIM + CONIC_SIZE times below)
    float outer = dL_dI * grad_dist;

    // Chain rule: dL/dcenter and dL/dconic via dD^2/dcenter and dD^2/dconic
    constexpr int CONIC_SIZE = conic_size<DIM>();

    // Compute dD^2/dd = 2 * Sigma^-1 @ d
    // Uses tri_index for O(1) packed index lookup (replaces O(DIM) row_start loop)
    float dD2_dd[DIM];
    #pragma unroll
    for (int i = 0; i < DIM; i++) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < DIM; j++) {
            // Access symmetric conic: for i<=j use tri_index(i,j), else tri_index(j,i)
            int ci = (i <= j) ? tri_index<DIM>(i, j) : tri_index<DIM>(j, i);
            sum += conic[ci] * d_vec[j];
        }
        dD2_dd[i] = 2.0f * sum;
    }

    // dL/dcenter = outer * dD^2/dd * (-1)
    #pragma unroll
    for (int i = 0; i < DIM; i++) {
        local_d_centers[i] -= outer * dD2_dd[i];
    }

    // dD^2/dconic - gradient w.r.t. packed upper triangle
    int conic_idx = 0;
    #pragma unroll
    for (int i = 0; i < DIM; i++) {
        // Diagonal
        local_d_conic[conic_idx] += outer * d_vec[i] * d_vec[i];
        conic_idx++;

        // Off-diagonals
        #pragma unroll
        for (int j = i + 1; j < DIM; j++) {
            local_d_conic[conic_idx] += outer * 2.0f * d_vec[i] * d_vec[j];
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
// PREPROCESS KERNEL
// =============================================================================

// Default batch size for splat loading (can be overridden via template parameter)
// Supported batch sizes: 32, 128, 256
// Larger batches = fewer global memory round-trips, better for memory-bound workloads
constexpr int DEFAULT_BATCH_SIZE = 128;

// Threshold for global splat handling (fraction of tiles)
constexpr float GLOBAL_SPLAT_THRESHOLD = 0.1f;

template <int DIM, typename InputDType = float>
__global__ void preprocess_kernel(
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ amps,
    const InputDType* __restrict__ sharpness,
    const InputDType* __restrict__ L_row_norms,
    int N,
    const int* __restrict__ shape,
    const int* __restrict__ tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    int* __restrict__ tile_counts,
    bool* __restrict__ global_flags,
    int* __restrict__ aabb_lo_cache,
    int* __restrict__ aabb_hi_cache,
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

    float amp = DTypeTraits<InputDType>::load(amps, splat_idx);
    float s = DTypeTraits<InputDType>::load(sharpness, splat_idx);

    // Load exact L_row_norms (precomputed in Python from Cholesky factors)
    // L_row_norms[i] = sqrt(sum_j L[i,j]^2) = sqrt(Sigma[i,i])
    float L_row_norms_local[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        L_row_norms_local[d] = DTypeTraits<InputDType>::load(L_row_norms, splat_idx * DIM + d);
    }

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

    // Compute AABB using exact L_row_norms
    AABB<DIM> aabb = compute_splat_aabb<DIM>(
        mu, L_row_norms_local, s, amp, truncate, intensity_floor,
        tile_size_arr, tile_dims_local, shape_local
    );

    // Check if AABB is empty (splat outside volume or culled)
    if (aabb.is_empty()) {
        global_flags[splat_idx] = false;
        // Write sentinel AABB (lo > hi signals empty to bin_kernel)
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            aabb_lo_cache[splat_idx * DIM + d] = 1;
            aabb_hi_cache[splat_idx * DIM + d] = 0;
        }
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

    // Cache AABB for bin_kernel (avoids recomputation)
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        aabb_lo_cache[splat_idx * DIM + d] = aabb.lo[d];
        aabb_hi_cache[splat_idx * DIM + d] = aabb.hi[d];
    }

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

template <int DIM>
__global__ void bin_kernel(
    const int* __restrict__ aabb_lo_cache,
    const int* __restrict__ aabb_hi_cache,
    const bool* __restrict__ global_flags,
    int N,
    const int* __restrict__ tile_dims,
    const int64_t* __restrict__ tile_offsets,
    int* __restrict__ tile_write_heads,
    int* __restrict__ tile_content,
    int64_t num_tiles
) {
    int splat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (splat_idx >= N) return;

    // Skip global splats (already flagged by preprocess_kernel)
    if (global_flags[splat_idx]) return;

    // Read cached AABB (computed by preprocess_kernel)
    AABB<DIM> aabb;
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        aabb.lo[d] = aabb_lo_cache[splat_idx * DIM + d];
        aabb.hi[d] = aabb_hi_cache[splat_idx * DIM + d];
    }

    if (aabb.is_empty()) return;

    int n_tiles = aabb.num_tiles();

    // Load tile_dims to local memory
    int tile_dims_local[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        tile_dims_local[d] = tile_dims[d];
    }

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

template <int DIM, int BATCH_SIZE = DEFAULT_BATCH_SIZE, typename InputDType = float>
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
    float* __restrict__ output
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
    __shared__ float s_centers[BATCH_SIZE * CENTER_STRIDE];
    __shared__ float s_conic[BATCH_SIZE * CONIC_SIZE];
    __shared__ float s_amps[BATCH_SIZE];
    __shared__ float s_sharpness[BATCH_SIZE];
    // OPTIMIZATION 1.2: Precompute effective truncation squared per splat
    __shared__ float s_truncate_sq[BATCH_SIZE];

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
            // Bitwise shifts are hardcoded for 8x8x8 = 512 pixels
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
            // Bitwise shifts are hardcoded for 16x16 = 256 pixels
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
        for (int batch_start = 0; batch_start < n_splats_in_tile; batch_start += BATCH_SIZE) {
            int batch_size = min(BATCH_SIZE, n_splats_in_tile - batch_start);

            // Cooperative load of splat batch into shared memory
            // OPTIMIZATION: Use DTypeTraits for dtype-aware loading (supports FP16->FP32 conversion)
            // This improves memory throughput for scattered reads by ~15-20%
            __syncthreads();
            for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
                int splat_idx = __ldg(&tile_content[tile_offset + batch_start + i]);

                // Load centers with dtype conversion (using padded stride for 3D)
                // OPTIMIZATION: Use vectorized loads for FP16 (reduces memory transactions)
                if constexpr (DIM == 3) {
                    float tmp_centers[3];
                    load_centers_3d<InputDType>(centers, splat_idx, tmp_centers);
                    s_centers[i * CENTER_STRIDE + 0] = tmp_centers[0];
                    s_centers[i * CENTER_STRIDE + 1] = tmp_centers[1];
                    s_centers[i * CENTER_STRIDE + 2] = tmp_centers[2];
                } else if constexpr (DIM == 2) {
                    float tmp_centers[2];
                    load_centers_2d<InputDType>(centers, splat_idx, tmp_centers);
                    s_centers[i * CENTER_STRIDE + 0] = tmp_centers[0];
                    s_centers[i * CENTER_STRIDE + 1] = tmp_centers[1];
                } else {
                    #pragma unroll
                    for (int d = 0; d < DIM; d++) {
                        s_centers[i * CENTER_STRIDE + d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
                    }
                }

                // Load conic with dtype conversion
                // OPTIMIZATION: Use vectorized loads for FP16 (reduces memory transactions)
                if constexpr (DIM == 3) {
                    float tmp_conic[6];
                    load_conic_3d<InputDType>(conic, splat_idx, tmp_conic);
                    #pragma unroll
                    for (int c = 0; c < 6; c++) {
                        s_conic[i * CONIC_SIZE + c] = tmp_conic[c];
                    }
                } else if constexpr (DIM == 2) {
                    float tmp_conic[3];
                    load_conic_2d<InputDType>(conic, splat_idx, tmp_conic);
                    s_conic[i * CONIC_SIZE + 0] = tmp_conic[0];
                    s_conic[i * CONIC_SIZE + 1] = tmp_conic[1];
                    s_conic[i * CONIC_SIZE + 2] = tmp_conic[2];
                } else {
                    #pragma unroll
                    for (int c = 0; c < CONIC_SIZE; c++) {
                        s_conic[i * CONIC_SIZE + c] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + c);
                    }
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

template <int DIM, int BATCH_SIZE = DEFAULT_BATCH_SIZE, typename InputDType = float>
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
    float* __restrict__ d_sharpness
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
    __shared__ float s_centers[BATCH_SIZE * CENTER_STRIDE];
    __shared__ float s_conic[BATCH_SIZE * CONIC_SIZE];
    __shared__ float s_amps[BATCH_SIZE];
    __shared__ float s_sharpness[BATCH_SIZE];
    __shared__ int s_splat_ids[BATCH_SIZE];
    // OPTIMIZATION 1.2: Precompute effective truncation squared per splat
    __shared__ float s_truncate_sq[BATCH_SIZE];

    // OPTIMIZATION 3.2: Per-tile gradient accumulators for shared memory reduction
    // This reduces atomicAdd operations from 176 to 11 per splat per tile (3D case)
    __shared__ float s_d_centers_tile[BATCH_SIZE * CENTER_STRIDE];
    __shared__ float s_d_conic_tile[BATCH_SIZE * CONIC_SIZE];
    __shared__ float s_d_amps_tile[BATCH_SIZE];
    __shared__ float s_d_sharpness_tile[BATCH_SIZE];

    // OPTIMIZATION 3.3: Cache grad_output in shared memory
    // This eliminates redundant global memory loads (each pixel loaded once per splat → once per tile)
    // Dimension-specific sizing to minimize shared memory waste while matching tile sizes:
    // 2D: 16x16=256, 3D: 8x8x8=512, 4D: 4^4=256, 6D: 3^6=729
    constexpr int MAX_TILE_PIXELS = (DIM == 2) ? 256 :
                                    (DIM == 3) ? 512 :
                                    (DIM == 4) ? 256 :
                                    (DIM == 5) ? 243 :
                                    (DIM == 6) ? 729 :
                                    (DIM == 7) ? 128 :
                                    (DIM == 8) ? 256 : 512;
    __shared__ float s_grad_output[MAX_TILE_PIXELS];

    // OPTIMIZATION: Hoist loop-invariant condition checks outside all loops
    // These are uniform across all threads in the block (no divergence)
    const bool use_fast_path_3d = (DIM == 3) && (tile_size == 8) &&
        (tile_extent[0] == 8) && (tile_extent[1] == 8) && (tile_extent[2] == 8);
    const bool use_fast_path_2d = (DIM == 2) && (tile_size == 16) &&
        (tile_extent[0] == 16) && (tile_extent[1] == 16);

    // OPTIMIZATION 3.3: Load grad_output into shared memory ONCE per tile
    // This eliminates ~50x redundant global memory loads (one load per pixel per splat → one per pixel)
    for (int px = threadIdx.x; px < tile_pixels; px += blockDim.x) {
        int voxel_coords[DIM];

        if constexpr (DIM == 3) {
            if (use_fast_path_3d) {
                int local_z = px & 7;
                int local_y = (px >> 3) & 7;
                int local_x = px >> 6;
                voxel_coords[0] = tile_origin[0] + local_x;
                voxel_coords[1] = tile_origin[1] + local_y;
                voxel_coords[2] = tile_origin[2] + local_z;
            } else {
                int remaining = px;
                #pragma unroll
                for (int d = DIM - 1; d >= 0; d--) {
                    voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
                    remaining /= tile_extent[d];
                }
            }
        } else if constexpr (DIM == 2) {
            if (use_fast_path_2d) {
                int local_y = px & 15;
                int local_x = px >> 4;
                voxel_coords[0] = tile_origin[0] + local_x;
                voxel_coords[1] = tile_origin[1] + local_y;
            } else {
                int remaining = px;
                #pragma unroll
                for (int d = DIM - 1; d >= 0; d--) {
                    voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
                    remaining /= tile_extent[d];
                }
            }
        } else {
            int remaining = px;
            #pragma unroll
            for (int d = DIM - 1; d >= 0; d--) {
                voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
                remaining /= tile_extent[d];
            }
        }

        int64_t global_px_idx = voxel_to_linear<DIM>(voxel_coords, shape);
        s_grad_output[px] = grad_output[global_px_idx];
    }
    __syncthreads();

    // =========================================================================
    // PRECOMPUTE pixel coordinates ONCE (reused across all batches and splats)
    // =========================================================================
    // Each thread handles ceil(tile_pixels / blockDim.x) pixels.
    // For common cases (3D: 512px/512threads, 2D: 256px/256threads) = 1 pixel/thread.
    // We precompute coordinates for the first pixel (covers the common case).
    // For multi-pixel threads, additional pixels recompute coords in the inner loop.
    constexpr int MAX_PIXELS_PER_THREAD = 4;  // Covers all DIM cases
    int n_my_pixels = 0;
    float precomp_px[MAX_PIXELS_PER_THREAD][DIM];
    float precomp_grad[MAX_PIXELS_PER_THREAD];
    int precomp_local_idx[MAX_PIXELS_PER_THREAD];

    for (int local_px_idx = threadIdx.x; local_px_idx < tile_pixels && n_my_pixels < MAX_PIXELS_PER_THREAD;
         local_px_idx += blockDim.x) {
        precomp_local_idx[n_my_pixels] = local_px_idx;

        int voxel_coords[DIM];
        if constexpr (DIM == 3) {
            if (use_fast_path_3d) {
                voxel_coords[2] = tile_origin[2] + (local_px_idx & 7);
                voxel_coords[1] = tile_origin[1] + ((local_px_idx >> 3) & 7);
                voxel_coords[0] = tile_origin[0] + (local_px_idx >> 6);
            } else {
                int remaining = local_px_idx;
                #pragma unroll
                for (int d = DIM - 1; d >= 0; d--) {
                    voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
                    remaining /= tile_extent[d];
                }
            }
        } else if constexpr (DIM == 2) {
            if (use_fast_path_2d) {
                voxel_coords[1] = tile_origin[1] + (local_px_idx & 15);
                voxel_coords[0] = tile_origin[0] + (local_px_idx >> 4);
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

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            precomp_px[n_my_pixels][d] = (float)voxel_coords[d];
        }
        precomp_grad[n_my_pixels] = s_grad_output[local_px_idx];
        n_my_pixels++;
    }

    // Process splats in batches
    for (int batch_start = 0; batch_start < n_splats_in_tile; batch_start += BATCH_SIZE) {
        int batch_size = min(BATCH_SIZE, n_splats_in_tile - batch_start);

        // Load splat batch
        // OPTIMIZATION: Use DTypeTraits for dtype-aware loading (supports FP16->FP32 conversion)
        // OPTIMIZATION: Use vectorized loads for FP16 (reduces memory transactions)
        __syncthreads();
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            int splat_idx = __ldg(&tile_content[tile_offset + batch_start + i]);
            s_splat_ids[i] = splat_idx;

            // Load centers with vectorized loads for 2D/3D
            if constexpr (DIM == 3) {
                float tmp_centers[3];
                load_centers_3d<InputDType>(centers, splat_idx, tmp_centers);
                s_centers[i * CENTER_STRIDE + 0] = tmp_centers[0];
                s_centers[i * CENTER_STRIDE + 1] = tmp_centers[1];
                s_centers[i * CENTER_STRIDE + 2] = tmp_centers[2];
            } else if constexpr (DIM == 2) {
                float tmp_centers[2];
                load_centers_2d<InputDType>(centers, splat_idx, tmp_centers);
                s_centers[i * CENTER_STRIDE + 0] = tmp_centers[0];
                s_centers[i * CENTER_STRIDE + 1] = tmp_centers[1];
            } else {
                #pragma unroll
                for (int d = 0; d < DIM; d++) {
                    s_centers[i * CENTER_STRIDE + d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
                }
            }

            // Load conic with vectorized loads for 2D/3D
            if constexpr (DIM == 3) {
                float tmp_conic[6];
                load_conic_3d<InputDType>(conic, splat_idx, tmp_conic);
                #pragma unroll
                for (int c = 0; c < 6; c++) {
                    s_conic[i * CONIC_SIZE + c] = tmp_conic[c];
                }
            } else if constexpr (DIM == 2) {
                float tmp_conic[3];
                load_conic_2d<InputDType>(conic, splat_idx, tmp_conic);
                s_conic[i * CONIC_SIZE + 0] = tmp_conic[0];
                s_conic[i * CONIC_SIZE + 1] = tmp_conic[1];
                s_conic[i * CONIC_SIZE + 2] = tmp_conic[2];
            } else {
                #pragma unroll
                for (int c = 0; c < CONIC_SIZE; c++) {
                    s_conic[i * CONIC_SIZE + c] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + c);
                }
            }

            float s = DTypeTraits<InputDType>::load(sharpness, splat_idx);
            s_amps[i] = DTypeTraits<InputDType>::load(amps, splat_idx);
            s_sharpness[i] = s;

            // OPTIMIZATION 1.2: Precompute effective truncation squared
            s_truncate_sq[i] = effective_truncate_sq(truncate, s);

            // OPTIMIZATION 3.2: Initialize gradient accumulators for this batch
            s_d_amps_tile[i] = 0.0f;
            s_d_sharpness_tile[i] = 0.0f;
            #pragma unroll
            for (int d = 0; d < CENTER_STRIDE; d++) {
                s_d_centers_tile[i * CENTER_STRIDE + d] = 0.0f;
            }
            #pragma unroll
            for (int c = 0; c < CONIC_SIZE; c++) {
                s_d_conic_tile[i * CONIC_SIZE + c] = 0.0f;
            }
        }
        __syncthreads();

        // Process each splat in batch
        for (int si = 0; si < batch_size; si++) {

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

            // Each thread processes its precomputed pixels (no redundant coord recomputation)
            for (int pi = 0; pi < n_my_pixels; pi++) {
                // OPTIMIZATION: Use precomputed pixel coordinates and gradients
                // This eliminates BATCH_SIZE-1 redundant coordinate computations per thread
                float dL_dI = precomp_grad[pi];

                if (dL_dI == 0.0f) continue;

                // Compute displacement (using padded stride)
                float d_vec[DIM];
                #pragma unroll
                for (int d = 0; d < DIM; d++) {
                    d_vec[d] = precomp_px[pi][d] - s_centers[si * CENTER_STRIDE + d];
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

            // OPTIMIZATION 3.2: Warp-level reduction to shared memory (fast atomics)
            // This accumulates per-batch instead of writing directly to global memory
            int lane = threadIdx.x % 32;

            // Reduce and write d_amp to shared memory
            float warp_d_amp = warp_reduce_sum(local_d_amp);
            if (lane == 0) {
                atomicAdd(&s_d_amps_tile[si], warp_d_amp);
            }

            // Reduce and write d_sharpness to shared memory
            float warp_d_sharpness = warp_reduce_sum(local_d_sharpness);
            if (lane == 0) {
                atomicAdd(&s_d_sharpness_tile[si], warp_d_sharpness);
            }

            // Reduce and write d_centers to shared memory
            #pragma unroll
            for (int d = 0; d < DIM; d++) {
                float warp_d_center = warp_reduce_sum(local_d_centers[d]);
                if (lane == 0) {
                    atomicAdd(&s_d_centers_tile[si * CENTER_STRIDE + d], warp_d_center);
                }
            }

            // Reduce and write d_conic to shared memory
            #pragma unroll
            for (int c = 0; c < CONIC_SIZE; c++) {
                float warp_d_conic = warp_reduce_sum(local_d_conic[c]);
                if (lane == 0) {
                    atomicAdd(&s_d_conic_tile[si * CONIC_SIZE + c], warp_d_conic);
                }
            }
        }

        // OPTIMIZATION 3.2: Cooperative write-back to global memory
        // This reduces atomicAdds from 176 to 11 per splat per tile (3D case)
        // 176 = 16 warps x (1 amp + 1 sharpness + 3 centers + 6 conic)
        // 11 = 1 x (1 amp + 1 sharpness + 3 centers + 6 conic)
        __syncthreads();
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            int splat_idx = s_splat_ids[i];
            atomicAdd(&d_amps[splat_idx], s_d_amps_tile[i]);
            atomicAdd(&d_sharpness[splat_idx], s_d_sharpness_tile[i]);
            #pragma unroll
            for (int d = 0; d < DIM; d++) {
                atomicAdd(&d_centers[splat_idx * DIM + d], s_d_centers_tile[i * CENTER_STRIDE + d]);
            }
            #pragma unroll
            for (int c = 0; c < CONIC_SIZE; c++) {
                atomicAdd(&d_conic[splat_idx * CONIC_SIZE + c], s_d_conic_tile[i * CONIC_SIZE + c]);
            }
        }
    }
}

#endif // CUDA_SPLATTING_KERNELS_CORE_CUH
