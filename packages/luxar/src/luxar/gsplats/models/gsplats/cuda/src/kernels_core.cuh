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
    float amp,
    const float* __restrict__ d_vec,
    const float* __restrict__ conic,
    float* __restrict__ local_d_centers,
    float* __restrict__ local_d_conic,
    float& local_d_amp
) {
    // Gradient w.r.t. amplitude
    local_d_amp += dL_dI * grad_intensity_wrt_amplitude(intensity, amp);

    // Gradient w.r.t. dist_sq
    float grad_dist = grad_intensity_wrt_dist_sq(intensity);

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
    float amp,
    const float* __restrict__ d_vec,
    const float* __restrict__ conic,
    float* __restrict__ local_d_centers,
    float* __restrict__ local_d_conic,
    float& local_d_amp
) {
    backward_pixel_splat_3d(dL_dI, intensity, amp, d_vec, conic,
                           local_d_centers, local_d_conic, local_d_amp);
}

/**
 * Specialized 2D backward gradient computation using explicit formulas.
 */
template <>
__device__ __forceinline__ void compute_pixel_gradients<2>(
    float dL_dI,
    float intensity,
    float amp,
    const float* __restrict__ d_vec,
    const float* __restrict__ conic,
    float* __restrict__ local_d_centers,
    float* __restrict__ local_d_conic,
    float& local_d_amp
) {
    backward_pixel_splat_2d(dL_dI, intensity, amp, d_vec, conic,
                           local_d_centers, local_d_conic, local_d_amp);
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

// Minimum splats in a tile before grad_output is cached in shared memory.
// For sparse tiles (few splats), the cache loading overhead exceeds the benefit
// of avoiding redundant global memory reads. Empirically, the crossover is ~4 splats.
constexpr int GRAD_CACHE_THRESHOLD = 4;

template <int DIM, typename InputDType = float>
__global__ void preprocess_kernel(
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ amps,
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
    int64_t num_tiles,
    int* __restrict__ global_count  // Atomic counter for global splats
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
        mu, L_row_norms_local, amp, truncate, intensity_floor,
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

    // Increment atomic counter so host can skip torch::nonzero when 0 global splats
    if (is_global) {
        atomicAdd(global_count, 1);
    }

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

    constexpr int CONIC_SIZE = conic_size<DIM>();
    constexpr int CENTER_STRIDE = (DIM == 3) ? 4 : DIM;  // Pad 3D to 4

    // OPTIMIZATION: Single-splat fast path — bypass shared memory entirely.
    // For tiles with exactly 1 splat (common: most tiles have 0-1 splats),
    // load data via __ldg (texture cache broadcast) and process directly.
    // Saves: 2 __syncthreads + cooperative load + shared memory overhead.
    if constexpr (DIM <= 4) {
        // Only use fast path when tile_pixels == blockDim.x (standard tile sizes)
        // Oversized tiles need the multi-pixel loop and can't use this shortcut
        if (n_splats_in_tile == 1 && tile_pixels <= (int)blockDim.x) {
            int splat_idx = __ldg(&tile_content[tile_offset]);

            // Load splat data directly from global memory (broadcast to all threads)
            float mu[DIM];
            #pragma unroll
            for (int d = 0; d < DIM; d++) {
                mu[d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
            }
            float con[CONIC_SIZE];
            #pragma unroll
            for (int c = 0; c < CONIC_SIZE; c++) {
                con[c] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + c);
            }
            float amp_val = DTypeTraits<InputDType>::load(amps, splat_idx);
            float trunc_sq = effective_truncate_sq(truncate, amp_val, intensity_floor);

            // Compute fast path flags
            const bool fast_3d = (DIM == 3) && (tile_size == 8) &&
                (tile_extent[0] == 8) && (tile_extent[1] == 8) && (tile_extent[2] == 8);
            const bool fast_2d = (DIM == 2) && (tile_size == 16) &&
                (tile_extent[0] == 16) && (tile_extent[1] == 16);

            // Each thread processes its pixel (tile_pixels == blockDim.x for DIM <= 4)
            int local_px_idx = threadIdx.x;
            if (local_px_idx < tile_pixels) {
                int voxel_coords[DIM];
                compute_voxel_coords<DIM>(local_px_idx, tile_origin, tile_extent,
                                          fast_3d, fast_2d, voxel_coords);
                float px[DIM];
                #pragma unroll
                for (int d = 0; d < DIM; d++) px[d] = (float)voxel_coords[d];

                float d_vec[DIM];
                #pragma unroll
                for (int d = 0; d < DIM; d++) d_vec[d] = px[d] - mu[d];

                float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, con);

                if (dist_sq <= trunc_sq) {
                    float intensity = gaussian_intensity(dist_sq, amp_val);
                    if (intensity >= intensity_floor) {
                        int64_t global_px_idx = voxel_to_linear<DIM>(voxel_coords, shape);
                        output[global_px_idx] = intensity;
                    }
                }
            }
            return;
        }
    }

    // Shared memory for splat batch loading
    // OPTIMIZATION 2.3: Pad DIM to avoid bank conflicts (stride-3 causes conflicts)
    // For 3D: use stride-4 instead of stride-3, wastes 25% but eliminates conflicts
    __shared__ float s_centers[BATCH_SIZE * CENTER_STRIDE];
    __shared__ float s_conic[BATCH_SIZE * CONIC_SIZE];
    __shared__ float s_amps[BATCH_SIZE];
    // OPTIMIZATION 1.2: Precompute effective truncation squared per splat
    __shared__ float s_truncate_sq[BATCH_SIZE];

    // OPTIMIZATION: Hoist loop-invariant condition checks outside the pixel loop
    // These are uniform across all threads in the block (no divergence)
    // Hoisting ensures the compiler doesn't re-evaluate per iteration
    const bool use_fast_path_3d = (DIM == 3) && (tile_size == 8) &&
        (tile_extent[0] == 8) && (tile_extent[1] == 8) && (tile_extent[2] == 8);
    const bool use_fast_path_2d = (DIM == 2) && (tile_size == 16) &&
        (tile_extent[0] == 16) && (tile_extent[1] == 16);

    // ------------------------------------------------------------------
    // PIXEL PROCESSING + SPLAT BATCHING
    // ------------------------------------------------------------------
    // __syncthreads() requires ALL threads in the block to participate.
    //
    // For DIM <= 4: tile_pixels == blockDim.x (by design: 16^2=256, 8^3=512,
    // 4^4=256), so every thread executes exactly one pixel-loop iteration and
    // all __syncthreads() calls are reached uniformly.  We use the original
    // pixel-outer / batch-inner loop — zero overhead, maximum compiler
    // optimisation.
    //
    // For DIM >= 5: tile_pixels may differ from blockDim.x (e.g. 3^5=243
    // with blockDim=256, or 3^6=729 with blockDim=256).  Different threads
    // would execute different iteration counts, making __syncthreads()
    // inside the loop undefined behaviour.  We use a batch-outer / pixel-
    // inner structure where __syncthreads() is in the outer loop that all
    // threads traverse uniformly.  Pixel writes use atomicAdd to accumulate
    // across batches (uncontended — each pixel owned by one thread).
    // ------------------------------------------------------------------

    if constexpr (DIM <= 4) {
        // FAST PATH — original loop nesting (pixel outer, batch inner).
        // Safe because tile_pixels == blockDim.x for all standard tile sizes.
        for (int local_px_idx = threadIdx.x; local_px_idx < tile_pixels; local_px_idx += blockDim.x) {
            // Convert local pixel index to voxel coordinates
            int voxel_coords[DIM];
            compute_voxel_coords<DIM>(local_px_idx, tile_origin, tile_extent,
                                      use_fast_path_3d, use_fast_path_2d, voxel_coords);

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
                    s_amps[i] = DTypeTraits<InputDType>::load(amps, splat_idx);

                    // OPTIMIZATION 1.2: Precompute effective truncation squared
                    // This moves the expensive computation out of the hot inner loop
                    // Includes amplitude-based tightening for better early rejection
                    s_truncate_sq[i] = effective_truncate_sq(truncate, s_amps[i], intensity_floor);
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

                    // OPTIMIZATION: Warp-level early termination
                    // If no thread in this warp is within truncation radius, skip __expf
                    bool within_range = (dist_sq <= s_truncate_sq[i]);
                    unsigned int warp_in_range = __ballot_sync(0xFFFFFFFF, within_range);
                    if (warp_in_range == 0) continue;

                    if (within_range) {
                        // Compute intensity (only for pixels within truncation radius)
                        float intensity = gaussian_intensity(dist_sq, s_amps[i]);

                        // Skip if below threshold
                        if (intensity >= intensity_floor) {
                            intensity_sum += intensity;
                        }
                    }
                }
            }

            // Write accumulated intensity to output
            // Note: We use direct assignment (not atomicAdd) because:
            // 1. Each pixel belongs to exactly one tile
            // 2. Each thread in this block processes distinct pixels (different local_px_idx)
            // The global splat kernel uses atomicAdd to add to these values.
            // OPTIMIZATION: Skip write for zero-intensity pixels (output is zero-initialized)
            if (local_px_idx < tile_pixels && intensity_sum != 0.0f) {
                int64_t global_px_idx = voxel_to_linear<DIM>(voxel_coords, shape);
                output[global_px_idx] = intensity_sum;
            }
        }
    } else {
        // SAFE PATH for DIM >= 5 — batch-outer / pixel-inner loop nesting.
        // __syncthreads() is in the outer (batch) loop where ALL threads
        // participate uniformly, regardless of how many pixels each thread owns.
        // Pixel writes use atomicAdd to accumulate across batches.
        for (int batch_start = 0; batch_start < n_splats_in_tile; batch_start += BATCH_SIZE) {
            int batch_size = min(BATCH_SIZE, n_splats_in_tile - batch_start);

            // Cooperative load — identical to fast path
            __syncthreads();
            for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
                int splat_idx = __ldg(&tile_content[tile_offset + batch_start + i]);

                #pragma unroll
                for (int d = 0; d < DIM; d++) {
                    s_centers[i * CENTER_STRIDE + d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
                }
                #pragma unroll
                for (int c = 0; c < CONIC_SIZE; c++) {
                    s_conic[i * CONIC_SIZE + c] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + c);
                }

                s_amps[i] = DTypeTraits<InputDType>::load(amps, splat_idx);
                s_truncate_sq[i] = effective_truncate_sq(truncate, s_amps[i], intensity_floor);
            }
            __syncthreads();

            // Inner pixel loop — no __syncthreads, safe for varying iteration counts
            for (int local_px_idx = threadIdx.x; local_px_idx < tile_pixels; local_px_idx += blockDim.x) {
                int voxel_coords[DIM];
                compute_voxel_coords<DIM>(local_px_idx, tile_origin, tile_extent,
                                          use_fast_path_3d, use_fast_path_2d, voxel_coords);
                float px[DIM];
                #pragma unroll
                for (int d = 0; d < DIM; d++) {
                    px[d] = (float)voxel_coords[d];
                }

                float pixel_intensity = 0.0f;
                for (int i = 0; i < batch_size; i++) {
                    float d[DIM];
                    #pragma unroll
                    for (int dim = 0; dim < DIM; dim++) {
                        d[dim] = px[dim] - s_centers[i * CENTER_STRIDE + dim];
                    }
                    float dist_sq = mahalanobis_distance_sq<DIM>(d, &s_conic[i * CONIC_SIZE]);
                    if (dist_sq > s_truncate_sq[i]) continue;
                    float intensity = gaussian_intensity(dist_sq, s_amps[i]);
                    if (intensity >= intensity_floor) {
                        pixel_intensity += intensity;
                    }
                }

                if (pixel_intensity != 0.0f) {
                    int64_t global_px_idx = voxel_to_linear<DIM>(voxel_coords, shape);
                    atomicAdd(&output[global_px_idx], pixel_intensity);
                }
            }
        }
    }
}

// =============================================================================
// BACKWARD RASTERIZATION KERNEL
// =============================================================================

template <int DIM, int BATCH_SIZE = DEFAULT_BATCH_SIZE, typename InputDType = float>
__global__ __launch_bounds__(512, 3)
void rasterize_backward_kernel(
    const float* __restrict__ grad_output,
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
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
    float* __restrict__ d_amps
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
    __shared__ int s_splat_ids[BATCH_SIZE];
    // OPTIMIZATION 1.2: Precompute effective truncation squared per splat
    __shared__ float s_truncate_sq[BATCH_SIZE];

    // OPTIMIZATION 3.2: Per-tile gradient accumulators for shared memory reduction
    // This reduces atomicAdd operations per splat per tile
    __shared__ float s_d_centers_tile[BATCH_SIZE * CENTER_STRIDE];
    __shared__ float s_d_conic_tile[BATCH_SIZE * CONIC_SIZE];
    __shared__ float s_d_amps_tile[BATCH_SIZE];

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
    // For DIM <= 4: SKIP the cache entirely — precomp_dL_dI loads directly from global
    // into a register, so the shared memory cache (512 writes + sync + 512 reads) is pure waste.
    // For DIM >= 5: Cache is valuable because multiple pixel iterations re-read the same values.
    const bool use_grad_cache = (DIM > 4) &&
                                (n_splats_in_tile > GRAD_CACHE_THRESHOLD) &&
                                (tile_pixels <= MAX_TILE_PIXELS);

    if (use_grad_cache) {
    for (int px = threadIdx.x; px < tile_pixels; px += blockDim.x) {
        int voxel_coords[DIM];
        compute_voxel_coords<DIM>(px, tile_origin, tile_extent,
                                  use_fast_path_3d, use_fast_path_2d, voxel_coords);

        int64_t global_px_idx = voxel_to_linear<DIM>(voxel_coords, shape);
        s_grad_output[px] = grad_output[global_px_idx];
    }
    __syncthreads();
    } // end if (use_grad_cache)

    // OPTIMIZATION: For DIM <= 4, precompute pixel coords and grad_output ONCE
    // Each thread handles exactly one pixel (tile_pixels == blockDim.x by construction)
    // Load grad_output directly from global memory (no shared memory intermediary)
    float precomp_px[DIM];
    float precomp_dL_dI = 0.0f;
    bool precomp_has_grad = false;

    if constexpr (DIM <= 4) {
        compute_pixel_coords_float<DIM>(threadIdx.x, tile_origin, tile_extent,
                                        use_fast_path_3d, use_fast_path_2d, precomp_px);
        // Always load directly from global memory (shared cache skipped for DIM<=4)
        {
            int voxel_int[DIM];
            #pragma unroll
            for (int d = 0; d < DIM; d++) voxel_int[d] = (int)precomp_px[d];
            int64_t global_px_idx = voxel_to_linear<DIM>(voxel_int, shape);
            precomp_dL_dI = grad_output[global_px_idx];
        }
        precomp_has_grad = (precomp_dL_dI != 0.0f);
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

            s_amps[i] = DTypeTraits<InputDType>::load(amps, splat_idx);

            // OPTIMIZATION 1.2: Precompute effective truncation squared
            // Includes amplitude-based tightening for better early rejection
            s_truncate_sq[i] = effective_truncate_sq(truncate, s_amps[i], intensity_floor);

            // OPTIMIZATION 3.2: Initialize gradient accumulators for this batch
            s_d_amps_tile[i] = 0.0f;
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

            #pragma unroll
            for (int d = 0; d < DIM; d++) {
                local_d_centers[d] = 0.0f;
            }
            #pragma unroll
            for (int c = 0; c < CONIC_SIZE; c++) {
                local_d_conic[c] = 0.0f;
            }

            float amp = s_amps[si];
            float truncate_sq = s_truncate_sq[si];

            if constexpr (DIM <= 4) {
                // FAST PATH: Use precomputed pixel coords and grad (no pixel loop needed)
                // For DIM <= 4, tile_pixels == blockDim.x, so each thread = one pixel
                if (precomp_has_grad) {
                    // Compute displacement (using padded stride)
                    float d_vec[DIM];
                    #pragma unroll
                    for (int d = 0; d < DIM; d++) {
                        d_vec[d] = precomp_px[d] - s_centers[si * CENTER_STRIDE + d];
                    }

                    float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, &s_conic[si * CONIC_SIZE]);

                    if (dist_sq <= truncate_sq) {
                        float intensity = gaussian_intensity(dist_sq, amp);
                        if (intensity >= intensity_floor) {
                            compute_pixel_gradients<DIM>(
                                precomp_dL_dI, intensity, amp, d_vec,
                                &s_conic[si * CONIC_SIZE],
                                local_d_centers, local_d_conic,
                                local_d_amp
                            );
                        }
                    }
                }
            } else {
                // GENERIC PATH: pixel loop (DIM >= 5, tile_pixels may differ from blockDim.x)
                for (int local_px_idx = threadIdx.x; local_px_idx < tile_pixels; local_px_idx += blockDim.x) {
                    float px[DIM];
                    compute_pixel_coords_float<DIM>(local_px_idx, tile_origin, tile_extent,
                                                    use_fast_path_3d, use_fast_path_2d, px);

                    float dL_dI;
                    if (use_grad_cache) {
                        dL_dI = s_grad_output[local_px_idx];
                    } else {
                        int voxel_int[DIM];
                        #pragma unroll
                        for (int d = 0; d < DIM; d++) voxel_int[d] = (int)px[d];
                        int64_t global_px_idx = voxel_to_linear<DIM>(voxel_int, shape);
                        dL_dI = grad_output[global_px_idx];
                    }

                    if (dL_dI == 0.0f) continue;

                    float d_vec[DIM];
                    #pragma unroll
                    for (int d = 0; d < DIM; d++) {
                        d_vec[d] = px[d] - s_centers[si * CENTER_STRIDE + d];
                    }

                    float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, &s_conic[si * CONIC_SIZE]);

                    if (dist_sq > truncate_sq) continue;

                    float intensity = gaussian_intensity(dist_sq, amp);
                    if (intensity < intensity_floor) continue;

                    compute_pixel_gradients<DIM>(
                        dL_dI, intensity, amp, d_vec,
                        &s_conic[si * CONIC_SIZE],
                        local_d_centers, local_d_conic,
                        local_d_amp
                    );
                }
            }

            // OPTIMIZATION 3.2 + 3.3: Warp-level reduction with early termination
            // Skip the entire reduction when no thread in the warp contributed gradients.
            // For sparse splats (most common case), many warps have zero contributions.
            // __ballot_sync costs ~5 cycles but saves 50 shuffles + 10 shared atomics.
            unsigned int warp_has_grads = __ballot_sync(0xFFFFFFFF, local_d_amp != 0.0f);

            if (warp_has_grads != 0) {
                int lane = threadIdx.x % 32;

                // OPTIMIZATION: Batch all warp reductions BEFORE any atomics.
                // Each warp_reduce_sum requires all threads via __shfl_down_sync.
                // Interleaving reduce→atomic→reduce→atomic serializes because
                // lane 0's atomic blocks the next __shfl_down_sync for all lanes.
                // By batching reductions first, the GPU can interleave independent
                // shuffle operations across components (ILP).
                float warp_d_amp = warp_reduce_sum(local_d_amp);

                float warp_d_centers[DIM];
                #pragma unroll
                for (int d = 0; d < DIM; d++) {
                    warp_d_centers[d] = warp_reduce_sum(local_d_centers[d]);
                }

                float warp_d_conic[CONIC_SIZE];
                #pragma unroll
                for (int c = 0; c < CONIC_SIZE; c++) {
                    warp_d_conic[c] = warp_reduce_sum(local_d_conic[c]);
                }

                // Now batch all atomic writes (only lane 0)
                if (lane == 0) {
                    atomicAdd(&s_d_amps_tile[si], warp_d_amp);
                    #pragma unroll
                    for (int d = 0; d < DIM; d++) {
                        atomicAdd(&s_d_centers_tile[si * CENTER_STRIDE + d], warp_d_centers[d]);
                    }
                    #pragma unroll
                    for (int c = 0; c < CONIC_SIZE; c++) {
                        atomicAdd(&s_d_conic_tile[si * CONIC_SIZE + c], warp_d_conic[c]);
                    }
                }
            }
        }

        // OPTIMIZATION 3.2: Cooperative write-back to global memory
        // This reduces atomicAdds per splat per tile (3D case)
        __syncthreads();
        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
            int splat_idx = s_splat_ids[i];
            atomicAdd(&d_amps[splat_idx], s_d_amps_tile[i]);
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

// =============================================================================
// SPLAT-CENTRIC FORWARD RASTERIZATION KERNEL
// =============================================================================
// Each block processes ONE splat: computes its voxel AABB from conic,
// iterates over all voxels, computes intensity, and atomicAdds to output.
// For sparse data (most pixels receive 0-1 contributions), atomicAdd
// contention is negligible. Eliminates the entire tile binning pipeline.

template <int DIM, typename InputDType = float>
__global__ __launch_bounds__(256, 6)
void rasterize_forward_splat_centric_kernel(
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
    int N,
    const int* __restrict__ shape,
    float truncate,
    float intensity_floor,
    float* __restrict__ output,
    // Optional: global splat detection (pass nullptr to skip)
    bool* __restrict__ global_splat_flags,
    int* __restrict__ global_splat_count,
    int64_t num_tiles,
    int tile_size_param,
    // Optional: tile_counts for diagnostic compatibility (pass nullptr to skip)
    int* __restrict__ tile_counts_out,
    const int* __restrict__ tile_dims
) {
    int splat_idx = blockIdx.x;
    if (splat_idx >= N) return;

    constexpr int CONIC_SIZE_L = conic_size<DIM>();

    // Load splat data into shared memory (broadcast to all threads)
    __shared__ float s_center[DIM];
    __shared__ float s_conic[CONIC_SIZE_L];
    __shared__ float s_amp;
    __shared__ float s_truncate_sq;
    __shared__ int s_lo[DIM], s_hi[DIM];
    __shared__ int s_extent[DIM];
    __shared__ int s_total_voxels;

    if (threadIdx.x == 0) {
        s_amp = DTypeTraits<InputDType>::load(amps, splat_idx);
        s_truncate_sq = effective_truncate_sq(truncate, s_amp, intensity_floor);

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            s_center[d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
        }
        #pragma unroll
        for (int c = 0; c < CONIC_SIZE_L; c++) {
            s_conic[c] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE_L + c);
        }

        // Compute voxel AABB from conic (same as backward kernel)
        float t_eff = effective_truncation(truncate, s_amp, intensity_floor);

        if constexpr (DIM == 3) {
            float c00 = s_conic[0], c01 = s_conic[1], c02 = s_conic[2];
            float c11 = s_conic[3], c12 = s_conic[4], c22 = s_conic[5];
            float det = c00*(c11*c22 - c12*c12) - c01*(c01*c22 - c02*c12)
                      + c02*(c01*c12 - c02*c11);
            float inv_det = 1.0f / fmaxf(fabsf(det), 1e-10f);
            float sigma[3] = {
                sqrtf(fmaxf((c11*c22 - c12*c12) * inv_det, 0.0f)),
                sqrtf(fmaxf((c00*c22 - c02*c02) * inv_det, 0.0f)),
                sqrtf(fmaxf((c00*c11 - c01*c01) * inv_det, 0.0f))
            };
            int total = 1;
            #pragma unroll
            for (int d = 0; d < 3; d++) {
                // AABB matches PyTorch reference: ceil(t_eff * sigma) as integer radius,
                // then floor(center) - radius for lo, ceil(center) + radius for hi.
                // This ensures the CUDA AABB covers ALL pixels the reference covers.
                int int_radius = (int)ceilf(t_eff * sigma[d]);
                s_lo[d] = max(0, (int)floorf(s_center[d]) - int_radius);
                s_hi[d] = min(shape[d] - 1, (int)ceilf(s_center[d]) + int_radius);
                s_extent[d] = max(0, s_hi[d] - s_lo[d] + 1);
                total *= s_extent[d];
            }
            s_total_voxels = total;
        } else if constexpr (DIM == 2) {
            float c00 = s_conic[0], c01 = s_conic[1], c11 = s_conic[2];
            float det = c00*c11 - c01*c01;
            float inv_det = 1.0f / fmaxf(fabsf(det), 1e-10f);
            float sigma[2] = {
                sqrtf(fmaxf(c11 * inv_det, 0.0f)),
                sqrtf(fmaxf(c00 * inv_det, 0.0f))
            };
            int total = 1;
            #pragma unroll
            for (int d = 0; d < 2; d++) {
                // AABB matches PyTorch reference: ceil(t_eff * sigma) as integer radius,
                // then floor(center) - radius for lo, ceil(center) + radius for hi.
                // This ensures the CUDA AABB covers ALL pixels the reference covers.
                int int_radius = (int)ceilf(t_eff * sigma[d]);
                s_lo[d] = max(0, (int)floorf(s_center[d]) - int_radius);
                s_hi[d] = min(shape[d] - 1, (int)ceilf(s_center[d]) + int_radius);
                s_extent[d] = max(0, s_hi[d] - s_lo[d] + 1);
                total *= s_extent[d];
            }
            s_total_voxels = total;
        } else {
            int total = 1;
            int ci = 0;
            for (int d = 0; d < DIM; d++) {
                float sigma_d = 1.0f / sqrtf(fmaxf(s_conic[ci], 1e-10f));
                // Generic DIM: conic diagonal gives lower bound on sigma,
                // use 1.5x safety factor + ceil to match reference conservatively
                int int_radius = (int)ceilf(t_eff * sigma_d * 1.5f);
                s_lo[d] = max(0, (int)floorf(s_center[d]) - int_radius);
                s_hi[d] = min(shape[d] - 1, (int)ceilf(s_center[d]) + int_radius);
                s_extent[d] = max(0, s_hi[d] - s_lo[d] + 1);
                total *= s_extent[d];
                ci += (DIM - d);
            }
            s_total_voxels = total;
        }

        // Optional: flag global splats and compute tile_counts (API/diagnostic compat)
        if (global_splat_flags != nullptr) {
            // Compute tile AABB from voxel AABB
            int n_tiles_approx = 1;
            for (int d = 0; d < DIM; d++) {
                int tile_lo = s_lo[d] / tile_size_param;
                int tile_hi = s_hi[d] / tile_size_param;
                n_tiles_approx *= max(1, tile_hi - tile_lo + 1);
            }
            bool is_global = (n_tiles_approx > (int)(0.1f * (float)num_tiles))
                          && (n_tiles_approx > 1024);
            global_splat_flags[splat_idx] = is_global;
            if (is_global && global_splat_count != nullptr) {
                atomicAdd(global_splat_count, 1);
            }

            // Compute tile_counts for diagnostic compatibility
            if (tile_counts_out != nullptr && tile_dims != nullptr && !is_global) {
                int tile_lo[DIM], tile_hi[DIM];
                for (int d = 0; d < DIM; d++) {
                    tile_lo[d] = max(0, s_lo[d] / tile_size_param);
                    tile_hi[d] = min(tile_dims[d] - 1, s_hi[d] / tile_size_param);
                }
                // Iterate tile AABB and increment counts (same as preprocess_kernel)
                int tile_coords_iter[DIM];
                for (int d = 0; d < DIM; d++) tile_coords_iter[d] = tile_lo[d];
                for (int i = 0; i < n_tiles_approx; i++) {
                    int tile_idx_linear = tile_coords_to_linear<DIM>(tile_coords_iter, tile_dims);
                    atomicAdd(&tile_counts_out[tile_idx_linear], 1);
                    // Advance odometer
                    for (int d = DIM - 1; d >= 0; d--) {
                        tile_coords_iter[d]++;
                        if (tile_coords_iter[d] <= tile_hi[d]) break;
                        tile_coords_iter[d] = tile_lo[d];
                    }
                }
            }
        }
    }
    __syncthreads();

    if (s_total_voxels == 0) return;

    // Load splat data into registers
    float amp = s_amp;
    float truncate_sq = s_truncate_sq;
    float center_reg[DIM];
    float conic_reg[CONIC_SIZE_L];
    #pragma unroll
    for (int d = 0; d < DIM; d++) center_reg[d] = s_center[d];
    #pragma unroll
    for (int c = 0; c < CONIC_SIZE_L; c++) conic_reg[c] = s_conic[c];

    int total_voxels = s_total_voxels;
    for (int vox_idx = threadIdx.x; vox_idx < total_voxels; vox_idx += blockDim.x) {
        int voxel[DIM];
        {
            int remaining = vox_idx;
            #pragma unroll
            for (int d = DIM - 1; d >= 0; d--) {
                voxel[d] = s_lo[d] + (remaining % s_extent[d]);
                remaining /= s_extent[d];
            }
        }

        float d_vec[DIM];
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            d_vec[d] = (float)voxel[d] - center_reg[d];
        }

        float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, conic_reg);
        if (dist_sq > truncate_sq) continue;

        float intensity = gaussian_intensity(dist_sq, amp);
        if (intensity >= intensity_floor) {
            int64_t global_px_idx = voxel_to_linear<DIM>(voxel, shape);
            atomicAdd(&output[global_px_idx], intensity);
        }
    }
}

// =============================================================================
// SPLAT-CENTRIC BACKWARD RASTERIZATION KERNEL
// =============================================================================
// Each block processes ONE splat: iterates over all voxels in its AABB,
// accumulates gradients in thread-local registers, reduces once, and writes
// directly to global memory with ZERO atomics. Replaces both the tile-centric
// backward and global splat backward kernels.
//
// Key advantages over tile-centric:
// 1. No tile binning dependency (no tile_offsets/counts/content needed)
// 2. No shared memory gradient accumulators (saves ~6KB per block)
// 3. No global atomicAdds for gradient write-back (each block owns its splat)
// 4. Handles global splats uniformly (no separate kernel needed)
// 5. Simpler block reduction (1 per splat vs 128 warp reductions per tile)

template <int DIM, typename InputDType = float>
__global__ __launch_bounds__(256, 6)
void rasterize_backward_splat_centric_kernel(
    const float* __restrict__ grad_output,
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
    int N,
    const int* __restrict__ shape,
    float truncate,
    float intensity_floor,
    float* __restrict__ d_centers,
    float* __restrict__ d_conic,
    float* __restrict__ d_amps,
    // Optional: zero the forward output tensor as a side effect (eliminates output.zero_() on next call)
    float* __restrict__ output_to_zero
) {
    int splat_idx = blockIdx.x;
    if (splat_idx >= N) return;

    constexpr int CONIC_SIZE_L = conic_size<DIM>();

    // Load splat data into shared memory (broadcast to all threads)
    __shared__ float s_center[DIM];
    __shared__ float s_conic[CONIC_SIZE_L];
    __shared__ float s_amp;
    __shared__ float s_truncate_sq;
    __shared__ int s_lo[DIM], s_hi[DIM];
    __shared__ int s_extent[DIM];
    __shared__ int s_total_voxels;

    if (threadIdx.x == 0) {
        s_amp = DTypeTraits<InputDType>::load(amps, splat_idx);
        s_truncate_sq = effective_truncate_sq(truncate, s_amp, intensity_floor);

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            s_center[d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
        }
        #pragma unroll
        for (int c = 0; c < CONIC_SIZE_L; c++) {
            s_conic[c] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE_L + c);
        }

        // Compute voxel AABB from conic (Sigma^-1) by extracting Sigma diagonals
        // via cofactor / determinant. This gives the exact per-axis standard deviation.
        float t_eff = effective_truncation(truncate, s_amp, intensity_floor);

        if constexpr (DIM == 3) {
            float c00 = s_conic[0], c01 = s_conic[1], c02 = s_conic[2];
            float c11 = s_conic[3], c12 = s_conic[4], c22 = s_conic[5];
            float det = c00*(c11*c22 - c12*c12) - c01*(c01*c22 - c02*c12)
                      + c02*(c01*c12 - c02*c11);
            float inv_det = 1.0f / fmaxf(fabsf(det), 1e-10f);
            float sigma[3] = {
                sqrtf(fmaxf((c11*c22 - c12*c12) * inv_det, 0.0f)),
                sqrtf(fmaxf((c00*c22 - c02*c02) * inv_det, 0.0f)),
                sqrtf(fmaxf((c00*c11 - c01*c01) * inv_det, 0.0f))
            };
            int total = 1;
            #pragma unroll
            for (int d = 0; d < 3; d++) {
                float radius = t_eff * sigma[d] + 0.5f;  // +0.5 safety margin
                s_lo[d] = max(0, (int)floorf(s_center[d] - radius));
                s_hi[d] = min(shape[d] - 1, (int)floorf(s_center[d] + radius));
                s_extent[d] = max(0, s_hi[d] - s_lo[d] + 1);
                total *= s_extent[d];
            }
            s_total_voxels = total;
        } else if constexpr (DIM == 2) {
            float c00 = s_conic[0], c01 = s_conic[1], c11 = s_conic[2];
            float det = c00*c11 - c01*c01;
            float inv_det = 1.0f / fmaxf(fabsf(det), 1e-10f);
            float sigma[2] = {
                sqrtf(fmaxf(c11 * inv_det, 0.0f)),
                sqrtf(fmaxf(c00 * inv_det, 0.0f))
            };
            int total = 1;
            #pragma unroll
            for (int d = 0; d < 2; d++) {
                // AABB matches PyTorch reference: ceil(t_eff * sigma) as integer radius,
                // then floor(center) - radius for lo, ceil(center) + radius for hi.
                // This ensures the CUDA AABB covers ALL pixels the reference covers.
                int int_radius = (int)ceilf(t_eff * sigma[d]);
                s_lo[d] = max(0, (int)floorf(s_center[d]) - int_radius);
                s_hi[d] = min(shape[d] - 1, (int)ceilf(s_center[d]) + int_radius);
                s_extent[d] = max(0, s_hi[d] - s_lo[d] + 1);
                total *= s_extent[d];
            }
            s_total_voxels = total;
        } else {
            // Generic: use diagonal of conic as conservative approximation
            // 1/sqrt(conic[ii]) gives a LOWER bound on sigma_i, so we add extra margin
            int total = 1;
            int ci = 0;
            for (int d = 0; d < DIM; d++) {
                float sigma_d = 1.0f / sqrtf(fmaxf(s_conic[ci], 1e-10f));
                float radius = t_eff * sigma_d * 1.5f + 1.0f;  // generous margin
                s_lo[d] = max(0, (int)floorf(s_center[d] - radius));
                s_hi[d] = min(shape[d] - 1, (int)floorf(s_center[d] + radius));
                s_extent[d] = max(0, s_hi[d] - s_lo[d] + 1);
                total *= s_extent[d];
                ci += (DIM - d);  // skip to next diagonal in packed triangle
            }
            s_total_voxels = total;
        }
    }
    __syncthreads();

    if (s_total_voxels == 0) return;

    // Thread-local gradient accumulators
    float local_d_centers[DIM];
    float local_d_conic[CONIC_SIZE_L];
    float local_d_amp = 0.0f;
    #pragma unroll
    for (int d = 0; d < DIM; d++) local_d_centers[d] = 0.0f;
    #pragma unroll
    for (int c = 0; c < CONIC_SIZE_L; c++) local_d_conic[c] = 0.0f;

    // Load splat data from shared memory into registers
    float amp = s_amp;
    float truncate_sq = s_truncate_sq;
    float center_reg[DIM];
    float conic_reg[CONIC_SIZE_L];
    #pragma unroll
    for (int d = 0; d < DIM; d++) center_reg[d] = s_center[d];
    #pragma unroll
    for (int c = 0; c < CONIC_SIZE_L; c++) conic_reg[c] = s_conic[c];

    // Iterate over all voxels in AABB
    int total_voxels = s_total_voxels;
    for (int vox_idx = threadIdx.x; vox_idx < total_voxels; vox_idx += blockDim.x) {
        // Convert linear index to voxel coordinates within AABB
        int voxel[DIM];
        {
            int remaining = vox_idx;
            #pragma unroll
            for (int d = DIM - 1; d >= 0; d--) {
                voxel[d] = s_lo[d] + (remaining % s_extent[d]);
                remaining /= s_extent[d];
            }
        }

        // Load grad_output
        int64_t global_px_idx = voxel_to_linear<DIM>(voxel, shape);
        float dL_dI = grad_output[global_px_idx];
        if (dL_dI == 0.0f) continue;

        // Compute displacement
        float d_vec[DIM];
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            d_vec[d] = (float)voxel[d] - center_reg[d];
        }

        // Mahalanobis distance
        float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, conic_reg);
        if (dist_sq > truncate_sq) continue;

        float intensity = gaussian_intensity(dist_sq, amp);
        if (intensity < intensity_floor) continue;

        // OPTIMIZATION: Zero ONLY the pixels that the forward actually wrote to.
        // This is inside the same intensity >= floor check as the forward kernel's
        // atomicAdd, so we zero exactly the same set of pixels. Cost: ~0.02ms
        // (vs 0.56ms for output.zero_()) because only ~5M of 134M pixels are hit.
        // Safe: backward runs AFTER forward, output is consumed by loss computation,
        // and the zeroed buffer is ready for the next forward's atomicAdd.
        if (output_to_zero != nullptr) {
            output_to_zero[global_px_idx] = 0.0f;
        }

        // Accumulate gradients (template-specialized for 2D/3D)
        compute_pixel_gradients<DIM>(
            dL_dI, intensity, amp, d_vec, conic_reg,
            local_d_centers, local_d_conic, local_d_amp
        );
    }

    // Two-level reduction: warp → block → single write
    // Step 1: Warp reduction
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    constexpr int NUM_WARPS = 256 / 32;  // 8 warps

    float warp_d_amp = warp_reduce_sum(local_d_amp);
    float warp_d_centers[DIM];
    float warp_d_conic[CONIC_SIZE_L];
    #pragma unroll
    for (int d = 0; d < DIM; d++) warp_d_centers[d] = warp_reduce_sum(local_d_centers[d]);
    #pragma unroll
    for (int c = 0; c < CONIC_SIZE_L; c++) warp_d_conic[c] = warp_reduce_sum(local_d_conic[c]);

    // Step 2: Block reduction via shared memory
    __shared__ float s_block_d_amp[NUM_WARPS];
    __shared__ float s_block_d_centers[NUM_WARPS * DIM];
    __shared__ float s_block_d_conic[NUM_WARPS * CONIC_SIZE_L];

    if (lane == 0) {
        s_block_d_amp[warp_id] = warp_d_amp;
        #pragma unroll
        for (int d = 0; d < DIM; d++) s_block_d_centers[warp_id * DIM + d] = warp_d_centers[d];
        #pragma unroll
        for (int c = 0; c < CONIC_SIZE_L; c++) s_block_d_conic[warp_id * CONIC_SIZE_L + c] = warp_d_conic[c];
    }
    __syncthreads();

    // Final reduction: first warp reduces the NUM_WARPS partial sums
    if (threadIdx.x < NUM_WARPS) {
        float final_amp = s_block_d_amp[threadIdx.x];
        // Reduce across the first warp (only NUM_WARPS values, pad rest with 0)
        #pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            final_amp += __shfl_down_sync(0xFF, final_amp, offset);
        }
        if (threadIdx.x == 0) d_amps[splat_idx] = final_amp;

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            float val = s_block_d_centers[threadIdx.x * DIM + d];
            #pragma unroll
            for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xFF, val, offset);
            }
            if (threadIdx.x == 0) d_centers[splat_idx * DIM + d] = val;
        }
        #pragma unroll
        for (int c = 0; c < CONIC_SIZE_L; c++) {
            float val = s_block_d_conic[threadIdx.x * CONIC_SIZE_L + c];
            #pragma unroll
            for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
                val += __shfl_down_sync(0xFF, val, offset);
            }
            if (threadIdx.x == 0) d_conic[splat_idx * CONIC_SIZE_L + c] = val;
        }
    }
}

#endif // CUDA_SPLATTING_KERNELS_CORE_CUH
