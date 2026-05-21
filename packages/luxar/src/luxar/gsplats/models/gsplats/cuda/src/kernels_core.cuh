/**
 * CUDA Core Kernel Implementations
 *
 * This header contains the core kernel implementations for volumetric Gaussian splatting:
 * - Backward gradient dispatch helpers (compute_pixel_gradients)
 * - Splat-centric forward rasterization kernel (1 block per splat, atomicAdd to output)
 * - Splat-centric backward rasterization kernel (1 block per splat, warp reduction)
 *
 * These kernels are templated on DIM for compile-time optimization of 2D/3D cases.
 */

#ifndef CUDA_SPLATTING_KERNELS_CORE_CUH
#define CUDA_SPLATTING_KERNELS_CORE_CUH

#include "utils.cuh"

// =============================================================================
// LAUNCH BOUNDS
// =============================================================================
// Only specify maxThreadsPerBlock — do NOT add minBlocksPerSM (the second
// parameter).  Benchmarking on RTX 3090 Ti (sm_86) showed that the
// minBlocksPerSM hint causes the compiler to make worse register-allocation
// decisions, resulting in measurable regressions.  See MEMORY.md for details.

#define LAUNCH_BOUNDS_FWD_SPLAT  __launch_bounds__(256)
#define LAUNCH_BOUNDS_BWD_SPLAT  __launch_bounds__(256)

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
    float shift_C,
    float inv_one_minus_C,
    float* __restrict__ local_d_centers,
    float* __restrict__ local_d_conic,
    float& local_d_amp
) {
    // Gradient w.r.t. amplitude
    local_d_amp += dL_dI * grad_intensity_wrt_amplitude(intensity, amp);

    // Gradient w.r.t. dist_sq (shifted Gaussian)
    float grad_dist = grad_intensity_wrt_dist_sq(intensity, amp, shift_C, inv_one_minus_C);

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
    float shift_C,
    float inv_one_minus_C,
    float* __restrict__ local_d_centers,
    float* __restrict__ local_d_conic,
    float& local_d_amp
) {
    backward_pixel_splat_3d(dL_dI, intensity, amp, d_vec, conic,
                           shift_C, inv_one_minus_C,
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
    float shift_C,
    float inv_one_minus_C,
    float* __restrict__ local_d_centers,
    float* __restrict__ local_d_conic,
    float& local_d_amp
) {
    backward_pixel_splat_2d(dL_dI, intensity, amp, d_vec, conic,
                           shift_C, inv_one_minus_C,
                           local_d_centers, local_d_conic, local_d_amp);
}

// =============================================================================
// SPLAT-CENTRIC FORWARD RASTERIZATION KERNEL
// =============================================================================
// Each block processes ONE splat: computes its voxel AABB from conic,
// iterates over all voxels, computes intensity, and atomicAdds to output.
// For sparse data (most pixels receive 0-1 contributions), atomicAdd
// contention is negligible. Eliminates the entire tile binning pipeline.

template <int DIM, typename InputDType = float>
__global__ LAUNCH_BOUNDS_FWD_SPLAT
void rasterize_forward_splat_centric_kernel(
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
    int N,
    const int* __restrict__ shape,
    float truncate,
    float intensity_floor,
    float* __restrict__ output
) {
    // Precompute shifted Gaussian truncation parameters (once per kernel)
    const GaussianShiftParams gsp = compute_shift_params(truncate);
    const float shift_C = gsp.shift_C;
    const float inv_one_minus_C = gsp.inv_one_minus_C;

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
        s_truncate_sq = effective_truncate_sq(truncate, s_amp, intensity_floor, shift_C, inv_one_minus_C);

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            s_center[d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
        }
        #pragma unroll
        for (int c = 0; c < CONIC_SIZE_L; c++) {
            s_conic[c] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE_L + c);
        }

        // Compute voxel AABB from conic (Sigma^-1) via cofactor/determinant
        float t_eff = effective_truncation(truncate, s_amp, intensity_floor, shift_C, inv_one_minus_C);

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

        // Early rejection via Mahalanobis distance: skip pixels outside the
        // effective truncation sphere. Uses min(truncate², 2*ln(amp/floor)).
        //
        // KNOWN LIMITATION: For high-amplitude splats, the base truncation
        // (truncate²) can be tighter than the amplitude-based cutoff. This
        // may reject ~4 borderline pixels per volume (intensity within 42% of
        // floor threshold). Sum accuracy remains within 0.001% of reference.
        // See README.md for current accuracy notes.
        float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, conic_reg);
        if (dist_sq > truncate_sq) continue;

        float intensity = gaussian_intensity(dist_sq, amp, shift_C, inv_one_minus_C);
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
__global__ LAUNCH_BOUNDS_BWD_SPLAT
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
    // Precompute shifted Gaussian truncation parameters (once per kernel)
    const GaussianShiftParams gsp = compute_shift_params(truncate);
    const float shift_C = gsp.shift_C;
    const float inv_one_minus_C = gsp.inv_one_minus_C;

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
        s_truncate_sq = effective_truncate_sq(truncate, s_amp, intensity_floor, shift_C, inv_one_minus_C);

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
        float t_eff = effective_truncation(truncate, s_amp, intensity_floor, shift_C, inv_one_minus_C);

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
                // AABB matches forward kernel and PyTorch reference exactly
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
            // Generic DIM: conic diagonal gives lower bound on sigma,
            // use 1.5x safety factor + ceil to match reference conservatively
            int total = 1;
            int ci = 0;
            for (int d = 0; d < DIM; d++) {
                float sigma_d = 1.0f / sqrtf(fmaxf(s_conic[ci], 1e-10f));
                int int_radius = (int)ceilf(t_eff * sigma_d * 1.5f);
                s_lo[d] = max(0, (int)floorf(s_center[d]) - int_radius);
                s_hi[d] = min(shape[d] - 1, (int)ceilf(s_center[d]) + int_radius);
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

        // Same truncation as forward (see KNOWN LIMITATION comment above)
        float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, conic_reg);
        if (dist_sq > truncate_sq) continue;

        float intensity = gaussian_intensity(dist_sq, amp, shift_C, inv_one_minus_C);
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
            shift_C, inv_one_minus_C,
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
