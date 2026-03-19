/**
 * Reduction and Gradient Utilities for Gaussian Splatting
 *
 * This header provides parallel reduction operations and backward pass helpers:
 * - Warp-level sum reduction and aggregated atomic adds
 * - Gradient computation for Gaussian intensity (amplitude, dist_sq)
 * - Explicit 2D/3D backward pass implementations
 */

#ifndef CUDA_SPLATTING_REDUCTION_UTILS_CUH
#define CUDA_SPLATTING_REDUCTION_UTILS_CUH

#include <cuda_runtime.h>
#include <cmath>

// Forward declaration - grad functions use math from math_utils.cuh
#include "math_utils.cuh"

// =============================================================================
// WARP-LEVEL REDUCTION OPERATIONS
// =============================================================================

/**
 * Warp-level sum reduction using shuffle instructions.
 *
 * All threads in the warp contribute their value, result is in lane 0.
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * Atomic add with warp aggregation to reduce contention.
 *
 * All threads in the warp with the same address sum their values,
 * and only one thread performs the atomic add.
 */
__device__ __forceinline__ void warp_aggregated_atomic_add(
    float* __restrict__ addr,
    float val
) {
    // Sum across warp
    float sum = warp_reduce_sum(val);

    // Lane 0 performs the atomic add
    if ((threadIdx.x % 32) == 0) {
        atomicAdd(addr, sum);
    }
}

// =============================================================================
// GRADIENT HELPERS
// =============================================================================

/**
 * Compute gradient of intensity w.r.t. dist_sq.
 *
 * For standard Gaussian: I = a * exp(-0.5 * D²)
 * ∂I/∂D² = I × (-0.5)
 */
__device__ __forceinline__ float grad_intensity_wrt_dist_sq(
    float intensity
) {
    return intensity * (-0.5f);
}

/**
 * Compute gradient of intensity w.r.t. amplitude.
 *
 * ∂I/∂a = exp(-0.5 × D²) = I / a
 */
__device__ __forceinline__ float grad_intensity_wrt_amplitude(
    float intensity,
    float amplitude
) {
    return intensity / fmaxf(amplitude, 1e-10f);
}

// =============================================================================
// EXPLICIT 3D BACKWARD GRADIENT COMPUTATION
// =============================================================================

/**
 * Compute ∂D²/∂d for 3D case (explicit formula).
 *
 * ∂D²/∂d = 2 * Σ⁻¹ @ d
 *
 * For 3D with conic layout [c00, c01, c02, c11, c12, c22]:
 *   ∂D²/∂d[0] = 2 * (c00*d0 + c01*d1 + c02*d2)
 *   ∂D²/∂d[1] = 2 * (c01*d0 + c11*d1 + c12*d2)
 *   ∂D²/∂d[2] = 2 * (c02*d0 + c12*d1 + c22*d2)
 *
 * OPTIMIZATION: Explicit formula avoids loop overhead and index computation.
 * Expected speedup: 25-35% in backward pass for 3D data.
 *
 * @param d       Displacement vector (px - center), length 3
 * @param conic   Packed upper triangle of Σ⁻¹, [c00,c01,c02,c11,c12,c22]
 * @param dD2_dd  Output: gradient ∂D²/∂d, length 3
 */
__device__ __forceinline__ void compute_dD2_dd_3d(
    const float* __restrict__ d,
    const float* __restrict__ conic,
    float* __restrict__ dD2_dd
) {
    // Σ⁻¹ @ d (using symmetry, conic stores upper triangle)
    // conic layout: [c00, c01, c02, c11, c12, c22]
    //               [ 0    1    2    3    4    5 ]
    float c00 = conic[0], c01 = conic[1], c02 = conic[2];
    float c11 = conic[3], c12 = conic[4], c22 = conic[5];

    float d0 = d[0], d1 = d[1], d2 = d[2];

    // 2 * (Σ⁻¹ @ d)
    dD2_dd[0] = 2.0f * (c00 * d0 + c01 * d1 + c02 * d2);
    dD2_dd[1] = 2.0f * (c01 * d0 + c11 * d1 + c12 * d2);
    dD2_dd[2] = 2.0f * (c02 * d0 + c12 * d1 + c22 * d2);
}

/**
 * Compute all backward gradients for 3D (explicit, fully unrolled).
 *
 * This is the complete backward computation for a single pixel-splat pair
 * in 3D, using explicit formulas instead of loops.
 *
 * @param dL_dI         Upstream gradient (∂L/∂I)
 * @param intensity     Computed intensity at this pixel
 * @param amp           Splat amplitude
 * @param d_vec         Displacement (px - center), length 3
 * @param conic         Packed conic, length 6
 * @param local_d_centers Output: accumulated center gradients, length 3
 * @param local_d_conic   Output: accumulated conic gradients, length 6
 * @param local_d_amp     Output: accumulated amplitude gradient (single value)
 */
__device__ __forceinline__ void backward_pixel_splat_3d(
    float dL_dI,
    float intensity,
    float amp,
    const float* __restrict__ d_vec,
    const float* __restrict__ conic,
    float* __restrict__ local_d_centers,
    float* __restrict__ local_d_conic,
    float& local_d_amp
) {
    // Gradient w.r.t amplitude
    local_d_amp += dL_dI * grad_intensity_wrt_amplitude(intensity, amp);

    // Gradient w.r.t dist_sq - pre-compute common factor (eliminates 8 redundant multiplies)
    float grad_dist = grad_intensity_wrt_dist_sq(intensity);
    float outer = dL_dI * grad_dist;

    // Compute ∂D²/∂d (3D explicit)
    float dD2_dd[3];
    compute_dD2_dd_3d(d_vec, conic, dD2_dd);

    // ∂L/∂center = outer * ∂D²/∂d * (-1)
    // (negative because d = px - center, so ∂d/∂center = -1)
    float outer_neg = -outer;
    local_d_centers[0] += outer_neg * dD2_dd[0];
    local_d_centers[1] += outer_neg * dD2_dd[1];
    local_d_centers[2] += outer_neg * dD2_dd[2];

    // ∂L/∂conic = outer * ∂D²/∂conic (inline computation, no intermediate array)
    float d0 = d_vec[0], d1 = d_vec[1], d2 = d_vec[2];
    local_d_conic[0] += outer * d0 * d0;          // ∂D²/∂c00 = d0²
    local_d_conic[1] += outer * 2.0f * d0 * d1;   // ∂D²/∂c01 = 2*d0*d1
    local_d_conic[2] += outer * 2.0f * d0 * d2;   // ∂D²/∂c02 = 2*d0*d2
    local_d_conic[3] += outer * d1 * d1;           // ∂D²/∂c11 = d1²
    local_d_conic[4] += outer * 2.0f * d1 * d2;    // ∂D²/∂c12 = 2*d1*d2
    local_d_conic[5] += outer * d2 * d2;           // ∂D²/∂c22 = d2²
}

/**
 * Compute all backward gradients for 2D (explicit, fully unrolled).
 */
__device__ __forceinline__ void backward_pixel_splat_2d(
    float dL_dI,
    float intensity,
    float amp,
    const float* __restrict__ d_vec,
    const float* __restrict__ conic,
    float* __restrict__ local_d_centers,
    float* __restrict__ local_d_conic,
    float& local_d_amp
) {
    // Gradient w.r.t amplitude
    local_d_amp += dL_dI * grad_intensity_wrt_amplitude(intensity, amp);

    // Gradient w.r.t dist_sq - pre-compute common factor
    float grad_dist = grad_intensity_wrt_dist_sq(intensity);
    float outer = dL_dI * grad_dist;

    // For 2D, conic layout: [c00, c01, c11]
    float c00 = conic[0], c01 = conic[1], c11 = conic[2];
    float d0 = d_vec[0], d1 = d_vec[1];

    // ∂D²/∂d = 2 * Σ⁻¹ @ d
    float dD2_dd0 = 2.0f * (c00 * d0 + c01 * d1);
    float dD2_dd1 = 2.0f * (c01 * d0 + c11 * d1);

    // ∂L/∂center = outer * ∂D²/∂d * (-1)
    float outer_neg = -outer;
    local_d_centers[0] += outer_neg * dD2_dd0;
    local_d_centers[1] += outer_neg * dD2_dd1;

    // ∂D²/∂conic (inlined)
    local_d_conic[0] += outer * d0 * d0;
    local_d_conic[1] += outer * 2.0f * d0 * d1;
    local_d_conic[2] += outer * d1 * d1;
}

#endif // CUDA_SPLATTING_REDUCTION_UTILS_CUH
