/**
 * Mathematical Utilities for Gaussian Splatting
 *
 * This header provides mathematical functions for Gaussian computation:
 * - Triangular matrix indexing for packed covariance matrices
 * - Mahalanobis distance calculation (generic + optimized 2D/3D)
 * - Standard Gaussian intensity computation
 * - Effective truncation radius for early rejection
 */

#ifndef CUDA_SPLATTING_MATH_UTILS_CUH
#define CUDA_SPLATTING_MATH_UTILS_CUH

#include <cuda_runtime.h>
#include <cmath>

// =============================================================================
// TRIANGULAR MATRIX UTILITIES
// =============================================================================

/**
 * Get number of elements in upper triangle of DxD symmetric matrix.
 * conic_size = D*(D+1)/2
 */
template <int DIM>
__host__ __device__ __forceinline__ constexpr int conic_size() {
    return DIM * (DIM + 1) / 2;
}

/**
 * Convert (row, col) indices to packed upper-triangle index.
 * Assumes row <= col (upper triangle).
 *
 * For 3D (row-major upper triangle):
 *   [c_00, c_01, c_02, c_11, c_12, c_22]
 *   (0,0)->0, (0,1)->1, (0,2)->2, (1,1)->3, (1,2)->4, (2,2)->5
 */
template <int DIM>
__device__ __forceinline__ constexpr int tri_index(int row, int col) {
    // Closed-form index into packed upper triangle (row-major order).
    // For DIM=3: [c00, c01, c02, c11, c12, c22]
    //   (0,0)->0, (0,1)->1, (0,2)->2, (1,1)->3, (1,2)->4, (2,2)->5
    //
    // Formula: row * DIM - row*(row+1)/2 + col
    // Equivalent: row * (2*DIM - row - 1) / 2 + col
    return row * (2 * DIM - row - 1) / 2 + col;
}

// =============================================================================
// MAHALANOBIS DISTANCE
// =============================================================================

/**
 * Compute Mahalanobis distance squared: D² = d^T @ Σ⁻¹ @ d
 *
 * @param d    Displacement vector (x - μ), length DIM
 * @param conic Packed upper-triangle of Σ⁻¹, length DIM*(DIM+1)/2
 * @return     Mahalanobis distance squared
 *
 * CRITICAL: Off-diagonal elements of conic contribute 2× due to symmetry.
 * The packed format stores each off-diagonal once, but it affects D² twice.
 */
template <int DIM>
__device__ __forceinline__ float mahalanobis_distance_sq(
    const float* __restrict__ d,
    const float* __restrict__ conic
) {
    float result = 0.0f;
    int idx = 0;

    // Compute d^T @ C @ d where C is symmetric
    // For packed upper triangle [c_00, c_01, c_02, ..., c_11, c_12, ..., c_22, ...]
    // D² = Σᵢ c_ii * d_i² + 2 * Σᵢ<ⱼ c_ij * d_i * d_j

    #pragma unroll
    for (int i = 0; i < DIM; i++) {
        // Diagonal: c_ii * d_i²
        result += d[i] * d[i] * conic[idx++];

        // Off-diagonals: 2 * c_ij * d_i * d_j for j > i
        #pragma unroll
        for (int j = i + 1; j < DIM; j++) {
            result += 2.0f * d[i] * d[j] * conic[idx++];
        }
    }

    return result;
}

// =============================================================================
// OPTIMIZED 2D/3D MAHALANOBIS DISTANCE (EXPLICIT FORMULAS)
// =============================================================================

/**
 * Hardcoded 2D Mahalanobis distance squared.
 *
 * Conic layout: [c00, c01, c11]
 *
 * D² = c00*d0² + 2*c01*d0*d1 + c11*d1²
 *
 * This explicit formula compiles to 5 FMA instructions, vs 6 loop iterations
 * with branch/counter overhead in the generic version.
 */
template <>
__device__ __forceinline__ float mahalanobis_distance_sq<2>(
    const float* __restrict__ d,
    const float* __restrict__ c
) {
    return c[0] * d[0] * d[0]
         + c[2] * d[1] * d[1]
         + 2.0f * c[1] * d[0] * d[1];
}

/**
 * Hardcoded 3D Mahalanobis distance squared.
 *
 * Conic layout: [c00, c01, c02, c11, c12, c22]
 *
 * D² = c00*d0² + c11*d1² + c22*d2²
 *    + 2*(c01*d0*d1 + c02*d0*d2 + c12*d1*d2)
 *
 * This explicit formula compiles to 9 FMA instructions, eliminating:
 * - Loop overhead (branch, counter, bounds check)
 * - Index computation
 * - Register pressure from loop variables
 *
 * Expected speedup: 20-30% in inner loops.
 */
template <>
__device__ __forceinline__ float mahalanobis_distance_sq<3>(
    const float* __restrict__ d,
    const float* __restrict__ c
) {
    // Diagonal contributions: c_ii * d_i²
    float diag = c[0] * d[0] * d[0]
               + c[3] * d[1] * d[1]
               + c[5] * d[2] * d[2];

    // Off-diagonal contributions: 2 * c_ij * d_i * d_j
    float off_diag = c[1] * d[0] * d[1]
                   + c[2] * d[0] * d[2]
                   + c[4] * d[1] * d[2];

    return diag + 2.0f * off_diag;
}

// =============================================================================
// GAUSSIAN INTENSITY COMPUTATION
// =============================================================================

/**
 * Compute standard Gaussian intensity.
 *
 * I(x) = a × exp(-0.5 × D²)
 *
 * where D² is Mahalanobis distance squared.
 *
 * OPTIMIZATION: Uses __expf() fast math intrinsic for ~15% speedup.
 * Slightly lower precision (~2 ULP vs 1 ULP) but acceptable for rendering.
 *
 * @param dist_sq   Mahalanobis distance squared (D²)
 * @param amplitude Amplitude (a)
 * @return          Gaussian intensity
 */
__device__ __forceinline__ float gaussian_intensity(
    float dist_sq,
    float amplitude
) {
    return amplitude * __expf(-0.5f * dist_sq);
}

// =============================================================================
// EFFECTIVE TRUNCATION RADIUS
// =============================================================================

/**
 * Compute effective truncation distance for standard Gaussian.
 *
 * radius = truncate * σ
 *
 * Additionally, account for amplitude-based culling:
 * Find t_max where a * exp(-0.5 * t²) = intensity_floor
 *   t_max = sqrt(2 * ln(a/intensity_floor))
 *
 * @param truncate        Base truncation radius (typically 3.0)
 * @param amplitude       Amplitude
 * @param intensity_floor Minimum intensity threshold
 * @return                Effective truncation in units of sqrt(eigenvalue)
 */
__device__ __forceinline__ float effective_truncation(
    float truncate,
    float amplitude,
    float intensity_floor
) {
    float t_base = truncate;

    // Amplitude-based truncation (where intensity drops below floor)
    float ratio = amplitude / fmaxf(intensity_floor, 1e-10f);
    float t_amp = 1e6f;  // Large default if amplitude check not needed
    if (ratio > 1.0f) {
        t_amp = sqrtf(2.0f * __logf(ratio));
    }

    // Use minimum of both truncations
    return fminf(t_base, t_amp);
}

/**
 * Compute effective truncation distance SQUARED for early rejection.
 *
 * OPTIMIZATION: This value can be precomputed per-splat during batch loading
 * into shared memory. The inner loop then only needs a simple comparison:
 *     if (dist_sq <= effective_truncate_sq) { ... }
 *
 * Combines two truncation criteria (taking the tighter bound):
 * 1. Base truncation: D² <= truncate²
 * 2. Amplitude-based: D² where intensity drops below intensity_floor
 *    Solving a * exp(-0.5 * D²) = floor => D² = 2*ln(a/floor)
 *
 * @param truncate        Base truncation radius (typically 3.0)
 * @param amplitude       Splat amplitude (for amplitude-based tightening)
 * @param intensity_floor Minimum intensity threshold
 * @return                Squared effective truncation distance (in Mahalanobis space)
 */
__device__ __forceinline__ float effective_truncate_sq(
    float truncate,
    float amplitude,
    float intensity_floor
) {
    float t_base_sq = truncate * truncate;

    // Amplitude-based tightening: find D² where intensity drops below floor
    // a * exp(-0.5 * D²) = floor => D² = 2*ln(a/floor)
    float ratio = amplitude / fmaxf(intensity_floor, 1e-10f);
    if (ratio <= 1.0f) {
        // amplitude <= intensity_floor: splat contributes nothing
        return 0.0f;
    }

    float t_amp_sq = 2.0f * __logf(ratio);
    return fminf(t_base_sq, t_amp_sq);
}

#endif // CUDA_SPLATTING_MATH_UTILS_CUH
