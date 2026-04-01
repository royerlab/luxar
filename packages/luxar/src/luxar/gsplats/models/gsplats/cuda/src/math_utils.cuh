/**
 * Mathematical Utilities for Gaussian Splatting
 *
 * This header provides mathematical functions for Gaussian computation:
 * - Triangular matrix indexing for packed covariance matrices
 * - Mahalanobis distance calculation (generic + optimized 2D/3D)
 * - Shifted Gaussian intensity with C⁰ continuous truncation
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
// SHIFTED GAUSSIAN TRUNCATION PARAMETERS
// =============================================================================

/**
 * Precomputed constants for shifted Gaussian truncation.
 *
 * The shifted Gaussian ensures C⁰ continuity at the truncation boundary:
 *
 *   I(x) = a · scale · max(0, exp(-0.5·D²) - C)
 *
 * where C = exp(-0.5·T²) is the boundary value, and scale = 1/(1-C)
 * preserves peak amplitude (I(0) = a).
 *
 * For the default T=3: C ≈ 0.01111, scale ≈ 1.01123.
 */
struct GaussianShiftParams {
    float shift_C;          // exp(-0.5 * T²) — boundary value to subtract
    float inv_one_minus_C;  // 1/(1-C) — peak-preserving rescale factor
};

/**
 * Compute shift parameters from truncation radius.
 * Call once per kernel launch (not per pixel).
 */
__device__ __forceinline__ GaussianShiftParams
compute_shift_params(float truncate) {
    float C = __expf(-0.5f * truncate * truncate);
    return {C, 1.0f / (1.0f - C)};
}

// =============================================================================
// GAUSSIAN INTENSITY COMPUTATION
// =============================================================================

/**
 * Compute shifted Gaussian intensity with C⁰ continuous truncation.
 *
 * I(x) = a · scale · max(0, exp(-0.5·D²) - C)
 *
 * where:
 *   C     = exp(-0.5·T²) — boundary value (shifts Gaussian to zero at cutoff)
 *   scale = 1/(1-C)      — rescales so I(0) = a (peak amplitude preserved)
 *   D²   = Mahalanobis distance squared
 *
 * This eliminates the discontinuity at D² = T² that the unshifted formula
 * a·exp(-0.5·D²) would produce when hard-truncated.
 *
 * OPTIMIZATION: Uses __expf() fast math intrinsic for ~15% speedup.
 * shift_C and inv_one_minus_C should be precomputed via compute_shift_params().
 *
 * @param dist_sq          Mahalanobis distance squared (D²)
 * @param amplitude        Amplitude (a)
 * @param shift_C          Precomputed exp(-0.5 * truncate²)
 * @param inv_one_minus_C  Precomputed 1/(1 - shift_C)
 * @return                 Shifted Gaussian intensity
 */
__device__ __forceinline__ float gaussian_intensity(
    float dist_sq,
    float amplitude,
    float shift_C,
    float inv_one_minus_C
) {
    return amplitude * inv_one_minus_C *
           fmaxf(__expf(-0.5f * dist_sq) - shift_C, 0.0f);
}

// =============================================================================
// EFFECTIVE TRUNCATION RADIUS
// =============================================================================

/**
 * Compute effective truncation distance for shifted Gaussian.
 *
 * Finds D where a·scale·(exp(-0.5·D²) - C) = intensity_floor, then
 * takes min(truncate, D) to get the tighter bound.
 *
 * Solving:  exp(-0.5·D²) = floor/(a·scale) + C
 *       →  D = sqrt(-2·ln(floor/(a·scale) + C))
 *
 * @param truncate        Base truncation radius (typically 3.0)
 * @param amplitude       Amplitude
 * @param intensity_floor Minimum intensity threshold
 * @param shift_C         Precomputed exp(-0.5 * truncate²)
 * @param inv_one_minus_C Precomputed 1/(1 - shift_C)
 * @return                Effective truncation in units of sqrt(eigenvalue)
 */
__device__ __forceinline__ float effective_truncation(
    float truncate,
    float amplitude,
    float intensity_floor,
    float shift_C,
    float inv_one_minus_C
) {
    float t_base = truncate;

    // Amplitude-based truncation: find D where shifted intensity = floor
    // a·scale·(exp(-0.5·D²) - C) = floor
    // exp(-0.5·D²) = floor/(a·scale) + C
    float threshold = intensity_floor / fmaxf(amplitude * inv_one_minus_C, 1e-10f) + shift_C;
    if (threshold >= 1.0f) {
        // Splat amplitude too low — entirely below floor
        return 0.0f;
    }

    float t_amp = sqrtf(-2.0f * __logf(threshold));
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
 * 2. Amplitude-based: D² where shifted intensity drops below intensity_floor
 *    Solving a·scale·(exp(-0.5·D²) - C) = floor
 *        →  D² = -2·ln(floor/(a·scale) + C)
 *
 * @param truncate        Base truncation radius (typically 3.0)
 * @param amplitude       Splat amplitude (for amplitude-based tightening)
 * @param intensity_floor Minimum intensity threshold
 * @param shift_C         Precomputed exp(-0.5 * truncate²)
 * @param inv_one_minus_C Precomputed 1/(1 - shift_C)
 * @return                Squared effective truncation distance (in Mahalanobis space)
 */
__device__ __forceinline__ float effective_truncate_sq(
    float truncate,
    float amplitude,
    float intensity_floor,
    float shift_C,
    float inv_one_minus_C
) {
    float t_base_sq = truncate * truncate;

    // Amplitude-based tightening for shifted Gaussian
    float threshold = intensity_floor / fmaxf(amplitude * inv_one_minus_C, 1e-10f) + shift_C;
    if (threshold >= 1.0f) {
        // amplitude too low: splat contributes nothing
        return 0.0f;
    }

    float t_amp_sq = -2.0f * __logf(threshold);
    return fminf(t_base_sq, t_amp_sq);
}

#endif // CUDA_SPLATTING_MATH_UTILS_CUH
