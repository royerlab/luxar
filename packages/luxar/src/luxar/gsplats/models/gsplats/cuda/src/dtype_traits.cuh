/**
 * FP16 (Half Precision) Support and Vectorized Load Helpers
 *
 * This header provides type traits and optimized load functions for
 * mixed-precision computation (FP16 storage, FP32 compute).
 *
 * Features:
 * - DTypeTraits templates for dtype-agnostic kernel code
 * - Vectorized load helpers for bandwidth-optimized FP16 access
 * - 2D/3D specialized loaders for centers and conics
 */

#ifndef CUDA_SPLATTING_DTYPE_TRAITS_CUH
#define CUDA_SPLATTING_DTYPE_TRAITS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cassert>

// =============================================================================
// FP16 (HALF PRECISION) SUPPORT
// =============================================================================

/**
 * Type traits for dtype-agnostic kernel code.
 *
 * Enables writing kernels that work with both float32 and float16 inputs
 * using a single code path. The load() function converts to float32 for
 * computation while allowing FP16 storage for bandwidth savings.
 *
 * Usage:
 *   float val = DTypeTraits<InputDType>::load(ptr, idx);
 */
template <typename T>
struct DTypeTraits;

/**
 * Float32 type traits - direct load, no conversion needed.
 */
template <>
struct DTypeTraits<float> {
    static constexpr bool is_fp16 = false;

    /**
     * Load float32 value using texture cache hint.
     */
    __device__ __forceinline__ static float load(const float* ptr, int idx) {
        return __ldg(&ptr[idx]);
    }

    /**
     * Vectorized load of 2 consecutive float32 values.
     */
    __device__ __forceinline__ static void load2(const float* ptr, int idx, float& a, float& b) {
        a = __ldg(&ptr[idx]);
        b = __ldg(&ptr[idx + 1]);
    }
};

/**
 * Float16 type traits - load and convert to float32 for computation.
 *
 * The mixed precision strategy (store FP16, compute FP32) provides:
 * - ~2x memory bandwidth improvement from FP16 storage
 * - Full FP32 numerical precision for computation
 * - No loss scaling needed for backward pass
 */
template <>
struct DTypeTraits<__half> {
    static constexpr bool is_fp16 = true;

    /**
     * Load FP16 value and convert to FP32 for computation.
     * Uses texture cache hint for better memory throughput.
     */
    __device__ __forceinline__ static float load(const __half* ptr, int idx) {
        return __half2float(__ldg(&ptr[idx]));
    }

    /**
     * Vectorized load of 2 consecutive FP16 values as FP32.
     * Uses __half2 for efficient 32-bit aligned load.
     *
     * IMPORTANT: idx must be even (4-byte aligned for __half2).
     * Callers must verify alignment before calling this function.
     * Unaligned access causes undefined behavior on CUDA.
     */
    __device__ __forceinline__ static void load2(const __half* ptr, int idx, float& a, float& b) {
        // Aligned load of 2 half values (4 bytes total)
        // Assert alignment in debug builds
        assert((idx & 1) == 0 && "load2 requires even index for __half2 alignment");
        __half2 h2 = *reinterpret_cast<const __half2*>(&ptr[idx]);
        a = __low2float(h2);
        b = __high2float(h2);
    }
};

/**
 * Helper to load a batch of values from FP16 or FP32 source to FP32 destination.
 *
 * @tparam InputDType Source data type (__half or float)
 * @param src         Source pointer
 * @param src_idx     Index in source array
 * @param dst         Destination float pointer
 * @param count       Number of values to load
 */
template <typename InputDType>
__device__ __forceinline__ void load_batch_to_float(
    const InputDType* __restrict__ src,
    int src_idx,
    float* __restrict__ dst,
    int count
) {
    #pragma unroll
    for (int i = 0; i < count; i++) {
        dst[i] = DTypeTraits<InputDType>::load(src, src_idx + i);
    }
}

// =============================================================================
// VECTORIZED LOAD HELPERS FOR FP16 OPTIMIZATION
// =============================================================================

/**
 * Load 3D center coordinates using vectorized loads.
 *
 * For FP16: Uses load2() for first 2 values (single 32-bit aligned read),
 *           then load() for 3rd value. Reduces memory transactions from 3 to 2.
 *
 * For FP32: Falls back to individual loads (already optimal for coalescing).
 *
 * @tparam InputDType Source data type (__half or float)
 * @param centers     Source centers array
 * @param splat_idx   Index of splat (0-based)
 * @param out         Output array, length 3
 */
template <typename InputDType>
__device__ __forceinline__ void load_centers_3d(
    const InputDType* __restrict__ centers,
    int splat_idx,
    float* __restrict__ out
) {
    int base = splat_idx * 3;
    if constexpr (DTypeTraits<InputDType>::is_fp16) {
        // Vectorized load: 2 + 1 pattern for 3 values
        // ONLY when base index is even (4-byte aligned for __half2)
        if ((base & 1) == 0) {
            DTypeTraits<InputDType>::load2(centers, base, out[0], out[1]);
            out[2] = DTypeTraits<InputDType>::load(centers, base + 2);
        } else {
            // Odd base index - unaligned, use scalar loads
            out[0] = DTypeTraits<InputDType>::load(centers, base);
            out[1] = DTypeTraits<InputDType>::load(centers, base + 1);
            out[2] = DTypeTraits<InputDType>::load(centers, base + 2);
        }
    } else {
        // FP32: individual loads with __ldg
        out[0] = DTypeTraits<InputDType>::load(centers, base);
        out[1] = DTypeTraits<InputDType>::load(centers, base + 1);
        out[2] = DTypeTraits<InputDType>::load(centers, base + 2);
    }
}

/**
 * Load 2D center coordinates using vectorized loads.
 *
 * For FP16: Uses load2() for both values (single 32-bit aligned read).
 * For FP32: Falls back to individual loads.
 *
 * @tparam InputDType Source data type (__half or float)
 * @param centers     Source centers array
 * @param splat_idx   Index of splat (0-based)
 * @param out         Output array, length 2
 */
template <typename InputDType>
__device__ __forceinline__ void load_centers_2d(
    const InputDType* __restrict__ centers,
    int splat_idx,
    float* __restrict__ out
) {
    if constexpr (DTypeTraits<InputDType>::is_fp16) {
        // Vectorized load: single load2() for 2 values
        DTypeTraits<InputDType>::load2(centers, splat_idx * 2, out[0], out[1]);
    } else {
        int base = splat_idx * 2;
        out[0] = DTypeTraits<InputDType>::load(centers, base);
        out[1] = DTypeTraits<InputDType>::load(centers, base + 1);
    }
}

/**
 * Load 3D conic (packed upper triangle, 6 elements) using vectorized loads.
 *
 * For FP16: Uses 3x load2() for 6 values. Reduces memory transactions from 6 to 3.
 * For FP32: Falls back to individual loads.
 *
 * Conic layout: [c00, c01, c02, c11, c12, c22]
 *
 * @tparam InputDType Source data type (__half or float)
 * @param conic       Source conic array
 * @param splat_idx   Index of splat (0-based)
 * @param out         Output array, length 6
 */
template <typename InputDType>
__device__ __forceinline__ void load_conic_3d(
    const InputDType* __restrict__ conic,
    int splat_idx,
    float* __restrict__ out
) {
    if constexpr (DTypeTraits<InputDType>::is_fp16) {
        // Vectorized load: 3x load2() for 6 values
        int base = splat_idx * 6;
        DTypeTraits<InputDType>::load2(conic, base + 0, out[0], out[1]);
        DTypeTraits<InputDType>::load2(conic, base + 2, out[2], out[3]);
        DTypeTraits<InputDType>::load2(conic, base + 4, out[4], out[5]);
    } else {
        int base = splat_idx * 6;
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            out[i] = DTypeTraits<InputDType>::load(conic, base + i);
        }
    }
}

/**
 * Load 2D conic (packed upper triangle, 3 elements) using vectorized loads.
 *
 * For FP16: Uses load2() + load() pattern (2 + 1).
 * For FP32: Falls back to individual loads.
 *
 * Conic layout: [c00, c01, c11]
 *
 * @tparam InputDType Source data type (__half or float)
 * @param conic       Source conic array
 * @param splat_idx   Index of splat (0-based)
 * @param out         Output array, length 3
 */
template <typename InputDType>
__device__ __forceinline__ void load_conic_2d(
    const InputDType* __restrict__ conic,
    int splat_idx,
    float* __restrict__ out
) {
    int base = splat_idx * 3;
    if constexpr (DTypeTraits<InputDType>::is_fp16) {
        // Vectorized load: 2 + 1 pattern for 3 values
        // ONLY when base index is even (4-byte aligned for __half2)
        if ((base & 1) == 0) {
            DTypeTraits<InputDType>::load2(conic, base, out[0], out[1]);
            out[2] = DTypeTraits<InputDType>::load(conic, base + 2);
        } else {
            // Odd base index - unaligned, use scalar loads
            out[0] = DTypeTraits<InputDType>::load(conic, base);
            out[1] = DTypeTraits<InputDType>::load(conic, base + 1);
            out[2] = DTypeTraits<InputDType>::load(conic, base + 2);
        }
    } else {
        out[0] = DTypeTraits<InputDType>::load(conic, base);
        out[1] = DTypeTraits<InputDType>::load(conic, base + 1);
        out[2] = DTypeTraits<InputDType>::load(conic, base + 2);
    }
}

/**
 * Load amplitude and sharpness using vectorized load.
 *
 * For FP16: Uses load2() for both values (single 32-bit aligned read).
 * For FP32: Falls back to individual loads.
 *
 * Note: Assumes amps and sharpness are stored contiguously per-splat.
 * If stored separately, use individual load() calls instead.
 *
 * @tparam InputDType Source data type (__half or float)
 * @param amps        Amplitude array
 * @param sharpness   Sharpness array
 * @param splat_idx   Index of splat (0-based)
 * @param out_amp     Output amplitude
 * @param out_sharp   Output sharpness
 */
template <typename InputDType>
__device__ __forceinline__ void load_amp_sharpness(
    const InputDType* __restrict__ amps,
    const InputDType* __restrict__ sharpness,
    int splat_idx,
    float& out_amp,
    float& out_sharp
) {
    // Amps and sharpness are separate arrays, so load individually
    out_amp = DTypeTraits<InputDType>::load(amps, splat_idx);
    out_sharp = DTypeTraits<InputDType>::load(sharpness, splat_idx);
}

#endif // CUDA_SPLATTING_DTYPE_TRAITS_CUH
