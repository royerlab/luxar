/**
 * Voxel Utilities for Gaussian Splatting
 *
 * This header provides voxel coordinate conversion functions
 * used by the splat-centric CUDA kernels.
 */

#ifndef CUDA_SPLATTING_VOXEL_UTILS_CUH
#define CUDA_SPLATTING_VOXEL_UTILS_CUH

// NVRTC has no toolkit/std headers: rtc_compat.cuh supplies what the kernels use.
#ifdef __CUDACC_RTC__
#include "rtc_compat.cuh"
#else
#include <cuda_runtime.h>
#include <cstdint>
#endif

/**
 * Convert voxel coordinates to linear pixel index.
 */
template <int DIM>
__device__ __forceinline__ int64_t voxel_to_linear(
    const int* __restrict__ voxel_coords,
    const int* __restrict__ shape
) {
    int64_t idx = 0;
    int64_t stride = 1;

    #pragma unroll
    for (int d = DIM - 1; d >= 0; d--) {
        idx += (int64_t)voxel_coords[d] * stride;
        stride *= (int64_t)shape[d];
    }

    return idx;
}

#endif // CUDA_SPLATTING_VOXEL_UTILS_CUH
