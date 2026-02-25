/**
 * CUDA Device Utilities for Gaussian Splatting - Umbrella Header
 *
 * This is the main header that includes all sub-headers. Include this file
 * to get access to all utility functions.
 *
 * Sub-headers:
 * - dtype_traits.cuh: FP16 support and vectorized load helpers (DTypeTraits)
 * - math_utils.cuh: Triangular matrix indexing, Mahalanobis distance, Gaussian intensity
 * - tile_utils.cuh: AABB computation, tile indexing, grid optimization
 * - reduction_utils.cuh: Warp-level reduction, gradient helpers, backward pass implementations
 */

#ifndef CUDA_SPLATTING_UTILS_CUH
#define CUDA_SPLATTING_UTILS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

// =============================================================================
// CONFIGURATION CONSTANTS
// =============================================================================

// Maximum supported dimensions
constexpr int MAX_DIM = 8;

// Maximum splats that can be handled in a single tile batch
constexpr int MAX_SPLATS_PER_BATCH = 256;

// Error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA kernel error at %s:%d - %s\n", __FILE__,     \
                    __LINE__, cudaGetErrorString(err));                        \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

// =============================================================================
// SUB-HEADER INCLUDES
// =============================================================================

// FP16 support and vectorized load helpers (must be first - others may depend on DTypeTraits)
#include "dtype_traits.cuh"

// Mathematical utilities (triangular matrix, Mahalanobis distance, Gaussian intensity)
#include "math_utils.cuh"

// Tile-based spatial utilities (AABB, tile indexing, grid optimization)
#include "tile_utils.cuh"

// Reduction and gradient utilities (warp reduction, backward pass helpers)
#include "reduction_utils.cuh"

#endif // CUDA_SPLATTING_UTILS_CUH
