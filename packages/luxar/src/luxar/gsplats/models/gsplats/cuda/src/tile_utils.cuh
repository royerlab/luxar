/**
 * Tile-based Spatial Utilities for Gaussian Splatting
 *
 * This header provides tile/voxel management functions:
 * - AABB computation for spatial binning
 * - Tile index conversion (linear <-> coordinates)
 * - 3D grid launch optimization for CUDA
 * - Optimized bitwise indexing for power-of-2 tiles
 */

#ifndef CUDA_SPLATTING_TILE_UTILS_CUH
#define CUDA_SPLATTING_TILE_UTILS_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// Forward declaration - effective_truncation is defined in math_utils.cuh
// We include it here for AABB computation
#include "math_utils.cuh"

// =============================================================================
// AABB (AXIS-ALIGNED BOUNDING BOX) COMPUTATION
// =============================================================================

/**
 * AABB structure for spatial binning.
 * Stores inclusive tile indices [lo, hi] for each dimension.
 */
template <int DIM>
struct AABB {
    int lo[DIM];  // Inclusive lower bounds (tile indices)
    int hi[DIM];  // Inclusive upper bounds (tile indices)

    __device__ __forceinline__ bool is_empty() const {
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            if (lo[d] > hi[d]) return true;
        }
        return false;
    }

    __device__ __forceinline__ int num_tiles() const {
        if (is_empty()) return 0;
        int count = 1;
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            count *= (hi[d] - lo[d] + 1);
        }
        return count;
    }
};

/**
 * Compute AABB for a single splat.
 *
 * The bounding box is computed in tile coordinates and clipped to valid range.
 *
 * @param mu            Splat center in voxel coordinates, length DIM
 * @param L_row_norms   Row norms of Cholesky factor (approximates σ per axis), length DIM
 * @param sharpness     Sharpness parameter
 * @param amplitude     Amplitude
 * @param truncate      Base truncation radius
 * @param intensity_floor Minimum intensity threshold
 * @param tile_size     Voxels per tile in each dimension
 * @param tile_dims     Number of tiles in each dimension
 * @param shape         Volume shape in voxels
 * @return              AABB in tile coordinates
 */
template <int DIM>
__device__ AABB<DIM> compute_splat_aabb(
    const float* __restrict__ mu,
    const float* __restrict__ L_row_norms,
    float sharpness,
    float amplitude,
    float truncate,
    float intensity_floor,
    const int* __restrict__ tile_size,
    const int* __restrict__ tile_dims,
    const int* __restrict__ shape
) {
    AABB<DIM> aabb;

    // Compute effective truncation
    float t_eff = effective_truncation(truncate, sharpness, amplitude, intensity_floor);

    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        // Radius in voxels = t_eff * sqrt(σ²_d) ≈ t_eff * L_row_norm_d
        float radius = t_eff * L_row_norms[d];

        // Voxel bounds
        float lo_voxel = mu[d] - radius;
        float hi_voxel = mu[d] + radius;

        // Clip to valid voxel range [0, shape-1]
        lo_voxel = fmaxf(lo_voxel, 0.0f);
        hi_voxel = fminf(hi_voxel, (float)(shape[d] - 1));

        // Convert to tile indices
        aabb.lo[d] = (int)floorf(lo_voxel / (float)tile_size[d]);
        aabb.hi[d] = (int)floorf(hi_voxel / (float)tile_size[d]);

        // Clip to valid tile range [0, tile_dims-1]
        aabb.lo[d] = max(aabb.lo[d], 0);
        aabb.hi[d] = min(aabb.hi[d], tile_dims[d] - 1);
    }

    return aabb;
}

/**
 * Compute L_row_norms from full Cholesky factor L.
 *
 * L_row_norm[i] = sqrt(sum_j L[i,j]²) ≈ sqrt(σ²_i)
 *
 * This approximates the standard deviation along each axis for AABB computation.
 */
template <int DIM>
__device__ __forceinline__ void compute_L_row_norms(
    const float* __restrict__ L,  // (DIM, DIM) in row-major
    float* __restrict__ L_row_norms
) {
    #pragma unroll
    for (int i = 0; i < DIM; i++) {
        float norm_sq = 0.0f;
        #pragma unroll
        for (int j = 0; j <= i; j++) {  // L is lower-triangular
            float val = L[i * DIM + j];
            norm_sq += val * val;
        }
        L_row_norms[i] = sqrtf(norm_sq);
    }
}

// =============================================================================
// TILE INDEX CONVERSION
// =============================================================================

/**
 * Convert tile coordinates to linear tile index.
 *
 * Uses row-major ordering: tile_idx = Σ coords[d] * stride[d]
 * where stride[d] = Π tile_dims[d+1:DIM]
 */
template <int DIM>
__device__ __forceinline__ int tile_coords_to_linear(
    const int* __restrict__ coords,
    const int* __restrict__ tile_dims
) {
    int idx = 0;
    int stride = 1;

    // Compute from last dimension to first
    #pragma unroll
    for (int d = DIM - 1; d >= 0; d--) {
        idx += coords[d] * stride;
        stride *= tile_dims[d];
    }

    return idx;
}

/**
 * Convert linear tile index to tile coordinates.
 */
template <int DIM>
__device__ __forceinline__ void linear_to_tile_coords(
    int linear_idx,
    const int* __restrict__ tile_dims,
    int* __restrict__ coords
) {
    #pragma unroll
    for (int d = DIM - 1; d >= 0; d--) {
        coords[d] = linear_idx % tile_dims[d];
        linear_idx /= tile_dims[d];
    }
}

// =============================================================================
// 3D GRID LAUNCH OPTIMIZATION
// =============================================================================

/**
 * Extract tile coordinates directly from blockIdx for 3D grid launch.
 *
 * When using dim3 grid launch (2D/3D dimensions), we can directly use
 * blockIdx.x/y/z as tile coordinates, eliminating expensive division
 * and modulo operations.
 *
 * For 3D: grid = dim3(tile_dims[2], tile_dims[1], tile_dims[0])
 *   blockIdx.x = tile_z (fastest), blockIdx.y = tile_y, blockIdx.z = tile_x (slowest)
 *
 * For 2D: grid = dim3(tile_dims[1], tile_dims[0], 1)
 *   blockIdx.x = tile_y (fastest), blockIdx.y = tile_x (slowest)
 *
 * Performance benefit: Division is 20-40 cycles, blockIdx read is 1 cycle.
 */

/**
 * Get tile coordinates and linear index from blockIdx for 3D volumes.
 * Uses direct blockIdx read instead of division/modulo.
 *
 * Grid launch order: dim3(tile_dims[2], tile_dims[1], tile_dims[0])
 *   blockIdx.x = tile_coords[2] (z, varies fastest)
 *   blockIdx.y = tile_coords[1] (y)
 *   blockIdx.z = tile_coords[0] (x, varies slowest)
 *
 * This matches the linear index formula from linear_to_tile_coords:
 *   tile_idx = coords[2] + coords[1]*tile_dims[2] + coords[0]*tile_dims[2]*tile_dims[1]
 */
__device__ __forceinline__ void get_tile_info_3d(
    const int* __restrict__ tile_dims,
    int* __restrict__ tile_coords,
    int& tile_idx
) {
    // Map blockIdx to tile coordinates (last dimension varies fastest)
    tile_coords[0] = blockIdx.z;  // x (slowest varying)
    tile_coords[1] = blockIdx.y;  // y
    tile_coords[2] = blockIdx.x;  // z (fastest varying)

    // Compute linear index matching linear_to_tile_coords formula
    tile_idx = blockIdx.x                              // coords[2]
             + blockIdx.y * tile_dims[2]               // coords[1] * tile_dims[2]
             + blockIdx.z * tile_dims[2] * tile_dims[1]; // coords[0] * tile_dims[2] * tile_dims[1]
}

/**
 * Get tile coordinates and linear index from blockIdx for 2D volumes.
 *
 * Grid launch order: dim3(tile_dims[1], tile_dims[0], 1)
 *   blockIdx.x = tile_coords[1] (y, varies fastest)
 *   blockIdx.y = tile_coords[0] (x, varies slowest)
 */
__device__ __forceinline__ void get_tile_info_2d(
    const int* __restrict__ tile_dims,
    int* __restrict__ tile_coords,
    int& tile_idx
) {
    tile_coords[0] = blockIdx.y;  // x (slowest varying)
    tile_coords[1] = blockIdx.x;  // y (fastest varying)

    tile_idx = blockIdx.x + blockIdx.y * tile_dims[1];
}

/**
 * Generic tile info extraction (falls back to linear_to_tile_coords).
 * Used for dimensions > 3 where we can't use 3D CUDA grid.
 */
template <int DIM>
__device__ __forceinline__ void get_tile_info_generic(
    const int* __restrict__ tile_dims,
    int* __restrict__ tile_coords,
    int& tile_idx
) {
    tile_idx = blockIdx.x;
    linear_to_tile_coords<DIM>(tile_idx, tile_dims, tile_coords);
}

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

// =============================================================================
// OPTIMIZED BITWISE INDEXING FOR POWER-OF-2 TILES
// =============================================================================

/**
 * Tile size constants for bitwise operations.
 *
 * For power-of-2 tile sizes, we can replace expensive integer division
 * and modulo with fast bitwise AND and shift operations:
 *   x % tile_size  =>  x & (tile_size - 1)
 *   x / tile_size  =>  x >> log2(tile_size)
 *
 * This provides ~10-15% speedup in the inner loop.
 */
constexpr int TILE_SIZE_2D = 16;   // 16×16 = 256 pixels
constexpr int TILE_SIZE_3D = 8;    // 8×8×8 = 512 pixels
constexpr int TILE_SIZE_4D = 4;    // 4×4×4×4 = 256 pixels

constexpr int TILE_BITS_2D = 4;    // log2(16)
constexpr int TILE_BITS_3D = 3;    // log2(8)
constexpr int TILE_BITS_4D = 2;    // log2(4)

constexpr int TILE_MASK_2D = 15;   // 16 - 1
constexpr int TILE_MASK_3D = 7;    // 8 - 1
constexpr int TILE_MASK_4D = 3;    // 4 - 1

/**
 * Optimized 3D: Convert local pixel index to tile-relative coordinates.
 *
 * For 8×8×8 tiles (512 pixels), local_idx in [0, 511]:
 *   local_z = local_idx & 0x7          (% 8)
 *   local_y = (local_idx >> 3) & 0x7   (/ 8 % 8)
 *   local_x = local_idx >> 6           (/ 64)
 *
 * Integer division is 20-40 cycles on GPU; bitwise ops are 1 cycle.
 */
__device__ __forceinline__ void local_idx_to_coords_3d_fast(
    int local_idx,
    int& local_x,
    int& local_y,
    int& local_z
) {
    local_z = local_idx & TILE_MASK_3D;
    local_y = (local_idx >> TILE_BITS_3D) & TILE_MASK_3D;
    local_x = local_idx >> (2 * TILE_BITS_3D);
}

/**
 * Optimized 3D: Convert tile-relative coordinates to local pixel index.
 */
__device__ __forceinline__ int coords_to_local_idx_3d_fast(
    int local_x,
    int local_y,
    int local_z
) {
    return (local_x << (2 * TILE_BITS_3D)) | (local_y << TILE_BITS_3D) | local_z;
}

/**
 * Optimized 2D: Convert local pixel index to tile-relative coordinates.
 *
 * For 16×16 tiles (256 pixels), local_idx in [0, 255]:
 *   local_y = local_idx & 0xF          (% 16)
 *   local_x = local_idx >> 4           (/ 16)
 */
__device__ __forceinline__ void local_idx_to_coords_2d_fast(
    int local_idx,
    int& local_x,
    int& local_y
) {
    local_y = local_idx & TILE_MASK_2D;
    local_x = local_idx >> TILE_BITS_2D;
}

/**
 * Optimized 2D: Convert tile-relative coordinates to local pixel index.
 */
__device__ __forceinline__ int coords_to_local_idx_2d_fast(
    int local_x,
    int local_y
) {
    return (local_x << TILE_BITS_2D) | local_y;
}

/**
 * Optimized 4D: Convert local pixel index to tile-relative coordinates.
 *
 * For 4×4×4×4 tiles (256 pixels), local_idx in [0, 255]:
 *   local_w = local_idx & 0x3               (% 4)
 *   local_z = (local_idx >> 2) & 0x3        (/ 4 % 4)
 *   local_y = (local_idx >> 4) & 0x3        (/ 16 % 4)
 *   local_x = local_idx >> 6                (/ 64)
 */
__device__ __forceinline__ void local_idx_to_coords_4d_fast(
    int local_idx,
    int& local_x,
    int& local_y,
    int& local_z,
    int& local_w
) {
    local_w = local_idx & TILE_MASK_4D;
    local_z = (local_idx >> TILE_BITS_4D) & TILE_MASK_4D;
    local_y = (local_idx >> (2 * TILE_BITS_4D)) & TILE_MASK_4D;
    local_x = local_idx >> (3 * TILE_BITS_4D);
}

/**
 * Optimized 3D: Get voxel coordinates from tile origin and local index.
 *
 * Combines tile origin lookup with fast bitwise local coordinate extraction.
 */
__device__ __forceinline__ void get_voxel_coords_3d_fast(
    int local_idx,
    const int* __restrict__ tile_origin,
    int* __restrict__ voxel_coords
) {
    int local_x, local_y, local_z;
    local_idx_to_coords_3d_fast(local_idx, local_x, local_y, local_z);
    voxel_coords[0] = tile_origin[0] + local_x;
    voxel_coords[1] = tile_origin[1] + local_y;
    voxel_coords[2] = tile_origin[2] + local_z;
}

/**
 * Check if tile size is a power of 2 for optimization eligibility.
 */
__host__ __device__ __forceinline__ constexpr bool is_power_of_2(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

#endif // CUDA_SPLATTING_TILE_UTILS_CUH
