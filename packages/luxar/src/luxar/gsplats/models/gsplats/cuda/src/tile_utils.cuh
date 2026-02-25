/**
 * Tile-based Spatial Utilities for Gaussian Splatting
 *
 * This header provides tile/voxel management functions:
 * - AABB computation for spatial binning
 * - Tile index conversion (linear <-> coordinates)
 * - 3D grid launch optimization for CUDA (get_tile_info_2d/3d)
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
// PIXEL COORDINATE EXTRACTION HELPERS
// =============================================================================

/**
 * Convert local pixel index to integer voxel coordinates within a tile.
 *
 * For 2D/3D with standard tile sizes (16x16, 8x8x8), uses bitwise operations
 * for fast coordinate extraction when the tile is full (not an edge tile).
 * Falls back to generic division/modulo for edge tiles or higher dimensions.
 *
 * OPTIMIZATION: Bitwise ops are 1 cycle vs 20-40 cycles for integer division.
 * The `if constexpr` branches are resolved at compile time — no runtime cost.
 *
 * @param local_px_idx    Thread-local pixel index within the tile
 * @param tile_origin     Starting voxel coordinate of the tile
 * @param tile_extent     Tile extent in each dimension (may be clipped at volume edges)
 * @param use_fast_path_3d Whether 3D bitwise fast path is valid (full 8x8x8 tile)
 * @param use_fast_path_2d Whether 2D bitwise fast path is valid (full 16x16 tile)
 * @param voxel_coords    Output: integer voxel coordinates
 */
template <int DIM>
__device__ __forceinline__ void compute_voxel_coords(
    int local_px_idx,
    const int* __restrict__ tile_origin,
    const int* __restrict__ tile_extent,
    bool use_fast_path_3d,
    bool use_fast_path_2d,
    int* __restrict__ voxel_coords
) {
    if constexpr (DIM == 3) {
        if (use_fast_path_3d) {
            // Bitwise ops for 8x8x8 tiles: idx & 7, (idx >> 3) & 7, idx >> 6
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
            // Bitwise ops for 16x16 tiles: idx & 15, idx >> 4
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
        // Generic path for DIM > 3
        int remaining = local_px_idx;
        #pragma unroll
        for (int d = DIM - 1; d >= 0; d--) {
            voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
            remaining /= tile_extent[d];
        }
    }
}

/**
 * Convert local pixel index to float pixel coordinates within a tile.
 * Calls compute_voxel_coords then converts to float.
 * Used by backward kernel where float coordinates are needed for displacement computation.
 */
template <int DIM>
__device__ __forceinline__ void compute_pixel_coords_float(
    int local_px_idx,
    const int* __restrict__ tile_origin,
    const int* __restrict__ tile_extent,
    bool use_fast_path_3d,
    bool use_fast_path_2d,
    float* __restrict__ px
) {
    int voxel_coords[DIM];
    compute_voxel_coords<DIM>(local_px_idx, tile_origin, tile_extent,
                              use_fast_path_3d, use_fast_path_2d, voxel_coords);
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        px[d] = (float)voxel_coords[d];
    }
}

#endif // CUDA_SPLATTING_TILE_UTILS_CUH
