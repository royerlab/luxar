/**
 * CUDA Kernel Launch Wrappers
 *
 * This header contains launch wrapper functions for all CUDA kernels.
 * These wrappers handle:
 * - Block/grid size configuration
 * - Shared memory allocation
 * - 2D/3D grid optimization for spatial locality
 *
 * Launch wrappers:
 * - launch_preprocess: Preprocess kernel (AABB and tile counting)
 * - launch_bin: Binning kernel (splat-to-tile assignment)
 * - launch_rasterize_forward: Forward rasterization
 * - launch_rasterize_backward: Backward rasterization
 * - launch_rasterize_global_forward: Global splat forward
 * - launch_rasterize_global_backward: Global splat backward
 */

#ifndef CUDA_SPLATTING_KERNEL_LAUNCHERS_CUH
#define CUDA_SPLATTING_KERNEL_LAUNCHERS_CUH

#include "utils.cuh"
#include "kernels_core.cuh"
#include "kernels_global.cuh"

#include <vector>

// =============================================================================
// CONFIGURATION
// =============================================================================

// Block sizes for different kernels
constexpr int PREPROCESS_BLOCK_SIZE = 256;
constexpr int BIN_BLOCK_SIZE = 256;
constexpr int RASTER_BLOCK_SIZE_2D = 256;  // 16x16 tile
constexpr int RASTER_BLOCK_SIZE_3D = 512;  // 8x8x8 tile
constexpr int RASTER_BLOCK_SIZE_DEFAULT = 256;
constexpr int GLOBAL_BLOCK_SIZE = 256;

/**
 * Compile-time block size for rasterization kernels based on dimension.
 * Used by __launch_bounds__ to inform the compiler of the actual max threads,
 * allowing better register allocation (avoid spills from assuming 1024 threads).
 */
template <int DIM>
constexpr int raster_block_size() {
    if constexpr (DIM == 3) return RASTER_BLOCK_SIZE_3D;
    else if constexpr (DIM == 2) return RASTER_BLOCK_SIZE_2D;
    else return RASTER_BLOCK_SIZE_DEFAULT;
}

// =============================================================================
// PREPROCESS KERNEL LAUNCHER
// =============================================================================

template <int DIM, typename InputDType = float>
void launch_preprocess(
    const InputDType* centers,
    const InputDType* amps,
    const InputDType* L_row_norms,
    int N,
    const int* shape,
    const int* tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    int* tile_counts,
    bool* global_flags,
    int* aabb_lo,
    int* aabb_hi,
    int64_t num_tiles,
    int* global_count,
    cudaStream_t stream
) {
    int block_size = PREPROCESS_BLOCK_SIZE;
    int num_blocks = (N + block_size - 1) / block_size;

    preprocess_kernel<DIM, InputDType><<<num_blocks, block_size, 0, stream>>>(
        centers, amps, L_row_norms, N,
        shape, tile_dims, tile_size, truncate, intensity_floor,
        tile_counts, global_flags, aabb_lo, aabb_hi, num_tiles, global_count
    );
}

// =============================================================================
// BINNING KERNEL LAUNCHER
// =============================================================================

template <int DIM>
void launch_bin(
    const int* aabb_lo,
    const int* aabb_hi,
    const bool* global_flags,
    int N,
    const int* tile_dims,
    const int64_t* tile_offsets,
    int* tile_write_heads,
    int* tile_content,
    int64_t num_tiles,
    cudaStream_t stream
) {
    int block_size = BIN_BLOCK_SIZE;
    int num_blocks = (N + block_size - 1) / block_size;

    bin_kernel<DIM><<<num_blocks, block_size, 0, stream>>>(
        aabb_lo, aabb_hi, global_flags, N,
        tile_dims, tile_offsets, tile_write_heads, tile_content, num_tiles
    );
}

// =============================================================================
// FORWARD RASTERIZATION KERNEL LAUNCHER
// =============================================================================

template <int DIM, int BATCH_SIZE = DEFAULT_BATCH_SIZE, typename InputDType = float>
void launch_rasterize_forward(
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    int N,
    const int* shape,
    const int* tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* tile_offsets,
    const int* tile_counts,
    const int* tile_content,
    float* output,
    int64_t num_tiles,
    const std::vector<int>& host_tile_dims,
    cudaStream_t stream
) {
    // Block size depends on dimension
    int block_size;
    if constexpr (DIM == 2) {
        block_size = RASTER_BLOCK_SIZE_2D;
    } else if constexpr (DIM == 3) {
        block_size = RASTER_BLOCK_SIZE_3D;
    } else {
        block_size = RASTER_BLOCK_SIZE_DEFAULT;
    }

    // Shared memory is statically allocated in the kernel using __shared__ arrays.
    // No dynamic shared memory needed (smem_size = 0).
    // Static allocation allows compile-time optimization and bank conflict avoidance.
    constexpr size_t smem_size = 0;

    // OPTIMIZATION: Use 3D grid for 2D/3D volumes
    // This improves L2 cache locality and eliminates division/modulo in tile coordinate extraction
    // Grid order: last dimension varies fastest to match linear index formula
    if constexpr (DIM == 3) {
        // Grid: (z, y, x) so blockIdx.x=z, blockIdx.y=y, blockIdx.z=x
        dim3 grid(host_tile_dims[2], host_tile_dims[1], host_tile_dims[0]);
        rasterize_forward_kernel<DIM, BATCH_SIZE, InputDType><<<grid, block_size, smem_size, stream>>>(
            centers, conic, amps, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content, output
        );
    } else if constexpr (DIM == 2) {
        // Grid: (y, x, 1) so blockIdx.x=y, blockIdx.y=x
        dim3 grid(host_tile_dims[1], host_tile_dims[0], 1);
        rasterize_forward_kernel<DIM, BATCH_SIZE, InputDType><<<grid, block_size, smem_size, stream>>>(
            centers, conic, amps, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content, output
        );
    } else {
        // For DIM > 3, use 1D grid
        int num_blocks = (int)num_tiles;
        rasterize_forward_kernel<DIM, BATCH_SIZE, InputDType><<<num_blocks, block_size, smem_size, stream>>>(
            centers, conic, amps, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content, output
        );
    }
}

// =============================================================================
// BACKWARD RASTERIZATION KERNEL LAUNCHER
// =============================================================================

template <int DIM, int BATCH_SIZE = DEFAULT_BATCH_SIZE, typename InputDType = float>
void launch_rasterize_backward(
    const float* grad_output,
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    int N,
    const int* shape,
    const int* tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* tile_offsets,
    const int* tile_counts,
    const int* tile_content,
    float* d_centers,
    float* d_conic,
    float* d_amps,
    int64_t num_tiles,
    const std::vector<int>& host_tile_dims,
    cudaStream_t stream
) {
    int block_size;
    if constexpr (DIM == 2) {
        block_size = RASTER_BLOCK_SIZE_2D;
    } else if constexpr (DIM == 3) {
        block_size = RASTER_BLOCK_SIZE_3D;
    } else {
        block_size = RASTER_BLOCK_SIZE_DEFAULT;
    }

    // Shared memory is statically allocated in the kernel using __shared__ arrays.
    // No dynamic shared memory needed (smem_size = 0).
    // Static allocation allows compile-time optimization and bank conflict avoidance.
    // Note: The backward kernel uses significantly more shared memory than forward
    // due to per-tile gradient accumulators and s_grad_output cache.
    constexpr size_t smem_size = 0;

    // OPTIMIZATION: Use 3D grid for 2D/3D volumes
    // Grid order: last dimension varies fastest to match linear index formula
    if constexpr (DIM == 3) {
        // Grid: (z, y, x) so blockIdx.x=z, blockIdx.y=y, blockIdx.z=x
        dim3 grid(host_tile_dims[2], host_tile_dims[1], host_tile_dims[0]);
        rasterize_backward_kernel<DIM, BATCH_SIZE, InputDType><<<grid, block_size, smem_size, stream>>>(
            grad_output, centers, conic, amps, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content,
            d_centers, d_conic, d_amps
        );
    } else if constexpr (DIM == 2) {
        // Grid: (y, x, 1) so blockIdx.x=y, blockIdx.y=x
        dim3 grid(host_tile_dims[1], host_tile_dims[0], 1);
        rasterize_backward_kernel<DIM, BATCH_SIZE, InputDType><<<grid, block_size, smem_size, stream>>>(
            grad_output, centers, conic, amps, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content,
            d_centers, d_conic, d_amps
        );
    } else {
        // For DIM > 3, use 1D grid
        int num_blocks = (int)num_tiles;
        rasterize_backward_kernel<DIM, BATCH_SIZE, InputDType><<<num_blocks, block_size, smem_size, stream>>>(
            grad_output, centers, conic, amps, N,
            shape, tile_dims, tile_size, truncate, intensity_floor,
            tile_offsets, tile_counts, tile_content,
            d_centers, d_conic, d_amps
        );
    }
}

// =============================================================================
// SPLAT-CENTRIC FORWARD KERNEL LAUNCHER
// =============================================================================

template <int DIM, typename InputDType = float>
void launch_rasterize_forward_splat_centric(
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    int N,
    const int* shape,
    float truncate,
    float intensity_floor,
    float* output,
    bool* global_splat_flags,
    int* global_splat_count,
    int64_t num_tiles,
    int tile_size,
    int* tile_counts_out,
    const int* tile_dims,
    cudaStream_t stream
) {
    if (N == 0) return;

    constexpr int BLOCK_SIZE = 256;
    int num_blocks = N;  // One block per splat

    rasterize_forward_splat_centric_kernel<DIM, InputDType><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        centers, conic, amps, N,
        shape, truncate, intensity_floor, output,
        global_splat_flags, global_splat_count, num_tiles, tile_size,
        tile_counts_out, tile_dims
    );
}

// =============================================================================
// SPLAT-CENTRIC BACKWARD KERNEL LAUNCHER
// =============================================================================

template <int DIM, typename InputDType = float>
void launch_rasterize_backward_splat_centric(
    const float* grad_output,
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    int N,
    const int* shape,
    float truncate,
    float intensity_floor,
    float* d_centers,
    float* d_conic,
    float* d_amps,
    cudaStream_t stream
) {
    if (N == 0) return;

    constexpr int BLOCK_SIZE = 256;
    int num_blocks = N;  // One block per splat

    rasterize_backward_splat_centric_kernel<DIM, InputDType><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        grad_output, centers, conic, amps, N,
        shape, truncate, intensity_floor,
        d_centers, d_conic, d_amps
    );
}

// =============================================================================
// GLOBAL SPLAT FORWARD KERNEL LAUNCHER
// =============================================================================

template <int DIM, typename InputDType = float>
void launch_rasterize_global_forward(
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    const int* global_splat_ids,
    int n_global_splats,
    const int* shape,
    float truncate,
    float intensity_floor,
    float* output,
    int64_t num_pixels,
    cudaStream_t stream
) {
    if (n_global_splats == 0) return;

    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (int)((num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);

    rasterize_global_forward_kernel<DIM, InputDType><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        centers, conic, amps,
        global_splat_ids, n_global_splats,
        shape, truncate, intensity_floor, output, num_pixels
    );
}

// =============================================================================
// GLOBAL SPLAT BACKWARD KERNEL LAUNCHER
// =============================================================================

template <int DIM, typename InputDType = float>
void launch_rasterize_global_backward(
    const float* grad_output,
    const InputDType* centers,
    const InputDType* conic,
    const InputDType* amps,
    const int* global_splat_ids,
    int n_global_splats,
    const int* shape,
    float truncate,
    float intensity_floor,
    float* d_centers,
    float* d_conic,
    float* d_amps,
    int64_t num_pixels,
    cudaStream_t stream
) {
    if (n_global_splats == 0) return;

    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (int)((num_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);

    rasterize_global_backward_kernel<DIM, InputDType><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        grad_output, centers, conic, amps,
        global_splat_ids, n_global_splats,
        shape, truncate, intensity_floor,
        d_centers, d_conic, d_amps, num_pixels
    );
}

#endif // CUDA_SPLATTING_KERNEL_LAUNCHERS_CUH
