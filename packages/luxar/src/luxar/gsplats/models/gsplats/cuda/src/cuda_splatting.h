/**
 * CUDA Gaussian Splatting - Header Declarations
 *
 * This header declares the public interface for the CUDA splatting backend.
 * It is used by both the CUDA dispatch layer (cuda_splatting.cu) and the
 * PyTorch C++ bindings (bindings.cpp).
 *
 * Architecture:
 *   Forward:  preprocess -> prefix_sum -> bin -> rasterize_fwd [-> global_fwd]
 *   Backward: rasterize_bwd [-> global_bwd]
 *
 * FP16 support: All kernels and launch wrappers are templated on InputDType
 * (float or __half). The dispatch layer in cuda_splatting.cu uses unified
 * template functions (dispatch_forward_impl<InputDType>, etc.) to avoid
 * FP32/FP16 code duplication. See kernel_launchers.cuh for the templated
 * launch wrappers.
 */

#ifndef CUDA_SPLATTING_H
#define CUDA_SPLATTING_H

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <tuple>

// =============================================================================
// CONFIGURATION CONSTANTS
// =============================================================================

// Maximum supported dimensions (2D to 8D)
constexpr int MIN_DIM = 2;
constexpr int MAX_SUPPORTED_DIM = 8;

// Maximum number of tiles (1M limit for safety)
constexpr int64_t MAX_TILES = 1000000;

// Default tile sizes per dimension
constexpr int DEFAULT_TILE_SIZE_2D = 16;
constexpr int DEFAULT_TILE_SIZE_3D = 8;
constexpr int DEFAULT_TILE_SIZE_4D = 4;
constexpr int DEFAULT_TILE_SIZE_HIGH = 2;

// =============================================================================
// BINNING STATE
// =============================================================================

/**
 * Workspace for tile binning operations.
 *
 * This structure holds temporary buffers used during the forward pass
 * and passed to the backward pass for gradient computation.
 *
 * OPTIMIZATION: Cached device tensors (shape_tensor, tile_dims_tensor) are
 * stored here for reuse in backward pass, avoiding redundant allocations
 * and host-to-device copies.
 */
struct BinningState {
    torch::Tensor tile_counts;      // (num_tiles,) int32 - splats per tile
    torch::Tensor tile_offsets;     // (num_tiles,) int64 - exclusive prefix sum
    torch::Tensor tile_content;     // (total_pairs,) int32 - splat IDs per tile
    torch::Tensor tile_write_heads; // (num_tiles,) int32 - atomic write positions

    // Scan temporary storage (persisted for reuse)
    torch::Tensor scan_temp_storage;
    size_t scan_temp_bytes;

    // Global splat handling (large splats)
    torch::Tensor global_splat_flags; // (N,) bool - is global splat
    torch::Tensor global_splat_ids;   // (num_global,) int32 - global splat IDs
    int num_global_splats;

    // AABB cache: avoids recomputing AABBs in bin_kernel
    torch::Tensor aabb_lo;          // (N, DIM) int32 - AABB lower bounds (tile coords)
    torch::Tensor aabb_hi;          // (N, DIM) int32 - AABB upper bounds (tile coords)

    // Metadata
    int64_t num_tiles;
    int64_t total_pairs;  // Total (tile, splat) pairs

    // OPTIMIZATION: Cached device tensors for backward pass reuse
    // These eliminate redundant host-to-device copies in backward pass
    torch::Tensor shape_tensor;     // (dim,) int32 - volume shape on device
    torch::Tensor tile_dims_tensor; // (dim,) int32 - tile dimensions on device
    int tile_size;                  // Cached tile size
};

// =============================================================================
// FORWARD PASS INTERFACE
// =============================================================================

/**
 * Forward pass: Render Gaussians to volume (FP32 inputs).
 *
 * @param centers         (N, d) float32 - splat centers in voxel coordinates
 * @param conic           (N, d*(d+1)/2) float32 - packed upper-triangle of Sigma^-1
 * @param amps            (N,) float32 - amplitudes
 * @param L_row_norms     (N, d) float32 - per-axis std dev from Cholesky row norms
 * @param shape           Target volume shape (d elements)
 * @param truncate        Base truncation radius
 * @param intensity_floor Minimum intensity threshold for culling
 * @param tile_size       Tile size for spatial binning
 * @param batch_size      Splat batch size for shared memory loading (32, 128, or 256)
 *
 * @return Tuple of 7 tensors (see cuda_splatting.cu for details)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
forward(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& L_row_norms,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size,
    const torch::Tensor& output_buffer = torch::Tensor()
);

// =============================================================================
// BACKWARD PASS INTERFACE
// =============================================================================

/**
 * Backward pass: Compute gradients (FP32 inputs).
 *
 * @return Tuple of 3 gradient tensors: d_centers, d_conic, d_amps
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size,
    const torch::Tensor& shape_tensor_cached = torch::Tensor(),
    const torch::Tensor& tile_dims_tensor_cached = torch::Tensor(),
    const torch::Tensor& output_to_zero = torch::Tensor()
);

// =============================================================================
// FP16 (HALF PRECISION) INTERFACE
// =============================================================================
//
// Mixed-precision variants: inputs are FP16, computation is FP32, output is FP32.
// This provides ~1.5-2x memory bandwidth improvement while maintaining precision.
//
// FP16 inputs are loaded directly from global memory and converted to FP32 during
// shared memory load (DTypeTraits::load()). All computation and gradients are FP32.
//
// The FP32 and FP16 implementations share a single templated dispatch layer
// (dispatch_forward_impl<InputDType>, dispatch_backward_impl<InputDType>)
// in cuda_splatting.cu. These thin wrappers provide a non-templated API for
// bindings.cpp.

/**
 * Forward pass with FP16 inputs.
 * Same signature as forward() but expects float16 input tensors.
 * Output is always FP32.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
forward_fp16(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& L_row_norms,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size,
    const torch::Tensor& output_buffer = torch::Tensor()
);

/**
 * Backward pass with FP16 inputs.
 * Gradients are always FP32 for numerical stability.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
backward_fp16(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    int batch_size,
    const torch::Tensor& shape_tensor_cached = torch::Tensor(),
    const torch::Tensor& tile_dims_tensor_cached = torch::Tensor(),
    const torch::Tensor& output_to_zero = torch::Tensor()
);

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Compute tile dimensions from volume shape and tile size.
 */
std::vector<int> compute_tile_dims(
    const std::vector<int64_t>& shape,
    int tile_size
);

/**
 * Compute total number of tiles.
 */
int64_t compute_num_tiles(const std::vector<int>& tile_dims);

/**
 * Validate input tensors.
 *
 * @param expected_dtype Expected dtype (torch::kFloat32 or torch::kFloat16)
 */
void validate_inputs(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    torch::ScalarType expected_dtype = torch::kFloat32
);

#endif // CUDA_SPLATTING_H
