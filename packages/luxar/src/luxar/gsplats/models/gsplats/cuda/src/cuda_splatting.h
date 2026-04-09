/**
 * CUDA Gaussian Splatting - Header Declarations
 *
 * This header declares the public interface for the CUDA splatting backend.
 * It is used by both the CUDA dispatch layer (cuda_splatting.cu) and the
 * PyTorch C++ bindings (bindings.cpp).
 *
 * Architecture:
 *   Forward:  splat-centric rasterization (1 block per splat, atomicAdd to output)
 *   Backward: splat-centric gradient computation (1 block per splat, warp reduction)
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


// =============================================================================
// FORWARD STATE
// =============================================================================

/**
 * State from forward pass, cached for backward pass reuse.
 *
 * Contains cached device tensors and empty placeholders for backward API
 * compatibility (tile_counts, tile_offsets, etc. are no longer computed
 * by the splat-centric forward kernel).
 */
struct BinningState {
    // Empty tensors for backward API compatibility
    torch::Tensor tile_counts;      // empty (0,) int32
    torch::Tensor tile_offsets;     // empty (0,) int64
    torch::Tensor tile_content;     // empty (0,) int32
    torch::Tensor global_splat_ids; // empty (0,) int32
    int num_global_splats;

    // Metadata
    int64_t num_tiles;

    // Cached device tensor for backward pass reuse
    torch::Tensor shape_tensor;     // (dim,) int32 - volume shape on device
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
 * @param tile_size       Tile size for spatial partitioning
 * @param batch_size      Unused (kept for API compatibility)
 *
 * @return Tuple of 2 tensors: (output, shape_tensor)
 */
std::tuple<torch::Tensor, torch::Tensor>
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
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    bool use_fp16,
    const torch::Tensor& shape_tensor_cached = torch::Tensor(),
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
std::tuple<torch::Tensor, torch::Tensor>
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
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    const torch::Tensor& shape_tensor_cached = torch::Tensor(),
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
