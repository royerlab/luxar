/**
 * CUDA Gaussian Splatting - Header Declarations
 *
 * This header declares the public interface for the CUDA splatting backend.
 * It is used by both the CUDA kernels and the PyTorch C++ bindings.
 *
 * Architecture:
 *   Forward:  preprocess → prefix_sum → bin → rasterize_fwd
 *   Backward: rasterize_bwd
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
 * Forward pass: Render Gaussians to volume.
 *
 * @param centers         (N, d) float32 - splat centers in voxel coordinates
 * @param conic           (N, d*(d+1)/2) float32 - packed upper-triangle of Σ⁻¹
 * @param amps            (N,) float32 - amplitudes
 * @param sharpness       (N,) float32 - sharpness parameters
 * @param shape           Target volume shape (d elements)
 * @param truncate        Base truncation radius
 * @param intensity_floor Minimum intensity threshold for culling
 * @param tile_size       Tile size for spatial binning
 *
 * @return Tuple of:
 *   - output: (prod(shape),) float32 - rendered volume (flattened)
 *   - tile_counts: (num_tiles,) int32 - splats per tile
 *   - tile_offsets: (num_tiles,) int64 - exclusive prefix sum
 *   - tile_content: (total_pairs,) int32 - splat IDs per tile
 *   - global_splat_ids: (num_global,) int32 - IDs of global splats (for backward)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size
);

// =============================================================================
// BACKWARD PASS INTERFACE
// =============================================================================

/**
 * Backward pass: Compute gradients.
 *
 * @param grad_output     (prod(shape),) float32 - upstream gradient
 * @param centers         (N, d) float32 - splat centers
 * @param conic           (N, d*(d+1)/2) float32 - packed conic
 * @param amps            (N,) float32 - amplitudes
 * @param sharpness       (N,) float32 - sharpness parameters
 * @param tile_offsets    (num_tiles,) int64 - from forward pass
 * @param tile_counts     (num_tiles,) int32 - from forward pass
 * @param tile_content    (total_pairs,) int32 - from forward pass
 * @param global_splat_ids (num_global,) int32 - from forward pass
 * @param shape           Target volume shape
 * @param truncate        Base truncation radius
 * @param intensity_floor Minimum intensity threshold
 * @param tile_size       Tile size
 *
 * @return Tuple of:
 *   - d_centers: (N, d) float32 - center gradients
 *   - d_conic: (N, d*(d+1)/2) float32 - conic gradients
 *   - d_amps: (N,) float32 - amplitude gradients
 *   - d_sharpness: (N,) float32 - sharpness gradients
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size
);

// =============================================================================
// FP16 (HALF PRECISION) FORWARD PASS INTERFACE
// =============================================================================

/**
 * Forward pass with FP16 inputs for improved memory bandwidth.
 *
 * This is the mixed-precision variant: inputs are FP16, computation is FP32,
 * output is FP32. This provides ~1.5-2x memory bandwidth improvement while
 * maintaining numerical precision.
 *
 * @param centers         (N, d) float16 - splat centers in voxel coordinates
 * @param conic           (N, d*(d+1)/2) float16 - packed upper-triangle of Σ⁻¹
 * @param amps            (N,) float16 - amplitudes
 * @param sharpness       (N,) float16 - sharpness parameters
 * @param shape           Target volume shape (d elements)
 * @param truncate        Base truncation radius
 * @param intensity_floor Minimum intensity threshold for culling
 * @param tile_size       Tile size for spatial binning
 *
 * @return Tuple of:
 *   - output: (prod(shape),) float32 - rendered volume (always FP32)
 *   - tile_counts: (num_tiles,) int32 - splats per tile
 *   - tile_offsets: (num_tiles,) int64 - exclusive prefix sum
 *   - tile_content: (total_pairs,) int32 - splat IDs per tile
 *   - global_splat_ids: (num_global,) int32 - IDs of global splats
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_fp16(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size
);

// =============================================================================
// FP16 (HALF PRECISION) BACKWARD PASS INTERFACE
// =============================================================================

/**
 * Backward pass with FP16 inputs.
 *
 * Inputs are FP16, gradients are always FP32 for numerical stability.
 * No loss scaling is required due to FP32 gradient accumulation.
 *
 * @param grad_output     (prod(shape),) float32 - upstream gradient (always FP32)
 * @param centers         (N, d) float16 - splat centers
 * @param conic           (N, d*(d+1)/2) float16 - packed conic
 * @param amps            (N,) float16 - amplitudes
 * @param sharpness       (N,) float16 - sharpness parameters
 * @param tile_offsets    (num_tiles,) int64 - from forward pass
 * @param tile_counts     (num_tiles,) int32 - from forward pass
 * @param tile_content    (total_pairs,) int32 - from forward pass
 * @param global_splat_ids (num_global,) int32 - from forward pass
 * @param shape           Target volume shape
 * @param truncate        Base truncation radius
 * @param intensity_floor Minimum intensity threshold
 * @param tile_size       Tile size
 *
 * @return Tuple of:
 *   - d_centers: (N, d) float32 - center gradients (always FP32)
 *   - d_conic: (N, d*(d+1)/2) float32 - conic gradients
 *   - d_amps: (N,) float32 - amplitude gradients
 *   - d_sharpness: (N,) float32 - sharpness gradients
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_fp16(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size
);

// =============================================================================
// DISPATCHER FUNCTIONS (by dimension)
// =============================================================================

// Forward dispatcher - selects optimized kernel based on dimension
void dispatch_forward(
    int dim,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    torch::Tensor& output,
    BinningState& state
);

// Backward dispatcher
void dispatch_backward(
    int dim,
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    torch::Tensor& d_centers,
    torch::Tensor& d_conic,
    torch::Tensor& d_amps,
    torch::Tensor& d_sharpness
);

// FP16 Forward dispatcher - uses FP16 inputs with FP32 compute
void dispatch_forward_fp16(
    int dim,
    const torch::Tensor& centers,    // float16
    const torch::Tensor& conic,      // float16
    const torch::Tensor& amps,       // float16
    const torch::Tensor& sharpness,  // float16
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    torch::Tensor& output,           // float32
    BinningState& state
);

// FP16 Backward dispatcher - uses FP16 inputs, produces FP32 gradients
void dispatch_backward_fp16(
    int dim,
    const torch::Tensor& grad_output, // float32
    const torch::Tensor& centers,     // float16
    const torch::Tensor& conic,       // float16
    const torch::Tensor& amps,        // float16
    const torch::Tensor& sharpness,   // float16
    const torch::Tensor& tile_offsets,
    const torch::Tensor& tile_counts,
    const torch::Tensor& tile_content,
    const torch::Tensor& global_splat_ids,
    const std::vector<int64_t>& shape,
    float truncate,
    float intensity_floor,
    int tile_size,
    torch::Tensor& d_centers,   // float32
    torch::Tensor& d_conic,     // float32
    torch::Tensor& d_amps,      // float32
    torch::Tensor& d_sharpness  // float32
);

// =============================================================================
// KERNEL LAUNCH WRAPPERS (templated by DIM)
// =============================================================================

// Preprocess: Compute AABBs and tile counts
template <int DIM>
void launch_preprocess(
    const float* centers,
    const float* conic,
    const float* amps,
    const float* sharpness,
    int N,
    const int* shape,
    const int* tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    int* tile_counts,
    bool* global_flags,
    int64_t num_tiles,
    cudaStream_t stream
);

// Binning: Assign splats to tiles
template <int DIM>
void launch_bin(
    const float* centers,
    const float* conic,
    const float* amps,
    const float* sharpness,
    int N,
    const int* shape,
    const int* tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* tile_offsets,
    int* tile_write_heads,
    int* tile_content,
    int64_t num_tiles,
    cudaStream_t stream
);

// Forward rasterization: Render splats to pixels
// OPTIMIZATION: For 2D/3D, uses dim3 grid for better cache locality
template <int DIM>
void launch_rasterize_forward(
    const float* centers,
    const float* conic,
    const float* amps,
    const float* sharpness,
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
);

// Backward rasterization: Compute gradients
// OPTIMIZATION: For 2D/3D, uses dim3 grid for better cache locality
template <int DIM>
void launch_rasterize_backward(
    const float* grad_output,
    const float* centers,
    const float* conic,
    const float* amps,
    const float* sharpness,
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
    float* d_sharpness,
    int64_t num_tiles,
    const std::vector<int>& host_tile_dims,
    cudaStream_t stream
);

// Global splat forward: Render global splats to all pixels
template <int DIM>
void launch_rasterize_global_forward(
    const float* centers,
    const float* conic,
    const float* amps,
    const float* sharpness,
    const int* global_splat_ids,
    int n_global_splats,
    const int* shape,
    float truncate,
    float* output,
    int64_t num_pixels,
    cudaStream_t stream
);

// Global splat backward: Compute gradients for global splats
template <int DIM>
void launch_rasterize_global_backward(
    const float* grad_output,
    const float* centers,
    const float* conic,
    const float* amps,
    const float* sharpness,
    const int* global_splat_ids,
    int n_global_splats,
    const int* shape,
    float truncate,
    float* d_centers,
    float* d_conic,
    float* d_amps,
    float* d_sharpness,
    int64_t num_pixels,
    cudaStream_t stream
);

// =============================================================================
// FP16 KERNEL LAUNCH WRAPPERS (templated by DIM)
// =============================================================================
//
// NOTE: These are placeholder declarations for future "true FP16 kernel" optimization.
// The current implementation (Phase 1) uses FP16→FP32 conversion at the API boundary
// via dispatch_forward_fp16() and dispatch_backward_fp16(), which convert tensors
// and call the regular FP32 kernels.
//
// Phase 2 would implement these functions to load FP16 data directly into shared
// memory and convert to FP32 only in registers, providing better memory bandwidth.
// Until then, these declarations are not instantiated and serve as documentation.
//

// FP16 Forward rasterization: Mixed precision (FP16 input, FP32 compute/output)
// NOT YET IMPLEMENTED - see dispatch_forward_fp16() for current implementation
template <int DIM>
void launch_rasterize_forward_fp16(
    const __half* centers,    // FP16 input
    const __half* conic,      // FP16 input
    const __half* amps,       // FP16 input
    const __half* sharpness,  // FP16 input
    int N,
    const int* shape,
    const int* tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* tile_offsets,
    const int* tile_counts,
    const int* tile_content,
    float* output,            // FP32 output
    int64_t num_tiles,
    const std::vector<int>& host_tile_dims,
    cudaStream_t stream
);

// FP16 Backward rasterization: Mixed precision (FP16 input, FP32 gradients)
// NOT YET IMPLEMENTED - see dispatch_backward_fp16() for current implementation
template <int DIM>
void launch_rasterize_backward_fp16(
    const float* grad_output, // FP32 upstream gradient
    const __half* centers,    // FP16 input
    const __half* conic,      // FP16 input
    const __half* amps,       // FP16 input
    const __half* sharpness,  // FP16 input
    int N,
    const int* shape,
    const int* tile_dims,
    int tile_size,
    float truncate,
    float intensity_floor,
    const int64_t* tile_offsets,
    const int* tile_counts,
    const int* tile_content,
    float* d_centers,         // FP32 gradients
    float* d_conic,
    float* d_amps,
    float* d_sharpness,
    int64_t num_tiles,
    const std::vector<int>& host_tile_dims,
    cudaStream_t stream
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
 * Validate input tensors (FP32 version).
 */
void validate_inputs(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const std::vector<int64_t>& shape
);

/**
 * Validate input tensors (FP16 version).
 */
void validate_inputs_fp16(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const std::vector<int64_t>& shape
);

#endif // CUDA_SPLATTING_H
