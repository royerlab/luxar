/**
 * PyTorch C++ Extension Bindings for CUDA Splatting Backend
 *
 * This file provides the pybind11 interface between Python and the CUDA kernels.
 * It exposes the forward() and backward() functions to Python.
 */

#include <torch/extension.h>
#include "cuda_splatting.h"

#include <vector>
#include <tuple>

// =============================================================================
// PYTHON-FACING FUNCTIONS
// =============================================================================

/**
 * Forward pass wrapper for Python.
 *
 * Converts Python types to C++ types and calls the CUDA forward function.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_wrapper(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& sharpness,
    const std::vector<int64_t>& shape,
    double truncate,
    double intensity_floor,
    int64_t tile_size
) {
    return forward(
        centers,
        conic,
        amps,
        sharpness,
        shape,
        (float)truncate,
        (float)intensity_floor,
        (int)tile_size
    );
}

/**
 * Backward pass wrapper for Python.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
backward_wrapper(
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
    double truncate,
    double intensity_floor,
    int64_t tile_size
) {
    return backward(
        grad_output,
        centers,
        conic,
        amps,
        sharpness,
        tile_offsets,
        tile_counts,
        tile_content,
        global_splat_ids,
        shape,
        (float)truncate,
        (float)intensity_floor,
        (int)tile_size
    );
}

// =============================================================================
// MODULE REGISTRATION
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"doc(
        CUDA backend for Gaussian splatting.

        This module provides GPU-accelerated forward and backward passes for
        rendering volumetric Gaussian splats. It supports 2D-8D volumes with
        tile-based rasterization and optimized gradient computation.

        Key Features:
        - Tile-based spatial binning (no depth sorting needed)
        - Warp-level gradient reduction
        - Shared memory batch loading
        - Support for generalized Gaussians (variable sharpness)

        See SPECIFICATIONS.md for detailed algorithm descriptions.
    )doc";

    m.def(
        "forward",
        &forward_wrapper,
        R"doc(
            Forward pass: Render Gaussians to volume.

            Parameters
            ----------
            centers : torch.Tensor
                (N, d) float32 - Splat centers in voxel coordinates
            conic : torch.Tensor
                (N, d*(d+1)/2) float32 - Packed upper-triangle of inverse covariance
            amps : torch.Tensor
                (N,) float32 - Amplitudes
            sharpness : torch.Tensor
                (N,) float32 - Sharpness parameters (s=2 for standard Gaussian)
            shape : List[int]
                Target volume shape (d elements)
            truncate : float
                Base truncation radius in standard deviations
            intensity_floor : float
                Minimum intensity threshold for culling
            tile_size : int
                Tile size for spatial binning

            Returns
            -------
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                - output: (prod(shape),) float32 - Rendered volume (flattened)
                - tile_counts: (num_tiles,) int32 - Splats per tile
                - tile_offsets: (num_tiles,) int64 - Exclusive prefix sum
                - tile_content: (total_pairs,) int32 - Splat IDs per tile
                - global_splat_ids: (num_global,) int32 - Global splat IDs (for backward)
        )doc",
        py::arg("centers"),
        py::arg("conic"),
        py::arg("amps"),
        py::arg("sharpness"),
        py::arg("shape"),
        py::arg("truncate"),
        py::arg("intensity_floor"),
        py::arg("tile_size")
    );

    m.def(
        "backward",
        &backward_wrapper,
        R"doc(
            Backward pass: Compute gradients.

            Parameters
            ----------
            grad_output : torch.Tensor
                (prod(shape),) float32 - Upstream gradient
            centers : torch.Tensor
                (N, d) float32 - Splat centers (from forward)
            conic : torch.Tensor
                (N, d*(d+1)/2) float32 - Packed conic (from forward)
            amps : torch.Tensor
                (N,) float32 - Amplitudes (from forward)
            sharpness : torch.Tensor
                (N,) float32 - Sharpness parameters (from forward)
            tile_offsets : torch.Tensor
                (num_tiles,) int64 - From forward pass
            tile_counts : torch.Tensor
                (num_tiles,) int32 - From forward pass
            tile_content : torch.Tensor
                (total_pairs,) int32 - From forward pass
            global_splat_ids : torch.Tensor
                (num_global,) int32 - Global splat IDs from forward pass
            shape : List[int]
                Target volume shape
            truncate : float
                Base truncation radius
            intensity_floor : float
                Minimum intensity threshold
            tile_size : int
                Tile size

            Returns
            -------
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                - d_centers: (N, d) float32 - Center gradients
                - d_conic: (N, d*(d+1)/2) float32 - Conic gradients
                - d_amps: (N,) float32 - Amplitude gradients
                - d_sharpness: (N,) float32 - Sharpness gradients
        )doc",
        py::arg("grad_output"),
        py::arg("centers"),
        py::arg("conic"),
        py::arg("amps"),
        py::arg("sharpness"),
        py::arg("tile_offsets"),
        py::arg("tile_counts"),
        py::arg("tile_content"),
        py::arg("global_splat_ids"),
        py::arg("shape"),
        py::arg("truncate"),
        py::arg("intensity_floor"),
        py::arg("tile_size")
    );

    // Version info
    m.attr("__version__") = "0.1.0";
    m.attr("__cuda_version__") = std::to_string(CUDART_VERSION);
}
