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
 * Returns a 2-tuple: (output, shape_tensor)
 * The second is a cached device tensor for backward pass reuse.
 */
std::tuple<torch::Tensor, torch::Tensor>
forward_wrapper(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const torch::Tensor& L_row_norms,
    const std::vector<int64_t>& shape,
    double truncate,
    double intensity_floor,
    int64_t tile_size,
    int64_t batch_size,
    bool use_fp16,
    const c10::optional<torch::Tensor>& output_buffer
) {
    torch::Tensor out_buf = output_buffer.value_or(torch::Tensor());

    if (use_fp16) {
        auto centers_fp16 = centers.dtype() == torch::kFloat16 ? centers : centers.to(torch::kFloat16);
        auto conic_fp16 = conic.dtype() == torch::kFloat16 ? conic : conic.to(torch::kFloat16);
        auto amps_fp16 = amps.dtype() == torch::kFloat16 ? amps : amps.to(torch::kFloat16);
        auto L_row_norms_fp16 = L_row_norms.dtype() == torch::kFloat16 ? L_row_norms : L_row_norms.to(torch::kFloat16);

        return forward_fp16(
            centers_fp16, conic_fp16, amps_fp16, L_row_norms_fp16,
            shape, (float)truncate, (float)intensity_floor,
            (int)tile_size, (int)batch_size, out_buf
        );
    }

    return forward(
        centers, conic, amps, L_row_norms,
        shape, (float)truncate, (float)intensity_floor,
        (int)tile_size, (int)batch_size, out_buf
    );
}

/**
 * Backward pass wrapper for Python.
 *
 * Gradients are always returned as FP32 for numerical stability.
 * Returns a 3-tuple: (d_centers, d_conic, d_amps)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
backward_wrapper(
    const torch::Tensor& grad_output,
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    const torch::Tensor& amps,
    const std::vector<int64_t>& shape,
    double truncate,
    double intensity_floor,
    bool use_fp16,
    const c10::optional<torch::Tensor>& shape_tensor_cached,
    const c10::optional<torch::Tensor>& output_to_zero
) {
    torch::Tensor shape_cached = shape_tensor_cached.value_or(torch::Tensor());
    torch::Tensor output_zero = output_to_zero.value_or(torch::Tensor());

    if (use_fp16) {
        auto centers_fp16 = centers.dtype() == torch::kFloat16 ? centers : centers.to(torch::kFloat16);
        auto conic_fp16 = conic.dtype() == torch::kFloat16 ? conic : conic.to(torch::kFloat16);
        auto amps_fp16 = amps.dtype() == torch::kFloat16 ? amps : amps.to(torch::kFloat16);

        return backward_fp16(
            grad_output, centers_fp16, conic_fp16, amps_fp16,
            shape, (float)truncate, (float)intensity_floor,
            shape_cached, output_zero
        );
    }

    return backward(
        grad_output, centers, conic, amps,
        shape, (float)truncate, (float)intensity_floor,
        use_fp16, shape_cached, output_zero
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
        splat-centric rasterization and optimized gradient computation.

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
            L_row_norms : torch.Tensor
                (N, d) float32 - Per-axis standard deviations from Cholesky row norms.
            shape : List[int]
                Target volume shape (d elements)
            truncate : float
                Base truncation radius in standard deviations
            intensity_floor : float
                Minimum intensity threshold for culling
            tile_size : int
                Tile size for spatial partitioning
            batch_size : int
                Unused (kept for API compatibility). Default: 128.
            use_fp16 : bool
                If True, use FP16 precision for inputs. Default: False.

            Returns
            -------
            Tuple of 2 tensors:
                - output: (prod(shape),) float32 - Rendered volume (flattened)
                - shape_tensor: (d,) int32 - Volume shape on device (for backward reuse)
        )doc",
        py::arg("centers"),
        py::arg("conic"),
        py::arg("amps"),
        py::arg("L_row_norms"),
        py::arg("shape"),
        py::arg("truncate"),
        py::arg("intensity_floor"),
        py::arg("tile_size"),
        py::arg("batch_size") = 128,
        py::arg("use_fp16") = false,
        py::arg("output_buffer") = py::none()
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
            shape : List[int]
                Target volume shape
            truncate : float
                Base truncation radius
            intensity_floor : float
                Minimum intensity threshold
            use_fp16 : bool
                Must match forward pass. Default: False.
            shape_tensor_cached : torch.Tensor, optional
                Cached device tensor from forward pass.

            Returns
            -------
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                - d_centers: (N, d) float32 - Center gradients
                - d_conic: (N, d*(d+1)/2) float32 - Conic gradients
                - d_amps: (N,) float32 - Amplitude gradients
        )doc",
        py::arg("grad_output"),
        py::arg("centers"),
        py::arg("conic"),
        py::arg("amps"),
        py::arg("shape"),
        py::arg("truncate"),
        py::arg("intensity_floor"),
        py::arg("use_fp16") = false,
        py::arg("shape_tensor_cached") = py::none(),
        py::arg("output_to_zero") = py::none()
    );

    // Version info
    m.attr("__version__") = "0.2.0";
    m.attr("__cuda_version__") = std::to_string(CUDART_VERSION);
}
