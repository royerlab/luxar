/**
 * PyTorch C++ Extension Bindings for NLM CUDA Backend
 *
 * Exposes nlm_denoise_2d() and nlm_denoise_3d() to Python via pybind11.
 */

#include <torch/extension.h>
#include "nlm_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA-accelerated Non-Local Means denoising";

    m.def(
        "nlm_denoise_2d",
        &nlm_denoise_2d,
        R"doc(
            Non-Local Means denoising for a 2D image.

            Parameters
            ----------
            input : torch.Tensor
                (H, W) float32 CUDA tensor
            h : float
                Filtering strength (> 0)
            patch_half : int
                Half the patch side length. Supported: 1 (patch_size=3) or 2 (patch_size=5).
            search_dist : int
                Half the search window side length. Supported: 5 or 7.

            Returns
            -------
            torch.Tensor
                (H, W) float32 denoised image
        )doc",
        py::arg("input"),
        py::arg("h"),
        py::arg("patch_half"),
        py::arg("search_dist")
    );

    m.def(
        "nlm_denoise_3d",
        &nlm_denoise_3d,
        R"doc(
            Non-Local Means denoising for a 3D volume.

            Parameters
            ----------
            input : torch.Tensor
                (D, H, W) float32 CUDA tensor
            h : float
                Filtering strength (> 0)
            patch_half : int
                Half the patch side length. Supported: 1 (patch_size=3) or 2 (patch_size=5).
            search_dist : int
                Half the search window side length. Supported: 5 or 7.

            Returns
            -------
            torch.Tensor
                (D, H, W) float32 denoised volume
        )doc",
        py::arg("input"),
        py::arg("h"),
        py::arg("patch_half"),
        py::arg("search_dist")
    );

    m.attr("__version__") = "0.1.0";
}
