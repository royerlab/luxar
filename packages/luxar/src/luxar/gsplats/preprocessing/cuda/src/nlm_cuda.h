/**
 * Non-Local Means CUDA Kernel — Public API
 *
 * Provides GPU-accelerated NLM denoising for 2D images and 3D volumes.
 * One thread per output voxel, with shared memory tiling for the
 * search + patch neighbourhood.
 */

#ifndef NLM_CUDA_H
#define NLM_CUDA_H

#include <torch/extension.h>

/**
 * NLM denoise for a 2D image.
 *
 * @param input       (H, W) float32 tensor
 * @param h           filtering strength (> 0)
 * @param patch_half  half-size of comparison patch (patch_size = 2*patch_half + 1)
 * @param search_dist half-size of search window
 * @return            (H, W) float32 denoised tensor
 */
torch::Tensor nlm_denoise_2d(
    const torch::Tensor& input,
    float h,
    int patch_half,
    int search_dist
);

/**
 * NLM denoise for a 3D volume.
 *
 * @param input       (D, H, W) float32 tensor
 * @param h           filtering strength (> 0)
 * @param patch_half  half-size of comparison patch
 * @param search_dist half-size of search window
 * @return            (D, H, W) float32 denoised tensor
 */
torch::Tensor nlm_denoise_3d(
    const torch::Tensor& input,
    float h,
    int patch_half,
    int search_dist
);

#endif // NLM_CUDA_H
