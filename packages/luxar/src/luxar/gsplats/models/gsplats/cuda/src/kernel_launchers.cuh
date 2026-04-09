/**
 * CUDA Kernel Launch Wrappers
 *
 * This header contains launch wrapper functions for the splat-centric CUDA kernels.
 * These wrappers handle block/grid size configuration.
 *
 * Launch wrappers:
 * - launch_rasterize_forward_splat_centric: Forward rasterization (1 block per splat)
 * - launch_rasterize_backward_splat_centric: Backward rasterization (1 block per splat)
 */

#ifndef CUDA_SPLATTING_KERNEL_LAUNCHERS_CUH
#define CUDA_SPLATTING_KERNEL_LAUNCHERS_CUH

#include "utils.cuh"
#include "kernels_core.cuh"

#include <vector>

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
    cudaStream_t stream
) {
    if (N == 0) return;

    constexpr int BLOCK_SIZE = 256;
    int num_blocks = N;  // One block per splat

    rasterize_forward_splat_centric_kernel<DIM, InputDType><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        centers, conic, amps, N,
        shape, truncate, intensity_floor, output
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
    float* output_to_zero,  // Optional: zero forward output as side effect
    cudaStream_t stream
) {
    if (N == 0) return;

    constexpr int BLOCK_SIZE = 256;
    int num_blocks = N;  // One block per splat

    rasterize_backward_splat_centric_kernel<DIM, InputDType><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        grad_output, centers, conic, amps, N,
        shape, truncate, intensity_floor,
        d_centers, d_conic, d_amps,
        output_to_zero
    );
}

#endif // CUDA_SPLATTING_KERNEL_LAUNCHERS_CUH
