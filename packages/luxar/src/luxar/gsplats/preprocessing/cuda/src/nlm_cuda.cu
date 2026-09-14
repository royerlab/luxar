/**
 * Non-Local Means CUDA Kernels — 2D and 3D implementations.
 *
 * Strategy: one thread per output voxel.  Each thread block loads a tile
 * (plus halo) into shared memory, then every thread computes its NLM
 * weighted average entirely from shared memory.
 *
 * Template parameters:
 *   PATCH_HALF  — half the patch side length (compile-time for unrolling)
 *   SEARCH_DIST — half the search window side length (compile-time)
 *
 * Pre-instantiated for 12 microscopy parameter combos (2 patch sizes x 6 search distances):
 *   PATCH_HALF = {1, 2}       → patch_size = {3, 5}
 *   SEARCH_DIST = {5, 7, 9, 11, 13, 15}
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

#include "nlm_cuda.h"

// ============================================================================
// 2D Kernel — shared memory tiled
// ============================================================================

// Block tile dimensions for 2D (threads per block = TILE_Y * TILE_X)
static constexpr int TILE_2D_Y = 16;
static constexpr int TILE_2D_X = 16;

template <int PATCH_HALF, int SEARCH_DIST>
__global__ void nlm_2d_kernel(
    const float* __restrict__ input,   // padded: (H_pad, W_pad)
    float* __restrict__ output,         // (H, W)
    const int H, const int W,
    const int H_pad, const int W_pad,
    const int pad,
    const float exp_scale              // = -1.0 / (patch_area * h^2)
) {
    constexpr int HALO = SEARCH_DIST + PATCH_HALF;
    constexpr int SMEM_Y = TILE_2D_Y + 2 * HALO;
    constexpr int SMEM_X = TILE_2D_X + 2 * HALO;

    __shared__ float smem[SMEM_Y][SMEM_X];

    // Block tile origin in padded image coordinates
    const int by0 = blockIdx.y * TILE_2D_Y;
    const int bx0 = blockIdx.x * TILE_2D_X;

    // Cooperative load of shared memory tile
    const int smem_area = SMEM_Y * SMEM_X;
    const int threads_per_block = TILE_2D_Y * TILE_2D_X;
    const int tid = threadIdx.y * TILE_2D_X + threadIdx.x;

    for (int i = tid; i < smem_area; i += threads_per_block) {
        const int sy = i / SMEM_X;
        const int sx = i % SMEM_X;
        int gy = by0 + sy;
        int gx = bx0 + sx;
        // Clamp to padded image bounds
        gy = min(max(gy, 0), H_pad - 1);
        gx = min(max(gx, 0), W_pad - 1);
        smem[sy][sx] = input[gy * W_pad + gx];
    }
    __syncthreads();

    // Output voxel coordinates
    const int oy = blockIdx.y * TILE_2D_Y + threadIdx.y;
    const int ox = blockIdx.x * TILE_2D_X + threadIdx.x;
    if (oy >= H || ox >= W) return;

    // Local coordinates in shared memory (center of search window)
    const int ly = threadIdx.y + HALO;
    const int lx = threadIdx.x + HALO;

    float weighted_sum = 0.0f;
    float weight_total = 0.0f;

    // Iterate over search window
    for (int dy = -SEARCH_DIST; dy <= SEARCH_DIST; dy++) {
        for (int dx = -SEARCH_DIST; dx <= SEARCH_DIST; dx++) {
            // Compute raw squared patch distance (NOT normalised)
            float dist_raw = 0.0f;

            #pragma unroll
            for (int py = -PATCH_HALF; py <= PATCH_HALF; py++) {
                #pragma unroll
                for (int px = -PATCH_HALF; px <= PATCH_HALF; px++) {
                    float c = smem[ly + py][lx + px];
                    float s = smem[ly + dy + py][lx + dx + px];
                    float diff = c - s;
                    dist_raw += diff * diff;
                }
            }

            // exp_scale = -1/(patch_area * h^2) folds normalisation + negation
            float w = expf(dist_raw * exp_scale);
            weighted_sum += w * smem[ly + dy][lx + dx];
            weight_total += w;
        }
    }

    output[oy * W + ox] = weighted_sum / fmaxf(weight_total, 1e-10f);
}


// ============================================================================
// 3D Kernel — shared memory tiled
// ============================================================================

// Block tile dimensions for 3D (threads per block = TILE_Z * TILE_Y * TILE_X)
static constexpr int TILE_3D_Z = 4;
static constexpr int TILE_3D_Y = 8;
static constexpr int TILE_3D_X = 8;

template <int PATCH_HALF, int SEARCH_DIST>
__global__ void nlm_3d_kernel(
    const float* __restrict__ input,   // original: (D, H, W) — NOT padded
    float* __restrict__ output,         // (D, H, W)
    const int D, const int H, const int W,
    const float exp_scale              // = -1.0 / (patch_vol * h^2)
) {
    constexpr int HALO = SEARCH_DIST + PATCH_HALF;
    constexpr int SMEM_Z = TILE_3D_Z + 2 * HALO;
    constexpr int SMEM_Y = TILE_3D_Y + 2 * HALO;
    constexpr int SMEM_X = TILE_3D_X + 2 * HALO;

    // Use dynamic shared memory to support larger halo sizes
    extern __shared__ float smem_flat[];

    // Index helper macro for 3D shared memory
    #define SMEM(z, y, x) smem_flat[(z) * SMEM_Y * SMEM_X + (y) * SMEM_X + (x)]

    // Block tile origin (output voxel coordinates, not padded)
    const int bz0 = blockIdx.z * TILE_3D_Z;
    const int by0 = blockIdx.y * TILE_3D_Y;
    const int bx0 = blockIdx.x * TILE_3D_X;

    // Cooperative load — clamp to original volume bounds (reflection at edges)
    const int smem_vol = SMEM_Z * SMEM_Y * SMEM_X;
    const int threads_per_block = TILE_3D_Z * TILE_3D_Y * TILE_3D_X;
    const int tid = threadIdx.z * TILE_3D_Y * TILE_3D_X
                  + threadIdx.y * TILE_3D_X
                  + threadIdx.x;

    for (int i = tid; i < smem_vol; i += threads_per_block) {
        const int sz = i / (SMEM_Y * SMEM_X);
        const int rem = i % (SMEM_Y * SMEM_X);
        const int sy = rem / SMEM_X;
        const int sx = rem % SMEM_X;

        // Map shared memory position to global coordinates (with halo offset)
        int gz = bz0 + sz - HALO;
        int gy = by0 + sy - HALO;
        int gx = bx0 + sx - HALO;

        // Reflect at boundaries: mirror coordinates into [0, dim-1]
        gz = abs(gz);
        gy = abs(gy);
        gx = abs(gx);
        if (gz >= D) gz = 2 * D - 2 - gz;
        if (gy >= H) gy = 2 * H - 2 - gy;
        if (gx >= W) gx = 2 * W - 2 - gx;
        gz = min(max(gz, 0), D - 1);
        gy = min(max(gy, 0), H - 1);
        gx = min(max(gx, 0), W - 1);

        SMEM(sz, sy, sx) = input[gz * H * W + gy * W + gx];
    }
    __syncthreads();

    // Output voxel
    const int oz = blockIdx.z * TILE_3D_Z + threadIdx.z;
    const int oy = blockIdx.y * TILE_3D_Y + threadIdx.y;
    const int ox = blockIdx.x * TILE_3D_X + threadIdx.x;
    if (oz >= D || oy >= H || ox >= W) return;

    // Local coords in shared memory
    const int lz = threadIdx.z + HALO;
    const int ly = threadIdx.y + HALO;
    const int lx = threadIdx.x + HALO;

    float weighted_sum = 0.0f;
    float weight_total = 0.0f;

    for (int dz = -SEARCH_DIST; dz <= SEARCH_DIST; dz++) {
        for (int dy = -SEARCH_DIST; dy <= SEARCH_DIST; dy++) {
            for (int dx = -SEARCH_DIST; dx <= SEARCH_DIST; dx++) {
                float dist_raw = 0.0f;

                #pragma unroll
                for (int pz = -PATCH_HALF; pz <= PATCH_HALF; pz++) {
                    #pragma unroll
                    for (int py = -PATCH_HALF; py <= PATCH_HALF; py++) {
                        #pragma unroll
                        for (int px = -PATCH_HALF; px <= PATCH_HALF; px++) {
                            float c = SMEM(lz + pz, ly + py, lx + px);
                            float s = SMEM(lz + dz + pz, ly + dy + py, lx + dx + px);
                            float diff = c - s;
                            dist_raw += diff * diff;
                        }
                    }
                }

                // exp_scale = -1/(patch_vol * h^2)
                float w = expf(dist_raw * exp_scale);
                weighted_sum += w * SMEM(lz + dz, ly + dy, lx + dx);
                weight_total += w;
            }
        }
    }

    output[oz * H * W + oy * W + ox] = weighted_sum / fmaxf(weight_total, 1e-10f);

    #undef SMEM
}


// ============================================================================
// 2D Launcher
// ============================================================================

template <int PATCH_HALF, int SEARCH_DIST>
static torch::Tensor launch_nlm_2d(
    const torch::Tensor& input_padded,
    int H, int W, int H_pad, int W_pad, int pad, float exp_scale
) {
    auto output = torch::empty({H, W}, input_padded.options());

    dim3 block(TILE_2D_X, TILE_2D_Y);
    dim3 grid(
        (W + TILE_2D_X - 1) / TILE_2D_X,
        (H + TILE_2D_Y - 1) / TILE_2D_Y
    );

    nlm_2d_kernel<PATCH_HALF, SEARCH_DIST><<<grid, block>>>(
        input_padded.data_ptr<float>(),
        output.data_ptr<float>(),
        H, W, H_pad, W_pad, pad, exp_scale
    );

    return output;
}


// ============================================================================
// 3D Launcher
// ============================================================================

template <int PATCH_HALF, int SEARCH_DIST>
static torch::Tensor launch_nlm_3d(
    const torch::Tensor& input,  // original (NOT padded)
    int D, int H, int W, float exp_scale
) {
    auto output = torch::empty({D, H, W}, input.options());

    dim3 block(TILE_3D_X, TILE_3D_Y, TILE_3D_Z);
    dim3 grid(
        (W + TILE_3D_X - 1) / TILE_3D_X,
        (H + TILE_3D_Y - 1) / TILE_3D_Y,
        (D + TILE_3D_Z - 1) / TILE_3D_Z
    );

    // Compute dynamic shared memory size
    constexpr int HALO = SEARCH_DIST + PATCH_HALF;
    constexpr int SMEM_Z = TILE_3D_Z + 2 * HALO;
    constexpr int SMEM_Y = TILE_3D_Y + 2 * HALO;
    constexpr int SMEM_X = TILE_3D_X + 2 * HALO;
    constexpr int smem_bytes = SMEM_Z * SMEM_Y * SMEM_X * sizeof(float);

    // Request extended shared memory if needed (Ampere+ supports up to 164KB)
    if constexpr (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            nlm_3d_kernel<PATCH_HALF, SEARCH_DIST>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes
        );
    }

    nlm_3d_kernel<PATCH_HALF, SEARCH_DIST><<<grid, block, smem_bytes>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        D, H, W, exp_scale
    );

    return output;
}


// ============================================================================
// Runtime dispatch macros
// ============================================================================

// Dispatch macros — instantiate for all supported (patch_half, search_dist) combos.
// Search distances 5,7 (original) + 9,11,13,15 (extended for microscopy).
// Larger search distances need more shared memory (Ampere+ for 3D with sd>=11).

#define NLM_DISPATCH_2D(ph, sd, input_padded, H, W, H_pad, W_pad, pad, exp_scale)  \
    if ((ph) == 1 && (sd) == 5)        return launch_nlm_2d<1,  5>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else if ((ph) == 1 && (sd) == 7)   return launch_nlm_2d<1,  7>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else if ((ph) == 1 && (sd) == 9)   return launch_nlm_2d<1,  9>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else if ((ph) == 1 && (sd) == 11)  return launch_nlm_2d<1, 11>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else if ((ph) == 1 && (sd) == 13)  return launch_nlm_2d<1, 13>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else if ((ph) == 1 && (sd) == 15)  return launch_nlm_2d<1, 15>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else if ((ph) == 2 && (sd) == 5)   return launch_nlm_2d<2,  5>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else if ((ph) == 2 && (sd) == 7)   return launch_nlm_2d<2,  7>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else if ((ph) == 2 && (sd) == 9)   return launch_nlm_2d<2,  9>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else if ((ph) == 2 && (sd) == 11)  return launch_nlm_2d<2, 11>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else if ((ph) == 2 && (sd) == 13)  return launch_nlm_2d<2, 13>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else if ((ph) == 2 && (sd) == 15)  return launch_nlm_2d<2, 15>(input_padded, H, W, H_pad, W_pad, pad, exp_scale); \
    else TORCH_CHECK(false, "Unsupported NLM params: patch_half=", ph, " search_dist=", sd, \
                     ". Supported: patch_size={3,5}, search_distance={5,7,9,11,13,15}")

#define NLM_DISPATCH_3D(ph, sd, input, D, H, W, exp_scale)  \
    if ((ph) == 1 && (sd) == 5)        return launch_nlm_3d<1,  5>(input, D, H, W, exp_scale); \
    else if ((ph) == 1 && (sd) == 7)   return launch_nlm_3d<1,  7>(input, D, H, W, exp_scale); \
    else if ((ph) == 1 && (sd) == 9)   return launch_nlm_3d<1,  9>(input, D, H, W, exp_scale); \
    else if ((ph) == 1 && (sd) == 11)  return launch_nlm_3d<1, 11>(input, D, H, W, exp_scale); \
    else if ((ph) == 1 && (sd) == 13)  return launch_nlm_3d<1, 13>(input, D, H, W, exp_scale); \
    else if ((ph) == 1 && (sd) == 15)  return launch_nlm_3d<1, 15>(input, D, H, W, exp_scale); \
    else if ((ph) == 2 && (sd) == 5)   return launch_nlm_3d<2,  5>(input, D, H, W, exp_scale); \
    else if ((ph) == 2 && (sd) == 7)   return launch_nlm_3d<2,  7>(input, D, H, W, exp_scale); \
    else if ((ph) == 2 && (sd) == 9)   return launch_nlm_3d<2,  9>(input, D, H, W, exp_scale); \
    else if ((ph) == 2 && (sd) == 11)  return launch_nlm_3d<2, 11>(input, D, H, W, exp_scale); \
    else if ((ph) == 2 && (sd) == 13)  return launch_nlm_3d<2, 13>(input, D, H, W, exp_scale); \
    else if ((ph) == 2 && (sd) == 15)  return launch_nlm_3d<2, 15>(input, D, H, W, exp_scale); \
    else TORCH_CHECK(false, "Unsupported NLM params: patch_half=", ph, " search_dist=", sd, \
                     ". Supported: patch_size={3,5}, search_distance={5,7,9,11,13,15}")


// ============================================================================
// Public C++ API (called from bindings.cpp)
// ============================================================================

torch::Tensor nlm_denoise_2d(
    const torch::Tensor& input,
    float h,
    int patch_half,
    int search_dist
) {
    TORCH_CHECK(input.dim() == 2, "nlm_denoise_2d expects 2D tensor, got ", input.dim(), "D");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(h > 0, "h must be positive");

    // Make the input's device current for allocations and kernel launches.
    const c10::cuda::CUDAGuard device_guard(input.device());

    const int H = input.size(0);
    const int W = input.size(1);
    const int pad = search_dist + patch_half;

    // Reflection padding via torch::nn::functional
    namespace F = torch::nn::functional;
    auto input_padded = F::pad(
        input.unsqueeze(0).unsqueeze(0),
        F::PadFuncOptions({pad, pad, pad, pad}).mode(torch::kReflect)
    ).squeeze(0).squeeze(0).contiguous();

    const int H_pad = input_padded.size(0);
    const int W_pad = input_padded.size(1);

    // Precompute exp scale: folds normalisation + negation into one constant
    // exp(-dist_norm / h^2) = exp(-dist_raw / (patch_area * h^2)) = exp(dist_raw * scale)
    const int ps = 2 * patch_half + 1;
    const float exp_scale = -1.0f / (static_cast<float>(ps * ps) * h * h);

    NLM_DISPATCH_2D(patch_half, search_dist, input_padded, H, W, H_pad, W_pad, pad, exp_scale);
}

torch::Tensor nlm_denoise_3d(
    const torch::Tensor& input,
    float h,
    int patch_half,
    int search_dist
) {
    TORCH_CHECK(input.dim() == 3, "nlm_denoise_3d expects 3D tensor, got ", input.dim(), "D");
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(h > 0, "h must be positive");

    // Also applies the per-device shared-memory opt-in to the input's device.
    const c10::cuda::CUDAGuard device_guard(input.device());

    const int D = input.size(0);
    const int H = input.size(1);
    const int W = input.size(2);

    // Ensure contiguous layout
    auto input_contig = input.contiguous();

    // Precompute exp scale: -1 / (patch_vol * h^2)
    const int ps = 2 * patch_half + 1;
    const float exp_scale = -1.0f / (static_cast<float>(ps * ps * ps) * h * h);

    // Kernel handles boundary reflection internally — no pre-padding needed
    NLM_DISPATCH_3D(patch_half, search_dist, input_contig, D, H, W, exp_scale);
}
