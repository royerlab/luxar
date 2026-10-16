/**
 * Non-Local Means CUDA kernels (torch-free).
 *
 * Shared by the nvcc-built extension (nlm_cuda.cu) and the runtime-compiled
 * NVRTC backend (nlm_nvrtc.py), so the two paths run the SAME device code.
 * Nothing here may include a toolkit or C++ standard header: NVRTC has
 * neither, and every builtin used (min, max, abs, expf, fmaxf) is intrinsic.
 *
 * Template parameters:
 *   PATCH_HALF  — half the patch side length (compile-time for unrolling)
 *   SEARCH_DIST — half the search window side length (compile-time)
 */

#ifndef LUXAR_NLM_KERNELS_CUH
#define LUXAR_NLM_KERNELS_CUH

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

#endif // LUXAR_NLM_KERNELS_CUH
