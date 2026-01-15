/**
 * CUDA Global Splat Kernel Implementations
 *
 * This header contains kernels for handling "global" splats that touch too many tiles
 * (>10% of total tiles AND >1024 tiles). Instead of tile-based binning, we process
 * all pixels for each global splat.
 *
 * These kernels are less efficient than tile-based processing but correct, and
 * global splats are expected to be rare in typical use cases.
 *
 * Kernels:
 * - rasterize_global_forward_kernel: Forward pass for global splats
 * - rasterize_global_backward_kernel: Backward pass for global splats
 */

#ifndef CUDA_SPLATTING_KERNELS_GLOBAL_CUH
#define CUDA_SPLATTING_KERNELS_GLOBAL_CUH

#include "utils.cuh"

// =============================================================================
// GLOBAL SPLAT FORWARD KERNEL
// =============================================================================

/**
 * Forward kernel for global splats.
 *
 * Each thread processes one pixel. For each pixel, we iterate over all global
 * splats and accumulate their intensity contributions.
 *
 * Template parameter DIM: dimensionality (2-8)
 */
template <int DIM, typename InputDType = float>
__global__ void rasterize_global_forward_kernel(
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
    const InputDType* __restrict__ sharpness,
    const int* __restrict__ global_splat_ids,  // Array of global splat indices
    int n_global_splats,                        // Number of global splats
    const int* __restrict__ shape,
    float truncate,
    float intensity_floor,  // Minimum intensity threshold (must match tile-based kernel)
    float* __restrict__ output,  // Output to add to (already has tile-based contributions)
    int64_t num_pixels
) {
    int64_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= num_pixels) return;

    constexpr int CONIC_SIZE = conic_size<DIM>();

    // Convert linear index to voxel coordinates
    int voxel_coords[DIM];
    {
        int64_t remaining = pixel_idx;
        #pragma unroll
        for (int d = DIM - 1; d >= 0; d--) {
            voxel_coords[d] = (int)(remaining % shape[d]);
            remaining /= shape[d];
        }
    }

    float px[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        px[d] = (float)voxel_coords[d];
    }

    float intensity_sum = 0.0f;

    // Process all global splats
    for (int i = 0; i < n_global_splats; i++) {
        int splat_idx = global_splat_ids[i];

        // Load splat data with dtype-aware conversion
        float mu[DIM];
        float c[CONIC_SIZE];

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            mu[d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
        }
        #pragma unroll
        for (int ci = 0; ci < CONIC_SIZE; ci++) {
            c[ci] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + ci);
        }
        float amp = DTypeTraits<InputDType>::load(amps, splat_idx);
        float s = DTypeTraits<InputDType>::load(sharpness, splat_idx);

        // Compute displacement
        float d_vec[DIM];
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            d_vec[d] = px[d] - mu[d];
        }

        // Compute Mahalanobis distance squared
        float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, c);

        // Early culling based on effective truncation
        float eff_trunc_sq = effective_truncate_sq(truncate, s);
        if (dist_sq > eff_trunc_sq) continue;

        // Compute intensity
        float intensity = gaussian_intensity(dist_sq, amp, s);

        // Skip if below threshold (must match tile-based kernel behavior)
        if (intensity >= intensity_floor) {
            intensity_sum += intensity;
        }
    }

    // Add to output using atomicAdd (tile-based kernel may have already written)
    if (intensity_sum > 0.0f) {
        atomicAdd(&output[pixel_idx], intensity_sum);
    }
}

// =============================================================================
// GLOBAL SPLAT BACKWARD KERNEL
// =============================================================================

/**
 * Backward kernel for global splats.
 *
 * Each thread processes one pixel. Gradients are accumulated using atomicAdd.
 */
template <int DIM, typename InputDType = float>
__global__ void rasterize_global_backward_kernel(
    const float* __restrict__ grad_output,
    const InputDType* __restrict__ centers,
    const InputDType* __restrict__ conic,
    const InputDType* __restrict__ amps,
    const InputDType* __restrict__ sharpness,
    const int* __restrict__ global_splat_ids,
    int n_global_splats,
    const int* __restrict__ shape,
    float truncate,
    float intensity_floor,  // Minimum intensity threshold (must match tile-based kernel)
    float* __restrict__ d_centers,
    float* __restrict__ d_conic,
    float* __restrict__ d_amps,
    float* __restrict__ d_sharpness,
    int64_t num_pixels
) {
    int64_t pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= num_pixels) return;

    constexpr int CONIC_SIZE = conic_size<DIM>();

    float grad_out = grad_output[pixel_idx];
    if (fabsf(grad_out) < 1e-10f) return;  // Skip if no gradient

    // Convert linear index to voxel coordinates
    int voxel_coords[DIM];
    {
        int64_t remaining = pixel_idx;
        #pragma unroll
        for (int d = DIM - 1; d >= 0; d--) {
            voxel_coords[d] = (int)(remaining % shape[d]);
            remaining /= shape[d];
        }
    }

    float px[DIM];
    #pragma unroll
    for (int d = 0; d < DIM; d++) {
        px[d] = (float)voxel_coords[d];
    }

    // Process all global splats
    for (int i = 0; i < n_global_splats; i++) {
        int splat_idx = global_splat_ids[i];

        // Load splat data with dtype-aware conversion
        float mu[DIM];
        float c[CONIC_SIZE];

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            mu[d] = DTypeTraits<InputDType>::load(centers, splat_idx * DIM + d);
        }
        #pragma unroll
        for (int ci = 0; ci < CONIC_SIZE; ci++) {
            c[ci] = DTypeTraits<InputDType>::load(conic, splat_idx * CONIC_SIZE + ci);
        }
        float amp = DTypeTraits<InputDType>::load(amps, splat_idx);
        float s = DTypeTraits<InputDType>::load(sharpness, splat_idx);

        // Compute displacement
        float d_vec[DIM];
        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            d_vec[d] = px[d] - mu[d];
        }

        // Compute Mahalanobis distance squared
        float dist_sq = mahalanobis_distance_sq<DIM>(d_vec, c);

        // Early culling
        float eff_trunc_sq = effective_truncate_sq(truncate, s);
        if (dist_sq > eff_trunc_sq) continue;

        // Compute intensity and gradients
        float intensity = gaussian_intensity(dist_sq, amp, s);
        if (intensity < intensity_floor) continue;  // Must match tile-based kernel behavior

        // d_amp = grad_out * (I / a)
        float local_d_amp = grad_out * (intensity / fmaxf(amp, 1e-10f));

        // Gradient w.r.t. dist_sq: dI/dD^2 = I x (-0.25 x s) x (D^2)^(s/2 - 1)
        // For s=2: dI/dD^2 = -0.5 x I
        float dist_sq_safe = fmaxf(dist_sq, 1e-12f);
        float dI_dD_sq;
        if (fabsf(s - 2.0f) < 1e-4f) {
            // Standard Gaussian: dI/dD^2 = -0.5 * intensity
            dI_dD_sq = -0.5f * intensity;
        } else {
            // Generalized Gaussian: dI/dD^2 = I x (-0.25 x s) x (D^2)^(s/2 - 1)
            float dist_pow_s_minus_1 = __powf(dist_sq_safe, s * 0.5f - 1.0f);
            dI_dD_sq = intensity * (-0.25f * s) * dist_pow_s_minus_1;
        }

        float outer_grad = grad_out * dI_dD_sq;

        // d_centers: dD^2/dmu = -dD^2/dd = -2 x Sigma^-1 @ d
        // Note: No factor of 2 on off-diagonal conic elements when computing Sigma^-1 @ d
        float local_d_centers[DIM];
        #pragma unroll
        for (int di = 0; di < DIM; di++) {
            float sum = 0.0f;
            // Sum over conic contributions (symmetric matrix in packed upper triangle)
            for (int dj = 0; dj < DIM; dj++) {
                int ci = (di <= dj) ?
                    (di * (2 * DIM - di - 1) / 2 + dj - di) :
                    (dj * (2 * DIM - dj - 1) / 2 + di - dj);
                sum += c[ci] * d_vec[dj];
            }
            // dL/dcenter = grad_out x dI_dD_sq x (-2) x (Sigma^-1 @ d)
            local_d_centers[di] = outer_grad * (-2.0f) * sum;
        }

        // d_conic: dD^2/dc_ij = d_i x d_j (diagonal) or 2 x d_i x d_j (off-diagonal)
        float local_d_conic[CONIC_SIZE];
        int ci = 0;
        #pragma unroll
        for (int di = 0; di < DIM; di++) {
            for (int dj = di; dj < DIM; dj++) {
                float factor = (di == dj) ? 1.0f : 2.0f;
                local_d_conic[ci] = outer_grad * factor * d_vec[di] * d_vec[dj];
                ci++;
            }
        }

        // d_sharpness: dI/ds = I x (-0.25) x (D^2)^(s/2) x ln(D^2)
        float local_d_sharpness = 0.0f;
        if (dist_sq > 1e-6f) {
            float dist_pow_s = __powf(dist_sq_safe, s * 0.5f);
            float log_dist_sq = __logf(dist_sq_safe);
            local_d_sharpness = grad_out * intensity * (-0.25f) * dist_pow_s * log_dist_sq;
        }

        // Accumulate gradients using atomicAdd
        atomicAdd(&d_amps[splat_idx], local_d_amp);
        atomicAdd(&d_sharpness[splat_idx], local_d_sharpness);

        #pragma unroll
        for (int d = 0; d < DIM; d++) {
            atomicAdd(&d_centers[splat_idx * DIM + d], local_d_centers[d]);
        }
        #pragma unroll
        for (int ci_idx = 0; ci_idx < CONIC_SIZE; ci_idx++) {
            atomicAdd(&d_conic[splat_idx * CONIC_SIZE + ci_idx], local_d_conic[ci_idx]);
        }
    }
}

#endif // CUDA_SPLATTING_KERNELS_GLOBAL_CUH
