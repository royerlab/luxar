// kernels.metal
// Splat-centric Metal kernels for 3D FP32 Gaussian splatting.
//
// Coordinate convention: all centers, conics, and output indices use Luxar's
// PyTorch/NumPy volume order [Z, Y, X].  For 3D packed conics this means:
//   [c_zz, c_zy, c_zx, c_yy, c_yx, c_xx]
// which is the standard row-major upper-triangle packing for axes [Z, Y, X].

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#define THREADGROUP_SIZE 64u

// ============================================================================
// Helpers
// ============================================================================

inline void atomic_add_float(device atomic_float* addr, float val) {
    atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
}

inline float shift_c(float truncate) {
    return fast::exp(-0.5f * truncate * truncate);
}

inline float gaussian_intensity(
    float dist_sq,
    float amp,
    float shift_C,
    float inv_one_minus_C
) {
    return amp * inv_one_minus_C * (fast::exp(-0.5f * dist_sq) - shift_C);
}

inline float effective_truncate_sq(
    float truncate,
    float amp,
    float intensity_floor,
    float shift_C,
    float inv_one_minus_C
) {
    float threshold = intensity_floor / max(amp * inv_one_minus_C, 1e-10f) + shift_C;
    if (threshold >= 1.0f) {
        return 0.0f;
    }
    return min(truncate * truncate, -2.0f * fast::log(threshold));
}

inline float effective_truncation(
    float truncate,
    float amp,
    float intensity_floor,
    float shift_C,
    float inv_one_minus_C
) {
    float threshold = intensity_floor / max(amp * inv_one_minus_C, 1e-10f) + shift_C;
    if (threshold >= 1.0f) {
        return 0.0f;
    }
    return min(truncate, fast::sqrt(-2.0f * fast::log(threshold)));
}

inline float mahalanobis_distance_sq_3d(
    float dz,
    float dy,
    float dx,
    float c00,
    float c01,
    float c02,
    float c11,
    float c12,
    float c22
) {
    return c00 * dz * dz + c11 * dy * dy + c22 * dx * dx
         + 2.0f * (c01 * dz * dy + c02 * dz * dx + c12 * dy * dx);
}

inline float3 sigma_diag_sqrt_from_conic_3d(
    float c00,
    float c01,
    float c02,
    float c11,
    float c12,
    float c22
) {
    float det = c00 * (c11 * c22 - c12 * c12)
              - c01 * (c01 * c22 - c02 * c12)
              + c02 * (c01 * c12 - c02 * c11);
    float inv_det = 1.0f / max(abs(det), 1e-10f);

    return float3(
        fast::sqrt(max((c11 * c22 - c12 * c12) * inv_det, 0.0f)),
        fast::sqrt(max((c00 * c22 - c02 * c02) * inv_det, 0.0f)),
        fast::sqrt(max((c00 * c11 - c01 * c01) * inv_det, 0.0f))
    );
}

inline void compute_aabb_3d_from_sigma(
    threadgroup int* lo,
    threadgroup int* extent,
    threadgroup uint& total_voxels,
    float center_z,
    float center_y,
    float center_x,
    float sigma_z,
    float sigma_y,
    float sigma_x,
    float t_eff,
    uint D,
    uint H,
    uint W
) {
    if (t_eff <= 0.0f) {
        lo[0] = lo[1] = lo[2] = 0;
        extent[0] = extent[1] = extent[2] = 0;
        total_voxels = 0;
        return;
    }

    int radius_z = int(ceil(t_eff * sigma_z));
    int radius_y = int(ceil(t_eff * sigma_y));
    int radius_x = int(ceil(t_eff * sigma_x));

    int lo_z = max(0, int(floor(center_z)) - radius_z);
    int lo_y = max(0, int(floor(center_y)) - radius_y);
    int lo_x = max(0, int(floor(center_x)) - radius_x);

    int hi_z = min(int(D) - 1, int(ceil(center_z)) + radius_z);
    int hi_y = min(int(H) - 1, int(ceil(center_y)) + radius_y);
    int hi_x = min(int(W) - 1, int(ceil(center_x)) + radius_x);

    lo[0] = lo_z;
    lo[1] = lo_y;
    lo[2] = lo_x;
    extent[0] = max(0, hi_z - lo_z + 1);
    extent[1] = max(0, hi_y - lo_y + 1);
    extent[2] = max(0, hi_x - lo_x + 1);
    // MET-3: compute the voxel total in uint so a wide splat in a 1290³+
    // volume (host validator hard-caps at 2e9 voxels) cannot silently
    // overflow a 32-bit signed integer and produce a negative loop bound.
    total_voxels = uint(extent[0]) * uint(extent[1]) * uint(extent[2]);
}

// Conic-input AABB: thin wrapper around compute_aabb_3d_from_sigma that
// derives the per-axis sigma from the packed conic. Used by the constrained
// (precomputed-conic) splat-centric kernels.
inline void compute_aabb_3d(
    threadgroup int* lo,
    threadgroup int* extent,
    threadgroup uint& total_voxels,
    float center_z,
    float center_y,
    float center_x,
    float c00,
    float c01,
    float c02,
    float c11,
    float c12,
    float c22,
    float t_eff,
    uint D,
    uint H,
    uint W
) {
    float3 sigma = sigma_diag_sqrt_from_conic_3d(c00, c01, c02, c11, c12, c22);
    compute_aabb_3d_from_sigma(
        lo, extent, total_voxels,
        center_z, center_y, center_x,
        sigma.x, sigma.y, sigma.z,
        t_eff, D, H, W);
}

// ============================================================================
// Utility kernels
// ============================================================================

kernel void zero_float_buffer(
    device float* output [[buffer(0)]],
    constant uint& total [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < total) {
        output[id] = 0.0f;
    }
}

// ============================================================================
// Optional 3D L -> conic conversion
// ============================================================================

kernel void compute_conic_from_L_3d(
    device const float* Ls [[buffer(0)]],
    device float* conic [[buffer(1)]],
    constant uint& n_splats [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_splats) {
        return;
    }

    int base = int(id) * 9;
    float l00 = Ls[base + 0];
    float l10 = Ls[base + 3];
    float l11 = Ls[base + 4];
    float l20 = Ls[base + 6];
    float l21 = Ls[base + 7];
    float l22 = Ls[base + 8];

    float k00 = 1.0f / (l00 + 1e-9f);
    float k11 = 1.0f / (l11 + 1e-9f);
    float k22 = 1.0f / (l22 + 1e-9f);
    float k10 = -l10 * k00 * k11;
    float k21 = -l21 * k11 * k22;
    float k20 = -(l20 * k00 + l21 * k10) * k22;

    float c00 = k00 * k00 + k10 * k10 + k20 * k20;
    float c01 = k10 * k11 + k20 * k21;
    float c02 = k20 * k22;
    float c11 = k11 * k11 + k21 * k21;
    float c12 = k21 * k22;
    float c22 = k22 * k22;

    int out = int(id) * 6;
    conic[out + 0] = c00;
    conic[out + 1] = c01;
    conic[out + 2] = c02;
    conic[out + 3] = c11;
    conic[out + 4] = c12;
    conic[out + 5] = c22;
}

// ============================================================================
// Splat-centric forward: one threadgroup owns one splat
// ============================================================================

kernel void rasterize_forward_splat_centric_3d(
    device const float* centers [[buffer(0)]],
    device const float* conic [[buffer(1)]],
    device const float* amps [[buffer(2)]],
    device atomic_float* output [[buffer(3)]],
    constant uint3& shape_dhw [[buffer(4)]],
    constant uint& n_splats [[buffer(5)]],
    constant float& truncate [[buffer(6)]],
    constant float& intensity_floor [[buffer(7)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint splat_id = tg_pos.x;
    if (splat_id >= n_splats) {
        return;
    }

    threadgroup float s_center[3];
    threadgroup float s_conic[6];
    threadgroup float s_amp;
    threadgroup float s_shift_C;
    threadgroup float s_inv_one_minus_C;
    threadgroup float s_truncate_sq;
    threadgroup int s_lo[3];
    threadgroup int s_extent[3];
    // MET-3: counter is uint to match compute_aabb_3d's `uint& total_voxels`
    // signature and to prevent a 1290³+ volume from overflowing the loop
    // bound. Threadgroup-storage initializers are technically UB per the
    // MSL spec; thread 0 always writes via compute_aabb_3d before the
    // barrier below, so other threads observe the assigned value.
    threadgroup uint s_total_voxels;

    if (tid == 0) {
        int base3 = int(splat_id) * 3;
        int base6 = int(splat_id) * 6;

        s_center[0] = centers[base3 + 0];
        s_center[1] = centers[base3 + 1];
        s_center[2] = centers[base3 + 2];
        for (uint k = 0; k < 6; ++k) {
            s_conic[k] = conic[base6 + int(k)];
        }
        s_amp = amps[splat_id];

        s_shift_C = shift_c(truncate);
        s_inv_one_minus_C = 1.0f / (1.0f - s_shift_C);
        s_truncate_sq = effective_truncate_sq(
            truncate, s_amp, intensity_floor, s_shift_C, s_inv_one_minus_C);
        float t_eff = effective_truncation(
            truncate, s_amp, intensity_floor, s_shift_C, s_inv_one_minus_C);

        compute_aabb_3d(
            s_lo,
            s_extent,
            s_total_voxels,
            s_center[0],
            s_center[1],
            s_center[2],
            s_conic[0],
            s_conic[1],
            s_conic[2],
            s_conic[3],
            s_conic[4],
            s_conic[5],
            t_eff,
            shape_dhw.x,
            shape_dhw.y,
            shape_dhw.z);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint total = s_total_voxels;
    if (total == 0u) {
        return;
    }

    int extent_yx = s_extent[1] * s_extent[2];
    uint H = shape_dhw.y;
    uint W = shape_dhw.z;

    for (uint local = tid; local < total; local += uint(THREADGROUP_SIZE)) {
        int local_int = int(local);
        int z = s_lo[0] + local_int / extent_yx;
        int rem = local_int - (z - s_lo[0]) * extent_yx;
        int y = s_lo[1] + rem / s_extent[2];
        int x = s_lo[2] + rem - (y - s_lo[1]) * s_extent[2];

        float dz = float(z) - s_center[0];
        float dy = float(y) - s_center[1];
        float dx = float(x) - s_center[2];
        float dist_sq = mahalanobis_distance_sq_3d(
            dz, dy, dx,
            s_conic[0], s_conic[1], s_conic[2],
            s_conic[3], s_conic[4], s_conic[5]);

        if (dist_sq > s_truncate_sq) {
            continue;
        }

        float intensity = gaussian_intensity(
            dist_sq, s_amp, s_shift_C, s_inv_one_minus_C);
        if (intensity < intensity_floor) {
            continue;
        }

        uint out_idx = uint(z) * H * W + uint(y) * W + uint(x);
        atomic_add_float(&output[out_idx], intensity);
    }
}

// ============================================================================
// Splat-centric backward: one threadgroup owns one splat gradient
// ============================================================================

kernel void rasterize_backward_splat_centric_3d(
    device const float* grad_output [[buffer(0)]],
    device const float* centers [[buffer(1)]],
    device const float* conic [[buffer(2)]],
    device const float* amps [[buffer(3)]],
    device float* d_centers [[buffer(4)]],
    device float* d_conic [[buffer(5)]],
    device float* d_amps [[buffer(6)]],
    constant uint3& shape_dhw [[buffer(7)]],
    constant uint& n_splats [[buffer(8)]],
    constant float& truncate [[buffer(9)]],
    constant float& intensity_floor [[buffer(10)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint splat_id = tg_pos.x;
    if (splat_id >= n_splats) {
        return;
    }

    threadgroup float s_center[3];
    threadgroup float s_conic[6];
    threadgroup float s_amp;
    threadgroup float s_shift_C;
    threadgroup float s_inv_one_minus_C;
    threadgroup float s_truncate_sq;
    threadgroup int s_lo[3];
    threadgroup int s_extent[3];
    // MET-3: see forward kernel — counter is uint to match
    // compute_aabb_3d's signature and prevent overflow.
    threadgroup uint s_total_voxels;

    threadgroup float tg_amp[THREADGROUP_SIZE];
    threadgroup float tg_centers[THREADGROUP_SIZE * 3];
    threadgroup float tg_conic[THREADGROUP_SIZE * 6];

    if (tid == 0) {
        int base3 = int(splat_id) * 3;
        int base6 = int(splat_id) * 6;

        s_center[0] = centers[base3 + 0];
        s_center[1] = centers[base3 + 1];
        s_center[2] = centers[base3 + 2];
        for (uint k = 0; k < 6; ++k) {
            s_conic[k] = conic[base6 + int(k)];
        }
        s_amp = amps[splat_id];

        s_shift_C = shift_c(truncate);
        s_inv_one_minus_C = 1.0f / (1.0f - s_shift_C);
        s_truncate_sq = effective_truncate_sq(
            truncate, s_amp, intensity_floor, s_shift_C, s_inv_one_minus_C);
        float t_eff = effective_truncation(
            truncate, s_amp, intensity_floor, s_shift_C, s_inv_one_minus_C);

        compute_aabb_3d(
            s_lo,
            s_extent,
            s_total_voxels,
            s_center[0],
            s_center[1],
            s_center[2],
            s_conic[0],
            s_conic[1],
            s_conic[2],
            s_conic[3],
            s_conic[4],
            s_conic[5],
            t_eff,
            shape_dhw.x,
            shape_dhw.y,
            shape_dhw.z);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_amp = 0.0f;
    float local_center_0 = 0.0f;
    float local_center_1 = 0.0f;
    float local_center_2 = 0.0f;
    float local_conic_0 = 0.0f;
    float local_conic_1 = 0.0f;
    float local_conic_2 = 0.0f;
    float local_conic_3 = 0.0f;
    float local_conic_4 = 0.0f;
    float local_conic_5 = 0.0f;

    uint total = s_total_voxels;
    if (total > 0u) {
        int extent_yx = s_extent[1] * s_extent[2];
        uint H = shape_dhw.y;
        uint W = shape_dhw.z;

        for (uint local = tid; local < total; local += uint(THREADGROUP_SIZE)) {
            int local_int = int(local);
            int z = s_lo[0] + local_int / extent_yx;
            int rem = local_int - (z - s_lo[0]) * extent_yx;
            int y = s_lo[1] + rem / s_extent[2];
            int x = s_lo[2] + rem - (y - s_lo[1]) * s_extent[2];

            uint out_idx = uint(z) * H * W + uint(y) * W + uint(x);
            float dL_dI = grad_output[out_idx];
            if (dL_dI == 0.0f) {
                continue;
            }

            float dz = float(z) - s_center[0];
            float dy = float(y) - s_center[1];
            float dx = float(x) - s_center[2];

            float c00 = s_conic[0];
            float c01 = s_conic[1];
            float c02 = s_conic[2];
            float c11 = s_conic[3];
            float c12 = s_conic[4];
            float c22 = s_conic[5];

            float dist_sq = mahalanobis_distance_sq_3d(
                dz, dy, dx, c00, c01, c02, c11, c12, c22);
            if (dist_sq > s_truncate_sq) {
                continue;
            }

            float intensity = gaussian_intensity(
                dist_sq, s_amp, s_shift_C, s_inv_one_minus_C);
            if (intensity < intensity_floor) {
                continue;
            }

            local_amp += dL_dI * intensity / max(s_amp, 1e-10f);

            float unshifted = intensity + s_amp * s_inv_one_minus_C * s_shift_C;
            float outer = dL_dI * unshifted * (-0.5f);

            float dD2_dz = 2.0f * (c00 * dz + c01 * dy + c02 * dx);
            float dD2_dy = 2.0f * (c01 * dz + c11 * dy + c12 * dx);
            float dD2_dx = 2.0f * (c02 * dz + c12 * dy + c22 * dx);

            local_center_0 -= outer * dD2_dz;
            local_center_1 -= outer * dD2_dy;
            local_center_2 -= outer * dD2_dx;

            local_conic_0 += outer * dz * dz;
            local_conic_1 += outer * 2.0f * dz * dy;
            local_conic_2 += outer * 2.0f * dz * dx;
            local_conic_3 += outer * dy * dy;
            local_conic_4 += outer * 2.0f * dy * dx;
            local_conic_5 += outer * dx * dx;
        }
    }

    tg_amp[tid] = local_amp;
    tg_centers[tid * 3 + 0] = local_center_0;
    tg_centers[tid * 3 + 1] = local_center_1;
    tg_centers[tid * 3 + 2] = local_center_2;
    tg_conic[tid * 6 + 0] = local_conic_0;
    tg_conic[tid * 6 + 1] = local_conic_1;
    tg_conic[tid * 6 + 2] = local_conic_2;
    tg_conic[tid * 6 + 3] = local_conic_3;
    tg_conic[tid * 6 + 4] = local_conic_4;
    tg_conic[tid * 6 + 5] = local_conic_5;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = THREADGROUP_SIZE >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tg_amp[tid] += tg_amp[tid + stride];
            tg_centers[tid * 3 + 0] += tg_centers[(tid + stride) * 3 + 0];
            tg_centers[tid * 3 + 1] += tg_centers[(tid + stride) * 3 + 1];
            tg_centers[tid * 3 + 2] += tg_centers[(tid + stride) * 3 + 2];
            tg_conic[tid * 6 + 0] += tg_conic[(tid + stride) * 6 + 0];
            tg_conic[tid * 6 + 1] += tg_conic[(tid + stride) * 6 + 1];
            tg_conic[tid * 6 + 2] += tg_conic[(tid + stride) * 6 + 2];
            tg_conic[tid * 6 + 3] += tg_conic[(tid + stride) * 6 + 3];
            tg_conic[tid * 6 + 4] += tg_conic[(tid + stride) * 6 + 4];
            tg_conic[tid * 6 + 5] += tg_conic[(tid + stride) * 6 + 5];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        int base3 = int(splat_id) * 3;
        int base6 = int(splat_id) * 6;
        d_centers[base3 + 0] = tg_centers[0];
        d_centers[base3 + 1] = tg_centers[1];
        d_centers[base3 + 2] = tg_centers[2];
        d_conic[base6 + 0] = tg_conic[0];
        d_conic[base6 + 1] = tg_conic[1];
        d_conic[base6 + 2] = tg_conic[2];
        d_conic[base6 + 3] = tg_conic[3];
        d_conic[base6 + 4] = tg_conic[4];
        d_conic[base6 + 5] = tg_conic[5];
        d_amps[splat_id] = tg_amp[0];
    }
}
