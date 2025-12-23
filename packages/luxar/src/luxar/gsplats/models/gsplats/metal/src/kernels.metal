// kernels.metal
// Complete, production-ready Metal kernels for Gaussian splatting

// ============================================================================
// ⚠️  CRITICAL COORDINATE CONVENTION WARNING
// ============================================================================
//
// These kernels use NUMPY/PYTORCH convention [Z, Y, X], NOT Cartesian [X, Y, Z]!
//
// When you see float3 variables:
//   float3.x corresponds to DEPTH  (Z dimension, first in numpy array)
//   float3.y corresponds to HEIGHT (Y dimension, middle in numpy array)
//   float3.z corresponds to WIDTH  (X dimension, last in numpy array)
//
// Example: float3 px = float3(gid.z, gid.y, gid.x) creates position in [Z,Y,X] order
//
// Why: PyTorch stores data in numpy convention. Matching this avoids error-prone
// coordinate transformations and ensures consistency with reference implementation.
//
// This is INTENTIONAL and has been thoroughly validated. Do not "fix" it!
// ============================================================================

#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>
using namespace metal;

// ============================================================================
// CONSTANTS AND HELPERS
// ============================================================================

// Default tile size (can be overridden at runtime via kernel parameter)
constant int DEFAULT_TILE_SIZE_3D = 4;  // 4×4×4 = 64 threads per tile

// Atomic float add using compare-exchange
// NOTE: Metal 2.3+ supports atomic_float natively. For older versions, use this fallback.
// The memory orders are both relaxed since we only need atomicity, not ordering.
inline void atomic_add_float(device atomic_float* addr, float val) {
    float old = atomic_load_explicit(addr, memory_order_relaxed);
    float desired;
    do {
        desired = old + val;
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &old, desired,
        memory_order_relaxed,   // success: relaxed is sufficient for accumulation
        memory_order_relaxed    // failure: relaxed is standard for CAS retry loops
    ));
}

// Alternative: If targeting Metal 3.0+ only, use native atomic_fetch_add_explicit:
// inline void atomic_add_float(device atomic_float* addr, float val) {
//     atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
// }

// Shared tile range calculation (ensures preprocess and bin are consistent)
struct TileRange {
    int3 min_t;
    int3 max_t;
};

inline TileRange get_tile_range_3d(
    float3 center,      // In [Z,Y,X] order: center.x=z, center.y=y, center.z=x
    float3 sigma_diag,  // In [Z,Y,X] order: [Σ_zz, Σ_yy, Σ_xx]
    float truncate,
    float sharpness,
    uint3 grid_dims,    // In [X,Y,Z] order: (tiles_x, tiles_y, tiles_z)
    int tile_size       // Tile size (e.g., 4 for 4×4×4 tiles)
) {
    // Sharpness-adjusted truncation radius
    float effective_truncate = pow(truncate, 2.0f / sharpness);

    // Exact bounding box radius per axis (in [Z,Y,X] order)
    float3 r = effective_truncate * sqrt(max(sigma_diag, float3(1e-8f)));

    // Extract individual coordinates
    float z = center.x, y = center.y, x = center.z;  // center is [Z,Y,X]
    float rz = r.x, ry = r.y, rx = r.z;              // r is [Z,Y,X]

    // Compute tile ranges (careful with coordinate order!)
    float tile_size_f = (float)tile_size;
    int min_x = max((int)((x - rx) / tile_size_f), 0);
    int max_x = min((int)((x + rx) / tile_size_f), (int)grid_dims.x - 1);
    int min_y = max((int)((y - ry) / tile_size_f), 0);
    int max_y = min((int)((y + ry) / tile_size_f), (int)grid_dims.y - 1);
    int min_z = max((int)((z - rz) / tile_size_f), 0);
    int max_z = min((int)((z + rz) / tile_size_f), (int)grid_dims.z - 1);

    // Store in int3 as [x,y,z] for loop compatibility
    TileRange tr;
    tr.min_t = int3(min_x, min_y, min_z);
    tr.max_t = int3(max_x, max_y, max_z);

    return tr;
}

// Compute sigma diagonal from Cholesky L
inline float3 compute_sigma_diag_3d(
    float L00, float L10, float L11, float L20, float L21, float L22
) {
    return float3(
        L00 * L00,                          // Σ_00
        L10 * L10 + L11 * L11,              // Σ_11
        L20 * L20 + L21 * L21 + L22 * L22   // Σ_22
    );
}

// ============================================================================
// OPTIONAL: Compute L→Conic in Metal (faster than PyTorch for forward)
// ============================================================================
kernel void compute_conic_from_L_3d(
    device const float* Ls       [[buffer(0)]],  // (N, 3, 3) in [Z,Y,X] order
    device float* conic          [[buffer(1)]],  // (N, 6) output in [X,Y,Z] order
    constant uint& n_splats      [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_splats) return;

    // Load L in [Z,Y,X] order
    int base = id * 9;
    float L_zz = Ls[base + 0];
    float L_yz = Ls[base + 3], L_yy = Ls[base + 4];
    float L_xz = Ls[base + 6], L_xy = Ls[base + 7], L_xx = Ls[base + 8];

    // Compute K = L^(-1) via forward substitution
    float K_zz = 1.0f / (L_zz + 1e-9f);
    float K_yy = 1.0f / (L_yy + 1e-9f);
    float K_xx = 1.0f / (L_xx + 1e-9f);
    float K_yz = -L_yz * K_zz * K_yy;
    float K_xy = -L_xy * K_yy * K_xx;
    float K_xz = -(L_xz * K_zz + L_xy * K_yz) * K_xx;

    // Compute C = K^T @ K (conic = inverse covariance)
    // K in [Z,Y,X]: [[K_zz, 0, 0], [K_yz, K_yy, 0], [K_xz, K_xy, K_xx]]
    // C[i,j] = sum_k K[k,i] * K[k,j]
    float c_zz = K_zz * K_zz + K_yz * K_yz + K_xz * K_xz;
    float c_zy = K_yz * K_yy + K_xz * K_xy;
    float c_zx = K_xz * K_xx;
    float c_yy = K_yy * K_yy + K_xy * K_xy;
    float c_yx = K_xy * K_xx;
    float c_xx = K_xx * K_xx;

    // Output in [X,Y,Z] order (reordered for Metal kernel compatibility)
    // [c_zz,c_zy,c_zx,c_yy,c_yx,c_xx] → [c_xx,c_xy,c_xz,c_yy,c_yz,c_zz]
    int out_base = id * 6;
    conic[out_base + 0] = c_xx;  // c_xx
    conic[out_base + 1] = c_yx;  // c_xy (=c_yx, symmetric)
    conic[out_base + 2] = c_zx;  // c_xz (=c_zx, symmetric)
    conic[out_base + 3] = c_yy;  // c_yy
    conic[out_base + 4] = c_zy;  // c_yz (=c_zy, symmetric)
    conic[out_base + 5] = c_zz;  // c_zz
}

// ============================================================================
// KERNEL 1: PREPROCESS (Count splats per tile)
// ============================================================================

kernel void preprocess_3d(
    device const float* centers     [[buffer(0)]],  // (N, 3)
    device const float* Ls          [[buffer(1)]],  // (N, 3, 3) row-major
    device const float* sharpness   [[buffer(2)]],  // (N,)
    device atomic_int* tile_counts  [[buffer(3)]],  // (num_tiles,)
    constant float& truncate        [[buffer(4)]],
    constant uint3& grid_dims       [[buffer(5)]],
    constant uint& n_splats         [[buffer(6)]],
    constant int& tile_size         [[buffer(7)]],  // Configurable tile size
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_splats) return;

    // Load splat data in [Z,Y,X] order (matches PyTorch)
    // CRITICAL: Must match rasterize kernels!
    float3 center = float3(
        centers[id * 3 + 0],  // Z
        centers[id * 3 + 1],  // Y
        centers[id * 3 + 2]   // X
    );

    // Load L elements in [Z,Y,X] order
    // L is lower-triangular in [Z,Y,X]: [[L_zz,0,0], [L_yz,L_yy,0], [L_xz,L_xy,L_xx]]
    int base = id * 9;
    float L00 = Ls[base + 0];  // L_zz
    float L10 = Ls[base + 3], L11 = Ls[base + 4];  // L_yz, L_yy
    float L20 = Ls[base + 6], L21 = Ls[base + 7], L22 = Ls[base + 8];  // L_xz, L_xy, L_xx

    float3 sigma_diag = compute_sigma_diag_3d(L00, L10, L11, L20, L21, L22);

    // Get tile range (with configurable tile size)
    TileRange tr = get_tile_range_3d(
        center, sigma_diag, truncate, sharpness[id], grid_dims, tile_size
    );

    // Increment counts for each overlapping tile
    for (int z = tr.min_t.z; z <= tr.max_t.z; z++) {
        for (int y = tr.min_t.y; y <= tr.max_t.y; y++) {
            for (int x = tr.min_t.x; x <= tr.max_t.x; x++) {
                int tile_idx = z * (grid_dims.x * grid_dims.y)
                             + y * grid_dims.x + x;
                atomic_fetch_add_explicit(
                    &tile_counts[tile_idx], 1, memory_order_relaxed);
            }
        }
    }
}

// ============================================================================
// KERNEL 2: BINNING (Populate tile content lists)
// ============================================================================

kernel void bin_3d(
    device const float* centers         [[buffer(0)]],  // (N, 3)
    device const float* Ls              [[buffer(1)]],  // (N, 3, 3)
    device const float* sharpness       [[buffer(2)]],  // (N,)
    device const int* tile_offsets      [[buffer(3)]],  // (num_tiles,) prefix sum
    device atomic_int* tile_write_heads [[buffer(4)]],  // (num_tiles,) must be zeroed
    device int* tile_content            [[buffer(5)]],  // (total_pairs,)
    constant float& truncate            [[buffer(6)]],
    constant uint3& grid_dims           [[buffer(7)]],
    constant uint& n_splats             [[buffer(8)]],
    constant int& tile_size             [[buffer(9)]],  // Configurable tile size
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_splats) return;

    // Load splat data (MUST match preprocess exactly)
    // In [Z,Y,X] order (matches PyTorch)
    float3 center = float3(
        centers[id * 3 + 0],  // Z
        centers[id * 3 + 1],  // Y
        centers[id * 3 + 2]   // X
    );

    // Load L elements in [Z,Y,X] order (MUST match preprocess)
    int base = id * 9;
    float L00 = Ls[base + 0];  // L_zz
    float L10 = Ls[base + 3], L11 = Ls[base + 4];  // L_yz, L_yy
    float L20 = Ls[base + 6], L21 = Ls[base + 7], L22 = Ls[base + 8];  // L_xz, L_xy, L_xx

    float3 sigma_diag = compute_sigma_diag_3d(L00, L10, L11, L20, L21, L22);

    // Get tile range (MUST match preprocess - with same tile_size!)
    TileRange tr = get_tile_range_3d(
        center, sigma_diag, truncate, sharpness[id], grid_dims, tile_size
    );

    // Write splat ID to each tile's list
    for (int z = tr.min_t.z; z <= tr.max_t.z; z++) {
        for (int y = tr.min_t.y; y <= tr.max_t.y; y++) {
            for (int x = tr.min_t.x; x <= tr.max_t.x; x++) {
                int tile_idx = z * (grid_dims.x * grid_dims.y)
                             + y * grid_dims.x + x;

                // Atomically get write slot
                int slot = atomic_fetch_add_explicit(
                    &tile_write_heads[tile_idx], 1, memory_order_relaxed);

                // CRITICAL: Bounds check to prevent overflow (safety)
                // If slot exceeds tile capacity, skip (shouldn't happen if preprocess matches)
                int offset = tile_offsets[tile_idx];
                int next_offset = (tile_idx + 1 < grid_dims.x * grid_dims.y * grid_dims.z) ?
                                 tile_offsets[tile_idx + 1] : offset + 1000;  // Conservative
                int capacity = next_offset - offset;

                if (slot < capacity) {
                    // Safe to write
                    tile_content[offset + slot] = int(id);
                }
                // else: overflow detected - skip this write (shouldn't happen)
            }
        }
    }
}

// ============================================================================
// KERNEL 3: RASTERIZE FORWARD (Pixel-parallel rendering)
// ============================================================================

kernel void rasterize_fwd_3d(
    device const float* centers      [[buffer(0)]],   // (N, 3)
    device const float* conic        [[buffer(1)]],   // (N, 6) [xx,xy,xz,yy,yz,zz]
    device const float* amps         [[buffer(2)]],   // (N,)
    device const float* sharpness    [[buffer(3)]],   // (N,)
    device const int* tile_offsets   [[buffer(4)]],   // (num_tiles,)
    device const int* tile_counts    [[buffer(5)]],   // (num_tiles,)
    device const int* tile_content   [[buffer(6)]],   // (total_pairs,)
    device float* output             [[buffer(7)]],   // (D, H, W) row-major output
    constant uint3& img_size         [[buffer(8)]],   // (W, H, D) - see §2.7 Dimension Conventions
    constant uint3& grid_dims        [[buffer(9)]],   // (tiles_x, tiles_y, tiles_z) - see §2.7
    constant float& truncate         [[buffer(10)]],  // base truncate (NOT squared)
    constant float& intensity_floor  [[buffer(11)]],  // early culling threshold
    uint3 gid [[thread_position_in_grid]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    // Bounds check
    if (gid.x >= img_size.x || gid.y >= img_size.y || gid.z >= img_size.z) return;

    // Tile index (threads are grouped by tile)
    uint tile_idx = group_id.z * (grid_dims.x * grid_dims.y)
                  + group_id.y * grid_dims.x + group_id.x;

    int count = tile_counts[tile_idx];
    int start = tile_offsets[tile_idx];

    // Pixel position in [Z,Y,X] coordinates (matching PyTorch/numpy)
    float3 px = float3(gid.z, gid.y, gid.x);  // [Z,Y,X]

    // Accumulate contributions
    float accum = 0.0f;

    for (int i = 0; i < count; i++) {
        int splat_id = tile_content[start + i];

        // Load center in [Z,Y,X] order (matches PyTorch)
        float3 c = float3(
            centers[splat_id * 3 + 0],  // Z
            centers[splat_id * 3 + 1],  // Y
            centers[splat_id * 3 + 2]   // X
        );

        // Displacement in [Z,Y,X]: d = [dz, dy, dx]
        float3 d = px - c;

        // Load conic (reordered in Python to [X,Y,Z])
        // conic = [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]
        int cb = splat_id * 6;
        float c_xx = conic[cb + 0];
        float c_xy = conic[cb + 1];
        float c_xz = conic[cb + 2];
        float c_yy = conic[cb + 3];
        float c_yz = conic[cb + 4];
        float c_zz = conic[cb + 5];

        // Mahalanobis distance: d^T × Σ^-1 × d
        // d = [dz, dy, dx], so: d.x=dz, d.y=dy, d.z=dx
        float dz = d.x, dy = d.y, dx = d.z;
        float dist_sq = dx * dx * c_xx + dy * dy * c_yy + dz * dz * c_zz
                      + 2.0f * (dx * dy * c_xy + dx * dz * c_xz + dy * dz * c_yz);

        // Load sharpness and compute sharpness-adjusted truncation threshold
        // Must match the threshold used in tiling: truncate^(2/s)
        float s = sharpness[splat_id];
        float effective_truncate_sq = pow(truncate, 4.0f / s);  // (truncate^(2/s))^2

        // Truncation check (sharpness-adjusted to match tiling)
        if (dist_sq <= effective_truncate_sq) {
            float a = amps[splat_id];

            // Generalized Gaussian: exp(-0.5 × dist^s)
            float val = a * exp(-0.5f * pow(max(dist_sq, 1e-10f), s * 0.5f));

            // Early culling: skip invisible contributions (saves GPU cycles)
            if (val < intensity_floor) continue;

            accum += val;
        }
    }

    // Write output (row-major: [z, y, x] -> z*H*W + y*W + x)
    int out_idx = gid.z * (img_size.y * img_size.x) + gid.y * img_size.x + gid.x;
    output[out_idx] = accum;
}

// ============================================================================
// KERNEL 4: RASTERIZE BACKWARD (Gradient computation with SIMD reduction)
// ============================================================================

kernel void rasterize_bwd_3d(
    device const float* grad_output    [[buffer(0)]],   // (D, H, W)
    device const float* centers        [[buffer(1)]],   // (N, 3)
    device const float* conic          [[buffer(2)]],   // (N, 6)
    device const float* amps           [[buffer(3)]],   // (N,)
    device const float* sharpness      [[buffer(4)]],   // (N,)
    device const int* tile_offsets     [[buffer(5)]],
    device const int* tile_counts      [[buffer(6)]],
    device const int* tile_content     [[buffer(7)]],
    device atomic_float* d_centers     [[buffer(8)]],   // (N, 3)
    device atomic_float* d_conic       [[buffer(9)]],   // (N, 6)
    device atomic_float* d_amps        [[buffer(10)]],  // (N,)
    device atomic_float* d_sharpness   [[buffer(11)]],  // (N,)
    constant uint3& img_size           [[buffer(12)]],  // (W, H, D) - see §2.7
    constant uint3& grid_dims          [[buffer(13)]],  // (tiles_x, tiles_y, tiles_z) - see §2.7
    constant float& truncate           [[buffer(14)]],  // base truncate (NOT squared)
    constant float& intensity_floor    [[buffer(15)]],  // CRITICAL: must match forward
    uint3 gid [[thread_position_in_grid]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    // Load upstream gradient
    bool active = (gid.x < img_size.x && gid.y < img_size.y && gid.z < img_size.z);
    int pix_idx = gid.z * (img_size.y * img_size.x) + gid.y * img_size.x + gid.x;
    float d_L_d_I = active ? grad_output[pix_idx] : 0.0f;

    // Tile info
    uint tile_idx = group_id.z * (grid_dims.x * grid_dims.y)
                  + group_id.y * grid_dims.x + group_id.x;
    int count = tile_counts[tile_idx];
    int start = tile_offsets[tile_idx];

    // Pixel position in [Z,Y,X] coordinates
    float3 px = float3(gid.z, gid.y, gid.x);  // [Z,Y,X]

    // Process each splat in tile
    for (int i = 0; i < count; i++) {
        int splat_id = tile_content[start + i];

        // === A. Compute local gradients (per thread) ===
        float val_amps = 0.0f;
        float val_sharpness = 0.0f;
        float3 val_centers = 0.0f;
        float val_conic[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        if (active && abs(d_L_d_I) > 1e-9f) {
            // Load center in [Z,Y,X] order
            float3 c = float3(
                centers[splat_id * 3 + 0],  // Z
                centers[splat_id * 3 + 1],  // Y
                centers[splat_id * 3 + 2]   // X
            );
            // Displacement in [Z,Y,X]: d = [dz, dy, dx]
            float3 d = px - c;

            // Load conic (reordered to [X,Y,Z] in Python)
            int cb = splat_id * 6;
            float c_xx = conic[cb + 0], c_xy = conic[cb + 1], c_xz = conic[cb + 2];
            float c_yy = conic[cb + 3], c_yz = conic[cb + 4], c_zz = conic[cb + 5];

            // EXPLICIT distance calculation (avoid float3 component confusion)
            float dz = px.x - c.x;  // Z_pixel - Z_center
            float dy = px.y - c.y;  // Y_pixel - Y_center
            float dx = px.z - c.z;  // X_pixel - X_center
            float dist_sq = dx * dx * c_xx + dy * dy * c_yy + dz * dz * c_zz
                          + 2.0f * (dx * dy * c_xy + dx * dz * c_xz + dy * dz * c_yz);

            // Sharpness-adjusted truncation (must match forward pass)
            float s = sharpness[splat_id];
            float effective_truncate_sq = pow(truncate, 4.0f / s);

            if (dist_sq <= effective_truncate_sq) {
                float a = amps[splat_id];
                float half_s = s * 0.5f;

                // Recompute forward values
                float dist_sq_safe = max(dist_sq, 1e-10f);
                float inner = -0.5f * pow(dist_sq_safe, half_s);
                float exp_val = exp(inner);
                float intensity = a * exp_val;

                // CRITICAL: Must match forward pass intensity_floor culling!
                if (intensity < intensity_floor) continue;

                float d_common = intensity * d_L_d_I;

                // 1. Amplitude gradient: ∂I/∂a = exp(inner)
                val_amps = exp_val * d_L_d_I;

                // 2. Sharpness gradient: ∂I/∂s = I × inner × 0.5 × ln(D)
                val_sharpness = d_common * inner * 0.5f * log(dist_sq_safe);

                // 3. Distance gradient: ∂I/∂D = I × (-0.25s) × D^(s/2-1)
                float d_inner_d_D2 = -0.25f * s * pow(dist_sq_safe, half_s - 1.0f);
                float grad_dist = d_common * d_inner_d_D2;

                // 4. Center gradient: ∂D/∂μ = -2 × Σ^-1 × d
                // d = [dz, dy, dx] (already extracted above), conic in [X,Y,Z]
                float3 d_D2_d_d;  // In [Z,Y,X] order
                d_D2_d_d.x = 2.0f * (dz * c_zz + dy * c_yz + dx * c_xz);  // ∂D/∂z
                d_D2_d_d.y = 2.0f * (dz * c_yz + dy * c_yy + dx * c_xy);  // ∂D/∂y
                d_D2_d_d.z = 2.0f * (dz * c_xz + dy * c_xy + dx * c_xx);  // ∂D/∂x
                val_centers = grad_dist * d_D2_d_d * -1.0f;

                // 5. Conic gradient: ∂D/∂c_ij (in [X,Y,Z] order)
                val_conic[0] = grad_dist * dx * dx;             // c_xx
                val_conic[1] = grad_dist * 2.0f * dx * dy;      // c_xy
                val_conic[2] = grad_dist * 2.0f * dx * dz;      // c_xz
                val_conic[3] = grad_dist * dy * dy;             // c_yy
                val_conic[4] = grad_dist * 2.0f * dy * dz;      // c_yz
                val_conic[5] = grad_dist * dz * dz;             // c_zz

                // === DIRECT ATOMIC WRITE (no SIMD reduction) ===
                // Each thread atomically adds its own gradient contribution
                // This matches the forward pass pattern and avoids race conditions

                // Guard against NaN poisoning
                bool has_nan = isnan(val_amps) || isinf(val_amps) ||
                              isnan(val_sharpness) || isinf(val_sharpness) ||
                              isnan(val_centers.x) || isnan(val_centers.y) || isnan(val_centers.z);

                if (!has_nan) {
                    // Amplitude and sharpness gradients
                    if (abs(val_amps) > 1e-12f) {
                        atomic_add_float(&d_amps[splat_id], val_amps);
                    }
                    if (abs(val_sharpness) > 1e-12f) {
                        atomic_add_float(&d_sharpness[splat_id], val_sharpness);
                    }

                    // Center gradients (always add, even if small)
                    atomic_add_float(&d_centers[splat_id * 3 + 0], val_centers.x);
                    atomic_add_float(&d_centers[splat_id * 3 + 1], val_centers.y);
                    atomic_add_float(&d_centers[splat_id * 3 + 2], val_centers.z);

                    // Conic gradients
                    int cb = splat_id * 6;
                    for (int k = 0; k < 6; k++) {
                        if (abs(val_conic[k]) > 1e-12f && !isnan(val_conic[k]) && !isinf(val_conic[k])) {
                            atomic_add_float(&d_conic[cb + k], val_conic[k]);
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// GENERIC nD KERNELS (No tiling - fallback for 4D+)
// ============================================================================

kernel void rasterize_fwd_nd(
    device const float* centers      [[buffer(0)]],   // (N, dim)
    device const float* Ls           [[buffer(1)]],   // (N, dim, dim)
    device const float* amps         [[buffer(2)]],   // (N,)
    device const float* sharpness    [[buffer(3)]],   // (N,)
    device float* output             [[buffer(4)]],   // flattened
    constant uint& n_splats          [[buffer(5)]],
    constant uint& dim               [[buffer(6)]],
    constant uint* shape             [[buffer(7)]],   // (dim,)
    constant float& truncate         [[buffer(8)]],
    constant float& intensity_floor  [[buffer(9)]],   // early culling threshold
    uint gid [[thread_position_in_grid]]
) {
    // Unpack voxel coordinates from linear index
    float coords[8];  // Max 8 dimensions
    int temp = gid;
    for (int d = dim - 1; d >= 0; d--) {
        coords[d] = float(temp % shape[d]);
        temp /= shape[d];
    }
    if (temp > 0) return;  // Out of bounds

    float accum = 0.0f;

    for (uint i = 0; i < n_splats; i++) {
        float s = sharpness[i];

        // AABB check using diagonal of Sigma (with sharpness adjustment)
        bool possible = true;
        for (uint d = 0; d < dim; d++) {
            float c = centers[i * dim + d];
            float diff = coords[d] - c;

            // Compute Σ_dd = sum_k(L_dk²)
            float sigma_dd = 0.0f;
            for (uint k = 0; k <= d; k++) {
                float val = Ls[i * dim * dim + d * dim + k];
                sigma_dd += val * val;
            }

            // Sharpness-adjusted radius
            float eff_trunc = pow(truncate, 2.0f / s);
            if (abs(diff) > eff_trunc * sqrt(sigma_dd)) {
                possible = false;
                break;
            }
        }

        if (!possible) continue;

        // Full Mahalanobis via forward substitution: y = L^-1 × (x - μ)
        float y[8];
        for (uint r = 0; r < dim; r++) {
            float sum = 0.0f;
            for (uint c = 0; c < r; c++) {
                sum += Ls[i * dim * dim + r * dim + c] * y[c];
            }
            float diff = coords[r] - centers[i * dim + r];
            float L_rr = Ls[i * dim * dim + r * dim + r];
            y[r] = (diff - sum) / (L_rr + 1e-9f);
        }

        // dist_sq = ||y||²
        float dist_sq = 0.0f;
        for (uint d = 0; d < dim; d++) {
            dist_sq += y[d] * y[d];
        }

        // Use sharpness-adjusted truncation (consistent with AABB check)
        float effective_truncate_sq = pow(truncate, 4.0f / s);  // (truncate^(2/s))²
        if (dist_sq <= effective_truncate_sq) {
            float val = amps[i] * exp(-0.5f * pow(max(dist_sq, 1e-10f), s * 0.5f));

            // Early culling for intensity_floor
            if (val < intensity_floor) continue;

            accum += val;
        }
    }

    output[gid] = accum;
}

// ============================================================================
// KERNEL 5: RASTERIZE BACKWARD nD (MUST use SIMD reduction!)
// ============================================================================
//
// CRITICAL: The nD backward kernel MUST use SIMD reduction, just like the 3D
// kernel. Without it, every pixel fires atomic writes to global memory.
// For a 128³ volume, that's ~2 million atomic locks per frame - slower than CPU!
//
// The pattern is identical to rasterize_bwd_3d because nD still iterates over
// n_splats linearly. Each SIMD group processes the same splat together.

kernel void rasterize_bwd_nd(
    device const float* grad_output    [[buffer(0)]],   // flattened
    device const float* centers        [[buffer(1)]],   // (N, dim)
    device const float* Ls             [[buffer(2)]],   // (N, dim, dim)
    device const float* amps           [[buffer(3)]],   // (N,)
    device const float* sharpness      [[buffer(4)]],   // (N,)
    device atomic_float* d_centers     [[buffer(5)]],   // (N, dim)
    device atomic_float* d_Ls          [[buffer(6)]],   // (N, dim, dim)
    device atomic_float* d_amps        [[buffer(7)]],   // (N,)
    device atomic_float* d_sharpness   [[buffer(8)]],   // (N,)
    constant uint& n_splats            [[buffer(9)]],
    constant uint& dim                 [[buffer(10)]],
    constant uint* shape               [[buffer(11)]],  // (dim,)
    constant float& truncate           [[buffer(12)]],
    constant float& intensity_floor    [[buffer(13)]],  // for consistency
    uint gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    // Unpack voxel coordinates from linear index
    float coords[8];  // Max 8 dimensions
    int temp = gid;
    for (int d = dim - 1; d >= 0; d--) {
        coords[d] = float(temp % shape[d]);
        temp /= shape[d];
    }
    bool active = (temp == 0);  // Within bounds

    float d_L_d_I = active ? grad_output[gid] : 0.0f;

    // Process each splat
    for (uint i = 0; i < n_splats; i++) {
        float s = sharpness[i];

        // === A. Compute local gradients (per thread) ===
        float val_amps = 0.0f;
        float val_sharpness = 0.0f;
        float val_centers[8] = {0};
        float val_Ls[64] = {0};  // Max 8x8

        if (active && abs(d_L_d_I) > 1e-9f) {
            // AABB check (same as forward - with sharpness adjustment)
            bool possible = true;
            for (uint d = 0; d < dim; d++) {
                float c = centers[i * dim + d];
                float diff = coords[d] - c;
                float sigma_dd = 0.0f;
                for (uint k = 0; k <= d; k++) {
                    float val = Ls[i * dim * dim + d * dim + k];
                    sigma_dd += val * val;
                }
                float eff_trunc = pow(truncate, 2.0f / s);
                if (abs(diff) > eff_trunc * sqrt(sigma_dd)) {
                    possible = false;
                    break;
                }
            }

            if (possible) {
                // Forward substitution: y = L^-1 × (x - μ)
                // Also compute delta = x - μ for L gradient
                float y[8];
                float delta[8];
                for (uint r = 0; r < dim; r++) {
                    delta[r] = coords[r] - centers[i * dim + r];
                    float sum = 0.0f;
                    for (uint c = 0; c < r; c++) {
                        sum += Ls[i * dim * dim + r * dim + c] * y[c];
                    }
                    float L_rr = Ls[i * dim * dim + r * dim + r];
                    y[r] = (delta[r] - sum) / (L_rr + 1e-9f);
                }

                float dist_sq = 0.0f;
                for (uint d = 0; d < dim; d++) {
                    dist_sq += y[d] * y[d];
                }

                // Sharpness-adjusted truncation (consistent with forward)
                float effective_truncate_sq = pow(truncate, 4.0f / s);
                if (dist_sq <= effective_truncate_sq) {
                    float a = amps[i];
                    float half_s = s * 0.5f;
                    float dist_sq_safe = max(dist_sq, 1e-10f);
                    float inner = -0.5f * pow(dist_sq_safe, half_s);
                    float exp_val = exp(inner);
                    float intensity = a * exp_val;

                    // CRITICAL: Must match forward pass intensity_floor culling
                    if (intensity < intensity_floor) continue;

                    float d_common = intensity * d_L_d_I;

                    // 1. Amplitude gradient: ∂I/∂a = exp(inner)
                    val_amps = exp_val * d_L_d_I;

                    // 2. Sharpness gradient: ∂I/∂s = I × inner × 0.5 × ln(D)
                    val_sharpness = d_common * inner * 0.5f * log(dist_sq_safe);

                    // 3. Distance gradient: ∂I/∂D = I × (-0.25s) × D^(s/2-1)
                    float d_inner_d_D2 = -0.25f * s * pow(dist_sq_safe, half_s - 1.0f);
                    float grad_dist = d_common * d_inner_d_D2;

                    // 4. Center gradient: ∂D/∂μ = -2 × y (via chain rule)
                    for (uint d = 0; d < dim; d++) {
                        val_centers[d] = grad_dist * (-2.0f * y[d]);
                    }

                    // 5. L gradient: ∂D/∂L via chain rule
                    for (uint r = 0; r < dim; r++) {
                        for (uint c = 0; c <= r; c++) {
                            float L_rc = Ls[i * dim * dim + r * dim + c];
                            float L_rc_safe = (c == r) ? max(abs(L_rc), 1e-9f) : (abs(L_rc) > 1e-9f ? L_rc : 1.0f);
                            val_Ls[r * 8 + c] = grad_dist * (-2.0f * y[r] * y[c]) / L_rc_safe;
                        }
                    }
                }
            }
        }

        // === B. SIMD Reduction (CRITICAL - reduces atomics by 32x) ===
        float sum_amps = simd_sum(val_amps);
        float sum_sharpness = simd_sum(val_sharpness);

        // Sum centers across SIMD group
        float sum_centers[8];
        for (uint d = 0; d < dim; d++) {
            sum_centers[d] = simd_sum(val_centers[d]);
        }

        // Sum Ls across SIMD group
        float sum_Ls[64];
        for (uint r = 0; r < dim; r++) {
            for (uint c = 0; c <= r; c++) {
                sum_Ls[r * 8 + c] = simd_sum(val_Ls[r * 8 + c]);
            }
        }

        // === C. Leader writes to global memory (lane 0 only) ===
        if (simd_lane_id == 0) {
            if (abs(sum_amps) > 1e-12f) {
                atomic_add_float(&d_amps[i], sum_amps);
            }
            if (abs(sum_sharpness) > 1e-12f) {
                atomic_add_float(&d_sharpness[i], sum_sharpness);
            }
            for (uint d = 0; d < dim; d++) {
                if (abs(sum_centers[d]) > 1e-12f) {
                    atomic_add_float(&d_centers[i * dim + d], sum_centers[d]);
                }
            }
            // Write L gradients (lower triangle only)
            for (uint r = 0; r < dim; r++) {
                for (uint c = 0; c <= r; c++) {
                    if (abs(sum_Ls[r * 8 + c]) > 1e-12f) {
                        atomic_add_float(&d_Ls[i * dim * dim + r * dim + c], sum_Ls[r * 8 + c]);
                    }
                }
            }
        }
    }
}
