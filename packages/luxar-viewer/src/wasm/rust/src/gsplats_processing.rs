//! GSplats Processing for nD → 3D Conversion
//!
//! High-performance implementations of GSplats processing operations:
//! - Mahalanobis distance computation (forward substitution)
//! - Cholesky submatrix extraction
//! - Batch attenuation computation for visibility filtering
//!
//! # Dimension Limits
//!
//! **Maximum supported dimensions: 16**
//!
//! This module uses fixed-size arrays for performance. Datasets with more than
//! 16 dimensions will trigger a panic with a clear error message.

use wasm_bindgen::prelude::*;

/// Maximum number of dimensions supported by WASM functions.
/// This limit exists because fixed-size arrays are used for performance.
/// If you need more dimensions, the code would need to use Vec<f32> with
/// pre-allocation, which has a small performance cost.
pub const MAX_SUPPORTED_DIMS: usize = 16;

/// Maximum packed Cholesky size for MAX_SUPPORTED_DIMS dimensions.
/// Formula: n * (n + 1) / 2 = 16 * 17 / 2 = 136
const MAX_PACKED_CHOLESKY_SIZE: usize = (MAX_SUPPORTED_DIMS * (MAX_SUPPORTED_DIMS + 1)) / 2;

/// Compute the packed index for a Cholesky element L[row, col].
/// Packed lower-triangular: [L00, L10, L11, L20, L21, L22, ...]
/// Formula: row * (row + 1) / 2 + col (for col <= row)
#[inline]
fn packed_index(row: usize, col: usize) -> usize {
    (row * (row + 1)) / 2 + col
}

/// Validate that the number of dimensions is within the supported limit.
/// Panics with a clear error message if the limit is exceeded.
#[inline]
fn validate_ndim(ndim: usize, function_name: &str) {
    if ndim > MAX_SUPPORTED_DIMS {
        panic!(
            "[WASM] {}: ndim={} exceeds maximum supported dimensions ({}). \
             Luxar WASM functions support up to {} dimensions. \
             For higher dimensions, use TypeScript fallback or reduce dataset dimensionality.",
            function_name, ndim, MAX_SUPPORTED_DIMS, MAX_SUPPORTED_DIMS
        );
    }
}

/// Compute Mahalanobis distance for a single point using packed Cholesky factor.
///
/// Given L (lower-triangular Cholesky of covariance),
/// Mahalanobis distance = ||L⁻¹ · (x - μ)||
///
/// We use forward substitution to solve L·y = d, then ||y|| is the Mahalanobis distance.
///
/// # Arguments
/// * `diff` - Difference vector (x - μ) for the dimensions [ndim]
/// * `packed_l` - Packed Cholesky factor [packedSize]
/// * `ndim` - Dimensionality of the Cholesky (max 16)
///
/// # Returns
/// Mahalanobis distance
///
/// # Panics
/// Panics if `ndim > 16`. Use TypeScript fallback for higher dimensions.
#[wasm_bindgen]
pub fn mahalanobis_distance(diff: &[f32], packed_l: &[f32], ndim: usize) -> f32 {
    validate_ndim(ndim, "mahalanobis_distance");

    // Forward substitution: solve L · y = diff
    // Using a small fixed-size array for common cases (up to 16 dims)
    let mut y = [0.0f32; MAX_SUPPORTED_DIMS];

    for i in 0..ndim {
        let mut val = diff[i];
        for j in 0..i {
            val -= packed_l[packed_index(i, j)] * y[j];
        }
        let diag = packed_l[packed_index(i, i)];
        y[i] = if diag > 1e-10 { val / diag } else { 0.0 };
    }

    // Compute ||y||
    let mut sum_sq = 0.0f32;
    for i in 0..ndim {
        sum_sq += y[i] * y[i];
    }
    sum_sq.sqrt()
}

/// Extract a Cholesky submatrix for specified dimensions.
///
/// # Arguments
/// * `packed` - Full packed Cholesky [packedSize]
/// * `keep_dims` - Indices of dimensions to keep (must be sorted ascending) [subNdim]
/// * `sub_ndim` - Number of dimensions to keep
/// * `output` - Output packed submatrix [subPackedSize]
#[wasm_bindgen]
pub fn extract_cholesky_submatrix(
    packed: &[f32],
    keep_dims: &[u32],
    sub_ndim: usize,
    output: &mut [f32],
) {
    let mut out_idx = 0;

    for sub_row in 0..sub_ndim {
        let orig_row = keep_dims[sub_row] as usize;
        for sub_col in 0..=sub_row {
            let orig_col = keep_dims[sub_col] as usize;
            output[out_idx] = packed[packed_index(orig_row, orig_col)];
            out_idx += 1;
        }
    }
}

/// Compute attenuation factors for all GSplats based on hidden dimension distance.
///
/// For each splat, computes:
/// 1. Difference vector in hidden dimensions
/// 2. Mahalanobis distance using hidden Cholesky submatrix
/// 3. Attenuation = exp(-0.5 * mahal^sharpness)
/// 4. Visibility = (amplitude * attenuation) >= threshold
///
/// # Arguments
/// * `positions` - Splat centers [splatCount * ndim]
/// * `cholesky` - Packed Cholesky factors [splatCount * packedSize]
/// * `amplitudes` - Splat amplitudes [splatCount]
/// * `sharpness` - Per-splat sharpness values [splatCount]
/// * `slice_position` - Current slice position [ndim]
/// * `hidden_dims` - Indices of hidden dimensions (sorted) [numHidden]
/// * `ndim` - Total dimensionality (max 16)
/// * `splat_count` - Number of splats
/// * `min_amplitude` - Visibility threshold
/// * `output_visibility` - Output visibility mask [splatCount]
/// * `output_attenuation` - Output attenuation factors [splatCount]
///
/// # Returns
/// Number of visible splats
///
/// # Panics
/// Panics if `ndim > 16`. Use TypeScript fallback for higher dimensions.
#[wasm_bindgen]
pub fn compute_gsplats_attenuation(
    positions: &[f32],
    cholesky: &[f32],
    amplitudes: &[f32],
    sharpness: &[f32],
    slice_position: &[f32],
    hidden_dims: &[u32],
    ndim: usize,
    splat_count: usize,
    min_amplitude: f32,
    output_visibility: &mut [u8],
    output_attenuation: &mut [f32],
) -> u32 {
    validate_ndim(ndim, "compute_gsplats_attenuation");

    let num_hidden = hidden_dims.len();
    let full_packed_size = (ndim * (ndim + 1)) / 2;

    // Temporary buffers (use fixed-size arrays for performance)
    let mut diff = [0.0f32; MAX_SUPPORTED_DIMS];
    let mut hidden_cholesky = [0.0f32; MAX_PACKED_CHOLESKY_SIZE];

    let mut visible_count = 0u32;

    for i in 0..splat_count {
        let center_offset = i * ndim;
        let cholesky_offset = i * full_packed_size;
        let splat_sharpness = sharpness[i];

        let attenuation = if num_hidden == 0 {
            // No hidden dimensions, full visibility
            1.0
        } else {
            // Compute difference vector in hidden dimensions
            for (h_idx, &dim) in hidden_dims.iter().enumerate() {
                let d = dim as usize;
                diff[h_idx] = slice_position[d] - positions[center_offset + d];
            }

            // Extract hidden Cholesky submatrix
            let mut out_idx = 0;
            for sub_row in 0..num_hidden {
                let orig_row = hidden_dims[sub_row] as usize;
                for sub_col in 0..=sub_row {
                    let orig_col = hidden_dims[sub_col] as usize;
                    hidden_cholesky[out_idx] =
                        cholesky[cholesky_offset + packed_index(orig_row, orig_col)];
                    out_idx += 1;
                }
            }

            // Compute Mahalanobis distance
            let mahal_dist =
                mahalanobis_distance_internal(&diff[..num_hidden], &hidden_cholesky, num_hidden);

            // Attenuation = exp(-0.5 * mahal^sharpness)
            (-0.5 * mahal_dist.powf(splat_sharpness)).exp()
        };

        output_attenuation[i] = attenuation;

        let attenuated_amplitude = amplitudes[i] * attenuation;
        let visible = attenuated_amplitude >= min_amplitude;
        output_visibility[i] = if visible { 1 } else { 0 };
        if visible {
            visible_count += 1;
        }
    }

    visible_count
}

/// Internal Mahalanobis distance (no WASM binding, avoids allocation)
/// Note: ndim is already validated by caller, no need to validate again
#[inline]
fn mahalanobis_distance_internal(diff: &[f32], packed_l: &[f32], ndim: usize) -> f32 {
    let mut y = [0.0f32; MAX_SUPPORTED_DIMS];

    for i in 0..ndim {
        let mut val = diff[i];
        for j in 0..i {
            val -= packed_l[packed_index(i, j)] * y[j];
        }
        let diag = packed_l[packed_index(i, i)];
        y[i] = if diag > 1e-10 { val / diag } else { 0.0 };
    }

    let mut sum_sq = 0.0f32;
    for i in 0..ndim {
        sum_sq += y[i] * y[i];
    }
    sum_sq.sqrt()
}

/// Extract 3D Cholesky submatrices for visible splats.
///
/// # Arguments
/// * `cholesky` - Packed Cholesky factors [splatCount * packedSize]
/// * `visibility` - Visibility mask [splatCount]
/// * `display_dims` - Display dimension indices (sorted) [3]
/// * `ndim` - Total dimensionality (max 16)
/// * `splat_count` - Number of splats
/// * `output` - Output 3D Cholesky factors [visibleCount * 6]
///
/// # Returns
/// Number of visible splats processed
///
/// # Panics
/// Panics if `ndim > 16`. Use TypeScript fallback for higher dimensions.
#[wasm_bindgen]
pub fn extract_visible_cholesky_3d(
    cholesky: &[f32],
    visibility: &[u8],
    display_dims: &[u32],
    ndim: usize,
    splat_count: usize,
    output: &mut [f32],
) -> u32 {
    validate_ndim(ndim, "extract_visible_cholesky_3d");

    let full_packed_size = (ndim * (ndim + 1)) / 2;
    let mut out_splat = 0u32;

    for i in 0..splat_count {
        if visibility[i] == 0 {
            continue;
        }

        let src_offset = i * full_packed_size;
        let dst_offset = (out_splat as usize) * 6;

        // Extract 3x3 Cholesky submatrix (6 elements)
        let mut out_idx = 0;
        for sub_row in 0..3 {
            let orig_row = display_dims[sub_row] as usize;
            for sub_col in 0..=sub_row {
                let orig_col = display_dims[sub_col] as usize;
                output[dst_offset + out_idx] = cholesky[src_offset + packed_index(orig_row, orig_col)];
                out_idx += 1;
            }
        }

        out_splat += 1;
    }

    out_splat
}

/// Compact amplitudes by visibility mask, applying attenuation.
///
/// # Arguments
/// * `amplitudes` - Original amplitudes [splatCount]
/// * `attenuation` - Attenuation factors [splatCount]
/// * `visibility` - Visibility mask [splatCount]
/// * `splat_count` - Number of splats
/// * `output` - Output attenuated amplitudes [visibleCount]
///
/// # Returns
/// Number of visible splats
#[wasm_bindgen]
pub fn compact_attenuated_amplitudes(
    amplitudes: &[f32],
    attenuation: &[f32],
    visibility: &[u8],
    splat_count: usize,
    output: &mut [f32],
) -> u32 {
    let mut out_idx = 0u32;

    for i in 0..splat_count {
        if visibility[i] != 0 {
            output[out_idx as usize] = amplitudes[i] * attenuation[i];
            out_idx += 1;
        }
    }

    out_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_index() {
        // [L00, L10, L11, L20, L21, L22, ...]
        assert_eq!(packed_index(0, 0), 0); // L00
        assert_eq!(packed_index(1, 0), 1); // L10
        assert_eq!(packed_index(1, 1), 2); // L11
        assert_eq!(packed_index(2, 0), 3); // L20
        assert_eq!(packed_index(2, 1), 4); // L21
        assert_eq!(packed_index(2, 2), 5); // L22
    }

    #[test]
    fn test_mahalanobis_distance_identity() {
        // Identity Cholesky (L = I): Mahalanobis = Euclidean
        // 3D: packed = [1, 0, 1, 0, 0, 1]
        let packed_l = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let diff = vec![3.0, 4.0, 0.0]; // Distance should be 5.0

        let dist = mahalanobis_distance(&diff, &packed_l, 3);
        assert!((dist - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_mahalanobis_distance_scaled() {
        // Scaled Cholesky: L = diag(2, 2, 2)
        // packed = [2, 0, 2, 0, 0, 2]
        // Mahalanobis = ||L⁻¹ · diff|| = ||diff / 2||
        let packed_l = vec![2.0, 0.0, 2.0, 0.0, 0.0, 2.0];
        let diff = vec![4.0, 0.0, 0.0]; // Mahalanobis should be 4/2 = 2

        let dist = mahalanobis_distance(&diff, &packed_l, 3);
        assert!((dist - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_extract_cholesky_submatrix() {
        // 4D Cholesky: 10 elements
        // [L00, L10, L11, L20, L21, L22, L30, L31, L32, L33]
        let packed = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Extract dims [0, 2] (2D submatrix)
        let keep_dims = vec![0, 2];
        let mut output = vec![0.0f32; 3]; // 2D packed = 3 elements

        extract_cholesky_submatrix(&packed, &keep_dims, 2, &mut output);

        // Expected: [L00, L20, L22] = [1.0, 4.0, 6.0]
        assert_eq!(output[0], 1.0); // L[0,0]
        assert_eq!(output[1], 4.0); // L[2,0]
        assert_eq!(output[2], 6.0); // L[2,2]
    }

    #[test]
    fn test_compute_gsplats_attenuation_no_hidden() {
        // 3D splats with no hidden dimensions
        let positions = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let cholesky = vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, // Splat 0
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, // Splat 1
        ];
        let amplitudes = vec![1.0, 0.5];
        let sharpness = vec![2.0, 2.0];
        let slice_pos = vec![0.0, 0.0, 0.0];
        let hidden_dims: Vec<u32> = vec![]; // No hidden dims

        let mut visibility = vec![0u8; 2];
        let mut attenuation = vec![0.0f32; 2];

        let count = compute_gsplats_attenuation(
            &positions,
            &cholesky,
            &amplitudes,
            &sharpness,
            &slice_pos,
            &hidden_dims,
            3,
            2,
            0.1,
            &mut visibility,
            &mut attenuation,
        );

        assert_eq!(count, 2); // Both visible
        assert_eq!(attenuation[0], 1.0); // No attenuation
        assert_eq!(attenuation[1], 1.0);
    }

    #[test]
    fn test_compute_gsplats_attenuation_with_hidden() {
        // 4D splats with dim 3 as hidden
        // Splat 0: at (0,0,0,0), visible
        // Splat 1: at (0,0,0,5), attenuated (far in hidden dim)
        let positions = vec![
            0.0, 0.0, 0.0, 0.0, // Splat 0
            0.0, 0.0, 0.0, 5.0, // Splat 1
        ];
        // 4D Cholesky: 10 elements, identity
        let cholesky = vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Splat 0
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Splat 1
        ];
        let amplitudes = vec![1.0, 1.0];
        let sharpness = vec![2.0, 2.0];
        let slice_pos = vec![0.0, 0.0, 0.0, 0.0];
        let hidden_dims = vec![3u32]; // Dim 3 is hidden

        let mut visibility = vec![0u8; 2];
        let mut attenuation = vec![0.0f32; 2];

        let count = compute_gsplats_attenuation(
            &positions,
            &cholesky,
            &amplitudes,
            &sharpness,
            &slice_pos,
            &hidden_dims,
            4,
            2,
            0.01, // Low threshold
            &mut visibility,
            &mut attenuation,
        );

        // Splat 0: mahal = 0, attenuation = 1.0
        assert_eq!(visibility[0], 1);
        assert!((attenuation[0] - 1.0).abs() < 1e-5);

        // Splat 1: mahal = 5.0, attenuation = exp(-0.5 * 25) ≈ 3.7e-6
        assert_eq!(visibility[1], 0); // Below threshold
        assert!(attenuation[1] < 0.001);

        assert_eq!(count, 1);
    }
}
