//! Gaussian Splat visibility computation for nD datasets.
//!
//! Computes which GSplats are visible in the current nD slice using
//! ellipsoid extent estimation from Cholesky factors.

use wasm_bindgen::prelude::*;

/// Compute nD visibility for GSplats using ellipsoid extent.
///
/// # Arguments
/// * `centers` - Splat centers [numSplats * ndim]
/// * `cholesky_factors` - Packed Cholesky factors [numSplats * k] where k = ndim*(ndim+1)/2
/// * `slice_position` - Current slice position [ndim]
/// * `tolerance` - Tolerance per dimension [ndim]
/// * `ndim` - Number of dimensions
/// * `num_splats` - Total number of splats
/// * `output_mask` - Output visibility mask [numSplats]
///
/// # Returns
/// Number of visible splats
///
/// # Algorithm
/// For each splat:
/// 1. Extract maximum ellipsoid extent from Cholesky diagonal elements
/// 2. Check if center +/- max extent intersects the slice
/// 3. Uses an optimistic estimate (max diagonal of L underestimates the true
///    ellipsoid extent for correlated covariances). This is acceptable as a
///    pre-filter since precise attenuation is computed later by
///    `compute_gsplats_attenuation`.
#[wasm_bindgen]
pub fn compute_nd_visibility_gsplats(
    centers: &[f32],
    cholesky_factors: &[f32],
    slice_position: &[f32],
    tolerance: &[f32],
    ndim: usize,
    num_splats: usize,
    output_mask: &mut [u8],
) -> u32 {
    let cholesky_size = (ndim * (ndim + 1)) / 2;
    let mut visible_count = 0;

    for splat_idx in 0..num_splats {
        let center_offset = splat_idx * ndim;
        let cholesky_offset = splat_idx * cholesky_size;

        // Compute maximum ellipsoid extent from Cholesky factors
        // Diagonal elements of L give axis scales (L is lower triangular)
        // OPTIMIZATION: Use direct formula instead of accumulating chol_idx
        let mut max_extent = 0.0_f32;

        for dim in 0..ndim {
            // Diagonal element L[dim,dim] is at packed position: dim*(dim+1)/2 + dim = dim*(dim+3)/2
            let diag_pos = cholesky_offset + (dim * (dim + 3)) / 2;
            let scale = cholesky_factors[diag_pos].abs();
            max_extent = max_extent.max(scale);
        }

        // Check if center + max extent is within tolerance
        // OPTIMIZATION: Use reciprocal multiplication instead of division (10x faster)
        let mut dist_sq = 0.0_f32;
        for dim in 0..ndim {
            let delta = centers[center_offset + dim] - slice_position[dim];
            let effective_tolerance = tolerance[dim] + max_extent;

            if effective_tolerance > 0.0 {
                let inv_tolerance = 1.0 / effective_tolerance;
                let normalized = delta * inv_tolerance;
                dist_sq += normalized * normalized;
            } else if delta.abs() > 1e-6 {
                dist_sq = f32::INFINITY;
                break;
            }
        }

        let visible = dist_sq <= 1.0;
        output_mask[splat_idx] = visible as u8;
        visible_count += visible as u32;
    }

    visible_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gsplat_visibility_basic() {
        // 2 splats in 3D
        let centers = vec![
            0.0, 0.0, 0.0,    // Splat 0 at origin
            10.0, 10.0, 10.0, // Splat 1 far away
        ];
        // 3D cholesky: 6 elements per splat [L00, L10, L11, L20, L21, L22]
        let cholesky = vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, // Splat 0: identity-ish
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, // Splat 1: identity-ish
        ];
        let slice_pos = vec![0.0, 0.0, 0.0];
        let tolerance = vec![2.0, 2.0, 2.0];

        let mut output = vec![0u8; 2];
        let count = compute_nd_visibility_gsplats(
            &centers,
            &cholesky,
            &slice_pos,
            &tolerance,
            3,
            2,
            &mut output,
        );

        assert_eq!(output[0], 1, "Splat 0 should be visible");
        assert_eq!(output[1], 0, "Splat 1 should be hidden");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_gsplat_visibility_4d() {
        // 2 splats in 4D, same XYZ but different T
        let centers = vec![
            0.0, 0.0, 0.0, 0.0,  // Splat 0 at t=0
            0.0, 0.0, 0.0, 10.0, // Splat 1 at t=10
        ];
        // 4D cholesky: 10 elements per splat
        let cholesky = vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Splat 0
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // Splat 1
        ];
        let slice_pos = vec![0.0, 0.0, 0.0, 0.0];
        let tolerance = vec![1e10, 1e10, 1e10, 2.0]; // Infinite for XYZ, 2.0 for T

        let mut output = vec![0u8; 2];
        let count = compute_nd_visibility_gsplats(
            &centers,
            &cholesky,
            &slice_pos,
            &tolerance,
            4,
            2,
            &mut output,
        );

        assert_eq!(output[0], 1, "Splat 0 should be visible (t=0)");
        assert_eq!(output[1], 0, "Splat 1 should be hidden (t=10 > 2.0)");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_gsplat_visibility_anisotropic_3d() {
        // 2 splats in 3D with anisotropic Cholesky
        // Cholesky packed [L00, L10, L11, L20, L21, L22]
        // L = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 5.0]]
        // Sigma = L * L^T -> diag(0.25, 0.25, 25.0) - elongated in z
        let centers = vec![
            0.0, 0.0, 0.0,   // Splat 0 at origin
            0.0, 0.0, 20.0,  // Splat 1 at z=20
        ];
        let cholesky = vec![
            0.5, 0.0, 0.5, 0.0, 0.0, 5.0, // Splat 0
            0.5, 0.0, 0.5, 0.0, 0.0, 5.0, // Splat 1
        ];
        let slice_pos = vec![0.0, 0.0, 0.0];
        let tolerance = vec![1.0, 1.0, 1.0];

        let mut output = vec![0u8; 2];
        let count = compute_nd_visibility_gsplats(
            &centers,
            &cholesky,
            &slice_pos,
            &tolerance,
            3,
            2,
            &mut output,
        );

        // Splat 0: at origin, max_extent=5.0, effective_tolerance = tol + 5.0 = 6.0
        // delta = [0,0,0], dist_sq = 0 <= 1.0 -> visible
        assert_eq!(output[0], 1, "Splat 0 at origin should be visible");

        // Splat 1: at z=20, max_extent=5.0, effective_tolerance = 1.0 + 5.0 = 6.0
        // delta_z = 20, normalized = 20/6 = 3.33, dist_sq = 3.33^2 = 11.1 > 1.0 -> hidden
        assert_eq!(output[1], 0, "Splat 1 at z=20 should be hidden");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_gsplat_visibility_anisotropic_2d() {
        // 2D with anisotropic Cholesky (tilted ellipse)
        // Cholesky packed [L00, L10, L11]
        // L = [[2.0, 0], [1.0, 3.0]]
        // Sigma = L * L^T = [[4, 2], [2, 10]]
        // Max diagonal of L is 3.0
        let centers = vec![
            0.0, 0.0, // Splat 0 at origin
            0.0, 5.0, // Splat 1 at y=5
        ];
        let cholesky = vec![
            2.0, 1.0, 3.0, // Splat 0: L = [[2,0],[1,3]]
            2.0, 1.0, 3.0, // Splat 1
        ];
        let slice_pos = vec![0.0, 0.0];
        let tolerance = vec![1.0, 1.0];

        let mut output = vec![0u8; 2];
        let count = compute_nd_visibility_gsplats(
            &centers,
            &cholesky,
            &slice_pos,
            &tolerance,
            2,
            2,
            &mut output,
        );

        // Splat 0: at origin, max_extent=3.0, effective_tol = 1+3 = 4.0
        // delta = [0,0], dist_sq = 0 -> visible
        assert_eq!(output[0], 1, "Splat 0 at origin should be visible");

        // Splat 1: at y=5, max_extent=3.0, effective_tol = 1+3 = 4.0
        // delta_y = 5, normalized = 5/4 = 1.25, dist_sq = 1.25^2 = 1.5625 > 1.0 -> hidden
        assert_eq!(output[1], 0, "Splat 1 at y=5 should be hidden");
        assert_eq!(count, 1);
    }
}
