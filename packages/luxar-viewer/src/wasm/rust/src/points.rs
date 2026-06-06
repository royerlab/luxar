//! Point visibility computation for nD datasets.
//!
//! Computes which points are visible in the current nD slice using
//! hypersphere intersection tests.

use wasm_bindgen::prelude::*;

use crate::common::validate_ndim;

/// Compute nD visibility for Points using hypersphere intersection.
///
/// # Arguments
/// * `positions` - Point positions [numPoints * ndim]
/// * `radii` - Point radii [numPoints]
/// * `slice_position` - Current slice position [ndim]
/// * `tolerance` - Tolerance per dimension [ndim]
/// * `ndim` - Number of dimensions
/// * `num_points` - Total number of points
/// * `output_mask` - Output visibility mask [numPoints] (1=visible, 0=hidden)
///
/// # Returns
/// Number of visible points
///
/// # Algorithm
/// For each point, compute normalized distance in nD space:
/// - Normalize each dimension by (tolerance + radius)
/// - Point is visible if sum of squared normalized distances <= 1.0
#[wasm_bindgen]
pub fn compute_nd_visibility_points(
    positions: &[f32],
    radii: &[f32],
    slice_position: &[f32],
    tolerance: &[f32],
    ndim: usize,
    num_points: usize,
    output_mask: &mut [u8],
) -> u32 {
    validate_ndim(ndim, "compute_nd_visibility_points");

    debug_assert!(
        output_mask.len() >= num_points,
        "output_mask too small: {} < {}",
        output_mask.len(),
        num_points
    );

    let mut visible_count = 0;

    // OPTIMIZATION: Cache visibility threshold (avoids branch in inner loop)
    const VISIBILITY_THRESHOLD: f32 = 1.0;
    const EPSILON: f32 = 1e-6;

    for pt_idx in 0..num_points {
        let pt_offset = pt_idx * ndim;
        let radius = radii[pt_idx];

        // Compute normalized distance in nD space
        let mut dist_sq = 0.0_f32;

        // OPTIMIZATION: Hoist branch-sensitive code outside inner loop where possible
        for dim in 0..ndim {
            let delta = positions[pt_offset + dim] - slice_position[dim];
            let effective_tolerance = tolerance[dim] + radius;

            // Direct division (not reciprocal-multiply) keeps the visibility
            // boundary (dist_sq ≈ VISIBILITY_THRESHOLD) bit-for-bit consistent
            // with the TS reference; the perf delta is negligible on modern FPUs.
            if effective_tolerance > 0.0 {
                let normalized = delta / effective_tolerance;
                dist_sq += normalized * normalized;
            } else if delta.abs() > EPSILON {
                // Point is far from slice with zero tolerance - not visible
                dist_sq = f32::INFINITY;
                break;
            }
        }

        // OPTIMIZATION: Minimize branching - visibility check only once
        let visible = dist_sq <= VISIBILITY_THRESHOLD;
        output_mask[pt_idx] = visible as u8;
        visible_count += visible as u32;
    }

    visible_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_visibility_3d() {
        // 3 points in 3D, all at same XY but different Z
        let positions = vec![
            0.0, 0.0, 0.0, // Point 0 at origin
            0.0, 0.0, 5.0, // Point 1 at z=5
            0.0, 0.0, 0.5, // Point 2 at z=0.5
        ];
        let radii = vec![0.5, 0.5, 0.5];
        let slice_pos = vec![0.0, 0.0, 0.0];
        let tolerance = vec![1.0, 1.0, 1.0];

        let mut output = vec![0u8; 3];
        let count = compute_nd_visibility_points(
            &positions,
            &radii,
            &slice_pos,
            &tolerance,
            3,
            3,
            &mut output,
        );

        assert_eq!(output[0], 1, "Point 0 should be visible (at origin)");
        assert_eq!(output[2], 1, "Point 2 should be visible (z=0.5)");
        assert!(count >= 2, "At least 2 points should be visible");
    }

    #[test]
    fn test_point_visibility_4d_hidden_dimension() {
        // 2 points in 4D, same XYZ but different T
        let positions = vec![
            0.0, 0.0, 0.0, 0.0, // Point 0 at t=0
            0.0, 0.0, 0.0, 10.0, // Point 1 at t=10 (far away)
        ];
        let radii = vec![0.5, 0.5];
        let slice_pos = vec![0.0, 0.0, 0.0, 0.0];
        // Infinite tolerance for XYZ, 1.0 for T
        let tolerance = vec![1e10, 1e10, 1e10, 1.0];

        let mut output = vec![0u8; 2];
        let count = compute_nd_visibility_points(
            &positions,
            &radii,
            &slice_pos,
            &tolerance,
            4,
            2,
            &mut output,
        );

        assert_eq!(output[0], 1, "Point 0 should be visible (t=0)");
        assert_eq!(output[1], 0, "Point 1 should be hidden (t=10)");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_edge_case_zero_tolerance() {
        // Point exactly at slice position with zero tolerance
        let positions = vec![0.0, 0.0, 0.0];
        let radii = vec![0.1];
        let slice_pos = vec![0.0, 0.0, 0.0];
        let tolerance = vec![0.0, 0.0, 0.0];

        let mut output = vec![0u8; 1];
        let count = compute_nd_visibility_points(
            &positions,
            &radii,
            &slice_pos,
            &tolerance,
            3,
            1,
            &mut output,
        );

        // With radius 0.1 and zero tolerance, point should still be visible
        assert_eq!(output[0], 1, "Point at exact position should be visible");
        assert_eq!(count, 1);
    }
}
