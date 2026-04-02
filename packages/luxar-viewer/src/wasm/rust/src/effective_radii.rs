//! Effective radius calculation for nD hypersphere slicing.
//!
//! When an nD hypersphere of radius R is intersected by a hyperplane at distance D,
//! the effective radius in the slice is: R_effective = sqrt(R² - D²)
//!
//! This module provides optimized calculation for millions of points.
//!
//! # Dimension Limits
//!
//! **Maximum supported dimensions: 16**
//!
//! This module uses fixed-size arrays for performance. Datasets with more than
//! 16 dimensions will trigger a panic with a clear error message.

use wasm_bindgen::prelude::*;

use crate::common::{validate_ndim, MAX_SUPPORTED_DIMS};

/// Calculate effective radii for nD points when sliced.
///
/// For each point, computes the effective radius visible in the 3D slice
/// based on the point's distance from the slice plane in hidden dimensions.
///
/// # Arguments
/// * `positions` - Point positions [numPoints * ndim] (flattened nD coordinates)
/// * `radii` - Original point radii [numPoints]
/// * `display_dims` - Dimensions to display (typically first 3) [numDisplayDims]
/// * `slice_position` - Current slice position [ndim]
/// * `spatial_extend_dims` - Which dims are spatial (1) vs discrete (0) [ndim]
/// * `ndim` - Total number of dimensions (max 16)
/// * `num_points` - Number of points
/// * `output` - Output effective radii [numPoints]
///
/// # Returns
/// Number of points with non-zero effective radius (visible points)
///
/// # Algorithm
/// For each point:
/// 1. Check discrete dimension match (exact or within tolerance)
/// 2. Calculate squared distance in non-displayed spatial dimensions
/// 3. Apply Pythagorean theorem: R_effective = sqrt(R² - D²)
/// 4. Set R_effective = 0 if point is outside original radius
///
/// # Panics
/// Panics if `ndim > 16`. Use TypeScript fallback for higher dimensions.
#[wasm_bindgen]
pub fn calculate_effective_radii(
    positions: &[f32],
    radii: &[f32],
    display_dims: &[u32],
    slice_position: &[f32],
    spatial_extend_dims: &[u8],
    ndim: usize,
    num_points: usize,
    output: &mut [f32],
) -> u32 {
    validate_ndim(ndim, "calculate_effective_radii");

    debug_assert!(
        output.len() >= num_points,
        "output too small: {} < {}",
        output.len(),
        num_points
    );

    let discrete_tolerance = 0.5_f32;
    let mut visible_count = 0u32;

    // Pre-convert display_dims to a lookup set for fast checking
    let mut is_display_dim = [false; MAX_SUPPORTED_DIMS];
    for &d in display_dims {
        if (d as usize) < MAX_SUPPORTED_DIMS {
            is_display_dim[d as usize] = true;
        }
    }

    for i in 0..num_points {
        let original_radius = radii[i];
        let pos_offset = i * ndim;

        // OPTIMIZATION: Fused loop - check discrete match AND compute spatial distance in one pass
        let mut discrete_match = true;
        let mut distance_squared = 0.0_f32;

        for d in 0..ndim {
            if is_display_dim[d] {
                continue; // Skip displayed dimensions
            }

            // Check if this is a spatial or discrete dimension
            let is_spatial = if d < spatial_extend_dims.len() {
                spatial_extend_dims[d] != 0
            } else {
                true // Default to spatial if not specified
            };

            let value = positions[pos_offset + d];
            let target = slice_position[d];

            if is_spatial {
                // Spatial dimension: accumulate squared distance
                let diff = value - target;
                distance_squared += diff * diff;
            } else {
                // Discrete dimension: must match exactly (within tolerance)
                if (value - target).abs() > discrete_tolerance {
                    discrete_match = false;
                    break;
                }
            }
        }

        if !discrete_match {
            output[i] = 0.0;
            continue;
        }

        // Apply Pythagorean theorem: R_effective = sqrt(R² - D²)
        let radius_squared = original_radius * original_radius;
        if distance_squared >= radius_squared {
            output[i] = 0.0;
        } else {
            output[i] = (radius_squared - distance_squared).sqrt();
            visible_count += 1;
        }
    }

    visible_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_radii_3d_no_hidden_dims() {
        // 3D case: all dimensions displayed, no effective radius change
        let positions = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let radii = vec![1.0, 0.5];
        let display_dims = vec![0, 1, 2];
        let slice_pos = vec![0.0, 0.0, 0.0];
        let spatial_extend = vec![1, 1, 1];
        let mut output = vec![0.0; 2];

        let visible = calculate_effective_radii(
            &positions,
            &radii,
            &display_dims,
            &slice_pos,
            &spatial_extend,
            3,
            2,
            &mut output,
        );

        // No hidden dims, so effective radii should equal original radii
        assert_eq!(visible, 2);
        assert!((output[0] - 1.0).abs() < 1e-6);
        assert!((output[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_effective_radii_5d_with_hidden_dims() {
        // 5D case: display dims [0,1,2], hidden spatial dims [3,4]
        // Point at (0,0,0, 0.6, 0.8) with radius 1.0
        // Distance in hidden dims = sqrt(0.6² + 0.8²) = 1.0
        // Effective radius = sqrt(1² - 1²) = 0
        let positions = vec![0.0, 0.0, 0.0, 0.6, 0.8];
        let radii = vec![1.0];
        let display_dims = vec![0, 1, 2];
        let slice_pos = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let spatial_extend = vec![1, 1, 1, 1, 1]; // All spatial
        let mut output = vec![0.0; 1];

        let visible = calculate_effective_radii(
            &positions,
            &radii,
            &display_dims,
            &slice_pos,
            &spatial_extend,
            5,
            1,
            &mut output,
        );

        // Point is at distance 1.0 from slice, radius is 1.0, so effective = 0
        assert_eq!(visible, 0);
        assert!((output[0]).abs() < 1e-6);
    }

    #[test]
    fn test_effective_radii_partial_intersection() {
        // 4D case: point partially intersects slice
        // Point at (0,0,0, 0.6) with radius 1.0
        // Distance in hidden dims = 0.6
        // Effective radius = sqrt(1² - 0.6²) = sqrt(0.64) = 0.8
        let positions = vec![0.0, 0.0, 0.0, 0.6];
        let radii = vec![1.0];
        let display_dims = vec![0, 1, 2];
        let slice_pos = vec![0.0, 0.0, 0.0, 0.0];
        let spatial_extend = vec![1, 1, 1, 1];
        let mut output = vec![0.0; 1];

        let visible = calculate_effective_radii(
            &positions,
            &radii,
            &display_dims,
            &slice_pos,
            &spatial_extend,
            4,
            1,
            &mut output,
        );

        assert_eq!(visible, 1);
        assert!((output[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_discrete_dimension_filtering() {
        // 4D case: dim 3 is discrete, not spatial
        // Point at (0,0,0, 5.0) should be filtered out (discrete mismatch)
        let positions = vec![
            0.0, 0.0, 0.0, 0.0, // Point 0: matches discrete dim
            0.0, 0.0, 0.0, 5.0, // Point 1: doesn't match discrete dim
        ];
        let radii = vec![1.0, 1.0];
        let display_dims = vec![0, 1, 2];
        let slice_pos = vec![0.0, 0.0, 0.0, 0.0];
        let spatial_extend = vec![1, 1, 1, 0]; // Dim 3 is discrete
        let mut output = vec![0.0; 2];

        let visible = calculate_effective_radii(
            &positions,
            &radii,
            &display_dims,
            &slice_pos,
            &spatial_extend,
            4,
            2,
            &mut output,
        );

        assert_eq!(visible, 1);
        assert!((output[0] - 1.0).abs() < 1e-6); // Point 0 visible
        assert!((output[1]).abs() < 1e-6); // Point 1 filtered
    }

    #[test]
    #[should_panic(expected = "exceeds maximum supported dimensions")]
    fn test_dimension_limit_validation() {
        // Test that ndim > 16 triggers a panic with clear error message
        // This ensures users get helpful feedback when exceeding limits
        let ndim = 17; // One above the MAX_SUPPORTED_DIMS (16)
        let positions = vec![0.0; ndim]; // Single point with 17 dimensions
        let radii = vec![1.0];
        let display_dims = vec![0, 1, 2];
        let slice_pos = vec![0.0; ndim];
        let spatial_extend = vec![1; ndim];
        let mut output = vec![0.0; 1];

        // This should panic because ndim > MAX_SUPPORTED_DIMS
        calculate_effective_radii(
            &positions,
            &radii,
            &display_dims,
            &slice_pos,
            &spatial_extend,
            ndim,
            1,
            &mut output,
        );
    }

    #[test]
    fn test_dimension_at_limit() {
        // Test that exactly 16 dimensions works fine (boundary case)
        let ndim = 16; // Exactly MAX_SUPPORTED_DIMS
        let positions = vec![0.0; ndim];
        let radii = vec![1.0];
        let display_dims = vec![0, 1, 2];
        let slice_pos = vec![0.0; ndim];
        let spatial_extend = vec![1; ndim];
        let mut output = vec![0.0; 1];

        // This should NOT panic - 16 dimensions is the max supported
        let visible = calculate_effective_radii(
            &positions,
            &radii,
            &display_dims,
            &slice_pos,
            &spatial_extend,
            ndim,
            1,
            &mut output,
        );

        // Point at origin should be fully visible
        assert_eq!(visible, 1);
        assert!((output[0] - 1.0).abs() < 1e-6);
    }
}
