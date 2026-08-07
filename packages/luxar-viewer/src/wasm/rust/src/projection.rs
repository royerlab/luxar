//! nD to 3D projection.
//!
//! Extracts 3D display coordinates from nD point data. Optimized for large
//! datasets.

use wasm_bindgen::prelude::*;

/// Extract 3D positions from nD positions using display dimension indices.
///
/// # Arguments
/// * `positions_nd` - Input nD positions [numPoints * ndim]
/// * `display_dims` - Which dimensions to display as X,Y,Z [3 or fewer]
/// * `ndim` - Total number of dimensions
/// * `num_points` - Number of points
/// * `output` - Output 3D positions [numPoints * 3]
///
/// # Example
/// For 5D data with display_dims=[0,2,4]:
/// - X = position[dim 0]
/// - Y = position[dim 2]
/// - Z = position[dim 4]
#[wasm_bindgen]
pub fn extract_3d_positions(
    positions_nd: &[f32],
    display_dims: &[u32],
    ndim: usize,
    num_points: usize,
    output: &mut [f32],
) {
    debug_assert!(
        output.len() >= num_points * 3,
        "output too small: {} < {}",
        output.len(),
        num_points * 3
    );

    let num_display_dims = display_dims.len().min(3);

    for i in 0..num_points {
        let src_offset = i * ndim;
        let dst_offset = i * 3;

        // Extract displayed dimensions
        for j in 0..num_display_dims {
            let dim_idx = display_dims[j] as usize;
            output[dst_offset + j] = positions_nd[src_offset + dim_idx];
        }

        // Fill remaining with zeros (e.g., 2D display -> z=0)
        for j in num_display_dims..3 {
            output[dst_offset + j] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_3d_positions_basic() {
        // 5D data: 2 points
        let positions_nd = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // point 0
            6.0, 7.0, 8.0, 9.0, 10.0, // point 1
        ];
        let display_dims = vec![0, 2, 4]; // X=dim0, Y=dim2, Z=dim4
        let mut output = vec![0.0f32; 6];

        extract_3d_positions(&positions_nd, &display_dims, 5, 2, &mut output);

        assert_eq!(output[0], 1.0); // point0.x = dim0
        assert_eq!(output[1], 3.0); // point0.y = dim2
        assert_eq!(output[2], 5.0); // point0.z = dim4
        assert_eq!(output[3], 6.0); // point1.x = dim0
        assert_eq!(output[4], 8.0); // point1.y = dim2
        assert_eq!(output[5], 10.0); // point1.z = dim4
    }

    #[test]
    fn test_extract_3d_positions_2d_display() {
        // 4D data displayed as 2D (z=0)
        let positions_nd = vec![1.0, 2.0, 3.0, 4.0];
        let display_dims = vec![0, 1]; // Only X and Y
        let mut output = vec![0.0f32; 3];

        extract_3d_positions(&positions_nd, &display_dims, 4, 1, &mut output);

        assert_eq!(output[0], 1.0); // x
        assert_eq!(output[1], 2.0); // y
        assert_eq!(output[2], 0.0); // z = 0 (default)
    }
}
