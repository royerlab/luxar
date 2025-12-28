//! Line segment visibility computation for nD datasets.
//!
//! Computes which line segments are visible in the current nD slice
//! by checking endpoint visibility.

use wasm_bindgen::prelude::*;

/// Check if a single vertex is visible in the nD slice.
///
/// # Arguments
/// * `vertices` - All vertex positions [numVertices * ndim]
/// * `vertex_idx` - Index of the vertex to check
/// * `width` - Width/radius of the vertex
/// * `slice_position` - Current slice position [ndim]
/// * `tolerance` - Tolerance per dimension [ndim]
/// * `ndim` - Number of dimensions
///
/// # Returns
/// true if vertex is visible, false otherwise
fn check_point_visibility(
    vertices: &[f32],
    vertex_idx: usize,
    width: f32,
    slice_position: &[f32],
    tolerance: &[f32],
    ndim: usize,
) -> bool {
    let offset = vertex_idx * ndim;
    let mut dist_sq = 0.0_f32;

    for dim in 0..ndim {
        let delta = vertices[offset + dim] - slice_position[dim];
        let effective_tolerance = tolerance[dim] + width;

        if effective_tolerance > 0.0 {
            let normalized = delta / effective_tolerance;
            dist_sq += normalized * normalized;
        } else if delta.abs() > 1e-6 {
            return false;
        }
    }

    dist_sq <= 1.0
}

/// Compute nD visibility for Lines by checking segment endpoints.
///
/// # Arguments
/// * `vertices` - Vertex positions [numVertices * ndim]
/// * `segments` - Segment indices [numSegments * 2] (pairs of vertex indices)
/// * `widths` - Per-vertex widths [numVertices]
/// * `slice_position` - Current slice position [ndim]
/// * `tolerance` - Tolerance per dimension [ndim]
/// * `ndim` - Number of dimensions
/// * `num_segments` - Total number of segments
/// * `output_mask` - Output visibility mask [numSegments]
///
/// # Returns
/// Number of visible segments
///
/// # Algorithm
/// A segment is visible if EITHER endpoint is visible in the nD slice.
/// Uses per-vertex widths for effective tolerance calculation.
#[wasm_bindgen]
pub fn compute_nd_visibility_lines(
    vertices: &[f32],
    segments: &[u32],
    widths: &[f32],
    slice_position: &[f32],
    tolerance: &[f32],
    ndim: usize,
    num_segments: usize,
    output_mask: &mut [u8],
) -> u32 {
    let mut visible_count = 0;

    for seg_idx in 0..num_segments {
        let v0_idx = segments[seg_idx * 2] as usize;
        let v1_idx = segments[seg_idx * 2 + 1] as usize;

        // Get per-vertex widths
        let width0 = widths[v0_idx];
        let width1 = widths[v1_idx];

        // Check if EITHER endpoint is visible
        let v0_visible =
            check_point_visibility(vertices, v0_idx, width0, slice_position, tolerance, ndim);
        let v1_visible =
            check_point_visibility(vertices, v1_idx, width1, slice_position, tolerance, ndim);

        let visible = v0_visible || v1_visible;
        output_mask[seg_idx] = if visible { 1 } else { 0 };
        if visible {
            visible_count += 1;
        }
    }

    visible_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_visibility_both_endpoints_visible() {
        // 1 segment with both endpoints near origin
        let vertices = vec![
            0.0, 0.0, 0.0, // Vertex 0
            0.5, 0.5, 0.5, // Vertex 1
        ];
        let segments = vec![0, 1]; // One segment
        let widths = vec![0.1, 0.1]; // Per-vertex
        let slice_pos = vec![0.0, 0.0, 0.0];
        let tolerance = vec![1.0, 1.0, 1.0];

        let mut output = vec![0u8; 1];
        let count = compute_nd_visibility_lines(
            &vertices,
            &segments,
            &widths,
            &slice_pos,
            &tolerance,
            3,
            1,
            &mut output,
        );

        assert_eq!(output[0], 1, "Segment should be visible");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_line_visibility_one_endpoint_visible() {
        // 1 segment with one endpoint far away
        let vertices = vec![
            0.0, 0.0, 0.0,    // Vertex 0 (near)
            10.0, 10.0, 10.0, // Vertex 1 (far)
        ];
        let segments = vec![0, 1];
        let widths = vec![0.1, 0.1];
        let slice_pos = vec![0.0, 0.0, 0.0];
        let tolerance = vec![1.0, 1.0, 1.0];

        let mut output = vec![0u8; 1];
        let count = compute_nd_visibility_lines(
            &vertices,
            &segments,
            &widths,
            &slice_pos,
            &tolerance,
            3,
            1,
            &mut output,
        );

        assert_eq!(output[0], 1, "Segment should be visible (one endpoint visible)");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_line_visibility_both_endpoints_hidden() {
        // 1 segment with both endpoints far away
        let vertices = vec![
            10.0, 10.0, 10.0, // Vertex 0 (far)
            20.0, 20.0, 20.0, // Vertex 1 (far)
        ];
        let segments = vec![0, 1];
        let widths = vec![0.1, 0.1];
        let slice_pos = vec![0.0, 0.0, 0.0];
        let tolerance = vec![1.0, 1.0, 1.0];

        let mut output = vec![0u8; 1];
        let count = compute_nd_visibility_lines(
            &vertices,
            &segments,
            &widths,
            &slice_pos,
            &tolerance,
            3,
            1,
            &mut output,
        );

        assert_eq!(output[0], 0, "Segment should be hidden (both endpoints far)");
        assert_eq!(count, 0);
    }
}
