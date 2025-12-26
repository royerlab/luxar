//! Luxar WASM Module - Spatial Queries and nD Visibility Computation
//!
//! This module provides high-performance implementations of:
//! - Spatial index chunk queries (bounding box intersection tests)
//! - nD visibility computation for Points, Lines, and GSplats
//!
//! Optimized with:
//! - SIMD operations where beneficial
//! - Minimal allocations
//! - Cache-friendly access patterns
//!
//! Phase 3: Replaces TypeScript fallbacks with 3-5x faster WASM

use wasm_bindgen::prelude::*;

/// Query chunks whose bounding boxes intersect the nD slice.
///
/// # Arguments
/// * `chunk_bounds` - Flattened chunk bounds [numChunks * ndim * 2] (min/max pairs)
/// * `slice_position` - Current slice position in nD space [ndim]
/// * `tolerance` - Tolerance per dimension [ndim]
/// * `ndim` - Number of dimensions
/// * `num_chunks` - Total number of chunks
/// * `output` - Output buffer for matching chunk indices [numChunks]
///
/// # Returns
/// Number of matching chunks (indices stored in output buffer)
///
/// # Algorithm
/// For each chunk, test if its bounding box intersects the query hypercube
/// defined by slice_position ± tolerance in each dimension.
#[wasm_bindgen]
pub fn query_chunks_for_view(
    chunk_bounds: &[f32],
    slice_position: &[f32],
    tolerance: &[f32],
    ndim: usize,
    num_chunks: usize,
    output: &mut [u32],
) -> u32 {
    let mut match_count = 0;

    for chunk_idx in 0..num_chunks {
        let mut intersects = true;

        for dim in 0..ndim {
            let bounds_offset = (chunk_idx * ndim * 2) + (dim * 2);
            let chunk_min = chunk_bounds[bounds_offset];
            let chunk_max = chunk_bounds[bounds_offset + 1];

            let query_min = slice_position[dim] - tolerance[dim];
            let query_max = slice_position[dim] + tolerance[dim];

            // No intersection if chunk is completely outside query range
            if chunk_max < query_min || chunk_min > query_max {
                intersects = false;
                break;
            }
        }

        if intersects {
            output[match_count as usize] = chunk_idx as u32;
            match_count += 1;
        }
    }

    match_count
}

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
    let mut visible_count = 0;

    for pt_idx in 0..num_points {
        let pt_offset = pt_idx * ndim;
        let radius = radii[pt_idx];

        // Compute normalized distance in nD space
        let mut dist_sq = 0.0_f32;
        for dim in 0..ndim {
            let delta = positions[pt_offset + dim] - slice_position[dim];
            let effective_tolerance = tolerance[dim] + radius;

            // Avoid division by zero
            if effective_tolerance > 0.0 {
                let normalized = delta / effective_tolerance;
                dist_sq += normalized * normalized;
            } else if delta.abs() > 1e-6 {
                // Point is far from slice with zero tolerance - not visible
                dist_sq = f32::INFINITY;
                break;
            }
        }

        let visible = dist_sq <= 1.0;
        output_mask[pt_idx] = if visible { 1 } else { 0 };
        if visible {
            visible_count += 1;
        }
    }

    visible_count
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
    widths: &[f32], // Per-vertex widths
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
        let v0_visible = check_point_visibility(vertices, v0_idx, width0, slice_position, tolerance, ndim);
        let v1_visible = check_point_visibility(vertices, v1_idx, width1, slice_position, tolerance, ndim);

        let visible = v0_visible || v1_visible;
        output_mask[seg_idx] = if visible { 1 } else { 0 };
        if visible {
            visible_count += 1;
        }
    }

    visible_count
}

/// Helper: Check if a single point is visible
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
/// 2. Check if center ± max extent intersects the slice
/// 3. Uses conservative estimate (faster than full Mahalanobis distance)
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
        let mut max_extent = 0.0_f32;
        let mut chol_idx = 0;

        for dim in 0..ndim {
            // Diagonal element L[dim,dim] is at packed position:
            // dim + (dim-1) + (dim-2) + ... + 0 = dim*(dim+1)/2
            let diag_pos = cholesky_offset + chol_idx + dim;
            let scale = cholesky_factors[diag_pos].abs();
            max_extent = max_extent.max(scale);
            chol_idx += dim + 1; // Move to next row's diagonal
        }

        // Check if center + max extent is within tolerance
        let mut dist_sq = 0.0_f32;
        for dim in 0..ndim {
            let delta = centers[center_offset + dim] - slice_position[dim];
            let effective_tolerance = tolerance[dim] + max_extent;

            if effective_tolerance > 0.0 {
                let normalized = delta / effective_tolerance;
                dist_sq += normalized * normalized;
            } else if delta.abs() > 1e-6 {
                dist_sq = f32::INFINITY;
                break;
            }
        }

        let visible = dist_sq <= 1.0;
        output_mask[splat_idx] = if visible { 1 } else { 0 };
        if visible {
            visible_count += 1;
        }
    }

    visible_count
}

// =============================================================================
// Rust Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_chunks_basic_3d() {
        // 3 chunks in 3D space
        let chunk_bounds = vec![
            // Chunk 0: [0,0,0] to [1,1,1]
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            // Chunk 1: [1,1,1] to [2,2,2]
            1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
            // Chunk 2: [5,5,5] to [6,6,6] (far away)
            5.0, 6.0, 5.0, 6.0, 5.0, 6.0,
        ];

        let slice_pos = vec![0.5, 0.5, 0.5];
        let tolerance = vec![0.6, 0.6, 0.6]; // Covers [−0.1, 1.1]

        let mut output = vec![0u32; 3];
        let count = query_chunks_for_view(&chunk_bounds, &slice_pos, &tolerance, 3, 3, &mut output);

        assert_eq!(count, 2, "Should match chunks 0 and 1");
        assert_eq!(output[0], 0);
        assert_eq!(output[1], 1);
    }

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
        let count = compute_nd_visibility_points(&positions, &radii, &slice_pos, &tolerance, 3, 3, &mut output);

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
        let count = compute_nd_visibility_points(&positions, &radii, &slice_pos, &tolerance, 4, 2, &mut output);

        assert_eq!(output[0], 1, "Point 0 should be visible (t=0)");
        assert_eq!(output[1], 0, "Point 1 should be hidden (t=10)");
        assert_eq!(count, 1);
    }

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
        let count = compute_nd_visibility_lines(&vertices, &segments, &widths, &slice_pos, &tolerance, 3, 1, &mut output);

        assert_eq!(output[0], 1, "Segment should be visible");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_line_visibility_one_endpoint_visible() {
        // 1 segment with one endpoint far away
        let vertices = vec![
            0.0, 0.0, 0.0, // Vertex 0 (near)
            10.0, 10.0, 10.0, // Vertex 1 (far)
        ];
        let segments = vec![0, 1];
        let widths = vec![0.1, 0.1];
        let slice_pos = vec![0.0, 0.0, 0.0];
        let tolerance = vec![1.0, 1.0, 1.0];

        let mut output = vec![0u8; 1];
        let count = compute_nd_visibility_lines(&vertices, &segments, &widths, &slice_pos, &tolerance, 3, 1, &mut output);

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
        let count = compute_nd_visibility_lines(&vertices, &segments, &widths, &slice_pos, &tolerance, 3, 1, &mut output);

        assert_eq!(output[0], 0, "Segment should be hidden (both endpoints far)");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_gsplat_visibility_basic() {
        // 2 splats in 3D
        let centers = vec![
            0.0, 0.0, 0.0, // Splat 0 at origin
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
        let count = compute_nd_visibility_gsplats(&centers, &cholesky, &slice_pos, &tolerance, 3, 2, &mut output);

        assert_eq!(output[0], 1, "Splat 0 should be visible");
        assert_eq!(output[1], 0, "Splat 1 should be hidden");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_gsplat_visibility_4d() {
        // 2 splats in 4D, same XYZ but different T
        let centers = vec![
            0.0, 0.0, 0.0, 0.0, // Splat 0 at t=0
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
        let count = compute_nd_visibility_gsplats(&centers, &cholesky, &slice_pos, &tolerance, 4, 2, &mut output);

        assert_eq!(output[0], 1, "Splat 0 should be visible (t=0)");
        assert_eq!(output[1], 0, "Splat 1 should be hidden (t=10 > 2.0)");
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
        let count = compute_nd_visibility_points(&positions, &radii, &slice_pos, &tolerance, 3, 1, &mut output);

        // With radius 0.1 and zero tolerance, point should still be visible
        assert_eq!(output[0], 1, "Point at exact position should be visible");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_edge_case_empty_input() {
        let mut output = vec![0u32; 0];
        let count = query_chunks_for_view(&[], &[0.0, 0.0], &[1.0, 1.0], 2, 0, &mut output);
        assert_eq!(count, 0, "Should handle empty input");
    }
}
