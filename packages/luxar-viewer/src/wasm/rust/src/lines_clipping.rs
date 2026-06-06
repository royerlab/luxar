//! Lines Clipping for nD → 3D Conversion
//!
//! This module provides WASM-accelerated line segment clipping for nD visualization.
//!
//! ## Clipping Algorithm (Liang-Barsky based)
//!
//! For each segment, handles 5 visibility cases:
//! - A: Both endpoints IN slice → Full segment visible
//! - B: P1 IN, P2 OUT → Clip P2 to boundary
//! - C: P1 OUT, P2 IN → Clip P1 to boundary
//! - D: Both OUT, opposite sides → Clip both (segment crosses slice)
//! - E: Both OUT, same side → Invisible
//!
//! ## Performance
//!
//! The batch function processes all segments in a single WASM call,
//! avoiding JS↔WASM call overhead per segment.

use wasm_bindgen::prelude::*;

use crate::common::{validate_ndim, MAX_SUPPORTED_DIMS, SEGMENT_PARALLEL_EPSILON};

/// Alias for readability within this module
const MAX_DIMS: usize = MAX_SUPPORTED_DIMS;

/// Clip a single segment to the nD slice and return interpolation parameters.
///
/// Returns (visible, t1, t2) where:
/// - visible: true if any part of segment intersects slice
/// - t1: interpolation parameter for clipped start (0.0 = original start)
/// - t2: interpolation parameter for clipped end (1.0 = original end)
///
/// The clipped points can be computed as:
/// - clipped_p1 = p1 + t1 * (p2 - p1)
/// - clipped_p2 = p1 + t2 * (p2 - p1)
#[wasm_bindgen]
pub fn clip_segment_single(
    p1: &[f32],
    p2: &[f32],
    slice_position: &[f32],
    tolerance: &[f32],
    display_dims: &[u32],
    ndim: usize,
) -> Vec<f32> {
    let mut t1: f32 = 0.0;
    let mut t2: f32 = 1.0;

    validate_ndim(ndim, "clip_segment_single");

    // OPTIMIZATION: Use fixed-size array instead of HashSet (zero allocation)
    let mut is_display_dim = [false; MAX_DIMS];
    for &d in display_dims {
        if (d as usize) < MAX_DIMS {
            is_display_dim[d as usize] = true;
        }
    }

    for dim in 0..ndim {
        if is_display_dim[dim] {
            continue; // Skip displayed dimensions
        }

        let tol = tolerance[dim];
        let slice_center = slice_position[dim];
        let slice_min = slice_center - tol;
        let slice_max = slice_center + tol;

        let v1 = p1[dim];
        let v2 = p2[dim];

        // Classify endpoints relative to slice
        let p1_in = v1 >= slice_min && v1 <= slice_max;
        let p2_in = v2 >= slice_min && v2 <= slice_max;

        if p1_in && p2_in {
            continue; // Both in - no clipping for this dimension
        }

        if !p1_in && !p2_in {
            // Both out - check if on same side (Case E: invisible)
            if (v1 < slice_min && v2 < slice_min) || (v1 > slice_max && v2 > slice_max) {
                return vec![0.0, 0.0, 0.0]; // [visible=0, t1, t2]
            }
            // Opposite sides - will clip both (Case D)
        }

        // Compute intersection parameters
        let dv = v2 - v1;
        if dv.abs() < SEGMENT_PARALLEL_EPSILON {
            continue; // Parallel to slice
        }

        // OPTIMIZATION: Avoid two divisions - use reciprocal multiplication
        let inv_dv = 1.0 / dv;
        let t_min = (slice_min - v1) * inv_dv;
        let t_max = (slice_max - v1) * inv_dv;

        // Clip t1 (entry) and t2 (exit)
        if dv > 0.0 {
            t1 = t1.max(t_min);
            t2 = t2.min(t_max);
        } else {
            t1 = t1.max(t_max);
            t2 = t2.min(t_min);
        }

        if t1 >= t2 {
            return vec![0.0, 0.0, 0.0]; // No valid range
        }
    }

    vec![1.0, t1, t2] // [visible=1, t1, t2]
}

/// Batch clip all segments and output visibility mask and interpolation parameters.
///
/// This is the main workhorse function - processes all segments in one WASM call.
///
/// # Arguments
/// - `positions`: Vertex positions [numVertices * ndim]
/// - `segments`: Segment indices [numSegments * 2]
/// - `slice_position`: Current slice position [ndim]
/// - `tolerance`: Per-dimension tolerance [ndim]
/// - `display_dims`: Which dimensions to display [numDisplayDims]
/// - `ndim`: Number of dimensions
/// - `num_segments`: Number of segments
/// - `output_visibility`: Output visibility mask [numSegments]
/// - `output_t1`: Output t1 parameters [numSegments]
/// - `output_t2`: Output t2 parameters [numSegments]
///
/// # Returns
/// Number of visible segments
#[wasm_bindgen]
pub fn clip_segments_batch(
    positions: &[f32],
    segments: &[u32],
    slice_position: &[f32],
    tolerance: &[f32],
    display_dims: &[u32],
    ndim: usize,
    num_segments: usize,
    output_visibility: &mut [u8],
    output_t1: &mut [f32],
    output_t2: &mut [f32],
) -> u32 {
    validate_ndim(ndim, "clip_segments_batch");

    debug_assert!(
        output_visibility.len() >= num_segments,
        "output_visibility too small: {} < {}",
        output_visibility.len(),
        num_segments
    );
    debug_assert!(
        output_t1.len() >= num_segments,
        "output_t1 too small: {} < {}",
        output_t1.len(),
        num_segments
    );
    debug_assert!(
        output_t2.len() >= num_segments,
        "output_t2 too small: {} < {}",
        output_t2.len(),
        num_segments
    );

    // OPTIMIZATION: Use fixed-size array instead of HashSet (zero allocation)
    let mut is_display_dim = [false; MAX_DIMS];
    for &d in display_dims {
        if (d as usize) < MAX_DIMS {
            is_display_dim[d as usize] = true;
        }
    }
    let mut visible_count: u32 = 0;

    for seg_idx in 0..num_segments {
        let v0 = segments[seg_idx * 2] as usize;
        let v1 = segments[seg_idx * 2 + 1] as usize;

        let p1_offset = v0 * ndim;
        let p2_offset = v1 * ndim;

        let mut t1: f32 = 0.0;
        let mut t2: f32 = 1.0;
        let mut visible = true;

        for dim in 0..ndim {
            if is_display_dim[dim] {
                continue;
            }

            let tol = tolerance[dim];
            let slice_center = slice_position[dim];
            let slice_min = slice_center - tol;
            let slice_max = slice_center + tol;

            let v1_val = positions[p1_offset + dim];
            let v2_val = positions[p2_offset + dim];

            let p1_in = v1_val >= slice_min && v1_val <= slice_max;
            let p2_in = v2_val >= slice_min && v2_val <= slice_max;

            if p1_in && p2_in {
                continue;
            }

            if !p1_in
                && !p2_in
                && ((v1_val < slice_min && v2_val < slice_min)
                    || (v1_val > slice_max && v2_val > slice_max))
            {
                visible = false;
                break;
            }

            let dv = v2_val - v1_val;
            if dv.abs() < SEGMENT_PARALLEL_EPSILON {
                continue;
            }

            // OPTIMIZATION: Avoid two divisions - use reciprocal multiplication
            let inv_dv = 1.0 / dv;
            let t_min = (slice_min - v1_val) * inv_dv;
            let t_max = (slice_max - v1_val) * inv_dv;

            if dv > 0.0 {
                t1 = t1.max(t_min);
                t2 = t2.min(t_max);
            } else {
                t1 = t1.max(t_max);
                t2 = t2.min(t_min);
            }

            if t1 >= t2 {
                visible = false;
                break;
            }
        }

        output_visibility[seg_idx] = if visible { 1 } else { 0 };
        output_t1[seg_idx] = t1;
        output_t2[seg_idx] = t2;

        if visible {
            visible_count += 1;
        }
    }

    visible_count
}

/// Interpolate clipped positions to 3D display space.
///
/// For visible segments, computes clipped 3D positions using t1/t2 parameters.
///
/// # Arguments
/// - `positions`: Vertex positions [numVertices * ndim]
/// - `segments`: Segment indices [numSegments * 2]
/// - `visibility`: Visibility mask [numSegments]
/// - `t1`: Start interpolation parameters [numSegments]
/// - `t2`: End interpolation parameters [numSegments]
/// - `display_dims`: Which dimensions to display [3]
/// - `ndim`: Number of dimensions
/// - `num_segments`: Total number of segments
/// - `output_start`: Output start positions [visibleCount * 3]
/// - `output_end`: Output end positions [visibleCount * 3]
///
/// # Returns
/// Number of visible segments written
#[wasm_bindgen]
pub fn interpolate_clipped_positions(
    positions: &[f32],
    segments: &[u32],
    visibility: &[u8],
    t1_params: &[f32],
    t2_params: &[f32],
    display_dims: &[u32],
    ndim: usize,
    num_segments: usize,
    output_start: &mut [f32],
    output_end: &mut [f32],
) -> u32 {
    let num_display = display_dims.len().min(3);
    let mut out_idx: usize = 0;

    for seg_idx in 0..num_segments {
        if visibility[seg_idx] == 0 {
            continue;
        }

        let v0 = segments[seg_idx * 2] as usize;
        let v1 = segments[seg_idx * 2 + 1] as usize;
        let t1 = t1_params[seg_idx];
        let t2 = t2_params[seg_idx];

        let p1_offset = v0 * ndim;
        let p2_offset = v1 * ndim;

        // Interpolate to clipped positions, then project to display dims
        for (out_d, &dim) in display_dims.iter().take(num_display).enumerate() {
            let d = dim as usize;
            let p1_val = positions[p1_offset + d];
            let p2_val = positions[p2_offset + d];

            // Clipped start: p1 + t1 * (p2 - p1)
            output_start[out_idx * 3 + out_d] = p1_val + t1 * (p2_val - p1_val);
            // Clipped end: p1 + t2 * (p2 - p1)
            output_end[out_idx * 3 + out_d] = p1_val + t2 * (p2_val - p1_val);
        }

        // Pad to 3D if fewer than 3 display dims
        for out_d in num_display..3 {
            output_start[out_idx * 3 + out_d] = 0.0;
            output_end[out_idx * 3 + out_d] = 0.0;
        }

        out_idx += 1;
    }

    out_idx as u32
}

/// Linear interpolation helper (scalar).
#[wasm_bindgen]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Linear interpolation for 3D vectors.
///
/// Returns interpolated vector as [x, y, z].
#[wasm_bindgen]
pub fn lerp_vec3(a: &[f32], b: &[f32], t: f32) -> Vec<f32> {
    vec![
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Calculate 3D Euclidean distance.
#[wasm_bindgen]
pub fn distance_3d(a: &[f32], b: &[f32]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Batch interpolate scalar attributes for visible segments.
///
/// Interpolates values using t1/t2 parameters and compacts to visible-only output.
///
/// # Arguments
/// - `values`: Per-vertex attribute values [numVertices]
/// - `segments`: Segment indices [numSegments * 2]
/// - `visibility`: Visibility mask [numSegments]
/// - `t1`: Start interpolation parameters [numSegments]
/// - `t2`: End interpolation parameters [numSegments]
/// - `num_segments`: Total number of segments
/// - `output_start`: Output interpolated start values [visibleCount]
/// - `output_end`: Output interpolated end values [visibleCount]
///
/// # Returns
/// Number of visible segments written
#[wasm_bindgen]
pub fn interpolate_scalars_batch(
    values: &[f32],
    segments: &[u32],
    visibility: &[u8],
    t1_params: &[f32],
    t2_params: &[f32],
    num_segments: usize,
    output_start: &mut [f32],
    output_end: &mut [f32],
) -> u32 {
    let mut out_idx: usize = 0;

    for seg_idx in 0..num_segments {
        if visibility[seg_idx] == 0 {
            continue;
        }

        let v0 = segments[seg_idx * 2] as usize;
        let v1 = segments[seg_idx * 2 + 1] as usize;
        let t1 = t1_params[seg_idx];
        let t2 = t2_params[seg_idx];

        let val0 = values[v0];
        let val1 = values[v1];

        output_start[out_idx] = val0 + t1 * (val1 - val0);
        output_end[out_idx] = val0 + t2 * (val1 - val0);

        out_idx += 1;
    }

    out_idx as u32
}

/// Batch interpolate RGB color attributes for visible segments.
///
/// # Arguments
/// - `colors`: Per-vertex RGB colors [numVertices * 3]
/// - `segments`: Segment indices [numSegments * 2]
/// - `visibility`: Visibility mask [numSegments]
/// - `t1`: Start interpolation parameters [numSegments]
/// - `t2`: End interpolation parameters [numSegments]
/// - `num_segments`: Total number of segments
/// - `output_start`: Output interpolated start colors [visibleCount * 3]
/// - `output_end`: Output interpolated end colors [visibleCount * 3]
///
/// # Returns
/// Number of visible segments written
#[wasm_bindgen]
pub fn interpolate_colors_batch(
    colors: &[f32],
    segments: &[u32],
    visibility: &[u8],
    t1_params: &[f32],
    t2_params: &[f32],
    num_segments: usize,
    output_start: &mut [f32],
    output_end: &mut [f32],
) -> u32 {
    let mut out_idx: usize = 0;

    for seg_idx in 0..num_segments {
        if visibility[seg_idx] == 0 {
            continue;
        }

        let v0 = segments[seg_idx * 2] as usize;
        let v1 = segments[seg_idx * 2 + 1] as usize;
        let t1 = t1_params[seg_idx];
        let t2 = t2_params[seg_idx];

        // OPTIMIZATION: Unrolled loop for RGB (better cache performance)
        let base0 = v0 * 3;
        let base1 = v1 * 3;
        let out_base = out_idx * 3;

        // Red
        let c0 = colors[base0];
        let c1 = colors[base1];
        output_start[out_base] = c0 + t1 * (c1 - c0);
        output_end[out_base] = c0 + t2 * (c1 - c0);

        // Green
        let c0 = colors[base0 + 1];
        let c1 = colors[base1 + 1];
        output_start[out_base + 1] = c0 + t1 * (c1 - c0);
        output_end[out_base + 1] = c0 + t2 * (c1 - c0);

        // Blue
        let c0 = colors[base0 + 2];
        let c1 = colors[base1 + 2];
        output_start[out_base + 2] = c0 + t1 * (c1 - c0);
        output_end[out_base + 2] = c0 + t2 * (c1 - c0);

        out_idx += 1;
    }

    out_idx as u32
}

/// Calculate 3D segment lengths for visible segments.
///
/// # Arguments
/// - `start_positions`: Clipped start positions [visibleCount * 3]
/// - `end_positions`: Clipped end positions [visibleCount * 3]
/// - `visible_count`: Number of visible segments
/// - `output`: Output segment lengths [visibleCount]
#[wasm_bindgen]
pub fn calculate_segment_lengths(
    start_positions: &[f32],
    end_positions: &[f32],
    visible_count: usize,
    output: &mut [f32],
) {
    debug_assert!(
        output.len() >= visible_count,
        "output too small: {} < {}",
        output.len(),
        visible_count
    );

    for i in 0..visible_count {
        let dx = end_positions[i * 3] - start_positions[i * 3];
        let dy = end_positions[i * 3 + 1] - start_positions[i * 3 + 1];
        let dz = end_positions[i * 3 + 2] - start_positions[i * 3 + 2];
        output[i] = (dx * dx + dy * dy + dz * dz).sqrt();
    }
}

/// Mark clipped endpoints (for cap factor adjustment).
///
/// # Arguments
/// - `visibility`: Visibility mask [numSegments]
/// - `t1`: Start interpolation parameters [numSegments]
/// - `t2`: End interpolation parameters [numSegments]
/// - `num_segments`: Total number of segments
/// - `output_start_clipped`: Output start clipped flags [visibleCount]
/// - `output_end_clipped`: Output end clipped flags [visibleCount]
///
/// # Returns
/// Number of visible segments written
#[wasm_bindgen]
pub fn mark_clipped_endpoints(
    visibility: &[u8],
    t1_params: &[f32],
    t2_params: &[f32],
    num_segments: usize,
    output_start_clipped: &mut [u8],
    output_end_clipped: &mut [u8],
) -> u32 {
    let mut out_idx: usize = 0;

    for seg_idx in 0..num_segments {
        if visibility[seg_idx] == 0 {
            continue;
        }

        output_start_clipped[out_idx] = if t1_params[seg_idx] > 0.0 { 1 } else { 0 };
        output_end_clipped[out_idx] = if t2_params[seg_idx] < 1.0 { 1 } else { 0 };

        out_idx += 1;
    }

    out_idx as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_segment_both_in() {
        // 4D segment, both endpoints in slice
        let p1 = vec![0.0, 0.0, 0.0, 5.2];
        let p2 = vec![10.0, 10.0, 10.0, 4.8];
        let slice_pos = vec![0.0, 0.0, 0.0, 5.0];
        let tolerance = vec![1e10, 1e10, 1e10, 0.5];
        let display_dims = vec![0, 1, 2];

        let result = clip_segment_single(&p1, &p2, &slice_pos, &tolerance, &display_dims, 4);

        assert_eq!(result[0], 1.0); // visible
        assert!((result[1] - 0.0).abs() < 1e-6); // t1 = 0
        assert!((result[2] - 1.0).abs() < 1e-6); // t2 = 1
    }

    #[test]
    fn test_clip_segment_both_out_same_side() {
        // Both endpoints below slice
        let p1 = vec![0.0, 0.0, 0.0, 0.0];
        let p2 = vec![10.0, 10.0, 10.0, 2.0];
        let slice_pos = vec![0.0, 0.0, 0.0, 5.0];
        let tolerance = vec![1e10, 1e10, 1e10, 0.5];
        let display_dims = vec![0, 1, 2];

        let result = clip_segment_single(&p1, &p2, &slice_pos, &tolerance, &display_dims, 4);

        assert_eq!(result[0], 0.0); // not visible
    }

    #[test]
    fn test_clip_segment_crosses_slice() {
        // Segment crosses slice in dim3
        let p1 = vec![0.0, 0.0, 0.0, 0.0];
        let p2 = vec![10.0, 10.0, 10.0, 10.0];
        let slice_pos = vec![0.0, 0.0, 0.0, 5.0];
        let tolerance = vec![1e10, 1e10, 1e10, 0.5];
        let display_dims = vec![0, 1, 2];

        let result = clip_segment_single(&p1, &p2, &slice_pos, &tolerance, &display_dims, 4);

        assert_eq!(result[0], 1.0); // visible
        assert!((result[1] - 0.45).abs() < 1e-6); // t1 = 0.45
        assert!((result[2] - 0.55).abs() < 1e-6); // t2 = 0.55
    }

    #[test]
    fn test_lerp() {
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
    }

    #[test]
    fn test_lerp_vec3() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![10.0, 20.0, 30.0];
        let result = lerp_vec3(&a, &b, 0.5);
        assert_eq!(result, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_distance_3d() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert_eq!(distance_3d(&a, &b), 5.0); // 3-4-5 triangle
    }

    #[test]
    fn test_interpolate_clipped_positions() {
        // 2 segments, 4D positions, display_dims=[0,1,2]
        // Segment 0: visible, t1=0.0, t2=0.5 (half clipped)
        // Segment 1: hidden (visibility=0)
        let ndim = 4;
        let positions: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, // v0
            10.0, 0.0, 0.0, 5.0, // v1
            0.0, 10.0, 0.0, 5.0, // v2
            10.0, 10.0, 0.0, 5.0, // v3
        ];
        let segments: Vec<u32> = vec![0, 1, 2, 3];
        let visibility: Vec<u8> = vec![1, 0];
        let t1_params: Vec<f32> = vec![0.0, 0.0];
        let t2_params: Vec<f32> = vec![0.5, 1.0];
        let display_dims: Vec<u32> = vec![0, 1, 2];

        // 1 visible segment × 3 components.
        let mut output_start = vec![0.0f32; 3];
        let mut output_end = vec![0.0f32; 3];

        let count = interpolate_clipped_positions(
            &positions,
            &segments,
            &visibility,
            &t1_params,
            &t2_params,
            &display_dims,
            ndim,
            2,
            &mut output_start,
            &mut output_end,
        );

        assert_eq!(count, 1);
        // Start: v0 + 0.0 * (v1 - v0) = [0, 0, 0]
        assert!((output_start[0] - 0.0).abs() < 1e-6);
        assert!((output_start[1] - 0.0).abs() < 1e-6);
        assert!((output_start[2] - 0.0).abs() < 1e-6);
        // End: v0 + 0.5 * (v1 - v0) = [5, 0, 0]
        assert!((output_end[0] - 5.0).abs() < 1e-6);
        assert!((output_end[1] - 0.0).abs() < 1e-6);
        assert!((output_end[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_scalars_batch() {
        // 3 segments, 2 visible (indices 0 and 2)
        let values: Vec<f32> = vec![1.0, 3.0, 5.0, 7.0];
        let segments: Vec<u32> = vec![0, 1, 1, 2, 2, 3];
        let visibility: Vec<u8> = vec![1, 0, 1];
        let t1_params: Vec<f32> = vec![0.0, 0.0, 0.25];
        let t2_params: Vec<f32> = vec![0.5, 1.0, 1.0];

        let mut output_start = vec![0.0f32; 2];
        let mut output_end = vec![0.0f32; 2];

        let count = interpolate_scalars_batch(
            &values,
            &segments,
            &visibility,
            &t1_params,
            &t2_params,
            3,
            &mut output_start,
            &mut output_end,
        );

        assert_eq!(count, 2);
        // Segment 0: v0=1.0, v1=3.0, t1=0.0, t2=0.5
        // start = 1.0 + 0.0*(3.0-1.0) = 1.0
        // end   = 1.0 + 0.5*(3.0-1.0) = 2.0
        assert!((output_start[0] - 1.0).abs() < 1e-6);
        assert!((output_end[0] - 2.0).abs() < 1e-6);
        // Segment 2: v2=5.0, v3=7.0, t1=0.25, t2=1.0
        // start = 5.0 + 0.25*(7.0-5.0) = 5.5
        // end   = 5.0 + 1.0*(7.0-5.0)  = 7.0
        assert!((output_start[1] - 5.5).abs() < 1e-6);
        assert!((output_end[1] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_colors_batch() {
        // 2 segments, both visible, RGB colors
        let colors: Vec<f32> = vec![
            1.0, 0.0, 0.0, // v0: red
            0.0, 1.0, 0.0, // v1: green
        ];
        let segments: Vec<u32> = vec![0, 1];
        let visibility: Vec<u8> = vec![1];
        let t1_params: Vec<f32> = vec![0.0];
        let t2_params: Vec<f32> = vec![1.0];

        let mut output_start = vec![0.0f32; 3];
        let mut output_end = vec![0.0f32; 3];

        let count = interpolate_colors_batch(
            &colors,
            &segments,
            &visibility,
            &t1_params,
            &t2_params,
            1,
            &mut output_start,
            &mut output_end,
        );

        assert_eq!(count, 1);
        // start at t=0: [1, 0, 0]
        assert!((output_start[0] - 1.0).abs() < 1e-6);
        assert!((output_start[1] - 0.0).abs() < 1e-6);
        assert!((output_start[2] - 0.0).abs() < 1e-6);
        // end at t=1: [0, 1, 0]
        assert!((output_end[0] - 0.0).abs() < 1e-6);
        assert!((output_end[1] - 1.0).abs() < 1e-6);
        assert!((output_end[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_segment_lengths() {
        // 3 segments with known distances
        let starts: Vec<f32> = vec![
            0.0, 0.0, 0.0, // seg0 start
            0.0, 0.0, 0.0, // seg1 start
            0.0, 0.0, 0.0, // seg2 start
        ];
        let ends: Vec<f32> = vec![
            3.0, 4.0, 0.0, // seg0 end -> length 5.0
            1.0, 0.0, 0.0, // seg1 end -> length 1.0
            0.0, 0.0, 0.0, // seg2 end -> length 0.0
        ];
        let mut output = vec![0.0f32; 3];

        calculate_segment_lengths(&starts, &ends, 3, &mut output);

        assert!((output[0] - 5.0).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
        assert!((output[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mark_clipped_endpoints() {
        // 4 segments, all visible, various clipping states
        let visibility: Vec<u8> = vec![1, 1, 1, 1];
        let t1_params: Vec<f32> = vec![0.0, 0.3, 0.0, 0.2];
        let t2_params: Vec<f32> = vec![1.0, 1.0, 0.7, 0.8];

        let mut out_start = vec![0u8; 4];
        let mut out_end = vec![0u8; 4];

        let count = mark_clipped_endpoints(
            &visibility,
            &t1_params,
            &t2_params,
            4,
            &mut out_start,
            &mut out_end,
        );

        assert_eq!(count, 4);
        // t1=0.0, t2=1.0 -> neither clipped
        assert_eq!(out_start[0], 0);
        assert_eq!(out_end[0], 0);
        // t1=0.3, t2=1.0 -> start clipped
        assert_eq!(out_start[1], 1);
        assert_eq!(out_end[1], 0);
        // t1=0.0, t2=0.7 -> end clipped
        assert_eq!(out_start[2], 0);
        assert_eq!(out_end[2], 1);
        // t1=0.2, t2=0.8 -> both clipped
        assert_eq!(out_start[3], 1);
        assert_eq!(out_end[3], 1);
    }

    #[test]
    fn test_clip_segments_batch() {
        // Two segments: one visible, one not
        let positions = vec![
            0.0, 0.0, 0.0, 5.0, // v0: in slice
            10.0, 10.0, 10.0, 5.0, // v1: in slice
            20.0, 20.0, 20.0, 0.0, // v2: out of slice
        ];
        let segments = vec![0, 1, 1, 2]; // seg0: v0-v1, seg1: v1-v2
        let slice_pos = vec![0.0, 0.0, 0.0, 5.0];
        let tolerance = vec![1e10, 1e10, 1e10, 0.5];
        let display_dims = vec![0, 1, 2];

        let mut visibility = vec![0u8; 2];
        let mut t1 = vec![0.0f32; 2];
        let mut t2 = vec![0.0f32; 2];

        let count = clip_segments_batch(
            &positions,
            &segments,
            &slice_pos,
            &tolerance,
            &display_dims,
            4,
            2,
            &mut visibility,
            &mut t1,
            &mut t2,
        );

        // seg0: both in slice -> visible, t1=0, t2=1
        assert_eq!(visibility[0], 1);
        assert!((t1[0] - 0.0).abs() < 1e-6);
        assert!((t2[0] - 1.0).abs() < 1e-6);

        // seg1: crosses slice (v1 at 5.0, v2 at 0.0) -> visible, clipped
        assert_eq!(visibility[1], 1);
        // t where dim3 crosses 4.5: 5.0 + t*(0.0-5.0) = 4.5 -> t = 0.1
        assert!((t1[1] - 0.0).abs() < 1e-6); // v1 is inside
        assert!((t2[1] - 0.1).abs() < 1e-6); // v2 clipped

        assert_eq!(count, 2);
    }
}
