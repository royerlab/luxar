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

    // The squared length is accumulated in f64: in f32 it overflows to
    // infinity once a component delta exceeds sqrt(f32::MAX) ≈ 1.8e19, while
    // the TypeScript mirror reads the same f32 inputs and computes in f64,
    // returning the true value — the two backends must agree (same fix as
    // the compute_cap_suppression direction loop below). The result is
    // stored back as f32.
    for i in 0..visible_count {
        let dx = end_positions[i * 3] as f64 - start_positions[i * 3] as f64;
        let dy = end_positions[i * 3 + 1] as f64 - start_positions[i * 3 + 1] as f64;
        let dz = end_positions[i * 3 + 2] as f64 - start_positions[i * 3 + 2] as f64;
        output[i] = (dx * dx + dy * dy + dz * dz).sqrt() as f32;
    }
}

/// Per-endpoint cap suppression in [0, 1] (drives the shader cap factor).
///
/// The line fragment shader dims each segment towards `0.5` at its own
/// endpoints, with each endpoint's ramp lifted by its own suppression
/// (`capFactor = min(mix(startRamp, 1.0, suppress_start), mix(endRamp, 1.0,
/// suppress_end))`). That dimming is
/// only correct where a neighbouring quad *overlaps* the endpoint and adds the
/// missing half back — the quads span exactly `[start, end]`, so collinear
/// neighbours tile instead of overlapping and the dimming becomes a dark notch
/// at every interior joint. This kernel computes, per endpoint, how much of
/// that dimming to suppress:
///
/// - `1.0` — a clipped endpoint (the slice boundary cut the polyline
///   mid-segment; the real endpoint is outside the slice, so no neighbour will
///   arrive to sum with) or a straight-through interior joint (quads tile, no
///   overlap, nothing to compensate).
/// - `0.0` — a free polyline end (keep the soft cap) or a sharp bend / branch
///   point, where the quads genuinely do overlap and the `0.5 + 0.5` sum is
///   what makes the joint come out flat.
///
/// Between those, the value is `clamp(-dot(away_a, away_b), 0, 1)` where
/// `away_*` is the unit direction from the shared vertex back along each
/// segment. A straight continuation gives opposite `away` vectors (dot -1 ->
/// suppression 1); a 90-degree-or-sharper turn gives dot >= 0 -> suppression 0,
/// preserving the pre-existing behaviour exactly. The overlap area between two
/// quads grows monotonically with the turn angle, so this interpolates between
/// the two regimes in the right direction. It is a per-endpoint scalar
/// approximating a spatially varying ideal, deliberately biased conservative
/// (never brighter than the previous behaviour at sharp angles).
///
/// Only **degree-2** vertices are treated as joints: at a branch point (3+
/// segments meeting) the quads all overlap near the hub and suppressing would
/// stack them into a bright nub, so those keep the cap.
///
/// A joint is recognised only between two endpoints that both actually *reach*
/// the shared vertex — a visible neighbour trimmed away from the vertex
/// (`t1 > 0` / `t2 < 1`) does not anchor a joint, and an invisible neighbour
/// does not either.
///
/// Joints are matched by vertex **index**, not by position. A chain whose
/// segments each carry their own duplicate copy of the shared point (what
/// `line_type="segments"` emits for abutting segments) is geometrically
/// continuous but has no shared index, so it keeps the cap at every joint and
/// still shows the notch. That is deliberate: a shared index means "the same
/// vertex of the same polyline", whereas position matching would also fuse two
/// unrelated lines that merely touch. Author connected geometry as
/// `line_type="polyline"` (or reuse indices) to get continuous joints.
///
/// # Cost
///
/// Unlike the other lines kernels, this one allocates inside wasm: `code_sum`
/// + `degree` (5 B per source vertex) and `dirs` (12 B per visible segment).
/// On the largest bundled lines scene (2.7M segments) `dirs` alone is
/// ~32.4 MB, and for polyline-shaped data (vertices ≈ segments) the two
/// vertex tables add another ~13.5 MB — a ~46 MB marginal high-water mark
/// (before allocator overhead) on the module's linear memory, on top of the
/// ~138 MB the four pre-existing lines kernels already reach through
/// wasm-bindgen slice marshalling — and wasm memory is never returned to the
/// OS, so it stays reserved for the worker's lifetime. Both alternatives are
/// worse: dropping `dirs` and normalising per endpoint-pair costs ~2x the
/// runtime of the whole lines projection, and quantising it breaks the
/// bit-exact agreement with the TypeScript mirror.
///
/// # Arguments
/// - `segments`: Vertex index pairs [numSegments * 2]
/// - `visibility`: Visibility mask [numSegments]
/// - `t1_params`: Start interpolation parameters [numSegments]
/// - `t2_params`: End interpolation parameters [numSegments]
/// - `num_segments`: Total number of segments
/// - `num_vertices`: Total number of source vertices (bounds the touch tables)
/// - `start_positions`: Clipped start positions [visibleCount * 3]
/// - `end_positions`: Clipped end positions [visibleCount * 3]
/// - `output_start`: Output start suppression [visibleCount]
/// - `output_end`: Output end suppression [visibleCount]
///
/// # Returns
/// Number of visible segments written
#[wasm_bindgen]
pub fn compute_cap_suppression(
    segments: &[u32],
    visibility: &[u8],
    t1_params: &[f32],
    t2_params: &[f32],
    num_segments: usize,
    num_vertices: usize,
    start_positions: &[f32],
    end_positions: &[f32],
    output_start: &mut [f32],
    output_end: &mut [f32],
) -> u32 {
    // Endpoint code: (out_idx << 1) | end_bit, end_bit 0 = start, 1 = end.
    // `code_sum` accumulates the codes of the endpoints landing exactly on each
    // vertex and `degree` counts them (saturating at 3, so branch points stay
    // distinguishable from ordinary joints). At degree 2 the partner is simply
    // `code_sum - my_code` — one scattered array instead of two, which halves
    // the cache traffic of this vertex-indexed pass. Both start zeroed: degree
    // gates every read, so a sentinel fill would be pure cost.
    let mut code_sum: Vec<i32> = vec![0; num_vertices];
    let mut degree: Vec<u8> = vec![0; num_vertices];

    let mut out_idx: usize = 0;
    for seg_idx in 0..num_segments {
        if visibility[seg_idx] == 0 {
            continue;
        }
        let code = (out_idx as i32) << 1;
        if t1_params[seg_idx] <= 0.0 {
            register_touch(
                segments[seg_idx * 2] as usize,
                code,
                num_vertices,
                &mut code_sum,
                &mut degree,
            );
        }
        if t2_params[seg_idx] >= 1.0 {
            register_touch(
                segments[seg_idx * 2 + 1] as usize,
                code | 1,
                num_vertices,
                &mut code_sum,
                &mut degree,
            );
        }
        out_idx += 1;
    }

    let visible_count = out_idx;

    // One unit direction per visible segment, computed ONCE (sequential, one
    // sqrt each). The joint test then needs no normalisation at all: the "away"
    // vector at an endpoint is +dir for a start and -dir for an end, so
    //   dot(away_mine, away_partner) = s_mine * s_partner * dot(dir_i, dir_p)
    // with s = +1 / -1. A degenerate (zero-length or non-finite) segment gets a
    // zero direction, whose dot is 0 → suppression 0 → the cap is kept, which is
    // exactly the wanted fallback. Normalising per endpoint-pair instead cost
    // ~4 sqrt and two scattered position reads per segment and dominated the
    // whole lines projection.
    // The squared length is accumulated in f64. In f32 it overflows to
    // infinity once a component delta exceeds sqrt(f32::MAX) ≈ 1.8e19, which
    // would zero the direction and silently drop the joint on a huge-coordinate
    // scene — and, worse, disagree with the TypeScript mirror (which reads the
    // same f32 inputs but computes in f64, so it does NOT overflow). That
    // divergence is observable: the >16D TS backend would render the joint
    // suppressed while the WASM path rendered it capped. f64 here is both the
    // scale-free answer and the one that keeps the two backends identical.
    let mut dirs: Vec<f32> = vec![0.0; visible_count * 3];
    for i in 0..visible_count {
        let o = i * 3;
        let dx = end_positions[o] as f64 - start_positions[o] as f64;
        let dy = end_positions[o + 1] as f64 - start_positions[o + 1] as f64;
        let dz = end_positions[o + 2] as f64 - start_positions[o + 2] as f64;
        let len = (dx * dx + dy * dy + dz * dz).sqrt();
        if len.is_finite() && len > 0.0 {
            dirs[o] = (dx / len) as f32;
            dirs[o + 1] = (dy / len) as f32;
            dirs[o + 2] = (dz / len) as f32;
        }
    }

    out_idx = 0;
    for seg_idx in 0..num_segments {
        if visibility[seg_idx] == 0 {
            continue;
        }
        let code = (out_idx as i32) << 1;

        output_start[out_idx] = if t1_params[seg_idx] > 0.0 {
            1.0
        } else {
            joint_suppression(
                segments[seg_idx * 2] as usize,
                code,
                num_vertices,
                &code_sum,
                &degree,
                &dirs,
            )
        };
        output_end[out_idx] = if t2_params[seg_idx] < 1.0 {
            1.0
        } else {
            joint_suppression(
                segments[seg_idx * 2 + 1] as usize,
                code | 1,
                num_vertices,
                &code_sum,
                &degree,
                &dirs,
            )
        };

        out_idx += 1;
    }

    out_idx as u32
}

/// Record one endpoint landing exactly on `vertex`, saturating degree at 3.
fn register_touch(
    vertex: usize,
    code: i32,
    num_vertices: usize,
    code_sum: &mut [i32],
    degree: &mut [u8],
) {
    if vertex >= num_vertices {
        return; // out-of-range index; upstream validation rejects these
    }
    let d = degree[vertex];
    if d < 2 {
        code_sum[vertex] += code;
        degree[vertex] = d + 1;
    } else {
        degree[vertex] = 3; // branch point — the sum is no longer meaningful
    }
}

/// Suppression for one unclipped endpoint sitting on `vertex`.
///
/// `dirs` holds one precomputed unit segment direction per visible segment.
fn joint_suppression(
    vertex: usize,
    my_code: i32,
    num_vertices: usize,
    code_sum: &[i32],
    degree: &[u8],
    dirs: &[f32],
) -> f32 {
    if vertex >= num_vertices || degree[vertex] != 2 {
        return 0.0; // free end, branch point, or unregistered — keep the cap
    }
    let partner = code_sum[vertex] - my_code;
    if partner == my_code {
        return 0.0; // self-segment registered both of its own endpoints here
    }
    let mo = ((my_code >> 1) as usize) * 3;
    let po = ((partner >> 1) as usize) * 3;
    if mo + 2 >= dirs.len() || po + 2 >= dirs.len() {
        return 0.0;
    }
    // f64 to stay bit-identical to the TypeScript mirror, which reads the same
    // f32 directions but accumulates in f64 (see the direction loop above).
    let dot = dirs[mo] as f64 * dirs[po] as f64
        + dirs[mo + 1] as f64 * dirs[po + 1] as f64
        + dirs[mo + 2] as f64 * dirs[po + 2] as f64;
    // away = +dir at a start endpoint, -dir at an end endpoint, so the product
    // of the two signs is +1 exactly when the endpoint bits agree.
    let sign = if (my_code & 1) == (partner & 1) {
        1.0
    } else {
        -1.0
    };
    (-(sign * dot)).clamp(0.0, 1.0) as f32
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

    /// Component deltas past sqrt(f32::MAX) ≈ 1.8e19 must not overflow to
    /// infinity: the squared length accumulates in f64 (matching the
    /// TypeScript mirror, which computes in f64 from the same f32 inputs).
    #[test]
    fn test_calculate_segment_lengths_huge_coordinates_no_f32_overflow() {
        let starts: Vec<f32> = vec![-1e30, 0.0, 0.0];
        let ends: Vec<f32> = vec![1e30, 0.0, 0.0];
        let mut output = vec![0.0f32; 1];

        calculate_segment_lengths(&starts, &ends, 1, &mut output);

        assert!(output[0].is_finite());
        // Exact expected value: the f64 delta of the two f32 inputs, rounded
        // back to f32 — bit-identical to the TypeScript mirror.
        let expected = (1e30f32 as f64 - (-1e30f32) as f64) as f32;
        assert_eq!(output[0], expected);
    }

    /// 4 disjoint segments (no shared vertices) → clipped-flag path only.
    #[test]
    fn test_compute_cap_suppression_clipped_flags() {
        let segments: Vec<u32> = (0..8).collect();
        let visibility: Vec<u8> = vec![1, 1, 1, 1];
        let t1_params: Vec<f32> = vec![0.0, 0.3, 0.0, 0.2];
        let t2_params: Vec<f32> = vec![1.0, 1.0, 0.7, 0.8];
        let mut start_pos = vec![0.0f32; 12];
        let mut end_pos = vec![0.0f32; 12];
        for i in 0..4 {
            start_pos[i * 3] = (i as f32) * 10.0;
            end_pos[i * 3] = (i as f32) * 10.0 + 1.0;
        }

        let mut out_start = vec![0.0f32; 4];
        let mut out_end = vec![0.0f32; 4];

        let count = compute_cap_suppression(
            &segments,
            &visibility,
            &t1_params,
            &t2_params,
            4,
            8,
            &start_pos,
            &end_pos,
            &mut out_start,
            &mut out_end,
        );

        assert_eq!(count, 4);
        // t1=0.0, t2=1.0 -> neither clipped, no neighbour -> free ends
        assert_eq!(out_start[0], 0.0);
        assert_eq!(out_end[0], 0.0);
        // t1=0.3, t2=1.0 -> start clipped
        assert_eq!(out_start[1], 1.0);
        assert_eq!(out_end[1], 0.0);
        // t1=0.0, t2=0.7 -> end clipped
        assert_eq!(out_start[2], 0.0);
        assert_eq!(out_end[2], 1.0);
        // t1=0.2, t2=0.8 -> both clipped
        assert_eq!(out_start[3], 1.0);
        assert_eq!(out_end[3], 1.0);
    }

    /// Straight-through joint suppresses; free outer ends keep the cap.
    #[test]
    fn test_compute_cap_suppression_straight_joint() {
        let segments: Vec<u32> = vec![0, 1, 1, 2];
        let visibility: Vec<u8> = vec![1, 1];
        let t1_params: Vec<f32> = vec![0.0, 0.0];
        let t2_params: Vec<f32> = vec![1.0, 1.0];
        let start_pos: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let end_pos: Vec<f32> = vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let mut out_start = vec![0.0f32; 2];
        let mut out_end = vec![0.0f32; 2];

        compute_cap_suppression(
            &segments,
            &visibility,
            &t1_params,
            &t2_params,
            2,
            3,
            &start_pos,
            &end_pos,
            &mut out_start,
            &mut out_end,
        );

        assert_eq!(out_start[0], 0.0);
        assert!((out_end[0] - 1.0).abs() < 1e-6);
        assert!((out_start[1] - 1.0).abs() < 1e-6);
        assert_eq!(out_end[1], 0.0);
    }

    /// A 90-degree bend keeps the cap; a branch point keeps the cap.
    #[test]
    fn test_compute_cap_suppression_bend_and_branch() {
        // v0 -> v1 along +x, then v1 -> v2 along +y.
        let mut out_start = vec![0.0f32; 2];
        let mut out_end = [0.0f32; 2];
        compute_cap_suppression(
            &[0, 1, 1, 2],
            &[1, 1],
            &[0.0, 0.0],
            &[1.0, 1.0],
            2,
            3,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            &mut out_start,
            &mut out_end,
        );
        assert_eq!(out_start[1], 0.0);

        // Three segments radiating from vertex 0 — a hub, not a joint.
        let mut hub = vec![0.0f32; 3];
        let mut hub_end = [0.0f32; 3];
        compute_cap_suppression(
            &[0, 1, 0, 2, 0, 3],
            &[1, 1, 1],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            3,
            4,
            &[0.0; 9],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &mut hub,
            &mut hub_end,
        );
        assert_eq!(hub, vec![0.0, 0.0, 0.0]);
    }

    /// A gentle 45-degree bend interpolates: suppression = cos(45°) ≈ 0.7071
    /// (mirrors the TypeScript reference test — the value must be genuinely
    /// fractional, not quantised to 0/1).
    #[test]
    fn test_compute_cap_suppression_fractional_bend() {
        let d = std::f32::consts::FRAC_1_SQRT_2;
        let mut out_start = vec![0.0f32; 2];
        let mut out_end = vec![0.0f32; 2];
        compute_cap_suppression(
            &[0, 1, 1, 2],
            &[1, 1],
            &[0.0, 0.0],
            &[1.0, 1.0],
            2,
            3,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0 + d, d, 0.0],
            &mut out_start,
            &mut out_end,
        );
        assert!((out_end[0] - d).abs() < 1e-5);
        assert!((out_start[1] - d).abs() < 1e-5);
    }

    /// Both segments END at the shared vertex (v0 -> v1 <- v2): opposing
    /// stored orientation, but geometrically a straight continuation — the
    /// endpoint-bit sign flip must still yield full suppression.
    #[test]
    fn test_compute_cap_suppression_opposing_orientation() {
        let mut out_start = vec![0.0f32; 2];
        let mut out_end = vec![0.0f32; 2];
        compute_cap_suppression(
            &[0, 1, 2, 1],
            &[1, 1],
            &[0.0, 0.0],
            &[1.0, 1.0],
            2,
            3,
            &[0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &mut out_start,
            &mut out_end,
        );
        assert!((out_end[0] - 1.0).abs() < 1e-6);
        assert!((out_end[1] - 1.0).abs() < 1e-6);
        assert_eq!(out_start[0], 0.0);
        assert_eq!(out_start[1], 0.0);
    }

    /// Non-contiguous visibility: the invisible middle segment must not shift
    /// the compacted output indexing — the two SURVIVING segments share vertex
    /// 1 and form a straight joint (guards against a source-order vs
    /// compacted-index mixup in the direction table).
    #[test]
    fn test_compute_cap_suppression_non_contiguous_visibility() {
        // seg0: v0 -> v1 (visible), seg1: v3 -> v4 (culled, disjoint),
        // seg2: v1 -> v2 (visible). Positions are compacted: 2 visible only.
        let mut out_start = vec![0.0f32; 2];
        let mut out_end = vec![0.0f32; 2];
        let count = compute_cap_suppression(
            &[0, 1, 3, 4, 1, 2],
            &[1, 0, 1],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            3,
            5,
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            &mut out_start,
            &mut out_end,
        );
        assert_eq!(count, 2);
        // Straight-through joint at v1; free outer ends keep the cap.
        assert_eq!(out_start[0], 0.0);
        assert!((out_end[0] - 1.0).abs() < 1e-6);
        assert!((out_start[1] - 1.0).abs() < 1e-6);
        assert_eq!(out_end[1], 0.0);
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
