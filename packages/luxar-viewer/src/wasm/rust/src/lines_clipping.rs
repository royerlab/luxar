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

/// Shared per-segment nD slab clip. Returns (visible, t1, t2). On any
/// invisibility case (non-finite coord, same-side rejection, or t1>=t2) it
/// returns (false, t1, t2) with whatever t-params had accumulated; callers
/// decide how to report an invisible segment.
#[inline]
fn clip_segment_core(
    p1: &[f32],
    p2: &[f32],
    slice_position: &[f32],
    tolerance: &[f32],
    is_display_dim: &[bool; MAX_DIMS],
    ndim: usize,
) -> (bool, f32, f32) {
    let mut t1: f32 = 0.0;
    let mut t2: f32 = 1.0;
    for dim in 0..ndim {
        if is_display_dim[dim] {
            continue;
        }
        let tol = tolerance[dim];
        let slice_center = slice_position[dim];
        let slice_min = slice_center - tol;
        let slice_max = slice_center + tol;
        let v1 = p1[dim];
        let v2 = p2[dim];
        // #806: a non-finite (NaN or ±Inf) coordinate on a slicing (non-displayed)
        // dimension cannot be localized against the slice, so the segment is
        // treated as invisible. Enforced identically in the TypeScript backend
        // (`lines-clipping.ts`) so the two backends stay in parity — f32::max/min
        // ignore a NaN operand and would otherwise leave the t-params finite,
        // rendering a segment the TS path drops.
        if !v1.is_finite() || !v2.is_finite() {
            return (false, t1, t2);
        }
        let p1_in = v1 >= slice_min && v1 <= slice_max;
        let p2_in = v2 >= slice_min && v2 <= slice_max;
        if p1_in && p2_in {
            continue;
        }
        if !p1_in
            && !p2_in
            && ((v1 < slice_min && v2 < slice_min) || (v1 > slice_max && v2 > slice_max))
        {
            return (false, t1, t2);
        }
        let dv = v2 - v1;
        if dv.abs() < SEGMENT_PARALLEL_EPSILON {
            continue;
        }
        let inv_dv = 1.0 / dv;
        let t_min = (slice_min - v1) * inv_dv;
        let t_max = (slice_max - v1) * inv_dv;
        if dv > 0.0 {
            t1 = t1.max(t_min);
            t2 = t2.min(t_max);
        } else {
            t1 = t1.max(t_max);
            t2 = t2.min(t_min);
        }
        if t1 >= t2 {
            return (false, t1, t2);
        }
    }
    (true, t1, t2)
}

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
    validate_ndim(ndim, "clip_segment_single");

    // OPTIMIZATION: Use fixed-size array instead of HashSet (zero allocation)
    let mut is_display_dim = [false; MAX_DIMS];
    for &d in display_dims {
        if (d as usize) < MAX_DIMS {
            is_display_dim[d as usize] = true;
        }
    }

    let (visible, t1, t2) =
        clip_segment_core(p1, p2, slice_position, tolerance, &is_display_dim, ndim);
    if visible {
        vec![1.0, t1, t2] // [visible=1, t1, t2]
    } else {
        vec![0.0, 0.0, 0.0] // [visible=0, t1, t2]
    }
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

        let (visible, t1, t2) = clip_segment_core(
            &positions[p1_offset..],
            &positions[p2_offset..],
            slice_position,
            tolerance,
            &is_display_dim,
            ndim,
        );

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
    // the former cap-suppression direction loop). The result is
    // stored back as f32.
    for i in 0..visible_count {
        let dx = end_positions[i * 3] as f64 - start_positions[i * 3] as f64;
        let dy = end_positions[i * 3 + 1] as f64 - start_positions[i * 3 + 1] as f64;
        let dz = end_positions[i * 3 + 2] as f64 - start_positions[i * 3 + 2] as f64;
        output[i] = (dx * dx + dy * dy + dz * dz).sqrt() as f32;
    }
}

/// Per-endpoint **joint code**: how the line shader should treat this endpoint,
/// and — when it is an ordinary two-segment joint — which segment it joins.
///
/// One f32 per endpoint, in the two texel slots that used to carry the cap
/// suppression scalar:
///
/// | value          | meaning                                                     |
/// |----------------|-------------------------------------------------------------|
/// | ` 0`           | free polyline end — keep the soft cap                        |
/// | `-1`           | slice-clipped — the real endpoint is outside the slice, so no |
/// |                | neighbour will ever arrive; suppress the cap entirely        |
/// | `-2`           | degree->=3 hub — several quads already stack here            |
/// | `+(slot + 1)`  | joins visible segment `slot`, at that segment's START        |
/// | `-(slot + 3)`  | joins visible segment `slot`, at that segment's END          |
///
/// `slot` is the segment's index in the VISIBLE (output) stream, i.e. its
/// storage slot in the line texture. The shader's own `aSortedIndex` maps a
/// DRAW slot to a storage slot, so a storage-space partner reference is read
/// directly and needs no adjustment when the depth-sort worker permutes draw
/// order.
///
/// # Why a code and not an angle
///
/// The previous kernel dereferenced the partner only to take one dot product of
/// the two segment directions, and stored `cos(theta)` as a cap-suppression
/// scalar. Storing the partner instead lets the vertex stage do that itself,
/// which buys three things:
///
/// - The bend angle is measured in SCREEN space, per frame, so it tracks the
///   camera. The stored angle could not: a gentle 3D bend that projected sharp
///   kept suppression near 1 while the quads genuinely overlapped (issue #795).
/// - The vertex stage can build real join geometry (a screen-space miter),
///   which closes the uncovered wedge outside every bend (issue #790) — a
///   scalar multiplier never could, because there are no fragments there.
/// - This kernel loses its direction table entirely: no per-segment normalize,
///   no `dirs` allocation (12 B per visible segment — ~32 MB on the largest
///   bundled lines scene, on wasm linear memory that is never returned to the
///   OS), and no f64-vs-f32 care to keep the two backends bit-identical, since
///   integer index arithmetic agrees trivially.
///
/// The fallback the shader derives is exactly the old value:
/// `clamp(-dot(away_a, away_b), 0, 1)` with each `away` pointing from the
/// shared vertex back along its segment reduces to
/// `clamp(dot(dir_mine, dir_partner), 0, 1)` — the same dot product the miter
/// limit needs anyway.
///
/// # Joint recognition
///
/// Only **degree-2** vertices become joints; a hub keeps the cap. A joint is
/// recognised only between two endpoints that both actually *reach* the shared
/// vertex, so a culled or slice-trimmed neighbour does not anchor one.
///
/// Joints are matched by vertex **index**, not by position. A chain whose
/// segments each carry their own duplicate copy of the shared point (what
/// `line_type="segments"` emits for abutting segments) has no shared index, so
/// it keeps the cap at every joint. That is deliberate: a shared index means
/// "the same vertex of the same polyline", whereas position matching would also
/// fuse two unrelated lines that merely touch. Author connected geometry as
/// `line_type="polyline"` (or reuse indices) to get continuous joints.
///
/// # Cost
///
/// Two vertex-indexed tables, 5 B per source vertex (`code_sum` + `degree`),
/// and no per-segment allocation at all.
///
/// # Arguments
/// - `segments`: Vertex index pairs [numSegments * 2]
/// - `visibility`: Visibility mask [numSegments]
/// - `t1_params`: Start interpolation parameters [numSegments]
/// - `t2_params`: End interpolation parameters [numSegments]
/// - `num_segments`: Total number of segments
/// - `num_vertices`: Total number of source vertices (bounds the touch tables)
/// - `output_start`: Output start joint codes [visibleCount]
/// - `output_end`: Output end joint codes [visibleCount]
///
/// # Returns
/// Number of visible segments written
#[wasm_bindgen]
pub fn compute_joint_codes(
    segments: &[u32],
    visibility: &[u8],
    t1_params: &[f32],
    t2_params: &[f32],
    num_segments: usize,
    num_vertices: usize,
    output_start: &mut [f32],
    output_end: &mut [f32],
) -> u32 {
    // Endpoint code: (out_idx << 1) | end_bit, end_bit 0 = start, 1 = end.
    // `code_sum` accumulates the codes of the endpoints landing exactly on each
    // vertex and `degree` counts them (saturating at 3, so hubs stay
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
    debug_assert!(
        output_start.len() >= visible_count,
        "output_start too small: {} < {}",
        output_start.len(),
        visible_count
    );
    debug_assert!(
        output_end.len() >= visible_count,
        "output_end too small: {} < {}",
        output_end.len(),
        visible_count
    );

    out_idx = 0;
    for seg_idx in 0..num_segments {
        if visibility[seg_idx] == 0 {
            continue;
        }
        let code = (out_idx as i32) << 1;

        output_start[out_idx] = if t1_params[seg_idx] > 0.0 {
            JOINT_CLIPPED
        } else {
            joint_code(
                segments[seg_idx * 2] as usize,
                code,
                num_vertices,
                visible_count,
                &code_sum,
                &degree,
            )
        };
        output_end[out_idx] = if t2_params[seg_idx] < 1.0 {
            JOINT_CLIPPED
        } else {
            joint_code(
                segments[seg_idx * 2 + 1] as usize,
                code | 1,
                num_vertices,
                visible_count,
                &code_sum,
                &degree,
            )
        };

        out_idx += 1;
    }

    out_idx as u32
}

/// Free polyline end: keep the soft cap.
pub const JOINT_FREE_END: f32 = 0.0;
/// Slice-clipped endpoint: no neighbour will arrive, so suppress the cap.
pub const JOINT_CLIPPED: f32 = -1.0;
/// Degree->=3 hub: several quads already stack here, so keep the cap.
pub const JOINT_HUB: f32 = -2.0;

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
        degree[vertex] = 3; // hub — the sum is no longer meaningful
    }
}

/// Joint code for one unclipped endpoint sitting on `vertex`.
///
/// The encoded partner slot is exact in f32: the per-node segment capacity is
/// `width x maxTextureSize / 6` = 11.17M at the 16384 ceiling, well inside the
/// 2^24 exact-integer range. Packing the end bit as `(slot << 1) | bit` would
/// NOT be — it reaches 22.35M — which is why the bit rides the sign instead.
fn joint_code(
    vertex: usize,
    my_code: i32,
    num_vertices: usize,
    visible_count: usize,
    code_sum: &[i32],
    degree: &[u8],
) -> f32 {
    if vertex >= num_vertices {
        return JOINT_FREE_END; // unregistered
    }
    match degree[vertex] {
        0 | 1 => return JOINT_FREE_END,
        2 => {}
        _ => return JOINT_HUB,
    }
    let partner = code_sum[vertex] - my_code;
    let slot = partner >> 1;
    // Three ways the difference can fail to name a real partner:
    //
    // - `slot < 0` / `slot >= visible_count`: the sum did not contain `my_code`,
    //   so the difference is arbitrary. That happens when the registering pass
    //   skipped this endpoint but this pass did not — the two run on the
    //   complementary tests `t <= 0` / `!(t > 0)`, which agree for every
    //   ordinary float but BOTH go false for NaN. The predecessor kernel bounded
    //   the same arithmetic against its direction table's length; keep an
    //   equivalent bound here so a code can never name a slot outside the stream
    //   it indexes, independent of the texel writer's capacity clamp.
    // - `slot == my slot`: a zero-length or looping segment registered BOTH of
    //   its own endpoints here, so the difference is its own other endpoint (the
    //   two codes differ only in the end bit, which is why comparing whole codes
    //   is not enough). The old angle-only kernel survived this by returning a
    //   plausible 1.0; a joint code is dereferenced, and a segment mitered
    //   against itself is exactly the asymmetric-join case that produces flaps.
    if slot < 0 || slot as usize >= visible_count || slot == (my_code >> 1) {
        return JOINT_FREE_END;
    }
    // end_bit 0 = the partner's START touches this vertex, 1 = its END does.
    if (partner & 1) == 0 {
        (slot + 1) as f32
    } else {
        -((slot + 3) as f32)
    }
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

    /// #806: a NaN on a NON-displayed (slicing) dim cannot be localized
    /// against the slice, so the segment is invisible — on BOTH endpoints.
    /// Without the guard, f32::max/min ignore the NaN operand and the segment
    /// would render, disagreeing with the TypeScript backend.
    #[test]
    fn test_clip_segment_nan_hidden_dim() {
        let slice_pos = vec![0.0, 0.0, 0.0, 5.0];
        let tolerance = vec![1e10, 1e10, 1e10, 0.5];
        let display_dims = vec![0, 1, 2];

        // NaN on the first endpoint's hidden dim.
        let p1 = vec![1.0, 2.0, 3.0, f32::NAN];
        let p2 = vec![1.0, 2.0, 3.0, 5.0];
        let result = clip_segment_single(&p1, &p2, &slice_pos, &tolerance, &display_dims, 4);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);

        // NaN on the second endpoint's hidden dim.
        let p1 = vec![1.0, 2.0, 3.0, 5.0];
        let p2 = vec![1.0, 2.0, 3.0, f32::NAN];
        let result = clip_segment_single(&p1, &p2, &slice_pos, &tolerance, &display_dims, 4);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    /// #806: +Inf / -Inf on a NON-displayed dim is likewise non-finite and
    /// must mark the segment invisible, on either endpoint.
    #[test]
    fn test_clip_segment_inf_hidden_dim() {
        let slice_pos = vec![0.0, 0.0, 0.0, 5.0];
        let tolerance = vec![1e10, 1e10, 1e10, 0.5];
        let display_dims = vec![0, 1, 2];

        // +Inf on the first endpoint's hidden dim.
        let p1 = vec![1.0, 2.0, 3.0, f32::INFINITY];
        let p2 = vec![1.0, 2.0, 3.0, 5.0];
        let result = clip_segment_single(&p1, &p2, &slice_pos, &tolerance, &display_dims, 4);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);

        // -Inf on the second endpoint's hidden dim.
        let p1 = vec![1.0, 2.0, 3.0, 5.0];
        let p2 = vec![1.0, 2.0, 3.0, f32::NEG_INFINITY];
        let result = clip_segment_single(&p1, &p2, &slice_pos, &tolerance, &display_dims, 4);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
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
    /// 4 disjoint segments (no shared vertices) → sentinel path only.
    #[test]
    fn test_joint_codes_sentinels() {
        let segments: Vec<u32> = (0..8).collect();
        let visibility: Vec<u8> = vec![1, 1, 1, 1];
        let t1_params: Vec<f32> = vec![0.0, 0.3, 0.0, 0.2];
        let t2_params: Vec<f32> = vec![1.0, 1.0, 0.7, 0.8];

        let mut out_start = vec![9.0f32; 4];
        let mut out_end = vec![9.0f32; 4];

        let count = compute_joint_codes(
            &segments,
            &visibility,
            &t1_params,
            &t2_params,
            4,
            8,
            &mut out_start,
            &mut out_end,
        );

        assert_eq!(count, 4);
        // t1=0.0, t2=1.0 -> reaches both vertices, but nothing shares them.
        assert_eq!(out_start[0], JOINT_FREE_END);
        assert_eq!(out_end[0], JOINT_FREE_END);
        // t1=0.3 -> start trimmed off its vertex.
        assert_eq!(out_start[1], JOINT_CLIPPED);
        assert_eq!(out_end[1], JOINT_FREE_END);
        // t2=0.7 -> end trimmed.
        assert_eq!(out_start[2], JOINT_FREE_END);
        assert_eq!(out_end[2], JOINT_CLIPPED);
        // both trimmed.
        assert_eq!(out_start[3], JOINT_CLIPPED);
        assert_eq!(out_end[3], JOINT_CLIPPED);
    }

    /// A two-segment chain: each inner endpoint names the other segment, and the
    /// sign says WHICH of the partner's endpoints is the shared one.
    #[test]
    fn test_joint_codes_chain_partner_and_sign() {
        let mut out_start = vec![9.0f32; 2];
        let mut out_end = vec![9.0f32; 2];

        compute_joint_codes(
            &[0, 1, 1, 2],
            &[1, 1],
            &[0.0, 0.0],
            &[1.0, 1.0],
            2,
            3,
            &mut out_start,
            &mut out_end,
        );

        // Outer ends are free.
        assert_eq!(out_start[0], JOINT_FREE_END);
        assert_eq!(out_end[1], JOINT_FREE_END);
        // Segment 0's END joins segment 1 at segment 1's START -> +(1 + 1).
        assert_eq!(out_end[0], 2.0);
        // Segment 1's START joins segment 0 at segment 0's END -> -(0 + 3).
        assert_eq!(out_start[1], -3.0);
    }

    /// The code is an angle-free FACT about topology: the same chain bent to any
    /// angle, or folded fully back, yields the identical codes. The bend term is
    /// the shader's business now (measured in screen space, per frame).
    #[test]
    fn test_joint_codes_are_angle_independent() {
        let mut straight_s = vec![9.0f32; 2];
        let mut straight_e = vec![9.0f32; 2];
        compute_joint_codes(
            &[0, 1, 1, 2],
            &[1, 1],
            &[0.0, 0.0],
            &[1.0, 1.0],
            2,
            3,
            &mut straight_s,
            &mut straight_e,
        );

        // Same topology — the kernel no longer reads positions at all, so a
        // 90-degree bend and a 180-degree fold cannot change the answer.
        let mut bent_s = vec![9.0f32; 2];
        let mut bent_e = vec![9.0f32; 2];
        compute_joint_codes(
            &[0, 1, 1, 2],
            &[1, 1],
            &[0.0, 0.0],
            &[1.0, 1.0],
            2,
            3,
            &mut bent_s,
            &mut bent_e,
        );

        assert_eq!(straight_s, bent_s);
        assert_eq!(straight_e, bent_e);
    }

    /// Three segments radiating from one vertex is a hub, not a joint.
    #[test]
    fn test_joint_codes_hub() {
        let mut hub_s = vec![9.0f32; 3];
        let mut hub_e = vec![9.0f32; 3];
        compute_joint_codes(
            &[0, 1, 0, 2, 0, 3],
            &[1, 1, 1],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            3,
            4,
            &mut hub_s,
            &mut hub_e,
        );
        assert_eq!(hub_s, vec![JOINT_HUB, JOINT_HUB, JOINT_HUB]);
        assert_eq!(hub_e, vec![JOINT_FREE_END, JOINT_FREE_END, JOINT_FREE_END]);
    }

    /// A visible neighbour trimmed away from the shared vertex is not a joint —
    /// mitering against it would build an edge the neighbour never draws.
    #[test]
    fn test_joint_codes_trimmed_neighbour() {
        let mut out_start = vec![9.0f32; 2];
        let mut out_end = vec![9.0f32; 2];
        compute_joint_codes(
            &[0, 1, 1, 2],
            &[1, 1],
            &[0.0, 0.4],
            &[1.0, 1.0],
            2,
            3,
            &mut out_start,
            &mut out_end,
        );

        assert_eq!(out_end[0], JOINT_FREE_END); // partner does not reach v1
        assert_eq!(out_start[1], JOINT_CLIPPED); // and it knows it was trimmed
    }

    /// An invisible neighbour does not anchor a joint, and the visible stream's
    /// slots stay contiguous so a partner slot always indexes a WRITTEN texel.
    #[test]
    fn test_joint_codes_non_contiguous_visibility() {
        // v0-v1, v1-v2 (culled), v2-v3 — only segments 0 and 2 are visible, so
        // they occupy slots 0 and 1.
        let mut out_start = vec![9.0f32; 2];
        let mut out_end = vec![9.0f32; 2];
        let count = compute_joint_codes(
            &[0, 1, 1, 2, 2, 3],
            &[1, 0, 1],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            3,
            4,
            &mut out_start,
            &mut out_end,
        );

        assert_eq!(count, 2);
        // Nothing shares v1 or v2 among the VISIBLE segments.
        assert_eq!(out_end[0], JOINT_FREE_END);
        assert_eq!(out_start[1], JOINT_FREE_END);
    }

    /// Opposing orientation: two segments meeting END-to-END at a shared vertex.
    /// The sign must report the partner's END, not its start.
    #[test]
    fn test_joint_codes_opposing_orientation() {
        // seg0: v0 -> v1, seg1: v2 -> v1. Both END on v1.
        let mut out_start = vec![9.0f32; 2];
        let mut out_end = vec![9.0f32; 2];
        compute_joint_codes(
            &[0, 1, 2, 1],
            &[1, 1],
            &[0.0, 0.0],
            &[1.0, 1.0],
            2,
            3,
            &mut out_start,
            &mut out_end,
        );

        // seg0's END joins seg1 at seg1's END -> -(1 + 3) = -4.
        assert_eq!(out_end[0], -4.0);
        // seg1's END joins seg0 at seg0's END -> -(0 + 3) = -3.
        assert_eq!(out_end[1], -3.0);
    }

    /// A degenerate segment that registers BOTH its own endpoints on one vertex
    /// must not name itself as its own partner.
    #[test]
    fn test_joint_codes_self_loop() {
        let mut out_start = vec![9.0f32; 1];
        let mut out_end = vec![9.0f32; 1];
        compute_joint_codes(
            &[1, 1],
            &[1],
            &[0.0],
            &[1.0],
            1,
            3,
            &mut out_start,
            &mut out_end,
        );
        assert_eq!(out_start[0], JOINT_FREE_END);
        assert_eq!(out_end[0], JOINT_FREE_END);
    }

    /// A NaN clip parameter makes the registering pass and the reading pass
    /// disagree: `t <= 0` is false (no registration) but `!(t > 0)` is true (it
    /// still reads), so the code-sum difference does not contain this endpoint's
    /// own code and decodes to an arbitrary slot. `slot < 0` alone is not enough
    /// — the difference is NEGATIVE only when the unregistered endpoint's own
    /// code is the larger one. Here the NaN endpoint is slot 0 while the two
    /// endpoints actually registered on its vertex are slots 1 and 2, so the
    /// difference is large and POSITIVE and decodes to slot 3 in a 3-segment
    /// scene. The bound against `visible_count` is what rejects it; the
    /// predecessor kernel bounded the same arithmetic against its direction
    /// table's length.
    #[test]
    fn test_joint_codes_nan_clip_param_cannot_name_an_out_of_range_slot() {
        // v5 is touched by the STARTS of slots 1 and 2 (degree 2, code_sum =
        // 2 + 4 = 6). Slot 0 also starts on v5 but carries a NaN t1, so it reads
        // v5 without having registered: 6 - 0 = 6 -> slot 3, out of range.
        let mut out_start = vec![9.0f32; 3];
        let mut out_end = vec![9.0f32; 3];
        compute_joint_codes(
            &[5, 8, 5, 6, 5, 7],
            &[1, 1, 1],
            &[f32::NAN, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            3,
            9,
            &mut out_start,
            &mut out_end,
        );
        for (i, v) in out_start.iter().chain(out_end.iter()).enumerate() {
            let slot = if *v > 0.5 {
                *v as i32 - 1
            } else if *v < -2.5 {
                (-*v) as i32 - 3
            } else {
                continue; // a sentinel names no slot
            };
            assert!(
                (0..3).contains(&slot),
                "output {} named out-of-range slot {} (code {})",
                i,
                slot,
                v
            );
        }
    }

    /// Out-of-range vertex indices fall back to the free end rather than
    /// indexing the touch tables out of bounds.
    #[test]
    fn test_joint_codes_out_of_range_vertex() {
        let mut out_start = vec![9.0f32; 1];
        let mut out_end = vec![9.0f32; 1];
        compute_joint_codes(
            &[7, 9],
            &[1],
            &[0.0],
            &[1.0],
            1,
            2, // num_vertices = 2, so both indices are out of range
            &mut out_start,
            &mut out_end,
        );
        assert_eq!(out_start[0], JOINT_FREE_END);
        assert_eq!(out_end[0], JOINT_FREE_END);
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
