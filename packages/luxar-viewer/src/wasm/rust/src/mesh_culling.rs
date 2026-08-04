//! Mesh Culling for nD → 3D Slicing
//!
//! This module provides WASM-accelerated nD visibility culling for `mesh`
//! (indexed triangle surface) nodes.
//!
//! ## Whole-triangle cull, not clipping
//!
//! Lines *clip* a segment against the nD slab and interpolate every attribute at
//! the clip parameter (`lines_clipping.rs`). The exact triangle equivalent is nD
//! polygon clipping: a triangle cut by the slab becomes a convex polygon needing
//! fan re-triangulation and per-new-vertex attribute interpolation, every frame
//! the slice moves.
//!
//! Mesh v1 deliberately does not do that. Instead:
//!
//! > A triangle is rendered **iff all three of its vertices pass the nD slab
//! > membership test.**
//!
//! The consequence is a ragged, triangle-quantized cut boundary rather than a
//! clean planar section. That is a documented v1 trade — see
//! `docs/specs/MESH_NODE_SPEC.md` §5.3. Exact nD triangle clipping is explicitly
//! out of scope (§9).
//!
//! ## The two kernels
//!
//! The work splits along the boundary between per-vertex and per-face state, so
//! a slice move recomputes only what changed:
//!
//! 1. [`mesh_vertex_visibility_mask`] — per-vertex slab membership → `u8` mask.
//! 2. [`compact_visible_faces`] — keep faces whose three vertices are all in.
//!
//! ## No vertex compaction
//!
//! [`compact_visible_faces`] writes **original, un-remapped** vertex indices.
//! On a slice change only the *index buffer* is rebuilt; the vertex attribute
//! buffers are uploaded once, in full, and left alone. `drawElements` never
//! fetches an unreferenced vertex, so culled vertices cost nothing to draw, and
//! the mesh is resident in full anyway (§7). This avoids a `vertex_remap` array,
//! the index remapping that goes with it, and re-uploading every attribute
//! buffer on each slice change. The only cost is VRAM for currently-invisible
//! vertices — bounded by the mesh size, which is already the resident working
//! set.
//!
//! It also avoids `projection::compact_by_mask`, which is `&[f32]`-only and so
//! could not compact the native `uint8`/`uint16` vertex colors the format
//! permits without a widening pass.
//!
//! ## Buffer-shape contract
//!
//! `ndim`, `num_vertices` and `num_faces` are trusted, and are `debug_assert`ed
//! rather than checked — matching every sibling kernel (`compact_by_mask`,
//! `count_visible`). The loader is required to reconcile the declared shapes with
//! the materialized array lengths before calling (`MESH_NODE_SPEC.md` §3.5
//! Stage 2), so a mismatch is a caller bug, not untrusted input.
//!
//! Be aware that the two backends fail *differently* when that contract is
//! broken, which matters when reading a bug report: Rust bounds-checks slice
//! indexing even in release, so an oversized count **traps** — and because the
//! crate is `panic = "abort"` the trap takes down the whole WASM module, not just
//! this node (observed as `RuntimeError: unreachable`). The TypeScript reference
//! instead reads `undefined`, fails the finite test, and silently culls. Neither
//! is corruption, but only one is loud.
//!
//! ## Parity
//!
//! Kept in 1:1 parity with `wasm/typescript/mesh-culling.ts`. That reference is
//! not merely a WASM-missing fallback — it is the production backend for
//! `ndim > 16`, because [`mesh_vertex_visibility_mask`] calls `validate_ndim`
//! and the crate is built `panic = "abort"`. See the dimension-limit note in
//! `lib.rs`.

use wasm_bindgen::prelude::*;

use crate::common::{validate_ndim, MAX_SUPPORTED_DIMS};

/// Alias for readability within this module
const MAX_DIMS: usize = MAX_SUPPORTED_DIMS;

/// Compute per-vertex nD slab membership.
///
/// For each non-displayed ("hidden") dimension `d`, with
/// `slice_min = slice_position[d] - tolerance[d]` and
/// `slice_max = slice_position[d] + tolerance[d]`, a vertex is **in** iff
/// `v[d] >= slice_min && v[d] <= slice_max` for *every* such `d`. Displayed
/// dimensions are skipped entirely (they are what gets rendered).
///
/// This is precisely the `p1_in` branch of
/// [`crate::lines_clipping::clip_segment_single`], applied per vertex.
///
/// # Non-finite coordinates
///
/// A `NaN` or `±Inf` coordinate on any hidden dimension makes the vertex
/// **invisible** — the #806 rule, enforced identically in both lines backends
/// and now both mesh backends.
///
/// The explicit `is_finite` test is load-bearing, not decorative. `NaN` fails
/// both comparisons and `±Inf` fails against any finite bound, so the bare
/// range test looks sufficient — but a caller passing a genuinely infinite
/// `tolerance` (rather than the finite `EXTEND_TO_ALL_TOLERANCE` sentinel,
/// `1e10`) makes `slice_max = +Inf`, and `+Inf <= +Inf` is **true**. Without
/// this test an infinite coordinate would then be reported visible.
///
/// The rule covers the *coordinate* only, and the slab parameters behave the
/// OPPOSITE way — worth knowing before reading a surprising screen:
///
/// - a `NaN` in `slice_position` or `tolerance` makes every **finite** vertex
///   visible, because `value < NaN` and `value > NaN` are both false, so the
///   slab test degenerates to "not non-finite". It fails OPEN, not closed.
/// - a **negative** `tolerance` inverts the slab (`min > max`) and culls
///   everything.
///
/// Both are caller bugs — `slice_position` and `tolerance` are viewer-computed,
/// not store-supplied — and both backends agree exactly (verified across the
/// full non-finite matrix), so neither is guarded here. Callers deriving a
/// tolerance from possibly-absent dimension metadata should not let a `NaN`
/// reach this kernel expecting it to be culled.
///
/// # Arguments
/// * `positions` - Vertex positions [num_vertices * ndim]
/// * `slice_position` - Current slice position [ndim]
/// * `tolerance` - Per-dimension tolerance [ndim]
/// * `display_dims` - Which dimensions are displayed [num_display_dims]
/// * `ndim` - Number of dimensions
/// * `num_vertices` - Number of vertices
/// * `output` - Output visibility mask [num_vertices] (1 = in, 0 = out)
///
/// # Returns
/// Number of visible vertices.
///
/// # Panics
/// Via `validate_ndim` when `ndim > 16`. Callers must route above that to the
/// TypeScript backend (`pickBackend` does this transparently).
#[wasm_bindgen]
pub fn mesh_vertex_visibility_mask(
    positions: &[f32],
    slice_position: &[f32],
    tolerance: &[f32],
    display_dims: &[u32],
    ndim: usize,
    num_vertices: usize,
    output: &mut [u8],
) -> u32 {
    validate_ndim(ndim, "mesh_vertex_visibility_mask");

    debug_assert!(
        output.len() >= num_vertices,
        "output too small: {} < {}",
        output.len(),
        num_vertices
    );
    debug_assert!(
        positions.len() >= num_vertices * ndim,
        "positions too small: {} < {}",
        positions.len(),
        num_vertices * ndim
    );

    // OPTIMIZATION: fixed-size array instead of a HashSet (zero allocation),
    // exactly as `lines_clipping` does.
    let mut is_display_dim = [false; MAX_DIMS];
    for &d in display_dims {
        if (d as usize) < MAX_DIMS {
            is_display_dim[d as usize] = true;
        }
    }

    // OPTIMIZATION: hoist the slab bounds out of the per-vertex loop. They are
    // vertex-independent, and only hidden dimensions contribute — so the inner
    // loop runs over `num_hidden` (usually 1: a timepoint or channel) instead of
    // over `ndim` with a display-dim branch on every step.
    //
    // This is load-bearing, not cargo-cult: measured 2.6x against the same
    // kernel with the bounds recomputed inside the vertex loop (1M vertices, 8D,
    // 5 hidden dims). Don't fold it back in for brevity.
    let mut hidden_dims = [0usize; MAX_DIMS];
    let mut slab_min = [0f32; MAX_DIMS];
    let mut slab_max = [0f32; MAX_DIMS];
    let mut num_hidden = 0usize;
    for dim in 0..ndim {
        if is_display_dim[dim] {
            continue;
        }
        hidden_dims[num_hidden] = dim;
        slab_min[num_hidden] = slice_position[dim] - tolerance[dim];
        slab_max[num_hidden] = slice_position[dim] + tolerance[dim];
        num_hidden += 1;
    }

    // Fast path (§5.5): with no hidden dimensions — the common plain-3D case —
    // every vertex trivially passes. Callers are expected to skip this kernel
    // entirely in that case; the branch is here so the kernel is still correct
    // (and cheap) if they don't.
    if num_hidden == 0 {
        for i in 0..num_vertices {
            output[i] = 1;
        }
        return num_vertices as u32;
    }

    let mut visible = 0u32;

    for v in 0..num_vertices {
        let base = v * ndim;
        let mut is_in = true;

        for h in 0..num_hidden {
            let value = positions[base + hidden_dims[h]];
            // See "Non-finite coordinates" above for why `is_finite` is not
            // subsumed by the range test.
            if !value.is_finite() || value < slab_min[h] || value > slab_max[h] {
                is_in = false;
                break;
            }
        }

        if is_in {
            output[v] = 1;
            visible += 1;
        } else {
            output[v] = 0;
        }
    }

    visible
}

/// Compact `faces` to those whose three vertices are all visible.
///
/// Writes **original (un-remapped)** vertex indices into `output` — see the
/// "No vertex compaction" note in the module docs.
///
/// # Winding
///
/// The authored per-face index order is preserved exactly, so this kernel is
/// winding-agnostic. Restoring front-facing winding under a reflected display
/// permutation is a separate post-pass owned by the caller (§5.4).
///
/// # Out-of-range indices
///
/// A face index `>= vertex_mask.len()` causes the whole face to be **dropped**
/// rather than indexing out of bounds.
///
/// This guard exists because the values come from the *store*, not from the
/// caller: the viewer loads arbitrary (externally produced, possibly corrupted)
/// datasets, and an out-of-bounds slice read here would not be a local error —
/// the crate is `panic = "abort"`, so the trap takes down the entire WASM
/// module, losing every other node's kernels with it. The TypeScript backend
/// would instead read `undefined` and silently diverge. Dropping the face keeps
/// both backends in parity and fails safe.
///
/// The loader validates face indices up front and fails the node with a
/// `LoaderError` before reaching here (§3.5 Stage 2), so on the sanctioned path
/// this guard is unreachable. It is defense in depth for a `panic = "abort"`
/// blast radius, not a substitute for that gate.
///
/// Note the asymmetry with the `debug_assert`s above, which is deliberate:
/// store-supplied *values* are range-checked in release builds, while *shape*
/// parameters (`num_faces`, buffer sizes) the caller computed are trusted, as
/// they are in every sibling kernel (`compact_by_mask`, `count_visible`).
///
/// # Arguments
/// * `faces` - Triangle vertex indices [num_faces * 3]
/// * `vertex_mask` - Per-vertex visibility from
///   [`mesh_vertex_visibility_mask`]; its **length defines the valid vertex
///   range**, so pass a view sized exactly `num_vertices` rather than a larger
///   reused scratch buffer
/// * `num_faces` - Number of triangles
/// * `output` - Output indices [num_faces * 3] worst case
///
/// # Returns
///
/// Number of visible faces written. **Slice `output` to `3 ×` this before use.**
/// Everything past that point is left untouched — deliberately, to avoid a second
/// pass over the buffer — so on a reused buffer it holds the *previous* frame's
/// indices, and on a fresh one it holds zeros. Uploading the whole buffer as an
/// index range therefore draws stale or degenerate triangles rather than
/// nothing, which is the failure this return value exists to prevent.
#[wasm_bindgen]
pub fn compact_visible_faces(
    faces: &[u32],
    vertex_mask: &[u8],
    num_faces: usize,
    output: &mut [u32],
) -> u32 {
    debug_assert!(
        faces.len() >= num_faces * 3,
        "faces too small: {} < {}",
        faces.len(),
        num_faces * 3
    );
    debug_assert!(
        output.len() >= num_faces * 3,
        "output too small: {} < {}",
        output.len(),
        num_faces * 3
    );

    let num_vertices = vertex_mask.len();
    let mut out_faces = 0usize;

    for f in 0..num_faces {
        let base = f * 3;
        let i0 = faces[base] as usize;
        let i1 = faces[base + 1] as usize;
        let i2 = faces[base + 2] as usize;

        // Range-check before indexing the mask — see "Out-of-range indices".
        if i0 >= num_vertices || i1 >= num_vertices || i2 >= num_vertices {
            continue;
        }

        if vertex_mask[i0] != 0 && vertex_mask[i1] != 0 && vertex_mask[i2] != 0 {
            let dst = out_faces * 3;
            // Authored order preserved (winding-agnostic).
            output[dst] = faces[base];
            output[dst + 1] = faces[base + 1];
            output[dst + 2] = faces[base + 2];
            out_faces += 1;
        }
    }

    out_faces as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4D positions, dim 3 hidden. Helper keeps the tests readable.
    fn mask_4d(positions: &[f32], slice_w: f32, tol_w: f32, num_vertices: usize) -> (Vec<u8>, u32) {
        let slice_pos = vec![0.0, 0.0, 0.0, slice_w];
        let tolerance = vec![1e10, 1e10, 1e10, tol_w];
        let display_dims = vec![0u32, 1, 2];
        let mut output = vec![0u8; num_vertices];
        let visible = mesh_vertex_visibility_mask(
            positions,
            &slice_pos,
            &tolerance,
            &display_dims,
            4,
            num_vertices,
            &mut output,
        );
        (output, visible)
    }

    // ========================================================================
    // mesh_vertex_visibility_mask
    // ========================================================================

    /// §5.5: `display_dims.len() == ndim` means no hidden dims, so every vertex
    /// is in regardless of slice position or tolerance.
    #[test]
    fn test_no_hidden_dims_all_visible() {
        let positions = vec![
            0.0, 0.0, 0.0, // v0
            1.0, 2.0, 3.0, // v1
            -5.0, 9.0, 100.0, // v2
        ];
        let slice_pos = vec![50.0, 50.0, 50.0];
        let tolerance = vec![0.0, 0.0, 0.0];
        let display_dims = vec![0u32, 1, 2];
        let mut output = vec![0u8; 3];

        let visible = mesh_vertex_visibility_mask(
            &positions,
            &slice_pos,
            &tolerance,
            &display_dims,
            3,
            3,
            &mut output,
        );

        assert_eq!(visible, 3);
        assert_eq!(output, vec![1, 1, 1]);
    }

    /// A vertex inside the slab is in; one outside is out.
    #[test]
    fn test_single_hidden_dim_in_and_out() {
        let positions = vec![
            1.0, 2.0, 3.0, 5.0, // v0: w == slice
            1.0, 2.0, 3.0, 9.0, // v1: w far away
        ];
        let (mask, visible) = mask_4d(&positions, 5.0, 0.5, 2);

        assert_eq!(visible, 1);
        assert_eq!(mask, vec![1, 0]);
    }

    /// The slab is CLOSED: exactly `slice ± tolerance` is inside. Half-cell
    /// discrete tolerances put on-grid geometry exactly on this boundary, so an
    /// exclusive comparison here would drop whole timepoints.
    #[test]
    fn test_slab_bounds_are_inclusive() {
        let positions = vec![
            1.0, 2.0, 3.0, 4.5, // v0: exactly slice_min
            1.0, 2.0, 3.0, 5.5, // v1: exactly slice_max
        ];
        let (mask, visible) = mask_4d(&positions, 5.0, 0.5, 2);

        assert_eq!(visible, 2);
        assert_eq!(mask, vec![1, 1]);
    }

    /// One ulp outside either bound is out — the boundary test is not sloppy.
    ///
    /// `next_down`/`next_up`, not `± f32::EPSILON`: EPSILON is the ulp at 1.0
    /// (1.19e-7), while the ulp at 4.5 is ~4.8e-7, so `4.5 - EPSILON` rounds
    /// straight back to 4.5 and the test would assert nothing.
    #[test]
    fn test_just_outside_slab_is_out() {
        let below = 4.5f32.next_down();
        let above = 5.5f32.next_up();
        assert_ne!(below, 4.5, "next_down must actually move");
        assert_ne!(above, 5.5, "next_up must actually move");
        let positions = vec![1.0, 2.0, 3.0, below, 1.0, 2.0, 3.0, above];
        let (mask, visible) = mask_4d(&positions, 5.0, 0.5, 2);

        assert_eq!(visible, 0);
        assert_eq!(mask, vec![0, 0]);
    }

    /// Zero tolerance degenerates to exact float equality. This is WHY mesh
    /// cannot reuse the Lines spatial tolerance of 0 (§5.2.1): with no
    /// interpolation, a zero-thickness slab renders essentially nothing.
    #[test]
    fn test_zero_tolerance_is_exact_equality() {
        let positions = vec![
            1.0, 2.0, 3.0, 5.0, // v0: exactly on the slice
            1.0, 2.0, 3.0, 5.000001, // v1: a hair off
        ];
        let (mask, visible) = mask_4d(&positions, 5.0, 0.0, 2);

        assert_eq!(visible, 1);
        assert_eq!(mask, vec![1, 0]);
    }

    /// A `NaN` slab PARAMETER fails OPEN — the opposite of a NaN coordinate.
    /// Both `value < NaN` and `value > NaN` are false, so the test degenerates to
    /// "not non-finite". Pinned so nobody adds a one-sided guard in one backend
    /// and silently breaks parity; the TS reference asserts the same.
    #[test]
    fn test_nan_slab_parameters_fail_open() {
        // A vertex far outside any sane slab, so only fail-open admits it.
        let positions = vec![1.0, 2.0, 3.0, 999.0];

        let (mask, visible) = mask_4d(&positions, f32::NAN, 0.5, 1);
        assert_eq!(visible, 1, "NaN slice position should fail OPEN");
        assert_eq!(mask, vec![1]);

        let (mask, visible) = mask_4d(&positions, 5.0, f32::NAN, 1);
        assert_eq!(visible, 1, "NaN tolerance should fail OPEN");
        assert_eq!(mask, vec![1]);
    }

    /// A NEGATIVE tolerance inverts the slab (`min > max`) and culls everything,
    /// including a vertex exactly on the slice.
    #[test]
    fn test_negative_tolerance_culls_everything() {
        let positions = vec![1.0, 2.0, 3.0, 5.0]; // exactly on the slice
        let (mask, visible) = mask_4d(&positions, 5.0, -1.0, 1);

        assert_eq!(visible, 0);
        assert_eq!(mask, vec![0]);
    }

    /// #806: NaN on a hidden dim cannot be localized against the slice → out.
    #[test]
    fn test_nan_on_hidden_dim_is_invisible() {
        let positions = vec![1.0, 2.0, 3.0, f32::NAN];
        let (mask, visible) = mask_4d(&positions, 5.0, 1e10, 1);

        assert_eq!(visible, 0);
        assert_eq!(mask, vec![0]);
    }

    /// #806: ±Inf on a hidden dim is likewise non-finite → out.
    #[test]
    fn test_inf_on_hidden_dim_is_invisible() {
        for w in [f32::INFINITY, f32::NEG_INFINITY] {
            let positions = vec![1.0, 2.0, 3.0, w];
            let (mask, visible) = mask_4d(&positions, 5.0, 1e10, 1);
            assert_eq!(visible, 0, "w={w} should be invisible");
            assert_eq!(mask, vec![0]);
        }
    }

    /// The case the bare range test would get WRONG: an infinite tolerance
    /// makes `slab_max = +Inf`, and `+Inf <= +Inf` is true. Only the explicit
    /// `is_finite` test keeps an infinite coordinate invisible.
    #[test]
    fn test_infinite_tolerance_still_rejects_infinite_coordinate() {
        let positions = vec![1.0, 2.0, 3.0, f32::INFINITY];
        let (mask, visible) = mask_4d(&positions, 5.0, f32::INFINITY, 1);

        assert_eq!(visible, 0);
        assert_eq!(mask, vec![0]);
    }

    /// The non-finite rule is scoped to HIDDEN dims. A NaN on a displayed
    /// dimension is a rendering problem, not a slicing one, and must not be
    /// silently culled here (the sibling loaders don't finite-scan positions
    /// either — §3.5).
    #[test]
    fn test_non_finite_on_displayed_dim_stays_visible() {
        let positions = vec![f32::NAN, 2.0, f32::INFINITY, 5.0];
        let (mask, visible) = mask_4d(&positions, 5.0, 0.5, 1);

        assert_eq!(visible, 1);
        assert_eq!(mask, vec![1]);
    }

    /// Membership is AND-ed across every hidden dimension.
    #[test]
    fn test_multiple_hidden_dims_are_anded() {
        // 5D, display [0,1,2], hidden dims 3 and 4.
        let positions = vec![
            0.0, 0.0, 0.0, 5.0, 7.0, // v0: both in
            0.0, 0.0, 0.0, 5.0, 9.0, // v1: dim 4 out
            0.0, 0.0, 0.0, 1.0, 7.0, // v2: dim 3 out
            0.0, 0.0, 0.0, 1.0, 9.0, // v3: both out
        ];
        let slice_pos = vec![0.0, 0.0, 0.0, 5.0, 7.0];
        let tolerance = vec![1e10, 1e10, 1e10, 0.5, 0.5];
        let display_dims = vec![0u32, 1, 2];
        let mut output = vec![0u8; 4];

        let visible = mesh_vertex_visibility_mask(
            &positions,
            &slice_pos,
            &tolerance,
            &display_dims,
            5,
            4,
            &mut output,
        );

        assert_eq!(visible, 1);
        assert_eq!(output, vec![1, 0, 0, 0]);
    }

    /// `display_dims` is a SET here: a permuted (non-ascending) order selects
    /// the same hidden dimensions. Which display dim maps to renderer X/Y/Z is
    /// the projection's business, not the cull's.
    #[test]
    fn test_display_dims_order_does_not_affect_membership() {
        let positions = vec![1.0, 2.0, 3.0, 9.0];
        let slice_pos = vec![0.0, 0.0, 0.0, 5.0];
        let tolerance = vec![1e10, 1e10, 1e10, 0.5];
        let mut ascending = vec![0u8; 1];
        let mut permuted = vec![0u8; 1];

        mesh_vertex_visibility_mask(
            &positions,
            &slice_pos,
            &tolerance,
            &[0, 1, 2],
            4,
            1,
            &mut ascending,
        );
        mesh_vertex_visibility_mask(
            &positions,
            &slice_pos,
            &tolerance,
            &[2, 0, 1],
            4,
            1,
            &mut permuted,
        );

        assert_eq!(ascending, permuted);
        assert_eq!(ascending, vec![0]);
    }

    /// A hidden dim can be any index, not just the trailing one.
    #[test]
    fn test_hidden_dim_can_be_leading() {
        // 4D, display [1,2,3] → dim 0 is hidden.
        let positions = vec![
            5.0, 1.0, 2.0, 3.0, // v0: in
            9.0, 1.0, 2.0, 3.0, // v1: out
        ];
        let slice_pos = vec![5.0, 0.0, 0.0, 0.0];
        let tolerance = vec![0.5, 1e10, 1e10, 1e10];
        let mut output = vec![0u8; 2];

        let visible = mesh_vertex_visibility_mask(
            &positions,
            &slice_pos,
            &tolerance,
            &[1, 2, 3],
            4,
            2,
            &mut output,
        );

        assert_eq!(visible, 1);
        assert_eq!(output, vec![1, 0]);
    }

    /// Fewer than 3 DISPLAY dims must work. The dimension hazard is two-sided
    /// (see the WASM 16-dimension note in `CLAUDE.md`): >16D panics, and a
    /// hardcoded sub-ndim of 3 has crashed a sibling kernel on 2D data before
    /// (#881). This kernel is structurally immune — it reads only hidden
    /// dimensions and never builds a display marginal — so the test exists to
    /// keep it that way.
    #[test]
    fn test_fewer_than_three_display_dims() {
        // 2D data, both dims shown → no hidden dims → all visible.
        let mut output = vec![0u8; 2];
        let visible = mesh_vertex_visibility_mask(
            &[0.0, 0.0, 5.0, 5.0],
            &[5.0, 5.0],
            &[0.5, 0.5],
            &[0, 1],
            2,
            2,
            &mut output,
        );
        assert_eq!(visible, 2);

        // 3D data, 2 dims shown → dim 2 hidden and discriminating.
        let mut output = vec![0u8; 2];
        let visible = mesh_vertex_visibility_mask(
            &[0.0, 0.0, 5.0, 0.0, 0.0, 9.0],
            &[0.0, 0.0, 5.0],
            &[1e10, 1e10, 0.5],
            &[0, 1],
            3,
            2,
            &mut output,
        );
        assert_eq!(visible, 1);
        assert_eq!(output, vec![1, 0]);

        // ZERO dims shown → every dimension hidden.
        let mut output = vec![0u8; 2];
        let visible = mesh_vertex_visibility_mask(
            &[5.0, 5.0, 9.0, 9.0],
            &[5.0, 5.0],
            &[0.5, 0.5],
            &[],
            2,
            2,
            &mut output,
        );
        assert_eq!(visible, 1);
        assert_eq!(output, vec![1, 0]);
    }

    /// `EXTEND_TO_ALL_TOLERANCE` (1e10) admits the whole finite axis.
    #[test]
    fn test_extend_to_all_tolerance_admits_everything_finite() {
        let positions = vec![
            1.0, 2.0, 3.0, -1e6, //
            1.0, 2.0, 3.0, 1e6, //
        ];
        let (mask, visible) = mask_4d(&positions, 0.0, 1e10, 2);

        assert_eq!(visible, 2);
        assert_eq!(mask, vec![1, 1]);
    }

    /// The mask is fully written, not OR-ed into: stale 1s from a reused
    /// buffer must be cleared.
    #[test]
    fn test_output_mask_is_overwritten_not_ored() {
        let positions = vec![1.0, 2.0, 3.0, 9.0];
        let slice_pos = vec![0.0, 0.0, 0.0, 5.0];
        let tolerance = vec![1e10, 1e10, 1e10, 0.5];
        let mut output = vec![1u8; 1]; // pre-seeded "visible"

        let visible = mesh_vertex_visibility_mask(
            &positions,
            &slice_pos,
            &tolerance,
            &[0, 1, 2],
            4,
            1,
            &mut output,
        );

        assert_eq!(visible, 0);
        assert_eq!(output, vec![0]);
    }

    #[test]
    fn test_zero_vertices() {
        let mut output: Vec<u8> = vec![];
        let visible = mesh_vertex_visibility_mask(
            &[],
            &[0.0, 0.0, 0.0, 5.0],
            &[1e10, 1e10, 1e10, 0.5],
            &[0, 1, 2],
            4,
            0,
            &mut output,
        );
        assert_eq!(visible, 0);
    }

    /// 16D is the documented WASM fast-path ceiling and must work.
    #[test]
    fn test_max_supported_dims_works() {
        let ndim = MAX_SUPPORTED_DIMS;
        let mut positions = vec![0.0f32; ndim];
        positions[15] = 5.0;
        let mut slice_pos = vec![0.0f32; ndim];
        slice_pos[15] = 5.0;
        let tolerance = vec![0.5f32; ndim];
        let mut output = vec![0u8; 1];

        let visible = mesh_vertex_visibility_mask(
            &positions,
            &slice_pos,
            &tolerance,
            &[0, 1, 2],
            ndim,
            1,
            &mut output,
        );

        assert_eq!(visible, 1);
    }

    /// Above 16D the kernel panics; `pickBackend` routes there to the TS
    /// backend instead.
    #[test]
    #[should_panic(expected = "exceeds maximum supported dimensions")]
    fn test_dimension_limit_validation() {
        let ndim = MAX_SUPPORTED_DIMS + 1;
        let positions = vec![0.0f32; ndim];
        let slice_pos = vec![0.0f32; ndim];
        let tolerance = vec![0.5f32; ndim];
        let mut output = vec![0u8; 1];

        mesh_vertex_visibility_mask(
            &positions,
            &slice_pos,
            &tolerance,
            &[0, 1, 2],
            ndim,
            1,
            &mut output,
        );
    }

    // ========================================================================
    // compact_visible_faces
    // ========================================================================

    #[test]
    fn test_compact_all_visible_keeps_every_face() {
        let faces = vec![0u32, 1, 2, 1, 2, 3];
        let mask = vec![1u8, 1, 1, 1];
        let mut output = vec![0u32; 6];

        let kept = compact_visible_faces(&faces, &mask, 2, &mut output);

        assert_eq!(kept, 2);
        assert_eq!(output, faces);
    }

    #[test]
    fn test_compact_drops_face_with_any_invisible_vertex() {
        // Face 0 = (0,1,2) all in; face 1 = (1,2,3) has v3 out.
        let faces = vec![0u32, 1, 2, 1, 2, 3];
        let mask = vec![1u8, 1, 1, 0];
        let mut output = vec![0u32; 6];

        let kept = compact_visible_faces(&faces, &mask, 2, &mut output);

        assert_eq!(kept, 1);
        assert_eq!(&output[..3], &[0, 1, 2]);
    }

    /// Every vertex of a face must be in — check each of the three positions
    /// individually so a partial predicate can't pass.
    #[test]
    fn test_compact_checks_all_three_vertices() {
        let faces = vec![7u32, 8, 9];
        for hidden in 7..=9usize {
            let mut mask = vec![1u8; 10];
            mask[hidden] = 0;
            let mut output = vec![0u32; 3];
            let kept = compact_visible_faces(&faces, &mask, 1, &mut output);
            assert_eq!(kept, 0, "hiding vertex {hidden} should drop the face");
        }
    }

    #[test]
    fn test_compact_none_visible() {
        let faces = vec![0u32, 1, 2];
        let mask = vec![0u8, 0, 0];
        let mut output = vec![0u32; 3];

        let kept = compact_visible_faces(&faces, &mask, 1, &mut output);

        assert_eq!(kept, 0);
    }

    /// Indices are ORIGINAL, not remapped to a compacted vertex array — the
    /// whole point of the no-compaction rule.
    #[test]
    fn test_compact_preserves_original_indices() {
        // Only the high-index face survives; its indices must stay 5,6,7.
        let faces = vec![0u32, 1, 2, 5, 6, 7];
        let mask = vec![0u8, 0, 0, 0, 0, 1, 1, 1];
        let mut output = vec![0u32; 6];

        let kept = compact_visible_faces(&faces, &mask, 2, &mut output);

        assert_eq!(kept, 1);
        assert_eq!(&output[..3], &[5, 6, 7]);
    }

    /// Authored per-face index order (and thus winding) is preserved verbatim.
    #[test]
    fn test_compact_preserves_winding_order() {
        let faces = vec![2u32, 0, 1]; // deliberately not sorted
        let mask = vec![1u8, 1, 1];
        let mut output = vec![0u32; 3];

        let kept = compact_visible_faces(&faces, &mask, 1, &mut output);

        assert_eq!(kept, 1);
        assert_eq!(&output[..3], &[2, 0, 1]);
    }

    /// Faces stay in their original relative order after compaction.
    #[test]
    fn test_compact_is_order_stable() {
        let faces = vec![
            0u32, 1, 2, // keep
            3, 4, 5, // drop
            6, 7, 8, // keep
        ];
        let mask = vec![1u8, 1, 1, 0, 1, 1, 1, 1, 1];
        let mut output = vec![0u32; 9];

        let kept = compact_visible_faces(&faces, &mask, 3, &mut output);

        assert_eq!(kept, 2);
        assert_eq!(&output[..6], &[0, 1, 2, 6, 7, 8]);
    }

    /// An out-of-range face index drops the face instead of trapping. With
    /// `panic = "abort"` an out-of-bounds read would take down the whole WASM
    /// module, not just this node.
    #[test]
    fn test_compact_drops_out_of_range_index_without_panicking() {
        // vertex_mask has 3 entries → valid indices are 0..2.
        //
        // The trailing VALID face is the point of the layout: a bad index must
        // `continue` past that face only. With `break`, one corrupt index early in
        // the array would silently discard every valid face after it — most of the
        // mesh vanishing with no error — and a test whose only valid face came
        // first could not tell the two apart.
        let faces = vec![
            0u32, 1, 2, // valid
            0, 1, 3, // index 3 out of range
            99, 0, 1, // wildly out of range
            2, 1, 0, // valid, AFTER the bad ones
        ];
        let mask = vec![1u8, 1, 1];
        let mut output = vec![0u32; 12];

        let kept = compact_visible_faces(&faces, &mask, 4, &mut output);

        assert_eq!(kept, 2, "the valid face after a bad one must survive");
        assert_eq!(&output[..6], &[0, 1, 2, 2, 1, 0]);
    }

    /// `u32::MAX` is the sentinel most likely to appear from a signed `-1`
    /// reinterpreted during an external store's integer coercion.
    #[test]
    fn test_compact_drops_u32_max_index() {
        let faces = vec![0u32, 1, u32::MAX];
        let mask = vec![1u8, 1, 1];
        let mut output = vec![0u32; 3];

        let kept = compact_visible_faces(&faces, &mask, 1, &mut output);

        assert_eq!(kept, 0);
    }

    /// `vertex_mask.len()` — not a separate count — is the valid-vertex
    /// authority, so an OVERSIZED reused scratch mask admits faces that index
    /// its stale tail. Pinned because the doc comment promises this, and because
    /// a caller reusing one buffer across meshes must slice it to
    /// `num_vertices` (the second half here) rather than pass it whole.
    #[test]
    fn test_compact_treats_mask_len_as_the_vertex_range() {
        // 4 real vertices, but a 16-slot scratch buffer whose tail is stale 1s.
        let mut scratch = vec![1u8; 16];
        for slot in scratch.iter_mut().take(4) {
            *slot = 0;
        }
        scratch[1] = 1; // the only genuinely visible real vertex
        let faces = vec![1u32, 1, 1, 9, 10, 11]; // 2nd face lives in the stale tail
        let mut output = vec![0u32; 6];

        // Passed WHOLE: the stale tail is treated as real, so both faces survive.
        let kept = compact_visible_faces(&faces, &scratch, 2, &mut output);
        assert_eq!(kept, 2, "oversized mask admits its stale tail");

        // Sliced to the real vertex count: the stale face is correctly dropped.
        let kept = compact_visible_faces(&faces, &scratch[..4], 2, &mut output);
        assert_eq!(kept, 1);
        assert_eq!(&output[..3], &[1, 1, 1]);
    }

    /// A degenerate face (repeated index) is not special-cased: it is kept iff
    /// its vertices are visible. Culling is not the place to fix topology.
    #[test]
    fn test_compact_keeps_degenerate_face_when_visible() {
        let faces = vec![1u32, 1, 1];
        let mask = vec![1u8, 1];
        let mut output = vec![0u32; 3];

        let kept = compact_visible_faces(&faces, &mask, 1, &mut output);

        assert_eq!(kept, 1);
        assert_eq!(&output[..3], &[1, 1, 1]);
    }

    #[test]
    fn test_compact_zero_faces() {
        let mut output: Vec<u32> = vec![];
        let kept = compact_visible_faces(&[], &[1u8], 0, &mut output);
        assert_eq!(kept, 0);
    }

    /// A non-1 truthy mask value counts as visible (`!= 0`, matching every
    /// sibling mask consumer such as `compact_by_mask`).
    #[test]
    fn test_compact_treats_any_nonzero_mask_as_visible() {
        let faces = vec![0u32, 1, 2];
        let mask = vec![2u8, 255, 7];
        let mut output = vec![0u32; 3];

        let kept = compact_visible_faces(&faces, &mask, 1, &mut output);

        assert_eq!(kept, 1);
    }

    // ========================================================================
    // The two kernels composed — the shape the loader actually uses
    // ========================================================================

    /// End-to-end: a 4-vertex quad (2 triangles) spanning two timepoints. The
    /// slice picks t=0, so only the triangle whose vertices all sit at t=0
    /// survives — and the surviving indices still address the FULL vertex
    /// array.
    #[test]
    fn test_mask_then_compact_end_to_end() {
        // 4D: x, y, z, t. v0..v2 at t=0, v3 at t=1.
        let positions = vec![
            0.0, 0.0, 0.0, 0.0, // v0 t=0
            1.0, 0.0, 0.0, 0.0, // v1 t=0
            0.0, 1.0, 0.0, 0.0, // v2 t=0
            1.0, 1.0, 0.0, 1.0, // v3 t=1
        ];
        let faces = vec![0u32, 1, 2, 1, 3, 2];
        let slice_pos = vec![0.0, 0.0, 0.0, 0.0];
        // Half-cell membership tolerance for a discrete axis of step 1.
        let tolerance = vec![1e10, 1e10, 1e10, 0.5];
        let mut mask = vec![0u8; 4];

        let visible_vertices = mesh_vertex_visibility_mask(
            &positions,
            &slice_pos,
            &tolerance,
            &[0, 1, 2],
            4,
            4,
            &mut mask,
        );
        assert_eq!(visible_vertices, 3);
        assert_eq!(mask, vec![1, 1, 1, 0]);

        let mut output = vec![0u32; 6];
        let kept = compact_visible_faces(&faces, &mask, 2, &mut output);

        assert_eq!(kept, 1);
        assert_eq!(&output[..3], &[0, 1, 2]);
    }
}
