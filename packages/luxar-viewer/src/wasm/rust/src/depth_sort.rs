//! Depth Sorting for Order-Dependent GSplat Blending
//!
//! Produces a back-to-front draw ordering for 3D splat centers under a given
//! model-view transform (Phase 2 of the depth-sorting plan,
//! `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §5). The ordering is the
//! permutation consumed by the `aSortedIndex` instance attribute: instance
//! `j` renders splat `ordering[j]`, so `ordering[0]` is drawn first
//! (farthest from the camera).
//!
//! ## Algorithm — normalized-key counting sort
//!
//! - **Pass 1**: camera-space `z` per splat (`z = mv·center`, column-major
//!   THREE.js model-view) into an f32 scratch, with a running min/max over
//!   the in-front (`z < 0`) splats.
//! - **Pass 2**: normalize each `z` to a uint16 key
//!   (`floor((z - zmin) / (zmax - zmin) * 65535)`) and build a 65536-bucket
//!   histogram. Camera view space looks down `-z`, so `zmin` is the farthest
//!   splat and key 0 is the **far bucket**; ascending key order is
//!   back-to-front by construction. Behind-camera splats (`z >= 0`) key to
//!   the far bucket — the vertex shader already degenerates them, so their
//!   position in the ordering is cosmetic.
//! - **Pass 3**: prefix-sum the histogram and stably scatter original
//!   indices, preserving input order within equal keys.
//!
//! Faster-looking variants were measured and REJECTED on worst-case
//! (spatially incoherent) 1M-splat data: a 2×8-bit LSD radix sort (the
//! 65536 write cursors already sit in L2 on target hardware, so halving
//! the cursor set doesn't pay for the doubled memory traffic) and
//! branchless multi-lane min/max + interleaved partial histograms (the 4×
//! histogram footprint cost more in cache pressure than the broken
//! dependency chains gained). The simple three-pass form above is the
//! fastest measured; see `perf-budget.test.ts` for the 50 M splats/s floor
//! (report-only unless `LUXAR_PERF_QUIET_HOST=1`).
//!
//! Raw f16-bit-pattern keys (the SparkJS trick) are deliberately NOT used:
//! Luxar scenes carry physical units spanning nm..km, and f16
//! overflow/underflow at those magnitudes collapses the ordering.
//! Per-sort min/max normalization is scale-invariant — the key resolution
//! adapts to whatever depth range the frame actually spans.
//!
//! ## Degenerate inputs
//!
//! `zmax == zmin` (all splats at one depth, a single in-front splat, or
//! every splat behind the camera) yields the **identity ordering** — any
//! permutation is equally correct, and identity keeps the GPU upload a
//! no-op-equivalent.
//!
//! This kernel takes already-projected 3D centers (the projection stage's
//! output), so `validate_ndim` / the 16-dimension cap are not involved.
//!
//! ## Per-shard geometry (cross-node depth ordering)
//!
//! Optionally the kernel also reports, for each of `shard_count` contiguous
//! equal-population ranges of the output ordering, both the local-space AABB of
//! the elements that range draws AND that range's view-space z interval at this
//! sort's pose — see `docs/guides/specs/CROSS_NODE_DEPTH_ORDERING_SPEC.md` §3.3
//! decision 3. Since the permutation is already back-to-front, a contiguous range
//! of it is a depth interval, which is what lets the main thread order ranges of
//! DIFFERENT nodes against each other.
//!
//! The VIEW-Z interval is the merge key; the box is only an extent (containment).
//! `write_shard_bounds` explains at length why a re-projected box CANNOT serve as
//! the key — it is the correction to a design that measurably popped.
//!
//! `shard_count == 0` skips the work entirely and leaves the output slices
//! untouched, so an unsharded node pays nothing.
//!
//! Two deliberate choices:
//!
//! - **AABBs, not centroid+radius.** `min`/`max` over `f32` involve no rounding
//!   whatsoever, so the TypeScript twin reproduces these bytes exactly without
//!   any of the `Math.fround` discipline the depth-key math needs. Centroid and
//!   radius are derived on the main thread, where there is no parity contract.
//! - **A separate pass in shard-major order, not accumulation inside pass 3.**
//!   Walking `ordering` sequentially per shard needs no integer division per
//!   element and — the reason it is worth a pass — is the SAME code for the
//!   identity fallback as for a real sort, so there is one behaviour to keep in
//!   parity instead of two.

use wasm_bindgen::prelude::*;

/// Number of distinct depth keys (full uint16 key range).
const DEPTH_SORT_BUCKETS: usize = 1 << 16;

/// Maximum key value (`DEPTH_SORT_BUCKETS - 1` as f32 for normalization).
const DEPTH_KEY_MAX: f32 = (DEPTH_SORT_BUCKETS - 1) as f32;

/// Per-shard geometry for each contiguous equal-population range of `ordering`:
/// a local-space AABB **and** the range's VIEW-space z interval at this sort's
/// pose.
///
/// `shard_bounds_min` / `shard_bounds_max` are `[shard_count * 3]` (x, y, z per
/// shard); `shard_view_z_min` / `shard_view_z_max` are `[shard_count]`. A shard
/// with no elements — reachable when `count < shard_count` — keeps the empty
/// sentinel (`min = +inf`, `max = -inf`), which the caller must read as "no
/// usable bounds" rather than as a degenerate point.
///
/// ## Why the VIEW-Z interval is the merge key rather than the box
///
/// A shard is a thin slab perpendicular to the view axis AT THIS POSE. Its
/// axis-aligned box is therefore thin along that axis but spans the whole node in
/// the other two, so re-projecting the box from a rotated camera gives something
/// FAT along the new view direction: adjacent shards' projected centres converge
/// and the key degenerates. No local-space geometric summary avoids that, because
/// the shard GROUPING is itself pose-dependent — a centroid degenerates exactly as
/// the box does. A fresh key over a stale grouping is incoherent, so the key is
/// taken from the pose the grouping was built at and held until the next sort
/// rebuilds both. The merge output is then constant between re-sorts.
///
/// **Honest scope**: this makes the key exact and pose-consistent, and it is the
/// right key on those grounds. It did NOT measurably change the popping it was
/// first written to fix — swapping back to the re-projected box gives identical
/// per-frame numbers on real data (neuromast, two volumetric nodes: spikiness
/// 1.07 either way). The dominant cause of that popping is elsewhere: shard
/// COUNT versus how finely the two nodes interleave in depth. Two co-extensive,
/// finely interleaved nodes have shard intervals that overlap almost completely
/// at practical counts, so the imposed order is near-arbitrary and flaps; raising
/// the count until the intervals separate removes it (measured on the adversarial
/// two-comb fixture: median per-frame change 8.18 at 16 shards, 4.97 at 256,
/// against a 4.34 unsharded baseline). Spec §4 rule 3 describes deriving the
/// count from the overlap extent; the policy does not yet implement it.
///
/// `z_view` is pass 1's per-element view-space z, indexed by ELEMENT (not by draw
/// slot), so it is read through `ordering`.
///
/// Non-finite centers are handled ASYMMETRICALLY, on purpose:
///
/// - **NaN is EXCLUDED** from its shard's box. Every comparison against NaN is
///   false, so it never updates a bound — and that is the behaviour we want: one
///   bad element among thousands must not poison a whole shard's box and cost it
///   its place in the depth merge. The NaN element is degenerate in the shader
///   anyway. A shard whose elements are ALL NaN keeps the empty sentinel, which
///   the caller already reads as "no usable bounds".
/// - **±Infinity PROPAGATES**, because those comparisons do succeed, leaving a
///   non-finite bound that the caller rejects exactly as the render-order pass
///   already rejects a non-finite bounding sphere.
///
/// Both are reachable in normal operation: a NaN view-z keys to the NEAR bucket
/// (`f32::min(NaN, 65535.0) == 65535`), so a NaN center lands in the LAST shard.
fn write_shard_bounds(
    centers3: &[f32],
    z_view: &[f32],
    ordering: &[u32],
    count: usize,
    shard_count: usize,
    shard_bounds_min: &mut [f32],
    shard_bounds_max: &mut [f32],
    shard_view_z_min: &mut [f32],
    shard_view_z_max: &mut [f32],
) {
    if shard_count == 0 {
        return;
    }
    for slot in shard_view_z_min.iter_mut().take(shard_count) {
        *slot = f32::INFINITY;
    }
    for slot in shard_view_z_max.iter_mut().take(shard_count) {
        *slot = f32::NEG_INFINITY;
    }
    for slot in shard_bounds_min.iter_mut().take(shard_count * 3) {
        *slot = f32::INFINITY;
    }
    for slot in shard_bounds_max.iter_mut().take(shard_count * 3) {
        *slot = f32::NEG_INFINITY;
    }
    if count == 0 {
        return;
    }

    // Equal-population shards: the same `ceil(count / shard_count)` the main
    // thread uses to size each shard's draw, so box `s` covers exactly the
    // elements shard `s` draws.
    let shard_size = count.div_ceil(shard_count);
    for s in 0..shard_count {
        let lo = s * shard_size;
        if lo >= count {
            break;
        }
        let hi = (lo + shard_size).min(count);
        let (mut min_x, mut min_y, mut min_z) = (f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let (mut max_x, mut max_y, mut max_z) =
            (f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut min_zv = f32::INFINITY;
        let mut max_zv = f32::NEG_INFINITY;
        for &element in &ordering[lo..hi] {
            let base = element as usize * 3;
            let (x, y, z) = (centers3[base], centers3[base + 1], centers3[base + 2]);
            // VIEW-space z at THIS sort's pose — the shard's true depth interval,
            // and the only sound cross-node merge key (see the fn doc).
            let zv = z_view[element as usize];
            if zv < min_zv {
                min_zv = zv;
            }
            if zv > max_zv {
                max_zv = zv;
            }
            // Plain comparisons, which are false for NaN — so a NaN center is
            // EXCLUDED from the box rather than poisoning it (see the fn doc).
            // `f32::min`/`f32::max` would behave the same here; the explicit form
            // is kept so the TypeScript twin can mirror it literally, since
            // `Math.min(NaN, x)` is NaN and would NOT match.
            if x < min_x {
                min_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if z < min_z {
                min_z = z;
            }
            if x > max_x {
                max_x = x;
            }
            if y > max_y {
                max_y = y;
            }
            if z > max_z {
                max_z = z;
            }
        }
        let out = s * 3;
        shard_bounds_min[out] = min_x;
        shard_bounds_min[out + 1] = min_y;
        shard_bounds_min[out + 2] = min_z;
        shard_bounds_max[out] = max_x;
        shard_bounds_max[out + 1] = max_y;
        shard_bounds_max[out + 2] = max_z;
        shard_view_z_min[s] = min_zv;
        shard_view_z_max[s] = max_zv;
    }
}

/// Sort splats back-to-front by camera-space depth.
///
/// # Arguments
/// - `centers3`: Projected 3D splat centers `[count * 3]` (x, y, z triplets)
/// - `model_view`: Column-major 4x4 model-view matrix `[16]`
///   (`camera.matrixWorldInverse × mesh.matrixWorld`, THREE.js layout)
/// - `ordering`: Output permutation `[count]` — `ordering[j]` is the original
///   splat index drawn at instance slot `j` (slot 0 = farthest)
/// - `count`: Number of splats
/// - `shard_count`: Number of contiguous equal-population ranges to report
///   bounds for; `0` skips that work and leaves the two slices untouched
/// - `shard_bounds_min` / `shard_bounds_max`: Output local-space AABBs
///   `[shard_count * 3]` — see {@link write_shard_bounds}
/// - `shard_view_z_min` / `shard_view_z_max`: Output per-shard VIEW-space z
///   interval `[shard_count]` at this sort's pose — the cross-node merge key
///
/// # Returns
/// Number of splats placed via depth keys, or `0` when the identity
/// fallback was taken (degenerate depth range — the ordering is still
/// fully written, and the shard bounds are still valid boxes of it, but they
/// carry no DEPTH meaning, so the caller must not merge them as depth
/// intervals).
#[wasm_bindgen]
pub fn sort_splats_by_depth(
    centers3: &[f32],
    model_view: &[f32],
    ordering: &mut [u32],
    count: usize,
    shard_count: usize,
    shard_bounds_min: &mut [f32],
    shard_bounds_max: &mut [f32],
    shard_view_z_min: &mut [f32],
    shard_view_z_max: &mut [f32],
) -> u32 {
    // Undersized buffers are SAFE in release too: the debug_asserts
    // below only add friendlier messages — Rust's slice bounds checks
    // trap (`unreachable`) before any out-of-bounds access, and a trap
    // does not poison the instance (verified empirically: subsequent
    // calls succeed), so a bad call degrades to one rejected worker RPC.
    debug_assert!(
        centers3.len() >= count * 3,
        "centers3 too small: {} < {}",
        centers3.len(),
        count * 3
    );
    debug_assert!(
        ordering.len() >= count,
        "ordering too small: {} < {}",
        ordering.len(),
        count
    );
    debug_assert!(
        model_view.len() >= 16,
        "model_view too small: {} < 16",
        model_view.len()
    );
    debug_assert!(
        shard_bounds_min.len() >= shard_count * 3,
        "shard_bounds_min too small: {} < {}",
        shard_bounds_min.len(),
        shard_count * 3
    );
    debug_assert!(
        shard_bounds_max.len() >= shard_count * 3,
        "shard_bounds_max too small: {} < {}",
        shard_bounds_max.len(),
        shard_count * 3
    );

    if count == 0 {
        // Still stamp the empty-box sentinels: a caller that reads the slices
        // without checking the return value must not see a stale previous frame.
        write_shard_bounds(
            centers3,
            &[],
            ordering,
            0,
            shard_count,
            shard_bounds_min,
            shard_bounds_max,
            shard_view_z_min,
            shard_view_z_max,
        );
        return 0;
    }

    // Column-major view-z row: z_view = m2·x + m6·y + m10·z + m14.
    let m2 = model_view[2];
    let m6 = model_view[6];
    let m10 = model_view[10];
    let m14 = model_view[14];

    // Pass 1: camera-space z per splat + min/max over in-front splats.
    // View space looks down -z, so in-front splats have z < 0.
    let mut z_scratch = vec![0.0f32; count];
    let mut z_min = f32::INFINITY;
    let mut z_max = f32::NEG_INFINITY;
    for i in 0..count {
        let base = i * 3;
        let z = m2 * centers3[base] + m6 * centers3[base + 1] + m10 * centers3[base + 2] + m14;
        z_scratch[i] = z;
        if z < 0.0 {
            z_min = z_min.min(z);
            z_max = z_max.max(z);
        }
    }

    // Degenerate depth range (single depth plane, <=1 in-front splat, or
    // everything behind the camera): identity ordering. Deliberately
    // `!(a > b)` rather than `a <= b` — it also catches NaN centers
    // (untouched ±INFINITY sentinels compare false either way, but a NaN
    // z_min/z_max must fall through to identity, not into key math).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(z_max > z_min) {
        for (j, slot) in ordering.iter_mut().enumerate().take(count) {
            *slot = j as u32;
        }
        write_shard_bounds(
            centers3,
            &z_scratch,
            ordering,
            count,
            shard_count,
            shard_bounds_min,
            shard_bounds_max,
            shard_view_z_min,
            shard_view_z_max,
        );
        return 0;
    }

    // Pass 2: normalized uint16 keys + histogram. zmin (farthest) -> key 0,
    // zmax (nearest in-front) -> key 65535; behind-camera -> far bucket 0.
    let inv_range = 1.0 / (z_max - z_min);
    let mut keys = vec![0u16; count];
    let mut histogram = vec![0u32; DEPTH_SORT_BUCKETS];
    for i in 0..count {
        let z = z_scratch[i];
        let key = if z >= 0.0 {
            0u16
        } else {
            // Saturating float->int cast also guards rounding past 65535.
            (((z - z_min) * inv_range * DEPTH_KEY_MAX).min(DEPTH_KEY_MAX)) as u16
        };
        keys[i] = key;
        histogram[key as usize] += 1;
    }

    // Prefix sum: bucket k's write cursor starts after all lower (farther)
    // buckets — ascending keys scatter back-to-front.
    let mut cursor: u32 = 0;
    let mut starts = histogram; // reuse the allocation in place
    for slot in starts.iter_mut() {
        let bucket_count = *slot;
        *slot = cursor;
        cursor += bucket_count;
    }

    // Pass 3: stable scatter (equal keys keep their input order).
    for i in 0..count {
        let bucket = keys[i] as usize;
        ordering[starts[bucket] as usize] = i as u32;
        starts[bucket] += 1;
    }

    write_shard_bounds(
        centers3,
        &z_scratch,
        ordering,
        count,
        shard_count,
        shard_bounds_min,
        shard_bounds_max,
        shard_view_z_min,
        shard_view_z_max,
    );

    count as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity model-view: view z == world z (camera at origin looking -z).
    const IDENTITY_MV: [f32; 16] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];

    /// Build centers with the given view-space z values (x = index, y = 0)
    /// so the identity model-view maps world z directly to view z.
    fn centers_with_z(zs: &[f32]) -> Vec<f32> {
        let mut centers = Vec::with_capacity(zs.len() * 3);
        for (i, &z) in zs.iter().enumerate() {
            centers.push(i as f32);
            centers.push(0.0);
            centers.push(z);
        }
        centers
    }

    fn sort(zs: &[f32]) -> (Vec<u32>, u32) {
        let centers = centers_with_z(zs);
        let mut ordering = vec![u32::MAX; zs.len()];
        let sorted = sort_splats_by_depth(
            &centers,
            &IDENTITY_MV,
            &mut ordering,
            zs.len(),
            0,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        );
        (ordering, sorted)
    }

    /// Sort + report per-shard bounds. Returns `(ordering, sorted, min, max)`.
    fn sort_sharded(zs: &[f32], shard_count: usize) -> (Vec<u32>, u32, Vec<f32>, Vec<f32>) {
        let (ordering, sorted, min, max, _, _) = sort_sharded_full(zs, shard_count);
        (ordering, sorted, min, max)
    }

    /// Sort + report per-shard boxes AND view-z intervals.
    #[allow(clippy::type_complexity)]
    fn sort_sharded_full(
        zs: &[f32],
        shard_count: usize,
    ) -> (Vec<u32>, u32, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let centers = centers_with_z(zs);
        let mut ordering = vec![u32::MAX; zs.len()];
        let mut min = vec![0.0f32; shard_count * 3];
        let mut max = vec![0.0f32; shard_count * 3];
        let mut zmin = vec![0.0f32; shard_count];
        let mut zmax = vec![0.0f32; shard_count];
        let sorted = sort_splats_by_depth(
            &centers,
            &IDENTITY_MV,
            &mut ordering,
            zs.len(),
            shard_count,
            &mut min,
            &mut max,
            &mut zmin,
            &mut zmax,
        );
        (ordering, sorted, min, max, zmin, zmax)
    }

    fn assert_is_permutation(ordering: &[u32], count: usize) {
        let mut seen = vec![false; count];
        for &idx in ordering {
            assert!((idx as usize) < count, "index {idx} out of range {count}");
            assert!(!seen[idx as usize], "index {idx} appears twice");
            seen[idx as usize] = true;
        }
    }

    /// The core contract: view z is non-decreasing along the ordering
    /// (farthest = most negative z first), with behind-camera splats
    /// allowed anywhere in the far bucket's prefix.
    fn assert_back_to_front(ordering: &[u32], zs: &[f32]) {
        let in_front: Vec<f32> = ordering
            .iter()
            .map(|&i| zs[i as usize])
            .filter(|&z| z < 0.0)
            .collect();
        for w in in_front.windows(2) {
            // Equal keys may swap sub-bucket z order; allow one-bucket slack.
            let bucket = |z: f32| {
                let zmin = zs
                    .iter()
                    .cloned()
                    .filter(|&z| z < 0.0)
                    .fold(f32::INFINITY, f32::min);
                let zmax = zs
                    .iter()
                    .cloned()
                    .filter(|&z| z < 0.0)
                    .fold(f32::NEG_INFINITY, f32::max);
                (((z - zmin) / (zmax - zmin)) * DEPTH_KEY_MAX) as i64
            };
            assert!(
                bucket(w[0]) <= bucket(w[1]),
                "ordering not back-to-front: z {} before z {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_back_to_front_basic() {
        // Farthest (most negative z) must come first.
        let zs = [-1.0, -10.0, -5.0, -2.0];
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 4);
        assert_is_permutation(&ordering, 4);
        assert_eq!(ordering, vec![1, 2, 3, 0]); // z: -10, -5, -2, -1
    }

    #[test]
    fn test_already_sorted_input() {
        let zs = [-10.0, -5.0, -2.0, -1.0];
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 4);
        assert_eq!(ordering, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_stability_equal_depths() {
        // Splats at identical depth keep their input order.
        let zs = [-5.0, -1.0, -5.0, -5.0, -1.0];
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 5);
        assert_eq!(ordering, vec![0, 2, 3, 1, 4]);
    }

    #[test]
    fn test_behind_camera_far_bucket() {
        // Behind-camera splats (z >= 0) land in the far bucket — drawn
        // before every in-front splat (the shader degenerates them anyway).
        let zs = [-1.0, 3.0, -10.0, 0.0];
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 4);
        assert_is_permutation(&ordering, 4);
        // Far bucket 0 holds behind-camera (1, 3) and the farthest splat
        // (2, whose z == zmin keys to 0); stable input order within the
        // bucket: 1, 2, 3. The nearest in-front splat (0) draws last.
        assert_eq!(ordering, vec![1, 2, 3, 0]);
    }

    #[test]
    fn test_identity_on_uniform_depth() {
        // zmax == zmin -> identity ordering, return 0.
        let zs = [-4.0, -4.0, -4.0];
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 0);
        assert_eq!(ordering, vec![0, 1, 2]);
    }

    #[test]
    fn test_identity_on_single_splat() {
        let zs = [-7.5];
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 0);
        assert_eq!(ordering, vec![0]);
    }

    #[test]
    fn test_identity_when_all_behind_camera() {
        let zs = [1.0, 2.0, 0.5];
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 0);
        assert_eq!(ordering, vec![0, 1, 2]);
    }

    #[test]
    fn test_empty_input() {
        let mut ordering: Vec<u32> = vec![];
        let sorted = sort_splats_by_depth(
            &[],
            &IDENTITY_MV,
            &mut ordering,
            0,
            0,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        );
        assert_eq!(sorted, 0);
    }

    #[test]
    fn test_nan_center_keys_to_near_bucket() {
        // A NaN view z never updates the bounds (NaN < 0.0 is false) and
        // never takes the behind-camera branch (NaN >= 0.0 is false), so
        // it reaches the key math where `f32::min(NaN, 65535.0)` returns
        // 65535 — the NEAR bucket. Pinned because the TS twin must
        // reproduce this exactly (Math.min(NaN, x) is NaN, so the twin
        // uses a `<` comparison instead); the NaN splat itself is
        // degenerate in the shader, but parity is exact-permutation.
        let zs = [f32::NAN, -3.0, -8.0];
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 3);
        assert_is_permutation(&ordering, 3);
        // Keys: idx2 (zmin) -> 0; idx1 (zmax) -> 65535; idx0 (NaN) ->
        // 65535. Bucket 65535 keeps input order: [0, 1].
        assert_eq!(ordering, vec![2, 0, 1]);
    }

    #[test]
    fn test_nanometer_scale_magnitudes() {
        // ~1e-6 coordinate magnitudes — raw f16 keys would underflow to a
        // single bucket; normalized keys must still order correctly.
        let zs = [-1.0e-6, -9.0e-6, -5.0e-6, -3.0e-6];
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 4);
        assert_is_permutation(&ordering, 4);
        assert_eq!(ordering, vec![1, 2, 3, 0]);
    }

    #[test]
    fn test_kilometer_scale_magnitudes() {
        // ~1e6 coordinate magnitudes — raw f16 keys would overflow to Inf.
        let zs = [-1.0e6, -9.0e6, -5.0e6, -3.0e6];
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 4);
        assert_is_permutation(&ordering, 4);
        assert_eq!(ordering, vec![1, 2, 3, 0]);
    }

    #[test]
    fn test_mixed_scale_depth_range() {
        // Depth range spanning 12 orders of magnitude stays a valid
        // back-to-front permutation (nearby splats may share a bucket).
        let zs = [-1.0e-6, -1.0e6, -1.0, -1.0e3, -1.0e-3];
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 5);
        assert_is_permutation(&ordering, 5);
        assert_back_to_front(&ordering, &zs);
        // The km-scale splat is unambiguously farthest.
        assert_eq!(ordering[0], 1);
    }

    #[test]
    fn test_model_view_translation_applied() {
        // A translation along view z shifts which splats are behind the
        // camera: mv translates z by -5, so world z = +2 -> view z = -3.
        let mut mv = IDENTITY_MV;
        mv[14] = -5.0;
        let centers = centers_with_z(&[2.0, -2.0, 4.0]);
        let mut ordering = vec![u32::MAX; 3];
        let sorted = sort_splats_by_depth(
            &centers,
            &mv,
            &mut ordering,
            3,
            0,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        );
        assert_eq!(sorted, 3);
        // View z: -3, -7, -1 -> back-to-front: 1, 0, 2.
        assert_eq!(ordering, vec![1, 0, 2]);
    }

    #[test]
    fn test_model_view_rotation_row_used() {
        // 90° rotation about y (column-major): view z comes from world x.
        // Column 0 = (0,0,-1), column 2 = (1,0,0).
        let mv: [f32; 16] = [
            0.0, 0.0, -1.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        // Centers along x: view z = -x.
        let centers = vec![
            1.0, 0.0, 0.0, // view z = -1 (nearest)
            9.0, 0.0, 0.0, // view z = -9 (farthest)
            5.0, 0.0, 0.0, // view z = -5
        ];
        let mut ordering = vec![u32::MAX; 3];
        let sorted = sort_splats_by_depth(
            &centers,
            &mv,
            &mut ordering,
            3,
            0,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        );
        assert_eq!(sorted, 3);
        assert_eq!(ordering, vec![1, 2, 0]);
    }

    // ---- per-shard bounds ------------------------------------------------

    #[test]
    fn test_shard_count_zero_leaves_slices_untouched() {
        // The unsharded production path must pay nothing and write nothing.
        let centers = centers_with_z(&[-1.0, -10.0, -5.0, -2.0]);
        let mut ordering = vec![u32::MAX; 4];
        let mut min = vec![7.0f32; 3];
        let mut max = vec![9.0f32; 3];
        let mut zmin = vec![7.0f32; 1];
        let mut zmax = vec![9.0f32; 1];
        sort_splats_by_depth(
            &centers,
            &IDENTITY_MV,
            &mut ordering,
            4,
            0,
            &mut min,
            &mut max,
            &mut zmin,
            &mut zmax,
        );
        assert_eq!(min, vec![7.0, 7.0, 7.0]);
        assert_eq!(max, vec![9.0, 9.0, 9.0]);
        assert_eq!(zmin, vec![7.0]);
        assert_eq!(zmax, vec![9.0]);
    }

    #[test]
    fn test_shard_bounds_single_shard_is_whole_node_box() {
        // S == 1 must reproduce the node's own AABB — the self-check that makes
        // the inert phase non-vacuous. x = index, y = 0, z as given.
        let zs = [-1.0, -10.0, -5.0, -2.0];
        let (_, sorted, min, max) = sort_sharded(&zs, 1);
        assert_eq!(sorted, 4);
        assert_eq!(min, vec![0.0, 0.0, -10.0]);
        assert_eq!(max, vec![3.0, 0.0, -1.0]);
    }

    #[test]
    fn test_shard_bounds_follow_the_sorted_order_not_storage_order() {
        // The whole point: box 0 covers the FARTHEST elements, which are not the
        // first ones in storage. zs = [-1, -10, -5, -2] sorts to [1, 2, 3, 0]
        // (z -10, -5, -2, -1), so with S = 2 shard 0 draws elements {1, 2}
        // (x = 1, 2) and shard 1 draws {3, 0} (x = 3, 0).
        let zs = [-1.0, -10.0, -5.0, -2.0];
        let (ordering, sorted, min, max) = sort_sharded(&zs, 2);
        assert_eq!(sorted, 4);
        assert_eq!(ordering, vec![1, 2, 3, 0]);
        // Shard 0: x in {1, 2}, z in {-10, -5}.
        assert_eq!(&min[0..3], &[1.0, 0.0, -10.0]);
        assert_eq!(&max[0..3], &[2.0, 0.0, -5.0]);
        // Shard 1: x in {3, 0}, z in {-2, -1}.
        assert_eq!(&min[3..6], &[0.0, 0.0, -2.0]);
        assert_eq!(&max[3..6], &[3.0, 0.0, -1.0]);
    }

    #[test]
    fn test_shard_bounds_partition_every_element_exactly_once() {
        // Union of the shard boxes must equal the whole-node box, for a shard
        // count that does NOT divide the element count evenly (7 / 3 -> 3, 3, 1).
        let zs = [-1.0, -7.0, -3.0, -5.0, -2.0, -6.0, -4.0];
        let (_, sorted, min, max) = sort_sharded(&zs, 3);
        assert_eq!(sorted, 7);
        let (whole_min, whole_max) = {
            let (_, _, m, x) = sort_sharded(&zs, 1);
            (m, x)
        };
        for axis in 0..3 {
            let union_min = (0..3)
                .map(|s| min[s * 3 + axis])
                .fold(f32::INFINITY, f32::min);
            let union_max = (0..3)
                .map(|s| max[s * 3 + axis])
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(union_min, whole_min[axis], "axis {axis} min");
            assert_eq!(union_max, whole_max[axis], "axis {axis} max");
        }
    }

    #[test]
    fn test_shard_view_z_intervals_are_ordered_and_contiguous() {
        // The cross-node merge key. Shards are contiguous ranges of a
        // back-to-front permutation, so their view-z intervals must ascend and
        // tile the node's whole depth range with no gap — that is what makes
        // them comparable against ANOTHER node's intervals.
        let zs = [-1.0, -8.0, -3.0, -6.0, -2.0, -7.0, -4.0, -5.0];
        let (_, sorted, _, _, zmin, zmax) = sort_sharded_full(&zs, 4);
        assert_eq!(sorted, 8);
        for s in 0..4 {
            assert!(zmin[s] <= zmax[s], "shard {s} interval is inverted");
        }
        for s in 1..4 {
            assert!(
                zmin[s] >= zmax[s - 1],
                "shard {s} starts at {} before shard {} ends at {}",
                zmin[s],
                s - 1,
                zmax[s - 1]
            );
        }
        // Farthest first, and the union covers the input range.
        assert_eq!(zmin[0], -8.0);
        assert_eq!(zmax[3], -1.0);
    }

    #[test]
    fn test_shard_view_z_uses_the_view_axis_not_world_z() {
        // The key must come from the SORT POSE. Under a 90-degree rotation about
        // y, view z is derived from world x — so the intervals must follow x,
        // which a world-space box could not tell you.
        let mv: [f32; 16] = [
            0.0, 0.0, -1.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        // Centers along x only: view z = -x, so x=8 is farthest.
        let mut centers = Vec::new();
        for x in [1.0f32, 8.0, 3.0, 6.0] {
            centers.extend_from_slice(&[x, 0.0, 0.0]);
        }
        let mut ordering = vec![u32::MAX; 4];
        let mut min = vec![0.0f32; 6];
        let mut max = vec![0.0f32; 6];
        let mut zmin = vec![0.0f32; 2];
        let mut zmax = vec![0.0f32; 2];
        let sorted = sort_splats_by_depth(
            &centers,
            &mv,
            &mut ordering,
            4,
            2,
            &mut min,
            &mut max,
            &mut zmin,
            &mut zmax,
        );
        assert_eq!(sorted, 4);
        // Back-to-front by view z: x = 8, 6, 3, 1 -> view z -8, -6, -3, -1.
        assert_eq!(ordering, vec![1, 3, 2, 0]);
        assert_eq!((zmin[0], zmax[0]), (-8.0, -6.0));
        assert_eq!((zmin[1], zmax[1]), (-3.0, -1.0));
    }

    #[test]
    fn test_shard_bounds_empty_shard_keeps_sentinel() {
        // More shards than elements: the tail shards get the empty-box sentinel,
        // which the caller must read as "no usable bounds".
        let zs = [-1.0, -2.0];
        let (_, _, min, max) = sort_sharded(&zs, 5);
        // shard_size = ceil(2/5) = 1, so shards 0 and 1 are populated.
        assert!(min[0].is_finite() && min[3].is_finite());
        for s in 2..5 {
            assert_eq!(min[s * 3], f32::INFINITY, "shard {s} min sentinel");
            assert_eq!(max[s * 3], f32::NEG_INFINITY, "shard {s} max sentinel");
        }
    }

    #[test]
    fn test_shard_bounds_written_on_identity_fallback() {
        // A degenerate depth range returns 0 but still writes valid boxes of the
        // identity ordering — they simply carry no depth meaning. The caller
        // keys off the return value, not off box emptiness.
        let zs = [-4.0, -4.0, -4.0, -4.0];
        let (ordering, sorted, min, max) = sort_sharded(&zs, 2);
        assert_eq!(sorted, 0);
        assert_eq!(ordering, vec![0, 1, 2, 3]);
        // Identity order, so shard 0 is elements {0, 1} and shard 1 is {2, 3}.
        assert_eq!(&min[0..3], &[0.0, 0.0, -4.0]);
        assert_eq!(&max[0..3], &[1.0, 0.0, -4.0]);
        assert_eq!(&min[3..6], &[2.0, 0.0, -4.0]);
        assert_eq!(&max[3..6], &[3.0, 0.0, -4.0]);
    }

    #[test]
    fn test_shard_bounds_empty_input_stamps_sentinels() {
        // No elements at all must still clear the slices, so a caller cannot
        // read a previous frame's boxes.
        let mut ordering: Vec<u32> = vec![];
        let mut min = vec![7.0f32; 6];
        let mut max = vec![9.0f32; 6];
        let mut zmin = vec![7.0f32; 2];
        let mut zmax = vec![9.0f32; 2];
        let sorted = sort_splats_by_depth(
            &[],
            &IDENTITY_MV,
            &mut ordering,
            0,
            2,
            &mut min,
            &mut max,
            &mut zmin,
            &mut zmax,
        );
        assert_eq!(sorted, 0);
        assert_eq!(min, vec![f32::INFINITY; 6]);
        assert_eq!(max, vec![f32::NEG_INFINITY; 6]);
    }

    #[test]
    fn test_shard_bounds_exclude_nan_center_without_poisoning_the_box() {
        // A NaN view z keys to the NEAR bucket (see
        // test_nan_center_keys_to_near_bucket), so a NaN center lands in the LAST
        // shard. That shard's box must stay a valid box of its FINITE elements:
        // one bad element among thousands must not cost a whole shard its place
        // in the depth merge.
        let zs = [f32::NAN, -3.0, -8.0, -6.0];
        let (ordering, sorted, min, max) = sort_sharded(&zs, 2);
        assert_eq!(sorted, 4);
        // Keys: idx2 (zmin) -> 0, idx3 -> mid, idx1 (zmax) and idx0 (NaN) ->
        // 65535 (stable within the bucket: 0 before 1).
        assert_eq!(ordering, vec![2, 3, 0, 1]);
        // Shard 1 draws elements {0 (NaN, x=0), 1 (z=-3, x=1)}. The NaN is
        // excluded, so the box is exactly element 1's z with both x values.
        assert_eq!(&min[3..6], &[0.0, 0.0, -3.0]);
        assert_eq!(&max[3..6], &[1.0, 0.0, -3.0]);
        assert!(
            min[5].is_finite() && max[5].is_finite(),
            "one NaN element must not poison its shard's box"
        );
        // Shard 0 is untouched by the NaN.
        assert_eq!(&min[0..3], &[2.0, 0.0, -8.0]);
        assert_eq!(&max[0..3], &[3.0, 0.0, -6.0]);
    }

    #[test]
    fn test_shard_bounds_all_nan_shard_keeps_sentinel() {
        // The other half of the NaN contract: when a shard has NO finite element
        // there is nothing to bound, so it must fall back to the empty sentinel
        // the caller already reads as "no usable bounds" — not to a garbage box.
        // Declared finite-first so the two NaNs (both key 65535, stable within
        // the bucket) end up alone in the last shard. Declaring them first
        // instead splits one into each shard, which is why the order matters
        // here: keys are [0, 65535, 65535, 65535] -> ordering [0, 1, 2, 3].
        let zs = [-8.0, -6.0, f32::NAN, f32::NAN];
        let (ordering, _, min, max) = sort_sharded(&zs, 2);
        assert_eq!(ordering, vec![0, 1, 2, 3]);
        // Shard 0 = elements {0, 1}, both finite.
        assert_eq!(&min[0..3], &[0.0, 0.0, -8.0]);
        assert_eq!(&max[0..3], &[1.0, 0.0, -6.0]);
        // Shard 1 = elements {2, 3}: their x/y ARE finite, so only the z axis
        // has nothing to bound and keeps its sentinel. The caller's guard rejects
        // a box with any non-finite bound, so this is still "no usable bounds".
        assert_eq!(min[5], f32::INFINITY, "all-NaN z keeps the min sentinel");
        assert_eq!(
            max[5],
            f32::NEG_INFINITY,
            "all-NaN z keeps the max sentinel"
        );
        assert!(min[5] > max[5], "an empty axis must read as min > max");
    }

    #[test]
    fn test_shard_bounds_propagate_infinite_center() {
        // Unlike NaN, ±Infinity comparisons DO succeed, so an infinite center
        // reaches the box and the caller's finite check rejects that shard. The
        // asymmetry is deliberate and worth pinning.
        let mut centers = centers_with_z(&[-1.0, -2.0, -3.0, -4.0]);
        centers[0] = f32::NEG_INFINITY; // element 0's x
        let mut ordering = vec![u32::MAX; 4];
        let mut min = vec![0.0f32; 3];
        let mut max = vec![0.0f32; 3];
        sort_splats_by_depth(
            &centers,
            &IDENTITY_MV,
            &mut ordering,
            4,
            1,
            &mut min,
            &mut max,
            &mut [0.0f32],
            &mut [0.0f32],
        );
        assert_eq!(min[0], f32::NEG_INFINITY);
        assert!(!min[0].is_finite());
    }

    #[test]
    fn test_large_random_permutation_monotone() {
        // Deterministic pseudo-random depths; verify permutation +
        // back-to-front monotonicity on the actual z values (buckets are
        // fine-grained enough at this range for strict monotone checks).
        let mut zs = Vec::with_capacity(10_000);
        let mut state: u32 = 0x1234_5678;
        for _ in 0..10_000 {
            // xorshift32
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let unit = (state as f32) / (u32::MAX as f32); // [0, 1]
            zs.push(-1.0 - unit * 999.0); // [-1000, -1]
        }
        let (ordering, sorted) = sort(&zs);
        assert_eq!(sorted, 10_000);
        assert_is_permutation(&ordering, 10_000);
        let mut prev = f32::NEG_INFINITY;
        for &idx in &ordering {
            let z = zs[idx as usize];
            // Allow equal keys (same bucket): z may regress by at most one
            // bucket width.
            let bucket_width = 999.0 / DEPTH_KEY_MAX;
            assert!(
                z >= prev - bucket_width,
                "z {z} regressed more than one bucket below {prev}"
            );
            prev = prev.max(z);
        }
    }
}
