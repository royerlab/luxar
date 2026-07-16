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
//! fastest measured; see `perf-budget.test.ts` for the enforced floor.
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

use wasm_bindgen::prelude::*;

/// Number of distinct depth keys (full uint16 key range).
const DEPTH_SORT_BUCKETS: usize = 1 << 16;

/// Maximum key value (`DEPTH_SORT_BUCKETS - 1` as f32 for normalization).
const DEPTH_KEY_MAX: f32 = (DEPTH_SORT_BUCKETS - 1) as f32;

/// Sort splats back-to-front by camera-space depth.
///
/// # Arguments
/// - `centers3`: Projected 3D splat centers `[count * 3]` (x, y, z triplets)
/// - `model_view`: Column-major 4x4 model-view matrix `[16]`
///   (`camera.matrixWorldInverse × mesh.matrixWorld`, THREE.js layout)
/// - `ordering`: Output permutation `[count]` — `ordering[j]` is the original
///   splat index drawn at instance slot `j` (slot 0 = farthest)
/// - `count`: Number of splats
///
/// # Returns
/// Number of splats placed via depth keys, or `0` when the identity
/// fallback was taken (degenerate depth range — the ordering is still
/// fully written).
#[wasm_bindgen]
pub fn sort_splats_by_depth(
    centers3: &[f32],
    model_view: &[f32],
    ordering: &mut [u32],
    count: usize,
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

    if count == 0 {
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
        let sorted = sort_splats_by_depth(&centers, &IDENTITY_MV, &mut ordering, zs.len());
        (ordering, sorted)
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
                let zmin = zs.iter().cloned().filter(|&z| z < 0.0).fold(f32::INFINITY, f32::min);
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
        let sorted = sort_splats_by_depth(&[], &IDENTITY_MV, &mut ordering, 0);
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
        let sorted = sort_splats_by_depth(&centers, &mv, &mut ordering, 3);
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
        let sorted = sort_splats_by_depth(&centers, &mv, &mut ordering, 3);
        assert_eq!(sorted, 3);
        assert_eq!(ordering, vec![1, 2, 0]);
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
