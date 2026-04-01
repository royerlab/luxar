# Rust WASM Test Suite Review

**Date**: 2026-03-31
**Reviewer**: Claude Opus 4.6 (1M context)
**Scope**: All `#[cfg(test)]` modules in `packages/luxar-viewer/src/wasm/rust/src/`
**Last Updated**: 2026-03-31

---

## Status

**No fixes have been applied.** All 21 findings (1 CRITICAL reclassified to HIGH, 5 HIGH, 9 MEDIUM, 6 LOW) remain open. No commits have touched the WASM Rust source files since this review was created. The findings below are unchanged from the original review.

---

## Executive Summary

The test suite is **generally solid** for the core mathematical operations, with good coverage of the happy path and some edge cases. The most significant gap is in **lines_clipping.rs**, which has the most complex logic but incomplete batch-function testing. Numerical tolerances are reasonable throughout. The biggest systemic weakness is the **absence of adversarial/boundary-condition tests** across most modules -- tests tend to verify "normal" inputs but rarely probe degenerate geometry, NaN/Inf propagation, or off-by-one indexing.

**Overall Quality**: B+ (Good foundation, meaningful gaps in edge-case coverage)

### Severity Summary

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 1 | Potential correctness bug in batch clipping test |
| HIGH | 5 | Missing edge-case coverage that could mask real bugs |
| MEDIUM | 9 | Gaps that reduce confidence but unlikely to cause production issues |
| LOW | 6 | Minor improvements and hardening |

---

## Module-by-Module Analysis

### 1. `common.rs` -- Shared Constants & Utilities

**Tests**: None (no `#[cfg(test)]` module)

**Finding [MEDIUM]**: `packed_index` and `validate_ndim` are tested indirectly through other modules, but `packed_index` is a critical function (incorrect indexing corrupts all Cholesky operations). While `gsplats_processing.rs` has a dedicated `test_packed_index`, `common.rs` itself has no unit tests.

**Recommendation**: Add a test in `common.rs` for `packed_index` boundary cases (row=0, col=0; row=15, col=15 for max dims; verify formula against manual enumeration for 4D).

---

### 2. `lines_clipping.rs` -- Liang-Barsky nD Line Clipping

**Tests**: 7 tests covering `clip_segment_single`, `lerp`, `lerp_vec3`, `distance_3d`, `clip_segments_batch`.

#### Finding [CRITICAL]: `test_clip_segments_batch` -- Incorrect expected value for seg1

Lines 615-618: The test asserts that segment 1 (v1 at dim3=5.0, v2 at dim3=0.0) is clipped with `t2 = 0.1`.

The comment says: "t where dim3 crosses 4.5: 5.0 + t*(0.0-5.0) = 4.5 -> t = 0.1". This math is correct: t = (5.0 - 4.5) / (5.0 - 0.0) = 0.1. **However**, the test asserts `t1[1] = 0.0`, meaning v1 (at dim3=5.0) is treated as "inside" the slice [4.5, 5.5]. This is correct since v1 is inside.

But for t2: the code should clip to the boundary where the segment exits the slice. v2 is at dim3=0.0, which is below the slice. So the segment exits at dim3=4.5. t for that crossing: `(slice_min - v1) / (v2 - v1) = (4.5 - 5.0) / (0.0 - 5.0) = (-0.5) / (-5.0) = 0.1`. So t2=0.1 is correct.

**On re-examination**: The math checks out. However, the test does NOT verify the `interpolate_clipped_positions`, `interpolate_scalars_batch`, `interpolate_colors_batch`, `calculate_segment_lengths`, or `mark_clipped_endpoints` functions at all. These are significant batch operations with no dedicated tests. **Reclassified to HIGH.**

#### Finding [HIGH]: No tests for `interpolate_clipped_positions`

This function projects clipped nD positions to 3D display space. It involves dimension selection, interpolation, and zero-padding for <3 display dims. No tests exist.

#### Finding [HIGH]: No tests for `interpolate_scalars_batch`, `interpolate_colors_batch`, `calculate_segment_lengths`, `mark_clipped_endpoints`

Four batch functions with zero test coverage. These perform interpolation and compaction operations that could easily have off-by-one or stride errors.

#### Finding [MEDIUM]: No test for Case D (both out, opposite sides) in `clip_segment_single`

The test `test_clip_segment_crosses_slice` tests a segment where one endpoint is "in" on the display dims (large tolerance) and crosses only in dim3. A true Case D test (both endpoints outside the slice on opposite sides of a non-displayed dimension) is not explicitly tested -- the existing crossing test has p1 at dim3=0 and p2 at dim3=10 with slice at [4.5, 5.5], so both are outside in dim3. This IS Case D. Good.

#### Finding [MEDIUM]: No multi-dimension clipping test

All tests clip on a single hidden dimension (dim3). No test clips on multiple hidden dimensions simultaneously, which exercises the t1/t2 intersection logic across dimensions.

#### Finding [LOW]: `lerp` and `lerp_vec3` tests are trivially simple

These tests verify basic interpolation at t=0, t=0.5, t=1.0 but not edge cases like t<0, t>1, NaN inputs, or very large values.

---

### 3. `projection.rs` -- nD to 3D Projection & Utilities

**Tests**: 6 tests covering `extract_3d_positions`, `calculate_bounds_3d`, `compact_by_mask`, `radii_to_visibility_mask`.

#### Finding [MEDIUM]: `compact_by_mask` only tests stride=3

The function has three code paths: stride=1, stride=3, and generic. Only stride=3 is tested. The stride=1 fast path (used for radii/sharpness compaction) has no dedicated test.

#### Finding [MEDIUM]: `count_visible` has no test

The `count_visible` function is exported but has no dedicated test.

#### Finding [LOW]: No test for `extract_3d_positions` with 1D display

The function handles <3 display dims by zero-padding. Only 2D and 5D->3D are tested. A 1D display case (single dimension) would exercise the most padding.

#### Finding [LOW]: `calculate_bounds_3d` -- no test for single point or NaN/Inf

Single-point case should return min==max. NaN positions could silently corrupt bounds (NaN comparisons return false for both min and max). No NaN-robustness test exists.

---

### 4. `effective_radii.rs` -- nD Hypersphere Slicing

**Tests**: 6 tests including a `#[should_panic]` test for dimension limit validation.

#### Finding [MEDIUM]: Boundary test for `distance_squared == radius_squared` (exact tangent)

When the distance equals the radius exactly, the effective radius should be 0.0 (the `>=` comparison at line 113 returns 0). The 5D test `test_effective_radii_5d_with_hidden_dims` does test this case (distance = 1.0, radius = 1.0), which is good. However, there is no test for the case where `distance_squared` is *very slightly less* than `radius_squared` (near-tangent), which would exercise numerical precision.

#### Finding [MEDIUM]: No test with mixed spatial and discrete dimensions in the same point set

`test_discrete_dimension_filtering` tests discrete filtering, and `test_effective_radii_partial_intersection` tests spatial distance. No test combines both (e.g., a point that passes the discrete filter but fails the spatial distance check, and vice versa).

#### Finding [LOW]: No test for `spatial_extend_dims` shorter than `ndim`

Line 84-87: If `spatial_extend_dims` is shorter than `ndim`, dimensions default to spatial. This fallback behavior is untested.

**Positive notes**: The `test_dimension_limit_validation` and `test_dimension_at_limit` tests are excellent boundary tests for the 16-dim limit. The mathematical values in `test_effective_radii_partial_intersection` (sqrt(1 - 0.36) = 0.8) are verified correct.

---

### 5. `gsplats_processing.rs` -- Mahalanobis Distance & Cholesky Marginals

**Tests**: 10 tests -- the most thorough test module in the suite.

**Positive notes**: This module has exemplary tests. The `test_marginal_vs_raw_extraction_difference` test directly verifies the correctness rationale for the marginal Cholesky implementation by proving that naive extraction gives wrong results. The `test_mahalanobis_with_marginal_cholesky` provides end-to-end mathematical verification.

#### Finding [MEDIUM]: `test_mahalanobis_with_marginal_cholesky` tolerance is too loose

Line 639: `assert!((dist - 0.5).abs() < 0.05)` -- a 10% relative tolerance for a well-defined mathematical operation. The forward substitution yields y[0]=0.5, y[1]=-0.0621, ||y||=0.5038. The assertion should use `1e-3` or tighter.

#### Finding [MEDIUM]: `test_attenuation_with_correlated_cholesky` uses a wide range assertion

Line 681: `attenuation[0] > 0.9 && attenuation[0] < 1.0` -- a 10% band. The expected value is ~0.970. A tighter bound (e.g., `0.96..0.98`) would catch regressions better.

#### Finding [LOW]: `compact_attenuated_amplitudes` has no dedicated test

This function multiplies amplitudes by attenuation and compacts. It is tested indirectly through the attenuation test, but a dedicated test with known values would be valuable.

#### Finding [LOW]: No test for degenerate Cholesky (near-zero diagonal)

The code handles `diag <= EPSILON` by substituting `CHOLESKY_EPSILON.sqrt()` or returning 0. No test verifies behavior with a nearly-singular covariance matrix.

---

### 6. `gsplats.rs` -- GSplat Visibility Pre-filter

**Tests**: 2 tests for `compute_nd_visibility_gsplats`.

#### Finding [HIGH]: Only 2 tests for a geometrically complex function

This function uses ellipsoid extent estimation from Cholesky diagonals combined with per-dimension tolerance normalization. Only a basic 3D and a 4D test exist.

#### Finding [HIGH]: No test for non-identity Cholesky (anisotropic splats)

Both tests use identity Cholesky factors. The function extracts `max_extent` from diagonal elements. For a Cholesky like `[[3, 0, 0], [1, 2, 0], [0, 0, 5]]`, the max diagonal is 5, making a splat visible at greater distances. This is entirely untested.

#### Finding [MEDIUM]: Visibility metric is a normalized-distance ellipsoid, but tests use extreme positions

Splat 1 in `test_gsplat_visibility_basic` is at (10,10,10) with tolerance=(2,2,2) and extent=1. The normalized distance squared would be (10/3)^2 * 3 = 33.3, far exceeding 1.0. A test with a splat near the visibility boundary would better exercise the math.

---

### 7. `decode.rs` -- Quantized & LUT Decoding

**Tests**: 8 tests covering all decode functions.

**Positive notes**: Good coverage of all function variants (u8, u16, scalar, row, log-space, broadcast). The `test_decode_broadcasted_short_value` edge case is a nice touch.

#### Finding [LOW]: `decode_log_scalar_u16` has no dedicated test

Only `decode_log_scalar_u8` is tested. The u16 variant uses `65535.0` as divisor and could have precision differences.

#### Finding [LOW]: No test for `decode_lut_scalar_u16`

Only the u8 variant is tested directly.

#### Finding [MEDIUM]: `decode_quantized_u8` tolerance is inconsistent

Line 179: `output[1]` asserts `5.02` with tolerance `0.1`. The exact value is `128/255 * 10 = 5.0196...`. A tighter tolerance (e.g., `1e-3`) would be more rigorous. Compare with the bounds assertions on lines 178 and 180 which use `0.01`.

---

### 8. `lines.rs` -- Line Segment Visibility

**Tests**: 3 tests for `compute_nd_visibility_lines`.

#### Finding [HIGH]: No nD test (only 3D)

All three tests use 3D data. The function operates in arbitrary nD space. A 4D+ test (e.g., two segments identical in XYZ but differing in T dimension) is missing and would verify the nD visibility logic.

#### Finding [MEDIUM]: No test for width-dependent visibility

All tests use `widths = [0.1, 0.1]`. The `check_point_visibility` function adds width to tolerance. A test where a point would be invisible with width=0 but visible with a large width would verify this critical behavior.

---

### 9. `points.rs` -- Point Visibility

**Tests**: 3 tests for `compute_nd_visibility_points`.

#### Finding [MEDIUM]: `test_point_visibility_3d` uses weak assertion for Point 1

Line 98: `assert!(count >= 2, "At least 2 points should be visible")` -- this is unnecessarily weak. Point 1 at z=5 with tolerance 1.0 and radius 0.5 has normalized distance = 5/1.5 = 3.33 per dim, so dist_sq = 3.33^2 = 11.1 >> 1.0 (not visible). The count should be exactly 2, and the assertion should be `assert_eq!(count, 2)`.

#### Finding [MEDIUM]: No test for boundary visibility (point exactly at tolerance edge)

No test places a point exactly at the visibility boundary (dist_sq = 1.0) to verify the `<=` vs `<` behavior.

---

### 10. `spatial.rs` -- Chunk AABB Queries

**Tests**: 2 tests for `query_chunks_for_view`.

#### Finding [HIGH]: Extremely thin test coverage for a spatial indexing function

Only 2 tests: one basic 3D and one empty-input edge case. Missing:

- **Boundary intersection**: chunk edge exactly touches query edge (tests the `<` vs `<=` logic at line 49 -- note: the code uses strict `<` / `>`, meaning touching boundaries are considered intersecting; this should be verified with a test)
- **nD test**: no test with ndim > 3
- **Single-dimension rejection**: chunk that intersects in all dims except one (verifies early-exit logic)
- **Large number of chunks**: no stress or ordering test

---

### 11. `lib.rs` -- Module Root

**Tests**: None (re-exports only)

No integration-level tests exist. While individual module tests are reasonable, there are no tests that chain operations (e.g., decode -> visibility -> projection -> bounds), which would catch interface mismatches.

**Finding [MEDIUM]**: No integration test combining visibility + projection + compaction pipeline.

---

## Cross-Cutting Concerns

### NaN/Inf Propagation [MEDIUM]

No module tests behavior when input contains NaN or Infinity values. In WASM, NaN propagation through `f32` operations is well-defined (IEEE 754), but comparison operators return `false` for NaN, which means:
- `NaN <= 1.0` is `false` (point invisible) -- probably correct behavior
- `NaN.min(x)` returns `x` in Rust -- could silently corrupt bounds calculations

A single NaN-robustness test per module would provide confidence.

### Empty Input Handling [LOW]

Most functions do not explicitly handle `num_points=0` or `num_segments=0`. The loops simply execute zero iterations, which is correct. `calculate_bounds_3d` does handle `num_points=0` explicitly (and is tested). No other empty-input tests exist except `spatial.rs`.

### Consistency of Visibility Metric [MEDIUM]

Three modules (`points.rs`, `lines.rs`, `gsplats.rs`) use the same normalized-distance-squared metric (`dist_sq <= 1.0`). This metric treats the tolerance as defining an **ellipsoid** in nD space. This is consistent, but:
- `effective_radii.rs` uses **Euclidean** distance in hidden dimensions (Pythagorean theorem), which is a **spherical** model
- The visibility functions use **per-dimension normalization**, which is an **ellipsoidal** model

These are different geometric models. The inconsistency is not a bug per se (they serve different purposes), but no test verifies that the two approaches agree for the isotropic case (equal tolerances).

---

## Recommendations (Priority Order)

1. **[CRITICAL->HIGH reclassified] Add tests for `interpolate_clipped_positions`, `interpolate_scalars_batch`, `interpolate_colors_batch`, `mark_clipped_endpoints`, `calculate_segment_lengths`** in `lines_clipping.rs`. These batch functions have zero coverage.

2. **[HIGH] Add nD tests for `lines.rs` and anisotropic Cholesky tests for `gsplats.rs`**. Both modules test only the simplest geometric configurations.

3. **[HIGH] Expand `spatial.rs` tests** to include boundary intersection, nD queries, and single-dimension rejection.

4. **[MEDIUM] Tighten tolerances** in `gsplats_processing.rs` tests (`test_mahalanobis_with_marginal_cholesky`: 0.05 -> 0.001, `test_attenuation_with_correlated_cholesky`: 0.9-1.0 -> 0.96-0.98).

5. **[MEDIUM] Add stride=1 and generic-stride tests** for `compact_by_mask` in `projection.rs`.

6. **[MEDIUM] Strengthen weak assertions**: `points.rs` line 98 (`>= 2` -> `== 2`), `decode.rs` line 179 (tighter tolerance).

7. **[MEDIUM] Add boundary-condition tests**: point exactly at visibility threshold, chunk edge touching query edge, near-tangent effective radius.

8. **[LOW] Add `count_visible` and `compact_attenuated_amplitudes` dedicated tests**.

9. **[LOW] Add NaN-input test** for at least `calculate_bounds_3d` and one visibility function.

10. **[LOW] Add an integration test** in `lib.rs` that chains decode -> visibility -> projection -> bounds.

---

## Verified Correct Mathematical Values

The following expected values in tests were manually verified:

| Test | Value | Verification |
|------|-------|-------------|
| `effective_radii::test_effective_radii_partial_intersection` | R_eff = 0.8 | sqrt(1.0 - 0.36) = sqrt(0.64) = 0.8 |
| `effective_radii::test_effective_radii_5d_with_hidden_dims` | R_eff = 0.0 | sqrt(0.36 + 0.64) = 1.0 = R, so R_eff = 0 |
| `gsplats_processing::test_mahalanobis_distance_identity` | dist = 5.0 | ||[3,4,0]|| = 5.0 |
| `gsplats_processing::test_mahalanobis_distance_scaled` | dist = 2.0 | ||[4/2, 0, 0]|| = 2.0 |
| `gsplats_processing::test_compute_marginal_cholesky_diagonal` | L_S = diag(2,5) | Sigma = diag(4,25), chol = diag(2,5) |
| `gsplats_processing::test_compute_marginal_cholesky_correlated` | L_S[1,1] = sqrt(16.25) | Sigma_22 = 16.5, L_S[1,0]^2 = 0.25, sqrt(16.25) = 4.031 |
| `lines_clipping::test_clip_segment_crosses_slice` | t1=0.45, t2=0.55 | (4.5-0)/10=0.45, (5.5-0)/10=0.55 |
| `lines_clipping::test_clip_segments_batch` seg1 | t2=0.1 | (4.5-5.0)/(0.0-5.0) = 0.1 |
| `decode::test_decode_quantized_u8` | output[1] ~ 5.02 | 128/255 * 10 = 5.0196 |

All mathematical expected values are **correct**.

---

## Recommended Next Batch

The following 5 issues are recommended for the next fix cycle, ordered by impact (highest first):

### 1. Add tests for batch functions in `lines_clipping.rs` [HIGH]

**Findings**: `interpolate_clipped_positions`, `interpolate_scalars_batch`, `interpolate_colors_batch`, `calculate_segment_lengths`, `mark_clipped_endpoints` -- five exported batch functions with **zero test coverage**. These perform nD-to-3D interpolation, color/scalar lerping, and segment compaction. Off-by-one or stride errors here would silently corrupt rendered line geometry.

**Effort**: Medium (requires constructing nD line data and verifying interpolated 3D output).

### 2. Add anisotropic Cholesky tests for `gsplats.rs` [HIGH]

**Finding**: Both existing `compute_nd_visibility_gsplats` tests use identity Cholesky factors. Real Gaussian splats are anisotropic. The `max_extent` extraction from non-diagonal Cholesky is completely untested, meaning a regression in extent computation would go undetected.

**Effort**: Low (add one test with a known non-identity Cholesky, verify visibility boundary).

### 3. Expand `spatial.rs` chunk query tests [HIGH]

**Finding**: Only 2 tests exist for `query_chunks_for_view` -- a basic 3D case and an empty-input case. Missing: boundary intersection (touching edges), nD queries, and single-dimension rejection. This function gates which data chunks are loaded, so false negatives mean missing geometry and false positives mean wasted bandwidth.

**Effort**: Low (add 3-4 small tests with crafted AABB configurations).

### 4. Add nD tests for `lines.rs` visibility [HIGH]

**Finding**: All 3 tests use 3D data only. The function operates in arbitrary nD. A 4D+ test (segments identical in XYZ but differing in a hidden dimension) would verify that nD visibility filtering actually works for the line geometry type.

**Effort**: Low (add one 4D test case to existing test structure).

### 5. Tighten assertion tolerances in `gsplats_processing.rs` [MEDIUM]

**Finding**: `test_mahalanobis_with_marginal_cholesky` uses 10% relative tolerance (0.05 on a value of 0.5038), and `test_attenuation_with_correlated_cholesky` uses a 10% band (0.9-1.0 for expected ~0.970). These are deterministic math operations -- tolerances should be 1e-3 or tighter to catch numerical regressions.

**Effort**: Trivial (change two assertion constants).
