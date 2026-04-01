# Code Review: Python Core Tests (`packages/luxar/src/luxar/core/tests/`)

**Reviewer:** Claude (automated review)
**Date:** 2026-03-31
**Scope:** All 18 test files in `core/tests/`, cross-referenced against source code in `core/`

---

## Status

**Last updated:** 2026-03-31

The fixes shipped in PR #53 (`test_reader_nodes.py` parent= API misuse and `test_roundtrip.py` shape-only assertions) apply to the **IO tests report**, not this core tests report. All findings below remain **UNFIXED**.

| Finding | Status |
|---------|--------|
| All 24 findings (1 CRITICAL, 5 HIGH, 10 MEDIUM, 8 LOW) | **UNFIXED** |

---

## Executive Summary

The test suite is **generally solid** with good coverage of the core API surface. The tests are well-organized, use proper fixtures (`tmp_path`), and exercise real code paths through the `LuxarZarrCompiler` rather than mocking. However, there are several notable gaps, a few correctness concerns, and significant redundancy between files. The most critical issue is that `compose()` in transforms has a subtle documentation/implementation inconsistency that the tests happen to match but could confuse future developers.

**Severity distribution:**
- CRITICAL: 1
- HIGH: 5
- MEDIUM: 10
- LOW: 8

---

## Per-File Analysis

### 1. `test_datanode_types.py`

**What it tests:** `DataNode`, `Lines`, `GSplats`, and the `add_gsplats_from_data`/`add_gsplats_from_file` workflows.

**Rigor: Good.** Tests cover all line types (polyline, segments, loop, indexed), GSplats in 2D/3D, broadcast amplitude/cholesky, and validation error paths.

**Findings:**

- **MEDIUM - Missing edge case: empty data arrays.** No test for zero-length positions/centers (e.g., `np.zeros((0, 3))`). This is a realistic scenario when filtering produces no results.

- **LOW - No test for Lines with both colors AND sharpness simultaneously.** Tests check each independently but never together.

- **LOW - `test_add_gsplats_broadcast_cholesky` has ambiguous assertion.** It only asserts `n_splats == 3` but doesn't verify the broadcast actually happened correctly (i.e., all splats got the same cholesky values in the zarr store).

---

### 2. `test_dimension_metadata.py`

**What it tests:** Dimension metadata creation, validation, and zarr persistence.

**Rigor: Moderate.** Tests basic dimension creation and zarr persistence but relies heavily on the `Dimensions` API tested more thoroughly in `test_dimensions.py`.

**Findings:**

- **LOW - Redundancy with `test_dimensions.py`.** `test_scene_dimension_persistence` partially duplicates `TestSceneDimensionMetadata.test_scene_dimension_metadata`. The file acknowledges this with comments about moved/removed tests.

- **MEDIUM - `validate_positions` tested here but lives in `luxar.validation.types`.** This function is not part of `core/` -- the test imports from `luxar.validation.types` but lives in `core/tests/`. This creates a misleading test location.

---

### 3. `test_dimensions.py`

**What it tests:** `Dimension` and `Dimensions` classes -- creation, validation, serialization, categorical dimensions.

**Rigor: Very Good.** Excellent use of `@pytest.mark.parametrize` for validation edge cases. Categorical dimension tests are thorough.

**Findings:**

- **MEDIUM - Missing test for `Dimensions.__len__`.** The `Dimensions` class implements `__len__()` (line 213 of dimensions.py) but no test exercises it. Only `ndim` is tested.

- **LOW - `test_from_positions` doesn't test >5 dimensions.** The auto-naming for dimensions beyond 3 generates `dim3`, `dim4`, etc. This path is exercised for 5D but not for higher counts where the naming pattern matters.

- **LOW - No test for `Dimension.scale` usage in computation.** Scale is validated (must be positive) and serialized, but no test verifies it affects any downstream computation. This may be by design if scale is only consumed by the viewer.

---

### 4. `test_dim_order.py`

**What it tests:** `dim_order` parameter for mapping data columns to scene dimensions when adding points, lines, and gsplats.

**Rigor: Excellent.** This is one of the strongest test files. It verifies stored zarr data with `np.testing.assert_allclose`, checks covariance matrix correctness after Cholesky embedding, and covers all validation error paths.

**Findings:**

- **HIGH - Cholesky permutation test (`test_gsplats_cholesky_reorder_preserves_covariance`) is mathematically rigorous** but only tests a single anisotropic case. There is no test for the degenerate case where `dim_order` is identity (no permutation needed), which would catch off-by-one errors in the mapping.

- **LOW - Missing test for `fill_sigma` with a non-default value and then verifying the actual variance in the stored data.** `test_3d_gsplats_to_4d_scene` does this for `fill_sigma={"Time": 0.5}` but only one case is tested.

---

### 5. `test_extend_to_all.py`

**What it tests:** The `extend_to_all` parameter for `add_points`.

**Rigor: Good.** Covers None, empty list, explicit list, "all" string, invalid value, and warning behavior.

**Findings:**

- **MEDIUM - `test_warning_when_none_and_candidates_detected` could be fragile.** It uses `pytest.warns(UserWarning, match="Time")` which matches any warning containing "Time". If other warnings are added that mention "Time", this could produce false passes or failures.

- **LOW - No test for `extend_to_all` with a displayed dimension name.** What happens if you pass `extend_to_all=["X"]` where X is a displayed dimension? The source code doesn't explicitly reject this, so the behavior is untested.

---

### 6. `test_group.py`

**What it tests:** `Group.add_points/add_lines/add_gsplats`, nested groups, backward compatibility (`parent=` parameter), multi-LOD gsplats.

**Rigor: Very Good.** Tests both the new `group.add_points()` pattern and the old `scene.add_points(parent=group)` pattern. Multi-LOD tests are thorough.

**Findings:**

- **MEDIUM - `TestGroupNotAttachedToScene.test_detached_group_raises` only tests `add_points`.** The same error should occur for `add_lines` and `add_gsplats` on a detached group, but these paths are not tested.

- **LOW - No test for deeply nested groups (3+ levels) with data at each level.** Only 2-level nesting is tested.

---

### 7. `test_gsplats_extend_to_all.py`

**What it tests:** `extend_to_all` for `add_gsplats`, mirroring `test_extend_to_all.py` for points.

**Rigor: Good.** Nearly identical structure to `test_extend_to_all.py` but adapted for gsplats.

**Findings:**

- **HIGH - Significant redundancy with `test_extend_to_all.py`.** 8 out of 11 test methods are structural duplicates of the points version, testing the same `_resolve_extend_to_all` code path in `Scene`. The shared logic (Scene._resolve_extend_to_all) is the same for points, lines, and gsplats. A parameterized fixture or shared test base would eliminate ~150 lines of duplication.

- **MEDIUM - `create_test_cholesky` helper generates random cholesky factors that may not be valid lower-triangular matrices.** `np.random.rand(n_splats, k) * 0.1` produces arbitrary values, not proper Cholesky factors. The diagonal elements could be near-zero, leading to degenerate covariance matrices. This doesn't affect the tests (they only check metadata/attributes), but it's misleading.

---

### 8. `test_hdr_colors.py`

**What it tests:** HDR color support, SDR-to-uint8 conversion, broadcasting, negative color rejection, channel count validation.

**Rigor: Good.** Covers the AUTO mode behavior (SDR -> uint8, HDR -> float32), broadcasting, precision, and error cases.

**Findings:**

- **HIGH - `test_hdr_colors_extreme` uses `compiler.write_points()` directly instead of `scene.add_points()`.** This bypasses the Scene's dimension validation. While it tests the compiler-level HDR path, it's testing a different API surface than the other tests in this file. Similarly, `test_negative_color_rejection` and `test_color_channel_count` use `compiler.write_points()`.

- **MEDIUM - `test_color_precision` uses `ArrayDecoder` for decoding but other tests read raw zarr data.** This inconsistency means some tests might pass even if the encoding/decoding pipeline has bugs, because they read the raw stored data rather than the decoded output.

- **LOW - `test_mixed_hdr_sdr_colors` assertion `assert np.all(stored_colors[sdr_mask] <= 1.0)` could be wrong.** If any SDR color value was exactly 1.0 before the `*= 10.0` operation on even indices, the odd-indexed entries are untouched and should be <= 1.0. But `np.random.rand` generates [0, 1), so max is ~0.9999. The assertion is technically correct but relies on the statistical impossibility of `rand()` returning exactly 1.0.

---

### 9. `test_node_properties.py`

**What it tests:** Node hierarchy properties (`num_children`, `is_leaf`, `is_root`), method chaining (`set_opacity`, `set_gamma`, `set_blending_mode`), equality/hashing, duplicate child name rejection, and `walk()`.

**Rigor: Very Good.** Thorough coverage of the Node class.

**Findings:**

- **LOW - `test_walk_returns_nodes` verifies node identity with `is` comparison.** This is correct behavior but could become fragile if the implementation ever returns copies instead of references.

- **LOW - No test for `set_intensity` or `set_offset` method chaining.** These are newer Node methods (visible in the source) but have no tests in this file or elsewhere in the test suite.

- **MEDIUM - Missing tests for `nd_transform` property.** The Node class has `nd_transform` getter/setter and `world_nd_transform` (lines 290-339 of node.py), but no test file exercises these at all.

---

### 10. `test_node_rendering.py`

**What it tests:** Rendering attributes (opacity, gamma, blending_mode) -- getters, setters, validation, persistence, and type conversion.

**Rigor: Good.** Tests string-to-float conversion for opacity and gamma, persistence to zarr, and validation error messages.

**Findings:**

- **MEDIUM - `test_rendering_attributes_in_add_points` uses `compiler.write_points()` instead of `scene.add_points()`.** Same issue as in `test_hdr_colors.py` -- testing a different API level.

- **MEDIUM - Missing tests for `intensity` and `offset` rendering attributes.** The Node class supports `intensity` (0.0-100.0) and `offset` (-10.0-10.0) properties, but they have zero test coverage in this file.

- **LOW - `test_blending_mode_getter_setter` only tests "normal", "additive", "max" but the source code also supports "opaque" and "luminous".** These newer blending modes are not tested.

---

### 11. `test_physical_units.py`

**What it tests:** Physical unit acceptance in dimensions, invalid unit rejection, config-type consistency, mixed units.

**Rigor: Adequate.** The parametrized test over all supported units is good.

**Findings:**

- **MEDIUM - `test_invalid_unit_rejection` tests `PhysicalUnit.validate()` directly, not through the `Dimension` constructor.** If the Dimension class stops calling validation, this test would still pass but the actual user-facing code path would be broken. The test should also verify that `Dimension("x", unit="invalid_unit")` raises.

- **LOW - Missing test for the empty string unit.** `Dimension("x", unit="")` is the default. Is empty string considered valid? The test doesn't cover this.

---

### 12. `test_review_fixes.py`

**What it tests:** Regression guards for 5 specific fixes from a systematic API review.

**Rigor: Excellent.** These are focused, surgical regression tests that guard specific bugs. Each class is well-documented with the fix it guards.

**Findings:**

- **LOW - `test_gsplatdata_in_all` checks `__all__` but doesn't verify the actual import works.** `test_gsplatdata_importable_from_luxar` does verify `hasattr(luxar, "GSplatData")`, so this is partially redundant.

- **LOW - Cross-scene hash inequality test (`test_same_name_different_scenes_different_hash`) could theoretically fail.** Hash collisions are possible by design, so `assert hash(g1) != hash(g2)` is not guaranteed. However, with the implementation using `id()` of the root node, collisions are astronomically unlikely.

---

### 13. `test_scene_advanced.py`

**What it tests:** Scene initialization errors, input handling, add_lines extend_to_all, add_gsplats validation, _analyze_extend_candidates, dimensions property, to_zarr, get_store_path.

**Rigor: Good.** Covers many of the Scene's internal methods and edge cases.

**Findings:**

- **MEDIUM - Uses `tempfile.TemporaryDirectory()` instead of `tmp_path` fixture.** Most test files use pytest's `tmp_path` fixture, but this file uses `tempfile.TemporaryDirectory()` manually. This is inconsistent and less clean (no automatic cleanup on test failure).

- **HIGH - `test_extend_to_all_invalid_value` tests `extend_to_all="invalid_value"` which is a string.** The source code (Scene._resolve_extend_to_all) checks `extend_to_all == "all"` first, then `isinstance(extend_to_all, list)`, then falls through to the ValueError. But the string `"invalid_value"` would pass the `== "all"` check as False, pass the `isinstance(list)` check as False, and raise ValueError. This is correct, but the test does NOT verify that `extend_to_all="all"` is accepted as valid -- that's tested elsewhere. However, what about `extend_to_all="All"` or `extend_to_all="ALL"` (case sensitivity)? The source code uses `== "all"` (case-sensitive), but no test verifies that "All" or "ALL" is rejected.

- **LOW - `test_dimensions_setter_new_dims` sets dimensions to 2D on a 3D scene but doesn't verify any consequences.** It only checks that the dimensions object was updated, not that subsequent add_points calls validate against the new dimensions.

---

### 14. `test_scene_methods.py`

**What it tests:** Scene string representation, finalization, dimension validation, range warnings, helpful error messages, metadata preservation, writer access, context manager exception handling.

**Rigor: Very Good.** The metadata preservation tests (`test_points_metadata_preservation`) are particularly valuable -- they catch a real constructor ordering bug documented in the codebase.

**Findings:**

- **MEDIUM - `test_dimension_range_warning` uses `warnings.catch_warnings(record=True)` with manual filtering.** This is correct but verbose. Using `pytest.warns()` would be more idiomatic and less error-prone.

- **LOW - `test_scene_context_manager_exception` verifies the zarr store has data after an exception, but doesn't verify the store is in a valid/complete state.** The `.zmetadata` consolidation may not happen on exception paths.

- **LOW - `test_scene_writer_access` tests `scene._writer is compiler` which is testing a private attribute.** This is fragile if the internal implementation changes.

---

### 15. `test_scene_structure.py`

**What it tests:** A roundtrip test of the Lorenz attractor demo.

**Rigor: Adequate.** Tests root attributes, hierarchy, dataset shapes, and chunk sizes.

**Findings:**

- **MEDIUM - This is really an integration test for `create_lorenz_attractor`, not a unit test of Scene structure.** It depends on the demo utility, which is outside `core/`. If the demo function changes its output format, this test breaks even though core is fine.

- **LOW - The test name `test_random_demo_roundtrip` doesn't match the file name `test_scene_structure.py`.** The file is in `core/tests/` but tests a utility from `utils/demos`.

---

### 16. `test_spatial_dimensions.py`

**What it tests:** The `spatial` flag on dimensions -- auto-determination, validation, serialization, compiler integration.

**Rigor: Very Good.** Thorough coverage of the spatial flag logic, including edge cases and auto-correction warnings.

**Findings:**

- **MEDIUM - `test_compiler_stores_spatial_metadata` has a conditional assertion.** Line 244: `if "spatial_extend_dims" in points_node.attrs:` means the test will silently pass even if the attribute is missing. This should be a hard assertion.

- **LOW - No test for spatial flag interaction with `extend_to_all`.** Spatial dimensions affect how data extends through space, but there's no test verifying that `extend_to_all` behaves differently for spatial vs non-spatial dimensions.

---

### 17. `test_transforms.py`

**What it tests:** Transform utilities (identity, translate, scale, rotate, compose, inverse, look_at, to_list/from_list), node integration, world_transform.

**Rigor: Excellent.** One of the best test files. The `test_compose_application_order` test is exemplary -- it verifies transform ordering with multiple cases and explicitly tests that reversed order gives different results.

**Findings:**

- **CRITICAL - The `compose` function's implementation has a subtle correctness concern that tests happen to match.** The docstring says "compose(T1, T2, T3) applies T1 first", and the implementation iterates in reverse doing `result = result @ transform`. Let's trace: starting with I, reversed gives [T3, T2, T1]. After T3: `result = I @ T3 = T3`. After T2: `result = T3 @ T2`. After T1: `result = T3 @ T2 @ T1`. Applied to vector v: `(T3 @ T2 @ T1) @ v = T3(T2(T1(v)))` -- T1 applied first. This is correct. However, the comment in the source says "Right-multiply: result = result @ next_transform" which is confusing because in the reversed loop, the "next transform" is actually the earlier one. The code is correct, but the documentation comment is misleading. The tests thoroughly verify correctness, so this is not a bug, but the confusing comment could lead to future breakage if someone "fixes" it.

- **MEDIUM - `test_look_at` only tests two configurations.** The look_at function has edge cases (eye == target, parallel up and forward vectors) that are not tested. These would likely cause division-by-zero.

- **LOW - `test_to_from_list` doesn't explicitly verify the THREE.js column-major format.** It tests roundtrip correctness, which is good, but doesn't verify that the intermediate list is in the correct column-major order for the viewer.

---

### 18. `test_viewer_config.py`

**What it tests:** `ViewerConfig`, `CameraConfig`, `UIConfig`, `DimensionsConfig`, `AnimationConfig` -- validation, serialization, round-trips, JSON I/O.

**Rigor: Very Good.** Comprehensive coverage of validation boundaries, round-trips, and the snapshot workflow.

**Findings:**

- **LOW - Missing test for `ViewerConfig.validate()` called directly.** The `validate()` method is called in `__post_init__` but can also be called independently (e.g., after mutation). No test calls `validate()` after modifying fields.

- **LOW - `test_from_dict_ignores_unknown_keys` is a good forward-compatibility test.** No issues here.

- **LOW - `CameraConfig.from_dict` uses `tuple(data["position"])` which would fail silently for non-3-element lists.** The validation in `__post_init__` catches this, but it's worth noting the reconstruction path is protected.

---

## Cross-Cutting Findings

### HIGH - Missing Coverage: `nd_transform` Property

The `Node` class has `nd_transform` getter/setter and `world_nd_transform` composition (50+ lines of code in node.py, lines 289-339), plus validation in `validation/nd_transforms.py`. **No test file exercises any of this.** This is a significant gap for a feature documented in the CLAUDE.md spec.

### HIGH - Missing Coverage: `intensity`, `offset`, and `layer` Properties

The `Node` class has `intensity`, `offset`, and `layer` properties with validation, but these have **zero test coverage** across all 18 test files.

### HIGH - Missing Coverage: `colormap` Property and `set_colormap`

The `Node` class has `colormap` getter/setter, `set_colormap` method, and associated validation (lines 568-615 of node.py). No test verifies this, even though colormap is documented as an active feature.

### MEDIUM - Inconsistent API Usage

Some tests use `scene.add_points()` (the Group-inherited API), while others use `compiler.write_points()` (the lower-level writer API). These are different code paths with different validation. Tests in `test_hdr_colors.py` and `test_node_rendering.py` mix both within the same file, which can mask bugs in either layer.

### MEDIUM - Redundancy Between extend_to_all Tests

`test_extend_to_all.py` (points) and `test_gsplats_extend_to_all.py` share ~80% identical logic testing the same `Scene._resolve_extend_to_all` method. `test_scene_advanced.py` also has `TestAddLinesExtendToAll` covering extend_to_all for lines. The shared resolution logic is tested 3 times.

### LOW - No Test for Node `__repr__`

The `Node.__repr__` method (line 653 of node.py) is marked with `# pragma: no cover` and has no test. While repr is cosmetic, it's part of the debugging experience.

---

## Summary of Most Important Findings

| # | Severity | File(s) | Finding | Status |
|---|----------|---------|---------|--------|
| 1 | CRITICAL | `test_transforms.py` / `transforms.py` | `compose()` internal comment is misleading; code is correct but fragile to "helpful" refactoring | UNFIXED |
| 2 | HIGH | (none) | `nd_transform` property has zero test coverage (~50 lines of untested code) | UNFIXED |
| 3 | HIGH | (none) | `intensity`, `offset`, `layer` Node properties have zero test coverage | UNFIXED |
| 4 | HIGH | (none) | `colormap` property and `set_colormap` have zero test coverage | UNFIXED |
| 5 | HIGH | `test_gsplats_extend_to_all.py` | ~150 lines of redundant tests duplicating `test_extend_to_all.py` | UNFIXED |
| 6 | HIGH | `test_hdr_colors.py` | Mixed API usage (`compiler.write_points` vs `scene.add_points`) tests different code paths | UNFIXED |
| 7 | MEDIUM | `test_spatial_dimensions.py` | Conditional assertion silently passes when attribute is missing | UNFIXED |
| 8 | MEDIUM | `test_scene_advanced.py` | Case-sensitivity of `extend_to_all="all"` not tested | UNFIXED |
| 9 | MEDIUM | `test_node_properties.py` | `nd_transform`, `intensity`, `offset` chain methods untested | UNFIXED |
| 10 | MEDIUM | `test_physical_units.py` | Invalid unit tested at wrong abstraction level | UNFIXED |

---

## Recommendations

1. **Add `test_nd_transforms.py`** to cover the nd_transform property, world_nd_transform composition, and validation. This is the largest untested feature.

2. **Add tests for `intensity`, `offset`, `layer`, and `colormap`** Node properties. These can go in `test_node_rendering.py` and `test_node_properties.py`.

3. **Consolidate extend_to_all tests** using a parameterized fixture that tests the shared `_resolve_extend_to_all` logic once, with geometry-specific add methods as parameters.

4. **Fix the conditional assertion** in `test_spatial_dimensions.py` line 244 -- change the `if` to a hard `assert`.

5. **Clarify the `compose()` implementation comment** in transforms.py to prevent future breakage by someone misunderstanding the loop direction.

6. **Standardize API usage in tests** -- prefer `scene.add_points()` over `compiler.write_points()` unless explicitly testing the compiler layer.

---

## Recommended Next Batch

The top 5 remaining issues worth fixing, ordered by impact:

### 1. Add `test_nd_transforms.py` (HIGH -- zero coverage on ~50 lines of shipped code)
**Files:** Create `packages/luxar/src/luxar/core/tests/test_nd_transforms.py`
**Why:** `nd_transform` and `world_nd_transform` are documented features with per-dimension affine/categorical transforms and hierarchical composition. Zero test coverage means regressions will be silent. This is the single largest untested feature in core.
**Effort:** Medium (need to test getter/setter, validation, composition through parent-child hierarchy, and categorical permutation).

### 2. Add tests for `intensity`, `offset`, `layer`, and `colormap` Node properties (HIGH -- zero coverage on active features)
**Files:** Extend `packages/luxar/src/luxar/core/tests/test_node_rendering.py` and `test_node_properties.py`
**Why:** These are user-facing properties with validation ranges (`intensity` 0-100, `offset` -10 to 10) and zarr persistence. `colormap` is an actively developed feature (TODO #14/#15). All have zero coverage.
**Effort:** Low-medium (properties follow the same pattern as `opacity`/`gamma` which are already tested -- can use those as templates).

### 3. Fix the conditional assertion in `test_spatial_dimensions.py` (MEDIUM -- silent test pass)
**File:** `packages/luxar/src/luxar/core/tests/test_spatial_dimensions.py`, line 244
**Why:** The `if "spatial_extend_dims" in points_node.attrs:` guard means the test silently passes if the attribute is never written. Change to `assert "spatial_extend_dims" in points_node.attrs` followed by the value assertion. One-line fix with high safety value.
**Effort:** Trivial.

### 4. Clarify the `compose()` implementation comment (CRITICAL risk -- correct code, misleading docs)
**File:** `packages/luxar/src/luxar/core/transforms.py`, line 271
**Why:** The comment "Right-multiply: result = result @ next_transform" inside a `reversed()` loop is confusing. A well-meaning developer could "fix" the reversal, breaking all transform composition. Adding a clarifying comment explaining why the loop is reversed would prevent this.
**Effort:** Trivial (comment-only change).

### 5. Standardize `test_hdr_colors.py` to use `scene.add_points()` (HIGH -- testing wrong API layer)
**File:** `packages/luxar/src/luxar/core/tests/test_hdr_colors.py`
**Why:** Three tests (`test_hdr_colors_extreme`, `test_negative_color_rejection`, `test_color_channel_count`) use `compiler.write_points()` which bypasses Scene validation. If bugs exist in the Scene-level HDR path, these tests won't catch them. Switch to `scene.add_points()` to test the user-facing API.
**Effort:** Low (change `compiler.write_points(...)` calls to `scene.add_points(...)` with appropriate dimension setup).
