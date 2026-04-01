# Code Review: Python Encoding & IO Test Suites

**Reviewer**: Claude Opus 4.6 (1M context)
**Date**: 2026-03-31
**Scope**: `packages/luxar/src/luxar/encoding/tests/` (6 files) and `packages/luxar/src/luxar/io/tests/` (12 files)

---

## Status (Updated 2026-03-31)

The following issues from the original review were **fixed in PR #53**:

| Fixed Issue | Original Severity | File | Details |
|-------------|-------------------|------|---------|
| `test_positions_and_colors` missing value assertions | HIGH | `test_roundtrip.py` | Now uses `np.lexsort` to match points and `np.testing.assert_allclose` with `atol=2.0/255` for colors. |
| `test_full_point_attributes` only checks shapes | HIGH | `test_roundtrip.py` | Now verifies positions, colors, radii, and sharpness values with appropriate tolerances (e.g., `atol=0.15` for uint8-quantized sharpness). |
| `test_nested_groups_no_duplicates` used incorrect API and asserted wrong behavior | CRITICAL | `test_reader_nodes.py` | Rewritten to use full paths (`"GroupA/GroupAPoints"`) instead of `parent=` kwarg. Assertions now check full hierarchical names. Misleading "writer bug" comments removed. |
| `test_compiler_with_memory_config` flaky due to unseeded RNG | MEDIUM (from cross-cutting #3) | `test_dtype_support.py` | Now uses `np.random.RandomState(42)` with a comment explaining the uint8/uint16 threshold sensitivity. |

**Remaining issue counts:**

| Severity | Original | Fixed | Remaining |
|----------|----------|-------|-----------|
| CRITICAL | 1 | 1 | 0 |
| HIGH | 5 | 2 | 3 |
| MEDIUM | 11 | 1 | 10 |
| LOW | 8 | 0 | 8 |

---

## Executive Summary

The test suites are generally **well-structured and thorough**, with meaningful assertions, good edge case coverage, and clear separation of concerns. The encoding tests are particularly strong, with excellent boundary testing and real-world scenario validation (e.g., gsplat amplitudes). However, several issues ranging from medium to high severity were identified, primarily around missing coverage, weak assertions that could mask regressions, and a few tests that enforce potentially incorrect behavior.

**Overall quality**: 7.5/10 (up from initial review, after PR #53 fixes)

---

## Encoding Tests

### test_decoder.py

**Purpose**: Tests ArrayDecoder class across all encoding types.

| Finding | Severity | Description |
|---------|----------|-------------|
| Missing LUT uint16 decoding test | MEDIUM | Only `lut_uint8` is tested. The decoder code path for larger LUTs (hypothetical `lut_uint16`) is untested. Currently the encoder caps at 256 unique values, so this is a future concern. |
| Missing log_scalar_uint16 decode test | MEDIUM | `test_decode_log_scalar_uint8` exists but no test for `log_scalar_uint16`, which is a separate code path in the decoder (same `_decode_log_scalar` method, but different `bits` value). |
| `test_decode_broadcasted_2d` weak assertion | LOW | Line 49: `assert np.allclose(decoded[0], [1.0, 0.5, 0.0])` -- the tolerance for `allclose` is default `atol=1e-8`, but SDR colors go through uint8 quantization. This test passes because broadcasting happens before color quantization, but the assertion is tighter than it needs to be and fragile to encoding changes. |
| No test for `rgb_uint16` decoding | MEDIUM | The decoder has an explicit branch for `rgb_uint16` but no test covers it. |
| No test for unknown encoding name | LOW | The decoder's `else` fallback (passthrough) is tested via `test_decode_float32_passthrough` and `test_decode_with_none_encoding`, but there is no test for an encoding name like `"future_encoding"` to verify graceful fallback. |

**Strengths**: Good array_ref recursive decode test. Error handling tests for missing target and missing zarr_root are solid.

---

### test_dynamic_range.py

**Purpose**: Tests dynamic range-based dtype selection.

| Finding | Severity | Description |
|---------|----------|-------------|
| Tests access private method `_compute_quantization_bits` | LOW | 13 tests directly call `encoder._compute_quantization_bits()`. While this tests the unit thoroughly, it is coupled to the internal API. If the method is renamed or refactored, all tests break. Acceptable given the critical nature of this logic. |
| `TestQuantizationErrorBounds.test_uint8_absolute_error` manual decoding | MEDIUM | Lines 240-252: The test manually decodes as `arr / 255.0 * max_val`, but this does NOT match the actual decoder logic in `_decode_bounded_scalar`, which uses `normalized * (max_val - min_val) + min_val`. For this specific test, `min_val` is 0 (positive scalar encoding maps [0, max]), so it happens to work. But if the encoding semantics change (e.g., bounded scalar with nonzero min), this manual decode would silently diverge from the real decoder. Should use `ArrayDecoder.decode()` instead. |
| No test for negative values in dynamic range | LOW | `_compute_quantization_bits` is documented for non-negative data, but there is no test verifying behavior when accidentally called with negative data. |

**Strengths**: Excellent boundary testing at 256 and 65536 thresholds. The gsplat amplitude scenario test (`TestGSplatAmplitudeScenario`) is a great example of testing a real-world failure mode. Edge cases (all zeros, single value, uniform data) are well covered.

---

### test_edge_cases.py

**Purpose**: Tests error paths, boundary conditions, unusual inputs.

| Finding | Severity | Description |
|---------|----------|-------------|
| `test_hdr_color_invalid_mode` is misleadingly named | LOW | The test name says "invalid mode" but actually tests a **valid** usage (CUSTOM mode with HDR). The test verifies the happy path, not an error path. |
| No test for BOUNDED_SCALAR with negative bounds | MEDIUM | The encoder likely supports BOUNDED_SCALAR with bounds like `(-10.0, 10.0)`. No test covers this. |
| `test_cholesky_memory_mode` and `test_unit_vector_memory_mode` are near-identical patterns | LOW | These four tests (two per type, with/without float16) follow the exact same pattern. Not a bug, but could be parametrized. |
| Missing test for COORDINATE with NaN | LOW | COORDINATE NaN validation is tested in test_encoder.py but not in edge cases. Not a gap per se, just noting the split. |

**Strengths**: Good coverage of all custom encoder types. The float16 toggle tests (`float16_allowed=True/False`) are important and well-done. The degenerate range test (bounds `(5.0, 5.0)`) correctly verifies broadcast fallback.

---

### test_encoder.py

**Purpose**: Tests ArrayEncoder class across all semantic types and modes.

| Finding | Severity | Description |
|---------|----------|-------------|
| `test_lut_not_used_for_uint8` -- assertion could be stronger | LOW | Line 147: Only checks `enc["name"] != "lut_uint8"`. Should also verify the actual encoding used (e.g., `assert enc["name"] == "uint8"`). |
| `test_non_uniform_not_broadcasted` -- weak assertion | MEDIUM | Line 68: Only checks `enc["name"] != "broadcasted"` for a 3-element array `[1.0, 2.0, 3.0]`. This small array will likely become a LUT. Test should assert the specific encoding used, not just "not broadcasted". |
| No test for CHOLESKY encoding in AUTO mode | MEDIUM | Only MEMORY mode is tested for CHOLESKY (in test_edge_cases.py). AUTO mode behavior is untested. |
| No test for very large arrays | LOW | All arrays are small (100-1000 elements). No test exercises the two-stage hashing path in the registry for large arrays through the full encoder pipeline. |

**Strengths**: Good input validation tests (NaN, Inf, negative colors/scalars). Broadcasting, LUT, array reference, and custom mode tests are comprehensive.

---

### test_registry.py

**Purpose**: Tests ArrayRefRegistry for deduplication.

| Finding | Severity | Description |
|---------|----------|-------------|
| `test_same_bytes_different_shape_is_duplicate` -- documents potentially surprising behavior | MEDIUM | The test explicitly documents that arrays with same bytes but different shapes ARE considered duplicates. The docstring explains the rationale (original_shape is stored in metadata). However, this behavior is fragile: if a future change to the encoder stops storing `original_shape`, data corruption could result. The test should be marked as a regression guard. |
| No test for hash collision handling in large array path | MEDIUM | The registry has collision handling in the large array path (lines 100-105 of registry.py). No test creates a quick-key collision to verify this code path. Creating a synthetic collision would require finding two arrays with same dtype, shape, and first-32KB hash but different full content -- difficult but possible. |
| No test for the `_full_map` fallback in the small array path | MEDIUM | Lines 121-127 and 137-143 of registry.py handle the case where a small array's simple_key doesn't match but its full hash is already in `_full_map`. This code path is untested. |

**Strengths**: Clear, well-organized tests. The lifecycle test (clear/reset) is good. The `test_different_dtype_not_duplicate` test is important for correctness.

---

### test_scalar_input.py

**Purpose**: Tests scalar input support (v0.6.0 feature).

| Finding | Severity | Description |
|---------|----------|-------------|
| `test_no_memory_allocation` is a conceptual test, not a real memory test | LOW | The test verifies `radii.nbytes == 4` (storage size), not actual Python memory allocation. The docstring claims "no intermediate arrays" but this is not actually verified. Would need memory profiling or mocking `np.full` to truly test this. |
| Integration test accesses private attributes | LOW | Lines 29-30: `assert scene._writer == compiler` -- tests internal implementation detail. If the scene's writer reference is refactored, this test breaks unnecessarily. |
| Excellent error handling coverage | -- | Every error path (missing n_elements, wrong type, wrong length, mismatched n_elements, non-uniform with n_elements) is tested. |

**Strengths**: Thorough consistency tests (scalar vs. array produce identical results). Integration tests with the full compiler pipeline are valuable. All semantic types are tested with scalar inputs.

---

## IO Tests

### test_compiler_colormap.py

**Purpose**: Tests colormap support across points, gsplats, and lines.

| Finding | Severity | Description |
|---------|----------|-------------|
| Cholesky factor construction is repetitive | LOW | The same 6 lines of cholesky packing code appear 6 times in this file. Should be extracted to a helper/fixture. |
| `test_gsplats_invalid_colormap_name_raises` -- no matplotlib fallback test | MEDIUM | The test checks that a completely invalid name raises. But there is no test for the boundary case: a name that is not a built-in colormap AND matplotlib is not installed. The `pytest.importorskip("matplotlib")` in `test_points_matplotlib_colormap_stored_as_custom` suggests matplotlib is optional, but the error path when matplotlib is missing and a non-built-in name is given is untested. |
| `test_points_scalars_aligned_after_spatial_ordering` -- tolerance may be too loose | LOW | `atol=0.05` is used for comparing scalar alignment. Given that scalars go through quantization, this tolerance should be validated against the actual quantization error bound for BOUNDED_SCALAR uint8, which is `range / 255 / 2`. For scalars in [0, 3] range, max error is ~0.006, so 0.05 is conservative but safe. |

**Strengths**: Excellent regression tests for the scalars reordering bug. The `TestScalarsReorderWithSpatialOrdering` class is a model for regression testing -- it creates a known correlation (scalar = sum of positions) and verifies it survives reordering.

---

### test_compiler_improvements.py

**Purpose**: Tests version, chunking, transforms, spatial ordering, validation, HDR, empty datasets, position bounds.

| Finding | Severity | Description |
|---------|----------|-------------|
| `test_hdr_colors_warning` -- fragile warning match | MEDIUM | `pytest.warns(UserWarning, match="HDR colors")` -- if the warning message text changes, this test silently stops catching it. Consider also asserting on the maximum color value in the warning. |
| `test_bounds_with_spatial_ordering` uses `np.random.seed(42)` | LOW | Global seed usage can cause flakiness if tests are reordered. Should use `np.random.RandomState(42)` or `np.random.default_rng(42)` instead. |
| `test_negative_colors_rejected` -- checks ValueError, not ValidationError | MEDIUM | The docstring says "Scene wraps ValidationError in ValueError", which is an implementation detail. If the wrapping changes, the test fails for the wrong reason. Should test for the base `ValidationError` or document why `ValueError` is expected. |
| `TestChunkBoundsZarrAlignment` -- excellent regression suite | -- | These tests are thorough and verify a critical invariant (zarr chunk alignment). |

**Strengths**: This is a well-organized omnibus test file. The `_calculate_intelligent_chunks` unit tests are good for testing internals directly. Transform conversion tests verify the critical NumPy-to-THREE.js matrix format conversion.

---

### test_compiler_integration.py

**Purpose**: Integration tests for progressive writing API.

| Finding | Severity | Description |
|---------|----------|-------------|
| `test_memory_efficiency` doesn't actually test memory | HIGH | The test writes 10 arrays of 1M points each and checks `"positions" not in metadata`. This verifies the API contract but does NOT verify memory usage. If the compiler internally cached all arrays, this test would still pass. A proper memory test would use `tracemalloc` or similar. The test name is misleading. |
| `test_error_handling_in_context` -- incomplete verification | MEDIUM | The test verifies the file exists after an error, but doesn't verify the file is in a valid/usable state. A follow-up assertion like `zarr.open_group(output_path, mode="r")` would verify the store is at least openable. |
| `test_hierarchical_scene_with_transforms` -- no value verification | MEDIUM | Lines 56-62: Checks that transform attributes exist but doesn't verify the actual transform values. This test would pass even if all transforms were identity matrices. |

**Strengths**: Good coverage of the context manager pattern. The test for `write_lines` rendering attribute defaults is a useful regression guard.

---

### test_compiler_nd_bounds.py

**Purpose**: Tests nD transform bounds expansion in compiler finalization.

| Finding | Severity | Description |
|---------|----------|-------------|
| No test for categorical dimension bounds | MEDIUM | The `nd_transform` spec supports categorical permutations, but all tests only cover affine transforms (`scale` + `offset`). No test verifies bounds expansion with categorical dimensions. |
| `test_bounds_expansion_hierarchical` -- limited depth | LOW | Only tests parent -> child (depth 2). No test for deeper nesting (grandparent -> parent -> child) to verify recursive nd_transform composition. |

**Strengths**: Excellent coverage of the affine transform case. The negative scale test is particularly good -- it verifies min/max are correctly flipped. The multi-node union test verifies correct scene-level bounds aggregation.

---

### test_io_metadata.py

**Purpose**: Tests compressor and format metadata.

| Finding | Severity | Description |
|---------|----------|-------------|
| Extremely minimal test file | HIGH | This entire file is ONE test function (19 lines). It only verifies that the Lorenz attractor demo writes zarr v2 with Blosc compression. This provides almost zero coverage of the IO metadata system. Missing: version metadata, dimension metadata, node type metadata, encoding metadata verification, custom compressor support. |
| Depends on demo function | MEDIUM | Uses `create_lorenz_attractor`, coupling this IO test to the demos module. If the demo changes, this test may break for unrelated reasons. |

**Strengths**: At least verifies the compression format is correct.

---

### test_ordering_lines.py

**Purpose**: Tests lines spatial indexing with dual ordering.

| Finding | Severity | Description |
|---------|----------|-------------|
| `test_order_lines_preserves_connectivity` -- weak assertion | HIGH | Lines 252-259: The test claims to verify connectivity preservation but only checks that indices are in valid range (`0 <= v_idx < 4`). It does NOT verify that the connectivity graph is isomorphic to the original. A proper test would verify that the set of edges (as unordered pairs of positions) is preserved after reordering. The comment on line 258 acknowledges this: "We can't check exact connectivity" -- but we actually CAN by comparing edge position-pairs. |
| No test for non-3D line ordering | MEDIUM | All integration tests use 3D lines. The dual ordering code supports arbitrary dimensions, but no test exercises 4D+ lines end-to-end. |

**Strengths**: Excellent unit tests for `convert_to_indexed` (all line types). The segment chunk bounds with width expansion test is thorough. The discrete dimension handling test for segments is important.

---

### test_ordering_points.py

**Purpose**: Tests chunk bounds computation for points with discrete dimensions.

| Finding | Severity | Description |
|---------|----------|-------------|
| No test for the actual Morton/Hilbert sorting of points | MEDIUM | This file only tests `compute_chunk_bounds_points`. The actual point sorting (`sort_points_spatial` or equivalent) is not directly tested here. It is tested indirectly through integration tests in test_compiler_improvements.py. |
| `test_discrete_chunks_dont_overlap` -- gap assertion is loose | LOW | Line 318: `gap >= -0.5` allows chunks to overlap by up to 0.5 in the discrete dimension. This is by design (discrete tolerance is +-0.5), but the comment on line 314-315 could be clearer about why this overlap is acceptable. |

**Strengths**: This is an excellent, focused test file. The `TestChunkBoundsDiscreteFiltering` class is particularly strong -- it simulates actual queries to verify that discrete dimension filtering works correctly. The scalar radii support test is a good edge case.

---

### test_progressive_writing.py

**Purpose**: Tests progressive writing architecture.

| Finding | Severity | Description |
|---------|----------|-------------|
| Accesses private attributes extensively | MEDIUM | Lines 28-30: `compiler.store_path`, `compiler.store`, `compiler._is_finalized`, `scene._writer`, `scene._dimensions` -- all private. These tests are tightly coupled to implementation details. |
| `test_no_memory_accumulation` -- same issue as integration test | MEDIUM | Checks `"positions" not in metadata` and `not hasattr(scene, "_positions")`, but doesn't actually measure memory. |
| `.zmetadata` check assumes zarr consolidation | LOW | Line 38: `assert ".zmetadata" in store.store` -- this is zarr v2 specific. If the project migrates to zarr v3, this assertion will need updating. |

**Strengths**: Good test for resizable dataset creation (streaming use case). The context manager test verifies finalization on exit.

---

### test_roundtrip.py

**Purpose**: Comprehensive round-trip tests for the Luxar zarr format.

| Finding | Severity | Description |
|---------|----------|-------------|
| `test_positions_only` -- sorts before comparing | MEDIUM | Lines 53-54: The test sorts both arrays before comparison to account for spatial reordering. This is correct but means the test cannot detect if spatial reordering corrupts individual point coordinates (e.g., swapping x and y). A better approach would be to match points by nearest-neighbor. |
| No round-trip test for lines | MEDIUM | The file has extensive point round-trip tests but zero line round-trip tests. Lines have a more complex encoding (dual ordering, segments, widths) that would benefit from round-trip verification. |
| No round-trip test for gsplats | MEDIUM | Same as above -- no gsplat round-trip tests despite gsplats being a first-class geometry type. |
| `test_hdr_colors_preserved` uses `enable_spatial_index=False` | LOW | This avoids the sorting issue but means HDR colors are not tested through the spatial ordering path. |

**Strengths**: Good API test coverage (`TestSceneAPI` class). Error handling tests are solid (file not found, not a scene, wrong type, nonexistent node). Dimension round-trip tests cover 3D, 5D, and categorical dimensions.

---

### test_writer_parent_parameter.py

**Purpose**: Guards against regression in hierarchy creation.

| Finding | Severity | Description |
|---------|----------|-------------|
| Uses emoji in output | LOW | Line 75: `aprint("\n... All nodes are in correct hierarchy!")` contains emoji. |
| Good documentation of investigation results | -- | The file header clearly documents that the "writer bug" was actually incorrect API usage. |

**Strengths**: Clear, focused tests. Both hierarchy creation and reader reflection are tested. The investigation result is well-documented.

---

### test_zarr_nd_chunking.py

**Purpose**: Tests nD zarr data handling and chunking optimization.

| Finding | Severity | Description |
|---------|----------|-------------|
| `test_4d_data_chunking_optimization` creates 1M points | LOW | This test creates 1M points (1000 timesteps * 1000 points), which may be slow in CI. Consider reducing to 100 timesteps. |
| `test_no_hardcoded_dimensions` -- uses `Dimensions.default_3d()` for non-3D data | MEDIUM | Lines 148-149: Creates 2D, 4D, 7D, 10D positions but always passes `Dimensions.default_3d()`. This means the dimension metadata doesn't match the actual data dimensionality. The test verifies data integrity but with inconsistent metadata. |
| `test_sparse_data_efficiency` doesn't actually test efficiency | MEDIUM | The test verifies data integrity but doesn't measure whether sparse data is stored efficiently (e.g., checking disk size or chunk access patterns). |

**Strengths**: Good variety of dimension counts tested. The cache interaction test (`test_optimal_chunk_cache_interaction`) verifies reasonable chunk sizes.

---

## Cross-Cutting Issues

### 1. Redundancy Between test_encoder.py and test_dynamic_range.py (LOW)

Both files test POSITIVE_SCALAR and BOUNDED_SCALAR with dynamic range selection. `test_dynamic_range.py` focuses on `_compute_quantization_bits` directly, while `test_encoder.py` tests through the `encode()` API. There is meaningful overlap in:
- `TestPositiveScalarDynamicRange` in test_dynamic_range.py vs. `TestPositiveScalarEncodingDynamicRange` in test_encoder.py
- `TestBoundedScalarDynamicRange` in test_dynamic_range.py vs. `TestBoundedScalarEncoding` in test_encoder.py

This redundancy is acceptable since they test at different abstraction levels.

### 2. No End-to-End Encoder-Decoder Test File (HIGH)

There is no dedicated file that tests `ArrayEncoder.encode()` followed by `ArrayDecoder.decode()` for every encoding type in a single pipeline. test_decoder.py does encode-then-decode, but test_encoder.py only checks the encoded zarr state. A comprehensive encode-decode round-trip test for each encoding type would catch mismatches between encoder output and decoder expectations.

### 3. Random Seeds and Test Determinism (MEDIUM)

Most tests use `np.random.rand()` or `np.random.randn()` without seeds. While this provides broader coverage through randomization, it can cause intermittent failures. The encoding tests are especially sensitive because quantization boundaries depend on the exact data distribution. One file was fixed in PR #53 (`test_dtype_support.py`), but the pattern remains widespread. Two files use `np.random.seed(42)` (test_dynamic_range.py line 133, test_compiler_improvements.py line 807), but most do not.

### 4. Missing Coverage: UNIT_VECTOR in AUTO mode (MEDIUM)

`SemanticType.UNIT_VECTOR` is tested only in MEMORY mode (test_edge_cases.py). No test covers AUTO mode, which is the default and most common usage.

### 5. Missing Coverage: `EncodingMode.MEMORY` for POSITIVE_SCALAR and BOUNDED_SCALAR (MEDIUM)

MEMORY mode is tested for COORDINATE, CHOLESKY, and UNIT_VECTOR, but not for POSITIVE_SCALAR or BOUNDED_SCALAR. These types have different encoding paths in MEMORY mode vs. AUTO mode.

---

## Recommendations (Priority Order)

1. **HIGH**: Create a dedicated encode-decode round-trip test that covers every encoding type (`bounded_scalar_uint8`, `bounded_scalar_uint16`, `log_scalar_uint8`, `log_scalar_uint16`, `rgb_uint8`, `rgb_uint16`, `lut_uint8`, `broadcasted`, `array_ref`, `float32`, `float16`, `none`).

2. **HIGH**: Fix `test_ordering_lines.py::test_order_lines_preserves_connectivity` to actually verify edge connectivity preservation, not just index range validity.

3. **HIGH**: Expand `test_io_metadata.py` beyond a single test. It should cover dimension metadata, encoding metadata, version metadata, and node type metadata.

4. **MEDIUM**: Add round-trip tests for Lines and GSplats geometry types.

5. **MEDIUM**: Test `_full_map` fallback paths in `ArrayRefRegistry` (small arrays with same dtype/shape but different content, then same content with different dtype/shape).

6. **MEDIUM**: Add UNIT_VECTOR and CHOLESKY tests for AUTO mode.

7. **MEDIUM**: Fix `test_zarr_nd_chunking.py::test_no_hardcoded_dimensions` to use matching Dimensions objects for non-3D data.

8. **LOW**: Extract cholesky factor construction helper in test_compiler_colormap.py to reduce repetition.

---

## Recommended Next Batch

The following 5 issues represent the highest-impact remaining work, ordered by value:

### 1. Create encode-decode round-trip test suite (HIGH -- Cross-Cutting #2)
**Why**: This is the single highest-leverage addition. The encoder and decoder are tested separately, but no test verifies they agree on every encoding type end-to-end. A mismatch (e.g., encoder writes `log_scalar_uint16` but decoder expects different normalization) would be invisible to current tests.
**Effort**: Medium. One new test file with a parametrized test over all ~12 encoding types.
**File**: New `packages/luxar/src/luxar/encoding/tests/test_encode_decode_roundtrip.py`

### 2. Fix line connectivity assertion (HIGH -- test_ordering_lines.py)
**Why**: The test claims to verify a critical invariant (spatial reordering preserves line connectivity) but actually only checks index bounds. A bug that scrambles connectivity would pass silently. This is a correctness-critical code path.
**Effort**: Low. Replace the range check with an edge-set comparison (extract position pairs from segments, compare as sets of frozensets).
**File**: `packages/luxar/src/luxar/io/tests/test_ordering_lines.py`

### 3. Expand test_io_metadata.py (HIGH)
**Why**: One 19-line test for the entire IO metadata system is a coverage gap. Version, dimension, encoding, and node-type metadata are all load-bearing for cross-version compatibility.
**Effort**: Medium. Add 4-5 focused tests covering each metadata category.
**File**: `packages/luxar/src/luxar/io/tests/test_io_metadata.py`

### 4. Add Lines and GSplats round-trip tests (MEDIUM -- test_roundtrip.py)
**Why**: Lines (dual ordering, segments, widths) and GSplats (cholesky factors, amplitudes) are first-class geometry types with complex encoding, yet have zero round-trip coverage. Points are well-tested after PR #53, but the other two types are not.
**Effort**: Medium. Two new test classes in `test_roundtrip.py`, following the existing Points pattern.
**File**: `packages/luxar/src/luxar/io/tests/test_roundtrip.py`

### 5. Test ArrayRefRegistry collision and fallback paths (MEDIUM -- test_registry.py)
**Why**: The registry's `_full_map` fallback and hash collision handling are untested code paths that handle data deduplication correctness. A bug here could cause silent data corruption (wrong array referenced) or unnecessary duplication (wasted storage).
**Effort**: Low-Medium. Create synthetic arrays that trigger the small-array `_full_map` fallback path.
**File**: `packages/luxar/src/luxar/encoding/tests/test_registry.py`
