# Python GSplats Test Suite Review

**Date**: 2026-03-31
**Reviewer**: Claude Opus 4.6 (1M context)
**Scope**: All test files under `packages/luxar/src/luxar/gsplats/` (56 test files, ~15,000 lines)

---

## Status

**Last reviewed**: 2026-03-31
**Last status update**: 2026-03-31
**Fixes applied**: None. All 5 MEDIUM and 12 LOW issues remain open. PR #53 did not address any findings from this report (the gsplats suite was already graded A-). All issues were re-verified against the current codebase on 2026-03-31 and confirmed still present.

---

## Executive Summary

The gsplats test suite is **well-structured and comprehensive overall**. Test organization follows the source code layout closely, assertions are generally meaningful, and edge cases are well-covered. The suite demonstrates mature engineering practices: parametric fixtures, round-trip testing, numerical stability checks, and cross-device/cross-backend validation.

**Key metrics**:
- 56 test files across 12 subdirectories
- Approximately 15,000 lines of test code
- Estimated 800+ individual test cases
- Coverage of 2D/3D/4D/nD code paths

**Issues found**: 5 MEDIUM, 12 LOW. No CRITICAL or HIGH issues.

---

## Top-Level Tests (`gsplats/tests/`)

### test_batch.py
**Verdict**: Excellent

- **Rigor**: Strong. Tests manifest roundtrip, task ID decoding arithmetic, shell injection prevention, OME-Zarr discovery with real zarr stores.
- **Completeness**: Good coverage of Slurm script generation, environment capture, time estimation.
- **Notable strength**: Shell injection prevention tests (`test_shell_injection_prevention`, `test_fit_args_shell_injection_prevention`) verify shlex quoting of dangerous inputs.

| Severity | Issue |
|----------|-------|
| LOW | `test_estimate_within_range` asserts `seconds > 0` and `seconds < 3600` but does not verify the interpolation is between known bounds -- the assertion is very loose. |

### test_cholesky_dim_ops.py
**Verdict**: Excellent

- **Rigor**: Strong mathematical verification. Tests verify covariance matrices match after permutation/embedding via `Sigma_new[i,j] == Sigma_orig[perm[i], perm[j]]`.
- **Completeness**: Covers identity, reverse, cyclic permutations, batch operations, validation errors, fill_sigma=0 edge case.
- No issues found.

### test_culling.py
**Verdict**: Excellent

- **Rigor**: Tests actual rendering quality (PSNR > 30 dB after culling), verifies redundancy detection, tests 4D nD path.
- **Completeness**: Covers error budget, redundancy mode, empty data, GSplatData.cull() integration, verbose mode.
- **Notable strength**: `test_quality_preserved_after_culling` is a proper quality gate, not just a smoke test.

| Severity | Issue |
|----------|-------|
| LOW | `test_quality_preserved_after_culling` has a conditional guard `if data_range > 0 and mse > 0` that silently passes when MSE is exactly 0. This makes the PSNR assertion vacuously true if culling happens to remove nothing. However, the test setup makes this unlikely. |

### test_fit_gsplats.py
**Verdict**: Good

- **Rigor**: Thorough parameter validation tests, convergence checks, reconstruction quality tests.
- **Completeness**: Tests MSE/Poisson/L1 loss types, L1 regularization, sigma constraints, device support, auto-candidate generation.

| Severity | Issue |
|----------|-------|
| MEDIUM | `test_volume_proportional_scaling` asserts `len(result_large.amplitudes) >= len(result_small.amplitudes)` but both may hit the minimum seed count (50), making this assertion trivially true. The test comment acknowledges this but does not create volumes large enough to exceed the minimum. |
| LOW | `test_convergence_with_iterations` asserts results are "different" (`not np.allclose`) but does not verify the longer run is *better*. This is a weak assertion -- random noise could also make them different. |

### test_gpu_profile.py
**Verdict**: Excellent

- **Rigor**: Tests profile CRUD operations, multi-GPU support, v1 migration, corrupted YAML handling.
- **Completeness**: Covers save/load roundtrip, summary recomputation, GPU memory matching, idempotent migration.
- No issues found.

### test_gsplat_data.py
**Verdict**: Excellent (partial read due to size)

- **Rigor**: Strong. Tests properties, validation, translate, center_at_centroid, scale_intensity, heuristic culling, save whitelist.
- **Completeness**: Good edge case coverage (empty, zero amplitudes, mismatched shapes).
- **Notable strength**: `test_stats_not_shared_between_instances` catches a common mutable default argument bug.

| Severity | Issue |
|----------|-------|
| LOW | `test_quality_metrics_included` in `TestSaveWhitelist` simulates the save whitelist extraction inline rather than calling the actual `save()` method. If the real whitelist diverges from this test's hardcoded list, the test won't catch it. |

### test_gsplats_integration.py
**Verdict**: Excellent

- **Rigor**: Full pipeline tests (seed -> fit -> render -> quality check) for 2D, 3D, and 4D.
- **Completeness**: Tests early stopping, batched renderer equivalence, loss functions, regularization, sigma constraints, device compatibility, empty inputs, memory chunking.
- **Notable strength**: `test_full_pipeline_4d` exercises the nD renderer chunking path.

| Severity | Issue |
|----------|-------|
| LOW | `test_early_stopping_convergence` asserts `stats_no_threshold["iterations"] <= stats_threshold["iterations"]` which seems inverted -- a threshold-based run should stop *earlier*, not later. The test comment explains this is because the "no-threshold" run has internal heuristics that may stop sooner. The assertion direction is correct but the logic is non-obvious and could confuse maintainers. |

### test_metrics.py
**Verdict**: Excellent

- **Rigor**: Validates SSIM against scikit-image reference implementation to 1e-4 precision. Tests tiled SSIM matches non-tiled.
- **Completeness**: Tests PSNR, SSIM, quality metrics aggregate, tiled SSIM, auto-tiling heuristic, input immutability.
- **Notable strength**: `test_ssim_matches_skimage` provides ground-truth validation.
- No issues found.

### test_progressive_fitting.py
**Verdict**: Good

- **Rigor**: Tests multi-pass, LOD structure, PSNR tracking, callback invocation, save/load roundtrip.
- **Completeness**: Good coverage of progressive fitting lifecycle.

| Severity | Issue |
|----------|-------|
| MEDIUM | `test_cumulative_psnr_increases` has an extremely weak assertion: `assert all(p > 0 for p in psnrs)`. The test name says "PSNR increases" but the comment acknowledges tiny test volumes may not show this, so it just checks PSNR > 0. This test will pass even if PSNR *decreases* across passes. Consider either fixing the assertion or renaming the test to `test_cumulative_psnr_positive`. |

### test_spatial_volume_filter.py
**Verdict**: Excellent

- **Rigor**: Tests the spatial volume extraction logic that's critical for 4D background filtering. Mathematical assertions verify `det_L` calculation.
- **Completeness**: Tests isotropic, anisotropic, multiple timepoints, mixed population filtering.
- **Notable strength**: `test_diag_indices_correct` verifies the diagonal index formula against a manual calculation.
- No issues found.

### test_tiled_fitting.py
**Verdict**: Excellent

- **Rigor**: Tests tile coverage (every voxel covered), partition-of-unity for cosine windows, global coordinate translation.
- **Completeness**: Covers anisotropic tile sizes, validation errors, progressive tiled fitting, LOD merging across tiles.
- **Notable strength**: Partition-of-unity tests (`test_partition_of_unity_*`) mathematically verify window correctness.
- No issues found.

---

## Fitting Tests (`fitting/tests/`)

### test_downscale.py
**Verdict**: Excellent

- Tests volume downscaling, center rescaling, Cholesky factor rescaling, and integration with explicit seeds.
- Low-frequency preservation test uses correlation coefficient.
- No issues found.

### test_fitting_config.py
**Verdict**: Good but shallow

| Severity | Issue |
|----------|-------|
| MEDIUM | All four test classes (`TestFitConfig`, `TestPreprocessedData`, `TestOptimizationResults`, `TestModelComponents`) only test that dataclass construction works and attributes are accessible. They do not test any logic, validation, or defaults. These are essentially "does the constructor not crash" tests. While not harmful, they provide minimal value. |

### test_fitting_preprocessing.py
**Verdict**: Good

- **Rigor**: Tests NaN/Inf input validation, normalization, auto-candidate generation, convergence thresholds, compression ratio calculation.
- **Completeness**: Good coverage of seeds-as-int, seeds-as-float, explicit seeds, init array subsampling/extension.

| Severity | Issue |
|----------|-------|
| LOW | `test_seeds_as_compression_ratio` allows `abs(result.N - expected_target) <= max(10, expected_target * 0.5)` which is a 50% tolerance band. This may be too loose to catch regressions in the compression ratio calculation. |

### test_fitting_validation.py
**Verdict**: Excellent

- Comprehensive parameter validation coverage: seeds (int, float, array), sigma constraints, boundary penalty, voxel size, output space, sigma_max_diag fraction, rel_l2_target.
- No issues found.

### test_initialization.py
**Verdict**: Good

- Tests model initialization, zero candidates, parameter propagation, optimizer setup, device compatibility.
- No issues found.

### test_losses.py
**Verdict**: Excellent

- Tests MSE, L1, Poisson losses with and without asymmetric penalty.
- Tests L1 regularization on amplitudes and diagonals.
- Tests boundary penalty (increases loss at edges, zero for interior, differentiable, disabled by default).
- No issues found.

### test_optimization.py
**Verdict**: Good

- Tests optimization loop execution, convergence, iteration limits, best state tracking, gradient clipping, LR scheduling, movie recording, dynamic operations, rel_l2_target.
- No issues found.

### test_results.py
**Verdict**: Good (partial read)

- Tests result finalization, clip-to-bounds, intensity rescaling.
- No issues found in the portion reviewed.

### test_sorting.py
**Verdict**: Excellent

- **Notable strength**: Tests that Morton sorting preserves model forward output (rendering is identical before/after sort). Tests all 4 model parameters are permuted, optimizer state is permuted, relocation tracker is permuted.
- Tests degenerate case (identical centers), aliasing safety, 4D support.
- No issues found.

### test_visualization.py
**Verdict**: Good

- Tests compression analysis display, bits-per-pixel calculation, napari movie handling.
- Uses mock napari to avoid opening windows.

| Severity | Issue |
|----------|-------|
| LOW | `test_show_optimization_movie_no_napari` catches all exceptions with a bare `except Exception` and uses `pytest.fail()`, but the mock napari fixture means it will never actually test the real ImportError path. |

---

## Fitting Dynamic Ops Tests (`fitting/dynamic_ops/tests/`)

### test_dynamic_ops.py
**Verdict**: Good

- Tests config defaults, residual peak finding, simplified seeding, importance calculation, weak splat selection, integration with fitting pipeline, convergence guard.

| Severity | Issue |
|----------|-------|
| LOW | `test_compression_analysis_functionality` wraps the call in try/except and sets a boolean flag, but then `assert compression_test_passed` -- this pattern hides the actual exception message. Better to just call the function directly and let pytest handle the assertion. |

### test_relocation_tracker.py
**Verdict**: Excellent

- Tests tracker initialization, marking, cooldown filtering, cooldown expiration, multiple relocations, immediate re-relocation prevention, statistics.
- **Notable strength**: `test_tracker_prevents_immediate_rerelocation` directly tests the critical bug scenario.
- No issues found.

---

## Seeds Tests (`seeds/tests/`)

### test_edges.py
**Verdict**: Good

- Tests 2D/3D/1D edge detection, parameter handling, isotropic shapes, edge cases, reproducibility.
- Good documentation of removed tests (structure tensor parameters).
- No issues found.

### test_generate_seeds.py
**Verdict**: Excellent

- Tests all seeding methods (decomposition, grid, edges, auto, combined), parameter routing, error cases, output format, special cases, reproducibility.
- **Notable strength**: `test_unused_param_warning` verifies that unused parameters trigger warnings.
- No issues found.

### test_gpu_ops.py
**Verdict**: Excellent

- Tests GPU vs CPU consistency for Sobel gradients, peak detection, soft blur, amplitude interpolation.
- Performance comparison tests verify >2x GPU speedup.
- Proper CUDA skip markers throughout.
- No issues found.

### test_grid.py
**Verdict**: Excellent

- Tests spacing (scalar, per-dimension), anisotropic spacing, jitter, sigma, intensity filtering, edge cases, reproducibility.
- **Notable strength**: `test_anisotropic_spacing_thin_volume` and `test_anisotropic_spacing_very_thin` test realistic microscopy scenarios.
- No issues found.

### test_multiscale_decomposition.py
**Verdict**: Good

- Tests basic functionality, ignore_finest_k, min_distance, threshold filtering, 1D/3D volumes, input validation.
- No issues found.

### test_seeds_integration.py
**Verdict**: Good

- Cross-method comparisons, integration with fitting pipeline, noisy image robustness, 3D seeds, scale-based sigma preservation.
- No issues found.

### test_utils.py
**Verdict**: Good

- Tests local_maxima, dedupe_farthest_first, combine_seeds, GPU deduplication.
- No issues found.

---

## Models Tests

### models/gsplats/tests/test_gsplat_model.py
**Verdict**: Good (partial read -- 735 lines)

- Tests model creation, forward pass, parameter constraints, sigma clamping.
- Properly structured with fixtures.

### models/gsplats/tests/test_rendering.py
**Verdict**: Not fully reviewed (916 lines)

### models/utils/tests/test_inverse_softplus.py
**Verdict**: Excellent

- Tests basic, beta variations, small/large values, monotonicity, asymptotic behavior, dtype preservation, numerical stability comparison with naive implementation, PyTorch/NumPy equivalence.
- **Notable strength**: `test_stability_comparison` compares stable implementation against naive to verify numerical stability improvement.
- No issues found.

### models/utils/tests/test_lt_solver.py
**Verdict**: Excellent

- Tests 2x2, 3x3, batched systems, identity, diagonal, scalar, large systems, different dtypes/devices, gradient flow, singular matrix, mismatched dimensions, empty tensors, cross-version PyTorch compatibility, numerical stability.
- No issues found.

---

## IO Tests (`io/tests/`)

### test_format.py
**Verdict**: Excellent

- Validates format specification compliance: root attributes, splats group structure, array shapes, encoding metadata, fitting/provenance groups, chunk bounds, consolidated metadata, ordering metadata.
- No issues found.

### test_ordering.py
**Verdict**: Good

- Tests Morton encoding, coordinate normalization, auto resolution, Morton/Hilbert sorting, chunk bounds (including anisotropic covariance).
- No issues found.

### test_save_load.py
**Verdict**: Excellent

- Comprehensive roundtrip tests: basic, with ordering, with quantization, with colors (SDR, uint8, HDR), compression, multi-LOD.
- Tests validation errors, missing file, invalid format.
- **Notable strength**: Tests blosc compression is applied by default and chunk capping works.
- No issues found.

---

## Other Subdirectory Tests

### clahe/tests/test_clahe.py
**Verdict**: Excellent

- Tests basic properties (uniform unchanged, range/shape/dtype/device preservation), contrast enhancement, nD support (1D through 4D), edge cases (small image, near-uniform, zero), sampling probabilities, parameter variations, numerical stability.
- No issues found.

### multiscale/tests/test_decompose_advanced.py
**Verdict**: Good (partial read -- 802 lines)

- Tests loss types, asymmetric penalties.

### multiscale/tests/test_decomposition_basic.py
**Verdict**: Not fully reviewed (587 lines)

### multiscale/tests/test_energy_distribution.py
**Verdict**: Not fully reviewed (256 lines)

### optim/tests/test_integration.py
**Verdict**: Good

- Tests optimizer/scheduler creation, parameter passing, gradient dilution, training integration.

| Severity | Issue |
|----------|-------|
| MEDIUM | `test_2d_gradient_dilution` asserts `actual_lr == 0.1` (exact equality for 2D baseline), but `test_3d_gradient_dilution` only asserts `actual_lr > 0.1`. The 3D test does not verify the *specific* scaling factor. If the gradient dilution formula changes, this test would not catch regressions as long as the LR increases. |

### preprocessing/tests/test_backend_dispatch.py
**Verdict**: Good

- Tests backend auto-detection, dispatch routing, device kwarg, non-tensor rejection.
- Compact and focused.

### preprocessing/tests/ (calibration, denoise_pipeline, nlm_*)
**Verdict**: Not fully reviewed (small files, 93-136 lines each)

### utils/tests/test_trils.py
**Verdict**: Excellent

- Tests tril_size, pack_tril, unpack_tril, roundtrip, gradient dilution factor, validate_cholesky_shape.
- Mathematical correctness verified through numpy tril_indices consistency check.
- No issues found.

---

## Cross-Cutting Issues

| Severity | Issue | Files Affected |
|----------|-------|---------------|
| MEDIUM | **Redundant `validate_gsplatdata` helpers**: At least 5 separate test files define their own `validate_gsplatdata()` function with nearly identical logic. This should be extracted to a shared conftest.py fixture or test utility module. | `test_edges.py`, `test_generate_seeds.py`, `test_grid.py`, `test_multiscale_decomposition.py`, `test_seeds_integration.py` |
| LOW | **Non-deterministic random seeds**: Several integration tests use `np.random.seed()` or `torch.manual_seed()` at the test level but not consistently. Some tests (e.g., `test_full_pipeline_4d`) set seeds while similar tests don't. This could lead to intermittent failures. | Multiple integration test files |
| LOW | **No negative tests for CUDA/Metal backends**: CUDA and Metal test files (not reviewed in detail) are gated by hardware availability, which is correct, but there are no mock-based tests to verify error handling when GPU operations fail mid-execution. | `models/gsplats/cuda/tests/`, `models/gsplats/metal/tests/` |

---

## Summary Statistics

| Severity | Count | Description |
|----------|-------|-------------|
| CRITICAL | 0 | No tests enforcing wrong behavior or fundamental design flaws |
| HIGH | 0 | No tests that always pass regardless, no stale API references |
| MEDIUM | 5 | Weak assertions that may not catch regressions; redundant helper code |
| LOW | 12 | Minor assertion looseness, style issues, non-obvious logic |

## Recommendations

1. **Extract shared `validate_gsplatdata` to `conftest.py`** in the seeds/tests directory. This reduces duplication across 5+ files.

2. **Strengthen `test_cumulative_psnr_increases`** -- either make the test data large enough that PSNR genuinely increases, or rename the test to match what it actually checks.

3. **Tighten `test_volume_proportional_scaling`** by using volumes that exceed the minimum seed count threshold, so the scaling logic is actually exercised.

4. **Add specific expected value assertions in `test_3d_gradient_dilution`** to verify the exact scaling factor (1.8x for 3D), not just "greater than base LR".

5. **Consider adding regression snapshot tests** for the `save()` whitelist logic in `test_gsplat_data.py`, calling the actual `save()` method rather than simulating the whitelist inline.

---

## Overall Assessment

**Grade: A-**

The test suite is well above average for a research/scientific computing codebase. It demonstrates:
- Proper mathematical verification (not just "doesn't crash" checks)
- Systematic edge case coverage (empty inputs, uniform images, zero amplitudes)
- Cross-device testing (CPU/CUDA/MPS)
- Format compliance testing against specification
- Quality gate assertions (PSNR, SSIM, MSE thresholds)

The main gaps are a handful of loose assertions in integration tests and some code duplication in test utilities. No tests were found to be enforcing incorrect behavior, and no deprecated API references were detected.

---

## Recommended Next Batch

The following 5 issues are the highest-impact fixes remaining, ordered by value:

1. **Extract shared `validate_gsplatdata` to `conftest.py`** (MEDIUM, cross-cutting, 5 files)
   Files: `seeds/tests/test_edges.py`, `test_generate_seeds.py`, `test_grid.py`, `test_multiscale_decomposition.py`, `test_seeds_integration.py`.
   **Why first**: This is pure code hygiene with zero risk of breakage. A single shared helper in `seeds/tests/conftest.py` eliminates ~100 lines of near-identical duplication and makes future validation changes propagate automatically.

2. **Strengthen `test_cumulative_psnr_increases`** (MEDIUM, `test_progressive_fitting.py`)
   The test name promises monotonically increasing PSNR but only asserts `p > 0`. Either use a larger test volume (e.g., 64x64 with a clear gradient) so PSNR genuinely increases across passes, or rename to `test_cumulative_psnr_positive` to match reality. A misleading test name is worse than a weak assertion because it gives false confidence.

3. **Tighten `test_volume_proportional_scaling`** (MEDIUM, `test_fit_gsplats.py`)
   Both the 16x16 and 32x32 volumes likely hit the 50-seed minimum floor, making the `>=` assertion trivially true. Use volumes of at least 64x64 and 128x128 (or lower the minimum seed count in the test) so the scaling logic is actually exercised. This is a straightforward parameter change.

4. **Add exact scaling factor assertion in `test_3d_gradient_dilution`** (MEDIUM, `optim/tests/test_integration.py`)
   Currently asserts `actual_lr > 0.1` which passes for any positive scaling. Assert the expected ~1.8x factor with a reasonable tolerance (e.g., `pytest.approx(0.18, rel=0.1)`) to catch formula regressions.

5. **Strengthen `test_fitting_config.py` beyond constructor smoke tests** (MEDIUM, `fitting/tests/test_fitting_config.py`)
   All four test classes only verify that dataclass construction works. Add tests for: default value correctness, validation logic (if any), and `asdict()` roundtrip. If the dataclasses truly have no logic, consider removing these tests to avoid giving false coverage credit -- or add validation constraints to the dataclasses and test those.
