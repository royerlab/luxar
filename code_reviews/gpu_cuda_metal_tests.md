# GPU Backend Test Suite Review: CUDA and Metal

**Reviewed by:** Claude Opus 4.6
**Date:** 2026-03-31
**Scope:** All test files in `cuda/tests/` (10 files) and `metal/tests/` (6 files)
**Source files reviewed:** `gsplat_model_cuda.py`, `gsplat_model_metal.py`

---

## Executive Summary

The GPU test suites are extensive and well-structured. The CUDA suite is significantly more mature, covering 2D-8D forward/backward, FP16, performance benchmarks, edge cases, and review-fix regression tests. The Metal suite is solid for 3D-only scope but lighter on numerical rigor. Key findings include: overly loose backward-pass tolerances that could mask real regressions, missing `torch.autograd.gradcheck` for CUDA (the most rigorous correctness check), dead code in numerical tests, redundant test coverage between files, and several performance tests that don't assert meaningful thresholds.

**Severity Legend:**
- **CRITICAL** -- Could mask real bugs or cause false passes
- **HIGH** -- Significant gap in coverage or correctness
- **MEDIUM** -- Improvement needed but not blocking
- **LOW** -- Minor cleanup or style issue

---

## CUDA Test Suite

### 1. `conftest.py` -- Shared Fixtures and Tolerances

**Severity: MEDIUM**

**Strengths:**
- Centralized `Tolerances` class is excellent -- single source of truth for numerical thresholds.
- `compute_L_row_norms` helper avoids duplication.
- `set_random_seed` autouse fixture ensures reproducibility.
- Good parameterized `splat_params_nd` fixture covering 2D/3D/4D.

**Issues:**

1. **`BACKWARD_SIGN_MATCH = 0.70` is too loose (MEDIUM).** A 30% sign disagreement rate between CUDA and reference gradients is dangerously permissive. For a correct backward kernel, sign agreement on non-negligible gradients should be >90%. At 70%, you could have a sign-flipped dimension and still pass. Recommend: tighten to 0.85 and add per-parameter thresholds (amplitude gradients should match better than center gradients).

2. **`COMPARISON_MAX_REL_DIFF = 0.15` is lenient for same-algorithm comparison (MEDIUM).** 15% max relative difference between CUDA and PyTorch for the same mathematical operation suggests a real discrepancy, not just floating-point noise. FP32 should agree to within ~1e-5 for identical algorithms. This tolerance is appropriately loose for forward pass (different code paths, tiling vs sequential), but the name doesn't distinguish the two cases. Recommend: add `FORWARD_SAME_ALGO_MAX_REL_DIFF = 0.01` for optimized-vs-generic path tests.

3. **`splat_params_nd` only covers 2D/3D/4D (LOW).** The CUDA backend supports up to 8D, but the parameterized fixture stops at 4D. While dedicated 7D/8D tests exist elsewhere, this fixture should match the supported range for completeness. At minimum, add 5D.

4. **`reference_model_factory` fixture uses `cuda_device` by default (LOW).** The docstring says "PyTorch reference models" but the default device is CUDA. For a true reference comparison, CPU should be the default to avoid comparing two CUDA code paths.

---

### 2. `test_cuda_forward.py` -- Forward Pass

**Severity: MEDIUM**

**Strengths:**
- Tests both 2D and 3D forward against PyTorch CPU reference.
- `TestCUDAKernelActivation` verifies the CUDA code path is actually taken (not silently falling back to PyTorch) -- excellent guard.
- Checks both correlation and relative difference metrics.

**Issues:**

1. **`test_cuda_forward_faster_than_cpu` uses `time.time()` instead of CUDA events (MEDIUM).** `time.time()` includes Python overhead and is subject to OS scheduling jitter. For GPU benchmarks, `torch.cuda.Event` with `elapsed_time()` is the standard approach for accurate measurement. The test also doesn't account for CPU memory allocation overhead in the first iteration.

2. **`test_cuda_kernels_return_different_from_fallback_timing` has a warning-only assertion (MEDIUM).** Line 463: `if custom_cuda_time > pytorch_cuda_time * 0.9: print("WARNING: ...")`. This should be `pytest.warns()` or at least a soft assertion. As-is, it silently passes even when CUDA kernels are slower than PyTorch, which defeats the purpose.

3. **Missing 4D+ forward tests in this file (LOW).** Forward tests only cover 2D and 3D. The 4D+ tests exist in `test_cuda_nd.py` and `test_cuda_comparison.py`, so this is a structural issue rather than a coverage gap.

4. **`test_forward_matches_cpu_3d` local tolerance (0.1) conflicts with `Tolerances.COMPARISON_MAX_REL_DIFF` (0.15) (LOW).** Line 195 uses a hardcoded `0.1` instead of the centralized constant. Either use the constant or document why this test needs a tighter tolerance.

---

### 3. `test_cuda_backward.py` -- Backward Pass

**Severity: HIGH**

**Strengths:**
- `TestOptimizedVsGenericPath` is a first-class testing strategy -- embedding 2D into 4D to cross-validate specialized vs generic code paths.
- Tests both isotropic and anisotropic L matrices.
- `_assert_paths_match` helper provides good diagnostic output.

**Issues:**

1. **`test_gradient_values_match_cpu` only checks amplitude gradients (HIGH).** Center gradients (`raw_mu`) and Cholesky factor gradients (`raw_L_diag`, `L_off`) are not compared. Since the backward kernel computes `d_centers`, `d_conic`, and `d_amps` in CUDA and then chains `d_conic -> d_Ls` in PyTorch, the center gradients are the most CUDA-specific and most likely to have bugs. This is a significant gap.

2. **Sign-match threshold of 0.70 for backward (HIGH).** As noted in conftest review, this is too loose. A backward kernel with 30% sign errors would not converge during optimization. The `test_gradient_numerical_accuracy_vs_reference` in `test_cuda_gradcheck.py` uses correlation > 0.5 which is even more permissive.

3. **No backward pass test for 2D (MEDIUM).** `test_gradcheck_3d` only tests 3D. While `test_cuda_gradcheck.py` covers 2D, this file's class `TestCUDABackward` should have parity.

4. **`_embed_2d_to_4d` uses fixed dummy variance of 0.1 (MEDIUM).** The 0.1 variance in dummy dimensions means the Gaussian isn't a perfect delta function in those dimensions, introducing a non-trivial attenuation factor. The test accounts for this via `rtol=0.10`, but the tolerance could be tighter with a smaller variance (e.g., 0.01). Comment explaining the tradeoff would help.

---

### 4. `test_cuda_comparison.py` -- CUDA vs PyTorch Reference

**Severity: MEDIUM**

**Strengths:**
- Comprehensive parametric testing across 2D/3D/4D for both isotropic and anisotropic splats.
- `test_single_centered_splat_matches` is an excellent sanity check -- verifies peak location and value for the simplest possible case.
- Direct backend-level tests (calling `cuda_splatting_backend.forward` directly) supplement model-level tests.

**Issues:**

1. **4D tests use same tolerances as 2D/3D but are known to have larger discrepancies (MEDIUM).** The `_assert_outputs_match` helper uses the same thresholds for all dimensions, but comments in the code acknowledge 4D has larger differences due to the generic nD path. Should use `COMPARISON_BOUNDARY_REL_DIFF` (0.30) for 4D instead of the default.

2. **`test_forward_isotropic_splats_4d` comment says "Strict tolerance -- this will fail" (MEDIUM).** Line 420-421: `# Strict tolerance - this will fail, flagging need for investigation`. This suggests the test was written expecting failure but is still asserting. If this known failure isn't resolved, it should be `pytest.xfail` with a ticket reference, not a hard assert that may randomly pass or fail.

3. **Missing backward comparison tests (MEDIUM).** This file only tests forward passes. A `test_backward_cuda_vs_pytorch_3d` comparing gradient values would complete the coverage.

---

### 5. `test_cuda_gradcheck.py` -- Gradient Correctness (Gold Standard)

**Severity: HIGH**

**Strengths:**
- Explicitly labeled as "GOLD STANDARD" for gradient verification.
- Tests edge cases: single splat, dense overlapping splats, boundary splats.
- `test_multi_iteration_stability` catches accumulation drift over 20 iterations.
- Tests both 2D and 3D via parametrize.

**Issues:**

1. **Does NOT actually use `torch.autograd.gradcheck` (CRITICAL).** Despite the file name and docstring claiming "gradcheck uses finite differences to numerically verify analytical gradients", no test in this file calls `torch.autograd.gradcheck()`. The tests only verify: (a) gradients are finite, (b) sign consistency > 80%, (c) correlation > 0.5. This misses the entire purpose of gradcheck -- finite-difference verification of the Jacobian. This is the single most important missing test in the CUDA suite.

   **Recommendation:** Add a true `torch.autograd.gradcheck` test for the `CUDASplatFunction.apply()` with small N and small shape (e.g., N=3, shape=(8,8,8)). Use `eps=1e-3` and `atol=1e-2` to account for FP32 CUDA precision. If the custom autograd backward is correct, this should pass. If it doesn't pass, that's a real bug.

2. **Correlation > 0.5 threshold is extremely loose (HIGH).** Line 159: `assert corr > 0.5`. A correlation of 0.5 means the CUDA and reference gradients explain only 25% of each other's variance. This would still pass with severely incorrect gradient computation. For parameter groups like `raw_a` (amplitude), correlation should be > 0.95. For `raw_mu` through the softplus chain, > 0.8 is reasonable.

3. **`test_multi_iteration_stability` only checks loss is changing, not decreasing (MEDIUM).** Line 265: `assert losses[0] != losses[-1]`. With `output.sum()` as the loss and SGD with lr=0.001, the loss should decrease (since all parameters contribute positively to the sum). Checking `losses[-1] < losses[0]` would be a stronger assertion. The current check would pass even if gradients are flipped and the loss increases.

---

### 6. `test_cuda_model.py` -- Model Class Tests

**Severity: LOW**

**Strengths:**
- Comprehensive coverage: creation, dimension validation, device validation, auto tile size, output shape.
- Edge cases: single splat, boundary splat, outside volume, 7D, 8D, very small splat, overlapping splats, negative amplitude, zero amplitude.
- `test_overlapping_splats` verifies linear superposition (N identical splats = N * single splat) -- important correctness invariant.
- `TestSpecializedVsGenericImplementations` cross-validates 2D vs 3D and 3D vs 4D code paths.

**Issues:**

1. **`test_negative_amplitude` checks trivially (LOW).** Line 487: `(output - single_output).abs().max().item()` computes a value but never asserts on it. This is dead code from a computation that was likely meant to be an assertion.

2. **`test_zero_amplitude_splats` may fail with softplus activation (MEDIUM).** The model uses `softplus(raw_a)` for amplitude, so `amps0=0.0` does NOT produce zero amplitude -- softplus(x) = ln(1+e^x) > 0 for all x. The assert `output.abs().max() < 1e-6` may be incorrect. Need to verify whether the model maps `amps0=0.0` to raw_a such that `softplus(raw_a) = 0`. If the model uses `raw_a = softplus_inverse(amps0)`, then `amps0=0.0` maps to `raw_a = -inf`, which is problematic.

3. **Missing `test_1d_rejection` (LOW).** The model validates `d >= 2`, but there's no test verifying that 1D input is rejected.

---

### 7. `test_cuda_nd.py` -- Higher Dimensions (4D+)

**Severity: MEDIUM**

**Strengths:**
- Thorough 4D diagnostics: single splat, conic computation, Mahalanobis consistency, binning verification, tile boundary positions.
- `test_4d_multiple_splats_detailed` with well-separated splats provides good diagnostic output.
- `TestGlobalSplatHandling` tests the global splat kernel path (splats covering >1024 tiles).

**Issues:**

1. **`cleanup_cuda_state` fixture calls `gc.collect()` and `torch.cuda.empty_cache()` (MEDIUM).** While this prevents state pollution, it adds significant overhead to every test and can mask real memory leaks. Consider moving this to only the tests that need it, or using a session-scoped cleanup.

2. **No 5D or 6D forward tests (MEDIUM).** Tests jump from 4D to the global splat test on 3D 128^3. The CUDA backend supports up to 8D, and the tile sizes differ (5D: tile=3, 6D: tile=3). Adding at least one 5D or 6D forward-vs-PyTorch comparison would catch dimension-specific bugs.

3. **Missing backward tests for 4D+ (HIGH).** All tests in this file are forward-only. The generic nD backward path is exercised but never validated against a reference for d >= 4. This is a gap because the generic backward uses `torch.linalg.solve_triangular` (in `cholesky_to_conic`'s nD path) which has different numerical characteristics than the explicit 2D/3D formulas.

4. **`test_global_splat_forward_matches_reference` uses 128^3 volume (LOW).** This test allocates ~8MB for the output tensor alone, which is fine for CI but may be slow or OOM on constrained environments. Consider adding `@pytest.mark.slow`.

---

### 8. `test_cuda_numerical.py` -- Numerical Accuracy

**Severity: MEDIUM**

**Strengths:**
- Pure mathematical tests for Mahalanobis distance, generalized Gaussian, gradient formulas, and conic gradients.
- Finite-difference verification of analytical gradient formulas.
- Tests standard (s=2), sharp (s>2), and soft (s<2) Gaussian profiles.

**Issues:**

1. **Tests are pure NumPy/math -- don't touch CUDA at all (MEDIUM).** Despite being in the CUDA test directory, none of these tests use the GPU. They verify mathematical formulas in Python. While valuable, they don't test whether the CUDA kernel implements these formulas correctly. Consider: (a) moving to a shared `tests/math/` directory, or (b) adding companion tests that extract intermediate values from the CUDA kernel and compare.

2. **Dead code: unused variables (LOW).**
   - Line 121: `a * np.exp(inner)` -- computed but never used (should be `intensity = a * np.exp(inner)`)
   - Line 209: `np.array([1.0, 0.0, 1.0])` -- created but never used

3. **`TestGeneralizedGaussian` tests sharpness parameter `s`, but the CUDA kernel hardcodes `s=2` (MEDIUM).** The source code `gsplat_model_cuda.py` does not expose a sharpness parameter. The `test_sharp_gaussian` and `test_soft_gaussian` tests validate math for a feature that isn't implemented in the CUDA backend. This is either dead coverage or forward-looking for a planned feature. Should be marked `@pytest.mark.skip` with a note, or removed.

4. **Missing 2D Mahalanobis distance test (LOW).** Only 3D Mahalanobis distance is tested. The CUDA kernel has specialized 2D code.

---

### 9. `test_cuda_performance.py` -- Performance Benchmarks

**Severity: LOW**

**Strengths:**
- Tests both absolute performance and scaling characteristics.
- Memory footprint test catches CUDA memory bloat.
- CPU baseline measurements provide context for speedup claims.

**Issues:**

1. **`test_cuda_speedup_small_3d` asserts `speedup > 0.5` (MEDIUM).** This means CUDA can be 2x SLOWER than CPU and still pass. For a "speedup" test, this is backward. The assertion should be `speedup > 1.0` at minimum for even small workloads, or the test should be renamed to `test_cuda_not_catastrophically_slow`.

2. **No CUDA Event timing (MEDIUM).** All benchmarks use `time.perf_counter()` with `torch.cuda.synchronize()`. While functional, CUDA Events provide sub-microsecond precision and eliminate Python scheduling noise. For benchmark tests that inform optimization decisions, this matters.

3. **`test_cuda_memory_footprint` uses `100x` overhead threshold (LOW).** Allowing 100x expected base memory is extremely permissive. A realistic forward pass should use 5-20x the input+output memory (accounting for tile buffers and intermediate computations). Tightening to 50x would still be safe while catching real memory issues.

4. **`test_scaling_with_volume_size_cpu` doesn't actually test scaling (LOW).** The comment on line 367 acknowledges this: "We only verify the test ran without errors." The linear scaling assertion was removed because it was too flaky. Either remove this test (it provides no value as-is) or implement a more robust scaling check.

---

### 10. `test_cuda_fp16.py` -- Half Precision

**Severity: MEDIUM**

**Strengths:**
- Tests FP16 accuracy against FP32 for both forward and backward (AMP mode).
- Uses 95th percentile relative difference instead of max (robust to outliers).
- Verifies output dtype is always FP32 regardless of input precision.
- Memory and timing characterization tests.

**Issues:**

1. **`test_backward_amp_matches_fp32` uses 50% median tolerance (HIGH).** A 50% median relative difference between FP32 and AMP gradients means half the gradient elements differ by >50%. While FP16 has limited precision, the median should be much tighter (~5-10%) for a correctly working AMP implementation. If this tolerance is necessary, it suggests a real precision issue in the backward pass under autocast.

2. **Missing `test_backward_fp16_blocked` (MEDIUM).** The source code blocks training with `use_fp16=True` (raises RuntimeError). This guard should have a test: `with pytest.raises(RuntimeError, match="Cannot train with use_fp16=True")`.

3. **`test_fp16_timing_comparison` is informational-only (LOW).** The test prints timing but has no assertion (the remaining code after line 399 was not shown but likely just prints). Performance tests without assertions add CI overhead without catching regressions.

---

### 11. `test_cuda_review_fixes.py` -- Regression Tests for Specific Fixes

**Severity: LOW**

**Strengths:**
- Excellent practice: each test targets a specific fix with clear documentation of what would break without the fix.
- `test_3d_oversized_tile_backward_no_crash` tests the buffer overflow guard -- a critical safety check.
- `test_3d_oversized_tile_forward_matches_default` verifies tile-size independence.
- `test_amplitude_near_floor_correctness` tests edge behavior at the intensity floor.

**Issues:**

1. **`test_amplitude_near_floor_correctness` creates splats at 1.01x and 0.99x intensity_floor (LOW).** These values are so close together that FP32 precision may not distinguish them reliably. Consider using 2x and 0.5x for clearer separation.

2. **Missing 2D buffer overflow test (LOW).** The test only covers 3D oversized tiles. The docstring mentions 2D (`MAX_TILE_PIXELS=256, tile_size=32 gives 1024 > 256`), but there's no corresponding 2D test.

---

## Metal Test Suite

### 12. `test_coordinate_transforms.py` -- Coordinate Convention Tests

**Severity: LOW**

**Strengths:**
- Excellent isolation of coordinate transforms: centers, L matrix, conic, gradients.
- Tests roundtrip reversibility.
- `test_conic_from_L` documents a subtle trap: permuting L then computing conic is NOT equivalent to computing conic then permuting -- important correctness check.
- End-to-end test compares Metal and PyTorch for asymmetric splat placement.

**Issues:**

1. **`TestCentersReordering` and `TestGradientsReordering` are redundant (LOW).** Both test the exact same operation: `[:, [2, 1, 0]]` permutation and its inverse. The gradient reordering is mathematically identical to the forward reordering (it's a self-inverse permutation). One test class would suffice.

2. **End-to-end test tolerance of 0.02 (2%) is tight for GPU comparison (LOW).** Metal vs CPU can differ by more than 2% in peak values due to per-pixel intensity culling. Consider using 5% or adding an explanation.

---

### 13. `test_metal_backend.py` -- Basic Metal Functionality

**Severity: LOW**

**Strengths:**
- Clean fixture-based design.
- Tests: execution, shape, device, dtype, value ranges.
- Edge cases: single splat, small volume, deterministic repeated calls.
- Backward pass: execution, gradient existence, gradient finiteness.

**Issues:**

1. **No numerical accuracy comparison against CPU (MEDIUM).** Forward tests only check `output.min() >= 0` and `output.max() > 0`. Unlike the CUDA test suite which compares against PyTorch reference outputs, this file has no such comparison. The accuracy comparison exists in `test_metal_numerical.py`, but basic sanity should also be in the core backend test.

2. **`test_backward_executes` asserts `True` (LOW).** Line 155: `assert True  # If we get here, it worked`. This is technically correct but confusing. Better to assert on a computed value (e.g., `assert loss.item() > 0`).

3. **Missing `test_non_3d_rejection` (LOW).** The Metal model raises `ValueError` for non-3D inputs, but this isn't tested here. It's a constructor validation that deserves a test.

---

### 14. `test_metal_conic.py` -- Metal L-to-Conic

**Severity: LOW**

**Strengths:**
- Thorough comparison between Metal and PyTorch conic computation.
- Tests diagonal, non-diagonal, batch, and coordinate ordering.
- Integration tests verify Metal conic in full forward/backward pipeline.
- `test_matches_pytorch_conic` compares full model output with and without Metal conic.

**Issues:**

1. **`test_batch_processing` tolerance 1e-4 may be too loose for FP32 (LOW).** The Metal kernel should match PyTorch to ~1e-6 for the same mathematical operation. If 1e-4 is needed, it suggests numerical differences in the Metal implementation that should be documented.

2. **Tests disabled feature (`use_metal_conic`) (LOW).** The Metal model defaults `use_metal_conic=False` with comment "DISABLED: Gradient bug found, investigating". The conic tests are still active, which is good for tracking the fix, but should be marked with a note about the known bug.

---

### 15. `test_metal_numerical.py` -- Numerical Accuracy and Gradients

**Severity: MEDIUM**

**Strengths:**
- `test_gradient_sign_correctness` is excellent -- directly validates the fix for the Y/X sign bug in the Metal kernel.
- `test_gradient_values_match_cpu_reference` compares Metal gradients against CPU render_gaussians -- the strongest correctness check in the Metal suite.
- `test_optimization_convergence` runs 30 iterations and checks convergence to target -- an integration-level gradient test.

**Issues:**

1. **`test_gradcheck_simple` skips on failure (HIGH).** Lines 331-332: `pytest.skip("gradcheck failed (expected for GPU - use integration tests)")`. This masks potential gradient bugs. `torch.autograd.gradcheck` is the gold standard for verifying backward correctness. If it fails, the gradients ARE wrong (within the tolerance). The fix should be to: (a) debug why gradcheck fails, (b) use `pytest.xfail` with a ticket reference, or (c) loosen tolerances until it passes. Skipping defeats the purpose entirely.

2. **`test_gradcheck_simple` runs on CPU device (MEDIUM).** Line 301: `device="cpu"`. This means the Metal kernel is never exercised by gradcheck -- the CPU/PyTorch fallback path is used instead. To test Metal gradients with gradcheck, the test would need to operate at the `MetalSplatFunction.apply()` level with MPS tensors, not through the model class on CPU.

3. **`test_gradient_values_match_cpu_reference` uses `MetalSplatFunction.apply()` directly (GOOD).** This is the correct approach -- it bypasses the model's parameter transformations and tests the raw autograd function. This is the most valuable test in the Metal suite.

4. **15% tolerance for gradient magnitude match (LOW).** Line 438: `rtol=0.15`. For GPU vs CPU on the same algorithm, 15% is reasonable but not great. If tightening reveals issues, they should be investigated.

---

### 16. `test_metal_performance.py` -- Performance Benchmarks

**Severity: LOW**

**Strengths:**
- Comprehensive: forward, backward, scaling with volume size, scaling with splat count.
- Memory leak test runs 50 iterations.
- No false speedup assertions (correctly notes Metal overhead may exceed benefit for small problems).

**Issues:**

1. **No performance assertions at all (MEDIUM).** Every test ends with `assert True` or no assertion. While the comment explains that Metal may be slower for small problems, the tests should assert at least for medium/large configurations where Metal is expected to provide benefit. Otherwise these are informational prints, not tests.

2. **`benchmark_backward` timing is inaccurate (LOW).** The backward timing includes `loss = output.sum()` computation (creating the loss graph) in the backward measurement. This inflates backward time. Should separate loss computation from backward timing.

---

### 17. `test_optimizer_compatibility.py` -- Optimizer Integration

**Severity: LOW**

**Strengths:**
- Tests property delegation (shape, dim, truncate, internal parameters).
- Tests both standard `torch.optim.Adam` and Luxar's `create_optimizer_and_scheduler`.
- Multi-step training iteration test catches state management issues.

**Issues:**

1. **`test_multiple_training_steps` doesn't check loss decreasing (LOW).** Only checks losses are finite. For a well-behaved model, loss should decrease over 5 steps of Adam. This is a stronger integration check.

2. **Missing `test_prune_` and `test_append_` (LOW).** The Metal model delegates these to the base model, but they're not tested. If the delegation is broken, optimization workflows that add/remove splats will fail silently.

---

## Cross-Suite Analysis

### Redundancy Between CUDA and Metal

**Severity: LOW**

The `cholesky_to_conic` function is duplicated between CUDA and Metal source files and tested independently in both suites. The implementations are nearly identical (same forward substitution formula). Consider:
1. Extracting `cholesky_to_conic` to a shared module.
2. Having a single set of conic tests that both suites reference.

### Missing Cross-Backend Tests

**Severity: MEDIUM**

There are no tests comparing CUDA and Metal outputs for the same input (on machines with both). While the two backends target different hardware, they implement the same algorithm and should produce identical results (within FP32 tolerance). A parametrized test that runs on whichever backend is available would catch algorithm-level divergences.

### Generalized Gaussian (Sharpness Parameter)

**Severity: MEDIUM**

`test_cuda_numerical.py` tests `s != 2` (sharp/soft Gaussian), but neither backend exposes or implements a sharpness parameter. Both hardcode `s=2` (standard Gaussian: `exp(-0.5 * dist^2)`). These tests validate math for an unimplemented feature. They should be either:
- Removed (dead coverage)
- Moved to a `future/` test directory
- Marked with `@pytest.mark.skip(reason="Sharpness parameter not yet implemented")`

### Metal's 3D-Only Limitation

**Severity: LOW (informational)**

Metal tests only cover 3D (by design -- the backend rejects non-3D input). CUDA tests cover 2D-8D. This is correctly scoped to the backends' capabilities. The Metal model's non-3D rejection should have a test (noted above).

---

## Summary of Findings by Severity

### CRITICAL (1)
| File | Issue |
|------|-------|
| `test_cuda_gradcheck.py` | File named "gradcheck" but never calls `torch.autograd.gradcheck` -- the most rigorous gradient verification is missing entirely |

### HIGH (4)
| File | Issue |
|------|-------|
| `test_cuda_backward.py` | Backward comparison only checks amplitude gradients; center and L gradients not compared |
| `test_cuda_gradcheck.py` | Correlation > 0.5 threshold is far too loose for gradient correctness |
| `test_cuda_nd.py` | No backward tests for 4D+ dimensions |
| `test_metal_numerical.py` | `test_gradcheck_simple` silently skips on failure, masking potential bugs |

### MEDIUM (16)
| File | Issue |
|------|-------|
| `conftest.py` | `BACKWARD_SIGN_MATCH = 0.70` too permissive |
| `conftest.py` | `COMPARISON_MAX_REL_DIFF = 0.15` too permissive for same-algorithm tests |
| `test_cuda_forward.py` | GPU benchmark uses `time.time()` instead of CUDA events |
| `test_cuda_forward.py` | Warning-only assertion for kernel activation check |
| `test_cuda_backward.py` | No 2D backward test |
| `test_cuda_backward.py` | Dummy-dimension variance too large for tight comparison |
| `test_cuda_comparison.py` | 4D uses same tolerance as 2D/3D; known-failure test not marked `xfail` |
| `test_cuda_comparison.py` | Missing backward comparison tests |
| `test_cuda_gradcheck.py` | Multi-iteration test doesn't check loss is decreasing |
| `test_cuda_model.py` | `test_zero_amplitude_splats` may be incorrect with softplus |
| `test_cuda_nd.py` | No 5D or 6D tests |
| `test_cuda_numerical.py` | Tests are pure Python math, never touch CUDA |
| `test_cuda_numerical.py` | `TestGeneralizedGaussian` tests unimplemented sharpness parameter |
| `test_cuda_fp16.py` | 50% median gradient tolerance for AMP is too loose |
| `test_metal_numerical.py` | `test_gradcheck_simple` runs on CPU, not Metal |
| `test_metal_performance.py` | No performance assertions for any configuration |

### LOW (15)
Various cleanup items: dead code, missing minor test cases, loose tolerances, structural issues. See individual file sections above.

---

## Recommended Priority Actions

1. **Add true `torch.autograd.gradcheck` for `CUDASplatFunction`** (CRITICAL). Small N, small shape, relaxed tolerances for FP32 CUDA. This is the single most impactful improvement.

2. **Tighten backward gradient tolerances** (HIGH). `BACKWARD_SIGN_MATCH` from 0.70 to 0.85; gradcheck correlation from 0.5 to 0.8 for most parameters.

3. **Extend backward comparison to all gradient components** (HIGH). Compare center and L gradients, not just amplitudes.

4. **Add 4D+ backward tests** (HIGH). The generic nD backward path is untested against any reference.

5. **Fix Metal gradcheck** (HIGH). Either make it pass on MPS device or use `pytest.xfail` with issue tracker reference.

6. **Add 5D/6D forward coverage** (MEDIUM). These dimensions use different tile sizes and exercise different code paths.

7. **Use CUDA events for benchmarks** (MEDIUM). Replace `time.perf_counter()` with `torch.cuda.Event` for accurate GPU timing.
