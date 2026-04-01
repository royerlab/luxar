# Luxar Test Suite Review -- Updated Summary (Post PR #53)

**Last updated**: 2026-03-31 (reports updated after PR #53 merged)
**PR #53 fixed**: 11 issues (4 CRITICAL, 4 HIGH, 2 MEDIUM, 1 flaky test)

---

## What PR #53 Fixed

| Fix | Severity | Report |
|-----|----------|--------|
| CUDA: Added real `torch.autograd.gradcheck` | CRITICAL | GPU CUDA/Metal |
| E2E error-recovery: Real error scenarios via route interception | CRITICAL | E2E Playwright |
| E2E data-monitor: Added missing assertions | CRITICAL | E2E Playwright |
| Python reader_nodes: Fixed parent= API misuse | CRITICAL | Encoding/IO |
| CUDA backward: Extended to all gradient components | HIGH | GPU CUDA/Metal |
| CUDA conftest: Tightened BACKWARD_SIGN_MATCH 0.70->0.85 | HIGH | GPU CUDA/Metal |
| E2E controls: Fixed no-op tests | HIGH | E2E Playwright |
| Python roundtrip: Added value assertions (not just shape) | HIGH | Encoding/IO |
| TS range-loader: Expanded from 2 to 42 tests | HIGH | TS Data |
| TS scene-loader/data-loading: Removed dead mocks | MEDIUM | TS Data |
| Python dtype_support: Seeded RNG for flaky test | MEDIUM | Validation/CLI |
| eslint: Added avoidEscape for prettier compat | CONFIG | N/A |

---

## Remaining Issues by Severity

| Severity | Count | Notes |
|----------|-------|-------|
| CRITICAL | 6 | Mostly E2E tests that verify nothing |
| HIGH | ~40 | Missing coverage, over-mocking, wrong assertions |
| MEDIUM | ~95 | Gaps, weak assertions, redundancy |
| LOW | ~60 | Style, minor improvements |

---

## Recommended Next Batch (Top 20, Prioritized)

### Tier 1: High-Impact, Low-Effort (quick wins)

| # | Issue | Domain | Severity | Effort | Report |
|---|-------|--------|----------|--------|--------|
| 1 | `real-dataset-loading.spec.ts`: fix `attrs.size` -> `attrs.radius` | E2E | CRITICAL | 1 line | E2E Playwright |
| 2 | `geometry-update-manager.test.ts`: fix `validateColorMode` no-op test | TS Unit | HIGH | 1 line | TS Data |
| 3 | `data-loading-monitor.test.ts`: fix vacuous button-click if-guards | TS Unit | MEDIUM | 3 lines | TS Data |
| 4 | `test_spatial_dimensions.py`: fix conditional `if attr in attrs` assertion | Python | MEDIUM | 1 line | Python Core |
| 5 | `gsplats_processing.rs`: tighten 10% tolerance to 1% for exact math | Rust | MEDIUM | trivial | Rust WASM |

### Tier 2: High-Impact, Medium-Effort (significant coverage gaps)

| # | Issue | Domain | Severity | Effort | Report |
|---|-------|--------|----------|----------|--------|
| 6 | `worker-wasm-integration.spec.ts`: rewrite to actually test Workers/WASM | E2E | CRITICAL | ~2hr | E2E Playwright |
| 7 | `spatial-index-accuracy.spec.ts`: add real accuracy assertions | E2E | CRITICAL | ~1hr | E2E Playwright |
| 8 | `performance-tracking.spec.ts`: fix baseline file race condition | E2E | CRITICAL | ~1hr | E2E Playwright |
| 9 | Add `nd_transform` property tests (getter, setter, world composition) | Python | HIGH | ~1hr | Python Core |
| 10 | Add `intensity`, `offset`, `layer`, `colormap` Node property tests | Python | HIGH | ~1hr | Python Core |
| 11 | Add 4D+ backward pass tests for CUDA | GPU | HIGH | ~1hr | GPU CUDA/Metal |
| 12 | Fix Metal gradcheck skip-on-failure (silent pass) | GPU | HIGH | ~30min | GPU CUDA/Metal |
| 13 | `lines_clipping.rs`: test 5 batch functions with zero coverage | Rust | HIGH | ~2hr | Rust WASM |
| 14 | `gsplats.rs`: test anisotropic Cholesky (only identity tested) | Rust | HIGH | ~1hr | Rust WASM |
| 15 | Integrate `scripts/test_*.py` ad-hoc scripts into pytest suite | Python | HIGH | ~1hr | Validation/CLI |

### Tier 3: Structural Improvements (reduce technical debt)

| # | Issue | Domain | Severity | Effort | Report |
|---|-------|--------|----------|--------|--------|
| 16 | HDR export error-recovery test: verify effects actually restored | TS Unit | CRITICAL | ~1hr | TS Rendering |
| 17 | `postprocessing-manager.test.ts`: replace 6+ vacuous `toBeDefined()` | TS Unit | HIGH | ~1hr | TS Rendering |
| 18 | Consolidate duplicate `validate_layer` tests across 2 files | Python | HIGH | ~30min | Validation/CLI |
| 19 | Fix silent `try/except` in `test_validation_module.py` | Python | MEDIUM | ~15min | Validation/CLI |
| 20 | Create encode-decode round-trip test covering all 12 encoding types | Python | HIGH | ~2hr | Encoding/IO |

---

## Per-Report Remaining Counts

| Report | CRITICAL | HIGH | MEDIUM | LOW |
|--------|----------|------|--------|-----|
| Python Core | 1 | 5 | 10 | 8 |
| Python Encoding/IO | 0 | 3 | 10 | 8 |
| Python GSplats | 0 | 0 | 5 | 12 |
| Python Validation/CLI | 0 | 3 | 8 | 7 |
| TS Data Unit | 1 | 7 | 9 | 6 |
| TS Rendering/Scene/Controls | 1 | 6 | 10 | 8 |
| TS UI/Cache/Misc | 0 | 3 | 5 | ~10 |
| E2E Playwright | 6 | 16 | 25 | 15 |
| Rust WASM | 0 | 5 | 9 | 6 |
| GPU CUDA/Metal | 0 | 2 | 14 | 15 |

---

## Individual Reports

| Report | File |
|--------|------|
| Python Core | [`python_core_tests.md`](python_core_tests.md) |
| Python Encoding/IO | [`python_encoding_io_tests.md`](python_encoding_io_tests.md) |
| Python GSplats | [`python_gsplats_tests.md`](python_gsplats_tests.md) |
| Python Validation/Utils/CLI | [`python_validation_utils_cli_tests.md`](python_validation_utils_cli_tests.md) |
| TS Data Unit | [`ts_data_unit_tests.md`](ts_data_unit_tests.md) |
| TS Rendering/Scene/Controls | [`ts_rendering_scene_controls_tests.md`](ts_rendering_scene_controls_tests.md) |
| TS UI/Cache/Misc | [`ts_ui_cache_misc_tests.md`](ts_ui_cache_misc_tests.md) |
| E2E Playwright | [`e2e_playwright_tests.md`](e2e_playwright_tests.md) |
| Rust WASM | [`rust_wasm_tests.md`](rust_wasm_tests.md) |
| GPU CUDA/Metal | [`gpu_cuda_metal_tests.md`](gpu_cuda_metal_tests.md) |
