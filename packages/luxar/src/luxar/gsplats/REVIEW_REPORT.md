# GSplats Core, I/O, Utils, CLAHE, Multiscale, and Optim Code Review Report

**Date:** 2026-02-24
**Reviewer:** Claude Opus 4.6
**Scope:** Top-level orchestration, I/O, Utils, CLAHE, Multiscale, and Optim packages

---

## Executive Summary

The gsplats codebase is well-structured overall, with good documentation, thoughtful API design, and thorough parameter validation in most areas. However, this review identified **3 CRITICAL**, **7 HIGH**, **12 MEDIUM**, and **8 LOW** severity issues across the reviewed files.

The most impactful findings are:
- **Division-by-zero crash** in `GSplatData.prune()` when all amplitudes are zero
- **Empty array crash** in `save_gsplats()` when saving zero splats (min/max on empty arrays)
- **Silent data corruption risk** in `load_gsplats()` where compressed archive extraction can misidentify zarr directories
- **Incorrect downsampling** in `decompose.py` for nD (n>3) data where `mode="nearest"` is used instead of area averaging
- **Variable shadowing** in `fit_multiscale_gsplats.py` where `truncate` parameter is overwritten inside a loop

---

## 1. Top-Level Orchestration

### 1.1 `__init__.py`

| # | Severity | Issue | Lines |
|---|----------|-------|-------|
| 1 | LOW | Missing `clahe` and `seeds` stub exports in fallback | 8, 88-89 |
| 2 | LOW | Fallback `_raise_gsplats_import_error` doesn't return `NoReturn` type | 23-27 |

### 1.2 `fit_gsplats.py`

| # | Severity | Issue | Lines |
|---|----------|-------|-------|
| 3 | MEDIUM | Inconsistent default values between `GaussianSplatFitter.fit()` and `fit_gaussian_splats()` | 86-109 vs 227-253 |
| 4 | MEDIUM | `ConstraintConfig` override silently ignores `amp_max` when it is `None` | 493-496 |
| 5 | LOW | Import inside function body for `aprint` and `display_compression_analysis` | 545, 556 |

Key default inconsistencies:

| Parameter | `GaussianSplatFitter.fit()` | `fit_gaussian_splats()` |
|-----------|---------------------------|------------------------|
| `max_eccentricity` | `None` | `10.0` |
| `sharpness_range` | `None` | `2.0` |
| `lr_reduction_factor` | `0.95` | `0.98` |
| `early_stop_patience` | `200` | `300` |
| `enable_dynamic_ops` | `False` | `True` |

### 1.3 `gsplat_data.py`

| # | Severity | Issue | Lines |
|---|----------|-------|-------|
| 6 | CRITICAL | `prune()` with `method="cumulative"` crashes on zero-amplitude data | 298 |
| 7 | HIGH | `prune()` with empty GSplatData (N=0) crashes | 267-298 |
| 8 | MEDIUM | `prune()` `amplitude_retention` stat is NaN when total amplitude is zero | 337 |
| 9 | MEDIUM | Mutable default for `stats` parameter in dataclass | 51 |
| 10 | MEDIUM | `translate()` shares references to mutable arrays | 165-170 |
| 11 | LOW | Missing `__len__` and `__repr__` on dataclass | 16 |

**Issue 6 detail (CRITICAL)**: When all splat amplitudes are zero, `cumsum_amps[-1]` is `0.0`, producing NaN from division by zero. Then `np.searchsorted` on NaN gives undefined behavior.

### 1.4 `fit_multiscale_gsplats.py`

| # | Severity | Issue | Lines |
|---|----------|-------|-------|
| 12 | HIGH | Variable `truncate` is shadowed inside loop, affects later iterations | 541, 584 |
| 13 | MEDIUM | Seed distribution rounding can drop below minimums | 146-152 |
| 14 | MEDIUM | `_distribute_seeds_by_maxima` ignores `maxima_per_scale` length mismatch silently | 114 |
| 15 | LOW | Unused variable `sharpness_combined` when `valid_params` is empty | 640-647 |

---

## 2. I/O Package

### 2.1 `save_gsplats.py`

| # | Severity | Issue | Lines |
|---|----------|-------|-------|
| 16 | CRITICAL | `min()`/`max()` on empty arrays crashes when `n_splats == 0` | 189-201 |
| 17 | HIGH | Temp directory leak if compression fails before `try` block | 105, 339-342 |
| 18 | MEDIUM | Version import always results in `"unknown"` | 22-26 |
| 19 | MEDIUM | No validation that `centers` is 2D | 118 |
| 20 | LOW | Hardcoded sharpness bounds `(0.0, 31.0)` | 290 |

### 2.2 `load_gsplats.py`

| # | Severity | Issue | Lines |
|---|----------|-------|-------|
| 21 | HIGH | `is_file()` check triggers extraction for any regular file, not just compressed archives | 100-102 |
| 22 | HIGH | Compressed archive temp directory is deleted while zarr data may still be in use | 188-191 |
| 23 | MEDIUM | `_extract_compressed_zarr` fallback picks first directory blindly | 59-61 |
| 24 | MEDIUM | No `tarfile` path traversal protection | 43-44 |

### 2.3 `inspect_gsplats.py`

| # | Severity | Issue | Lines |
|---|----------|-------|-------|
| 25 | MEDIUM | `format_gsplats_info` uses wrong key names for ordering resolution | 156-158 |
| 26 | MEDIUM | `render_gsplats_to_volume` placed in wrong module | 200-261 |
| 27 | LOW | Uncompressed size estimate assumes float32 colors (12 bytes) but uint8 is 3 bytes | 122 |

**Issue 25 detail**: `info.get("morton_resolution", "unknown")` and `info.get("hilbert_resolution", "unknown")` will always return "unknown" because `inspect_gsplats_zarr()` stores the resolution under the key `"ordering_resolution"`.

---

## 3. Utils Package (`trils.py`)

| # | Severity | Issue | Lines |
|---|----------|-------|-------|
| 28 | LOW | `pack_tril` and `unpack_tril` use Python loops instead of vectorized operations | 130-135, 172-176 |
| 29 | LOW | No input validation in `pack_tril` / `unpack_tril` | 128, 166 |

---

## 4. CLAHE Package (`clahe_core.py`)

| # | Severity | Issue | Lines |
|---|----------|-------|-------|
| 30 | HIGH | Tile-based processing with no interpolation creates block artifacts | 112-153 |
| 31 | MEDIUM | `torch.histc` bin edge semantics differ from `torch.searchsorted` | 124, 145-147 |
| 32 | MEDIUM | Performance: Python loop over all tiles scales poorly for large nD volumes | 112 |
| 33 | LOW | Docstring claims "Intensity range preserved" but implementation rescales | 155-157 |

---

## 5. Multiscale Package (`decompose.py`)

| # | Severity | Issue | Lines |
|---|----------|-------|-------|
| 34 | HIGH | `_downsample_to_scale` uses "nearest" for area mode in nD (n>3) | 345-346 |
| 35 | MEDIUM | `_cubic_upsample_recursive` fails for non-power-of-2 mixed factors | 143-169 |
| 36 | MEDIUM | Energy conservation not guaranteed due to non-negativity clamping | 487 |
| 37 | MEDIUM | `initialize_uniform` energy calculation assumes area-averaging preserves sum | 570-574 |
| 38 | LOW | `movie_max_frames` default overwritten from `None` to `10000` | 1138-1139 |
| 39 | LOW | Dead `# type: ignore[no-untyped-call]` comment | 1182 |

**Issue 34 detail (HIGH)**: When `mode="area"` and `ndim > 3`, the code falls back to `"nearest"` interpolation. Both the `if` and `else` branches do exactly the same thing, making the conditional dead code.

---

## 6. Optim Package (`integration.py`)

| # | Severity | Issue | Lines |
|---|----------|-------|-------|
| 40 | MEDIUM | `model` parameter untyped, accesses `.shape` without validation | 62 |
| 41 | MEDIUM | `**extra_kwargs` silently ignored, hides typos | 30 |
| 42 | LOW | No logging of effective LR after gradient dilution compensation | 63 |

---

## 7. Cross-Cutting Concerns

### I/O Round-Trip Correctness

1. **Sharpness default injection**: Loading without sharpnesses creates defaults of `2.0`. Re-saving explicitly stores these.
2. **Ordering is not round-trippable**: Hilbert reordering on save creates slightly different files each time.
3. **Encoding precision loss**: `ArrayEncoder`/`ArrayDecoder` may apply quantization. `load(save(data)) != data` in general.

---

## Overall Assessment

### Strengths

1. Excellent documentation with thorough docstrings
2. Clean separation of concerns in the fitting pipeline
3. Good hardware abstraction (CUDA, MPS, CPU) with graceful fallbacks
4. Defensive programming in decomposition (convergence tracking, best-state restoration)
5. Format versioning in I/O for forward compatibility

### Areas for Improvement

1. **Edge case handling**: Several CRITICAL/HIGH issues stem from not handling empty arrays (N=0) or zero values
2. **I/O robustness**: Compressed archive handling has multiple fragility points
3. **Performance for high dimensions**: CLAHE and multiscale use Python loops that scale poorly with dimensionality
4. **API consistency**: Dual interface with different defaults is a maintenance burden
5. **Test coverage for edge cases**: N=0, all-zero data, and >3D data appear insufficiently tested

### Priority Recommendations

1. **Immediate (CRITICAL)**: Fix `prune()` division-by-zero and `save_gsplats()` empty array crash
2. **Soon (HIGH)**: Fix nD downsampling fallback, file detection logic, variable shadowing, block artifacts
3. **Next sprint (MEDIUM)**: Align defaults between APIs, add N=0 guards, improve CLAHE performance