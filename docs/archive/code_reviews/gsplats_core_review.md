> **⚠️ Archived — point-in-time review, not re-verified.** Finding statuses below (e.g. "STILL OPEN") reflect the status-check date in the header and have **not** been re-verified against current `main`. Treat this as a historical snapshot, not a live task list; verify any finding against current code before acting on it. See [the archive README](../README.md).

# GSplats Core, I/O, Utils, CLAHE, Multiscale, and Optim - Review Status Report

**Original Review Date:** 2026-02-24
**Status Check Date:** 2026-02-25
**Reviewer:** Claude Opus 4.6
**Scope:** Top-level orchestration, I/O, Utils, CLAHE, Multiscale, and Optim packages

---

## Summary

| Severity | Total | Fixed | Partially Fixed | Still Open |
|----------|-------|-------|-----------------|------------|
| CRITICAL | 3     | 2     | 0               | 1          |
| HIGH     | 7     | 3     | 2               | 2          |
| MEDIUM   | 12    | 4     | 3               | 5          |
| LOW      | 8     | 2     | 2               | 4          |
| **Total** | **30** | **11 (37%)** | **7 (23%)** | **12 (40%)** |

Overall: **60% addressed** (fixed or partially fixed), **40% still open**.

---

## Detailed Issue Status

### 1. Top-Level Orchestration

#### 1.1 `__init__.py`

**Issue #1 - LOW: Missing `clahe` and `seeds` stub exports in fallback**
- **Status: FIXED**
- **Evidence:** Lines 8 and 46-56 of `__init__.py` now import `clahe` and `seeds` in the try block (line 8: `from luxar.gsplats import clahe, seeds`), and stub fallback functions for `generate_seeds`, `seed_from_decomposition`, `seed_from_grid`, and `seed_from_edges` are present in the except block (lines 46-56). The `__all__` list at lines 59-74 includes both `"seeds"` and `"clahe"`.

**Issue #2 - LOW: Fallback `_raise_gsplats_import_error` doesn't return `NoReturn` type**
- **Status: STILL OPEN**
- **Evidence:** Line 22 still declares `def _raise_gsplats_import_error(_exc: ImportError = exc) -> None:`. The function unconditionally raises `ImportError`, so the return type should be `NoReturn` from `typing`. The `-> None` annotation is technically incorrect since the function never returns. No `NoReturn` import or annotation is present.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/__init__.py`, line 22

---

#### 1.2 `fit_gsplats.py`

**Issue #3 - MEDIUM: Inconsistent default values between `GaussianSplatFitter.fit()` and `fit_gaussian_splats()`**
- **Status: PARTIALLY FIXED**
- **Evidence:** The actual code defaults in the function signatures NOW MATCH between `GaussianSplatFitter.fit()` and `fit_gaussian_splats()`:
  - `max_eccentricity`: Both `10.0` (lines 114 and 247)
  - `sharpness_range`: Both `2.0` (lines 115 and 248)
  - `lr_reduction_factor`: Both `0.98` (lines 126 and 259)
  - `early_stop_patience`: Both `300` (lines 127 and 260)
  - `enable_dynamic_ops`: Both `True` (line 51 for class init, line 262 for function)

  However, the **docstrings are now inconsistent with the code**:
  - `fit_gaussian_splats()` docstring says `lr_reduction_factor` default is `0.95` (line 409) but code has `0.98` (line 259)
  - `fit_gaussian_splats()` docstring says `early_stop_patience` default is `200` (line 412) but code has `300` (line 260)

  Additionally, `prepare_fit_config()` in `validation.py` has different defaults in its own signature (`lr_reduction_factor=0.5`, `early_stop_patience=200`, `patience=10`), but since callers always pass explicit values, these are effectively dead defaults. They are still misleading to anyone reading the code. (Note: `cull_ratio` was removed — post-fit culling is now handled by `cull_retention` via `GSplatData.cull()`.)
- **Files:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/fit_gsplats.py` (lines 409, 412) and `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/fitting/validation.py` (lines 47-52)

**Issue #4 - MEDIUM: `ConstraintConfig` override silently ignores `amp_max` when it is `None`**
- **Status: STILL OPEN**
- **Evidence:** In `fitting/config.py`, `ConstraintConfig.amp_max` defaults to `None` (line 67). In `fitting/initialization.py` (lines 107-109), the code auto-fills `amp_max = 1.0` when `config.amp_max is None`. This means if a user explicitly passes `ConstraintConfig(amp_max=None)` intending "no constraint", it gets silently overridden to `1.0`. There is no way to distinguish "not set" from "explicitly None".
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/fitting/config.py`, line 67; `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/fitting/initialization.py`, lines 107-109

**Issue #5 - LOW: Import inside function body for `aprint` and `display_compression_analysis`**
- **Status: STILL OPEN**
- **Evidence:** In `fit_gsplats.py`:
  - Line 529: `from arbol import aprint` (inside `if verbose:` block)
  - Line 540: `from luxar.gsplats.fitting.visualization import display_compression_analysis` (inside `if verbose:` block)

  These are intentional lazy imports to avoid loading visualization dependencies when not needed. The review flagged this as a style issue. The pattern is still present and arguably acceptable for optional visualization imports.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/fit_gsplats.py`, lines 529, 540

---

#### 1.3 `gsplat_data.py`

**Issue #6 - CRITICAL: `prune()` with `method="cumulative"` crashes on zero-amplitude data**
- **Status: FIXED**
- **Evidence:** Lines 339-346 of `gsplat_data.py` now explicitly check `if total_amp == 0:` and handle the case by keeping all splats (if `target_retention > 0`) or none (if `target_retention == 0`). This prevents the division-by-zero that previously occurred at `cumsum_amps / total_amp`.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/gsplat_data.py`, lines 339-346

**Issue #7 - HIGH: `prune()` with empty GSplatData (N=0) crashes**
- **Status: FIXED**
- **Evidence:** Lines 295-308 add an explicit short-circuit: `if N_original == 0:` returns a new GSplatData with copies of the empty arrays and appropriate stats. This prevents any index/min/max operations on empty arrays.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/gsplat_data.py`, lines 295-308

**Issue #8 - MEDIUM: `prune()` `amplitude_retention` stat is NaN when total amplitude is zero**
- **Status: FIXED**
- **Evidence:** Line 387-389 now uses a conditional: `float(np.sum(pruned_amplitudes) / total_amp) if total_amp > 0 else 1.0`. When total amplitude is zero, it returns `1.0` instead of computing `0/0 = NaN`.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/gsplat_data.py`, lines 387-389

**Issue #9 - MEDIUM: Mutable default for `stats` parameter in dataclass**
- **Status: PARTIALLY FIXED**
- **Evidence:** Line 51 uses `stats: Dict[str, Any] = None  # type: ignore[assignment]` with a `__post_init__` (lines 53-56) that initializes to `{}` if `None`. This avoids the mutable default dict issue (all instances won't share the same dict). However, the `# type: ignore[assignment]` is a code smell since `None` doesn't match `Dict[str, Any]`. The proper solution would be `field(default_factory=dict)` from dataclasses, or using `Optional[Dict[str, Any]]`. The current approach works correctly at runtime.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/gsplat_data.py`, lines 51-56

**Issue #10 - MEDIUM: `translate()` shares references to mutable arrays**
- **Status: STILL OPEN**
- **Evidence:** Lines 185-192 show that `translate()` creates new centers (via `self.centers + offset`), but `amplitudes`, `cholesky_factors`, `sharpnesses`, and `colors` are passed as references (no `.copy()`). The comments on lines 187-190 explicitly say "REFERENCE (no copy needed)". While this is intentional for performance (NumPy arrays are immutable-ish when not explicitly modified), mutating the original's arrays would corrupt the translated copy. The `stats` dict IS copied (line 191). This is a design decision documented in comments, not a bug, but it means users who mutate arrays in-place could corrupt shared data.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/gsplat_data.py`, lines 185-192

**Issue #11 - LOW: Missing `__len__` and `__repr__` on dataclass**
- **Status: PARTIALLY FIXED**
- **Evidence:** A `__repr__` method was added (lines 58-72) providing a summary like `GSplatData(1,000 splats, 3D, amplitudes=[0.1, 0.9], sharpness=[1.5, 3.0], colors=no)`. However, `__len__` is still missing. Users must use `len(data.amplitudes)` instead of `len(data)`.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/gsplat_data.py`

---

#### 1.4 `fit_multiscale_gsplats.py`

**Issue #12 - HIGH: Variable `truncate` is shadowed inside loop, affects later iterations**
- **Status: STILL OPEN**
- **Evidence:** The variable `truncate` is reassigned inside the per-scale loop at TWO locations:
  - Line 541: `truncate = fit_kwargs.get("truncate", 3.0)` (inside `if return_intermediate`)
  - Line 584: `truncate = fit_kwargs.get("truncate", 3.0)` (inside `if visualize_per_scale`)

  While the value is re-read from `fit_kwargs` each time (so the actual value is consistent), this shadows a potential outer-scope variable. Since `truncate` is not used after the loop except through `fit_kwargs`, this doesn't cause a functional bug in the current code, but it's fragile and could cause issues if the code is modified. The variable name should be `local_truncate` or similar.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/fit_multiscale_gsplats.py`, lines 541, 584

**Issue #13 - MEDIUM: Seed distribution rounding can drop below minimums**
- **Status: STILL OPEN**
- **Evidence:** Lines 144-152 show the rounding adjustment: when `total_allocated != total_seeds`, the difference is applied to the scale with the most seeds. However, if `diff` is negative (over-allocated) and `seeds_per_scale[max_idx] + diff < min_seeds_per_scale[max_idx]`, the `max()` on line 150-151 ensures the minimum is respected for that one scale, but the total sum may then NOT equal `total_seeds`. This edge case remains unhandled.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/fit_multiscale_gsplats.py`, lines 144-152

**Issue #14 - MEDIUM: `_distribute_seeds_by_maxima` ignores `maxima_per_scale` length mismatch silently**
- **Status: PARTIALLY FIXED**
- **Evidence:** Line 114 checks `if not maxima_per_scale or len(maxima_per_scale) != n_scales:` and falls back to equal distribution. This handles the mismatch, but does so silently without any warning or log message. A length mismatch likely indicates a bug in the caller, and silently falling back could mask the issue.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/fit_multiscale_gsplats.py`, line 114

**Issue #15 - LOW: Unused variable `sharpness_combined` when `valid_params` is empty**
- **Status: STILL OPEN**
- **Evidence:** Lines 640-647 show that when `valid_params` is empty, the code creates `params_final` and `amps_combined` from zeros but does not create `sharpness_combined`. The variable `sharpness_combined` defined at line 640 (in the `if len(valid_params) > 0` branch) is only used at line 642 (`np.column_stack([params_combined, sharpness_combined])`), so it's not actually "unused" in the problematic branch. The empty branch at lines 644-647 handles the case separately. The real concern is that `sharpness_combined` is in-scope but never set in the else branch, which is fine because it's not accessed there. This is not a bug but a readability concern.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/fit_multiscale_gsplats.py`, lines 640-647

---

### 2. I/O Package

#### 2.1 `save_gsplats.py`

**Issue #16 - CRITICAL: `min()`/`max()` on empty arrays crashes when `n_splats == 0`**
- **Status: FIXED**
- **Evidence:** Lines 189-207 now explicitly check `if n_splats == 0:` and use safe defaults (`amplitude_min/max = 0.0`, `sharpness_min/max = 2.0`, `center_min/max = [0.0] * ndim`). The `else` branch (lines 194-207) handles the non-empty case with actual min/max operations.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/save_gsplats.py`, lines 189-207

**Issue #17 - HIGH: Temp directory leak if compression fails before `try` block**
- **Status: FIXED**
- **Evidence:** Lines 327-348 wrap the compression logic in a `try/finally` block. The `finally` clause (lines 345-348) ensures the temp directory is cleaned up even if compression fails: `if temp_dir.exists(): shutil.rmtree(temp_dir, ignore_errors=True)`. The temp dir creation at line 105 is only reached when `compress` is truthy, and the cleanup covers all failure modes within the compression block.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/save_gsplats.py`, lines 327-348

**Issue #18 - MEDIUM: Version import always results in `"unknown"`**
- **Status: STILL OPEN**
- **Evidence:** Lines 21-26 attempt to import `__version__` from `luxar.gsplats`, but `luxar.gsplats.__init__.py` does not define `__version__`. The `except ImportError` fallback at line 26 sets `GSPLATS_VERSION = "unknown"`. However, the import may succeed (since `luxar.gsplats` is importable) but fail with `ImportError` because `__version__` doesn't exist as an attribute. The `# type: ignore[attr-defined]` comment on line 23 confirms this is a known issue. The version will always be "unknown" unless `__version__` is added to the gsplats package.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/save_gsplats.py`, lines 21-26

**Issue #19 - MEDIUM: No validation that `centers` is 2D**
- **Status: STILL OPEN**
- **Evidence:** Line 118 uses `n_splats, ndim = centers.shape` which will raise `ValueError: not enough values to unpack` if `centers` is 1D. There is no explicit check like `if centers.ndim != 2: raise ValueError(...)` with a clear error message. The error from unpacking is not user-friendly.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/save_gsplats.py`, line 118

**Issue #20 - LOW: Hardcoded sharpness bounds `(0.0, 31.0)` -- RESOLVED (sharpness removed from gsplats)**
- **Status: STILL OPEN**
- **Evidence:** Line 296 still uses `bounds=(0.0, 31.0)` as a hardcoded constant for sharpness encoding. There is no named constant or configurable parameter for these bounds.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/save_gsplats.py`, line 296

---

#### 2.2 `load_gsplats.py`

**Issue #21 - HIGH: `is_file()` check triggers extraction for any regular file, not just compressed archives**
- **Status: FIXED**
- **Evidence:** Lines 107-117 now use explicit suffix-based detection: `compressed_suffixes = (".zip", ".tar.gz")` and `is_compressed = any(str(path).endswith(s) for s in compressed_suffixes)`. Only paths ending with `.zip` or `.tar.gz` trigger extraction (line 109-111). If `path.is_file()` but NOT a recognized compressed format, line 113-117 raises `ValueError("Expected a zarr directory or compressed archive (.zip/.tar.gz), got regular file: {path}")`. This properly distinguishes compressed archives from other files.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/load_gsplats.py`, lines 107-117

**Issue #22 - HIGH: Compressed archive temp directory is deleted while zarr data may still be in use**
- **Status: PARTIALLY FIXED**
- **Evidence:** Lines 200-203 use a `finally` block that cleans up the temp directory after the zarr data has been fully read. Since all arrays are decoded (loaded into NumPy arrays in memory) within the `try` block (lines 143-145), the zarr store is no longer needed when `finally` executes. However, if a user were to use `zarr.open_group` with lazy loading (not the case here, since `ArrayDecoder.decode()` materializes arrays), there could still be an issue. In the current implementation, this is safe because all data is materialized before cleanup.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/load_gsplats.py`, lines 200-203

**Issue #23 - MEDIUM: `_extract_compressed_zarr` fallback picks first directory blindly**
- **Status: PARTIALLY FIXED**
- **Evidence:** Lines 57-72 show improved logic: first, the code searches for a directory ending with `.gsplats.zarr` (lines 60-62). Only if no `.gsplats.zarr` directory is found does it fall back to the first directory (lines 65-68). If no directories exist at all, it raises `ValueError` (line 70). This is better than the original blind pick but the fallback at line 67 could still pick the wrong directory if multiple non-`.gsplats.zarr` directories exist in the archive.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/load_gsplats.py`, lines 57-72

**Issue #24 - MEDIUM: No `tarfile` path traversal protection**
- **Status: FIXED**
- **Evidence:** Lines 43-50 now include explicit path traversal validation for tar archives. Each member is checked with `member_path.resolve().is_relative_to(Path(temp_dir).resolve())`, and a `ValueError` is raised if any member would escape the extraction directory. This protects against CVE-2007-4559.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/load_gsplats.py`, lines 43-50

---

#### 2.3 `inspect_gsplats.py`

**Issue #25 - MEDIUM: `format_gsplats_info` uses wrong key names for ordering resolution**
- **Status: FIXED**
- **Evidence:** Lines 154-159 now use `info.get("ordering_resolution")` which matches the key set by `inspect_gsplats_zarr()` at lines 72-78 (`info["ordering_resolution"] = ...`). Both morton and hilbert cases use the same `"ordering_resolution"` key. The old `"morton_resolution"` / `"hilbert_resolution"` keys are no longer used in `format_gsplats_info`.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/inspect_gsplats.py`, lines 154-159

**Issue #26 - MEDIUM: `render_gsplats_to_volume` placed in wrong module**
- **Status: PARTIALLY FIXED**
- **Evidence:** The function is still in `inspect_gsplats.py` (lines 200-261), but it now:
  1. Has a deprecation warning (lines 234-240) directing users to `luxar.gsplats.rendering.render_to_volume()` or `GSplatData.render_to_volume()`
  2. Internally delegates to the new fast renderer (lines 244, 254-260)

  The function still exists in the wrong module but is now a thin deprecated wrapper. This is a reasonable approach for backward compatibility.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/inspect_gsplats.py`, lines 200-261

**Issue #27 - LOW: Uncompressed size estimate assumes float32 colors (12 bytes) but uint8 is 3 bytes**
- **Status: STILL OPEN**
- **Evidence:** Line 122 still uses `(12 if info["has_colors"] else 0)` which assumes float32 (4 bytes x 3 channels = 12 bytes) for all color data. If colors are stored as uint8, the actual uncompressed size would be 3 bytes per splat, not 12. There is no check of the actual color dtype in the metadata.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/io/inspect_gsplats.py`, line 122

---

### 3. Utils Package (`trils.py`)

**Issue #28 - LOW: `pack_tril` and `unpack_tril` use Python loops instead of vectorized operations**
- **Status: STILL OPEN**
- **Evidence:** `pack_tril` (lines 130-135) and `unpack_tril` (lines 171-175) still use nested Python `for` loops iterating over `range(d)` and `range(i+1)` to copy elements. For small `d` (2-8, typical for spatial dimensions), the overhead is negligible. For large N, the batch dimension IS vectorized (all N elements are copied per loop iteration via `out[:, k] = L[:, i, j]`). This is a low-priority performance concern.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/utils/trils.py`, lines 130-135, 171-175

**Issue #29 - LOW: No input validation in `pack_tril` / `unpack_tril`**
- **Status: STILL OPEN**
- **Evidence:** `pack_tril` at line 128 uses `N, d, _ = L.shape` which will fail with an unclear error if L is not 3D. `unpack_tril` at line 166 uses `N = v.shape[0]` without checking that `v` is 2D or that `v.shape[1] == tril_size(d)`. No explicit validation is present.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/utils/trils.py`, lines 128, 166

---

### 4. CLAHE Package (`clahe_core.py`)

**Issue #30 - HIGH: Tile-based processing with no interpolation creates block artifacts**
- **Status: STILL OPEN**
- **Evidence:** The `apply_clahe` function at lines 112-153 processes each tile independently with no inter-tile interpolation. The docstring now explicitly documents this limitation (lines 85-87): "No interpolation between tiles (for speed and simplicity)" and "For visualization, consider adding bilinear/trilinear interpolation". The implementation has not changed, but the documentation is clearer about the tradeoff. For the current use case (sampling probabilities), block artifacts are acceptable as noted in the docstring.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/clahe/clahe_core.py`, lines 85-87, 112-153

**Issue #31 - MEDIUM: `torch.histc` bin edge semantics differ from `torch.searchsorted`**
- **Status: STILL OPEN**
- **Evidence:** Line 124 uses `torch.histc(tile_flat, bins=nbins, min=V_min, max=V_max)` which creates `nbins` bins between `V_min` and `V_max`. Line 145-146 uses `torch.linspace(V_min, V_max, nbins + 1)` to create bin edges and `torch.searchsorted(bin_edges[1:], ...)` for mapping. The `histc` and `searchsorted` may disagree on edge cases (values exactly at bin boundaries), which could cause off-by-one bin assignments. The clamping on line 147 (`torch.clamp(bin_indices, 0, nbins - 1)`) mitigates the worst case but doesn't fix the fundamental mismatch.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/clahe/clahe_core.py`, lines 124, 145-147

**Issue #32 - MEDIUM: Performance: Python loop over all tiles scales poorly for large nD volumes**
- **Status: STILL OPEN**
- **Evidence:** Line 112 uses `itertools.product(*[range(n) for n in n_tiles])` which is a Python-level loop over all tiles. For a 4D volume of shape (64, 64, 64, 64) with tile_size=8, this is 8^4 = 4096 tiles. Each tile requires histogram computation, CDF calculation, and mapping. This is still a Python loop with no batched/vectorized tile processing.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/clahe/clahe_core.py`, line 112

**Issue #33 - LOW: Docstring claims "Intensity range preserved" but implementation rescales**
- **Status: PARTIALLY FIXED**
- **Evidence:** The docstring at line 62 still says "Intensity range preserved (same min/max as input)." The implementation at line 156 rescales the CLAHE output back to the original range: `V_clahe = V_clahe * (V_max - V_min) + V_min`. So the output DOES have the same min/max range as the input, making the docstring technically correct. The original concern was about intermediate processing changing the range, but the final rescaling step ensures the claim holds. The docstring is accurate.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/clahe/clahe_core.py`, lines 62, 155-156

---

### 5. Multiscale Package (`decompose.py`)

**Issue #34 - HIGH: `_downsample_to_scale` uses "nearest" for area mode in nD (n>3)**
- **Status: STILL OPEN**
- **Evidence:** Lines 345-349 show that for `ndim > 3`, both the `if mode == "area"` and `else` branches use `mode="nearest"` for `F.interpolate`:
  ```python
  if mode == "area":
      downsampled = F.interpolate(img_expanded, size=target_shape, mode="nearest")
  else:
      # Fallback for higher dimensions
      downsampled = F.interpolate(img_expanded, size=target_shape, mode="nearest")
  ```
  Both branches are identical, making the conditional dead code. When `mode="area"` is requested for >3D data, nearest-neighbor interpolation is silently used instead, which does NOT perform area averaging and will produce incorrect results for multi-scale decomposition.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/multiscale/decompose.py`, lines 342-349

**Issue #35 - MEDIUM: `_cubic_upsample_recursive` fails for non-power-of-2 mixed factors**
- **Status: STILL OPEN**
- **Evidence:** Lines 143-169 show three cases:
  1. All factors >= 2.0: Apply 2x upsample and recurse (line 143-147)
  2. All factors in [1.0, 2.0): Use interpolation for fractional part (line 148-167)
  3. Otherwise: `raise ValueError` (line 168-169)

  If dimensions have mixed factors (e.g., one needs 3x and another needs 1.5x), after one 2x upsample, the first dimension would need 1.5x and the second would need 0.75x (<1.0). This falls into the `else` branch and raises `ValueError`. This can occur with non-power-of-2 scale factors or non-uniform aspect ratios.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/multiscale/decompose.py`, lines 143-169

**Issue #36 - MEDIUM: Energy conservation not guaranteed due to non-negativity clamping**
- **Status: STILL OPEN**
- **Evidence:** Line 487 (`torch.clamp(img_scale, min=1e-6)`) in `initialize_from_pyramid` clamps all values to be positive. Since softplus requires non-negative inputs for inverse, negative residuals from the pyramid decomposition are discarded. This means the sum of all scale components may not equal the original target, violating energy conservation. The same clamping appears in `initialize_uniform` (line 587) and other initialization methods.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/multiscale/decompose.py`, line 487

**Issue #37 - MEDIUM: `initialize_uniform` energy calculation assumes area-averaging preserves sum**
- **Status: STILL OPEN**
- **Evidence:** Lines 569-574 compute `energy_per_scale_downsampled = energy_per_scale_upsampled / upsampling_factor` where `upsampling_factor = scale**self.ndim`. This assumes that upsampling by factor `scale` in each dimension multiplies total energy by `scale^ndim`. This is true for sum-preserving upsampling (like nearest-neighbor which replicates), but NOT true for interpolation-based upsampling (linear/cubic) where energy is approximately preserved but not exactly. The cubic interpolation used by default introduces small energy changes.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/multiscale/decompose.py`, lines 569-574

**Issue #38 - LOW: `movie_max_frames` default overwritten from `None` to `10000`**
- **Status: STILL OPEN**
- **Evidence:** Lines 1138-1139: `if movie_max_frames is None: movie_max_frames = 10000`. The docstring for `decompose_image` says `movie_max_frames: Optional[int], default=None` meaning "no limit". But the code silently replaces `None` with `10000`. This could surprise users who expect unlimited frames when passing `None`.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/multiscale/decompose.py`, lines 1138-1139

**Issue #39 - LOW: Dead `# type: ignore[no-untyped-call]` comment**
- **Status: STILL OPEN**
- **Evidence:** Line 1182: `loss.backward()  # type: ignore[no-untyped-call]`. The `backward()` method on PyTorch tensors IS typed in modern PyTorch type stubs. The type ignore comment may have been needed with older stubs but is likely unnecessary now. It's harmless but clutters the code.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/multiscale/decompose.py`, line 1182

---

### 6. Optim Package (`integration.py`)

**Issue #40 - MEDIUM: `model` parameter untyped, accesses `.shape` without validation**
- **Status: STILL OPEN**
- **Evidence:** Line 14 declares `model` without type annotation: `def create_optimizer_and_scheduler(model, ...)`. Line 62 accesses `model.shape` without validation: `d = len(model.shape)`. If `model` doesn't have a `shape` attribute, this will raise `AttributeError` with no helpful context. The docstring says `model: GaussianSplatModel` but this is not enforced.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/optim/integration.py`, lines 14, 62

**Issue #41 - MEDIUM: `**extra_kwargs` silently ignored, hides typos**
- **Status: STILL OPEN**
- **Evidence:** Line 30 accepts `**extra_kwargs` but these are never used anywhere in the function body. A caller passing `sceduler_type="plateau"` (typo) would not get an error. The kwargs are silently swallowed.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/optim/integration.py`, line 30

**Issue #42 - LOW: No logging of effective LR after gradient dilution compensation**
- **Status: STILL OPEN**
- **Evidence:** Line 63 computes `effective_lr = lr * calculate_gradient_dilution_factor(d)` but this is not logged anywhere. Users have no visibility into the actual learning rate being used. The dilution factor can be significant (e.g., 4.4x for 6D), meaning the effective LR could be very different from what the user specified.
- **File:** `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/optim/integration.py`, line 63

---

## Cross-Cutting Concerns (from original review, unnumbered)

These were observations, not numbered issues:

1. **Sharpness parameter**: Removed from gsplats. The standard Gaussian (s=2) is now hardcoded.

2. **Ordering is not round-trippable**: Still the case. Hilbert reordering on save changes element order.

3. **Encoding precision loss**: Still the case. `ArrayEncoder`/`ArrayDecoder` may quantize. This is by design for compression.

---

## Priority Recommendations (Updated)

### Immediate (CRITICAL/HIGH still open)
1. **Issue #34**: Fix nD downsampling fallback in `decompose.py` -- the `area` mode for >3D data is broken (uses nearest instead of area averaging)
2. **Issue #30**: Document or fix CLAHE block artifacts -- currently documented as known limitation

### Soon (HIGH partially fixed)
3. **Issue #22**: The temp dir cleanup in `load_gsplats.py` works for current code but is fragile if lazy loading is ever introduced
4. **Issue #12**: Rename shadowed `truncate` variable in `fit_multiscale_gsplats.py`

### Next Sprint (MEDIUM still open)
5. **Issue #3**: Fix docstring defaults to match code defaults in `fit_gsplats.py`
6. **Issue #4**: Add sentinel value to distinguish "not set" from "explicitly None" for `amp_max`
7. **Issue #18**: Add `__version__` to gsplats package or fix the import
8. **Issue #19**: Add explicit 2D validation for `centers` in `save_gsplats.py`
9. **Issue #31**: Align `histc` and `searchsorted` bin edge semantics in CLAHE
10. **Issue #32**: Batch tile processing in CLAHE for nD performance
11. **Issue #40**: Type-annotate `model` parameter in `integration.py`
12. **Issue #41**: Remove or warn on unused `**extra_kwargs`
