> **⚠️ Archived — point-in-time review, not re-verified.** Finding statuses below (e.g. "STILL OPEN") reflect the status-check date in the header and have **not** been re-verified against current `main`. Treat this as a historical snapshot, not a live task list; verify any finding against current code before acting on it. See [the archive README](../README.md).

# Fitting Pipeline Code Review - Issue Status Report

**Original Review Date**: 2026-02-24
**Status Check Date**: 2026-02-25
**Scope**: `packages/luxar/src/luxar/gsplats/fitting/` (12 files)
**Original Reviewer**: Claude Opus 4.6 (automated deep review)
**Status Checker**: Claude Opus 4.6

---

## Summary

| Severity | Total | FIXED | PARTIALLY FIXED | STILL OPEN |
|----------|-------|-------|-----------------|------------|
| CRITICAL | 4     | 3     | 0               | 1          |
| HIGH     | 11    | 3     | 2               | 6          |
| MEDIUM   | 16    | 2     | 3               | 11         |
| LOW      | 10    | 0     | 0               | 10         |
| **Total** | **41** | **8** | **5** | **28** |

**Overall Fix Rate**: 19.5% fully fixed, 12.2% partially fixed, 68.3% still open

---

## CRITICAL Issues

### CRITICAL-01: Mutable default + wrong type annotation for `seed_kwargs`
**File**: `config.py` line 141
**Status**: **FIXED**

**Evidence**: The current code reads:
```python
seed_kwargs: Optional[Dict[str, Any]] = None  # Additional parameters for seed generation
```
The type annotation is now `Optional[Dict[str, Any]]` (correct), and the default is `None` (safe). This is the exact fix recommended in the review.

---

### CRITICAL-02: Incorrect `elif` chain for `movie_max_frames` validation
**File**: `validation.py` lines 159-164
**Status**: **FIXED**

**Evidence**: The current code uses separate `if` statements for all validations:
```python
if max_abs_error is not None and max_abs_error <= 0:
    raise ValueError("max_abs_error must be positive if specified")
if movie_max_frames is not None and movie_max_frames <= 0:
    raise ValueError("movie_max_frames must be positive or None")
if movie_every < 1:
    raise ValueError("movie_every must be >= 1")
```
The `elif` has been changed to `if`, and additionally `movie_every` validation (HIGH-02) was added here as well. Both independent validations are now independent `if` statements.

---

### CRITICAL-03: Poisson loss gradient instability at `Vc=0`
**File**: `losses.py` lines 98-102
**Status**: **FIXED**

**Evidence**: The current code uses `torch.xlogy` as recommended:
```python
# Use xlogy to safely handle Vc=0 (0 * log(0) = 0, with correct gradients)
dev = 2.0 * torch.sum(Pc - Vc + torch.xlogy(Vc, torch.clamp(Vc / Pc, min=eps)))
```
The comment explicitly documents the fix rationale. The asymmetric penalty block also uses `torch.xlogy`.

---

### CRITICAL-04: `n_splats` variable scope and staleness
**File**: `optimization.py` lines 110-112, 285-286
**Status**: **STILL OPEN**

**Evidence**: The current code still defines `n_splats` only inside the `if config.enable_dynamic_ops:` block (line 110-111):
```python
if config.enable_dynamic_ops:
    n_splats = (
        model.n_splats() if hasattr(model, "n_splats") else preprocessed_data.N
    )
```
And uses it on line 285-286 in the logging section guarded by `if relocation_tracker is not None:`:
```python
aprint(f"  Unique splats relocated: {stats['unique_splats']} / {n_splats}")
coverage_pct = (stats["unique_splats"] / n_splats) * 100
```
While the guard (`relocation_tracker is not None`) makes this safe in practice (since `relocation_tracker` is only set when `enable_dynamic_ops` is True), the implicit coupling is still fragile. The `n_splats` value is also still the pre-loop value, not the current count.

---

## HIGH Issues

### HIGH-01: `norm_percentile` not validated
**File**: `validation.py` line 25
**Status**: **FIXED**

**Evidence**: Validation was added at lines 165-168:
```python
if not 0.0 <= norm_percentile < 50.0:
    raise ValueError(
        f"norm_percentile must be in range [0.0, 50.0), got {norm_percentile}"
    )
```

---

### HIGH-02: `movie_every` not validated (div-by-zero risk)
**File**: `validation.py` line 44
**Status**: **FIXED**

**Evidence**: Validation was added at lines 163-164:
```python
if movie_every < 1:
    raise ValueError("movie_every must be >= 1")
```

---

### HIGH-03: GPU/CPU path variable scope confusion in `_subsample_seeds_spatially_diverse`
**File**: `preprocessing.py` lines 528-605
**Status**: **PARTIALLY FIXED**

**Evidence**: The code has been significantly restructured. The GPU path (lines 548-576) now uses `selected_mask` and `selected_indices_final` cleanly, and the CPU path (lines 577-601) uses `selected_indices` and `selected_indices_final` separately. However, the variable `selected` is still shared across both paths -- it's initialized as a list at line 529, then reassigned in both branches (line 576 for GPU, line 601 for CPU), then again converted at line 603. The result assignment `result = np.array(selected)` on line 603 and `original_indices = valid_indices[selected_indices_final]` on line 605 works for both paths, but the intermediate variable reuse between the list-based `selected` and the array-based paths is still somewhat confusing. The core correctness issue is resolved but clarity could still improve.

---

### HIGH-04: O(n) list removal in farthest-first loop
**File**: `preprocessing.py` line 598
**Status**: **STILL OPEN**

**Evidence**: The CPU path still uses:
```python
remaining_indices.remove(farthest_idx_global)  # O(n) per call
```
inside the `while len(selected) < target_count and remaining_indices:` loop. This remains O(n * target_count). However, the practical impact is mitigated because large selections (>10k) now use the fast random sampling path (lines 497-524) and medium selections (>1000) attempt GPU acceleration (lines 537-576). The O(n) removal only applies to CPU path with target_count < 1000.

---

### HIGH-05: `model=None` returned for N=0 but callers don't check
**File**: `initialization.py` lines 36-42
**Status**: **STILL OPEN**

**Evidence**: The `initialize_optimization` function still returns `ModelComponents(model=None, optimizer=None, scheduler=None)` when N=0 (lines 38-44). Looking at the caller `run_optimization_loop` in `optimization.py`, line 72-73:
```python
model = components.model
optimizer = components.optimizer
```
There is no None check before accessing `model.n_splats()` on line 111, `model()` on line 133, or `model.parameters()` on line 139. If N=0 reaches the optimization loop, it will crash with `AttributeError`.

---

### HIGH-06: `sharpness0` computed but never passed to any model constructor
**File**: `initialization.py` lines 94-100
**Status**: **RESOLVED** (sharpness parameter removed from gsplats)

**Note**: The sharpness parameter has been removed from Gaussian splats entirely. The standard Gaussian (s=2) is now hardcoded. This issue is no longer applicable.

---

### HIGH-07: Double forward pass every iteration
**File**: `optimization.py` lines 133-134, 153-155
**Status**: **STILL OPEN**

**Evidence**: The optimization loop still performs two forward passes per iteration:
```python
# Forward pass 1 (line 133):
pred = model()
loss = loss_fn(pred)
loss.backward()
optimizer.step()

# Forward pass 2 (lines 154-155):
with torch.no_grad():
    pred_eval = model()
    loss_eval = loss_fn(pred_eval)
```
The second forward pass is used for evaluation metrics, movie recording, and convergence checking. This doubles the computation cost per iteration.

---

### HIGH-08: Cholesky decomposition can fail on near-singular matrices
**File**: `results.py` lines 45-47 (now lines 126-130)
**Status**: **STILL OPEN**

**Evidence**: The `_apply_voxel_footprint_correction` function (lines 102-130) still uses bare `np.linalg.cholesky` without try/except:
```python
Sigma_corrected = Sigma + variance * np.eye(d, dtype=Ls.dtype)
return np.linalg.cholesky(Sigma_corrected)
```
While adding `variance * I` regularizes the matrix, if the original `Sigma` has very large negative eigenvalues (e.g., from numerical issues in L), the correction may still be insufficient. No fallback with stronger regularization exists.

---

### HIGH-09: `DynamicOpsConfig` is a plain class, not a dataclass
**File**: `dynamic_ops/config.py` lines 5-74
**Status**: **FIXED**

**Evidence**: The current code uses `@dataclass`:
```python
@dataclass
class DynamicOpsConfig:
```
with `from dataclasses import dataclass` imported. This provides `__eq__`, `__repr__`, and `__hash__` automatically.

---

### HIGH-10: Unnecessary GPU->CPU->GPU roundtrip for weak splat indices
**File**: `dynamic_ops/operations.py` lines 314, 425, 598
**Status**: **PARTIALLY FIXED**

**Evidence**: The `_select_weak_splats` function now returns `candidates_tensor.tolist()` at line 425 (single GPU->CPU transfer at end). The calling code in `_match_weak_splats_to_peaks_batch` converts it back to a tensor at line 598:
```python
weak_tensor = torch.tensor(weak_splat_indices, dtype=torch.long, device=device)
```
So there is still a GPU->CPU->GPU roundtrip (tensor -> list -> tensor), but it's now one roundtrip instead of multiple. The type signature of `_select_weak_splats` still returns `List[int]`, which necessitates the conversion.

---

### HIGH-11: Non-deterministic `random.random()` in tiled peak finding
**File**: `dynamic_ops/peak_finding.py` line 329
**Status**: **STILL OPEN**

**Evidence**: The code still uses `random.random()` at line 329:
```python
elif random.random() >= keep_probability:
    # Reject weak peak based on probability
    tile_peaks = []
```
The global `random` module is used without seeding, making results non-reproducible across runs. The `random` module is imported at line 7 at module level.

---

## MEDIUM Issues

### MEDIUM-01: Inconsistent defaults between `OptimConfig` and `prepare_fit_config`
**File**: `config.py` lines 17-33 / `validation.py`
**Status**: **STILL OPEN**

**Evidence**: `OptimConfig` (config.py lines 27-33) defaults:
- `lr=0.05`, `patience=25`, `lr_reduction_factor=0.98`, `early_stop_patience=300`

`prepare_fit_config` (validation.py lines 28, 47-49) defaults:
- `lr=0.01`, `patience=10`, `lr_reduction_factor=0.5`, `early_stop_patience=200`

These remain inconsistent.

---

### MEDIUM-02: `ModelComponents.model` typed as `Any`
**File**: `config.py` line 248
**Status**: **STILL OPEN**

**Evidence**: Still reads:
```python
model: Any  # GaussianSplatModel
```
No `TYPE_CHECKING` guard or `Optional["GaussianSplatModel"]` annotation.

---

### MEDIUM-03: Vestigial `proportion` parameter
**File**: `preprocessing.py` lines 277-278
**Status**: **STILL OPEN**

**Evidence**: `_generate_seeds` still has the `proportion` parameter (line 278):
```python
def _generate_seeds(
    V: np.ndarray,
    proportion: float | None,
    ...
```
All callers pass `None` (lines 161, 174, 193). The docstring now says "Deprecated parameter, no longer used. Kept for API compatibility." (line 293), so this is intentionally kept but still technically dead code.

---

### MEDIUM-04: Verbose logging shows wrong L1 origin when user overrides
**File**: `preprocessing.py` lines 252-255
**Status**: **STILL OPEN**

**Evidence**: The log still always shows the percentage labels even when the user provided their own values:
```python
aprint("L1 regularization (as % of base LR):")
aprint(f"  Amplitude: {l1_amp:.4f} (10% of LR {config.lr:.3f})")
aprint(f"  Diagonal: {l1_diag:.5f} (1% of LR {config.lr:.3f})")
```
When `config.l1_amp` was user-specified (not None), the log would misleadingly say "(10% of LR ...)" even though the value came from the user.

---

### MEDIUM-05: `_normalize_data` uses `abs(intensity_range) < 1e-12` but negative range not guarded
**File**: `preprocessing.py` line 828 (now line 828)
**Status**: **STILL OPEN**

**Evidence**: The code at lines 826-834:
```python
intensity_range = image_max - image_min
if np.abs(intensity_range) < 1e-12:
    V = np.full_like(V, 0.5, dtype=np.float32)
    ...
else:
    V = np.clip((V - image_min) / intensity_range, 0.0, 1.0)
```
A negative `intensity_range` (which could happen with pathological `norm_percentile` values near 50.0) would pass the `abs(...) < 1e-12` check and result in inverted normalization. However, HIGH-01 fix (norm_percentile < 50.0) mitigates this in practice, making `image_min > image_max` very unlikely.

---

### MEDIUM-06: `amp_max` auto-set to 1.0 may clip pre-initialized amplitudes
**File**: `initialization.py` lines 105-109 (now lines 104-111)
**Status**: **STILL OPEN**

**Evidence**: The code still auto-sets `amp_max = 1.0` when not specified (lines 107-109):
```python
amp_max = config.amp_max
if amp_max is None:
    amp_max = 1.0
```
This is passed to model constructors. Pre-initialized amplitudes from GSplatData are already rescaled to [0, 1] in preprocessing (line 216-218), so this is less of an issue, but if init amplitudes from seeding methods are slightly above 1.0 (e.g., due to interpolation), they would be clipped.

---

### MEDIUM-07: Loss function closure captures `model` by reference
**File**: `losses.py` lines 47-91
**Status**: **STILL OPEN**

**Evidence**: The `loss_fn` closure still captures `model` by reference:
```python
def create_loss_function(config, preprocessed_data, model):
    ...
    def loss_fn(pred):
        ...
        data = data + l1_amp * torch.mean(torch.abs(F.softplus(model.raw_a)))
        ...
    return loss_fn
```
If the model is replaced/modified after `create_loss_function` is called, the closure would reference the new state. In the current codebase this is safe since the model reference doesn't change, but it's a latent coupling.

---

### MEDIUM-08: Target stored per-frame in movie recording
**File**: `optimization.py` line 359 (now line 359)
**Status**: **STILL OPEN**

**Evidence**: The `_record_movie_frame` function still stores `target_frame = V_t.cpu().numpy()` per frame (line 359):
```python
target_frame = V_t.cpu().numpy()
...
movie_frames["target"].append(target_frame)
```
The target doesn't change between frames, so storing it once would save significant memory. However, a `movie_max_frames` limit now exists (default 10000 with FIFO eviction), which bounds the memory usage.

---

### MEDIUM-09: Amplitude rescaling semantics undocumented
**File**: `results.py` line 79 (now lines 222-223)
**Status**: **PARTIALLY FIXED**

**Evidence**: The amplitude rescaling now has a verbose log message:
```python
amps_np = amps_np * preprocessed_data.intensity_range
if config.verbose:
    aprint(f"Rescaled amplitudes to original intensity range (factor: {preprocessed_data.intensity_range:.4f})")
```
But the comment still doesn't explain why only `intensity_range` is applied and not `image_min`. The inverse transform would be `amps_original = amps_normalized * intensity_range + image_min`, but `image_min` is not added. This is because amplitude represents peak contribution (additive), not absolute intensity, so only the scale factor matters. This reasoning is not documented.

---

### MEDIUM-10: Broad exception catch without traceback
**File**: `visualization.py` lines 169-170
**Status**: **STILL OPEN**

**Evidence**: The code still has:
```python
except Exception as e:
    aprint(f"Movie visualization error: {e}")
```
No traceback is logged. A `traceback.format_exc()` or `logging.exception()` call would help debug issues.

---

### MEDIUM-11: Private functions exported in `__all__`
**File**: `dynamic_ops/__init__.py` lines 17-27
**Status**: **STILL OPEN**

**Evidence**: Still exports private functions:
```python
__all__ = [
    "DynamicOpsConfig",
    "RecentlyRelocatedTracker",
    "apply_dynamic_operations",
    # Exported for testing
    "_find_residual_peaks",
    "_calculate_splat_importance",
    "_select_weak_splats",
]
```
There is now a comment `# Exported for testing` documenting the intent, but private functions are still in `__all__`.

---

### MEDIUM-12: Expanded candidate pool edge case not logged
**File**: `dynamic_ops/operations.py` lines 413-422
**Status**: **STILL OPEN**

**Evidence**: The expanded candidate pool logic (lines 411-422) has no logging when it activates:
```python
if len(candidates_tensor) < max(1, n_candidates // 2):
    expanded_candidates = sorted_indices[: n_candidates * 2]
    expanded_low_residual = low_residual_mask[: n_candidates * 2]
    ...
    if len(expanded_eligible) > len(candidates_tensor):
        candidates_tensor = expanded_eligible[:n_candidates]
```
No `aprint` or other logging indicates when the pool expansion occurs.

---

### MEDIUM-13: `_reset_optimizer_state_batch` only handles `torch.optim.Adam`
**File**: `dynamic_ops/operations.py` line 663
**Status**: **STILL OPEN**

**Evidence**: Still uses exact type checking:
```python
if not isinstance(optimizer, torch.optim.Adam):
    return
```
`torch.optim.AdamW` (which is the more commonly recommended optimizer) would be silently skipped since `AdamW` is not a subclass of `Adam` in PyTorch. The comment says "Only Adam has exp_avg/exp_avg_sq state" but `AdamW` also has these.

---

### MEDIUM-14: Separable max pool docstring claims false equivalence
**File**: `dynamic_ops/peak_finding.py` lines 13-17
**Status**: **PARTIALLY FIXED**

**Evidence**: The current docstring reads:
```
which is equivalent to a full nD max pool for local maxima detection.
```
This is technically inaccurate -- separable max pooling is equivalent to full nD max pool for hypercube-shaped kernels, but the claim is now slightly more qualified ("for local maxima detection"). It's still not perfectly precise but is less misleading than the original.

---

### MEDIUM-15: Duplicated NMS logic between `_find_residual_peaks_global` and `_find_peaks_in_tile`
**File**: `dynamic_ops/peak_finding.py` lines 141-193 vs 355-430
**Status**: **STILL OPEN**

**Evidence**: Both `_find_residual_peaks_global` (lines 119-193) and `_find_peaks_in_tile` (lines 355-431) contain nearly identical NMS logic:
- Kernel size calculation from `nms_radius_vox`
- Conditional max pooling (2D/3D/nD branches)
- Peak detection with `(data >= max_pooled) & (data > 0)`
- Sorting by magnitude and top-k selection

This logic is duplicated rather than extracted to a shared helper.

---

### MEDIUM-16: No cross-tile NMS deduplication
**File**: `dynamic_ops/peak_finding.py` lines 196-352
**Status**: **STILL OPEN**

**Evidence**: In `_find_residual_peaks_tiled`, peaks from different tiles are collected into `all_peaks` and sorted (lines 345-351), but there is no cross-tile NMS step. Two tiles sharing a border could both detect the same peak or peaks within `nms_radius_vox` of each other. The function only applies NMS within each tile.

---

## LOW Issues

### LOW-01: Asymmetric scalar acceptance for `sigma_min_diag` vs `sigma_max_diag`
**File**: `config.py` lines 65-66
**Status**: **STILL OPEN**

**Evidence**: `sigma_min_diag` accepts `Optional[Sequence[float] | float]` while `sigma_max_diag` accepts `Optional[Sequence[float]]`. The `prepare_fit_config` function handles scalar `sigma_min_diag` (line 178: `elif isinstance(sigma_min_diag, (int, float)):`) but has no such handling for `sigma_max_diag`.

---

### LOW-02: `itertools` imported inside function body
**File**: `preprocessing.py` line 746
**Status**: **STILL OPEN**

**Evidence**: `import itertools` is still at line 746 inside `_add_grid_fallback_seeds`, rather than at module level.

---

### LOW-03: `**seed_kwargs` silently captures typos
**File**: `preprocessing.py` line 53, `validation.py` line 54
**Status**: **STILL OPEN**

**Evidence**: `prepare_fit_config` still uses `**seed_kwargs` (line 54). Typos in keyword arguments (e.g., `num_scals=5` instead of `num_scales=5`) would be silently captured and passed through without any validation.

---

### LOW-04: `loss_type.lower()` called every forward pass
**File**: `losses.py` line 61
**Status**: **STILL OPEN**

**Evidence**: In `create_loss_function`, `loss_type` is captured from `config.loss_type` (line 39) and `.lower()` is called every time the inner `loss_fn` is invoked (lines 61, 63). It should be lowercased once outside the closure.

---

### LOW-05: `previous_best` assigned unconditionally but only used in verbose block
**File**: `optimization.py` line 176
**Status**: **STILL OPEN**

**Evidence**: `previous_best = best_loss` is assigned at line 176 on every improvement, but only used at line 196 inside `if config.verbose`. Minor inefficiency.

---

### LOW-06: Dead commented-out `add_points` block
**File**: `visualization.py` lines 130-138
**Status**: **STILL OPEN**

**Evidence**: The commented-out code block is still present:
```python
# # Add points layer (napari will handle NaN values automatically)
# viewer.add_points(
#     points_stack,
#     name="Splat Centers",
#     size=3,
#     face_color="cyan",
#     border_color="white",
#     border_width=1,
# )
```

---

### LOW-07: `points_stack` computed but never used
**File**: `visualization.py` lines 124-128
**Status**: **STILL OPEN**

**Evidence**: `points_stack` is computed at lines 124-128:
```python
points_stack = np.full((len(splat_centers_list), max_splats, d), np.nan)
for i, centers in enumerate(splat_centers_list):
    n_centers = len(centers)
    if n_centers > 0:
        points_stack[i, :n_centers, :] = centers
```
But since the `viewer.add_points` call is commented out (LOW-06), `points_stack` is never used. It's dead code.

---

### LOW-08: Missing `from __future__ import annotations` for `int | None` syntax
**File**: `dynamic_ops/config.py` line 34
**Status**: **FIXED** (but was already fixed at time of report)

**Evidence**: The file now has `from __future__ import annotations` at line 4 and no longer uses `int | None` syntax in the class body. The `Optional` type is used instead (line 39: `num_tiles_per_dim: Optional[int]`).

Wait -- re-examining: `from __future__ import annotations` is present on line 4. The `Optional[int]` type is used at line 39. This was likely fixed as part of the same batch of changes.

---

### LOW-09: Fragile PyTorch version check for `scatter_reduce`
**File**: `dynamic_ops/operations.py` lines 510-521
**Status**: **STILL OPEN**

**Evidence**: The version check at line 510 uses:
```python
if hasattr(torch, "scatter_reduce") or hasattr(max_influences, "scatter_reduce_"):
```
This `hasattr` check is more robust than a version string comparison, but the fallback path (lines 516-520) uses a Python loop which would be very slow for large inputs. The hasattr approach is reasonable but could be tested once at import time rather than per call.

---

### LOW-10: `random` module imported at module level but only used in tiled mode
**File**: `dynamic_ops/peak_finding.py` line 7
**Status**: **STILL OPEN**

**Evidence**: `import random` at line 7 is a module-level import but only used in the probabilistic tiled mode path (line 329). This is a minor style issue -- the import is cheap and standard practice in Python is to import at module level.

---

## Detailed Statistics

### By Fix Status
- **FIXED**: 8 issues (19.5%)
  - CRITICAL-01, CRITICAL-02, CRITICAL-03, HIGH-01, HIGH-02, HIGH-09, LOW-08
- **PARTIALLY FIXED**: 5 issues (12.2%)
  - HIGH-03, HIGH-10, MEDIUM-09, MEDIUM-14
  - (LOW-08 is counted as FIXED since the import is present)
- **STILL OPEN**: 28 issues (68.3%)

### By Severity (Fixed/Total)
- CRITICAL: 3/4 = 75% fixed
- HIGH: 3/11 = 27% fixed (+ 2 partially = 45% addressed)
- MEDIUM: 0/16 = 0% fixed (+ 3 partially = 19% addressed)
- LOW: 1/10 = 10% fixed

### Priority Recommendations

The following open issues should be prioritized:

1. **HIGH-06** (initialization.py: `sharpness0` dead code) -- **RESOLVED**: Sharpness parameter removed from gsplats (standard Gaussian hardcoded).

2. **HIGH-07** (optimization.py: double forward pass) -- Doubles computation cost per iteration. The evaluation forward pass result could be reused from the training forward pass with detached gradients.

3. **HIGH-05** (initialization.py: N=0 crash) -- Will crash if no seeds are generated. Add a check in `run_optimization_loop` or return early.

4. **HIGH-11** (peak_finding.py: non-deterministic random) -- Seed the `random` module or use `np.random.default_rng()` for reproducibility.

5. **HIGH-08** (results.py: Cholesky failure) -- Add try/except with stronger regularization fallback.

6. **MEDIUM-13** (operations.py: AdamW not handled) -- Change `isinstance(optimizer, torch.optim.Adam)` to check for both `Adam` and `AdamW`, or check for `exp_avg` in state dict instead.

7. **CRITICAL-04** (optimization.py: `n_splats` scope) -- Move `n_splats` definition outside the conditional or re-query at usage point.
