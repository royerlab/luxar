> **⚠️ Archived — point-in-time review, not re-verified.** Finding statuses below (e.g. "STILL OPEN") reflect the status-check date in the header and have **not** been re-verified against current `main`. Treat this as a historical snapshot, not a live task list; verify any finding against current code before acting on it. See [the archive README](../README.md).

# Seeds Sub-Package Review Status Report

**Original Review**: `gsplats/seeds/REVIEW_REPORT.md` (2026-02-24, commit `164c16c`)
**Status Check Date**: 2026-02-25
**Reviewer**: Claude Opus 4.6 (1M context)
**Branch**: `feat/comprehensive-codebase-review-improvements`

---

## Summary

| Status | Count | Percentage |
|--------|-------|------------|
| FIXED | 6 | 24% |
| PARTIALLY FIXED | 4 | 16% |
| STILL OPEN | 15 | 60% |
| **Total** | **25** | |

6 of 25 issues have been fully resolved. 4 more are partially addressed. 15 remain open.

---

## Issue-by-Issue Analysis

### CRITICAL (1)

#### GPU1: GPU Sobel uses `[-1,0,1]` instead of full separable Sobel with perpendicular smoothing -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/gpu_ops.py`, line 228

**Evidence**: The GPU Sobel implementation still uses only the differentiation kernel `[-1, 0, 1]` without perpendicular smoothing. The CPU path uses `ndi.sobel()` which applies a full separable Sobel kernel (differentiation + smoothing along perpendicular axes). Current code at line 228:

```python
sobel_kernel = torch.tensor([-1.0, 0.0, 1.0], device=V_tensor.device)
```

The docstring at lines 203-225 still describes it as "Sobel gradient" but only uses central differences. The suggested fix (applying `smooth_kernel = [1.0, 2.0, 1.0] / 4.0` along perpendicular axes) has not been implemented. GPU and CPU paths will produce different results for the same input.

---

### HIGH (5)

#### M2: `_dedupe_with_scales_and_energies` is O(N^2) without KD-tree -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/multiscale_decomposition.py`, lines 279-287

**Evidence**: The function still uses the naive O(N^2) greedy algorithm. Lines 279-287:

```python
for i in range(len(coords_sorted)):
    if not kept_mask[i]:
        continue
    diffs = coords_sorted[i + 1 :] - coords_sorted[i]
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    nearby = distances < min_distance
    kept_mask[i + 1 :][nearby] = False
```

No KD-tree is used. The function has not been refactored to use `dedupe_farthest_first` from `utils.py`.

#### E1: KD-tree rebuilt after every accepted point -- PARTIALLY FIXED

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/edges.py`, lines 374-377

**Evidence**: The code now has an optimization for the last few candidates (lines 356-361) using a simple distance check instead of rebuilding the tree. However, the main loop still rebuilds the KD-tree after every accepted point (line 377):

```python
if n_selected > 0 and n_samples - n_selected > 3:
    tree = cKDTree(selected_array[:n_selected])
```

Compare with `utils.py` line 333 which rebuilds every 100 seeds. The early-termination optimization helps for the tail end, but the core O(N * M log M) complexity from per-acceptance rebuilds remains for the bulk of the selection.

#### E2: KD-tree rebuild logic inconsistent with `utils.py` pattern -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/edges.py` vs `utils.py`

**Evidence**: `utils.py` (lines 332-335) rebuilds every 100 seeds and checks recently-added points explicitly:

```python
if n_selected % 100 == 0:
    tree = cKDTree(selected_array[:n_selected])
    last_tree_rebuild_at = n_selected
```

`edges.py` (line 377) rebuilds after every acceptance:

```python
tree = cKDTree(selected_array[:n_selected])
```

The two files still use fundamentally different strategies. No unification has occurred.

#### GPU3: `squeeze()` returns scalar for N=1 input -- FIXED

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/gpu_ops.py`, line 453

**Evidence**: The code now uses `reshape(-1)` instead of `squeeze()`. Line 452-453:

```python
# Use reshape(-1) instead of squeeze() to handle N=1 correctly
return sampled.reshape(-1)
```

This correctly handles the N=1 case.

#### GR2: Amplitude threshold compared against pre-scaled amplitudes -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/grid.py`, lines 221-229

**Evidence**: Amplitudes are still scaled before threshold comparison. Lines 221-229:

```python
amplitudes = (
    _sample_amplitudes(V, grid_coords, device=device) * SEED_AMPLITUDE_SCALE
)

# Apply intensity threshold
if exclude_below is not None:
    mask = amplitudes >= exclude_below
elif exclude_below_percentile is not None:
    threshold = np.percentile(V, exclude_below_percentile)
    mask = amplitudes >= threshold
```

The `exclude_below` threshold is still compared against amplitudes already scaled by `SEED_AMPLITUDE_SCALE` (0.9x). A user passing `exclude_below=1.0` effectively filters at `1.0 / 0.9 ~ 1.11` in the original image. The threshold should be applied before scaling.

---

### MEDIUM (11)

#### G1: `_empty_gsplatdata(2)` hardcoded to 2D -- FIXED

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/generate.py`

**Evidence**: The `_combine_gsplatdata` function now accepts an `ndim` parameter (line 410) and passes it through. Line 300:

```python
return _combine_gsplatdata(results, min_distance, device=device, ndim=V.ndim)
```

And line 421:

```python
if len(results) == 0:
    return _empty_gsplatdata(ndim)
```

Additionally, the early return at line 295 also uses `V.ndim`:

```python
if len(results) == 0:
    return _empty_gsplatdata(V.ndim)
```

This issue is fully resolved.

#### G6: `verbose`/`device` consumed inconsistently via `.get()` and routing -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/generate.py`, lines 181-182

**Evidence**: Lines 181-182 still use `.get()` (which leaves them in kwargs), and then the routing loop at lines 226-236 also picks them up:

```python
verbose = kwargs.get("verbose", False)  # Don't pop - methods may use it
device = kwargs.get("device", None)  # Don't pop - methods may use it
```

The comments explain the intent ("methods may use it"), but both `verbose` and `device` are still in `passthrough_params` (lines 221-222) and also in the method-specific param sets (e.g., `decomposition_params`). The dual treatment remains confusing.

#### M1: Coordinate mapping adds 0.5 offset at scale=1 -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/multiscale_decomposition.py`, line 201

**Evidence**: The formula is unchanged:

```python
seeds_full_res = peaks.astype(float) * scale_factor + scale_factor / 2.0
```

For `scale_factor=1`, this still shifts peaks by 0.5 voxels. No special-case handling for scale=1 has been added.

#### M3: Seeds may lie outside image bounds -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/multiscale_decomposition.py`, line 201

**Evidence**: The coordinate mapping at line 201 can produce coordinates that exceed image boundaries. The clipping at line 237 only clips coordinates for the amplitude lookup:

```python
seeds_int = np.clip(np.round(seeds).astype(int), 0, np.array(V.shape) - 1)
```

But the actual `seeds` array returned at line 249 is not clipped:

```python
centers=seeds.astype(np.float32),
```

Seeds outside image bounds are still returned.

#### M4: Mutable default argument for `scales` parameter -- FIXED

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/multiscale_decomposition.py`, line 25

**Evidence**: The parameter now uses `Optional[List[int]] = None` with a guard clause at lines 92-93:

```python
scales: Optional[List[int]] = None,
...
if scales is None:
    scales = [1, 2, 4, 8, 16, 32, 64]
```

This correctly avoids the mutable default argument pitfall.

#### GR1: `np.arange` float step may produce extra point -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/grid.py`, line 182

**Evidence**: The code is unchanged:

```python
dim_coords = np.arange(start, end + spacing_arr[dim] / 2.0, spacing_arr[dim])
```

No explicit clipping after `arange` has been added. The `end + spacing_arr[dim] / 2.0` upper bound can still include an extra point due to floating-point rounding.

#### GR3: Cross-module import of private `_sample_amplitudes` -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/grid.py`, line 219

**Evidence**: The import is still from `edges.py`:

```python
from luxar.gsplats.seeds.edges import _sample_amplitudes
```

`_sample_amplitudes` has not been moved to `utils.py`.

#### U1: `_dedupe_simple` marks self as "used" (confusing) -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/utils.py`, lines 385-387

**Evidence**: The code is unchanged:

```python
diffs = coords_sorted - c
dist2 = np.sum(diffs**2, axis=1) if coords_sorted.ndim > 1 else diffs**2
used |= dist2 < (min_distance**2)
```

The distance computation includes self (distance=0), marking `used[i] = True`. No clarifying comment has been added. While functionally correct (the seed is already selected), it remains confusing.

#### U2: `local_maxima` returns all pixels on flat plateaus -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/utils.py`, lines 191-192

**Evidence**: The peak criteria is unchanged:

```python
peaks_mask = (img == max_f) & (img >= thresh)
```

On flat regions where all values are identical, every voxel matches this condition. No documentation warning or handling has been added.

#### GPU4: Memory check uses total memory, not free memory -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/gpu_ops.py`, lines 528-529

**Evidence**: Still uses `props.total_memory`:

```python
props = torch.cuda.get_device_properties(device_id)
available = props.total_memory * 0.8  # Use 80% as safety margin
```

`torch.cuda.mem_get_info(device_id)` is not used. The check does not account for memory already in use.

#### X1: No `min_distance > 0` validation in `edges.py` -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/edges.py`, line 22

**Evidence**: `seed_from_edges` validates `edge_threshold_rel` (line 108) but does not validate `min_distance`. There is no `min_distance <= 0` check. Compare with `multiscale_decomposition.py` line 125-126 which does validate:

```python
if min_distance <= 0:
    raise ValueError("min_distance must be positive")
```

#### X2: Amplitude sampling inconsistent across methods -- STILL OPEN

**Evidence**: The inconsistency persists:
- `multiscale_decomposition.py` line 237-238: Uses nearest-neighbor indexing (integer coordinates) for amplitude lookup:
  ```python
  seeds_int = np.clip(np.round(seeds).astype(int), 0, np.array(V.shape) - 1)
  amplitudes = V[tuple(seeds_int.T)].astype(np.float32) * SEED_AMPLITUDE_SCALE
  ```
- `edges.py` and `grid.py`: Use bilinear interpolation via `_sample_amplitudes` which calls `ndi.map_coordinates(..., order=1)`.

---

### LOW (8)

#### G2: Duplicate `_empty_gsplatdata` function -- STILL OPEN

**Evidence**: `_empty_gsplatdata` is defined identically in both:
- `generate.py` lines 303-311
- `edges.py` lines 169-177

Neither has been refactored to a shared location in `utils.py`.

#### G3: Stale try/except for edges import -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/generate.py`, lines 280-291

**Evidence**: The try/except block with the comment "Lazy import since edges.py may not exist yet" still exists:

```python
# Lazy import since edges.py may not exist yet
try:
    from luxar.gsplats.seeds.edges import seed_from_edges
    result = seed_from_edges(V, **edges_kwargs)
    results.append(result)
except ImportError:
    import warnings
    warnings.warn(
        "Edge seeding not available yet (edges.py not implemented)",
        UserWarning,
    )
```

The comment is stale since `edges.py` is fully implemented.

#### G4: Redundant import in `_auto_combine` -- STILL OPEN

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/generate.py`, line 357

**Evidence**: `seed_from_edges` is imported again inside `_auto_combine` with a try/except at line 357:

```python
from luxar.gsplats.seeds.edges import seed_from_edges
```

This is redundant since `seed_from_edges` is already imported in the elif branch at line 281, and `edges.py` clearly exists.

#### G5: Emoji in verbose output -- PARTIALLY FIXED

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/generate.py`, lines 365, 386, 402

**Evidence**: Emoji characters are still present in verbose output at lines 365, 386, and 402:

```python
aprint(f"✓ Generated {len(seeds_edges.centers)} edge seeds")
...
aprint(f"✓ Generated {len(seeds_grid.centers)} grid seeds")
...
aprint(f"✓ Final: {len(result.centers)} seeds after deduplication")
```

Per CLAUDE.md: "Only use emojis if the user explicitly requests it." The checkmark character is still used.

#### E3: `_sample_amplitudes` GPU fallback warning has wrong stacklevel -- PARTIALLY FIXED

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/edges.py`, line 277

**Evidence**: The stacklevel is still `2`:

```python
warnings.warn(
    f"GPU interpolation not supported for {V.ndim}D volumes. Using CPU.",
    RuntimeWarning,
    stacklevel=2,
)
```

This points to `_sample_amplitudes` itself rather than the public API caller. It arguably should be `3` to reach the public function that called `_sample_amplitudes`. However, this is a minor concern as the warning message is sufficiently descriptive.

#### U3: `combine_seeds` may fail on zero-length 1D arrays -- PARTIALLY FIXED

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/utils.py`, lines 467-470

**Evidence**: The code now has a guard for `arr is not None` and accesses `arr.shape[1]`, but it does not guard against 1D arrays explicitly:

```python
for arr in candidate_arrays:
    if arr is not None:
        return cast(np.ndarray, np.zeros((0, arr.shape[1]), dtype=float))
return cast(np.ndarray, np.zeros((0, 2), dtype=float))  # Default to 2D
```

If all arrays have shape `(0,)` (1D), `arr.shape[1]` will still raise `IndexError`. No `arr.ndim >= 2` guard exists.

#### U4: Function name `dedupe_farthest_first` misleading -- FIXED

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/utils.py`, line 210

**Evidence**: While the function is still named `dedupe_farthest_first`, the docstring (lines 221-231) now clearly explains it uses "simple greedy selection" and explicitly states:

```
This simple greedy approach is ~50x faster than farthest-first selection
and produces equivalent results for Gaussian splatting, since the optimizer
will adjust positions during fitting anyway.
```

The name is historical, but the documentation now clarifies the actual algorithm. Considered fixed by documentation.

#### GPU5-7: Minor GPU issues -- STILL OPEN

**Evidence**:
- **GPU5** (line 125): `50**3` threshold in `should_use_gpu` is unchanged and still undocumented in terms of empirical justification.
- **GPU6** (lines 480-491): Memory estimates are unchanged. The multipliers (5x, 3x, 2x) are rough and ndim-independent.
- **GPU7** (lines 322-325): `top_k` in `local_maxima_gpu` uses `torch.topk` which returns in descending value order, while CPU `local_maxima` uses `np.argsort(vals)[-top_k:]` which returns in ascending index order. The ordering inconsistency persists.

#### X3: `device` parameter accepted but ignored in deduplication -- FIXED

**File**: `/home/royer/PycharmProjects/luxar/packages/luxar/src/luxar/gsplats/seeds/utils.py`, line 214

**Evidence**: The function now has clear documentation explaining that `device` is intentionally ignored (lines 243-245):

```python
device : str, optional
    Device parameter (ignored for deduplication).
    Deduplication always uses CPU with KD-tree as it's fastest for typical
    seed counts. CPU completes 16K seeds in ~0.77s vs 27s before optimization.
```

And lines 276-279:

```python
# Note: GPU deduplication was removed because CPU with KD-tree is faster
# for typical seed counts (<100K). The GPU version had O(N^2) complexity
# and significant transfer overhead. CPU completes 16K seeds in ~0.77s.
# If device is specified, we ignore it for deduplication and use CPU.
```

The behavior is now explicitly documented as a deliberate design choice.

---

## Summary by Severity

### CRITICAL (1 issue)
- **0 FIXED, 0 PARTIALLY, 1 OPEN** (0% resolved)
- GPU1 (GPU Sobel kernel mismatch) remains the most important unresolved issue.

### HIGH (5 issues)
- **1 FIXED, 1 PARTIALLY, 3 OPEN** (20% fully resolved)
- GPU3 (squeeze for N=1) is fixed.
- E1 (KD-tree rebuild) has a partial optimization for the tail end.
- M2, E2, GR2 remain open.

### MEDIUM (11 issues)
- **2 FIXED, 0 PARTIALLY, 9 OPEN** (18% fully resolved)
- G1 (hardcoded 2D) and M4 (mutable default) are fixed.
- All others remain open.

### LOW (8 issues)
- **3 FIXED, 3 PARTIALLY, 2 OPEN** (38% fully resolved)
- U4 (misleading name) and X3 (device ignored) are fixed by documentation.
- G5 (emoji), E3 (stacklevel), U3 (1D arrays) are partially addressed.
- G2, G3, G4, GPU5-7 remain open.

---

## Recommended Next Steps (Priority Order)

1. **GPU1** (CRITICAL): Implement full separable Sobel kernel on GPU. This is the only correctness-affecting issue and impacts all GPU users.
2. **GR2** (HIGH): Apply threshold before amplitude scaling. One-line reorder fix.
3. **E1/E2** (HIGH): Unify KD-tree rebuild strategy in `edges.py` to match `utils.py` (rebuild every 100 points).
4. **M2** (HIGH): Replace O(N^2) dedup in `multiscale_decomposition.py` with `dedupe_farthest_first` from `utils.py`.
5. **X1** (MEDIUM): Add `min_distance > 0` validation to `edges.py`.
6. **GR3** (MEDIUM): Move `_sample_amplitudes` to `utils.py`.
7. **G2** (LOW): Consolidate duplicate `_empty_gsplatdata` into `utils.py`.
8. **G3/G4** (LOW): Remove stale try/except guards for edges import.
