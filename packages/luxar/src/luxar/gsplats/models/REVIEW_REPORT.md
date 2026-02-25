# Code Review Report: gsplats/models & rendering Sub-packages

**Reviewer**: Claude Opus 4.6 (1M context)
**Date**: 2026-02-24
**Scope**: All `.py`, `.cu`, `.cuh`, `.h`, `.cpp`, `.metal` files in `models/` and `rendering/`
**Total files reviewed**: 28

## Executive Summary

The codebase is **well-architected and production-quality** overall. The three-backend design (CPU/PyTorch, CUDA, Metal) is clean, with consistent math across backends. The CUDA kernel code is particularly impressive with excellent optimization techniques (tile-based rasterization, warp reduction, shared memory batching, FP16 support). However, there are several issues ranging from numerical subtleties to potential correctness bugs, especially around coordinate convention handling in the Metal backend and a few edge cases in the rendering core.

**Overall Quality Rating**: 7.5/10

| Severity  | Count |
|-----------|-------|
| CRITICAL  | 2     |
| HIGH      | 7     |
| MEDIUM    | 14    |
| LOW       | 12    |

---

## CRITICAL Issues (Fix Immediately)

### 1. Metal backward fallback returns None for all gradients

**File**: `models/gsplats/metal/gsplat_model_metal.py`, lines 397-403

```python
else:
    return (None, None, None, None, None, None, None, None, None)
```

The comment states "Autograd will handle it," but this is **incorrect**. When `MetalSplatFunction.apply()` wraps the fallback `render_gaussians()` call, PyTorch's autograd will call THIS `backward()` method -- it does NOT automatically fall back to the inner graph. The output from `apply()` has `MetalSplatFunction` as its `grad_fn`, not the `render_gaussians` graph. This means training with PyTorch fallback (Metal unavailable but `GaussianSplatModelMetal` is used) produces **zero gradients**, causing silent training failure.

**Suggested fix**: Mirror the CUDA backend's approach at `models/gsplats/cuda/gsplat_model_cuda.py` lines 288-326, which correctly recomputes with gradient tracking in the fallback path.

### 2. `cholesky_to_conic` additive epsilon can produce incorrect results

**Files**:
- `models/gsplats/cuda/gsplat_model_cuda.py`, lines 56-57
- `models/gsplats/metal/gsplat_model_metal.py`, lines 71-76
- `models/gsplats/metal/src/kernels.metal`, lines 125-127

```python
K00 = 1.0 / (L00 + 1e-9)
```

For very small valid diagonal values (e.g., `L00 = 1e-8`), the result is `K00 = 1/(1e-8 + 1e-9) = 9.1e7`, noticeably different from the correct `K00 = 1e8` (~9% error). For negative values (shouldn't happen but could during optimization), `L00 = -1e-9` gives `K00 = Inf`.

**Suggested fix**: Use `torch.clamp(L00, min=1e-8)` instead of `L00 + 1e-9`.

---

## HIGH Issues (Fix Before Next Release)

### 3. Grid cache has no eviction policy
**File**: `models/gsplats/rendering_core.py`, line 87-88. The `_GRID_CACHE` dict grows unboundedly, accumulating GPU memory. Add an LRU eviction policy or maximum size.

### 4. Metal 3D backward kernel lacks SIMD reduction
**File**: `models/gsplats/metal/src/kernels.metal`, line 484. The 3D backward uses "DIRECT ATOMIC WRITE (no SIMD reduction)" while the nD backward kernel states without it performance is "slower than CPU." The 3D kernel should be updated to match.

### 5. Debug `os.environ` probes in Metal backward hot path
**File**: `models/gsplats/metal/gsplat_model_metal.py`, lines 285-341. `import os` and `os.environ.get("DEBUG_METAL_GRADIENTS")` run inside every backward call. Move to module level or remove entirely.

### 6. Metal `cholesky_to_conic` variable naming is documented as misleading
**File**: `models/gsplats/metal/gsplat_model_metal.py`, lines 53, 79-86. Variables named `c_xx` actually correspond to the Z,Z element. Rename to index-based (`c_00`) or coordinate-based (`c_zz`).

### 7. CUDA `CUDASplatFunction` saves unnecessary tensors in fallback path
**File**: `models/gsplats/cuda/gsplat_model_cuda.py`, lines 200-218. When CUDA backend is unavailable, 6 tensors are saved for backward but the fallback recomputes everything. Wasted memory.

### 8. CUDA fast math intrinsics may affect training convergence
**File**: `models/gsplats/cuda/src/math_utils.cuh`, lines 215-227. `__expf`/`__powf` have ~2 ULP error and don't handle denormals. May affect convergence for edge-case splat configurations.

### 9. Global backward kernel duplicates gradient math (DRY violation)
**File**: `models/gsplats/cuda/src/kernels_global.cuh`, lines 226-269. Implements gradient computation inline instead of calling `compute_pixel_gradients<DIM>()` from `reduction_utils.cuh`.

---

## MEDIUM Issues (14 total)

1. **`_build_L` Python loops** (`gsplat_model.py:231,255`): Could use `torch.diag_embed` and advanced indexing
2. **AABB computes radii twice** (`rendering_core.py:290-325`): When `intensity_floor > 0`, initial radii computation is wasted
3. **Variable name collision** (`rendering_wrappers.py:75`): Local `result` shadows parameter `result: GSplatData`
4. **Metal `__init__.py` side effects at import** (`metal/__init__.py:299`): Triggers MPS validation and potential 2-min build
5. **Dead code `_params_cache`** (`gsplat_model_metal.py:498-500`): Initialized but never read
6. **Fragile parameter count check** (`gsplat_model_metal.py:462-473`): Hardcoded `> 13` triggers spurious warning on refactor
7. **`_to_internal_params` CPU round-trip** (`gsplat_model.py:358-362`): Uses NumPy version when PyTorch `stable_inverse_softplus_torch` exists
8. **`inverse_softplus` produces `-inf` for `y=0`** (`inverse_softplus.py:47-52`): Warns but returns `-inf` silently
9. **`sys.path` modification is permanent** (`cuda/__init__.py:30-32`): Could cause import collisions
10. **Metal nD backward L gradient approximation** (`kernels.metal:730`): Division by `1.0f` for near-zero off-diagonals
11. **Forward substitution clamping** (`rendering_core.py:171`): Silently masks degenerate Cholesky factors
12. **`render_to_volume` doesn't use accelerated backends** (`volume_rendering.py:37`): Always uses PyTorch path
13. **CUDA backward shared memory contention** (`kernels_core.cuh:924`): All warps contend on same `si` index
14. **`GaussianSplatModelCUDA` breaks isinstance** (`gsplat_model_cuda.py:333`): Uses composition, so `isinstance(model, GaussianSplatModel)` is False

---

## LOW Issues (12 total)

1. `current_params` allocates shape tensor each call (`gsplat_model.py:299`)
2. `replace_with`/`prune_`/`append_` break optimizer state (`gsplat_model.py:391-441`)
3. Variable name shadowing `s` in nD renderer (`rendering_core.py:575`)
4. `_group_by_box` redundant GPU-CPU sync (`rendering_core.py:230`)
5. `render_gaussians_batched` is a no-op wrapper (`rendering_wrappers.py:153`)
6. `build.py` assumes GPU device 0 (`cuda/build.py:48`)
7. `benchmark_forward` doesn't use `torch.no_grad()` (`benchmark.py:58`)
8. `benchmark.py` mutable default argument (`benchmark.py:152`)
9. Metal MPS tensors not freed during import validation (`metal/__init__.py:247`)
10. Dtype preservation no-op in `inverse_softplus` (`inverse_softplus.py:40-44`)
11. `stable_inverse_softplus_torch` has no input validation (`inverse_softplus.py:74-118`)
12. Overly broad `except Exception` in `lt_solver.py` (`lt_solver.py:55`) -- should be `except AttributeError`

---

## Cross-Backend Consistency

### Numerical Epsilon Inconsistencies

| Location | Epsilon | Purpose |
|----------|---------|---------|
| `rendering_core.py:171` | `1e-6` | Clamp Cholesky diagonal |
| `gsplat_model_cuda.py:56` | `1e-9` | Additive epsilon for L inversion |
| `gsplat_model_metal.py:71` | `1e-9` | Additive epsilon for L inversion |
| `math_utils.cuh:220` | `1e-12f` | Clamp dist_sq in gaussian_intensity |
| `rendering_core.py:400` | `1e-10` | Clamp expo for pow() stability |

These inconsistent values can cause subtle numerical differences when comparing backend outputs. Recommend standardizing or documenting the rationale.

### Coordinate Convention Complexity

The Metal backend's `[Z,Y,X] <-> [X,Y,Z]` conic reordering (handled in Python via `conic[:, [5, 4, 2, 3, 1, 0]]`) is the most error-prone part of the codebase. The CUDA backend avoids this entirely by using natural dimension ordering throughout.

---

## Recommendations

1. **Fix the two CRITICAL issues immediately** -- the Metal fallback backward bug causes silent training failure
2. **Standardize epsilon handling** across all three backends
3. **Add cross-backend numerical comparison tests** for forward AND backward passes
4. **Remove debug probes** from Metal backward -- they belong in test code
5. **Add SIMD reduction** to Metal 3D backward kernel
6. **Add LRU eviction** to the grid cache
7. **Use `stable_inverse_softplus_torch`** in `_to_internal_params` to avoid CPU-GPU round-trips