> **⚠️ Archived — point-in-time review, not re-verified.** Finding statuses below (e.g. "STILL OPEN") reflect the status-check date in the header and have **not** been re-verified against current `main`. Treat this as a historical snapshot, not a live task list; verify any finding against current code before acting on it. See [the archive README](../README.md).

# Code Review Verification Report: gsplats/models & rendering Sub-packages

**Verification Date**: 2026-02-25
**Original Review Date**: 2026-02-24
**Scope**: Verification of all 35 issues from the original review against current codebase
**Verified By**: Claude Opus 4.6 (1M context)

---

## Summary

| Severity | Total | Fixed | Partially Fixed | Still Open |
|----------|-------|-------|-----------------|------------|
| CRITICAL | 2     | 0     | 0               | 2          |
| HIGH     | 7     | 2     | 0               | 5          |
| MEDIUM   | 14    | 1     | 1               | 12         |
| LOW      | 12    | 1     | 0               | 11         |
| **Total**| **35**| **4** | **1**           | **30**     |

**Overall Fix Rate**: 11.4% fixed, 2.9% partially fixed, 85.7% still open.

---

## CRITICAL Issues

### CRITICAL 1: Metal backward fallback returns None for all gradients
**Status**: STILL OPEN

**File**: `models/gsplats/metal/gsplat_model_metal.py`, lines 397-403

**Evidence**: The current code at line 397-403 still reads:
```python
else:
    # Fallback: PyTorch forward was used, so PyTorch Autograd handled it
    # We don't need to compute gradients - they're already in the graph
    # Just return None for all outputs (Autograd will handle it)
    # NOTE: This should never be called if forward used PyTorch fallback,
    # because render_gaussians is already tracked by Autograd
    return (None, None, None, None, None, None, None, None, None)
```

The comment claims Autograd handles it, but when `MetalSplatFunction.apply()` wraps the call, this backward IS what Autograd calls. The CUDA backend at `gsplat_model_cuda.py` lines 303-341 correctly handles the fallback by recomputing with gradient tracking:
```python
else:
    # Fallback: recompute forward with gradient tracking and use autograd
    from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
    with torch.enable_grad():
        centers_grad = centers.detach().clone().requires_grad_(True)
        ...
        output = render_gaussians(shape, centers_grad, ...)
    grads = torch.autograd.grad(outputs=output, inputs=[centers_grad, ...], ...)
```

The Metal fallback path silently produces zero gradients, causing training failure when Metal is unavailable.

---

### CRITICAL 2: `cholesky_to_conic` additive epsilon can produce incorrect results
**Status**: STILL OPEN

**Files**:
- `models/gsplats/cuda/gsplat_model_cuda.py`, lines 56-57, 77-79
- `models/gsplats/metal/gsplat_model_metal.py`, lines 71-73, 94
- `models/gsplats/metal/src/kernels.metal`, lines 125-127

**Evidence**: All three backends still use `L00 + 1e-9` additive epsilon:

CUDA (`gsplat_model_cuda.py`):
```python
K00 = 1.0 / (L00 + 1e-9)
K11 = 1.0 / (L11 + 1e-9)
K22 = 1.0 / (L22 + 1e-9)
```

Metal Python (`gsplat_model_metal.py`):
```python
K00 = 1.0 / (L00 + 1e-9)
K11 = 1.0 / (L11 + 1e-9)
K22 = 1.0 / (L22 + 1e-9)
```

Metal Shader (`kernels.metal`):
```metal
float K_zz = 1.0f / (L_zz + 1e-9f);
float K_yy = 1.0f / (L_yy + 1e-9f);
float K_xx = 1.0f / (L_xx + 1e-9f);
```

The recommendation was to use `torch.clamp(L00, min=1e-8)` (or `max(L_zz, 1e-8f)` in Metal) instead of additive epsilon. This prevents the ~9% error for small valid diagonals (e.g., `L00=1e-8`) and the `Inf` result for negative values.

---

## HIGH Issues

### HIGH 3: Grid cache has no eviction policy
**Status**: STILL OPEN

**File**: `models/gsplats/rendering_core.py`, lines 87-98

**Evidence**: The `_GRID_CACHE` dict at line 87-89 still has no size limit or LRU eviction:
```python
_GRID_CACHE: Dict[
    Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], Tuple[torch.Tensor, torch.Tensor]
] = {}
```
A `clear_grid_cache()` function exists at line 92-98, but there is no automatic eviction. The cache grows unboundedly with each unique (device, dtype, strides, box_shape) combination.

---

### HIGH 4: Metal 3D backward kernel lacks SIMD reduction
**Status**: STILL OPEN

**File**: `models/gsplats/metal/src/kernels.metal`, line 484

**Evidence**: The 3D backward kernel `rasterize_bwd_3d` at line 484 still contains the comment:
```metal
// === DIRECT ATOMIC WRITE (no SIMD reduction) ===
// Each thread atomically adds its own gradient contribution
```
It writes gradients directly with per-thread atomic operations. Meanwhile, the nD backward kernel `rasterize_bwd_nd` at lines 737-776 correctly uses SIMD reduction:
```metal
// === B. SIMD Reduction (CRITICAL - reduces atomics by 32x) ===
float sum_amps = simd_sum(val_amps);
...
if (simd_lane_id == 0) {
    atomic_add_float(&d_amps[i], sum_amps);
}
```

---

### HIGH 5: Debug `os.environ` probes in Metal backward hot path
**Status**: STILL OPEN

**File**: `models/gsplats/metal/gsplat_model_metal.py`, lines 285-341

**Evidence**: The backward method still imports `os` and queries `os.environ` inside the backward pass:
```python
# line 285
import os

if os.environ.get("DEBUG_METAL_GRADIENTS"):
    ...  # extensive debug printing from line 288-311, 323-341, 368-395
```
This `import os` statement and multiple `os.environ.get("DEBUG_METAL_GRADIENTS")` checks execute on every backward call. There are 5 separate `os.environ.get("DEBUG_METAL_GRADIENTS")` checks throughout the backward method.

---

### HIGH 6: Metal `cholesky_to_conic` variable naming is documented as misleading
**Status**: FIXED

**File**: `models/gsplats/metal/gsplat_model_metal.py`, lines 53-86

**Evidence**: The CUDA backend has been updated to use index-based naming (`c_00`, `c_01`, etc.) at lines 85-92:
```python
c_00 = K00 * K00 + K10 * K10 + K20 * K20
c_01 = K10 * K11 + K20 * K21
c_02 = K20 * K22
c_11 = K11 * K11 + K21 * K21
c_12 = K21 * K22
c_22 = K22 * K22
```

However, the Metal backend still uses the misleading `c_xx`, `c_xy`, `c_zz` naming (lines 79-86) with a WARNING docstring at line 52-53 explaining the mismatch. The issue recommended renaming to index-based. The CUDA version was fixed; the Metal version still carries the warning but has not been renamed.

**Revised Status**: PARTIALLY FIXED (CUDA side fixed, Metal side still has misleading names with a warning comment).

Wait -- re-reading the issue: it says "Rename to index-based (`c_00`) or coordinate-based (`c_zz`)." The Metal backend uses `c_xx` etc. which the review called "misleading" because `c_xx` "actually corresponds to the Z,Z element." The CUDA backend now uses `c_00` (index-based). The Metal backend still uses the misleading names with a warning docstring. This is still open for Metal.

**Status**: PARTIALLY FIXED (CUDA fixed to index-based naming; Metal still uses misleading coordinate names with a warning docstring)

---

### HIGH 7: CUDA `CUDASplatFunction` saves unnecessary tensors in fallback path
**Status**: STILL OPEN

**File**: `models/gsplats/cuda/gsplat_model_cuda.py`, lines 205-226

**Evidence**: The forward method at line 206 always saves 6 tensors:
```python
ctx.save_for_backward(centers, Ls, Ls_for_conic, conic, amps)
```
And also caches FP16 kernel tensors at lines 208-211:
```python
ctx.centers_kernel = centers_kernel
ctx.conic_kernel = conic_kernel
ctx.amps_kernel = amps_kernel
// sharpness_kernel removed (standard Gaussian hardcoded)
```

These are saved regardless of whether the CUDA backend was used (line 166) or the PyTorch fallback was used (line 191). The fallback backward path at lines 303-341 recomputes everything from scratch, so the cached kernel tensors are wasted memory in that path.

---

### HIGH 8: CUDA fast math intrinsics may affect training convergence
**Status**: STILL OPEN

**File**: `models/gsplats/cuda/src/math_utils.cuh`, lines 179, 187, 190

**Evidence**: The code still uses `__expf()` and `__powf()` fast math intrinsics:
```c
return amplitude * __expf(-0.5f * dist_sq);           // line 179
float dist_pow_s = __powf(dist_sq_safe, sharpness * 0.5f);  // line 187
return amplitude * __expf(-0.5f * dist_pow_s);         // line 190
```
These have ~2 ULP error and don't handle denormals. The code comment at lines 162-163 acknowledges this is acceptable for rendering but not for exact numerical computation.

---

### HIGH 9: Global backward kernel duplicates gradient math (DRY violation)
**Status**: STILL OPEN

**File**: `models/gsplats/cuda/src/kernels_global.cuh`, lines 226-269

**Evidence**: The global backward kernel still implements gradient computation inline at lines 226-269:
```c
// d_amp = grad_out * (I / a)
local_d_amp = grad_out * (intensity / fmaxf(amp, 1e-10f));
...
// d_centers: dD^2/dmu = -2 x Sigma^-1 @ d
for (int di = 0; di < DIM; di++) {
    float sum = 0.0f;
    for (int dj = 0; dj < DIM; dj++) {
        ...
    }
    local_d_centers[di] = outer_grad * (-2.0f) * sum;
}
```

Meanwhile, the core kernels at `kernels_core.cuh` line 793-798 properly use the `compute_pixel_gradients<DIM>()` function from `reduction_utils.cuh`. The global kernel should call `compute_pixel_gradients<DIM>()` instead of duplicating the math.

---

## MEDIUM Issues

### MEDIUM 1: `_build_L` Python loops
**Status**: STILL OPEN

**File**: `models/gsplats/gsplat_model.py`, lines 231, 255-267

**Evidence**: The code still uses Python for-loops to fill the diagonal and off-diagonal elements:
```python
# line 231
for i in range(d):
    L[:, i, i] = diag[:, i]

# line 255-267
for i in range(d):
    for j in range(i):
        ...
        L[:, i, j] = off_val
        k += 1
```
Could be replaced with `torch.diag_embed()` and advanced indexing.

---

### MEDIUM 2: AABB computes radii twice
**Status**: FIXED

**File**: `models/gsplats/rendering_core.py`, lines 244-329

**Evidence**: The review referenced `rendering_core.py:290-325` where initial radii computation was wasted when `intensity_floor > 0`. The current code at lines 244-329 uses a helper function `_compute_aabb_with_intensity_floor()` which computes radii once (lines 291-296), then conditionally shrinks them if `intensity_floor > 0` using `torch.minimum(radii, shrink)` at line 320. The initial computation is NOT wasted -- it's used as the upper bound and the intensity-based shrink can only reduce it. This is correct behavior. However, looking more carefully, the `lo` and `hi` are computed twice (lines 299-303 and lines 321-325). The second computation only applies if `intensity_floor > 0` and uses the potentially shrunk radii.

On closer examination, the code was refactored into a helper function but the double-computation of `lo`/`hi` still exists. The initial `lo`/`hi` at lines 299-303 are overwritten at lines 321-325 when intensity_floor > 0. So the first `lo`/`hi` computation is indeed wasted in the intensity_floor case.

**Revised Status**: STILL OPEN (the double lo/hi computation still exists, though radii computation is now correct)

---

### MEDIUM 3: Variable name collision `result` shadows parameter
**Status**: STILL OPEN

**File**: `models/gsplats/rendering_wrappers.py`, line 75

**Evidence**: The function `render_gaussians_numpy` has parameter `result: GSplatData` at line 28, and then at line 75 reassigns it:
```python
result = render_gaussians(
    shape,
    centers_torch,
    ...
)
```
The local `result` at line 75 shadows the parameter `result: GSplatData` from line 28. The rendered tensor overwrites the input GSplatData reference.

---

### MEDIUM 4: Metal `__init__.py` side effects at import
**Status**: STILL OPEN

**File**: `models/gsplats/metal/__init__.py`, line 299

**Evidence**: At module level (line 299):
```python
if sys.platform == "darwin":
    # First validate MPS interop works
    _mps_interop_valid = _validate_mps_interop()
```
This triggers MPS device validation at import time (creating MPS tensors, checking storage pointers). If the extension is not found, `_auto_build_extension()` is called which can take up to 2 minutes (subprocess with timeout=120 at line 196).

---

### MEDIUM 5: Dead code `_params_cache`
**Status**: STILL OPEN

**File**: `models/gsplats/metal/gsplat_model_metal.py`, lines 498-500

**Evidence**: At lines 498-500 (in `__init__`):
```python
# Cache for current_params() (23% speedup!)
self._params_cache = None
self._params_cache_valid = False
```
And line 502-504:
```python
def _invalidate_cache(self) -> None:
    """Invalidate cached parameters (call after optimizer.step())."""
    self._params_cache_valid = False
```
But `_params_cache` is never read or populated. The `current_params()` at line 537-538 delegates directly to `self._base.current_params()` without checking the cache.

---

### MEDIUM 6: Fragile parameter count check
**Status**: STILL OPEN

**File**: `models/gsplats/metal/gsplat_model_metal.py`, lines 462-473

**Evidence**: At lines 462-473:
```python
import inspect
base_params = inspect.signature(GaussianSplatModel.__init__).parameters
if len(base_params) > 13:  # Expected: ~12 parameters (self + 11 init params)
    import warnings
    warnings.warn(
        "GaussianSplatModel has more parameters than expected. "
        ...
    )
```
This hardcoded `> 13` check will trigger spurious warnings whenever `GaussianSplatModel.__init__` gains a new parameter.

---

### MEDIUM 7: `_to_internal_params` CPU round-trip
**Status**: STILL OPEN

**File**: `models/gsplats/gsplat_model.py`, lines 358-362, 377-381

**Evidence**: The method still uses NumPy `stable_inverse_softplus` requiring CPU round-trips:
```python
# line 358-362
L_diag_raw = torch.tensor(
    stable_inverse_softplus(diag.detach().cpu().numpy()),
    device=device,
    dtype=torch.float32,
)

# line 377-381
amp_raw = torch.tensor(
    stable_inverse_softplus(amps.detach().cpu().numpy()),
    device=device,
    dtype=torch.float32,
)
```
The `stable_inverse_softplus_torch` function exists in `inverse_softplus.py` at line 74 but is not used here.

---

### MEDIUM 8: `inverse_softplus` produces `-inf` for `y=0`
**Status**: STILL OPEN

**File**: `models/utils/inverse_softplus.py`, lines 47-52

**Evidence**: At lines 47-52:
```python
if np.any(y <= 0):
    import warnings
    warnings.warn(
        "Inverse softplus input contains non-positive values", RuntimeWarning
    )
```
It warns but then continues to compute `np.log(np.expm1(beta_y[small_mask]))` which produces `-inf` for `y=0` (since `expm1(0) = 0` and `log(0) = -inf`). No special handling for zero values.

---

### MEDIUM 9: `sys.path` modification is permanent
**Status**: STILL OPEN

**File**: `models/gsplats/cuda/__init__.py`, lines 30-32

**Evidence**: At lines 30-32:
```python
_cuda_dir = Path(__file__).parent
if str(_cuda_dir) not in sys.path:
    sys.path.insert(0, str(_cuda_dir))
```
This permanently adds the CUDA directory to `sys.path` at the front, which could cause import collisions with other modules that share a name with files in the CUDA directory.

---

### MEDIUM 10: Metal nD backward L gradient approximation
**Status**: STILL OPEN

**File**: `models/gsplats/metal/src/kernels.metal`, line 729-730

**Evidence**: At lines 728-730:
```metal
for (uint c = 0; c <= r; c++) {
    float L_rc = Ls[i * dim * dim + r * dim + c];
    float L_rc_safe = (c == r) ? max(abs(L_rc), 1e-9f) : (abs(L_rc) > 1e-9f ? L_rc : 1.0f);
    val_Ls[r * 8 + c] = grad_dist * (-2.0f * y[r] * y[c]) / L_rc_safe;
}
```
For near-zero off-diagonals, `L_rc_safe` becomes `1.0f` (a somewhat arbitrary fallback), which produces an approximation rather than the correct gradient.

---

### MEDIUM 11: Forward substitution clamping
**Status**: STILL OPEN

**File**: `models/gsplats/rendering_core.py`, lines 171, 191-193

**Evidence**: At lines 171, 191-193:
```python
y0 = d0 / torch.clamp(l11, min=1e-6)        # line 171
y0 = d0 / torch.clamp(l11, min=1e-6)        # line 191
y1 = (d1 - l21 * y0) / torch.clamp(l22, min=1e-6)  # line 192
y2 = (d2 - l31 * y0 - l32 * y1) / torch.clamp(l33, min=1e-6)  # line 193
```
The clamping silently masks degenerate Cholesky factors (diagonal elements near zero) instead of raising an error or warning.

---

### MEDIUM 12: `render_to_volume` doesn't use accelerated backends
**Status**: STILL OPEN

**File**: `rendering/volume_rendering.py`, lines 124-134

**Evidence**: The `render_to_volume` function at lines 124-134 always uses the PyTorch `render_gaussians()` from `rendering_core.py`:
```python
rendered_t = render_gaussians(
    shape=shape,
    centers=centers_t,
    Ls=Ls_t,
    amps=amps_t,
    sharpness=sharpness_t,
    truncate=truncate,
    intensity_floor=intensity_floor,
    chunk_size=chunk_size,
)
```
It does not use `CUDASplatFunction` or `MetalSplatFunction` even when CUDA/Metal backends are available. However, it does move tensors to the appropriate device (CUDA/MPS/CPU) at line 88-94, so it benefits from GPU tensor operations in the PyTorch renderer. The custom CUDA/Metal kernels (tile-based rasterization) are not used.

---

### MEDIUM 13: CUDA backward shared memory contention
**Status**: STILL OPEN

**File**: `models/gsplats/cuda/src/kernels_core.cuh`, lines 806-808

**Evidence**: At lines 806-808, all warps in a tile block contend on the same shared memory index `si`:
```c
float warp_d_amp = warp_reduce_sum(local_d_amp);
if (lane == 0) {
    atomicAdd(&s_d_amps_tile[si], warp_d_amp);
}
```
Multiple warps within the same block process the same splat (`si`) simultaneously, meaning warp-level leaders all atomicAdd to the same shared memory locations. However, shared memory atomics are relatively fast on modern GPUs (single-cycle on SM 8.0+), so this is more of a theoretical concern than a practical bottleneck.

---

### MEDIUM 14: `GaussianSplatModelCUDA` breaks isinstance
**Status**: STILL OPEN

**File**: `models/gsplats/cuda/gsplat_model_cuda.py`, line 348

**Evidence**: `GaussianSplatModelCUDA` at line 348 inherits from `torch.nn.Module`:
```python
class GaussianSplatModelCUDA(torch.nn.Module):
```
It uses composition (wrapping a `GaussianSplatModel` via `self._base`) rather than inheritance. This means `isinstance(model, GaussianSplatModel)` returns `False` for a `GaussianSplatModelCUDA` instance. Same pattern is used by `GaussianSplatModelMetal` at line 409.

---

## LOW Issues

### LOW 1: `current_params` allocates shape tensor each call
**Status**: STILL OPEN

**File**: `models/gsplats/gsplat_model.py`, line 299

**Evidence**: At line 299:
```python
shape = torch.tensor(self.shape, dtype=torch.float32, device=u.device)
```
This allocates a new tensor every time `current_params()` is called (which happens every forward pass). It could be cached as a buffer in `__init__`.

---

### LOW 2: `replace_with`/`prune_`/`append_` break optimizer state
**Status**: STILL OPEN

**File**: `models/gsplats/gsplat_model.py`, lines 391-441

**Evidence**: All three methods create new `torch.nn.Parameter` objects (e.g., line 402):
```python
self.raw_mu = torch.nn.Parameter(raw_mu)
```
This detaches from any existing optimizer's parameter groups. The optimizer still references the old `Parameter` objects. No warning or documentation about this behavior.

---

### LOW 3: Variable name shadowing `s` in nD renderer
**Status**: STILL OPEN

**File**: `models/gsplats/rendering_core.py`, line 575

**Evidence**: At line 203, `s` is used for strides computation:
```python
s = [1]
for i in range(d - 1, 0, -1):
    s.insert(0, s[0] * shape[i])
```
And at line 570 in the nD renderer loop, `s` is reused for sharpness:
```python
s = sharpness[idx]  # (K,)
```
These are in different scopes (different functions), so this is not actually a collision. Let me re-check the original claim... The review says "Variable name shadowing `s` in nD renderer (`rendering_core.py:575`)." Looking at line 575:
```python
ranges = [
    torch.arange(s, device=device, dtype=torch.float32) for s in box_shape
]
```
Here `s` in the list comprehension shadows the outer `s = sharpness[idx]` at line 570. This is a genuine shadowing within the nD render loop.

---

### LOW 4: `_group_by_box` redundant GPU-CPU sync
**Status**: STILL OPEN

**File**: `models/gsplats/rendering_core.py`, line 231

**Evidence**: At line 231:
```python
uniq_cpu = uniq.cpu().tolist()
```
This transfers the unique sizes array to CPU. The comment at line 230 says "Only transfer the small unique array to CPU for dict keys" which is an optimization over the alternative. However, there is still one sync point here. This is minor since the unique array is typically very small.

---

### LOW 5: `render_gaussians_batched` is a no-op wrapper
**Status**: STILL OPEN

**File**: `models/gsplats/rendering_wrappers.py`, lines 153-192

**Evidence**: At lines 190-192:
```python
return render_gaussians(
    shape, centers, Ls, amps, sharpness, truncate, intensity_floor, chunk_size
)
```
The function is a pure pass-through to `render_gaussians` with identical parameters and no additional logic.

---

### LOW 6: `build.py` assumes GPU device 0
**Status**: STILL OPEN

**File**: `models/gsplats/cuda/build.py`, line 48

**Evidence**: At line 48:
```python
major, minor = torch.cuda.get_device_capability()
```
This calls `get_device_capability()` without specifying a device, defaulting to device 0. On multi-GPU systems, this may not compile for the correct architecture if the user intends to use a different GPU.

---

### LOW 7: `benchmark_forward` doesn't use `torch.no_grad()`
**Status**: FIXED

**File**: `models/gsplats/cuda/benchmark.py`, line 85

**Evidence**: The current code at line 85 wraps the entire benchmark in `torch.no_grad()`:
```python
def benchmark_forward(...) -> float:
    ...
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            ...
```
This has been fixed since the original review.

---

### LOW 8: `benchmark.py` mutable default argument
**Status**: STILL OPEN

**File**: `models/gsplats/cuda/benchmark.py`, line 180

**Evidence**: At line 180:
```python
def run_benchmark(
    configs: List[Tuple[int, Tuple[int, ...], str]] = None,
    ...
```
The default is `None` (not mutable), so this is technically safe. However, the type annotation says `List[...]` but the default is `None`. The correct annotation should be `Optional[List[...]] = None`.

Note: Re-examining the original issue -- it said "mutable default argument" at line 152. The current line 180 uses `= None` which is the standard Python pattern. If the original code had `= []`, it has been changed to `= None`. This appears fixed or was a false positive.

**Revised Status**: FIXED (uses `= None` default, not a mutable default)

---

### LOW 9: Metal MPS tensors not freed during import validation
**Status**: STILL OPEN

**File**: `models/gsplats/metal/__init__.py`, lines 247-264

**Evidence**: In `_validate_mps_interop()` at lines 247-264:
```python
t = torch.randn(10, device="mps")
storage_ptr = t.untyped_storage().data_ptr()
...
t_full = torch.randn(100, device="mps")
t_slice = t_full[25:75]
...
t_contig = t_full.contiguous()
```
These MPS tensors are created but never explicitly freed. They rely on garbage collection after the function scope ends. On MPS, explicit cleanup with `del` and `torch.mps.empty_cache()` would be more reliable.

---

### LOW 10: Dtype preservation no-op in `inverse_softplus`
**Status**: STILL OPEN

**File**: `models/utils/inverse_softplus.py`, lines 40-44

**Evidence**: At lines 38-44:
```python
y = np.asarray(y)
original_dtype = y.dtype

# Convert to float for computation (preserving original precision)
if y.dtype == np.float64:
    y = y.astype(np.float64)  # No-op: already float64
else:
    y = y.astype(np.float32)
```
The `y.dtype == np.float64` branch calls `y.astype(np.float64)` which is a no-op when `y` is already float64.

---

### LOW 11: `stable_inverse_softplus_torch` has no input validation
**Status**: STILL OPEN

**File**: `models/utils/inverse_softplus.py`, lines 74-118

**Evidence**: The torch version at lines 74-118 performs no input validation:
```python
def stable_inverse_softplus_torch(y: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    beta_y = beta * y
    large_mask = beta_y >= 50.0
    result = torch.where(
        large_mask,
        y,
        torch.log(torch.expm1(beta_y)) / beta,
    )
    return result
```
No check for non-positive values (unlike the NumPy version which warns). For `y <= 0`, `torch.expm1(beta_y)` returns values in `[-1, 0)`, and `torch.log()` of these produces `NaN`/`-inf` silently.

---

### LOW 12: Overly broad `except Exception` in `lt_solver.py`
**Status**: STILL OPEN

**File**: `models/utils/lt_solver.py`, line 55

**Evidence**: At line 52-58:
```python
try:
    return cast(torch.Tensor, torch.linalg.solve_triangular(L, B, upper=False))
except Exception:
    X, _ = torch.triangular_solve(B, L, upper=False)
    return cast(torch.Tensor, X)
```
The `except Exception` catches all exceptions, not just `AttributeError` (which would indicate the function doesn't exist in older PyTorch). This could mask genuine runtime errors like CUDA OOM, invalid tensor shapes, or singular matrices.

---

## Cross-Backend Consistency

### Numerical Epsilon Inconsistencies
**Status**: STILL OPEN

The epsilon values remain inconsistent across backends:

| Location | Epsilon | Purpose |
|----------|---------|---------|
| `rendering_core.py:171` | `1e-6` | Clamp Cholesky diagonal |
| `gsplat_model_cuda.py:56` | `1e-9` | Additive epsilon for L inversion |
| `gsplat_model_metal.py:71` | `1e-9` | Additive epsilon for L inversion |
| `math_utils.cuh:184` | `1e-12f` | Clamp dist_sq in gaussian_intensity |
| `rendering_core.py:400` | `1e-10` | Clamp expo for pow() stability |
| `kernels.metal:583` | `1e-9f` | Forward substitution in nD kernel |

No standardization or rationale documentation has been added.

---

## Recommendations Status

| Recommendation | Status |
|---------------|--------|
| 1. Fix two CRITICAL issues | NOT DONE |
| 2. Standardize epsilon handling | NOT DONE |
| 3. Add cross-backend numerical comparison tests | NOT VERIFIED (may exist in test files) |
| 4. Remove debug probes from Metal backward | NOT DONE |
| 5. Add SIMD reduction to Metal 3D backward | NOT DONE |
| 6. Add LRU eviction to grid cache | NOT DONE |
| 7. Use `stable_inverse_softplus_torch` in `_to_internal_params` | NOT DONE |
