# GSplat Fitting Loop Bottleneck Analysis

**Date:** 2026-04-08  
**Scope:** Per-iteration overhead in `run_optimization_loop()` and scaling to 1000^3 3D volumes  
**Focus:** Non-CUDA bottlenecks (the forward/backward CUDA kernels are assumed fast)

---

## Executive Summary

The CUDA forward and backward kernels are indeed fast, but significant overhead exists in the **Python-side scaffolding** that wraps every iteration. For typical volumes (128^3 to 512^3), several of these are noticeable. For 1000^3 volumes (~10^9 voxels, 4 GB float32), some become **dominant or infeasible**.

**Top 5 bottlenecks by per-iteration impact:**

| # | Bottleneck | Where | Impact at 1000^3 | Fix Difficulty |
|---|-----------|-------|-------------------|----------------|
| B1 | `_build_L()` Python loops + tensor alloc every forward | `gsplat_model.py:186-280` | ~ms per call, called 2-3x/iter | Medium |
| B2 | Loss function creates 3-4 full-volume temporaries | `losses.py:143-163` | 12-16 GB transient GPU mem | Medium |
| B3 | `current_params()` re-called in loss for boundary penalty | `losses.py:95` | Doubles `_build_L()` cost | Easy |
| B4 | Output volume `torch.zeros()` allocation every forward | `rendering_core.py:435` | 4 GB alloc per forward call | Medium |
| B5 | Eval metrics `.item()` forces GPU sync every 25 iters | `optimization.py:94` | Stalls GPU pipeline | Easy |

---

## Detailed Findings

### B1: `_build_L()` — Python loops and tensor allocation on every forward pass [CRITICAL]

**File:** `gsplat_model.py:186-280`  
**Called by:** `current_params()` → `forward()` → every training iteration

```python
def _build_L(self) -> torch.Tensor:
    N, d = self.raw_L_diag.shape
    diag = self.sigma_min_diag + F.softplus(self.raw_L_diag)  # (N, d)
    
    # ... eccentricity constraints (more tensor ops) ...
    
    L = torch.zeros((N, d, d), ...)  # <-- ALLOC: N×d×d float32 EVERY CALL
    
    for i in range(d):                # <-- PYTHON LOOP over dimensions
        L[:, i, i] = diag[:, i]
    
    for i in range(d):                # <-- NESTED PYTHON LOOP
        for j in range(i):
            # ... constraint logic per off-diagonal ...
            L[:, i, j] = off_val
    
    return L
```

**Problems:**
1. **Allocates `(N, d, d)` zeros tensor every call.** For N=100K, d=3: 3.6 MB per call. Not huge, but CUDA malloc has overhead and fragments the allocator.
2. **Python for-loops** iterate d + d*(d-1)/2 times (6 iterations for 3D, 10 for 4D). Each iteration dispatches a CUDA kernel for the slice assignment. Python loop overhead + kernel launch overhead compound.
3. **Called 2-3 times per iteration:**
   - Once in `model()` forward pass (training)
   - Once in `model()` forward pass (eval, every 25 iters)
   - Once in `loss_fn()` if `boundary_penalty > 0` (see B3)
   - Once in `best_state` tracking via `model.current_params()` when loss improves

**Impact:** For 3D, ~6 Python-dispatched CUDA kernels + 1 allocation per call. At 2-3 calls/iter over 1000 iterations = 6000-9000 unnecessary kernel launches.

**Fix:** Vectorize `_build_L()` to eliminate Python loops:
```python
# Vectorized diagonal fill (no loop):
idx = torch.arange(d, device=diag.device)
L = torch.zeros((N, d, d), ...)
L[:, idx, idx] = diag

# Vectorized off-diagonal fill using tril_indices:
row_idx, col_idx = torch.tril_indices(d, d, offset=-1)
L[:, row_idx, col_idx] = self.L_off  # (or constrained values)
```
Also consider caching the `L` tensor and reusing the buffer via `L.zero_()` instead of `torch.zeros()`.

---

### B2: Loss function creates multiple full-volume temporaries [CRITICAL at scale]

**File:** `losses.py:143-163`

```python
def _compute_l1_loss(pred, target, asymmetric_penalty):
    l1_error = torch.abs(pred - target)       # TEMP 1: full volume copy
    over_prediction_mask = pred > target       # TEMP 2: full volume bool
    data = torch.mean(torch.where(             # TEMP 3: full volume copy
        over_prediction_mask,
        asymmetric_penalty * l1_error,         # TEMP 4: full volume copy
        l1_error,
    ))
    return data
```

**Memory at 1000^3:** Each temporary is 4 GB (float32) or 1 GB (bool). Peak transient: ~13 GB for L1 loss alone, on top of the 4 GB target + 4 GB prediction already in memory.

**Total GPU memory for 1000^3 volume:**
| Tensor | Size | Lifetime |
|--------|------|----------|
| V_target | 4 GB | Permanent |
| pred (forward output) | 4 GB | Forward + backward |
| pred - target | 4 GB | Loss computation |
| abs(pred - target) | 4 GB | Loss computation |
| over_prediction_mask | 1 GB | Loss computation |
| torch.where result | 4 GB | Loss computation |
| grad_output (backward) | 4 GB | Backward pass |
| **Peak total** | **~25 GB** | |

This exceeds RTX 3090/4090 (24 GB) and is tight on A100-40GB.

**Fix:** Apply `torch.compile` to `_compute_l1_loss` (like already done for Poisson loss). This would fuse the element-wise ops into a single kernel, eliminating intermediate tensor allocations:
```python
@torch.compile(fullgraph=False)
def _compute_l1_loss(pred, target, asymmetric_penalty):
    ...
```
For the MSE loss path too. This alone could save ~12 GB of transient memory at 1000^3.

Alternatively, implement a chunked loss computation that processes the volume in tiles.

---

### B3: `current_params()` called redundantly in loss function [EASY FIX]

**File:** `losses.py:94-110`

```python
if boundary_penalty is not None and boundary_penalty > 0:
    centers, L, _ = model.current_params()   # <-- CALLS _build_L() AGAIN!
    sigma_diag = torch.sum(L * L, dim=2)
    ...
```

`current_params()` is already called in `model.forward()` (line 434 of `gsplat_model.py`). Calling it again in the loss function triggers a second full `_build_L()` computation. With boundary penalty enabled, this doubles the Python-loop overhead from B1.

**Fix:** Pass the already-computed `centers, Ls, amps` from the forward pass into the loss function, or cache the last `current_params()` result on the model.

---

### B4: Output volume allocated from scratch every forward pass [SIGNIFICANT at scale]

**File:** `rendering_core.py:435` (PyTorch renderer), CUDA also allocates output

```python
out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
```

For 1000^3: **4 GB allocation per forward call.** Called 1-2 times per iteration (training + eval). CUDA's memory allocator caches freed blocks, but fragmentation and allocation overhead still matter.

**Fix:** Pre-allocate the output buffer once and reuse it:
```python
# In model __init__:
self._output_buffer = torch.zeros(shape, dtype=torch.float32, device=device)

# In forward:
self._output_buffer.zero_()  # Much faster than torch.zeros()
```

---

### B5: Eval metrics force GPU synchronization [MODERATE]

**File:** `optimization.py:94,96`

```python
def _compute_eval_metrics(pred, target):
    diff = pred - target                                    # 4 GB temp
    max_abs_err = torch.max(torch.abs(diff)).item()         # <-- GPU SYNC
    rel_l2 = float(torch.linalg.norm(diff.reshape(-1))     # <-- GPU SYNC
              / (torch.linalg.norm(target.reshape(-1)) + 1e-12))
    del diff
    return max_abs_err, rel_l2
```

`.item()` and `float()` both force CUDA synchronization, stalling the GPU pipeline until the reduction completes. This happens every 25 iterations.

**Additional issue:** `diff` is a 4 GB tensor at 1000^3 (on top of pred and target already in memory).

**Fix:** Keep metrics as GPU tensors and only sync when actually logging or checking convergence thresholds:
```python
def _compute_eval_metrics(pred, target):
    diff = pred - target
    max_abs_err_t = torch.max(torch.abs(diff))     # stays on GPU
    diff_norm = torch.linalg.norm(diff.reshape(-1))
    target_norm = torch.linalg.norm(target.reshape(-1))
    rel_l2_t = diff_norm / (target_norm + 1e-12)
    del diff
    return max_abs_err_t, rel_l2_t  # Return tensors, sync later
```

---

### B6: `current_params()` creates shape tensor every call [MINOR but frequent]

**File:** `gsplat_model.py:306`

```python
def current_params(self):
    u = torch.sigmoid(self.raw_mu)
    shape = torch.tensor(self.shape, dtype=torch.float32, device=u.device)  # <-- ALLOC
    centers = u * torch.clamp(shape - 1.0, min=1.0)
    ...
```

Creates a small GPU tensor from a Python tuple every call. It's small (~12 bytes for 3D) but involves a CPU→GPU transfer and kernel launch. Called 2-3 times per iteration.

**Fix:** Cache as a buffer in `__init__`:
```python
self.register_buffer('_shape_tensor', torch.tensor(shape, dtype=torch.float32))
```

---

### B7: Cholesky → Conic conversion clones + recomputes in backward [CUDA path]

**File:** `gsplat_model_cuda.py:161,280-286`

**Forward:**
```python
Ls_for_conic = Ls.detach().clone().requires_grad_(True)  # <-- CLONE: 3.6 MB
conic = _cholesky_to_conic_compiled(Ls_for_conic)
```

**Backward:**
```python
with torch.enable_grad():
    conic_recomputed = _cholesky_to_conic_compiled(Ls_for_conic)  # <-- RECOMPUTE
(d_Ls,) = torch.autograd.grad(conic_recomputed, Ls_for_conic, d_conic)
```

The conic conversion is computed **twice** per iteration (once in forward, once in backward for the chain rule). The clone in forward is also unnecessary if you structure the graph correctly.

**Fix:** Compute analytical gradients for the L → conic mapping (it's just matrix inversion derivatives) instead of relying on autograd. The 3D case is fully explicit and the analytical Jacobian is straightforward.

---

### B8: Morton sort does CPU roundtrip [PERIODIC, every N iters]

**File:** `sorting.py:43-58`

```python
centers_np = centers.detach().cpu().numpy()    # GPU → CPU transfer
morton_codes = morton_encode_nd(grid_coords, bits_per_dim)  # CPU computation
sort_indices = np.argsort(morton_codes)        # CPU sort
perm = torch.tensor(sort_indices, ..., device=centers.device)  # CPU → GPU
```

For 100K splats this is fast (~1ms). For 1M splats it could reach ~50ms. Done every `sort_splats_interval` iterations (default: unclear, but periodic).

**Fix:** Implement Morton encoding + sort on GPU using bit-interleaving kernels.

---

### B9: Dynamic ops peak finding creates large intermediate tensors [PERIODIC]

**File:** `peak_finding.py:143-192`

```python
residual_positive = torch.clamp(residual, min=0)  # Full volume copy (4 GB at 1000^3)
max_pooled = torch.nn.functional.max_pool3d(...)   # Another full volume (4 GB)
is_peak = (residual_positive >= max_pooled) & (residual_positive > 0)  # Bool (1 GB)
```

For 1000^3: ~9 GB of transient memory just for peak finding, on top of the existing volume tensors. This runs every `step_every` iterations (default: 50).

**Fix:** Chunked/tiled peak finding that processes the volume in blocks.

---

### B10: `_to_internal_params()` does CPU roundtrip for inverse softplus [DYNAMIC OPS]

**File:** `gsplat_model.py:344-348,363-367`

```python
L_diag_raw = torch.tensor(
    stable_inverse_softplus(diag_shifted.detach().cpu().numpy()),  # GPU→CPU→numpy→CPU→GPU
    device=device, dtype=torch.float32,
)
```

Both `_to_internal_params()` and `replace_with()` transfer tensors to CPU for `stable_inverse_softplus`, then back to GPU. This happens during dynamic ops (splat relocation) and sorting.

**Fix:** Implement `stable_inverse_softplus` as a pure PyTorch operation (it's just `log(exp(x) - 1)` with numerical guards).

---

## Scaling Analysis: 1000^3 Volume Feasibility

### Memory Budget

| Component | Size | Notes |
|-----------|------|-------|
| Target volume V_t | 4.0 GB | Permanent, on GPU |
| Forward output | 4.0 GB | Freed after backward |
| Loss intermediates | 8-12 GB | Multiple full-volume temps (B2) |
| Backward grad_output | 4.0 GB | CUDA backward needs it |
| Model parameters | ~5 MB | 100K splats × (3+9+1) × 4 bytes |
| Optimizer state | ~10 MB | Adam: 2× model params |
| Eval forward output | 4.0 GB | Every 25 iters (freed quickly) |
| **Total peak** | **~24-28 GB** | Exceeds most consumer GPUs |

### Verdict

1000^3 is **infeasible on 24 GB GPUs** (RTX 3090/4090) without memory optimization. It's **tight on A100-40GB** and comfortable on **A100-80GB / H100**.

### Recommendations for 1000^3 scale

1. **`torch.compile` the L1/MSE loss** — eliminates ~8 GB of transient memory via kernel fusion [DONE]
2. **Chunked loss computation** — process loss in tiles to cap peak memory
3. **Gradient checkpointing** — trade compute for memory in the backward pass
4. ~~**Pre-allocated output buffer** — avoid 4 GB allocation per forward~~ **REJECTED (2026-04-08):** Benchmarked and reverted. PyTorch autograd requires `.clone()` on the output when reusing a buffer (in-place modification detection), which negates the savings. The CUDA caching allocator already makes `torch::zeros` cheap. Net result was 5-17% regression. See comment in `gsplat_model_cuda.py:forward()`.
5. **Consider tiled fitting** (`fit_tiled_gsplats.py`) — already supported, processes tiles independently

---

## Per-Iteration Cost Breakdown (3D, 512^3, 100K splats, CUDA backend)

| Phase | Key Operations | Est. Time | Bottleneck? |
|-------|---------------|-----------|-------------|
| `zero_grad()` | Zero optimizer buffers | <0.01 ms | No |
| `current_params()` | sigmoid + `_build_L()` + softplus | ~0.5-1 ms | **B1: Python loops** |
| `CUDASplatFunction.forward()` | L→conic + CUDA kernel | ~2-10 ms | Mostly CUDA (fast) |
| `loss_fn(pred)` | L1/MSE + optional boundary | ~0.5-2 ms | **B2: temporaries, B3: redundant _build_L** |
| `loss.backward()` | CUDA backward + chain rule | ~5-15 ms | **B7: conic recompute** |
| `clip_grad_norm_()` | Single norm computation | ~0.01 ms | No |
| `optimizer.step()` | Fused Adam update | ~0.1 ms | No |
| `scheduler.step()` | LR comparison | <0.01 ms | No |
| **Best-state tracking** | Tensor comparison + clone | ~0.1 ms | Minor |
| **Eval (every 25)** | 2nd forward + metrics | ~15-25 ms | **B4: alloc, B5: sync** |
| **Dynamic ops (every 50)** | Peak finding + relocation | ~5-50 ms | **B9, B10** |
| **Sorting (periodic)** | CPU roundtrip | ~1-5 ms | **B8** |

**Estimated overhead from non-CUDA bottlenecks: ~2-4 ms per iteration** (B1+B2+B3+B6), which is **20-40% of a 10 ms CUDA kernel**. At 1000 iterations, that's 2-4 seconds of pure overhead.

---

## Priority Recommendations

### P0 — Do first (highest impact, easiest)
1. **`torch.compile` the L1 and MSE loss functions** (like Poisson already is) — eliminates memory pressure from B2
2. **Cache `_shape_tensor`** in model `__init__` — trivial fix for B6
3. **Remove redundant `current_params()` in boundary penalty** — pass cached values (B3)

### P1 — High impact
4. **Vectorize `_build_L()`** — replace Python loops with `tril_indices` + advanced indexing (B1)
5. **Pre-allocate output buffer** in renderer (B4)
6. **Implement `stable_inverse_softplus` in pure PyTorch** — eliminate CPU roundtrips in dynamic ops (B10)

### P2 — Important for 1000^3 scale
7. **Chunked loss computation** for large volumes (B2 at scale)
8. **GPU-side Morton encoding** for sorting (B8)
9. **Analytical L→conic gradients** to avoid recomputation in CUDA backward (B7)

### P3 — Nice to have
10. **Keep eval metrics as GPU tensors** until logging (B5)
11. **Chunked peak finding** in dynamic ops (B9)

---

## Appendix: Files Analyzed

| File | Lines | Role |
|------|-------|------|
| `fitting/optimization.py` | 489 | Main optimization loop |
| `models/gsplats/gsplat_model.py` | 444 | Base model, `_build_L`, `current_params` |
| `models/gsplats/cuda/gsplat_model_cuda.py` | 808 | CUDA model wrapper |
| `models/gsplats/rendering_core.py` | 645 | PyTorch renderer |
| `fitting/losses.py` | 189 | Loss functions |
| `fitting/initialization.py` | 296 | Model/optimizer init |
| `fitting/sorting.py` | 91 | Morton order sorting |
| `fitting/dynamic_ops/operations.py` | 783 | Splat relocation |
| `fitting/dynamic_ops/peak_finding.py` | 437 | Residual peak detection |
| `optim/integration.py` | 124 | Optimizer factory |
| `fitting/preprocessing.py` | ~400 | Data prep (one-time) |
