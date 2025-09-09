Great—thanks for sharing the code. I read all four modules end‑to‑end:

* **`gsplat_model.py`** – learnable n‑D oriented Gaussian splats + differentiable renderer (`render_gaussians`), plus NumPy/PyTorch wrappers.
* **`fit_gsplats.py`** – optimization loop (`GaussianSplatFitter.fit` + `fit_gaussian_splats`).
* **`candidates.py`** – multi‑strategy (Gaussian/DoG + grid) peak proposal and de‑duplication.
* **`demo_splats.py`** – Napari demo and compression visualization.

Below is a **critical, performance‑focused review** with concrete, code‑level suggestions. I grouped them by (i) code improvements, (ii) speed optimizations, and (iii) memory reductions. Where helpful I include drop‑in snippets.

---

## 0) Hot path recap (what dominates runtime/memory)

* **Renderer (`render_gaussians`)**
  For each group of splats with the same AABB shape, it builds a base grid (size $P=\prod h_i$), forms $\Delta = \text{base} + \mathrm{lo} - \mu$ $(K,d,P)$, does a **batched triangular solve** $L\,y=\Delta$ (shape $K\times d\times P$), then `exp(-0.5*||y||^2)` and a single `index_add_` into a flattened output.
  **Peak memory here** is the `(K,d,P)` buffers (`delta` and `y`). **Peak time** is triangular solve + scatter.

* **Fitter (`fit`)**
  One forward/backward per iteration; optional AMP/compile already enabled. Best‑state checkpointing doubles param memory.

* **Candidates**
  `_local_maxima` uses `maximum_filter` with a dense footprint (large intermediate arrays). `_dedupe` is greedy $O(N^2)$.

---

## 1) Code improvements (correctness & maintainability)

1. **Avoid recomputing full params just to get amplitudes in the loss.**
   In `loss_fn` you call `model.current_params()` only to L1‑penalize amplitudes; this also rebuilds `L` and computes centers. Use the raw parameter directly:

   ```python
   # replace inside loss_fn
   if l1_amp > 0:
       data = data + l1_amp * torch.mean(torch.abs(F.softplus(model.raw_a)))
   ```

   This shaves measurable time per iteration, especially with many splats.

2. **Vectorize `_build_L` (remove Python loops).**
   Your loop over `d` to fill diagonals and strict lower triangle is fine for small `d`, but vectorization is cleaner and scales better:

   ```python
   i, j = torch.tril_indices(d, d, offset=-1, device=self.raw_L_diag.device)
   L = torch.zeros((N, d, d), dtype=torch.float32, device=self.raw_L_diag.device)
   L[:, torch.arange(d), torch.arange(d)] = diag
   if self.L_off.numel():
       L[:, i, j] = self.L_off  # L_off is (N, n_off)
   ```

3. **Clear docstring mismatch for precision parameterization.**
   `fit_gaussian_splats` advertises a precision‑matrix path (`use_precision_parameterization`) which is **not implemented** in the current code path. Either implement it (see §2.5) or remove the claim to avoid confusion.

4. **Small API polish.**

   * `render_gaussians_numpy` correctly returns `np.ndarray` (good). Keep that consistent; the PyTorch wrapper returns `torch.Tensor`.
   * Add a `dtype` kwarg (default `torch.float32`) to the model and renderer; respect it everywhere to make AMP/half easy to control.
   * Cache `shape` as a `torch.Size` in the model and keep a prebuilt `shape_t_long` buffer to avoid repeated `torch.tensor(shape, device=...)` allocations.

---

## 2) Speed optimizations (faster fitting & rendering)

### 2.1 Grouping splats by AABB size without CPU `.tolist()`

Current `_group_by_box` copies `(hi-lo)` to CPU and builds a Python dict. For large `N` this host sync is costly. Use `torch.unique(..., dim=0)` on GPU:

```python
def _group_by_box(lo: torch.Tensor, hi: torch.Tensor):
    sizes = (hi - lo).to(torch.int32)  # (N, d)
    uniq, inv = torch.unique(sizes, dim=0, return_inverse=True)
    groups = {}
    # NB: only small uniq count goes to CPU for dict keys
    uniq_cpu = uniq.cpu().tolist()
    for g, key in enumerate(map(tuple, uniq_cpu)):
        idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)
        groups[key] = idx
    return groups
```

This keeps the heavy part on device and scales much better.

### 2.2 Reuse more precomputations across groups

* Precompute once **outside** the group loop:

  ```python
  strides = _linear_strides(shape, device)         # (d,)
  base_idx_all = (lo.to(torch.long) * strides).sum(dim=1)  # (N,)
  ```

  Then inside:

  ```python
  base_idx = base_idx_all[idx]  # instead of recomputing
  ```

* Also move `strides_f = strides.to(torch.float32)` **out of** the loop.

### 2.3 Compute `lin_offsets` without float matmul

Avoid `@` with float casts; do pure integer ops:

```python
# base is (d, P) float because of later delta, but build an int view cheaply:
base_long = torch.stack([g.reshape(-1).to(torch.long) for g in grids], dim=0)  # (d, P)
lin_offsets = (base_long.T * strides).sum(dim=1)  # (P,), long
```

This removes a large float matmul on big boxes.

### 2.4 **Chunk the P dimension** (biggest win for stability and large boxes)

The peak tensor size is `(K, d, P)`. For large boxes (e.g., 3D, loose truncation), that explodes. Process points in chunks:

```python
P = base.shape[1]
P_chunk = 131072  # tune or make a parameter
for p0 in range(0, P, P_chunk):
    p1 = min(P, p0 + P_chunk)
    base_slice = base[:, p0:p1]                 # (d, Pc)
    lin_off_slice = lin_offsets[p0:p1]          # (Pc,)
    delta = base_slice[None,:,:] + lo_f[:,:,None] - mu[:,:,None]     # (K,d,Pc)
    y = torch.linalg.solve_triangular(L, delta, upper=False)          # (K,d,Pc)
    expo = (y*y).sum(dim=1)                    # (K,Pc)
    vals = torch.exp(-0.5 * expo) * a[:, None] # (K,Pc)
    idx_flat = (base_idx[:, None] + lin_off_slice[None, :]).reshape(-1)
    out_flat.index_add_(0, idx_flat, vals.reshape(-1))
```

This keeps memory bounded and often improves kernel scheduling (you’ll see smoother utilization in a profiler).

> You can also chunk along **K** if you have many splats in one box.

### 2.5 Optional: **Precision‑matrix parameterization** (removes triangular solve)

On several backends (notably MPS), `solve_triangular` is slow. You can parameterize the **precision** matrix $Q=\Sigma^{-1}$ via its Cholesky $U$ (lower‑triangular with positive diagonal), then compute:

$$
\|U\,\Delta\|^2 = \Delta^\top Q \Delta
$$

which is just a **batched matmul** instead of a triangular solve:

* Replace covariance Cholesky parameters (`Lcov`) with precision Cholesky (`U`):

  ```python
  # U diag = sigma_min_inv + softplus(raw) ensures invertible precision
  # Constraint anisotropy (optional): clamp diag or use spectral penalty (§3.4).
  ```

* In the renderer per group:

  ```python
  y = torch.matmul(U, delta)          # (K,d,d) @ (K,d,Pc) -> (K,d,Pc)
  expo = (y*y).sum(dim=1)             # (K,Pc)
  vals = torch.exp(-0.5*expo) * a[:,None]
  ```

This change keeps the same big‑O (both are $O(d^2 P)$) but typically runs faster on GPUs/Metal because matmuls are heavily optimized. It also simplifies AMP.

> Your wrapper already advertises this mode; wiring it in would be a real‑world speedup.

### 2.6 Cache per‑shape base grids and linear offsets across **iterations**

Within a single optimization run, AABB **shapes** tend to fall in a small set. Cache:

```python
# module-global or LRU
_GRID_CACHE = {}  # key: (device, dtype, box_shape) -> (base, base_long, lin_offsets)

def _get_base(box_shape, strides, device, dtype):
    key = (device.type, str(dtype), tuple(int(s) for s in box_shape))
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    ranges = [torch.arange(s, device=device) for s in box_shape]
    grids  = torch.meshgrid(*ranges, indexing="ij")
    base   = torch.stack([g.reshape(-1).to(dtype) for g in grids], dim=0)
    base_l = torch.stack([g.reshape(-1).to(torch.long) for g in grids], dim=0)
    linoff = (base_l.T * _linear_strides(box_shape, device)).sum(dim=1)
    _GRID_CACHE[key] = (base, base_l, linoff)
    return _GRID_CACHE[key]
```

Then just call `_get_base(...)` inside the group loop. This significantly trims per‑iteration overhead.

### 2.7 Optional splat pruning during training

If you use `l1_amp`, you can periodically **skip** negligibly small splats to save compute:

```python
with torch.no_grad():
    amps = F.softplus(model.raw_a)
    keep = amps > prune_thresh  # e.g., 1e-4 or percentile-based
model.mask = keep  # maintain a boolean mask and apply to raw params in forward
```

It’s a trade‑off (non‑smooth), but with conservative thresholds it can cut cost a lot as training progresses.

---

## 3) Memory reductions

1. **Chunk the P dimension** (see §2.4). This is the single most effective control over peak memory in the renderer.

2. **AMP for forward pass (in training and inference).**
   You already support AMP in the fitter; ensure the renderer is covered by autocast. With half precision on CUDA, the `(K,d,Pc)` tensors are halved in size.

3. **Move best‑state checkpoint to CPU.**
   Storing `best_state = model.state_dict()` on GPU doubles parameter VRAM. Immediately move it to CPU:

   ```python
   best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
   # ...
   model.load_state_dict(best_state)  # PyTorch will move tensors back automatically
   ```

   This frees GPU memory during the remainder of training.

4. **Avoid temporary CPU copies in grouping** (see §2.1). Reduces host memory & copy overhead.

5. **Candidates: use `size=` instead of `footprint=` in `maximum_filter`.**
   Building a full boolean footprint array of shape $(2r+1)^d$ is wasteful; `size=[2r+1]*d` uses \~no extra memory and is often faster:

   ```python
   max_f = ndi.maximum_filter(img, size=[2*radius+1]*img.ndim, mode="nearest")
   ```

6. **Candidates: faster de‑duplication.**
   `_dedupe` is $O(N^2)$. Use a KD‑Tree or voxel hashing to find neighbors within `min_dist` in (near) $O(N \log N)$:

   ```python
   from scipy.spatial import cKDTree
   tree = cKDTree(coords)
   keep = np.ones(len(coords), dtype=bool)
   for i in range(len(coords)):
       if not keep[i]: continue
       nbrs = tree.query_ball_point(coords[i], r=min_dist)
       nbrs.remove(i)
       keep[nbrs] = False
   return coords[keep]
   ```

   (Or do a uniform grid hash for pure‑NumPy.)

7. **Cache grid‑centroid coordinates in the intensity grid pass.**
   In the grid step you rebuild `np.meshgrid` for each anchor; pre‑make a relative 0‑based grid once for the chosen box size and add offsets per anchor to compute the weighted centroid faster and with lower allocation.

---

## 4) Numerical/algorithmic knobs (both speed & quality)

1. **Anisotropy constraint (prevents degenerate covariances).**
   Add a gentle penalty on the **log condition number** of Σ (or precision), or simply clamp diag ranges more tightly. This stabilizes training, reduces very large boxes (fewer points in P), and improves runtime.

2. **Amplitude‑aware AABB shrinking: make it opt‑in during training.**
   Using amplitude to shrink the box is good for inference, but during training it introduces discrete box changes. Consider disabling (or using `amps.detach()`) for early iterations, then enabling later to reduce cost.

3. **Learning rate schedule tweaks.**
   You’re already using `ReduceLROnPlateau`. Consider a short warm‑up for smoother early training, which can reduce overshoot and thus extra iterations.

---

## 5) Micro‑optimizations & cleanup

* Hoist repeated small allocations (e.g., `torch.tensor(shape, device, dtype=long)` and `shape_arr`) out of tight code paths.
* Prefer `out.reshape(-1)` over `out.view(-1)` only when you cannot guarantee contiguity (you can here, so `.view(-1)` is fine).
* Replace `for g in grids: P *= g.numel()` with `P = int(np.prod(box_shape))`.
* `zero_grad(set_to_none=True)` – already good.
* Set TF32 where appropriate on Ampere+:

  ```python
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  ```

---

## 6) Candidate generation: speed & robustness checklist

* Use `size=` in max‑filter (see §3.5).
* Reuse blurred images between Gaussian and DoG where possible (you already do two `gaussian_filter` calls per sigma; cache the first if you reuse it).
* Apply `np.float32` early to avoid accidental float64 arrays.
* Limit `peaks_per_scale` aggressively at coarse scales (big neighborhoods are expensive anyway).
* If `min_dist` is large, prefer KD‑Tree/voxel hashing over greedy $O(N^2)$.

---

## 7) Example patches (concise)

**(A) Faster grouping + precomputations**

```python
# --- before group loop ---
strides = _linear_strides(shape, device)
base_idx_all = (lo.to(torch.long) * strides).sum(dim=1)

# GPU grouping
sizes = (hi - lo).to(torch.int32)
uniq, inv = torch.unique(sizes, dim=0, return_inverse=True)

for g in range(uniq.shape[0]):
    box_shape = tuple(int(x) for x in uniq[g].tolist())
    idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)
    mu, L, a = centers[idx], Ls[idx], amps[idx]
    lo_f = lo[idx].to(torch.float32)
    base_idx = base_idx_all[idx]

    # cached base + lin_offsets
    base, base_l, lin_offsets = _get_base(box_shape, strides, device, dtype=torch.float32)

    # chunked P
    P, P_chunk = base.shape[1], 131072
    for p0 in range(0, P, P_chunk):
        p1 = min(P, p0 + P_chunk)
        base_slice = base[:, p0:p1]
        lin_slice  = lin_offsets[p0:p1]
        delta = base_slice[None,:,:] + lo_f[:,:,None] - mu[:,:,None]   # (K,d,Pc)
        y = torch.linalg.solve_triangular(L, delta, upper=False)
        vals = torch.exp(-0.5 * (y*y).sum(dim=1)) * a[:, None]
        idx_flat = (base_idx[:, None] + lin_slice[None, :]).reshape(-1)
        out_flat.index_add_(0, idx_flat, vals.reshape(-1))
```

**(B) Precision parameterization in the renderer (no triangular solve)**

```python
# if parameterized with precision Cholesky U (lower-tri, positive diag):
y = torch.matmul(U, delta)                 # (K,d,d) @ (K,d,Pc) -> (K,d,Pc)
expo = (y*y).sum(dim=1)
vals = torch.exp(-0.5 * expo) * a[:, None]
```

(Plus the corresponding model parameterization and initialization to ensure U gives reasonable initial Σ⁻¹.)

**(C) Move best‑state to CPU**

```python
if current_loss < best_loss:
    best_loss = current_loss
    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
# ...
if best_state is not None:
    model.load_state_dict(best_state)
```

**(D) `maximum_filter` with `size=`**

```python
max_f = ndi.maximum_filter(img, size=[2*radius+1]*img.ndim, mode="nearest")
```

**(E) KD‑Tree dedupe**

```python
from scipy.spatial import cKDTree
def _dedupe(coords, min_dist):
    if len(coords) == 0: return coords.astype(float)
    tree = cKDTree(coords)
    keep = np.ones(len(coords), dtype=bool)
    for i in range(len(coords)):
        if not keep[i]: continue
        for j in tree.query_ball_point(coords[i], r=min_dist):
            if j > i: keep[j] = False
    return coords[keep].astype(float)
```

---

## 8) What to expect (order‑of‑magnitude)

* **Grouping on GPU + precomputations**: 1.2–1.5× faster per forward at large N.
* **Chunked P**: same FLOPs, **much** lower peak memory; avoids OOM on large 3D AABBs; mild speedup from better kernel scheduling.
* **Precision param (matmul)**: backend‑dependent; on MPS often 1.5–3× faster than triangular solves; on CUDA often modest speedup but better AMP behavior.
* **KD‑Tree dedupe + `size=` filter**: 3–10× faster candidate generation on big arrays.

---

## 9) Final checklist to implement safely

* Add unit tests comparing outputs (MSE/Poisson loss, forward images) before/after refactors.
* Guard chunk size with a heuristic based on available VRAM (`torch.cuda.mem_get_info`) if on CUDA.
* Keep the amplitude‑aware truncation the same by default; make its use during training a flag.
* If adding the precision path, keep both code paths selectable and cross‑validate on a suite of images.

If you want, I can produce a PR‑style patch set that implements (2.1–2.4, 3.3) in your current codebase and a minimal precision‑param switch; just say the word and I’ll draft it inline.
