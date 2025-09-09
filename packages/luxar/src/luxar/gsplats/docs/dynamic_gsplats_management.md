
## What your current code already does (quick readback)

* **Model & parameterization** (`GaussianSplatModel` in `gsplat_model.py`)

  * Centers are **logit‑sigmoid** parameterized so they stay inside the image/volume.
  * Covariances use **Cholesky** $Σ = L Lᵀ$, with diagonal constrained via **inverse‑softplus**; off‑diagonals are free.
  * Amplitudes are **non‑negative** via softplus.
  * Forward uses **AABB truncation** and specialized 2D/3D fast paths with per‑tile chunking for memory.

* **Fitting loop** (`fit_gsplats.py`)

  * Adam on all parameters; MSE or Poisson deviance losses; optional L1 on amplitudes.
  * LR scheduler, gradient clipping, early stopping, mixed precision, optional `torch.compile`.
  * **No dynamic point management** yet (no prune/seed/merge/split).

This is a great base: numerically sound parameterization and a fast renderer.

---

## What works in the wild (condensed evidence)

* **Adaptive density control** in 3D Gaussian Splatting alternates optimization with *densification* (duplicate/split high‑gradient, large‑footprint splats) and *pruning* (low opacity/weak points) — a big part of why 3DGS trains quickly and reaches high fidelity. Variants refine the criteria using **per‑pixel error** and different gradient measures. ([arXiv][1], [docs.gsplat.studio][2])

* **Split–Merge EM** for Gaussian mixtures is a classic, effective way to escape poor local optima: *split* overly broad components and *merge* nearly redundant ones using principled criteria (e.g., symmetric KL). ([MLG Cambridge][3], [SpringerLink][4], [ScienceDirect][5])

* **Seeding from residuals**: adding new components at **LoG/DoG or scale‑normalized Laplacian** peaks (or simply local maxima of |residual|) is standard in blob detection and gives good, scale‑aware initializations; anisotropy can be set from a **structure tensor** around the peak. ([scikit-image.org][6], [cvl-umass.github.io][7])

* **Similarity metrics** between Gaussians for merge decisions: **symmetric KL** or **2‑Wasserstein** distance have closed forms for Gaussians; both are stable with Cholesky factors. ([Cross Validated][8], [djalil.chafai.net][9])

---

## Concrete plan (with drop‑in code)

We’ll add **four dynamic operations** that trigger periodically during training:

1. **Prune**: remove splats that are (a) *too weak* (amplitude and usage gradient low), (b) *too tiny* (at the minimum σ with tiny mass), or (c) *negligible contributors* (EMA of |∂L/∂a| below threshold).
2. **Seed**: add splats at **top‑K local maxima of |residual|**, with optional anisotropic init from a local **structure tensor**; amplitude initialized by a one‑shot projection of the residual onto the candidate Gaussian.
3. **Merge**: pairwise **merge near‑duplicate splats** (close in center and shape) using a **symmetric KL** threshold; merged splat uses **moment matching** (mass‑/amplitude‑weighted).
4. **Split**: for **large, high‑error** splats, split along the principal eigenvector of Σ (largest eigenvalue), slightly shrink each child’s Σ, and distribute amplitude (optionally mass‑preserving).

### A. Minimal extensions to `GaussianSplatModel` (append/prune/replace)

Add these methods to `gsplat_model.py` (inside the class) to convert external params to internal raw tensors, and to **prune/append/replace** the parameter sets:

```python
# --- inside GaussianSplatModel ----------------------------------------------

@torch.no_grad()
def _to_internal_params(self,
                        centers: torch.Tensor,   # (N,d)
                        Ls: torch.Tensor,        # (N,d,d)
                        amps: torch.Tensor       # (N,)
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert external (μ, L, a) to raw learnable params (raw_mu, L_diag_raw, L_off, amp_raw)."""
    device = self.raw_mu.device
    d = centers.shape[1]
    # centers -> raw_mu (logit in [0,1] coords)
    shape_arr = torch.tensor(self.shape, device=device, dtype=torch.float32)
    u = torch.clamp(centers / torch.clamp(shape_arr - 1.0, min=1.0), 1e-6, 1.0 - 1e-6)
    raw_mu = torch.log(u) - torch.log(1.0 - u)

    # L -> diag/off raw (diag via inverse-softplus)
    diag = torch.diagonal(Ls, dim1=1, dim2=2)  # (N,d)
    # Avoid zero/neg
    eps = 1e-6
    diag = torch.clamp(diag, min=eps)
    from luxar.gsplats.models.utils.inverse_softplus import stable_inverse_softplus
    L_diag_raw = torch.tensor(stable_inverse_softplus(diag.detach().cpu().numpy()),
                              device=device, dtype=torch.float32)

    # Pack off-diagonals (row-major, below diag)
    off_elems = []
    for i in range(d):
        for j in range(i):
            off_elems.append(Ls[:, i, j])
    L_off = torch.stack(off_elems, dim=1) if len(off_elems) else torch.zeros((centers.shape[0], 0), device=device)

    # amps -> amp_raw
    amps = torch.clamp(amps, min=0.0)
    amp_raw = torch.tensor(stable_inverse_softplus(amps.detach().cpu().numpy()),
                           device=device, dtype=torch.float32)
    return raw_mu, L_diag_raw, L_off, amp_raw

@torch.no_grad()
def replace_with(self, centers: torch.Tensor, Ls: torch.Tensor, amps: torch.Tensor) -> None:
    """Hard replace the whole parameter set."""
    raw_mu, L_diag_raw, L_off, amp_raw = self._to_internal_params(centers, Ls, amps)
    self.raw_mu = torch.nn.Parameter(raw_mu)
    self.L_diag_raw = torch.nn.Parameter(L_diag_raw)
    self.L_off = torch.nn.Parameter(L_off)
    self.amp_raw = torch.nn.Parameter(amp_raw)

@torch.no_grad()
def prune_(self, keep_mask: torch.Tensor) -> None:
    """Keep only indices where keep_mask is True."""
    self.raw_mu = torch.nn.Parameter(self.raw_mu[keep_mask])
    self.L_diag_raw = torch.nn.Parameter(self.L_diag_raw[keep_mask])
    self.L_off = torch.nn.Parameter(self.L_off[keep_mask])
    self.amp_raw = torch.nn.Parameter(self.amp_raw[keep_mask])

@torch.no_grad()
def append_(self, centers_new: torch.Tensor, Ls_new: torch.Tensor, amps_new: torch.Tensor) -> None:
    """Append new splats to the tail."""
    if centers_new.numel() == 0:
        return
    raw_mu, L_diag_raw, L_off, amp_raw = self._to_internal_params(centers_new, Ls_new, amps_new)
    self.raw_mu = torch.nn.Parameter(torch.cat([self.raw_mu, raw_mu], dim=0))
    self.L_diag_raw = torch.nn.Parameter(torch.cat([self.L_diag_raw, L_diag_raw], dim=0))
    self.L_off = torch.nn.Parameter(torch.cat([self.L_off, L_off], dim=0))
    self.amp_raw = torch.nn.Parameter(torch.cat([self.amp_raw, amp_raw], dim=0))

def n_splats(self) -> int:
    return int(self.raw_mu.shape[0])
```

> **Note**: after any `prune_`/`append_`/`replace_with` you must **rebuild the optimizer** (Adam state is invalidated). The trainer below will handle that.

---

### B. Dynamic manager (drop into `fit_gsplats.py`)

Add the following utilities above your `FitGaussianSplats` class (or inside it as `@staticmethod` helpers). They are **dimension‑aware** for 2D/3D, and degrade gracefully for n>3.

```python
import math
from typing import List, Tuple

def _structure_tensor_eigs_nd(res: torch.Tensor, sigma: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate structure tensor eigenvectors/values for 2D/3D residual to orient new Gaussians.
    Returns (eigvals, eigvecs) per voxel for the top-1 direction (we only need principal dir).
    We compute simple Sobel-like gradients and locally smooth by Gaussian blurs (approx via pooling).
    """
    d = res.ndim
    assert d in (2, 3), "Structure tensor helper implemented for 2D/3D."
    # gradients
    if d == 2:
        ky = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=res.dtype, device=res.device)/8.0
        kx = ky.t()
        gy = torch.nn.functional.conv2d(res[None,None], ky[None,None], padding=1)[0,0]
        gx = torch.nn.functional.conv2d(res[None,None], kx[None,None], padding=1)[0,0]
        g = (gx, gy)
    else:
        # 3D: approximate finite differences
        gx = res.roll(-1, dims=2) - res.roll(1, dims=2)
        gy = res.roll(-1, dims=1) - res.roll(1, dims=1)
        gz = res.roll(-1, dims=0) - res.roll(1, dims=0)
        g = (gx/2, gy/2, gz/2)

    # components of structure tensor J = ⟨∇I ∇Iᵀ⟩_window
    # Smooth with a simple box or Gaussian-like pooling (fast & differentiable)
    def _smooth(x):
        if d == 2:
            return torch.nn.functional.avg_pool2d(x[None,None], kernel_size=5, stride=1, padding=2)[0,0]
        else:
            return torch.nn.functional.avg_pool3d(x[None,None], kernel_size=3, stride=1, padding=1)[0,0]

    if d == 2:
        Jxx = _smooth(g[0]*g[0]); Jxy = _smooth(g[0]*g[1]); Jyy = _smooth(g[1]*g[1])
        # eigen decomposition of 2x2 at each voxel: principal eigenpair
        # λ = (Jxx+Jyy ± sqrt((Jxx-Jyy)^2 + 4Jxy^2))/2 ; v = normalized
        tr = Jxx + Jyy
        det = Jxx*Jyy - Jxy*Jxy
        tmp = torch.sqrt(torch.clamp((Jxx-Jyy)**2 + 4*Jxy*Jxy, min=1e-12))
        lam1 = (tr + tmp)/2
        # principal eigenvector (avoid zero)
        vx = torch.where(torch.abs(Jxy) > 1e-12, lam1 - Jyy, torch.ones_like(Jxy))
        vy = torch.where(torch.abs(Jxy) > 1e-12, Jxy, torch.zeros_like(Jxy))
        norm = torch.sqrt(torch.clamp(vx*vx + vy*vy, min=1e-12))
        v1 = torch.stack([vy/norm, vx/norm], dim=0)  # (2,H,W) in y,x order
        return lam1, v1
    else:
        # 3D: build 3x3 tensors and use eigh in small neighborhoods (costly but OK for few seeds)
        # For simplicity we only return placeholders; actual orientation will be computed locally at the seed voxel.
        return None, None

def _local_maxima(res_abs: torch.Tensor, k: int, min_dist: int, thr: float) -> List[Tuple[int,...]]:
    """Top-k local maxima indices in |residual| above threshold using max-pooling NMS (2D/3D)."""
    d = res_abs.ndim
    if d == 1:
        values, idx = torch.topk(res_abs, k)
        coords = [(int(i),) for i in idx.tolist() if values[idx==i] > thr]
        return coords
    ks = [min_dist]*d
    if d == 2:
        mx = torch.nn.functional.max_pool2d(res_abs[None,None], kernel_size=ks, stride=1, padding=min_dist//2)[0,0]
    elif d == 3:
        mx = torch.nn.functional.max_pool3d(res_abs[None,None], kernel_size=ks, stride=1, padding=min_dist//2)[0,0]
    else:
        # Fallback: just take global top-k
        values, flat_idx = torch.topk(res_abs.reshape(-1), k)
        coords = list(torch.unravel_index(flat_idx, res_abs.shape))
        return list(zip(*[c.tolist() for c in coords]))
    mask = (res_abs >= mx) & (res_abs >= thr)
    ys = torch.nonzero(mask, as_tuple=False)
    if ys.shape[0] == 0:
        return []
    vals = res_abs[tuple(ys.t())]
    topk = min(k, ys.shape[0])
    v, order = torch.topk(vals, topk)
    sel = ys[order]
    return [tuple(int(i) for i in s.tolist()) for s in sel]

def _moment_match_merge(mu1, L1, a1, mu2, L2, a2):
    """Merge two Gaussians via moment matching; return (mu, L, a)."""
    a = a1 + a2
    if a <= 0:  # degenerate
        return mu1, L1, torch.zeros_like(a1)
    mu = (a1*mu1 + a2*mu2) / a
    # Σ = Σ_k + (μ_k-μ)(μ_k-μ)ᵀ (amplitude-weighted)
    S1 = L1 @ L1.transpose(-1, -2)
    S2 = L2 @ L2.transpose(-1, -2)
    d1 = (mu1 - mu).unsqueeze(-1); d2 = (mu2 - mu).unsqueeze(-1)
    Sigma = (a1*(S1 + d1@d1.transpose(-1,-2)) + a2*(S2 + d2@d2.transpose(-1,-2))) / a
    # Ensure PD
    # add a tiny jitter for stability
    d = mu.shape[-1]
    Sigma = Sigma + 1e-6*torch.eye(d, device=Sigma.device)[None]
    L = torch.linalg.cholesky(Sigma)
    return mu, L, a

def _sym_kl_gaussians(mu1, L1, mu2, L2):
    """Symmetric KL between N(mu1,Σ1) and N(mu2,Σ2); all tensors are (d,) or (d,d)."""
    # KL(N1||N2) = 0.5[ tr(Σ2^-1 Σ1) + (μ2-μ1)^T Σ2^-1 (μ2-μ1) - d + ln(detΣ2/detΣ1) ]
    d = mu1.shape[0]
    # Σ2^{-1} Σ1  → Frobenius norm of solve(L2, L1)
    Y21 = torch.linalg.solve_triangular(L2, L1, upper=False)          # L2^{-1} L1
    tr21 = torch.sum(Y21*Y21)                                         # ||·||_F^2 = tr(...)
    dm = (mu2 - mu1)
    y = torch.linalg.solve_triangular(L2, dm, upper=False)            # L2^{-1}(μ2-μ1)
    q21 = torch.dot(y, y)
    logdet1 = 2.0*torch.log(torch.diagonal(L1)).sum()
    logdet2 = 2.0*torch.log(torch.diagonal(L2)).sum()
    kl12 = 0.5*(tr21 + q21 - d + (logdet2 - logdet1))

    # reverse
    Y12 = torch.linalg.solve_triangular(L1, L2, upper=False)
    tr12 = torch.sum(Y12*Y12)
    y2 = torch.linalg.solve_triangular(L1, -dm, upper=False)
    q12 = torch.dot(y2, y2)
    kl21 = 0.5*(tr12 + q12 - d + (logdet1 - logdet2))
    return 0.5*(kl12 + kl21)

def _estimate_amp_from_residual(shape, center, L, residual, truncate=3.0):
    """
    One-shot amplitude estimate a ≈ <r,g>/<g,g> where g is a unit-amplitude Gaussian.
    Uses model's renderer on a single splat; residual may be the full tensor on device.
    """
    from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians
    centers = center[None]
    Ls = L[None]
    amps = torch.ones((1,), device=center.device, dtype=torch.float32)
    g = render_gaussians(shape, centers, Ls, amps, truncate=truncate)
    num = torch.sum(residual * g)
    den = torch.sum(g * g) + 1e-12
    a = torch.clamp(num / den, min=0.0)  # non-negative model
    return a
```

Now the **dynamic step** that runs every `N` iterations:

```python
class _DynamicOpsConfig:
    # Pruning
    amp_abs_min: float = 1e-4          # absolute amplitude cutoff
    amp_grad_ema_min: float = 1e-5     # minimal EMA(|∂L/∂a|) to keep
    min_volume_det: float = 1e-6       # drop if det(Σ) tiny AND amp small
    patience_decay: int = 15           # #iters of low grad before prune
    # Seeding
    max_add_per_step: int = 64
    residual_quantile: float = 0.98    # seed above this |residual| quantile
    nms_radius_vox: int = 7
    seed_sigma_vox: float = 1.2
    # Merging
    merge_dist_vox: float = 1.5        # centers within this (in voxels)
    merge_symkl_max: float = 0.1
    max_merges_per_step: int = 64
    # Splitting
    split_eig_thr: float = 2.5         # sqrt(λ_max) threshold (voxels)
    split_error_quantile: float = 0.95 # split only if local |res| high
    split_shrink: float = 0.75         # child L = shrink * parent L
    split_offset_alpha: float = 0.6    # offset = alpha * sqrt(λ_max) * v1
    # Scheduling
    step_every: int = 10               # run dynamics every N iters
    do_prune: bool = True
    do_seed: bool = True
    do_merge: bool = True
    do_split: bool = True

class _DynamicState:
    def __init__(self):
        self.amp_grad_ema = None        # torch.Tensor [N]
        self.low_grad_counters = None   # torch.LongTensor [N]
```

**Integrate in your training loop** (inside `fit()` right after `opt.step()` when grads exist). Insert **once** after `opt.step()`:

```python
# inside FitGaussianSplats.fit(), after opt.step() / scaler.step():
if dyn_state.amp_grad_ema is None or dyn_state.amp_grad_ema.shape[0] != model.n_splats():
    N = model.n_splats()
    dyn_state.amp_grad_ema = torch.zeros(N, device=device)
    dyn_state.low_grad_counters = torch.zeros(N, dtype=torch.long, device=device)

# Update EMA of |∂L/∂a| using chain rule: dL/da = dL/da_raw * softplus'(a_raw)
if model.amp_raw.grad is not None:
    # softplus'(x) = sigmoid(x)
    dL_da = torch.sigmoid(model.amp_raw) * model.amp_raw.grad.abs()
    dyn_state.amp_grad_ema = 0.9 * dyn_state.amp_grad_ema + 0.1 * dL_da.detach()
    low = (dyn_state.amp_grad_ema < cfg.amp_grad_ema_min).to(torch.long)
    dyn_state.low_grad_counters = dyn_state.low_grad_counters + low
    # reset counters where gradient is healthy
    dyn_state.low_grad_counters = dyn_state.low_grad_counters * (low > 0).to(torch.long)
```

Then, **every `cfg.step_every` iterations**, run the dynamic pass:

```python
if it % cfg.step_every == 0:
    with torch.no_grad():
        pred = model()  # current reconstruction
        residual = (V_t - pred)
        res_abs = residual.abs()

        # === PRUNE ===
        if cfg.do_prune:
            centers, Ls, amps = model.current_params()
            d = centers.shape[1]
            diag = torch.diagonal(Ls, dim1=1, dim2=2)  # (N,d)
            det = torch.prod(diag, dim=1)**2           # since det(Σ)= (∏ diag(L))^2
            weak = amps < cfg.amp_abs_min
            tiny = (det < cfg.min_volume_det) & (amps < 10*cfg.amp_abs_min)
            stale = dyn_state.low_grad_counters >= cfg.patience_decay
            keep = ~(weak | tiny | stale)
            if keep.sum() < keep.numel():
                model.prune_(keep)
                # shrink state
                dyn_state.amp_grad_ema = dyn_state.amp_grad_ema[keep]
                dyn_state.low_grad_counters = dyn_state.low_grad_counters[keep]
                # Rebuild optimizer & scheduler after topology change
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)

        # === MERGE ===
        if cfg.do_merge and model.n_splats() > 1:
            centers, Ls, amps = model.current_params()
            N = centers.shape[0]
            # candidate pairs: use distance-based culling
            # (simple O(N^2) for moderate N; replace with grid hashing if N is large)
            if N <= 3000:
                dists = torch.cdist(centers, centers)
                cand = torch.nonzero((dists < cfg.merge_dist_vox) & (torch.triu(torch.ones_like(dists), 1) > 0), as_tuple=False)
            else:
                cand = torch.empty((0,2), dtype=torch.long, device=device)
            merged = []
            take = []
            for (i,j) in cand.tolist():
                if i in merged or j in merged:
                    continue
                d = centers.shape[1]
                skl = _sym_kl_gaussians(centers[i], Ls[i], centers[j], Ls[j])
                if skl.item() <= cfg.merge_symkl_max:
                    mu,L,a = _moment_match_merge(centers[i], Ls[i], amps[i], centers[j], Ls[j], amps[j])
                    take.append((i,j,mu,L,a))
                    merged.extend([i,j])
                    if len(take) >= cfg.max_merges_per_step:
                        break
            if take:
                keep_mask = torch.ones(model.n_splats(), dtype=torch.bool, device=device)
                for i,j,mu,L,a in take:
                    keep_mask[i] = False
                    keep_mask[j] = False
                model.prune_(keep_mask)
                # append merged ones
                centers_new = torch.stack([t[2] for t in take], dim=0)
                Ls_new = torch.stack([t[3] for t in take], dim=0)
                amps_new = torch.stack([t[4] for t in take], dim=0)
                model.append_(centers_new, Ls_new, amps_new)
                # reset opt
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
                # reset state
                dyn_state.amp_grad_ema = None
                dyn_state.low_grad_counters = None

        # === SPLIT ===
        if cfg.do_split and model.n_splats() > 0:
            centers, Ls, amps = model.current_params()
            d = centers.shape[1]
            # find candidates with large principal axis AND high residual where they live
            Sigma = Ls @ Ls.transpose(-1,-2)  # (N,d,d)
            eigvals, eigvecs = torch.linalg.eigh(Sigma)   # ascending
            lam_max = eigvals[:, -1].clamp(min=1e-12)
            v1 = eigvecs[:, :, -1]
            rad = lam_max.sqrt()                           # std along v1
            large = rad > cfg.split_eig_thr

            # sample residual at each center (clamp within bounds)
            grid_idx = torch.round(centers).long().T       # (d,N)
            for dd in range(d):
                grid_idx[dd] = torch.clamp(grid_idx[dd], 0, res_abs.shape[dd]-1)
            local_res = res_abs[tuple(grid_idx)]
            high_err = local_res >= torch.quantile(res_abs, cfg.split_error_quantile)
            to_split = torch.nonzero(large & high_err, as_tuple=False).squeeze(1)

            if to_split.numel() > 0:
                mu_children = []
                L_children = []
                a_children = []
                kill_mask = torch.ones(model.n_splats(), dtype=torch.bool, device=device)
                for idx in to_split.tolist():
                    mu0 = centers[idx]
                    L0 = Ls[idx]
                    a0 = amps[idx]
                    offset = cfg.split_offset_alpha * rad[idx] * v1[idx]
                    # child covariances (shrink)
                    Lc = cfg.split_shrink * L0
                    # child centers
                    muA = mu0 - offset
                    muB = mu0 + offset
                    # distribute amplitude; compensate shrink to roughly preserve integral
                    dV = (1.0 / (cfg.split_shrink ** d))
                    aA = 0.5 * a0 * dV
                    aB = 0.5 * a0 * dV
                    mu_children.extend([muA, muB])
                    L_children.extend([Lc, Lc])
                    a_children.extend([aA, aB])
                    kill_mask[idx] = False

                model.prune_(kill_mask)
                model.append_(torch.stack(mu_children), torch.stack(L_children), torch.stack(a_children))
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
                dyn_state.amp_grad_ema = None
                dyn_state.low_grad_counters = None

        # === SEED ===
        if cfg.do_seed:
            # threshold for residual peaks
            thr = torch.quantile(res_abs, cfg.residual_quantile)
            coords = _local_maxima(res_abs, k=cfg.max_add_per_step, min_dist=cfg.nms_radius_vox, thr=thr)
            if len(coords) > 0:
                d = len(model.shape)
                centers_new = []
                Ls_new = []
                amps_new = []
                for c in coords:
                    mu = torch.tensor(c, dtype=torch.float32, device=device)
                    # Init isotropic L; optionally orient via structure tensor (2D)
                    Linit = torch.diag(torch.full((d,), cfg.seed_sigma_vox, device=device))
                    if d == 2:
                        # approximate principal axis from structure tensor at the seed
                        lam1, v1 = _structure_tensor_eigs_nd(res_abs)
                        if lam1 is not None:
                            v = v1[:, c[0], c[1]]  # (2,)
                            # build rotation matrix to align yx axes to v (optional small anisotropy boost)
                            # keep it simple: slightly elongate along v
                            R = torch.stack([v, torch.tensor([-v[1], v[0]], device=device)], dim=1)  # 2x2
                            A = torch.diag(torch.tensor([1.15*cfg.seed_sigma_vox, 0.85*cfg.seed_sigma_vox], device=device))
                            Linit = R @ A
                    a_est = _estimate_amp_from_residual(model.shape, mu, Linit, residual, truncate=model.truncate)
                    # Only keep if meaningful
                    if a_est.item() > cfg.amp_abs_min:
                        centers_new.append(mu)
                        Ls_new.append(Linit)
                        amps_new.append(a_est)
                if centers_new:
                    model.append_(torch.stack(centers_new), torch.stack(Ls_new), torch.stack(amps_new))
                    opt = torch.optim.Adam(model.parameters(), lr=lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
                    dyn_state.amp_grad_ema = None
                    dyn_state.low_grad_counters = None
```

**How to enable**: at the beginning of `fit()`, create:

```python
cfg = _DynamicOpsConfig()
dyn_state = _DynamicState()
```

You can expose `cfg` as arguments to `fit_gaussian_splats(...)` so you can tune thresholds or turn features on/off.

> **Why these choices?**
>
> * *Prune*: mirrors 3DGS practice of removing low‑opacity/low‑use points, but we use **amplitude and gradient‑of‑use** proxies appropriate for your additive intensity model. ([docs.gsplat.studio][2])
> * *Seed*: follows **pixel‑error driven densification** suggested by recent 3DGS revisits, using top‑K local maxima of |residual|; LoG/DoG can replace the max‑pool NMS if desired. ([arXiv][10], [scikit-image.org][6])
> * *Merge*: split–merge EM recommends merging based on likelihood change; **symmetric KL** is a good, fast surrogate with closed form using Cholesky. ([MLG Cambridge][3], [Cross Validated][8])
> * *Split*: 3DGS splits large, high‑gradient splats; we generalize to n‑D by splitting **along the top eigenvector** of Σ where residual is large. ([docs.gsplat.studio][2])

---

## Practical knobs & defaults (start here)

* **Every 10 iters**: run dynamics.
* **Prune** if `amp < 1e-4`, or `det(Σ) < 1e-6 and amp < 1e-3`, or **EMA(|∂L/∂a|)** below `1e-5` for **15** checks.
* **Seed** top‑64 peaks with NMS radius 7 voxels above the **98th percentile** of |residual|; init `σ ≈ 1.2 vox`.
* **Merge** pairs within **1.5 vox** with `symKL < 0.1` (cap at 64 merges/step).
* **Split** if `√λ_max > 2.5 vox` *and* local |residual| is above **95th percentile**; two children with `L_child = 0.75 L`, centers at `±0.6 σ_max v1`, amplitudes adjusted by `1/(shrink^d)` to roughly preserve integral.

These defaults work well as a starting point; adjust for your dataset’s SNR and scale.

---

## Extra improvements (small changes, big gains)

1. **Anneal σ‑bounds** (coarse‑to‑fine): start with larger `sigma_min_diag` (e.g., 1.5–2.0 vox), and reduce linearly over training to your final bound (e.g., 0.7). Encourages capturing broad trends first, then detail (like MIP‑Nerf schedules). ([arXiv][1])

2. **Amplitude L1 warm‑start**: start with `l1_amp ≈ 1e-3` for 50–100 iters to encourage pruning, then decay → `0`. This yields sparser solutions earlier, making merge/split decisions cleaner.

3. **Trust‑region steps for centers**: clamp center updates per step to ≤ 0.5·σ\_min along each axis (cheap guard against overshooting in narrow Gaussians).

4. **Occasional “NNLS‑like” amplitude refresh (optional)**: every \~50–100 iters, freeze μ, L and perform 5–10 closed‑form **amplitude projection** steps $a_k ← ⟨r,g_k⟩/⟨g_k,g_k⟩, clamped ≥0$. It often reduces ringing and speeds convergence.

5. **Error‑aware truncation**: modestly increase `truncate` from 3.0 to 3.5 during the error‑driven phase (seeding/splitting), then return to 3.0 for speed. This reduces bias at edges during growth.

---

## Small integration example

In your `fit_gaussian_splats(...)`, after constructing `fitter = FitGaussianSplats(...)`, pass dynamic flags:

```python
params, amps, stats = fitter.fit(
    V=V,
    centers_overcomplete=centers_overcomplete,
    n_iters=n_iters,
    lr=lr,
    loss_type=loss_type,
    l1_amp=L1_AMP,
    truncate=truncate,
    # dynamic config (expose as kwargs in your API if you like)
    # dynamic_cfg=dict(step_every=10, do_merge=True, do_split=True, do_seed=True, do_prune=True)
)
```

(If you prefer, store `cfg` as part of `FitGaussianSplats.__init__` and wire it similarly to `compile_model`.)

---

## Why this should help

* **Fewer, better splats**: continuous pruning + merging fights parameter bloat. (This mirrors opacity‑based pruning in 3DGS.) ([docs.gsplat.studio][2])
* **Faster convergence**: seeding at **residual peaks** reduces the optimizer’s burden to move faraway splats; splitting aligns to local anisotropy, improving edge/detail modeling quickly. ([arXiv][10], [scikit-image.org][6])
* **Stability**: EMA of |∂L/∂a| provides a use‑aware criterion that’s robust to noisy amplitudes; symmetric KL and moment matching make merges principled. ([Cross Validated][8])

---

## Notes & caveats

* Changing topology requires **re‑creating the optimizer**. The snippets above reset Adam & scheduler when we prune/append/merge/split. That’s simplest and works well in practice.
* Your renderer already has an `intensity_floor` parameter; if you find many splats contribute tiny values, consider increasing it slightly during growth, then lowering for final refinement.
* If your data exhibit **negative residual structures** you wish to model explicitly, you could extend amplitudes to **signed** (e.g., two softplus parameters a⁺, a⁻ with a = a⁺ − a⁻ and a small TV/L1 penalty on a⁻). For now, the non‑negative model works well with positive residual seeding.

---

## References (selected)

* **3DGS & density control (prune/split/duplicate)**: Kerbl et al., 2023; gsplat default strategy; error‑driven revisions. ([arXiv][1], [docs.gsplat.studio][2])
* **Learned pruning** (masking): LP‑3DGS. ([shengjie-xu.com][11], [NeurIPS Proceedings][12])
* **Split–Merge EM** for GMMs (foundational for merge/split logic). ([MLG Cambridge][3], [SpringerLink][4])
* **Seeding at scale‑normalized maxima**: LoG/DoG blob detection; structure tensor orientation. ([scikit-image.org][6], [cvl-umass.github.io][7])
* **Symmetric KL / Gaussian distances** (for merges). ([Cross Validated][8], [djalil.chafai.net][9])

---

If you’d like, I can fold these snippets directly into your two files (`gsplat_model.py` and `fit_gsplats.py`) so you can run a quick A/B: baseline vs. dynamic. Just say which thresholds you want exposed as function args, and I’ll wire them cleanly.

[1]: https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
[2]: https://docs.gsplat.studio/main/apis/strategy.html?utm_source=chatgpt.com "Densification - gsplat documentation"
[3]: https://mlg.eng.cam.ac.uk/pub/pdf/UedNakGha00b.pdf?utm_source=chatgpt.com "Split and Merge EM Algorithm for Improving Gaussian Mixture Density ..."
[4]: https://link.springer.com/article/10.1023/A%3A1008155703044?utm_source=chatgpt.com "Split and Merge EM Algorithm for Improving Gaussian Mixture Density ..."
[5]: https://www.sciencedirect.com/science/article/pii/S0031320303000591?utm_source=chatgpt.com "EM algorithms for Gaussian mixtures with split-and-merge operation"
[6]: https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_blob.html?utm_source=chatgpt.com "Blob Detection — skimage 0.25.2 documentation - scikit-image"
[7]: https://cvl-umass.github.io/intro-cv-spring-2025/assets/slides/lec11_blob_detection.pdf?utm_source=chatgpt.com "lec11_blob_detection"
[8]: https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians?utm_source=chatgpt.com "KL divergence between two multivariate Gaussians"
[9]: https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/?utm_source=chatgpt.com "Wasserstein distance between two Gaussians – Libres pensées d'un ..."
[10]: https://arxiv.org/abs/2404.06109?utm_source=chatgpt.com "[2404.06109] Revising Densification in Gaussian Splatting"
[11]: https://shengjie-xu.com/pdfs/CMSC848B_project.pdf?utm_source=chatgpt.com "Event3DGS: Event-based 3D Gaussian Splatting for Real-Time Radiance ..."
[12]: https://proceedings.neurips.cc/paper_files/paper/2024/file/dd51dbce305433cd60910dc5b0147be4-Paper-Conference.pdf?utm_source=chatgpt.com "LP-3DGS: Learning to Prune 3D Gaussian Splatting"
