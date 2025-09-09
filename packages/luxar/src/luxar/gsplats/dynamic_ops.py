# dynamic_ops.py

"""
Dynamic Gaussian Splat Management Operations

This module implements sophisticated dynamic operations for adaptive Gaussian splatting,
allowing the model to automatically adjust its complexity during optimization.

## Core Operations

### 1. Prune: Intelligent Splat Removal
Removes splats that contribute minimally to reconstruction quality:
- **Weak splats**: Amplitude below threshold (insignificant contribution)
- **Tiny splats**: At minimum size with low mass (over-constrained)
- **Stagnant splats**: Low gradient activity over time (optimization stuck)

### 2. Seed: Adaptive Splat Addition
Adds new splats at locations with high reconstruction error:
- **Residual analysis**: Identifies peaks in |target - reconstruction|
- **Structure tensor**: Orients new splats along principal gradient directions (2D)
- **Amplitude estimation**: Initializes amplitude via least-squares projection
- **Non-maximum suppression**: Prevents clustering of new splats

### 3. Merge: Redundancy Elimination
Combines near-duplicate splats using principled criteria:
- **Spatial proximity**: Centers within distance threshold
- **Shape similarity**: Symmetric KL divergence between Gaussian distributions
- **Moment matching**: Preserves first and second moments of merged distribution

### 4. Split: Large Splat Subdivision
Divides poorly-fitting large splats along their principal axis:
- **Size criterion**: Principal eigenvalue above threshold
- **Error criterion**: High residual in splat's spatial region
- **Eigenvector splitting**: Split along direction of maximum variance
- **Volume preservation**: Maintain approximate total mass

## Mathematical Framework

### Gaussian Parameterization
Each splat is represented as N(μ, Σ) where:
- μ ∈ ℝᵈ: center position
- Σ = LLᵀ: covariance via Cholesky decomposition
- a ∈ ℝ₊: amplitude (height/mass)

### Key Algorithms
- **Symmetric KL divergence**: D_KL(P||Q) + D_KL(Q||P) for merge decisions
- **Structure tensor**: Local gradient orientation for anisotropic seeding
- **Moment matching**: μ_new = (a₁μ₁ + a₂μ₂)/(a₁ + a₂) for merging
- **Principal component analysis**: Split along max eigenvalue direction

## Performance Considerations

### Computational Complexity
- Prune: O(N) - linear in number of splats
- Seed: O(HW log K) - proportional to image size, K candidates
- Merge: O(N²) naive, O(N log N) with spatial indexing
- Split: O(N) - eigenvalue decomposition per candidate

### Memory Efficiency
- Operations modify model parameters in-place when possible
- Batch processing reduces memory fragmentation
- Gradient state preservation for unchanged splats

These operations are designed to run periodically (every 20-100 iterations)
during optimization to maintain optimal model complexity and improve convergence.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


class DynamicOpsConfig:
    """Configuration for dynamic Gaussian splat operations."""

    def __init__(self):
        # Pruning parameters
        self.amp_abs_min: float = 1e-4  # absolute amplitude cutoff
        self.min_volume_det: float = 1e-6  # drop if det(Σ) tiny AND amp small

        # Seeding parameters
        self.max_add_per_step: int = 1
        self.residual_quantile: float = 0.5  # seed above this |residual| quantile
        self.nms_radius_vox: int = 7
        self.seed_sigma_vox: float = 1.2

        # Merging parameters
        self.merge_dist_vox: float = 1.5  # centers within this (in voxels)
        self.merge_symkl_max: float = 0.01
        self.max_merges_per_step: int = 8

        # Splitting parameters
        self.split_eig_thr: float = 2.5  # sqrt(λ_max) threshold (voxels)
        self.split_error_quantile: float = 0.95  # split only if local |res| high
        self.split_shrink: float = 0.75  # child L = shrink * parent L
        self.split_offset_alpha: float = 0.6  # offset = alpha * sqrt(λ_max) * v1

        # Scheduling parameters
        self.step_every: int = 50  # run dynamics every N iters
        self.do_prune: bool = True
        self.do_seed: bool = True
        self.do_merge: bool = False
        self.do_split: bool = False


def _structure_tensor_eigs_nd(
    res: torch.Tensor, sigma: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Approximate structure tensor eigenvectors/values for 2D/3D residual to orient new Gaussians.
    Returns (eigvals, eigvecs) per voxel for the top-1 direction (we only need principal dir).
    We compute simple Sobel-like gradients and locally smooth by Gaussian blurs (approx via pooling).
    """
    d = res.ndim
    assert d in (2, 3), "Structure tensor helper implemented for 2D/3D."

    # gradients
    if d == 2:
        ky = (
            torch.tensor(
                [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=res.dtype, device=res.device
            )
            / 8.0
        )
        kx = ky.t()
        gy = torch.nn.functional.conv2d(res[None, None], ky[None, None], padding=1)[
            0, 0
        ]
        gx = torch.nn.functional.conv2d(res[None, None], kx[None, None], padding=1)[
            0, 0
        ]
        g = (gx, gy)
    else:
        # 3D: approximate finite differences
        gx = res.roll(-1, dims=2) - res.roll(1, dims=2)
        gy = res.roll(-1, dims=1) - res.roll(1, dims=1)
        gz = res.roll(-1, dims=0) - res.roll(1, dims=0)
        g = (gx / 2, gy / 2, gz / 2)

    # components of structure tensor J = ⟨∇I ∇Iᵀ⟩_window
    # Smooth with a simple box or Gaussian-like pooling (fast & differentiable)
    def _smooth(x):
        if d == 2:
            return torch.nn.functional.avg_pool2d(
                x[None, None], kernel_size=5, stride=1, padding=2
            )[0, 0]
        else:
            return torch.nn.functional.avg_pool3d(
                x[None, None], kernel_size=3, stride=1, padding=1
            )[0, 0]

    if d == 2:
        Jxx = _smooth(g[0] * g[0])
        Jxy = _smooth(g[0] * g[1])
        Jyy = _smooth(g[1] * g[1])
        # eigen decomposition of 2x2 at each voxel: principal eigenpair
        # λ = (Jxx+Jyy ± sqrt((Jxx-Jyy)^2 + 4Jxy^2))/2 ; v = normalized
        tr = Jxx + Jyy
        _det = Jxx * Jyy - Jxy * Jxy  # Keep for potential future use
        tmp = torch.sqrt(torch.clamp((Jxx - Jyy) ** 2 + 4 * Jxy * Jxy, min=1e-12))
        lam1 = (tr + tmp) / 2
        # principal eigenvector (avoid zero)
        vx = torch.where(torch.abs(Jxy) > 1e-12, lam1 - Jyy, torch.ones_like(Jxy))
        vy = torch.where(torch.abs(Jxy) > 1e-12, Jxy, torch.zeros_like(Jxy))
        norm = torch.sqrt(torch.clamp(vx * vx + vy * vy, min=1e-12))
        v1 = torch.stack([vy / norm, vx / norm], dim=0)  # (2,H,W) in y,x order
        return lam1, v1
    else:
        # 3D: build 3x3 tensors and use eigh in small neighborhoods (costly but OK for few seeds)
        # For simplicity we only return placeholders; actual orientation will be computed locally at the seed voxel.
        return None, None


def _local_maxima(
    res_abs: torch.Tensor, k: int, min_dist: int, thr: float
) -> List[Tuple[int, ...]]:
    """Top-k local maxima indices in |residual| above threshold using max-pooling NMS (2D/3D)."""
    d = res_abs.ndim
    if d == 1:
        values, idx = torch.topk(res_abs, k)
        coords = [(int(i),) for i in idx.tolist() if values[idx == i] > thr]
        return coords

    # Ensure kernel size is odd for symmetric padding
    ks = [min_dist if min_dist % 2 == 1 else min_dist + 1 for _ in range(d)]
    pad = [k // 2 for k in ks]

    if d == 2:
        mx = torch.nn.functional.max_pool2d(
            res_abs[None, None], kernel_size=ks, stride=1, padding=pad
        )[0, 0]
    elif d == 3:
        mx = torch.nn.functional.max_pool3d(
            res_abs[None, None], kernel_size=ks, stride=1, padding=pad
        )[0, 0]
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
    mu = (a1 * mu1 + a2 * mu2) / a
    # Σ = Σ_k + (μ_k-μ)(μ_k-μ)ᵀ (amplitude-weighted)
    S1 = L1 @ L1.T
    S2 = L2 @ L2.T
    d1 = (mu1 - mu).unsqueeze(-1)
    d2 = (mu2 - mu).unsqueeze(-1)
    Sigma = (a1 * (S1 + d1 @ d1.T) + a2 * (S2 + d2 @ d2.T)) / a
    # Ensure PD
    # add a tiny jitter for stability
    d = mu.shape[-1]
    Sigma = Sigma + 1e-6 * torch.eye(d, device=Sigma.device)
    L = torch.linalg.cholesky(Sigma)
    return mu, L, a


def _sym_kl_gaussians(mu1, L1, mu2, L2):
    """Symmetric KL between N(mu1,Σ1) and N(mu2,Σ2); all tensors are (d,) or (d,d)."""
    # KL(N1||N2) = 0.5[ tr(Σ2^-1 Σ1) + (μ2-μ1)^T Σ2^-1 (μ2-μ1) - d + ln(detΣ2/detΣ1) ]
    d = mu1.shape[0]
    # Σ2^{-1} Σ1  → Frobenius norm of solve(L2, L1)
    Y21 = torch.linalg.solve_triangular(L2, L1, upper=False)  # L2^{-1} L1
    tr21 = torch.sum(Y21 * Y21)  # ||·||_F^2 = tr(...)
    dm = (mu2 - mu1).unsqueeze(-1)  # Make it (d, 1) for triangular solve
    y = torch.linalg.solve_triangular(L2, dm, upper=False)  # L2^{-1}(μ2-μ1)
    q21 = torch.dot(y.squeeze(), y.squeeze())
    logdet1 = 2.0 * torch.log(torch.diagonal(L1)).sum()
    logdet2 = 2.0 * torch.log(torch.diagonal(L2)).sum()
    kl12 = 0.5 * (tr21 + q21 - d + (logdet2 - logdet1))

    # reverse
    Y12 = torch.linalg.solve_triangular(L1, L2, upper=False)
    tr12 = torch.sum(Y12 * Y12)
    y2 = torch.linalg.solve_triangular(L1, (-dm), upper=False)
    q12 = torch.dot(y2.squeeze(), y2.squeeze())
    kl21 = 0.5 * (tr12 + q12 - d + (logdet1 - logdet2))
    return 0.5 * (kl12 + kl21)


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


def apply_dynamic_operations(
    model,
    optimizer,
    scheduler,
    V_t: torch.Tensor,
    cfg: DynamicOpsConfig,
    lr: float,
    device: torch.device,
    verbose: bool = False,
    napari_debug: bool = True,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, bool]:
    """
    Apply all enabled dynamic operations (prune, merge, split, seed).
    Returns new optimizer, scheduler, and whether operations occurred.

    Args:
        napari_debug: If True, display before/after visualizations using napari

    Returns:
        tuple: (optimizer, scheduler, operations_occurred)
               operations_occurred=True if any splats were added/removed/merged
    """
    from arbol import aprint, asection

    with torch.no_grad():
        pred = model()  # current reconstruction
        residual = V_t - pred
        res_abs = residual.abs()

        # Track if we modified model topology (requires optimizer rebuild)
        topology_changed = False
        operations_performed = []

        # Track operation details for visualization
        operation_details = {
            "pruned_locations": [],
            "seeded_locations": [],
            "merged_pairs": [],
            "split_locations": [],
        }

        # === PRUNE ===
        if cfg.do_prune and model.n_splats() > 0:
            centers, Ls, amps = model.current_params()
            diag = torch.diagonal(Ls, dim1=1, dim2=2)  # (N,d)
            det = torch.prod(diag, dim=1) ** 2  # since det(Σ)= (∏ diag(L))^2
            weak = amps < cfg.amp_abs_min
            tiny = (det < cfg.min_volume_det) & (amps < 10 * cfg.amp_abs_min)

            prune_mask = weak | tiny
            keep = ~prune_mask
            n_pruned = prune_mask.sum().item()

            if n_pruned > 0:
                n_weak = weak.sum().item()
                n_tiny = tiny.sum().item()

                # Store pruned locations for visualization
                pruned_centers = centers[prune_mask].cpu().numpy()
                operation_details["pruned_locations"] = pruned_centers.tolist()

                model.prune_(keep)
                topology_changed = True

                prune_details = f"Pruned {n_pruned} splats"
                if n_weak > 0:
                    prune_details += f" (weak:{n_weak}"
                if n_tiny > 0:
                    prune_details += (
                        f", tiny:{n_tiny}" if n_weak > 0 else f" (tiny:{n_tiny}"
                    )
                if n_weak > 0 or n_tiny > 0:
                    prune_details += ")"
                operations_performed.append(prune_details)

        # === MERGE ===
        if cfg.do_merge and model.n_splats() > 1:
            centers, Ls, amps = model.current_params()
            N = centers.shape[0]
            # candidate pairs: use distance-based culling
            # (simple O(N^2) for moderate N; replace with grid hashing if N is large)
            if N <= 3000:
                dists = torch.cdist(centers, centers)
                cand = torch.nonzero(
                    (dists < cfg.merge_dist_vox)
                    & (torch.triu(torch.ones_like(dists), 1) > 0),
                    as_tuple=False,
                )
            else:
                cand = torch.empty((0, 2), dtype=torch.long, device=device)

            merged = []
            take = []
            n_candidates = len(cand)
            n_kl_rejected = 0

            for i, j in cand.tolist():
                if i in merged or j in merged:
                    continue
                d = centers.shape[1]
                skl = _sym_kl_gaussians(centers[i], Ls[i], centers[j], Ls[j])
                if skl.item() <= cfg.merge_symkl_max:
                    mu, L, a = _moment_match_merge(
                        centers[i], Ls[i], amps[i], centers[j], Ls[j], amps[j]
                    )
                    take.append((i, j, mu, L, a))
                    merged.extend([i, j])
                    if len(take) >= cfg.max_merges_per_step:
                        break
                else:
                    n_kl_rejected += 1

            if take:
                n_merged_pairs = len(take)
                n_removed = 2 * n_merged_pairs  # Each merge removes 2 splats

                # Store merge pairs for visualization
                merge_pairs = []
                for i, j, mu, L, a in take:
                    pair_info = {
                        "old_centers": [
                            centers[i].cpu().numpy().tolist(),
                            centers[j].cpu().numpy().tolist(),
                        ],
                        "new_center": mu.cpu().numpy().tolist(),
                    }
                    merge_pairs.append(pair_info)
                operation_details["merged_pairs"] = merge_pairs

                keep_mask = torch.ones(
                    model.n_splats(), dtype=torch.bool, device=device
                )
                for i, j, mu, L, a in take:
                    keep_mask[i] = False
                    keep_mask[j] = False
                model.prune_(keep_mask)
                # append merged ones
                centers_new = torch.stack([t[2] for t in take], dim=0)
                Ls_new = torch.stack([t[3] for t in take], dim=0)
                amps_new = torch.stack([t[4] for t in take], dim=0)
                model.append_(centers_new, Ls_new, amps_new)
                topology_changed = True

                merge_details = (
                    f"Merged {n_merged_pairs} pairs ({n_removed}→{n_merged_pairs})"
                )
                if n_candidates > n_merged_pairs:
                    merge_details += f" [{n_kl_rejected} KL-rejected]"
                operations_performed.append(merge_details)

        # === SPLIT ===
        if cfg.do_split and model.n_splats() > 0:
            centers, Ls, amps = model.current_params()
            d = centers.shape[1]
            # find candidates with large principal axis AND high residual where they live
            Sigma = Ls @ Ls.transpose(-1, -2)  # (N,d,d)
            eigvals, eigvecs = torch.linalg.eigh(Sigma)  # ascending
            lam_max = eigvals[:, -1].clamp(min=1e-12)
            v1 = eigvecs[:, :, -1]
            rad = lam_max.sqrt()  # std along v1
            large = rad > cfg.split_eig_thr

            # sample residual at each center (clamp within bounds)
            grid_idx = torch.round(centers).long().T  # (d,N)
            for dd in range(d):
                grid_idx[dd] = torch.clamp(grid_idx[dd], 0, res_abs.shape[dd] - 1)
            local_res = res_abs[tuple(grid_idx)]
            high_err = local_res >= torch.quantile(res_abs, cfg.split_error_quantile)

            n_large = large.sum().item()
            n_high_err = high_err.sum().item()
            to_split = torch.nonzero(large & high_err, as_tuple=False).squeeze(1)

            if to_split.numel() > 0:
                n_split = len(to_split)
                mu_children = []
                L_children = []
                a_children = []
                kill_mask = torch.ones(
                    model.n_splats(), dtype=torch.bool, device=device
                )

                # Store split information for visualization
                split_info = []
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
                    dV = 1.0 / (cfg.split_shrink**d)
                    aA = 0.5 * a0 * dV
                    aB = 0.5 * a0 * dV
                    mu_children.extend([muA, muB])
                    L_children.extend([Lc, Lc])
                    a_children.extend([aA, aB])
                    kill_mask[idx] = False

                    # Store for visualization
                    split_info.append(
                        {
                            "parent_center": mu0.cpu().numpy().tolist(),
                            "child_centers": [
                                muA.cpu().numpy().tolist(),
                                muB.cpu().numpy().tolist(),
                            ],
                        }
                    )

                operation_details["split_locations"] = split_info

                model.prune_(kill_mask)
                model.append_(
                    torch.stack(mu_children),
                    torch.stack(L_children),
                    torch.stack(a_children),
                )
                topology_changed = True

                split_details = f"Split {n_split} splats ({n_split}→{2 * n_split})"
                if n_large > n_split or n_high_err > n_split:
                    split_details += f" [large:{n_large}, high-err:{n_high_err}]"
                operations_performed.append(split_details)

        # === SEED ===
        if cfg.do_seed:
            # threshold for residual peaks
            thr = torch.quantile(res_abs, cfg.residual_quantile)
            coords = _local_maxima(
                res_abs, k=cfg.max_add_per_step, min_dist=cfg.nms_radius_vox, thr=thr
            )
            n_candidates = len(coords)

            if n_candidates > 0:
                d = len(model.shape)
                centers_new = []
                Ls_new = []
                amps_new = []
                n_too_weak = 0
                n_too_close = 0

                # Get current splat centers and their effective radii for proximity check
                if model.n_splats() > 0:
                    current_centers, current_Ls, current_amps = model.current_params()
                    # Estimate effective radius as 50% of the largest eigenvalue (standard deviation)
                    current_Sigma = current_Ls @ current_Ls.transpose(-1, -2)
                    current_eigvals = torch.linalg.eigvals(current_Sigma).real
                    current_radii = 0.5 * torch.sqrt(
                        torch.max(current_eigvals, dim=-1)[0]
                    )  # 50% of max std
                else:
                    current_centers = torch.empty((0, d), device=device)
                    current_radii = torch.empty((0,), device=device)

                for c in coords:
                    mu = torch.tensor(c, dtype=torch.float32, device=device)
                    # Init isotropic L; optionally orient via structure tensor (2D)
                    Linit = torch.diag(
                        torch.full((d,), cfg.seed_sigma_vox, device=device)
                    )
                    if d == 2:
                        # approximate principal axis from structure tensor at the seed
                        lam1, v1 = _structure_tensor_eigs_nd(res_abs)
                        if lam1 is not None:
                            v = v1[:, c[0], c[1]]  # (2,)
                            # build rotation matrix to align yx axes to v (optional small anisotropy boost)
                            # keep it simple: slightly elongate along v
                            R = torch.stack(
                                [v, torch.tensor([-v[1], v[0]], device=device)], dim=1
                            )  # 2x2
                            A = torch.diag(
                                torch.tensor(
                                    [
                                        1.15 * cfg.seed_sigma_vox,
                                        0.85 * cfg.seed_sigma_vox,
                                    ],
                                    device=device,
                                )
                            )
                            Linit = R @ A
                    a_est = _estimate_amp_from_residual(
                        model.shape, mu, Linit, residual, truncate=model.truncate
                    )

                    # Check amplitude threshold
                    if a_est.item() <= cfg.amp_abs_min:
                        n_too_weak += 1
                        continue

                    # Check proximity to existing splats (avoid seeding within 50% core radius)
                    too_close = False
                    if len(current_centers) > 0:
                        distances = torch.norm(current_centers - mu.unsqueeze(0), dim=1)
                        min_allowed_dist = (
                            current_radii  # 50% of max std dev (core region)
                        )
                        if torch.any(distances < min_allowed_dist):
                            too_close = True
                            n_too_close += 1

                    if not too_close:
                        centers_new.append(mu)
                        Ls_new.append(Linit)
                        amps_new.append(a_est)

                if centers_new:
                    n_seeded = len(centers_new)

                    # Store seeded locations for visualization
                    seeded_centers = [
                        center.cpu().numpy().tolist() for center in centers_new
                    ]
                    operation_details["seeded_locations"] = seeded_centers

                    model.append_(
                        torch.stack(centers_new),
                        torch.stack(Ls_new),
                        torch.stack(amps_new),
                    )
                    topology_changed = True

                    seed_details = f"Seeded {n_seeded} splats"
                    rejections = []
                    if n_too_weak > 0:
                        rejections.append(f"{n_too_weak} too-weak")
                    if n_too_close > 0:
                        rejections.append(f"{n_too_close} too-close")
                    if rejections:
                        seed_details += f" [{', '.join(rejections)}]"
                    operations_performed.append(seed_details)

        # Report operations performed
        if verbose and operations_performed:
            if len(operations_performed) == 1:
                aprint(f"Dynamic ops: {operations_performed[0]}")
            else:
                with asection("Dynamic Operations"):
                    for op in operations_performed:
                        aprint(f"• {op}")

        # Handle optimizer state when topology changes
        if topology_changed:
            # Check if we're using per-splat optimizer
            if hasattr(optimizer, "splat_states"):
                # Per-splat optimizer - no need to rebuild, just sync state
                _sync_per_splat_optimizer(optimizer, scheduler, verbose)
            else:
                # Standard PyTorch optimizer - need to rebuild
                optimizer, scheduler = _update_optimizer_for_topology_change(
                    model, optimizer, scheduler, lr, verbose
                )

        # Visual debugging with napari
        if napari_debug and (topology_changed or operations_performed):
            _show_napari_debug(
                model,
                V_t,
                residual,
                operations_performed,
                topology_changed,
                operation_details,
            )

    return optimizer, scheduler, topology_changed


def _show_napari_debug(
    model, V_t, residual, operations_performed, topology_changed, operation_details
):
    """
    Display napari debug visualization showing:
    - Target image
    - Current reconstruction
    - Residual
    - Splat locations and properties
    - Visual markers for operations performed (seeded, pruned, merged, split)
    """
    try:
        import napari
        import numpy as np

        # Get current model state
        with torch.no_grad():
            centers, ls, amps = model.current_params()
            pred = model()

        # Convert to numpy
        target = V_t.cpu().numpy()
        prediction = pred.cpu().numpy()
        residual_np = residual.cpu().numpy()
        centers_np = centers.cpu().numpy()
        amps_np = amps.cpu().numpy()

        # Create a fresh viewer for each debug session
        operations_summary = f"Ops: {', '.join(operations_performed) if operations_performed else 'None'}"
        viewer = napari.Viewer(title=f"Dynamic Operations Debug - {operations_summary}")

        # Add image layers
        viewer.add_image(target, name="Target", colormap="viridis", opacity=0.7)
        viewer.add_image(
            prediction,
            name="Current Reconstruction",
            colormap="plasma",
            opacity=0.7,
            visible=False,
        )
        viewer.add_image(
            np.abs(residual_np), name="Residual (abs)", colormap="hot", opacity=0.8
        )

        # Add splat centers as points
        if len(centers_np) > 0:
            # Simple color mapping without matplotlib - use amplitude directly
            normalized_amps = amps_np / (amps_np.max() + 1e-8)
            # Create RGB colors: blue for low amplitude, red for high amplitude
            colors = np.zeros((len(normalized_amps), 4))  # RGBA
            colors[:, 0] = normalized_amps  # Red channel
            colors[:, 2] = 1.0 - normalized_amps  # Blue channel
            colors[:, 3] = 1.0  # Alpha

            viewer.add_points(
                centers_np, name="Splat Centers", face_color=colors, size=8
            )

        # Add operation-specific visualizations
        if operation_details["pruned_locations"]:
            pruned_np = np.array(operation_details["pruned_locations"])
            viewer.add_points(
                pruned_np, name="🗑️ Pruned Splats", face_color="red", size=15
            )

        if operation_details["seeded_locations"]:
            seeded_np = np.array(operation_details["seeded_locations"])
            viewer.add_points(
                seeded_np, name="🌱 Seeded Splats", face_color="lime", size=15
            )

        if operation_details["merged_pairs"]:
            # Show merge old locations and new locations
            merge_old_points = []
            merge_new_points = []
            for pair in operation_details["merged_pairs"]:
                old1, old2 = pair["old_centers"]
                new = pair["new_center"]
                merge_old_points.extend([old1, old2])
                merge_new_points.append(new)

            if merge_old_points:
                viewer.add_points(
                    np.array(merge_old_points),
                    name="🔄 Merged (old)",
                    face_color="orange",
                    size=12,
                )

            if merge_new_points:
                viewer.add_points(
                    np.array(merge_new_points),
                    name="🔄 Merged (new)",
                    face_color="darkorange",
                    size=12,
                )

        if operation_details["split_locations"]:
            # Show split parent and children locations
            split_parent_points = []
            split_child_points = []
            for split in operation_details["split_locations"]:
                parent = split["parent_center"]
                child1, child2 = split["child_centers"]
                split_parent_points.append(parent)
                split_child_points.extend([child1, child2])

            if split_parent_points:
                viewer.add_points(
                    np.array(split_parent_points),
                    name="🍴 Split Parents",
                    face_color="purple",
                    border_color=12,
                )

            if split_child_points:
                viewer.add_points(
                    np.array(split_child_points),
                    name="🍴 Split Children",
                    face_color="violet",
                    size=10,
                )

        # Add text overlay with operations info
        info_text = f"Splats: {len(amps_np)}\n"
        info_text += f"Topology changed: {topology_changed}\n"
        if operations_performed:
            info_text += "Operations:\n"
            for op in operations_performed:
                info_text += f"  • {op}\n"
        else:
            info_text += "No operations performed\n"

        # Add MSE info
        mse = np.mean((target - prediction) ** 2)
        info_text += f"MSE: {mse:.6f}"

        viewer.text_overlay.text = info_text
        viewer.text_overlay.visible = True

        # Show message and run napari - will block until window is closed
        from arbol import aprint

        aprint("🔍 Napari debug window opened - close window to continue...")
        napari.run()  # Blocks until viewer is closed

    except ImportError:
        from arbol import aprint

        aprint("⚠ napari not available for debug visualization")
    except Exception as e:
        from arbol import aprint

        aprint(f"⚠ napari debug error: {e}")


def _update_optimizer_for_topology_change(
    model, old_optimizer, old_scheduler, original_lr, verbose
):
    """
    Update optimizer and scheduler when model topology changes.

    Key insight: After topology changes (adding splats), the model needs fresh momentum
    and a reset learning rate schedule to properly integrate new parameters.

    The fundamental problem: PyTorch optimizers store momentum per parameter tensor.
    When we add/remove splats, parameter tensors change size, invalidating optimizer state.
    """
    from arbol import aprint

    # Use a more conservative learning rate - halfway between current and original
    current_lr = (
        old_optimizer.param_groups[0]["lr"]
        if hasattr(old_optimizer, "param_groups")
        else original_lr
    )

    # After adding new splats, we need a reasonable LR to integrate them
    # Too low = new splats never optimize, too high = disruption
    if current_lr < original_lr * 0.1:
        # If LR has been heavily reduced, boost it back up for new splats
        reset_lr = original_lr * 0.5  # Conservative reset
        if verbose:
            aprint(
                f"  → LR boosted for new splat integration: {current_lr:.6f} → {reset_lr:.6f}"
            )
    else:
        reset_lr = current_lr

    # Create new optimizer (loses ALL momentum)
    new_optimizer = torch.optim.Adam(model.parameters(), lr=reset_lr)

    # Create fresh scheduler - after topology change, we want to give the model
    # time to integrate new splats before reducing LR again
    new_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        new_optimizer, mode="min", factor=0.5, patience=10
    )

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        aprint(f"  → Fresh optimizer: lr={reset_lr:.6f}, {n_params} parameters")
        aprint("  → Fresh scheduler (patience reset for new splat integration)")
        aprint(
            f"  ⚠ ALL {model.n_splats()} splats lose momentum (causes global disruption)"
        )

    return new_optimizer, new_scheduler


def _sync_per_splat_optimizer(optimizer, scheduler, verbose):
    """
    Sync per-splat optimizer state after topology changes.

    The beauty of per-splat optimizers: no rebuilding needed!
    The optimizer automatically handles topology changes via its
    add_splats() and remove_splats() methods during model operations.
    """
    from arbol import aprint

    if verbose:
        n_splats = optimizer.model.n_splats()
        n_states = len(optimizer.splat_states)

        if n_states != n_splats:
            # This shouldn't happen if model operations are calling optimizer sync correctly
            aprint(f"  ⚠ Splat count mismatch: model={n_splats}, optimizer={n_states}")
            # Force re-sync
            optimizer._initialize_all_splats()
            n_states = len(optimizer.splat_states)

        # Get learning rate statistics
        lrs = optimizer.get_effective_learning_rates()
        lr_mean = float(torch.mean(lrs))
        lr_min = float(torch.min(lrs))
        lr_max = float(torch.max(lrs))

        aprint(
            f"  ✓ Per-splat optimizer: {n_splats} splats, LR: {lr_mean:.6f} (min={lr_min:.6f}, max={lr_max:.6f})"
        )
        aprint("  ✓ Momentum preserved for unchanged splats - no global disruption!")
