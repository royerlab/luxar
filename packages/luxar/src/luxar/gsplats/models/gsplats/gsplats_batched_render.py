from typing import Dict, List, Sequence, Tuple

import torch


def _group_by_box(
    lo: torch.Tensor, hi: torch.Tensor
) -> Dict[Tuple[int, ...], torch.Tensor]:
    """
    Group splats by their AABB shape so we can reuse a single base grid per group.
    Returns: dict { box_shape_tuple : idx_tensor }  (CPU tuple keys, GPU indices)
    """
    sizes = (hi - lo).to(torch.long)  # (N, d)
    # Move tiny metadata to CPU for hashing
    sizes_cpu = sizes.cpu().tolist()
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for k, s in enumerate(sizes_cpu):
        key = tuple(int(x) for x in s)
        groups.setdefault(key, []).append(k)
    return {
        k: torch.tensor(v, device=lo.device, dtype=torch.long)
        for k, v in groups.items()
    }


def _linear_strides(shape: Sequence[int], device) -> torch.Tensor:
    """Row-major linear strides for an nD tensor with given shape."""
    d = len(shape)
    s = [1]
    for i in range(d - 1, 0, -1):
        s.insert(0, s[0] * shape[i])
    return torch.tensor(s, device=device, dtype=torch.long)  # (d,)


def render_gaussians_batched(
    shape: Sequence[int],
    centers: torch.Tensor,  # (N, d) voxel coords
    Ls: torch.Tensor,  # (N, d, d) lower-tri
    amps: torch.Tensor,  # (N,)
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,  # for amplitude-aware culling (see §2)
) -> torch.Tensor:
    """
    Fast vectorized renderer.
    Strategy: compute each splat's box [lo,hi), group by box size, build one base grid per group,
    batched triangular solve (L y = Δ^T), then one scatter_add_ per group.
    Everything is differentiable.
    """
    device = centers.device
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    out_flat = out.view(-1)
    strides = _linear_strides(shape, device)  # (d,)

    # --- AABB per splat ---
    # Σ_ii = row-wise sum(L^2); r_i = ceil(truncate * sqrt(Σ_ii))
    sigma_diag = torch.sum(Ls * Ls, dim=2)  # (N, d)
    radii = torch.clamp(
        (truncate * torch.sqrt(torch.clamp(sigma_diag, 1e-8))).ceil().to(torch.long),
        min=1,
    )
    lo = torch.clamp((centers - radii).floor().to(torch.long), min=0)  # (N, d)
    hi = torch.minimum(
        (centers + radii).ceil().to(torch.long) + 1,
        torch.tensor(shape, device=device).to(torch.long),
    )  # (N, d)

    # (Optional) **amplitude-aware shrinking** (see §2) – avoids giant boxes for tiny a
    if intensity_floor is not None and intensity_floor > 0:
        # t_max per splat solves: a * exp(-0.5 * t^2) >= intensity_floor  ->  t <= sqrt(2 log(a/eps))
        eps = torch.tensor(intensity_floor, device=device, dtype=torch.float32)
        a = torch.clamp(amps, min=1e-12)
        tmax = torch.sqrt(
            torch.clamp(2.0 * torch.log(torch.clamp(a / eps, min=1.0)), min=0.0)
        )  # (N,)
        # shrink radii = min(current, ceil(tmax * sqrt(Σ_ii)))
        shrink = torch.clamp(
            (tmax[:, None] * torch.sqrt(torch.clamp(sigma_diag, 1e-8)))
            .ceil()
            .to(torch.long),
            min=1,
        )
        radii = torch.minimum(radii, shrink)
        lo = torch.clamp((centers - radii).floor().to(torch.long), min=0)
        hi = torch.minimum(
            (centers + radii).ceil().to(torch.long) + 1,
            torch.tensor(shape, device=device).to(torch.long),
        )

    # Drop empty boxes (rare but safe)
    valid = (hi > lo).all(1)
    if not torch.all(valid):
        centers = centers[valid]
        Ls = Ls[valid]
        amps = amps[valid]
        lo = lo[valid]
        hi = hi[valid]
        sigma_diag = sigma_diag[valid]

    if centers.numel() == 0:
        return out

    # --- Group by box size for reuse of base grid ---
    groups = _group_by_box(lo, hi)  # { (h1,..,hd) : idx }
    for box_shape, idx in groups.items():
        # Splat subset
        mu = centers[idx]  # (K, d)
        L = Ls[idx]  # (K, d, d)
        a = amps[idx]  # (K,)

        # Build base grid (one per group, on device)
        # coords_i = [0, 1, ..., h_i-1]  -> broadcast to P points
        ranges = [
            torch.arange(s, device=device, dtype=torch.float32) for s in box_shape
        ]
        grids = torch.meshgrid(*ranges, indexing="ij")
        P = 1
        for g in grids:
            P *= g.numel()
        # per-axis base coords flattened (P,)
        base = torch.stack([g.reshape(-1) for g in grids], dim=0)  # (d, P)

        # Per-splat Δ = base + lo - μ   (broadcast to (K, d, P))
        lo_f = lo[idx].to(torch.float32)  # (K, d)
        delta = base[None, :, :] + lo_f[:, :, None] - mu[:, :, None]  # (K, d, P)

        # Solve L y = Δ  (batched lower-tri solve with P RHS per splat)
        # torch.linalg.solve_triangular supports batch dims: (K, d, d) x (K, d, P) -> (K, d, P)
        try:
            y = torch.linalg.solve_triangular(L, delta, upper=False)
        except Exception:
            # older PyTorch
            y, _ = torch.triangular_solve(delta, L, upper=False)

        # Exponent and values: exp(-0.5 * ||y||^2) * a
        expo = torch.sum(y * y, dim=1)  # (K, P)
        vals = torch.exp(-0.5 * expo) * a[:, None]  # (K, P)

        # Compute flattened indices once for the group:
        # lin_offsets = base dot strides  (P,)
        strides_f = strides.to(torch.float32)
        lin_offsets = (base.T @ strides_f).to(torch.long)  # (P,)
        # base index per splat = lo dot strides
        base_idx = (lo[idx].to(torch.long) * strides).sum(dim=1)  # (K,)

        # Absolute flat indices (K, P) -> (K*P,)
        idx_flat = (base_idx[:, None] + lin_offsets[None, :]).reshape(-1)
        vals_flat = vals.reshape(-1)

        # One scatter-add per group
        out_flat.index_add_(0, idx_flat, vals_flat)

    return out
