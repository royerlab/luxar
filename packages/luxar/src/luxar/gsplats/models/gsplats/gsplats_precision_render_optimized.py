"""
Optimized Precision Matrix Gaussian Splatting Renderer

This module implements the TRUE precision-only pipeline using triangular operations:
- Rendering: ||U(x-μ)||² via triangular matrix multiplication (no full precision matrix)
- AABB: tight per-axis bounds via triangular solves (no matrix inversion)
- Numerically stable and computationally optimal

Key insight: Work entirely with upper triangular Cholesky U where Λ = U^T @ U
"""
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


def _compute_tight_aabb_radii(U: torch.Tensor, truncate: float) -> torch.Tensor:
    """
    Compute tight per-axis radii using triangular solves.
    
    The ellipsoid {x: (x-μ)^T Λ (x-μ) ≤ t²} has half-width along axis i:
    r_i = t * ||U^{-T} e_i||₂
    
    We compute this via triangular solve: U^T y = e_i, then r_i = t * ||y||
    
    Parameters
    ----------
    U : torch.Tensor, shape (N, d, d)
        Upper triangular Cholesky factors where Λ = U^T @ U
    truncate : float
        Truncation factor t
        
    Returns
    -------
    torch.Tensor, shape (N, d)
        Tight radii per axis for each Gaussian
    """
    N, d, _ = U.shape
    device = U.device
    
    # Create identity matrix for e_i vectors
    eye = torch.eye(d, device=device, dtype=U.dtype)  # (d, d)
    
    # Solve U^T @ Y = I for each Gaussian
    # This gives us Y = U^{-T} where Y[:, i] = U^{-T} e_i
    U_T = U.transpose(-1, -2)  # (N, d, d) - transpose to lower triangular
    Y = torch.linalg.solve_triangular(U_T, eye.expand(N, -1, -1), upper=False)  # (N, d, d)
    
    # Compute ||Y[:, i]|| for each axis i (column norms of each matrix)
    # Y[n, :, i] is the i-th column of U_n^{-T}, we want its norm
    radii_per_axis = torch.norm(Y, dim=1)  # (N, d) - norm along middle dimension
    
    # Scale by truncation factor
    return truncate * radii_per_axis


def render_gaussians_precision_optimized(
    shape: Sequence[int],
    centers: torch.Tensor,  # (N, d) voxel coords
    Us: torch.Tensor,  # (N, d, d) upper-triangular Cholesky factors  
    amps: torch.Tensor,  # (N,)
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,
) -> torch.Tensor:
    """
    Optimized precision-only Gaussian renderer using triangular operations.
    
    TRUE precision-only pipeline:
    1. Rendering: g(x) = a * exp(-0.5 * ||U(x-μ)||²) via triangular matmul
    2. AABB: tight per-axis bounds r_i = t * ||U^{-T} e_i|| via triangular solves
    3. No matrix inversions or full precision matrices needed
    
    This is numerically stable and computationally optimal.
    
    Parameters
    ----------
    shape : Sequence[int]
        Output tensor shape
    centers : torch.Tensor, shape (N, d)  
        Gaussian center positions in voxel coordinates
    Us : torch.Tensor, shape (N, d, d)
        Upper-triangular Cholesky factors where precision Λ = U^T @ U
    amps : torch.Tensor, shape (N,)
        Gaussian amplitudes
    truncate : float, default=3.0
        Truncation radius in standard deviations
    intensity_floor : float, default=1e-5
        Minimum intensity for amplitude-aware culling
        
    Returns
    -------
    torch.Tensor
        Rendered output tensor with same shape as input shape
    """
    device = centers.device
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    out_flat = out.view(-1)
    strides = _linear_strides(shape, device)  # (d,)

    # --- Tight AABB per splat using triangular solves ---
    radii = _compute_tight_aabb_radii(Us, truncate)  # (N, d)
    
    # Clamp to at least 1 voxel
    radii = torch.clamp(radii.ceil().to(torch.long), min=1)
    
    lo = torch.clamp((centers - radii).floor().to(torch.long), min=0)  # (N, d)
    hi = torch.minimum(
        (centers + radii).ceil().to(torch.long) + 1,
        torch.tensor(shape, device=device).to(torch.long),
    )  # (N, d)

    # (Optional) amplitude-aware shrinking using triangular solves
    if intensity_floor is not None and intensity_floor > 0:
        eps = torch.tensor(intensity_floor, device=device, dtype=torch.float32)
        a = torch.clamp(amps, min=1e-12)
        tmax = torch.sqrt(
            torch.clamp(2.0 * torch.log(torch.clamp(a / eps, min=1.0)), min=0.0)
        )  # (N,)
        
        # Compute shrunk radii using the same triangular solve approach
        shrink_radii = _compute_tight_aabb_radii(Us, 1.0) * tmax[:, None]  # (N, d)
        shrink_radii = torch.clamp(shrink_radii.ceil().to(torch.long), min=1)
        
        radii = torch.minimum(radii, shrink_radii)
        lo = torch.clamp((centers - radii).floor().to(torch.long), min=0)
        hi = torch.minimum(
            (centers + radii).ceil().to(torch.long) + 1,
            torch.tensor(shape, device=device).to(torch.long),
        )

    # Drop empty boxes
    valid = (hi > lo).all(1)
    if not torch.all(valid):
        centers = centers[valid]
        Us = Us[valid]
        amps = amps[valid]
        lo = lo[valid]
        hi = hi[valid]

    if centers.numel() == 0:
        return out

    # --- Group by box size for reuse of base grid ---
    groups = _group_by_box(lo, hi)  # { (h1,..,hd) : idx }
    for box_shape, idx in groups.items():
        # Splat subset
        mu = centers[idx]  # (K, d)
        U = Us[idx]  # (K, d, d) upper triangular
        a = amps[idx]  # (K,)

        # Build base grid (one per group, on device)
        ranges = [
            torch.arange(s, device=device, dtype=torch.float32) for s in box_shape
        ]
        grids = torch.meshgrid(*ranges, indexing="ij")
        P = 1
        for g in grids:
            P *= g.numel()
        base = torch.stack([g.reshape(-1) for g in grids], dim=0)  # (d, P)

        # Per-splat Δ = base + lo - μ   (broadcast to (K, d, P))
        lo_f = lo[idx].to(torch.float32)  # (K, d)
        delta = base[None, :, :] + lo_f[:, :, None] - mu[:, :, None]  # (K, d, P)

        # *** OPTIMIZED PRECISION-ONLY COMPUTATION ***
        # Core insight: g(x) = a * exp(-0.5 * ||U(x-μ)||²)
        # Compute y = U @ delta via triangular matmul (much faster than full precision matrix)
        
        # Reshape for batched triangular matmul: delta (K, d, P) -> (K*P, d, 1)
        K, d, P = delta.shape
        delta_reshaped = delta.permute(0, 2, 1).reshape(K * P, d, 1)  # (K*P, d, 1)
        
        # Expand U for batched operation: (K, d, d) -> (K*P, d, d)
        U_expanded = U.unsqueeze(1).expand(K, P, d, d).reshape(K * P, d, d)
        
        # Batched triangular matrix multiply: y = U @ delta
        y = torch.bmm(U_expanded, delta_reshaped).squeeze(-1)  # (K*P, d)
        
        # Compute ||y||² for each point
        y_norm_sq = torch.sum(y * y, dim=1)  # (K*P,)
        
        # Reshape back and compute Gaussian values
        expo = y_norm_sq.reshape(K, P)  # (K, P)
        vals = torch.exp(-0.5 * expo) * a[:, None]  # (K, P)

        # Compute flattened indices once for the group
        strides_f = strides.to(torch.float32)
        lin_offsets = (base.T @ strides_f).to(torch.long)  # (P,)
        base_idx = (lo[idx].to(torch.long) * strides).sum(dim=1)  # (K,)

        # Absolute flat indices (K, P) -> (K*P,)
        idx_flat = (base_idx[:, None] + lin_offsets[None, :]).reshape(-1)
        vals_flat = vals.reshape(-1)

        # One scatter-add per group
        out_flat.index_add_(0, idx_flat, vals_flat)

    return out


def render_gaussians_precision_sequential_optimized(
    shape: Sequence[int],
    centers: torch.Tensor,  # (N, d) voxel coords
    Us: torch.Tensor,  # (N, d, d) upper triangular Cholesky factors
    amps: torch.Tensor,  # (N,)
    truncate: float = 3.0,
) -> torch.Tensor:
    """
    Sequential optimized precision renderer (for debugging/verification).
    
    Same precision-only approach as batched version but processes one Gaussian at a time.
    """
    device = centers.device
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    N, d = centers.shape
    shape_t = torch.tensor(shape, dtype=torch.long, device=device)
    
    # Pre-compute radii for all splats using triangular solves
    radii_all = _compute_tight_aabb_radii(Us, truncate)  # (N, d)
    
    for k in range(N):
        mu = centers[k]  # (d,)
        U = Us[k]  # (d, d) upper triangular
        a = amps[k]  # scalar
        radii = radii_all[k]  # (d,)
        
        # AABB radius per axis (already computed via triangular solve)
        r = torch.clamp(radii.ceil().long(), min=1)
        
        # Local bounding box with bounds checking
        lo = torch.clamp((mu - r).floor().long(), min=0)
        hi = torch.minimum((mu + r).ceil().long() + 1, shape_t)
        
        # Skip degenerate boxes
        if torch.any(hi <= lo):
            continue
            
        # Build local grid
        ranges = [
            torch.arange(lo[i], hi[i], device=device, dtype=torch.float32)
            for i in range(d)
        ]
        grids = torch.meshgrid(*ranges, indexing="ij")
        P = 1
        for g in grids:
            P *= g.numel()
        if P == 0:
            continue
        pts = torch.stack([g.reshape(-1) for g in grids], dim=1)  # (P, d)
        delta = pts - mu.unsqueeze(0)  # (P, d)
        
        # *** OPTIMIZED PRECISION-ONLY COMPUTATION ***
        # Triangular matmul: y = delta @ U^T (since U is upper triangular)
        y = delta @ U.T  # (P, d)
        
        # Compute ||y||² directly
        y_norm_sq = torch.sum(y * y, dim=1)  # (P,)
        
        G = torch.exp(-0.5 * y_norm_sq) * a  # (P,)
        
        # Scatter-add into canvas
        slicer = tuple(
            slice(int(lo[i].item()), int(hi[i].item())) for i in range(d)
        )
        out[slicer] = out[slicer] + G.reshape(
            [int(hi[i] - lo[i]) for i in range(d)]
        )
    
    return out