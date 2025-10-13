# gsplat_model.py

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from luxar.gsplats.models.utils.inverse_softplus import stable_inverse_softplus


def _calculate_optimal_chunk_size(
    K: int, d: int, device: torch.device, dtype: torch.dtype
) -> int:
    """
    Calculate optimal P chunk size based on available memory and tensor dimensions.

    This function determines the largest chunk size that can safely fit in memory,
    considering the actual memory footprint of the (K, d, P_chunk) tensors used
    in the renderer (delta, y, and intermediate results).

    Parameters
    ----------
    K : int
        Number of splats in the current group
    d : int
        Number of dimensions
    device : torch.device
        Target device for computation
    dtype : torch.dtype
        Data type for tensors

    Returns
    -------
    int
        Optimal chunk size for P dimension, bounded to reasonable range
    """
    # Get available memory with safety margin
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            free_mem, _ = torch.cuda.mem_get_info(device)
            available_mem = int(free_mem * 0.6)  # 60% safety margin for CUDA
        except Exception:
            # Fallback if memory info unavailable
            available_mem = 2 * (1024**3)  # 2GB conservative estimate
    else:
        # Conservative estimates for CPU/MPS
        if device.type == "mps":
            available_mem = 4 * (1024**3)  # 4GB for Apple Silicon unified memory
        else:
            available_mem = 8 * (1024**3)  # 8GB for CPU

    # Calculate memory footprint per P element
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    # Account for: delta (K,d,P), y (K,d,P), expo (K,P), vals (K,P), plus overhead
    memory_per_p = K * (2 * d + 2) * bytes_per_element * 1.5  # 1.5x overhead factor

    if memory_per_p <= 0:
        return 131072  # Fallback default

    # Calculate maximum P_chunk that fits in available memory
    max_p_chunk = max(1, int(available_mem / memory_per_p))

    # Clamp to reasonable range for performance
    min_chunk = 1024  # Minimum for kernel efficiency
    max_chunk = 1024**2  # Maximum to avoid very large kernel launches

    # Use default if calculation seems unreasonable
    if max_p_chunk < min_chunk:
        return 131072  # Fallback to current default

    return min(max_chunk, max_p_chunk)


# ===== [ADD] Fast-path helpers for 2D/3D =====================================

# Simple process-wide cache for base grids and linear offsets.
# Keyed by (device, dtype, strides_tuple, box_shape_tuple).
_GRID_CACHE: Dict[
    Tuple[str, str, Tuple[int, ...], Tuple[int, ...]], Tuple[torch.Tensor, torch.Tensor]
] = {}


def _cached_base_and_offsets(
    box_shape: Sequence[int],
    strides: torch.Tensor,  # (d,), long
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      base: (d, P) float tensor with coordinates [0..h_i-1] mesh, flattened.
      lin_offsets: (P,) long tensor of row-major flat offsets for this box_shape.
    """
    key = (
        device.type,
        str(dtype),
        tuple(int(s) for s in strides.tolist()),
        tuple(int(s) for s in box_shape),
    )
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]

    ranges = [torch.arange(int(s), device=device, dtype=dtype) for s in box_shape]
    grids = torch.meshgrid(*ranges, indexing="ij")  # list of d arrays
    base = torch.stack([g.reshape(-1) for g in grids], dim=0)  # (d, P)

    # Compute row-major offsets once for this box shape.
    base_l = torch.stack([g.reshape(-1).to(torch.long) for g in grids], dim=0)  # (d, P)
    lin_offsets = (base_l.T * strides).sum(dim=1)  # (P,)

    _GRID_CACHE[key] = (base, lin_offsets)
    return _GRID_CACHE[key]


@torch.jit.ignore  # jit-able but optional; ignore keeps it simple if torch.compile() is used outside
def _group_by_box_gpu(
    lo: torch.Tensor, hi: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-friendly grouping by AABB size.
    Returns (uniq_sizes, inv), where uniq_sizes is (G, d) and inv is (N,).
    """
    sizes = (hi - lo).to(torch.int32)  # (N, d)

    # MPS doesn't support torch.unique with dim argument, fallback to CPU
    if sizes.device.type == "mps":
        sizes_cpu = sizes.cpu()
        uniq, inv = torch.unique(sizes_cpu, dim=0, return_inverse=True)
        uniq, inv = uniq.to(sizes.device), inv.to(sizes.device)
    else:
        uniq, inv = torch.unique(sizes, dim=0, return_inverse=True)

    return uniq, inv


# ---- Explicit forward-substitution for 2D/3D (no linalg kernels) ------------


def _fwd_norm2_2d(L: torch.Tensor, d0: torch.Tensor, d1: torch.Tensor) -> torch.Tensor:
    """
    Solve L y = [d0, d1]^T for each splat (batched) and return ||y||^2.
    L: (K, 2, 2), d0/d1: (K, P)
    Returns: (K, P)
    """
    l11 = L[:, 0, 0].unsqueeze(1)  # (K,1)
    l21 = L[:, 1, 0].unsqueeze(1)
    l22 = L[:, 1, 1].unsqueeze(1)

    y0 = d0 / torch.clamp(l11, min=1e-12)
    y1 = (d1 - l21 * y0) / torch.clamp(l22, min=1e-12)
    return y0.mul(y0).add_(y1.mul(y1))


def _fwd_norm2_3d(
    L: torch.Tensor, d0: torch.Tensor, d1: torch.Tensor, d2: torch.Tensor
) -> torch.Tensor:
    """
    Solve L y = [d0, d1, d2]^T for each splat (batched) and return ||y||^2.
    L: (K, 3, 3), d0/d1/d2: (K, P)
    Returns: (K, P)
    """
    l11 = L[:, 0, 0].unsqueeze(1)  # (K,1)
    l21 = L[:, 1, 0].unsqueeze(1)
    l22 = L[:, 1, 1].unsqueeze(1)
    l31 = L[:, 2, 0].unsqueeze(1)
    l32 = L[:, 2, 1].unsqueeze(1)
    l33 = L[:, 2, 2].unsqueeze(1)

    y0 = d0 / torch.clamp(l11, min=1e-12)
    y1 = (d1 - l21 * y0) / torch.clamp(l22, min=1e-12)
    y2 = (d2 - l31 * y0 - l32 * y1) / torch.clamp(l33, min=1e-12)
    return y0.mul(y0).add_(y1.mul(y1)).add_(y2.mul(y2))


# ---- Specialized renderers ---------------------------------------------------


def _render_gaussians_2d(
    shape: Sequence[int],
    centers: torch.Tensor,  # (N,2)
    Ls: torch.Tensor,  # (N,2,2)
    amps: torch.Tensor,  # (N,)
    sharpness: torch.Tensor,  # (N,)
    truncate: float,
    intensity_floor: float,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    device = centers.device
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    out_flat = out.view(-1)
    strides = _linear_strides(shape, device)  # (2,)

    # AABB per splat (sharpness-adjusted for generalized Gaussian exp(-0.5 * r^s))
    sigma_diag = torch.sum(Ls * Ls, dim=2)  # (N,2)
    effective_truncate = truncate ** (
        2.0 / sharpness
    )  # (N,) - sharpness-adjusted radius
    radii = torch.clamp(
        (effective_truncate[:, None] * torch.sqrt(torch.clamp(sigma_diag, 1e-8)))
        .ceil()
        .to(torch.long),
        min=1,
    )
    lo = torch.clamp((centers - radii).floor().to(torch.long), min=0)
    hi = torch.minimum(
        (centers + radii).ceil().to(torch.long) + 1,
        torch.tensor(shape, device=device, dtype=torch.long),
    )

    if intensity_floor is not None and intensity_floor > 0:
        eps = torch.tensor(intensity_floor, device=device, dtype=torch.float32)
        a = torch.clamp(amps, min=1e-12)
        log_ratio = torch.clamp(2.0 * torch.log(torch.clamp(a / eps, min=1.0)), min=0.0)
        tmax = torch.pow(
            log_ratio, 1.0 / sharpness
        )  # (N,) - sharpness-adjusted threshold
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
            torch.tensor(shape, device=device, dtype=torch.long),
        )

    valid = (hi > lo).all(1)
    if not torch.all(valid):
        centers, Ls, amps, sharpness = (
            centers[valid],
            Ls[valid],
            amps[valid],
            sharpness[valid],
        )
        lo, hi = lo[valid], hi[valid]
        if centers.numel() == 0:
            return out

    # Group on GPU
    uniq, inv = _group_by_box_gpu(lo, hi)

    for g in range(uniq.shape[0]):
        box_shape = uniq[g].tolist()  # [h0, h1]
        idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)

        mu = centers[idx]  # (K,2)
        L = Ls[idx]  # (K,2,2)
        a = amps[idx]  # (K,)
        s = sharpness[idx]  # (K,)
        lo_sel = lo[idx]  # (K,2)

        base, lin_offsets = _cached_base_and_offsets(
            box_shape, strides, device, dtype=torch.float32
        )  # (2,P), (P,)
        base_idx = (lo_sel.to(torch.long) * strides).sum(dim=1)  # (K,)

        P = base.shape[1]
        # Calculate optimal chunk size for memory management (2D)
        P_chunk = chunk_size or _calculate_optimal_chunk_size(
            K=len(idx), d=2, device=device, dtype=torch.float32
        )
        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)
            # Δ = base + lo - μ
            d0 = base[0, p0:p1][None, :] + lo_sel[:, 0:1] - mu[:, 0:1]  # (K,Pc)
            d1 = base[1, p0:p1][None, :] + lo_sel[:, 1:2] - mu[:, 1:2]  # (K,Pc)

            # ||y||^2 via explicit forward-substitution
            expo = _fwd_norm2_2d(L, d0, d1)  # (K,Pc)

            # Apply sharpness: exp(-0.5 * ||y||^s) where s = sharpness
            # ||y||^s = (||y||^2)^(s/2) = expo^(s/2)
            # Clamp expo to avoid log(0) in gradients of pow(expo, s/2)
            expo_safe = torch.clamp(expo, min=1e-10)
            vals = torch.exp(-0.5 * torch.pow(expo_safe, s[:, None] / 2.0)) * a[:, None]

            idx_flat = (base_idx[:, None] + lin_offsets[p0:p1][None, :]).reshape(-1)
            out_flat.index_add_(0, idx_flat, vals.reshape(-1))

    return out


def _render_gaussians_3d(
    shape: Sequence[int],
    centers: torch.Tensor,  # (N,3)
    Ls: torch.Tensor,  # (N,3,3)
    amps: torch.Tensor,  # (N,)
    sharpness: torch.Tensor,  # (N,)
    truncate: float,
    intensity_floor: float,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    device = centers.device
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    out_flat = out.view(-1)
    strides = _linear_strides(shape, device)  # (3,)

    # AABB per splat (sharpness-adjusted for generalized Gaussian exp(-0.5 * r^s))
    sigma_diag = torch.sum(Ls * Ls, dim=2)  # (N,3)
    effective_truncate = truncate ** (
        2.0 / sharpness
    )  # (N,) - sharpness-adjusted radius
    radii = torch.clamp(
        (effective_truncate[:, None] * torch.sqrt(torch.clamp(sigma_diag, 1e-8)))
        .ceil()
        .to(torch.long),
        min=1,
    )
    lo = torch.clamp((centers - radii).floor().to(torch.long), min=0)
    hi = torch.minimum(
        (centers + radii).ceil().to(torch.long) + 1,
        torch.tensor(shape, device=device, dtype=torch.long),
    )

    if intensity_floor is not None and intensity_floor > 0:
        eps = torch.tensor(intensity_floor, device=device, dtype=torch.float32)
        a = torch.clamp(amps, min=1e-12)
        log_ratio = torch.clamp(2.0 * torch.log(torch.clamp(a / eps, min=1.0)), min=0.0)
        tmax = torch.pow(
            log_ratio, 1.0 / sharpness
        )  # (N,) - sharpness-adjusted threshold
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
            torch.tensor(shape, device=device, dtype=torch.long),
        )

    valid = (hi > lo).all(1)
    if not torch.all(valid):
        centers, Ls, amps, sharpness = (
            centers[valid],
            Ls[valid],
            amps[valid],
            sharpness[valid],
        )
        lo, hi = lo[valid], hi[valid]
        if centers.numel() == 0:
            return out

    # Group on GPU
    uniq, inv = _group_by_box_gpu(lo, hi)

    for g in range(uniq.shape[0]):
        box_shape = uniq[g].tolist()  # [h0, h1, h2]
        idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)

        mu = centers[idx]  # (K,3)
        L = Ls[idx]  # (K,3,3)
        a = amps[idx]  # (K,)
        s = sharpness[idx]  # (K,)
        lo_sel = lo[idx]  # (K,3)

        base, lin_offsets = _cached_base_and_offsets(
            box_shape, strides, device, dtype=torch.float32
        )  # (3,P),(P,)
        base_idx = (lo_sel.to(torch.long) * strides).sum(dim=1)  # (K,)

        P = base.shape[1]
        # Calculate optimal chunk size for memory management (3D)
        P_chunk = chunk_size or _calculate_optimal_chunk_size(
            K=len(idx), d=3, device=device, dtype=torch.float32
        )
        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)
            # Δ = base + lo - μ
            d0 = base[0, p0:p1][None, :] + lo_sel[:, 0:1] - mu[:, 0:1]  # (K,Pc)
            d1 = base[1, p0:p1][None, :] + lo_sel[:, 1:2] - mu[:, 1:2]
            d2 = base[2, p0:p1][None, :] + lo_sel[:, 2:3] - mu[:, 2:3]

            expo = _fwd_norm2_3d(L, d0, d1, d2)  # (K,Pc)

            # Apply sharpness: exp(-0.5 * ||y||^s) where s = sharpness
            # ||y||^s = (||y||^2)^(s/2) = expo^(s/2)
            # Clamp expo to avoid log(0) in gradients of pow(expo, s/2)
            expo_safe = torch.clamp(expo, min=1e-10)
            vals = torch.exp(-0.5 * torch.pow(expo_safe, s[:, None] / 2.0)) * a[:, None]

            idx_flat = (base_idx[:, None] + lin_offsets[p0:p1][None, :]).reshape(-1)
            out_flat.index_add_(0, idx_flat, vals.reshape(-1))

    return out


# ===== [END ADD] =============================================================


class GaussianSplatModel(nn.Module):
    """
    PyTorch model for n-dimensional oriented Gaussian splats with full covariance matrices.

    This model represents a collection of oriented Gaussian functions (splats) that can be
    optimized to reconstruct images or volumes. Each splat is parameterized by:

    1. **Center position**: Constrained to image domain via sigmoid parameterization
    2. **Covariance matrix**: Represented via Cholesky decomposition L where Σ = L @ L^T
    3. **Amplitude**: Non-negative scalar via softplus activation

    Mathematical formulation:
        Each splat k contributes: a_k * exp(-0.5 * (x-μ_k)^T @ Σ_k^{-1} @ (x-μ_k))

    Computational optimizations:
        - Avoids explicit matrix inversion by solving triangular system L @ y = (x-μ)
        - Uses AABB truncation for efficient rendering
        - Batched operations for multiple splats

    Parameters
    ----------
    shape : Sequence[int]
        Dimensions of the target image/volume to reconstruct.
    centers0 : np.ndarray, shape (N, d)
        Initial center positions in voxel coordinates.
    L0 : np.ndarray, shape (N, d, d)
        Initial lower-triangular Cholesky factors.
    amps0 : np.ndarray, shape (N,)
        Initial amplitude values.
    sigma_min_diag : Sequence[float]
        Minimum diagonal values for Cholesky factor (prevents degeneracy).
    sigma_max_diag : Sequence[float], optional
        Maximum diagonal values for Cholesky factor (prevents over-smoothing).
    truncate : float, default=3.0
        Truncation radius in standard deviations for computational efficiency.
    device : torch.device, optional
        PyTorch device for computations.
    """

    def __init__(
        self,
        shape: Sequence[int],
        centers0: np.ndarray,  # (N, d) voxel coords
        L0: np.ndarray,  # (N, d, d) lower-triangular init (e.g., diag(sigmas))
        amps0: np.ndarray,  # (N,)
        sigma_min_diag: Sequence[float],  # per-axis minimal diag(L) (≈ σ floor)
        sigma_max_diag: Optional[Sequence[float]] = None,
        truncate: float = 3.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.dim = len(shape)
        self.truncate = float(truncate)
        N = centers0.shape[0]
        d = self.dim

        # Auto-detect best performing device: CUDA → CPU
        # Note: MPS is supported but currently slower than CPU for typical workloads
        if device is not None:
            device = device
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # ---- Center parameterization: sigmoid ensures centers stay within image bounds ----
        # Transform initial centers to sigmoid parameter space
        shape_arr = np.array(self.shape, dtype=np.float32)
        # Normalize to [0,1] coordinates (with small margin to avoid sigmoid saturation)
        u0 = np.clip(centers0 / np.maximum(shape_arr - 1.0, 1.0), 1e-6, 1 - 1e-6)
        # Inverse sigmoid (logit) transformation: raw_mu -> sigmoid(raw_mu) = u0
        raw_mu0 = np.log(u0) - np.log(1.0 - u0)  # logit function
        self.raw_mu = nn.Parameter(
            torch.tensor(raw_mu0, dtype=torch.float32, device=device)
        )

        # ---- Cholesky factor parameterization: ensure positive definiteness ----
        L0 = np.asarray(L0, dtype=np.float32)
        assert L0.shape == (N, d, d), (
            f"Expected L0 shape ({N}, {d}, {d}), got {L0.shape}"
        )
        # Extract diagonal elements (must be positive for valid Cholesky decomposition)
        diag0 = np.diagonal(L0, axis1=1, axis2=2)  # Shape: (N, d)

        # Pack off-diagonal elements (lower triangle only, since upper triangle is zero)
        # Store in row-major order: (i,j) for i>j (below diagonal)
        off_idx = [(i, j) for i in range(d) for j in range(i)]
        off0 = (
            np.stack([L0[:, i, j] for (i, j) in off_idx], axis=1)
            if len(off_idx)
            else np.zeros((N, 0), np.float32)
        )

        sigma_min_diag = np.asarray(sigma_min_diag, dtype=np.float32)
        assert sigma_min_diag.shape == (d,), "sigma_min_diag must be length d"

        if sigma_max_diag is not None:
            sigma_max_diag = np.asarray(sigma_max_diag, dtype=np.float32)
            assert sigma_max_diag.shape == (d,), "sigma_max_diag must be length d"
            self.sigma_max_diag = torch.tensor(
                sigma_max_diag, dtype=torch.float32, device=device
            )
        else:
            self.sigma_max_diag = None

        # Parameterize diagonal elements: sigma_min + softplus(raw) ensures positivity
        # Use inverse softplus to initialize raw parameters from desired diagonal values
        raw_diag0 = stable_inverse_softplus(
            np.maximum(diag0, sigma_min_diag) - sigma_min_diag
        )
        self.raw_L_diag = nn.Parameter(
            torch.tensor(raw_diag0, dtype=torch.float32, device=device)
        )

        # Off-diagonal elements can be any real number (no constraints)
        self.L_off = nn.Parameter(
            torch.tensor(off0, dtype=torch.float32, device=device)
        )

        # Store minimum diagonal constraint as non-trainable tensor
        self.sigma_min_diag = torch.tensor(
            sigma_min_diag, dtype=torch.float32, device=device
        )

        # ---- Amplitude parameterization: softplus ensures non-negativity ----
        # Initialize raw parameters using inverse softplus from desired amplitudes
        raw_a0 = stable_inverse_softplus(np.maximum(amps0, 1e-6))  # Avoid log(0)
        self.raw_a = nn.Parameter(
            torch.tensor(raw_a0, dtype=torch.float32, device=device)
        )

        # ---- Sharpness parameterization: exponential mapping s = 2 * exp(s') ----
        # Initialize to 0, which gives s = 2 * exp(0) = 2 (standard Gaussian)
        # s' > 0 → sharper edges, s' < 0 → softer edges
        # This allows symmetric exploration around standard Gaussian with L1 regularization
        sharpness_offsets0 = np.zeros(N, dtype=np.float32)
        self.sharpness_offsets_raw = nn.Parameter(
            torch.tensor(sharpness_offsets0, dtype=torch.float32, device=device)
        )

    def _build_L(self) -> torch.Tensor:
        """
        Reconstruct lower-triangular Cholesky factors from learnable parameters.

        Combines constrained diagonal elements with free off-diagonal elements to
        form valid lower-triangular matrices for covariance parameterization.

        Returns
        -------
        torch.Tensor, shape (N, d, d)
            Lower-triangular Cholesky factors where L @ L^T gives covariance matrices.
            Diagonal elements are guaranteed to be >= sigma_min_diag.
        """
        N, d = self.raw_L_diag.shape
        # Construct positive diagonal elements: min_constraint + softplus(raw_param)
        diag = self.sigma_min_diag + F.softplus(self.raw_L_diag)  # Shape: (N, d)

        # Apply maximum constraint if specified (prevents over-smoothing)
        if self.sigma_max_diag is not None:
            diag = torch.minimum(diag, self.sigma_max_diag)

        # Initialize lower-triangular matrices (zeros above diagonal)
        L = torch.zeros((N, d, d), dtype=torch.float32, device=diag.device)

        # Fill diagonal elements (constrained to be positive)
        for i in range(d):
            L[:, i, i] = diag[:, i]

        # Fill off-diagonal elements below diagonal (unconstrained)
        # Unpack from row-major storage order
        k = 0
        for i in range(d):
            for j in range(i):  # j < i (below diagonal)
                L[:, i, j] = self.L_off[:, k]
                k += 1
        return L

    def current_params(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract current parameter values from the model's learnable parameters.

        Applies all transformations to convert raw parameters to their final forms:
        - Centers: sigmoid transformation to ensure bounds
        - Cholesky factors: reconstruction from diagonal/off-diagonal components
        - Amplitudes: softplus transformation to ensure non-negativity
        - Sharpness: exponential mapping s = 2 * exp(s') to ensure positivity

        Returns
        -------
        centers : torch.Tensor, shape (N, d)
            Center coordinates in voxel units, bounded within image domain.
        L : torch.Tensor, shape (N, d, d)
            Lower-triangular Cholesky factors where covariance Σ = L @ L^T.
        amps : torch.Tensor, shape (N,)
            Non-negative amplitude values for each splat.
        sharpness : torch.Tensor, shape (N,)
            Sharpness values for each splat (s = 2 * exp(s')).
            s = 2 is standard Gaussian, s > 2 is sharper, s < 2 is softer.
        """
        # Transform raw parameters to normalized coordinates [0,1]
        u = torch.sigmoid(self.raw_mu)

        # Scale to actual voxel coordinates within image bounds
        shape = torch.tensor(self.shape, dtype=torch.float32, device=u.device)
        centers = u * torch.clamp(shape - 1.0, min=1.0)  # Maps [0,1] -> [0, shape-1]

        # Reconstruct Cholesky factors and apply amplitude transformation
        L = self._build_L()
        amps = F.softplus(self.raw_a)  # Ensures non-negative amplitudes

        # Apply exponential mapping for sharpness: s = 2 * exp(s')
        # This ensures s > 0 always, with s = 2 when s' = 0 (standard Gaussian)
        # Clamp s' to [-2.5, 2.5] for numerical stability: gives s in range [0.16, 24.5]
        # This provides wide sharpness variation while preventing numerical overflow
        sharpness_clamped = torch.clamp(self.sharpness_offsets_raw, min=-2.5, max=2.5)
        sharpness = 2.0 * torch.exp(sharpness_clamped)

        return centers, L, amps, sharpness

    # ===== Dynamic Management Methods =========================================

    @torch.no_grad()
    def _to_internal_params(
        self,
        centers: torch.Tensor,  # (N,d)
        Ls: torch.Tensor,  # (N,d,d)
        amps: torch.Tensor,  # (N,)
        sharpness: Optional[torch.Tensor] = None,  # (N,) optional
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert external (μ, L, a, s) to raw learnable params (raw_mu, L_diag_raw, L_off, amp_raw, sharpness_raw)."""
        device = self.raw_mu.device
        d = centers.shape[1]

        # centers -> raw_mu (logit in [0,1] coords)
        shape_arr = torch.tensor(self.shape, device=device, dtype=torch.float32)
        u = torch.clamp(
            centers / torch.clamp(shape_arr - 1.0, min=1.0), 1e-6, 1.0 - 1e-6
        )
        raw_mu = torch.log(u) - torch.log(1.0 - u)

        # L -> diag/off raw (diag via inverse-softplus)
        diag = torch.diagonal(Ls, dim1=1, dim2=2)  # (N,d)
        # Avoid zero/neg
        eps = 1e-6
        diag = torch.clamp(diag, min=eps)
        L_diag_raw = torch.tensor(
            stable_inverse_softplus(diag.detach().cpu().numpy()),
            device=device,
            dtype=torch.float32,
        )

        # Pack off-diagonals (row-major, below diag)
        off_elems = []
        for i in range(d):
            for j in range(i):
                off_elems.append(Ls[:, i, j])
        L_off = (
            torch.stack(off_elems, dim=1)
            if len(off_elems)
            else torch.zeros((centers.shape[0], 0), device=device)
        )

        # amps -> amp_raw
        amps = torch.clamp(amps, min=0.0)
        amp_raw = torch.tensor(
            stable_inverse_softplus(amps.detach().cpu().numpy()),
            device=device,
            dtype=torch.float32,
        )

        # sharpness -> sharpness_raw (inverse of s = 2 * exp(s'))
        # s' = log(s / 2)
        if sharpness is None:
            # Default to standard Gaussian (s = 2, s' = 0)
            sharpness_raw = torch.zeros(
                centers.shape[0], device=device, dtype=torch.float32
            )
        else:
            sharpness = torch.clamp(sharpness, min=1e-6)  # Avoid log(0)
            sharpness_raw = torch.log(sharpness / 2.0)

        return raw_mu, L_diag_raw, L_off, amp_raw, sharpness_raw

    @torch.no_grad()
    def replace_with(
        self,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
        sharpness: Optional[torch.Tensor] = None,
    ) -> None:
        """Hard replace the whole parameter set."""
        raw_mu, L_diag_raw, L_off, amp_raw, sharpness_raw = self._to_internal_params(
            centers, Ls, amps, sharpness
        )
        self.raw_mu = torch.nn.Parameter(raw_mu)
        self.raw_L_diag = torch.nn.Parameter(L_diag_raw)
        self.L_off = torch.nn.Parameter(L_off)
        self.raw_a = torch.nn.Parameter(amp_raw)
        self.sharpness_offsets_raw = torch.nn.Parameter(sharpness_raw)

    @torch.no_grad()
    def prune_(self, keep_mask: torch.Tensor) -> None:
        """Keep only indices where keep_mask is True."""
        self.raw_mu = torch.nn.Parameter(self.raw_mu[keep_mask])
        self.raw_L_diag = torch.nn.Parameter(self.raw_L_diag[keep_mask])
        self.L_off = torch.nn.Parameter(self.L_off[keep_mask])
        self.raw_a = torch.nn.Parameter(self.raw_a[keep_mask])
        self.sharpness_offsets_raw = torch.nn.Parameter(
            self.sharpness_offsets_raw[keep_mask]
        )

    @torch.no_grad()
    def append_(
        self,
        centers_new: torch.Tensor,
        Ls_new: torch.Tensor,
        amps_new: torch.Tensor,
        sharpness_new: Optional[torch.Tensor] = None,
    ) -> None:
        """Append new splats to the tail."""
        if centers_new.numel() == 0:
            return
        raw_mu, L_diag_raw, L_off, amp_raw, sharpness_raw = self._to_internal_params(
            centers_new, Ls_new, amps_new, sharpness_new
        )
        self.raw_mu = torch.nn.Parameter(torch.cat([self.raw_mu, raw_mu], dim=0))
        self.raw_L_diag = torch.nn.Parameter(
            torch.cat([self.raw_L_diag, L_diag_raw], dim=0)
        )
        self.L_off = torch.nn.Parameter(torch.cat([self.L_off, L_off], dim=0))
        self.raw_a = torch.nn.Parameter(torch.cat([self.raw_a, amp_raw], dim=0))
        self.sharpness_offsets_raw = torch.nn.Parameter(
            torch.cat([self.sharpness_offsets_raw, sharpness_raw], dim=0)
        )

    def n_splats(self) -> int:
        """Return current number of splats."""
        return int(self.raw_mu.shape[0])

    # =======================================================================

    @staticmethod
    def _sigma_diag_from_L(L: torch.Tensor) -> torch.Tensor:
        """
        Compute diag(Σ) for Σ = L L^T given L (d, d) or (N, d, d):
        Σ_ii = sum_j L[i,j]^2 (row-wise sum of squares).
        """
        if L.dim() == 2:
            return torch.sum(L * L, dim=1)  # (d,)
        else:
            return torch.sum(L * L, dim=2)  # (N, d)

    def forward(self) -> torch.Tensor:
        """
        Render all splats using AABB truncation at 'truncate' sigmas.
        Avoids explicit Σ^{-1} by solving L y = (x-μ) and using ||y||^2.
        Applies per-splat sharpness via generalized Gaussian: exp(-0.5 * ||y||^s).
        """
        device = self.raw_mu.device
        torch.zeros(self.shape, dtype=torch.float32, device=device)
        centers, Ls, amps, sharpness = self.current_params()

        return render_gaussians(
            self.shape,
            centers,
            Ls,
            amps,
            sharpness,
            truncate=self.truncate,
            intensity_floor=1e-5,
        )


def _group_by_box(
    lo: torch.Tensor, hi: torch.Tensor
) -> Dict[Tuple[int, ...], torch.Tensor]:
    """
    Group splats by their AABB shape so we can reuse a single base grid per group.
    Returns: dict { box_shape_tuple : idx_tensor }  (CPU tuple keys, GPU indices)

    Optimized to minimize GPU-CPU synchronization by using torch.unique on GPU.
    """
    sizes = (hi - lo).to(torch.int32)  # (N, d)

    # Use torch.unique on GPU to find unique sizes and group indices
    uniq, inv = torch.unique(sizes, dim=0, return_inverse=True)

    groups: Dict[Tuple[int, ...], torch.Tensor] = {}
    # Only transfer the small unique array to CPU for dict keys
    uniq_cpu = uniq.cpu().tolist()

    for g, key in enumerate(map(tuple, uniq_cpu)):
        # Find indices for this group on GPU
        idx = torch.nonzero(inv == g, as_tuple=False).squeeze(1)
        groups[key] = idx

    return groups


def _linear_strides(shape: Sequence[int], device) -> torch.Tensor:
    """Row-major linear strides for an nD tensor with given shape."""
    d = len(shape)
    s = [1]
    for i in range(d - 1, 0, -1):
        s.insert(0, s[0] * shape[i])
    return torch.tensor(s, device=device, dtype=torch.long)  # (d,)


def render_gaussians(
    shape: Sequence[int],
    centers: torch.Tensor,  # (N, d) voxel coords
    Ls: torch.Tensor,  # (N, d, d) lower-tri
    amps: torch.Tensor,  # (N,)
    sharpness: torch.Tensor,  # (N,) sharpness values (s = 2 * exp(s'))
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,  # for amplitude-aware culling (see §2)
    chunk_size: Optional[int] = None,  # P-dimension chunk size for memory control
) -> torch.Tensor:
    """
    Fast vectorized renderer with 2D/3D fast-paths and per-splat sharpness.
    Falls back to the generic nD implementation for d != 2 and d != 3.

    Renders Gaussians with generalized falloff: exp(-0.5 * ||y||^s) where s is sharpness.
    s = 2 is standard Gaussian, s > 2 is sharper, s < 2 is softer.
    """
    d = len(shape)
    if d == 2:
        return _render_gaussians_2d(
            shape, centers, Ls, amps, sharpness, truncate, intensity_floor, chunk_size
        )
    if d == 3:
        return _render_gaussians_3d(
            shape, centers, Ls, amps, sharpness, truncate, intensity_floor, chunk_size
        )

    # --- keep existing nD implementation below unchanged ---
    device = centers.device
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)
    out_flat = out.view(-1)
    strides = _linear_strides(shape, device)  # (d,)

    # --- AABB per splat ---
    # Σ_ii = row-wise sum(L^2); r_i = ceil(truncate * sqrt(Σ_ii))
    # Adjust truncate for sharpness: for generalized Gaussian exp(-0.5 * r^s),
    # to reach same threshold as truncate*σ for s=2, we need: r = (truncate^2)^(1/s) * σ
    sigma_diag = torch.sum(Ls * Ls, dim=2)  # (N, d)
    effective_truncate = truncate ** (
        2.0 / sharpness
    )  # (N,) - sharpness-adjusted radius
    radii = torch.clamp(
        (effective_truncate[:, None] * torch.sqrt(torch.clamp(sigma_diag, 1e-8)))
        .ceil()
        .to(torch.long),
        min=1,
    )
    lo = torch.clamp((centers - radii).floor().to(torch.long), min=0)  # (N, d)
    hi = torch.minimum(
        (centers + radii).ceil().to(torch.long) + 1,
        torch.tensor(shape, device=device).to(torch.long),
    )  # (N, d)

    # (Optional) **amplitude-aware shrinking** (see §2) – avoids giant boxes for tiny a
    if intensity_floor is not None and intensity_floor > 0:
        # t_max per splat solves: a * exp(-0.5 * t^s) >= intensity_floor  ->  t <= (2 log(a/eps))^(1/s)
        # For s=2: t <= sqrt(2 log(a/eps)) (original formula)
        eps = torch.tensor(intensity_floor, device=device, dtype=torch.float32)
        a = torch.clamp(amps, min=1e-12)
        log_ratio = torch.clamp(2.0 * torch.log(torch.clamp(a / eps, min=1.0)), min=0.0)
        tmax = torch.pow(
            log_ratio, 1.0 / sharpness
        )  # (N,) - sharpness-adjusted threshold
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
        sharpness = sharpness[valid]
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
        s = sharpness[idx]  # (K,)

        # Build base grid (one per group, on device)
        # coords_i = [0, 1, ..., h_i-1]  -> broadcast to P points
        ranges = [
            torch.arange(s, device=device, dtype=torch.float32) for s in box_shape
        ]
        grids = torch.meshgrid(*ranges, indexing="ij")
        P = int(np.prod(box_shape))  # More efficient than loop
        # per-axis base coords flattened (P,)
        base = torch.stack([g.reshape(-1) for g in grids], dim=0)  # (d, P)

        # Precompute base indices and linear offsets for the group
        lo_f = lo[idx].to(torch.float32)  # (K, d)
        strides_f = strides.to(torch.float32)
        lin_offsets = (base.T @ strides_f).to(torch.long)  # (P,)
        base_idx = (lo[idx].to(torch.long) * strides).sum(dim=1)  # (K,)

        # *** MEMORY OPTIMIZATION: Process P dimension in chunks to prevent OOM ***
        # Calculate optimal chunk size based on available memory and tensor dimensions
        P_chunk = chunk_size or _calculate_optimal_chunk_size(
            K=len(idx), d=d, device=device, dtype=torch.float32
        )
        for p0 in range(0, P, P_chunk):
            p1 = min(P, p0 + P_chunk)

            # Process chunk: base coordinates and indices for this chunk
            base_chunk = base[:, p0:p1]  # (d, Pc)
            lin_offsets_chunk = lin_offsets[p0:p1]  # (Pc,)

            # Per-splat Δ = base + lo - μ   (broadcast to (K, d, Pc))
            delta = (
                base_chunk[None, :, :] + lo_f[:, :, None] - mu[:, :, None]
            )  # (K, d, Pc)

            # Solve L y = Δ  (batched lower-tri solve with Pc RHS per splat)
            try:
                y = torch.linalg.solve_triangular(L, delta, upper=False)
            except Exception:
                # older PyTorch fallback
                y, _ = torch.triangular_solve(delta, L, upper=False)

            # Exponent and values: exp(-0.5 * ||y||^s) * a where s is sharpness
            # ||y||^s = (||y||^2)^(s/2) = expo^(s/2)
            expo = torch.sum(y * y, dim=1)  # (K, Pc)
            # Clamp expo to avoid log(0) in gradients of pow(expo, s/2)
            expo_safe = torch.clamp(expo, min=1e-10)
            vals = (
                torch.exp(-0.5 * torch.pow(expo_safe, s[:, None] / 2.0)) * a[:, None]
            )  # (K, Pc)

            # Absolute flat indices (K, Pc) -> (K*Pc,)
            idx_flat = (base_idx[:, None] + lin_offsets_chunk[None, :]).reshape(-1)
            vals_flat = vals.reshape(-1)

            # Accumulate into output
            out_flat.index_add_(0, idx_flat, vals_flat)

    return out


def render_gaussians_numpy(
    shape: Sequence[int],
    params_full: np.ndarray,
    amps: np.ndarray,
    truncate: float = 3.0,
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """CPU NumPy output wrapper around torch renderer (no grads)."""
    from luxar.gsplats.utils.trils import tril_size, unpack_tril

    # Input validation
    if params_full.size == 0:
        return np.zeros(shape, dtype=np.float32)

    # Determine dimensionality and unpack parameters
    d = len(shape)
    tril_elements = tril_size(d)

    if params_full.shape[1] != d + tril_elements:
        raise ValueError(
            f"params_full should have {d + tril_elements} columns for {d}D data, "
            f"got {params_full.shape[1]}"
        )

    # Split parameters: centers (first d columns) + packed Cholesky (remaining columns)
    centers = params_full[:, :d].astype(np.float32)
    packed_L = params_full[:, d:].astype(np.float32)

    # Unpack Cholesky factors
    ls = unpack_tril(packed_L, d)

    # Convert to PyTorch tensors (CPU, no gradients needed)
    centers_torch = torch.tensor(centers, dtype=torch.float32, device="cpu")
    ls_torch = torch.tensor(ls, dtype=torch.float32, device="cpu")
    amps_torch = torch.tensor(
        amps.astype(np.float32), dtype=torch.float32, device="cpu"
    )
    # Default to standard Gaussian sharpness (s = 2.0) for packed parameter wrapper
    sharpness_torch = torch.full(
        (centers_torch.shape[0],), 2.0, dtype=torch.float32, device="cpu"
    )

    # Render using the PyTorch function
    with torch.no_grad():
        result = render_gaussians(
            shape,
            centers_torch,
            ls_torch,
            amps_torch,
            sharpness_torch,
            truncate=truncate,
            chunk_size=chunk_size,
        )

    return result.cpu().numpy()


def render_gaussians_pytorch(
    shape: Sequence[int],
    params_full: np.ndarray,
    amps: np.ndarray,
    truncate: float = 3.0,
    device: str = "cpu",
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """PyTorch wrapper for rendering gaussians with packed parameters."""
    from luxar.gsplats.utils.trils import tril_size, unpack_tril

    # Input validation
    if params_full.size == 0:
        return torch.zeros(shape, dtype=torch.float32, device=device)

    # Determine dimensionality and unpack parameters
    d = len(shape)
    tril_elements = tril_size(d)

    if params_full.shape[1] != d + tril_elements:
        raise ValueError(
            f"params_full should have {d + tril_elements} columns for {d}D data, "
            f"got {params_full.shape[1]}"
        )

    # Split parameters: centers (first d columns) + packed Cholesky (remaining columns)
    centers = params_full[:, :d].astype(np.float32)
    packed_L = params_full[:, d:].astype(np.float32)

    # Unpack Cholesky factors
    ls = unpack_tril(packed_L, d)

    # Convert to PyTorch tensors
    centers_torch = torch.tensor(centers, dtype=torch.float32, device=device)
    ls_torch = torch.tensor(ls, dtype=torch.float32, device=device)
    amps_torch = torch.tensor(
        amps.astype(np.float32), dtype=torch.float32, device=device
    )
    # Default to standard Gaussian sharpness (s = 2.0) for packed parameter wrapper
    sharpness_torch = torch.full(
        (centers_torch.shape[0],), 2.0, dtype=torch.float32, device=device
    )

    # Render using the PyTorch function
    result = render_gaussians(
        shape,
        centers_torch,
        ls_torch,
        amps_torch,
        sharpness_torch,
        truncate=truncate,
        chunk_size=chunk_size,
    )

    return result


def render_gaussians_batched(
    shape: Sequence[int],
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    sharpness: Optional[torch.Tensor] = None,
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Batched wrapper for render_gaussians - identical functionality."""
    # Default to standard Gaussian (s = 2.0) if sharpness not provided
    if sharpness is None:
        sharpness = torch.full(
            (centers.shape[0],),
            2.0,
            dtype=torch.float32,
            device=centers.device,
        )
    return render_gaussians(
        shape, centers, Ls, amps, sharpness, truncate, intensity_floor, chunk_size
    )
