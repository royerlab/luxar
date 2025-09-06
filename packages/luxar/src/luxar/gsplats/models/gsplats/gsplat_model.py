# gsplat_model.py

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from luxar.gsplats.models.utils.inverse_softplus import stable_inverse_softplus


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

    def current_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract current parameter values from the model's learnable parameters.

        Applies all transformations to convert raw parameters to their final forms:
        - Centers: sigmoid transformation to ensure bounds
        - Cholesky factors: reconstruction from diagonal/off-diagonal components
        - Amplitudes: softplus transformation to ensure non-negativity

        Returns
        -------
        centers : torch.Tensor, shape (N, d)
            Center coordinates in voxel units, bounded within image domain.
        L : torch.Tensor, shape (N, d, d)
            Lower-triangular Cholesky factors where covariance Σ = L @ L^T.
        amps : torch.Tensor, shape (N,)
            Non-negative amplitude values for each splat.
        """
        # Transform raw parameters to normalized coordinates [0,1]
        u = torch.sigmoid(self.raw_mu)

        # Scale to actual voxel coordinates within image bounds
        shape = torch.tensor(self.shape, dtype=torch.float32, device=u.device)
        centers = u * torch.clamp(shape - 1.0, min=1.0)  # Maps [0,1] -> [0, shape-1]

        # Reconstruct Cholesky factors and apply amplitude transformation
        L = self._build_L()
        amps = F.softplus(self.raw_a)  # Ensures non-negative amplitudes

        return centers, L, amps

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
        """
        device = self.raw_mu.device
        out = torch.zeros(self.shape, dtype=torch.float32, device=device)
        centers, Ls, amps = self.current_params()

        return render_gaussians(
            self.shape,
            centers,
            Ls,
            amps,
            truncate=self.truncate,
            intensity_floor=1e-5,
        )




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



def render_gaussians(
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

def render_gaussians_numpy(
    shape: Sequence[int],
    params_full: np.ndarray,
    amps: np.ndarray,
    truncate: float = 3.0,
) -> np.ndarray:
    """CPU NumPy output wrapper around torch renderer (no grads)."""
    from luxar.gsplats.utils.trils import unpack_tril, tril_size
    
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
    amps_torch = torch.tensor(amps.astype(np.float32), dtype=torch.float32, device="cpu")
    
    # Render using the PyTorch function
    with torch.no_grad():
        result = render_gaussians(
            shape, centers_torch, ls_torch, amps_torch, truncate=truncate
        )
    
    return result.cpu().numpy()


def render_gaussians_pytorch(
    shape: Sequence[int],
    params_full: np.ndarray,
    amps: np.ndarray,
    truncate: float = 3.0,
    device: str = "cpu",
) -> torch.Tensor:
    """PyTorch wrapper for rendering gaussians with packed parameters."""
    from luxar.gsplats.utils.trils import unpack_tril, tril_size
    
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
    amps_torch = torch.tensor(amps.astype(np.float32), dtype=torch.float32, device=device)
    
    # Render using the PyTorch function
    result = render_gaussians(
        shape, centers_torch, ls_torch, amps_torch, truncate=truncate
    )
    
    return result


def render_gaussians_batched(
    shape: Sequence[int],
    centers: torch.Tensor,
    Ls: torch.Tensor, 
    amps: torch.Tensor,
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,
) -> torch.Tensor:
    """Batched wrapper for render_gaussians - identical functionality."""
    return render_gaussians(shape, centers, Ls, amps, truncate, intensity_floor)
