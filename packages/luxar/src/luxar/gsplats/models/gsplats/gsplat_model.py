from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from luxar.gsplats.models.gsplats.gsplats_batched_render import (
    render_gaussians_batched,
)
from luxar.gsplats.models.utils.inverse_softplus import stable_inverse_softplus
from luxar.gsplats.models.utils.lt_solver import solve_lower_triangular


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
    batched : bool, default=True
        Whether to use batched rendering implementation for better performance.
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
        batched: bool = True,
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.dim = len(shape)
        self.truncate = float(truncate)
        N = centers0.shape[0]
        d = self.dim
        self.batched = batched

        device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

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

        # Use batched implementation for better performance
        if self.batched:
            return render_gaussians_batched(
                self.shape,
                centers,
                Ls,
                amps,
                truncate=self.truncate,
                intensity_floor=1e-5,
            )

        # Fall back to sequential processing only when batched mode is disabled
        # Note: Consider enabling batched mode for better performance
        N, d = centers.shape
        shape_t = torch.tensor(self.shape, dtype=torch.long, device=device)

        # Pre-compute sigma_diag for all splats to reduce redundant calculations
        sigma_diag_all = self._sigma_diag_from_L(Ls)  # (N, d)

        for k in range(N):
            mu = centers[k]  # (d,)
            L = Ls[k]  # (d, d) lower-tri
            a = amps[k]  # scalar

            # Use pre-computed sigma_diag to avoid redundant calculations
            sigma_diag = sigma_diag_all[k]  # (d,)

            # AABB radius per axis: r_i = truncate * sqrt(Σ_ii), Σ = L L^T
            r = torch.clamp(
                (self.truncate * torch.sqrt(torch.clamp(sigma_diag, min=1e-8)))
                .ceil()
                .long(),
                min=1,
            )

            # Local bounding box [lo, hi) with bounds checking
            lo = torch.clamp((mu - r).floor().long(), min=0)
            hi = torch.minimum((mu + r).ceil().long() + 1, shape_t)

            # Skip degenerate boxes (empty or invalid)
            if torch.any(hi <= lo):
                continue  # empty box

            # Build local grid (P points), flatten to (P, d)
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

            # Solve L y = delta^T  -> y has shape (d, P)
            y = solve_lower_triangular(L, delta.T)

            # Mahalanobis exponent = sum of squares per column
            expo = torch.sum(y * y, dim=0)  # (P,)
            G = torch.exp(-0.5 * expo) * a  # (P,)

            # Scatter-add into canvas
            slicer = tuple(
                slice(int(lo[i].item()), int(hi[i].item())) for i in range(d)
            )
            out[slicer] = out[slicer] + G.reshape(
                [int(hi[i] - lo[i]) for i in range(d)]
            )

        return out
