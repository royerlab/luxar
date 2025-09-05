"""
Optimized Precision Matrix Parameterized Gaussian Splat Model

This model uses the TRUE precision-only approach with upper triangular Cholesky:
- Store: μ, U (upper triangular), a where Λ = U^T @ U  
- Render: g(x) = a * exp(-0.5 * ||U(x-μ)||²) via triangular matmul
- AABB: r_i = t * ||U^{-T} e_i|| via triangular solves
- No matrix inversions needed anywhere

This is the numerically optimal and computationally efficient approach.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from luxar.gsplats.models.gsplats.gsplats_precision_render_optimized import (
    render_gaussians_precision_optimized,
)
from luxar.gsplats.models.utils.inverse_softplus import stable_inverse_softplus


class GaussianSplatPrecisionModelOptimized(nn.Module):
    """
    Optimized PyTorch model for n-dimensional oriented Gaussian splats using 
    upper triangular precision Cholesky parameterization.

    This model represents Gaussian functions in the true precision-only form:
        G(x) = a * exp(-0.5 * ||U(x-μ)||²)
    where:
        - μ: center position  
        - U: upper triangular Cholesky factor of precision matrix Λ = U^T @ U
        - a: amplitude

    Key advantages:
    - Numerically stable (no matrix inversions)
    - Computationally optimal (triangular operations only)
    - Memory efficient (no full precision/covariance matrices stored)

    Parameters
    ----------
    shape : Sequence[int]
        Dimensions of the target image/volume to reconstruct.
    centers0 : np.ndarray, shape (N, d)
        Initial center positions in voxel coordinates.
    U0 : np.ndarray, shape (N, d, d)
        Initial upper-triangular Cholesky factors for precision matrices.
    amps0 : np.ndarray, shape (N,)
        Initial amplitude values.
    precision_min_diag : Sequence[float]
        Minimum diagonal values for precision Cholesky factor.
    precision_max_diag : Sequence[float], optional
        Maximum diagonal values for precision Cholesky factor.
    truncate : float, default=3.0
        Truncation radius in standard deviations for computational efficiency.
    device : torch.device, optional
        PyTorch device for computations.
    """

    def __init__(
        self,
        shape: Sequence[int],
        centers0: np.ndarray,  # (N, d) voxel coords
        U0: np.ndarray,  # (N, d, d) upper-triangular precision Cholesky factors
        amps0: np.ndarray,  # (N,)
        precision_min_diag: Sequence[float],  # min diag(U) for precision
        precision_max_diag: Optional[Sequence[float]] = None,
        max_aspect_ratio: Optional[float] = None,  # constrain anisotropy
        truncate: float = 3.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.dim = len(shape)
        self.truncate = float(truncate)
        N = centers0.shape[0]
        d = self.dim

        # Auto-detect device
        if device is not None:
            device = device
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # ---- Center parameterization ----
        shape_arr = np.array(self.shape, dtype=np.float32)
        u0 = np.clip(centers0 / np.maximum(shape_arr - 1.0, 1.0), 1e-6, 1 - 1e-6)
        raw_mu0 = np.log(u0) - np.log(1.0 - u0)  # logit function
        self.raw_mu = nn.Parameter(
            torch.tensor(raw_mu0, dtype=torch.float32, device=device)
        )

        # ---- Upper triangular Cholesky factor parameterization ----
        U0 = np.asarray(U0, dtype=np.float32)
        assert U0.shape == (N, d, d), f"Expected U0 shape ({N}, {d}, {d}), got {U0.shape}"
        
        # Ensure U0 is upper triangular
        for i in range(N):
            U0[i] = np.triu(U0[i])  # Zero out lower triangle
        
        # Extract diagonal elements
        diag0 = np.diagonal(U0, axis1=1, axis2=2)  # Shape: (N, d)

        # Pack upper-triangular off-diagonal elements
        off_idx = [(i, j) for i in range(d) for j in range(i+1, d)]  # Upper triangle
        off0 = (
            np.stack([U0[:, i, j] for (i, j) in off_idx], axis=1)
            if len(off_idx)
            else np.zeros((N, 0), np.float32)
        )

        precision_min_diag = np.asarray(precision_min_diag, dtype=np.float32)
        assert precision_min_diag.shape == (d,), "precision_min_diag must be length d"

        if precision_max_diag is not None:
            precision_max_diag = np.asarray(precision_max_diag, dtype=np.float32)
            assert precision_max_diag.shape == (d,), "precision_max_diag must be length d"
            self.precision_max_diag = torch.tensor(
                precision_max_diag, dtype=torch.float32, device=device
            )
        else:
            self.precision_max_diag = None

        # Parameterize diagonal elements for precision matrix
        raw_diag0 = stable_inverse_softplus(
            np.maximum(diag0, precision_min_diag) - precision_min_diag
        )
        self.raw_U_diag = nn.Parameter(
            torch.tensor(raw_diag0, dtype=torch.float32, device=device)
        )

        # Off-diagonal elements (upper triangle)
        self.U_off = nn.Parameter(
            torch.tensor(off0, dtype=torch.float32, device=device)
        )

        self.precision_min_diag = torch.tensor(
            precision_min_diag, dtype=torch.float32, device=device
        )
        
        # Optional anisotropy constraint
        self.max_aspect_ratio = max_aspect_ratio

        # ---- Amplitude parameterization ----
        raw_a0 = stable_inverse_softplus(np.maximum(amps0, 1e-6))
        self.raw_a = nn.Parameter(
            torch.tensor(raw_a0, dtype=torch.float32, device=device)
        )

    def _build_U(self) -> torch.Tensor:
        """
        Reconstruct upper-triangular Cholesky factors for precision matrices.
        
        Returns
        -------
        torch.Tensor, shape (N, d, d)
            Upper-triangular Cholesky factors where Λ = U^T @ U gives precision matrices.
        """
        N, d = self.raw_U_diag.shape
        # Construct positive diagonal elements
        diag = self.precision_min_diag + F.softplus(self.raw_U_diag)  # Shape: (N, d)

        # Apply maximum constraint if specified
        if self.precision_max_diag is not None:
            diag = torch.minimum(diag, self.precision_max_diag)

        # Initialize upper-triangular matrices
        U = torch.zeros((N, d, d), dtype=torch.float32, device=diag.device)

        # Fill diagonal elements
        for i in range(d):
            U[:, i, i] = diag[:, i]

        # Fill off-diagonal elements (upper triangle only)
        k = 0
        for i in range(d):
            for j in range(i+1, d):  # j > i (above diagonal)
                if self.max_aspect_ratio is not None:
                    # Apply tight anisotropy constraint
                    # For 2D case: constrain |U[i,j]| ≤ α * sqrt(U[i,i] * U[j,j])
                    # where α is chosen to limit condition number of precision matrix
                    # For max aspect ratio R, use α ≈ (R-1)/(R+1) which is tighter
                    alpha = (self.max_aspect_ratio - 1) / (self.max_aspect_ratio + 1)
                    diag_geom_mean = torch.sqrt(diag[:, i] * diag[:, j])
                    max_off_val = alpha * diag_geom_mean
                    
                    U[:, i, j] = torch.clamp(
                        self.U_off[:, k],
                        -max_off_val,
                        max_off_val
                    )
                else:
                    # No constraint
                    U[:, i, j] = self.U_off[:, k]
                k += 1
        return U

    def current_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract current parameter values from the model's learnable parameters.

        Returns
        -------
        centers : torch.Tensor, shape (N, d)
            Center coordinates in voxel units.
        U : torch.Tensor, shape (N, d, d)
            Upper-triangular Cholesky factors where precision Λ = U^T @ U.
        amps : torch.Tensor, shape (N,)
            Non-negative amplitude values for each splat.
        """
        # Transform to coordinates
        u = torch.sigmoid(self.raw_mu)
        shape = torch.tensor(self.shape, dtype=torch.float32, device=u.device)
        centers = u * torch.clamp(shape - 1.0, min=1.0)

        # Reconstruct precision Cholesky factors
        U = self._build_U()
        amps = F.softplus(self.raw_a)

        return centers, U, amps

    def get_precision_matrices(self) -> torch.Tensor:
        """
        Get the precision matrices Λ = U^T @ U.
        
        Returns
        -------
        torch.Tensor, shape (N, d, d)
            Precision matrices.
        """
        _, U, _ = self.current_params()
        return U.transpose(-1, -2) @ U
    
    def get_covariance_matrices(self) -> torch.Tensor:
        """
        Get the covariance matrices Σ = Λ^{-1} for compatibility.
        
        Returns
        -------
        torch.Tensor, shape (N, d, d) 
            Covariance matrices (inverse of precision matrices).
        """
        Lambda = self.get_precision_matrices()
        return torch.inverse(Lambda)

    @staticmethod
    def _precision_diag_from_U(U: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal of precision matrix Λ = U^T @ U.
        Λ_ii = sum_j U[j,i]^2 (column-wise sum of squares for upper triangular).
        """
        if U.dim() == 2:
            return torch.sum(U * U, dim=0)  # (d,) - sum over rows for each column
        else:
            return torch.sum(U * U, dim=1)  # (N, d) - sum over rows for each column

    def forward(self) -> torch.Tensor:
        """
        Render all splats using optimized precision-only approach.
        Direct computation: exp(-0.5 * ||U(x-μ)||²) with triangular operations only!
        """
        centers, Us, amps = self.current_params()
        
        return render_gaussians_precision_optimized(
            self.shape,
            centers, 
            Us,
            amps,
            truncate=self.truncate,
            intensity_floor=1e-5,
        )