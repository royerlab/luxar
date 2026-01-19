# gsplat_model.py
"""
Gaussian Splat Model for n-dimensional oriented Gaussian splatting.

This module contains the PyTorch model class for optimizing collections of
oriented Gaussian functions to reconstruct images and volumes.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
from luxar.gsplats.models.utils.inverse_softplus import stable_inverse_softplus


class GaussianSplatModel(nn.Module):
    """
    PyTorch model for n-dimensional oriented Gaussian splats with full covariance matrices.

    This model represents a collection of oriented Gaussian functions (splats) that can be
    optimized to reconstruct images or volumes. Each splat is parameterized by:

    1. **Center position**: Constrained to image domain via sigmoid parameterization
    2. **Covariance matrix**: Represented via Cholesky decomposition L where Σ = L @ L^T
    3. **Amplitude**: Non-negative scalar via softplus activation
    4. **Sharpness**: Generalized Gaussian falloff via exponential mapping s = 2 * exp(s')

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
    amp_max : float, optional
        Maximum amplitude value. Prevents amplitude explosion during optimization,
        especially with aggressive compression (few splats). Since images are
        normalized to [0, 1], a value of 1.0 matches the max possible intensity.
    max_eccentricity : float, optional
        Maximum allowed eccentricity (ratio of largest to smallest eigenvalue
        of the covariance matrix Σ = L @ L^T). This bounds the actual shape
        elongation of the Gaussian splats. For example, max_eccentricity=4.0
        means the longest axis can be at most 2x the shortest (since
        eccentricity is the variance ratio, axis ratio = sqrt(eccentricity)).
    sharpness_range : tuple[float, float] | float, optional
        Range for sharpness values. If a tuple (min, max), sharpness is clamped
        to this range. If a single float, sharpness is fixed to that value.
        Sharpness of 2.0 is standard Gaussian; higher values give sharper edges.
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
        amp_max: Optional[float] = None,  # Maximum amplitude (prevents explosion)
        max_eccentricity: Optional[float] = None,  # Max ratio of longest/shortest axis
        sharpness_range: Optional[
            tuple[float, float] | float
        ] = None,  # Sharpness constraints
        truncate: float = 3.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.shape = tuple(shape)
        self.dim = len(shape)
        self.truncate = float(truncate)
        N = centers0.shape[0]
        d = self.dim

        # Auto-detect best performing device: CUDA → CPU
        # Note: MPS is supported but currently slower than CPU for typical workloads
        if device is None:
            if torch.cuda.is_available():
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

        # Store amplitude maximum constraint (prevents amplitude explosion during optimization)
        self.amp_max: float | None = amp_max

        # Store eccentricity constraint (limits ratio of longest to shortest axis)
        self.max_eccentricity: float | None = max_eccentricity

        # Store sharpness range constraint
        # Can be tuple (min, max) or fixed float value
        self.sharpness_range: tuple[float, float] | float | None = sharpness_range

        # ---- Cholesky factor parameterization: ensure positive definiteness ----
        self.sigma_max_diag: torch.Tensor | None
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

        sigma_min_diag_arr = np.asarray(sigma_min_diag, dtype=np.float32)
        assert sigma_min_diag_arr.shape == (d,), "sigma_min_diag must be length d"

        if sigma_max_diag is not None:
            sigma_max_diag_arr = np.asarray(sigma_max_diag, dtype=np.float32)
            assert sigma_max_diag_arr.shape == (d,), "sigma_max_diag must be length d"
            self.sigma_max_diag = torch.tensor(
                sigma_max_diag_arr, dtype=torch.float32, device=device
            )
        else:
            self.sigma_max_diag = None

        # Parameterize diagonal elements: sigma_min + softplus(raw) ensures positivity
        # Use inverse softplus to initialize raw parameters from desired diagonal values
        raw_diag0 = stable_inverse_softplus(
            np.maximum(diag0, sigma_min_diag_arr) - sigma_min_diag_arr
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

        # Apply eccentricity constraint if specified
        # Two-part constraint for efficiency:
        # 1. Constrain diagonal ratio: max(diag)/min(diag) <= sqrt(max_eccentricity)
        # 2. Constrain off-diagonal magnitude relative to diagonal (see below)
        if self.max_eccentricity is not None:
            max_ratio = float(self.max_eccentricity) ** 0.5
            min_diag = diag.min(dim=1, keepdim=True).values  # (N, 1)
            max_allowed_diag = min_diag * max_ratio
            diag = torch.minimum(diag, max_allowed_diag)

        # Initialize lower-triangular matrices (zeros above diagonal)
        L = torch.zeros((N, d, d), dtype=torch.float32, device=diag.device)

        # Fill diagonal elements (constrained to be positive)
        for i in range(d):
            L[:, i, i] = diag[:, i]

        # Fill off-diagonal elements below diagonal
        # Apply eccentricity constraint on off-diagonals to prevent elongation
        k = 0

        # Precompute gamma for eccentricity constraint if needed
        # For 2D, the exact relationship between gamma and eccentricity E is:
        #   E = ((2 + γ²) + γ√(γ² + 4)) / ((2 + γ²) - γ√(γ² + 4))
        # Solving for γ gives: γ ≈ sqrt(E-1) / k where k varies with E
        # k values: E=2 → k=2.87, E=4 → k=2.45, E→∞ → k=2.41
        # Tight approximation: k = 2.4 + 0.5/sqrt(E-1)
        # For d dimensions, scale by (d-1)^0.7 to account for multiple off-diagonals
        # (power 0.7 empirically gives tightest bounds across dimensions)
        off_gamma = None
        if self.max_eccentricity is not None and d > 1:
            E = float(self.max_eccentricity)
            sqrt_E_minus_1 = (E - 1.0) ** 0.5
            # k varies with E: tighter for small E, looser for large E
            k_2d = 2.4 + 0.5 / sqrt_E_minus_1 if sqrt_E_minus_1 > 0 else 3.0
            # Scale for dimension with power 0.7
            off_gamma = sqrt_E_minus_1 / (k_2d * (d - 1) ** 0.7)

        for i in range(d):
            for j in range(i):  # j < i (below diagonal)
                off_val = self.L_off[:, k]

                # Constrain off-diagonal elements to limit eccentricity
                # |L[i,j]| <= γ * min(L[i,i], L[j,j])
                if off_gamma is not None:
                    min_diag_ij = torch.minimum(diag[:, i], diag[:, j])
                    max_off = off_gamma * min_diag_ij
                    off_val = torch.clamp(off_val, min=-max_off, max=max_off)

                L[:, i, j] = off_val
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

        # Apply maximum amplitude constraint if specified (prevents explosion during optimization)
        if self.amp_max is not None:
            amps = torch.clamp(amps, max=self.amp_max)

        # Apply exponential mapping for sharpness: s = 2 * exp(s')
        # This ensures s > 0 always, with s = 2 when s' = 0 (standard Gaussian)
        # Clamp s' to [-2.5, 2.5] for numerical stability: gives s in range [0.16, 24.5]
        # This provides wide sharpness variation while preventing numerical overflow
        sharpness_clamped = torch.clamp(self.sharpness_offsets_raw, min=-2.5, max=2.5)
        sharpness = 2.0 * torch.exp(sharpness_clamped)

        # Apply sharpness range constraint if specified
        if self.sharpness_range is not None:
            if isinstance(self.sharpness_range, (int, float)):
                # Fixed sharpness value
                sharpness = torch.full_like(sharpness, float(self.sharpness_range))
            else:
                # Tuple (min, max) - clamp to range
                sharpness = torch.clamp(
                    sharpness,
                    min=float(self.sharpness_range[0]),
                    max=float(self.sharpness_range[1]),
                )

        return centers, L, amps, sharpness

    # ===== Dynamic Management Methods =========================================

    @torch.no_grad()
    def _to_internal_params(
        self,
        centers: torch.Tensor,  # (N,d)
        Ls: torch.Tensor,  # (N,d,d)
        amps: torch.Tensor,  # (N,)
        sharpness: torch.Tensor,  # (N,) required
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
        sharpness = torch.clamp(sharpness, min=1e-6)  # Avoid log(0)
        sharpness_raw = torch.log(sharpness / 2.0)

        return raw_mu, L_diag_raw, L_off, amp_raw, sharpness_raw

    @torch.no_grad()
    def replace_with(
        self,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
        sharpness: torch.Tensor,
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
        sharpness_new: torch.Tensor,
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
