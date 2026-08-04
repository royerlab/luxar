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
from arbol import aprint

from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
from luxar.gsplats.models.utils.inverse_softplus import (
    stable_inverse_softplus,
    stable_inverse_softplus_torch,
)
from luxar.gsplats.utils import resolve_torch_device
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS


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
    truncate : float, default=DEFAULT_TRUNCATION_RADIUS
        Truncation radius in standard deviations for computational efficiency.
    device : str or torch.device, optional
        PyTorch device for computations. Explicit values override auto-detection.
    use_cuda : bool, default=True
        Allow CUDA during auto-detection when ``device`` is not provided.
    use_metal : bool, default=True
        Allow MPS/Metal during auto-detection when ``device`` is not provided.
    """

    # Class-level type annotations for register_buffer attributes.
    # These override mypy's default `Tensor | Module` inference from register_buffer().
    sigma_min_diag: torch.Tensor
    sigma_max_diag: torch.Tensor | None
    voxel_size: torch.Tensor | None
    _diag_idx: torch.Tensor
    _tril_rows: torch.Tensor
    _tril_cols: torch.Tensor
    _shape_f32: torch.Tensor

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
        truncate: float = DEFAULT_TRUNCATION_RADIUS,
        voxel_size: Optional[np.ndarray] = None,
        device: Optional[str | torch.device] = None,
        use_cuda: bool = True,
        use_metal: bool = True,
    ) -> None:
        super().__init__()
        self.shape = tuple(shape)
        self.dim = len(shape)
        self.truncate = float(truncate)
        N = centers0.shape[0]
        d = self.dim

        # Auto-detect best available accelerator consistently with the fitting API:
        # CUDA → MPS/Metal → CPU, honoring explicit accelerator opt-out flags.
        resolved_device = resolve_torch_device(
            device,
            use_cuda=use_cuda,
            use_metal=use_metal,
        )

        aprint(f"GaussianSplatModel: using device '{resolved_device}'")
        device = resolved_device

        # Store voxel_size for physical-space constraint enforcement
        # register_buffer ensures it moves with .to() calls
        if voxel_size is not None:
            self.register_buffer(
                "voxel_size",
                torch.tensor(
                    np.asarray(voxel_size, dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                ),
                persistent=False,
            )
        else:
            self.voxel_size = None

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

        sigma_min_diag_arr = np.asarray(sigma_min_diag, dtype=np.float32)
        assert sigma_min_diag_arr.shape == (d,), "sigma_min_diag must be length d"

        if sigma_max_diag is not None:
            sigma_max_diag_arr = np.asarray(sigma_max_diag, dtype=np.float32)
            assert sigma_max_diag_arr.shape == (d,), "sigma_max_diag must be length d"
            self.register_buffer(
                "sigma_max_diag",
                torch.tensor(sigma_max_diag_arr, dtype=torch.float32, device=device),
                persistent=False,
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

        # Store minimum diagonal constraint as non-trainable buffer
        self.register_buffer(
            "sigma_min_diag",
            torch.tensor(sigma_min_diag, dtype=torch.float32, device=device),
            persistent=False,
        )

        # ---- Amplitude parameterization: softplus ensures non-negativity ----
        # Initialize raw parameters using inverse softplus from desired amplitudes
        raw_a0 = stable_inverse_softplus(np.maximum(amps0, 1e-6))  # Avoid log(0)
        self.raw_a = nn.Parameter(
            torch.tensor(raw_a0, dtype=torch.float32, device=device)
        )

        # ---- Cached tensors for vectorized _build_L (avoids Python loops) ----
        # register_buffer with persistent=False: moves with .to(), not in state_dict
        self.register_buffer(
            "_diag_idx", torch.arange(d, device=device), persistent=False
        )
        if d > 1:
            tril = torch.tril_indices(d, d, offset=-1, device=device)
            self.register_buffer("_tril_rows", tril[0], persistent=False)
            self.register_buffer("_tril_cols", tril[1], persistent=False)
        else:
            self.register_buffer(
                "_tril_rows",
                torch.empty(0, dtype=torch.long, device=device),
                persistent=False,
            )
            self.register_buffer(
                "_tril_cols",
                torch.empty(0, dtype=torch.long, device=device),
                persistent=False,
            )
        # Cache shape as float32 tensor for current_params
        self.register_buffer(
            "_shape_f32",
            torch.tensor(self.shape, dtype=torch.float32, device=device),
            persistent=False,
        )

    def _build_L(self) -> torch.Tensor:
        """
        Reconstruct lower-triangular Cholesky factors from learnable parameters.

        Combines constrained diagonal elements with free off-diagonal elements to
        form valid lower-triangular matrices for covariance parameterization.

        Uses vectorized indexing (no Python loops) for performance.

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

        # Apply eccentricity constraint on diagonal if specified
        # Constrain diagonal ratio: max(diag)/min(diag) <= sqrt(max_eccentricity)
        # When voxel_size is set, eccentricity is evaluated in physical space.
        if self.max_eccentricity is not None:
            max_ratio = float(self.max_eccentricity) ** 0.5
            if self.voxel_size is not None:
                diag_phys = diag * self.voxel_size  # (N, d)
                min_phys = diag_phys.min(dim=1, keepdim=True).values  # (N, 1)
                max_allowed_vox = (min_phys * max_ratio) / self.voxel_size  # (N, d)
                diag = torch.minimum(diag, max_allowed_vox)
                # Re-enforce sigma_min_diag: physical-space clamping can push
                # voxel-space diag below the floor for high-voxel-size axes
                diag = torch.maximum(diag, self.sigma_min_diag)
            else:
                min_diag = diag.min(dim=1, keepdim=True).values  # (N, 1)
                max_allowed_diag = min_diag * max_ratio
                diag = torch.minimum(diag, max_allowed_diag)

        # Build L matrix using vectorized indexing (no Python loops)
        # Use diag.dtype to support both FP32 (training) and FP16 (inference)
        L = torch.zeros((N, d, d), dtype=diag.dtype, device=diag.device)

        # Fill all diagonal elements at once
        L[:, self._diag_idx, self._diag_idx] = diag

        # Fill off-diagonal elements below diagonal (vectorized)
        if self.L_off.shape[1] > 0:
            off_val: torch.Tensor = self.L_off  # (N, num_off)

            # Apply eccentricity constraint on off-diagonals to prevent elongation
            # |L[i,j]| <= γ * min(L[i,i], L[j,j])
            if self.max_eccentricity is not None and d > 1:
                E = float(self.max_eccentricity)
                sqrt_E_minus_1 = (E - 1.0) ** 0.5
                k_2d = 2.4 + 0.5 / sqrt_E_minus_1 if sqrt_E_minus_1 > 0 else 3.0
                off_gamma = sqrt_E_minus_1 / (k_2d * (d - 1) ** 0.7)

                rows = self._tril_rows  # (num_off,)
                cols = self._tril_cols  # (num_off,)

                if self.voxel_size is not None:
                    # Physical-space constraint:
                    # |vs[i]*L[i,j]| <= γ * min(vs[i]*L[i,i], vs[j]*L[j,j])
                    min_phys_ij = torch.minimum(
                        diag[:, rows] * self.voxel_size[rows],
                        diag[:, cols] * self.voxel_size[cols],
                    )  # (N, num_off)
                    max_off = off_gamma * min_phys_ij / self.voxel_size[rows]
                else:
                    min_diag_ij = torch.minimum(
                        diag[:, rows], diag[:, cols]
                    )  # (N, num_off)
                    max_off = off_gamma * min_diag_ij

                off_val = torch.clamp(off_val, min=-max_off, max=max_off)

            L[:, self._tril_rows, self._tril_cols] = off_val

        return L

    def current_params(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        centers = u * torch.clamp(
            self._shape_f32 - 1.0, min=1.0
        )  # Maps [0,1] -> [0, shape-1]

        # Reconstruct Cholesky factors and apply amplitude transformation
        L = self._build_L()
        amps = F.softplus(self.raw_a)  # Ensures non-negative amplitudes

        # Apply maximum amplitude constraint if specified (prevents explosion during optimization)
        if self.amp_max is not None:
            amps = torch.clamp(amps, max=self.amp_max)

        return centers, L, amps

    # ===== Dynamic Management Methods =========================================

    @torch.no_grad()
    def _to_internal_params(
        self,
        centers: torch.Tensor,  # (N,d)
        Ls: torch.Tensor,  # (N,d,d)
        amps: torch.Tensor,  # (N,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert external (μ, L, a) to raw learnable params (raw_mu, L_diag_raw, L_off, amp_raw)."""
        device = self.raw_mu.device
        d = centers.shape[1]

        # centers -> raw_mu (logit in [0,1] coords)
        shape_arr = torch.tensor(self.shape, device=device, dtype=torch.float32)
        u = torch.clamp(
            centers / torch.clamp(shape_arr - 1.0, min=1.0), 1e-6, 1.0 - 1e-6
        )
        raw_mu = torch.log(u) - torch.log(1.0 - u)

        # L -> diag/off raw (diag via inverse-softplus, GPU-only)
        diag = torch.diagonal(Ls, dim1=1, dim2=2)  # (N,d)
        # Subtract sigma_min_diag before inverse softplus (matches _build_L: sigma_min + softplus(raw))
        eps = 1e-6
        diag_shifted = torch.clamp(diag - self.sigma_min_diag, min=eps)
        L_diag_raw = stable_inverse_softplus_torch(diag_shifted)

        # Pack off-diagonals (row-major, below diag) — vectorized
        if d > 1:
            L_off = Ls[:, self._tril_rows, self._tril_cols]  # (N, num_off)
        else:
            L_off = torch.zeros((centers.shape[0], 0), device=device)

        # amps -> amp_raw (GPU-only, no CPU roundtrip)
        amp_raw = stable_inverse_softplus_torch(torch.clamp(amps, min=1e-6))

        return raw_mu, L_diag_raw, L_off, amp_raw

    @torch.no_grad()
    def replace_with(
        self,
        centers: torch.Tensor,
        Ls: torch.Tensor,
        amps: torch.Tensor,
    ) -> None:
        """Hard-replace the whole parameter set in-place.

        .. warning::
            Reassigning ``nn.Parameter`` attributes invalidates any optimizer
            state (Adam moments, momentum buffers, etc.) registered against
            the *previous* parameter tensors. Callers that intend to keep
            training after a ``replace_with`` must rebuild the optimizer —
            ``initialize_optimization`` is the canonical entry point.

            The current best-state restore path
            (``optimization.py::_restore_best_state``) is safe because it
            runs purely under ``torch.no_grad()`` and does NOT call
            ``optimizer.step()`` afterwards: it only re-evaluates the loss
            so the reported metrics match the restored parameters.
        """
        raw_mu, L_diag_raw, L_off, amp_raw = self._to_internal_params(centers, Ls, amps)
        self.raw_mu = torch.nn.Parameter(raw_mu)
        self.raw_L_diag = torch.nn.Parameter(L_diag_raw)
        self.L_off = torch.nn.Parameter(L_off)
        self.raw_a = torch.nn.Parameter(amp_raw)

    @torch.no_grad()
    def prune_(self, keep_mask: torch.Tensor) -> None:
        """Keep only indices where keep_mask is True."""
        self.raw_mu = torch.nn.Parameter(self.raw_mu[keep_mask])
        self.raw_L_diag = torch.nn.Parameter(self.raw_L_diag[keep_mask])
        self.L_off = torch.nn.Parameter(self.L_off[keep_mask])
        self.raw_a = torch.nn.Parameter(self.raw_a[keep_mask])

    @torch.no_grad()
    def append_(
        self,
        centers_new: torch.Tensor,
        Ls_new: torch.Tensor,
        amps_new: torch.Tensor,
    ) -> None:
        """Append new splats to the tail."""
        if centers_new.numel() == 0:
            return
        raw_mu, L_diag_raw, L_off, amp_raw = self._to_internal_params(
            centers_new, Ls_new, amps_new
        )
        self.raw_mu = torch.nn.Parameter(torch.cat([self.raw_mu, raw_mu], dim=0))
        self.raw_L_diag = torch.nn.Parameter(
            torch.cat([self.raw_L_diag, L_diag_raw], dim=0)
        )
        self.L_off = torch.nn.Parameter(torch.cat([self.L_off, L_off], dim=0))
        self.raw_a = torch.nn.Parameter(torch.cat([self.raw_a, amp_raw], dim=0))

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
        Avoids explicit Sigma^{-1} by solving L y = (x-mu) and using ||y||^2.
        Standard Gaussian falloff: exp(-0.5 * ||y||^2).
        """
        centers, Ls, amps = self.current_params()

        return render_gaussians(
            self.shape,
            centers,
            Ls,
            amps,
            truncate=self.truncate,
            intensity_floor=1e-5,
        )
