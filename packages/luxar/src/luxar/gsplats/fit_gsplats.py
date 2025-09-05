# pytorch_splats_full.py
# Oriented (full-covariance) Gaussian splats in nD with PyTorch
# - Positive-definite Σ via Cholesky L (lower-triangular)
# - Centers inside volume via sigmoid parameterization
# - Amplitudes via softplus (>= 0)
# - Stable rendering using triangular solves (no cholesky_inverse required)
#
# Public API:
#   - fit_gaussian_splats_torch_full(...)
#
# Author: (you)
# License: MIT (or your choice)

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.utils.trils import pack_tril, tril_size

# -------------------------------
# Fitting / optimization (nD)
# -------------------------------


def fit_gaussian_splats(
    V: np.ndarray,
    centers_overcomplete: np.ndarray,  # (N, d) float voxel coords
    init_sigma_vox: float = 1.5,  # isotropic init σ for L = diag(sigmas)
    n_iters: int = 300,
    lr: float = 0.2,
    loss_type: str = "mse",  # "mse" or "poisson"
    l1_amp: float = 0.0,  # L1 on amplitudes for sparsity
    sigma_min_diag: Optional[Sequence[float]] = None,
    sigma_max_diag: Optional[Sequence[float]] = None,
    truncate: float = 3.0,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit n-dimensional oriented Gaussian splats to reconstruct input image/volume.

    This function optimizes a collection of oriented Gaussian splats to approximate
    the input image. Each splat is parameterized by:
    - Center position (bounded to image domain via sigmoid)
    - Full covariance matrix (via Cholesky decomposition for positive definiteness)
    - Non-negative amplitude (via softplus activation)

    The optimization uses Adam optimizer with configurable loss functions and
    regularization terms.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume to reconstruct. Will be normalized to [0,1].
    centers_overcomplete : np.ndarray, shape (N, d)
        Initial candidate center positions in voxel coordinates (float).
        Typically from find_candidates_overcomplete_nd().
    init_sigma_vox : float, default=1.5
        Initial isotropic standard deviation for Gaussian splats (in voxels).
        Used to initialize diagonal elements of Cholesky factor L.
    n_iters : int, default=300
        Number of optimization iterations.
    lr : float, default=0.2
        Learning rate for Adam optimizer.
    loss_type : str, default="mse"
        Loss function type: "mse" for mean squared error or "poisson" for
        Poisson deviance loss (better for count/photon data).
    l1_amp : float, default=0.0
        L1 regularization coefficient on splat amplitudes for sparsity.
        Higher values encourage fewer active splats.
    sigma_min_diag : Sequence[float], optional
        Minimum diagonal values for Cholesky factor L along each axis (in voxels).
        Defaults to [0.5]*d to prevent degenerate splats.
    sigma_max_diag : Sequence[float], optional
        Maximum diagonal values for Cholesky factor L along each axis (in voxels).
        If None, no upper bound is enforced.
    truncate : float, default=3.0
        Truncation radius in standard deviations. Splats are rendered only within
        truncate*sigma from their centers for computational efficiency.
    device : str, optional
        PyTorch device specification ("cpu", "cuda", "mps", etc.).
        If None, automatically selects CUDA if available, otherwise CPU.
    verbose : bool, default=True
        Whether to print optimization progress.

    Returns
    -------
    params_full : np.ndarray, shape (N, d + d*(d+1)//2), dtype=float32
        Concatenated parameters for each splat: [center_coords, packed_cholesky_L].
        Center coordinates are in voxel units.
        Packed Cholesky contains lower-triangular L in row-major order.
    amps : np.ndarray, shape (N,), dtype=float32
        Non-negative amplitude values for each splat.

    Notes
    -----
    The input image V is automatically normalized using robust percentiles (1st and 99th)
    to improve optimization stability. The covariance matrix Σ = L @ L^T where L is the
    lower triangular Cholesky factor.
    """
    # Input validation
    V = np.asarray(V, dtype=np.float32)
    if V.size == 0:
        raise ValueError("Input image V cannot be empty")
    if V.ndim == 0:
        raise ValueError("Input image V must have at least 1 dimension")

    centers_overcomplete = np.asarray(centers_overcomplete, dtype=np.float32)
    if centers_overcomplete.ndim != 2:
        raise ValueError("centers_overcomplete must be a 2D array")
    if centers_overcomplete.shape[1] != V.ndim:
        raise ValueError(
            f"centers_overcomplete must have {V.ndim} columns to match image dimensions"
        )

    # Validate hyperparameters
    if init_sigma_vox <= 0:
        raise ValueError("init_sigma_vox must be positive")
    if n_iters <= 0:
        raise ValueError("n_iters must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if loss_type not in ["mse", "poisson"]:
        raise ValueError("loss_type must be 'mse' or 'poisson'")
    if l1_amp < 0:
        raise ValueError("l1_amp must be non-negative")
    if truncate <= 0:
        raise ValueError("truncate must be positive")

    # Robust normalization using percentiles to handle outliers
    # This is more stable than min/max normalization for real data
    image_min = np.percentile(V, 1)  # 1st percentile as robust minimum
    image_max = np.percentile(V, 99)  # 99th percentile as robust maximum

    # Handle uniform images to prevent division by zero
    if np.abs(image_max - image_min) < 1e-12:
        # For uniform images, map to middle of [0,1] range
        V = np.full_like(V, 0.5, dtype=np.float32)
        if verbose:
            print("Warning: Input image is nearly uniform, using constant value 0.5")
    else:
        # Normalize image to [0, 1] range for optimization stability
        V = (V - image_min) / (image_max - image_min)
        V = np.clip(V, 0.0, 1.0)  # Ensure strict bounds

    # Extract problem dimensions
    d = V.ndim  # Number of spatial dimensions
    N = int(centers_overcomplete.shape[0])  # Number of candidate splats

    # Handle edge case of no candidates
    if N == 0:
        return np.zeros((0, d + tril_size(d)), np.float32), np.zeros((0,), np.float32)

    # Initialize Cholesky factors as isotropic covariances
    # L0 = diag(init_sigma_vox) creates diagonal matrices with equal variances
    L0 = np.zeros((N, d, d), dtype=np.float32)
    for i in range(d):
        L0[:, i, i] = init_sigma_vox  # Set diagonal elements

    # Initialize amplitudes from image intensities at candidate centers
    # Round coordinates and clamp to valid indices
    idx = np.clip(np.round(centers_overcomplete).astype(int), 0, np.array(V.shape) - 1)
    amps0 = V[tuple(idx.T)]  # Extract intensity values as initial amplitudes

    # Set minimum diagonal constraints to prevent degenerate splats
    # Default: half-voxel minimum size along each axis
    if sigma_min_diag is None:
        sigma_min_diag = [0.5] * d
    else:
        if len(sigma_min_diag) != d:
            raise ValueError(
                f"sigma_min_diag must have length {d} to match image dimensions"
            )
        if any(s <= 0 for s in sigma_min_diag):
            raise ValueError("All sigma_min_diag values must be positive")

    # Validate maximum diagonal constraints if provided
    if sigma_max_diag is not None:
        if len(sigma_max_diag) != d:
            raise ValueError(
                f"sigma_max_diag must have length {d} to match image dimensions"
            )
        if any(s <= 0 for s in sigma_max_diag):
            raise ValueError("All sigma_max_diag values must be positive")
        if any(s_max <= s_min for s_max, s_min in zip(sigma_max_diag, sigma_min_diag)):
            raise ValueError(
                "All sigma_max_diag values must be greater than corresponding sigma_min_diag values"
            )

    # Select PyTorch device (auto-detect if not specified)
    device_t = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Transfer normalized image data to selected device
    V_t = torch.tensor(V, dtype=torch.float32, device=device_t)

    # Build model:
    model = GaussianSplatModel(
        shape=V.shape,
        centers0=centers_overcomplete,
        L0=L0,
        amps0=amps0,
        sigma_min_diag=sigma_min_diag,
        sigma_max_diag=sigma_max_diag,
        truncate=truncate,
        device=device_t,
    )

    # Setup optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss function
    def loss_fn(pred: torch.Tensor) -> torch.Tensor:
        if loss_type.lower() == "poisson":
            eps = 1e-8
            Vc = torch.clamp(V_t, min=0.0)
            Pc = torch.clamp(pred, min=eps)
            dev = 2.0 * torch.sum(
                Pc - Vc + Vc * torch.log(torch.clamp(Vc / Pc, min=eps))
            )
            data = dev / V_t.numel()
        else:
            data = F.mse_loss(pred, V_t)
        if l1_amp > 0:
            _, _, a = model.current_params()
            data = data + l1_amp * torch.mean(torch.abs(a))
        return data

    # Best state tracking
    best_loss = float("inf")
    best_state = None

    # Main optimization loop
    for it in range(1, n_iters + 1):
        opt.zero_grad(set_to_none=True)
        pred = model()
        loss = loss_fn(pred)
        loss.backward()
        opt.step()

        with torch.no_grad():
            rel = torch.linalg.norm((pred - V_t).reshape(-1)) / (
                torch.linalg.norm(V_t.reshape(-1)) + 1e-12
            )
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (it % max(1, n_iters // 10) == 0 or it <= 5):
            print(
                f"[{it:4d}/{n_iters}] loss={loss.item():.5g}  relL2={float(rel):.4f}  N={N}"
            )

    # Restore best state and export params
    if best_state is not None:
        model.load_state_dict(best_state)

    # Extract params
    with torch.no_grad():
        centers, Ls, amps = model.current_params()
        centers_np = centers.cpu().numpy()
        Ls_np = Ls.cpu().numpy()
        amps_np = amps.cpu().numpy()
        params_full = np.concatenate([centers_np, pack_tril(Ls_np)], axis=1)

    return params_full.astype(np.float32), amps_np.astype(np.float32)
