# fit_gsplats.py

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from arbol import aprint, asection

from luxar.gsplats.dynamic_ops import DynamicOpsConfig, apply_dynamic_operations
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.optim import create_per_splat_optimizer_setup
from luxar.gsplats.utils.trils import pack_tril, tril_size


class GaussianSplatFitter:
    """
    Advanced Gaussian splat fitter with per-splat optimizer.

    This class uses the per-splat Adam optimizer to maintain momentum
    for individual splats during dynamic operations, providing smooth
    optimization without global disruption.

    Parameters
    ----------
    device : str, optional
        PyTorch device ('cpu', 'cuda', 'mps'). Auto-detects if None.
    enable_dynamic_ops : bool, default=False
        Enable dynamic operations (seeding, splitting, pruning).
    dynamic_config : DynamicOpsConfig, optional
        Configuration for dynamic operations.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        enable_dynamic_ops: bool = False,
        dynamic_config: Optional[DynamicOpsConfig] = None,
    ):
        # Auto-detect best performing device: CUDA → CPU
        # Note: MPS is supported but currently slower than CPU for typical workloads
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Dynamic operations configuration
        self.enable_dynamic_ops = enable_dynamic_ops
        self.dynamic_config = dynamic_config or DynamicOpsConfig()


    def fit(
        self,
        V: np.ndarray,
        centers_overcomplete: np.ndarray,
        init_sigma_vox: float = 1.5,
        n_iters: int = 1000,
        lr: float = 0.2,
        loss_type: str = "mse",
        asymmetric_penalty: Optional[float] = 10.0,
        l1_amp: float = 0.0,
        sigma_min_diag: Optional[Sequence[float]] = None,
        sigma_max_diag: Optional[Sequence[float]] = None,
        truncate: float = 3.0,
        verbose: bool = True,
        max_abs_error: Optional[float] = None,
        gradient_clip: Optional[float] = 1.0,
        napari_movie: bool = False,
        movie_every: int = 1,
        movie_max_frames: Optional[int] = None,
        scheduler_type: str = "plateau",
        patience: int = 10,
        factor: float = 0.5,
        dynamic_ops_verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fit Gaussian splats using per-splat Adam optimizer.

        See fit_gaussian_splats() for full parameter documentation.

        Returns
        -------
        params_full : np.ndarray
            Splat parameters [centers, packed_L].
        amps : np.ndarray
            Splat amplitudes.
        stats : dict
            Optimization statistics (time, iterations, convergence).
        """
        start_time = time.time()

        # Movie frame limit
        if movie_max_frames is None:
            movie_max_frames = float('inf')  # No limit

        # Input validation and normalization
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
        if max_abs_error is not None and max_abs_error <= 0:
            raise ValueError("max_abs_error must be positive if specified")
        elif movie_max_frames <= 0:
            raise ValueError("movie_max_frames must be positive or None")

        # Robust normalization
        image_min = np.percentile(V, 1)
        image_max = np.percentile(V, 99)

        if np.abs(image_max - image_min) < 1e-12:
            V = np.full_like(V, 0.5, dtype=np.float32)
            if verbose:
                aprint("Warning: Input image is nearly uniform")
        else:
            V = np.clip((V - image_min) / (image_max - image_min), 0.0, 1.0)

        d = V.ndim
        N = int(centers_overcomplete.shape[0])

        if N == 0:
            return (
                np.zeros((0, d + tril_size(d)), np.float32),
                np.zeros((0,), np.float32),
                {},
            )

        # Initialize parameters
        L0 = np.zeros((N, d, d), dtype=np.float32)
        for i in range(d):
            L0[:, i, i] = init_sigma_vox

        idx = np.clip(
            np.round(centers_overcomplete).astype(int), 0, np.array(V.shape) - 1
        )
        amps0 = V[tuple(idx.T)]

        if sigma_min_diag is None:
            sigma_min_diag = [0.5] * d
        else:
            if len(sigma_min_diag) != d:
                raise ValueError(f"sigma_min_diag must have length {d}")
            if any(s <= 0 for s in sigma_min_diag):
                raise ValueError("All sigma_min_diag values must be positive")

        if sigma_max_diag is not None:
            if len(sigma_max_diag) != d:
                raise ValueError(f"sigma_max_diag must have length {d}")
            if any(s <= 0 for s in sigma_max_diag):
                raise ValueError("All sigma_max_diag values must be positive")
            if any(
                s_max <= s_min for s_max, s_min in zip(sigma_max_diag, sigma_min_diag)
            ):
                raise ValueError("sigma_max_diag must be greater than sigma_min_diag")

        # Move to device
        V_t = torch.tensor(V, dtype=torch.float32, device=self.device)

        # Build model
        model = GaussianSplatModel(
            shape=V.shape,
            centers0=centers_overcomplete,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            sigma_max_diag=sigma_max_diag,
            truncate=truncate,
            device=self.device,
        )

        # Setup per-splat optimizer
        opt, scheduler, coordinator = create_per_splat_optimizer_setup(
            model,
            lr=lr,
            scheduler_type=scheduler_type,
            patience=patience,
            factor=factor,
        )

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

                # Apply asymmetric penalty if specified
                if asymmetric_penalty is not None:
                    over_prediction_mask = pred > V_t
                    # Compute additional penalty for over-prediction regions only
                    # This penalizes regions where we predict more intensity than target
                    over_prediction_dev = 2.0 * torch.sum(
                        over_prediction_mask * (Pc - Vc + Vc * torch.log(torch.clamp(Vc / Pc, min=eps)))
                    )
                    # Add (F-1) times the over-prediction loss to get total F times penalty
                    data = data + (asymmetric_penalty - 1.0) * over_prediction_dev / V_t.numel()
            else:
                # MSE loss
                squared_error = (pred - V_t) ** 2
                if asymmetric_penalty is not None:
                    # Asymmetric MSE: heavily penalize over-prediction (pred > target)
                    # This addresses the fundamental asymmetry in additive Gaussian models:
                    # - Under-prediction (pred < target): Easy to fix by adding more Gaussians
                    # - Over-prediction (pred > target): Hard to fix, requires reducing/moving splats
                    over_prediction_mask = pred > V_t
                    data = torch.mean(
                        torch.where(over_prediction_mask,
                                   asymmetric_penalty * squared_error,  # F times penalty
                                   squared_error)  # Normal penalty
                    )
                else:
                    data = F.mse_loss(pred, V_t)

            if l1_amp > 0:
                # Use raw parameters directly to avoid rebuilding L matrices and centers
                data = data + l1_amp * torch.mean(torch.abs(F.softplus(model.raw_a)))
            return data

        # Tracking
        best_loss = float("inf")

        # Movie recording setup (only if enabled)
        movie_frames = None
        if napari_movie:
            movie_frames = {
                "target": [],
                "reconstruction": [],
                "residual": [],
                "iterations": [],
                "splat_centers": [],
            }

        # Main optimization loop
        actual_iters = 0
        for it in range(1, n_iters + 1):
            actual_iters = it

            # Forward pass with per-splat optimizer
            opt.zero_grad()
            pred = model()
            loss = loss_fn(pred)
            loss.backward()

            # Gradient clipping for stability
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            opt.step()

            # Learning rate scheduling (detach to avoid warning)
            scheduler.step(loss.detach())

            # Tracking
            current_loss = loss.item()

            # Movie frame recording (only if enabled and at specified intervals)
            if napari_movie and movie_frames is not None and it % movie_every == 0:
                with torch.no_grad():
                    # Memory-bounded recording: remove oldest frames if limit exceeded
                    if len(movie_frames["target"]) >= movie_max_frames:
                        # Remove oldest frame (FIFO)
                        for key in ["target", "reconstruction", "residual", "splat_centers", "iterations"]:
                            movie_frames[key].pop(0)

                    # Store frames as numpy arrays (detached from computation graph)
                    target_frame = V_t.cpu().numpy()
                    pred_frame = pred.detach().cpu().numpy()
                    residual_frame = torch.abs(V_t - pred.detach()).cpu().numpy()

                    # Record current splat centers
                    centers, _, _ = model.current_params()
                    centers_frame = centers.detach().cpu().numpy()

                    movie_frames["target"].append(target_frame)
                    movie_frames["reconstruction"].append(pred_frame)
                    movie_frames["residual"].append(residual_frame)
                    movie_frames["splat_centers"].append(centers_frame)
                    movie_frames["iterations"].append(it)

            # Update best loss tracking
            if current_loss < best_loss:
                best_loss = current_loss

            # Convergence check using maximum absolute error
            if max_abs_error is not None:
                with torch.no_grad():
                    current_max_abs_error = torch.max(torch.abs(pred - V_t)).item()
                    if current_max_abs_error < max_abs_error:
                        if verbose:
                            aprint(f"Converged at iteration {it} (max_abs_error={current_max_abs_error:.6f} < {max_abs_error})")
                        break

            # Dynamic operations (seeding, splitting, pruning)
            if self.enable_dynamic_ops and it % self.dynamic_config.step_every == 0:
                opt, scheduler, topology_changed = apply_dynamic_operations(
                    model,
                    opt,
                    scheduler,
                    V_t,  # target
                    pred,  # current prediction
                    self.dynamic_config,
                    lr,
                    max_abs_error_threshold=max_abs_error or float('inf'),
                    device=self.device,
                    verbose=dynamic_ops_verbose,
                )

            # Logging (update N after potential dynamic ops)
            N = model.n_splats() if hasattr(model, "n_splats") else N
            if verbose and (it % max(1, n_iters // 10) == 0 or it <= 5):
                with torch.no_grad():
                    rel = torch.linalg.norm((pred - V_t).reshape(-1)) / (
                        torch.linalg.norm(V_t.reshape(-1)) + 1e-12
                    )
                    # Calculate max absolute error for display
                    current_max_abs_error = torch.max(torch.abs(pred - V_t)).item()
                aprint(
                    f"[{it:4d}/{n_iters}] loss={current_loss:.5g}  "
                    f"relL2={float(rel):.4f}  maxAbsErr={current_max_abs_error:.5g}  N={N}"
                )

        # No need to restore state with per-splat optimizer

        # Extract parameters
        with torch.no_grad():
            centers, Ls, amps = model.current_params()
            centers_np = centers.cpu().numpy()
            Ls_np = Ls.cpu().numpy()
            amps_np = amps.cpu().numpy()
            params_full = np.concatenate([centers_np, pack_tril(Ls_np)], axis=1)

        # Compute statistics
        end_time = time.time()
        stats = {
            "time_seconds": end_time - start_time,
            "iterations": actual_iters,
            "final_loss": best_loss,
            "converged": actual_iters < n_iters,
            "n_splats": N,
        }

        # Show optimization movie (only if enabled and frames were recorded)
        if (
            napari_movie
            and movie_frames is not None
            and len(movie_frames["target"]) > 0
        ):
            _show_optimization_movie(movie_frames, V.shape)

        # Calculate and display compression ratio
        if verbose:
            _display_compression_analysis(V, params_full, amps_np)

        return params_full.astype(np.float32), amps_np.astype(np.float32), stats


def fit_gaussian_splats(
    V: np.ndarray,
    centers_overcomplete: np.ndarray,
    init_sigma_vox: float = 1.5,
    n_iters: int = 1000,
    lr: float = 0.2,
    loss_type: str = "mse",
    asymmetric_penalty: Optional[float] = 10.0,
    l1_amp: float = 0.0,
    sigma_min_diag: Optional[Sequence[float]] = None,
    sigma_max_diag: Optional[Sequence[float]] = None,
    truncate: float = 3.0,
    device: Optional[str] = None,
    verbose: bool = True,
    # Optimization parameters
    max_abs_error: Optional[float] = None,
    gradient_clip: Optional[float] = 1.0,
    # Per-splat optimizer parameters
    scheduler_type: str = "plateau",
    patience: int = 10,
    factor: float = 0.9,
    # Dynamic operations parameters
    enable_dynamic_ops: bool = True,
    dynamic_config: Optional[DynamicOpsConfig] = None,
    dynamic_ops_verbose: bool = False,
    napari_movie: bool = True,
    movie_every: int = 1,
    movie_max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit n-dimensional oriented Gaussian splats to reconstruct input image/volume.

    This function uses the per-splat Adam optimizer to maintain momentum
    for individual splats during dynamic operations, providing smooth
    optimization without global disruption.

    The optimization uses:
    - Per-splat Adam optimizer with individual learning rates
    - Center position (bounded to image domain via sigmoid)
    - Non-negative amplitude (via softplus activation)
    - Covariance matrix Σ = L @ L^T where L is the Cholesky factor
    - Efficient rendering via batched triangular solve

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image/volume to reconstruct. Will be normalized to [0,1].
    centers_overcomplete : np.ndarray, shape (N, d)
        Initial candidate center positions in voxel coordinates (float).
        Typically from find_candidates_overcomplete_nd().
    init_sigma_vox : float, default=1.5
        Initial isotropic standard deviation for Gaussian splats (in voxels).
    n_iters : int, default=1000
        Maximum number of optimization iterations. Default is generous to allow 
        max_abs_error convergence criterion to work effectively.
    lr : float, default=0.2
        Learning rate for Adam optimizer.
    loss_type : str, default="mse"
        Loss function: "mse" or "poisson" (better for count/photon data).
    asymmetric_penalty : float, default=10.0
        Over-prediction penalty factor for asymmetric loss. Multiplies loss for regions
        where pred > target by this factor. Set to None to disable asymmetric loss.
        Default 10.0 heavily penalizes over-prediction since non-negative Gaussian sums
        cannot easily reduce intensity, making under-prediction easier to correct.
    l1_amp : float, default=0.0
        L1 regularization coefficient on splat amplitudes for sparsity.
    sigma_min_diag : Sequence[float], optional
        Minimum diagonal values for Cholesky factor L along each axis.
        Defaults to [0.5]*d to prevent degenerate splats.
    sigma_max_diag : Sequence[float], optional
        Maximum diagonal values for Cholesky factor L along each axis.
    truncate : float, default=3.0
        Truncation radius in standard deviations for rendering efficiency.
    device : str, optional
        PyTorch device ("cpu", "cuda", "mps"). Auto-detects if None.
    verbose : bool, default=True
        Whether to print optimization progress.
    max_abs_error : float or None, default=None
        Maximum absolute error threshold for convergence. If specified, 
        optimization stops when max(|prediction - target|) < max_abs_error.
        If None, only n_iters limit applies.
    gradient_clip : float or None, default=1.0
        Maximum gradient norm for clipping. None disables clipping.
    scheduler_type : str, default="plateau"
        Type of learning rate scheduler ("plateau" or "exponential").
    patience : int, default=10
        Scheduler patience for plateau scheduler.
    factor : float, default=0.5
        Learning rate reduction factor for scheduler.
    enable_dynamic_ops : bool, default=True
        Enable dynamic operations (seeding, splitting, pruning).
    dynamic_config : DynamicOpsConfig, optional
        Configuration for dynamic operations. Uses defaults if None.
    dynamic_ops_verbose : bool, default=False
        Enable detailed console logging for dynamic operations. Shows residual analysis,
        seeding attempts, splitting decisions, and pruning operations.
    napari_movie : bool, default=True
        Record optimization movie for napari visualization.
    movie_every : int, default=1
        Record movie frame every N iterations.
    movie_max_frames : int, default=None (infinite)
        Maximum number of movie frames to store in memory. Older frames are automatically
        removed when this limit is exceeded, preventing memory exhaustion during long optimizations.

    Returns
    -------
    params_full : np.ndarray, shape (N, d + d*(d+1)//2), dtype=float32
        Concatenated parameters for each splat: [center_coords, packed_cholesky_L].
    amps : np.ndarray, shape (N,), dtype=float32
        Non-negative amplitude values for each splat.
    stats : dict
        Optimization statistics including time, iterations, convergence status.

    Notes
    -----
    The optimization uses per-splat Adam optimizer which provides:
    - Individual learning rates per splat
    - Momentum preservation during dynamic operations
    - Smooth optimization trajectory without global disruption
    - Early stopping for improved efficiency
    - Adaptive learning rate scheduling

    This approach is particularly beneficial when dynamic operations
    (prune, seed, merge, split) are enabled.
    """

    with asection("Fitting Gaussian Splats"):
        # Use per-splat optimizer
        fitter = GaussianSplatFitter(
            device=device,
            enable_dynamic_ops=enable_dynamic_ops,
            dynamic_config=dynamic_config,
        )

        # Fit and extract results
        params, amps, stats = fitter.fit(
            V=V,
            centers_overcomplete=centers_overcomplete,
            init_sigma_vox=init_sigma_vox,
            n_iters=n_iters,
            lr=lr,
            loss_type=loss_type,
            asymmetric_penalty=asymmetric_penalty,
            l1_amp=l1_amp,
            dynamic_ops_verbose=dynamic_ops_verbose,
            sigma_min_diag=sigma_min_diag,
            sigma_max_diag=sigma_max_diag,
            truncate=truncate,
            verbose=verbose,
            max_abs_error=max_abs_error,
            gradient_clip=gradient_clip,
            napari_movie=napari_movie,
            movie_every=movie_every,
            movie_max_frames=movie_max_frames,
            scheduler_type=scheduler_type,
            patience=patience,
            factor=factor,
        )

        if verbose:
            with asection("Optimization Complete"):
                aprint(f"Time: {stats['time_seconds']:.2f} seconds")
                aprint(f"Iterations: {stats['iterations']}/{n_iters}")
                if stats["converged"]:
                    aprint(
                        f"✓ Converged (saved {n_iters - stats['iterations']} iterations)"
                    )

        # Calculate and display compression ratio
        if verbose:
            _display_compression_analysis(V, params, amps)

        return params, amps, stats


def _display_compression_analysis(V: np.ndarray, params: np.ndarray, amps: np.ndarray):
    """
    Calculate and display compression ratio analysis.

    Compares the storage requirements of the original image vs the Gaussian splat representation.
    """
    from arbol import aprint, asection

    with asection("Compression Analysis"):
        # Original image storage (assuming float32)
        original_bytes = V.size * 4  # 4 bytes per float32
        original_bits = original_bytes * 8

        # Gaussian splat representation storage
        # params contains: centers (d floats) + packed L matrix (tril_size(d) floats)
        # amps contains: amplitudes (1 float per splat)
        n_splats = len(amps)
        d = len(V.shape)

        from luxar.gsplats.utils.trils import tril_size
        floats_per_splat = d + tril_size(d) + 1  # centers + covariance + amplitude
        splat_bytes = n_splats * floats_per_splat * 4  # 4 bytes per float32
        splat_bits = splat_bytes * 8

        # Calculate compression metrics
        compression_ratio = original_bytes / splat_bytes if splat_bytes > 0 else float('inf')
        compression_percent = (1.0 - splat_bytes / original_bytes) * 100.0 if original_bytes > 0 else 0.0
        bits_per_pixel = splat_bits / V.size

        aprint(f"Original image: {original_bytes:,} bytes ({original_bits:,} bits)")
        aprint(f"Splat representation: {splat_bytes:,} bytes ({splat_bits:,} bits)")
        aprint(f"Compression ratio: {compression_ratio:.2f}:1")
        aprint(f"Space savings: {compression_percent:.1f}%")
        aprint(f"Bits per pixel: {bits_per_pixel:.3f} (original: 32.000)")
        aprint(f"Storage efficiency: {n_splats} splats ({floats_per_splat} floats each)")


def _show_optimization_movie(movie_frames, shape):
    """
    Display napari viewer with optimization movie showing target, reconstruction, and residual over time.
    """
    try:
        import napari
        import numpy as np
        from arbol import aprint

        aprint("🎬 Creating optimization movie visualization...")

        # Convert lists to 4D arrays (time, y, x) for 2D or (time, z, y, x) for 3D
        target_stack = np.array(movie_frames["target"])
        reconstruction_stack = np.array(movie_frames["reconstruction"])
        residual_stack = np.array(movie_frames["residual"])
        splat_centers_list = movie_frames["splat_centers"]
        iterations = movie_frames["iterations"]

        # Create napari viewer with time series
        viewer = napari.Viewer(title=f"Optimization Movie ({len(iterations)} frames)")

        # Add image stacks as layers
        viewer.add_image(target_stack, name="Target", colormap="magma")

        viewer.add_image(
            reconstruction_stack,
            name="Reconstruction",
            colormap="magma",
        )

        viewer.add_image(residual_stack, name="Residual", colormap="hot")

        # Add splat centers as points that change over time
        # Create a stack of points data for napari (time, n_points, n_dims)
        # Pad all frames to have the same number of points (use max)
        max_splats = max(len(centers) for centers in splat_centers_list)
        d = len(shape)

        # Create padded points array: (n_frames, max_splats, d)
        points_stack = np.full((len(splat_centers_list), max_splats, d), np.nan)
        for i, centers in enumerate(splat_centers_list):
            n_centers = len(centers)
            if n_centers > 0:
                points_stack[i, :n_centers, :] = centers

        # # Add points layer (napari will handle NaN values automatically)
        # viewer.add_points(
        #     points_stack,
        #     name="Splat Centers",
        #     size=3,
        #     face_color="cyan",
        #     border_color="white",
        #     border_width=1,
        # )

        # Set up the time slider
        viewer.dims.axis_labels = ["iteration"] + [
            f"dim_{i}" for i in range(len(shape))
        ]

        # Add text overlay with movie information
        info_text = "Optimization Movie\n"
        info_text += f"Frames: {len(iterations)}\n"
        info_text += f"Iterations: {iterations[0]} → {iterations[-1]}\n"
        info_text += f"Shape: {shape}\n\n"
        info_text += "Use time slider to scrub through optimization\n"
        info_text += "Toggle layers to compare target/reconstruction/residual"

        viewer.text_overlay.text = info_text
        viewer.text_overlay.visible = True

        aprint(
            f"🎬 Movie ready: {len(iterations)} frames from iterations {iterations[0]} to {iterations[-1]}"
        )
        aprint("Use the time slider to scrub through optimization progress!")
        aprint("Toggle layer visibility to compare target/reconstruction/residual")
        aprint("Close window to continue...")

        # Run napari - blocks until window is closed
        napari.run()

    except ImportError:
        from arbol import aprint

        aprint("⚠ napari not available for movie visualization")
    except Exception as e:
        from arbol import aprint

        aprint(f"⚠ Movie visualization error: {e}")
