"""
Optimized Precision Matrix Gaussian Splat Fitting

This module implements the TRUE precision-only optimization pipeline:
- Uses upper triangular Cholesky U where Λ = U^T @ U
- Rendering via triangular matmul: ||U(x-μ)||²  
- AABB via triangular solves: r_i = t * ||U^{-T} e_i||
- No matrix inversions anywhere in the pipeline

This provides maximum numerical stability and computational efficiency.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from arbol import aprint, asection

from luxar.gsplats.models.gsplats.gsplat_precision_model_optimized import (
    GaussianSplatPrecisionModelOptimized,
)
from luxar.gsplats.utils.trils import pack_tril, tril_size


class GaussianSplatPrecisionFitterOptimized:
    """
    Optimized precision matrix Gaussian splat fitter using triangular operations.
    
    This fitter implements the true precision-only approach with:
    - Upper triangular Cholesky parameterization
    - Triangular matrix operations for rendering  
    - Triangular solves for AABB computation
    - No matrix inversions needed anywhere
    
    This provides maximum performance and numerical stability.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        compile_model: bool = False,
        use_mixed_precision: bool = False,
        output_format: str = 'covariance',  # For backward compatibility
        max_aspect_ratio: Optional[float] = 2.5,  # Constrain anisotropy (reasonable ellipses)
    ):
        # Auto-detect best performing device: CUDA → CPU → MPS
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.compile_model = compile_model
        self.use_mixed_precision = use_mixed_precision
        self.output_format = output_format  # 'precision' or 'covariance'
        self.max_aspect_ratio = max_aspect_ratio
        
        # Only create scaler if both mixed precision requested AND CUDA available
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if use_mixed_precision and self.device.type == "cuda"
            else None
        )

    def _detect_convergence(self, losses: list, patience: int, threshold: float) -> bool:
        """Detect convergence from loss history."""
        if len(losses) < patience + 5:
            return False

        recent = losses[-patience:]
        older = losses[-patience-5:-5]
        
        if len(older) == 0:
            return False

        avg_recent = float(np.mean(recent))
        avg_older = float(np.mean(older))
        
        if avg_older == 0.0:
            return avg_recent < threshold
        
        rel_improvement = abs(avg_older - avg_recent) / avg_older
        variance_norm = float(np.var(recent)) / max(float(avg_recent**2), 1e-6)
        
        converged = bool(rel_improvement < threshold and variance_norm < threshold)
        return converged

    def fit(
        self,
        V: np.ndarray,
        centers_overcomplete: np.ndarray,
        init_sigma_vox: float = 1.5,  # Initial Gaussian width (same as traditional)
        n_iters: int = 300,
        lr: float = 0.2,
        loss_type: str = "mse",
        l1_amp: float = 0.0,
        precision_min_diag: Optional[Sequence[float]] = None,
        precision_max_diag: Optional[Sequence[float]] = None,
        truncate: float = 3.0,
        verbose: bool = True,
        early_stopping: bool = True,
        early_stop_patience: int = 20,
        convergence_threshold: float = 1e-3,
        gradient_clip: Optional[float] = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fit Gaussian splats using optimized precision matrix parameterization.
        
        This implementation uses the true precision-only approach with triangular
        operations for maximum performance and numerical stability.
        """
        import time
        start_time = time.time()

        if verbose:
            with asection("🔥 Optimized Precision Matrix Fitting"):
                aprint("Using triangular operations only - no matrix inversions!")

        # Input validation
        V = np.asarray(V, dtype=np.float32)
        if V.ndim < 1:
            raise ValueError("Input image V must have at least 1 dimension")

        centers_overcomplete = np.asarray(centers_overcomplete, dtype=np.float32)
        if centers_overcomplete.ndim != 2:
            raise ValueError("centers_overcomplete must be a 2D array")
        if centers_overcomplete.shape[1] != V.ndim:
            raise ValueError(
                f"centers_overcomplete must have {V.ndim} columns to match image dimensions"
            )

        # Convert covariance initialization to precision Cholesky  
        init_precision_diag = 1.0 / init_sigma_vox if init_sigma_vox > 0 else 1.0
        
        # Validate hyperparameters
        if init_sigma_vox <= 0:
            raise ValueError("init_sigma_vox must be positive")
        if n_iters <= 0:
            raise ValueError("n_iters must be positive")

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
            return np.zeros((0, d + tril_size(d)), np.float32), np.zeros((0,), np.float32), {}

        # Initialize UPPER triangular precision matrix Cholesky factors
        U0_precision = np.zeros((N, d, d), dtype=np.float32)
        for i in range(d):
            U0_precision[:, i, i] = init_precision_diag  # Diagonal only

        # Initialize amplitudes from image intensities
        idx = np.clip(np.round(centers_overcomplete).astype(int), 0, np.array(V.shape) - 1)
        amps0 = V[tuple(idx.T)]

        if precision_min_diag is None:
            precision_min_diag = [0.01] * d  # Very small minimum precision (allows large spread)
        else:
            if len(precision_min_diag) != d:
                raise ValueError(f"precision_min_diag must have length {d}")
            if any(s <= 0 for s in precision_min_diag):
                raise ValueError("All precision_min_diag values must be positive")

        if precision_max_diag is not None:
            if len(precision_max_diag) != d:
                raise ValueError(f"precision_max_diag must have length {d}")
            if any(s <= 0 for s in precision_max_diag):
                raise ValueError("All precision_max_diag values must be positive")

        # Move to device
        V_t = torch.tensor(V, dtype=torch.float32, device=self.device)

        # Build optimized precision model
        model = GaussianSplatPrecisionModelOptimized(
            shape=V.shape,
            centers0=centers_overcomplete,
            U0=U0_precision,
            amps0=amps0,
            precision_min_diag=precision_min_diag,
            precision_max_diag=precision_max_diag,
            max_aspect_ratio=self.max_aspect_ratio,
            truncate=truncate,
            device=self.device,
        )

        # Compile if requested (CUDA only)
        if self.compile_model and self.device.type == "cuda":
            try:
                model.forward = torch.compile(model.forward, mode="default")
                if verbose:
                    aprint("✅ Model compiled with torch.compile")
            except Exception as e:
                if verbose:
                    aprint(f"⚠️  torch.compile failed: {e}")

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        # Training loop with early stopping
        loss_history = []
        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for iteration in range(n_iters):
            optimizer.zero_grad()

            if self.scaler is not None:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    pred = model()
                    
                    if loss_type == "mse":
                        loss = F.mse_loss(pred, V_t)
                    elif loss_type == "poisson":
                        loss = F.poisson_nll_loss(pred, V_t, log_input=False, reduction='mean')
                    else:
                        raise ValueError(f"Unknown loss_type: {loss_type}")
                    
                    if l1_amp > 0:
                        _, _, amps = model.current_params()
                        loss += l1_amp * torch.mean(torch.abs(amps))

                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                if gradient_clip is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # Regular precision
                pred = model()
                
                if loss_type == "mse":
                    loss = F.mse_loss(pred, V_t)
                elif loss_type == "poisson":
                    loss = F.poisson_nll_loss(pred, V_t, log_input=False, reduction='mean')
                else:
                    raise ValueError(f"Unknown loss_type: {loss_type}")
                
                if l1_amp > 0:
                    _, _, amps = model.current_params()
                    loss += l1_amp * torch.mean(torch.abs(amps))

                loss.backward()
                
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                optimizer.step()

            loss_val = float(loss.item())
            loss_history.append(loss_val)
            scheduler.step(loss_val)

            # Track best model
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
                aprint(f"[{iteration + 1}/{n_iters}] loss={loss_val:.5g} (best)  N={N}")
            else:
                patience_counter += 1


            # Early stopping check
            if early_stopping:
                converged = self._detect_convergence(
                    loss_history, early_stop_patience, convergence_threshold
                )
                if converged:
                    if verbose:
                        aprint(f"✅ Converged after {iteration + 1} iterations")
                    break

        # Restore best model
        if best_state is not None:
            model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # Extract final parameters
        with torch.no_grad():
            centers, Us_precision, amps = model.current_params()
            centers_np = centers.cpu().numpy()
            amps_np = amps.cpu().numpy()

            if self.output_format == 'covariance':
                # Convert precision Cholesky to covariance Cholesky for compatibility
                # Λ = U^T @ U (precision matrix)
                # Σ = Λ^{-1} (covariance matrix)  
                # L_cov such that Σ = L_cov @ L_cov^T
                
                # Get covariance matrices
                Sigma = model.get_covariance_matrices()  # (N, d, d)
                
                # Compute Cholesky of covariance matrices  
                Ls_covariance = torch.linalg.cholesky(Sigma)  # (N, d, d)
                Ls_np = Ls_covariance.cpu().numpy()
                
                if verbose:
                    aprint("Converted precision parameters to covariance format")
            else:
                # Keep precision format (but convert U to lower triangular for compatibility)
                # Convert upper triangular U to lower triangular L where M = L @ L^T = U^T @ U
                Us_np = Us_precision.cpu().numpy()
                # L = U^T converts upper triangular to lower triangular
                Ls_np = np.transpose(Us_np, (0, 2, 1))

        # Pack into traditional format: [centers, packed_triangular]
        params_full = np.zeros((len(centers_np), d + tril_size(d)), dtype=np.float32)
        params_full[:, :d] = centers_np
        
        # Pack lower triangular matrices (pack_tril expects batch format)
        packed_tril = pack_tril(Ls_np)  # (N, tril_size)
        params_full[:, d:] = packed_tril

        # Compute final statistics
        end_time = time.time()
        stats = {
            'iterations': len(loss_history),
            'converged': early_stopping and len(loss_history) < n_iters,
            'final_loss': float(best_loss),
            'time_seconds': end_time - start_time,
            'device_used': str(self.device),
            'optimization_approach': 'precision_triangular_optimized',
            'loss_history': loss_history,
        }

        return params_full.astype(np.float32), amps_np.astype(np.float32), stats


def fit_gaussian_splats_precision_optimized(
    V: np.ndarray,
    centers_overcomplete: np.ndarray,
    init_sigma_vox: float = 1.5,
    n_iters: int = 300,
    lr: float = 0.2,
    loss_type: str = "mse",
    l1_amp: float = 0.001,
    precision_min_diag: Optional[Sequence[float]] = None,
    precision_max_diag: Optional[Sequence[float]] = None,
    truncate: float = 3.0,
    device: Optional[str] = None,
    verbose: bool = True,
    # Optimization parameters
    early_stopping: bool = True,
    early_stop_patience: int = 50,
    convergence_threshold: float = 1e-6,
    gradient_clip: Optional[float] = 1.0,
    compile_model: bool = False,
    use_mixed_precision: bool = False,
    output_format: str = 'covariance',
    max_aspect_ratio: Optional[float] = 2.5,  # Constrain anisotropy (reasonable ellipses)
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Fit Gaussian splats using optimized precision matrix parameterization.
    
    This is the high-performance entry point that uses triangular operations
    exclusively for maximum speed and numerical stability.
    
    Uses the true precision-only approach:
    - Rendering: g(x) = a * exp(-0.5 * ||U(x-μ)||²) via triangular matmul
    - AABB: tight bounds r_i = t * ||U^{-T} e_i|| via triangular solves  
    - No matrix inversions in the critical path
    
    Parameters are same as traditional approach for drop-in compatibility.
    """
    
    # Create optimized fitter
    fitter = GaussianSplatPrecisionFitterOptimized(
        device=device,
        compile_model=compile_model,
        use_mixed_precision=use_mixed_precision,
        output_format=output_format,
        max_aspect_ratio=max_aspect_ratio,
    )

    # Fit and extract results
    params, amps, stats = fitter.fit(
        V=V,
        centers_overcomplete=centers_overcomplete,
        init_sigma_vox=init_sigma_vox,
        n_iters=n_iters,
        lr=lr,
        loss_type=loss_type,
        l1_amp=l1_amp,
        precision_min_diag=precision_min_diag,
        precision_max_diag=precision_max_diag,
        truncate=truncate,
        verbose=verbose,
        early_stopping=early_stopping,
        early_stop_patience=early_stop_patience,
        convergence_threshold=convergence_threshold,
        gradient_clip=gradient_clip,
    )

    return params, amps, stats