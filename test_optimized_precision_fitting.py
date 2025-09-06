#!/usr/bin/env python3
"""
Test script for optimized precision-only Gaussian splatting implementation.
"""

import numpy as np
import torch
from arbol import aprint, asection

def test_optimized_precision_fitting():
    """Test the optimized precision-only fitting pipeline."""
    
    with asection("🔥 Testing Optimized Precision Fitting"):
        
        # Generate synthetic 2D data
        shape = (64, 64)
        aprint(f"Creating synthetic test data: {shape}")
        
        # Create a simple test pattern with a few Gaussians
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        V = np.zeros(shape, dtype=np.float32)
        
        # Add some Gaussians manually
        centers_true = np.array([[20, 20], [30, 40], [45, 15]], dtype=np.float32)
        sigmas_true = [3.0, 2.5, 4.0]
        amps_true = [1.0, 0.8, 1.2]
        
        for i, (center, sigma, amp) in enumerate(zip(centers_true, sigmas_true, amps_true)):
            dy = y - center[0]
            dx = x - center[1]
            r_sq = (dy**2 + dx**2) / (sigma**2)
            V += amp * np.exp(-0.5 * r_sq)
        
        # Add some noise
        np.random.seed(42)
        V += 0.05 * np.random.randn(*shape)
        V = np.clip(V, 0, None)
        
        aprint(f"Data range: [{V.min():.3f}, {V.max():.3f}]")
        aprint(f"Data sum: {V.sum():.3f}")
        
        # Test centers (slightly perturbed from true)
        centers_init = centers_true + 2.0 * np.random.randn(*centers_true.shape)
        aprint(f"Initial centers:\n{centers_init}")
        
        # Test both MSE and Poisson loss
        for loss_type in ["mse", "poisson"]:
            with asection(f"Testing {loss_type.upper()} loss"):
                
                # Import the optimized fitter
                from luxar.gsplats.fit_gsplats_precision_optimized import (
                    fit_gaussian_splats_precision_optimized
                )
                
                try:
                    # Use small data for Poisson to avoid normalization issues
                    test_data = V * 10.0 if loss_type == "poisson" else V
                    
                    params, amps, stats = fit_gaussian_splats_precision_optimized(
                        V=test_data,
                        centers_overcomplete=centers_init,
                        init_sigma_vox=2.0,
                        n_iters=50,  # Small for testing
                        lr=0.1,
                        loss_type=loss_type,
                        l1_amp=0.001,
                        verbose=True,
                        device="cpu",  # Use CPU for reliable testing
                        early_stopping=True,
                        early_stop_patience=10,
                        # Test regularizers
                        off_diagonal_penalty=1e-3,
                        condition_penalty=1e-4,
                        energy_penalty=1e-4,
                    )
                    
                    aprint(f"✅ {loss_type.upper()} fitting completed successfully")
                    aprint(f"   Final loss: {stats['final_loss']:.6f}")
                    aprint(f"   Iterations: {stats['iterations']}")
                    aprint(f"   Converged: {stats['converged']}")
                    aprint(f"   Device used: {stats['device_used']}")
                    aprint(f"   Optimization approach: {stats['optimization_approach']}")
                    
                    # Check outputs
                    assert params.shape == (len(centers_init), 2 + 3), f"Wrong params shape: {params.shape}"
                    assert amps.shape == (len(centers_init),), f"Wrong amps shape: {amps.shape}"
                    assert len(stats['compression_scores']) == len(centers_init)
                    assert len(stats['compression_ranking']) == len(centers_init)
                    
                    fitted_centers = params[:, :2]
                    aprint(f"   Fitted centers:\n{fitted_centers}")
                    aprint(f"   Center errors: {np.linalg.norm(fitted_centers - centers_true, axis=1)}")
                    aprint(f"   Final amplitudes: {amps}")
                    aprint(f"   Compression scores: {stats['compression_scores']}")
                    
                except Exception as e:
                    aprint(f"❌ {loss_type.upper()} fitting failed: {e}")
                    raise
        
        # Test renderer directly
        with asection("Testing Optimized Renderer"):
            from luxar.gsplats.models.gsplats.gsplats_precision_render_optimized import (
                render_gaussians_precision_optimized
            )
            
            try:
                # Create simple test data
                centers = torch.tensor([[32.0, 32.0]], dtype=torch.float32)
                # Simple isotropic Gaussian: U = diag([1/σ, 1/σ])
                Us = torch.tensor([[[0.5, 0.0], [0.0, 0.5]]], dtype=torch.float32)  # σ = 2.0
                amps = torch.tensor([1.0], dtype=torch.float32)
                
                result = render_gaussians_precision_optimized(
                    shape=(64, 64),
                    centers=centers,
                    Us=Us,
                    amps=amps,
                    truncate=3.0
                )
                
                aprint(f"✅ Renderer test successful")
                aprint(f"   Output shape: {result.shape}")
                aprint(f"   Output range: [{result.min():.6f}, {result.max():.6f}]")
                aprint(f"   Output sum: {result.sum():.6f}")
                aprint(f"   Peak location: {torch.unravel_index(result.argmax(), result.shape)}")
                
                # Check that peak is near the center
                peak_y, peak_x = torch.unravel_index(result.argmax(), result.shape)
                center_error = torch.sqrt((peak_y - 32)**2 + (peak_x - 32)**2)
                aprint(f"   Peak error from center: {center_error:.3f} pixels")
                assert center_error < 2.0, f"Peak too far from center: {center_error}"
                
            except Exception as e:
                aprint(f"❌ Renderer test failed: {e}")
                raise
        
        # Test triangular solve robustness
        with asection("Testing MPS-Safe Triangular Solve"):
            from luxar.gsplats.models.gsplats.gsplats_precision_render_optimized import (
                _solve_triangular_safe, _compute_tight_aabb_radii
            )
            
            try:
                # Test the safe triangular solve
                U = torch.tensor([[[2.0, 1.0], [0.0, 1.5]]], dtype=torch.float32)  # Upper triangular
                b = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)   # Identity
                
                result = _solve_triangular_safe(U.transpose(-1, -2), b, upper=False)
                aprint(f"✅ Safe triangular solve test passed")
                aprint(f"   Result shape: {result.shape}")
                
                # Test AABB radii computation
                radii = _compute_tight_aabb_radii(U, truncate=3.0)
                aprint(f"✅ AABB radii computation test passed")
                aprint(f"   Radii shape: {radii.shape}")
                aprint(f"   Radii values: {radii}")
                
            except Exception as e:
                aprint(f"❌ Triangular solve test failed: {e}")
                raise
        
        aprint("🎉 All optimized precision fitting tests passed!")

if __name__ == "__main__":
    test_optimized_precision_fitting()