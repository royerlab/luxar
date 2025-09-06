#!/usr/bin/env python3
"""
Debug loss computation and identify amplitude scaling issues.
"""

import numpy as np
import torch
from arbol import aprint, asection

def test_loss_computation():
    """Test if loss computation is correct in both approaches."""
    
    with asection("🔍 Loss Computation Debug"):
        
        # Create simple test case
        shape = (10, 10)
        
        # Simple uniform image
        image = np.full(shape, 0.5, dtype=np.float32)
        aprint(f"Test image - uniform value: {image[0, 0]:.3f}")
        aprint(f"Test image - sum: {image.sum():.3f}")
        
        # Single center at image center
        center = np.array([[5.0, 5.0]], dtype=np.float32)
        
        # Test with a known Gaussian
        sigma = 2.0
        amplitude = 0.3
        
        # Manual rendering to check our understanding
        aprint(f"\n=== Manual Gaussian Rendering ===")
        y = np.arange(shape[0]).astype(np.float32)
        x = np.arange(shape[1]).astype(np.float32)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        dy = Y - center[0, 0]
        dx = X - center[0, 1]
        r_sq = (dy ** 2 + dx ** 2) / (sigma ** 2)
        
        gaussian_manual = amplitude * np.exp(-0.5 * r_sq)
        aprint(f"Manual Gaussian sum: {gaussian_manual.sum():.6f}")
        aprint(f"Manual Gaussian max: {gaussian_manual.max():.6f}")
        
        # Test traditional renderer
        aprint(f"\n=== Traditional Renderer Test ===")
        try:
            from luxar.gsplats.models.gsplats.gsplats_render import render_gaussians_full_numpy
            from luxar.gsplats.utils.trils import pack_tril
            
            # Create traditional format parameters  
            # Covariance matrix for sigma=2: [[4, 0], [0, 4]]
            # Cholesky: [[2, 0], [0, 2]]
            L_cov = np.array([[[sigma, 0.0], [0.0, sigma]]], dtype=np.float32)
            
            d = 2
            tril_size = d * (d + 1) // 2
            params_trad = np.zeros((1, d + tril_size), dtype=np.float32)
            params_trad[:, :d] = center
            params_trad[:, d:] = pack_tril(L_cov)
            
            amps_trad = np.array([amplitude], dtype=np.float32)
            
            result_trad = render_gaussians_full_numpy(shape, params_trad, amps_trad, truncate=3.0)
            
            aprint(f"Traditional renderer sum: {result_trad.sum():.6f}")
            aprint(f"Traditional renderer max: {result_trad.max():.6f}")
            aprint(f"Difference vs manual: {np.mean((result_trad - gaussian_manual) ** 2):.8f}")
            
        except Exception as e:
            aprint(f"Traditional renderer failed: {e}")
        
        # Test optimized precision renderer
        aprint(f"\n=== Optimized Precision Renderer Test ===")
        try:
            from luxar.gsplats.models.gsplats.gsplats_precision_render_optimized import render_gaussians_precision_optimized
            
            centers_torch = torch.from_numpy(center)
            amps_torch = torch.from_numpy(amps_trad)
            
            # Convert sigma to precision Cholesky (upper triangular)
            # Σ = [[σ², 0], [0, σ²]], Λ = Σ^{-1} = [[1/σ², 0], [0, 1/σ²]]
            # U such that Λ = U^T @ U, so U = [[1/σ, 0], [0, 1/σ]]
            precision_diag = 1.0 / sigma
            U = torch.tensor([[[precision_diag, 0.0], [0.0, precision_diag]]], dtype=torch.float32)
            
            result_opt = render_gaussians_precision_optimized(shape, centers_torch, U, amps_torch, truncate=3.0)
            result_opt_np = result_opt.numpy()
            
            aprint(f"Optimized renderer sum: {result_opt_np.sum():.6f}")
            aprint(f"Optimized renderer max: {result_opt_np.max():.6f}")
            aprint(f"Difference vs manual: {np.mean((result_opt_np - gaussian_manual) ** 2):.8f}")
            
        except Exception as e:
            aprint(f"Optimized renderer failed: {e}")
        
        # Test fitting with this simple case
        aprint(f"\n=== Fitting Test with Known Target ===")
        
        target_image = gaussian_manual.copy()  # Use the manual Gaussian as target
        
        # Test traditional fitting
        aprint("Traditional fitting:")
        try:
            from luxar.gsplats.fit_gsplats import fit_gaussian_splats
            
            params_fit_trad, amps_fit_trad, stats_fit_trad = fit_gaussian_splats(
                V=target_image,
                centers_overcomplete=center,
                init_sigma_vox=sigma,
                n_iters=100,
                lr=0.1,
                verbose=False,
                use_precision_parameterization=False,
            )
            
            aprint(f"  Final loss: {stats_fit_trad['final_loss']:.8f}")
            aprint(f"  Fitted amplitude: {amps_fit_trad[0]:.6f} (target: {amplitude:.6f})")
            aprint(f"  Fitted center: {params_fit_trad[0, :2]} (target: {center[0]})")
            
            # Render fitted result
            result_fit_trad = render_gaussians_full_numpy(shape, params_fit_trad, amps_fit_trad, truncate=3.0)
            aprint(f"  Fitted result sum: {result_fit_trad.sum():.6f} (target: {target_image.sum():.6f})")
            
        except Exception as e:
            aprint(f"  Failed: {e}")
        
        # Test optimized fitting 
        aprint("Optimized fitting:")
        try:
            from luxar.gsplats.fit_gsplats_precision_optimized import fit_gaussian_splats_precision_optimized
            
            params_fit_opt, amps_fit_opt, stats_fit_opt = fit_gaussian_splats_precision_optimized(
                V=target_image,
                centers_overcomplete=center,
                init_sigma_vox=sigma,
                n_iters=100,
                lr=0.1,
                verbose=False,
                output_format='covariance',
            )
            
            aprint(f"  Final loss: {stats_fit_opt['final_loss']:.8f}")
            aprint(f"  Fitted amplitude: {amps_fit_opt[0]:.6f} (target: {amplitude:.6f})")
            aprint(f"  Fitted center: {params_fit_opt[0, :2]} (target: {center[0]})")
            
            # Render fitted result
            result_fit_opt = render_gaussians_full_numpy(shape, params_fit_opt, amps_fit_opt, truncate=3.0)
            aprint(f"  Fitted result sum: {result_fit_opt.sum():.6f} (target: {target_image.sum():.6f})")
            
        except Exception as e:
            aprint(f"  Failed: {e}")
        
        aprint(f"\n=== Analysis Complete ===")

if __name__ == "__main__":
    test_loss_computation()