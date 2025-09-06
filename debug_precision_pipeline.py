#!/usr/bin/env python3
"""
Debug the precision pipeline step by step to find where reconstruction breaks down.
"""

import numpy as np
import torch
from arbol import aprint, asection

def debug_precision_pipeline():
    """Trace through precision pipeline step by step."""
    
    with asection("🔍 Precision Pipeline Debug"):
        
        # Create super simple test case
        shape = (32, 32)
        
        # Single Gaussian at center
        center_gt = np.array([16.0, 16.0])
        sigma_gt = 3.0
        amp_gt = 0.8
        
        # Generate perfect target image
        y = np.arange(shape[0]).astype(np.float32)
        x = np.arange(shape[1]).astype(np.float32)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        dy = Y - center_gt[0]
        dx = X - center_gt[1]
        r_sq = (dy ** 2 + dx ** 2) / (sigma_gt ** 2)
        
        target = amp_gt * np.exp(-0.5 * r_sq)
        
        aprint(f"Target image - sum: {target.sum():.6f}")
        aprint(f"Target image - max: {target.max():.6f}")
        aprint(f"Target center: {center_gt}")
        
        # Test 1: Direct precision renderer with known correct parameters
        with asection("Step 1: Test precision renderer with ground truth"):
            try:
                from luxar.gsplats.models.gsplats.gsplats_precision_render_optimized import render_gaussians_precision_optimized
                
                centers_torch = torch.tensor([[center_gt]], dtype=torch.float32)  # (1, 1, 2)
                amps_torch = torch.tensor([amp_gt], dtype=torch.float32)
                
                # Ground truth precision matrix: Λ = diag([1/σ², 1/σ²])
                # Upper triangular Cholesky: U = diag([1/σ, 1/σ])
                precision_val = 1.0 / sigma_gt
                U = torch.tensor([[[precision_val, 0.0], [0.0, precision_val]]], dtype=torch.float32)
                
                result_direct = render_gaussians_precision_optimized(
                    shape, centers_torch, U, amps_torch, truncate=3.0
                )
                result_direct_np = result_direct.numpy()
                
                aprint(f"Direct precision render - sum: {result_direct_np.sum():.6f}")
                aprint(f"Direct precision render - max: {result_direct_np.max():.6f}")
                aprint(f"Error vs target: {np.mean((target - result_direct_np) ** 2):.8f}")
                
                if np.mean((target - result_direct_np) ** 2) < 1e-6:
                    aprint("✅ Direct precision renderer works correctly")
                    direct_renderer_works = True
                else:
                    aprint("❌ Direct precision renderer has issues")
                    direct_renderer_works = False
                    
            except Exception as e:
                aprint(f"❌ Direct precision renderer failed: {e}")
                import traceback
                traceback.print_exc()
                direct_renderer_works = False
        
        # Test 2: Precision fitter with single candidate at correct location
        with asection("Step 2: Test precision fitter with perfect initialization"):
            if direct_renderer_works:
                try:
                    from luxar.gsplats.fit_gsplats_precision_optimized import fit_gaussian_splats_precision_optimized
                    
                    # Single candidate at correct location
                    centers_candidates = np.array([center_gt], dtype=np.float32).reshape(1, -1)
                    
                    params_fit, amps_fit, stats_fit = fit_gaussian_splats_precision_optimized(
                        V=target,
                        centers_overcomplete=centers_candidates,
                        init_sigma_vox=sigma_gt,  # Start with correct sigma
                        n_iters=100,
                        lr=0.1,  # Conservative learning rate
                        l1_amp=0.0,  # No regularization for this test
                        verbose=False,
                        output_format='covariance',  # Convert back to covariance format
                    )
                    
                    aprint(f"Fit result - final loss: {stats_fit['final_loss']:.8f}")
                    aprint(f"Fit result - fitted center: {params_fit[0, :2]}")
                    aprint(f"Fit result - fitted amplitude: {amps_fit[0]:.6f}")
                    
                    # Render fitted result
                    from luxar.gsplats.models.gsplats.gsplats_render import render_gaussians_full_numpy
                    result_fit = render_gaussians_full_numpy(shape, params_fit, amps_fit, truncate=3.0)
                    
                    aprint(f"Fitted render - sum: {result_fit.sum():.6f}")
                    aprint(f"Fitted render - max: {result_fit.max():.6f}")
                    
                    fit_error = np.mean((target - result_fit) ** 2)
                    aprint(f"Fitted error vs target: {fit_error:.8f}")
                    
                    if fit_error < 1e-4:
                        aprint("✅ Precision fitter works with perfect initialization")
                        fitter_works = True
                    else:
                        aprint("❌ Precision fitter has issues even with perfect initialization")
                        fitter_works = False
                        
                        # Debug the fitted parameters
                        from luxar.gsplats.utils.trils import unpack_tril
                        L_fitted = unpack_tril(params_fit[:, 2:], 2)[0]
                        sigma_fitted = np.sqrt(np.diag(L_fitted @ L_fitted.T))
                        
                        aprint(f"Debug - fitted sigma: {sigma_fitted} (target: [{sigma_gt}, {sigma_gt}])")
                        aprint(f"Debug - center error: {np.linalg.norm(params_fit[0, :2] - center_gt):.6f}")
                        aprint(f"Debug - amplitude error: {abs(amps_fit[0] - amp_gt):.6f}")
                    
                except Exception as e:
                    aprint(f"❌ Precision fitter failed: {e}")
                    import traceback
                    traceback.print_exc()
                    fitter_works = False
        
        # Test 3: Traditional fitter comparison
        with asection("Step 3: Compare with traditional fitter"):
            try:
                from luxar.gsplats.fit_gsplats import fit_gaussian_splats
                
                centers_candidates = np.array([center_gt], dtype=np.float32).reshape(1, -1)
                
                params_trad, amps_trad, stats_trad = fit_gaussian_splats(
                    V=target,
                    centers_overcomplete=centers_candidates,
                    init_sigma_vox=sigma_gt,
                    n_iters=100,
                    lr=0.1,
                    l1_amp=0.0,
                    verbose=False,
                    use_precision_parameterization=False,  # Traditional approach
                )
                
                result_trad = render_gaussians_full_numpy(shape, params_trad, amps_trad, truncate=3.0)
                trad_error = np.mean((target - result_trad) ** 2)
                
                aprint(f"Traditional - final loss: {stats_trad['final_loss']:.8f}")
                aprint(f"Traditional - sum: {result_trad.sum():.6f}")
                aprint(f"Traditional - error: {trad_error:.8f}")
                
                if direct_renderer_works and fitter_works:
                    if trad_error < fit_error * 10:  # Traditional should be similar quality
                        aprint("✅ Both approaches have similar quality")
                    else:
                        aprint(f"❌ Traditional is much better ({trad_error:.2e} vs {fit_error:.2e})")
                        aprint("This confirms precision pipeline has a bug!")
                
            except Exception as e:
                aprint(f"❌ Traditional fitter failed: {e}")
        
        # Test 4: Inspect precision-to-covariance conversion
        with asection("Step 4: Debug precision-to-covariance conversion"):
            if direct_renderer_works:
                try:
                    # Test the conversion pipeline manually
                    from luxar.gsplats.models.gsplats.gsplat_precision_model_optimized import GaussianSplatPrecisionModelOptimized
                    
                    # Create model with known parameters
                    U_test = np.array([[[precision_val, 0.0], [0.0, precision_val]]], dtype=np.float32)
                    
                    model_test = GaussianSplatPrecisionModelOptimized(
                        shape=shape,
                        centers0=centers_candidates,
                        U0=U_test,
                        amps0=np.array([amp_gt]),
                        truncate=3.0,
                        device=torch.device('cpu'),
                    )
                    
                    # Test precision matrices
                    Lambda = model_test.get_precision_matrices()
                    aprint(f"Model precision matrix:\n{Lambda[0]}")
                    
                    # Test covariance matrices
                    Sigma = model_test.get_covariance_matrices()
                    aprint(f"Model covariance matrix:\n{Sigma[0]}")
                    
                    # Expected covariance: [[σ², 0], [0, σ²]]
                    expected_cov = np.array([[sigma_gt**2, 0.0], [0.0, sigma_gt**2]])
                    cov_error = np.linalg.norm(Sigma[0].numpy() - expected_cov)
                    aprint(f"Covariance conversion error: {cov_error:.8f}")
                    
                    if cov_error < 1e-6:
                        aprint("✅ Precision-to-covariance conversion is correct")
                    else:
                        aprint("❌ Precision-to-covariance conversion has errors")
                        aprint(f"Expected:\n{expected_cov}")
                        aprint(f"Got:\n{Sigma[0].numpy()}")
                
                except Exception as e:
                    aprint(f"❌ Conversion test failed: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    debug_precision_pipeline()