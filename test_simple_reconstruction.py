#!/usr/bin/env python3
"""
Test reconstruction with a single known Gaussian to isolate any issues.
"""

import numpy as np
import torch
from arbol import aprint, asection

def test_single_gaussian_reconstruction():
    """Test reconstruction with a single, well-positioned Gaussian."""
    
    with asection("🎯 Single Gaussian Reconstruction Test"):
        
        # Create simple test with one Gaussian at known location
        shape = (20, 20)
        center_gt = np.array([10.0, 10.0], dtype=np.float32)  # Center of image
        amp_gt = 0.5
        sigma_gt = 2.0  # Standard deviation
        
        # Generate ground truth image
        y = np.arange(shape[0]).astype(np.float32)
        x = np.arange(shape[1]).astype(np.float32)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        # Gaussian: exp(-0.5 * ((y-cy)² + (x-cx)²) / σ²)
        dy = Y - center_gt[0]
        dx = X - center_gt[1]
        r_sq = (dy ** 2 + dx ** 2) / (sigma_gt ** 2)
        
        image = amp_gt * np.exp(-0.5 * r_sq)
        
        aprint(f"Ground truth image - shape: {image.shape}")
        aprint(f"Ground truth image - sum: {image.sum():.6f}")
        aprint(f"Ground truth image - max: {image.max():.6f}")
        aprint(f"Ground truth center: {center_gt}")
        aprint(f"Ground truth amplitude: {amp_gt}")
        aprint(f"Ground truth sigma: {sigma_gt}")
        
        # Use a single candidate center at the correct location
        centers_overcomplete = np.array([center_gt], dtype=np.float32)
        
        # Test parameters for fast convergence
        n_iters = 50
        init_sigma_vox = sigma_gt  # Initialize close to ground truth
        lr = 0.2
        
        # Test with traditional approach
        with asection("🔄 Traditional Approach"):
            try:
                from luxar.gsplats.fit_gsplats import fit_gaussian_splats
                
                params_trad, amps_trad, stats_trad = fit_gaussian_splats(
                    V=image,
                    centers_overcomplete=centers_overcomplete,
                    init_sigma_vox=init_sigma_vox,
                    n_iters=n_iters,
                    lr=lr,
                    verbose=False,
                    use_precision_parameterization=False,
                )
                
                aprint(f"Traditional - Final loss: {stats_trad['final_loss']:.8f}")
                aprint(f"Traditional - Fitted center: {params_trad[0, :2]}")
                aprint(f"Traditional - Fitted amplitude: {amps_trad[0]:.6f}")
                
                # Extract covariance matrix from traditional result
                from luxar.gsplats.utils.trils import unpack_tril
                L_trad = unpack_tril(params_trad[:, 2:], 2)[0]  # (2, 2) lower triangular
                Sigma_trad = L_trad @ L_trad.T
                sigma_trad = np.sqrt(np.diag(Sigma_trad))
                aprint(f"Traditional - Fitted sigma: {sigma_trad}")
                
                # Render traditional result
                from luxar.gsplats.models.gsplats.gsplats_render import render_gaussians_full_numpy
                result_trad = render_gaussians_full_numpy(shape, params_trad, amps_trad, truncate=3.0)
                
                traditional_success = True
                
            except Exception as e:
                aprint(f"❌ Traditional approach failed: {e}")
                traditional_success = False
                result_trad = None
        
        # Test with optimized precision approach
        with asection("⚡ Optimized Precision Approach"):
            try:
                from luxar.gsplats.fit_gsplats_precision_optimized import fit_gaussian_splats_precision_optimized
                
                params_opt, amps_opt, stats_opt = fit_gaussian_splats_precision_optimized(
                    V=image,
                    centers_overcomplete=centers_overcomplete,
                    init_sigma_vox=init_sigma_vox,
                    n_iters=n_iters,
                    lr=lr,
                    verbose=False,
                    output_format='covariance',  # For fair comparison
                )
                
                aprint(f"Optimized - Final loss: {stats_opt['final_loss']:.8f}")
                aprint(f"Optimized - Fitted center: {params_opt[0, :2]}")
                aprint(f"Optimized - Fitted amplitude: {amps_opt[0]:.6f}")
                
                # Extract covariance matrix from optimized result
                from luxar.gsplats.utils.trils import unpack_tril
                L_opt = unpack_tril(params_opt[:, 2:], 2)[0]  # (2, 2) lower triangular
                Sigma_opt = L_opt @ L_opt.T
                sigma_opt = np.sqrt(np.diag(Sigma_opt))
                aprint(f"Optimized - Fitted sigma: {sigma_opt}")
                
                # Render optimized result
                result_opt = render_gaussians_full_numpy(shape, params_opt, amps_opt, truncate=3.0)
                
                optimized_success = True
                
            except Exception as e:
                aprint(f"❌ Optimized approach failed: {e}")
                optimized_success = False
                result_opt = None
        
        # Compare results
        if traditional_success and optimized_success:
            with asection("📊 Comparison"):
                
                # Compare to ground truth
                error_gt = np.mean((image - image) ** 2)  # Should be 0
                error_trad = np.mean((image - result_trad) ** 2)
                error_opt = np.mean((image - result_opt) ** 2)
                
                aprint(f"Ground truth sum: {image.sum():.6f}")
                aprint(f"Traditional reconstruction sum: {result_trad.sum():.6f}")
                aprint(f"Optimized reconstruction sum: {result_opt.sum():.6f}")
                
                aprint(f"Traditional MSE vs ground truth: {error_trad:.8f}")
                aprint(f"Optimized MSE vs ground truth: {error_opt:.8f}")
                
                # Compare reconstructions directly
                direct_diff = np.mean((result_trad - result_opt) ** 2)
                max_diff = np.max(np.abs(result_trad - result_opt))
                rel_diff = np.linalg.norm(result_trad - result_opt) / np.linalg.norm(image)
                
                aprint(f"Direct MSE: {direct_diff:.8f}")
                aprint(f"Direct max diff: {max_diff:.8f}")
                aprint(f"Direct relative L2 error: {rel_diff:.8f}")
                
                # Compare fitted parameters
                center_diff = np.linalg.norm(params_trad[0, :2] - params_opt[0, :2])
                amp_diff = abs(amps_trad[0] - amps_opt[0])
                
                aprint(f"Center difference: {center_diff:.6f}")
                aprint(f"Amplitude difference: {amp_diff:.6f}")
                
                # Assessment
                good_fit_traditional = error_trad < 1e-6
                good_fit_optimized = error_opt < 1e-6
                similar_results = direct_diff < 1e-4
                
                aprint(f"\n=== ASSESSMENT ===")
                aprint(f"Traditional fit quality: {'✅ Good' if good_fit_traditional else '❌ Poor'}")
                aprint(f"Optimized fit quality: {'✅ Good' if good_fit_optimized else '❌ Poor'}")
                aprint(f"Result similarity: {'✅ Similar' if similar_results else '❌ Different'}")
                
                if good_fit_traditional and good_fit_optimized and similar_results:
                    aprint("🎉 SINGLE GAUSSIAN TEST PASSED!")
                    return True
                else:
                    aprint("💥 SINGLE GAUSSIAN TEST FAILED!")
                    
                    # Detailed analysis if test fails
                    if not similar_results:
                        aprint("\n📊 DETAILED DIFFERENCE ANALYSIS:")
                        diff_map = np.abs(result_trad - result_opt)
                        aprint(f"Max difference location: {np.unravel_index(np.argmax(diff_map), diff_map.shape)}")
                        aprint(f"Mean absolute difference: {np.mean(diff_map):.8f}")
                        
                        # Check if both reconstructions have correct peak
                        peak_trad = np.unravel_index(np.argmax(result_trad), result_trad.shape)
                        peak_opt = np.unravel_index(np.argmax(result_opt), result_opt.shape)
                        peak_gt = np.unravel_index(np.argmax(image), image.shape)
                        
                        aprint(f"Ground truth peak at: {peak_gt}")
                        aprint(f"Traditional peak at: {peak_trad}")  
                        aprint(f"Optimized peak at: {peak_opt}")
                    
                    return False
        else:
            aprint("❌ Cannot compare - one or both approaches failed")
            return False

if __name__ == "__main__":
    success = test_single_gaussian_reconstruction()
    exit(0 if success else 1)