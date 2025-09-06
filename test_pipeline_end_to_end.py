#!/usr/bin/env python3
"""
End-to-end test of optimized precision pipeline vs traditional approach.
"""

import numpy as np
import torch
from arbol import aprint, asection

def create_test_image():
    """Create a simple synthetic test image with known Gaussian structure."""
    
    # Create 2D test image with two clear Gaussians
    shape = (32, 32)  # Small for fast testing
    x = np.arange(shape[1]).astype(np.float32)
    y = np.arange(shape[0]).astype(np.float32)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([Y.ravel(), X.ravel()], axis=1)  # (H*W, 2)
    
    # Ground truth Gaussians
    centers_gt = np.array([[10.0, 12.0], [20.0, 20.0]], dtype=np.float32)
    amps_gt = np.array([0.8, 0.6], dtype=np.float32)
    
    # Ground truth covariance matrices (symmetric positive definite)
    Sigma_gt = np.array([
        [[4.0, 1.0], [1.0, 2.0]],      # Elliptical Gaussian
        [[3.0, -0.5], [-0.5, 1.5]]     # Another elliptical Gaussian  
    ], dtype=np.float32)
    
    # Generate synthetic image
    image = np.zeros(shape, dtype=np.float32)
    for i, (center, amp, sigma) in enumerate(zip(centers_gt, amps_gt, Sigma_gt)):
        # For each pixel, compute Gaussian value
        for py in range(shape[0]):
            for px in range(shape[1]):
                delta = np.array([py, px], dtype=np.float32) - center
                quad_form = delta @ np.linalg.inv(sigma) @ delta
                if quad_form < 9.0:  # Truncation
                    image[py, px] += amp * np.exp(-0.5 * quad_form)
    
    # Add small amount of noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.01, shape).astype(np.float32)
    image = np.clip(image + noise, 0.0, 1.0)
    
    return image, centers_gt, amps_gt, Sigma_gt

def test_pipeline_comparison():
    """Compare traditional and optimized precision pipelines end-to-end."""
    
    with asection("🧪 End-to-End Pipeline Test"):
        
        # Create test data
        aprint("Creating synthetic test image...")
        image, centers_gt, amps_gt, Sigma_gt = create_test_image()
        
        aprint(f"Image shape: {image.shape}")
        aprint(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
        aprint(f"Image sum: {image.sum():.3f}")
        aprint(f"Ground truth centers: {centers_gt}")
        aprint(f"Ground truth amplitudes: {amps_gt}")
        
        # Generate overcomplete candidate centers (grid sampling)
        spacing = 2  # Sample every 2 pixels
        y_coords = np.arange(2, image.shape[0] - 2, spacing).astype(np.float32)
        x_coords = np.arange(2, image.shape[1] - 2, spacing).astype(np.float32)
        YY, XX = np.meshgrid(y_coords, x_coords, indexing='ij')
        centers_overcomplete = np.column_stack([YY.ravel(), XX.ravel()])
        
        aprint(f"Overcomplete centers: {len(centers_overcomplete)} candidates")
        
        # Test parameters
        n_iters = 100
        init_sigma_vox = 2.0
        lr = 0.1
        
        # Test 1: Traditional covariance approach
        with asection("🔄 Traditional Covariance Approach"):
            try:
                from luxar.gsplats.fit_gsplats import fit_gaussian_splats
                
                params_trad, amps_trad, stats_trad = fit_gaussian_splats(
                    V=image,
                    centers_overcomplete=centers_overcomplete,
                    init_sigma_vox=init_sigma_vox,
                    n_iters=n_iters,
                    lr=lr,
                    verbose=False,
                    early_stopping=True,
                    use_precision_parameterization=False,  # Use traditional
                )
                
                aprint(f"Traditional - Final loss: {stats_trad['final_loss']:.6f}")
                aprint(f"Traditional - Iterations: {stats_trad['iterations']}")
                aprint(f"Traditional - Converged: {stats_trad['converged']}")
                aprint(f"Traditional - Time: {stats_trad['time_seconds']:.2f}s")
                aprint(f"Traditional - Active Gaussians: {len(params_trad)}")
                
                # Render result for comparison
                from luxar.gsplats.models.gsplats.gsplats_render import render_gaussians_full_numpy
                result_trad = render_gaussians_full_numpy(
                    image.shape, params_trad, amps_trad, truncate=3.0
                )
                
                traditional_success = True
                
            except Exception as e:
                aprint(f"❌ Traditional approach failed: {e}")
                traditional_success = False
                result_trad = None
        
        # Test 2: Optimized precision approach
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
                    early_stopping=True,
                    output_format='covariance',  # For fair comparison
                )
                
                aprint(f"Optimized - Final loss: {stats_opt['final_loss']:.6f}")
                aprint(f"Optimized - Iterations: {stats_opt['iterations']}")
                aprint(f"Optimized - Converged: {stats_opt['converged']}")
                aprint(f"Optimized - Time: {stats_opt['time_seconds']:.2f}s")
                aprint(f"Optimized - Active Gaussians: {len(params_opt)}")
                
                # Render result for comparison
                result_opt = render_gaussians_full_numpy(
                    image.shape, params_opt, amps_opt, truncate=3.0
                )
                
                optimized_success = True
                
            except Exception as e:
                aprint(f"❌ Optimized approach failed: {e}")
                optimized_success = False
                result_opt = None
        
        # Compare results if both succeeded
        if traditional_success and optimized_success:
            with asection("📊 Results Comparison"):
                
                # Compare reconstruction quality
                error_trad = np.mean((image - result_trad) ** 2)
                error_opt = np.mean((image - result_opt) ** 2)
                
                aprint(f"Original image sum: {image.sum():.6f}")
                aprint(f"Traditional reconstruction sum: {result_trad.sum():.6f}")
                aprint(f"Optimized reconstruction sum: {result_opt.sum():.6f}")
                
                aprint(f"Traditional MSE vs original: {error_trad:.8f}")
                aprint(f"Optimized MSE vs original: {error_opt:.8f}")
                
                # Compare reconstructions directly
                direct_diff = np.mean((result_trad - result_opt) ** 2)
                max_diff = np.max(np.abs(result_trad - result_opt))
                
                aprint(f"Direct comparison MSE: {direct_diff:.8f}")
                aprint(f"Direct comparison max diff: {max_diff:.8f}")
                
                # Performance comparison
                speedup = stats_trad['time_seconds'] / stats_opt['time_seconds']
                aprint(f"Performance speedup: {speedup:.2f}x")
                
                # Compare final parameters
                param_diff = np.mean((params_trad - params_opt) ** 2)
                amp_diff = np.mean((amps_trad - amps_opt) ** 2)
                
                aprint(f"Parameter MSE: {param_diff:.8f}")
                aprint(f"Amplitude MSE: {amp_diff:.8f}")
                
                # Success criteria
                reconstruction_match = direct_diff < 1e-4  # Reconstructions should be very similar
                quality_reasonable = error_opt < 0.01      # Should reconstruct well
                speedup_achieved = speedup > 0.5          # Should not be much slower
                
                aprint("\n=== FINAL ASSESSMENT ===")
                
                if reconstruction_match:
                    aprint("✅ Reconstructions match closely")
                else:
                    aprint("❌ Reconstructions differ significantly")
                
                if quality_reasonable:
                    aprint("✅ Reconstruction quality is good")
                else:
                    aprint("❌ Reconstruction quality is poor")
                    
                if speedup_achieved:
                    aprint("✅ Performance is acceptable")
                else:
                    aprint("❌ Performance is significantly worse")
                
                overall_success = reconstruction_match and quality_reasonable and speedup_achieved
                
                if overall_success:
                    aprint("🎉 END-TO-END TEST PASSED!")
                    return True
                else:
                    aprint("💥 END-TO-END TEST FAILED!")
                    return False
                    
        else:
            aprint("❌ Cannot compare - one or both approaches failed")
            return False

if __name__ == "__main__":
    success = test_pipeline_comparison()
    exit(0 if success else 1)