#!/usr/bin/env python3
"""
Compare reconstruction quality between traditional and precision approaches.
"""

import numpy as np
from arbol import aprint, asection

def test_reconstruction_comparison():
    """Compare reconstruction quality side by side."""
    
    with asection("🔄 Reconstruction Quality Comparison"):
        
        # Use the same synthetic data as in my earlier tests
        # but with more realistic parameters
        shape = (64, 64)  # Larger for more detail
        
        # Create a more complex synthetic image with multiple structures
        aprint("Creating synthetic test image...")
        
        # Ground truth with multiple Gaussian structures
        centers_gt = np.array([
            [16.0, 16.0], [48.0, 16.0], [32.0, 48.0], [16.0, 48.0], [48.0, 48.0]
        ], dtype=np.float32)
        
        amps_gt = np.array([0.8, 0.6, 0.7, 0.5, 0.9], dtype=np.float32)
        sigmas_gt = [3.0, 2.5, 4.0, 2.0, 3.5]  # Different sizes
        
        # Generate synthetic image
        image = np.zeros(shape, dtype=np.float32)
        y = np.arange(shape[0]).astype(np.float32)
        x = np.arange(shape[1]).astype(np.float32)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        for center, amp, sigma in zip(centers_gt, amps_gt, sigmas_gt):
            dy = Y - center[0]
            dx = X - center[1]
            r_sq = (dy ** 2 + dx ** 2) / (sigma ** 2)
            mask = r_sq <= 9.0  # 3-sigma truncation
            image[mask] += amp * np.exp(-0.5 * r_sq[mask])
        
        # Add small amount of noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)
        
        aprint(f"Test image - shape: {image.shape}")
        aprint(f"Test image - range: [{image.min():.3f}, {image.max():.3f}]")
        aprint(f"Test image - sum: {image.sum():.3f}")
        aprint(f"Ground truth centers: {len(centers_gt)} Gaussians")
        
        # Create overcomplete centers (dense grid)
        spacing = 4  # Every 4 pixels
        y_coords = np.arange(4, shape[0] - 4, spacing).astype(np.float32)
        x_coords = np.arange(4, shape[1] - 4, spacing).astype(np.float32)
        YY, XX = np.meshgrid(y_coords, x_coords, indexing='ij')
        centers_overcomplete = np.column_stack([YY.ravel(), XX.ravel()])
        
        aprint(f"Candidate centers: {len(centers_overcomplete)} positions")
        
        # Test parameters
        n_iters = 200
        init_sigma_vox = 2.5
        lr = 0.1
        l1_amp = 0.001  # Light sparsity to prune weak splats
        
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
                    l1_amp=l1_amp,
                    verbose=True,
                    use_precision_parameterization=False,  # Traditional
                )
                
                aprint(f"Traditional - Final loss: {stats_trad['final_loss']:.8f}")
                aprint(f"Traditional - Active splats: {len(params_trad)}")
                aprint(f"Traditional - Time: {stats_trad['time_seconds']:.2f}s")
                
                # Count significant amplitudes
                sig_amps_trad = np.sum(amps_trad > 0.1)
                aprint(f"Traditional - Significant splats (amp > 0.1): {sig_amps_trad}")
                
                # Render result
                from luxar.gsplats.models.gsplats.gsplats_render import render_gaussians_full_numpy
                result_trad = render_gaussians_full_numpy(shape, params_trad, amps_trad, truncate=3.0)
                
                # Quality metrics
                mse_trad = np.mean((image - result_trad) ** 2)
                aprint(f"Traditional - Reconstruction MSE: {mse_trad:.8f}")
                aprint(f"Traditional - Result sum: {result_trad.sum():.6f} (target: {image.sum():.6f})")
                
                traditional_success = True
                
            except Exception as e:
                aprint(f"❌ Traditional approach failed: {e}")
                import traceback
                traceback.print_exc()
                traditional_success = False
        
        # Test 2: Precision parameterization (current default)
        with asection("⚡ Precision Parameterization Approach"):
            try:
                from luxar.gsplats.fit_gsplats import fit_gaussian_splats
                
                params_prec, amps_prec, stats_prec = fit_gaussian_splats(
                    V=image,
                    centers_overcomplete=centers_overcomplete,
                    init_sigma_vox=init_sigma_vox,
                    n_iters=n_iters,
                    lr=lr,
                    l1_amp=l1_amp,
                    verbose=True,
                    use_precision_parameterization=True,  # Precision (default)
                )
                
                aprint(f"Precision - Final loss: {stats_prec['final_loss']:.8f}")
                aprint(f"Precision - Active splats: {len(params_prec)}")
                aprint(f"Precision - Time: {stats_prec['time_seconds']:.2f}s")
                
                # Count significant amplitudes
                sig_amps_prec = np.sum(amps_prec > 0.1)
                aprint(f"Precision - Significant splats (amp > 0.1): {sig_amps_prec}")
                
                # Render result
                result_prec = render_gaussians_full_numpy(shape, params_prec, amps_prec, truncate=3.0)
                
                # Quality metrics
                mse_prec = np.mean((image - result_prec) ** 2)
                aprint(f"Precision - Reconstruction MSE: {mse_prec:.8f}")
                aprint(f"Precision - Result sum: {result_prec.sum():.6f} (target: {image.sum():.6f})")
                
                precision_success = True
                
            except Exception as e:
                aprint(f"❌ Precision approach failed: {e}")
                import traceback
                traceback.print_exc()
                precision_success = False
        
        # Compare results
        if traditional_success and precision_success:
            with asection("📊 Quality Comparison"):
                
                # Direct comparison metrics
                aprint(f"Loss comparison:")
                aprint(f"  Traditional: {stats_trad['final_loss']:.8f}")
                aprint(f"  Precision:   {stats_prec['final_loss']:.8f}")
                aprint(f"  Precision is {'better' if stats_prec['final_loss'] < stats_trad['final_loss'] else 'worse'}")
                
                aprint(f"\nReconstruction MSE:")
                aprint(f"  Traditional: {mse_trad:.8f}")
                aprint(f"  Precision:   {mse_prec:.8f}")
                aprint(f"  Precision is {'better' if mse_prec < mse_trad else 'worse'}")
                
                aprint(f"\nNumber of significant splats:")
                aprint(f"  Traditional: {sig_amps_trad}")
                aprint(f"  Precision:   {sig_amps_prec}")
                
                aprint(f"\nSum conservation:")
                aprint(f"  Target:      {image.sum():.6f}")
                aprint(f"  Traditional: {result_trad.sum():.6f} (error: {abs(result_trad.sum() - image.sum()):.6f})")
                aprint(f"  Precision:   {result_prec.sum():.6f} (error: {abs(result_prec.sum() - image.sum()):.6f})")
                
                # Performance comparison
                speedup = stats_trad['time_seconds'] / stats_prec['time_seconds']
                aprint(f"\nPerformance:")
                aprint(f"  Traditional: {stats_trad['time_seconds']:.2f}s")
                aprint(f"  Precision:   {stats_prec['time_seconds']:.2f}s")
                aprint(f"  Speedup:     {speedup:.2f}x")
                
                # Overall assessment
                quality_issue = mse_prec > mse_trad * 2  # Precision is significantly worse
                if quality_issue:
                    aprint(f"\n💥 QUALITY ISSUE CONFIRMED!")
                    aprint(f"   Precision approach has {mse_prec/mse_trad:.2f}x higher MSE")
                    aprint(f"   This explains the 'weird reconstructions' in the visualization")
                else:
                    aprint(f"\n✅ Quality is comparable between approaches")
                
                return not quality_issue
        
        else:
            aprint("❌ Cannot compare - one or both approaches failed")
            return False

if __name__ == "__main__":
    success = test_reconstruction_comparison()
    exit(0 if success else 1)