#!/usr/bin/env python3
"""
Debug why Gaussian shapes (covariance matrices) aren't being learned properly.
"""

import numpy as np
import torch
from arbol import aprint, asection

def debug_shape_learning():
    """Debug the shape learning in precision vs traditional approaches."""
    
    with asection("🔍 Shape Learning Debug"):
        
        # Create test with diverse Gaussian shapes
        shape = (64, 64)
        
        # Create ground truth with different covariances
        centers_gt = np.array([
            [20.0, 20.0],  # Large isotropic
            [20.0, 44.0],  # Small isotropic  
            [44.0, 20.0],  # Wide horizontal
            [44.0, 44.0],  # Tall vertical
        ], dtype=np.float32)
        
        amps_gt = np.array([0.8, 0.6, 0.7, 0.5], dtype=np.float32)
        
        # Different covariance matrices
        covariances_gt = np.array([
            [[9.0, 0.0], [0.0, 9.0]],     # Large isotropic (σ=3.0)
            [[1.0, 0.0], [0.0, 1.0]],     # Small isotropic (σ=1.0)  
            [[16.0, 0.0], [0.0, 4.0]],    # Wide horizontal (σx=4.0, σy=2.0)
            [[4.0, 0.0], [0.0, 16.0]],    # Tall vertical (σx=2.0, σy=4.0)
        ], dtype=np.float32)
        
        # Generate synthetic image
        image = np.zeros(shape, dtype=np.float32)
        y = np.arange(shape[0]).astype(np.float32)
        x = np.arange(shape[1]).astype(np.float32)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        for center, amp, cov in zip(centers_gt, amps_gt, covariances_gt):
            dy = Y - center[0]
            dx = X - center[1]
            
            # Compute quadratic form (y-μ)ᵀ Σ⁻¹ (y-μ)
            cov_inv = np.linalg.inv(cov)
            delta = np.stack([dy.ravel(), dx.ravel()], axis=1)  # (H*W, 2)
            quad_forms = np.sum(delta @ cov_inv * delta, axis=1)  # (H*W,)
            quad_forms = quad_forms.reshape(shape)
            
            mask = quad_forms <= 9.0  # 3-sigma truncation
            image[mask] += amp * np.exp(-0.5 * quad_forms[mask])
        
        aprint(f"Ground truth shapes:")
        for i, cov in enumerate(covariances_gt):
            eigenvals, _ = np.linalg.eigh(cov)
            sigmas = np.sqrt(eigenvals)
            aprint(f"  Gaussian {i}: σ = [{sigmas[0]:.1f}, {sigmas[1]:.1f}]")
        
        aprint(f"Test image - sum: {image.sum():.3f}")
        
        # Use exact centers as candidates (no position learning needed)
        centers_candidates = centers_gt.copy()
        
        # Test traditional approach
        with asection("🔄 Traditional Shape Learning"):
            try:
                from luxar.gsplats.fit_gsplats import fit_gaussian_splats
                from luxar.gsplats.utils.trils import unpack_tril
                
                params_trad, amps_trad, stats_trad = fit_gaussian_splats(
                    V=image,
                    centers_overcomplete=centers_candidates,
                    init_sigma_vox=2.5,  # Start with medium size
                    n_iters=200,
                    lr=0.1,
                    l1_amp=0.0,  # No sparsity for shape learning test
                    sigma_min_diag=[0.5, 0.5],  # Allow small shapes
                    sigma_max_diag=None,  # Allow large shapes
                    verbose=False,
                    use_precision_parameterization=False,
                )
                
                aprint(f"Traditional - Final loss: {stats_trad['final_loss']:.6f}")
                aprint(f"Traditional - Learned shapes:")
                
                L_matrices = unpack_tril(params_trad[:, 2:], 2)  # (N, 2, 2) lower triangular
                for i, L in enumerate(L_matrices):
                    cov = L @ L.T
                    eigenvals, _ = np.linalg.eigh(cov)
                    sigmas = np.sqrt(eigenvals)
                    aprint(f"  Gaussian {i}: σ = [{sigmas[0]:.1f}, {sigmas[1]:.1f}] (target: [{np.sqrt(covariances_gt[i].diagonal())[0]:.1f}, {np.sqrt(covariances_gt[i].diagonal())[1]:.1f}])")
                
                traditional_success = True
                
            except Exception as e:
                aprint(f"❌ Traditional failed: {e}")
                traditional_success = False
        
        # Test precision approach
        with asection("⚡ Precision Shape Learning"):
            try:
                from luxar.gsplats.fit_gsplats import fit_gaussian_splats
                from luxar.gsplats.utils.trils import unpack_tril
                
                params_prec, amps_prec, stats_prec = fit_gaussian_splats(
                    V=image,
                    centers_overcomplete=centers_candidates,
                    init_sigma_vox=2.5,  # Start with medium size
                    n_iters=200,
                    lr=0.1,
                    l1_amp=0.0,  # No sparsity for shape learning test
                    sigma_min_diag=[0.5, 0.5],  # Allow small shapes
                    sigma_max_diag=None,  # Allow large shapes
                    verbose=False,
                    use_precision_parameterization=True,
                )
                
                aprint(f"Precision - Final loss: {stats_prec['final_loss']:.6f}")
                aprint(f"Precision - Learned shapes:")
                
                L_matrices = unpack_tril(params_prec[:, 2:], 2)  # (N, 2, 2) lower triangular
                for i, L in enumerate(L_matrices):
                    cov = L @ L.T
                    eigenvals, _ = np.linalg.eigh(cov)
                    sigmas = np.sqrt(eigenvals)
                    aprint(f"  Gaussian {i}: σ = [{sigmas[0]:.1f}, {sigmas[1]:.1f}] (target: [{np.sqrt(covariances_gt[i].diagonal())[0]:.1f}, {np.sqrt(covariances_gt[i].diagonal())[1]:.1f}])")
                
                precision_success = True
                
            except Exception as e:
                aprint(f"❌ Precision failed: {e}")
                precision_success = False
        
        # Compare shape learning quality
        if traditional_success and precision_success:
            with asection("📊 Shape Learning Analysis"):
                
                def compute_shape_errors(params, target_covariances):
                    """Compute shape learning errors."""
                    L_matrices = unpack_tril(params[:, 2:], 2)
                    errors = []
                    for i, (L, target_cov) in enumerate(zip(L_matrices, target_covariances)):
                        learned_cov = L @ L.T
                        # Use Frobenius norm of difference
                        error = np.linalg.norm(learned_cov - target_cov, 'fro')
                        errors.append(error)
                    return np.array(errors)
                
                trad_errors = compute_shape_errors(params_trad, covariances_gt)
                prec_errors = compute_shape_errors(params_prec, covariances_gt)
                
                aprint(f"Shape learning errors (Frobenius norm):")
                aprint(f"Traditional: {trad_errors}")
                aprint(f"Precision:   {prec_errors}")
                
                aprint(f"Average shape error:")
                aprint(f"Traditional: {np.mean(trad_errors):.3f}")
                aprint(f"Precision:   {np.mean(prec_errors):.3f}")
                
                if np.mean(prec_errors) > np.mean(trad_errors) * 2:
                    aprint("💥 PRECISION APPROACH LEARNS SHAPES MUCH WORSE!")
                    aprint("This explains the 'too small and same shape' issue")
                elif np.mean(trad_errors) > np.mean(prec_errors) * 2:
                    aprint("💥 TRADITIONAL APPROACH LEARNS SHAPES MUCH WORSE!")
                else:
                    aprint("✅ Both approaches learn shapes similarly")
                    aprint("The issue may be elsewhere (constraints, initialization, etc.)")

if __name__ == "__main__":
    debug_shape_learning()