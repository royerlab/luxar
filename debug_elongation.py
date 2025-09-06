#!/usr/bin/env python3
"""
Debug why Gaussians become extremely elongated.
"""

import numpy as np
import torch
from arbol import aprint, asection

def debug_elongation_issue():
    """Debug extreme elongation in precision approach."""
    
    with asection("🔍 Elongation Debug"):
        
        # Create test with blob-like data (should prefer circular/moderate ellipses)
        from skimage import data, filters
        
        blobs = data.binary_blobs(
            length=64, blob_size_fraction=0.08, n_dim=2, volume_fraction=0.2, rng=42
        ).astype(float)
        V = filters.gaussian(blobs, 1.0)  # Smooth blobs
        V = (V / V.max()).astype(np.float32)
        
        aprint(f"Test image - blob-like data")
        aprint(f"Shape: {V.shape}, Range: [{V.min():.3f}, {V.max():.3f}]")
        
        # Generate candidates
        from luxar.gsplats.candidates import find_candidates_overcomplete_nd
        centers = find_candidates_overcomplete_nd(
            V,
            spacing=(3, 3),
            peaks_per_scale=50,  # Fewer candidates for clearer analysis
            add_intensity_grid=False,
        )
        
        aprint(f"Using {len(centers)} candidate centers")
        
        # Test with precision approach
        with asection("⚡ Precision Approach Analysis"):
            try:
                from luxar.gsplats.fit_gsplats import fit_gaussian_splats
                from luxar.gsplats.utils.trils import unpack_tril
                
                params_prec, amps_prec, stats_prec = fit_gaussian_splats(
                    V=V,
                    centers_overcomplete=centers,
                    init_sigma_vox=2.0,
                    n_iters=150,
                    lr=0.1,
                    l1_amp=0.001,  # Light regularization
                    sigma_min_diag=[0.5, 0.5],  # Prevent too small
                    sigma_max_diag=None,        # Allow large (after our fix)
                    verbose=False,
                    use_precision_parameterization=True,
                )
                
                aprint(f"Final loss: {stats_prec['final_loss']:.6f}")
                aprint(f"Active Gaussians: {len(params_prec)}")
                
                # Analyze shapes
                L_matrices = unpack_tril(params_prec[:, 2:], 2)
                
                aspect_ratios = []
                eigenvalue_ratios = []
                max_sigmas = []
                
                for i, L in enumerate(L_matrices):
                    if amps_prec[i] > 0.1:  # Only analyze significant Gaussians
                        cov = L @ L.T
                        eigenvals, _ = np.linalg.eigh(cov)
                        eigenvals = np.sort(eigenvals)[::-1]  # Descending order
                        
                        sigmas = np.sqrt(eigenvals)
                        aspect_ratio = sigmas[0] / sigmas[1] if sigmas[1] > 1e-6 else float('inf')
                        eigenvalue_ratio = eigenvals[0] / eigenvals[1] if eigenvals[1] > 1e-6 else float('inf')
                        
                        aspect_ratios.append(aspect_ratio)
                        eigenvalue_ratios.append(eigenvalue_ratio)
                        max_sigmas.append(sigmas[0])
                        
                        if aspect_ratio > 5.0:  # Very elongated
                            aprint(f"  Elongated Gaussian {i}: σ = [{sigmas[0]:.1f}, {sigmas[1]:.1f}], ratio = {aspect_ratio:.1f}")
                
                aspect_ratios = np.array(aspect_ratios)
                max_sigmas = np.array(max_sigmas)
                
                aprint(f"\nShape Statistics (significant Gaussians only):")
                aprint(f"  Count: {len(aspect_ratios)}")
                aprint(f"  Aspect ratios - mean: {np.mean(aspect_ratios):.2f}, max: {np.max(aspect_ratios):.1f}")
                aprint(f"  Max sigma - mean: {np.mean(max_sigmas):.2f}, max: {np.max(max_sigmas):.1f}")
                aprint(f"  Very elongated (ratio > 5): {np.sum(aspect_ratios > 5.0)}/{len(aspect_ratios)}")
                aprint(f"  Extremely elongated (ratio > 10): {np.sum(aspect_ratios > 10.0)}/{len(aspect_ratios)}")
                
            except Exception as e:
                aprint(f"❌ Precision approach failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Compare with traditional approach
        with asection("🔄 Traditional Approach Comparison"):
            try:
                params_trad, amps_trad, stats_trad = fit_gaussian_splats(
                    V=V,
                    centers_overcomplete=centers,
                    init_sigma_vox=2.0,
                    n_iters=150,
                    lr=0.1,
                    l1_amp=0.001,
                    sigma_min_diag=[0.5, 0.5],
                    sigma_max_diag=None,
                    verbose=False,
                    use_precision_parameterization=False,
                )
                
                # Analyze traditional shapes
                L_matrices = unpack_tril(params_trad[:, 2:], 2)
                
                trad_aspect_ratios = []
                trad_max_sigmas = []
                
                for i, L in enumerate(L_matrices):
                    if amps_trad[i] > 0.1:
                        cov = L @ L.T
                        eigenvals, _ = np.linalg.eigh(cov)
                        eigenvals = np.sort(eigenvals)[::-1]
                        
                        sigmas = np.sqrt(eigenvals)
                        aspect_ratio = sigmas[0] / sigmas[1] if sigmas[1] > 1e-6 else float('inf')
                        
                        trad_aspect_ratios.append(aspect_ratio)
                        trad_max_sigmas.append(sigmas[0])
                
                trad_aspect_ratios = np.array(trad_aspect_ratios)
                trad_max_sigmas = np.array(trad_max_sigmas)
                
                aprint(f"Traditional Shape Statistics:")
                aprint(f"  Count: {len(trad_aspect_ratios)}")
                aprint(f"  Aspect ratios - mean: {np.mean(trad_aspect_ratios):.2f}, max: {np.max(trad_aspect_ratios):.1f}")
                aprint(f"  Max sigma - mean: {np.mean(trad_max_sigmas):.2f}, max: {np.max(trad_max_sigmas):.1f}")
                aprint(f"  Very elongated (ratio > 5): {np.sum(trad_aspect_ratios > 5.0)}/{len(trad_aspect_ratios)}")
                
            except Exception as e:
                aprint(f"❌ Traditional approach failed: {e}")
        
        # Analysis and recommendations
        aprint(f"\n=== ELONGATION ANALYSIS ===")
        if len(aspect_ratios) > 0 and len(trad_aspect_ratios) > 0:
            prec_elongated = np.sum(aspect_ratios > 5.0) / len(aspect_ratios)
            trad_elongated = np.sum(trad_aspect_ratios > 5.0) / len(trad_aspect_ratios)
            
            aprint(f"Precision elongated fraction: {prec_elongated:.2%}")
            aprint(f"Traditional elongated fraction: {trad_elongated:.2%}")
            
            if prec_elongated > trad_elongated * 2:
                aprint("💥 PRECISION APPROACH CREATES MORE ELONGATED SHAPES!")
                aprint("Possible causes:")
                aprint("  1. Off-diagonal elements unconstrained")
                aprint("  2. Different optimization landscape favors extreme ratios")
                aprint("  3. Numerical issues in precision matrix parameterization")
                aprint("  4. Need aspect ratio constraints or regularization")
            else:
                aprint("✅ Both approaches have similar elongation patterns")

if __name__ == "__main__":
    debug_elongation_issue()