#!/usr/bin/env python3
"""
Test anisotropy constraint with max aspect ratio of 2.0.
"""

import numpy as np
from arbol import aprint, asection

def test_anisotropy_constraint():
    """Test that anisotropy constraint prevents extreme elongation."""
    
    with asection("🔧 Anisotropy Constraint Test"):
        
        # Create test data that might naturally lead to elongated shapes
        from skimage import data, filters
        
        # Create elongated blobs by stretching
        blobs = data.binary_blobs(length=64, blob_size_fraction=0.1, n_dim=2, volume_fraction=0.15, rng=42)
        
        # Create elongated structures by applying anisotropic Gaussian filter
        V_base = filters.gaussian(blobs.astype(float), sigma=(1.0, 3.0))  # Stretch in y-direction
        V = (V_base / V_base.max()).astype(np.float32)
        
        aprint(f"Test image with elongated structures")
        aprint(f"Shape: {V.shape}, Range: [{V.min():.3f}, {V.max():.3f}]")
        
        # Generate candidates
        from luxar.gsplats.candidates import find_candidates_overcomplete_nd
        centers = find_candidates_overcomplete_nd(
            V,
            spacing=(4, 4),
            peaks_per_scale=40,
            add_intensity_grid=False,
        )
        
        aprint(f"Using {len(centers)} candidate centers")
        
        # Test WITHOUT anisotropy constraint (allow extreme elongation)
        with asection("❌ Without Anisotropy Constraint"):
            try:
                from luxar.gsplats.fit_gsplats import fit_gaussian_splats
                from luxar.gsplats.utils.trils import unpack_tril
                
                params_unconstrained, amps_unconstrained, stats_unconstrained = fit_gaussian_splats(
                    V=V,
                    centers_overcomplete=centers,
                    init_sigma_vox=2.0,
                    n_iters=100,
                    lr=0.1,
                    l1_amp=0.001,
                    max_aspect_ratio=None,  # No constraint
                    verbose=False,
                )
                
                # Analyze shapes
                L_matrices = unpack_tril(params_unconstrained[:, 2:], 2)
                
                aspect_ratios = []
                for i, L in enumerate(L_matrices):
                    if amps_unconstrained[i] > 0.05:
                        cov = L @ L.T
                        eigenvals, _ = np.linalg.eigh(cov)
                        eigenvals = np.sort(eigenvals)[::-1]
                        sigmas = np.sqrt(eigenvals)
                        aspect_ratio = sigmas[0] / sigmas[1] if sigmas[1] > 1e-6 else float('inf')
                        aspect_ratios.append(aspect_ratio)
                
                aspect_ratios = np.array(aspect_ratios)
                
                aprint(f"Results without constraint:")
                aprint(f"  Final loss: {stats_unconstrained['final_loss']:.6f}")
                aprint(f"  Significant Gaussians: {len(aspect_ratios)}")
                aprint(f"  Aspect ratios - mean: {np.mean(aspect_ratios):.2f}, max: {np.max(aspect_ratios):.1f}")
                aprint(f"  Very elongated (>5:1): {np.sum(aspect_ratios > 5.0)}")
                aprint(f"  Extremely elongated (>10:1): {np.sum(aspect_ratios > 10.0)}")
                
            except Exception as e:
                aprint(f"❌ Unconstrained test failed: {e}")
                aspect_ratios = np.array([])
        
        # Test WITH anisotropy constraint (max 2:1 ratio)
        with asection("✅ With Anisotropy Constraint (max 2:1)"):
            try:
                params_constrained, amps_constrained, stats_constrained = fit_gaussian_splats(
                    V=V,
                    centers_overcomplete=centers,
                    init_sigma_vox=2.0,
                    n_iters=100,
                    lr=0.1,
                    l1_amp=0.001,
                    max_aspect_ratio=2.0,  # 2:1 constraint
                    verbose=False,
                )
                
                # Analyze shapes
                L_matrices = unpack_tril(params_constrained[:, 2:], 2)
                
                constrained_aspect_ratios = []
                for i, L in enumerate(L_matrices):
                    if amps_constrained[i] > 0.05:
                        cov = L @ L.T
                        eigenvals, _ = np.linalg.eigh(cov)
                        eigenvals = np.sort(eigenvals)[::-1]
                        sigmas = np.sqrt(eigenvals)
                        aspect_ratio = sigmas[0] / sigmas[1] if sigmas[1] > 1e-6 else float('inf')
                        constrained_aspect_ratios.append(aspect_ratio)
                
                constrained_aspect_ratios = np.array(constrained_aspect_ratios)
                
                aprint(f"Results with constraint (max 2:1):")
                aprint(f"  Final loss: {stats_constrained['final_loss']:.6f}")
                aprint(f"  Significant Gaussians: {len(constrained_aspect_ratios)}")
                aprint(f"  Aspect ratios - mean: {np.mean(constrained_aspect_ratios):.2f}, max: {np.max(constrained_aspect_ratios):.1f}")
                aprint(f"  Very elongated (>5:1): {np.sum(constrained_aspect_ratios > 5.0)}")
                aprint(f"  Extremely elongated (>10:1): {np.sum(constrained_aspect_ratios > 10.0)}")
                aprint(f"  Violating constraint (>2:1): {np.sum(constrained_aspect_ratios > 2.1)}")  # Small tolerance
                
            except Exception as e:
                aprint(f"❌ Constrained test failed: {e}")
                import traceback
                traceback.print_exc()
                constrained_aspect_ratios = np.array([])
        
        # Analysis
        if len(aspect_ratios) > 0 and len(constrained_aspect_ratios) > 0:
            with asection("📊 Constraint Effectiveness"):
                
                unconstrained_violations = np.sum(aspect_ratios > 2.1)
                constrained_violations = np.sum(constrained_aspect_ratios > 2.1)
                
                aprint(f"Constraint violations (>2.1:1):")
                aprint(f"  Without constraint: {unconstrained_violations}/{len(aspect_ratios)} ({100*unconstrained_violations/len(aspect_ratios):.1f}%)")
                aprint(f"  With constraint: {constrained_violations}/{len(constrained_aspect_ratios)} ({100*constrained_violations/len(constrained_aspect_ratios):.1f}%)")
                
                if constrained_violations == 0:
                    aprint("✅ Constraint successfully prevents extreme elongation!")
                elif constrained_violations < unconstrained_violations:
                    aprint("✅ Constraint reduces elongation significantly")
                else:
                    aprint("❌ Constraint not working effectively")

if __name__ == "__main__":
    test_anisotropy_constraint()