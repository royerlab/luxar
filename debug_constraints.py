#!/usr/bin/env python3
"""
Debug what constraints are actually being applied in precision optimization.
"""

import numpy as np
import torch
from arbol import aprint, asection

def debug_precision_constraints():
    """Debug precision matrix constraints."""
    
    with asection("🔧 Precision Constraints Debug"):
        
        # Test what constraints are passed to precision fitter
        aprint("=== Testing Demo Parameters ===")
        
        sigma_min_diag = [0.6, 0.6]  # From demo
        sigma_max_diag = None
        
        # Conversion logic from fit_gsplats.py
        precision_min_diag = None
        precision_max_diag = None
        
        if sigma_min_diag is not None:
            precision_max_diag = [1.0 / max(s, 1e-3) for s in sigma_min_diag]
        if sigma_max_diag is not None:
            precision_min_diag = [1.0 / max(s, 1e-3) for s in sigma_max_diag]
            
        aprint(f"sigma_min_diag: {sigma_min_diag}")
        aprint(f"sigma_max_diag: {sigma_max_diag}")
        aprint(f"precision_min_diag: {precision_min_diag}")  
        aprint(f"precision_max_diag: {precision_max_diag}")
        
        # Test default behavior when no sigma constraints
        aprint(f"\n=== Testing No Sigma Constraints ===")
        precision_min_diag_default = [0.5, 0.5]  # From fit_gsplats_precision_optimized.py line 168
        precision_max_diag_default = None
        
        aprint(f"Default precision_min_diag: {precision_min_diag_default}")
        aprint(f"Default precision_max_diag: {precision_max_diag_default}")
        
        # What do these constraints mean in terms of sigma?
        aprint(f"\n=== Constraint Implications ===")
        
        if precision_max_diag:
            min_sigma_from_max_precision = [1.0/p for p in precision_max_diag]
            aprint(f"precision_max_diag {precision_max_diag} → min σ = {min_sigma_from_max_precision}")
            
        if precision_min_diag_default:
            max_sigma_from_min_precision = [1.0/p for p in precision_min_diag_default]
            aprint(f"precision_min_diag {precision_min_diag_default} → max σ = {max_sigma_from_min_precision}")
        
        # Test what happens during precision model creation
        aprint(f"\n=== Testing Precision Model Constraints ===")
        
        # Simulate model creation
        shape = (32, 32)
        centers = np.array([[16.0, 16.0]], dtype=np.float32)
        U0 = np.array([[[0.4, 0.0], [0.0, 0.4]]], dtype=np.float32)  # σ = 2.5 (precision = 0.4)
        amps0 = np.array([0.5])
        
        try:
            from luxar.gsplats.models.gsplats.gsplat_precision_model_optimized import GaussianSplatPrecisionModelOptimized
            
            model = GaussianSplatPrecisionModelOptimized(
                shape=shape,
                centers0=centers,
                U0=U0,
                amps0=amps0,
                precision_min_diag=precision_min_diag_default,  # [0.5, 0.5]
                precision_max_diag=precision_max_diag,          # [1.67, 1.67] from demo
                truncate=3.0,
                device=torch.device('cpu'),
            )
            
            aprint(f"Model created successfully")
            aprint(f"Model precision_min_diag: {model.precision_min_diag}")
            aprint(f"Model precision_max_diag: {model.precision_max_diag}")
            
            # Test current parameters
            centers_out, U_out, amps_out = model.current_params()
            aprint(f"Initial U diagonal: {torch.diag(U_out[0])}")
            
            # What sigma does this correspond to?
            precision_diag = torch.diag(U_out[0]) ** 2
            sigma_from_precision = 1.0 / torch.sqrt(precision_diag)
            aprint(f"Initial σ from precision: {sigma_from_precision}")
            
            # Test precision matrices
            Lambda = model.get_precision_matrices()
            Sigma = model.get_covariance_matrices()
            sigma_from_cov = torch.sqrt(torch.diag(Sigma[0]))
            aprint(f"Initial σ from covariance: {sigma_from_cov}")
            
        except Exception as e:
            aprint(f"Model creation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_precision_constraints()