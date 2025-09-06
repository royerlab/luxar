#!/usr/bin/env python3
"""
Debug why the anisotropy constraint isn't working properly.
"""

import numpy as np
import torch
from arbol import aprint, asection

def debug_constraint_math():
    """Debug the constraint mathematical relationship."""
    
    with asection("🔍 Constraint Math Debug"):
        
        aprint("Testing constraint formula: max_off_diag_ratio = sqrt(R² - 1) / R")
        
        for max_ratio in [1.5, 2.0, 3.0, 5.0]:
            if max_ratio > 1.0:
                max_off_ratio = np.sqrt(max_ratio**2 - 1) / max_ratio
                aprint(f"Max aspect ratio {max_ratio:.1f}:1 → max_off_diag_ratio = {max_off_ratio:.3f}")
            
        aprint(f"\nFor max_aspect_ratio = 2.0:")
        aprint(f"  max_off_diag_ratio = sqrt(4-1)/2 = sqrt(3)/2 = {np.sqrt(3)/2:.3f}")
        
        # Test with actual matrix
        aprint(f"\n=== Testing with example matrix ===")
        
        # Create a precision matrix that should give aspect ratio ~2
        diag1, diag2 = 1.0, 1.0  # Equal diagonal elements
        off_diag = 0.5  # Off-diagonal element
        
        U = np.array([[diag1, off_diag], [0.0, diag2]])
        Lambda = U.T @ U
        Sigma = np.linalg.inv(Lambda)
        
        eigenvals, _ = np.linalg.eigh(Sigma)
        eigenvals = np.sort(eigenvals)[::-1]
        sigmas = np.sqrt(eigenvals)
        aspect_ratio = sigmas[0] / sigmas[1]
        
        aprint(f"Test matrix U:")
        aprint(f"  U = [[{diag1}, {off_diag}], [0, {diag2}]]")
        aprint(f"  Off-diagonal ratio: {off_diag / np.sqrt(diag1 * diag2):.3f}")
        aprint(f"  Resulting aspect ratio: {aspect_ratio:.2f}")
        
        # Test constraint formula
        max_allowed_off = (np.sqrt(3)/2) * np.sqrt(diag1 * diag2)
        aprint(f"  Max allowed off-diag for 2:1 ratio: {max_allowed_off:.3f}")
        aprint(f"  Would this be constrained? {off_diag > max_allowed_off}")

def test_constraint_in_model():
    """Test the constraint directly in the precision model."""
    
    with asection("🔧 Model Constraint Test"):
        
        try:
            from luxar.gsplats.models.gsplats.gsplat_precision_model_optimized import GaussianSplatPrecisionModelOptimized
            
            # Create model with known parameters
            shape = (32, 32)
            centers = np.array([[16.0, 16.0]], dtype=np.float32)
            
            # Create U with large off-diagonal that should be constrained
            U0 = np.array([[[1.0, 1.0], [0.0, 1.0]]], dtype=np.float32)  # Large off-diagonal
            amps0 = np.array([0.5])
            precision_min_diag = [0.01, 0.01]
            
            aprint(f"Initial U matrix:")
            aprint(f"  U[0] = \n{U0[0]}")
            
            # Calculate initial aspect ratio
            Lambda0 = U0[0].T @ U0[0]
            Sigma0 = np.linalg.inv(Lambda0)
            eigenvals0, _ = np.linalg.eigh(Sigma0)
            eigenvals0 = np.sort(eigenvals0)[::-1]
            initial_aspect = np.sqrt(eigenvals0[0] / eigenvals0[1])
            aprint(f"  Initial aspect ratio: {initial_aspect:.2f}")
            
            # Test without constraint
            model_unconstrained = GaussianSplatPrecisionModelOptimized(
                shape=shape,
                centers0=centers,
                U0=U0,
                amps0=amps0,
                precision_min_diag=precision_min_diag,
                max_aspect_ratio=None,  # No constraint
                truncate=3.0,
                device=torch.device('cpu'),
            )
            
            centers_out, U_out, amps_out = model_unconstrained.current_params()
            aprint(f"Without constraint:")
            aprint(f"  U[0] = \n{U_out[0].detach()}")
            
            # Calculate aspect ratio
            Lambda_unc = U_out[0].T @ U_out[0]
            Sigma_unc = torch.inverse(Lambda_unc)
            eigenvals_unc, _ = torch.linalg.eigh(Sigma_unc)
            eigenvals_unc = torch.sort(eigenvals_unc, descending=True)[0]
            aspect_unc = torch.sqrt(eigenvals_unc[0] / eigenvals_unc[1])
            aprint(f"  Aspect ratio: {aspect_unc:.2f}")
            
            # Test with constraint
            model_constrained = GaussianSplatPrecisionModelOptimized(
                shape=shape,
                centers0=centers,
                U0=U0,
                amps0=amps0,
                precision_min_diag=precision_min_diag,
                max_aspect_ratio=2.0,  # 2:1 constraint
                truncate=3.0,
                device=torch.device('cpu'),
            )
            
            centers_out2, U_out2, amps_out2 = model_constrained.current_params()
            aprint(f"With 2:1 constraint:")
            aprint(f"  U[0] = \n{U_out2[0].detach()}")
            
            # Calculate aspect ratio
            Lambda_con = U_out2[0].T @ U_out2[0]
            Sigma_con = torch.inverse(Lambda_con)
            eigenvals_con, _ = torch.linalg.eigh(Sigma_con)
            eigenvals_con = torch.sort(eigenvals_con, descending=True)[0]
            aspect_con = torch.sqrt(eigenvals_con[0] / eigenvals_con[1])
            aprint(f"  Aspect ratio: {aspect_con:.2f}")
            
            if aspect_con <= 2.1:  # Small tolerance
                aprint("✅ Constraint is working!")
            else:
                aprint("❌ Constraint is not working properly")
                
        except Exception as e:
            aprint(f"❌ Model test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_constraint_math()
    test_constraint_in_model()