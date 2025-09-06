#!/usr/bin/env python3
"""
Add anisotropy constraints to prevent extreme elongation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from arbol import aprint, asection

def design_anisotropy_constraint():
    """Design different approaches to constrain anisotropy."""
    
    with asection("🔧 Anisotropy Constraint Design"):
        
        aprint("=== Approach 1: Eigenvalue Ratio Constraint ===")
        aprint("Constrain λ_max / λ_min ≤ max_aspect_ratio²")
        aprint("For precision matrix Λ = U^T @ U")
        aprint("Can be implemented as regularization term in loss")
        
        aprint("\n=== Approach 2: Off-diagonal Magnitude Constraint ===")
        aprint("Limit |U[i,j]| for i≠j relative to diagonal elements")
        aprint("Simple to implement in precision parameterization")
        
        aprint("\n=== Approach 3: Condition Number Constraint ===")
        aprint("Constrain cond(Σ) = λ_max / λ_min ≤ max_condition")
        aprint("Most mathematically principled")
        
        # Test implementation of Approach 2 (simplest)
        aprint("\n=== Testing Off-diagonal Constraint Implementation ===")
        
        def constrain_precision_anisotropy(U, max_off_diag_ratio=0.5):
            """
            Constrain off-diagonal elements relative to diagonal.
            
            Parameters
            ----------
            U : torch.Tensor, shape (N, d, d)
                Upper triangular precision Cholesky matrices
            max_off_diag_ratio : float
                Maximum |U[i,j]| / U[i,i] for i≠j
                
            Returns
            -------
            torch.Tensor
                Constrained U matrices
            """
            N, d, _ = U.shape
            U_constrained = U.clone()
            
            for i in range(d):
                for j in range(i+1, d):  # Upper triangular off-diagonals
                    # Constrain |U[i,j]| ≤ max_off_diag_ratio * U[i,i]
                    max_val = max_off_diag_ratio * U_constrained[:, i, i]
                    U_constrained[:, i, j] = torch.clamp(
                        U_constrained[:, i, j], 
                        -max_val, 
                        max_val
                    )
            
            return U_constrained
        
        # Test with example matrices
        torch.manual_seed(42)
        U_test = torch.randn(3, 2, 2)  # 3 test matrices
        U_test = torch.triu(U_test)    # Make upper triangular
        U_test[:, 0, 0] = torch.abs(U_test[:, 0, 0]) + 0.1  # Positive diagonal
        U_test[:, 1, 1] = torch.abs(U_test[:, 1, 1]) + 0.1
        
        aprint(f"Original U matrices:")
        for i in range(3):
            aprint(f"  U[{i}] = \n{U_test[i]}")
            
            # Compute aspect ratio
            Lambda = U_test[i].T @ U_test[i]
            Sigma = torch.inverse(Lambda)
            eigenvals, _ = torch.linalg.eigh(Sigma)
            eigenvals = torch.sort(eigenvals, descending=True)[0]
            aspect_ratio = torch.sqrt(eigenvals[0] / eigenvals[1])
            aprint(f"    Aspect ratio: {aspect_ratio:.2f}")
        
        # Apply constraint
        U_constrained = constrain_precision_anisotropy(U_test, max_off_diag_ratio=0.3)
        
        aprint(f"\nConstrained U matrices (max_off_diag_ratio=0.3):")
        for i in range(3):
            aprint(f"  U[{i}] = \n{U_constrained[i]}")
            
            # Compute new aspect ratio
            Lambda = U_constrained[i].T @ U_constrained[i]
            Sigma = torch.inverse(Lambda)
            eigenvals, _ = torch.linalg.eigh(Sigma)
            eigenvals = torch.sort(eigenvals, descending=True)[0]
            aspect_ratio = torch.sqrt(eigenvals[0] / eigenvals[1])
            aprint(f"    New aspect ratio: {aspect_ratio:.2f}")

def add_anisotropy_constraint_to_model():
    """Show how to integrate anisotropy constraint into the precision model."""
    
    with asection("🔧 Integration into Precision Model"):
        
        aprint("The anisotropy constraint can be added to GaussianSplatPrecisionModelOptimized")
        aprint("in the _build_U() method:")
        
        code_example = '''
def _build_U(self, max_anisotropy_ratio=3.0) -> torch.Tensor:
    """
    Build U with optional anisotropy constraint.
    
    Parameters
    ----------
    max_anisotropy_ratio : float
        Maximum aspect ratio (σ_max / σ_min) allowed.
        Controls off-diagonal constraint: max_off_diag = sqrt(max_ratio² - 1) / max_ratio
    """
    N, d = self.raw_U_diag.shape
    diag = self.precision_min_diag + F.softplus(self.raw_U_diag)
    
    if self.precision_max_diag is not None:
        diag = torch.minimum(diag, self.precision_max_diag)

    U = torch.zeros((N, d, d), dtype=torch.float32, device=diag.device)
    
    # Fill diagonal elements
    for i in range(d):
        U[:, i, i] = diag[:, i]

    # Fill off-diagonal elements with anisotropy constraint
    k = 0
    max_off_diag_ratio = np.sqrt(max_anisotropy_ratio**2 - 1) / max_anisotropy_ratio
    
    for i in range(d):
        for j in range(i+1, d):
            # Constrain relative to geometric mean of diagonal elements
            diag_geometric_mean = torch.sqrt(diag[:, i] * diag[:, j])
            max_off_val = max_off_diag_ratio * diag_geometric_mean
            
            U[:, i, j] = torch.clamp(
                self.U_off[:, k],
                -max_off_val,
                max_off_val
            )
            k += 1
    return U
        '''
        
        aprint(code_example)
        
        aprint(f"\nAlternatively, add it as a parameter to the fitter:")
        aprint(f"fit_gaussian_splats(..., max_aspect_ratio=5.0)")

if __name__ == "__main__":
    design_anisotropy_constraint()
    add_anisotropy_constraint_to_model()