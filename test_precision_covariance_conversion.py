#!/usr/bin/env python3
"""
Test precision-to-covariance conversion mathematical correctness.
"""

import numpy as np
import torch
from arbol import aprint

def test_precision_covariance_conversion():
    """Test that precision->covariance conversion is mathematically correct."""
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    aprint("=== Testing Precision-to-Covariance Conversion ===")
    
    # Create test upper triangular Cholesky factors U
    # For 2D case with 3 test Gaussians
    N, d = 3, 2
    U = torch.zeros(N, d, d, dtype=torch.float32)
    
    # Test case 1: Simple diagonal 
    U[0] = torch.tensor([[2.0, 0.0], [0.0, 1.5]], dtype=torch.float32)
    
    # Test case 2: Upper triangular with off-diagonal
    U[1] = torch.tensor([[1.5, 0.8], [0.0, 2.2]], dtype=torch.float32)
    
    # Test case 3: Another upper triangular
    U[2] = torch.tensor([[3.0, -0.5], [0.0, 1.0]], dtype=torch.float32)
    
    aprint(f"Test upper triangular U matrices:\n{U}")
    
    # Step 1: Compute precision matrices Λ = U^T @ U
    Lambda = U.transpose(-1, -2) @ U
    aprint(f"Precision matrices Λ = U^T @ U:\n{Lambda}")
    
    # Step 2: Compute covariance matrices Σ = Λ^{-1}
    Sigma = torch.inverse(Lambda)
    aprint(f"Covariance matrices Σ = Λ^{-1}:\n{Sigma}")
    
    # Step 3: Compute Cholesky of covariance L_cov such that Σ = L_cov @ L_cov^T
    L_cov = torch.linalg.cholesky(Sigma)
    aprint(f"Covariance Cholesky L_cov:\n{L_cov}")
    
    # Verification 1: Check Σ = L_cov @ L_cov^T
    Sigma_reconstructed = L_cov @ L_cov.transpose(-1, -2)
    sigma_error = torch.norm(Sigma - Sigma_reconstructed)
    aprint(f"Σ reconstruction error: {sigma_error:.8f}")
    
    # Verification 2: Check Λ @ Σ = I
    identity_check = Lambda @ Sigma
    identity_error = torch.norm(identity_check - torch.eye(d).unsqueeze(0))
    aprint(f"Λ @ Σ = I error: {identity_error:.8f}")
    
    # Verification 3: Test the full conversion pipeline
    aprint("\n=== Testing Full Conversion Pipeline ===")
    
    # Simulate the optimized model conversion
    Us_precision = U  # Upper triangular from optimization
    
    # Method 1: Convert to covariance format (as done in optimized fitter)
    Sigma_method1 = torch.inverse(Us_precision.transpose(-1, -2) @ Us_precision)
    Ls_covariance_method1 = torch.linalg.cholesky(Sigma_method1)
    
    # Method 2: Convert to precision format (transpose U to lower triangular)
    Ls_precision_method2 = Us_precision.transpose(-1, -2)  # L = U^T
    
    aprint(f"Method 1 - Covariance Cholesky:\n{Ls_covariance_method1}")
    aprint(f"Method 2 - Precision Cholesky (U^T):\n{Ls_precision_method2}")
    
    # Test: Both methods should produce the same quadratic form for rendering
    test_point = torch.tensor([1.5, 2.0], dtype=torch.float32)  # Random test point
    center = torch.tensor([1.0, 1.0], dtype=torch.float32)      # Random center
    delta = test_point - center
    
    aprint(f"\nTest point: {test_point}, Center: {center}, Delta: {delta}")
    
    for i in range(N):
        # Method 1: Traditional covariance approach 
        # (x-μ)^T @ Σ^{-1} @ (x-μ) = (x-μ)^T @ Λ @ (x-μ)
        quad_traditional = delta @ Lambda[i] @ delta
        
        # Method 2: Direct precision rendering
        # ||U @ (x-μ)||² = ||U @ delta||²
        y = Us_precision[i] @ delta
        quad_precision = torch.sum(y * y)
        
        # Method 3: Using converted covariance Cholesky
        # solve L_cov @ z = delta, then ||z||²
        # But this would require solve_triangular which we want to avoid
        
        aprint(f"Gaussian {i}:")
        aprint(f"  Traditional quad form: {quad_traditional:.6f}")  
        aprint(f"  Precision quad form: {quad_precision:.6f}")
        aprint(f"  Difference: {abs(quad_traditional - quad_precision):.8f}")
        
        if abs(quad_traditional - quad_precision) > 1e-6:
            aprint(f"  ❌ Quadratic forms don't match!")
        else:
            aprint(f"  ✅ Quadratic forms match!")
    
    # Test the pack/unpack operations used in the fitter
    aprint("\n=== Testing Pack/Unpack Operations ===")
    
    try:
        from luxar.gsplats.utils.trils import pack_tril, unpack_tril, tril_size
        
        # Test packing covariance Cholesky
        Ls_np = Ls_covariance_method1.numpy()
        packed_tril = pack_tril(Ls_np)
        aprint(f"Packed triangular shape: {packed_tril.shape}")
        aprint(f"Expected tril_size: {tril_size(d)}")
        
        # Test unpacking
        Ls_unpacked = unpack_tril(packed_tril, d)
        unpack_error = np.linalg.norm(Ls_np - Ls_unpacked)
        aprint(f"Pack/unpack error: {unpack_error:.8f}")
        
        if unpack_error < 1e-6:
            aprint("✅ Pack/unpack operations work correctly!")
        else:
            aprint("❌ Pack/unpack operations have errors!")
            
    except Exception as e:
        aprint(f"Pack/unpack test failed: {e}")
    
    aprint("\n=== Conversion Test Complete ===")
    
    # Summary
    all_tests_pass = (
        sigma_error < 1e-6 and 
        identity_error < 1e-6 and
        all(abs(quad_traditional - quad_precision) < 1e-6 for quad_traditional, quad_precision in [
            (delta @ Lambda[i] @ delta, torch.sum((Us_precision[i] @ delta) ** 2)) for i in range(N)
        ])
    )
    
    if all_tests_pass:
        aprint("🎉 All conversion tests PASSED!")
        return True
    else:
        aprint("💥 Some conversion tests FAILED!")
        return False

if __name__ == "__main__":
    test_precision_covariance_conversion()