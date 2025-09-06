#!/usr/bin/env python3
"""
Test render_gaussians_full_numpy function for mathematical correctness.
"""

import numpy as np
import torch
from arbol import aprint, asection

def test_render_gaussians_full_numpy():
    """Comprehensive test of render_gaussians_full_numpy function."""
    
    with asection("🧮 Testing render_gaussians_full_numpy"):
        
        from luxar.gsplats.models.gsplats.gsplats_render import render_gaussians_full_numpy
        from luxar.gsplats.utils.trils import pack_tril
        
        # Test case 1: Single centered Gaussian
        with asection("Test 1: Single Centered Gaussian"):
            shape = (21, 21)  # Odd dimensions for clear center
            center = np.array([[10.0, 10.0]], dtype=np.float32)  # Exact center
            amplitude = 1.0
            sigma = 3.0
            
            # Create covariance matrix Σ = [[σ², 0], [0, σ²]]
            # Cholesky: L = [[σ, 0], [0, σ]]
            L = np.array([[[sigma, 0.0], [0.0, sigma]]], dtype=np.float32)
            
            # Pack into parameter format
            d = 2
            tril_size = d * (d + 1) // 2
            params = np.zeros((1, d + tril_size), dtype=np.float32)
            params[:, :d] = center
            params[:, d:] = pack_tril(L)
            
            amps = np.array([amplitude], dtype=np.float32)
            
            # Render with function
            result = render_gaussians_full_numpy(shape, params, amps, truncate=3.0)
            
            # Manual calculation for verification
            y = np.arange(shape[0]).astype(np.float32)
            x = np.arange(shape[1]).astype(np.float32)
            Y, X = np.meshgrid(y, x, indexing='ij')
            
            dy = Y - center[0, 0]
            dx = X - center[0, 1]
            r_sq = (dy ** 2 + dx ** 2) / (sigma ** 2)
            
            # Apply truncation manually (same as in renderer)
            mask = r_sq <= (3.0 ** 2)  # truncate=3.0
            manual = np.zeros_like(r_sq)
            manual[mask] = amplitude * np.exp(-0.5 * r_sq[mask])
            
            aprint(f"Rendered result sum: {result.sum():.6f}")
            aprint(f"Manual calculation sum: {manual.sum():.6f}")
            aprint(f"Peak value rendered: {result.max():.6f}")
            aprint(f"Peak value manual: {manual.max():.6f}")
            aprint(f"Peak location rendered: {np.unravel_index(np.argmax(result), result.shape)}")
            aprint(f"Peak location manual: {np.unravel_index(np.argmax(manual), manual.shape)}")
            
            diff = np.abs(result - manual)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            aprint(f"Max absolute difference: {max_diff:.8f}")
            aprint(f"Mean absolute difference: {mean_diff:.8f}")
            
            test1_pass = max_diff < 1e-6
            aprint(f"Test 1 result: {'✅ PASS' if test1_pass else '❌ FAIL'}")
        
        # Test case 2: Off-center Gaussian with correlation
        with asection("Test 2: Correlated Gaussian"):
            shape = (20, 20)
            center = np.array([[6.0, 8.0]], dtype=np.float32)  # Off-center
            amplitude = 0.7
            
            # Correlated covariance matrix
            # Σ = [[4.0, 1.5], [1.5, 2.0]]
            # Compute Cholesky decomposition manually
            sigma_matrix = np.array([[4.0, 1.5], [1.5, 2.0]], dtype=np.float32)
            L_manual = np.linalg.cholesky(sigma_matrix).astype(np.float32)
            L = np.array([L_manual], dtype=np.float32)  # Add batch dimension
            
            aprint(f"Covariance matrix:\n{sigma_matrix}")
            aprint(f"Cholesky factor:\n{L_manual}")
            
            # Pack parameters
            params = np.zeros((1, d + tril_size), dtype=np.float32)
            params[:, :d] = center
            params[:, d:] = pack_tril(L)
            
            amps = np.array([amplitude], dtype=np.float32)
            
            # Render with function
            result = render_gaussians_full_numpy(shape, params, amps, truncate=3.0)
            
            # Manual calculation using full covariance
            y = np.arange(shape[0]).astype(np.float32)
            x = np.arange(shape[1]).astype(np.float32)
            Y, X = np.meshgrid(y, x, indexing='ij')
            
            coords = np.stack([Y.ravel(), X.ravel()], axis=1)  # (H*W, 2)
            deltas = coords - center[0]  # (H*W, 2)
            
            # Compute quadratic form: (x-μ)ᵀ Σ⁻¹ (x-μ)
            sigma_inv = np.linalg.inv(sigma_matrix)
            quad_forms = np.sum(deltas @ sigma_inv * deltas, axis=1)  # (H*W,)
            
            # Apply truncation
            truncate_sq = 3.0 ** 2
            mask = quad_forms <= truncate_sq
            
            manual_flat = np.zeros_like(quad_forms)
            manual_flat[mask] = amplitude * np.exp(-0.5 * quad_forms[mask])
            manual = manual_flat.reshape(shape)
            
            aprint(f"Rendered result sum: {result.sum():.6f}")
            aprint(f"Manual calculation sum: {manual.sum():.6f}")
            aprint(f"Peak value rendered: {result.max():.6f}")
            aprint(f"Peak value manual: {manual.max():.6f}")
            
            diff = np.abs(result - manual)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            aprint(f"Max absolute difference: {max_diff:.8f}")
            aprint(f"Mean absolute difference: {mean_diff:.8f}")
            
            test2_pass = max_diff < 1e-5  # Slightly higher tolerance for more complex case
            aprint(f"Test 2 result: {'✅ PASS' if test2_pass else '❌ FAIL'}")
        
        # Test case 3: Multiple Gaussians
        with asection("Test 3: Multiple Gaussians"):
            shape = (16, 16)
            centers = np.array([[4.0, 4.0], [12.0, 12.0]], dtype=np.float32)
            amps = np.array([0.8, 0.6], dtype=np.float32)
            
            # Two different Gaussians
            L1 = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)  # Isotropic
            L2 = np.array([[1.5, 0.0], [0.8, 1.0]], dtype=np.float32)  # Anisotropic
            Ls = np.array([L1, L2], dtype=np.float32)
            
            # Pack parameters
            N = 2
            params = np.zeros((N, d + tril_size), dtype=np.float32)
            params[:, :d] = centers
            params[:, d:] = pack_tril(Ls)
            
            # Render with function
            result = render_gaussians_full_numpy(shape, params, amps, truncate=3.0)
            
            # Manual calculation: render each Gaussian separately and sum
            manual = np.zeros(shape, dtype=np.float32)
            
            for i in range(N):
                center = centers[i]
                L_i = Ls[i]
                amp = amps[i]
                
                # Single Gaussian parameters
                params_single = np.zeros((1, d + tril_size), dtype=np.float32)
                params_single[:, :d] = center.reshape(1, -1)
                params_single[:, d:] = pack_tril(L_i.reshape(1, d, d))
                amps_single = np.array([amp], dtype=np.float32)
                
                # Render single Gaussian
                single_result = render_gaussians_full_numpy(shape, params_single, amps_single, truncate=3.0)
                manual += single_result
            
            aprint(f"Rendered result sum: {result.sum():.6f}")
            aprint(f"Manual sum (additive): {manual.sum():.6f}")
            aprint(f"Peak value rendered: {result.max():.6f}")
            aprint(f"Peak value manual: {manual.max():.6f}")
            
            diff = np.abs(result - manual)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            aprint(f"Max absolute difference: {max_diff:.8f}")
            aprint(f"Mean absolute difference: {mean_diff:.8f}")
            
            test3_pass = max_diff < 1e-6
            aprint(f"Test 3 result: {'✅ PASS' if test3_pass else '❌ FAIL'}")
        
        # Overall assessment
        overall_pass = test1_pass and test2_pass and test3_pass
        
        aprint(f"\n=== OVERALL ASSESSMENT ===")
        if overall_pass:
            aprint("🎉 render_gaussians_full_numpy is CORRECT!")
            return True
        else:
            aprint("💥 render_gaussians_full_numpy has ISSUES!")
            return False

if __name__ == "__main__":
    success = test_render_gaussians_full_numpy()
    exit(0 if success else 1)