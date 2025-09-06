#!/usr/bin/env python3
"""
Compare optimized precision renderer with traditional approach.
"""

import numpy as np
import torch
from arbol import aprint

# Create simple test case
torch.manual_seed(42)
np.random.seed(42)

# Simple 2D case with known parameters
shape = (20, 20)
centers = torch.tensor([[10.0, 10.0], [5.0, 15.0]], dtype=torch.float32)  # 2 Gaussians
amps = torch.tensor([1.0, 0.8], dtype=torch.float32)

# Traditional covariance Cholesky (lower triangular)
L_cov = torch.tensor([
    [[2.0, 0.0], [1.0, 1.5]],  # Lower triangular
    [[1.5, 0.0], [0.5, 2.0]]
], dtype=torch.float32)

aprint("=== Test Setup ===")
aprint(f"Image shape: {shape}")
aprint(f"Gaussians: {len(centers)}")
aprint(f"Centers:\n{centers}")
aprint(f"Amplitudes: {amps}")
aprint(f"Covariance Cholesky factors:\n{L_cov}")

# Test traditional renderer
aprint("\n=== Traditional Renderer ===")
try:
    from luxar.gsplats.models.gsplats.gsplats_render import render_gaussians_full_numpy
    from luxar.gsplats.utils.trils import pack_tril
    
    # Convert to traditional format: [centers, packed_triangular]
    centers_np = centers.numpy()  # (N, d)
    L_cov_np = L_cov.numpy()     # (N, d, d)
    
    N, d = centers_np.shape
    tril_size = d * (d + 1) // 2
    params_traditional = np.zeros((N, d + tril_size), dtype=np.float32)
    params_traditional[:, :d] = centers_np
    
    packed = pack_tril(L_cov_np)  # Pack lower triangular
    params_traditional[:, d:] = packed
    
    result_traditional = render_gaussians_full_numpy(shape, params_traditional, amps.numpy(), truncate=3.0)
    result_traditional = torch.from_numpy(result_traditional)
    aprint(f"Traditional result shape: {result_traditional.shape}")
    aprint(f"Traditional result range: [{result_traditional.min():.6f}, {result_traditional.max():.6f}]")
    aprint(f"Traditional result sum: {result_traditional.sum():.6f}")
except Exception as e:
    aprint(f"Traditional renderer failed: {e}")
    result_traditional = None

# Convert to precision parameterization
aprint("\n=== Converting to Precision Parameterization ===")
# Σ = L_cov @ L_cov^T, Λ = Σ^{-1}, U^T @ U = Λ
Sigma = L_cov @ L_cov.transpose(-1, -2)
Lambda = torch.inverse(Sigma)
U = torch.linalg.cholesky(Lambda).transpose(-1, -2)  # Upper triangular

aprint(f"Covariance matrices:\n{Sigma}")
aprint(f"Precision matrices:\n{Lambda}")  
aprint(f"Precision Cholesky (upper triangular):\n{U}")

# Test optimized precision renderer
aprint("\n=== Optimized Precision Renderer ===")
try:
    from luxar.gsplats.models.gsplats.gsplats_precision_render_optimized import render_gaussians_precision_optimized
    result_optimized = render_gaussians_precision_optimized(shape, centers, U, amps, truncate=3.0)
    aprint(f"Optimized result shape: {result_optimized.shape}")
    aprint(f"Optimized result range: [{result_optimized.min():.6f}, {result_optimized.max():.6f}]")
    aprint(f"Optimized result sum: {result_optimized.sum():.6f}")
except Exception as e:
    aprint(f"Optimized renderer failed: {e}")
    result_optimized = None

# Compare results
if result_traditional is not None and result_optimized is not None:
    aprint("\n=== Comparison ===")
    diff = torch.abs(result_traditional - result_optimized)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    rel_error = torch.norm(diff) / torch.norm(result_traditional)
    
    aprint(f"Max absolute difference: {max_diff:.8f}")
    aprint(f"Mean absolute difference: {mean_diff:.8f}")
    aprint(f"Relative L2 error: {rel_error:.8f}")
    
    if rel_error < 1e-5:
        aprint("✅ Renderers produce nearly identical results!")
    else:
        aprint("❌ Renderers produce different results!")
        
        # Find where the largest differences occur
        max_idx = torch.unravel_index(torch.argmax(diff), diff.shape)
        aprint(f"Largest difference at pixel {max_idx}: traditional={result_traditional[max_idx]:.6f}, optimized={result_optimized[max_idx]:.6f}")

# Test specific mathematical operations
aprint("\n=== Testing Core Math ===")
test_point = torch.tensor([12.0, 8.0])  # Random test point
gaussian_idx = 0

aprint(f"Test point: {test_point}")
aprint(f"Gaussian {gaussian_idx} center: {centers[gaussian_idx]}")

delta = test_point - centers[gaussian_idx]
aprint(f"Delta: {delta}")

# Traditional: exp(-0.5 * delta^T @ Σ^{-1} @ delta)
quad_traditional = delta @ Lambda[gaussian_idx] @ delta
g_traditional = amps[gaussian_idx] * torch.exp(-0.5 * quad_traditional)

# Precision: exp(-0.5 * ||U @ delta||²)
y = U[gaussian_idx] @ delta
quad_precision = torch.sum(y * y)
g_precision = amps[gaussian_idx] * torch.exp(-0.5 * quad_precision)

aprint(f"Traditional quadratic form: {quad_traditional:.6f}")
aprint(f"Precision quadratic form: {quad_precision:.6f}")
aprint(f"Traditional Gaussian value: {g_traditional:.6f}")
aprint(f"Precision Gaussian value: {g_precision:.6f}")
aprint(f"Gaussian value error: {abs(g_traditional - g_precision):.8f}")

if abs(g_traditional - g_precision) < 1e-6:
    aprint("✅ Core math is correct!")
else:
    aprint("❌ Core math has errors!")