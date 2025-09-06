#!/usr/bin/env python3
"""
Test truncation logic differences between manual and renderer.
"""

import numpy as np
import torch
from arbol import aprint, asection

def test_truncation_differences():
    """Compare truncation logic between manual calculation and renderer."""
    
    with asection("🎯 Truncation Logic Analysis"):
        
        # Simple test case
        center = np.array([10.0, 10.0])
        sigma = 3.0
        truncate = 3.0
        shape = (21, 21)
        
        aprint(f"Center: {center}")
        aprint(f"Sigma: {sigma}")
        aprint(f"Truncate: {truncate}")
        
        # Method 1: Manual radius-based truncation
        aprint("\n=== Manual Radius-based Truncation ===")
        y = np.arange(shape[0]).astype(np.float32)
        x = np.arange(shape[1]).astype(np.float32)
        Y, X = np.meshgrid(y, x, indexing='ij')
        
        dy = Y - center[0]
        dx = X - center[1]
        r_sq = (dy ** 2 + dx ** 2) / (sigma ** 2)
        
        manual_mask = r_sq <= (truncate ** 2)
        manual_count = np.sum(manual_mask)
        
        aprint(f"Manual truncation: pixels within r² ≤ {truncate**2}")
        aprint(f"Manual pixel count: {manual_count}")
        
        # Method 2: AABB-based truncation (as used by renderer)
        aprint("\n=== AABB-based Truncation (Renderer) ===")
        # From renderer logic:
        # sigma_diag = [σ², σ²] 
        # r = ceil(truncate * sqrt(sigma_diag)) = [ceil(3.0 * σ), ceil(3.0 * σ)]
        # AABB: [center - r, center + r + 1)
        
        sigma_diag = np.array([sigma**2, sigma**2])  # Diagonal of covariance matrix
        r = np.ceil(truncate * np.sqrt(sigma_diag)).astype(int)
        
        aprint(f"Sigma diagonal: {sigma_diag}")
        aprint(f"AABB radius per axis: {r}")
        
        lo = np.clip((center - r).astype(int), 0, None)
        hi = np.minimum((center + r).astype(int) + 1, shape)
        
        aprint(f"AABB bounds: [{lo[0]}:{hi[0]}, {lo[1]}:{hi[1]}]")
        
        # Count pixels in AABB
        aabb_count = (hi[0] - lo[0]) * (hi[1] - lo[1])
        aprint(f"AABB pixel count: {aabb_count}")
        
        # Create AABB mask
        aabb_mask = np.zeros(shape, dtype=bool)
        aabb_mask[lo[0]:hi[0], lo[1]:hi[1]] = True
        
        # Compare masks
        aprint(f"\n=== Mask Comparison ===")
        aprint(f"Manual mask pixels: {np.sum(manual_mask)}")
        aprint(f"AABB mask pixels: {np.sum(aabb_mask)}")
        
        # Pixels in AABB but not in manual radius
        extra_pixels = aabb_mask & ~manual_mask
        aprint(f"Extra pixels in AABB (outside radius): {np.sum(extra_pixels)}")
        
        # Pixels in manual radius but not in AABB (should be 0 if AABB is conservative)
        missing_pixels = manual_mask & ~aabb_mask
        aprint(f"Missing pixels in AABB (inside radius): {np.sum(missing_pixels)}")
        
        # Show some example extra pixels
        if np.sum(extra_pixels) > 0:
            extra_coords = np.where(extra_pixels)
            aprint(f"First few extra pixel coordinates:")
            for i in range(min(5, len(extra_coords[0]))):
                py, px = extra_coords[0][i], extra_coords[1][i]
                dist_sq = ((py - center[0])**2 + (px - center[1])**2) / (sigma**2)
                aprint(f"  ({py}, {px}): distance² = {dist_sq:.3f} (threshold: {truncate**2})")
        
        # Test actual rendering with both approaches
        aprint(f"\n=== Rendering Comparison ===")
        
        amplitude = 1.0
        
        # Manual calculation (radius-based truncation)
        manual_result = np.zeros(shape, dtype=np.float32)
        manual_result[manual_mask] = amplitude * np.exp(-0.5 * r_sq[manual_mask])
        
        # Simulate AABB calculation (what renderer actually does)
        aabb_result = np.zeros(shape, dtype=np.float32)
        
        # Only process pixels in AABB
        y_aabb = np.arange(lo[0], hi[0]).astype(np.float32)
        x_aabb = np.arange(lo[1], hi[1]).astype(np.float32)
        Y_aabb, X_aabb = np.meshgrid(y_aabb, x_aabb, indexing='ij')
        
        dy_aabb = Y_aabb - center[0]
        dx_aabb = X_aabb - center[1]
        r_sq_aabb = (dy_aabb ** 2 + dx_aabb ** 2) / (sigma ** 2)
        
        aabb_result[lo[0]:hi[0], lo[1]:hi[1]] = amplitude * np.exp(-0.5 * r_sq_aabb)
        
        aprint(f"Manual result sum: {manual_result.sum():.6f}")
        aprint(f"AABB result sum: {aabb_result.sum():.6f}")
        aprint(f"Difference: {abs(manual_result.sum() - aabb_result.sum()):.6f}")
        
        # Test with actual renderer
        from luxar.gsplats.models.gsplats.gsplats_render import render_gaussians_full_numpy
        from luxar.gsplats.utils.trils import pack_tril
        
        # Create parameters for renderer
        L = np.array([[[sigma, 0.0], [0.0, sigma]]], dtype=np.float32)
        d = 2
        tril_size = d * (d + 1) // 2
        params = np.zeros((1, d + tril_size), dtype=np.float32)
        params[:, :d] = center.reshape(1, -1)
        params[:, d:] = pack_tril(L)
        amps = np.array([amplitude], dtype=np.float32)
        
        renderer_result = render_gaussians_full_numpy(shape, params, amps, truncate=truncate)
        
        aprint(f"Renderer result sum: {renderer_result.sum():.6f}")
        aprint(f"AABB vs Renderer difference: {abs(aabb_result.sum() - renderer_result.sum()):.8f}")
        
        # This should be very close if my AABB logic matches the renderer
        if abs(aabb_result.sum() - renderer_result.sum()) < 1e-6:
            aprint("✅ AABB simulation matches renderer!")
        else:
            aprint("❌ AABB simulation differs from renderer")

if __name__ == "__main__":
    test_truncation_differences()