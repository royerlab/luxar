#!/usr/bin/env python
"""
Example: Using Metal Acceleration for Gaussian Splatting

Demonstrates how to use the Metal-accelerated backend for substantial speedup
on Apple Silicon (M-series chips); actual speedup depends on chip and workload.
"""

from __future__ import annotations

import numpy as np
import torch
from arbol import aprint, asection

from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.metal import is_metal_available


def create_synthetic_volume(size: int = 64, n_blobs: int = 5) -> np.ndarray:
    """Create a synthetic test volume with Gaussian blobs."""
    volume = np.zeros((size, size, size), dtype=np.float32)

    np.random.seed(42)
    for _ in range(n_blobs):
        center = np.random.randint(size // 4, 3 * size // 4, size=3)
        sigma = np.random.uniform(3, 6)
        amp = np.random.uniform(0.5, 1.0)

        z, y, x = np.ogrid[:size, :size, :size]
        dist_sq = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
        volume += amp * np.exp(-dist_sq / (2 * sigma**2))

    return volume


def main():
    """Run example with Metal acceleration."""
    with asection("Metal Acceleration Example"):
        # Check if Metal is available
        if is_metal_available():
            aprint("✓ Metal backend available")
            aprint(
                "  Will use Metal acceleration for substantial speedup (chip-dependent)!"
            )
        else:
            aprint("✗ Metal backend not available")
            aprint("  Using CPU PyTorch (slower but works everywhere)")

        # Create synthetic data
        with asection("Creating synthetic volume"):
            volume = create_synthetic_volume(size=64, n_blobs=5)
            aprint(f"Volume shape: {volume.shape}")
            aprint(f"Volume range: [{volume.min():.3f}, {volume.max():.3f}]")

        # Fit with Metal (if available)
        device = (
            "mps"
            if is_metal_available() and torch.backends.mps.is_available()
            else "cpu"
        )
        backend = "Metal" if device == "mps" else "CPU"

        with asection(f"Fitting with {backend} backend"):
            result = fit_gaussian_splats(
                volume,
                seeds=200,  # Number of splats
                n_iters=100,  # Iterations
                device=device,  # 'mps' selects Metal automatically on Apple Silicon
                verbose=True,
            )

            aprint("✓ Fitting complete!")
            aprint(f"  Final splats: {result.centers.shape[0]}")
            aprint(f"  Optimization time: {result.stats['time_seconds']:.2f}s")
            aprint(f"  Iterations: {result.stats['iterations']}")
            aprint(f"  Converged: {result.stats.get('converged', 'N/A')}")

        # Performance note
        if device == "mps":
            aprint(
                "\n💡 Note: Metal acceleration provided substantial speedup over CPU!"
            )
            aprint("   For even larger volumes (128³+), expect even better speedup.")
        else:
            aprint("\n💡 Note: To enable Metal acceleration:")
            aprint("   1. Use macOS with Apple Silicon (M1/M2/M3/M4)")
            aprint(
                "   2. Install Metal backend: cd metal && python setup.py build_ext --inplace"
            )
            aprint("   3. Use device='mps' when calling fit_gaussian_splats()")


if __name__ == "__main__":
    main()
