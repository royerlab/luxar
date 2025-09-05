#!/usr/bin/env python
"""
Quick benchmark to verify optimization improvements are working.
"""

import time

import numpy as np
from arbol import Arbol, aprint, asection

from luxar.gsplats.candidates import find_candidates_overcomplete_nd
from luxar.gsplats.fit_gsplats import GaussianSplatFitter

# Setup
Arbol.max_depth = 2


def create_test_data(size=64):
    """Create simple test data."""
    data = np.zeros((size, size), dtype=np.float32)
    # Add a few Gaussians
    for _ in range(3):
        y = np.random.randint(10, size-10)
        x = np.random.randint(10, size-10)
        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        data += 0.8 * np.exp(-((yy-y)**2 + (xx-x)**2) / (2 * 5**2))
    return np.clip(data, 0, 1)


def main():
    aprint("Gaussian Splatting Performance Verification")
    aprint("=" * 50)

    # Create test data
    V = create_test_data(64)
    candidates = find_candidates_overcomplete_nd(
        V, scales=(1.0, 2.0, 3.0), peaks_per_scale=30
    )
    aprint(f"Test image: {V.shape}, Candidates: {len(candidates)}")

    # Test 1: With optimizations (default)
    with asection("WITH Optimizations (early_stopping=True)"):
        fitter = GaussianSplatFitter()
        start = time.time()
        params1, amps1, stats1 = fitter.fit(
            V,
            centers_overcomplete=candidates,
            n_iters=150,
            early_stopping=True,
            early_stop_patience=15,
            verbose=False,
        )
        time1 = time.time() - start
        aprint(f"Time: {time1:.2f}s")
        aprint(f"Iterations: {stats1['iterations']}/150")
        aprint(f"Converged: {stats1['converged']}")

    # Test 2: Without optimizations
    with asection("WITHOUT Optimizations (early_stopping=False)"):
        start = time.time()
        params2, amps2, stats2 = fitter.fit(
            V,
            centers_overcomplete=candidates,
            n_iters=150,
            early_stopping=False,
            verbose=False,
        )
        time2 = time.time() - start
        aprint(f"Time: {time2:.2f}s")
        aprint(f"Iterations: {stats2['iterations']}/150")

    # Summary
    aprint("\n" + "=" * 50)
    aprint("SUMMARY:")
    if stats1['iterations'] < stats2['iterations']:
        speedup = time2 / time1
        saved = stats2['iterations'] - stats1['iterations']
        aprint("✓ Optimizations working!")
        aprint(f"  Speedup: {speedup:.1f}x")
        aprint(f"  Iterations saved: {saved} ({saved/stats2['iterations']*100:.0f}%)")
    else:
        aprint("Note: Early stopping didn't trigger on this simple case")
        aprint("(This can happen when the optimization converges very late)")
    aprint("=" * 50)


if __name__ == "__main__":
    main()
