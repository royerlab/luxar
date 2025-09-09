#!/usr/bin/env python
"""
Demo script showing the benefits of dynamic Gaussian splat operations.

This script compares standard fitting vs dynamic operations on a synthetic
test case to demonstrate improvements in convergence speed and final quality.
"""

import time

import numpy as np
from arbol import Arbol, aprint, asection

from luxar.gsplats.candidates import find_candidates_overcomplete_nd
from luxar.gsplats.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians_numpy

# Setup
Arbol.max_depth = 3


def create_test_image(size=64, n_blobs=5, add_noise=False):
    """Create synthetic test image with Gaussian blobs."""
    data = np.zeros((size, size), dtype=np.float32)

    # Add random Gaussian blobs
    for _ in range(n_blobs):
        center_y = np.random.randint(8, size - 8)
        center_x = np.random.randint(8, size - 8)
        sigma = np.random.uniform(2.5, 4.5)
        amplitude = np.random.uniform(0.4, 0.9)

        yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        blob = amplitude * np.exp(-((yy - center_y)**2 + (xx - center_x)**2) / (2 * sigma**2))
        data += blob

    if add_noise:
        data += np.random.normal(0, 0.05, data.shape)

    return np.clip(data, 0, 1)


def benchmark_fitting_methods(image, candidates, n_iters=150, verbose=True):
    """Compare standard vs dynamic fitting approaches."""
    results = {}

    # Standard fitting (baseline)
    with asection("Standard Fitting (Baseline)"):
        start_time = time.time()
        params_std, amps_std, stats_std = fit_gaussian_splats(
            image,
            centers_overcomplete=candidates,
            n_iters=n_iters,
            lr=0.15,
            verbose=verbose,
            early_stopping=True,
            enable_dynamic_ops=False,
        )
        std_time = time.time() - start_time

        # Compute reconstruction
        recon_std = render_gaussians_numpy(image.shape, params_std, amps_std)
        mse_std = np.mean((image - recon_std)**2)

        results['standard'] = {
            'time': std_time,
            'iterations': stats_std['iterations'],
            'converged': stats_std['converged'],
            'n_splats': len(amps_std),
            'mse': mse_std,
            'params': params_std,
            'amps': amps_std,
            'reconstruction': recon_std
        }

        if verbose:
            aprint(f"Time: {std_time:.2f}s")
            aprint(f"Iterations: {stats_std['iterations']}/{n_iters}")
            aprint(f"Final splats: {len(amps_std)}")
            aprint(f"MSE: {mse_std:.6f}")

    # Dynamic operations fitting
    with asection("Dynamic Operations Fitting"):
        # Configure dynamic operations
        cfg = DynamicOpsConfig()
        cfg.step_every = 8  # Run every 8 iterations
        cfg.max_add_per_step = 32
        cfg.max_merges_per_step = 32
        cfg.residual_quantile = 0.96  # More selective seeding
        cfg.merge_dist_vox = 1.8  # Slightly more aggressive merging
        cfg.amp_abs_min = 5e-5  # Prune very weak splats

        start_time = time.time()
        params_dyn, amps_dyn, stats_dyn = fit_gaussian_splats(
            image,
            centers_overcomplete=candidates,
            n_iters=n_iters,
            lr=0.15,
            verbose=verbose,
            early_stopping=True,
            enable_dynamic_ops=True,
            dynamic_config=cfg,
        )
        dyn_time = time.time() - start_time

        # Compute reconstruction
        recon_dyn = render_gaussians_numpy(image.shape, params_dyn, amps_dyn)
        mse_dyn = np.mean((image - recon_dyn)**2)

        results['dynamic'] = {
            'time': dyn_time,
            'iterations': stats_dyn['iterations'],
            'converged': stats_dyn['converged'],
            'n_splats': len(amps_dyn),
            'mse': mse_dyn,
            'params': params_dyn,
            'amps': amps_dyn,
            'reconstruction': recon_dyn
        }

        if verbose:
            aprint(f"Time: {dyn_time:.2f}s")
            aprint(f"Iterations: {stats_dyn['iterations']}/{n_iters}")
            aprint(f"Final splats: {len(amps_dyn)}")
            aprint(f"MSE: {mse_dyn:.6f}")

    return results


def analyze_results(results, verbose=True):
    """Analyze and report comparison results."""
    std = results['standard']
    dyn = results['dynamic']

    if verbose:
        with asection("Performance Analysis"):
            # Time comparison
            speedup = std['time'] / dyn['time'] if dyn['time'] > 0 else float('inf')
            aprint(f"Time speedup: {speedup:.2f}x")
            if speedup > 1:
                aprint("✓ Dynamic ops are faster")
            else:
                aprint("⚠ Dynamic ops are slower (may indicate need for tuning)")

            # Iteration comparison
            iter_saved = std['iterations'] - dyn['iterations']
            iter_pct = 100 * iter_saved / std['iterations'] if std['iterations'] > 0 else 0
            aprint(f"Iterations saved: {iter_saved} ({iter_pct:.1f}%)")

            # Quality comparison
            mse_improvement = (std['mse'] - dyn['mse']) / std['mse'] * 100
            aprint(f"MSE change: {mse_improvement:+.2f}%")
            if mse_improvement > 0:
                aprint("✓ Dynamic ops achieve better reconstruction")
            elif abs(mse_improvement) < 5:
                aprint("≈ Similar reconstruction quality")
            else:
                aprint("⚠ Standard method achieved better reconstruction")

            # Efficiency comparison
            splat_efficiency = dyn['n_splats'] / std['n_splats']
            aprint(f"Splat count ratio: {splat_efficiency:.2f}")
            if splat_efficiency < 1:
                aprint("✓ Dynamic ops use fewer splats for similar quality")
            else:
                aprint("⚠ Dynamic ops use more splats")

    return {
        'speedup': speedup,
        'iterations_saved': iter_saved,
        'mse_improvement_pct': mse_improvement,
        'splat_efficiency': splat_efficiency
    }


def main():
    aprint("Dynamic Gaussian Splat Operations Demo")
    aprint("=" * 50)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create test data
    with asection("Generating Test Data"):
        image = create_test_image(size=64, n_blobs=6, add_noise=True)
        aprint(f"Image size: {image.shape}")
        aprint(f"Data range: [{image.min():.3f}, {image.max():.3f}]")

        # Find candidate positions
        candidates = find_candidates_overcomplete_nd(
            image,
            scales=(1.0, 2.0, 3.0),
            peaks_per_scale=25,
            percentile_thresh=75,
            min_dist=2.5,
        )
        aprint(f"Initial candidates: {len(candidates)}")

    # Run comparison
    results = benchmark_fitting_methods(image, candidates, n_iters=120, verbose=True)

    # Analyze results
    analysis = analyze_results(results, verbose=True)

    # Summary
    with asection("Summary"):
        aprint("Key Benefits of Dynamic Operations:")
        if analysis['speedup'] > 1.1:
            aprint(f"• {analysis['speedup']:.1f}x faster convergence")
        if analysis['iterations_saved'] > 5:
            pct = 100 * analysis['iterations_saved'] / 120
            aprint(f"• {analysis['iterations_saved']} fewer iterations ({pct:.0f}% reduction)")
        if analysis['mse_improvement_pct'] > 2:
            aprint(f"• {analysis['mse_improvement_pct']:.1f}% better reconstruction quality")
        if analysis['splat_efficiency'] < 0.9:
            aprint(f"• {100*(1-analysis['splat_efficiency']):.0f}% fewer splats needed")

        aprint("\nDynamic operations provide:")
        aprint("• Automatic pruning of weak/redundant splats")
        aprint("• Intelligent seeding at high-error regions")
        aprint("• Merging of near-duplicate splats")
        aprint("• Splitting of large, poorly-fitting splats")
        aprint("• Adaptive model complexity during optimization")


if __name__ == "__main__":
    main()
