#!/usr/bin/env python3
"""Tiled fitting analysis: compare tiled vs monolithic fitting.

Measures quality (PSNR), memory, and time for kidney_dapi dataset.
"""

import json
import sys
import time
from pathlib import Path

import torch

ANALYSIS_DIR = Path(__file__).parent
RESULTS_DIR = ANALYSIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ANALYSIS_DIR.parent))
from _shared import compute_psnr  # noqa: E402


def measure_peak_memory():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def run_monolithic(volume, n_seeds=8000, n_iters=2000):
    """Fit the full volume monolithically and compute metrics."""
    from luxar.gsplats import fit_gaussian_splats
    from luxar.gsplats.rendering.volume_rendering import render_to_volume

    reset_peak_memory()
    t0 = time.time()

    result = fit_gaussian_splats(
        volume, seeds=n_seeds, n_iters=n_iters,
        loss_type='l1', early_stop_patience=200,
        enable_dynamic_ops=True, cull_retention=0.999, verbose=False,
    )

    elapsed = time.time() - t0
    peak_mem = measure_peak_memory()

    # Render and compute PSNR
    recon = render_to_volume(result, shape=volume.shape)
    psnr = compute_psnr(volume, recon)

    return {
        'method': 'monolithic',
        'n_seeds': n_seeds,
        'n_splats': result.n_splats,
        'psnr_db': round(psnr, 2),
        'wall_time_s': round(elapsed, 1),
        'peak_gpu_mb': round(peak_mem, 1),
    }


def run_tiled(volume, tile_size, overlap, n_seeds=2000, n_iters=2000):
    """Fit using tiled approach."""
    from luxar.gsplats.fit_tiled_gsplats import fit_tiled
    from luxar.gsplats.rendering.volume_rendering import render_to_volume

    reset_peak_memory()
    t0 = time.time()

    result = fit_tiled(
        volume, tile_size=tile_size, overlap=overlap,
        seeds=n_seeds, n_iters=n_iters,
        loss_type='l1', early_stop_patience=200,
        enable_dynamic_ops=True, cull_retention=0.999, verbose=False,
    )

    elapsed = time.time() - t0
    peak_mem = measure_peak_memory()

    recon = render_to_volume(result, shape=volume.shape)
    psnr = compute_psnr(volume, recon)

    return {
        'method': f'tiled_{tile_size}_{overlap}',
        'tile_size': tile_size,
        'overlap': overlap,
        'n_seeds': n_seeds,
        'n_splats': result.n_splats,
        'psnr_db': round(psnr, 2),
        'wall_time_s': round(elapsed, 1),
        'peak_gpu_mb': round(peak_mem, 1),
    }


def main():
    sys.path.insert(0, str(ANALYSIS_DIR.parent / "splat_count_vs_quality"))
    from arbol import aprint, asection

    import datasets as ds_module

    with asection("Loading kidney_dapi dataset"):
        volume, meta = ds_module.DATASETS['kidney_dapi']()
        aprint(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")

    results = []

    # Monolithic at different seed counts
    for n_seeds in [4000, 8000, 16000]:
        with asection(f"Monolithic fitting ({n_seeds} seeds)"):
            try:
                r = run_monolithic(volume, n_seeds=n_seeds, n_iters=2000)
                aprint(f"  PSNR: {r['psnr_db']:.1f} dB, Time: {r['wall_time_s']:.1f}s, "
                       f"GPU: {r['peak_gpu_mb']:.0f} MB, Splats: {r['n_splats']}")
                results.append(r)
            except Exception as e:
                aprint(f"  ERROR: {e}")

    # Tiled fitting
    tile_configs = [
        (8, 2, 2000),
        (12, 3, 3000),
    ]
    for tile_size, overlap, seeds in tile_configs:
        with asection(f"Tiled fitting (tile={tile_size}, overlap={overlap})"):
            try:
                r = run_tiled(volume, tile_size=tile_size, overlap=overlap,
                              n_seeds=seeds, n_iters=2000)
                aprint(f"  PSNR: {r['psnr_db']:.1f} dB, Time: {r['wall_time_s']:.1f}s, "
                       f"GPU: {r['peak_gpu_mb']:.0f} MB, Splats: {r['n_splats']}")
                results.append(r)
            except Exception as e:
                aprint(f"  ERROR: {e}")

    out_path = RESULTS_DIR / "tiled_comparison.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['method']:25s}  PSNR: {r['psnr_db']:5.1f} dB  "
              f"Time: {r['wall_time_s']:6.1f}s  GPU: {r['peak_gpu_mb']:6.0f} MB  "
              f"Splats: {r['n_splats']}")


if __name__ == '__main__':
    main()
