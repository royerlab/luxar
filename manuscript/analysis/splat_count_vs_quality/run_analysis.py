#!/usr/bin/env python3
"""Runner: splat count vs reconstruction quality analysis.

For each splat count in SPLAT_COUNTS, this script:
  1. Fits non-progressive Gaussian splats (or loads from cache)
  2. Renders back to the original volume shape
  3. Computes quality metrics (PSNR, SSIM, MSE, ...)
  4. Saves representative z-slices for the plotting script
  5. Appends results to a TSV file

Resumable: re-running skips already-completed splat counts.

Usage::

    hatch run python manuscript/analysis/splat_count_vs_quality/run_analysis.py
    hatch run python manuscript/analysis/splat_count_vs_quality/run_analysis.py --dataset opencell_map4_ch1
"""

from __future__ import annotations

# Enable MPS fallback before any torch import
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import csv
import gc
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from arbol import Arbol, aprint, asection

# Ensure the analysis package is importable
sys.path.insert(0, str(Path(__file__).parent))

from datasets import DATASETS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPLAT_COUNTS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]

# Z-slice indices for representative slices (~quartiles of Z=51)
SLICE_PERCENTILES = [0.25, 0.50, 0.75]

# Fitting hyperparameters — fixed across all splat counts
FIT_KWARGS = dict(
    n_iters=5000,
    lr=0.01,
    loss_type="l1",
    early_stop_patience=500,
    enable_dynamic_ops=True,
    cull_retention=0.99,
    seed_method="auto",
    verbose=True,
)

TSV_COLUMNS = [
    "dataset",
    "seeds_requested",
    "n_splats_final",
    "psnr_db",
    "ssim",
    "mse",
    "rel_l2",
    "max_abs_error",
    "compression_ratio",
    "fit_time_s",
    "n_iters_actual",
    "converged",
    "timestamp",
]

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slice_indices_for_volume(volume: np.ndarray) -> list[int]:
    """Compute representative z-slice indices from SLICE_PERCENTILES."""
    z_size = volume.shape[0]
    return [int(p * (z_size - 1)) for p in SLICE_PERCENTILES]


def get_completed_seeds(tsv_path: Path, dataset_key: str) -> set[int]:
    """Return set of seeds_requested already recorded in the TSV."""
    if not tsv_path.exists():
        return set()
    completed = set()
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("dataset") == dataset_key:
                completed.add(int(row["seeds_requested"]))
    return completed


def append_row(tsv_path: Path, row: dict) -> None:
    """Append a single row to the TSV, writing header if file is new/empty."""
    write_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
    with open(tsv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_slices(
    volume: np.ndarray,
    recon: np.ndarray,
    z_indices: list[int],
    path: Path,
) -> None:
    """Save target and reconstruction slices as compressed npz."""
    np.savez_compressed(
        str(path),
        target_slices=volume[z_indices],
        recon_slices=recon[z_indices],
        z_indices=np.array(z_indices),
    )


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def run_single(
    dataset_key: str,
    volume: np.ndarray,
    seeds: int,
    device: str,
    z_indices: list[int],
) -> None:
    """Fit, render, measure, and save results for one splat count."""
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.metrics import compute_quality_metrics
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor
    from luxar.gsplats.utils.trils import tril_size

    ds_dir = RESULTS_DIR / dataset_key
    gsplats_dir = ds_dir / "gsplats"
    slices_dir = ds_dir / "slices"
    tsv_path = ds_dir / "metrics.tsv"

    gsplats_dir.mkdir(parents=True, exist_ok=True)
    slices_dir.mkdir(parents=True, exist_ok=True)

    cache_file = gsplats_dir / f"n{seeds:06d}.gsplats.zarr.zip"
    slices_file = slices_dir / f"n{seeds:06d}_slices.npz"

    # 1. Fit or load from cache
    if cache_file.exists():
        aprint(f"Loading cached gsplats from {cache_file.name}")
        gsplat_data = GSplatData.load(cache_file)
    else:
        aprint(f"Fitting with seeds={seeds:,} ...")
        gsplat_data = fit_gaussian_splats(
            volume, seeds=seeds, device=device, **FIT_KWARGS
        )
        gsplat_data.save(
            cache_file,
            compress="zip",
            zip_deflate=True,
            include_fitting_info=True,
        )

    n_final = gsplat_data.n_splats
    aprint(f"Final splat count: {n_final:,}")

    # 2. Render and compute metrics (always recomputed for consistency)
    with torch.no_grad():
        recon_tensor = render_to_volume_tensor(
            gsplat_data, shape=volume.shape, device=device
        )
        V_tensor = torch.from_numpy(volume.astype(np.float32)).to(
            recon_tensor.device
        )
        metrics = compute_quality_metrics(recon_tensor, V_tensor)
        recon_np = recon_tensor.cpu().numpy()

    aprint(
        f"PSNR: {metrics['psnr_db']:.2f} dB, "
        f"SSIM: {metrics['ssim']:.4f}, "
        f"MSE: {metrics['mse']:.6g}"
    )

    # 3. Save representative slices
    save_slices(volume, recon_np, z_indices, slices_file)

    # 4. Compression ratio: volume voxels / (n_splats * floats_per_splat)
    d = volume.ndim
    floats_per_splat = d + tril_size(d) + 1  # center + cholesky + amplitude
    cr = volume.size / (n_final * floats_per_splat) if n_final > 0 else float("inf")

    # 5. Extract fitting stats
    stats = gsplat_data.stats if gsplat_data.stats else {}
    fit_time = stats.get("time_seconds", 0.0)
    n_iters_actual = stats.get("iterations", 0)
    converged = stats.get("converged", False)

    # 6. Append to TSV
    row = {
        "dataset": dataset_key,
        "seeds_requested": seeds,
        "n_splats_final": n_final,
        "psnr_db": f"{metrics['psnr_db']:.4f}",
        "ssim": f"{metrics['ssim']:.6f}",
        "mse": f"{metrics['mse']:.8f}",
        "rel_l2": f"{metrics['rel_l2']:.6f}",
        "max_abs_error": f"{metrics['max_abs_error']:.6f}",
        "compression_ratio": f"{cr:.2f}",
        "fit_time_s": f"{fit_time:.2f}",
        "n_iters_actual": n_iters_actual,
        "converged": converged,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    append_row(tsv_path, row)

    # 7. Free GPU memory
    del recon_tensor, V_tensor, recon_np, gsplat_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Splat count vs quality analysis runner"
    )
    parser.add_argument(
        "--dataset",
        default="opencell_map4_ch0",
        choices=list(DATASETS.keys()),
        help="Dataset key from the registry (default: opencell_map4_ch0)",
    )
    parser.add_argument(
        "--counts",
        type=str,
        default=None,
        help="Comma-separated splat counts (overrides default list)",
    )
    args = parser.parse_args()

    Arbol.max_depth = 10
    dataset_key = args.dataset
    counts = (
        [int(c) for c in args.counts.split(",")]
        if args.counts
        else SPLAT_COUNTS
    )

    from luxar.utils.demos import detect_device

    device = detect_device()

    with asection(f"Loading dataset: {dataset_key}"):
        volume, metadata = DATASETS[dataset_key]()
        aprint(f"Shape: {metadata['shape']}, ndim: {metadata['ndim']}")

    z_indices = slice_indices_for_volume(volume)
    aprint(f"Representative z-slices: {z_indices}")

    ds_dir = RESULTS_DIR / dataset_key
    tsv_path = ds_dir / "metrics.tsv"
    completed = get_completed_seeds(tsv_path, dataset_key)

    for seeds in counts:
        if seeds in completed:
            aprint(f"Skipping seeds={seeds:,} (already completed)")
            continue
        with asection(f"Seeds = {seeds:,}"):
            run_single(dataset_key, volume, seeds, device, z_indices)

    aprint(f"\nAll splat counts completed. Results: {tsv_path}")


if __name__ == "__main__":
    main()
