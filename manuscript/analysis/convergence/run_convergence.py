#!/usr/bin/env python3
"""Convergence analysis: reconstruction quality vs iteration count.

For each splat count and a series of iteration checkpoints, fits Gaussian
splats and measures PSNR. This reveals convergence dynamics — how fast
quality saturates for different model capacities.

Usage::

    hatch run python manuscript/analysis/convergence/run_convergence.py
    hatch run python manuscript/analysis/convergence/run_convergence.py --dataset kidney_dapi
"""

from __future__ import annotations

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from arbol import Arbol, aprint, asection

sys.path.insert(0, str(Path(__file__).parent.parent / "splat_count_vs_quality"))
from datasets import DATASETS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Representative splat counts
SPLAT_COUNTS = [1000, 4000, 16000, 64000, 256000]

# Iteration checkpoints — logarithmically spaced for good coverage
ITER_CHECKPOINTS = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000]

# Fitting hyperparameters — same as main analysis but NO early stopping
# (we want to see the full curve, not just where it stops)
FIT_KWARGS = dict(
    lr=0.01,
    loss_type="l1",
    early_stop_patience=None,  # Disable early stopping
    enable_dynamic_ops=True,
    cull_retention=0.999,
    seed_method="auto",
    verbose=False,  # Quiet — we're running many times
)

TSV_COLUMNS = [
    "dataset",
    "seeds_requested",
    "n_iters",
    "n_splats_final",
    "psnr_db",
    "ssim",
    "mse",
    "rel_l2",
    "fit_time_s",
]

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_completed(tsv_path: Path, dataset_key: str) -> set[tuple[int, int]]:
    """Return set of (seeds, n_iters) pairs already in the TSV."""
    if not tsv_path.exists():
        return set()
    done = set()
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("dataset") == dataset_key:
                done.add((int(row["seeds_requested"]), int(row["n_iters"])))
    return done


def append_row(tsv_path: Path, row: dict) -> None:
    write_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
    with open(tsv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def run_checkpoint(
    dataset_key: str,
    volume: np.ndarray,
    seeds: int,
    n_iters: int,
    device: str,
) -> None:
    """Fit at a specific iteration count and record quality metrics."""
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.metrics import compute_quality_metrics
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    ds_dir = RESULTS_DIR / dataset_key
    tsv_path = ds_dir / "convergence.tsv"
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Fit with exact iteration count, no early stopping
    gsplat_data = fit_gaussian_splats(
        volume, seeds=seeds, device=device, n_iters=n_iters, **FIT_KWARGS
    )

    n_final = gsplat_data.n_splats

    # Render and compute metrics
    with torch.no_grad():
        recon = render_to_volume_tensor(gsplat_data, shape=volume.shape, device=device)
        V_tensor = torch.from_numpy(volume.astype(np.float32)).to(recon.device)
        metrics = compute_quality_metrics(recon, V_tensor)

    stats = gsplat_data.stats if gsplat_data.stats else {}

    row = {
        "dataset": dataset_key,
        "seeds_requested": seeds,
        "n_iters": n_iters,
        "n_splats_final": n_final,
        "psnr_db": f"{metrics['psnr_db']:.4f}",
        "ssim": f"{metrics['ssim']:.6f}",
        "mse": f"{metrics['mse']:.8f}",
        "rel_l2": f"{metrics['rel_l2']:.6f}",
        "fit_time_s": f"{stats.get('time_seconds', 0.0):.2f}",
    }
    append_row(tsv_path, row)

    aprint(
        f"  iters={n_iters:5d}  splats={n_final:6d}  "
        f"PSNR={metrics['psnr_db']:.2f} dB  "
        f"SSIM={metrics['ssim']:.4f}  "
        f"time={stats.get('time_seconds', 0.0):.1f}s"
    )

    # Free GPU memory
    del recon, V_tensor, gsplat_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Convergence analysis: PSNR vs iteration count"
    )
    parser.add_argument(
        "--dataset",
        default="opencell_map4_ch0",
        choices=list(DATASETS.keys()),
        help="Dataset key (default: opencell_map4_ch0)",
    )
    parser.add_argument(
        "--counts",
        type=str,
        default=None,
        help="Comma-separated splat counts",
    )
    parser.add_argument(
        "--iters",
        type=str,
        default=None,
        help="Comma-separated iteration checkpoints",
    )
    args = parser.parse_args()

    Arbol.max_depth = 10
    dataset_key = args.dataset
    counts = (
        [int(c) for c in args.counts.split(",")]
        if args.counts
        else SPLAT_COUNTS
    )
    checkpoints = (
        [int(i) for i in args.iters.split(",")]
        if args.iters
        else ITER_CHECKPOINTS
    )

    from luxar.utils.demos import detect_device

    device = detect_device()

    with asection(f"Loading dataset: {dataset_key}"):
        volume, metadata = DATASETS[dataset_key]()
        aprint(f"Shape: {metadata['shape']}, ndim: {metadata['ndim']}")

    ds_dir = RESULTS_DIR / dataset_key
    tsv_path = ds_dir / "convergence.tsv"
    completed = get_completed(tsv_path, dataset_key)

    for seeds in counts:
        with asection(f"Seeds = {seeds:,}"):
            for n_iters in checkpoints:
                if (seeds, n_iters) in completed:
                    aprint(f"  Skipping iters={n_iters} (done)")
                    continue
                run_checkpoint(dataset_key, volume, seeds, n_iters, device)

    aprint(f"\nConvergence analysis complete. Results: {tsv_path}")


if __name__ == "__main__":
    main()
