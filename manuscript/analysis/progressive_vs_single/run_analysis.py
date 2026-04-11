#!/usr/bin/env python3
"""Progressive vs single-pass fitting analysis.

Compares reconstruction quality for the same total splat and iteration budget
split across 1, 2, 4, or 8 progressive passes.

Usage::

    hatch run python manuscript/analysis/progressive_vs_single/run_analysis.py
    hatch run python manuscript/analysis/progressive_vs_single/run_analysis.py --dataset kidney_dapi
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).parent.parent / "splat_count_vs_quality"))
from datasets import DATASETS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPLAT_BUDGET = 32000
ITER_BUDGET = 10000
PASS_COUNTS = [1, 2, 4, 8]

# Z-slice indices for montage (~quartiles)
SLICE_PERCENTILES = [0.25, 0.50, 0.75]

# Shared fitting kwargs
SHARED_KWARGS = dict(
    lr=0.01,
    loss_type="l1",
    cull_retention=0.999,
    enable_dynamic_ops=True,
    seed_method="auto",
    verbose=True,
)

TSV_COLUMNS = [
    "dataset",
    "n_passes",
    "splats_budget",
    "iters_budget",
    "n_splats_final",
    "psnr_db",
    "ssim",
    "mse",
    "rel_l2",
    "max_abs_error",
    "fit_time_s",
    "timestamp",
]

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_completed(tsv_path: Path, dataset_key: str) -> set[int]:
    if not tsv_path.exists():
        return set()
    done = set()
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("dataset") == dataset_key:
                done.add(int(row["n_passes"]))
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


def run_single_pass(volume: np.ndarray, device: str) -> tuple:
    """Fit with a single pass using fit_gaussian_splats."""
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats

    gsplat_data = fit_gaussian_splats(
        volume,
        seeds=SPLAT_BUDGET,
        n_iters=ITER_BUDGET,
        early_stop_patience=None,  # No early stopping — use full budget
        **SHARED_KWARGS,
        device=device,
    )
    return gsplat_data


def run_progressive(volume: np.ndarray, n_passes: int, device: str) -> tuple:
    """Fit with N progressive passes."""
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats

    splats_per_pass = SPLAT_BUDGET // n_passes
    iters_per_pass = ITER_BUDGET // n_passes

    gsplat_data = fit_progressive_gaussian_splats(
        volume,
        max_splats=SPLAT_BUDGET,
        max_splats_per_pass=splats_per_pass,
        iters_per_pass=iters_per_pass,
        max_passes=n_passes,
        psnr_patience=0.0,  # Force all passes to run
        **SHARED_KWARGS,
        device=device,
    )
    return gsplat_data


def run_condition(
    dataset_key: str,
    volume: np.ndarray,
    n_passes: int,
    device: str,
) -> None:
    """Run one condition and save results."""
    from luxar.gsplats.metrics import compute_quality_metrics
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    ds_dir = RESULTS_DIR / dataset_key
    ds_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = ds_dir / "progressive.tsv"

    # Fit
    if n_passes == 1:
        gsplat_data = run_single_pass(volume, device)
    else:
        gsplat_data = run_progressive(volume, n_passes, device)

    n_final = gsplat_data.n_splats
    stats = gsplat_data.stats if gsplat_data.stats else {}

    # Render and compute metrics
    with torch.no_grad():
        recon = render_to_volume_tensor(gsplat_data, shape=volume.shape, device=device)
        V_t = torch.from_numpy(volume.astype(np.float32)).to(recon.device)
        metrics = compute_quality_metrics(recon, V_t)

    # Save representative slices for montage
    recon_np = recon.cpu().numpy()
    z_size = volume.shape[0]
    z_indices = [int(p * (z_size - 1)) for p in SLICE_PERCENTILES]
    slices_dir = ds_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(slices_dir / f"pass{n_passes:02d}_slices.npz"),
        target_slices=volume[z_indices],
        recon_slices=recon_np[z_indices],
        z_indices=np.array(z_indices),
    )

    aprint(
        f"  {n_passes}-pass: {n_final:,} splats, "
        f"PSNR={metrics['psnr_db']:.2f} dB, "
        f"SSIM={metrics['ssim']:.4f}, "
        f"time={stats.get('time_seconds', 0.0):.1f}s"
    )

    row = {
        "dataset": dataset_key,
        "n_passes": n_passes,
        "splats_budget": SPLAT_BUDGET,
        "iters_budget": ITER_BUDGET,
        "n_splats_final": n_final,
        "psnr_db": f"{metrics['psnr_db']:.4f}",
        "ssim": f"{metrics['ssim']:.6f}",
        "mse": f"{metrics['mse']:.8f}",
        "rel_l2": f"{metrics['rel_l2']:.6f}",
        "max_abs_error": f"{metrics['max_abs_error']:.6f}",
        "fit_time_s": f"{stats.get('time_seconds', 0.0):.2f}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    append_row(tsv_path, row)

    # Free GPU memory
    del recon, recon_np, V_t, gsplat_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Progressive vs single-pass fitting analysis"
    )
    parser.add_argument(
        "--dataset",
        default="opencell_map4_ch0",
        choices=list(DATASETS.keys()),
        help="Dataset key (default: opencell_map4_ch0)",
    )
    args = parser.parse_args()

    Arbol.max_depth = 10
    dataset_key = args.dataset

    from luxar.utils.demos import detect_device

    device = detect_device()

    with asection(f"Loading dataset: {dataset_key}"):
        volume, metadata = DATASETS[dataset_key]()
        aprint(f"Shape: {metadata['shape']}, ndim: {metadata['ndim']}")

    ds_dir = RESULTS_DIR / dataset_key
    tsv_path = ds_dir / "progressive.tsv"
    completed = get_completed(tsv_path, dataset_key)

    aprint(f"Budget: {SPLAT_BUDGET:,} splats, {ITER_BUDGET:,} iterations")

    for n_passes in PASS_COUNTS:
        if n_passes in completed:
            aprint(f"Skipping {n_passes}-pass (already completed)")
            continue
        with asection(f"{n_passes}-pass ({SPLAT_BUDGET // n_passes:,} splats/pass, {ITER_BUDGET // n_passes:,} iters/pass)"):
            run_condition(dataset_key, volume, n_passes, device)

    aprint(f"\nResults: {tsv_path}")


if __name__ == "__main__":
    main()
