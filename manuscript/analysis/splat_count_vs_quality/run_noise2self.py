#!/usr/bin/env python3
"""Noise2Self-inspired analysis: optimal splat count via held-out loss.

Determines where Gaussian splatting transitions from fitting signal to
fitting noise by masking a fraction of voxels, replacing them with a
blind-spot (donut) median, fitting splats to the modified volume, and
evaluating reconstruction at the masked positions against the original
noisy values.

The held-out loss minimum indicates the optimal splat count — beyond
that, adding splats captures noise rather than signal.

Based on: Batson & Royer, "Noise2Self: Blind Denoising by
Self-Supervision", ICML 2019.

Usage::

    hatch run python manuscript/analysis/splat_count_vs_quality/run_noise2self.py
    hatch run python manuscript/analysis/splat_count_vs_quality/run_noise2self.py --mask-fraction 0.1
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
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).parent))

from datasets import DATASETS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPLAT_COUNTS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]

SLICE_PERCENTILES = [0.25, 0.50, 0.75]

FIT_KWARGS = dict(
    n_iters=20000,
    lr=0.01,
    loss_type="l1",
    early_stop_patience=500,
    enable_dynamic_ops=True,
    cull_retention=0.999,
    seed_method="auto",
    verbose=True,
)

TSV_COLUMNS = [
    "dataset",
    "seeds_requested",
    "n_splats_final",
    "mask_fraction",
    "held_out_mse",
    "held_out_psnr_db",
    "train_mse",
    "train_psnr_db",
    "full_mse",
    "full_psnr_db",
    "fit_time_s",
    "n_iters_actual",
    "timestamp",
]

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Donut median (blind-spot replacement)
# ---------------------------------------------------------------------------


def donut_median_3d(volume: np.ndarray, radius: int = 1) -> np.ndarray:
    """Compute donut median: median of neighbors excluding the center voxel.

    For each voxel, computes the median of all voxels within a (2r+1)^3 cube
    excluding the center. This is the blind-spot estimator from Noise2Self.

    Parameters
    ----------
    volume : np.ndarray
        3D float32 volume.
    radius : int
        Radius of the neighborhood cube (default: 1 → 3x3x3 neighborhood).

    Returns
    -------
    np.ndarray
        Volume where each voxel is the donut median of its neighbors.
    """
    # Strategy: for a 3x3x3 neighborhood (radius=1), the donut has 26 neighbors.
    # We use the identity: median of 27 values with center replaced by median
    # of 26 neighbors ≈ median of 26 neighbors when center is not an outlier.
    # But for correctness, we compute it explicitly using shifted volumes.

    d = volume.ndim
    size = 2 * radius + 1

    # Collect all neighbor values (excluding center) for each voxel
    # For 3D radius=1: 26 neighbors
    shifts = []
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue  # Skip center (blind spot)
                shifts.append((dz, dy, dx))

    # Pad volume to handle boundaries
    padded = np.pad(volume, radius, mode="reflect")

    # Stack all shifted views
    neighbors = np.empty(
        (len(shifts),) + volume.shape, dtype=volume.dtype
    )
    for i, (dz, dy, dx) in enumerate(shifts):
        sz = radius + dz
        sy = radius + dy
        sx = radius + dx
        neighbors[i] = padded[
            sz : sz + volume.shape[0],
            sy : sy + volume.shape[1],
            sx : sx + volume.shape[2],
        ]

    # Median along the neighbor axis
    return np.median(neighbors, axis=0).astype(volume.dtype)


def create_masked_volume(
    volume: np.ndarray,
    mask_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a volume with masked voxels replaced by donut median.

    Parameters
    ----------
    volume : np.ndarray
        Original noisy volume, float32 in [0, 1].
    mask_fraction : float
        Fraction of voxels to mask (default: 5%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    masked_volume : np.ndarray
        Volume with masked voxels replaced by donut median.
    mask : np.ndarray
        Boolean mask (True = held-out voxel).
    original_values : np.ndarray
        Original values at masked positions.
    """
    rng = np.random.RandomState(seed)
    mask = rng.random(volume.shape) < mask_fraction

    with asection(f"Creating blind-spot masked volume ({mask.sum():,} masked voxels, {mask_fraction:.1%})"):
        # Compute donut median for the entire volume
        aprint("Computing donut median (3x3x3 blind-spot)...")
        donut = donut_median_3d(volume, radius=1)

        # Replace masked voxels with their donut median
        masked_volume = volume.copy()
        masked_volume[mask] = donut[mask]

        original_values = volume[mask].copy()
        aprint(f"Masked {mask.sum():,} / {volume.size:,} voxels ({mask.mean():.1%})")

    return masked_volume, mask, original_values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slice_indices_for_volume(volume: np.ndarray) -> list[int]:
    z_size = volume.shape[0]
    return [int(p * (z_size - 1)) for p in SLICE_PERCENTILES]


def get_completed_seeds(tsv_path: Path, dataset_key: str) -> set[int]:
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
    write_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
    with open(tsv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def run_single(
    dataset_key: str,
    masked_volume: np.ndarray,
    original_volume: np.ndarray,
    mask: np.ndarray,
    seeds: int,
    device: str,
) -> None:
    """Fit to masked volume, evaluate on held-out pixels."""
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    ds_dir = RESULTS_DIR / f"{dataset_key}_n2s"
    gsplats_dir = ds_dir / "gsplats"
    tsv_path = ds_dir / "metrics_n2s.tsv"

    gsplats_dir.mkdir(parents=True, exist_ok=True)

    cache_file = gsplats_dir / f"n{seeds:06d}.gsplats.zarr.zip"

    # 1. Fit to the masked volume (or load from cache)
    if cache_file.exists():
        aprint(f"Loading cached gsplats from {cache_file.name}")
        gsplat_data = GSplatData.load(cache_file)
    else:
        aprint(f"Fitting to masked volume with seeds={seeds:,} ...")
        gsplat_data = fit_gaussian_splats(
            masked_volume, seeds=seeds, device=device, **FIT_KWARGS
        )
        gsplat_data.save(
            cache_file,
            compress="zip",
            zip_deflate=True,
            include_fitting_info=True,
        )

    n_final = gsplat_data.n_splats
    aprint(f"Final splat count: {n_final:,}")

    # 2. Render reconstruction
    with torch.no_grad():
        recon_tensor = render_to_volume_tensor(
            gsplat_data, shape=original_volume.shape, device=device
        )
        recon_np = recon_tensor.cpu().numpy()

    # 3. Compute held-out loss (masked pixels vs original noisy values)
    recon_masked = recon_np[mask]
    original_masked = original_volume[mask]
    held_out_mse = float(np.mean((recon_masked - original_masked) ** 2))
    held_out_psnr = 10 * np.log10(1.0 / held_out_mse) if held_out_mse > 0 else float("inf")

    # 4. Compute train loss (unmasked pixels vs original noisy values)
    train_mask = ~mask
    recon_train = recon_np[train_mask]
    original_train = original_volume[train_mask]
    train_mse = float(np.mean((recon_train - original_train) ** 2))
    train_psnr = 10 * np.log10(1.0 / train_mse) if train_mse > 0 else float("inf")

    # 5. Full volume loss (for reference)
    full_mse = float(np.mean((recon_np - original_volume) ** 2))
    full_psnr = 10 * np.log10(1.0 / full_mse) if full_mse > 0 else float("inf")

    aprint(
        f"Held-out PSNR: {held_out_psnr:.2f} dB, "
        f"Train PSNR: {train_psnr:.2f} dB, "
        f"Full PSNR: {full_psnr:.2f} dB"
    )

    # 6. Extract fitting stats
    stats = gsplat_data.stats if gsplat_data.stats else {}

    # 7. Append to TSV
    row = {
        "dataset": dataset_key,
        "seeds_requested": seeds,
        "n_splats_final": n_final,
        "mask_fraction": f"{mask.mean():.4f}",
        "held_out_mse": f"{held_out_mse:.8f}",
        "held_out_psnr_db": f"{held_out_psnr:.4f}",
        "train_mse": f"{train_mse:.8f}",
        "train_psnr_db": f"{train_psnr:.4f}",
        "full_mse": f"{full_mse:.8f}",
        "full_psnr_db": f"{full_psnr:.4f}",
        "fit_time_s": f"{stats.get('time_seconds', 0.0):.2f}",
        "n_iters_actual": stats.get("iterations", 0),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    append_row(tsv_path, row)

    # 8. Free GPU memory
    del recon_tensor, recon_np, gsplat_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Noise2Self analysis: optimal splat count via held-out loss"
    )
    parser.add_argument(
        "--dataset",
        default="opencell_map4_ch0",
        choices=list(DATASETS.keys()),
        help="Dataset key (default: opencell_map4_ch0)",
    )
    parser.add_argument(
        "--mask-fraction",
        type=float,
        default=0.05,
        help="Fraction of voxels to mask (default: 0.05)",
    )
    parser.add_argument(
        "--counts",
        type=str,
        default=None,
        help="Comma-separated splat counts (overrides default list)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mask generation (default: 42)",
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

    # Create masked volume (done once, reused for all splat counts)
    masked_volume, mask, original_values = create_masked_volume(
        volume, mask_fraction=args.mask_fraction, seed=args.seed
    )

    ds_dir = RESULTS_DIR / f"{dataset_key}_n2s"
    tsv_path = ds_dir / "metrics_n2s.tsv"
    completed = get_completed_seeds(tsv_path, dataset_key)

    for seeds in counts:
        if seeds in completed:
            aprint(f"Skipping seeds={seeds:,} (already completed)")
            continue
        with asection(f"Seeds = {seeds:,}"):
            run_single(
                dataset_key, masked_volume, volume, mask, seeds, device
            )

    aprint(f"\nNoise2Self analysis complete. Results: {tsv_path}")


if __name__ == "__main__":
    main()
