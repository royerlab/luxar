#!/usr/bin/env python3
"""Estimate the noise floor for each dataset.

Computes the noise standard deviation using multiple robust estimators
and derives the theoretical PSNR ceiling. Results are saved to a single
shared TSV at ``results/noise_floor.tsv``.

Usage::

    hatch run python manuscript/analysis/splat_count_vs_quality/run_noise_floor.py --all
    hatch run python manuscript/analysis/splat_count_vs_quality/run_noise_floor.py --dataset kidney_dapi
    hatch run python manuscript/analysis/splat_count_vs_quality/run_noise_floor.py --verify
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

sys.path.insert(0, str(Path(__file__).parent))

from datasets import DATASETS
from noise_floor import estimate_noise_floor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TSV_COLUMNS = [
    "dataset",
    "sigma_laplacian",
    "sigma_mad_haar",
    "sigma_mad_wavelet",
    "sigma_background",
    "sigma_ensemble",
    "noise_variance",
    "psnr_max_db",
    "volume_shape",
    "methods_used",
    "timestamp",
]

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_completed(tsv_path: Path) -> set[str]:
    if not tsv_path.exists():
        return set()
    done = set()
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            done.add(row["dataset"])
    return done


def append_row(tsv_path: Path, row: dict) -> None:
    write_header = not tsv_path.exists() or tsv_path.stat().st_size == 0
    with open(tsv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_COLUMNS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def fmt_sigma(val) -> str:
    if val is None:
        return "None"
    return f"{val:.8f}"


# ---------------------------------------------------------------------------
# Synthetic verification
# ---------------------------------------------------------------------------


def verify_synthetic():
    """Run a synthetic test to validate the noise estimators."""
    from scipy.ndimage import gaussian_filter

    aprint("Synthetic verification: smooth signal + known Gaussian noise")

    rng = np.random.RandomState(42)
    sigma_true = 0.03
    signal = gaussian_filter(rng.rand(64, 64, 64).astype(np.float32), sigma=5)
    noise = (rng.randn(64, 64, 64) * sigma_true).astype(np.float32)
    noisy = np.clip(signal + noise, 0, 1).astype(np.float32)

    result = estimate_noise_floor(noisy)

    aprint(f"  True sigma:       {sigma_true:.6f}")
    aprint(f"  Laplacian:        {result['sigma_laplacian']:.6f}")
    aprint(f"  Haar MAD:         {result['sigma_mad_haar']:.6f}")
    aprint(f"  Wavelet MAD:      {fmt_sigma(result['sigma_mad_wavelet'])}")
    aprint(f"  Background:       {fmt_sigma(result['sigma_background'])}")
    aprint(f"  Ensemble:         {result['sigma_ensemble']:.6f}")
    aprint(f"  PSNR_max:         {result['psnr_max_db']:.2f} dB")

    rel_error = abs(result["sigma_ensemble"] - sigma_true) / sigma_true
    aprint(f"  Relative error:   {rel_error:.1%}")

    if rel_error < 0.20:
        aprint("  PASS (within 20%)")
    else:
        aprint("  WARN: relative error > 20% — investigate")

    # Check cross-method consistency
    sigmas = [result["sigma_laplacian"], result["sigma_mad_haar"]]
    if result["sigma_mad_wavelet"] is not None:
        sigmas.append(result["sigma_mad_wavelet"])
    if result["sigma_background"] is not None:
        sigmas.append(result["sigma_background"])

    spread = max(sigmas) / min(sigmas) if min(sigmas) > 0 else float("inf")
    aprint(f"  Method spread:    {spread:.2f}x (max/min)")
    if spread > 2.0:
        aprint("  WARN: methods disagree by >2x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Estimate noise floor for each dataset"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=list(DATASETS.keys()),
        help="Single dataset to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all datasets",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run synthetic verification test",
    )
    args = parser.parse_args()

    Arbol.max_depth = 10

    if args.verify:
        verify_synthetic()
        return

    if args.all:
        keys = list(DATASETS.keys())
    elif args.dataset:
        keys = [args.dataset]
    else:
        keys = ["opencell_map4_ch0"]

    tsv_path = RESULTS_DIR / "noise_floor.tsv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed = get_completed(tsv_path)

    for key in keys:
        if key in completed:
            aprint(f"Skipping {key} (already computed)")
            continue

        with asection(f"Noise floor: {key}"):
            volume, metadata = DATASETS[key]()
            aprint(f"Shape: {volume.shape}, voxels: {volume.size:,}")

            result = estimate_noise_floor(volume)

            aprint(f"  Laplacian σ:   {result['sigma_laplacian']:.6f}")
            aprint(f"  Haar MAD σ:    {result['sigma_mad_haar']:.6f}")
            aprint(f"  Wavelet MAD σ: {fmt_sigma(result['sigma_mad_wavelet'])}")
            aprint(f"  Background σ:  {fmt_sigma(result['sigma_background'])}")
            aprint(f"  Ensemble σ:    {result['sigma_ensemble']:.6f}")
            aprint(f"  PSNR_max:      {result['psnr_max_db']:.2f} dB")

            # Check cross-method consistency
            sigmas = [v for v in [
                result["sigma_laplacian"],
                result["sigma_mad_haar"],
                result["sigma_mad_wavelet"],
                result["sigma_background"],
            ] if v is not None and v > 0]
            if len(sigmas) >= 2:
                spread = max(sigmas) / min(sigmas)
                if spread > 2.0:
                    aprint(f"  WARNING: methods disagree by {spread:.1f}x")

            shape_str = "x".join(str(s) for s in volume.shape)
            row = {
                "dataset": key,
                "sigma_laplacian": fmt_sigma(result["sigma_laplacian"]),
                "sigma_mad_haar": fmt_sigma(result["sigma_mad_haar"]),
                "sigma_mad_wavelet": fmt_sigma(result["sigma_mad_wavelet"]),
                "sigma_background": fmt_sigma(result["sigma_background"]),
                "sigma_ensemble": fmt_sigma(result["sigma_ensemble"]),
                "noise_variance": f"{result['noise_variance']:.10f}",
                "psnr_max_db": f"{result['psnr_max_db']:.4f}",
                "volume_shape": shape_str,
                "methods_used": ",".join(result["methods_used"]),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            append_row(tsv_path, row)

    aprint(f"\nResults: {tsv_path}")


if __name__ == "__main__":
    main()
