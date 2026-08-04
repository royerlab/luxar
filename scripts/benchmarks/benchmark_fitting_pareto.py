#!/usr/bin/env python3
"""Pareto-dominance benchmark for single-pass gsplat fitting.

Fits on 3 diverse microscopy datasets with a fixed splat/iteration budget,
reports (median_PSNR, median_time), and checks Pareto dominance against a
stored baseline.

Exit code:
  0 — New result Pareto-dominates (or no baseline yet)
  1 — Not Pareto-dominant (regression in at least one axis)

Outputs a single metric line:  METRIC=<median_psnr>

Usage::

    hatch run python scripts/benchmarks/benchmark_fitting_pareto.py
    hatch run python scripts/benchmarks/benchmark_fitting_pareto.py --budget 4000 --iters 2000
    hatch run python scripts/benchmarks/benchmark_fitting_pareto.py --baseline path/to/baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Dataset loaders (inline to avoid manuscript dependency)
# ---------------------------------------------------------------------------


def _load_kidney_dapi() -> tuple[np.ndarray, str]:
    from skimage.data import kidney

    raw = kidney()
    V = raw[:, :, :, 0].astype(np.float32)
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    return V, "kidney_dapi (16x512x512)"


def _load_kidney_actin() -> tuple[np.ndarray, str]:
    from skimage.data import kidney

    raw = kidney()
    V = raw[:, :, :, 2].astype(np.float32)
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    return V, "kidney_actin (16x512x512)"


def _load_opencell_ch0() -> tuple[np.ndarray, str]:
    import tempfile
    import urllib.request

    import tifffile

    cache_dir = Path.home() / ".cache" / "luxar" / "gsplats_opencell_map4"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tiff_path = cache_dir / "opencell_map4_stack.tif"
    if not tiff_path.exists():
        url = (
            "https://czb-opencell.s3.amazonaws.com/microscopy/raw/"
            "MAP4_ENSG00000047849/"
            "OC-FOV_MAP4_ENSG00000047849_CID000828_FID00002848_stack.tif"
        )
        print("Downloading OpenCell MAP4 TIFF (~70 MB)...")
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        os.close(fd)
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, str(tiff_path))
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    data = tifffile.imread(str(tiff_path))
    if data.ndim == 4 and data.shape[1] == 2:
        V = data[:, 0].astype(np.float32)
    elif data.ndim == 4 and data.shape[0] == 2:
        V = data[0].astype(np.float32)
    else:
        raise ValueError(f"Unexpected shape: {data.shape}")
    lo, hi = np.percentile(V, [1.0, 99.5])
    V = np.clip(V, lo, hi)
    V = ((V - lo) / (hi - lo + 1e-8)).astype(np.float32)
    return V, "opencell_ch0 (51x600x600)"


DATASET_LOADERS = [
    _load_kidney_dapi,
    _load_kidney_actin,
    _load_opencell_ch0,
]

# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------


def run_single_fit(
    V: np.ndarray,
    n_splats: int,
    n_iters: int,
    device: str,
) -> tuple[float, float]:
    """Fit and return (psnr_db, wall_time_seconds)."""
    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.metrics import compute_quality_metrics
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    t0 = time.perf_counter()
    gsplat_data = fit_gaussian_splats(
        V,
        seeds=n_splats,
        n_iters=n_iters,
        early_stop_patience=500,
        verbose=False,
    )
    fit_time = time.perf_counter() - t0

    # Render back and compute PSNR
    recon = render_to_volume_tensor(gsplat_data, V.shape, device=device)
    target = torch.as_tensor(V, dtype=torch.float32, device=recon.device)
    metrics = compute_quality_metrics(recon, target)
    psnr = float(metrics["psnr_db"])

    # Cleanup
    del gsplat_data, recon, target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return psnr, fit_time


def run_benchmark(
    n_splats: int, n_iters: int, device: str
) -> tuple[float, float, list[dict]]:
    """Run on all datasets, return (median_psnr, median_time, per_dataset_results)."""
    results = []
    for loader in DATASET_LOADERS:
        V, name = loader()
        print(f"  Fitting {name}: {n_splats} splats, {n_iters} iters...", flush=True)
        psnr, wall_time = run_single_fit(V, n_splats, n_iters, device)
        print(f"    PSNR={psnr:.2f} dB, time={wall_time:.1f}s")
        results.append({"dataset": name, "psnr_db": psnr, "time_s": wall_time})
        del V

    psnrs = [r["psnr_db"] for r in results]
    times = [r["time_s"] for r in results]
    return float(np.median(psnrs)), float(np.median(times)), results


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------

# Significance thresholds: ignore improvements smaller than these
PSNR_EPS = 0.05  # dB — below this is noise
TIME_EPS_FRAC = 0.02  # 2% — below this is measurement jitter


def pareto_dominates(
    new_psnr: float,
    new_time: float,
    old_psnr: float,
    old_time: float,
) -> bool:
    """Check if (new_psnr, new_time) Pareto-dominates (old_psnr, old_time).

    Both axes must be >= old (within tolerance), and at least one must be
    strictly and significantly better.
    """
    # "at least as good" checks (with tolerance for noise)
    psnr_ok = new_psnr >= old_psnr - PSNR_EPS
    time_ok = new_time <= old_time * (1 + TIME_EPS_FRAC)

    if not (psnr_ok and time_ok):
        return False

    # "at least one strictly better" check
    psnr_better = new_psnr > old_psnr + PSNR_EPS
    time_better = new_time < old_time * (1 - TIME_EPS_FRAC)

    return psnr_better or time_better


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Pareto fitting benchmark")
    parser.add_argument("--budget", type=int, default=8000, help="Splat count")
    parser.add_argument("--iters", type=int, default=15000, help="Max iterations")
    parser.add_argument(
        "--baseline",
        type=str,
        default="scripts/benchmarks/data/pareto_baseline.json",
        help="Path to baseline JSON",
    )
    parser.add_argument("--device", type=str, default=None, help="torch device")
    args = parser.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    baseline_path = Path(args.baseline)

    print("=== Pareto Fitting Benchmark ===")
    print(f"Budget: {args.budget} splats, {args.iters} iters, device={device}")
    print()

    median_psnr, median_time, per_dataset = run_benchmark(
        args.budget, args.iters, device
    )

    print()
    print(f"--- Aggregate (median across {len(per_dataset)} datasets) ---")
    print(f"  median_PSNR = {median_psnr:.4f} dB")
    print(f"  median_time = {median_time:.2f} s")

    # Load baseline if exists
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        old_psnr = baseline["median_psnr"]
        old_time = baseline["median_time"]
        print()
        print("--- Baseline ---")
        print(f"  median_PSNR = {old_psnr:.4f} dB")
        print(f"  median_time = {old_time:.2f} s")

        dominates = pareto_dominates(median_psnr, median_time, old_psnr, old_time)
        print()
        if dominates:
            print("PARETO: NEW DOMINATES BASELINE ✓")
        else:
            dpsnr = median_psnr - old_psnr
            dtime = median_time - old_time
            print(
                f"PARETO: NOT DOMINANT ✗  (ΔPSNR={dpsnr:+.4f} dB, Δtime={dtime:+.2f} s)"
            )
    else:
        dominates = True  # First run — accept as baseline
        print()
        print("PARETO: No baseline found — this run becomes the baseline.")

    # Save new result if dominant
    if dominates:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(
                {
                    "median_psnr": median_psnr,
                    "median_time": median_time,
                    "per_dataset": per_dataset,
                    "budget": args.budget,
                    "iters": args.iters,
                    "device": device,
                },
                f,
                indent=2,
            )

    # Output the metric line (for autoresearch extraction)
    print()
    print(f"METRIC={median_psnr:.4f}")

    sys.exit(0 if dominates else 1)


if __name__ == "__main__":
    main()
