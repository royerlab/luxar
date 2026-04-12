#!/usr/bin/env python3
"""Pareto-dominance benchmark for progressive (multi-pass) gsplat fitting.

Fits on 3 diverse microscopy datasets with a fixed budget, reports
(median_PSNR, median_time), and checks Pareto dominance against stored baseline.

Exit code:
  0 — New result Pareto-dominates (or no baseline yet)
  1 — Not Pareto-dominant

Usage::

    hatch run python scripts/benchmarks/benchmark_progressive_pareto.py
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
# Dataset loaders (same as single-pass benchmark)
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
        print(f"Downloading OpenCell MAP4 TIFF (~70 MB)...")
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
    max_splats: int,
    max_splats_per_pass: int,
    iters_per_pass: int,
    device: str,
) -> tuple[float, float]:
    """Fit progressively and return (psnr_db, wall_time_seconds)."""
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
    from luxar.gsplats.metrics import compute_quality_metrics
    from luxar.gsplats.rendering.volume_rendering import render_to_volume_tensor

    t0 = time.perf_counter()
    gsplat_data = fit_progressive_gaussian_splats(
        V,
        max_splats=max_splats,
        max_splats_per_pass=max_splats_per_pass,
        iters_per_pass=iters_per_pass,
        psnr_patience=0.3,
        max_passes=4,
        device=device,
        verbose=False,
    )
    fit_time = time.perf_counter() - t0

    # Render combined result and compute PSNR
    recon = render_to_volume_tensor(gsplat_data, V.shape, device=device)
    target = torch.as_tensor(V, dtype=torch.float32, device=recon.device)
    metrics = compute_quality_metrics(recon, target)
    psnr = float(metrics["psnr_db"])

    del gsplat_data, recon, target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return psnr, fit_time


def run_benchmark(
    max_splats: int,
    max_splats_per_pass: int,
    iters_per_pass: int,
    device: str,
) -> tuple[float, float, list[dict]]:
    """Run on all datasets, return (median_psnr, median_time, per_dataset)."""
    results = []
    for loader in DATASET_LOADERS:
        V, name = loader()
        print(
            f"  Fitting {name}: {max_splats} total splats, "
            f"{max_splats_per_pass}/pass, {iters_per_pass} iters/pass...",
            flush=True,
        )
        psnr, wall_time = run_single_fit(
            V, max_splats, max_splats_per_pass, iters_per_pass, device
        )
        print(f"    PSNR={psnr:.2f} dB, time={wall_time:.1f}s")
        results.append({"dataset": name, "psnr_db": psnr, "time_s": wall_time})
        del V

    psnrs = [r["psnr_db"] for r in results]
    times = [r["time_s"] for r in results]
    return float(np.median(psnrs)), float(np.median(times)), results


# ---------------------------------------------------------------------------
# Pareto dominance (same as single-pass)
# ---------------------------------------------------------------------------

PSNR_EPS = 0.05
TIME_EPS_FRAC = 0.02


def pareto_dominates(
    new_psnr: float, new_time: float, old_psnr: float, old_time: float,
) -> bool:
    psnr_ok = new_psnr >= old_psnr - PSNR_EPS
    time_ok = new_time <= old_time * (1 + TIME_EPS_FRAC)
    if not (psnr_ok and time_ok):
        return False
    psnr_better = new_psnr > old_psnr + PSNR_EPS
    time_better = new_time < old_time * (1 - TIME_EPS_FRAC)
    return psnr_better or time_better


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Progressive Pareto benchmark")
    parser.add_argument("--max-splats", type=int, default=16000)
    parser.add_argument("--splats-per-pass", type=int, default=8000)
    parser.add_argument("--iters-per-pass", type=int, default=5000)
    parser.add_argument(
        "--baseline", type=str,
        default="scripts/benchmarks/data/progressive_pareto_baseline.json",
    )
    parser.add_argument("--device", type=str, default=None)
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

    print(f"=== Progressive Pareto Fitting Benchmark ===")
    print(
        f"Budget: {args.max_splats} total, {args.splats_per_pass}/pass, "
        f"{args.iters_per_pass} iters/pass, device={device}"
    )
    print()

    median_psnr, median_time, per_dataset = run_benchmark(
        args.max_splats, args.splats_per_pass, args.iters_per_pass, device
    )

    print()
    print(f"--- Aggregate (median across {len(per_dataset)} datasets) ---")
    print(f"  median_PSNR = {median_psnr:.4f} dB")
    print(f"  median_time = {median_time:.2f} s")

    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        old_psnr = baseline["median_psnr"]
        old_time = baseline["median_time"]
        print()
        print(f"--- Baseline ---")
        print(f"  median_PSNR = {old_psnr:.4f} dB")
        print(f"  median_time = {old_time:.2f} s")

        dominates = pareto_dominates(median_psnr, median_time, old_psnr, old_time)
        print()
        if dominates:
            print("PARETO: NEW DOMINATES BASELINE")
        else:
            dpsnr = median_psnr - old_psnr
            dtime = median_time - old_time
            print(
                f"PARETO: NOT DOMINANT "
                f"(dPSNR={dpsnr:+.4f} dB, dtime={dtime:+.2f} s)"
            )
    else:
        dominates = True
        print()
        print("PARETO: No baseline found — this run becomes the baseline.")

    if dominates:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(
                {
                    "median_psnr": median_psnr,
                    "median_time": median_time,
                    "per_dataset": per_dataset,
                    "max_splats": args.max_splats,
                    "splats_per_pass": args.splats_per_pass,
                    "iters_per_pass": args.iters_per_pass,
                    "device": device,
                },
                f,
                indent=2,
            )

    print()
    print(f"METRIC={median_psnr:.4f}")
    sys.exit(0 if dominates else 1)


if __name__ == "__main__":
    main()
