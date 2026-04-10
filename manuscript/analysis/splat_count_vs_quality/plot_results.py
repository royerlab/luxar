#!/usr/bin/env python3
"""Plotting: splat count vs quality PDF figures.

Reads the metrics TSV and cached slices produced by ``run_analysis.py``
and generates two publication-quality PDF figures:

  1. **fig_quality_curves.pdf** — PSNR, SSIM, and fitting time vs splat count
  2. **fig_slice_montage.pdf** — representative z-slices at selected splat counts

No GPU required. Usage::

    hatch run python manuscript/analysis/splat_count_vs_quality/plot_results.py
    hatch run python manuscript/analysis/splat_count_vs_quality/plot_results.py --dataset opencell_map4_ch1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"

# Subset of splat counts shown in the montage (avoids overcrowding)
MONTAGE_COUNTS = [1000, 4000, 16000, 64000, 128000]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_metrics(dataset_key: str) -> pd.DataFrame:
    """Load and sort the metrics TSV."""
    tsv_path = RESULTS_DIR / dataset_key / "metrics.tsv"
    if not tsv_path.exists():
        print(f"ERROR: {tsv_path} not found. Run run_analysis.py first.")
        sys.exit(1)
    df = pd.read_csv(tsv_path, sep="\t")
    return df.sort_values("seeds_requested").reset_index(drop=True)


def load_slices(dataset_key: str, seeds: int) -> dict[str, np.ndarray]:
    """Load cached slices npz for a given splat count."""
    path = RESULTS_DIR / dataset_key / "slices" / f"n{seeds:06d}_slices.npz"
    return dict(np.load(str(path)))


# ---------------------------------------------------------------------------
# Figure 1: Quality curves
# ---------------------------------------------------------------------------


def _format_count(x: float, _pos=None) -> str:
    """Format splat count as '1K', '16K', '128K', etc."""
    if x >= 1000:
        return f"{int(x / 1000)}K"
    return str(int(x))


def plot_quality_curves(df: pd.DataFrame, dataset_key: str, output_path: Path):
    """PSNR, SSIM, and fitting time vs splat count (log x-axis)."""
    fig, (ax_psnr, ax_ssim, ax_time) = plt.subplots(
        1, 3, figsize=(13, 4), constrained_layout=True
    )

    x = df["n_splats_final"].values

    # --- Left: PSNR ---
    ax_psnr.plot(x, df["psnr_db"].values, "o-", color="#2563eb", linewidth=1.5,
                 markersize=5)
    ax_psnr.set_xscale("log", base=2)
    ax_psnr.xaxis.set_major_formatter(ticker.FuncFormatter(_format_count))
    ax_psnr.set_xlabel("Number of Gaussians")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.grid(True, alpha=0.25, linewidth=0.5)

    # --- Center: SSIM ---
    ax_ssim.plot(x, df["ssim"].values, "s-", color="#d97706", linewidth=1.5,
                 markersize=5)
    ax_ssim.set_xscale("log", base=2)
    ax_ssim.xaxis.set_major_formatter(ticker.FuncFormatter(_format_count))
    ax_ssim.set_xlabel("Number of Gaussians")
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.grid(True, alpha=0.25, linewidth=0.5)

    # --- Right: Fitting time ---
    ax_time.plot(x, df["fit_time_s"].values, "^-", color="#059669", linewidth=1.5,
                 markersize=5)
    ax_time.set_xscale("log", base=2)
    ax_time.xaxis.set_major_formatter(ticker.FuncFormatter(_format_count))
    ax_time.set_xlabel("Number of Gaussians")
    ax_time.set_ylabel("Fitting time (s)")
    ax_time.grid(True, alpha=0.25, linewidth=0.5)

    # Secondary top axis: compression ratio
    for ax in [ax_psnr, ax_ssim, ax_time]:
        ax_top = ax.secondary_xaxis(
            "top",
            functions=(
                lambda n: df["compression_ratio"].values[0]
                * df["n_splats_final"].values[0]
                / np.clip(n, 1, None),
                lambda cr: df["compression_ratio"].values[0]
                * df["n_splats_final"].values[0]
                / np.clip(cr, 1e-6, None),
            ),
        )
        ax_top.set_xlabel("Compression ratio", fontsize=8)
        ax_top.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v:.0f}x")
        )
        ax_top.tick_params(labelsize=7)

    fig.suptitle(
        f"Reconstruction Quality vs Gaussian Count — {dataset_key}",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Slice montage
# ---------------------------------------------------------------------------


def plot_slice_montage(
    df: pd.DataFrame, dataset_key: str, output_path: Path
):
    """Grid of representative z-slices at selected splat counts + error map."""
    # Determine which montage counts are actually available
    available_seeds = set(df["seeds_requested"].astype(int).values)
    montage_counts = [c for c in MONTAGE_COUNTS if c in available_seeds]
    if not montage_counts:
        print("WARNING: No montage counts available in the TSV. Skipping montage.")
        return

    # Load reference slices (target is the same in all files)
    data0 = load_slices(dataset_key, montage_counts[0])
    target_slices = data0["target_slices"]
    z_indices = data0["z_indices"]
    n_slices = len(z_indices)

    # Columns: Target | recon_1 | ... | recon_N | |Error| at max
    n_cols = 1 + len(montage_counts) + 1
    cell_w, cell_h = 2.2, 2.4
    fig, axes = plt.subplots(
        n_slices,
        n_cols,
        figsize=(cell_w * n_cols, cell_h * n_slices),
        constrained_layout=True,
    )
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_slices):
        z_idx = int(z_indices[row])

        # Column 0: target
        axes[row, 0].imshow(target_slices[row], cmap="gray", vmin=0, vmax=1)
        if row == 0:
            axes[row, 0].set_title("Target", fontsize=8, fontweight="bold")
        axes[row, 0].set_ylabel(f"z = {z_idx}", fontsize=8)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # Columns 1..N: reconstructions
        for col_off, seeds in enumerate(montage_counts):
            col = col_off + 1
            data = load_slices(dataset_key, seeds)
            recon_slices = data["recon_slices"]
            row_df = df[df["seeds_requested"] == seeds].iloc[0]
            psnr = float(row_df["psnr_db"])

            axes[row, col].imshow(recon_slices[row], cmap="gray", vmin=0, vmax=1)
            if row == 0:
                label = _format_count(seeds)
                axes[row, col].set_title(
                    f"{label}\n{psnr:.1f} dB", fontsize=7
                )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        # Last column: absolute error for highest count
        data_max = load_slices(dataset_key, montage_counts[-1])
        error = np.abs(target_slices[row] - data_max["recon_slices"][row])
        im = axes[row, -1].imshow(error, cmap="inferno", vmin=0, vmax=0.15)
        if row == 0:
            axes[row, -1].set_title("|Error|", fontsize=8, fontweight="bold")
        axes[row, -1].set_xticks([])
        axes[row, -1].set_yticks([])

    # Shared colorbar for error maps
    fig.colorbar(
        im,
        ax=axes[:, -1].tolist(),
        fraction=0.06,
        pad=0.02,
        label="Absolute error",
    )

    fig.suptitle(
        f"Reconstruction Slices — {dataset_key}",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot splat count vs quality figures"
    )
    parser.add_argument(
        "--dataset",
        default="opencell_map4_ch0",
        help="Dataset key (must match run_analysis.py)",
    )
    args = parser.parse_args()

    dataset_key = args.dataset
    df = load_metrics(dataset_key)
    print(f"Loaded {len(df)} rows from {dataset_key}/metrics.tsv")

    out_dir = RESULTS_DIR / dataset_key
    plot_quality_curves(
        df, dataset_key, out_dir / "fig_quality_curves.pdf"
    )
    plot_slice_montage(
        df, dataset_key, out_dir / "fig_slice_montage.pdf"
    )


if __name__ == "__main__":
    main()
