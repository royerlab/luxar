#!/usr/bin/env python3
"""Plotting: splat count vs quality PDF figures.

Reads the metrics TSV and cached slices produced by ``run_analysis.py``
and generates publication-quality PDF figures:

  1. **fig_quality_curves.pdf** — PSNR, SSIM, fitting time, and culling vs splat count
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
MONTAGE_COUNTS = [1000, 4000, 16000, 64000, 256000, 512000]

# Dataset display names (for figure titles)
DATASET_LABELS = {
    "opencell_map4_ch0": "OpenCell MAP4 — Hoechst (Nuclei)",
    "opencell_map4_ch1": "OpenCell MAP4 — GFP (Microtubules)",
    "kidney_dapi": "Mouse Kidney — DAPI (Nuclei)",
    "kidney_actin": "Mouse Kidney — Phalloidin (Actin)",
    "organoid_ch0": "Organoid — Channel 0",
    "celegans_t100": "C. elegans Embryo — t=100",
    "tribolium": "Tribolium Embryo (Light-Sheet)",
    "opencell_lmnb1_ch0": "OpenCell LMNB1 — Hoechst (Nuclei)",
    "opencell_lmnb1_ch1": "OpenCell LMNB1 — GFP (Nuclear Lamina)",
    "cells3d_nuclei": "HeLa Cells — Nuclei",
    "cells3d_membrane": "HeLa Cells — Membrane",
    "acto3d_heart_nuclei": "Mouse Heart — Nuclei (Light-Sheet)",
}

# Consistent styling
_STYLE = dict(linewidth=1.5, markersize=5)
_COLORS = {
    "psnr": "#2563eb",
    "ssim": "#d97706",
    "time": "#059669",
    "cull": "#9333ea",
}


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


def _dataset_label(dataset_key: str) -> str:
    """Human-readable label for a dataset key."""
    return DATASET_LABELS.get(dataset_key, dataset_key)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_count(x: float, _pos=None) -> str:
    """Format splat count as '1K', '16K', '128K', etc."""
    if x >= 1000:
        return f"{int(x / 1000)}K"
    return str(int(x))


def _make_compression_top_axis(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Add a secondary top x-axis showing compression ratio."""
    # Compute constant: volume_voxels / floats_per_splat = cr * n_splats
    # Use first row to calibrate
    k = float(df["compression_ratio"].iloc[0]) * float(df["n_splats_final"].iloc[0])

    ax_top = ax.secondary_xaxis(
        "top",
        functions=(
            lambda n: k / np.clip(n, 1, None),
            lambda cr: k / np.clip(cr, 1e-6, None),
        ),
    )
    ax_top.set_xlabel("Compression ratio", fontsize=7, labelpad=4)
    ax_top.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0f}x" if v >= 1 else f"{v:.1f}x")
    )
    ax_top.tick_params(labelsize=7)


# ---------------------------------------------------------------------------
# Figure 1: Quality curves (2x2 grid)
# ---------------------------------------------------------------------------


def plot_quality_curves(df: pd.DataFrame, dataset_key: str, output_path: Path):
    """PSNR, SSIM, fitting time, and culling vs splat count."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    ax_psnr, ax_ssim = axes[0]
    ax_time, ax_cull = axes[1]

    x = df["n_splats_final"].values
    x_req = df["seeds_requested"].values

    # --- PSNR ---
    ax_psnr.plot(x, df["psnr_db"].values, "o-", color=_COLORS["psnr"], **_STYLE)
    ax_psnr.set_xscale("log", base=2)
    ax_psnr.xaxis.set_major_formatter(ticker.FuncFormatter(_format_count))
    ax_psnr.set_xlabel("Gaussian Splats (effective)")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.grid(True, alpha=0.2, linewidth=0.5)

    # Overlay noise floor if available
    try:
        from noise_floor import load_noise_floor

        nf = load_noise_floor(dataset_key)
        if nf is not None:
            ax_psnr.axhline(
                nf["psnr_max_db"], color="black", linestyle="--",
                linewidth=1.0, alpha=0.5,
                label=f"Noise floor ({nf['psnr_max_db']:.1f} dB)",
            )
            ax_psnr.legend(fontsize=7, loc="lower right")
    except ImportError:
        pass

    _make_compression_top_axis(ax_psnr, df)

    # --- SSIM ---
    ax_ssim.plot(x, df["ssim"].values, "s-", color=_COLORS["ssim"], **_STYLE)
    ax_ssim.set_xscale("log", base=2)
    ax_ssim.xaxis.set_major_formatter(ticker.FuncFormatter(_format_count))
    ax_ssim.set_xlabel("Gaussian Splats (effective)")
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.grid(True, alpha=0.2, linewidth=0.5)
    _make_compression_top_axis(ax_ssim, df)

    # --- Fitting time ---
    ax_time.plot(x, df["fit_time_s"].values, "^-", color=_COLORS["time"], **_STYLE)
    ax_time.set_xscale("log", base=2)
    ax_time.xaxis.set_major_formatter(ticker.FuncFormatter(_format_count))
    ax_time.set_xlabel("Gaussian Splats (effective)")
    ax_time.set_ylabel("Fitting time (s)")
    ax_time.grid(True, alpha=0.2, linewidth=0.5)
    _make_compression_top_axis(ax_time, df)

    # --- Culling ---
    cull_pct = (1 - x / x_req) * 100
    ax_cull.plot(x_req, cull_pct, "D-", color=_COLORS["cull"], **_STYLE)
    ax_cull.set_xscale("log", base=2)
    ax_cull.xaxis.set_major_formatter(ticker.FuncFormatter(_format_count))
    ax_cull.set_xlabel("Seeds requested")
    ax_cull.set_ylabel("Splats culled (%)")
    ax_cull.grid(True, alpha=0.2, linewidth=0.5)
    ax_cull.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    # Annotate final counts
    for xr, xf, pct in zip(x_req, x, cull_pct):
        ax_cull.annotate(
            f"{_format_count(xf)}",
            (xr, pct),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=6,
            ha="center",
            color=_COLORS["cull"],
        )

    fig.suptitle(
        f"Reconstruction Quality vs Gaussian Count\n{_dataset_label(dataset_key)}",
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
    cell_w, cell_h = 2.4, 2.6
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
            ssim = float(row_df["ssim"])
            cr = float(row_df["compression_ratio"])

            axes[row, col].imshow(recon_slices[row], cmap="gray", vmin=0, vmax=1)
            if row == 0:
                label = _format_count(seeds)
                axes[row, col].set_title(
                    f"{label}\n{psnr:.1f} dB  |  SSIM {ssim:.3f}\n{cr:.0f}x compression",
                    fontsize=6.5,
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
        f"Reconstruction Slices — {_dataset_label(dataset_key)}",
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
