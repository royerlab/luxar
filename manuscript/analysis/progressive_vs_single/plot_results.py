#!/usr/bin/env python3
"""Plot progressive vs single-pass comparison.

Usage::

    hatch run python manuscript/analysis/progressive_vs_single/plot_results.py
    hatch run python manuscript/analysis/progressive_vs_single/plot_results.py --dataset kidney_dapi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"

DATASET_LABELS = {
    "opencell_map4_ch0": "OpenCell MAP4 — Hoechst (Nuclei)",
    "opencell_map4_ch1": "OpenCell MAP4 — GFP (Microtubules)",
    "kidney_dapi": "Mouse Kidney — DAPI (Nuclei)",
    "kidney_actin": "Mouse Kidney — Phalloidin (Actin)",
    "organoid_ch0": "Organoid — Channel 0",
    "celegans_t100": "C. elegans Embryo — t=100",
    "tribolium": "Tribolium Embryo (Light-Sheet)",
}

_COLORS = ["#2563eb", "#059669", "#d97706", "#dc2626"]


def _dataset_label(key: str) -> str:
    return DATASET_LABELS.get(key, key)


def load_data(dataset_key: str) -> pd.DataFrame:
    tsv_path = RESULTS_DIR / dataset_key / "progressive.tsv"
    if not tsv_path.exists():
        print(f"ERROR: {tsv_path} not found. Run run_analysis.py first.")
        sys.exit(1)
    return pd.read_csv(tsv_path, sep="\t").sort_values("n_passes")


def plot_comparison(df: pd.DataFrame, dataset_key: str, output_path: Path):
    """Bar chart comparison: PSNR, SSIM, and time for each pass count."""
    fig, (ax_psnr, ax_ssim, ax_time) = plt.subplots(
        1, 3, figsize=(13, 5), constrained_layout=True
    )

    passes = df["n_passes"].values
    x = np.arange(len(passes))
    width = 0.6
    labels = [f"{p}-pass" for p in passes]

    # PSNR
    bars = ax_psnr.bar(x, df["psnr_db"].values, width, color=_COLORS[:len(passes)])
    ax_psnr.set_xticks(x)
    ax_psnr.set_xticklabels(labels)
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    # Annotate bars
    for bar, val in zip(bars, df["psnr_db"].values):
        ax_psnr.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8,
        )

    # SSIM
    bars = ax_ssim.bar(x, df["ssim"].values, width, color=_COLORS[:len(passes)])
    ax_ssim.set_xticks(x)
    ax_ssim.set_xticklabels(labels)
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    for bar, val in zip(bars, df["ssim"].values):
        ax_ssim.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8,
        )

    # Fitting time
    bars = ax_time.bar(x, df["fit_time_s"].values, width, color=_COLORS[:len(passes)])
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels)
    ax_time.set_ylabel("Fitting time (s)")
    ax_time.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    for bar, val in zip(bars, df["fit_time_s"].values):
        ax_time.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.0f}s", ha="center", va="bottom", fontsize=8,
        )

    budget_str = f"{df['splats_budget'].iloc[0] // 1000}K splats, {df['iters_budget'].iloc[0] // 1000}K iters"
    fig.suptitle(
        f"Progressive vs Single-Pass — {_dataset_label(dataset_key)}\n"
        f"Fixed budget: {budget_str}",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_slice_montage(df: pd.DataFrame, dataset_key: str, output_path: Path):
    """Slice montage: rows = z-slices, cols = Target + each pass count + |Error| at best."""
    slices_dir = RESULTS_DIR / dataset_key / "slices"
    pass_counts = sorted(df["n_passes"].astype(int).values)

    # Check which slices are available
    available = []
    for p in pass_counts:
        f = slices_dir / f"pass{p:02d}_slices.npz"
        if f.exists():
            available.append(p)
    if not available:
        print(f"No slice data found in {slices_dir}. Skipping montage.")
        return

    # Load target from first available
    data0 = dict(np.load(str(slices_dir / f"pass{available[0]:02d}_slices.npz")))
    target_slices = data0["target_slices"]
    z_indices = data0["z_indices"]
    n_slices = len(z_indices)

    # Find best pass (highest PSNR)
    best_row = df.loc[df["psnr_db"].idxmax()]
    best_passes = int(best_row["n_passes"])

    # Columns: Target | 1-pass | 2-pass | 4-pass | 8-pass | |Error| at best
    n_cols = 1 + len(available) + 1
    cell_w, cell_h = 2.4, 2.6
    fig, axes = plt.subplots(
        n_slices, n_cols,
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

        # Columns 1..N: reconstructions for each pass count
        for col_off, p in enumerate(available):
            col = col_off + 1
            data = dict(np.load(str(slices_dir / f"pass{p:02d}_slices.npz")))
            recon_slices = data["recon_slices"]
            row_df = df[df["n_passes"] == p].iloc[0]
            psnr = float(row_df["psnr_db"])
            ssim = float(row_df["ssim"])

            axes[row, col].imshow(recon_slices[row], cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title(
                    f"{p}-pass\n{psnr:.1f} dB | SSIM {ssim:.3f}",
                    fontsize=6.5,
                )
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        # Last column: error map for best pass
        data_best = dict(np.load(str(slices_dir / f"pass{best_passes:02d}_slices.npz")))
        error = np.abs(target_slices[row] - data_best["recon_slices"][row])
        im = axes[row, -1].imshow(error, cmap="inferno", vmin=0, vmax=0.15)
        if row == 0:
            axes[row, -1].set_title("|Error|", fontsize=8, fontweight="bold")
        axes[row, -1].set_xticks([])
        axes[row, -1].set_yticks([])

    fig.colorbar(
        im, ax=axes[:, -1].tolist(), fraction=0.06, pad=0.02,
        label="Absolute error",
    )
    fig.suptitle(
        f"Progressive Slice Comparison — {_dataset_label(dataset_key)}",
        fontsize=11, fontweight="bold",
    )
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot progressive comparison")
    parser.add_argument("--dataset", default="opencell_map4_ch0", help="Dataset key")
    args = parser.parse_args()

    dataset_key = args.dataset
    df = load_data(dataset_key)
    print(f"Loaded {len(df)} conditions for {dataset_key}")

    out_dir = RESULTS_DIR / dataset_key
    plot_comparison(df, dataset_key, out_dir / "fig_progressive_comparison.pdf")
    plot_slice_montage(df, dataset_key, out_dir / "fig_progressive_slices.pdf")


if __name__ == "__main__":
    main()
