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
    "opencell_lmnb1_ch0": "OpenCell LMNB1 — Hoechst (Nuclei)",
    "opencell_lmnb1_ch1": "OpenCell LMNB1 — GFP (Nuclear Lamina)",
    "cells3d_nuclei": "HeLa Cells — Nuclei",
    "cells3d_membrane": "HeLa Cells — Membrane",
    "acto3d_heart_nuclei": "Mouse Heart — Nuclei (Light-Sheet)",
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


def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicates: compute mean and std for each n_passes condition."""
    agg = (
        df.groupby("n_passes")
        .agg(
            psnr_db_mean=("psnr_db", "mean"),
            psnr_db_std=("psnr_db", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            fit_time_s_mean=("fit_time_s", "mean"),
            fit_time_s_std=("fit_time_s", "std"),
            max_abs_error_mean=("max_abs_error", "mean"),
            max_abs_error_std=("max_abs_error", "std"),
            n_replicates=("psnr_db", "count"),
            splats_budget=("splats_budget", "first"),
            iters_budget=("iters_budget", "first"),
        )
        .reset_index()
        .sort_values("n_passes")
    )
    for col in agg.columns:
        if col.endswith("_std"):
            agg[col] = agg[col].fillna(0.0)
    return agg


def plot_comparison(df: pd.DataFrame, dataset_key: str, output_path: Path):
    """Bar chart comparison: PSNR, SSIM, and time for each pass count."""
    agg = aggregate_data(df)

    fig, (ax_psnr, ax_ssim, ax_time) = plt.subplots(
        1, 3, figsize=(13, 5), constrained_layout=True
    )

    passes = agg["n_passes"].values
    x = np.arange(len(passes))
    width = 0.6
    labels = [f"{p}-pass" for p in passes]
    err_kw = {"linewidth": 0.8, "capsize": 3}

    # PSNR
    bars = ax_psnr.bar(
        x, agg["psnr_db_mean"].values, width,
        yerr=agg["psnr_db_std"].values, color=_COLORS[:len(passes)],
        error_kw=err_kw,
    )
    ax_psnr.set_xticks(x)
    ax_psnr.set_xticklabels(labels)
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    for bar, val, n in zip(bars, agg["psnr_db_mean"].values, agg["n_replicates"].values):
        y_top = bar.get_height()
        ax_psnr.text(
            bar.get_x() + bar.get_width() / 2, y_top + 0.1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8,
        )
        ax_psnr.text(
            bar.get_x() + bar.get_width() / 2, 0,
            f"n={n}", ha="center", va="bottom", fontsize=6, color="0.5",
        )

    # SSIM
    bars = ax_ssim.bar(
        x, agg["ssim_mean"].values, width,
        yerr=agg["ssim_std"].values, color=_COLORS[:len(passes)],
        error_kw=err_kw,
    )
    ax_ssim.set_xticks(x)
    ax_ssim.set_xticklabels(labels)
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    for bar, val in zip(bars, agg["ssim_mean"].values):
        ax_ssim.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8,
        )

    # Fitting time
    bars = ax_time.bar(
        x, agg["fit_time_s_mean"].values, width,
        yerr=agg["fit_time_s_std"].values, color=_COLORS[:len(passes)],
        error_kw=err_kw,
    )
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(labels)
    ax_time.set_ylabel("Fitting time (s)")
    ax_time.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    for bar, val in zip(bars, agg["fit_time_s_mean"].values):
        ax_time.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.0f}s", ha="center", va="bottom", fontsize=8,
        )

    budget_str = f"{agg['splats_budget'].iloc[0] // 1000}K splats, {agg['iters_budget'].iloc[0] // 1000}K iters"
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
    agg = aggregate_data(df)
    slices_dir = RESULTS_DIR / dataset_key / "slices"
    pass_counts = sorted(df["n_passes"].astype(int).unique())

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
    slice_axis = int(data0["slice_axis"]) if "slice_axis" in data0 else 0
    axis_label = ["z", "y", "x"][slice_axis] if slice_axis < 3 else f"ax{slice_axis}"
    z_indices = data0["z_indices"]
    n_slices = len(z_indices)

    # Find best pass (highest mean PSNR)
    best_passes = int(agg.loc[agg["psnr_db_mean"].idxmax(), "n_passes"])

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
        axes[row, 0].set_ylabel(f"{axis_label} = {z_idx}", fontsize=8)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # Columns 1..N: reconstructions for each pass count
        for col_off, p in enumerate(available):
            col = col_off + 1
            data = dict(np.load(str(slices_dir / f"pass{p:02d}_slices.npz")))
            recon_slices = data["recon_slices"]
            agg_row = agg[agg["n_passes"] == p].iloc[0]
            psnr = float(agg_row["psnr_db_mean"])
            ssim = float(agg_row["ssim_mean"])

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
