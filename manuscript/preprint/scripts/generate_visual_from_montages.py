#!/usr/bin/env python3
"""Generate visual comparison figure from pre-computed analysis results.

Fallback script that assembles the main-paper visual comparison figure
directly from the cached slice NPZ files produced by
``manuscript/analysis/splat_count_vs_quality/run_analysis.py``.

No GPU required -- this only reads NumPy arrays and creates a matplotlib figure.

Usage::

    hatch run python manuscript/paper/generate_visual_from_montages.py
    hatch run python manuscript/paper/generate_visual_from_montages.py --output figs/visual_comparison/visual_comparison.pdf
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANALYSIS_RESULTS = (
    Path(__file__).resolve().parent.parent.parent
    / "analysis"
    / "splat_count_vs_quality"
    / "results"
)

# Datasets to include (row order). Each must have results from run_analysis.py.
DATASET_KEYS = ["kidney_dapi", "organoid_ch0", "tribolium"]

# Splat counts to show (must exist in the metrics TSV / slices directory).
# These are seeds_requested values -- the TSV maps them to actual splat counts.
TARGET_COUNTS = [4000, 32000, 128000]

# Short display names for row labels
DATASET_LABELS = {
    "kidney_dapi": "Mouse kidney\n(confocal, DAPI)",
    "organoid_ch0": "Organoid\n(confocal)",
    "tribolium": "Tribolium embryo\n(light-sheet)",
}

# Column headers
COLUMN_HEADERS = ["Original", "4K splats", "32K splats", "128K splats"]

# Approximate pixel sizes (um/voxel) for scale bars
# Kidney: scikit-image sample, ~0.5 um XY
# Organoid: CZI confocal, ~0.5 um XY
# Tribolium: 0.381 um isotropic (Preibisch et al.)
PIXEL_SIZES_UM = {
    "kidney_dapi": 0.5,
    "organoid_ch0": 0.5,
    "tribolium": 0.381,
}
SCALE_BAR_UM = {
    "kidney_dapi": 50,
    "organoid_ch0": 50,
    "tribolium": 100,
}

# Figure dimensions
FIG_WIDTH = 7.2  # inches (two-column)
FONT_SIZE = 7
PANEL_LABELS = "abcdefghijkl"

# Which of the 3 cached slices to show (0=25th pct, 1=50th pct, 2=75th pct)
SLICE_INDEX = 1  # middle slice


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_metrics(dataset_key: str) -> pd.DataFrame:
    """Load the metrics TSV for a dataset."""
    tsv_path = ANALYSIS_RESULTS / dataset_key / "metrics.tsv"
    if not tsv_path.exists():
        print(f"ERROR: {tsv_path} not found. Run run_analysis.py first.")
        sys.exit(1)
    return pd.read_csv(tsv_path, sep="\t").sort_values("seeds_requested").reset_index(drop=True)


def load_slices(dataset_key: str, seeds: int) -> dict[str, np.ndarray]:
    """Load the cached slice NPZ for a given splat count."""
    path = ANALYSIS_RESULTS / dataset_key / "slices" / f"n{seeds:06d}_slices.npz"
    if not path.exists():
        print(f"ERROR: {path} not found. Run run_analysis.py with seeds={seeds}.")
        sys.exit(1)
    return dict(np.load(str(path)))


def get_psnr(df: pd.DataFrame, seeds_requested: int) -> float:
    """Look up the PSNR for a specific seeds_requested value."""
    row = df[df["seeds_requested"] == seeds_requested]
    if row.empty:
        print(f"WARNING: No metrics row for seeds_requested={seeds_requested}")
        return 0.0
    return float(row.iloc[0]["psnr_db"])


def get_ssim(df: pd.DataFrame, seeds_requested: int) -> float:
    """Look up the SSIM for a specific seeds_requested value."""
    row = df[df["seeds_requested"] == seeds_requested]
    if row.empty:
        return 0.0
    return float(row.iloc[0]["ssim"])


def get_compression_ratio(df: pd.DataFrame, seeds_requested: int) -> float:
    """Look up the compression ratio for a specific seeds_requested value."""
    row = df[df["seeds_requested"] == seeds_requested]
    if row.empty:
        return 0.0
    return float(row.iloc[0]["compression_ratio"])


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


def generate_figure(output_path: Path) -> None:
    """Assemble the visual comparison figure from pre-computed data."""
    n_rows = len(DATASET_KEYS)
    n_cols = 1 + len(TARGET_COUNTS)  # Original + reconstructions

    # First pass: collect all slice data and determine aspect ratios
    all_slices: list[list[np.ndarray]] = []
    psnr_values: list[list[float | None]] = []
    ssim_values: list[list[float | None]] = []
    cr_values: list[list[float | None]] = []

    for ds_key in DATASET_KEYS:
        df = load_metrics(ds_key)

        # Load the original (target) slice from the first available count
        first_count = TARGET_COUNTS[0]
        data = load_slices(ds_key, first_count)
        target_slice = data["target_slices"][SLICE_INDEX]

        row_slices = [target_slice]
        row_psnr: list[float | None] = [None]
        row_ssim: list[float | None] = [None]
        row_cr: list[float | None] = [None]

        for seeds in TARGET_COUNTS:
            data = load_slices(ds_key, seeds)
            recon_slice = data["recon_slices"][SLICE_INDEX]
            psnr = get_psnr(df, seeds)
            ssim = get_ssim(df, seeds)
            cr = get_compression_ratio(df, seeds)

            row_slices.append(recon_slice)
            row_psnr.append(psnr)
            row_ssim.append(ssim)
            row_cr.append(cr)

        all_slices.append(row_slices)
        psnr_values.append(row_psnr)
        ssim_values.append(row_ssim)
        cr_values.append(row_cr)

    # Compute aspect ratios for proper sizing
    row_aspects = []
    for row_slices in all_slices:
        h, w = row_slices[0].shape
        row_aspects.append(h / w)

    # Column width (all equal)
    col_width = (FIG_WIDTH - 0.6) / n_cols
    row_heights = [col_width * asp for asp in row_aspects]
    fig_height = sum(row_heights) + 0.8

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_WIDTH, fig_height),
        gridspec_kw={
            "height_ratios": row_aspects,
            "wspace": 0.03,
            "hspace": 0.08,
            "left": 0.08,
            "right": 0.99,
            "top": 0.94,
            "bottom": 0.02,
        },
    )

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            img = all_slices[row][col]

            ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])

            # Panel label (a-l)
            panel_idx = row * n_cols + col
            if panel_idx < len(PANEL_LABELS):
                ax.text(
                    0.02, 0.98, PANEL_LABELS[panel_idx],
                    transform=ax.transAxes,
                    fontsize=FONT_SIZE + 1,
                    fontweight="bold",
                    color="white",
                    va="top", ha="left",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="black",
                        alpha=0.6,
                        edgecolor="none",
                    ),
                )

            # PSNR + CR annotation (reconstruction panels only)
            if col > 0 and psnr_values[row][col] is not None:
                psnr = psnr_values[row][col]
                cr = cr_values[row][col]
                label_text = f"{psnr:.1f} dB"
                if cr is not None and cr > 1:
                    label_text += f"\n{cr:.0f}x"
                ax.text(
                    0.98, 0.02, label_text,
                    transform=ax.transAxes,
                    fontsize=FONT_SIZE - 0.5,
                    color="white",
                    va="bottom", ha="right",
                    linespacing=1.1,
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="black",
                        alpha=0.65,
                        edgecolor="none",
                    ),
                )

            # Scale bar (first column only, bottom-left)
            if col == 0:
                ds_key = DATASET_KEYS[row]
                px_size = PIXEL_SIZES_UM.get(ds_key, 1.0)
                bar_um = SCALE_BAR_UM.get(ds_key, 50)
                bar_px = bar_um / px_size
                h_img, w_img = img.shape[:2]

                # Position: bottom-left. Text ABOVE bar (text first, bar underneath).
                margin_x = int(w_img * 0.05)
                bar_thickness = 3  # Fixed thickness for consistency across panels

                # Bar at very bottom
                bar_y = h_img - int(h_img * 0.06)
                bar_x0 = margin_x
                bar_x1 = bar_x0 + int(bar_px)

                # Draw bar
                ax.plot([bar_x0, bar_x1], [bar_y, bar_y],
                        color='white', linewidth=bar_thickness,
                        solid_capstyle='butt', zorder=10)
                # Text ABOVE the bar (clear separation)
                ax.text((bar_x0 + bar_x1) / 2, bar_y - int(h_img * 0.025),
                        f'{bar_um} \u00b5m', color='white',
                        fontsize=FONT_SIZE - 0.5, ha='center', va='bottom',
                        fontweight='bold',
                        zorder=10)

            # Column headers (top row only)
            if row == 0:
                ax.set_title(
                    COLUMN_HEADERS[col],
                    fontsize=FONT_SIZE,
                    fontweight="bold",
                    pad=4,
                )

            # Row labels (first column only)
            if col == 0:
                ax.set_ylabel(
                    DATASET_LABELS[DATASET_KEYS[row]],
                    fontsize=FONT_SIZE,
                    rotation=90,
                    labelpad=8,
                    va="center",
                )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    global SLICE_INDEX, DATASET_KEYS, TARGET_COUNTS, COLUMN_HEADERS

    parser = argparse.ArgumentParser(
        description="Assemble visual comparison figure from pre-computed analysis results"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--slice-index", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--counts", nargs="+", type=int, default=None)
    args = parser.parse_args()

    SLICE_INDEX = args.slice_index
    if args.datasets:
        DATASET_KEYS = args.datasets
    if args.counts:
        TARGET_COUNTS = args.counts

    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent / "figs" / "visual_comparison" / "visual_comparison.pdf"
    )

    # Validate
    for ds_key in DATASET_KEYS:
        ds_dir = ANALYSIS_RESULTS / ds_key
        if not ds_dir.exists():
            print(f"ERROR: Results directory not found: {ds_dir}")
            sys.exit(1)
        for count in TARGET_COUNTS:
            slices_path = ds_dir / "slices" / f"n{count:06d}_slices.npz"
            if not slices_path.exists():
                print(f"ERROR: Missing slice data: {slices_path}")
                sys.exit(1)

    def _format_count(n: int) -> str:
        return f"{n // 1000}K splats" if n >= 1000 else f"{n} splats"

    COLUMN_HEADERS = ["Original"] + [_format_count(c) for c in TARGET_COUNTS]

    for ds_key in DATASET_KEYS:
        if ds_key not in DATASET_LABELS:
            DATASET_LABELS[ds_key] = ds_key.replace("_", " ").title()

    print(f"Output: {output_path}")
    print(f"Datasets: {DATASET_KEYS}")
    print(f"Counts: {TARGET_COUNTS}")
    print(f"Slice index: {SLICE_INDEX}")

    generate_figure(output_path=output_path)


if __name__ == "__main__":
    main()
