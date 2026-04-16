#!/usr/bin/env python3
"""Generate visual comparison figure: original vs Gaussian splat reconstructions.

Creates a publication-quality PDF figure showing original microscopy volume
slices alongside Gaussian splat reconstructions at different splat counts
(4K, 32K, 128K) for three representative datasets.

The script fits Gaussian splats from scratch, renders reconstructions,
extracts representative slices, and assembles the composite figure.

Usage::

    hatch run python manuscript/paper/generate_visual_comparisons.py
    hatch run python manuscript/paper/generate_visual_comparisons.py --device cuda
    hatch run python manuscript/paper/generate_visual_comparisons.py --device cpu  # slow!
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from arbol import aprint, asection

# Ensure the analysis package is importable (for the dataset registry)
ANALYSIS_DIR = Path(__file__).resolve().parent.parent.parent / "analysis" / "splat_count_vs_quality"
sys.path.insert(0, str(ANALYSIS_DIR))

from datasets import DATASETS  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Datasets to include (row order in the figure)
DATASET_KEYS = ["kidney_dapi", "organoid_ch0", "tribolium"]

# Splat counts to compare (column order, after the "Original" column)
SPLAT_COUNTS = [4000, 32000, 128000]

# Short display names for datasets (used as row labels)
DATASET_LABELS = {
    "kidney_dapi": "Mouse kidney\n(confocal, DAPI)",
    "organoid_ch0": "Organoid\n(confocal)",
    "tribolium": "Tribolium embryo\n(light-sheet)",
}

# Column headers
COLUMN_HEADERS = ["Original", "4K splats", "32K splats", "128K splats"]

# Fitting hyperparameters -- fast settings for manuscript generation
FIT_KWARGS = dict(
    n_iters=2000,
    lr=0.01,
    loss_type="l1",
    early_stop_patience=200,
    enable_dynamic_ops=True,
    cull_retention=0.999,
    seed_method="auto",
    verbose=True,
)

# Figure dimensions (two-column width)
FIG_WIDTH = 7.2  # inches (Nature/Science two-column width)
FONT_SIZE = 7
PANEL_LABELS = "abcdefghijkl"

# Slice percentile for picking the representative slice
SLICE_PERCENTILE = 0.50  # middle slice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pick_best_slice_axis(shape: tuple[int, ...]) -> int:
    """Pick the slicing axis that produces the most square 2D slices."""
    best_axis = 0
    best_ratio = float("inf")
    for ax in range(len(shape)):
        other = [shape[i] for i in range(len(shape)) if i != ax]
        if len(other) < 2:
            continue
        ratio = max(other) / (min(other) + 1e-9)
        if ratio < best_ratio:
            best_ratio = ratio
            best_axis = ax
    return best_axis


def extract_middle_slice(volume: np.ndarray, axis: int) -> np.ndarray:
    """Extract the middle slice along the given axis."""
    idx = volume.shape[axis] // 2
    slc = [slice(None)] * volume.ndim
    slc[axis] = idx
    return volume[tuple(slc)]


def compute_psnr(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio in dB."""
    mse = float(np.mean((original - reconstruction) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def fit_and_render(
    volume: np.ndarray, seeds: int, device: str
) -> np.ndarray:
    """Fit Gaussian splats and render back to a volume.

    Returns the reconstructed volume as a float32 ndarray.
    """
    import torch

    from luxar.gsplats.fit_gsplats import fit_gaussian_splats
    from luxar.gsplats.rendering.volume_rendering import render_to_volume

    gsplat_data = fit_gaussian_splats(
        volume, seeds=seeds, device=device, **FIT_KWARGS
    )

    with torch.no_grad():
        recon = render_to_volume(gsplat_data, shape=volume.shape, device=device)

    # Cleanup to free GPU memory
    del gsplat_data
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return recon


# ---------------------------------------------------------------------------
# Main figure generation
# ---------------------------------------------------------------------------


def generate_figure(device: str, output_path: Path) -> None:
    """Generate the visual comparison figure."""
    n_rows = len(DATASET_KEYS)
    n_cols = 1 + len(SPLAT_COUNTS)  # Original + reconstructions

    # Collect all slice data first
    # slices[row][col] = 2D array, psnr_values[row][col] = float (col=0 is original)
    all_slices: list[list[np.ndarray]] = []
    psnr_values: list[list[float | None]] = []

    for ds_key in DATASET_KEYS:
        with asection(f"Processing dataset: {ds_key}"):
            # Load the dataset
            volume, metadata = DATASETS[ds_key]()
            aprint(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")

            # Determine best slice axis and extract original slice
            axis = pick_best_slice_axis(volume.shape)
            aprint(f"Slice axis: {axis} (shape along other axes: "
                   f"{[volume.shape[i] for i in range(volume.ndim) if i != axis]})")

            original_slice = extract_middle_slice(volume, axis)
            aprint(f"Original slice shape: {original_slice.shape}")

            row_slices = [original_slice]
            row_psnr: list[float | None] = [None]  # No PSNR for original

            for seeds in SPLAT_COUNTS:
                with asection(f"Fitting {seeds:,} splats"):
                    recon = fit_and_render(volume, seeds=seeds, device=device)
                    recon_slice = extract_middle_slice(recon, axis)
                    psnr = compute_psnr(original_slice, recon_slice)
                    aprint(f"PSNR: {psnr:.2f} dB")

                    row_slices.append(recon_slice)
                    row_psnr.append(psnr)

                    del recon
                    gc.collect()

            all_slices.append(row_slices)
            psnr_values.append(row_psnr)

            del volume
            gc.collect()

    # --- Assemble figure ---
    with asection("Assembling figure"):
        # Compute aspect ratios for proper sizing
        # Use the first slice of each row to determine row aspect ratio
        row_aspects = []
        for row_slices in all_slices:
            h, w = row_slices[0].shape
            row_aspects.append(h / w)

        # Column width (all equal)
        col_width = (FIG_WIDTH - 0.6) / n_cols  # leave room for row labels
        row_heights = [col_width * asp for asp in row_aspects]
        fig_height = sum(row_heights) + 0.8  # room for column headers + bottom margin

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

                ax.imshow(img, cmap="inferno", vmin=0, vmax=1,
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

                # PSNR annotation (reconstruction panels only)
                if col > 0 and psnr_values[row][col] is not None:
                    psnr = psnr_values[row][col]
                    ax.text(
                        0.98, 0.02, f"{psnr:.1f} dB",
                        transform=ax.transAxes,
                        fontsize=FONT_SIZE,
                        color="white",
                        va="bottom", ha="right",
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            facecolor="black",
                            alpha=0.6,
                            edgecolor="none",
                        ),
                    )

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
        aprint(f"Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual comparison figure for Luxar manuscript"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device (cuda, mps, cpu). Auto-detects if not specified.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PDF path (default: figs/visual_comparison/visual_comparison.pdf)",
    )
    args = parser.parse_args()

    # Auto-detect device
    device = args.device
    if device is None:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    output_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent / "figs" / "visual_comparison" / "visual_comparison.pdf"
    )

    aprint(f"Device: {device}")
    aprint(f"Output: {output_path}")
    aprint(f"Datasets: {DATASET_KEYS}")
    aprint(f"Splat counts: {SPLAT_COUNTS}")

    generate_figure(device=device, output_path=output_path)


if __name__ == "__main__":
    main()
