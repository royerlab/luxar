#!/usr/bin/env python3
"""Plot convergence curves: PSNR and SSIM vs iteration for each splat count.

Usage::

    hatch run python manuscript/analysis/convergence/plot_convergence.py
    hatch run python manuscript/analysis/convergence/plot_convergence.py --dataset kidney_dapi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

_COLORS = [
    "#2563eb", "#059669", "#d97706", "#dc2626", "#9333ea",
    "#0891b2", "#be185d", "#65a30d",
]


def _format_count(x: float, _pos=None) -> str:
    if x >= 1000:
        return f"{int(x / 1000)}K"
    return str(int(x))


def _dataset_label(key: str) -> str:
    return DATASET_LABELS.get(key, key)


def load_convergence(dataset_key: str) -> pd.DataFrame:
    tsv_path = RESULTS_DIR / dataset_key / "convergence.tsv"
    if not tsv_path.exists():
        print(f"ERROR: {tsv_path} not found. Run run_convergence.py first.")
        sys.exit(1)
    return pd.read_csv(tsv_path, sep="\t")


def plot_convergence(df: pd.DataFrame, dataset_key: str, output_path: Path):
    """PSNR and SSIM vs iteration, one curve per splat count."""
    fig, (ax_psnr, ax_ssim, ax_time) = plt.subplots(
        1, 3, figsize=(15, 5), constrained_layout=True
    )

    seeds_list = sorted(df["seeds_requested"].unique())

    for i, seeds in enumerate(seeds_list):
        sub = df[df["seeds_requested"] == seeds].sort_values("n_iters")
        color = _COLORS[i % len(_COLORS)]
        label = f"{_format_count(seeds)} Gaussian Splats"

        # PSNR vs iteration
        ax_psnr.plot(
            sub["n_iters"], sub["psnr_db"],
            "o-", color=color, linewidth=1.5, markersize=4, label=label,
        )

        # SSIM vs iteration
        ax_ssim.plot(
            sub["n_iters"], sub["ssim"],
            "s-", color=color, linewidth=1.5, markersize=4, label=label,
        )

        # PSNR vs wall-clock time
        ax_time.plot(
            sub["fit_time_s"], sub["psnr_db"],
            "^-", color=color, linewidth=1.5, markersize=4, label=label,
        )

    # --- Left: PSNR vs iterations ---
    ax_psnr.set_xlabel("Iterations")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_xscale("log")
    ax_psnr.legend(fontsize=7, loc="lower right")
    ax_psnr.grid(True, alpha=0.2, linewidth=0.5)

    # --- Center: SSIM vs iterations ---
    ax_ssim.set_xlabel("Iterations")
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.set_xscale("log")
    ax_ssim.legend(fontsize=7, loc="lower right")
    ax_ssim.grid(True, alpha=0.2, linewidth=0.5)

    # --- Right: PSNR vs wall-clock time ---
    ax_time.set_xlabel("Fitting time (s)")
    ax_time.set_ylabel("PSNR (dB)")
    ax_time.legend(fontsize=7, loc="lower right")
    ax_time.grid(True, alpha=0.2, linewidth=0.5)

    fig.suptitle(
        f"Convergence — {_dataset_label(dataset_key)}",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot convergence curves")
    parser.add_argument(
        "--dataset",
        default="opencell_map4_ch0",
        help="Dataset key",
    )
    args = parser.parse_args()

    dataset_key = args.dataset
    df = load_convergence(dataset_key)
    n_seeds = df["seeds_requested"].nunique()
    n_iters = df["n_iters"].nunique()
    print(f"Loaded {len(df)} rows: {n_seeds} splat counts x {n_iters} checkpoints")

    out_dir = RESULTS_DIR / dataset_key
    plot_convergence(df, dataset_key, out_dir / "fig_convergence.pdf")


if __name__ == "__main__":
    main()
