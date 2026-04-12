#!/usr/bin/env python3
"""Plot Noise2Self analysis: train vs held-out loss curves.

Reads the N2S metrics TSV and generates a figure showing where the
model transitions from fitting signal to fitting noise.

Usage::

    hatch run python manuscript/analysis/splat_count_vs_quality/plot_noise2self.py
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

DATASET_LABELS = {
    "opencell_map4_ch0": "OpenCell MAP4 — Hoechst (Nuclei)",
    "opencell_map4_ch1": "OpenCell MAP4 — GFP (Microtubules)",
    "kidney_dapi": "Mouse Kidney — DAPI (Nuclei)",
    "kidney_actin": "Mouse Kidney — Phalloidin (Actin)",
    "organoid_ch0": "Organoid — Channel 0",
    "celegans_t100": "C. elegans Embryo — t=100",
    "tribolium": "Tribolium Embryo (Light-Sheet)",
}


def _format_count(x: float, _pos=None) -> str:
    if x >= 1000:
        return f"{int(x / 1000)}K"
    return str(int(x))


def _dataset_label(key: str) -> str:
    return DATASET_LABELS.get(key, key)


def load_metrics(dataset_key: str) -> pd.DataFrame:
    tsv_path = RESULTS_DIR / f"{dataset_key}_n2s" / "metrics_n2s.tsv"
    if not tsv_path.exists():
        print(f"ERROR: {tsv_path} not found. Run run_noise2self.py first.")
        sys.exit(1)
    df = pd.read_csv(tsv_path, sep="\t")
    return df.sort_values("seeds_requested").reset_index(drop=True)


def plot_noise2self(df: pd.DataFrame, dataset_key: str, output_path: Path):
    """Train vs held-out loss, gap (noise absorption), and culling."""
    fig, (ax_psnr, ax_mse, ax_cull) = plt.subplots(
        1, 3, figsize=(15, 4.5), constrained_layout=True
    )

    x = df["n_splats_final"].values
    x_req = df["seeds_requested"].values

    # --- Left: PSNR (train vs held-out) ---
    ax_psnr.plot(
        x, df["train_psnr_db"].values, "o-",
        color="#2563eb", linewidth=1.5, markersize=5, label="Train (unmasked)",
    )
    ax_psnr.plot(
        x, df["held_out_psnr_db"].values, "s-",
        color="#dc2626", linewidth=1.5, markersize=5, label="Held-out (masked)",
    )
    ax_psnr.plot(
        x, df["full_psnr_db"].values, "^--",
        color="#6b7280", linewidth=1.0, markersize=4, alpha=0.6, label="Full volume",
    )

    # Mark the held-out PSNR peak — the optimal splat count.
    held_psnr = df["held_out_psnr_db"].values
    peak_idx = int(np.argmax(held_psnr))
    peak_val = held_psnr[peak_idx]
    final_val = held_psnr[-1]
    drop = peak_val - final_val

    # Only annotate if the peak is meaningful (not the last point, and >0.2 dB drop)
    if peak_idx < len(x) - 1 and drop > 0.2:
        cx, cy = x[peak_idx], peak_val
        ax_psnr.axvline(cx, color="#9333ea", linestyle=":", alpha=0.5)
        ax_psnr.annotate(
            f"Held-out peak\n{_format_count(cx)} splats\n{peak_val:.1f} dB",
            (cx, cy),
            textcoords="offset points",
            xytext=(-70, -30),
            fontsize=8,
            color="#9333ea",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#9333ea", alpha=0.6),
        )
        crossover_idx = peak_idx
    else:
        crossover_idx = None

    ax_psnr.set_xscale("log", base=2)
    ax_psnr.xaxis.set_major_formatter(ticker.FuncFormatter(_format_count))
    ax_psnr.set_xlabel("Number of Gaussians")
    ax_psnr.set_ylabel("PSNR (dB)")
    # Overlay noise floor if available
    try:
        from noise_floor import load_noise_floor

        _nf = load_noise_floor(dataset_key)
        if _nf is not None:
            ax_psnr.axhline(
                _nf["psnr_max_db"], color="black", linestyle="--",
                linewidth=1.0, alpha=0.5,
                label=f"Noise floor ({_nf['psnr_max_db']:.1f} dB)",
            )
    except ImportError:
        _nf = None

    ax_psnr.legend(fontsize=8, loc="lower right")
    ax_psnr.grid(True, alpha=0.2, linewidth=0.5)

    # --- Right: MSE (train vs held-out) ---
    ax_mse.plot(
        x, df["train_mse"].values, "o-",
        color="#2563eb", linewidth=1.5, markersize=5, label="Train (unmasked)",
    )
    ax_mse.plot(
        x, df["held_out_mse"].values, "s-",
        color="#dc2626", linewidth=1.5, markersize=5, label="Held-out (masked)",
    )

    # Shade the gap (noise absorption region)
    ax_mse.fill_between(
        x,
        df["train_mse"].values,
        df["held_out_mse"].values,
        alpha=0.12,
        color="#dc2626",
        label="Gap (noise fitting)",
    )

    # Mark 50% noise crossover on MSE plot too
    if crossover_idx is not None:
        ax_mse.axvline(x[crossover_idx], color="#9333ea", linestyle=":", alpha=0.5)

    ax_mse.set_xscale("log", base=2)
    ax_mse.xaxis.set_major_formatter(ticker.FuncFormatter(_format_count))
    ax_mse.set_xlabel("Number of Gaussians")
    ax_mse.set_ylabel("MSE")
    # Overlay noise variance if available
    if _nf is not None:
        ax_mse.axhline(
            _nf["noise_variance"], color="black", linestyle="--",
            linewidth=1.0, alpha=0.5,
            label=f"Noise variance ({_nf['noise_variance']:.5f})",
        )

    ax_mse.legend(fontsize=8, loc="upper right")
    ax_mse.grid(True, alpha=0.2, linewidth=0.5)

    # --- Right: Culling (requested vs final) ---
    cull_pct = (1 - x / x_req) * 100
    ax_cull.plot(
        x_req, cull_pct, "D-",
        color="#9333ea", linewidth=1.5, markersize=5,
    )
    ax_cull.set_xscale("log", base=2)
    ax_cull.xaxis.set_major_formatter(ticker.FuncFormatter(_format_count))
    ax_cull.set_xlabel("Seeds requested")
    ax_cull.set_ylabel("Splats culled (%)")
    ax_cull.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax_cull.grid(True, alpha=0.2, linewidth=0.5)
    # Annotate final counts
    for xr, xf, pct in zip(x_req, x, cull_pct):
        ax_cull.annotate(
            f"{_format_count(xf)}",
            (xr, pct),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=6,
            ha="center",
            color="#9333ea",
        )

    fig.suptitle(
        f"Noise2Self Analysis — Signal vs Noise Fitting\n{_dataset_label(dataset_key)}",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Noise2Self train vs held-out analysis"
    )
    parser.add_argument(
        "--dataset",
        default="opencell_map4_ch0",
        help="Dataset key (must match run_noise2self.py)",
    )
    args = parser.parse_args()

    dataset_key = args.dataset
    df = load_metrics(dataset_key)
    print(f"Loaded {len(df)} rows from {dataset_key}_n2s/metrics_n2s.tsv")

    out_dir = RESULTS_DIR / f"{dataset_key}_n2s"
    plot_noise2self(df, dataset_key, out_dir / "fig_noise2self.pdf")


if __name__ == "__main__":
    main()
