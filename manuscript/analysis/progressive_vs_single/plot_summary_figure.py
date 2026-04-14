#!/usr/bin/env python3
"""Generate summary supplementary figure for progressive vs single-pass analysis.

Produces a compact two-panel figure:
  (a) PSNR difference (2-pass minus 1-pass) for all 12 datasets
  (b) Wall-clock time speedup (1-pass / 2-pass)

Usage::

    hatch run python manuscript/analysis/progressive_vs_single/plot_summary_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_PATH = (
    Path(__file__).parent.parent.parent
    / "preprint"
    / "figs"
    / "suppfig"
    / "progressive.pdf"
)

# Dataset order: group by modality, then alphabetically within group
DATASETS = [
    # Confocal
    "opencell_map4_ch0",
    "opencell_map4_ch1",
    "opencell_lmnb1_ch0",
    "opencell_lmnb1_ch1",
    "kidney_dapi",
    "kidney_actin",
    # Widefield
    "cells3d_nuclei",
    "cells3d_membrane",
    # Spinning-disk
    "organoid_ch0",
    # Embryo / developmental
    "celegans_t100",
    # Light-sheet
    "tribolium",
    "acto3d_heart_nuclei",
]

SHORT_LABELS = {
    "opencell_map4_ch0": "OpenCell MAP4\nnuclei",
    "opencell_map4_ch1": "OpenCell MAP4\nmicrotubules",
    "opencell_lmnb1_ch0": "OpenCell LMNB1\nnuclei",
    "opencell_lmnb1_ch1": "OpenCell LMNB1\nlamina",
    "kidney_dapi": "Kidney\nDAPI",
    "kidney_actin": "Kidney\nactin",
    "cells3d_nuclei": "HeLa\nnuclei",
    "cells3d_membrane": "HeLa\nmembrane",
    "organoid_ch0": "Organoid",
    "celegans_t100": "C. elegans\nembryo",
    "tribolium": "Tribolium\nembryo",
    "acto3d_heart_nuclei": "Mouse\nheart",
}

MODALITY_SPANS = [
    (0, 6, "Confocal"),
    (6, 8, "Widefield"),
    (8, 9, "Spinning-disk"),
    (9, 10, "Developmental"),
    (10, 12, "Light-sheet"),
]


def load_all() -> pd.DataFrame:
    """Load and aggregate results for all datasets."""
    rows = []
    for ds in DATASETS:
        tsv = RESULTS_DIR / ds / "progressive.tsv"
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep="\t")
        agg = df.groupby("n_passes").agg(
            psnr_mean=("psnr_db", "mean"),
            psnr_std=("psnr_db", "std"),
            ssim_mean=("ssim", "mean"),
            time_mean=("fit_time_s", "mean"),
            n_reps=("psnr_db", "count"),
        ).reset_index()
        agg["psnr_std"] = agg["psnr_std"].fillna(0.0)

        p1 = agg[agg["n_passes"] == 1].iloc[0]
        p2 = agg[agg["n_passes"] == 2].iloc[0]

        rows.append({
            "dataset": ds,
            "label": SHORT_LABELS[ds],
            "psnr_1": p1["psnr_mean"],
            "psnr_2": p2["psnr_mean"],
            "psnr_delta": p2["psnr_mean"] - p1["psnr_mean"],
            "psnr_1_std": p1["psnr_std"],
            "psnr_2_std": p2["psnr_std"],
            "ssim_1": p1["ssim_mean"],
            "ssim_2": p2["ssim_mean"],
            "time_1": p1["time_mean"],
            "time_2": p2["time_mean"],
            "speedup": p1["time_mean"] / p2["time_mean"],
        })
    return pd.DataFrame(rows)


def plot_summary(data: pd.DataFrame, output: Path):
    """Two-panel horizontal summary figure."""
    n = len(data)
    y = np.arange(n)

    fig, (ax_psnr, ax_speed) = plt.subplots(
        1, 2, figsize=(7.0, 3.8),
        gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.45},
    )

    # --- Panel (a): PSNR delta (2-pass minus 1-pass) ---
    deltas = data["psnr_delta"].values
    colors = ["#059669" if d > 0 else "#6366f1" for d in deltas]
    ax_psnr.barh(y, deltas, height=0.65, color=colors, edgecolor="white",
                 linewidth=0.3)

    # Propagated uncertainty for delta
    delta_std = np.sqrt(data["psnr_1_std"].values**2 + data["psnr_2_std"].values**2)
    ax_psnr.errorbar(deltas, y, xerr=delta_std, fmt="none", ecolor="0.4",
                     elinewidth=0.7, capsize=2)

    ax_psnr.axvline(0, color="0.3", linewidth=0.6, zorder=0)
    ax_psnr.set_yticks(y)
    ax_psnr.set_yticklabels(data["label"].values, fontsize=6.5)
    ax_psnr.set_xlabel("PSNR change (dB)\n2-pass minus 1-pass", fontsize=8)
    ax_psnr.invert_yaxis()
    ax_psnr.grid(True, axis="x", alpha=0.15, linewidth=0.5)
    ax_psnr.set_title("a", fontsize=10, fontweight="bold", loc="left", pad=4)

    # Annotate values
    for i, (d, s) in enumerate(zip(deltas, delta_std)):
        offset = 0.05 if d >= 0 else -0.05
        ha = "left" if d >= 0 else "right"
        ax_psnr.text(d + offset, i, f"{d:+.1f}", va="center", ha=ha, fontsize=6)

    # Add modality shading bands and labels
    _band_colors = ["#f0f0f0", "#ffffff"]
    for idx, (start, end, label) in enumerate(MODALITY_SPANS):
        ax_psnr.axhspan(start - 0.4, end - 0.6, color=_band_colors[idx % 2],
                        zorder=0, linewidth=0)
        ax_speed.axhspan(start - 0.4, end - 0.6, color=_band_colors[idx % 2],
                         zorder=0, linewidth=0)
    # Thin separator lines between modality groups
    for start, end, label in MODALITY_SPANS[1:]:
        for ax in (ax_psnr, ax_speed):
            ax.axhline(start - 0.5, color="0.75", linewidth=0.5, zorder=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#059669", label="2-pass better"),
        Patch(facecolor="#6366f1", label="1-pass better"),
    ]
    ax_psnr.legend(handles=legend_elements, fontsize=6, loc="lower right",
                   framealpha=0.8)

    # --- Panel (b): Time speedup ---
    speedups = data["speedup"].values
    ax_speed.barh(y, speedups, height=0.65, color="#2563eb", edgecolor="white",
                  linewidth=0.3, alpha=0.85)
    ax_speed.axvline(1, color="0.3", linewidth=0.6, zorder=0, linestyle="--")
    ax_speed.set_yticks(y)
    ax_speed.set_yticklabels([])
    ax_speed.set_xlabel("Wall-clock speedup\n(1-pass time / 2-pass time)", fontsize=8)
    ax_speed.invert_yaxis()
    ax_speed.grid(True, axis="x", alpha=0.15, linewidth=0.5)
    ax_speed.set_title("b", fontsize=10, fontweight="bold", loc="left", pad=4)

    for i, s in enumerate(speedups):
        ax_speed.text(s + 0.1, i, f"{s:.1f}x", va="center", ha="left", fontsize=6)

    fig.savefig(str(output), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    data = load_all()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_summary(data, OUTPUT_PATH)


if __name__ == "__main__":
    main()
