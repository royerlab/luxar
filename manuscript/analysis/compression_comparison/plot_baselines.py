#!/usr/bin/env python3
"""Plot rate-distortion comparison: GSplats vs traditional compression."""

import matplotlib

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
FIGS_DIR = Path(__file__).parent.parent.parent / "preprint" / "figs"

DATASET_NAMES = {
    'kidney_dapi': 'Kidney DAPI',
    'organoid_ch0': 'Organoid',
    'celegans_t100': 'C. elegans',
    'tribolium': 'Tribolium',
    'opencell_map4_ch0': 'OpenCell nuclei',
}

LOSSLESS_COLORS = {
    'blosc-lz4-shuffle': '#E69F00',       # orange
    'blosc-zstd-shuffle': '#56B4E9',      # sky blue
    'blosc-zstd-bitshuffle': '#009E73',   # green
}

LOSSLESS_LABELS = {
    'blosc-lz4-shuffle': 'LZ4+shuffle',
    'blosc-zstd-shuffle': 'zstd+shuffle',
    'blosc-zstd-bitshuffle': 'zstd+bitshuffle',
}


def main():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 7,
        'axes.linewidth': 0.4, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    df = pd.read_csv(RESULTS_DIR / "compression_baselines.tsv", sep='\t')

    datasets = ['kidney_dapi', 'organoid_ch0', 'tribolium']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[df['dataset'] == ds]

        # GSplats curve (solid blue)
        gsplats = sub[sub['method'] == 'gsplats'].sort_values('bpv')
        ax.plot(gsplats['bpv'], gsplats['psnr_db'], '-o',
                color='#0072B2', markersize=3, linewidth=1.2,
                markeredgewidth=0.3, markeredgecolor='white',
                label='GSplats', zorder=5)

        # Quantization curve (dashed grey)
        quant = sub[sub['method'].str.startswith('quant')].sort_values('bpv')
        ax.plot(quant['bpv'], quant['psnr_db'], '--s',
                color='#999999', markersize=3, linewidth=0.9,
                markeredgewidth=0.3, markeredgecolor='white',
                label='Quantize+zstd', zorder=3)

        # Lossless baselines: each in a distinct color
        lossless = sub[sub['type'] == 'lossless']
        for _, lr in lossless.iterrows():
            method = lr['method']
            color = LOSSLESS_COLORS.get(method, '#cccccc')
            label = LOSSLESS_LABELS.get(method, method) if i == 0 else None
            ax.axvline(x=lr['bpv'], color=color, linewidth=0.7,
                       linestyle=':', zorder=1, label=label)

        ax.set_xscale('log')
        ax.set_xlabel('Bits per voxel')
        if i == 0:
            ax.set_ylabel('PSNR (dB)')
        ax.set_title(DATASET_NAMES[ds], fontsize=8, fontweight='bold', pad=3)
        ax.set_ylim(18, 52)

        if i == 0:
            ax.legend(frameon=False, fontsize=5, loc='lower right',
                      handlelength=1.5, handletextpad=0.3)

        ax.text(-0.10, 1.06, chr(ord('a') + i), transform=ax.transAxes,
                fontsize=10, fontweight='bold')

        # Annotations for panel b only
        if i == 1:
            ax.annotate('directly\nrenderable', xy=(0.15, 37),
                        fontsize=5, color='#0072B2', ha='center',
                        fontweight='bold', fontstyle='italic')
            ax.annotate('requires\ndecompression', xy=(5, 49),
                        fontsize=5, color='#999999', ha='center',
                        fontstyle='italic')

    plt.tight_layout()
    out = FIGS_DIR / "suppfig_compression.pdf"
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
