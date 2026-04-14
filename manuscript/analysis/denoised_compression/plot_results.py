#!/usr/bin/env python3
"""Plot denoised compression comparison: raw PSNR vs held-out PSNR.

Shows that quantization's apparent advantage in raw PSNR largely reflects
noise preservation, while GSplats selectively capture signal.
"""

import matplotlib

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
FIGS_DIR = Path(__file__).parent.parent.parent / "preprint" / "figs"

NAMES = {
    'kidney_dapi': 'Kidney DAPI',
    'organoid_ch0': 'Organoid',
    'celegans_t100': 'C. elegans',
    'tribolium': 'Tribolium',
    'opencell_map4_ch0': 'OpenCell nuclei',
}


def main():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 7,
        'axes.linewidth': 0.4, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    df = pd.read_csv(RESULTS_DIR / "denoised_comparison.tsv", sep='\t')

    datasets = ['kidney_dapi', 'organoid_ch0', 'tribolium']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))

    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[(df['dataset'] == ds) & (df['method'] != 'noise_floor')]
        nf = df[(df['dataset'] == ds) & (df['method'] == 'noise_floor')]

        # GSplats: show both raw and held-out PSNR
        gs = sub[sub['method'] == 'gsplats'].sort_values('bpv')
        ax.plot(gs['bpv'], gs['psnr_raw'], '-o',
                color='#0072B2', markersize=3, linewidth=1.0,
                markeredgewidth=0.3, markeredgecolor='white',
                label='GSplats (raw)', zorder=5, alpha=0.4)
        ax.plot(gs['bpv'], gs['psnr_heldout'], '-o',
                color='#0072B2', markersize=3.5, linewidth=1.2,
                markeredgewidth=0.3, markeredgecolor='white',
                label='GSplats (held-out)', zorder=6)

        # Mark N2S-optimal (held-out peak)
        if not gs.empty:
            peak_idx = gs['psnr_heldout'].idxmax()
            last_idx = gs.index[-1]
            if peak_idx != last_idx:
                ax.plot(gs.loc[peak_idx, 'bpv'], gs.loc[peak_idx, 'psnr_heldout'],
                        '*', color='#0072B2', markersize=10,
                        markeredgecolor='black', markeredgewidth=0.3, zorder=7)

        # Quantization: raw PSNR only (= held-out for non-learning methods)
        quant = sub[sub['method'].str.startswith('quant')].sort_values('bpv')
        ax.plot(quant['bpv'], quant['psnr_raw'], '--s',
                color='#999999', markersize=3, linewidth=0.9,
                markeredgewidth=0.3, markeredgecolor='white',
                label='Quantize+zstd', zorder=3)

        # Noise floor
        if not nf.empty:
            nf_val = float(nf['psnr_heldout'].values[0])
            if nf_val < 65:
                ax.axhline(y=nf_val, color='#D55E00', linewidth=0.6,
                           linestyle=':', label='Noise floor', zorder=1)

        ax.set_xscale('log')
        ax.set_xlabel('Bits per voxel')
        if i == 0:
            ax.set_ylabel('PSNR (dB)')
        ax.set_title(NAMES[ds], fontsize=8, fontweight='bold', pad=3)
        ax.set_ylim(18, 52)

        if i == 0:
            ax.legend(frameon=False, fontsize=5, loc='lower right',
                      handlelength=1.5, handletextpad=0.3)

        ax.text(-0.10, 1.06, chr(ord('a') + i), transform=ax.transAxes,
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = FIGS_DIR / "suppfig_compression.pdf"
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
