#!/usr/bin/env python3
"""Plot compression comparison at CV-optimal splat counts."""

import matplotlib

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
FIGS_DIR = Path(__file__).parent.parent.parent / "preprint" / "figs" / "suppfig"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'kidney_dapi': '#0072B2', 'kidney_actin': '#D55E00',
    'opencell_map4_ch0': '#009E73', 'opencell_map4_ch1': '#CC79A7',
    'organoid_ch0': '#8B6914', 'celegans_t100': '#56B4E9',
    'tribolium': '#E69F00', 'cells3d_nuclei': '#666666',
    'cells3d_membrane': '#882255', 'opencell_lmnb1_ch0': '#117733',
    'opencell_lmnb1_ch1': '#AA4499', 'acto3d_heart_nuclei': '#44AA99',
}
NAMES = {
    'kidney_dapi': 'Kidney DAPI', 'kidney_actin': 'Kidney actin',
    'opencell_map4_ch0': 'OpenCell nuclei', 'opencell_map4_ch1': 'OpenCell MT',
    'organoid_ch0': 'Organoid', 'celegans_t100': 'C. elegans',
    'tribolium': 'Tribolium', 'cells3d_nuclei': 'Cells3D nuclei',
    'cells3d_membrane': 'Cells3D membrane', 'opencell_lmnb1_ch0': 'LMNB1 ch0',
    'opencell_lmnb1_ch1': 'LMNB1 ch1', 'acto3d_heart_nuclei': 'Heart nuclei',
}


def main():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 7,
        'axes.linewidth': 0.4, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42,
    })

    df = pd.read_csv(RESULTS_DIR / "compression_at_n2s_optimal.tsv", sep='\t')
    bpv = pd.read_csv(RESULTS_DIR / "bits_per_voxel_all.tsv", sep='\t')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # Panel a: Raw vs splat file sizes at CV-optimal count
    summary = df.sort_values('raw_MB', ascending=True)

    y = np.arange(len(summary))
    bar_h = 0.35

    ax1.barh(y + bar_h/2, summary['raw_MB'], bar_h, color='#cccccc',
             label='Raw (float32)', edgecolor='none')
    ax1.barh(y - bar_h/2, summary['splat_MB'], bar_h, color='#0072B2',
             label='GSplats (CV optimal)', edgecolor='none')

    # Add CV-optimal count and CR annotations
    for i, (_, row) in enumerate(summary.iterrows()):
        ax1.text(row['splat_MB'] * 1.3, i - bar_h/2,
                 f"{row['cr_vs_raw']:.0f}x",
                 fontsize=5, va='center', color='#0072B2', fontweight='bold')

    ax1.set_yticks(y)
    ax1.set_yticklabels([f"{NAMES.get(d, d)}" for d in summary['dataset']],
                         fontsize=6)
    ax1.set_xlabel('File size (MB)')
    ax1.set_xscale('log')
    ax1.legend(frameon=False, fontsize=6, loc='lower right')
    ax1.text(-0.02, 1.05, 'a', transform=ax1.transAxes, fontsize=10, fontweight='bold')

    # Panel b: Bits per voxel vs PSNR (rate-distortion curves)
    core = ['kidney_dapi', 'kidney_actin', 'opencell_map4_ch0', 'organoid_ch0',
            'celegans_t100', 'tribolium']
    for ds in core:
        sub = bpv[bpv['dataset'] == ds].sort_values('bpv_splat')
        ax2.plot(sub['bpv_splat'], sub['psnr_db'], '-o',
                 color=COLORS.get(ds, '#333'), label=NAMES.get(ds, ds),
                 markersize=3, linewidth=0.9, markeredgewidth=0.3, markeredgecolor='white')

        # Mark the CV-optimal point with a star
        opt_row = df[df['dataset'] == ds]
        if not opt_row.empty and opt_row['n2s_genuine_peak'].values[0]:
            opt_bpv = opt_row['bpv_splat'].values[0]
            opt_psnr = opt_row['psnr_db'].values[0]
            ax2.plot(opt_bpv, opt_psnr, '*', color=COLORS.get(ds, '#333'),
                     markersize=8, markeredgecolor='black', markeredgewidth=0.3, zorder=5)

    ax2.set_xlabel('Bits per voxel')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_xscale('log')
    ax2.legend(frameon=False, fontsize=5.5, ncol=2, loc='lower right')
    ax2.text(-0.02, 1.05, 'b', transform=ax2.transAxes, fontsize=10, fontweight='bold')

    plt.tight_layout()
    out = FIGS_DIR / "compression_cv_optimal.pdf"
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
