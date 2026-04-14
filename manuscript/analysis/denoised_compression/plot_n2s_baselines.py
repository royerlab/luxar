#!/usr/bin/env python3
"""Plot cross-validation-based compression comparison: GSplats vs H.265 (6 datasets)."""

import matplotlib

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
FIGS_DIR = Path(__file__).parent.parent.parent / "preprint" / "figs" / "suppfig"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

NAMES = {
    'kidney_dapi': 'Kidney DAPI',
    'kidney_actin': 'Kidney actin',
    'organoid_ch0': 'Organoid',
    'celegans_t100': 'C. elegans',
    'tribolium': 'Tribolium',
    'opencell_map4_ch0': 'OpenCell nuclei',
    'opencell_map4_ch1': 'OpenCell MT',
    'cells3d_nuclei': 'Cells3D nuclei',
}

# 6 datasets: 3 noisy confocal (top row) + 3 varied (bottom row)
PLOT_DATASETS = [
    ['kidney_dapi', 'kidney_actin', 'opencell_map4_ch1'],
    ['organoid_ch0', 'cells3d_nuclei', 'tribolium'],
]


def main():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 6.5,
        'axes.linewidth': 0.4, 'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

    df = pd.read_csv(RESULTS_DIR / "n2s_baselines.tsv", sep='\t')

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 4.2))

    panel_idx = 0
    for row_i, row_datasets in enumerate(PLOT_DATASETS):
        for col_i, ds in enumerate(row_datasets):
            ax = axes[row_i, col_i]
            sub = df[(df['dataset'] == ds) & (df['method'] != 'noise_floor')]
            nf = df[(df['dataset'] == ds) & (df['method'] == 'noise_floor')]

            # GSplats: held-out PSNR (solid blue, main curve)
            gs = sub[sub['method'] == 'gsplats'].sort_values('bpv')
            if not gs.empty:
                # Raw PSNR (faded)
                ax.plot(gs['bpv'], gs['psnr_raw'], '--',
                        color='#0072B2', linewidth=0.6, alpha=0.3,
                        label='GSplats (raw)', zorder=2)
                # Held-out PSNR (solid)
                ax.plot(gs['bpv'], gs['psnr_heldout'], '-o',
                        color='#0072B2', markersize=2.5, linewidth=1.0,
                        markeredgewidth=0.2, markeredgecolor='white',
                        label='GSplats (N2S)', zorder=5)
                # Mark N2S-optimal
                peak_idx = gs['psnr_heldout'].idxmax()
                last_idx = gs.index[-1]
                if peak_idx != last_idx:
                    ax.plot(gs.loc[peak_idx, 'bpv'], gs.loc[peak_idx, 'psnr_heldout'],
                            '*', color='#0072B2', markersize=8,
                            markeredgecolor='black', markeredgewidth=0.3, zorder=7)

            # H.265: held-out + raw
            h265 = sub[sub['method'] == 'h265'].sort_values('bpv')
            if not h265.empty:
                # Raw PSNR (faded)
                ax.plot(h265['bpv'], h265['psnr_raw'], '--',
                        color='#D55E00', linewidth=0.6, alpha=0.3,
                        label='H.265 (raw)', zorder=2)
                # Held-out PSNR (solid)
                ax.plot(h265['bpv'], h265['psnr_heldout'], '-s',
                        color='#D55E00', markersize=2.5, linewidth=1.0,
                        markeredgewidth=0.2, markeredgecolor='white',
                        label='H.265 (N2S)', zorder=4)
                # Mark H.265 optimal
                h265_peak = h265['psnr_heldout'].idxmax()
                h265_last = h265.index[-1]
                if h265_peak != h265_last:
                    ax.plot(h265.loc[h265_peak, 'bpv'],
                            h265.loc[h265_peak, 'psnr_heldout'],
                            '*', color='#D55E00', markersize=8,
                            markeredgecolor='black', markeredgewidth=0.3, zorder=7)

            # Noise floor
            if not nf.empty:
                nf_val = float(nf['psnr_heldout'].values[0])
                if nf_val < 65:
                    ax.axhline(y=nf_val, color='#999999', linewidth=0.5,
                               linestyle=':', zorder=1)

            ax.set_xscale('log')
            if row_i == 1:
                ax.set_xlabel('Bits per voxel')
            if col_i == 0:
                ax.set_ylabel('PSNR (dB)')
            ax.set_title(NAMES[ds], fontsize=7, fontweight='bold', pad=2)

            # Legend only in first panel
            if panel_idx == 0:
                ax.legend(frameon=False, fontsize=4.5, loc='lower right',
                          handlelength=1.5, handletextpad=0.3)

            ax.text(-0.12, 1.06, chr(ord('a') + panel_idx), transform=ax.transAxes,
                    fontsize=9, fontweight='bold')
            panel_idx += 1

    plt.tight_layout(h_pad=1.0)
    out = FIGS_DIR / "compression.pdf"
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
