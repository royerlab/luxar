#!/usr/bin/env python3
"""Generate combined rate-distortion + cross-validation figure (Fig 3).

Merges the previous Fig 3 (rate-distortion) and Fig 4 (cross-validation)
into a single two-row figure for more efficient use of page space.

Layout:
  Row 1: a) PSNR vs splats  b) SSIM vs splats  c) Compression vs PSNR
  Row 2: d) CV kidney DAPI   e) CV tribolium     f) Noise-capacity scatter
"""

import matplotlib

matplotlib.use('Agg')
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# Paths — add manuscript/analysis/ to sys.path for shared utilities
SCRIPT_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis" / "splat_count_vs_quality"
sys.path.insert(0, str(SCRIPT_DIR.parent / "analysis"))
from cv_optimal import find_cv_optimal_idx  # noqa: E402

RESULTS_DIR = ANALYSIS_DIR / "results"
FIGS_DIR = SCRIPT_DIR / "figs" / "quantitative_analysis"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Style
FONTSIZE = 6.5
FONTSIZE_LABEL = 7
FONTSIZE_TITLE = 7.5
LINEWIDTH = 0.9
MARKERSIZE = 2.5
TWO_COL_WIDTH = 7.2

COLORS = {
    'kidney_dapi': '#0072B2',
    'kidney_actin': '#D55E00',
    'opencell_map4_ch0': '#009E73',
    'opencell_map4_ch1': '#CC79A7',
    'organoid_ch0': '#8B6914',
    'celegans_t100': '#56B4E9',
    'tribolium': '#E69F00',
}

DISPLAY_NAMES = {
    'kidney_dapi': 'Kidney DAPI',
    'kidney_actin': 'Kidney actin',
    'opencell_map4_ch0': 'MAP4 (Hoechst)',
    'opencell_map4_ch1': 'MAP4 (GFP)',
    'organoid_ch0': 'Organoid',
    'celegans_t100': 'C. elegans',
    'tribolium': 'Tribolium',
}

CORE_DATASETS = [
    'kidney_dapi', 'kidney_actin',
    'opencell_map4_ch0', 'opencell_map4_ch1',
    'organoid_ch0', 'celegans_t100', 'tribolium',
]


def setup():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': FONTSIZE,
        'axes.labelsize': FONTSIZE_LABEL,
        'axes.titlesize': FONTSIZE_TITLE,
        'xtick.labelsize': FONTSIZE,
        'ytick.labelsize': FONTSIZE,
        'legend.fontsize': FONTSIZE - 1,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'axes.linewidth': 0.4,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'lines.linewidth': LINEWIDTH,
        'lines.markersize': MARKERSIZE,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def load_metrics(ds):
    p = RESULTS_DIR / ds / "metrics.tsv"
    return pd.read_csv(p, sep='\t').sort_values('seeds_requested') if p.exists() else pd.DataFrame()


def load_cv(ds):
    p = RESULTS_DIR / f"{ds}_n2s" / "metrics_n2s.tsv"
    return pd.read_csv(p, sep='\t').sort_values('seeds_requested') if p.exists() else pd.DataFrame()


def load_noise_floor():
    p = RESULTS_DIR / "noise_floor.tsv"
    return pd.read_csv(p, sep='\t') if p.exists() else pd.DataFrame()


def add_label(ax, label, x=-0.14, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=FONTSIZE_TITLE + 2, fontweight='bold', va='top', ha='left')


def fmt_x(ax):
    ax.set_xscale('log')
    ax.set_xlim(0.8, 600)
    ax.set_xticks([1, 10, 100, 500])
    ax.set_xticklabels(['1K', '10K', '100K', '500K'])
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def main():
    setup()

    fig = plt.figure(figsize=(TWO_COL_WIDTH, 4.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.50,
                           left=0.06, right=0.99, top=0.95, bottom=0.08)

    nf = load_noise_floor()

    # ===== ROW 1: Rate-distortion =====

    # Panel a: PSNR vs splat count
    ax = fig.add_subplot(gs[0, 0])
    for ds in CORE_DATASETS:
        df = load_metrics(ds)
        if df.empty:
            continue
        ax.plot(df['n_splats_final']/1000, df['psnr_db'],
                '-o', color=COLORS[ds], label=DISPLAY_NAMES[ds],
                markersize=MARKERSIZE, markeredgewidth=0.3, markeredgecolor='white', zorder=3)
    fmt_x(ax)
    ax.set_xlabel('Splat count')
    ax.set_ylabel('PSNR (dB)')
    ax.set_ylim(18, 46)
    add_label(ax, 'a')

    # Panel b: SSIM vs splat count (legend goes here - more space)
    ax = fig.add_subplot(gs[0, 1])
    for ds in CORE_DATASETS:
        df = load_metrics(ds)
        if df.empty:
            continue
        ax.plot(df['n_splats_final']/1000, df['ssim'],
                '-o', color=COLORS[ds], label=DISPLAY_NAMES[ds],
                markersize=MARKERSIZE, markeredgewidth=0.3, markeredgecolor='white', zorder=3)
    fmt_x(ax)
    ax.set_xlabel('Splat count')
    ax.set_ylabel('SSIM')
    ax.set_ylim(0.2, 1.02)
    ax.legend(loc='lower right', frameon=False,
              fontsize=FONTSIZE - 1.5, ncol=2, handlelength=1.0, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.15, columnspacing=0.5)
    add_label(ax, 'b')

    # Panel c: Compression ratio vs PSNR
    ax = fig.add_subplot(gs[0, 2])
    for ds in CORE_DATASETS:
        df = load_metrics(ds)
        if df.empty:
            continue
        mask = df['compression_ratio'] > 1
        ax.plot(df.loc[mask, 'compression_ratio'], df.loc[mask, 'psnr_db'],
                '-o', color=COLORS[ds],
                markersize=MARKERSIZE, markeredgewidth=0.3, markeredgecolor='white', zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel('Compression ratio')
    ax.set_ylabel('PSNR (dB)')
    ax.set_ylim(18, 46)
    ax.invert_xaxis()
    add_label(ax, 'c')

    # ===== ROW 2: Cross-validation =====

    # Panel d: Kidney DAPI (overfitting)
    ax = fig.add_subplot(gs[1, 0])
    ds = 'kidney_dapi'
    n2s = load_cv(ds)
    c = COLORS[ds]
    if not n2s.empty:
        ax.plot(n2s['n_splats_final']/1000, n2s['train_psnr_db'],
                '-s', color=c, markersize=MARKERSIZE, markeredgewidth=0.3,
                markeredgecolor='white', label='Train', zorder=3)
        ax.plot(n2s['n_splats_final']/1000, n2s['held_out_psnr_db'],
                '--o', color=c, markersize=MARKERSIZE, markeredgewidth=0.3,
                markeredgecolor='white', label='Held-out', zorder=3)
        if not nf.empty:
            nf_val = nf[nf['dataset']==ds]['psnr_max_db'].values[0]
            ax.axhline(y=nf_val, color='#999999', linestyle=':', linewidth=0.6, label='Noise floor')
        # Star at the CV-optimal point (clear peak or plateau onset)
        opt_pos = find_cv_optimal_idx(n2s['held_out_psnr_db'].values)
        opt_iloc = n2s.index[opt_pos]
        px = n2s.loc[opt_iloc, 'n_splats_final']/1000
        py = n2s.loc[opt_iloc, 'held_out_psnr_db']
        ax.plot(px, py, '*', color='black', markersize=7, zorder=5)
        # Shade overfitting (past the absolute peak)
        abs_peak_idx = n2s['held_out_psnr_db'].idxmax()
        abs_px = n2s.loc[abs_peak_idx, 'n_splats_final']/1000
        mask = n2s['n_splats_final']/1000 > abs_px
        if mask.any():
            ax.fill_between(n2s.loc[mask, 'n_splats_final']/1000,
                            n2s.loc[mask, 'held_out_psnr_db'],
                            n2s.loc[mask, 'train_psnr_db'],
                            alpha=0.12, color='#D55E00')
            # Show gap at the last (512K) point for consistency with text
            last_idx = mask.values.nonzero()[0][-1]
            gap = n2s.iloc[last_idx]['train_psnr_db'] - n2s.iloc[last_idx]['held_out_psnr_db']
            ax.annotate(f'{gap:.1f} dB\ngap', fontsize=FONTSIZE-1,
                        xy=(n2s.iloc[last_idx]['n_splats_final']/1000,
                            (n2s.iloc[last_idx]['train_psnr_db']+n2s.iloc[last_idx]['held_out_psnr_db'])/2),
                        ha='center', va='center', color='#D55E00', fontweight='bold')
    fmt_x(ax)
    ax.set_xlabel('Splat count')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Kidney DAPI (confocal)', fontweight='bold', pad=2)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='#cccccc',
              framealpha=0.95, fontsize=FONTSIZE-1.5, handlelength=1.0,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.15)
    add_label(ax, 'd')

    # Panel e: Tribolium (no overfitting)
    ax = fig.add_subplot(gs[1, 1])
    ds = 'tribolium'
    n2s = load_cv(ds)
    c = COLORS[ds]
    if not n2s.empty:
        ax.plot(n2s['n_splats_final']/1000, n2s['train_psnr_db'],
                '-s', color=c, markersize=MARKERSIZE, markeredgewidth=0.3,
                markeredgecolor='white', label='Train', zorder=3)
        ax.plot(n2s['n_splats_final']/1000, n2s['held_out_psnr_db'],
                '--o', color=c, markersize=MARKERSIZE, markeredgewidth=0.3,
                markeredgecolor='white', label='Held-out', zorder=3)
        if not nf.empty:
            nf_val = nf[nf['dataset']==ds]['psnr_max_db'].values[0]
            if nf_val < 65:
                ax.axhline(y=nf_val, color='#999999', linestyle=':', linewidth=0.6, label='Noise floor')
    fmt_x(ax)
    ax.set_xlabel('Splat count')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Tribolium (light-sheet)', fontweight='bold', pad=2)
    ax.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='#cccccc',
              framealpha=0.95, fontsize=FONTSIZE-1.5, handlelength=1.0,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.15)
    add_label(ax, 'e')

    # Panel f: Optimal splat count vs noise
    ax = fig.add_subplot(gs[1, 2])
    if not nf.empty:
        # Collect data - EXCLUDE Tribolium (signal-limited, no CV peak;
        # multi-view deconvolution breaks pixel-independence assumption)
        plot_data = []
        for ds in CORE_DATASETS:
            if ds == 'tribolium':
                continue  # No meaningful held-out peak
            n2s = load_cv(ds)
            nfr = nf[nf['dataset']==ds]
            if n2s.empty or nfr.empty:
                continue
            sigma = nfr['sigma_ensemble'].values[0]
            if sigma == 0:
                continue
            # Check for genuine overfitting: held-out must decline from peak
            abs_peak_idx = n2s['held_out_psnr_db'].idxmax()
            last_idx = n2s.index[-1]
            if abs_peak_idx == last_idx:
                continue  # No decline = no meaningful peak
            opt_pos = find_cv_optimal_idx(n2s['held_out_psnr_db'].values)
            opt = n2s.iloc[opt_pos]['n_splats_final']/1000
            plot_data.append((ds, sigma*100, opt))
            ax.scatter(sigma*100, opt, color=COLORS[ds], s=30, zorder=5,
                       edgecolors='white', linewidths=0.4)

        # Direct labels with manually tuned offsets (no arrows - cleaner per Tufte)
        # With Tribolium excluded, the y-range is ~16K to ~124K on log scale
        offsets = {
            'kidney_actin': (6, 4),       # top-left area, label right
            'kidney_dapi': (-5, 5),       # mid-left, label left-above (room above)
            'opencell_map4_ch0': (6, 5),  # right area, label right
            'opencell_map4_ch1': (6, -7), # far right, label right-below
            'organoid_ch0': (6, -6),      # far left, label RIGHT (not left - avoid axis clip)
            'celegans_t100': (6, -6),     # middle, label right-below
        }
        for ds, sigma, opt in plot_data:
            off = offsets.get(ds, (5, 3))
            ax.annotate(
                DISPLAY_NAMES[ds],
                xy=(sigma, opt),
                xytext=off, textcoords='offset points',
                fontsize=FONTSIZE - 1.5, color=COLORS[ds], fontweight='bold',
                ha='left' if off[0] > 0 else 'right',
            )

    ax.set_xlabel('Noise level ($\\sigma \\times 100$)')
    ax.set_ylabel('Optimal splat count (K)')
    ax.set_title('Noise limits capacity', fontweight='bold', pad=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Clean tick labels: explicit positions and labels
    ax.set_yticks([16, 28, 63, 124])
    ax.set_yticklabels(['16', '28', '63', '124'])
    ax.set_ylim(12, 180)
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xticks([0.2, 0.5, 1, 3])
    ax.set_xticklabels(['0.2', '0.5', '1', '3'])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    add_label(ax, 'f')

    out = FIGS_DIR / "quantitative_analysis.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == '__main__':
    main()
