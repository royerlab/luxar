#!/usr/bin/env python3
"""Generate publication-quality figures for the Luxar manuscript.

Reads analysis results from manuscript/analysis/ and generates composite
figures suitable for a Nature Methods Correspondence / BioRxiv preprint.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = SCRIPT_DIR.parent / "analysis" / "splat_count_vs_quality"
RESULTS_DIR = ANALYSIS_DIR / "results"
FIGS_DIR = SCRIPT_DIR / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# Style constants
FONTSIZE = 7
FONTSIZE_LABEL = 7.5
FONTSIZE_TITLE = 8
LINEWIDTH = 1.0
MARKERSIZE = 3
DPI = 300

# Column width for Nature Methods (mm -> inches)
COL_WIDTH = 3.5  # inches (single column ~89mm)
TWO_COL_WIDTH = 7.2  # inches (two columns ~183mm)

# Colorblind-friendly palette (Okabe-Ito based, darkened for legibility)
COLORS = {
    'kidney_dapi': '#0072B2',       # Blue
    'kidney_actin': '#D55E00',      # Vermillion
    'opencell_map4_ch0': '#009E73', # Bluish green
    'opencell_map4_ch1': '#CC79A7', # Reddish purple
    'organoid_ch0': '#8B6914',      # Dark yellow/gold (NOT yellow)
    'celegans_t100': '#56B4E9',     # Sky blue
    'tribolium': '#E69F00',         # Orange
    'cells3d_nuclei': '#666666',    # Grey
    'cells3d_membrane': '#882255',  # Wine
    'opencell_lmnb1_ch0': '#117733',# Forest green
    'opencell_lmnb1_ch1': '#AA4499',# Rose
    'acto3d_heart_nuclei': '#44AA99',# Teal
}

DISPLAY_NAMES = {
    'kidney_dapi': 'Kidney DAPI',
    'kidney_actin': 'Kidney actin',
    'opencell_map4_ch0': 'OpenCell nuclei',
    'opencell_map4_ch1': 'OpenCell MT',
    'organoid_ch0': 'Organoid',
    'celegans_t100': 'C. elegans',
    'tribolium': 'Tribolium',
    'cells3d_nuclei': 'Cells3D nuclei',
    'cells3d_membrane': 'Cells3D membrane',
    'opencell_lmnb1_ch0': 'LMNB1 ch0',
    'opencell_lmnb1_ch1': 'LMNB1 ch1',
    'acto3d_heart_nuclei': 'Heart nuclei',
}

MODALITIES = {
    'kidney_dapi': 'Confocal',
    'kidney_actin': 'Confocal',
    'opencell_map4_ch0': 'Spinning-disk',
    'opencell_map4_ch1': 'Spinning-disk',
    'organoid_ch0': 'Confocal',
    'celegans_t100': 'Confocal',
    'tribolium': 'Light-sheet',
    'cells3d_nuclei': 'Confocal',
    'cells3d_membrane': 'Confocal',
    'opencell_lmnb1_ch0': 'Spinning-disk',
    'opencell_lmnb1_ch1': 'Spinning-disk',
    'acto3d_heart_nuclei': 'Confocal',
}

CORE_DATASETS = [
    'kidney_dapi', 'kidney_actin',
    'opencell_map4_ch0', 'opencell_map4_ch1',
    'organoid_ch0', 'celegans_t100', 'tribolium',
]


def setup_matplotlib():
    """Configure matplotlib for publication quality."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': FONTSIZE,
        'axes.labelsize': FONTSIZE_LABEL,
        'axes.titlesize': FONTSIZE_TITLE,
        'xtick.labelsize': FONTSIZE,
        'ytick.labelsize': FONTSIZE,
        'legend.fontsize': FONTSIZE - 1,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,
        'lines.linewidth': LINEWIDTH,
        'lines.markersize': MARKERSIZE,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def load_metrics(dataset_name: str) -> pd.DataFrame:
    path = RESULTS_DIR / dataset_name / "metrics.tsv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep='\t')
    return df.sort_values('seeds_requested')


def load_n2s_metrics(dataset_name: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{dataset_name}_n2s" / "metrics_n2s.tsv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep='\t')
    return df.sort_values('seeds_requested')


def load_noise_floor() -> pd.DataFrame:
    path = RESULTS_DIR / "noise_floor.tsv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep='\t')


def add_panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=FONTSIZE_TITLE + 2, fontweight='bold',
            va='top', ha='left')


def format_splat_axis(ax):
    """Format x-axis for splat count (log scale, K notation)."""
    ax.set_xscale('log')
    ax.set_xlim(0.8, 600)
    ax.set_xticks([1, 10, 100, 500])
    ax.set_xticklabels(['1K', '10K', '100K', '500K'])
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def figure_1_pipeline():
    """
    Figure 1: Pipeline overview - improved with better visual design.
    """
    setup_matplotlib()

    fig = plt.figure(figsize=(TWO_COL_WIDTH, 2.0))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.92])
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.2, 2.2)
    ax.axis('off')

    # Color scheme for pipeline stages
    stage_colors = [
        ('#E3F2FD', '#1565C0'),  # Light blue bg, dark blue border
        ('#E8F5E9', '#2E7D32'),  # Light green bg, dark green border
        ('#FFF3E0', '#E65100'),  # Light orange bg, dark orange border
        ('#F3E5F5', '#6A1B9A'),  # Light purple bg, dark purple border
        ('#FCE4EC', '#C62828'),  # Light red bg, dark red border
    ]

    steps = [
        (0.8, 'Microscopy\nVolume', '3D/4D/nD\nConfocal, light-sheet\nOME-Zarr, TIFF'),
        (2.8, 'Seed\nGeneration', 'Edge detection\nGPU-accelerated\nSobel + peak finding'),
        (4.8, 'Gaussian Splat\nFitting', 'Per-splat Adam\nAsymmetric loss\nCUDA / tiled'),
        (6.8, 'Zarr\nEncoding', 'Spatial indexing\nChunked + compressed\n.gsplats.zarr'),
        (8.8, 'Web\nViewer', 'WebGL + HDR\nnD navigation\nProgressive loading'),
    ]

    box_w, box_h = 1.6, 1.3
    for i, (x, label, sublabel) in enumerate(steps):
        bg_color, border_color = stage_colors[i]

        # Main box with rounded corners (approximated)
        rect = plt.Rectangle(
            (x - box_w / 2, 1 - box_h / 2),
            box_w, box_h,
            facecolor=bg_color, edgecolor=border_color,
            linewidth=1.0, zorder=2, joinstyle='round',
        )
        ax.add_patch(rect)

        # Main label
        ax.text(x, 1.15, label, ha='center', va='center',
                fontsize=FONTSIZE + 0.5, fontweight='bold',
                color=border_color, zorder=3)

        # Sub-label
        ax.text(x, 0.55, sublabel, ha='center', va='center',
                fontsize=FONTSIZE - 1.5, color='#555555',
                linespacing=1.15, zorder=3)

    # Arrows between boxes
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + box_w / 2 + 0.02
        x2 = steps[i + 1][0] - box_w / 2 - 0.02
        ax.annotate('', xy=(x2, 1), xytext=(x1, 1),
                     arrowprops=dict(arrowstyle='->', color='#333333',
                                     lw=1.5, connectionstyle='arc3,rad=0'))

    # Title strip
    ax.text(5.0, 2.1, 'LUXAR  PIPELINE', ha='center', va='center',
            fontsize=FONTSIZE + 2, fontweight='bold', color='#333333',
            style='italic')

    out_path = FIGS_DIR / "fig1_pipeline.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def figure_2_rate_distortion():
    """
    Figure 2: Quantitative analysis (3 panels).
    a) PSNR vs splat count
    b) SSIM vs splat count
    c) Compression ratio vs PSNR
    """
    setup_matplotlib()

    fig = plt.figure(figsize=(TWO_COL_WIDTH, 2.3))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35,
                           left=0.06, right=0.99, top=0.90, bottom=0.20)

    noise_floor_df = load_noise_floor()

    # --- Panel a: PSNR vs splat count ---
    ax_a = fig.add_subplot(gs[0, 0])
    for ds in CORE_DATASETS:
        df = load_metrics(ds)
        if df.empty:
            continue
        color = COLORS[ds]
        ax_a.plot(df['n_splats_final'] / 1000, df['psnr_db'],
                  '-o', color=color, label=DISPLAY_NAMES[ds],
                  markersize=MARKERSIZE, linewidth=LINEWIDTH,
                  markeredgewidth=0.4, markeredgecolor='white', zorder=3)
    format_splat_axis(ax_a)
    ax_a.set_xlabel('Splat count')
    ax_a.set_ylabel('PSNR (dB)')
    ax_a.set_ylim(18, 46)
    ax_a.legend(loc='lower right', frameon=True, fancybox=False,
                edgecolor='#cccccc', framealpha=0.95,
                fontsize=FONTSIZE - 1.5, ncol=1,
                handlelength=1.2, handletextpad=0.3,
                borderpad=0.25, labelspacing=0.2,
                columnspacing=0.5)
    add_panel_label(ax_a, 'a')

    # --- Panel b: SSIM vs splat count ---
    ax_b = fig.add_subplot(gs[0, 1])
    for ds in CORE_DATASETS:
        df = load_metrics(ds)
        if df.empty:
            continue
        color = COLORS[ds]
        ax_b.plot(df['n_splats_final'] / 1000, df['ssim'],
                  '-o', color=color, label=DISPLAY_NAMES[ds],
                  markersize=MARKERSIZE, linewidth=LINEWIDTH,
                  markeredgewidth=0.4, markeredgecolor='white', zorder=3)
    format_splat_axis(ax_b)
    ax_b.set_xlabel('Splat count')
    ax_b.set_ylabel('SSIM')
    ax_b.set_ylim(0.2, 1.02)
    add_panel_label(ax_b, 'b')

    # --- Panel c: Compression ratio vs PSNR ---
    ax_c = fig.add_subplot(gs[0, 2])
    for ds in CORE_DATASETS:
        df = load_metrics(ds)
        if df.empty or 'compression_ratio' not in df.columns:
            continue
        color = COLORS[ds]
        # Only show points with CR > 1
        mask = df['compression_ratio'] > 1
        ax_c.plot(df.loc[mask, 'compression_ratio'], df.loc[mask, 'psnr_db'],
                  '-o', color=color, label=DISPLAY_NAMES[ds],
                  markersize=MARKERSIZE, linewidth=LINEWIDTH,
                  markeredgewidth=0.4, markeredgecolor='white', zorder=3)
    ax_c.set_xscale('log')
    ax_c.set_xlabel('Compression ratio')
    ax_c.set_ylabel('PSNR (dB)')
    ax_c.set_ylim(18, 46)
    ax_c.invert_xaxis()  # Higher compression on left
    add_panel_label(ax_c, 'c')

    out_path = FIGS_DIR / "fig2_rate_distortion.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def figure_3_noise2self():
    """
    Figure 3: Noise2Self model selection (3 panels).
    a) Kidney DAPI (strong overfitting)
    b) Tribolium (no overfitting)
    c) Optimal splat count vs noise level
    """
    setup_matplotlib()

    fig = plt.figure(figsize=(TWO_COL_WIDTH, 2.3))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                           left=0.06, right=0.99, top=0.88, bottom=0.20)

    noise_floor_df = load_noise_floor()

    # --- Panel a: Kidney DAPI ---
    ax_a = fig.add_subplot(gs[0, 0])
    ds = 'kidney_dapi'
    n2s = load_n2s_metrics(ds)
    color = COLORS[ds]

    if not n2s.empty:
        ax_a.plot(n2s['n_splats_final'] / 1000, n2s['train_psnr_db'],
                  '-s', color=color, linewidth=LINEWIDTH,
                  markersize=MARKERSIZE, markeredgewidth=0.4,
                  markeredgecolor='white', label='Train', zorder=3)
        ax_a.plot(n2s['n_splats_final'] / 1000, n2s['held_out_psnr_db'],
                  '--o', color=color, linewidth=LINEWIDTH,
                  markersize=MARKERSIZE, markeredgewidth=0.4,
                  markeredgecolor='white', label='Held-out', zorder=3)

        # Noise floor
        if not noise_floor_df.empty:
            nf_row = noise_floor_df[noise_floor_df['dataset'] == ds]
            if not nf_row.empty:
                nf_val = nf_row['psnr_max_db'].values[0]
                ax_a.axhline(y=nf_val, color='#999999', linestyle=':',
                             linewidth=0.7, label='Noise floor', zorder=1)

        # Mark optimal
        peak_idx = n2s['held_out_psnr_db'].idxmax()
        peak_x = n2s.loc[peak_idx, 'n_splats_final'] / 1000
        peak_y = n2s.loc[peak_idx, 'held_out_psnr_db']
        ax_a.plot(peak_x, peak_y, '*', color='black',
                  markersize=8, zorder=5)

        # Shade overfitting region
        overfit_mask = n2s['n_splats_final'] / 1000 > peak_x
        if overfit_mask.any():
            ax_a.fill_between(
                n2s.loc[overfit_mask, 'n_splats_final'] / 1000,
                n2s.loc[overfit_mask, 'held_out_psnr_db'],
                n2s.loc[overfit_mask, 'train_psnr_db'],
                alpha=0.12, color='#D55E00', zorder=1
            )
            # Add "overfitting" annotation
            mid_idx = overfit_mask.values.nonzero()[0][len(overfit_mask.values.nonzero()[0]) // 2]
            mid_x = n2s.iloc[mid_idx]['n_splats_final'] / 1000
            mid_y_train = n2s.iloc[mid_idx]['train_psnr_db']
            mid_y_held = n2s.iloc[mid_idx]['held_out_psnr_db']
            gap = mid_y_train - mid_y_held
            ax_a.annotate(f'{gap:.1f} dB\ngap',
                          xy=(mid_x, (mid_y_train + mid_y_held) / 2),
                          fontsize=FONTSIZE - 1.5, ha='center', va='center',
                          color='#D55E00', fontweight='bold')

    format_splat_axis(ax_a)
    ax_a.set_xlabel('Splat count')
    ax_a.set_ylabel('PSNR (dB)')
    ax_a.set_title('Kidney DAPI (confocal)', fontsize=FONTSIZE_TITLE,
                    fontweight='bold', pad=3)
    ax_a.legend(loc='upper left', frameon=True, fancybox=False,
                edgecolor='#cccccc', framealpha=0.95,
                fontsize=FONTSIZE - 1.5, handlelength=1.2,
                handletextpad=0.3, borderpad=0.25, labelspacing=0.2)
    add_panel_label(ax_a, 'a')

    # --- Panel b: Tribolium ---
    ax_b = fig.add_subplot(gs[0, 1])
    ds = 'tribolium'
    n2s = load_n2s_metrics(ds)
    color = COLORS[ds]

    if not n2s.empty:
        ax_b.plot(n2s['n_splats_final'] / 1000, n2s['train_psnr_db'],
                  '-s', color=color, linewidth=LINEWIDTH,
                  markersize=MARKERSIZE, markeredgewidth=0.4,
                  markeredgecolor='white', label='Train', zorder=3)
        ax_b.plot(n2s['n_splats_final'] / 1000, n2s['held_out_psnr_db'],
                  '--o', color=color, linewidth=LINEWIDTH,
                  markersize=MARKERSIZE, markeredgewidth=0.4,
                  markeredgecolor='white', label='Held-out', zorder=3)

        if not noise_floor_df.empty:
            nf_row = noise_floor_df[noise_floor_df['dataset'] == ds]
            if not nf_row.empty:
                nf_val = nf_row['psnr_max_db'].values[0]
                if nf_val < 65:
                    ax_b.axhline(y=nf_val, color='#999999', linestyle=':',
                                 linewidth=0.7, label='Noise floor', zorder=1)

    format_splat_axis(ax_b)
    ax_b.set_xlabel('Splat count')
    ax_b.set_ylabel('PSNR (dB)')
    ax_b.set_title('Tribolium (light-sheet)', fontsize=FONTSIZE_TITLE,
                    fontweight='bold', pad=3)
    ax_b.legend(loc='lower right', frameon=True, fancybox=False,
                edgecolor='#cccccc', framealpha=0.95,
                fontsize=FONTSIZE - 1.5, handlelength=1.2,
                handletextpad=0.3, borderpad=0.25, labelspacing=0.2)
    add_panel_label(ax_b, 'b')

    # --- Panel c: Optimal splat count vs noise ---
    ax_c = fig.add_subplot(gs[0, 2])

    if not noise_floor_df.empty:
        plot_data = []
        for ds in CORE_DATASETS:
            n2s = load_n2s_metrics(ds)
            nf_row = noise_floor_df[noise_floor_df['dataset'] == ds]
            if n2s.empty or nf_row.empty:
                continue
            sigma = nf_row['sigma_ensemble'].values[0]
            if sigma == 0:
                continue
            peak_idx = n2s['held_out_psnr_db'].idxmax()
            optimal_count = n2s.loc[peak_idx, 'n_splats_final'] / 1000
            plot_data.append((ds, sigma, optimal_count))

        # Plot with offset labels
        for ds, sigma, opt_count in plot_data:
            color = COLORS.get(ds, '#333333')
            ax_c.scatter(sigma * 100, opt_count, color=color,
                         s=35, zorder=5, edgecolors='white', linewidths=0.5)

        # Add labels with smart positioning to avoid overlap
        label_offsets = {
            'kidney_dapi': (-40, -2),
            'kidney_actin': (6, 4),
            'opencell_map4_ch0': (6, -8),
            'opencell_map4_ch1': (6, -8),
            'organoid_ch0': (-35, 8),
            'celegans_t100': (-40, -8),
            'tribolium': (6, 6),
        }
        for ds, sigma, opt_count in plot_data:
            color = COLORS.get(ds, '#333333')
            offset = label_offsets.get(ds, (5, 3))
            ax_c.annotate(DISPLAY_NAMES.get(ds, ds),
                          xy=(sigma * 100, opt_count),
                          xytext=offset, textcoords='offset points',
                          fontsize=FONTSIZE - 2, color=color,
                          fontweight='bold')

    ax_c.set_xlabel('Noise level ($\\sigma \\times 100$)')
    ax_c.set_ylabel('Optimal splat count (K)')
    ax_c.set_title('Noise governs capacity', fontsize=FONTSIZE_TITLE,
                    fontweight='bold', pad=3)
    ax_c.set_xscale('log')
    ax_c.set_yscale('log')
    add_panel_label(ax_c, 'c')

    out_path = FIGS_DIR / "fig3_noise2self.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def figure_table_summary():
    """Generate LaTeX table for the paper."""
    noise_floor_df = load_noise_floor()

    rows = []
    for ds in CORE_DATASETS:
        df = load_metrics(ds)
        n2s = load_n2s_metrics(ds)
        if df.empty:
            continue

        name = DISPLAY_NAMES[ds]
        modality = MODALITIES[ds]

        nf_row = noise_floor_df[noise_floor_df['dataset'] == ds] if not noise_floor_df.empty else pd.DataFrame()
        shape = nf_row['volume_shape'].values[0] if not nf_row.empty else '?'
        nf_psnr = nf_row['psnr_max_db'].values[0] if not nf_row.empty else 0

        # Format shape with \times
        shape_fmt = shape.replace('x', '$\\times$') if isinstance(shape, str) else str(shape)

        # Best PSNR
        best_psnr = df['psnr_db'].max()

        # At 32K splats
        row_32k = df[df['seeds_requested'] == 32000]
        psnr_32k = row_32k['psnr_db'].values[0] if not row_32k.empty else 0
        ssim_32k = row_32k['ssim'].values[0] if not row_32k.empty else 0
        cr_32k = row_32k['compression_ratio'].values[0] if not row_32k.empty else 0

        # N2S optimal
        if not n2s.empty:
            peak_idx = n2s['held_out_psnr_db'].idxmax()
            optimal_count = n2s.loc[peak_idx, 'n_splats_final'] / 1000
        else:
            optimal_count = 0

        rows.append({
            'name': name,
            'modality': modality,
            'shape': shape_fmt,
            'psnr_32k': f'{psnr_32k:.1f}',
            'ssim_32k': f'{ssim_32k:.3f}',
            'cr_32k': f'{cr_32k:.0f}$\\times$',
            'best_psnr': f'{best_psnr:.1f}',
            'nf_psnr': f'{nf_psnr:.1f}' if 0 < nf_psnr < 100 else '--',
            'n2s_opt': f'{optimal_count:.0f}K' if optimal_count > 0 else '--',
        })

    latex_path = FIGS_DIR / "table_summary.tex"
    with open(latex_path, 'w') as f:
        f.write('\\begin{table*}[t]\n')
        f.write('\\centering\n')
        f.write('\\small\n')
        f.write('\\caption{\\textbf{Gaussian splat reconstruction quality across microscopy modalities.} ')
        f.write('PSNR, SSIM, and compression ratio (CR) at 32K splats; best PSNR at 512K splats; ')
        f.write('estimated noise floor; and Noise2Self-optimal splat count. ')
        f.write('Volumes normalized to $[0, 1]$.}\n')
        f.write('\\label{tab:summary}\n')
        f.write('\\begin{tabular}{@{}llccccccc@{}}\n')
        f.write('\\toprule\n')
        f.write('Dataset & Modality & Shape & ')
        f.write('\\makecell{PSNR\\\\@32K} & \\makecell{SSIM\\\\@32K} & ')
        f.write('\\makecell{CR\\\\@32K} & ')
        f.write('\\makecell{Best\\\\PSNR} & \\makecell{Noise\\\\floor} & ')
        f.write('\\makecell{N2S\\\\optimal} \\\\\n')
        f.write('\\midrule\n')
        for row in rows:
            f.write(f"  {row['name']} & {row['modality']} & "
                    f"\\footnotesize{{{row['shape']}}} & "
                    f"{row['psnr_32k']} & {row['ssim_32k']} & "
                    f"{row['cr_32k']} & {row['best_psnr']} & "
                    f"{row['nf_psnr']} & {row['n2s_opt']} \\\\\n")
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table*}\n')
    print(f"  Saved: {latex_path}")


def figure_convergence_summary():
    """Supplementary: Convergence behavior."""
    setup_matplotlib()

    conv_dir = SCRIPT_DIR.parent / "analysis" / "convergence" / "results"
    if not conv_dir.exists():
        print("  Skipping convergence (no results)")
        return

    showcase = [ds for ds in ['kidney_dapi', 'kidney_actin', 'opencell_map4_ch0', 'opencell_map4_ch1']
                if (conv_dir / ds / "convergence.tsv").exists()]
    if not showcase:
        print("  Skipping convergence (no data)")
        return

    splat_colors = {
        1000: '#56B4E9', 4000: '#009E73', 16000: '#E69F00',
        64000: '#D55E00', 256000: '#CC79A7',
    }

    n = len(showcase)
    fig, axes = plt.subplots(1, n, figsize=(TWO_COL_WIDTH, 2.0))
    if n == 1:
        axes = [axes]

    for i, ds in enumerate(showcase):
        ax = axes[i]
        df = pd.read_csv(conv_dir / ds / "convergence.tsv", sep='\t')
        for n_seeds, group in df.groupby('seeds_requested'):
            group = group.sort_values('n_iters')
            color = splat_colors.get(n_seeds, '#999999')
            ax.plot(group['n_iters'], group['psnr_db'],
                    '-o', color=color, linewidth=LINEWIDTH,
                    markersize=MARKERSIZE - 1,
                    markeredgewidth=0.3, markeredgecolor='white',
                    label=f'{n_seeds // 1000}K')
        ax.set_xlabel('Iterations')
        if i == 0:
            ax.set_ylabel('PSNR (dB)')
        ax.set_title(DISPLAY_NAMES.get(ds, ds), fontsize=FONTSIZE_TITLE,
                      fontweight='bold', pad=3)
        ax.set_xscale('log')
        if i == 0:
            ax.legend(title='Splats', frameon=True, fancybox=False,
                      edgecolor='#cccccc', framealpha=0.95,
                      fontsize=FONTSIZE - 2, title_fontsize=FONTSIZE - 1,
                      handlelength=1.0, handletextpad=0.2,
                      borderpad=0.25, labelspacing=0.15)
        add_panel_label(ax, chr(ord('a') + i))

    plt.tight_layout()
    out_path = FIGS_DIR / "suppfig_convergence.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def figure_4_n2s_all_datasets():
    """
    Supplementary Figure: N2S curves for ALL core datasets.
    Shows train vs held-out PSNR with noise floor for each dataset.
    """
    setup_matplotlib()

    noise_floor_df = load_noise_floor()
    n_ds = len(CORE_DATASETS)
    ncols = 4
    nrows = (n_ds + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(TWO_COL_WIDTH, nrows * 2.0))
    axes_flat = axes.flatten()

    for i, ds in enumerate(CORE_DATASETS):
        ax = axes_flat[i]
        n2s = load_n2s_metrics(ds)
        color = COLORS[ds]

        if not n2s.empty:
            ax.plot(n2s['n_splats_final'] / 1000, n2s['train_psnr_db'],
                    '-s', color=color, linewidth=LINEWIDTH,
                    markersize=MARKERSIZE - 1, markeredgewidth=0.3,
                    markeredgecolor='white', label='Train')
            ax.plot(n2s['n_splats_final'] / 1000, n2s['held_out_psnr_db'],
                    '--o', color=color, linewidth=LINEWIDTH,
                    markersize=MARKERSIZE - 1, markeredgewidth=0.3,
                    markeredgecolor='white', label='Held-out')

            # Noise floor
            if not noise_floor_df.empty:
                nf_row = noise_floor_df[noise_floor_df['dataset'] == ds]
                if not nf_row.empty:
                    nf_val = nf_row['psnr_max_db'].values[0]
                    if 0 < nf_val < 65:
                        ax.axhline(y=nf_val, color='#999999', linestyle=':',
                                   linewidth=0.6)

            # Mark peak only if it's a genuine peak (not at last point)
            peak_idx = n2s['held_out_psnr_db'].idxmax()
            last_idx = n2s.index[-1]
            if peak_idx != last_idx:
                peak_x = n2s.loc[peak_idx, 'n_splats_final'] / 1000
                peak_y = n2s.loc[peak_idx, 'held_out_psnr_db']
                ax.plot(peak_x, peak_y, '*', color='black', markersize=6, zorder=5)

        format_splat_axis(ax)
        ax.set_title(DISPLAY_NAMES[ds], fontsize=FONTSIZE, fontweight='bold', pad=2)
        if i % ncols == 0:
            ax.set_ylabel('PSNR (dB)')
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel('Splat count')
        if i == 0:
            ax.legend(fontsize=FONTSIZE - 2, frameon=True, fancybox=False,
                      edgecolor='#cccccc', handlelength=1.0, handletextpad=0.2)

    # Hide unused axes
    for j in range(len(CORE_DATASETS), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out_path = FIGS_DIR / "suppfig_n2s_all.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    print("Generating Luxar manuscript figures...")
    print()

    print("Figure 1: Pipeline")
    figure_1_pipeline()
    print()

    print("Figure 2: Rate-distortion")
    figure_2_rate_distortion()
    print()

    print("Figure 3: Noise2Self")
    figure_3_noise2self()
    print()

    print("Table 1: Summary")
    figure_table_summary()
    print()

    print("Supp: Convergence")
    figure_convergence_summary()
    print()

    print("Supp: N2S all datasets")
    figure_4_n2s_all_datasets()
    print()

    print("Done!")
