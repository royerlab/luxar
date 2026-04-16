"""Shared utilities for Luxar manuscript analysis and figure generation.

Consolidates duplicated constants, metric functions, and matplotlib helpers
that were previously copy-pasted across 15+ analysis and figure scripts.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

# Colorblind-friendly palette (Okabe-Ito based, darkened for legibility)
COLORS = {
    'kidney_dapi': '#0072B2',
    'kidney_actin': '#D55E00',
    'opencell_map4_ch0': '#009E73',
    'opencell_map4_ch1': '#CC79A7',
    'organoid_ch0': '#8B6914',
    'celegans_t100': '#56B4E9',
    'tribolium': '#E69F00',
    'cells3d_nuclei': '#666666',
    'cells3d_membrane': '#882255',
    'opencell_lmnb1_ch0': '#117733',
    'opencell_lmnb1_ch1': '#AA4499',
    'acto3d_heart_nuclei': '#44AA99',
}

DISPLAY_NAMES = {
    'kidney_dapi': 'Kidney DAPI',
    'kidney_actin': 'Kidney actin',
    'opencell_map4_ch0': 'MAP4 (Hoechst)',
    'opencell_map4_ch1': 'MAP4 (GFP)',
    'organoid_ch0': 'Organoid',
    'celegans_t100': 'C. elegans',
    'tribolium': 'Tribolium',
    'cells3d_nuclei': 'Cells3D nuclei',
    'cells3d_membrane': 'Cells3D membrane',
    'opencell_lmnb1_ch0': 'LMNB1 (Hoechst)',
    'opencell_lmnb1_ch1': 'LMNB1 (GFP)',
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

# 7 core datasets shown in main-paper figures and Table 1
CORE_DATASETS = [
    'kidney_dapi', 'kidney_actin',
    'opencell_map4_ch0', 'opencell_map4_ch1',
    'organoid_ch0', 'celegans_t100', 'tribolium',
]

# All 12 datasets
ALL_DATASETS = list(COLORS.keys())

# ---------------------------------------------------------------------------
# Figure style constants (Nature Methods format)
# ---------------------------------------------------------------------------

FONTSIZE = 7
FONTSIZE_LABEL = 7.5
FONTSIZE_TITLE = 8
LINEWIDTH = 1.0
MARKERSIZE = 3
DPI = 300

COL_WIDTH = 3.5     # inches (single column ~89mm)
TWO_COL_WIDTH = 7.2  # inches (two columns ~183mm)

# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def compute_psnr(original, reconstructed, mask=None):
    """Compute PSNR (dB) between two volumes, optionally on masked voxels only.

    Volumes are assumed to be normalized to [0, 1].
    """
    if mask is not None:
        original = original[mask]
        reconstructed = reconstructed[mask]
    mse = np.mean(
        (original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2
    )
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def compute_splat_bytes(n_splats, ndim=3):
    """Compute uncompressed byte size of a Gaussian splat dataset.

    Each splat stores: center (ndim) + amplitude (1) + Cholesky factors
    (ndim*(ndim+1)/2), all as float32 (4 bytes each).
    """
    cholesky_params = ndim * (ndim + 1) // 2
    params_per_splat = ndim + 1 + cholesky_params
    return n_splats * params_per_splat * 4


def generate_holdout_mask(shape, fraction=0.05, seed=42):
    """Generate a random boolean mask for held-out cross-validation.

    Returns a mask where ``True`` marks held-out voxels (fraction of total).
    """
    rng = np.random.RandomState(seed)
    return rng.random(shape) < fraction


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------


def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker  # noqa: F401

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


def add_panel_label(ax, label, x=-0.12, y=1.05):
    """Add a bold panel label (a, b, c, ...) to a matplotlib axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=FONTSIZE_TITLE + 2, fontweight='bold',
            va='top', ha='left')


def format_splat_axis(ax):
    """Format x-axis for splat count (log scale, K notation)."""
    import matplotlib.ticker as ticker

    ax.set_xscale('log')
    ax.set_xlim(0.8, 600)
    ax.set_xticks([1, 10, 100, 500])
    ax.set_xticklabels(['1K', '10K', '100K', '500K'])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
