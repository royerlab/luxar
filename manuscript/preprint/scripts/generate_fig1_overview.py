#!/usr/bin/env python3
"""Generate Figure 1: Luxar system overview.

Layout:
  Top row: a) Python code (white bg)  |  b) Pipeline with visual elements
  Bottom row: c) 4 viewer screenshots (white bg, tightly cropped)
"""

import matplotlib

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent.parent
FIGS_DIR = SCRIPT_DIR / "figs"
RESULTS_DIR = SCRIPT_DIR.parent / "analysis" / "splat_count_vs_quality" / "results"

FONTSIZE = 6.5
TWO_COL_WIDTH = 7.2


def add_label(ax, label, x=-0.06, y=1.06):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')


def draw_code_panel(ax):
    """Panel a: Python code on white background with subtle border."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Light gray background
    rect = plt.Rectangle((0.0, 0.0), 1.0, 1.0,
                          facecolor='#f8f8f8', edgecolor='#dddddd',
                          linewidth=0.5, zorder=0)
    ax.add_patch(rect)

    mono = 'DejaVu Sans Mono'
    fs = 5.5
    lh = 0.072

    lines = [
        ('from luxar import LuxarZarrCompiler', '#8b008b'),
        ('from luxar import Dimensions', '#8b008b'),
        ('', None),
        ('dims = Dimensions(["x","y","z","t"])', '#333333'),
        ('with LuxarZarrCompiler("out.zarr") as c:', '#333333'),
        ('  s = c.create_scene(dimensions=dims)', '#333333'),
        ('', None),
        ('  # Three geometry types', '#228b22'),
        ('  s.add_points("cells", pos, colors)', '#0055aa'),
        ('  s.add_gsplats("vol", ctrs, amps)', '#0055aa'),
        ('  s.add_lines("trk", verts, widths)', '#0055aa'),
        ('', None),
        ('# $ luxar serve out.zarr --viewer', '#228b22'),
    ]

    y = 0.94
    for text, color in lines:
        if not text:
            y -= lh * 0.4
            continue
        ax.text(0.03, y, text, fontsize=fs, fontfamily=mono, color=color,
                va='top', transform=ax.transAxes, zorder=2)
        y -= lh


def draw_pipeline_with_visuals(ax):
    """Panel b: Pipeline with small visual thumbnails at each step."""
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 2.5)
    ax.axis('off')

    # Load small thumbnails from cached data for visual elements
    # Step 1: Input volume (original slice)
    orig_img = None
    p = RESULTS_DIR / "kidney_dapi" / "slices" / "n032000_slices.npz"
    if p.exists():
        d = np.load(str(p))
        orig_img = d['target_slices'][1]

    # Step 3: Reconstruction (splat result)
    recon_img = None
    if p.exists():
        d = np.load(str(p))
        recon_img = d['recon_slices'][1]

    nodes = [
        (1.2, 'Volume', '#1565C0', orig_img),
        (3.5, 'Fit splats', '#2E7D32', recon_img),
        (5.8, 'Zarr archive', '#E65100', None),
        (8.1, 'Web viewer', '#C62828', None),
    ]

    thumb_size = 0.55  # in data coords
    for x, label, color, img in nodes:
        # Small thumbnail or icon
        if img is not None:
            # Show a small data thumbnail
            thumb_extent = [x - thumb_size, x + thumb_size,
                            1.05 - thumb_size, 1.05 + thumb_size]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1, aspect='auto',
                      extent=thumb_extent, zorder=2, interpolation='bilinear')
        else:
            # Simple colored dot as icon
            circle = plt.Circle((x, 1.05), 0.3, facecolor=color, alpha=0.12,
                                edgecolor=color, linewidth=0.5, zorder=2)
            ax.add_patch(circle)
            # Icon text
            if 'Zarr' in label:
                ax.text(x, 1.05, '.zarr', ha='center', va='center',
                        fontsize=5, fontweight='bold', color=color, zorder=3)
            elif 'Web' in label:
                ax.text(x, 1.05, 'WWW', ha='center', va='center',
                        fontsize=5, fontweight='bold', color=color, zorder=3,
                        fontfamily='sans-serif')

        # Label below
        ax.text(x, 0.25, label, ha='center', va='center',
                fontsize=6.5, fontweight='bold', color=color)

    # Arrows
    arrow_y = 1.05
    for i in range(len(nodes) - 1):
        x1 = nodes[i][0] + thumb_size + 0.15
        x2 = nodes[i+1][0] - thumb_size - 0.15
        ax.annotate('', xy=(x2, arrow_y), xytext=(x1, arrow_y),
                     arrowprops=dict(arrowstyle='->', color='#999999', lw=1.0))

    # Sub-labels
    subs = [
        (1.2, '3D/4D/nD'),
        (3.5, 'Per-splat Adam'),
        (5.8, 'Chunked, indexed'),
        (8.1, 'HDR, nD nav'),
    ]
    for x, s in subs:
        ax.text(x, -0.1, s, ha='center', va='center',
                fontsize=4.5, color='#999999', style='italic')


def draw_screenshots(axes_row):
    """Panel c: 4 viewer screenshots - tightly cropped, no borders."""
    screenshots = [
        ('viewer_gsplats_organoid_paper.png', 'GSplats: Organoid'),
        ('viewer_storm_microtubules_paper.png', 'Points: STORM'),
        ('viewer_spiral_galaxy_paper.png', 'Points: Galaxy'),
        ('viewer_lsystem_forest_paper.png', 'Lines: L-system'),
    ]

    for i, (fname, title) in enumerate(screenshots):
        ax = axes_row[i]
        img_path = FIGS_DIR / fname

        if img_path.exists():
            img = mpimg.imread(str(img_path))
            # Additional center-crop to remove whitespace margins
            h, w = img.shape[:2]
            margin_h = int(h * 0.05)
            margin_w = int(w * 0.05)
            img = img[margin_h:h-margin_h, margin_w:w-margin_w]
            ax.imshow(img, aspect='equal')

        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        ax.set_xlabel(title, fontsize=5.5, fontweight='bold',
                       color='#333333', labelpad=2)


def main():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial'],
        'font.size': FONTSIZE,
        'axes.linewidth': 0.3,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    fig = plt.figure(figsize=(TWO_COL_WIDTH, 3.5))

    # Two-row layout - tighter, screenshots get more space
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[0.9, 1.0],
                              hspace=0.08, left=0.01, right=0.99,
                              top=0.97, bottom=0.03)

    # Top row: code (left) + pipeline (right)
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                           width_ratios=[1.0, 1.3], wspace=0.06)

    ax_code = fig.add_subplot(top[0, 0])
    draw_code_panel(ax_code)
    add_label(ax_code, 'a', x=0.01, y=1.04)

    ax_pipe = fig.add_subplot(top[0, 1])
    draw_pipeline_with_visuals(ax_pipe)
    add_label(ax_pipe, 'b', x=-0.02, y=1.04)

    # Bottom row: 4 screenshots
    bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1],
                                              wspace=0.03)

    sc_axes = [fig.add_subplot(bottom[0, i]) for i in range(4)]
    draw_screenshots(sc_axes)
    add_label(sc_axes[0], 'c', x=-0.04, y=1.06)

    out = FIGS_DIR / "fig1_overview.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
