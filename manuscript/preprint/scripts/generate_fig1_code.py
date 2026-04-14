#!/usr/bin/env python3
"""Generate the code snippet panel for Figure 1.

Shows the minimal Python code to go from numpy arrays to web visualization.
"""

import matplotlib

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt

FIGS_DIR = Path(__file__).parent / "figs"

CODE = '''import numpy as np
from luxar import LuxarZarrCompiler, Dimensions, Dimension

dims = Dimensions([
    Dimension("x", unit="um", display=True),
    Dimension("y", unit="um", display=True),
    Dimension("z", unit="um", display=True),
])

with LuxarZarrCompiler("scene.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_points("nuclei", positions, colors=colors)
    scene.add_gsplats("volume", centers, amplitudes, cholesky)
    scene.add_lines("tracks", vertices, widths=widths)

# luxar serve scene.zarr --viewer --open'''


def main():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Dark background for code
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                          facecolor='#1E1E2E', edgecolor='#444466',
                          linewidth=0.8, zorder=1)
    ax.add_patch(rect)

    # Simple syntax coloring
    lines = CODE.strip().split('\n')
    y_start = 0.93
    line_height = 0.052
    fontsize = 5.5
    mono = 'DejaVu Sans Mono'

    keywords = {'import', 'from', 'as', 'with'}
    strings_start = ['"', "'"]

    for i, line in enumerate(lines):
        y = y_start - i * line_height
        if y < 0.03:
            break

        if line.startswith('#'):
            ax.text(0.05, y, line, fontsize=fontsize, fontfamily=mono,
                    color='#6A9955', va='top', transform=ax.transAxes, zorder=2)
        elif line.strip() == '':
            continue
        else:
            # Basic colorization
            colored_line = line
            # Just render the whole line with a base color
            # and highlight keywords
            parts = line.split(' ')
            x_pos = 0.05
            for j, part in enumerate(parts):
                if part in keywords:
                    color = '#C586C0'  # purple for keywords
                elif part.startswith('"') or part.startswith("'"):
                    color = '#CE9178'  # orange for strings
                elif part.endswith('(') or '.' in part:
                    color = '#DCDCAA'  # yellow for functions
                elif part in ['=', '+', '-', '*']:
                    color = '#D4D4D4'
                else:
                    color = '#9CDCFE'  # light blue for variables

                ax.text(x_pos, y, part + ' ', fontsize=fontsize, fontfamily=mono,
                        color=color, va='top', transform=ax.transAxes, zorder=2)
                # Approximate character width
                x_pos += len(part + ' ') * 0.0125

    # Title bar
    ax.text(0.5, 0.99, 'Python API', fontsize=6, fontfamily='sans-serif',
            color='#888899', va='top', ha='center', transform=ax.transAxes,
            fontweight='bold', zorder=2)

    out = FIGS_DIR / "fig1_code_snippet.pdf"
    fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
