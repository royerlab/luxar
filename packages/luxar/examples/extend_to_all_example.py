#!/usr/bin/env python3
"""Extend-to-All Example — control visibility across non-displayed dimensions.

This example demonstrates the ``extend_to_all`` parameter on
``add_points``/``add_lines``/``add_gsplats``, which controls whether
data defined at a specific slice of a non-displayed dimension should
remain visible as the user slices through that dimension:

- ``extend_to_all=[]`` — (the typical authoring choice) data is only
  visible at exactly the slice it was defined at. Time-varying point
  clouds use this.
- ``extend_to_all=['time']`` — data is visible across all values of
  the named dimension, even though it was only defined at one slice.
  Useful for axis markers, scale bars, reference grids, fixed labels.
- ``extend_to_all=None`` (default) — Luxar warns if it detects that
  data lives in only a small portion of a non-displayed dimension's
  range, suggesting the explicit choice.

Scene: a "scan-line" of moving dots that walks through 6 timepoints
(``extend_to_all=[]``), and a stationary set of X/Y/Z axis markers
that are defined once but visible at every timepoint
(``extend_to_all=['time']``).

Educational value:
- Discover the three values of ``extend_to_all``.
- Understand the trade-off between data size (broadcast across all
  slices) and explicit per-slice authoring.
- Pattern for adding persistent reference geometry to a nD scene.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main() -> None:
    """Walk a moving scan-line through time alongside fixed axis markers."""
    output_path = get_examples_output_dir() / "extend_to_all_example.zarr"
    aprint(f"Writing extend_to_all example to {output_path}")

    n_timepoints = 6
    dims = Dimensions(
        [
            Dimension("x", display=True, range=(-6.0, 6.0)),
            Dimension("y", display=True, range=(-6.0, 6.0)),
            Dimension("z", display=True, range=(-6.0, 6.0)),
            Dimension(
                "time",
                display=False,
                range=(0, n_timepoints - 1),
                step=1.0,
                discrete=True,
            ),
        ]
    )

    # 1. Per-timepoint moving scan-line. Each timepoint gets its own
    #    set of 50 points; with extend_to_all=[] they are visible only
    #    at that exact time slice.
    n_per_step = 50
    rng = np.random.default_rng(seed=0)
    scan_positions = []
    scan_colors = []
    for t in range(n_timepoints):
        # Line at y=z=0, walking from x=-5 to x=+5 across timepoints.
        x = (t / (n_timepoints - 1)) * 10.0 - 5.0
        positions = np.zeros((n_per_step, 4), dtype=np.float32)
        positions[:, 0] = x + rng.normal(0.0, 0.1, n_per_step)
        positions[:, 1] = rng.normal(0.0, 0.3, n_per_step)
        positions[:, 2] = rng.normal(0.0, 0.3, n_per_step)
        positions[:, 3] = float(t)
        scan_positions.append(positions)
        # Hue progresses with t.
        scan_colors.append(
            np.tile(
                [t / (n_timepoints - 1), 1.0 - t / (n_timepoints - 1), 0.4],
                (n_per_step, 1),
            ).astype(np.float32)
        )
    scan_positions = np.vstack(scan_positions)
    scan_colors = np.vstack(scan_colors)

    # 2. Fixed axis markers — 5 points along each of X, Y, Z. These are
    #    defined at time=0 only, but with extend_to_all=['time'] they
    #    stay visible as the user walks through every timepoint.
    axis_points = []
    axis_colors = []
    for axis, color in enumerate([[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]]):
        for v in np.linspace(-5.0, 5.0, 5):
            p = [0.0, 0.0, 0.0, 0.0]  # x, y, z, time
            p[axis] = float(v)
            axis_points.append(p)
            axis_colors.append(color)
    axis_positions = np.array(axis_points, dtype=np.float32)
    axis_colors_arr = np.array(axis_colors, dtype=np.float32)

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=dims)

        # extend_to_all=[] : per-time scan-line, only visible at its slice.
        scene.add_points(
            "scan_line",
            scan_positions,
            colors=scan_colors,
            radii=0.18,
            extend_to_all=[],
        )

        # extend_to_all=['time'] : axis markers, visible at every timepoint.
        scene.add_points(
            "axis_markers",
            axis_positions,
            colors=axis_colors_arr,
            radii=0.3,
            sharpness=4.0,
            extend_to_all=["time"],
        )

        add_explainer(
            scene,
            title="Extend-to-All Visibility",
            body=(
                "The scan-line uses <code>extend_to_all=[]</code> so it shows "
                "only at its own time slice; the axis markers use "
                "<code>extend_to_all=['time']</code> so they persist across all "
                "frames. Press <code>4</code> then <code>[</code>/<code>]</code> "
                "to walk time."
            ),
            observe=[
                "The colored scan-line dots move along X as time advances.",
                "The red/green/blue axis markers stay fixed at every timepoint.",
                "Only one scan-line cluster is visible per slice.",
            ],
            observe_label="Look for",
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")
    aprint("Press 4 then [/] to walk time. The axis markers stay; the scan-line moves.")


if __name__ == "__main__":
    main()
