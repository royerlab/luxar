#!/usr/bin/env python3
"""Lines Primitive QA Example - a visual test grid for the line primitives.

A small scene of lines in every behavior-critical configuration, laid out as
a labeled grid (row by row, top to bottom):

- row 1 (joins):  gentle wave | 90-degree zigzag | extreme 150-degree zigzag
- row 2 (width):  taper thin->thick | alternating widths | thin + thick pair
- row 3 (color):  red->blue gradient | sharpness 0 / 0.5 / 1 | rainbow polyline
- row 4 (3D):     tilt fan (in-plane -> along-z: goes END-ON when orbited)
                  | 12-spoke hub | circle
- row 5 (blend):  X crossing (additive stacking at the intersection)

Educational value:
- The living QA artifact for the ?linePrimitive= renderer toggle (#1352):
  open the same scene with ?linePrimitive=capsule, screen-space, and
  volumetric side by side and compare against the overlay's expectations.
- Joins must tile seamlessly (no gaps, wedges, or slivers), end-on segments
  must render as stable round discs, hairlines must hold the ~1.5 px floor.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir

WHITE = np.array([255, 255, 255, 255], dtype=np.uint8)


def v3(pts) -> np.ndarray:
    """Coerce a point list to a float32 (N, 3) array."""
    return np.asarray(pts, dtype=np.float32)


def solid(n: int, rgba: np.ndarray = WHITE) -> np.ndarray:
    """A solid per-vertex color array of n copies of rgba."""
    return np.tile(np.asarray(rgba, dtype=np.uint8), (n, 1))


def add(scene, name, pts, widths, y_off, x_off, colors=None, sharpness=None):
    """Add one polyline at a grid offset (shared style for every case)."""
    p = v3(pts)
    p[:, 0] += x_off
    p[:, 1] += y_off
    scene.add_lines(
        name,
        p,
        widths=widths,
        colors=colors if colors is not None else solid(len(p)),
        sharpness=sharpness if sharpness is not None else 0.5,
        blending_mode="additive",
        opacity=0.85,
    )
    aprint(name)


def main():
    output_path = get_examples_output_dir() / "lines_primitive_qa_example.luxar.zarr"

    dims = Dimensions(
        [
            Dimension(name="x", unit="px", range=(-2.0, 10.0), display=True),
            Dimension(name="y", unit="px", range=(-14.0, 2.0), display=True),
            Dimension(name="z", unit="px", range=(-2.0, 2.0), display=True),
        ]
    )

    with asection("Authoring line-primitive QA scene"):
        with LuxarZarrCompiler(str(output_path)) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # --- row 1: joins ---
            wave = [[i * 0.4, 0.35 * np.sin(i * 0.9), 0] for i in range(9)]
            add(scene, "join-gentle-wave", wave, 0.10, 0.0, 0.0)

            zig = [[i * 0.45, 0.45 * (i % 2), 0] for i in range(8)]
            add(scene, "join-zigzag-90deg", zig, 0.10, 0.0, 4.0)

            # Extreme bends: ~150-degree direction change per vertex — the
            # capsule primitive's half-disc joints must fill the fold tips.
            sharpzig = [[i * 0.18, 0.8 * (i % 2), 0] for i in range(8)]
            add(scene, "join-extreme-150deg", sharpzig, 0.10, 0.0, 8.0)

            # --- row 2: width ---
            add(
                scene,
                "width-taper",
                [[0, 0, 0], [3, 0, 0]],
                np.array([0.01, 0.35], dtype=np.float32),
                -2.5,
                0.0,
            )
            altw = [[i * 0.5, 0, 0] for i in range(7)]
            add(
                scene,
                "width-alternating",
                altw,
                np.array([0.03, 0.25, 0.03, 0.25, 0.03, 0.25, 0.03], dtype=np.float32),
                -2.5,
                4.0,
            )
            add(scene, "width-thin", [[0, 0.3, 0], [3, 0.3, 0]], 0.01, -2.5, 8.0)
            add(scene, "width-thick", [[0, -0.3, 0], [3, -0.3, 0]], 0.30, -2.5, 8.0)

            # --- row 3: color + sharpness ---
            add(
                scene,
                "color-gradient",
                [[0, 0, 0], [3, 0, 0]],
                0.15,
                -5.0,
                0.0,
                colors=np.array(
                    [[255, 40, 40, 255], [40, 90, 255, 255]], dtype=np.uint8
                ),
            )
            for i, s in enumerate([0.0, 0.5, 1.0]):
                add(
                    scene,
                    f"sharpness-{s:.1f}",
                    [[0, -i * 0.45, 0], [3, -i * 0.45, 0]],
                    0.16,
                    -5.0,
                    4.0,
                    sharpness=float(s),
                )
            rainbow_pts = [[i * 0.45, 0.3 * np.sin(i * 1.1), 0] for i in range(8)]
            hues = np.stack(
                [
                    np.linspace(255, 40, 8),
                    np.abs(np.linspace(-255, 255, 8)) * 0.7,
                    np.linspace(40, 255, 8),
                    np.full(8, 255),
                ],
                axis=1,
            ).astype(np.uint8)
            add(
                scene,
                "color-rainbow-polyline",
                rainbow_pts,
                0.12,
                -5.0,
                8.0,
                colors=hues,
            )

            # --- row 4: 3D / end-on ---
            for i in range(5):
                # 0 (in-plane) .. 90 degrees (along z = END-ON in front view)
                a = i * (np.pi / 8)
                seg = [[0, 0, 0], [1.6 * np.cos(a), 0, 1.6 * np.sin(a)]]
                add(scene, f"tilt-fan-{i * 22}deg", seg, 0.08, -8.0, 0.0 + i * 0.35)

            hub = []
            for k in range(12):
                a = k * np.pi / 6
                hub += [[0, 0, 0], [1.1 * np.cos(a), 1.1 * np.sin(a), 0]]
            scene.add_lines(
                "hub-12-spokes",
                v3(hub) + np.array([5.0, -8.5, 0], dtype=np.float32),
                widths=0.06,
                colors=solid(len(hub)),
                line_type="segments",
                blending_mode="additive",
                opacity=0.85,
            )
            aprint("hub-12-spokes")

            circ = [
                [np.cos(a) * 1.0, np.sin(a) * 1.0, 0]
                for a in np.linspace(0, 2 * np.pi, 33)
            ]
            add(scene, "circle-smooth", circ, 0.08, -8.5, 8.5)

            # --- row 5: crossing / blending ---
            add(scene, "cross-A", [[0, -1, 0], [3, 1, 0]], 0.18, -12.0, 0.0)
            add(scene, "cross-B", [[0, 1, 0], [3, -1, 0]], 0.18, -12.0, 0.0)

            add_explainer(
                scene,
                title="Line-primitive QA grid",
                body=(
                    "Five rows of line cases, top to bottom. Compare the same "
                    "scene under <code>?linePrimitive=capsule</code>, "
                    "<code>screen-space</code>, and <code>volumetric</code>."
                ),
                observe=[
                    "<strong>Row 1 — joins</strong>: gentle wave, 90&#176; zigzag, "
                    "extreme 150&#176; zigzag. Corners must tile seamlessly: no "
                    "gaps, no double-bright wedges, no slivers even at the "
                    "extreme bends (each capsule end carries its half of the "
                    "joint disc, so folds stay round at any angle).",
                    "<strong>Row 2 — width</strong>: smooth thin&#8594;thick taper; "
                    "alternating widths form clean diamonds; the hairline stays "
                    "crisp (~1.5 px floor), the thick bar shows a soft "
                    "gaussian-like cross-profile.",
                    "<strong>Row 3 — color + sharpness</strong>: red&#8594;blue "
                    "gradient without banding; the three bars are sharpness "
                    "0 / 0.5 / 1 &#8212; spiky core, gaussian, boxy; the rainbow "
                    "follows the vertices.",
                    "<strong>Row 4 — 3D (ORBIT THIS)</strong>: the fan tilts from "
                    "in-plane to along-z &#8212; end-on segments must render as "
                    "stable round discs, never flickering slivers, while you "
                    "rotate. Hub center sums bright but stays round; the circle "
                    "stays smooth.",
                    "<strong>Row 5 — crossing</strong>: the X doubles in "
                    "brightness at the overlap (additive) with no seam.",
                ],
                width=0.26,
            )
            aprint("explainer overlay")

    aprint(f"Scene saved to: {output_path}")
    aprint(f"\nTo view: luxar serve --viewer {output_path}")
    aprint("Then open the viewer with ?linePrimitive=capsule (or volumetric) to A/B.")


if __name__ == "__main__":
    main()
