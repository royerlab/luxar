#!/usr/bin/env python3
"""Self-Contained Demo: 3D Hilbert Space-Filling Curve.

Visualize the 3D Hilbert curve — a continuous, self-similar polyline that
visits every cell of a 2^n × 2^n × 2^n grid exactly once, with consecutive
cells always sharing a face. Stepping through orders 1..max_order via a
slider reveals the curve's recursive self-similarity: each higher order is
built from 8 rotated/reflected copies of the previous one, glued together
into a single Hamiltonian path.

================================================================================
WHAT IS A HILBERT CURVE?
================================================================================

A space-filling curve is a continuous map from a 1D interval onto a higher-
dimensional region. The Hilbert curve (David Hilbert, 1891; generalized to
3D by many authors since) has the remarkable locality property: nearby
points along the 1D curve stay nearby in 3D, and vice versa. That makes it
a workhorse for cache-efficient memory layouts (Morton/Hilbert ordering),
spatial database indexing, image dithering, and load-balancing in HPC.

ALGORITHM
---------
Skilling's "Programming the Hilbert curve" (AIP Conf. Proc. 707, 2004) —
vectorized over the index axis. For order p in n=3 dimensions, we map each
of 8^p indices through:
    1. Index → transpose form (MSB-first bit interleaving).
    2. Gray-decode: H ^= H/2 across the transposed bit columns.
    3. "Undo excess work": a chain of bit-XORs and exchanges parameterized
       by Q = 2, 4, 8, ..., 2^(p-1) that unrolls the recursive rotations
       baked into Hilbert's construction.

The result is a (8^p, 3) array of integer coordinates such that any two
consecutive rows differ by exactly 1 in exactly one axis. We verify both
properties at runtime as a sanity check.

VISUAL ENCODING
---------------
- Each order is a single ``polyline`` Lines node — one continuous strand,
  no breaks. Vertex positions are scaled to the unit cube and centered at
  the origin so all orders share a common viewing volume.
- Color: HSV hue swept from 0° to 360° along the traversal index, so the
  curve fades smoothly through red → yellow → green → cyan → blue →
  magenta → red as it fills space.
- Width: very thin (constant 0.003) to keep the recursive lattice
  structure visible at higher orders.
- A slider on the ``order`` dimension lets you step through 1..max_order
  one at a time so the recursion is visible: start at the 8-vertex base,
  watch each level subdivide and rotate.

NAVIGATION
----------
- Press '1' to focus the order slider, then '[' / ']' to step orders.
- Each order is its own Lines node — toggle individually from the Layers
  panel if you want to compare two side-by-side.

Usage:
    python -m luxar.demos.demo_hilbert_curve_3d
    python -m luxar.demos.demo_hilbert_curve_3d --max-order 5
    python -m luxar.demos.demo_hilbert_curve_3d --no-serve
"""

from __future__ import annotations

DEMO_META = {
    "key": "hilbert_curve_3d",
    "title": "Hilbert Curve 3D",
    "description": "The 3D Hilbert space-filling curve as a single polyline, steppable through recursion orders.",
    "category": "synthetic",
    "geometry": "lines",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["hilbert_curve_3d"],
    # Procedurally generated: no external dataset, nothing to credit.
    "citation": None,
}

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import DimensionsConfig, ViewerConfig
from luxar.demos import add_demo_caption, launch_viewer, parse_int_arg
from luxar.demos._lod_policy import stream_ladder
from luxar.utils.lod_breakpoints import (
    DEFAULT_MAX_ADDITIVE_COMMIT,
    parse_stream_chunk,
    sliced_ladder_first_chunk,
    stream_cuts,
)
from luxar.utils.paths import get_demos_output_dir

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_MAX_ORDER = 6  # Order 6 = 262,144 vertices (largest comfortable size)
LINE_WIDTH = 0.0015  # In normalized cube units; thin enough not to drown the curve


def hilbert_ladder(n_vertices: int) -> dict[str, Any]:
    """Give each hidden-order curve a useful first frame.

    Each leaf occupies one ``order`` coordinate, so a measured stop count is 1,
    but stepping the hidden dimension still replaces the whole curve and resets
    it to rung 0. Arm the sliced policy's 1/8 share explicitly: order 6 opens at
    65,536 of 524,286 segment vertices (12.5%), and its largest increment stays
    below the 900,000-vertex whole-node cap.
    """
    ladder = stream_ladder(n_vertices, geometry="lines")
    first_chunk = sliced_ladder_first_chunk(
        parse_stream_chunk(ladder["counts"]),
        elements=n_vertices,
        slices=2,
    )
    cuts = stream_cuts(n_vertices, first_chunk)
    largest_increment = max(
        (cut - previous for previous, cut in zip([0, *cuts[:-1]], cuts, strict=True)),
        default=0,
    )
    if largest_increment > DEFAULT_MAX_ADDITIVE_COMMIT:
        raise ValueError(
            f"Hilbert ladder resolves a {largest_increment:,}-vertex commit, above "
            f"the {DEFAULT_MAX_ADDITIVE_COMMIT:,}-vertex ceiling"
        )
    ladder["counts"] = f"stream:{first_chunk}"
    return ladder


# -----------------------------------------------------------------------------
# Hilbert curve generation
# -----------------------------------------------------------------------------


def hilbert_curve_3d(order: int) -> np.ndarray:
    """Generate the 3D Hilbert space-filling curve at the given order.

    Returns an ``(8^order, 3)`` array of integer coordinates in
    ``[0, 2^order)``, where consecutive rows differ by exactly 1 in
    exactly one axis (Hamiltonian path on the integer lattice).

    Implementation: Skilling's TransposetoAxes algorithm
    ("Programming the Hilbert curve", AIP Conf. Proc. 707, 2004),
    vectorized over the index axis with NumPy.
    """
    if order == 0:
        return np.array([[0, 0, 0]], dtype=np.int64)

    n = 3
    p = order
    n_pts = 8**order
    indices = np.arange(n_pts, dtype=np.int64)

    # Step 1: index → transpose form (MSB-first bit interleaving).
    # Bit (p-1-k) of X[i] = bit ((p-1-k)*n + (n-1-i)) of the index.
    X = np.zeros((n, n_pts), dtype=np.int64)
    for k in range(p):
        for i in range(n):
            bit = (indices >> ((p - 1 - k) * n + (n - 1 - i))) & 1
            X[i] |= bit << (p - 1 - k)

    # Step 2: Gray decode (H ^ (H/2)) on transposed columns.
    t = X[n - 1] >> 1
    for i in range(n - 1, 0, -1):
        X[i] ^= X[i - 1]
    X[0] ^= t

    # Step 3: "Undo excess work" — Skilling's bit-twiddling unrolling of
    # the recursive sub-cube rotations. For p=1 the loop is empty.
    N_max = 1 << p
    Q = 2
    while Q != N_max:
        P_val = Q - 1
        for i in range(n - 1, -1, -1):
            mask = (X[i] & Q).astype(bool)
            t_arr = (X[0] ^ X[i]) & P_val
            X[0] = np.where(mask, X[0] ^ P_val, X[0] ^ t_arr)
            X[i] = np.where(mask, X[i], X[i] ^ t_arr)
        Q <<= 1

    return X.T


def _verify_hilbert(coords: np.ndarray, order: int) -> None:
    """Cheap runtime sanity check. Raises if Skilling's output is malformed."""
    side = 1 << order
    assert coords.shape == (8**order, 3), f"shape {coords.shape}"
    assert coords.min() == 0 and coords.max() == side - 1
    if order >= 1:
        diffs = np.abs(np.diff(coords, axis=0)).sum(axis=1)
        assert (diffs == 1).all(), "non-adjacent jumps"
        flat = coords[:, 0] * side * side + coords[:, 1] * side + coords[:, 2]
        assert len(np.unique(flat)) == 8**order, "duplicate cells visited"


# -----------------------------------------------------------------------------
# Color
# -----------------------------------------------------------------------------


def _hsv_to_rgb_along_curve(n_pts: int) -> np.ndarray:
    """Sweep hue 0..1 across n_pts points, full saturation/value.

    Returns an (n_pts, 3) float32 array in [0, 1].
    """
    h = np.linspace(0.0, 1.0, n_pts, endpoint=False, dtype=np.float32)
    # Vectorized HSV → RGB (S = V = 1):
    # rgb = (1 - sat * max(0, min(c, 1)))  for c = abs(((h*6 + offset) mod 6) - 3) - 1
    # With S=V=1 this simplifies considerably.
    h6 = h * 6.0
    r = np.clip(np.abs(((h6 + 0.0) % 6.0) - 3.0) - 1.0, 0.0, 1.0)
    g = np.clip(np.abs(((h6 + 4.0) % 6.0) - 3.0) - 1.0, 0.0, 1.0)
    b = np.clip(np.abs(((h6 + 2.0) % 6.0) - 3.0) - 1.0, 0.0, 1.0)
    return np.stack([r, g, b], axis=1).astype(np.float32)


# -----------------------------------------------------------------------------
# Geometry → scene
# -----------------------------------------------------------------------------


def _curve_to_unit_cube(coords: np.ndarray, order: int) -> np.ndarray:
    """Scale lattice coords [0, 2^order) → centered unit cube [-0.5, 0.5)."""
    side = float(1 << order)
    if side == 1.0:
        return coords.astype(np.float32) - 0.5
    return ((coords.astype(np.float32) + 0.5) / side - 0.5).astype(np.float32)


#: The order the scene opens on (its slot is ``OPENING_ORDER - 1``); clamped to
#: the highest order actually built when ``--max-order`` is smaller.
OPENING_ORDER = 4


def opening_slot(orders: list[int]) -> int:
    """Category slot of :data:`OPENING_ORDER` among ``orders`` (clamped)."""
    return max(
        0,
        min(
            len(orders) - 1,
            orders.index(min(orders, key=lambda o: abs(o - OPENING_ORDER))),
        ),
    )


def build_scene(output_path: Path, max_order: int) -> int:
    """Write the multi-order Hilbert curve scene. Returns total vertex count."""
    orders = list(range(1, max_order + 1))
    total_pts = 0

    with asection("Building Hilbert curve scene"):
        dims = Dimensions(
            [
                Dimension(
                    "order",
                    unit="",
                    categories=[f"order {o}" for o in orders],
                    display=False,
                    description="Recursive depth (8^order vertices)",
                ),
                Dimension("x", unit="", display=True),
                Dimension("y", unit="", display=True),
                Dimension("z", unit="", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    # Thin lines lose detail at CSS resolution; see ViewerConfig.allow_high_dpr.
                    allow_high_dpr=True,
                    cinematic_mode=True,
                    # Open on order 4 (2026-09-10 review: "a bit more
                    # interesting from the get go" than the 8-vertex order 1
                    # the viewer would otherwise start on). `current_step` is
                    # indexed by ABSOLUTE dimension; `order` is dimension 0 and
                    # its categories are slots 0..N-1, so order k is slot k-1.
                    dimensions=DimensionsConfig(
                        current_step=[float(opening_slot(orders)), 0.0, 0.0, 0.0],
                        selected_dimension=0,
                    ),
                ),
            )

            for slot, order in enumerate(orders):
                coords = hilbert_curve_3d(order)
                _verify_hilbert(coords, order)
                xyz = _curve_to_unit_cube(coords, order)
                colors = _hsv_to_rgb_along_curve(len(xyz))

                # Attach this curve to its own slot on the order dim.
                vertices_4d = np.empty((len(xyz), 4), dtype=np.float32)
                vertices_4d[:, 0] = float(slot)
                vertices_4d[:, 1:] = xyz
                segment_vertices = np.empty(
                    (2 * (len(vertices_4d) - 1), 4), dtype=np.float32
                )
                segment_vertices[0::2] = vertices_4d[:-1]
                segment_vertices[1::2] = vertices_4d[1:]
                segment_colors = np.empty((len(segment_vertices), 3), dtype=np.float32)
                segment_colors[0::2] = colors[:-1]
                segment_colors[1::2] = colors[1:]

                aprint(
                    f"  order {order}: {len(xyz):,} vertices "
                    f"({len(xyz) - 1:,} segments)"
                )

                scene.add_lines(
                    f"Hilbert order {order}",
                    vertices=segment_vertices,
                    widths=LINE_WIDTH,
                    colors=segment_colors,
                    line_type="segments",
                    sharpness=0.5,
                    opacity=0.95,
                    intensity=0.55,
                    blending_mode="luminous",
                    layer=True,
                    additive_lod=hilbert_ladder(len(segment_vertices)),
                )
                total_pts += len(xyz)

            # Title (top-left)
            scene.add_text(
                "3D Hilbert Curve",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.7)",
                blend_mode="difference",
            )

            # Per-order caption
            for slot, order in enumerate(orders):
                n = 8**order
                side = 1 << order
                scene.add_text(
                    f"order {order}  ·  {n:,} vertices  ·  {side}×{side}×{side} grid",
                    position=(0.02, 0.97),
                    anchor="bottom-left",
                    font_size=0.018,
                    color="#ffcc44",
                    visible_range={"order": slot},
                    transition="fade",
                    transition_duration=0.2,
                )

            # Footer / nav hint
            add_demo_caption(
                scene, "← [  •  ] →   step through orders", DEMO_META.get("citation")
            )

        aprint(f"  ✓ {len(orders)} orders, {total_pts:,} total vertices")

    return total_pts


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    argv = sys.argv[1:]
    max_order = parse_int_arg("max-order", DEFAULT_MAX_ORDER, argv)
    if max_order < 1:
        raise SystemExit("--max-order must be ≥ 1")
    if max_order > 7:
        aprint(
            f"⚠ max-order {max_order} → {8**max_order:,} vertices at the top "
            f"slot. Browser GPU may struggle past order 7."
        )

    aprint("=" * 70)
    aprint("3D HILBERT SPACE-FILLING CURVE")
    aprint("=" * 70)
    aprint(f"Orders 1..{max_order}  (top order = {8**max_order:,} vertices)")
    aprint("")

    if "--no-serve" in argv:
        output_path = get_demos_output_dir() / "hilbert_curve_3d.luxar.zarr"
        n = build_scene(output_path, max_order=max_order)
        aprint(f"Dataset generated at {output_path} ({n:,} total vertices)")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_hilbert_") as tmpdir:
        output_path = Path(tmpdir) / "hilbert_curve_3d.luxar.zarr"
        n = build_scene(output_path, max_order=max_order)

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint(f"  Press '1' then '[' / ']' to step through orders 1..{max_order}")
        aprint("  Layers panel: toggle individual orders for side-by-side")
        aprint("")
        aprint(f"  Total vertices across all orders: {n:,}")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
