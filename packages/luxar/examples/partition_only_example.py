#!/usr/bin/env python3
"""Partition-Only Example — spatial decomposition without any LOD.

This example demonstrates the **pure partition** path: a points cloud
partitioned into N spatial parts (via the BSP heuristic from PR #319)
*without* any LOD wrapping. Companion to ``partition_of_lod_example.py``,
which mixes partition + LOD; this one isolates the partition feature alone so
its behaviour is unambiguous.

Two layers are authored:

1. ``auto_partition`` — a single ``add_points`` call on a 30k-point cloud,
   relying on the compiler-level ``auto_partition_max_elements`` threshold
   to trigger automatic spatial partitioning. The user gets back a
   ``kind='partition'`` group with several leaf children; the layers panel
   still shows one logical layer with an ``N parts`` badge.

2. ``manual_partition`` — an explicit ``scene.add_partition_group(...)`` whose
   ``part_<i>`` children are authored by hand, each a distinct flat hue in
   its own X-band. Because partition is *invisible* on its own (it only
   drives per-tile frustum culling, not appearance), the per-part colours
   are the visible confirmation that the layer is carved into spatial
   tiles — you literally see the coloured bands. Each child passes
   ``partition=False`` so it stays a leaf instead of being re-partitioned
   by the compiler's ``auto_partition_max_elements`` threshold.

Educational value:
- See ``auto_partition_max_elements`` on the compiler vs explicit
  ``add_partition_group`` on a Group/Scene side-by-side.
- Confirm that partition groups are about **spatial partitioning + frustum
  culling** in the viewer — not about quality (no LOD here).
- Foundation for ``partition_of_lod_example.py``, which composes partition
  with LOD.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def make_long_strip(n_points: int, seed: int = 0) -> np.ndarray:
    """A long strip along X — guarantees the BSP carves along that axis."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-20.0, 20.0, n_points).astype(np.float32)
    y = rng.uniform(-1.0, 1.0, n_points).astype(np.float32)
    z = rng.uniform(-1.0, 1.0, n_points).astype(np.float32)
    return np.column_stack([x, y, z])


def main() -> None:
    """Author two partition-only layers (auto and manual)."""
    output_path = get_examples_output_dir() / "partition_only_example.luxar.zarr"
    aprint(f"Writing partition-only example to {output_path}")

    # auto_partition_max_elements caps a single Points node at 8k elements.
    # 30k points / 8k cap → ~4 spatial parts.
    with LuxarZarrCompiler(output_path, auto_partition_max_elements=8_000) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # 1. Auto-partition: bare add_points; the compiler wraps the result
        # in a kind=partition group because the input exceeds the threshold.
        positions_auto = make_long_strip(n_points=30_000, seed=0)
        x_norm = (positions_auto[:, 0] - positions_auto[:, 0].min()) / (
            positions_auto[:, 0].max() - positions_auto[:, 0].min()
        )
        colors_auto = np.column_stack(
            [x_norm, 1.0 - x_norm, np.full_like(x_norm, 0.4)]
        ).astype(np.float32)
        scene.add_points(
            "auto_partition",
            positions_auto,
            colors=colors_auto,
            radii=0.05,
            layer=True,
        )

        # 2. Manual partition: hand-built add_partition_group whose part_<i>
        # children are authored explicitly — one distinctly-coloured flat
        # band per spatial tile. This is the *visible* demonstration: the
        # auto layer above is a single hue gradient (you can't see where the
        # BSP cut), whereas here each tile is a solid colour so the spatial
        # decomposition is unmistakable in the viewer.
        manual_wrapper = scene.add_partition_group(
            "manual_partition",
            display_type="points",
            max_elements=8_000,
            layer=True,
        )
        positions_manual = make_long_strip(n_points=30_000, seed=1)
        positions_manual[:, 1] += 4.0  # offset Y so the two strips don't overlap
        # A categorical palette — one solid colour per tile.
        palette = np.array(
            [
                [0.90, 0.20, 0.25],  # red
                [0.20, 0.70, 0.35],  # green
                [0.25, 0.45, 0.95],  # blue
                [0.95, 0.75, 0.20],  # amber
            ],
            dtype=np.float32,
        )
        n_tiles = palette.shape[0]
        x_manual = positions_manual[:, 0]
        edges = np.linspace(x_manual.min(), x_manual.max(), n_tiles + 1)
        for i in range(n_tiles):
            lo, hi = edges[i], edges[i + 1]
            mask = (x_manual >= lo) & (
                x_manual < hi if i < n_tiles - 1 else x_manual <= hi
            )
            tile_pos = positions_manual[mask]
            tile_colors = np.tile(palette[i], (tile_pos.shape[0], 1))
            manual_wrapper.add_points(
                f"part_{i}",
                tile_pos,
                colors=tile_colors,
                radii=0.05,
                # Stay a leaf — don't let auto_partition re-split this tile.
                partition=False,
            )

        add_explainer(
            scene,
            title="Spatial Partitioning (No LOD)",
            body=(
                "Both layers are carved into spatial parts for per-tile frustum "
                "culling only — partitioning never changes quality or count. "
                "<code>auto_partition</code> uses the compiler threshold; "
                "<code>manual_partition</code> uses an explicit "
                "<code>add_partition_group</code>."
            ),
            observe=[
                "Both layers show an 'N parts' badge in the layers panel.",
                "auto_partition is one smooth X-gradient (BSP cuts are invisible).",
                "manual_partition shows 4 solid-coloured bands, one per tile.",
                "Detail never refines when zooming — there is no LOD here.",
            ],
            observe_label="Look for",
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")
    aprint(
        "Two layers, each with an 'N parts' badge:\n"
        "  - auto_partition: one X-gradient hue (BSP cuts are invisible — "
        "partition only affects culling, not colour).\n"
        "  - manual_partition: 4 solid-coloured tiles — the visible proof "
        "the layer is spatially partitioned."
    )


if __name__ == "__main__":
    main()
