#!/usr/bin/env python3
"""Split-Only Example — spatial decomposition without any LOD.

This example demonstrates the **pure split** path: a points cloud
partitioned into N spatial parts (via the BSP heuristic from PR #319)
*without* any LOD wrapping. Companion to ``split_of_lod_example.py``,
which mixes split + LOD; this one isolates the split feature alone so
its behaviour is unambiguous.

Two layers are authored:

1. ``auto_split`` — a single ``add_points`` call on a 30k-point cloud,
   relying on the compiler-level ``auto_split_max_elements`` threshold
   to trigger automatic spatial partitioning. The user gets back a
   ``kind='split'`` group with several leaf children; the layers panel
   still shows one logical layer with an ``N parts`` badge.

2. ``manual_split`` — an explicit ``scene.add_split_group(...)`` with a
   matching ``max_elements`` parameter, hand-fed a colour-tinted strip
   so that each spatial part comes out with a distinct hue (visible
   confirmation that the BSP actually carved spatial sub-regions).

Educational value:
- See ``auto_split_max_elements`` on the compiler vs explicit
  ``add_split_group`` on a Group/Scene side-by-side.
- Confirm that split groups are about **spatial partitioning + frustum
  culling** in the viewer — not about quality (no LOD here).
- Foundation for ``split_of_lod_example.py``, which composes split
  with LOD.
"""

import numpy as np
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
    """Author two split-only layers (auto and manual)."""
    output_path = get_examples_output_dir() / "split_only_example.zarr"
    aprint(f"Writing split-only example to {output_path}")

    # auto_split_max_elements caps a single Points node at 8k elements.
    # 30k points / 8k cap → ~4 spatial parts.
    with LuxarZarrCompiler(
        output_path, auto_split_max_elements=8_000
    ) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # 1. Auto-split: bare add_points; the compiler wraps the result
        # in a kind=split group because the input exceeds the threshold.
        positions_auto = make_long_strip(n_points=30_000, seed=0)
        x_norm = (positions_auto[:, 0] - positions_auto[:, 0].min()) / (
            positions_auto[:, 0].max() - positions_auto[:, 0].min()
        )
        colors_auto = np.column_stack(
            [x_norm, 1.0 - x_norm, np.full_like(x_norm, 0.4)]
        ).astype(np.float32)
        scene.add_points(
            "auto_split",
            positions_auto,
            colors=colors_auto,
            radii=0.05,
            layer=True,
        )

        # 2. Manual split: hand-built add_split_group, then a single
        # add_points call inside it. Same shape, but the wrapper is
        # explicit in the scene graph instead of compiler-induced.
        manual_wrapper = scene.add_split_group(
            "manual_split",
            display_type="points",
            max_elements=8_000,
            layer=True,
        )
        positions_manual = make_long_strip(n_points=30_000, seed=1)
        positions_manual[:, 1] += 4.0  # offset Y so the two strips don't overlap
        colors_manual = np.full(
            (positions_manual.shape[0], 3), [0.4, 0.7, 1.0], dtype=np.float32
        )
        manual_wrapper.add_points(
            "ribbon",
            positions_manual,
            colors=colors_manual,
            radii=0.05,
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")
    aprint("Two layers in the panel; each shows an 'N parts' badge.")


if __name__ == "__main__":
    main()
