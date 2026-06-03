#!/usr/bin/env python3
"""Partition-of-LOD Example — kind=partition wrapping kind=lod, with UX polish.

This example demonstrates the four UX-polish items landed in PR #319:

1. **Auto-partition heuristic** — ``LuxarZarrCompiler(auto_partition_max_elements=N)``
   automatically wraps oversize ``add_points`` / ``add_gsplats`` inputs
   in a ``kind=partition`` Group. User-explicit ``partition=`` always wins.
2. **Combined badge** — the layers panel shows ``N parts × M LODs`` on
   the wrapper header when a Partition layer contains nested LOD groups.
3. **Broadcast LOD dropdown** — the "Active level" dropdown on the
   Partition wrapper updates every nested ``lod_group`` in lock-step. Ragged
   ladders clamp per-group automatically.
4. **Partition-aware picking** — hovering an inner ``part_<i>`` reports the
   outermost ``kind=partition`` wrapper's path (matches the layers-panel
   "outermost-as-layer" convention).

Layout: one wrapper layer ``ribbon`` containing a strip of points the
auto-partition partitions into several spatial parts; under each part the
example explicitly nests a hand-built ``add_lod_group`` with two LOD
levels (coarse + fine) so the broadcast + combined badge surface.

Educational value:
- See the ``[N parts × M LODs]`` combined badge in the layers panel.
- Switch the "Active level" dropdown and observe ALL parts switch
  together — proof the broadcast wires through.
- Hover any visible point: tooltip path = ``/ribbon`` (the wrapper),
  not ``/ribbon/part_3/lod_fine``.

PR β (#319) — Specialized-group UX polish.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def make_strip(n: int = 50_000, seed: int = 0):
    """A long horizontal strip — guarantees the BSP partitions along X."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-20, 20, n)  # long axis
    y = rng.uniform(-1.5, 1.5, n)
    z = rng.uniform(-1.5, 1.5, n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def main() -> None:
    output_path = get_examples_output_dir() / "partition_of_lod_example.zarr"

    # Auto-partition kicks in when an add_points call exceeds the threshold.
    # The default balanced median BSP halves recursively, so 50k points with
    # a 12k cap → 8 equal parts (50k → 25k → 12.5k ≤ 12k after 3 levels).
    with LuxarZarrCompiler(
        str(output_path), auto_partition_max_elements=12_000
    ) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # === 1. Auto-partition Points wrapper ===
        # The user calls add_points as normal — no partition= argument. The
        # compiler-level threshold kicks in, BSP partitions into N
        # spatial parts, and the result is a kind=partition Group named
        # ``ribbon`` with one Points child per part. The user sees one
        # logical layer in the panel.
        positions = make_strip(n=50_000)
        # Coloring stretches across X so each spatial part has a
        # distinct hue → easy to confirm each part is rendered.
        x_norm = (positions[:, 0] - positions[:, 0].min()) / (
            positions[:, 0].max() - positions[:, 0].min()
        )
        colors = np.stack(
            [x_norm, 1.0 - x_norm, np.full_like(x_norm, 0.5)], axis=1
        ).astype(np.float32)
        scene.add_points(
            "ribbon",
            positions,
            colors=colors,
            radii=0.05,
            layer=True,
        )

        # === 2. A hand-built Partition-of-LOD layer ===
        # Showcases the broadcast dropdown + combined badge. The layers
        # panel only surfaces a node as a selectable layer when it is
        # marked ``layer=True`` (see layer-state.ts::walkSceneGraph), and
        # the broadcast dropdown renders only for a selected ``kind=partition``
        # layer that wraps nested ``kind=lod`` groups. So this partition group
        # itself carries ``layer=True`` — no plain-group wrapper needed.
        #
        # auto-partition's children are bare leaves, so the ``ribbon`` layer
        # above can't surface the broadcast feature; this hand-built tree
        # mixes both: a kind=partition wrapping two kind=lod ladders.
        outer = scene.add_partition_group(
            "ribbon_lod",
            display_type="points",
            max_elements=12_000,
            layer=True,
        )
        # Build 2 parts by slicing the strip.
        for i, (lo, hi) in enumerate([(-20.0, 0.0), (0.0, 20.0)]):
            half = positions[(positions[:, 0] >= lo) & (positions[:, 0] < hi)]
            half_colors = colors[(positions[:, 0] >= lo) & (positions[:, 0] < hi)]
            part = outer.add_lod_group(f"part_{i}")
            # Coarse: every-Nth subsample. Fine: full resolution.
            #
            # ``partition=False`` opts each LOD level out of the compiler's
            # auto-partition heuristic. Without it the ~24k-point ``lod_fine``
            # leaf exceeds ``auto_partition_max_elements`` and gets wrapped in
            # a kind=partition group — which *drops* the ``min_pixel_size``
            # threshold, leaving both levels at 0 and breaking the
            # view-driven (auto) LOD selector. Keeping them as plain leaves
            # preserves the coarse(0)→fine(200px) threshold ladder.
            part.add_points(
                "lod_coarse",
                half[::8],
                colors=half_colors[::8],
                radii=0.10,
                min_pixel_size=0.0,
                partition=False,
            )
            part.add_points(
                "lod_fine",
                half,
                colors=half_colors,
                radii=0.05,
                min_pixel_size=200.0,
                partition=False,
            )

        add_explainer(
            scene,
            title="Partition Wrapping LOD",
            body=(
                "<code>ribbon</code> is auto-partitioned into spatial parts for "
                "frustum culling; <code>ribbon_lod</code> nests a "
                "<code>kind=lod</code> ladder under each part, so it carries a "
                "combined parts × LODs badge and a broadcast level dropdown."
            ),
            observe=[
                "Both layers show an 'N parts' badge in the layers panel.",
                "ribbon_lod shows a '[2 parts x 2 LODs]' badge plus an Active-level dropdown.",
                "Switching the dropdown coarsens or refines all parts in lock-step.",
                "Hovering a point reports the wrapper path, never the inner part.",
            ],
            observe_label="Look for",
        )

        aprint(
            "Created two layers:\n"
            "  - ribbon: 50k points auto-partition into N spatial parts "
            "(combined badge: 'N parts').\n"
            "  - ribbon_lod: hand-built kind=partition wrapping 2 nested "
            "kind=lod groups\n"
            "    → expect '[2 parts × 2 LODs]' badge + broadcast dropdown."
        )

    aprint(f"\nScene saved to: {output_path}")
    aprint(f"To view: luxar serve --viewer {output_path}")
    aprint(
        "  → Open the layers panel.\n"
        "  → 'ribbon' shows an 'N parts' badge.\n"
        "  → Select 'ribbon_lod': it shows a '[2 parts × 2 LODs]' badge\n"
        "     AND an Active-level dropdown in the controls panel. Switch\n"
        "     to 'lock to level 0': both parts coarsen. Switch to 'lock\n"
        "     to level 1': both refine.\n"
        "  → Hover any point: tooltip nodeName = '/ribbon' or\n"
        "     '/ribbon_lod', never the inner part."
    )


if __name__ == "__main__":
    main()
