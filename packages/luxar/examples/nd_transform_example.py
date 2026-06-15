#!/usr/bin/env python3
"""nD Transform Example — per-dimension affine and permutation transforms.

This example demonstrates the ``nd_transform`` parameter on
``add_points`` / ``add_lines`` / ``add_gsplats`` / ``add_group``. Unlike
the spatial 4x4 ``transform`` (which only affects displayed dims), the
``nd_transform`` operates *per-dimension* on **non-displayed** dimensions
and supports two forms:

- **Affine**: ``{"dim_name": {"scale": s, "offset": o}}`` — maps a local
  coordinate ``x`` into world ``s * x + o``. Use for ordinal dimensions
  like time, depth, wavelength.
- **Permutation**: ``{"dim_name": {"permutation": [i0, i1, ...]}}`` —
  reorders categorical labels so local category ``k`` becomes world
  category ``permutation[k]``. Use for categorical dimensions like
  channel, species, condition.

Scene: two "instruments" observe the same 5D space-and-time data, but
their local Time and Channel axes are misaligned. ``nd_transform`` on
each instrument's group re-projects local indices into a shared world
coordinate system without rewriting positions.

Educational value:
- Understand the distinction between the 4x4 spatial ``transform`` and
  the per-dimension ``nd_transform``.
- See affine and permutation forms side-by-side.
- See how a single time slider then drives both instruments coherently.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def make_grid(t_values, c_values, rng: np.random.Generator) -> np.ndarray:
    """Build a 5D point cloud spanning the given time and channel indices.

    Returns positions as ``(N, 5)`` in dim order ``(X, Y, Z, Time, Channel)``.
    """
    positions = []
    for t in t_values:
        for c in c_values:
            xyz = rng.uniform(-3.0, 3.0, (20, 3))
            t_col = np.full((20, 1), t, dtype=np.float32)
            c_col = np.full((20, 1), c, dtype=np.float32)
            positions.append(np.hstack([xyz, t_col, c_col]))
    return np.vstack(positions).astype(np.float32)


def main() -> None:
    """Show two instruments aligned into a common world frame via nd_transform."""
    output_path = get_examples_output_dir() / "nd_transform_example.luxar.zarr"
    aprint(f"Writing nd_transform example to {output_path}")

    # World coordinate system: 10 timepoints, 3 categorical channels
    # ('red', 'green', 'blue').
    dims = Dimensions(
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
            Dimension(
                "time",
                unit="s",
                display=False,
                range=(0, 9),
                discrete=True,
                step=1.0,
            ),
            Dimension(
                "channel",
                unit="",
                display=False,
                # Passing categories=[...] auto-sets discrete=True and
                # auto-derives range from len(categories).
                categories=["red", "green", "blue"],
            ),
        ]
    )

    rng = np.random.default_rng(seed=0)

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=dims)

        # Instrument A — the reference. Local indices already match world
        # indices, so no nd_transform is needed.
        instrument_a = scene.add_group("instrument_a", layer=True)
        positions_a = make_grid(t_values=range(5), c_values=range(3), rng=rng)
        # Translate spatially so the two instruments don't overlap.
        positions_a[:, 0] -= 5.0
        instrument_a.add_points(
            "samples",
            positions_a,
            colors=[0.9, 0.4, 0.4],  # broadcast single RGB across all points
            radii=0.15,
        )

        # Instrument B — captured later (local t=0..4 corresponds to
        # world t=5..9) and labels channels in the reverse order
        # (local 0='blue', local 1='green', local 2='red'). We fix both
        # with a single nd_transform on the wrapping group: children
        # inherit the alignment.
        instrument_b = scene.add_group(
            "instrument_b",
            layer=True,
            nd_transform={
                # Affine on the ordinal Time dim: local t → world (t + 5).
                "time": {"offset": 5.0},
                # Permutation on the categorical Channel dim:
                # local 0 → world 2 (blue), local 1 → world 1 (green),
                # local 2 → world 0 (red).
                "channel": {"permutation": [2, 1, 0]},
            },
        )
        positions_b = make_grid(t_values=range(5), c_values=range(3), rng=rng)
        positions_b[:, 0] += 5.0
        instrument_b.add_points(
            "samples",
            positions_b,
            colors=[0.4, 0.6, 0.9],  # broadcast single RGB across all points
            radii=0.15,
        )

        add_explainer(
            scene,
            title="Per-Dimension nd_transform",
            body=(
                "Two instruments share one world frame: an affine "
                "<code>offset</code> aligns Instrument B's local time to "
                "world time, and a <code>permutation</code> re-maps its "
                "channel labels — no positions are rewritten."
            ),
            observe=[
                "Press <code>4</code> for time, then <code>[</code> / "
                "<code>]</code>: both clusters step in lockstep.",
                "Instrument A (left) shows world t=0–4; B (right) shows world t=5–9.",
                "Press <code>5</code> for channel: the permutation reorders "
                "Instrument B's colours.",
            ],
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")
    aprint("Press 4 then [/] to walk time; both instruments stay in lockstep.")
    aprint("Press 5 to switch channels; permutation reorders Instrument B.")


if __name__ == "__main__":
    main()
