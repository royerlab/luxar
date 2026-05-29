#!/usr/bin/env python3
"""Dimensions Builders Example — convenience constructors on ``Dimensions``.

This example demonstrates the four built-in convenience builders on
``Dimensions``, each of which returns a fully-configured collection
without typing out every ``Dimension(...)`` declaration:

- ``Dimensions.default_2d()`` — ``(x, y)`` in pixel units.
- ``Dimensions.default_3d()`` — ``(x, y, z)`` in generic units (most
  common; appears in nearly every other example).
- ``Dimensions.default_timeseries(n_timepoints, time_unit='s')`` —
  ``(t, x, y, z)`` with ``t`` non-displayed and discrete.
- ``Dimensions.default_multichannel(n_channels)`` — ``(c, x, y, z)``
  with ``c`` non-displayed and discrete.
- ``Dimensions.from_positions(positions)`` — infer dimensionality from
  the positions array shape.

The example writes four small scenes, one per builder, into a single
zarr archive (one per group) so you can compare their dimension
structure in one viewer.

Educational value:
- Stop typing the same boilerplate ``Dimension(...)`` calls.
- Discover the right tool for the common patterns: 2D, 3D, time
  series, multichannel.
- See ``from_positions`` for the case where you already have an N-D
  positions array and don't want to hand-write the dimension list.
"""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def show(name: str, dims: Dimensions) -> None:
    """Pretty-print a Dimensions object."""
    summary = ", ".join(
        f"{d.name}({'displayed' if d.display else 'navigable'})" for d in dims.dimensions
    )
    aprint(f"  {name:30s} → {len(dims)} dims: [{summary}]")


def main() -> None:
    """Write four small scenes, one per Dimensions builder."""
    output_path = get_examples_output_dir() / "dimensions_builders_example.zarr"
    aprint(f"Writing dimensions-builders example to {output_path}")

    # Show what each builder produces.
    rng = np.random.default_rng(seed=0)
    inferred_positions = rng.normal(0.0, 1.0, (100, 4)).astype(np.float32)

    show("default_2d()", Dimensions.default_2d())
    show("default_3d()", Dimensions.default_3d())
    show("default_timeseries(n=10)", Dimensions.default_timeseries(n_timepoints=10))
    show("default_multichannel(n=4)", Dimensions.default_multichannel(n_channels=4))
    show("from_positions(N,4)", Dimensions.from_positions(inferred_positions))

    # Pick one builder for the final scene — default_timeseries is the
    # most "interesting" because it includes a navigable dimension.
    dims = Dimensions.default_timeseries(n_timepoints=8, time_unit="frame")

    n_per_frame = 200
    all_positions = []
    all_colors = []
    for t in range(8):
        xyz = rng.normal(0.0, 1.0, (n_per_frame, 3)).astype(np.float32)
        # default_timeseries puts t FIRST: dims are (t, x, y, z).
        positions = np.hstack(
            [np.full((n_per_frame, 1), t, dtype=np.float32), xyz]
        )
        all_positions.append(positions)
        all_colors.append(
            np.full((n_per_frame, 3), [t / 7.0, 1.0 - t / 7.0, 0.5], dtype=np.float32)
        )

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_points(
            "frames",
            np.vstack(all_positions),
            colors=np.vstack(all_colors),
            radii=0.1,
            sharpness=2.0,
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")
    aprint("Press 1 then [/] to walk the time dimension.")


if __name__ == "__main__":
    main()
