#!/usr/bin/env python3
"""Categorical Dimensions Example — discrete labels with category metadata.

This example demonstrates:

- ``Dimension(name='species', categories=['cat', 'dog', 'bird'], display=False)``
  — declares a non-displayed dimension whose values are labels, not
  continuous coordinates. Setting ``categories=[...]`` auto-implies
  ``discrete=True`` and derives ``range=(0, len(categories)-1)``.
- The viewer renders the dimension slider with the category names
  instead of bare integers.
- Combined with ``nd_transform={'species': {'permutation': [...]}}`` to
  remap a node's local category order into the scene's world order
  (see ``nd_transform_example.py`` for the cross-instrument case).

Scene: 200 points scattered across (X, Y, Z) ∈ [-3, 3]³, each tagged
with one of three species labels. Slicing the species dimension shows
only the points in that category.

Educational value:
- Discover the ``categories=`` shorthand on ``Dimension``.
- See category labels in the viewer's slider UI.
- Foundation for category-based filtering of multi-label datasets.
"""

import numpy as np
from arbol import aprint

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def main() -> None:
    """Author a small scene with a categorical non-displayed dimension."""
    output_path = get_examples_output_dir() / "categorical_dimensions_example.zarr"
    aprint(f"Writing categorical-dimensions example to {output_path}")

    rng = np.random.default_rng(seed=0)
    n_per_species = 200
    species_names = ["cat", "dog", "bird"]

    # Build (N, 4) positions where the 4th column is the species index.
    positions_per = []
    colors_per = []
    palette = np.array(
        [
            [0.95, 0.45, 0.25],  # warm orange — cat
            [0.30, 0.55, 0.95],  # blue — dog
            [0.55, 0.85, 0.35],  # green — bird
        ],
        dtype=np.float32,
    )
    for species_idx, _name in enumerate(species_names):
        xyz = rng.normal(0.0, 1.0, (n_per_species, 3)).astype(np.float32)
        species_col = np.full((n_per_species, 1), species_idx, dtype=np.float32)
        positions_per.append(np.hstack([xyz, species_col]))
        colors_per.append(np.tile(palette[species_idx], (n_per_species, 1)))
    positions = np.vstack(positions_per).astype(np.float32)
    colors = np.vstack(colors_per).astype(np.float32)

    dims = Dimensions(
        [
            Dimension("x", unit="", display=True),
            Dimension("y", unit="", display=True),
            Dimension("z", unit="", display=True),
            Dimension(
                "species",
                display=False,
                # Passing categories=[...] is enough — discrete=True and
                # range=(0, 2) are auto-derived. The viewer's slider
                # shows 'cat' / 'dog' / 'bird' instead of 0 / 1 / 2.
                categories=species_names,
            ),
        ]
    )

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_points(
            "labelled",
            positions,
            colors=colors,
            radii=0.10,
            sharpness=2.0,
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")
    aprint("Press 4 then [/] to walk species labels (cat/dog/bird).")


if __name__ == "__main__":
    main()
