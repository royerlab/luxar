#!/usr/bin/env python3
"""GSplats LOD Example — substitutive level-of-detail on a fitted splat set.

This example demonstrates the **substitutive LOD** path for Gaussian
splats: a coarse-to-fine pyramid where each coarser level *replaces*
its finer level with a smaller set of synthesized representative
splats. The viewer picks a level based on projected pixel size.

This complements the four existing LOD examples
(``progressive_points_lines_example.py``, ``split_of_lod_example.py``,
``lines_split_and_sampling_example.py``, ``energy_breakpoints_example.py``)
which all focus on **additive** LOD (each level adds new elements on
top of the previous one). Substitutive and additive are independent
axes — together they form the 2-D LOD pyramid described in the
gsplats specs.

Pipeline:
1. Fit a small synthetic volume to get a base ``GSplatData``.
2. Build a substitutive ladder with ``lod_group=dict(compression_factor=4, levels=2)``,
   producing 3 levels (the original + 2 coarsened copies).
3. The viewer renders the appropriate level given the camera distance;
   zooming in switches to finer levels automatically.

Educational value:
- See substitutive LOD authored from a single ``add_gsplats_from_data``
  call — no manual ladder construction.
- Understand the ``compression_factor`` × ``levels`` parameters.
- Set up a starting template before reaching for the lower-level
  ``make_substitutive_lod`` / ``make_lod_pyramid`` APIs.
"""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.gsplats import fit_gaussian_splats
from luxar.utils.paths import get_examples_output_dir


def make_synthetic_volume(size: int = 48, seed: int = 0) -> np.ndarray:
    """Build a 3D volume with ~12 randomly-placed Gaussian blobs."""
    rng = np.random.default_rng(seed)
    n_blobs = 12
    centers = rng.integers(size // 5, 4 * size // 5, size=(n_blobs, 3))
    sigmas = rng.uniform(1.5, 3.0, size=n_blobs)
    amps = rng.uniform(0.5, 1.0, size=n_blobs)
    z_idx, y_idx, x_idx = np.ogrid[:size, :size, :size]
    volume = np.zeros((size, size, size), dtype=np.float32)
    for (cz, cy, cx), sigma, amp in zip(centers, sigmas, amps):
        d2 = (z_idx - cz) ** 2 + (y_idx - cy) ** 2 + (x_idx - cx) ** 2
        volume += amp * np.exp(-d2 / (2.0 * sigma**2))
    return volume.astype(np.float32)


def main() -> None:
    """Fit a tiny volume and ship it as a substitutive LOD pyramid."""
    output_path = get_examples_output_dir() / "gsplats_lod_example.zarr"
    aprint(f"Writing substitutive-LOD example to {output_path}")

    volume = make_synthetic_volume(size=48, seed=0)
    aprint(f"Synthetic volume: shape={volume.shape}")

    # 80 seeds × 80 iterations is enough to produce a non-trivial splat
    # set for a 12-blob volume on CPU in a few seconds.
    result = fit_gaussian_splats(
        volume,
        seeds=80,
        n_iters=80,
        device="cpu",
        verbose=False,
    )
    aprint(f"Base fit: {result.centers.shape[0]} splats")

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        # Build a substitutive ladder: each coarser level has roughly
        # N/4 of the previous level's splats. With levels=2 we get the
        # original + 2 coarsened copies = 3 substitutive levels.
        # The viewer auto-switches based on projected pixel size.
        scene.add_gsplats_from_data(
            "lod_blobs",
            result,
            lod_group=dict(
                compression_factor=4,
                levels=2,
                method="kmeans_lloyd",
            ),
            opacity=1.0,
            blending_mode="additive",
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")
    aprint("Zoom in/out — the layers panel should show 3 substitutive levels.")


if __name__ == "__main__":
    main()
