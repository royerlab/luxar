#!/usr/bin/env python3
"""Points Substitutive LOD Example — coarse levels rendered as Gaussian splats.

Points coarsen *additively* by decimation (drop points), which flickers and dims
a dense cloud when you zoom out. ``substitutive_lod=`` instead synthesises
**mass-preserving Gaussian splats** for the coarse levels: each point is lifted
to an isotropic Gaussian and the gsplat substitutive pipeline builds
fewer-but-larger representatives. The finest LOD child stays the original Points
node; coarser children are gsplats, assembled as one ``kind=lod`` Group.

Why it looks seamless:
- The point kernel at the default sharpness is a true Gaussian — the same shape
  as the gsplat kernel — so there is no "pop" at the LOD transition.
- The lift is calibrated (σ = 2R/T, amplitude = peak-matched) so a single lifted
  splat renders like its point, and coarse-level amplitudes are rescaled to
  conserve total render-light → no brightness change on zoom-out.

Layout: a dense blobby cloud authored two ways for comparison —
- ``substitutive``: coarse levels are synthesised gsplats (this feature).
- ``additive``: coarse levels are decimated points (the baseline).

Educational value:
- Zoom OUT on ``substitutive``: the cloud stays smooth and equally bright as the
  viewer swaps in the coarse gsplat levels.
- Zoom OUT on ``additive``: the decimated points thin out and the cloud dims.
- Both report type "points" in the layers panel (the substitutive group's finest
  child is the Points node; display_type is derived from it).
"""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def make_cloud(n: int = 60000, seed: int = 0):
    """A dense cloud: several gaussian blobs over a faint uniform haze."""
    rng = np.random.default_rng(seed)
    n_blobs = 12
    blob_centers = rng.uniform(-60, 60, (n_blobs, 3))
    n_blob = int(n * 0.8)
    blob_pts = (
        blob_centers[rng.integers(0, n_blobs, n_blob)]
        + rng.normal(0, 6, (n_blob, 3))
    ).astype(np.float32)
    haze = rng.uniform(-70, 70, (n - n_blob, 3)).astype(np.float32)
    positions = np.concatenate([blob_pts, haze], axis=0)
    # Warm blobs, cool haze.
    colors = np.concatenate(
        [
            np.tile(np.array([1.0, 0.6, 0.2], np.float32), (n_blob, 1)),
            np.tile(np.array([0.2, 0.4, 1.0], np.float32), (n - n_blob, 1)),
        ]
    )
    radii = np.full(positions.shape[0], 0.8, dtype=np.float32)
    return positions, colors, radii


def main() -> None:
    output_path = (
        get_examples_output_dir() / "points_substitutive_lod_example.luxar.zarr"
    )

    with LuxarZarrCompiler(str(output_path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        positions, colors, radii = make_cloud()

        # === 1. Substitutive LOD — coarse levels are synthesised gsplats. ===
        scene.add_points(
            "substitutive",
            positions,
            colors=colors,
            radii=radii,
            layer=True,
            substitutive_lod=dict(compression_factor=4, levels=3, device="cpu"),
        )

        # === 2. Additive LOD (baseline) — coarse levels are decimated points. ===
        shifted = positions.copy()
        shifted[:, 0] += 160.0  # offset so the two layers don't overlap
        scene.add_points(
            "additive",
            shifted,
            colors=colors,
            radii=radii,
            layer=True,
            additive_lod=dict(n_lods=4, method="random", seed=0),
        )

    aprint(f"\nScene saved to: {output_path}")
    aprint(f"To view: luxar serve --viewer {output_path}")
    aprint(
        "  → Zoom OUT on 'substitutive': coarse gsplat levels keep the cloud\n"
        "     smooth and equally bright (mass-preserving).\n"
        "  → Zoom OUT on 'additive': decimated points thin and the cloud dims.\n"
        "  → Both report type 'points' in the layers panel."
    )


if __name__ == "__main__":
    main()
