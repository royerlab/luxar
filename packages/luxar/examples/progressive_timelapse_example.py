#!/usr/bin/env python3
"""Progressive Timelapse Example - additive LODs across a discrete time dimension.

This example demonstrates:
- A 4D points scene (x, y, z + hidden discrete ``frame`` dimension)
- Additive LOD ladders (``add_points(additive_lod=...)``) combined with
  frame-based navigation
- The viewer's SliceCache ("S-cache"): scrubbing back to an already-visited
  frame restores its decoded slice instantly instead of re-streaming LODs

Educational value:
- Learn how progressive (additive-LOD) loading composes with discrete
  dimension navigation
- Watch the Data Loading Monitor's Cache tab: the SLICE CACHE section's hit
  count rises as you scrub back and forth through frames
- Canonical E2E fixture for the S-cache revisit behavior
  (``slice-cache.spec.ts``)
"""

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir

N_FRAMES = 8
POINTS_PER_FRAME = 40_000
N_LODS = 4


def create_frame_cloud(rng: np.random.Generator, frame: int) -> np.ndarray:
    """A Gaussian ball that drifts and tightens as the frame advances.

    The drift makes each frame visually distinct (easy manual verification);
    the tightening varies density so LOD ladders differ per frame.
    """
    sigma = 0.55 - 0.03 * frame
    xyz = rng.normal(0.0, sigma, size=(POINTS_PER_FRAME, 3)).astype(np.float32)
    xyz[:, 0] += 0.18 * frame  # drift in +x
    frame_col = np.full((POINTS_PER_FRAME, 1), float(frame), dtype=np.float32)
    return np.concatenate([xyz, frame_col], axis=1)


def main() -> None:
    output_path = get_examples_output_dir() / "progressive_timelapse_example.luxar.zarr"

    with asection(f"Building progressive timelapse example at {output_path}"):
        rng = np.random.default_rng(42)
        positions = np.concatenate(
            [create_frame_cloud(rng, k) for k in range(N_FRAMES)]
        )
        aprint(f"{len(positions)} points across {N_FRAMES} frames")

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=Dimensions(
                    [
                        Dimension(name="x", range=(-2.0, 3.5), display=True),
                        Dimension(name="y", range=(-2.0, 2.0), display=True),
                        Dimension(name="z", range=(-2.0, 2.0), display=True),
                        Dimension(
                            name="frame",
                            range=(0.0, float(N_FRAMES - 1)),
                            step=1.0,
                            display=False,
                            discrete=True,
                        ),
                    ]
                ),
            )

            scene.add_points(
                "timelapse_points",
                positions=positions,
                radii=np.full(len(positions), 0.012, dtype=np.float32),
                additive_lod=dict(n_lods=N_LODS),
            )

        aprint(f"Done: {N_FRAMES} frames x {POINTS_PER_FRAME} points, {N_LODS} additive LODs")


if __name__ == "__main__":
    main()
