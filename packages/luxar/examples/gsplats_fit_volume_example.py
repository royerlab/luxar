#!/usr/bin/env python3
"""GSplats Fit-Volume Example — fit a Gaussian-splat representation to a tiny volume.

This example demonstrates the **fit-from-volume** workflow:

1. Build a small synthetic 32³ volume containing a few isotropic 3D
   Gaussian blobs (think: a stand-in for a confocal microscopy stack).
2. Run ``fit_gaussian_splats`` to recover a sparse splat representation.
3. Drop the resulting ``GSplatData`` into a scene via
   ``scene.add_gsplats_from_data(...)``.

The volume is intentionally minimal (32³, 4 blobs, 50 iterations) so the
example runs in seconds on CPU. Production-quality fits live in the
``luxar gsplat fit`` CLI and the gsplats demos under
``packages/luxar/src/luxar/demos/demo_gsplats_*.py``.

Educational value:
- See the full hand-off from a volume array to a scene-ready splat
  representation without leaving Python.
- Discover how ``add_gsplats_from_data`` consumes a ``GSplatData``
  object (the same object the CLI's ``luxar gsplat fit`` writes to
  ``.gsplats.zarr``).
"""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.gsplats import fit_gaussian_splats
from luxar.utils.paths import get_examples_output_dir


def make_synthetic_volume(size: int = 32, seed: int = 0) -> np.ndarray:
    """Build a 3D volume containing a small number of Gaussian blobs.

    Returns a ``(size, size, size)`` float32 array with values roughly in
    [0, 1]. The fitter recovers each blob as one (or a few) splats.
    """
    rng = np.random.default_rng(seed)
    n_blobs = 4
    centers = rng.integers(size // 4, 3 * size // 4, size=(n_blobs, 3))
    sigmas = rng.uniform(2.0, 4.0, size=n_blobs)
    amps = rng.uniform(0.6, 1.0, size=n_blobs)

    z_idx, y_idx, x_idx = np.ogrid[:size, :size, :size]
    volume = np.zeros((size, size, size), dtype=np.float32)
    for (cz, cy, cx), sigma, amp in zip(centers, sigmas, amps):
        d2 = (z_idx - cz) ** 2 + (y_idx - cy) ** 2 + (x_idx - cx) ** 2
        volume += amp * np.exp(-d2 / (2.0 * sigma**2))
    return volume.astype(np.float32)


def main() -> None:
    """Fit Gaussian splats to a synthetic volume and write the resulting scene."""
    output_path = get_examples_output_dir() / "gsplats_fit_volume_example.zarr"
    aprint(f"Writing fitted-gsplat example to {output_path}")

    volume = make_synthetic_volume(size=32, seed=0)
    aprint(f"Synthetic volume: shape={volume.shape}, range=[{volume.min():.2f}, {volume.max():.2f}]")

    # Tiny fit — 50 iterations is enough for 4 isotropic blobs.
    # ``device='cpu'`` keeps the example dependency-free; switch to
    # 'mps' / 'cuda' for real volumes.
    result = fit_gaussian_splats(
        volume,
        seeds=12,
        n_iters=50,
        device="cpu",
        verbose=False,
    )
    aprint(
        f"Fit complete: {result.centers.shape[0]} splats, "
        f"time={result.stats.get('time_seconds', 0.0):.2f}s, "
        f"iters={result.stats.get('iterations', 'n/a')}"
    )

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats_from_data(
            "fitted_blobs",
            result,
            opacity=1.0,
            blending_mode="additive",
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")


if __name__ == "__main__":
    main()
