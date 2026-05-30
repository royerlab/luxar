#!/usr/bin/env python3
"""GSplats Basic Example — three hand-authored 3D Gaussian splats.

This example demonstrates:
- The third first-class geometry type: ``add_gsplats`` (after Points and Lines).
- The four arrays that define a Gaussian splat:
  - ``centers``: (N, 3) — splat position in scene space.
  - ``amplitudes``: (N,) or scalar — peak intensity at the center.
  - ``cholesky_factors``: (N, 6) — packed lower-triangular factor of the
    inverse covariance matrix. Defines the splat's shape and orientation.
    For a 3D splat the 6 packed entries are the lower-triangular order
    ``[L00, L10, L11, L20, L21, L22]``; for an axis-aligned splat with
    standard deviations (sx, sy, sz) the packing is
    ``[1/sx, 0, 1/sy, 0, 0, 1/sz]``.
  - ``colors``: (N, 3) — per-splat RGB.

The scene contains three splats with deliberately different shapes:
- a near-spherical splat (isotropic Cholesky),
- an axis-aligned elongated splat (anisotropic diagonal Cholesky),
- a tilted ellipsoidal splat (off-diagonal Cholesky entry).

Educational value:
- Understand the minimal authoring path for Gaussian splats without
  needing a fitting pipeline.
- See how the Cholesky packing controls shape and orientation.
- Use as a starting template before reaching for ``fit_gaussian_splats``
  (covered in ``gsplats_fit_volume_example.py``).
"""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def pack_lower_triangular(L: np.ndarray) -> np.ndarray:
    """Pack a 3x3 lower-triangular matrix into the 6-element row-major form.

    Luxar stores Cholesky factors in row-major lower-triangular order:
    ``[L00, L10, L11, L20, L21, L22]`` (see
    ``luxar.gsplats.utils.trils.tril_size``).

    Args:
        L: A ``(3, 3)`` lower-triangular numpy array.

    Returns:
        Packed ``(6,)`` float32 array.
    """
    return np.array(
        [L[0, 0], L[1, 0], L[1, 1], L[2, 0], L[2, 1], L[2, 2]],
        dtype=np.float32,
    )


def isotropic_cholesky(sigma: float) -> np.ndarray:
    """Packed Cholesky for an isotropic Gaussian with standard deviation ``sigma``.

    The Cholesky factor of the precision matrix ``Σ^{-1} = (1/σ²) I`` is
    ``(1/σ) I``, so the packed form is ``[1/σ, 0, 1/σ, 0, 0, 1/σ]``.
    """
    inv = 1.0 / sigma
    return pack_lower_triangular(np.diag([inv, inv, inv]))


def axis_aligned_cholesky(sx: float, sy: float, sz: float) -> np.ndarray:
    """Packed Cholesky for an axis-aligned anisotropic Gaussian."""
    return pack_lower_triangular(np.diag([1.0 / sx, 1.0 / sy, 1.0 / sz]))


def tilted_cholesky(sx: float, sy: float, sz: float, off_xy: float) -> np.ndarray:
    """Packed Cholesky for a Gaussian with a single off-diagonal coupling.

    Adding a non-zero L[1, 0] entry couples the X and Y axes of the
    precision matrix, tilting the splat in the XY plane.
    """
    L = np.diag([1.0 / sx, 1.0 / sy, 1.0 / sz])
    L[1, 0] = off_xy
    return pack_lower_triangular(L)


def main() -> None:
    """Build a scene with three hand-authored 3D Gaussian splats."""
    output_path = get_examples_output_dir() / "gsplats_basic_example.zarr"
    aprint(f"Writing hand-authored gsplat scene to {output_path}")

    centers = np.array(
        [
            [-3.0, 0.0, 0.0],  # spherical, red
            [0.0, 0.0, 0.0],  # axis-aligned elongated, green
            [3.0, 0.0, 0.0],  # tilted, blue
        ],
        dtype=np.float32,
    )

    amplitudes = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    cholesky_factors = np.stack(
        [
            isotropic_cholesky(sigma=0.8),
            axis_aligned_cholesky(sx=0.4, sy=1.6, sz=0.4),
            tilted_cholesky(sx=0.6, sy=0.6, sz=1.2, off_xy=0.5),
        ],
        axis=0,
    ).astype(np.float32)

    colors = np.array(
        [
            [1.0, 0.30, 0.30],
            [0.30, 1.0, 0.30],
            [0.30, 0.50, 1.0],
        ],
        dtype=np.float32,
    )

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats(
            "trio",
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=cholesky_factors,
            colors=colors,
            opacity=1.0,
            blending_mode="additive",
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")


if __name__ == "__main__":
    main()
