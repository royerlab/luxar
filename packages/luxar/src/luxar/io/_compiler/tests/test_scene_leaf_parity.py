"""Scene ⇄ standalone leaf-parity test.

The core "no parallel writer" invariant: a standalone ``.gsplats.zarr`` leaf
(written by :func:`luxar.gsplats.io.save_gsplats.write_gsplats_tree` via the
shared walker) is structurally identical to the gsplats leaf the *scene*
compiler writes for the same data — because both go through the same
``gsplat_assembly`` free functions. This guards against the two paths drifting
apart (the §8 "dual divergent writers" risk).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import zarr

from luxar.core.dimensions import Dimensions
from luxar.encoding import EncodingMode
from luxar.gsplats.io import save_gsplats
from luxar.io.compiler import LuxarZarrCompiler


def _splats(n: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    k = 6
    chol = np.zeros((n, k), dtype=np.float32)
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
    return (
        (rng.uniform(0, 50, size=(n, 3))).astype(np.float32),
        rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32),
        chol,
    )


# Authoritative leaf attrs that must match between the two write paths.
_PARITY_ATTRS = (
    "type",
    "n_splats",
    "ndim",
    "has_colors",
    "ordering",
    "center_bounds",
    "position_bounds",
    "amplitude_range",
    "opacity",
    "gamma",
    "intensity",
    "offset",
    "blending_mode",
    "truncation_radius",
)


def test_standalone_leaf_matches_scene_leaf():
    centers, amplitudes, cholesky = _splats(64)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1) Scene leaf via the compiler (the production scene write path).
        #    The compiler spatially orders by default (hilbert); match that on
        #    the standalone side so the row order — and thus arrays — coincide.
        scene_path = tmp / "scene.zarr"
        with LuxarZarrCompiler(scene_path, encoding_mode=EncodingMode.PRECISION) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats(
                "g", centers=centers, amplitudes=amplitudes, cholesky_factors=cholesky,
            )
        scene_leaf = zarr.open_group(str(scene_path), mode="r")["g"]

        # 2) Standalone leaf via the shared walker (same hilbert ordering).
        std_path = tmp / "standalone.gsplats.zarr"
        save_gsplats(
            path=std_path, centers=centers, amplitudes=amplitudes,
            cholesky_factors=cholesky, ordering="hilbert",
            encoding_mode=EncodingMode.PRECISION,
        )
        std_leaf = zarr.open_group(str(std_path), mode="r")

        # Arrays: byte-identical (same data, same encoding, no reordering).
        for arr in ("centers", "amplitudes", "cholesky_factors"):
            np.testing.assert_array_equal(
                std_leaf[arr][:], scene_leaf[arr][:], err_msg=f"{arr} differs"
            )

        # Authoritative leaf attrs: identical between the two write paths.
        for key in _PARITY_ATTRS:
            assert key in scene_leaf.attrs, f"scene leaf missing {key}"
            assert key in std_leaf.attrs, f"standalone leaf missing {key}"
            assert std_leaf.attrs[key] == scene_leaf.attrs[key], (
                f"attr {key!r} differs: standalone={std_leaf.attrs[key]!r} "
                f"scene={scene_leaf.attrs[key]!r}"
            )


def test_standalone_leaf_matches_scene_leaf_with_colors_and_ordering():
    centers, amplitudes, cholesky = _splats(128, seed=3)
    colors = np.random.default_rng(9).uniform(0, 1, size=(128, 3)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scene_path = tmp / "scene.zarr"
        with LuxarZarrCompiler(scene_path, encoding_mode=EncodingMode.PRECISION) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats(
                "g", centers=centers, amplitudes=amplitudes, cholesky_factors=cholesky,
                colors=colors, ordering="hilbert",
            )
        scene_leaf = zarr.open_group(str(scene_path), mode="r")["g"]

        std_path = tmp / "standalone.gsplats.zarr"
        save_gsplats(
            path=std_path, centers=centers, amplitudes=amplitudes,
            cholesky_factors=cholesky, colors=colors, ordering="hilbert",
            encoding_mode=EncodingMode.PRECISION,
        )
        std_leaf = zarr.open_group(str(std_path), mode="r")

        # Same ordering method → same Hilbert sort → identical arrays + chunking.
        for arr in ("centers", "amplitudes", "cholesky_factors", "colors", "chunk_bounds"):
            np.testing.assert_array_equal(
                std_leaf[arr][:], scene_leaf[arr][:], err_msg=f"{arr} differs"
            )
        for arr in ("centers", "amplitudes", "cholesky_factors"):
            assert std_leaf[arr].chunks == scene_leaf[arr].chunks, f"{arr} chunks differ"
        assert std_leaf.attrs["has_colors"] is True
        assert std_leaf.attrs["chunk_size"] == scene_leaf.attrs["chunk_size"]
