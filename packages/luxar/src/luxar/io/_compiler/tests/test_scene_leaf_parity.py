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
    "has_label_ids",
    "label_vocabulary",
    "ordering",
    "center_bounds",
    "position_bounds",
    "amplitude_range",
    "opacity",
    "absorption",
    "gamma",
    "intensity",
    "offset",
    # NOTE: blending_mode is intentionally absent — the writers no longer
    # stamp a default (it has no identity value, so a stamped default would
    # shadow ancestor-set modes under the viewer's nearest-setter-wins
    # composition). Both paths share one writer, so parity still holds.
    "truncation_radius",
)


def test_standalone_leaf_matches_scene_leaf():
    centers, amplitudes, cholesky = _splats(64)
    label_ids = (np.arange(len(centers)) % 5).astype(np.uint8)
    label_vocabulary = {index: f"class-{index}" for index in range(5)}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1) Scene leaf via the compiler (the production scene write path).
        #    The compiler spatially orders by default (hilbert); match that on
        #    the standalone side so the row order — and thus arrays — coincide.
        scene_path = tmp / "scene.luxar.zarr"
        with LuxarZarrCompiler(scene_path, encoding_mode=EncodingMode.PRECISION) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                label_ids=label_ids,
                label_vocabulary=label_vocabulary,
            )
        scene_leaf = zarr.open_group(str(scene_path), mode="r")["g"]

        # 2) Standalone leaf via the shared walker (same hilbert ordering).
        std_path = tmp / "standalone.gsplats.zarr"
        save_gsplats(
            path=std_path,
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=cholesky,
            label_ids=label_ids,
            label_vocabulary=label_vocabulary,
            ordering="hilbert",
            encoding_mode=EncodingMode.PRECISION,
        )
        std_leaf = zarr.open_group(str(std_path), mode="r")

        # Arrays: byte-identical (same data, same encoding, no reordering).
        for arr in (
            "centers",
            "amplitudes",
            "cholesky_factors_diag",
            "cholesky_factors_offdiag",
            "label_ids",
        ):
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


def test_public_writer_accepts_label_id_sequences(tmp_path: Path) -> None:
    centers, amplitudes, cholesky = _splats(8)
    scene_path = tmp_path / "scene.luxar.zarr"

    with LuxarZarrCompiler(scene_path, encoding_mode=EncodingMode.PRECISION) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        compiler.write_gsplats(
            "g",
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=cholesky,
            label_ids=[0] * len(centers),
            label_vocabulary={0: "zero"},
        )

    group = zarr.open_group(str(scene_path), mode="r")["g"]
    np.testing.assert_array_equal(group["label_ids"][:], [0])
    assert group.attrs["label_vocabulary"] == {"0": "zero"}


def test_standalone_leaf_matches_scene_leaf_under_the_centers_sigma_rail():
    """Parity must also hold for a decision taken INSIDE the shared writer.

    Every other case here writes at ``PRECISION``, where
    ``_resolve_centers_encoding_mode`` returns immediately — so the centers
    sigma rail (#1748), the one place the writer overrides the caller's
    encoding mode, was never exercised by the parity invariant at all. It is
    also the one decision that depends on a *sibling* array (the Cholesky), so
    a path that lost the Cholesky, or called the rail with a different axis
    count, would diverge here and nowhere else.

    The fixture is the merge case the rail exists for: a continuous 4th axis
    (no grid to snap to) carrying a 4% ``sigma=0`` track sub-population.
    """
    import pytest

    from luxar.core.dimensions import Dimension
    from luxar.encoding.decoder import ArrayDecoder

    n, n_tracks = 5_000, 200
    rng = np.random.default_rng(1748)
    times = rng.random(n) * 4.0
    times[-n_tracks:] = rng.integers(0, 5, size=n_tracks).astype(np.float64)
    centers = np.column_stack([rng.random((n, 3)) * 10.0, times]).astype(np.float32)
    chol = np.zeros((n, 10), dtype=np.float32)
    chol[:, [0, 2, 5, 9]] = rng.uniform(0.5, 1.5, size=(n, 4))
    chol[:, 9] = 3.0
    chol[-n_tracks:, 9] = 1e-7
    amplitudes = rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32)

    dims = Dimensions(
        [
            Dimension("X", display=True),
            Dimension("Y", display=True),
            Dimension("Z", display=True),
            Dimension("Time", display=False, range=(0.0, 4.0)),
        ]
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        scene_path = tmp / "scene.luxar.zarr"
        with LuxarZarrCompiler(scene_path, encoding_mode=EncodingMode.AUTO) as c:
            scene = c.create_scene(dimensions=dims)
            with pytest.warns(UserWarning, match=r"axis 3.*sigma"):
                scene.add_gsplats(
                    "g",
                    centers=centers,
                    amplitudes=amplitudes,
                    cholesky_factors=chol,
                    extend_to_all=[],
                )
        scene_leaf = zarr.open_group(str(scene_path), mode="r")["g"]

        std_path = tmp / "standalone.gsplats.zarr"
        with pytest.warns(UserWarning, match=r"axis 3.*sigma"):
            save_gsplats(
                path=std_path,
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=chol,
                ordering="hilbert",
                encoding_mode=EncodingMode.AUTO,
            )
        std_leaf = zarr.open_group(str(std_path), mode="r")

        # The rail really did fire (otherwise this asserts parity of the
        # ordinary uint16 path the other tests already cover).
        scene_enc = dict(scene_leaf["centers"].attrs["encoding"])
        std_enc = dict(std_leaf["centers"].attrs["encoding"])
        assert scene_enc["name"] == "float32"
        assert std_enc == scene_enc

        # Compared as row MULTISETS. The two paths do not agree on the row
        # order for this fixture and are not meant to: the scene knows Time is
        # a discrete dimension and groups by it, while the standalone writer
        # auto-detects a barrier from the centers alone and finds none on a
        # continuous axis. Row order is the subject of the other tests here;
        # what the rail owes is that both stored the same VALUES, exactly.
        decoder = ArrayDecoder()
        std_rows = np.asarray(decoder.decode(std_leaf["centers"], zarr_root=std_leaf))
        scene_rows = np.asarray(
            decoder.decode(scene_leaf["centers"], zarr_root=scene_leaf)
        )
        np.testing.assert_array_equal(
            std_rows[np.lexsort(std_rows.T)], scene_rows[np.lexsort(scene_rows.T)]
        )
        # ...and that those values are the originals, bit for bit.
        np.testing.assert_array_equal(
            scene_rows[np.lexsort(scene_rows.T)], centers[np.lexsort(centers.T)]
        )


def _sublod(n: int, seed: int):
    """A full-array AdditiveSubLOD (3D, diagonal Cholesky)."""
    from luxar.gsplats.gsplat_data import AdditiveSubLOD

    rng = np.random.default_rng(seed)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
    return AdditiveSubLOD(
        centers=rng.uniform(0, 50, size=(n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32),
        cholesky_factors=chol,
    )


def test_scene_additive_ladder_matches_standalone():
    """A scene additive ladder (now written through the SAME walker as the
    standalone writer) is byte-identical to the standalone one — child arrays,
    chunking, attrs, and ``n_additive_sublods``."""
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    data = GSplatData(additive_sublods=[_sublod(80, 1), _sublod(30, 2)])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scene_path = tmp / "scene.luxar.zarr"
        with LuxarZarrCompiler(scene_path, encoding_mode=EncodingMode.PRECISION) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("g", data)
        scene_leaf = zarr.open_group(str(scene_path), mode="r")["g"]

        std_path = tmp / "standalone.gsplats.zarr"
        write_gsplats_tree(
            std_path,
            data.tree,
            ordering="hilbert",
            encoding_mode=EncodingMode.PRECISION,
        )
        std_leaf = zarr.open_group(str(std_path), mode="r")

        assert scene_leaf.attrs["n_additive_sublods"] == 2
        assert std_leaf.attrs["n_additive_sublods"] == 2
        for i in range(2):
            for arr in (
                "centers",
                "amplitudes",
                "cholesky_factors_diag",
                "cholesky_factors_offdiag",
            ):
                np.testing.assert_array_equal(
                    std_leaf[f"additive_{i}"][arr][:],
                    scene_leaf[f"additive_{i}"][arr][:],
                    err_msg=f"additive_{i}/{arr} differs",
                )


def test_scene_lod_group_matches_standalone():
    """A scene kind=lod group (built by the structural recursion) matches the
    standalone kind=lod group (built by the tree walker): same child_<i> arrays,
    same per-child ``coverage_fraction``, same group ``kind``/``default_level``.

    The two are written by *different* code, so this guards real drift."""
    from luxar.gsplats.gsplat_data import GSplatData, SubstitutiveLevel
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    pyramid = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(additive_sublods=[_sublod(64, 10)], level_index=0),
            SubstitutiveLevel(
                additive_sublods=[_sublod(16, 11)],
                compression_factor=4,
                level_index=1,
            ),
        ]
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scene_path = tmp / "scene.luxar.zarr"
        with LuxarZarrCompiler(scene_path, encoding_mode=EncodingMode.PRECISION) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_data("g", pyramid)
        scene_lod = zarr.open_group(str(scene_path), mode="r")["g"]

        std_path = tmp / "standalone.gsplats.zarr"
        write_gsplats_tree(
            std_path,
            pyramid.tree,
            ordering="hilbert",
            encoding_mode=EncodingMode.PRECISION,
        )
        std_lod = zarr.open_group(str(std_path), mode="r")

        # Both are kind=lod with 2 children, coarsest=child_0.
        assert scene_lod.attrs["kind"] == "lod"
        assert std_lod.attrs["kind"] == "lod"
        # The viewer's initial level is the COARSEST child (child_0) on BOTH
        # paths — a progressive-load hint. Loading the finest by default would
        # render "backwards" (eager full-res). Lock it to 0, not just equal.
        assert scene_lod.attrs["default_level"] == 0
        assert std_lod.attrs["default_level"] == 0
        for i in range(2):
            sc, st = scene_lod[f"child_{i}"], std_lod[f"child_{i}"]
            assert sc.attrs["coverage_fraction"] == st.attrs["coverage_fraction"], (
                f"child_{i} coverage_fraction differs"
            )
            for arr in (
                "centers",
                "amplitudes",
                "cholesky_factors_diag",
                "cholesky_factors_offdiag",
            ):
                np.testing.assert_array_equal(
                    st[arr][:], sc[arr][:], err_msg=f"child_{i}/{arr} differs"
                )


def test_scene_partition_matches_standalone():
    """A scene kind=partition group (raw-array BSP wrapper) matches the standalone
    one (GSplatData.to_spatial_partition -> tree walker): same part_<i> arrays,
    same kind/max_elements. Same centers + same median BSP -> same parts."""
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    rng = np.random.default_rng(7)
    # Two separated clusters so the BSP split is deterministic.
    a = rng.uniform(0, 10, size=(40, 3)).astype(np.float32)
    b = (
        np.array([100.0, 100.0, 100.0], dtype=np.float32)
        + rng.uniform(0, 10, size=(40, 3))
    ).astype(np.float32)
    centers = np.concatenate([a, b], axis=0)
    n = centers.shape[0]
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    amplitudes = np.ones(n, dtype=np.float32)
    data = GSplatData(centers=centers, amplitudes=amplitudes, cholesky_factors=chol)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scene_path = tmp / "scene.luxar.zarr"
        with LuxarZarrCompiler(scene_path, encoding_mode=EncodingMode.PRECISION) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=chol,
                partition={"max_elements": 40, "rule": "median"},
            )
        scene_part = zarr.open_group(str(scene_path), mode="r")["g"]

        std_path = tmp / "standalone.gsplats.zarr"
        write_gsplats_tree(
            std_path,
            data.to_spatial_partition(max_elements=40, rule="median"),
            ordering="hilbert",
            encoding_mode=EncodingMode.PRECISION,
        )
        std_part = zarr.open_group(str(std_path), mode="r")

        assert scene_part.attrs["kind"] == "partition"
        assert std_part.attrs["kind"] == "partition"
        assert scene_part.attrs["max_elements"] == std_part.attrs["max_elements"]
        n_scene = sum(1 for k in scene_part if str(k).startswith("part_"))
        n_std = sum(1 for k in std_part if str(k).startswith("part_"))
        assert n_scene == n_std and n_scene >= 2
        for i in range(n_scene):
            for arr in (
                "centers",
                "amplitudes",
                "cholesky_factors_diag",
                "cholesky_factors_offdiag",
            ):
                np.testing.assert_array_equal(
                    std_part[f"part_{i}"][arr][:],
                    scene_part[f"part_{i}"][arr][:],
                    err_msg=f"part_{i}/{arr} differs",
                )


def test_standalone_leaf_matches_scene_leaf_with_colors_and_ordering():
    # Deliberately large enough that the chunk_size atom is SMALLER than the
    # array. At the previous n=128 the atom equalled the row count, so every
    # array got one full-array chunk on both paths and the chunk-parity
    # assertions below could not fail — they passed for the wrong reason. A real
    # divergence (per-array byte-budget sizing on one path only) was invisible
    # at that size and only shows up once arrays span several atoms.
    n = 20_000
    centers, amplitudes, cholesky = _splats(n, seed=3)
    colors = np.random.default_rng(9).uniform(0, 1, size=(n, 3)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        scene_path = tmp / "scene.luxar.zarr"
        with LuxarZarrCompiler(scene_path, encoding_mode=EncodingMode.PRECISION) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
            )
        scene_leaf = zarr.open_group(str(scene_path), mode="r")["g"]

        std_path = tmp / "standalone.gsplats.zarr"
        save_gsplats(
            path=std_path,
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=cholesky,
            colors=colors,
            ordering="hilbert",
            encoding_mode=EncodingMode.PRECISION,
        )
        std_leaf = zarr.open_group(str(std_path), mode="r")

        # Same ordering method → same Hilbert sort → identical arrays + chunking.
        for arr in (
            "centers",
            "amplitudes",
            "cholesky_factors_diag",
            "cholesky_factors_offdiag",
            "colors",
            "chunk_bounds",
        ):
            np.testing.assert_array_equal(
                std_leaf[arr][:], scene_leaf[arr][:], err_msg=f"{arr} differs"
            )
        for arr in (
            "centers",
            "amplitudes",
            "cholesky_factors_diag",
            "cholesky_factors_offdiag",
        ):
            assert std_leaf[arr].chunks == scene_leaf[arr].chunks, (
                f"{arr} chunks differ"
            )
        assert std_leaf.attrs["has_colors"] is True
        assert std_leaf.attrs["chunk_size"] == scene_leaf.attrs["chunk_size"]
