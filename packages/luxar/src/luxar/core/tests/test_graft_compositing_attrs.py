"""Attr routing when a nested `.gsplats.zarr` is GRAFTED into a scene.

``graft_gsplat_node`` (``core/group/gsplats_pipeline/from_io.py``) is the third
writer of a multi-node gsplats subtree, next to
``lod_dispatch.add_gsplats_as_lod_group_impl`` and the ``adders/``
partition-wrapping path. All three must route
:data:`~luxar.core.group.compositing.COMPOSITING_ATTRS` the same way — onto the
WRAPPER only — because the viewer resolves them down the ancestry.

``blending_mode`` is the one that bites: it is *nearest-setter-wins*, not
multiplicative. Re-stamping it on each grafted part (which the graft path used
to do "belt-and-suspenders") makes every part SHADOW the wrapper, so the layers
panel's single Blend control becomes inert on exactly the part-based recipes
(``tiles`` / ``overview`` / ``adaptive``) while flat/stream/levels layers keep
responding.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import write_gsplats_tree


def _two_clusters(n_per: int = 30) -> GSplatData:
    """Two well-separated clusters so the BSP split is unambiguous."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 10, size=(n_per, 3)).astype(np.float32)
    b = (
        np.array([100.0, 100.0, 100.0], dtype=np.float32)
        + rng.uniform(0, 10, size=(n_per, 3))
    ).astype(np.float32)
    centers = np.concatenate([a, b], axis=0)
    n = centers.shape[0]
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    return GSplatData(
        centers=centers,
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )


def _partition_file(dirpath: Path) -> Path:
    """A standalone kind=partition ``.gsplats.zarr`` (what ``fit --tiling`` writes)."""
    path = dirpath / "part.gsplats.zarr"
    write_gsplats_tree(
        path,
        _two_clusters(30).to_spatial_partition(max_elements=30),
        ordering="none",
        encoding_mode=EncodingMode.PRECISION,
    )
    return path


def _scene_dims() -> Dimensions:
    return Dimensions(
        [
            Dimension("x", unit="px", display=True),
            Dimension("y", unit="px", display=True),
            Dimension("z", unit="px", display=True),
        ]
    )


def test_grafted_partition_keeps_blending_mode_on_the_wrapper_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        src = _partition_file(tmpdir)
        scene_path = tmpdir / "scene.luxar.zarr"

        with LuxarZarrCompiler(
            scene_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(dimensions=_scene_dims())
            scene.add_gsplats_from_file(
                "tiles",
                str(src),
                opacity=1.0,
                absorption=1.0,
                blending_mode="volumetric",
                layer=True,
            )

        root = zarr.open_group(str(scene_path), mode="r")
        wrapper = root["tiles"]
        assert wrapper.attrs["kind"] == "partition"
        # The layer — and the ONLY node that may set the mode.
        assert wrapper.attrs["blending_mode"] == "volumetric"
        assert wrapper.attrs["layer"] is True

        part_names = [k for k in wrapper.group_keys() if str(k).startswith("part_")]
        assert len(part_names) >= 2, "expected a real BSP split"
        for name in part_names:
            part_attrs = dict(wrapper[name].attrs)
            # A stamped copy here is nearest-setter-wins → it shadows the
            # wrapper and the layer's Blend control stops doing anything.
            assert "blending_mode" not in part_attrs, (
                f"part {name} shadows the wrapper's blending_mode"
            )
            # `layer` likewise stays on the wrapper: the parts are internal
            # structure, not user-facing layers.
            assert "layer" not in part_attrs


def test_grafted_partition_parts_are_not_exposed_as_layers() -> None:
    """Only the wrapper is a layer, so the panel shows ONE row for the graft."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        src = _partition_file(tmpdir)
        scene_path = tmpdir / "scene.luxar.zarr"

        with LuxarZarrCompiler(
            scene_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(dimensions=_scene_dims())
            scene.add_gsplats_from_file("tiles", str(src), layer=True)

        root = zarr.open_group(str(scene_path), mode="r")

        n_layers = 0

        def _visit(group: zarr.Group) -> None:
            nonlocal n_layers
            if group.attrs.get("layer") is True:
                n_layers += 1
            for name in group.group_keys():
                _visit(group[name])

        _visit(root)
        assert n_layers == 1


# ────────────────────────────────────────────────────────────────────────
# Topology-aware coverage_fraction fallback (parity with the standalone writer)
# ────────────────────────────────────────────────────────────────────────


def _one_part_adaptive():
    """The shape ``gsplat lod --recipe adaptive`` emits below ``max_elements``:
    a ``kind=partition`` holding a SINGLE per-part lod group."""
    from luxar.gsplats.lod.recipes import RecipeParams, build_recipe
    from luxar.gsplats.tree import GSplatPartition, without_meta_key

    node = build_recipe(
        _two_clusters(60),
        "adaptive",
        # max_elements unset → the default 1,000,000, so the BSP never splits.
        RecipeParams(compression_factor=4, levels=2, device="cpu", seed=0),
    )
    assert isinstance(node, GSplatPartition) and len(node.children) == 1
    # Scrubbed, so the graft's FALLBACK derivation is what gets exercised.
    return without_meta_key(node, "coverage_fraction")


def test_grafted_one_part_partition_uses_the_whole_object_anchor() -> None:
    """A one-part partition is not a tiling, so the ladder under it keeps the
    whole-object anchor (finest 1.0) — the same rule
    ``gsplat_tree.write_gsplat_node`` applies, which is what makes a
    file → scene graft agree with a standalone rewrite."""
    from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node

    with tempfile.TemporaryDirectory() as tmp:
        scene_path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(
            scene_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(dimensions=_scene_dims())
            graft_gsplat_node(scene, name="adaptive", node=_one_part_adaptive())

        root = zarr.open_group(str(scene_path), mode="r")
        lod = root["adaptive"]["part_0"]
        covs = [
            float(lod[k].attrs["coverage_fraction"])
            for k in sorted(lod.group_keys(), key=lambda s: int(s.split("_")[1]))
        ]
        assert covs[0] == 0.0
        assert covs[-1] == 1.0, f"expected the whole-object anchor; got {covs}"
