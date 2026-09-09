"""Structural keyword parity for the three public gsplat adders."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group import Group
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.io.compiler import LuxarZarrCompiler


def _data(n_splats: int = 16) -> GSplatData:
    rng = np.random.RandomState(2482)
    centers = rng.uniform(0, 100, size=(n_splats, 3)).astype(np.float32)
    cholesky = np.zeros((n_splats, 6), dtype=np.float32)
    cholesky[:, [0, 2, 5]] = 1.0
    return GSplatData(
        centers=centers,
        amplitudes=np.ones(n_splats, dtype=np.float32),
        cholesky_factors=cholesky,
    )


def test_public_gsplat_adders_declare_uniform_structure_keywords() -> None:
    for method_name in (
        "add_gsplats",
        "add_gsplats_from_data",
        "add_gsplats_from_file",
    ):
        parameters = inspect.signature(getattr(Group, method_name)).parameters
        assert "substitutive_lod" in parameters
        assert "additive_lod" in parameters

    file_parameters = inspect.signature(Group.add_gsplats_from_file).parameters
    assert "partition" in file_parameters
    assert "flatten" in file_parameters


def test_add_gsplats_composes_substitutive_and_additive_lod(tmp_path) -> None:
    data = _data()
    output_path = tmp_path / "scene.luxar.zarr"

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        node = scene.add_gsplats(
            "splats",
            data.centers,
            1.0,
            data.cholesky_factors,
            colors=(0.2, 0.4, 0.6),
            substitutive_lod=dict(compression_factor=2, levels=1),
            additive_lod=dict(n_lods=2),
        )
        assert isinstance(node, Group)

    stored = zarr.open_group(output_path, mode="r")["splats"]
    assert stored.attrs["kind"] == "lod"
    assert sorted(stored.keys()) == ["child_0", "child_1"]
    assert all(stored[name].attrs["n_additive_sublods"] == 2 for name in stored)
    assert stored["child_1"].attrs["n_splats"] == 16


def test_substitutive_lod_alias_refuses_ambiguous_double_spec(tmp_path) -> None:
    with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(
            ValueError,
            match="Pass only one of substitutive_lod= and lod_group=",
        ):
            scene.add_gsplats_from_data(
                "splats",
                _data(),
                substitutive_lod=False,
                lod_group=False,
            )


def test_add_gsplats_from_file_flattens_then_restructures(tmp_path) -> None:
    source = _data()
    source_path = tmp_path / "partitioned.gsplats.zarr"
    write_gsplats_tree(
        source_path,
        source.to_spatial_partition(max_elements=4),
        ordering="none",
    )
    output_path = tmp_path / "scene.luxar.zarr"

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats_from_file(
            "splats",
            source_path,
            flatten=True,
            additive_lod=dict(n_lods=2),
        )

    stored = zarr.open_group(output_path, mode="r")["splats"]
    assert stored.attrs["type"] == "gsplats"
    assert stored.attrs["n_splats"] == source.n_splats
    assert stored.attrs["n_additive_sublods"] == 2
