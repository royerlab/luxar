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


def test_add_gsplats_composes_partition_under_substitutive_lod(tmp_path) -> None:
    data = _data()
    output_path = tmp_path / "scene.luxar.zarr"

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats(
            "splats",
            data.centers,
            data.amplitudes,
            data.cholesky_factors,
            partition=dict(max_elements=4),
            substitutive_lod=dict(compression_factor=2, levels=1),
        )

    stored = zarr.open_group(output_path, mode="r")["splats"]
    assert stored.attrs["kind"] == "lod"
    assert all(stored[name].attrs["kind"] == "partition" for name in stored)
    assert (
        sum(stored["child_1"][name].attrs["n_splats"] for name in stored["child_1"])
        == data.n_splats
    )


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


def test_substitutive_lod_alias_accepts_explicit_none_defaults(tmp_path) -> None:
    with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        node = scene.add_gsplats_from_data(
            "splats", _data(), substitutive_lod=None, lod_group=None
        )

    assert node.name == "splats"


def test_explicit_none_does_not_shadow_substitutive_alias(tmp_path) -> None:
    with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        node = scene.add_gsplats_from_data(
            "splats",
            _data(),
            substitutive_lod=None,
            lod_group=dict(compression_factor=2, levels=1),
        )

    assert isinstance(node, Group)
    assert node.attrs["kind"] == "lod"


@pytest.mark.parametrize("keyword", ["lod_group", "substitutive_lod"])
def test_nested_file_accepts_false_substitutive_bypass(tmp_path, keyword) -> None:
    source_path = tmp_path / "partitioned.gsplats.zarr"
    write_gsplats_tree(
        source_path,
        _data().to_spatial_partition(max_elements=4),
        ordering="none",
    )

    with LuxarZarrCompiler(tmp_path / f"{keyword}.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        node = scene.add_gsplats_from_file("splats", source_path, **{keyword: False})

    assert isinstance(node, Group)
    assert node.attrs["kind"] == "partition"


def test_array_lod_preserves_payload_values_and_dtypes(tmp_path) -> None:
    from luxar.core.group.group import _gsplat_data_from_arrays

    data = _data()
    amplitudes = np.linspace(0.1, 1.6, data.n_splats, dtype=np.float32)
    colors = (0.2, 0.4, 0.6)
    actual = _gsplat_data_from_arrays(
        data.centers,
        amplitudes,
        data.cholesky_factors,
        colors,
        None,
        None,
    )

    np.testing.assert_allclose(actual.amplitudes, amplitudes)
    np.testing.assert_allclose(actual.colors, np.broadcast_to(colors, (16, 3)))
    assert actual.amplitudes.dtype == np.float32
    assert actual.colors.dtype == np.float32


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"colors": np.ones((15, 3), dtype=np.float32)}, "Number of colors"),
        ({"amplitudes": np.ones(15, dtype=np.float32)}, "doesn't match n_splats"),
        (
            {"cholesky_factors": np.ones((16, 5), dtype=np.float32)},
            r"expected \(16, 6\)",
        ),
    ],
)
def test_array_lod_keeps_flat_channel_validation(tmp_path, kwargs, match) -> None:
    data = _data()
    call = {
        "centers": data.centers,
        "amplitudes": data.amplitudes,
        "cholesky_factors": data.cholesky_factors,
        **kwargs,
    }
    with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match=match) as error:
            scene.add_gsplats("splats", additive_lod=dict(n_lods=2), **call)

    assert str(error.value).startswith("Could not add gsplats 'splats':")


def test_array_substitutive_lod_rejects_nonfinite_centers_cleanly(tmp_path) -> None:
    data = _data()
    centers = data.centers.copy()
    centers[0, 0] = np.nan
    with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="positions: Contains 1 NaN or Inf"):
            scene.add_gsplats(
                "splats",
                centers,
                data.amplitudes,
                data.cholesky_factors,
                substitutive_lod=dict(levels=1),
            )


@pytest.mark.parametrize("keyword", ["labels", "keys"])
def test_array_additive_lod_reports_unsupported_annotations(tmp_path, keyword) -> None:
    data = _data()
    with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="additive ladder"):
            scene.add_gsplats(
                "splats",
                data.centers,
                data.amplitudes,
                data.cholesky_factors,
                additive_lod=dict(n_lods=2),
                **{keyword: [str(i) for i in range(data.n_splats)]},
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


def test_array_adder_keeps_partition_plus_additive_refusal_transactional(
    tmp_path,
) -> None:
    output_path = tmp_path / "scene.luxar.zarr"
    data = _data()

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(
            ValueError,
            match="partition= is not supported alongside an additive_lod= ladder",
        ):
            scene.add_gsplats(
                "splats",
                data.centers,
                data.amplitudes,
                data.cholesky_factors,
                partition=dict(max_elements=4),
                additive_lod=dict(n_lods=2),
            )

    assert "splats" not in zarr.open_group(output_path, mode="r")


def test_nested_file_requires_flatten_before_relod(tmp_path) -> None:
    source_path = tmp_path / "partitioned.gsplats.zarr"
    write_gsplats_tree(
        source_path,
        _data().to_spatial_partition(max_elements=4),
        ordering="none",
    )
    output_path = tmp_path / "scene.luxar.zarr"

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="flatten=True is required"):
            scene.add_gsplats_from_file(
                "splats",
                source_path,
                substitutive_lod=dict(compression_factor=2, levels=1),
            )

    assert "splats" not in zarr.open_group(output_path, mode="r")
