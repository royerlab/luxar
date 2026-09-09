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
from luxar.gsplats.lod.additive import make_additive_lod
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


@pytest.mark.parametrize(
    "colors",
    [
        np.array([[0.2, 0.4, 0.6]], dtype=np.float32),
        np.array([[0.2, 0.4, 0.6, 0.8]], dtype=np.float32),
        (0.2, 0.4, 0.6),
        (255, 0, 0),
    ],
)
@pytest.mark.parametrize("lod_keyword", ["additive_lod", "substitutive_lod"])
def test_array_lod_preserves_broadcast_color_storage(
    tmp_path, colors, lod_keyword
) -> None:
    data = _data()
    channels = np.asarray(colors).shape[-1]
    lod_spec = (
        dict(n_lods=2)
        if lod_keyword == "additive_lod"
        else dict(compression_factor=2, levels=1)
    )
    output_path = tmp_path / "scene.luxar.zarr"

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats(
            "plain",
            data.centers,
            data.amplitudes,
            data.cholesky_factors,
            colors=colors,
        )
        scene.add_gsplats(
            "structured",
            data.centers,
            data.amplitudes,
            data.cholesky_factors,
            colors=colors,
            **{lod_keyword: lod_spec},
        )

    stored = zarr.open_group(output_path, mode="r")
    plain_colors = np.asarray(stored["plain"]["colors"])
    structured_leaf = (
        stored["structured"]["additive_1"]
        if lod_keyword == "additive_lod"
        else stored["structured"]["child_1"]
    )
    structured_colors = np.asarray(structured_leaf["colors"])
    assert plain_colors.shape == structured_colors.shape == (1, channels)
    assert plain_colors.dtype == structured_colors.dtype == np.float32
    np.testing.assert_allclose(structured_colors, plain_colors)


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("lod_group", dict(compression_factor=2, levels=1)),
        ("normalize_amplitudes", True),
    ],
)
@pytest.mark.parametrize("lod_keyword", ["additive_lod", "substitutive_lod"])
def test_array_lod_keeps_unknown_attr_refusal(
    tmp_path, attr, value, lod_keyword
) -> None:
    data = _data()
    lod_spec = (
        dict(n_lods=2)
        if lod_keyword == "additive_lod"
        else dict(compression_factor=2, levels=1)
    )

    errors = []
    for name, kwargs in (
        ("plain", {attr: value}),
        ("structured", {attr: value, lod_keyword: lod_spec}),
    ):
        with LuxarZarrCompiler(tmp_path / f"{name}.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError) as error:
                scene.add_gsplats(
                    "splats",
                    data.centers,
                    data.amplitudes,
                    data.cholesky_factors,
                    **kwargs,
                )
        errors.append(str(error.value))

    assert errors[0] == errors[1]
    assert errors[0].startswith(
        f"Could not add gsplats 'splats': Unknown node attribute '{attr}'"
    )


@pytest.mark.parametrize(
    ("attr", "value", "specialized_parent"),
    [
        ("join", "miter", False),
        ("metalness", 0.5, False),
        ("layer_order", 2, True),
    ],
)
@pytest.mark.parametrize("lod_keyword", ["additive_lod", "substitutive_lod"])
def test_array_lod_keeps_compositing_attr_refusal(
    tmp_path, attr, value, specialized_parent, lod_keyword
) -> None:
    data = _data()
    lod_spec = (
        dict(n_lods=2)
        if lod_keyword == "additive_lod"
        else dict(compression_factor=2, levels=1)
    )

    errors = []
    for name, kwargs in (
        ("plain", {attr: value}),
        ("structured", {attr: value, lod_keyword: lod_spec}),
    ):
        with LuxarZarrCompiler(tmp_path / f"{name}.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            parent = scene.add_lod_group("lod") if specialized_parent else None
            if specialized_parent:
                kwargs["coverage_fraction"] = 0.0
            with pytest.raises(ValueError) as error:
                scene.add_gsplats(
                    "splats",
                    data.centers,
                    data.amplitudes,
                    data.cholesky_factors,
                    parent=parent,
                    **kwargs,
                )
        errors.append(str(error.value))

    assert errors[0] == errors[1]
    assert errors[0].startswith("Could not add gsplats 'splats':")


@pytest.mark.parametrize("lod_keyword", ["additive_lod", "substitutive_lod"])
def test_array_lod_rejects_mismatched_partition_parent(tmp_path, lod_keyword) -> None:
    data = _data()
    lod_spec = (
        dict(n_lods=2)
        if lod_keyword == "additive_lod"
        else dict(compression_factor=2, levels=1)
    )

    errors = []
    for name, kwargs in (
        ("plain", {}),
        ("structured", {lod_keyword: lod_spec}),
    ):
        with LuxarZarrCompiler(tmp_path / f"{name}.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            parent = scene.add_partition_group(
                "wrap", display_type="points", max_elements=100
            )
            with pytest.raises(ValueError) as error:
                scene.add_gsplats(
                    "part_0",
                    data.centers,
                    data.amplitudes,
                    data.cholesky_factors,
                    parent=parent,
                    **kwargs,
                )
        errors.append(str(error.value))

    assert errors[0] == errors[1]
    assert errors[0].startswith(
        "Could not add gsplats 'part_0': Cannot add gsplats 'part_0' to a "
        "kind=partition group declared display_type='points'."
    )


@pytest.mark.parametrize(
    ("attr", "value", "specialized_parent"),
    [
        ("join", "miter", False),
        ("metalness", 0.5, False),
        ("layer_order", 2, True),
    ],
)
def test_nested_file_rejects_invalid_compositing_attrs(
    tmp_path, attr, value, specialized_parent
) -> None:
    source_path = tmp_path / "partitioned.gsplats.zarr"
    write_gsplats_tree(
        source_path,
        _data().to_spatial_partition(max_elements=4),
        ordering="none",
    )

    with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        parent = scene.add_lod_group("lod") if specialized_parent else None
        kwargs = {attr: value}
        if specialized_parent:
            kwargs["coverage_fraction"] = 0.0
        with pytest.raises(ValueError) as error:
            scene.add_gsplats_from_file("splats", source_path, parent=parent, **kwargs)

    assert str(error.value).startswith("Could not add gsplats 'splats':")


@pytest.mark.parametrize("source_kind", ["nested", "matrix_ladder"])
def test_structured_file_rejects_mismatched_partition_parent(
    tmp_path, source_kind
) -> None:
    source_path = tmp_path / f"{source_kind}.gsplats.zarr"
    if source_kind == "nested":
        write_gsplats_tree(
            source_path,
            _data().to_spatial_partition(max_elements=4),
            ordering="none",
        )
    else:
        make_additive_lod(_data(), n_lods=2).save(source_path, ordering="none")

    with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        parent = scene.add_partition_group(
            "wrap", display_type="points", max_elements=100
        )
        with pytest.raises(
            ValueError,
            match=(
                "Could not add gsplats 'part_0': Cannot add gsplats 'part_0' to a "
                "kind=partition group declared display_type='points'."
            ),
        ):
            scene.add_gsplats_from_file("part_0", source_path, parent=parent)


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


@pytest.mark.parametrize("lod_keyword", ["additive_lod", "substitutive_lod"])
def test_array_lod_keeps_flat_centers_shape_error(tmp_path, lod_keyword) -> None:
    data = _data()
    lod_spec = (
        dict(n_lods=2)
        if lod_keyword == "additive_lod"
        else dict(compression_factor=2, levels=1)
    )
    errors = []
    for name, kwargs in (
        ("plain", {}),
        ("structured", {lod_keyword: lod_spec}),
    ):
        with LuxarZarrCompiler(tmp_path / f"{name}.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError) as error:
                scene.add_gsplats(
                    "splats",
                    data.centers.ravel(),
                    data.amplitudes,
                    data.cholesky_factors,
                    **kwargs,
                )
        errors.append(str(error.value))

    assert errors[0] == errors[1]


@pytest.mark.parametrize(
    ("spec", "error_type", "message"),
    [
        (True, ValueError, "substitutive_lod=True"),
        (3, TypeError, r"substitutive_lod \(or lod_group alias\) must be"),
    ],
)
def test_array_substitutive_lod_errors_name_public_keyword(
    tmp_path, spec, error_type, message
) -> None:
    data = _data()
    with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(error_type, match=message):
            scene.add_gsplats(
                "splats",
                data.centers,
                data.amplitudes,
                data.cholesky_factors,
                substitutive_lod=spec,
            )


@pytest.mark.parametrize("lod_keyword", ["additive_lod", "substitutive_lod"])
@pytest.mark.parametrize("value", [0, 0.0, np.bool_(False)])
def test_array_lod_does_not_treat_numeric_zero_as_false(
    tmp_path, lod_keyword, value
) -> None:
    data = _data()
    with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(TypeError, match=lod_keyword):
            scene.add_gsplats(
                "splats",
                data.centers,
                data.amplitudes,
                data.cholesky_factors,
                **{lod_keyword: value},
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


def test_add_gsplats_from_file_flattens_substitutive_pyramid(tmp_path) -> None:
    from luxar.gsplats.lod.substitutive import make_substitutive_lod

    source = _data()
    source_path = tmp_path / "pyramid.gsplats.zarr"
    pyramid = make_substitutive_lod(
        source, compression_factor=2, levels=1, device="cpu"
    )
    write_gsplats_tree(source_path, pyramid.tree, ordering="none")
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
