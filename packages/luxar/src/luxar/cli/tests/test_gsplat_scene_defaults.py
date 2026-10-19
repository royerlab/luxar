"""A fitted store added to a scene with defaults is the scene ``gsplat convert`` builds.

``add_gsplats_from_file``'s fitted default (``fitted_defaults=None``) and
``fitted_scene_defaults`` both resolve through
:func:`luxar.gsplats.scene_defaults.resolve_fitted_scene`, the function convert
calls, so the two scenes agree attribute for attribute (and, with no centering,
byte for byte: the same ``content_hash``). A store that names no columns, an
explicit argument, a specialized parent group and ``fitted_defaults=False`` all
keep the plain adder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from luxar import Dimension, Dimensions, LuxarZarrCompiler, ViewerConfig
from luxar._zarr_compat import read_node_attrs
from luxar.cli import app
from luxar.cli.tests._testing import normalized_cli_output
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.scene_defaults import fitted_scene_defaults
from luxar.gsplats.tree import iter_leaves
from luxar.io._compiler.gsplat_tree import read_gsplat_node


def _fit(
    path: Path, *, source_dtype: str = "uint16", fitted: bool = True, n: int = 600
) -> Path:
    """A dense (z, y, x) store; ``fitted`` adds fitting provenance."""
    rng = np.random.default_rng(7)
    centers = np.column_stack(
        [rng.uniform(0, 40, n), rng.normal(10, 2, n), rng.normal(30, 2, n)]
    ).astype(np.float32)
    stats: Dict[str, Any] = {}
    if fitted:
        stats = {"source_shape": [40, 20, 40], "source_dtype": source_dtype}
    GSplatData(
        centers=centers,
        amplitudes=(rng.uniform(0.5, 1.0, n) * 200.0).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1.5, 0, 1.5, 0, 0, 1.5], dtype=np.float32), (n, 1)
        ),
        stats=stats,
    ).save(path)
    return path


def _partition(src: Path, dst: Path) -> Path:
    result = CliRunner().invoke(
        app, ["gsplat", "partition", str(src), str(dst), "--parts", "2"]
    )
    assert result.exit_code == 0, normalized_cli_output(result)
    return dst


def _convert(src: Path, out: Path, *extra: str) -> None:
    result = CliRunner().invoke(
        app, ["gsplat", "convert", str(src), str(out), "--no-center", *extra]
    )
    assert result.exit_code == 0, normalized_cli_output(result)


def _bare(
    src: Path,
    out: Path,
    *,
    viewer_config: Optional[ViewerConfig] = None,
    **kwargs: Any,
) -> None:
    """The documented notebook call: default dims, then the bare adder."""
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(
            dimensions=Dimensions.default_3d(), viewer_config=viewer_config
        )
        scene.add_gsplats_from_file("gsplats", src, **kwargs)


def _helper(src: Path, out: Path, **options: Any) -> None:
    defaults = fitted_scene_defaults(src, **options)
    with LuxarZarrCompiler(
        out, gsplat_amplitude_bits=defaults.amplitude_bits
    ) as compiler:
        scene = compiler.create_scene(
            dimensions=defaults.dimensions, viewer_config=defaults.viewer_config
        )
        scene.add_gsplats_from_file("gsplats", src, **defaults.attrs)


def _root(out: Path) -> Dict[str, Any]:
    return dict(zarr.open_group(str(out), mode="r").attrs)


def _centers(out: Path, name: str = "gsplats") -> np.ndarray:
    root = zarr.open_group(str(out), mode="r")
    node = read_gsplat_node(root[name], root)
    return np.concatenate(
        [
            np.asarray(s.centers)
            for leaf in iter_leaves(node)
            for s in leaf.additive_sublods
        ]
    )


def _assert_same_scene(a: Path, b: Path) -> None:
    root_a, root_b = _root(a), _root(b)
    assert root_a["scene_dimensions"] == root_b["scene_dimensions"]
    assert root_a.get("viewer_config") == root_b.get("viewer_config")
    assert read_node_attrs(a / "gsplats") == read_node_attrs(b / "gsplats")
    np.testing.assert_array_equal(np.sort(_centers(a), 0), np.sort(_centers(b), 0))
    assert root_a["content_hash"] == root_b["content_hash"]


@pytest.mark.parametrize("source_dtype", ["uint16", "uint8"])
def test_bare_adder_matches_convert(tmp_path: Path, source_dtype: str) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr", source_dtype=source_dtype)
    _convert(src, tmp_path / "convert.luxar.zarr")
    _bare(src, tmp_path / "bare.luxar.zarr")
    _assert_same_scene(tmp_path / "convert.luxar.zarr", tmp_path / "bare.luxar.zarr")

    # And it is the scene the tester expected: x, y, z dims, y-down XY camera,
    # an auto-exposed gain in data units, listed as a layer.
    root = _root(tmp_path / "bare.luxar.zarr")
    names = [d["name"] for d in root["scene_dimensions"]["dimensions"]]
    assert names == ["x", "y", "z"]
    camera = root["viewer_config"]["camera"]
    assert camera["view_axis"] == "xy"
    assert list(camera["up"]) == [0.0, -1.0, 0.0]
    attrs = read_node_attrs(tmp_path / "bare.luxar.zarr" / "gsplats")
    assert attrs["amplitude_normalization_mode"] == "projected"
    assert attrs["display_gain"] < 1.0 / 200.0
    assert attrs["layer"] is True


def test_uint8_source_needs_the_8_bit_tier(tmp_path: Path) -> None:
    """Without convert's tier the same scene hashes differently."""
    src = _fit(tmp_path / "fit.gsplats.zarr", source_dtype="uint8")
    _bare(src, tmp_path / "bare.luxar.zarr")
    defaults = fitted_scene_defaults(src)
    assert defaults.amplitude_bits == 8
    out = tmp_path / "untiered.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(
            dimensions=defaults.dimensions, viewer_config=defaults.viewer_config
        )
        scene.add_gsplats_from_file(
            "gsplats", src, fitted_defaults=False, **defaults.attrs
        )
    bare_hash = _root(tmp_path / "bare.luxar.zarr")["content_hash"]
    assert _root(out)["content_hash"] != bare_hash


def test_appearance_kwargs_match_convert_flags(tmp_path: Path) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr")
    _convert(
        src,
        tmp_path / "convert.luxar.zarr",
        "--colormap",
        "plasma",
        "--tone-mapping",
        "ACES",
    )
    _bare(
        src,
        tmp_path / "bare.luxar.zarr",
        viewer_config=ViewerConfig(tone_mapping="ACES"),
        colormap="plasma",
    )
    _assert_same_scene(tmp_path / "convert.luxar.zarr", tmp_path / "bare.luxar.zarr")


@pytest.mark.parametrize("source_dtype", ["uint16", "uint8"])
def test_helper_matches_convert(tmp_path: Path, source_dtype: str) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr", source_dtype=source_dtype)
    convert = tmp_path / "convert.luxar.zarr"
    _convert(src, convert, "--colormap", "plasma", "--view", "xz")
    _helper(src, tmp_path / "helper.luxar.zarr", colormap="plasma", view="xz")
    _assert_same_scene(convert, tmp_path / "helper.luxar.zarr")


def test_partition_store_is_grafted_without_dim_order(tmp_path: Path) -> None:
    part = _partition(_fit(tmp_path / "fit.gsplats.zarr"), tmp_path / "p.gsplats.zarr")
    assert "dim_order" not in fitted_scene_defaults(part).attrs
    convert = tmp_path / "convert.luxar.zarr"
    _convert(part, convert)
    _bare(part, tmp_path / "bare.luxar.zarr")
    _assert_same_scene(convert, tmp_path / "bare.luxar.zarr")
    _helper(part, tmp_path / "helper.luxar.zarr")
    _assert_same_scene(convert, tmp_path / "helper.luxar.zarr")


def test_unfitted_store_keeps_the_plain_adder(tmp_path: Path) -> None:
    src = _fit(tmp_path / "plain.gsplats.zarr", fitted=False)
    _bare(src, tmp_path / "auto.luxar.zarr")
    _bare(src, tmp_path / "off.luxar.zarr", fitted_defaults=False)
    assert _root(tmp_path / "auto.luxar.zarr") == _root(tmp_path / "off.luxar.zarr")
    assert "display_gain" not in read_node_attrs(tmp_path / "auto.luxar.zarr/gsplats")


def test_opt_out_is_the_old_adder(tmp_path: Path) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr")
    _bare(src, tmp_path / "off.luxar.zarr", fitted_defaults=False)
    root = _root(tmp_path / "off.luxar.zarr")
    assert root["scene_dimensions"] == Dimensions.default_3d().to_dict()
    assert "viewer_config" not in root
    attrs = read_node_attrs(tmp_path / "off.luxar.zarr" / "gsplats")
    assert "display_gain" not in attrs
    assert "amplitude_normalization_factor" in attrs
    assert not attrs.get("layer", False)


def test_explicit_normalization_keeps_the_old_exposure(tmp_path: Path) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr")
    out = tmp_path / "out.luxar.zarr"
    _bare(src, out, normalize_amplitudes=True, layer=False)
    attrs = read_node_attrs(out / "gsplats")
    assert "display_gain" not in attrs
    assert "amplitude_normalization_factor" in attrs
    assert attrs["layer"] is False
    # The layout still applies: those arguments say nothing about it.
    root = _root(out)
    assert [d["name"] for d in root["scene_dimensions"]["dimensions"]] == [
        "x",
        "y",
        "z",
    ]
    assert root["viewer_config"]["camera"]["view_axis"] == "xy"


def test_second_channel_reuses_the_layout(tmp_path: Path) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr")
    out = tmp_path / "two.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats_from_file("a", src)
        dims_after_first = scene.dimensions.to_dict()
        scene.add_gsplats_from_file("b", src, colormap="plasma")
        assert scene.dimensions.to_dict() == dims_after_first
    gain_a = read_node_attrs(out / "a")["display_gain"]
    assert read_node_attrs(out / "b")["display_gain"] == pytest.approx(gain_a, rel=0.5)
    np.testing.assert_array_equal(
        np.sort(_centers(out, "a"), 0), np.sort(_centers(out, "b"), 0)
    )


def test_a_scene_laid_out_by_the_caller_is_left_alone(tmp_path: Path) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr")
    dims = Dimensions(
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ]
    )
    out = tmp_path / "custom.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_gsplats_from_file("gsplats", src)
    root = _root(out)
    assert root["scene_dimensions"] == dims.to_dict()
    assert "viewer_config" not in root
    attrs = read_node_attrs(out / "gsplats")
    # Node-level exposure still applies, predicted for this scene...
    assert attrs["amplitude_normalization_mode"] == "projected"
    assert attrs["layer"] is True
    # ...but the columns keep their positional mapping: nothing moves.
    _assert_positional(out, "gsplats", src)


def _stored_columns(src: Path) -> np.ndarray:
    return GSplatData.load(src).centers.astype(np.float64)


def _assert_positional(out: Path, name: str, src: Path) -> None:
    np.testing.assert_allclose(
        np.sort(_centers(out, name).astype(np.float64), 0),
        np.sort(_stored_columns(src), 0),
        atol=1e-3,
    )


def test_fresh_scene_remaps_the_columns(tmp_path: Path) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr")
    _bare(src, tmp_path / "bare.luxar.zarr")
    expected = np.sort(_stored_columns(src)[:, ::-1], 0)  # z, y, x -> x, y, z
    np.testing.assert_allclose(
        np.sort(_centers(tmp_path / "bare.luxar.zarr").astype(np.float64), 0),
        expected,
        atol=1e-3,
    )


def test_default_scene_with_geometry_keeps_positional_centers(tmp_path: Path) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr")
    out = tmp_path / "mixed.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("pts", np.zeros((4, 3), dtype=np.float32))
        scene.add_gsplats_from_file("gsplats", src)
        assert scene.dimensions.to_dict() == Dimensions.default_3d().to_dict()
    assert "viewer_config" not in _root(out)
    _assert_positional(out, "gsplats", src)
    assert read_node_attrs(out / "gsplats")["amplitude_normalization_mode"] == (
        "projected"
    )


def test_an_explicit_transform_is_an_orientation_choice(tmp_path: Path) -> None:
    from luxar import transforms

    src = _fit(tmp_path / "fit.gsplats.zarr")
    out = tmp_path / "scaled.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats_from_file(
            "gsplats", src, transform=transforms.scale(4.0, 1.0, 1.0)
        )
        assert scene.dimensions.to_dict() == Dimensions.default_3d().to_dict()
    assert "viewer_config" not in _root(out)
    _assert_positional(out, "gsplats", src)


def test_specialized_parent_keeps_the_plain_adder(tmp_path: Path) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr")
    out = tmp_path / "lod.luxar.zarr"
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        group = scene.add_group("levels", kind="lod")
        group.add_gsplats_from_file("fine", src, coverage_fraction=0.5)
        assert scene.dimensions.to_dict() == Dimensions.default_3d().to_dict()
    assert "display_gain" not in read_node_attrs(out / "levels" / "fine")


def test_a_refused_add_leaves_the_scene_as_it_was(tmp_path: Path) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr")
    with LuxarZarrCompiler(tmp_path / "x.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError):
            scene.add_gsplats_from_file("gsplats", src, centers=np.zeros((2, 3)))
        assert scene.dimensions.to_dict() == Dimensions.default_3d().to_dict()
        assert scene.viewer_config is None


def test_empty_tree_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = _partition(_fit(tmp_path / "fit.gsplats.zarr"), tmp_path / "p.gsplats.zarr")
    monkeypatch.setattr("luxar.gsplats.tree.center_bounds", lambda node: None)
    with pytest.raises(ValueError, match="holds no splats"):
        fitted_scene_defaults(part)


def test_bad_view_raises_value_error(tmp_path: Path) -> None:
    src = _fit(tmp_path / "fit.gsplats.zarr")
    with pytest.raises(ValueError, match="--view"):
        fitted_scene_defaults(src, view="ab")


@pytest.mark.parametrize("fitted", [True, False])
def test_the_store_is_loaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fitted: bool
) -> None:
    import importlib

    loader = importlib.import_module("luxar.gsplats.io.load_gsplats")
    src = _fit(tmp_path / "fit.gsplats.zarr", fitted=fitted)
    real = loader.load_gsplat_node
    calls = []

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(loader, "load_gsplat_node", counting)
    _bare(src, tmp_path / "once.luxar.zarr")
    assert len(calls) == 1
