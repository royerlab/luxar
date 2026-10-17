"""``gsplat convert`` opens fitted image volumes in the image convention.

A fit stores its center columns in array order (z, y, x). Convert names them,
maps them onto x, y, z-ordered scene dimensions through ``dim_order`` (carrying
the Cholesky factors with the centers), and authors an opening camera that faces
the XY plane with y pointing down — the Fiji/napari view — as a CAMERA choice,
leaving every coordinate as fitted. ``--dim-order`` and ``--view`` override.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from luxar._zarr_compat import consolidate as zc_consolidate
from luxar._zarr_compat import open_group as zc_open_group
from luxar.cli import app
from luxar.cli.gsplat_ops.scene_orientation import (
    parse_dim_order,
    resolve_convert_orientation,
)
from luxar.cli.tests._testing import normalized_cli_output
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import fit_axis_dimension_metadata
from luxar.io._compiler.gsplat_tree import read_gsplat_node

# Column order is (z, y, x): z spans 0..4, y 0..20, x 0..100, so a positional
# mapping is distinguishable from a remapped one by the range alone.
_SPANS = np.array([4.0, 20.0, 100.0], dtype=np.float32)
# Per-column sigma (z=1, y=2, x=3), so the Cholesky permutation is observable.
_CHOL_ZYX = np.array([1.0, 0.0, 2.0, 0.0, 0.0, 3.0], dtype=np.float32)


def _fit_like(
    path: Path,
    *,
    ndim: int = 3,
    stats: Optional[Dict[str, Any]] = None,
) -> Path:
    rng = np.random.default_rng(0)
    n = 12
    spans = _SPANS[-ndim:]
    centers = (rng.random((n, ndim)) * spans).astype(np.float32)
    centers[0] = 0.0
    centers[1] = spans
    chol = _CHOL_ZYX if ndim == 3 else np.array([2.0, 0.0, 3.0], dtype=np.float32)
    data = GSplatData(
        centers=centers,
        amplitudes=rng.random(n).astype(np.float32) + 0.1,
        cholesky_factors=np.tile(chol, (n, 1)),
        stats=stats if stats is not None else {"source_shape": [8] * ndim},
    )
    data.save(path)
    return path


def _convert(runner: CliRunner, src: Path, out: Path, *extra: str) -> None:
    result = runner.invoke(
        app, ["gsplat", "convert", str(src), str(out), "--no-center", *extra]
    )
    assert result.exit_code == 0, normalized_cli_output(result)


def _scene(out: Path) -> zarr.Group:
    return zarr.open_group(str(out), mode="r")


def _dim_names(out: Path) -> List[str]:
    return [d["name"] for d in _scene(out).attrs["scene_dimensions"]["dimensions"]]


def _camera(out: Path) -> Optional[Dict[str, Any]]:
    return (_scene(out).attrs.get("viewer_config") or {}).get("camera")


def _leaf(out: Path) -> Any:
    root = _scene(out)
    node = read_gsplat_node(root["gsplats"], root)
    while not hasattr(node, "additive_sublods"):
        node = node.children[0]
    return node.additive_sublods[0]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestFitStampsAxisNames:
    def test_single_content_box_keeps_explicit_axes_through_checkpoint(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        source = tmp_path / "volume.npy"
        np.save(
            source, np.random.default_rng(42).random((8, 12, 16)).astype(np.float32)
        )
        store = tmp_path / "fit.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(source),
                str(store),
                "--axes",
                "x,y,z",
                "--tiling",
                "content",
                "--k-star-ref",
                "200",
                "--n-features-ref",
                "30",
                "--seeds",
                "100",
                "--iters",
                "20",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        attrs = zarr.open_group(str(store), mode="r").attrs
        assert attrs["type"] == "gsplats"
        assert [entry["name"] for entry in attrs["dimension_metadata"]] == [
            "x",
            "y",
            "z",
        ]

        out = tmp_path / "scene.luxar.zarr"
        _convert(runner, store, out)
        assert _dim_names(out) == ["x", "y", "z"]
        assert _camera(out) is None
        np.testing.assert_allclose(
            np.ptp(np.asarray(_leaf(out).centers), axis=0),
            np.ptp(GSplatData.load(store).centers, axis=0),
            rtol=0.02,
        )

    @pytest.mark.parametrize(
        ("axes", "shape", "expected"),
        [
            ("x,y,z", (8, 12, 16), ["x", "y", "z"]),
            ("c,depth,width,height", (1, 8, 12, 16), ["z", "x", "y"]),
            ("x,y", (12, 16), ["x", "y"]),
        ],
    )
    def test_fit_axes_names_kept_columns_and_convert_uses_them(
        self,
        runner: CliRunner,
        tmp_path: Path,
        axes: str,
        shape: tuple[int, ...],
        expected: List[str],
    ) -> None:
        rng = np.random.default_rng(42)
        source = tmp_path / "volume.npy"
        np.save(source, rng.random(shape).astype(np.float32))
        store = tmp_path / "fit.gsplats.zarr"
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(source),
                str(store),
                "--axes",
                axes,
                "--tiling",
                "none",
                "--floor",
                "none",
                "--seeds",
                "100",
                "--iters",
                "20",
                "--device",
                "cpu",
            ],
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        fitted = zarr.open_group(str(store), mode="r")
        attrs = fitted.attrs
        assert [entry["name"] for entry in attrs["dimension_metadata"]] == expected
        assert len(fitted["fitting"].attrs["source_shape"]) == len(expected)

        out = tmp_path / "scene.luxar.zarr"
        _convert(runner, store, out)
        assert _dim_names(out) == sorted(expected)
        source_spans = np.ptp(GSplatData.load(store).centers, axis=0)
        scene_spans = np.ptp(np.asarray(_leaf(out).centers), axis=0)
        np.testing.assert_allclose(
            scene_spans,
            [source_spans[expected.index(name)] for name in sorted(expected)],
            rtol=0.02,
        )
        if expected == ["x", "y", "z"] or expected == ["x", "y"]:
            assert _camera(out) is None

    def test_fit_store_root_names_its_columns_in_array_order(
        self, tmp_path: Path
    ) -> None:
        store = _fit_like(tmp_path / "fit.gsplats.zarr")
        attrs = zarr.open_group(str(store), mode="r").attrs
        assert attrs["dimension_metadata"] == [
            {"name": "z"},
            {"name": "y"},
            {"name": "x"},
        ]

    def test_two_dimensional_fit_is_y_x(self, tmp_path: Path) -> None:
        store = _fit_like(tmp_path / "fit2d.gsplats.zarr", ndim=2)
        attrs = zarr.open_group(str(store), mode="r").attrs
        assert attrs["dimension_metadata"] == [{"name": "y"}, {"name": "x"}]

    def test_no_fit_provenance_no_stamp(self, tmp_path: Path) -> None:
        store = _fit_like(tmp_path / "plain.gsplats.zarr", stats={})
        assert "dimension_metadata" not in zarr.open_group(str(store)).attrs

    def test_explicit_dimension_metadata_wins(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(1)
        data = GSplatData(
            centers=rng.random((4, 3)).astype(np.float32),
            amplitudes=np.ones(4, dtype=np.float32),
            cholesky_factors=np.tile(_CHOL_ZYX, (4, 1)),
            stats={"source_shape": [4, 4, 4]},
        )
        custom = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        out = tmp_path / "custom.gsplats.zarr"
        data.save(out, root_attrs={"dimension_metadata": custom})
        assert zarr.open_group(str(out)).attrs["dimension_metadata"] == custom

    @pytest.mark.parametrize(
        ("info", "ndim", "expected"),
        [
            (None, 3, None),
            ({"source_shape": [4, 4, 4]}, None, None),
            # A stacked timelapse: 4 center columns, 3-D source grid.
            ({"source_shape": [4, 4, 4]}, 4, None),
            # No conventional names beyond 3-D.
            ({"source_shape": [2, 4, 4, 4]}, 4, None),
            ({"source_shape": [4, 4]}, 2, [{"name": "y"}, {"name": "x"}]),
        ],
    )
    def test_fit_axis_dimension_metadata_rules(
        self, info: Any, ndim: Any, expected: Any
    ) -> None:
        assert fit_axis_dimension_metadata(info, ndim) == expected


class TestConvertImageConvention:
    @pytest.mark.parametrize("extra", [(), ("--dim-order", "x,y,z")])
    def test_centered_ranges_enclose_written_float32_centers(
        self, runner: CliRunner, tmp_path: Path, extra: tuple[str, ...]
    ) -> None:
        rng = np.random.default_rng(0)
        centers = rng.uniform(-25, 25, (12, 3)).astype(np.float32)
        data = GSplatData(
            centers=centers,
            amplitudes=(rng.random(12) + 0.1).astype(np.float32),
            cholesky_factors=np.tile(_CHOL_ZYX, (12, 1)),
            stats={"source_shape": [8, 8, 8]},
        )
        src = tmp_path / "fit.gsplats.zarr"
        data.save(src)
        out = tmp_path / "scene.luxar.zarr"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = runner.invoke(
                app, ["gsplat", "convert", str(src), str(out), *extra]
            )

        assert result.exit_code == 0, normalized_cli_output(result)
        assert not [
            warning
            for warning in caught
            if "outside declared range" in str(warning.message)
        ]
        written = np.asarray(_leaf(out).centers)
        for i, dim in enumerate(_scene(out).attrs["scene_dimensions"]["dimensions"]):
            assert dim["range"][0] <= written[:, i].min()
            assert written[:, i].max() <= dim["range"][1]

    def test_default_maps_zyx_columns_onto_xyz_and_faces_xy_y_down(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        src = _fit_like(tmp_path / "fit.gsplats.zarr")
        out = tmp_path / "scene.luxar.zarr"
        _convert(runner, src, out)

        assert _dim_names(out) == ["x", "y", "z"]
        assert _camera(out) == {"up": [0.0, -1.0, 0.0], "view_axis": "xy"}
        # Each scene dim carries its own column's range: x is the 0..100 column.
        dims = _scene(out).attrs["scene_dimensions"]["dimensions"]
        spans = [d["range"][1] - d["range"][0] for d in dims]
        np.testing.assert_allclose(spans, [100.0, 20.0, 4.0], rtol=1e-5)
        # The node's centers are remapped too, and the Cholesky factors with
        # them: sigma along scene x is the x column's 3, along z the z's 1.
        leaf = _leaf(out)
        centers = np.asarray(leaf.centers)
        np.testing.assert_allclose(np.ptp(centers, axis=0), [100, 20, 4], rtol=1e-3)
        chol = np.asarray(leaf.cholesky_factors)[0]
        np.testing.assert_allclose(chol[[0, 2, 5]], [3.0, 2.0, 1.0], rtol=1e-2)

    def test_dim_order_xyz_opts_out(self, runner: CliRunner, tmp_path: Path) -> None:
        src = _fit_like(tmp_path / "fit.gsplats.zarr")
        out = tmp_path / "scene.luxar.zarr"
        _convert(runner, src, out, "--dim-order", "x,y,z")

        assert _dim_names(out) == ["x", "y", "z"]
        assert _camera(out) is None
        dims = _scene(out).attrs["scene_dimensions"]["dimensions"]
        spans = [d["range"][1] - d["range"][0] for d in dims]
        np.testing.assert_allclose(spans, [4.0, 20.0, 100.0], rtol=1e-5)

    def test_view_xz(self, runner: CliRunner, tmp_path: Path) -> None:
        src = _fit_like(tmp_path / "fit.gsplats.zarr")
        out = tmp_path / "scene.luxar.zarr"
        _convert(runner, src, out, "--view", "xz")

        assert _dim_names(out) == ["x", "y", "z"]
        assert _camera(out) == {"up": [0.0, 0.0, -1.0], "view_axis": "xz"}

    def test_legacy_fit_without_names_assumes_array_order(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # A fit written before the axis-name stamp: fitting provenance, no names.
        src = _fit_like(tmp_path / "legacy.gsplats.zarr")
        root = zc_open_group(str(src), mode="r+")
        del root.attrs["dimension_metadata"]
        zc_consolidate(root)
        assert "dimension_metadata" not in zarr.open_group(str(src)).attrs
        out = tmp_path / "scene.luxar.zarr"
        _convert(runner, src, out)
        assert _dim_names(out) == ["x", "y", "z"]
        assert _camera(out) == {"up": [0.0, -1.0, 0.0], "view_axis": "xy"}

    def test_unnamed_store_without_fit_provenance_stays_positional(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        # Geometric x, y, z data written as .gsplats.zarr must not be relabelled.
        src = _fit_like(tmp_path / "geometric.gsplats.zarr", stats={})
        out = tmp_path / "scene.luxar.zarr"
        _convert(runner, src, out)
        assert _dim_names(out) == ["x", "y", "z"]
        assert _camera(out) is None
        dims = _scene(out).attrs["scene_dimensions"]["dimensions"]
        spans = [d["range"][1] - d["range"][0] for d in dims]
        np.testing.assert_allclose(spans, [4.0, 20.0, 100.0], rtol=1e-5)

    def test_classical_import_keeps_positional_xyz(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        src = _fit_like(
            tmp_path / "imported.gsplats.zarr",
            stats={"interop": {"format": "inria_ply"}},
        )
        out = tmp_path / "scene.luxar.zarr"
        _convert(runner, src, out)
        assert _dim_names(out) == ["x", "y", "z"]
        assert _camera(out) is None
        dims = _scene(out).attrs["scene_dimensions"]["dimensions"]
        spans = [d["range"][1] - d["range"][0] for d in dims]
        np.testing.assert_allclose(spans, [4.0, 20.0, 100.0], rtol=1e-5)

    def test_two_dimensional_fit(self, runner: CliRunner, tmp_path: Path) -> None:
        src = _fit_like(tmp_path / "fit2d.gsplats.zarr", ndim=2)
        out = tmp_path / "scene.luxar.zarr"
        _convert(runner, src, out)
        assert _dim_names(out) == ["x", "y"]
        assert _camera(out) == {"up": [0.0, -1.0, 0.0], "view_axis": "xy"}

    def test_partition_keeps_column_order_and_faces_xy_by_name(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        src = _fit_like(tmp_path / "fit.gsplats.zarr")
        part = tmp_path / "part.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "partition", str(src), str(part), "--parts", "2"]
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        out = tmp_path / "scene.luxar.zarr"
        _convert(runner, part, out)
        # A graft cannot be remapped: the dims stay in column order, named
        # honestly, and the camera's view_axis finds x and y by name.
        assert _dim_names(out) == ["z", "y", "x"]
        assert _camera(out) == {"up": [0.0, -1.0, 0.0], "view_axis": "xy"}

    @pytest.mark.parametrize(
        "extra",
        [["--dim-order", "z,y"], ["--dim-order", "z,z,x"], ["--view", "zy"]],
    )
    def test_bad_orientation_flags_are_usage_errors(
        self, runner: CliRunner, tmp_path: Path, extra: List[str]
    ) -> None:
        src = _fit_like(tmp_path / "fit.gsplats.zarr")
        out = tmp_path / "scene.luxar.zarr"
        result = runner.invoke(app, ["gsplat", "convert", str(src), str(out), *extra])
        assert result.exit_code != 0
        assert not out.exists()


class TestMergeKeepsAxisNames:
    def test_as_dimension_stamps_spatial_names_plus_the_new_axis(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        a = _fit_like(tmp_path / "t0.gsplats.zarr")
        b = _fit_like(tmp_path / "t1.gsplats.zarr")
        merged = tmp_path / "tl.gsplats.zarr"
        result = runner.invoke(
            app,
            ["gsplat", "merge", str(a), str(b), "-o", str(merged), "--as-dimension"]
            + ["--values", "0,1", "--dimension-name", "time"],
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        attrs = zarr.open_group(str(merged), mode="r").attrs
        assert [d["name"] for d in attrs["dimension_metadata"]] == [
            "z",
            "y",
            "x",
            "time",
        ]
        out = tmp_path / "scene.luxar.zarr"
        _convert(runner, merged, out)
        assert _dim_names(out) == ["x", "y", "z", "time"]
        dims = _scene(out).attrs["scene_dimensions"]["dimensions"]
        assert [d["display"] for d in dims] == [True, True, True, False]
        assert _camera(out) == {"up": [0.0, -1.0, 0.0], "view_axis": "xy"}

    def test_concatenate_keeps_agreed_names(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        a = _fit_like(tmp_path / "a.gsplats.zarr")
        b = _fit_like(tmp_path / "b.gsplats.zarr")
        merged = tmp_path / "ab.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "merge", str(a), str(b), "-o", str(merged)]
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        names = zarr.open_group(str(merged)).attrs["dimension_metadata"]
        assert [d["name"] for d in names] == ["z", "y", "x"]

    def test_disagreeing_inputs_are_left_unnamed(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        a = _fit_like(tmp_path / "a.gsplats.zarr")
        b = _fit_like(tmp_path / "b.gsplats.zarr", stats={})
        merged = tmp_path / "ab.gsplats.zarr"
        result = runner.invoke(
            app, ["gsplat", "merge", str(a), str(b), "-o", str(merged)]
        )
        assert result.exit_code == 0, normalized_cli_output(result)
        assert "dimension_metadata" not in zarr.open_group(str(merged)).attrs


class TestResolveConvertOrientation:
    def test_parse_dim_order(self) -> None:
        assert parse_dim_order(" z, y ,x", 3) == ["z", "y", "x"]
        assert parse_dim_order(None, 3) is None
        with pytest.raises(ValueError, match="names 2 columns"):
            parse_dim_order("z,y", 3)

    def test_timelapse_metadata_keeps_time_hidden_last(self) -> None:
        extent = np.array([[0, 0, 0, 0], [4, 20, 100, 9]], dtype=np.float32)
        metadata = [{"name": n} for n in ("z", "y", "x", "time")]
        orientation = resolve_convert_orientation(extent, metadata)
        assert [d.display for d in orientation.dimensions.dimensions] == [
            True,
            True,
            True,
            False,
        ]
        assert orientation.dimensions.names == ["x", "y", "z", "time"]
        assert orientation.dim_order == ["z", "y", "x", "time"]
        assert orientation.camera is not None
        assert orientation.camera.view_axis == "xy"
        assert orientation.camera.up == (0.0, -1.0, 0.0)

    @pytest.mark.parametrize("source", ["flag", "metadata"])
    def test_hidden_axis_first_is_hidden_by_name(self, source: str) -> None:
        # t first: positional display would draw t and hide x.
        extent = np.array([[0, 0, 0, 0], [9, 4, 20, 100]], dtype=np.float32)
        names = ["t", "z", "y", "x"]
        orientation = resolve_convert_orientation(
            extent,
            [{"name": n} for n in names] if source == "metadata" else None,
            dim_order=",".join(names) if source == "flag" else None,
        )
        dims = orientation.dimensions
        assert dims.names == ["x", "y", "z", "t"]
        assert [d.display for d in dims.dimensions] == [True, True, True, False]
        assert dims.dimensions[3].discrete
        assert dims.dimensions[3].step == 9.0
        assert orientation.dim_order == names
        assert orientation.camera is not None
        assert orientation.camera.up == (0.0, -1.0, 0.0)

    def test_colliding_view_axes_are_refused(self) -> None:
        # A graft named a,x,b: `x` is drawn along world Y and
        # the missing `y` falls back to world Y too — no plane to face.
        extent = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
        with pytest.raises(ValueError, match="same displayed axis"):
            resolve_convert_orientation(
                extent, dim_order="a,x,b", view="xy", remap=False
            )

    def test_array_order_default_needs_fit_provenance(self) -> None:
        extent = np.array([[0, 0, 0], [4, 20, 100]], dtype=np.float32)
        plain = resolve_convert_orientation(extent)
        assert plain.camera is None and plain.dim_order is None
        fitted = resolve_convert_orientation(
            extent, fitting_info={"source_shape": [5, 21, 101]}
        )
        assert fitted.dim_order == ["z", "y", "x"]

    def test_view_without_image_order_uses_up_positive(self) -> None:
        extent = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
        orientation = resolve_convert_orientation(extent, dim_order="x,y,z", view="yz")
        assert orientation.dim_order is None
        assert orientation.camera is not None
        assert orientation.camera.up == (0.0, 0.0, 1.0)

    def test_graft_xz_up_follows_the_z_dims_world_axis(self) -> None:
        extent = np.array([[0, 0, 0], [4, 20, 100]], dtype=np.float32)
        orientation = resolve_convert_orientation(
            extent, view="xz", remap=False, fitting_info={"source_shape": [4, 4, 4]}
        )
        assert orientation.dimensions.names == ["z", "y", "x"]
        # The z dim is drawn along world X, so "z down" is up = -X.
        assert orientation.camera is not None
        assert orientation.camera.up == (-1.0, 0.0, 0.0)
