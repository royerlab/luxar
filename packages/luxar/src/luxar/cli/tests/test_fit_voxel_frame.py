"""``fit`` / ``cal`` voxel spacing (``--voxel-size`` / ``--physical``) and ``--region``.

Covers the shared resolver in :mod:`luxar.cli.gsplat_ops.voxel_frame` (parsing,
precedence, the one-line report, the save-time offset and stamps) and the CLI end
to end on an OME-TIFF and an ImageJ TIFF written here with tifffile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.gsplat_ops.voxel_frame import (
    SOURCE_CONFIG,
    SOURCE_FLAG,
    SOURCE_NGFF,
    FitFrame,
    VoxelSpacing,
    apply_fit_frame,
    format_region,
    normalized_region,
    parse_region,
    parse_voxel_size,
    resolve_voxel_spacing,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.utils.trils import tril_size

runner = CliRunner()

#: Blob center (z, y, x) in FULL-volume voxel indices.
_BLOB = (8.0, 14.0, 18.0)
_SHAPE = (16, 32, 32)
_ANISO = (0.29, 0.108, 0.108)


def _blob_volume() -> np.ndarray:
    z, y, x = np.mgrid[: _SHAPE[0], : _SHAPE[1], : _SHAPE[2]].astype(np.float32)
    r2 = (z - _BLOB[0]) ** 2 / 4 + (y - _BLOB[1]) ** 2 / 6 + (x - _BLOB[2]) ** 2 / 6
    return (1000.0 * np.exp(-r2 / 2) + 5.0).astype(np.uint16)


def _ome_tiff(path: Path) -> Path:
    import tifffile

    tifffile.imwrite(
        path,
        _blob_volume(),
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeZ": _ANISO[0],
            "PhysicalSizeY": _ANISO[1],
            "PhysicalSizeX": _ANISO[2],
        },
    )
    return path


def _imagej_tiff(path: Path) -> Path:
    import tifffile

    tifffile.imwrite(
        path,
        _blob_volume(),
        imagej=True,
        resolution=(1 / 0.25, 1 / 0.25),
        metadata={"axes": "ZYX", "spacing": 1.5, "unit": "um"},
    )
    return path


def _fitting_attrs(store: Path) -> Dict[str, Any]:
    from luxar._zarr_compat import read_node_attrs

    return dict(read_node_attrs(store / "fitting"))


def _root_attrs(store: Path) -> Dict[str, Any]:
    from luxar._zarr_compat import read_node_attrs

    return dict(read_node_attrs(store))


def _centroid(store: Path) -> np.ndarray:
    data = GSplatData.load(store)
    weights = np.asarray(data.amplitudes, dtype=np.float64)
    return (data.centers * weights[:, None]).sum(0) / weights.sum()


def _fit(src: Path, out: Path, *extra: str) -> Any:
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(src),
            str(out),
            "--preset",
            "draft",
            "--iters",
            "80",
            "--seeds",
            "12",
            "--device",
            "cpu",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    return result


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_region_spans_open_sides_and_full_axes(self) -> None:
        region = parse_region("4:12,8:,:20")
        assert region == (slice(4, 12), slice(8, None), slice(None, 20))
        assert parse_region(":,,2:3") == (slice(None), slice(None), slice(2, 3))
        assert format_region(region) == "4:12,8:,:20"
        assert parse_region(None) is None

    @pytest.mark.parametrize("spec", ["1:2:3", "-4:8", "8:4", "a:b", "5"])
    def test_region_rejects_steps_negatives_empty_and_garbage(self, spec: str) -> None:
        with pytest.raises(typer.BadParameter):
            parse_region(spec)

    def test_voxel_size_parses_and_rejects_non_positive(self) -> None:
        assert parse_voxel_size("0.29, 0.108,0.108") == _ANISO
        for bad in ("1,0,1", "1,x,1", "1,-2,1", "nan,1,1"):
            with pytest.raises(typer.BadParameter):
                parse_voxel_size(bad)

    def test_normalized_region_uses_the_loaded_shape(self) -> None:
        region = parse_region("4:12,8:,:20")
        assert normalized_region(region, (8, 24, 20)) == ((4, 12), (8, 32), (0, 20))

    def test_normalized_region_refuses_a_squeezed_or_miscounted_crop(self) -> None:
        with pytest.raises(typer.BadParameter, match="names 3 axes"):
            normalized_region(parse_region("4:5,:,:"), (24, 20))
        with pytest.raises(typer.BadParameter, match="selects nothing"):
            normalized_region(parse_region("40:,:,:"), (0, 24, 20))


# ---------------------------------------------------------------------------
# Precedence + reporting
# ---------------------------------------------------------------------------


class TestResolveVoxelSpacing:
    def _resolve(self, tmp_path: Path, **kwargs: Any) -> Any:
        defaults: Dict[str, Any] = dict(
            input_path=tmp_path / "vol.npy",
            source_info={},
            ndim=3,
            explicit=None,
            physical=False,
            config={},
        )
        defaults.update(kwargs)
        return resolve_voxel_spacing(**defaults)

    def _tiff_info(self) -> Dict[str, Any]:
        return {
            "voxel_size": list(_ANISO),
            "voxel_unit": "um",
            "voxel_size_source": "ome",
        }

    def test_explicit_beats_config_and_metadata(self, tmp_path: Path, capsys) -> None:
        spacing = self._resolve(
            tmp_path,
            explicit=(3.0, 1.0, 1.0),
            physical=True,
            config={"voxel_size": [9.0, 9.0, 9.0]},
            source_info=self._tiff_info(),
        )
        assert spacing == VoxelSpacing((3.0, 1.0, 1.0), "um", SOURCE_FLAG)
        out = capsys.readouterr().out
        assert "3×1×1 µm (from --voxel-size; takes precedence" in out

    def test_config_beats_metadata(self, tmp_path: Path) -> None:
        spacing = self._resolve(
            tmp_path,
            physical=True,
            config={"voxel_size": 2.0},
            source_info=self._tiff_info(),
        )
        assert spacing == VoxelSpacing((2.0, 2.0, 2.0), None, SOURCE_CONFIG)

    def test_physical_reads_tiff_metadata(self, tmp_path: Path, capsys) -> None:
        spacing = self._resolve(tmp_path, physical=True, source_info=self._tiff_info())
        assert spacing is not None
        assert spacing.values == _ANISO and spacing.unit == "um"
        assert (
            "Voxel size Z,Y,X = 0.29×0.108×0.108 µm (from OME-TIFF metadata)"
            in capsys.readouterr().out
        )

    def test_metadata_is_inert_without_physical(self, tmp_path: Path, capsys) -> None:
        assert self._resolve(tmp_path, source_info=self._tiff_info()) is None
        assert capsys.readouterr().out == ""

    def test_hint_names_an_anisotropic_spacing(self, tmp_path: Path, capsys) -> None:
        assert self._resolve(tmp_path, source_info=self._tiff_info(), hint=True) is None
        out = capsys.readouterr().out
        assert "Hint: this stack has voxel size Z,Y,X = 0.29×0.108×0.108 µm" in out
        assert "pass --physical to fit in µm" in out

    def test_no_hint_for_isotropic_metadata(self, tmp_path: Path, capsys) -> None:
        info = {**self._tiff_info(), "voxel_size": [0.5, 0.5, 0.5]}
        self._resolve(tmp_path, source_info=info, hint=True)
        assert capsys.readouterr().out == ""

    def test_physical_without_metadata_warns(self, tmp_path: Path, capsys) -> None:
        assert self._resolve(tmp_path, physical=True) is None
        assert "⚠ --physical: no usable voxel size" in capsys.readouterr().out

    def test_physical_refuses_voxel_output_space(self, tmp_path: Path) -> None:
        with pytest.raises(typer.BadParameter, match="output_space: real"):
            self._resolve(tmp_path, physical=True, config={"output_space": "voxel"})

    def test_explicit_count_must_match_the_volume(self, tmp_path: Path) -> None:
        with pytest.raises(typer.BadParameter, match="gives 2 values") as info:
            self._resolve(tmp_path, explicit=(1.0, 1.0))
        assert "tile grid" not in str(info.value)

    def test_physical_on_an_unlabelled_4d_tiff_suggests_axes(
        self, tmp_path: Path, capsys
    ) -> None:
        import tifffile

        from luxar.io.volume import load_volume

        src = tmp_path / "tl.ome.tif"
        tifffile.imwrite(
            src,
            np.ones((2, 4, 8, 8), np.uint16),
            ome=True,
            metadata={
                "axes": "TZYX",
                "PhysicalSizeZ": 2.0,
                "PhysicalSizeY": 0.5,
                "PhysicalSizeX": 0.5,
            },
        )
        info: Dict[str, Any] = {}
        volume = load_volume(src, info=info)
        capsys.readouterr()
        assert (
            self._resolve(
                tmp_path,
                input_path=src,
                source_info=info,
                ndim=volume.ndim,
                physical=True,
            )
            is None
        )
        out = capsys.readouterr().out
        assert "4-D volume" in out and "--axes t,z,y,x" in out

    def test_physical_reads_ngff_scale(self, tmp_path: Path) -> None:
        from luxar._zarr_compat import create_array, open_group

        store = tmp_path / "ngff.zarr"
        root = open_group(store, mode="w")
        create_array(root, "0", data=np.zeros((4, 8, 8), np.float32), compressor="auto")
        axes = [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"]
        transforms = [{"type": "scale", "scale": [2.0, 0.5, 0.5]}]
        root.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": axes,
                    "datasets": [
                        {"path": "0", "coordinateTransformations": transforms}
                    ],
                }
            ],
        }
        spacing = self._resolve(tmp_path, input_path=store, physical=True)
        assert spacing == VoxelSpacing((2.0, 0.5, 0.5), "um", SOURCE_NGFF)


# ---------------------------------------------------------------------------
# Save-time frame
# ---------------------------------------------------------------------------


def _leaf(centers: np.ndarray) -> GSplatData:
    n, d = centers.shape
    chol = np.zeros((n, tril_size(d)), np.float32)
    diag = [i * (i + 1) // 2 + i for i in range(d)]
    chol[:, diag] = 1.0
    return GSplatData(
        centers=centers.astype(np.float32),
        amplitudes=np.ones(n, np.float32),
        cholesky_factors=chol,
        stats={"source_shape": [8, 24, 20]},
    )


class TestFitFrame:
    def test_offset_is_scaled_only_for_physical_output(self) -> None:
        spacing = VoxelSpacing((2.0, 0.5, 0.5), "um", SOURCE_FLAG)
        region = ((4, 12), (8, 32), (0, 20))
        assert np.allclose(FitFrame(spacing, True, region).offset(), [8.0, 4.0, 0.0])
        assert np.allclose(FitFrame(spacing, False, region).offset(), [4, 8, 0])
        assert FitFrame(spacing, True, ((0, 8), (0, 8), (0, 8))).offset() is None

    def test_leaf_is_translated_and_stamped(self) -> None:
        frame = FitFrame(
            VoxelSpacing((2.0, 1.0, 1.0), "um", SOURCE_FLAG),
            True,
            ((4, 12), (8, 32), (0, 20)),
        )
        moved = apply_fit_frame(_leaf(np.array([[1.0, 2.0, 3.0]])), frame)
        assert np.allclose(moved.centers, [[9.0, 10.0, 3.0]])
        assert moved.stats["source_shape"] == [8, 24, 20]  # survives the move
        assert moved.stats["source_region"] == [[4, 12], [8, 32], [0, 20]]
        assert moved.stats["voxel_size"] == [2.0, 1.0, 1.0]
        assert moved.stats["physical_coordinates"] is True
        assert moved.stats["voxel_unit"] == "um"

    def test_partition_split_planes_move_with_the_parts(self) -> None:
        from luxar.gsplats.tree import GSplatPartition, iter_default_leaves

        parts = [
            _leaf(np.array([[1.0, 1.0, 1.0]])).tree,
            _leaf(np.array([[1.0, 1.0, 6.0]])).tree,
        ]
        node = GSplatPartition(
            children=parts,
            bsp_tree={
                "axis": 2,
                "split": 4.0,
                "left": {"part": 0},
                "right": {"part": 1},
            },
        )
        frame = FitFrame(None, False, ((0, 8), (0, 8), (10, 30)))
        moved = apply_fit_frame(node, frame)
        xs = [
            float(GSplatData.from_tree(leaf).centers[0, 2])
            for leaf in iter_default_leaves(moved)
        ]
        assert xs == [11.0, 16.0]
        assert moved.bsp_tree["split"] == 14.0
        assert moved.meta["fit_stats"]["source_region"] == [[0, 8], [0, 8], [10, 30]]


# ---------------------------------------------------------------------------
# CLI end to end
# ---------------------------------------------------------------------------


class TestFitCli:
    @pytest.mark.parametrize(
        ("ngff_unit", "stored_unit", "scene_unit"),
        [("nanometer", "nm", "nm"), ("angstrom", "angstrom", "units")],
    )
    def test_physical_ngff_with_axes_converts_with_unit(
        self, tmp_path: Path, ngff_unit: str, stored_unit: str, scene_unit: str
    ) -> None:
        import zarr

        src = tmp_path / "vol.zarr"
        root = zarr.open_group(str(src), mode="w", zarr_format=2)
        root.create_array("0", data=_blob_volume(), chunks=_SHAPE)
        root.attrs["multiscales"] = [
            {
                "axes": [
                    {"name": name, "type": "space", "unit": ngff_unit} for name in "zyx"
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [290, 108, 108]}
                        ],
                    }
                ],
            }
        ]
        fitted = tmp_path / "physical.gsplats.zarr"
        _fit(src, fitted, "--axes", "z,y,x", "--physical")
        assert _fitting_attrs(fitted)["voxel_unit"] == stored_unit
        assert _root_attrs(fitted)["dimension_metadata"] == [
            {"name": name, **({"unit": scene_unit} if scene_unit != "units" else {})}
            for name in "zyx"
        ]
        scene = tmp_path / "scene.luxar.zarr"
        result = runner.invoke(app, ["gsplat", "convert", str(fitted), str(scene)])
        assert result.exit_code == 0, result.output
        assert [
            d["unit"] for d in _root_attrs(scene)["scene_dimensions"]["dimensions"]
        ] == [scene_unit] * 3

    def test_physical_ome_tiff_stamps_and_scales(self, tmp_path: Path) -> None:
        src = _ome_tiff(tmp_path / "vol.ome.tif")
        voxel = tmp_path / "voxel.gsplats.zarr"
        physical = tmp_path / "physical.gsplats.zarr"
        hinted = _fit(src, voxel)
        assert "pass --physical" in hinted.output
        result = _fit(src, physical, "--physical")
        assert "from OME-TIFF metadata" in result.output

        attrs = _fitting_attrs(physical)
        assert attrs["voxel_size"] == pytest.approx(list(_ANISO))
        assert attrs["voxel_unit"] == "um"
        assert attrs["voxel_size_source"] == "OME-TIFF metadata"
        assert attrs["physical_coordinates"] is True
        assert _fitting_attrs(voxel)["physical_coordinates"] is False
        names = _root_attrs(physical)["dimension_metadata"]
        assert names == [{"name": a, "unit": "um"} for a in "zyx"]
        # Same fit, physical frame: centroids differ by exactly the spacing.
        assert np.allclose(
            _centroid(physical), _centroid(voxel) * np.asarray(_ANISO), atol=0.05
        )

    def test_physical_imagej_tiff(self, tmp_path: Path) -> None:
        src = _imagej_tiff(tmp_path / "vol.tif")
        out = tmp_path / "out.gsplats.zarr"
        _fit(src, out, "--physical")
        attrs = _fitting_attrs(out)
        assert attrs["voxel_size"] == pytest.approx([1.5, 0.25, 0.25])
        assert attrs["voxel_size_source"] == "ImageJ metadata"

    def test_explicit_voxel_size_wins_over_metadata(self, tmp_path: Path) -> None:
        src = _ome_tiff(tmp_path / "vol.ome.tif")
        out = tmp_path / "out.gsplats.zarr"
        _fit(src, out, "--physical", "--voxel-size", "3,1,1")
        attrs = _fitting_attrs(out)
        assert attrs["voxel_size"] == [3.0, 1.0, 1.0]
        assert attrs["voxel_size_source"] == "--voxel-size"

    def test_voxel_size_on_a_npy(self, tmp_path: Path) -> None:
        src = tmp_path / "vol.npy"
        np.save(src, _blob_volume())
        out = tmp_path / "out.gsplats.zarr"
        _fit(src, out, "--voxel-size", "2,1,1")
        assert _fitting_attrs(out)["voxel_size"] == [2.0, 1.0, 1.0]
        assert _centroid(out)[0] == pytest.approx(2 * _BLOB[0], abs=0.6)

    def test_region_places_the_crop_in_full_volume_coordinates(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "vol.npy"
        np.save(src, _blob_volume())
        out = tmp_path / "crop.gsplats.zarr"
        result = _fit(src, out, "--region", "2:14,6:,10:28")
        assert "Region: z 2:14, y 6:32, x 10:28" in result.output

        attrs = _fitting_attrs(out)
        assert attrs["source_region"] == [[2, 14], [6, 32], [10, 28]]
        assert attrs["source_shape"] == [12, 26, 18]
        # The blob is found where it is in the FULL volume, not the crop.
        assert np.allclose(_centroid(out), _BLOB, atol=0.6)

    def test_region_with_physical_offsets_in_physical_units(
        self, tmp_path: Path
    ) -> None:
        src = _ome_tiff(tmp_path / "vol.ome.tif")
        out = tmp_path / "crop.gsplats.zarr"
        _fit(src, out, "--physical", "--region", "2:14,6:,10:28")
        expected = np.asarray(_BLOB) * np.asarray(_ANISO)
        assert np.allclose(_centroid(out), expected, atol=0.6 * max(_ANISO))

    def test_region_and_tile_region_are_exclusive(self, tmp_path: Path) -> None:
        src = tmp_path / "vol.npy"
        np.save(src, _blob_volume())
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(src),
                str(tmp_path / "o.gsplats.zarr"),
                "--region",
                "0:8,:,:",
                "--tile",
                "0/1",
                "--tile-region",
                "0:8,0:32,0:32",
                "--tile-volume-shape",
                "16,32,32",
            ],
        )
        assert result.exit_code != 0
        assert "--tile" in result.output

    def test_region_is_refused_with_a_hand_run_tile(self, tmp_path: Path) -> None:
        src = tmp_path / "vol.npy"
        np.save(src, _blob_volume())
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(src),
                str(tmp_path / "o.gsplats.zarr"),
                "--region",
                "0:8,:,:",
                "--tile",
                "0/2",
            ],
        )
        assert result.exit_code == 2
        assert "cannot be combined with --tile" in result.output

    def test_too_many_region_spans_is_a_usage_error(self, tmp_path: Path) -> None:
        src = _ome_tiff(tmp_path / "vol.ome.tif")
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(src),
                str(tmp_path / "o.gsplats.zarr"),
                "--region",
                "0:8,:,:,:",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "names 4 axes but the volume has 3" in result.output

    def test_voxel_size_count_mismatch_is_plain(self, tmp_path: Path) -> None:
        src = tmp_path / "vol.npy"
        np.save(src, _blob_volume())
        result = runner.invoke(
            app,
            [
                "gsplat",
                "fit",
                str(src),
                str(tmp_path / "o.gsplats.zarr"),
                "--voxel-size",
                "1,1",
            ],
        )
        assert result.exit_code == 2, result.output
        assert "gives 2 values but the volume is 3-D" in result.output
        assert "tile grid" not in result.output

    def test_parallel_uniform_region_matches_sequential(self, tmp_path: Path) -> None:
        src = tmp_path / "vol.npy"
        np.save(src, _blob_volume())
        common = (
            "--region",
            "2:14,4:,:",
            "--tiling",
            "uniform",
            "--tile-size",
            "16",
            "--overlap",
            "4",
            "--voxel-size",
            "2,1,1",
        )
        seq, par = tmp_path / "seq.gsplats.zarr", tmp_path / "par.gsplats.zarr"
        _fit(src, seq, *common, "-j", "1")
        _fit(src, par, *common, "-j", "2")

        def centers(store: Path) -> np.ndarray:
            from luxar.gsplats.io.load_gsplats import load_gsplat_node
            from luxar.gsplats.tree import iter_default_leaves

            node, _ = load_gsplat_node(store, include_stats=False)
            stacked = np.concatenate(
                [
                    GSplatData.from_tree(leaf).centers
                    for leaf in iter_default_leaves(node)
                ]
            )
            return stacked[np.lexsort(stacked.T[::-1])]

        assert np.allclose(centers(seq), centers(par), atol=1e-4)
        assert _fitting_attrs(par)["source_region"] == [[2, 14], [4, 32], [0, 32]]

    def test_content_physical_converts_to_um_image_orientation(
        self, tmp_path: Path
    ) -> None:
        src = _ome_tiff(tmp_path / "vol.ome.tif")
        fitted = tmp_path / "content.gsplats.zarr"
        _fit(
            src,
            fitted,
            "--physical",
            "--tiling",
            "content",
            "--k-star-ref",
            "40",
            "--n-features-ref",
            "4",
            "--min-leaf",
            "8",
            "--max-leaf",
            "16",
            "--cell",
            "4",
            "-j",
            "1",
        )
        assert _fitting_attrs(fitted)["source_shape"] == list(_SHAPE)
        scene = tmp_path / "content.luxar.zarr"
        result = runner.invoke(app, ["gsplat", "convert", str(fitted), str(scene)])
        assert result.exit_code == 0, result.output
        attrs = _root_attrs(scene)
        dims = attrs["scene_dimensions"]["dimensions"]
        # The z,y,x columns are named (with their unit) and mapped onto x, y, z
        # scene dimensions in µm.
        assert [(d["name"], d["unit"]) for d in dims] == [
            ("x", "um"),
            ("y", "um"),
            ("z", "um"),
        ]
        assert _root_attrs(fitted)["dimension_metadata"] == [
            {"name": a, "unit": "um"} for a in "zyx"
        ]
        camera = attrs["viewer_config"]["camera"]
        assert camera["view_axis"] == "xy" and camera["up"] == [0.0, -1.0, 0.0]


class TestCalCli:
    def _cal(self, src: Path, out: Path, *extra: str, config: Path) -> Any:
        result = runner.invoke(
            app,
            [
                "gsplat",
                "cal",
                str(src),
                str(out),
                "--k-grid",
                "6,12",
                "--preset",
                "draft",
                "--config",
                str(config),
                "--device",
                "cpu",
                "--no-rd-model",
                "--quiet",
                *extra,
            ],
        )
        assert result.exit_code == 0, result.output
        return json.loads(out.read_text())

    @pytest.fixture
    def config(self, tmp_path: Path) -> Path:
        path = tmp_path / "fast.yaml"
        path.write_text("n_iters: 60\n")
        return path

    def test_physical_records_spacing_and_scores_on_the_grid(
        self, tmp_path: Path, config: Path
    ) -> None:
        src = _ome_tiff(tmp_path / "vol.ome.tif")
        plain = self._cal(src, tmp_path / "plain.json", config=config)
        physical = self._cal(src, tmp_path / "phys.json", "--physical", config=config)
        frame = physical["input_frame"]
        assert frame["voxel_size"] == pytest.approx(list(_ANISO))
        assert frame["physical_coordinates"] is False
        assert plain["input_frame"] is None
        # Fits are rendered on the voxel grid either way: a physical-unit fit
        # scored off-grid would collapse the reconstruction PSNR.
        assert min(physical["full_psnr_db"]) > 25.0

    def test_voxel_size_and_region(self, tmp_path: Path, config: Path) -> None:
        src = tmp_path / "vol.npy"
        np.save(src, _blob_volume())
        data = self._cal(
            src,
            tmp_path / "cal.json",
            "--voxel-size",
            "2,1,1",
            "--region",
            "2:14,:,:",
            config=config,
        )
        assert data["input_frame"]["voxel_size"] == [2.0, 1.0, 1.0]
        assert data["input_frame"]["source_region"] == [[2, 14], [0, 32], [0, 32]]
        assert data["volume_shape"] == [12, 32, 32]

    def test_region_excludes_auto_region(self, tmp_path: Path, config: Path) -> None:
        src = tmp_path / "vol.npy"
        np.save(src, _blob_volume())
        result = runner.invoke(
            app,
            [
                "gsplat",
                "cal",
                str(src),
                str(tmp_path / "c.json"),
                "--region",
                "0:8,:,:",
                "--auto-region",
            ],
        )
        assert result.exit_code == 2, result.output


def test_worker_argv_forwards_spacing_and_region() -> None:
    from luxar.gsplats.fit_tiled_parallel import build_worker_cmd
    from luxar.gsplats.planner.fit_planned_parallel import _default_worker_cmd_builder

    cmd = build_worker_cmd(
        ["luxar"],
        "in.npy",
        "out.zarr",
        0,
        4,
        16,
        4,
        voxel_size=(0.29, 0.1, 0.1),
        region="2:14,:,:",
    )
    assert float(cmd[cmd.index("--voxel-size") + 1].split(",")[0]) == 0.29
    assert cmd[cmd.index("--worker-region") + 1] == "2:14,:,:"
    assert "--region" not in cmd  # a hand-run --tile refuses the public flag
    builder = _default_worker_cmd_builder("in.npy", "plan.json", region="2:14,:,:")
    argv = builder(0, Path("box.zarr"))
    assert argv[argv.index("--worker-region") + 1] == "2:14,:,:"
