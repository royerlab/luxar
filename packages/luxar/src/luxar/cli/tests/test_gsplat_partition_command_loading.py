"""Partition-loading contracts for flat and read-only gsplat commands."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    n_splats = 6
    cholesky = np.zeros((n_splats, 6), dtype=np.float32)
    cholesky[:, [0, 2, 5]] = 0.5
    data = GSplatData(
        centers=np.array(
            [
                [2, 2, 2],
                [3, 2, 2],
                [4, 2, 2],
                [8, 8, 8],
                [9, 8, 8],
                [10, 8, 8],
            ],
            dtype=np.float32,
        ),
        amplitudes=np.linspace(0.2, 1.0, n_splats, dtype=np.float32),
        cholesky_factors=cholesky,
    )

    partition_path = tmp_path / "partition.gsplats.zarr"
    flat_path = tmp_path / "flat.gsplats.zarr"
    target_path = tmp_path / "target.npy"
    write_gsplats_tree(
        partition_path,
        data.to_spatial_partition(max_elements=n_splats // 2),
        fitting_info={
            "image_min": 600.0,
            "psnr_db": 33.3,
            "fitter_name": "test-fitter",
        },
    )
    data.save(flat_path)
    np.save(target_path, np.zeros((12, 12, 12), dtype=np.float32))
    return partition_path, flat_path, target_path


def test_render_partition_matches_flat_default_selection(tmp_path: Path) -> None:
    from luxar.gsplats.io import load_default_gsplats

    partition_path, _, _ = _write_inputs(tmp_path)
    output_path = tmp_path / "render.npy"

    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "render",
            str(partition_path),
            str(output_path),
            "--shape",
            "12,12,12",
            "--device",
            "cpu",
            "--truncate",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    actual = np.load(output_path)
    expected = load_default_gsplats(partition_path).render_to_volume(
        shape=(12, 12, 12),
        device="cpu",
        truncate=3,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert "Loaded 6 splats (3D)" in result.output


def test_render_flat_does_not_allocate_accumulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, flat_path, _ = _write_inputs(tmp_path)
    output_path = tmp_path / "render-flat.npy"
    original_zeros = np.zeros

    def reject_output_accumulator(
        shape: tuple[int, ...], *args: object, **kwargs: object
    ) -> np.ndarray:
        dtype = kwargs.get("dtype", args[0] if args else float)
        if tuple(shape) == (12, 12, 12) and np.dtype(dtype) == np.dtype(np.float32):
            raise AssertionError("flat render allocated an output accumulator")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", reject_output_accumulator)

    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "render",
            str(flat_path),
            str(output_path),
            "--shape",
            "12,12,12",
            "--device",
            "cpu",
            "--truncate",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert np.load(output_path).shape == (12, 12, 12)


def test_render_auto_shape_uses_default_level_bounds(tmp_path: Path) -> None:
    from luxar.gsplats.gsplat_data import GSplatData, SubstitutiveLevel

    def make_data(centers: np.ndarray) -> GSplatData:
        n_splats = len(centers)
        cholesky = np.zeros((n_splats, 6), dtype=np.float32)
        cholesky[:, [0, 2, 5]] = 0.5
        return GSplatData(
            centers=centers,
            amplitudes=np.ones(n_splats, dtype=np.float32),
            cholesky_factors=cholesky,
        )

    fine = make_data(np.array([[0, 0, 0], [3, 3, 3], [6, 6, 6]], dtype=np.float32))
    coarse = make_data(np.array([[18, 18, 18]], dtype=np.float32))
    levels = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(additive_sublods=fine.additive_sublods, level_index=0),
            SubstitutiveLevel(additive_sublods=coarse.additive_sublods, level_index=1),
        ]
    )
    input_path = tmp_path / "levels.gsplats.zarr"
    output_path = tmp_path / "render-levels.npy"
    levels.save(input_path)

    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "render",
            str(input_path),
            str(output_path),
            "--device",
            "cpu",
            "--truncate",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert np.load(output_path).shape == (7, 7, 7)
    assert "Auto shape from bounding box: (7, 7, 7)" in result.output


def test_default_partition_load_retains_requested_root_stats(tmp_path: Path) -> None:
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io import load_default_gsplats
    from luxar.gsplats.io.load_gsplats import load_gsplat_node

    partition_path, _, _ = _write_inputs(tmp_path)

    node, stats = load_gsplat_node(partition_path, include_stats=True)
    selected = GSplatData.from_default_selection(node, stats=stats)
    data = load_default_gsplats(partition_path, include_stats=True)

    assert selected.n_splats == 6
    assert selected.stats["image_min"] == 600.0
    assert data.n_splats == 6
    assert data.n_substitutive == 1
    assert data.n_additive_sublods == 1
    assert data.stats["image_min"] == 600.0


def test_flatten_keeps_root_stats_out_of_leaf_lod_stats(tmp_path: Path) -> None:
    import zarr

    partition_path, _, _ = _write_inputs(tmp_path)
    output_path = tmp_path / "flattened.gsplats.zarr"
    source = zarr.open_group(str(partition_path), mode="r")
    source_timestamp = source.attrs["timestamp"]

    result = CliRunner().invoke(
        app,
        ["gsplat", "flatten", str(partition_path), str(output_path)],
    )

    assert result.exit_code == 0, result.output
    output = zarr.open_group(str(output_path), mode="r")
    lod_stats = dict(output.attrs.get("lod_stats", {}))
    assert source_timestamp not in lod_stats.values()
    assert "timestamp" not in lod_stats
    assert output["fitting"].attrs["psnr_db"] == 33.3
    assert output["fitting"].attrs["fitter_name"] == "test-fitter"
    assert output["pipeline"].attrs["image_min"] == 600.0


def test_flatten_summarizes_partition_slot_provenance(tmp_path: Path) -> None:
    import zarr

    partition_path, _, _ = _write_inputs(tmp_path)
    output_path = tmp_path / "flattened.gsplats.zarr"
    source = zarr.open_group(str(partition_path), mode="a")
    source["fitting"].attrs["part_provenance"] = [
        {
            "coordinate": 0.0,
            "fit_reference": {"kind": "acquisition"},
            "fitting": {
                "source_shape": [10, 20, 30],
                "source_declared": True,
                "source_dtype": "uint16",
                "source_bytes": 3000,
                "source_stored_bytes": 2500,
                "source_voxels": 1500,
                "time_seconds": 2.5,
                "part_provenance": [
                    {"coordinate": 0.0, "fitting": {"source_bytes": 1500}}
                ],
            },
        },
        {
            "coordinate": 1.0,
            "fit_reference": {"kind": "acquisition"},
            "fitting": {
                "source_shape": [10, 20, 30],
                "source_declared": True,
                "source_dtype": "uint16",
                "source_bytes": 4000,
                "source_stored_bytes": 3500,
                "source_voxels": 2000,
                "time_seconds": 3.5,
                "part_provenance": [
                    {"coordinate": 1.0, "fitting": {"source_bytes": 2000}}
                ],
            },
        },
    ]
    assert "part_provenance" in source["fitting"].attrs

    result = CliRunner().invoke(
        app,
        ["gsplat", "flatten", str(partition_path), str(output_path)],
    )

    assert result.exit_code == 0, result.output
    output = zarr.open_group(str(output_path), mode="r")
    assert output["fitting"].attrs["part_provenance"] == [
        {
            "part_count": 2,
            "fit_reference": {"kind": "acquisition"},
            "fitting": {
                "source_shape": [10, 20, 30],
                "source_declared": True,
                "source_dtype": "uint16",
                "source_bytes": 7000,
                "source_stored_bytes": 6000,
                "source_voxels": 3500,
                "time_seconds": 6.0,
            },
        }
    ]


def test_flatten_carries_unanimous_partition_source_figures_once(
    tmp_path: Path,
) -> None:
    import zarr

    partition_path, _, _ = _write_inputs(tmp_path)
    output_path = tmp_path / "flattened.gsplats.zarr"
    source = zarr.open_group(str(partition_path), mode="a")
    source["fitting"].attrs["part_provenance"] = [
        {
            "coordinate": 0.0,
            "fit_reference": {"kind": "acquisition"},
            "fitting": {
                "source_shape": [100, 84, 580, 576],
                "source_declared": True,
                "source_dtype": "float32",
                "source_bytes": 11_225_088_000,
                "source_stored_bytes": 10_000_000_000,
                "source_voxels": 2_806_272_000,
                "time_seconds": 2.5,
            },
        },
        {
            "coordinate": 1.0,
            "fit_reference": {"kind": "acquisition"},
            "fitting": {
                "source_shape": [100, 84, 580, 576],
                "source_declared": True,
                "source_dtype": "float32",
                "source_bytes": 11_225_088_000,
                "source_stored_bytes": 10_000_000_000,
                "source_voxels": 2_806_272_000,
                "time_seconds": 3.5,
            },
        },
    ]

    result = CliRunner().invoke(
        app,
        ["gsplat", "flatten", str(partition_path), str(output_path)],
    )

    assert result.exit_code == 0, result.output
    output = zarr.open_group(str(output_path), mode="r")
    assert output["fitting"].attrs["part_provenance"] == [
        {
            "part_count": 2,
            "fit_reference": {"kind": "acquisition"},
            "fitting": {
                "source_shape": [100, 84, 580, 576],
                "source_declared": True,
                "source_dtype": "float32",
                "source_bytes": 11_225_088_000,
                "source_stored_bytes": 10_000_000_000,
                "source_voxels": 2_806_272_000,
                "time_seconds": 6.0,
            },
        }
    ]


def test_flatten_summarizes_single_part_partition_slot_provenance(
    tmp_path: Path,
) -> None:
    import zarr

    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.tree import GSplatPartition

    partition_path, _, _ = _write_inputs(tmp_path)
    node, _ = load_gsplat_node(partition_path)
    assert isinstance(node, GSplatPartition)

    single_part_path = tmp_path / "single-part.gsplats.zarr"
    provenance = [
        {
            "coordinate": 1.0,
            "fitting": {
                "source_shape": [10, 20, 30],
                "source_dtype": "uint16",
                "source_bytes": 3000,
                "source_voxels": 1500,
                "time_seconds": 2.5,
            },
        }
    ]
    write_gsplats_tree(
        single_part_path,
        GSplatPartition(children=[node.children[0]], max_elements=node.max_elements),
        fitting_info={"part_provenance": provenance},
    )
    output_path = tmp_path / "flattened.gsplats.zarr"

    result = CliRunner().invoke(
        app,
        ["gsplat", "flatten", str(single_part_path), str(output_path)],
    )

    assert result.exit_code == 0, result.output
    output = zarr.open_group(str(output_path), mode="r")
    assert output["fitting"].attrs["part_provenance"] == [
        {
            "part_count": 1,
            "fitting": {
                "source_shape": [10, 20, 30],
                "source_dtype": "uint16",
                "source_bytes": 3000,
                "source_voxels": 1500,
                "time_seconds": 2.5,
            },
        }
    ]


def test_flatten_summary_omits_partial_or_disputed_figures(tmp_path: Path) -> None:
    import zarr

    partition_path, _, _ = _write_inputs(tmp_path)
    output_path = tmp_path / "flattened.gsplats.zarr"
    source = zarr.open_group(str(partition_path), mode="a")
    source["fitting"].attrs["part_provenance"] = [
        {
            "coordinate": 0.0,
            "fitting": {
                "source_shape": [10, 20, 30],
                "source_declared": True,
                "source_dtype": "uint16",
                "source_bytes": 3000,
                "source_voxels": 1500,
                "time_seconds": 2.5,
            },
        },
        {
            "coordinate": 1.0,
            "fitting": {
                "source_shape": [5, 20, 30],
                "source_declared": True,
                "source_dtype": "float32",
                "source_voxels": 2000,
                "time_seconds": 3.5,
            },
        },
    ]

    result = CliRunner().invoke(
        app,
        ["gsplat", "flatten", str(partition_path), str(output_path)],
    )

    assert result.exit_code == 0, result.output
    output = zarr.open_group(str(output_path), mode="r")
    assert output["fitting"].attrs["part_provenance"] == [
        {
            "part_count": 2,
            "fitting": {
                "source_voxels": 3500,
                "time_seconds": 6.0,
            },
        }
    ]


def test_flatten_preserves_non_slot_provenance_on_partition(tmp_path: Path) -> None:
    import zarr

    partition_path, _, _ = _write_inputs(tmp_path)
    output_path = tmp_path / "flattened.gsplats.zarr"
    source = zarr.open_group(str(partition_path), mode="a")
    provenance = [
        {
            "coordinate": float(index),
            "fit_reference": {"kind": "acquisition"},
            "fitting": {
                "psnr_db": 40.0 + index,
                "time_seconds": 1.0 + index,
            },
        }
        for index in range(3)
    ]
    source["fitting"].attrs["part_provenance"] = provenance

    result = CliRunner().invoke(
        app,
        ["gsplat", "flatten", str(partition_path), str(output_path)],
    )

    assert result.exit_code == 0, result.output
    output = zarr.open_group(str(output_path), mode="r")
    assert output["fitting"].attrs["part_provenance"] == provenance


def test_flatten_preserves_single_leaf_part_provenance(tmp_path: Path) -> None:
    import zarr

    _, flat_path, _ = _write_inputs(tmp_path)
    output_path = tmp_path / "flattened.gsplats.zarr"
    source = zarr.open_group(str(flat_path), mode="a")
    provenance = [
        {"coordinate": 0.0, "fitting": {"source_bytes": 3000}},
        {"coordinate": 1.0, "fitting": {"source_bytes": 3000}},
    ]
    source.require_group("fitting").attrs["part_provenance"] = provenance

    result = CliRunner().invoke(
        app,
        ["gsplat", "flatten", str(flat_path), str(output_path)],
    )

    assert result.exit_code == 0, result.output
    output = zarr.open_group(str(output_path), mode="r")
    assert output["fitting"].attrs["part_provenance"] == provenance


@pytest.mark.parametrize(
    ("case", "command_name"),
    [
        ("cull", "cull"),
        ("cull-target", "cull"),
        ("filter", "filter"),
        ("slice", "slice"),
        ("decimate", "decimate"),
        ("merge", "merge"),
        ("partition", "partition"),
    ],
)
def test_matrix_rewrite_commands_reject_partition_without_traceback(
    tmp_path: Path,
    case: str,
    command_name: str,
) -> None:
    partition_path, flat_path, target_path = _write_inputs(tmp_path)
    output_path = tmp_path / f"{case}-output.gsplats.zarr"

    args = {
        "cull": [
            "gsplat",
            "cull",
            str(partition_path),
            str(output_path),
        ],
        "cull-target": [
            "gsplat",
            "cull",
            str(partition_path),
            str(output_path),
            "--target",
            str(target_path),
            "--device",
            "cpu",
        ],
        "filter": [
            "gsplat",
            "filter",
            str(partition_path),
            str(output_path),
            "--amplitude-min",
            "0",
        ],
        "slice": [
            "gsplat",
            "slice",
            str(partition_path),
            str(output_path),
            "0:12,0:12,0:12",
        ],
        "decimate": [
            "gsplat",
            "decimate",
            str(partition_path),
            str(output_path),
            "--target",
            "2",
            "--device",
            "cpu",
        ],
        "merge": [
            "gsplat",
            "merge",
            str(partition_path),
            str(flat_path),
            "--output",
            str(output_path),
        ],
        "partition": [
            "gsplat",
            "partition",
            str(partition_path),
            str(output_path),
            "--max-elements",
            "3",
        ],
    }[case]

    result = CliRunner().invoke(app, args)
    combined_output = result.output + (result.stderr or "")

    assert result.exit_code == 1
    assert f"{partition_path.name}: 'luxar gsplat {command_name}'" in combined_output
    assert "needs a flat (matrix-shaped) store" in combined_output
    assert (
        f"'luxar gsplat flatten {partition_path.name} flat.gsplats.zarr'"
        in combined_output
    )
    assert "not matrix-shaped" not in combined_output
    assert "Traceback" not in combined_output
    assert not output_path.exists()


def test_napari_partition_uses_default_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partition_path, _, _ = _write_inputs(tmp_path)
    added: dict[str, np.ndarray] = {}

    class Viewer:
        def __init__(self, *, title: str) -> None:
            assert title == f"GSplats: {partition_path.name}"

        def add_image(self, volume: np.ndarray, **_: object) -> None:
            added["volume"] = volume

        def add_points(self, centers: np.ndarray, **_: object) -> None:
            added["centers"] = centers

    monkeypatch.setitem(
        sys.modules,
        "napari",
        SimpleNamespace(Viewer=Viewer, run=lambda: None),
    )

    result = CliRunner().invoke(app, ["gsplat", "napari", str(partition_path)])
    combined_output = result.output + (result.stderr or "")

    assert result.exit_code == 0, combined_output
    assert added["centers"].shape == (6, 3)
    assert added["volume"].max() > 0
    assert "Loaded 6 splats (3D)" in combined_output
    assert "Traceback" not in combined_output
