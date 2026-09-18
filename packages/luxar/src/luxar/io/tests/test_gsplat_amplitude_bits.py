"""GSplat amplitude tiers stay scoped to gsplat writes."""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler


def _encoding_name(array: Any) -> str:
    return str(array.attrs["encoding"]["name"])


def test_gsplat_amplitude_bits_do_not_change_points_or_lines(tmp_path: Path) -> None:
    path = tmp_path / "scene.luxar.zarr"
    values = np.geomspace(1.0, 1000.0, 32).astype(np.float32)
    positions = np.column_stack(
        [np.linspace(0.0, 1.0, 32), np.zeros(32), np.zeros(32)]
    ).astype(np.float32)
    cholesky = np.tile(np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (32, 1))

    with LuxarZarrCompiler(path, gsplat_amplitude_bits=8) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("points", positions, radii=values)
        scene.add_lines("lines", positions, widths=values)
        scene.add_gsplats(
            "gsplats",
            centers=positions,
            amplitudes=values,
            cholesky_factors=cholesky,
        )

    root = zarr.open_group(str(path), mode="r")
    assert np.dtype(root["gsplats/amplitudes"].dtype) == np.dtype(np.uint8)
    assert _encoding_name(root["gsplats/amplitudes"]) == "geolog_scalar_uint8"
    assert _encoding_name(root["points/radii"]) != "geolog_scalar_uint8"
    assert _encoding_name(root["lines/widths"]) != "geolog_scalar_uint8"


@pytest.mark.parametrize("value", [0, 7, 9, "automatic"])
def test_gsplat_amplitude_bits_reject_invalid_values(
    tmp_path: Path, value: Any
) -> None:
    with pytest.raises(ValueError, match="gsplat_amplitude_bits"):
        LuxarZarrCompiler(tmp_path / "scene.luxar.zarr", gsplat_amplitude_bits=value)


def test_auto_resolves_each_gsplat_data_node_from_its_source_dtype(
    tmp_path: Path,
) -> None:
    from luxar.gsplats.gsplat_data import GSplatData

    positions = np.column_stack(
        [np.linspace(0.0, 1.0, 32), np.zeros(32), np.zeros(32)]
    ).astype(np.float32)
    amplitudes = np.geomspace(1e-3, 1.0, 32).astype(np.float32)
    cholesky = np.tile(np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (32, 1))

    def data(source_dtype: str) -> GSplatData:
        return GSplatData(
            centers=positions,
            amplitudes=amplitudes,
            cholesky_factors=cholesky,
            stats={"source_dtype": source_dtype},
        )

    path = tmp_path / "scene.luxar.zarr"
    with LuxarZarrCompiler(path, gsplat_amplitude_bits="auto") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats_from_data("u8", data("uint8"), normalize_amplitudes=False)
        scene.add_gsplats_from_data("f32", data("float32"), normalize_amplitudes=False)

    root = zarr.open_group(str(path), mode="r")
    assert _encoding_name(root["u8/amplitudes"]) == "geolog_scalar_uint8"
    assert np.dtype(root["f32/amplitudes"].dtype) == np.dtype(np.uint16)


@pytest.mark.parametrize("amplitude_bits", [8, 16])
def test_explicit_amplitude_bits_preserve_deduplication(
    tmp_path: Path, amplitude_bits: Literal[8, 16]
) -> None:
    positions = np.column_stack(
        [np.linspace(0.0, 1.0, 32), np.zeros(32), np.zeros(32)]
    ).astype(np.float32)
    amplitudes = np.geomspace(1e-3, 1.0, 32).astype(np.float32)
    cholesky = np.tile(np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (32, 1))

    path = tmp_path / "scene.luxar.zarr"
    with LuxarZarrCompiler(path, gsplat_amplitude_bits=amplitude_bits) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        for name in ("first", "second"):
            scene.add_gsplats(
                name,
                centers=positions,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
            )

    root = zarr.open_group(str(path), mode="r")
    encoding = root["second/amplitudes"].attrs["encoding"]
    assert encoding["name"] == "array_ref"
    assert encoding["target"] == "first/amplitudes"


def test_auto_resolves_gsplat_file_from_recorded_source_dtype(tmp_path: Path) -> None:
    from luxar.gsplats.gsplat_data import GSplatData

    positions = np.column_stack(
        [np.linspace(0.0, 1.0, 32), np.zeros(32), np.zeros(32)]
    ).astype(np.float32)
    source = tmp_path / "source.gsplats.zarr"
    GSplatData(
        centers=positions,
        amplitudes=np.geomspace(1e-3, 1.0, 32).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32), (32, 1)
        ),
        stats={"source_dtype": "uint8"},
    ).save(source, amplitude_bits="auto")

    path = tmp_path / "scene.luxar.zarr"
    with LuxarZarrCompiler(path, gsplat_amplitude_bits="auto") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats_from_file("u8", source, normalize_amplitudes=False)

    root = zarr.open_group(str(path), mode="r")
    assert _encoding_name(root["u8/amplitudes"]) == "geolog_scalar_uint8"


def test_optimize_keeps_source_matched_amplitude_encoding(tmp_path: Path) -> None:
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.io.optimize import optimize_store

    positions = np.column_stack(
        [np.linspace(0.0, 1.0, 64), np.zeros(64), np.zeros(64)]
    ).astype(np.float32)
    source = tmp_path / "source.luxar.zarr"
    with LuxarZarrCompiler(source, gsplat_amplitude_bits="auto") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats_from_data(
            "u8",
            GSplatData(
                centers=positions,
                amplitudes=np.geomspace(1e-3, 1.0, 64).astype(np.float32),
                cholesky_factors=np.tile(
                    np.array([1.0, 0, 1.0, 0, 0, 1.0], dtype=np.float32),
                    (64, 1),
                ),
                stats={"source_dtype": "uint8"},
            ),
            normalize_amplitudes=False,
        )

    optimized = tmp_path / "optimized.luxar.zarr"
    optimize_store(source, optimized, target_bytes=64, verify=True)

    before = zarr.open_group(str(source), mode="r")["u8/amplitudes"]
    after = zarr.open_group(str(optimized), mode="r")["u8/amplitudes"]
    assert _encoding_name(before) == "geolog_scalar_uint8"
    assert _encoding_name(after) == "geolog_scalar_uint8"
    assert np.dtype(after.dtype) == np.dtype(np.uint8)
