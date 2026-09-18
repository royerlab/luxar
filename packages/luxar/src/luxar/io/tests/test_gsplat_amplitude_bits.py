"""GSplat amplitude tiers stay scoped to gsplat writes."""

from pathlib import Path
from typing import Any

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


@pytest.mark.parametrize("value", [0, 7, 9, "auto"])
def test_gsplat_amplitude_bits_reject_invalid_values(
    tmp_path: Path, value: Any
) -> None:
    with pytest.raises(ValueError, match="gsplat_amplitude_bits"):
        LuxarZarrCompiler(tmp_path / "scene.luxar.zarr", gsplat_amplitude_bits=value)
