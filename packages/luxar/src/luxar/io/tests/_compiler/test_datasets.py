"""Direct unit tests for the dataset serializers in luxar.io._compiler.dataset_writers."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.encoding import ArrayDecoder, ArrayEncoder, EncodingMode
from luxar.io._compiler.context import DatasetCtx
from luxar.io._compiler.dataset_writers.colors import write_colors
from luxar.io._compiler.dataset_writers.positions import write_positions
from luxar.io._compiler.dataset_writers.scalars import (
    write_bounded_scalar,
    write_positive_scalar,
    write_radii,
    write_scalars,
    write_sharpness,
)
from luxar.io.reader import DEFAULT_COMP


def _ctx() -> DatasetCtx:
    return DatasetCtx(
        encoder=ArrayEncoder(float16_allowed=False),
        encoding_mode=EncodingMode.AUTO,
        compressor=DEFAULT_COMP,
    )


def _group() -> zarr.Group:
    return zarr.group()


def test_write_positions_roundtrip() -> None:
    g = _group()
    pos = np.random.rand(50, 3).astype(np.float32)
    write_positions(g, pos, None, _ctx())
    assert "positions" in g
    # AUTO positions are uint16 per-axis fixed-point — decode before comparing.
    assert g["positions"].attrs["encoding"]["name"] == "linear_perchannel_u16"
    decoded = ArrayDecoder().decode(g["positions"], g)
    np.testing.assert_allclose(
        decoded, pos, atol=float(np.ptp(pos, axis=0).max()) / 65535 * 2
    )


def test_write_colors_records_data_range() -> None:
    g = _group()
    colors = np.random.rand(40, 3).astype(np.float32)
    write_colors(g, colors, None, 40, _ctx())
    assert "colors" in g
    lo, hi = g.attrs["color_data_range"]
    assert lo == float(colors.min())
    assert hi == float(colors.max())


def test_write_positive_scalar_returns_max() -> None:
    g = _group()
    radii = np.array([0.1, 0.5, 0.3, 0.9], dtype=np.float32)
    out = write_positive_scalar(g, radii, "radii", None, 4, _ctx())
    assert out == pytest.approx(0.9, abs=1e-6)
    assert "radii" in g


def test_write_radii_and_sharpness_wrappers() -> None:
    g = _group()
    n = 16
    write_positions(g, np.random.rand(n, 3).astype(np.float32), None, _ctx())
    rmax = write_radii(g, np.full(n, 2.0, dtype=np.float32), None, n, _ctx())
    smax = write_sharpness(g, np.full(n, 1.5, dtype=np.float32), None, n, _ctx())
    assert rmax == 2.0
    assert smax == 1.5
    assert "radii" in g and "sharpnesses" in g


def test_write_bounded_scalar_clamps_into_bounds() -> None:
    g = _group()
    data = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    out = write_bounded_scalar(g, data, "sharpnesses", (0.0, 1.0), None, 3, _ctx())
    assert out == float(data.max())
    assert "sharpnesses" in g


def test_write_scalars_records_range_and_requires_position() -> None:
    g = _group()
    write_positions(g, np.random.rand(10, 3).astype(np.float32), None, _ctx())
    scalars = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    write_scalars(g, scalars, None, 10, _ctx())
    assert "scalars" in g
    lo, hi = g.attrs["scalar_data_range"]
    assert lo == 0.0 and hi == 1.0


def test_write_scalars_without_position_raises() -> None:
    g = _group()
    try:
        write_scalars(g, np.zeros(4, dtype=np.float32), None, 4, _ctx())
    except RuntimeError as exc:
        assert "No position data" in str(exc)
    else:  # pragma: no cover - guard
        raise AssertionError("expected RuntimeError when no position dataset present")
