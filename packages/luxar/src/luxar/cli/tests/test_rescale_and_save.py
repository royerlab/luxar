"""Tests for the ``rescale_and_save`` empty-tile write branch.

``rescale_and_save`` (``cli/gsplat_ops/fitting/fit_utils.py``) has a special
branch for an empty single tile fit under ``--allow-empty-tile``: the gsplats
writer enforces a no-empty policy, so instead of erroring the function drops a
``<output_path>.empty`` marker (containing ``"0 splats\n"``) and skips the
store write. The parallel orchestrator treats that marker as a legitimately
skipped tile at merge time. No other test calls ``rescale_and_save``, so this
file pins both the empty-marker branch and — as a guard — that a non-empty leaf
still saves a real store (proving the branch is conditional, not always-on).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.gsplat_ops.fitting.fit_utils import FitPipelineCtx, rescale_and_save
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.utils.trils import tril_size

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _empty_result(ndim: int) -> GSplatData:
    """A fresh 0-splat GSplatData leaf (replicates the tiled-fit test helper)."""
    return GSplatData(
        centers=np.zeros((0, ndim), dtype=np.float32),
        amplitudes=np.zeros((0,), dtype=np.float32),
        cholesky_factors=np.zeros((0, tril_size(ndim)), dtype=np.float32),
        stats={"time_seconds": 0.0},
    )


def _one_splat_result(ndim: int) -> GSplatData:
    """A minimal but valid 1-splat GSplatData leaf (identity Cholesky)."""
    chol = np.zeros((1, tril_size(ndim)), dtype=np.float32)
    k = 0  # packed lower-triangular row-major: set the diagonal to 1
    for i in range(ndim):
        for j in range(i + 1):
            if i == j:
                chol[0, k] = 1.0
            k += 1
    return GSplatData(
        centers=np.full((1, ndim), 1.0, dtype=np.float32),
        amplitudes=np.ones((1,), dtype=np.float32),
        cholesky_factors=chol,
        stats={"time_seconds": 0.0},
    )


def _tile_ctx(output_path: Path) -> FitPipelineCtx:
    """A minimal FitPipelineCtx exercising the empty-tile save branch.

    Only the fields ``rescale_and_save`` reads are given meaningful values;
    everything else is None/inert.
    """
    kwargs: dict = {f.name: None for f in dataclasses.fields(FitPipelineCtx)}
    kwargs.update(
        output_path=output_path,
        allow_empty_tile=True,
        tile="1/4",
        compress=None,
        verbose=False,
    )
    return FitPipelineCtx(**kwargs)


def test_empty_tile_writes_marker(tmp_path: Path) -> None:
    """An empty ``--allow-empty-tile`` tile drops a ``.empty`` marker, no store."""
    output_path = tmp_path / "tile.gsplats.zarr"
    ctx = _tile_ctx(output_path)
    empty = _empty_result(3)

    result, n_splats, is_leaf = rescale_and_save(ctx, empty, None)

    assert result is empty
    assert n_splats == 0
    assert is_leaf is True

    marker = Path(str(output_path) + ".empty")
    assert marker.exists()
    assert marker.read_text() == "0 splats\n"
    # The store itself must NOT be written for an empty tile.
    assert not output_path.exists()


def test_non_empty_leaf_saves_normally(tmp_path: Path) -> None:
    """Guard: a non-empty leaf saves a real store and writes no ``.empty`` marker.

    Proves the empty-marker branch is conditional on ``n_splats == 0``, not
    always taken under ``--allow-empty-tile``.
    """
    output_path = tmp_path / "tile.gsplats.zarr"
    ctx = _tile_ctx(output_path)
    one = _one_splat_result(3)

    result, n_splats, is_leaf = rescale_and_save(ctx, one, None)

    assert result is one
    assert n_splats == 1
    assert is_leaf is True

    marker = Path(str(output_path) + ".empty")
    assert not marker.exists()
    assert output_path.exists()


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_fit_tile_allow_empty_writes_marker(tmp_path: Path) -> None:
    """End-to-end: ``gsplat fit --tile 0/M --allow-empty-tile`` on an all-background
    origin tile drops the ``.empty`` marker via the real CLI.

    The hand-built-ctx tests above pin ``rescale_and_save`` itself; this pins the
    CLI seam — that ``--allow-empty-tile`` actually threads through the typer
    option into ``FitPipelineCtx`` (the argv/sbatch tests only string-match the
    flag on the generator side, so a rename that broke the real worker save path
    would slip past them). The origin tile is all zeros, so ``fit_tile`` windows
    it to near-zero and skips the fitter (no fit runs — fast).
    """
    vol = tmp_path / "vol.npy"
    v = np.zeros((48, 48), dtype=np.float32)
    yy, xx = np.ogrid[:48, :48]
    v += np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / 20.0).astype(
        np.float32
    )  # blob in tile 3
    v[:24, :24] = 0.0  # origin tile (index 0) is exactly background — no leaked tail
    np.save(vol, v)

    out = tmp_path / "tile.gsplats.zarr"
    result = CliRunner().invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tile",
            "0/4",
            "--tile-size",
            "24",
            "--overlap",
            "0",
            "--allow-empty-tile",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    marker = Path(str(out) + ".empty")
    assert marker.exists()
    assert not out.exists()
