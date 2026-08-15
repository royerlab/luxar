"""Tests for the ``rescale_and_save`` empty-tile write branch and its rescale.

``rescale_and_save`` (``cli/gsplat_ops/fitting/fit_utils.py``) has a special
branch for an empty single tile fit under ``--allow-empty-tile``: the gsplats
writer enforces a no-empty policy, so instead of erroring the function drops a
``<output_path>.empty`` marker (containing ``"0 splats\n"``) and skips the
store write. The parallel orchestrator treats that marker as a legitimately
skipped tile at merge time. No other test calls ``rescale_and_save``, so this
file pins both the empty-marker branch and — as a guard — that a non-empty leaf
still saves a real store (proving the branch is conditional, not always-on).

It also pins the ``--downscale`` rescale the function performs on the way to
that save: a downscaled tile worker's splats are scaled back to original
coordinates, and doing so must not cost the worker's fitted
``truncation_radius`` — the live #1624 defect — nor, by construction, an
additive ladder it might one day carry.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.gsplat_ops.fitting.fit_utils import FitPipelineCtx, rescale_and_save
from luxar.gsplats.fitting.downscale import rescale_centers, rescale_cholesky_packed
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData
from luxar.gsplats.utils.trils import tril_size
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS

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


def _one_splat_result(
    ndim: int, truncation_radius: float = DEFAULT_TRUNCATION_RADIUS
) -> GSplatData:
    """A minimal but valid 1-splat GSplatData leaf (identity Cholesky).

    ``truncation_radius`` is what a ``truncate:`` in the fit config puts here
    (``fitting/results.py`` passes ``config.truncate`` straight through).
    """
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
        truncation_radius=truncation_radius,
    )


def _diagonal_chol(n: int, ndim: int, scale: float) -> np.ndarray:
    """``n`` packed Cholesky factors with ``scale`` on the diagonal."""
    chol = np.zeros((n, tril_size(ndim)), dtype=np.float32)
    k = 0
    for i in range(ndim):
        for j in range(i + 1):
            if i == j:
                chol[:, k] = scale
            k += 1
    return chol


def _ladder_result(
    ndim: int = 3, counts: tuple[int, ...] = (2, 3), truncation_radius: float = 2.5
) -> GSplatData:
    """A 2-rung additive ladder, every rung distinguishable from its sibling.

    The shape a laddered leaf would have. Per-rung splat counts, centers, sigma,
    colors and stats all differ, so a collapse to the concatenated top-level
    arrays cannot be mistaken for a faithful round-trip. Note that no fitter
    produces this today — see ``test_downscale_rescale_keeps_the_additive_ladder``
    for what that test does and does not prove.
    """
    lods = [
        AdditiveSubLOD(
            centers=(
                np.arange(n * ndim, dtype=np.float32).reshape(n, ndim) + 10.0 * rung
            ),
            amplitudes=np.full((n,), 1.0 + rung, dtype=np.float32),
            cholesky_factors=_diagonal_chol(n, ndim, 1.0 + rung),
            colors=np.full((n, 3), 40 * (rung + 1), dtype=np.uint8),
            stats={"pass_index": rung, "psnr_db": 20.0 + rung},
            truncation_radius=truncation_radius,
        )
        for rung, n in enumerate(counts)
    ]
    return GSplatData.from_additive_sublods(lods, stats={"time_seconds": 1.25})


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


def test_downscale_rescale_keeps_a_non_default_truncation_radius(
    tmp_path: Path,
) -> None:
    """The LIVE #1624 symptom: a fitted ``truncate:`` survives the rescale, and is
    saved.

    ``truncate:`` is a documented YAML key (``gsplat fit --dump-config`` emits
    it) that lands on the result in ``fitting/results.py``, so this is reachable
    with one flag: ``fit --tile k/M --downscale 2 --config 'truncate: 3.5'``.

    FAILS pre-fix: the rescale rebuilt a plain ``GSplatData`` from the top-level
    arrays without passing the radius, so the store recorded the DEFAULT 2.75
    instead of 3.5 — on the plain non-progressive path too. Consequence beyond
    the wrong number: ``concatenate`` (``_data/composition.py``) refuses to
    compose non-empty tiles whose radii differ, so a downscaled tile could not
    merge with an un-downscaled sibling.
    """
    output_path = tmp_path / "tile.gsplats.zarr"
    src = _one_splat_result(3, truncation_radius=3.5)
    assert src.truncation_radius != DEFAULT_TRUNCATION_RADIUS  # the test has teeth

    result, n_splats, _ = rescale_and_save(_tile_ctx(output_path), src, (2, 2, 2))

    assert n_splats == 1
    assert result.truncation_radius == 3.5
    assert np.array_equal(result.centers, rescale_centers(src.centers, (2, 2, 2)))
    # On DISK too: that is the copy the merge reloads and cross-checks.
    assert GSplatData.load(output_path).truncation_radius == 3.5


def test_downscale_rescale_keeps_the_additive_ladder(tmp_path: Path) -> None:
    """Pins a by-construction guarantee: the rescale maps per sub-LOD (#1624).

    What this DOES prove: ``rescale_and_save`` preserves a multi-rung leaf's
    structure — rung counts, per-rung colors, stats and truncation radius — where
    the pre-fix rebuild from the CONCATENATED top-level arrays collapsed it to a
    single sub-LOD (measured: 2 rungs in, 1 out).

    What it does NOT prove: that any caller currently hits that. No fitter
    reachable from this line returns a ladder — ``fit_progressive_gaussian_splats``
    ends with ``return final_result.flattened()`` — so the ladder path is a
    guarantee held in reserve, not a fixed symptom. The reachable defect is the
    truncation-radius reset pinned by the test above; this one keeps the
    structure-preserving property from silently regressing if a fitter ever stops
    flattening.
    """
    output_path = tmp_path / "tile.gsplats.zarr"
    src = _ladder_result()
    factors = (2, 2, 2)

    result, n_splats, is_leaf = rescale_and_save(_tile_ctx(output_path), src, factors)

    assert is_leaf is True
    assert n_splats == src.n_splats == 5
    # The structure itself: both rungs, with their own counts.
    assert result.n_additive_sublods == 2
    assert [lod.n_splats for lod in result.additive_sublods] == [2, 3]
    # Per-rung geometry is exactly the two rescale helpers applied per rung.
    for out_lod, in_lod in zip(result.additive_sublods, src.additive_sublods):
        assert np.array_equal(out_lod.centers, rescale_centers(in_lod.centers, factors))
        assert np.array_equal(
            out_lod.cholesky_factors,
            rescale_cholesky_packed(in_lod.cholesky_factors, factors),
        )
        assert out_lod.centers.dtype == np.float32
        # Everything the rung carries alongside its geometry.
        assert np.array_equal(out_lod.amplitudes, in_lod.amplitudes)
        assert out_lod.colors is not None
        assert np.array_equal(out_lod.colors, in_lod.colors)
        assert out_lod.stats == in_lod.stats
        assert out_lod.truncation_radius == 2.5
    assert result.truncation_radius == 2.5
    # The caller reads this straight off the returned result (fit.py summary).
    assert result.stats["time_seconds"] == 1.25

    # And the ladder has to be on DISK, because that is what the parallel merge
    # reloads from each worker's store.
    assert GSplatData.load(output_path).n_additive_sublods == 2


def test_downscale_rescale_of_a_single_rung_leaf(tmp_path: Path) -> None:
    """No-regression control: a plain (unladdered) leaf still comes back rescaled.

    The pre-fix path handled exactly this shape correctly, so it must keep
    working — the fix must not have traded one collapse for another.
    """
    output_path = tmp_path / "tile.gsplats.zarr"
    one = _one_splat_result(3)

    result, n_splats, _ = rescale_and_save(_tile_ctx(output_path), one, (2, 3, 4))

    assert n_splats == 1
    assert result.n_additive_sublods == 1
    assert np.array_equal(result.centers, rescale_centers(one.centers, (2, 3, 4)))
    assert np.array_equal(
        result.cholesky_factors,
        rescale_cholesky_packed(one.cholesky_factors, (2, 3, 4)),
    )
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
