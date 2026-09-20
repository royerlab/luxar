"""``--seeds`` is a whole-volume budget that a tiled fit splits across tiles.

A default ``luxar gsplat cal`` reports a WHOLE-VOLUME K*, and the documented
pipeline is ``cal`` -> ``fit --seeds K*``. Before #1556 every tile of a fit was
handed the SAME ``--seeds K``, so the realized total was roughly ``K x n_tiles``
(a real report: ``--seeds 256000`` auto-tiled into 21 tiles produced 2,449,962
splats). :func:`split_seeds_across_tiles` now divides an integer budget across
the grid.

The unit tests pin the helper's arithmetic; the CLI tests pin the value actually
handed down on both tiled entry points, by patching only the fitting boundary
(``fit_tile`` / ``fit_tiled``) and recording its ``seeds=`` kwarg — the real
typer command, volume load, tiling resolution and grid math all run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.gsplat_ops.fitting.fit_utils import (
    announce_seed_split_lower_bound,
    split_seeds_across_tiles,
)
from luxar.cli.tests._testing import normalized_cli_output
from luxar.gsplats.batch.manifest import allocate_weighted_integer_seeds
from luxar.gsplats.fit_tiled_gsplats import count_nonempty_tiles
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.tiling import compute_tile_specs
from luxar.gsplats.utils.trils import tril_size

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

runner = CliRunner()

# A 48x48 volume tiled at tile_size=24 / overlap=4 gives a 3x3 grid.
_N_TILES = 9


def test_weighted_integer_seeds_preserve_budget_and_tie_order() -> None:
    assert allocate_weighted_integer_seeds(10, [1.0, 1.0, 1.0]) == (4, 3, 3)
    assert allocate_weighted_integer_seeds(100, [1.0, 3.0, 0.0, 0.0]) == (
        25,
        75,
        0,
        0,
    )
    assert allocate_weighted_integer_seeds(10, [1.0, 1.0, 98.0]) == (1, 1, 8)


def test_weighted_integer_seeds_keep_one_per_nonempty_tile_below_floor() -> None:
    assert allocate_weighted_integer_seeds(2, [1.0, 3.0, 2.0, 0.0]) == (1, 1, 1, 0)


# ---------------------------------------------------------------- unit tests


def test_int_budget_is_divided_across_tiles() -> None:
    """The reported regression: 256,000 over 21 tiles is ~12k per tile, not 256k."""
    assert split_seeds_across_tiles(256_000, 21) == 12_191  # ceil(256000 / 21)


def test_int_budget_rounds_up() -> None:
    """A non-divisible budget rounds UP rather than truncating the request."""
    assert split_seeds_across_tiles(100, 8) == 13  # 100 / 8 == 12.5 -> ceil 13


def test_single_tile_is_a_no_op() -> None:
    """One tile IS the whole volume — the budget passes through untouched."""
    assert split_seeds_across_tiles(256_000, 1) == 256_000


@pytest.mark.parametrize("n_tiles", [0, -1])
def test_non_positive_tile_count_is_a_no_op(n_tiles: int) -> None:
    """Defensive: a degenerate grid count must never divide (or divide by zero)."""
    assert split_seeds_across_tiles(5000, n_tiles) == 5000


def test_float_ratio_passes_through_unchanged() -> None:
    """A compression ratio is scale-free: it is a fraction of whatever voxels it
    is applied to, so per tile it already means the same density it means
    whole-volume. Dividing it by the tile count would be wrong."""
    assert split_seeds_across_tiles(0.02, 21) == 0.02


def test_auto_passes_through_unchanged() -> None:
    """``--seeds auto`` (None) is sized per tile by the fitter itself."""
    assert split_seeds_across_tiles(None, 21) is None


def test_positive_budget_smaller_than_tile_count_gives_one_per_tile() -> None:
    """A positive K < N still seeds every tile, so the total is N, not K.

    ``ceil`` gives this for free — the point of the test is the SEMANTICS (never
    zero seeds for a tile), which is what keeps the existing tiled CLI tests
    (``--seeds 10`` across up to 16 tiles) fitting anything at all. It is also
    the one case where the realized total *exceeds* the request.
    """
    assert split_seeds_across_tiles(10, 16) == 1
    assert split_seeds_across_tiles(1, 1000) == 1


@pytest.mark.parametrize("bad", [0, -5])
def test_non_positive_budget_passes_through_unchanged(bad: int) -> None:
    """An invalid K must reach the fitter intact so its own validation fires.

    ``fit_gaussian_splats`` rejects a non-positive int count with "seeds as int
    must be positive" (``gsplats/fitting/validation.py``). Rounding it up to 1
    here would swallow that error on tiled fits only — ``--seeds 0 --tiling none``
    would fail while ``--seeds 0 --tiling uniform`` silently fit one splat per
    tile. See ``test_cli_zero_seeds_still_rejected_when_tiled`` for the
    end-to-end half of this.
    """
    assert split_seeds_across_tiles(bad, 16) == bad


def test_split_prints_a_notice(capsys: pytest.CaptureFixture[str]) -> None:
    """The division is never silent — it names budget, per-tile count and grid."""
    split_seeds_across_tiles(256_000, 21)
    out = capsys.readouterr().out
    assert "256,000" in out and "12,191" in out and "21 non-empty tiles" in out


def test_single_nonempty_tile_prints_sparse_grid_notice(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A one-tile divisor is still visible when it came from a larger grid."""
    assert split_seeds_across_tiles(200, 1, grid_tiles=25) == 200
    out = capsys.readouterr().out
    assert "200 per tile across 1 non-empty tile (25 grid tiles)" in out


def test_no_notice_when_nothing_is_split(capsys: pytest.CaptureFixture[str]) -> None:
    """Guard: the pass-through cases print nothing (the notice is conditional)."""
    split_seeds_across_tiles(256_000, 1)
    split_seeds_across_tiles(0.02, 21)
    split_seeds_across_tiles(None, 21)
    assert capsys.readouterr().out == ""


def test_lower_bound_notice_does_not_claim_exact_split(
    capsys: pytest.CaptureFixture[str],
) -> None:
    announce_seed_split_lower_bound(200, 25)
    out = capsys.readouterr().out
    assert "at least 8 per non-empty tile" in out
    assert "exact non-empty count" in out


# ------------------------------------------------------------------ CLI seam


def _make_volume(path: Path) -> None:
    """A small 2D image with a few Gaussian blobs (mirrors the tiled CLI tests)."""
    v = np.zeros((48, 48), np.float32)
    yy, xx = np.ogrid[:48, :48]
    for cy, cx in [(12, 12), (36, 36), (12, 36), (36, 12)]:
        v += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 20.0).astype(np.float32)
    np.save(path, v)


def _make_sparse_volume(path: Path) -> None:
    """The issue's one-corner case: four windowed tiles contain all signal."""
    volume = np.zeros((96, 96), np.float32)
    volume[:24, :24] = 1.0
    np.save(path, volume)


def _make_single_tile_sparse_volume(path: Path) -> None:
    """A smaller corner signal survives only the first tile of a 25-tile grid."""
    volume = np.zeros((96, 96), np.float32)
    volume[:16, :16] = 1.0
    np.save(path, volume)


def _make_floor_sparse_volume(path: Path) -> None:
    """A pedestal fills the grid, but the resolved floor leaves one corner."""
    volume = np.full((96, 96), 2.0, np.float32)
    volume[:24, :24] = 10.0
    np.save(path, volume)


def _one_splat_result() -> GSplatData:
    """A minimal valid 2D leaf so the CLI's save + summary path completes."""
    chol = np.zeros((1, tril_size(2)), dtype=np.float32)
    chol[0, 0] = 1.0  # packed lower-triangular: [l00, l10, l11]
    chol[0, 2] = 1.0
    return GSplatData(
        centers=np.full((1, 2), 1.0, dtype=np.float32),
        amplitudes=np.ones((1,), dtype=np.float32),
        cholesky_factors=chol,
        stats={"time_seconds": 0.0},
    )


def test_grid_is_nine_tiles() -> None:
    """Pins the grid the two CLI tests below divide against (48/24/4 -> 3x3)."""
    assert len(compute_tile_specs((48, 48), 24, 4)) == _N_TILES


def test_sparse_corner_counts_only_tiles_that_will_fit() -> None:
    volume = np.zeros((96, 96), np.float32)
    volume[:24, :24] = 1.0
    specs = compute_tile_specs(volume.shape, 24, 4)

    assert len(specs) == 25
    assert count_nonempty_tiles(volume, specs, applied_floor=None) == 4


def test_resolved_floor_excludes_pedestal_only_tiles() -> None:
    volume = np.full((96, 96), 2.0, np.float32)
    volume[:24, :24] = 10.0
    specs = compute_tile_specs(volume.shape, 24, 4)

    assert count_nonempty_tiles(volume, specs, applied_floor=None) == len(specs)
    assert count_nonempty_tiles(volume, specs, applied_floor=3.0) == 4


def test_all_empty_scan_falls_back_to_grid_count() -> None:
    volume = np.zeros((48, 48), np.float32)
    specs = compute_tile_specs(volume.shape, 24, 4)

    assert count_nonempty_tiles(volume, specs, applied_floor=None) == len(specs)


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_sequential_tiled_hands_exact_seed_counts_and_folded_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-process path receives one exact count per folded-grid tile."""
    seen: dict[str, Any] = {}

    def _fake_fit_tiled(volume: Any, **kwargs: Any) -> GSplatData:
        seen.update(kwargs)
        return _one_splat_result()

    monkeypatch.setattr("luxar.gsplats.fit_tiled_gsplats.fit_tiled", _fake_fit_tiled)

    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    out = tmp_path / "seq.gsplats.zarr"
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "--flat",
            "--seeds",
            "90",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["fold_tile_slivers"] is True
    assert sum(seen["tile_seed_counts"]) == 90
    assert len(set(seen["tile_seed_counts"])) > 1


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_sequential_ratio_skips_seed_divisor_floor_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ratio needs no non-empty divisor, so the CLI must not resolve its floor."""
    seen: dict[str, Any] = {}

    def _fake_fit_tiled(volume: Any, **kwargs: Any) -> GSplatData:
        seen.update(kwargs)
        return _one_splat_result()

    def _unexpected_floor_probe(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ratio seeds must not trigger a divisor floor probe")

    monkeypatch.setattr("luxar.gsplats.fit_tiled_gsplats.fit_tiled", _fake_fit_tiled)
    monkeypatch.setattr(
        "luxar.gsplats.fitting.preprocessing.resolve_volume_floor_denoised",
        _unexpected_floor_probe,
    )

    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(tmp_path / "ratio.gsplats.zarr"),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "--flat",
            "--floor",
            "5",
            "--seeds",
            "0.02",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["seeds"] == 0.02


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_sequential_sparse_fit_splits_over_nonempty_tiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def _fake_fit_tiled(volume: Any, **kwargs: Any) -> GSplatData:
        seen.update(kwargs)
        return _one_splat_result()

    monkeypatch.setattr("luxar.gsplats.fit_tiled_gsplats.fit_tiled", _fake_fit_tiled)

    volume = tmp_path / "sparse.npy"
    _make_sparse_volume(volume)
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(volume),
            str(tmp_path / "sparse.gsplats.zarr"),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "--flat",
            "--floor",
            "none",
            "--seeds",
            "200",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    counts = seen["tile_seed_counts"]
    assert sum(counts) == 200
    assert sum(count > 0 for count in counts) == 4
    assert max(counts) > min(count for count in counts if count > 0)
    assert seen["fold_tile_slivers"] is True
    assert "occupancy-weighted across 4 non-empty tiles" in result.output


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_sequential_single_nonempty_tile_announces_sparse_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def _fake_fit_tiled(volume: Any, **kwargs: Any) -> GSplatData:
        seen.update(kwargs)
        return _one_splat_result()

    monkeypatch.setattr("luxar.gsplats.fit_tiled_gsplats.fit_tiled", _fake_fit_tiled)

    volume = tmp_path / "single-tile-sparse.npy"
    _make_single_tile_sparse_volume(volume)
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(volume),
            str(tmp_path / "single-tile-sparse.gsplats.zarr"),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "--flat",
            "--floor",
            "none",
            "--seeds",
            "200",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["seeds"] == 200
    assert sum(seen["tile_seed_counts"]) == 200
    assert sum(count > 0 for count in seen["tile_seed_counts"]) == 1
    assert "occupancy-weighted across 1 non-empty tiles" in result.output


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_sequential_divisor_reuses_resolved_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def _fake_fit_tiled(volume: Any, **kwargs: Any) -> GSplatData:
        seen.update(kwargs)
        return _one_splat_result()

    monkeypatch.setattr("luxar.gsplats.fit_tiled_gsplats.fit_tiled", _fake_fit_tiled)

    volume = tmp_path / "floor-sparse.npy"
    _make_floor_sparse_volume(volume)
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(volume),
            str(tmp_path / "floor-sparse.gsplats.zarr"),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "--flat",
            "--floor",
            "3",
            "--seeds",
            "200",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["seeds"] == 200
    assert sum(seen["tile_seed_counts"]) == 200
    assert sum(count > 0 for count in seen["tile_seed_counts"]) == 4
    assert seen["floor"] == 3.0
    assert seen["_floor_resolved"] is True
    assert seen["_norm_range_resolved"] is True
    assert "occupancy-weighted across 4 non-empty tiles" in result.output


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_single_tile_worker_splits_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fit --tile 3/16 --seeds 90`` fits ONE tile of the REAL grid, so it
    gets 10 — not 90, and not 6.

    The denominator is deliberately WRONG (16 against a 9-tile grid — the case
    the command already prints "using actual grid count" for). Splitting by the
    user's ``tile_total`` would give ``ceil(90/16) == 6``; only splitting by
    ``len(specs)`` gives 10. This is the path every parallel ``-j`` worker and
    every uniform-mode ``batch-fit`` array task re-enters, which is why the
    whole-volume count is forwarded to them verbatim.
    """
    seen: dict[str, Any] = {}

    def _fake_fit_tile(volume: Any, spec: Any, **kwargs: Any) -> GSplatData:
        seen.update(kwargs)
        return _one_splat_result()

    monkeypatch.setattr("luxar.gsplats.fit_tiled_gsplats.fit_tile", _fake_fit_tile)

    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    out = tmp_path / "tile3.gsplats.zarr"
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tile",
            "3/16",  # stale denominator; the real grid is _N_TILES == 9
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "--seeds",
            "90",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "using actual grid count" in result.output.lower()
    assert seen["seeds"] == 10  # ceil(90 / 9); a tile_total split would be 6


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_sparse_worker_uses_same_nonempty_divisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def _fake_fit_tile(volume: Any, spec: Any, **kwargs: Any) -> GSplatData:
        seen.update(kwargs)
        return _one_splat_result()

    monkeypatch.setattr("luxar.gsplats.fit_tiled_gsplats.fit_tile", _fake_fit_tile)

    volume = tmp_path / "sparse.npy"
    _make_sparse_volume(volume)
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(volume),
            str(tmp_path / "tile.gsplats.zarr"),
            "--tile",
            "0/25",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "--floor",
            "none",
            "--seeds",
            "200",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["seeds"] == 50
    assert "4 non-empty tiles (25 grid tiles)" in result.output


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_whole_volume_fit_keeps_full_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard: an untiled fit still gets the full ``--seeds`` (no stray division)."""
    seen: dict[str, Any] = {}

    def _fake_fit(volume: Any, **kwargs: Any) -> GSplatData:
        seen.update(kwargs)
        return _one_splat_result()

    # The command does `from luxar.gsplats import fit_gaussian_splats` at call
    # time, so the package-level name is the binding it will resolve.
    monkeypatch.setattr("luxar.gsplats.fit_gaussian_splats", _fake_fit)

    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    out = tmp_path / "whole.gsplats.zarr"
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "none",
            "--seeds",
            "90",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["seeds"] == 90


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_parallel_forwards_exact_counts_and_folded_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``-j`` parent computes once and hands each worker its exact share."""
    captured: dict[str, Any] = {}

    def _fake_parallel(**kwargs: Any) -> GSplatData:
        captured.update(kwargs)
        return _one_splat_result()

    monkeypatch.setattr(
        "luxar.gsplats.fit_tiled_parallel.fit_tiled_parallel", _fake_parallel
    )

    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    out = tmp_path / "par.gsplats.zarr"
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(out),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "-j",
            "2",
            "--flat",
            "--seeds",
            "90",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["num_tiles"] == _N_TILES

    commands = [
        captured["worker_cmd_builder"](i, _N_TILES, tmp_path / f"t{i}")
        for i in range(_N_TILES)
    ]
    counts = [int(cmd[cmd.index("--tile-seed-count") + 1]) for cmd in commands]
    assert sum(counts) == 90
    assert all("--fold-tile-slivers" in cmd for cmd in commands)
    assert "occupancy-weighted across 9 non-empty tiles" in result.output


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_parallel_downscale_announces_conservative_seed_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent must not scan full-resolution data with downscaled tile specs."""
    captured: dict[str, Any] = {}

    def _fake_parallel(**kwargs: Any) -> GSplatData:
        captured.update(kwargs)
        return _one_splat_result()

    monkeypatch.setattr(
        "luxar.gsplats.fit_tiled_parallel.fit_tiled_parallel", _fake_parallel
    )

    vol = tmp_path / "vol.npy"
    _make_volume(vol)
    result = runner.invoke(
        app,
        [
            "gsplat",
            "fit",
            str(vol),
            str(tmp_path / "downscaled.gsplats.zarr"),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "--downscale",
            "1,2",
            "-j",
            "2",
            "--flat",
            "--seeds",
            "90",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["num_tiles"] == 3
    commands = [
        captured["worker_cmd_builder"](i, 3, tmp_path / f"downscaled-{i}")
        for i in range(3)
    ]
    counts = [int(cmd[cmd.index("--tile-seed-count") + 1]) for cmd in commands]
    assert sum(counts) == 90
    assert all("--fold-tile-slivers" in cmd for cmd in commands)
    assert "occupancy-weighted across 3 non-empty tiles" in result.output


@pytest.mark.skipif(not HAS_TORCH, reason="the fit CLI imports the torch fitter")
def test_cli_zero_seeds_still_rejected_when_tiled(tmp_path: Path) -> None:
    """``--seeds 0`` must fail on a TILED fit exactly as it does whole-volume.

    Nothing is patched here — the real fitter's validation must be the thing
    that fires. If the helper floored a non-positive budget to 1 per tile, the
    tiled run would silently fit instead, so the two paths would disagree about
    what valid input is.
    """
    vol = tmp_path / "vol.npy"
    _make_volume(vol)

    whole = runner.invoke(
        app,
        # fmt: off
        [
            "gsplat",
            "fit",
            str(vol),
            str(tmp_path / "a.gsplats.zarr"),
            "--tiling",
            "none",
            "--seeds",
            "0",
            "--device",
            "cpu",
        ],
        # fmt: on
    )
    tiled = runner.invoke(
        app,
        # fmt: off
        [
            "gsplat",
            "fit",
            str(vol),
            str(tmp_path / "b.gsplats.zarr"),
            "--tiling",
            "uniform",
            "--tile-size",
            "24",
            "--overlap",
            "4",
            "--flat",
            "--seeds",
            "0",
            "--device",
            "cpu",
        ],
        # fmt: on
    )

    whole_output = normalized_cli_output(whole)
    tiled_output = normalized_cli_output(tiled)
    assert whole.exit_code != 0, whole_output
    assert tiled.exit_code != 0, tiled_output
    assert "seeds as int must be positive" in tiled_output
