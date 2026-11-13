"""An opt-in tile amplitude floor bounds large tiled fits without dark seams."""

import numpy as np
import pytest

from luxar.gsplats import fit_tiled
from luxar.gsplats.gsplat_data import GSplatData


@pytest.mark.parametrize("partition", [False, True])
def test_tile_amplitude_floor_removes_negligible_splats_before_merge(
    partition: bool,
) -> None:
    y, x = np.mgrid[:24, :48]
    image = np.exp(-((y - 12) ** 2 / 50 + (x - 24) ** 2 / 450)).astype(np.float32)
    kwargs = dict(
        tile_size=32,
        overlap=8,
        seeds=128,
        n_iters=20,
        device="cpu",
        verbose=False,
        cull_retention=None,
        partition=partition,
    )

    original = fit_tiled(image, **kwargs)
    trimmed = fit_tiled(image, tile_amplitude_floor=1e-5, **kwargs)
    original_splats = (
        GSplatData.from_default_selection(original) if partition else original
    )
    trimmed_splats = (
        GSplatData.from_default_selection(trimmed) if partition else trimmed
    )
    stats = trimmed.meta["fit_stats"] if partition else trimmed.stats

    near_zero = int(np.count_nonzero(original_splats.amplitudes < 1e-5))
    assert near_zero > 0
    assert trimmed.n_splats == original.n_splats - near_zero
    assert np.all(trimmed_splats.amplitudes >= 1e-5)
    assert sum(stats["splats_per_tile"]) == trimmed.n_splats
    assert stats["tile_amplitude_floor"] == 1e-5
    assert stats["tile_amplitude_pruned"] == near_zero
    reconstruction = trimmed_splats.render_to_volume(shape=image.shape, device="cpu")
    seam_mse = np.mean((reconstruction[:, 24:32] - image[:, 24:32]) ** 2)
    neighbour_mse = np.mean((reconstruction[:, 16:24] - image[:, 16:24]) ** 2)
    assert seam_mse <= neighbour_mse


def test_tile_amplitude_floor_is_invariant_to_input_scale() -> None:
    y, x = np.mgrid[:24, :48]
    image = np.exp(-((y - 12) ** 2 / 50 + (x - 24) ** 2 / 450)).astype(np.float32)
    kwargs = dict(
        tile_size=32,
        overlap=8,
        seeds=128,
        n_iters=20,
        device="cpu",
        verbose=False,
        cull_retention=None,
    )

    pruned = []
    for scale in (1.0, 1000.0):
        source = image * scale
        original = fit_tiled(source, **kwargs)
        trimmed = fit_tiled(source, tile_amplitude_floor=1e-5, **kwargs)
        span = original.stats["image_max"] - original.stats["image_min"]
        expected = int(np.count_nonzero(original.amplitudes < 1e-5 * span))
        assert expected > 0
        assert trimmed.n_splats == original.n_splats - expected
        assert np.all(trimmed.amplitudes >= 1e-5 * span)
        assert trimmed.stats["tile_amplitude_pruned"] == expected
        pruned.append(expected)
    assert pruned[0] == pruned[1]


@pytest.mark.parametrize("floor", [-1.0, np.nan, np.inf])
def test_tile_amplitude_floor_rejects_invalid_values(floor: float) -> None:
    with pytest.raises(ValueError, match="tile_amplitude_floor"):
        fit_tiled(np.ones((4, 4), dtype=np.float32), tile_amplitude_floor=floor)


def _tile_store(path, stats: dict) -> None:
    """Save a tiny 2D tile store carrying the given fit stats."""
    n = 4
    chol = np.zeros((n, 3), dtype=np.float32)
    chol[:, 0] = 1.0
    chol[:, 2] = 1.0
    GSplatData(
        centers=np.random.default_rng(0).random((n, 2)).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
        stats=dict(stats),
    ).save(path)


def test_empty_tile_marker_round_trips_the_pruned_count(tmp_path) -> None:
    from luxar.gsplats.fit_tiled_gsplats import (
        read_empty_tile_marker_pruned,
        write_empty_tile_marker,
    )

    marker = tmp_path / "tile_0.gsplats.zarr.empty"
    write_empty_tile_marker(marker, 17)
    assert marker.read_text().splitlines()[0] == "0 splats"
    assert read_empty_tile_marker_pruned(marker) == 17

    write_empty_tile_marker(marker, None)
    assert marker.read_text() == "0 splats\n"
    assert read_empty_tile_marker_pruned(marker) is None

    for legacy in ("", "0 splats\n", "0 splats\ntile_amplitude_pruned=x\n"):
        marker.write_text(legacy)
        assert read_empty_tile_marker_pruned(marker) is None


def test_parallel_reload_sums_worker_counts_without_refloor(tmp_path) -> None:
    """The worker's stamped count is authoritative, markers included."""
    from luxar.gsplats.fit_tiled_gsplats import write_empty_tile_marker
    from luxar.gsplats.fit_tiled_parallel import _reload_tile_results

    paths = [tmp_path / f"tile_{i}.gsplats.zarr" for i in range(2)]
    # A stamped store with no `intensity_range`: re-flooring it would raise
    # and misreport the tile as corrupt.
    _tile_store(paths[0], {"tile_amplitude_pruned": 5})
    write_empty_tile_marker(tmp_path / "tile_1.gsplats.zarr.empty", 3)

    results, missing, corrupt, pruned = _reload_tile_results(paths, 2, 1e-5)
    assert (missing, corrupt) == ([], [])
    assert [r.n_splats for r in results] == [4, 0]
    assert pruned == 8


def test_parallel_reload_reports_unknown_count_for_legacy_tiles(tmp_path) -> None:
    """A tile saved before the count was stamped makes the total unknown."""
    from luxar.gsplats.fit_tiled_parallel import _reload_tile_results

    store = tmp_path / "tile_0.gsplats.zarr"
    _tile_store(store, {})
    _, _, corrupt, pruned = _reload_tile_results([store], 2, 1e-5)
    assert corrupt == []
    assert pruned is None

    marker_tile = tmp_path / "tile_1.gsplats.zarr"
    (tmp_path / "tile_1.gsplats.zarr.empty").write_text("0 splats\n")
    _, _, _, pruned = _reload_tile_results([marker_tile], 2, 1e-5)
    assert pruned is None

    # Without the floor nothing is counted at all.
    _, _, _, pruned = _reload_tile_results([store], 2, None)
    assert pruned is None
