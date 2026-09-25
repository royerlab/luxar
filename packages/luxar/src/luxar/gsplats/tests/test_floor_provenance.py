"""Background-floor provenance: one key name, one location, every writer path.

Issue #1175. ``docs/specs/GSPLATS_ZARR_FORMAT.md`` claims every fit persists its
normalization block (``floor`` / ``image_min`` / ``image_max`` /
``intensity_range``) into the store's ``pipeline/`` group. It used to be true of
the flat non-tiled fit only: a ``kind=partition`` merge stamped a differently
named ``applied_floor`` onto a node ``meta`` key the writer discarded, a content
fit and a progressive fit recorded nothing at all, and ``concatenate`` built a
fresh stats dict that dropped every tile's block on the way to the merge.

These tests pin the contract at the two ends that matter: what a producer puts
in memory, and what survives a save→load round trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.load_gsplats import load_gsplat_node, load_gsplats
from luxar.gsplats.io.save_gsplats import (
    NORMALIZATION_STATS_KEYS,
    agreed_normalization_stats,
    write_gsplats_tree,
)
from luxar.gsplats.utils.trils import tril_size

BLOCK = {
    "floor": 110.0,
    "floor_strategy": "specimen",
    "image_min": 110.0,
    "image_max": 4095.0,
    "intensity_range": 3985.0,
}


def test_specimen_preprocessing_records_selected_branch() -> None:
    pytest.importorskip("torch")
    from luxar.gsplats.fit_gsplats import GaussianSplatFitter
    from luxar.gsplats.fitting.config import FitParameters
    from luxar.gsplats.fitting.preprocessing import preprocess_data
    from luxar.gsplats.fitting.validation import prepare_fit_config

    rng = np.random.default_rng(1910)
    volume = (
        np.concatenate(
            [
                rng.normal(204.0, 5.0, 3_000),
                rng.normal(675.0, 25.0, 2_000),
                rng.normal(2500.0, 350.0, 300),
            ]
        )
        .astype(np.float32)
        .reshape(53, 10, 10)
    )

    config = prepare_fit_config(
        GaussianSplatFitter(),
        FitParameters(
            V=volume,
            seeds=np.array([[26.0, 5.0, 5.0]], dtype=np.float32),
            floor="specimen",
            verbose=False,
        ),
    )
    preprocessed = preprocess_data(config)

    assert preprocessed.floor == pytest.approx(675.0, abs=15.0)
    assert preprocessed.floor_strategy == "specimen"


def test_denoised_resolver_returns_strategy_from_resolving_sample() -> None:
    from luxar.gsplats.fitting.preprocessing import (
        resolve_volume_floor_denoised_with_strategy,
    )

    rng = np.random.default_rng(1910)
    volume = np.concatenate(
        [
            rng.normal(204.0, 5.0, 30_000),
            rng.normal(675.0, 25.0, 20_000),
            rng.normal(2500.0, 350.0, 3_000),
        ]
    ).astype(np.float32)

    level, strategy = resolve_volume_floor_denoised_with_strategy(
        volume, "specimen", guard_numeric=True
    )

    assert level == pytest.approx(675.0, abs=15.0)
    assert strategy == "specimen"


def _leaf(n: int = 6, ndim: int = 3, stats: Optional[dict] = None) -> GSplatData:
    """A tiny valid splat set (positive Cholesky diagonal)."""
    rng = np.random.default_rng(0)
    chol = np.zeros((n, tril_size(ndim)), dtype=np.float32)
    k = 0
    for i in range(ndim):
        for j in range(i + 1):
            if i == j:
                chol[:, k] = 1.0
            k += 1
    return GSplatData(
        centers=(rng.random((n, ndim)).astype(np.float32) * 10.0),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
        stats=dict(stats or {}),
    )


# ────────────────────────────────────────────────────────────────────────
# The agreement rule
# ────────────────────────────────────────────────────────────────────────


class TestAgreementRule:
    def test_unanimous_block_is_carried(self) -> None:
        assert agreed_normalization_stats([dict(BLOCK), dict(BLOCK)]) == BLOCK

    def test_disagreement_drops_only_the_disputed_key(self) -> None:
        other = {**BLOCK, "floor": 7.0}
        agreed = agreed_normalization_stats([dict(BLOCK), other])
        assert "floor" not in agreed
        assert agreed["image_max"] == 4095.0

    def test_silent_input_casts_no_vote(self) -> None:
        """An empty/skipped tile records no bounds; that must not erase them."""
        assert agreed_normalization_stats([dict(BLOCK), {"skipped": True}]) == BLOCK

    def test_unanimous_null_floor_is_carried_as_null(self) -> None:
        """``floor=None`` means "no pedestal removed" — information, not absence."""
        agreed = agreed_normalization_stats([{"floor": None}, {"floor": None}])
        assert agreed == {"floor": None}

    def test_nothing_recorded_yields_nothing(self) -> None:
        assert agreed_normalization_stats([{}, {"n_splats": 3}]) == {}


class TestConcatenate:
    def test_carries_the_block_its_inputs_agree_on(self) -> None:
        merged = GSplatData.concatenate([_leaf(stats=dict(BLOCK)) for _ in range(3)])
        for key in NORMALIZATION_STATS_KEYS:
            assert merged.stats[key] == BLOCK[key]

    def test_omits_a_floor_its_inputs_disagree_on(self) -> None:
        merged = GSplatData.concatenate(
            [_leaf(stats={"floor": 110.0}), _leaf(stats={"floor": 3.0})]
        )
        assert "floor" not in merged.stats


# ────────────────────────────────────────────────────────────────────────
# Save → load
# ────────────────────────────────────────────────────────────────────────


def _pipeline_attrs(path: Path) -> dict:
    """The store's ``pipeline/`` group attrs, read back off disk."""
    from luxar._zarr_compat import open_group

    root = open_group(str(path), mode="r")
    assert "pipeline" in root, "no pipeline/ group was written"
    return dict(root["pipeline"].attrs)


class TestRoundTrip:
    def test_flat_leaf(self, tmp_path: Path) -> None:
        path = tmp_path / "flat.gsplats.zarr"
        _leaf(stats=dict(BLOCK)).save(path)

        assert _pipeline_attrs(path)["floor"] == 110.0
        stats = load_gsplats(path, include_stats=True).stats
        for key in NORMALIZATION_STATS_KEYS:
            assert stats[key] == BLOCK[key]

    def test_partition(self, tmp_path: Path) -> None:
        """A partition has no flat stats dict; the block rides on the root meta."""
        node = GSplatData.partition_from_regions([_leaf(), _leaf()])
        node.meta.update(BLOCK)

        path = tmp_path / "part.gsplats.zarr"
        write_gsplats_tree(path, node)

        assert _pipeline_attrs(path)["floor"] == 110.0
        reloaded, stats = load_gsplat_node(path, include_stats=True)
        for key in NORMALIZATION_STATS_KEYS:
            assert stats[key] == BLOCK[key]
        # The new sibling group must not be mistaken for a part on read.
        assert len(reloaded.children) == 2

    def test_partition_null_floor_is_written_not_dropped(self, tmp_path: Path) -> None:
        node = GSplatData.partition_from_regions([_leaf(), _leaf()])
        node.meta["floor"] = None

        path = tmp_path / "part_none.gsplats.zarr"
        write_gsplats_tree(path, node)

        attrs = _pipeline_attrs(path)
        assert "floor" in attrs and attrs["floor"] is None

    def test_explicit_pipeline_info_wins_over_node_meta(self, tmp_path: Path) -> None:
        node = GSplatData.partition_from_regions([_leaf(), _leaf()])
        node.meta["floor"] = 110.0

        path = tmp_path / "explicit.gsplats.zarr"
        write_gsplats_tree(path, node, pipeline_info={"floor": 42.0})

        assert _pipeline_attrs(path)["floor"] == 42.0

    def test_a_real_fit_tile_result_survives_the_round_trip(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """The subprocess-merge tests below write their tile stores by hand, so
        they pin the merge but say nothing about the worker feeding it. This
        pins the other half: what the real ``fit --tile i/M`` door
        (:func:`fit_tile`) stamps is what a reload gets back."""
        pytest.importorskip("torch")
        import luxar.gsplats.fit_tiled_gsplats as ftg
        from luxar.gsplats.tiling import compute_tile_specs

        monkeypatch.setattr(ftg, "fit_gaussian_splats", _one_splat_stub())

        rng = np.random.RandomState(5)
        volume = rng.normal(100.0, 1.0, size=(96, 96)).astype(np.float32)
        volume[40:80, 40:80] += 150.0
        spec = compute_tile_specs(volume.shape, 48, 16)[0]

        path = tmp_path / "tile.gsplats.zarr"
        ftg.fit_tile(volume, spec, floor=100.0, verbose=False).save(path)

        stats = load_gsplats(path, include_stats=True).stats
        assert stats["floor"] == pytest.approx(100.0)
        # In the VOLUME's units, not the already-subtracted tile's.
        assert stats["image_min"] == pytest.approx(100.0)
        assert stats["image_max"] == pytest.approx(250.0)

    def test_a_plain_tree_still_writes_no_pipeline_group(self, tmp_path: Path) -> None:
        """The lift must not conjure a pipeline/ group onto every tree."""
        from luxar._zarr_compat import open_group

        path = tmp_path / "bare.gsplats.zarr"
        write_gsplats_tree(path, GSplatData.partition_from_regions([_leaf(), _leaf()]))
        assert "pipeline" not in open_group(str(path), mode="r")


# ────────────────────────────────────────────────────────────────────────
# Producers
# ────────────────────────────────────────────────────────────────────────


def _one_splat_stub(ndim_default: int = 2) -> Any:
    """Stub fitter returning a 1-splat result, so a merge has something to merge."""

    def stub(tile_data: np.ndarray, **kwargs: Any) -> GSplatData:
        ndim = tile_data.ndim
        chol = np.zeros((1, tril_size(ndim)), dtype=np.float32)
        k = 0
        for i in range(ndim):
            for j in range(i + 1):
                if i == j:
                    chol[0, k] = 1.0
                k += 1
        return GSplatData(
            centers=np.full((1, ndim), 1.0, dtype=np.float32),
            amplitudes=np.ones((1,), dtype=np.float32),
            cholesky_factors=chol,
            # What a real inner fit reports: it ran with floor="none" on data
            # whose pedestal was already removed, so its own bounds are
            # post-subtraction and its own `floor` is None.
            stats={
                "time_seconds": 0.0,
                "floor": None,
                "image_min": 0.0,
                "image_max": 150.0,
                "intensity_range": 150.0,
            },
        )

    return stub


@pytest.mark.parametrize("partition", [False, True])
def test_tiled_fit_persists_the_level_it_subtracted(
    monkeypatch, tmp_path: Path, partition: bool
) -> None:
    """The tiled merge — flat leaf AND kind=partition — reaches pipeline/floor."""
    pytest.importorskip("torch")
    import luxar.gsplats.fit_tiled_gsplats as ftg

    monkeypatch.setattr(ftg, "fit_gaussian_splats", _one_splat_stub())

    rng = np.random.RandomState(3)
    volume = rng.normal(100.0, 1.0, size=(96, 96)).astype(np.float32)
    volume[40:80, 40:80] += 150.0

    result = ftg.fit_tiled(
        volume,
        tile_size=48,
        overlap=16,
        floor=100.0,
        cull_retention=None,
        partition=partition,
        verbose=False,
    )

    path = tmp_path / f"tiled_{partition}.gsplats.zarr"
    if partition:
        write_gsplats_tree(path, result)
        _, stats = load_gsplat_node(path, include_stats=True)
    else:
        result.save(path)
        stats = load_gsplats(path, include_stats=True).stats

    assert stats["floor"] == pytest.approx(100.0)
    # And the bounds are in the INPUT volume's units, not the tile's
    # post-subtraction ones (the stub fitted data whose min it saw as 0.0).
    assert stats["image_min"] == pytest.approx(100.0)
    assert stats["image_max"] == pytest.approx(250.0)


def test_progressive_fit_records_the_level_it_subtracted(tmp_path: Path) -> None:
    """Every progressive pass runs with floor='none', so only the overall stats
    can say what pedestal was removed up front — and it has to SURVIVE the
    save, which is the whole point of #1175."""
    pytest.importorskip("torch")
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats

    rng = np.random.default_rng(1)
    V = np.full((16, 16, 16), 100.0, np.float32)
    V += rng.normal(0, 1.0, V.shape).astype(np.float32)
    V[0, 0, 0] = 0.0
    V[6:10, 6:10, 6:10] += 300.0

    result = fit_progressive_gaussian_splats(
        V,
        floor=100.0,
        max_splats=40,
        max_splats_per_pass=20,
        iters_per_pass=15,
        residual_pass_min_iters=15,
        max_passes=2,
        device="cpu",
        verbose=False,
    )

    assert result.stats["floor"] == pytest.approx(100.0)
    # Pass 0 saw the already-subtracted volume; the recorded bounds are shifted
    # back into the input's own units, so image_min is the applied level.
    assert result.stats["image_min"] == pytest.approx(100.0, abs=1e-3)
    assert result.stats["image_max"] > 300.0

    # No second, contradicting copy: `pass_stats` rides verbatim into the same
    # `pipeline/` group, and each pass's own block describes the ALREADY
    # floor-subtracted array it was handed (floor=None, image_min≈0).
    for entry in result.stats["pass_stats"]:
        assert not set(entry) & set(NORMALIZATION_STATS_KEYS), entry

    path = tmp_path / "progressive.gsplats.zarr"
    result.save(path)
    attrs = _pipeline_attrs(path)
    assert attrs["floor"] == pytest.approx(100.0)
    assert attrs["image_min"] == pytest.approx(100.0, abs=1e-3)
    reloaded = load_gsplats(path, include_stats=True).stats
    assert reloaded["floor"] == pytest.approx(100.0)


def test_progressive_and_flat_agree_on_a_floor_below_the_data_minimum() -> None:
    """Progressive and flat fits resolve the same effective baseline."""
    pytest.importorskip("torch")
    from luxar.gsplats import fit_gaussian_splats
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats

    rng = np.random.default_rng(7)
    # Minimum is ~98, far above the requested floor of 5.
    V = np.full((12, 12, 12), 100.0, np.float32)
    V += rng.normal(0, 0.5, V.shape).astype(np.float32)
    V[4:8, 4:8, 4:8] += 300.0
    V[0, 0, 0] = 5000.0

    common: dict[str, Any] = dict(
        floor=5.0, norm_percentile=1.0, device="cpu", verbose=False
    )
    progressive = fit_progressive_gaussian_splats(
        V,
        max_splats=20,
        max_splats_per_pass=20,
        iters_per_pass=10,
        residual_pass_min_iters=10,
        max_passes=2,
        **common,
    )
    flat = fit_gaussian_splats(V, n_splats=20, iterations=10, **common)

    assert flat.stats["floor"] > 5.0
    assert progressive.stats["floor"] == pytest.approx(flat.stats["floor"], abs=1e-3)
    assert progressive.stats["floor"] == pytest.approx(
        progressive.stats["image_min"], abs=1e-3
    )
    assert progressive.stats["image_max"] == pytest.approx(
        flat.stats["image_max"], abs=1e-3
    )


def test_progressive_shifts_a_supplied_range_after_floor_subtraction() -> None:
    pytest.importorskip("torch")
    from luxar.gsplats import fit_gaussian_splats
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats

    V = np.full((8, 8, 8), 100.0, np.float32)
    V[2:6, 2:6, 2:6] = 350.0
    common: dict[str, Any] = dict(
        floor=5.0,
        norm_range=(10.0, 400.0),
        device="cpu",
        verbose=False,
    )
    progressive = fit_progressive_gaussian_splats(
        V,
        max_splats=8,
        max_splats_per_pass=8,
        iters_per_pass=2,
        residual_pass_min_iters=2,
        max_passes=1,
        **common,
    )
    flat = fit_gaussian_splats(V, n_splats=8, iterations=2, **common)

    for key in ("floor", "image_min", "image_max", "intensity_range"):
        assert progressive.stats[key] == pytest.approx(flat.stats[key], abs=1e-3)


def test_progressive_fit_announces_floor_suppression_once(capsys) -> None:
    pytest.importorskip("torch")
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats

    V = np.full((8, 8, 8), 100.0, np.float32)
    V[2:6, 2:6, 2:6] = 350.0
    fit_progressive_gaussian_splats(
        V,
        floor=5.0,
        max_splats=8,
        max_splats_per_pass=8,
        iters_per_pass=2,
        residual_pass_min_iters=2,
        max_passes=1,
        device="cpu",
        verbose=True,
    )

    output = capsys.readouterr().out
    assert output.count("Floor suppression:") == 1


def test_progressive_fit_records_a_disabled_floor_as_null() -> None:
    pytest.importorskip("torch")
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats

    rng = np.random.default_rng(2)
    V = rng.random((12, 12, 12)).astype(np.float32)

    result = fit_progressive_gaussian_splats(
        V,
        floor="none",
        max_splats=20,
        max_splats_per_pass=20,
        iters_per_pass=10,
        residual_pass_min_iters=10,
        max_passes=1,
        device="cpu",
        verbose=False,
    )

    assert "floor" in result.stats and result.stats["floor"] is None


# ────────────────────────────────────────────────────────────────────────
# The subprocess merge paths (`fit -j`, `batch-fit merge`)
# ────────────────────────────────────────────────────────────────────────


_TILE_BLOCK = {
    "floor": 100.0,
    "image_min": 100.0,
    "image_max": 250.0,
    "intensity_range": 150.0,
    "time_seconds": 0.0,
}


@pytest.mark.parametrize("partition", [False, True])
def test_parallel_tiled_merge_recovers_the_level_from_its_tiles(
    tmp_path: Path, partition: bool
) -> None:
    """``fit --tiling uniform -j N`` fits each tile in a worker PROCESS and
    reloads it from disk, so the merge has no ``applied_floor`` of its own.

    Reloading without stats made every tile silent and the merge stamped
    ``floor: null`` — an affirmative "no pedestal was removed" — onto a store
    whose workers had each removed one.
    """
    import sys

    from luxar.gsplats.fit_tiled_parallel import fit_tiled_parallel

    def _builder(idx: int, num_tiles: int, out_path: Path) -> list[str]:
        # Stand in for the worker subprocess: write the tile store the real
        # `fit --tile i/M` would write, then exit cleanly.
        _leaf(ndim=2, stats=dict(_TILE_BLOCK)).save(out_path)
        return [sys.executable, "-c", "pass"]

    merged = fit_tiled_parallel(
        num_tiles=4,
        jobs=2,
        tmp_dir=tmp_path / "tiles",
        worker_cmd_builder=_builder,
        volume_shape=(96, 96),
        tile_size=48,
        overlap=16,
        progressive=False,
        cull_retention=None,
        verbose=False,
        partition=partition,
    )

    target = merged.meta if partition else merged.stats
    assert target["floor"] == pytest.approx(100.0)
    assert target["image_min"] == pytest.approx(100.0)


def test_batch_merge_records_the_level_the_plan_pinned(tmp_path: Path) -> None:
    """``batch-fit merge`` streams parts from a generator, so there is no root
    node whose ``meta`` the tree writer could promote — the level has to come
    off the manifest, which is the only place it survives."""
    from luxar.gsplats.batch.manifest import BatchManifest
    from luxar.gsplats.batch.merge_orchestrator import _batch_floor_stats

    pinned = BatchManifest(input_path="in.zarr", floor_level=110.0)
    assert _batch_floor_stats(pinned) == {"floor": 110.0}

    disabled = BatchManifest(input_path="in.zarr", fit_args={"floor": "none"})
    assert _batch_floor_stats(disabled) == {"floor": None}

    # A forwarded SPEC means each task resolved its own level: there is no one
    # answer, and `floor: null` would be a false one.
    unpinned = BatchManifest(input_path="in.zarr", fit_args={"floor": "auto"})
    assert _batch_floor_stats(unpinned) == {}
    assert _batch_floor_stats(BatchManifest(input_path="in.zarr")) == {}


def test_batch_merge_writes_the_block_for_a_single_tile_run(
    monkeypatch, tmp_path: Path
) -> None:
    """The K==1 branch saves a bare leaf via ``GSplatData.save``, which derives
    ``pipeline/`` from ``stats`` — a different seam from the streaming one."""
    import luxar.gsplats.batch.merge_orchestrator as mo
    from luxar.gsplats.batch.manifest import BatchManifest

    manifest = BatchManifest(
        input_path="in.zarr", n_tiles=1, n_timepoints=1, n_channels=1, floor_level=110.0
    )
    (tmp_path / "tiles").mkdir()

    def _fake_part(*args: Any, **kwargs: Any) -> GSplatData:
        return _leaf(ndim=3, stats={"time_seconds": 1.0})

    monkeypatch.setattr(mo, "_build_part_for_tile", _fake_part)
    final = mo._merge_partition(manifest, tmp_path, None, True, False)

    assert _pipeline_attrs(final)["floor"] == pytest.approx(110.0)


# ────────────────────────────────────────────────────────────────────────
# Degenerate merges
# ────────────────────────────────────────────────────────────────────────


def test_a_merge_with_no_tiles_still_answers_the_floor_question() -> None:
    """A merge with nothing to merge still records the level IT applied — an
    ABSENT key reads as "this artifact does not know", which would be false."""
    from luxar.gsplats.fit_tiled_gsplats import merge_tile_results

    merged = merge_tile_results(
        [],
        volume_shape=(32, 32),
        tile_size=16,
        overlap=4,
        num_tiles=0,
        progressive=False,
        cull_retention=None,
        elapsed=0.0,
        verbose=False,
        applied_floor=100.0,
    )
    assert merged.stats["floor"] == pytest.approx(100.0)


def test_a_partition_merge_with_no_surviving_region_still_answers() -> None:
    """Every tile empty → the partition branch bails before it builds a node,
    and used to return ``stats={}``."""
    from luxar.gsplats.fit_tiled_gsplats import merge_tile_results

    merged = merge_tile_results(
        [_leaf(n=0, ndim=2, stats=dict(_TILE_BLOCK)) for _ in range(2)],
        volume_shape=(32, 32),
        tile_size=16,
        overlap=4,
        num_tiles=2,
        progressive=False,
        cull_retention=None,
        elapsed=0.0,
        verbose=False,
        partition=True,
        applied_floor=None,
    )
    # Recovered from the tiles, since this call applied no level of its own.
    assert merged.stats["floor"] == pytest.approx(100.0)


def test_a_merge_that_knows_nothing_stays_silent() -> None:
    """No level of its own and tiles that record none (an older luxar's tile
    store) → say nothing. ``floor: null`` would assert no pedestal was removed,
    which the merge is in no position to claim."""
    from luxar.gsplats.fit_tiled_gsplats import merge_tile_results

    merged = merge_tile_results(
        [_leaf(n=1, ndim=2) for _ in range(2)],
        volume_shape=(32, 32),
        tile_size=16,
        overlap=4,
        num_tiles=2,
        progressive=False,
        cull_retention=None,
        elapsed=0.0,
        verbose=False,
        applied_floor=None,
    )
    assert "floor" not in merged.stats


def test_an_all_empty_concatenate_does_not_promote_the_first_input() -> None:
    """The all-empty branch used to copy input 0's stats wholesale, which is
    exactly what the fresh merge-stat contract exists to prevent."""
    merged = GSplatData.concatenate(
        [
            _leaf(n=0, ndim=3, stats={"floor": 110.0, "fitter_name": "x"}),
            _leaf(n=0, ndim=3, stats={"floor": 3.0}),
        ]
    )
    assert "floor" not in merged.stats
    assert "fitter_name" not in merged.stats
    assert merged.stats["concatenated_from"] == 2
    assert merged.stats["splats_per_source"] == [0, 0]
