"""Scale counts follow original fitter-space classifications through reductions."""

import numpy as np
import pytest

from luxar.gsplats.fitting.scale_population import (
    MASK_KEY,
    classify_scales,
    decode_population_bits,
    encode_population_bits,
    finish_population_stats,
    merge_population_stats,
    reduce_population_stats,
    stamp_population_stats,
    sum_population_counts,
)
from luxar.gsplats.gsplat_data import GSplatData


def _source(scales: list[float], amplitudes: list[float]) -> GSplatData:
    n = len(scales)
    raw = np.zeros((n, 2, 2), dtype=np.float32)
    raw[:, 0, 0] = scales
    raw[:, 1, 1] = scales
    stats = {
        "scale_diagnostic_tolerance_vox": 0.01,
        "fit_init_sigma_diag_vox": [1.0, 1.0],
        "relocation_init_sigma_vox": 0.5,
        "sigma_min_diag_vox": [0.25, 0.25],
        "n_splats": n,
        "dynamic_ops_relocation_events": 3,
        "dynamic_ops_unique_splats_relocated": 2,
    }
    relocated = np.array([i % 2 == 0 for i in range(n)])
    bits = classify_scales(raw, stats, relocated)
    stamp_population_stats(stats, bits)
    stats[MASK_KEY] = encode_population_bits(bits)
    # The delivered factors have a different scale after correction and units.
    # Re-measuring these would give different counts from the fitter-space masks.
    return GSplatData(
        centers=np.arange(n * 2, dtype=np.float32).reshape(n, 2)[::-1].copy(),
        amplitudes=np.asarray(amplitudes, np.float32),
        cholesky_factors=np.stack(
            [np.array([s * 2, 0, s * 2], np.float32) for s in scales]
        ),
        stats=stats,
    )


def test_cull_reduces_original_scale_masks_by_actual_kept_indices() -> None:
    source = _source([0.5, 1.0, 0.25, 0.5], [9.0, 1.0, 8.0, 0.1])
    kept: list[np.ndarray] = []
    delivered = source.cull(method="cumulative", retention=0.9, keep_mask_out=kept)
    assert kept[0].tolist() == [True, False, True, False]
    reduce_population_stats(source.stats, delivered.stats, kept[0])
    assert delivered.stats["splats_near_relocation_init_sigma_count"] == 1
    assert delivered.stats["splats_near_relocation_init_sigma_relocated_count"] == 1
    assert delivered.stats["splats_near_relocation_init_sigma_unrelocated_count"] == 0
    assert delivered.stats["splats_near_sigma_min_count"] == 1
    assert delivered.stats["splats_near_fit_init_sigma_count"] == 0
    assert delivered.stats["splats_near_relocation_init_sigma_fraction"] == 0.5
    assert np.array_equal(
        decode_population_bits(delivered.stats[MASK_KEY], delivered.n_splats),
        decode_population_bits(source.stats[MASK_KEY], source.n_splats)[kept[0]],
    )
    finish_population_stats(delivered.stats)
    assert MASK_KEY not in delivered.stats


def test_flat_merge_global_cull_and_partition_counts_match_delivered_set() -> None:
    first = _source([0.5, 1.0], [9.0, 0.1])
    second = _source([0.25, 0.5], [8.0, 1.0])
    merged = GSplatData.concatenate([first, second])
    merge_population_stats([first.stats, second.stats], merged.stats)
    assert merged.stats["splats_near_relocation_init_sigma_count"] == 2
    kept: list[np.ndarray] = []
    delivered = merged.cull(method="cumulative", retention=0.9, keep_mask_out=kept)
    reduce_population_stats(merged.stats, delivered.stats, kept[0])
    assert delivered.stats["splats_near_relocation_init_sigma_count"] == 1
    assert delivered.stats["splats_near_relocation_init_sigma_fraction"] == 0.5
    assert delivered.stats["dynamic_ops_relocation_events"] == 6

    root: dict[str, object] = {}
    sum_population_counts(
        [first.stats, second.stats], root, first.n_splats + second.n_splats
    )
    assert root["splats_near_relocation_init_sigma_count"] == 2
    assert root["splats_near_relocation_init_sigma_fraction"] == 0.5
    assert root["dynamic_ops_relocation_events"] == 6

    different_topology: dict[str, object] = {}
    sum_population_counts([first.stats, second.stats], different_topology, 8)
    assert "splats_near_relocation_init_sigma_count" not in different_topology
    assert different_topology["dynamic_ops_relocation_events"] == 6


def test_invalid_mask_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="length"):
        decode_population_bits(encode_population_bits(np.ones(2, np.uint8)), 3)


def test_relocation_classification_uses_effective_reseed_scale() -> None:
    raw = np.zeros((2, 2, 2), np.float32)
    raw[:, 0, 0] = [2.0, 0.5]
    raw[:, 1, 1] = [2.0, 0.5]
    stats = {
        "scale_diagnostic_tolerance_vox": 0.01,
        "relocation_init_sigma_vox": 0.5,
        "relocation_reseed_sigma_vox_effective": 2.0,
        "splats_near_relocation_init_sigma_relocated_count": 0,
    }
    bits = classify_scales(raw, stats, np.array([True, False]))
    stamp_population_stats(stats, bits)
    assert stats["splats_near_relocation_init_sigma_count"] == 1
    assert stats["splats_near_relocation_init_sigma_relocated_count"] == 1
    assert stats["splats_near_relocation_init_sigma_unrelocated_count"] == 0


def test_intermediate_tile_store_round_trips_classifications(tmp_path) -> None:
    source = _source([0.5, 1.0, 0.25], [3.0, 2.0, 1.0])
    path = tmp_path / "tile.gsplats.zarr"
    source.save(path)
    loaded = GSplatData.load(path, include_stats=True)
    assert loaded.stats[MASK_KEY] == source.stats[MASK_KEY]
    np.testing.assert_allclose(loaded.centers, source.centers, atol=1e-4)
    assert np.array_equal(
        decode_population_bits(loaded.stats[MASK_KEY], loaded.n_splats),
        decode_population_bits(source.stats[MASK_KEY], source.n_splats),
    )


def test_real_fit_does_not_write_population_bits_as_lod_stats(tmp_path) -> None:
    import zarr

    from luxar.gsplats.fit_gsplats import fit_gaussian_splats

    yy, xx = np.mgrid[:12, :12]
    volume = np.exp(-((yy - 6) ** 2 + (xx - 6) ** 2) / 8).astype(np.float32)
    result = fit_gaussian_splats(
        volume,
        seeds=8,
        n_iters=3,
        device="cpu",
        verbose=False,
        enable_dynamic_ops=False,
    )
    assert MASK_KEY not in result.stats
    assert MASK_KEY in result.additive_sublods[0].stats
    result.tree.meta["stats"] = {
        "quality": 0.9,
        MASK_KEY: result.additive_sublods[0].stats[MASK_KEY],
    }

    path = tmp_path / "fit.gsplats.zarr"
    result.save(path)
    assert all(MASK_KEY not in item.read_text() for item in path.rglob("zarr.json"))
    root = zarr.open_group(str(path), mode="r")
    assert root.attrs["level_stats"] == {"quality": 0.9}
    assert root["fitting"].attrs["splats_near_fit_init_sigma_count"] == result.stats[
        "splats_near_fit_init_sigma_count"
    ]


@pytest.mark.parametrize(
    "partition,expected_count,expected_splats", [(False, 1, 2), (True, 2, 3)]
)
def test_reloaded_tiles_report_only_delivered_populations(
    tmp_path, partition: bool, expected_count: int, expected_splats: int
) -> None:
    from luxar.gsplats.fit_tiled_gsplats import merge_tile_results

    sources = [
        _source([0.5, 1.0], [9.0, 0.1]),
        _source([0.25, 0.5], [8.0, 1.0]),
    ]
    loaded = []
    for index, source in enumerate(sources):
        path = tmp_path / f"tile_{index}.gsplats.zarr"
        source.save(path)
        loaded.append(GSplatData.load(path, include_stats=True))
    result = merge_tile_results(
        loaded,
        volume_shape=(2, 4),
        tile_size=2,
        overlap=0,
        num_tiles=2,
        progressive=False,
        cull_retention=0.9,
        elapsed=1.0,
        verbose=False,
        partition=partition,
    )
    stats = result.meta["fit_stats"] if partition else result.stats
    assert stats["n_splats"] == expected_splats
    assert stats["splats_near_relocation_init_sigma_count"] == expected_count
    assert stats["splats_near_relocation_init_sigma_fraction"] == pytest.approx(
        expected_count / expected_splats
    )
    assert stats["dynamic_ops_relocation_events"] == 6
    assert MASK_KEY not in stats
