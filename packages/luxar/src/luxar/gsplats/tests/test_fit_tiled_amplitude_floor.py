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
