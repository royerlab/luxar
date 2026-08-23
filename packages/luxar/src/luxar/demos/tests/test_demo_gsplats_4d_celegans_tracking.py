"""Tests for the C. elegans demo's cached 4D assembly pipeline."""

from __future__ import annotations

import numpy as np

from luxar.demos import demo_gsplats_4d_celegans_tracking as demo
from luxar.gsplats.gsplat_data import GSplatData


def _tiny_fit(seed: int) -> GSplatData:
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(-1.0, 1.0, (200, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, 200).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (200, 1)).astype(np.float32),
    )


def test_cached_4d_stages_return_the_stored_stream_ladders(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(demo, "CACHE_DIR", tmp_path)

    combined = demo.combine_timepoints_to_4d([_tiny_fit(seed) for seed in range(3)])
    combined_cache = tmp_path / (
        f"celegans_s{demo.SAMPLE_INDEX}_combined_4d_3tp.gsplats.zarr.zip"
    )

    assert combined_cache.exists()
    assert combined.n_additive_sublods == 4

    combined_without_lod = combined.flattened()
    assert combined_without_lod.n_additive_sublods == 1

    filtered = demo.filter_background_splats(combined_without_lod)
    threshold = f"{demo.SPEC_BRIGHTNESS_THRESHOLD:.4f}".replace(".", "p")
    filtered_cache = tmp_path / (
        f"celegans_s{demo.SAMPLE_INDEX}_filtered_4d_{combined.n_splats}n_"
        f"sb{threshold}.gsplats.zarr.zip"
    )

    assert filtered_cache.exists()
    assert filtered.n_additive_sublods == 4
