"""Seed order must not depend on how a platform's sort orders equal keys.

NumPy's default ``argsort`` (and ``torch.topk``) leave the order of ties
unspecified, and x86 and arm64 really do order them differently, so the same
CPU fit used to produce different seeds -- and a 1 dB different result -- on
Linux and macOS. Every order-defining sort in the seeding and top-up path now
breaks ties by input (C-order) index; these tests pin that contract on one
machine by checking orders against the canonical stable one.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.fitting.dynamic_ops.operations import _select_weak_splats
from luxar.gsplats.fitting.dynamic_ops.peak_finding import _top_k_stable
from luxar.gsplats.fitting.preprocessing import (
    _add_grid_fallback_seeds,
    _subsample_seeds_spatially_diverse,
)
from luxar.gsplats.seeds import dedupe_farthest_first, generate_seeds
from luxar.gsplats.seeds.edges import _poisson_disk_sample_weighted
from luxar.gsplats.seeds.multiscale_decomposition import (
    _dedupe_with_scales_and_energies,
)
from luxar.gsplats.seeds.peaks import seed_from_peaks, top_k_indices_stable
from luxar.gsplats.seeds.utils import argsort_descending, local_maxima


def _tied_volume() -> np.ndarray:
    """Symmetric Gaussian blobs over a flat pedestal: many exactly equal values."""
    shape = (24, 24, 80)
    volume = np.zeros(shape, dtype=np.float32)
    volume[:, :, :36] = 0.05
    zz, yy, xx = np.indices(shape)
    for center, amplitude, sigma in (
        ((8, 8, 52), 1.0, 2.0),
        ((16, 16, 62), 0.9, 2.5),
        ((8, 16, 72), 0.8, 1.8),
        ((16, 7, 55), 0.7, 2.2),
    ):
        distance_squared = sum(
            (axis - c) ** 2 for axis, c in zip((zz, yy, xx), center, strict=True)
        )
        volume += amplitude * np.exp(-distance_squared / (2 * sigma**2)).astype(
            np.float32
        )
    return volume


def _assert_brightest_first_ties_by_index(
    values: np.ndarray, indices: np.ndarray
) -> None:
    """(-value, index) must be strictly increasing along the returned order."""
    order = np.lexsort((indices, -values))
    np.testing.assert_array_equal(order, np.arange(len(values)))


@pytest.mark.parametrize(
    "dtype", [np.float32, np.float64, np.int8, np.int64, np.uint8, np.uint64]
)
def test_argsort_descending_breaks_ties_by_input_index(dtype: type) -> None:
    values = np.array([1, 3, 3, 2, 3], dtype=dtype)
    np.testing.assert_array_equal(argsort_descending(values), [1, 2, 4, 3, 0])


def test_argsort_descending_integer_extremes_do_not_wrap() -> None:
    lo8 = np.array([-128, 5, 0, 5], dtype=np.int8)
    np.testing.assert_array_equal(argsort_descending(lo8), [1, 3, 2, 0])
    i64 = np.iinfo(np.int64)
    big = np.array([i64.min, i64.max, i64.max - 1, i64.max], dtype=np.int64)
    np.testing.assert_array_equal(argsort_descending(big), [1, 3, 2, 0])
    u64 = np.array([2**63 + 1, 5, 2**64 - 1, 2**63 + 1], dtype=np.uint64)
    np.testing.assert_array_equal(argsort_descending(u64), [2, 0, 3, 1])
    flags = np.array([False, True, False, True])
    np.testing.assert_array_equal(argsort_descending(flags), [1, 3, 0, 2])
    assert len(argsort_descending(np.array([], dtype=np.int64))) == 0


def test_argsort_descending_puts_nan_last() -> None:
    values = np.array([np.nan, 1.0, np.nan, 2.0, 1.0])
    np.testing.assert_array_equal(argsort_descending(values), [3, 1, 4, 0, 2])


def test_local_maxima_top_k_keeps_lowest_c_order_ties_and_returns_c_order() -> None:
    image = np.zeros((51, 51), dtype=np.float32)
    image[1::3, 1::3] = 5.0
    image[49, 49] = 6.0

    peaks = local_maxima(image, radius=1, thresh=1.0, top_k=8)

    expected = np.array([(1, c) for c in range(1, 20, 3)] + [(49, 49)])
    np.testing.assert_array_equal(peaks, expected)


def test_edge_candidates_are_brightest_first_with_c_order_ties() -> None:
    # Two density levels over a 3D grid: thousands of exact ties, large enough
    # that an unstable introsort does reorder them.
    rng = np.random.default_rng(0)
    density = rng.integers(1, 3, size=(20, 20, 20)).astype(np.float32)
    sampled = _poisson_disk_sample_weighted(
        density, np.ones_like(density, dtype=bool), n_samples=800, min_distance=1.0
    )
    coords = sampled.astype(int)
    assert len(coords) == 800
    _assert_brightest_first_ties_by_index(
        density[tuple(coords.T)], np.ravel_multi_index(tuple(coords.T), density.shape)
    )


@pytest.mark.parametrize("n", [30, 400])  # the simple (<50) and the KD-tree path
def test_dedupe_keeps_tied_seeds_in_input_order(n: int) -> None:
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 50, size=(n, 3))
    intensities = rng.integers(0, 3, size=n).astype(np.float32)
    _, kept = dedupe_farthest_first(coords, min_distance=1.5, intensities=intensities)
    _assert_brightest_first_ties_by_index(intensities[kept], kept)


def test_generate_seeds_order_is_pinned_on_tied_input() -> None:
    # A cross-platform pin: before stable sorts, macOS returned
    # [14,15,61], [18,15,61], [18,15,63], ... and Linux [18,15,61], [14,15,61], ...
    seeds = generate_seeds(_tied_volume(), n_seeds=359, device="cpu", verbose=False)
    centers = np.asarray(seeds.centers)
    assert len(centers) == 90
    np.testing.assert_array_equal(
        centers[:4], [[18, 15, 61], [14, 15, 61], [18, 15, 63], [14, 15, 63]]
    )


@pytest.mark.parametrize("k", [1, 2, 3])
def test_tile_top_k_breaks_ties_by_lowest_index(k: int) -> None:
    x = torch.tensor([[1.0, 3.0, 3.0, 0.0, 3.0], [-np.inf, 2.0, 2.0, 2.0, 1.0]])
    values, idx = _top_k_stable(x, k)
    expected = torch.tensor([[1, 2, 4], [1, 2, 3]])[:, :k]
    assert torch.equal(idx, expected)
    assert torch.equal(values, torch.gather(x, 1, expected))


def test_tile_top_k_matches_a_full_stable_sort_on_heavily_tied_rows() -> None:
    gen = torch.Generator().manual_seed(0)
    x = torch.randint(-1, 4, (64, 40), generator=gen).float()
    x[x < 0] = -np.inf  # padding (non-peaks); its order is unspecified
    for k in range(2, 41):
        values, idx = _top_k_stable(x, k)
        ref = torch.sort(x, dim=1, descending=True, stable=True)
        finite = torch.isfinite(ref.values[:, :k])
        assert torch.equal(values, ref.values[:, :k])
        assert torch.equal(idx[finite], ref.indices[:, :k][finite])


def _canonical(values: np.ndarray) -> np.ndarray:
    """Brightest first, ties by ascending index: the order every site must give."""
    return np.lexsort((np.arange(len(values)), -values))


def test_grid_fallback_top_up_is_brightest_first_with_c_order_ties() -> None:
    volume = np.ones((40, 40, 40), dtype=np.float32)
    volume[:, :, 20:] = 2.0
    existing = np.zeros((0, 3))
    combined, _ = _add_grid_fallback_seeds(volume, 500, existing, verbose=False)
    added = combined.astype(int)
    assert len(added) == 500
    intensities = volume[tuple(added.T)]
    _assert_brightest_first_ties_by_index(
        intensities, np.ravel_multi_index(tuple(added.T), volume.shape)
    )


def test_fast_subsample_draws_from_the_canonical_top_candidates() -> None:
    n, target = 300, 100
    seeds = np.stack([np.arange(n, dtype=float)] * 3, axis=1)
    intensities = np.where(np.arange(n) % 3 == 0, 2.0, 1.0)
    _, picked = _subsample_seeds_spatially_diverse(
        seeds,
        intensities,
        target,
        verbose=False,
        return_indices=True,
        smart_subsample_threshold=10,
    )
    top = _canonical(intensities)[: int(target * 1.2)]
    expected = np.random.default_rng(seed=42).choice(top, size=target, replace=False)
    np.testing.assert_array_equal(picked, expected)


def test_multiscale_dedupe_keeps_tied_energies_in_input_order() -> None:
    n = 200
    coords = np.stack([np.arange(n, dtype=float) * 10.0, np.zeros(n)], axis=1)
    energies = np.where(np.arange(n) % 4 == 0, 3.0, 1.0)
    scales = np.arange(n, dtype=float)  # carries the input index through
    _, kept_scales, kept_energies = _dedupe_with_scales_and_energies(
        coords, scales, energies, min_distance=1.0
    )
    np.testing.assert_array_equal(kept_scales, _canonical(energies))
    np.testing.assert_array_equal(kept_energies, energies[_canonical(energies)])


def test_peak_seeds_are_brightest_first_with_c_order_ties() -> None:
    rng = np.random.default_rng(2)
    volume = rng.integers(1, 3, size=(10, 10, 10)).astype(np.float32)
    seeds = seed_from_peaks(volume, n_seeds=None, init_sigma=1.5, device="cpu")
    coords = np.asarray(seeds.centers).astype(int)
    assert len(coords) == volume.size
    _assert_brightest_first_ties_by_index(
        volume[tuple(coords.T)], np.ravel_multi_index(tuple(coords.T), volume.shape)
    )


def test_peak_prefilter_keeps_the_lowest_index_ties_at_the_cut() -> None:
    values = torch.tensor([1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 0.0, 2.0])
    np.testing.assert_array_equal(top_k_indices_stable(values, 4), [1, 2, 3, 4])
    np.testing.assert_array_equal(top_k_indices_stable(values, 2), [1, 3])
    np.testing.assert_array_equal(top_k_indices_stable(values, 9), np.arange(8))
    # Large enough that topk's own choice among ties is not lowest-index.
    tied = torch.randint(0, 3, (1000,), generator=torch.Generator().manual_seed(0))
    tied = tied.float()
    expected = torch.sort(torch.sort(tied, descending=True, stable=True)[1][:500])
    assert torch.equal(top_k_indices_stable(tied, 500), expected.values)


def test_weak_splats_are_weakest_first_with_lowest_index_ties() -> None:
    n = 300
    importance = torch.where(torch.arange(n) % 3 == 0, 2.0, 1.0)
    centers = torch.zeros((n, 3))
    residual = torch.zeros((4, 4, 4))
    weak = _select_weak_splats(importance, centers, residual, 50.0)
    expected = torch.sort(importance, stable=True)[1][: n // 2]
    assert torch.equal(weak, expected)
