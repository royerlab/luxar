"""The grid fallback must survive asking for a single seed on a thin volume.

`_add_grid_fallback_seeds` sizes its grid from ``needed * 4``. With ``needed``
of 1 the spacing is roughly ``0.63 * (volume ** (1/ndim))``, which on a CUBE
always leaves at least one grid point per axis — but on an anisotropic volume
the spacing easily exceeds the short axis, ``spacing // 2`` lands past its end,
every ``np.arange`` comes back empty, and ``np.array([])`` collapses to shape
``(0,)`` instead of ``(0, ndim)``. The spatial query then rejects it with
"query must have shape (Q, 3); got (0,)".

The anisotropy matters and is the whole point of these fixtures: a cubic volume
does NOT reproduce it (64³ yields spacing 41 and 8 grid points), so a test
written on a cube passes without ever exercising the guard. Measured triggers:
(4, 256, 256) and (2, 512, 512) both yield zero grid points.

Not hypothetical — this took down the entire storm_3d_microtubules tiled fit
29 minutes in, on a tile whose auto seeding produced 5,999 of a 6,000 target.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..preprocessing import _add_grid_fallback_seeds

# Thin first axis: the shape class that actually degenerates.
THIN_SHAPE = (4, 256, 256)


def _grid_point_count(shape: tuple[int, ...], needed: int) -> int:
    """Mirror the initial spacing calculation in ``_add_grid_fallback_seeds``.

    Keep this formula aligned with that implementation: its purpose is to prove
    the fixture still reaches the empty first-pass grid before the dense retry.
    """
    volume = int(np.prod(shape))
    ndim = len(shape)
    spacing = max(int(np.ceil((volume / (needed * 4)) ** (1.0 / ndim))), 1)
    ranges = [np.arange(spacing // 2, s, spacing) for s in shape]
    return len(list(itertools.product(*ranges)))


def test_fixture_actually_degenerates() -> None:
    """Guard the guard: if this shape stops producing an empty grid, the tests
    below would pass vacuously and the regression could return unnoticed."""
    assert _grid_point_count(THIN_SHAPE, needed=1) == 0
    # And confirm a cube would NOT have caught it.
    assert _grid_point_count((64, 64, 64), needed=1) > 0


def test_single_seed_shortfall_on_thin_volume_does_not_raise() -> None:
    volume = np.zeros(THIN_SHAPE, dtype=np.float32)
    volume[2, 128, 128] = 1.0
    existing = np.array([[1.0, 10.0, 10.0], [3.0, 200.0, 200.0]], dtype=float)

    target_count = len(existing) + 1
    seeds, spacing = _add_grid_fallback_seeds(
        volume, target_count=target_count, existing_seeds=existing, verbose=False
    )

    assert seeds.ndim == 2
    assert seeds.shape[1] == len(THIN_SHAPE)
    assert len(seeds) == target_count
    assert spacing > 0


def test_thin_volume_with_no_existing_seeds() -> None:
    """The other order: nothing to filter against, and an empty grid."""
    volume = np.zeros(THIN_SHAPE, dtype=np.float32)
    empty = np.empty((0, 3), dtype=float)

    target_count = 1
    seeds, spacing = _add_grid_fallback_seeds(
        volume, target_count=target_count, existing_seeds=empty, verbose=False
    )
    assert seeds.ndim == 2
    assert seeds.shape[1] == len(THIN_SHAPE)
    assert len(seeds) == target_count
    assert spacing > 0


def test_large_shortfall_still_fills() -> None:
    """A normal shortfall must keep working — the guard must not short-circuit it."""
    volume = np.zeros((64, 64, 64), dtype=np.float32)
    existing = np.array([[5.0, 5.0, 5.0]], dtype=float)

    seeds, _ = _add_grid_fallback_seeds(
        volume, target_count=200, existing_seeds=existing, verbose=False
    )
    assert len(seeds) > len(existing), "fallback should have added grid seeds"
    assert seeds.shape[1] == 3


def test_sparse_clustered_seeds_large_target_is_fast() -> None:
    """Regression for the ``gsplat cal`` stall at K=128000.

    The top-up filters its grid against the existing seeds. When those sit in
    a few tight clusters inside a mostly empty volume, almost every grid point
    is far from all of them, and the former kNN filter brute-forced the WHOLE
    seed set per such query: O(Q*N), 29 s at this size (32k target) and
    minutes at the reported 128k. The range test is O((Q + N) log N) and runs
    in well under a second here; the bound is generous so a slow CI runner
    still passes while the quadratic version cannot.
    """
    import time

    shape = (65, 312, 462)
    rng = np.random.default_rng(0)
    volume = rng.random(shape, dtype=np.float32)
    centers = np.array([[s * f for s in shape] for f in (0.3, 0.5, 0.7)])
    existing = np.concatenate(
        [c + rng.normal(0, 6, size=(1000, 3)) for c in centers]
    ).astype(np.float32)

    target_count = 32_000
    t0 = time.perf_counter()
    seeds, spacing = _add_grid_fallback_seeds(
        volume, target_count=target_count, existing_seeds=existing, verbose=False
    )
    elapsed = time.perf_counter() - t0

    assert len(seeds) == target_count
    np.testing.assert_array_equal(seeds[: len(existing)], existing)
    added = seeds[len(existing) :]
    # The min-distance guarantee the filter exists for still holds.
    from scipy.spatial import cKDTree

    nearest, _ = cKDTree(existing).query(added, k=1)
    assert nearest.min() > max(1.0, spacing * 0.3)
    assert elapsed < 5.0, f"grid top-up took {elapsed:.1f}s (quadratic regression?)"


def test_range_filter_matches_former_knn_predicate() -> None:
    """The KD-tree range test keeps exactly the grid points the kNN filter
    kept (``distance > min_distance``), integer-coordinate ties included."""
    from luxar.gsplats.spatial_hash import BatchedSpatialHashGrid

    from ..preprocessing import _has_point_within

    rng = np.random.default_rng(1)
    for trial in range(8):
        shape = rng.integers(20, 60, size=3)
        spacing = int(rng.integers(2, 9))
        n = int(rng.integers(5, 800))
        if trial % 2:
            pts = rng.integers(0, shape, size=(n, 3)).astype(np.float32)
        else:
            pts = (rng.random((n, 3)) * shape).astype(np.float32)
        ranges = [np.arange(spacing // 2, s, spacing) for s in shape]
        queries = np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, 3)
        min_distance = max(1.0, spacing * 0.3)
        grid = BatchedSpatialHashGrid.from_points(
            pts, cell_size=max(spacing, min_distance * 2.0), device="cpu"
        )
        distances, _ = grid.query_knn(queries.astype(np.float32), k=1)
        old_keep = distances[:, 0] > min_distance
        new_keep = ~_has_point_within(pts, queries.astype(float), min_distance)
        np.testing.assert_array_equal(new_keep, old_keep)
