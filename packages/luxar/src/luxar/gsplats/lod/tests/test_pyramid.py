"""Tests for the 2-D LOD pyramid builder.

Covers ``make_lod_pyramid`` from ``luxar.gsplats.lod.pyramid`` — the
one-shot substitutive × additive matrix builder.
"""

from __future__ import annotations

import numpy as np

from luxar.gsplats.gsplat_data import GSplatData


def _make_random_gsplat(n: int = 64, ndim: int = 3, seed: int = 0) -> GSplatData:
    """Random anisotropic Gaussian splats with overlap."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n, ndim)).astype(np.float32) * 1.5
    amplitudes = (np.abs(rng.standard_normal(n)) + 0.5).astype(np.float32)
    tril = ndim * (ndim + 1) // 2
    chol = (rng.standard_normal((n, tril)) * 0.1).astype(np.float32)
    diag_idx = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag_idx] = np.abs(chol[:, diag_idx]) + 0.4
    return GSplatData(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=chol,
    )


def test_make_lod_pyramid_full_matrix() -> None:
    """`make_lod_pyramid` produces a [levels+1, n_additive_lods] matrix."""
    from luxar.gsplats.lod import make_lod_pyramid

    data = _make_random_gsplat(n=64, ndim=3, seed=14)
    pyr = make_lod_pyramid(
        data,
        compression_factor=4,
        levels=2,
        substitutive_method="kmeans_lloyd",
        n_additive_lods=3,
        additive_method="self_energy",
        device="cpu",
        seed=0,
    )
    assert pyr.n_substitutive == 3
    # Each substitutive level has its own additive ladder (subject to clamping).
    for s in range(3):
        lev = pyr.substitutive_levels[s]
        assert lev.n_additive_lods >= 1
        # Stats propagated:
        assert lev.compression_factor == 4**s
        assert lev.level_index == s
        if s == 0:
            assert lev.parent_method is None
        else:
            assert lev.parent_method == "kmeans_lloyd"


def test_make_lod_pyramid_defaults_to_auto_substitutive_method() -> None:
    """The library default for `substitutive_method` is `auto` (matches
    `make_substitutive_lod`), not the legacy `kmeans_lloyd`."""
    import inspect

    from luxar.gsplats.lod import make_lod_pyramid

    assert (
        inspect.signature(make_lod_pyramid).parameters["substitutive_method"].default
        == "auto"
    )

    # And it runs end-to-end without an explicit method (resolves via `auto`).
    data = _make_random_gsplat(n=64, ndim=3, seed=21)
    pyr = make_lod_pyramid(
        data, compression_factor=4, levels=1, n_additive_lods=2, device="cpu"
    )
    assert pyr.n_substitutive == 2


def test_pyramid_stream_ladders_are_sibling_aware() -> None:
    """Levels with a coarser sibling get a raised stream base so an upgrade's
    committed prefix passes the sibling's total within two chunks; the
    coarsest level keeps the user's small fast-first-paint base."""
    import math

    from luxar.gsplats.lod import make_lod_pyramid

    data = _make_random_gsplat(n=256, ndim=3, seed=7)
    pyr = make_lod_pyramid(
        data,
        compression_factor=4,
        levels=2,
        substitutive_method="kmeans_lloyd",
        additive_method="self_energy",
        breakpoints="stream:4",
        device="cpu",
        seed=0,
    )
    coarsest = pyr.n_substitutive - 1
    for s, lev in enumerate(pyr.substitutive_levels):
        chunk = lev.additive_sublods[0].stats["lod_stream_chunk_splats"]
        if s == coarsest:
            assert chunk == 4  # eager fast-first-paint level: user base kept
        else:
            assert chunk == max(4, math.ceil(lev.n_splats_total / 8.0))
    # The headline invariant: each finer level's committed prefix reaches its
    # coarser sibling's TOTAL within two network chunks (measured pathology:
    # shared small bases pushed this to 2-3 chunks from the ladder END).
    for s in range(coarsest):
        sibling_total = pyr.substitutive_levels[s + 1].n_splats_total
        cum = 0
        chunks_needed = 0
        for sub in pyr.substitutive_levels[s].additive_sublods:
            cum += sub.n_splats
            chunks_needed += 1
            if cum >= sibling_total:
                break
        assert cum >= sibling_total
        assert chunks_needed <= 2
