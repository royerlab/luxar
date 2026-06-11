"""Per-substitutive-level filter/cull semantics (decision 6).

``filter_by`` / ``slice_by`` must apply their criteria to EVERY substitutive
level and preserve the pyramid — never silently collapse to the default level.
A raw ``filter(mask)`` (sized to the default level) can't be applied per-level,
so it must warn loudly instead of silently dropping coarser levels.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel


def _sub(
    n: int, seed: int = 0, amp_lo: float = 0.1, amp_hi: float = 1.0
) -> AdditiveSubLOD:
    rng = np.random.default_rng(seed)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
    return AdditiveSubLOD(
        centers=rng.uniform(0, 100, size=(n, 3)).astype(np.float32),
        amplitudes=rng.uniform(amp_lo, amp_hi, size=(n,)).astype(np.float32),
        cholesky_factors=chol,
    )


def _pyramid() -> GSplatData:
    return GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(additive_sublods=[_sub(100, 0)], level_index=0),
            SubstitutiveLevel(
                additive_sublods=[_sub(40, 1)], compression_factor=4, level_index=1
            ),
            SubstitutiveLevel(
                additive_sublods=[_sub(10, 2)], compression_factor=16, level_index=2
            ),
        ]
    )


def test_filter_by_preserves_all_substitutive_levels():
    pyr = _pyramid()
    assert pyr.n_substitutive == 3
    # A bbox that keeps roughly half of each level.
    out = pyr.filter_by(bbox=[(0.0, 50.0), (0.0, 100.0), (0.0, 100.0)])
    # The pyramid is preserved (not collapsed to a single level).
    assert out.n_substitutive == 3
    # Per-level metadata survives.
    assert [lvl.compression_factor for lvl in out.substitutive_levels] == [1, 4, 16]
    assert [lvl.level_index for lvl in out.substitutive_levels] == [0, 1, 2]
    # Every level was actually filtered (fewer or equal splats than the source).
    for src, dst in zip(pyr.substitutive_levels, out.substitutive_levels):
        assert dst.n_splats_total <= src.n_splats_total
    # And the bbox was applied to each level's centers.
    for lvl in out.substitutive_levels:
        c = lvl.additive_sublods[0].centers
        if c.shape[0]:
            assert c[:, 0].max() <= 50.0 + 1e-6


def test_cull_preserves_all_substitutive_levels():
    """cull() must apply per-level and preserve the pyramid (decision 6) —
    not collapse to the default level via self.filter() (the regression the
    review caught: cull's own warning text advertised it as pyramid-preserving
    while it silently dropped coarser levels)."""
    pyr = _pyramid()
    assert pyr.n_substitutive == 3
    out = pyr.cull(method="cumulative", retention=0.9)
    # All three substitutive levels survive (was 1 before the fix).
    assert out.n_substitutive == 3
    # Per-level metadata is preserved.
    assert [lvl.compression_factor for lvl in out.substitutive_levels] == [1, 4, 16]
    assert [lvl.level_index for lvl in out.substitutive_levels] == [0, 1, 2]
    # Each level was actually culled (cumulative keeps <= the source count).
    for src, dst in zip(pyr.substitutive_levels, out.substitutive_levels):
        assert 0 < dst.n_splats_total <= src.n_splats_total
    assert out.stats.get("culled") is True


def test_cull_single_substitutive_unchanged_shape():
    single = GSplatData.from_additive_sublods([_sub(50, 0)])
    out = single.cull(method="cumulative", retention=1.0)  # keep all amplitude
    assert out.n_substitutive == 1


def test_slice_by_preserves_pyramid():
    pyr = _pyramid()
    sliced = pyr.slice_by([slice(0, 50), slice(None, None), slice(None, None)])
    assert sliced.n_substitutive == 3


def test_filter_by_single_substitutive_unchanged_shape():
    single = GSplatData.from_additive_sublods([_sub(50, 0)])
    out = single.filter_by(amplitude_min=0.0)  # keep all
    assert out.n_substitutive == 1
    assert out.n_splats == 50


def test_raw_filter_mask_warns_on_multi_substitutive():
    pyr = _pyramid()
    mask = np.ones(pyr.n_splats, dtype=bool)  # sized to default (finest) level
    with pytest.warns(UserWarning, match="default substitutive level"):
        out = pyr.filter(mask)
    # raw mask collapses to a single level (documented, now loud)
    assert out.n_substitutive == 1


def test_raw_filter_mask_no_warn_single_substitutive():
    single = GSplatData.from_additive_sublods([_sub(30, 0)])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning would fail
        out = single.filter(np.ones(30, dtype=bool))
    assert out.n_splats == 30
