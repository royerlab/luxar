"""Tests for the GSplats-specific LOD axis resolvers (``lod/gsplats.py``).

Covers ``resolve_substitutive_axis_gsplats`` (the ``lod_group=`` kwarg) and
``resolve_additive_axis_gsplats`` (the ``additive_lod=`` kwarg) against real
``GSplatData`` — the GSplats peers of ``resolve_additive_axis_points`` /
``resolve_additive_axis_lines`` (tested in ``test_points.py`` / ``test_lines.py``).

The geometry-agnostic kind=lod ``Group`` machinery (builder, validation,
``coverage_fractions``, display-type resolution) is exercised in
``test_lod_group.py``. End-to-end ``add_gsplats_from_data(lod_group=...)``
round-trips live in ``luxar.core.tests.group.test_group``.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.core.group.lod.gsplats import (
    resolve_additive_axis_gsplats,
    resolve_substitutive_axis_gsplats,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod import make_additive_lod, make_substitutive_lod


def _make_random_gsplat(n: int = 64, ndim: int = 3, seed: int = 0) -> GSplatData:
    """Random anisotropic Gaussian splats with overlap (single substitutive level)."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n, ndim)).astype(np.float32) * 1.5
    amplitudes = (np.abs(rng.standard_normal(n)) + 0.5).astype(np.float32)
    tril = ndim * (ndim + 1) // 2
    chol = (rng.standard_normal((n, tril)) * 0.1).astype(np.float32)
    diag_idx = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag_idx] = np.abs(chol[:, diag_idx]) + 0.4
    return GSplatData(centers=centers, amplitudes=amplitudes, cholesky_factors=chol)


@pytest.fixture(scope="module")
def flat() -> GSplatData:
    """A single-substitutive, single-additive dataset (``fit`` output shape)."""
    return _make_random_gsplat(n=64, seed=0)


@pytest.fixture(scope="module")
def pyramid() -> GSplatData:
    """A multi-substitutive pyramid (n_substitutive > 1) for stored-path tests."""
    return make_substitutive_lod(
        _make_random_gsplat(n=64, seed=1), levels=2, device="cpu"
    )


# ────────────────────────────────────────────────────────────────────────
# resolve_substitutive_axis_gsplats — the ``lod_group=`` kwarg
# ────────────────────────────────────────────────────────────────────────


class TestResolveSubstitutiveAxisGsplats:
    def test_none_is_passthrough(self, flat) -> None:
        data, cov = resolve_substitutive_axis_gsplats(flat, None)
        assert data is flat
        assert cov is None

    def test_true_requires_multi_substitutive(self, flat) -> None:
        with pytest.raises(ValueError, match="n_substitutive"):
            resolve_substitutive_axis_gsplats(flat, True)

    def test_true_passes_pyramid_through(self, pyramid) -> None:
        data, cov = resolve_substitutive_axis_gsplats(pyramid, True)
        assert data.n_substitutive == pyramid.n_substitutive
        assert cov is None

    def test_false_collapses_to_finest(self, pyramid) -> None:
        assert pyramid.n_substitutive > 1
        data, _ = resolve_substitutive_axis_gsplats(pyramid, False)
        assert data.n_substitutive == 1

    def test_false_single_level_noop(self, flat) -> None:
        data, _ = resolve_substitutive_axis_gsplats(flat, False)
        assert data is flat

    def test_dict_explicit_coverage_fractions(self, pyramid) -> None:
        data, cov = resolve_substitutive_axis_gsplats(
            pyramid, {"coverage_fractions": [0.0, 0.2, 1.0]}
        )
        assert cov == [0.0, 0.2, 1.0]

    def test_dict_non_monotonic_coverage_fractions_raises(self, pyramid) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            resolve_substitutive_axis_gsplats(
                pyramid, {"coverage_fractions": [0.0, 0.5, 0.1]}
            )

    def test_dict_coverage_fractions_out_of_range_raises(self, pyramid) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            resolve_substitutive_axis_gsplats(
                pyramid, {"coverage_fractions": [0.0, 2.0]}
            )

    def test_dict_empty_coverage_fractions_raises_clean_error(self, pyramid) -> None:
        # Empty explicit list → actionable ValueError, not an IndexError from the
        # [0]/[-1] range check (regression: deep-double-check).
        with pytest.raises(ValueError, match="non-empty"):
            resolve_substitutive_axis_gsplats(pyramid, {"coverage_fractions": []})

    def test_dict_stored_pyramid_rejects_compute_kwargs(self, pyramid) -> None:
        # Compute kwargs on an already-built pyramid (without recompute) must
        # not be silently ignored.
        with pytest.raises(ValueError, match="recompute"):
            resolve_substitutive_axis_gsplats(pyramid, {"levels": 2})

    def test_dict_computes_on_single_level(self, flat) -> None:
        data, _ = resolve_substitutive_axis_gsplats(
            flat, {"levels": 2, "device": "cpu"}
        )
        assert data.n_substitutive >= 2

    def test_invalid_spec_type_raises(self, flat) -> None:
        with pytest.raises(TypeError, match="None, bool, or dict"):
            resolve_substitutive_axis_gsplats(flat, 1.5)  # type: ignore[arg-type]


# ────────────────────────────────────────────────────────────────────────
# resolve_additive_axis_gsplats — the ``additive_lod=`` kwarg
# ────────────────────────────────────────────────────────────────────────


class TestResolveAdditiveAxisGsplats:
    def test_none_is_passthrough(self, flat) -> None:
        assert resolve_additive_axis_gsplats(flat, None) is flat

    def test_true_requires_existing_ladder(self, flat) -> None:
        # flat data has a single additive sub-LOD per substitutive level.
        with pytest.raises(ValueError, match="additive ladder"):
            resolve_additive_axis_gsplats(flat, True)

    def test_true_passes_existing_ladder_through(self, flat) -> None:
        laddered = make_additive_lod(flat, n_lods=3)
        result = resolve_additive_axis_gsplats(laddered, True)
        assert result.n_additive_sublods == 3

    def test_false_flattens_to_single_sublod(self, flat) -> None:
        laddered = make_additive_lod(flat, n_lods=3)
        result = resolve_additive_axis_gsplats(laddered, False)
        assert result.n_additive_sublods == 1

    def test_dict_computes_ladder(self, flat) -> None:
        result = resolve_additive_axis_gsplats(flat, {"n_lods": 3})
        assert result.n_additive_sublods == 3

    def test_dict_breakpoints_pass_through(self, flat) -> None:
        # Cumulative-count breakpoints flow through to make_additive_lod and
        # determine the level count (closes the convenience-path gap).
        result = resolve_additive_axis_gsplats(flat, {"breakpoints": [30, 64]})
        assert result.n_additive_sublods == 2

    def test_dict_counts_clamp_per_substitutive_level(self, pyramid) -> None:
        """REGRESSION: explicit ``counts:`` breakpoints larger than a COARSER
        substitutive level (smaller by K^s) used to abort the whole build with
        'largest breakpoint exceeds N'. They now clamp per level — mirroring the
        CLI per-part sites (recipes/pyramid/gsplat additive)."""
        assert pyramid.n_substitutive > 1
        coarsest = min(
            pyramid.at_substitutive(s).n_splats for s in range(pyramid.n_substitutive)
        )
        # A count that exceeds every coarser level but fits the finest (== the
        # full dataset N — larger would be a typo and abort, see the next test).
        big = pyramid.n_splats
        assert coarsest < big
        result = resolve_additive_axis_gsplats(
            pyramid, {"breakpoints": [1, big]}
        )  # must NOT raise
        # Every level keeps all its splats and gains a (clamped) ladder.
        assert result.n_substitutive == pyramid.n_substitutive
        for s in range(result.n_substitutive):
            lvl = result.at_substitutive(s)
            assert sum(sub.n_splats for sub in lvl.additive_sublods) == lvl.n_splats

    def test_dict_counts_exceeding_whole_group_raise(self, pyramid) -> None:
        """Counts exceeding the WHOLE group (the finest substitutive level) are
        a dataset-scale typo and must still abort loudly — the per-level clamp
        never masks them."""
        with pytest.raises(ValueError, match="exceeds N="):
            resolve_additive_axis_gsplats(
                pyramid, {"breakpoints": [pyramid.n_splats + 1]}
            )

    def test_dict_stream_breakpoints_per_level(self, pyramid) -> None:
        """A ``stream:<c>`` spec sizes each substitutive level's ladder against
        ITS OWN N through the convenience API."""
        result = resolve_additive_axis_gsplats(pyramid, {"breakpoints": "stream:8"})
        for s in range(result.n_substitutive):
            lvl = result.at_substitutive(s)
            incs = [sub.n_splats for sub in lvl.additive_sublods]
            assert sum(incs) == lvl.n_splats
            assert lvl.additive_sublods[0].stats["lod_breakpoints_kind"] == "stream"

    def test_invalid_spec_type_raises(self, flat) -> None:
        with pytest.raises(TypeError, match="None, bool, or dict"):
            resolve_additive_axis_gsplats(flat, 1.5)  # type: ignore[arg-type]

    def test_spatial_method_rejected(self, flat) -> None:
        """B8-G3/[P8]: documents the deliberate cross-geometry asymmetry —
        unlike Points/Lines (which order additive LODs spatially and accept
        ``method='poisson-disk'``/``'spatial-uniform'``), the GSplats additive
        ladder is energy/greedy-based. A spatial method name is forwarded to
        ``make_additive_lod`` and rejected, rather than silently ignored."""
        with pytest.raises(ValueError, match="method must be one of"):
            resolve_additive_axis_gsplats(flat, {"method": "poisson-disk"})
