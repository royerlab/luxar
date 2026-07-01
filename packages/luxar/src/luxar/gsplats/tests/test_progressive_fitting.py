"""Tests for progressive Gaussian splat fitting.

The progressive fitter returns a *single-LOD* :class:`GSplatData`
containing all splats from all passes.  Per-pass detail is surfaced
through the ``on_pass_complete`` callback and the
``stats['pass_stats']`` / ``stats['pass_psnrs']`` / ``stats['n_passes']``
fields on the returned dataset.  To build a streamable LOD ladder, run
:func:`luxar.gsplats.lod.make_additive_lod` on the result.

Speed
-----
Most assertions probe disjoint facets of the same fit.  Those tests
share the ``shared_progressive_fit`` module-scoped fixture defined in
``conftest.py`` so the underlying fit runs once per module.  Tests whose
parameters drive a *binding* behavior (e.g. ``max_splats=150`` as a hard
cap) keep their own fit but pass ``residual_pass_min_iters=iters_per_pass``
to disable the 500-iter production floor for fast residual passes.

The production floor itself (default ``residual_pass_min_iters=500``) is
covered by ``test_residual_pass_min_iters_default_honored`` (marked
``slow``).
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import pytest

from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData

# Multi-pass progressive CPU fitting (tens of seconds per test). Slow → CI runs
# `-m "not slow"`; the full suite runs locally pre-push.
pytestmark = pytest.mark.slow


def _make_synthetic_volume(shape=(32, 32), seed=42):
    """Create a synthetic volume with blobs at multiple scales."""
    V = np.zeros(shape, dtype=np.float32)

    # Large blob
    center = np.array(shape) // 2
    for idx in np.ndindex(shape):
        dist = np.sqrt(sum((i - c) ** 2 for i, c in zip(idx, center)))
        V[idx] += np.exp(-(dist**2) / (2 * (shape[0] / 6) ** 2))

    # Small blob offset
    center2 = np.array(shape) // 4
    for idx in np.ndindex(shape):
        dist = np.sqrt(sum((i - c) ** 2 for i, c in zip(idx, center2)))
        V[idx] += 0.5 * np.exp(-(dist**2) / (2 * (shape[0] / 12) ** 2))

    return V


class TestProgressiveFitting:
    """Tests for fit_progressive_gaussian_splats.

    Tests that probe disjoint facets of the *same* multi-pass fit share
    the ``shared_progressive_fit`` fixture (see ``conftest.py``).  Tests
    whose parameters drive a binding behavior keep their own fits.
    """

    # ── Tests sharing the module-scoped fixture ──────────────────

    def test_basic_progressive_fit_2d(self, shared_progressive_fit):
        """Basic progressive fitting on a small 2D volume."""
        result = shared_progressive_fit.result
        assert isinstance(result, GSplatData)
        assert result.n_splats > 0
        # Single flattened LOD post-decoupling.
        assert result.n_additive_sublods == 1
        assert result.ndim == 2

    def test_multi_pass(self, shared_progressive_fit):
        """Verify multiple passes are executed (recorded in stats).

        The shared fixture uses ``max_splats=300, max_splats_per_pass=100,
        psnr_patience=0.01`` — chosen specifically to guarantee ≥2 passes.
        """
        result = shared_progressive_fit.result
        assert result.stats["n_passes"] >= 2
        assert result.n_splats > 0

    def test_per_pass_stats_recorded(self, shared_progressive_fit):
        """Per-pass stats are surfaced through ``stats['pass_stats']``.

        The progressive fitter no longer exposes per-pass AdditiveSubLOD
        intermediates on the returned dataset (it is flattened).  We
        therefore verify the per-pass detail through the dedicated
        ``pass_stats`` list, which mirrors what was previously stored
        on each LOD.
        """
        result = shared_progressive_fit.result
        n_passes = result.stats["n_passes"]
        pass_stats = result.stats["pass_stats"]
        assert isinstance(pass_stats, list)
        assert len(pass_stats) == n_passes
        for i, ps in enumerate(pass_stats):
            assert ps["pass_index"] == i

    def test_cumulative_passes_produce_positive_psnr(self, shared_progressive_fit):
        """Verify each progressive pass produces a positive PSNR value.

        Note: monotonic PSNR increase cannot be guaranteed on tiny test
        volumes because later passes may slightly degrade quality due to
        overshoot from few splats and few iterations.  We therefore only
        assert that every per-pass PSNR is positive (i.e. the fit is
        better than pure noise).
        """
        result = shared_progressive_fit.result
        if result.stats["n_passes"] >= 2:
            psnrs = result.stats["pass_psnrs"]
            assert all(p > 0 for p in psnrs)

    def test_callback_invoked(self, shared_progressive_fit):
        """Verify on_pass_complete callback is called once per pass.

        The shared fixture installs a recording callback; this test
        inspects the immutable ``callback_log`` tuple it produced.
        """
        result = shared_progressive_fit.result
        log = shared_progressive_fit.callback_log
        assert len(log) == result.stats["n_passes"]
        assert log[0][0] == 0  # First pass index is 0

    def test_stats(self, shared_progressive_fit):
        """Verify overall stats are populated.

        Shared fixture uses ``cull_retention=None`` for deterministic
        splat counts.
        """
        result = shared_progressive_fit.result
        assert result.stats["fitter_name"] == "progressive"
        assert result.stats["n_passes"] >= 1
        assert result.stats["n_splats"] == result.n_splats
        assert "time_seconds" in result.stats
        assert "psnr_db" in result.stats
        assert "stop_reason" in result.stats
        assert "pass_psnrs" in result.stats
        assert "pass_splats" in result.stats

    def test_save_load_roundtrip(self, shared_progressive_fit):
        """Verify the (single-LOD) result can be saved and loaded.

        Pure I/O on the frozen shared artifact — no mutation possible.
        """
        result = shared_progressive_fit.result
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            result.save(str(path), ordering="none")
            loaded = GSplatData.load(str(path), include_stats=True)

            assert loaded.n_splats == result.n_splats

    def test_adaptive_seed_reduction_tracked(self, shared_progressive_fit):
        """Per-pass ``seeds_requested`` / ``splats_after_culling`` are
        carried in ``stats['pass_stats']``.

        On the flattened return value, per-pass AdditiveSubLOD intermediates
        are not preserved; ``stats['pass_stats']`` is the canonical
        source of per-pass detail.  The shared fixture uses
        ``cull_retention=None`` so per-pass counts are deterministic.
        """
        result = shared_progressive_fit.result
        pass_stats = result.stats["pass_stats"]
        assert len(pass_stats) >= 1
        for ps in pass_stats:
            assert "seeds_requested" in ps
            assert "splats_after_culling" in ps

    # ── Tests with binding parameters (own fits) ─────────────────

    @pytest.mark.slow
    def test_max_splats_respected(self):
        """Verify max_splats limit is respected (binding parameter)."""
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=150,
            max_splats_per_pass=100,
            iters_per_pass=30,
            psnr_patience=0.01,
            residual_pass_min_iters=30,
            verbose=False,
        )
        assert result.n_splats <= 150

    @pytest.mark.slow
    def test_psnr_patience_stops_early(self):
        """Verify PSNR patience causes early stopping (binding parameter)."""
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=10000,  # Very high limit
            max_splats_per_pass=200,
            iters_per_pass=100,
            psnr_patience=5.0,  # Very high patience = stop early
            residual_pass_min_iters=100,
            verbose=False,
        )
        # Should stop after a few passes since ΔPSNR < 5 dB.
        assert result.stats["n_passes"] <= 5
        assert result.stats.get("stop_reason") in (
            "psnr_patience",
            "residual_negligible",
        )

    @pytest.mark.slow
    def test_max_passes_limits_passes(self):
        """Verify max_passes caps the number of passes (binding parameter)."""
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=10000,
            max_splats_per_pass=100,
            iters_per_pass=30,
            max_passes=2,
            psnr_patience=0.01,
            residual_pass_min_iters=30,
            verbose=False,
        )
        assert result.stats["n_passes"] <= 2
        # May stop by max_passes or psnr_patience (if residual thresholded to zero)
        assert result.stats.get("stop_reason") in (
            "max_passes",
            "psnr_patience",
            "residual_negligible",
        )

    def test_max_passes_zero_raises(self):
        """Verify max_passes=0 raises ValueError."""
        V = _make_synthetic_volume(shape=(32, 32))
        with pytest.raises(ValueError, match="max_passes must be >= 1"):
            fit_progressive_gaussian_splats(
                V,
                max_splats=100,
                max_splats_per_pass=50,
                max_passes=0,
                verbose=False,
            )

    @pytest.mark.slow
    def test_cull_retention_removes_weak_splats(self):
        """Verify post-fit cull_retention reduces splat count.

        Comparison fit — needs *two* runs, one with culling and one without.
        """
        V = _make_synthetic_volume(shape=(32, 32))
        result_culled = fit_progressive_gaussian_splats(
            V,
            max_splats=200,
            max_splats_per_pass=200,
            iters_per_pass=50,
            max_passes=1,
            cull_retention=0.5,
            residual_pass_min_iters=50,
            verbose=False,
        )
        result_no_cull = fit_progressive_gaussian_splats(
            V,
            max_splats=200,
            max_splats_per_pass=200,
            iters_per_pass=50,
            max_passes=1,
            cull_retention=None,
            residual_pass_min_iters=50,
            verbose=False,
        )
        assert result_culled.n_splats <= result_no_cull.n_splats

    # ── Production-floor regression guard ────────────────────────

    @pytest.mark.slow
    def test_residual_pass_min_iters_default_honored(self):
        """Regression guard: residual-pass floor MUST default to 500.

        Every *other* test in this file passes
        ``residual_pass_min_iters=iters_per_pass`` to opt out of the floor
        for speed.  Without this guard, the production code path (default
        floor active) would no longer be exercised at all.  Lowering the
        default silently would degrade production fit quality — residual
        passes fitting fine detail can need this many iters to converge.
        """
        # 1. Signature default — definitive guard against accidental change.
        sig = inspect.signature(fit_progressive_gaussian_splats)
        assert sig.parameters["residual_pass_min_iters"].default == 500, (
            "residual_pass_min_iters default must remain 500 — see the "
            "parameter docstring in fit_progressive_gsplats.py."
        )

        # 2. Behavioral check — the floor actually fires on residual passes.
        # iters_per_pass=20 is well below the floor; disable early-stop so
        # actual_iters reflects the n_iters budget the floor sets.
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=300,
            max_splats_per_pass=100,
            iters_per_pass=20,
            psnr_patience=0.01,
            cull_retention=None,
            early_stop_patience=None,  # disable → actual_iters tracks n_iters
            verbose=False,
            # residual_pass_min_iters left at default (500)
        )
        pass_stats = result.stats["pass_stats"]
        assert len(pass_stats) >= 2, (
            "need ≥2 passes to exercise the residual-pass floor"
        )
        for i, ps in enumerate(pass_stats[1:], start=1):
            assert ps["iterations"] >= 200, (
                f"Pass {i} ran only {ps['iterations']} iters — "
                "residual_pass_min_iters floor was not honored "
                "(expected ≥200; production floor is 500)."
            )
