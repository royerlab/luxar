"""Shared fixtures for gsplats tests.

The progressive / tiled-progressive fitters are the slowest paths in the
suite.  This conftest groups tests that probe disjoint facets of a single
fit into ``module``-scoped fixtures so the fit runs once per module rather
than once per test.  Combined with the production-safe
``residual_pass_min_iters`` knob (default 500 in production; lowered to
``iters_per_pass`` here), this drops the gsplats fitting wallclock from
~14 min to ~1 min without removing any assertion.

The floor itself is still exercised by the slow-marked test
``test_residual_pass_min_iters_default_honored`` in
``test_progressive_fitting.py``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_random_seed():
    """Safety net for tests that rely on global RNG state."""
    seed = 42
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    yield


# ---------------------------------------------------------------------------
# Shared progressive fit
# ---------------------------------------------------------------------------


def _make_synthetic_volume(shape=(32, 32), seed=42):
    """Two-blob synthetic 2D volume (kept in sync with test_progressive_fitting)."""
    V = np.zeros(shape, dtype=np.float32)

    center = np.array(shape) // 2
    for idx in np.ndindex(shape):
        dist = np.sqrt(sum((i - c) ** 2 for i, c in zip(idx, center)))
        V[idx] += np.exp(-(dist**2) / (2 * (shape[0] / 6) ** 2))

    center2 = np.array(shape) // 4
    for idx in np.ndindex(shape):
        dist = np.sqrt(sum((i - c) ** 2 for i, c in zip(idx, center2)))
        V[idx] += 0.5 * np.exp(-(dist**2) / (2 * (shape[0] / 12) ** 2))

    return V


class SharedProgressiveFit(NamedTuple):
    """Result of the module-scoped progressive fit.

    ``callback_log`` is a tuple (immutable) of ``(pass_index, n_splats, psnr)``
    triples — one entry per pass — recorded by an ``on_pass_complete``
    callback handed to the fitter.  Tests should treat both fields as
    read-only.
    """

    result: Any  # GSplatData
    callback_log: tuple[tuple[int, int, float], ...]


@pytest.fixture(scope="module")
def shared_progressive_fit() -> SharedProgressiveFit:
    """One progressive fit shared by tests that check disjoint result facets.

    Fixed parameters (chosen to satisfy every dependent test simultaneously):

    * ``max_splats=300, max_splats_per_pass=100`` — guarantees ≥2 passes,
      required by ``test_multi_pass``.
    * ``iters_per_pass=30, residual_pass_min_iters=30`` — opt out of the
      production 500-iter floor for fast tests.
    * ``psnr_patience=0.01`` — low so ΔPSNR doesn't dominate.
    * ``cull_retention=None`` — required by ``test_stats`` and
      ``test_adaptive_seed_reduction_tracked`` for deterministic counts.
    * ``on_pass_complete`` records into ``callback_log`` for
      ``test_callback_invoked``.

    Do not parametrize this fixture or change its parameters without
    re-checking every test in ``test_progressive_fitting.py`` that consumes
    it — the binding parameter values are load-bearing for those assertions.
    """
    if not HAS_TORCH:
        pytest.skip("torch not available")

    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
    from luxar.gsplats.gsplat_data import AdditiveSubLOD

    volume = _make_synthetic_volume(shape=(32, 32))
    log: list[tuple[int, int, float]] = []

    def _record(pass_idx: int, lod: AdditiveSubLOD, psnr: float) -> None:
        log.append((pass_idx, lod.n_splats, psnr))

    result = fit_progressive_gaussian_splats(
        volume,
        max_splats=300,
        max_splats_per_pass=100,
        iters_per_pass=30,
        psnr_patience=0.01,
        cull_retention=None,
        residual_pass_min_iters=30,
        on_pass_complete=_record,
        verbose=False,
    )
    return SharedProgressiveFit(result=result, callback_log=tuple(log))


# ---------------------------------------------------------------------------
# Shared tiled-progressive fit
# ---------------------------------------------------------------------------


class SharedTiledProgressiveFit(NamedTuple):
    """Result of the module-scoped tiled+progressive fit (read-only)."""

    result: Any  # GSplatData


@pytest.fixture(scope="module")
def shared_tiled_progressive_fit() -> SharedTiledProgressiveFit:
    """One tiled+progressive fit shared by the two TestTiledProgressive tests.

    The two original tests (``test_tiled_progressive_basic``,
    ``test_tiled_progressive_stats``) used parameter sets that differed only
    in ``seeds`` (150 vs 100); neither's assertions depend on that
    difference.  This fixture uses ``seeds=150`` (the larger of the two)
    plus ``residual_pass_min_iters=30`` for fast residual passes.
    """
    if not HAS_TORCH:
        pytest.skip("torch not available")

    from luxar.gsplats.fit_tiled_gsplats import fit_tiled

    volume = np.random.RandomState(42).rand(32, 32).astype(np.float32)
    result = fit_tiled(
        volume,
        tile_size=16,
        overlap=4,
        progressive=True,
        max_splats_per_pass=50,
        psnr_patience=0.01,
        max_passes=3,
        seeds=150,
        iters_per_pass=30,
        residual_pass_min_iters=30,
        verbose=False,
    )
    return SharedTiledProgressiveFit(result=result)
