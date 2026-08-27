"""Finalize-time ``amplitude_data_range`` harmonization (#1691).

Covers the two writer-stamped mass statistics
(:func:`~luxar.io._compiler.gsplat_assembly.compute_amplitude_mass_stats`) and
the pass that consumes them
(:func:`~luxar.io._compiler.finalize.amplitude_window.
harmonize_gsplat_amplitude_windows`) across every gsplat structure shape: a
substitutive ``kind=lod`` ladder (writer-named and hand-authored), levels
carrying their default ``additive_<i>`` stream ladders, a ``kind=partition``
(in-memory and streamed, with flat and with ``kind=lod`` parts), the
``adaptive`` partition-of-ladders, the ``overview`` lod-over-partition, a ladder
whose finest child is Points, the no-stats legacy fallback, the
scale/degeneracy guards, and the scene-compiler path.

Two of them are built by the SHIPPED producer
(:mod:`luxar.gsplats.lod.recipes`) rather than by hand, so the pass stays
coupled to what ``gsplat lod`` actually writes — including the
constant-amplitude ``levels`` ladder that ``gsplat import`` produces, whose
finest level is windowed degenerately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray

from luxar._zarr_compat import open_group
from luxar.core.dimensions import Dimensions
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel
from luxar.gsplats.io.save_gsplats import write_gsplats_tree, write_partition_streaming
from luxar.gsplats.lod.recipes import RecipeParams, build_recipe
from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatNode, GSplatPartition
from luxar.io._compiler.finalize.amplitude_window import (
    _SCALE_BOUND,
    harmonize_gsplat_amplitude_windows,
)
from luxar.io._compiler.gsplat_assembly import compute_amplitude_mass_stats
from luxar.io.compiler import LuxarZarrCompiler

_DIAG_3D = [0, 2, 5]  # packed lower-triangular diagonal indices for ndim == 3


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────


def _chol_from_dets(dets: NDArray[np.float64]) -> NDArray[np.float32]:
    """Isotropic 3D Cholesky factors with ``Π diag(L) == dets`` per row."""
    n = dets.shape[0]
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, _DIAG_3D] = np.cbrt(dets)[:, None].astype(np.float32)
    return chol


def _closed_form(
    amps: NDArray[np.float32], chol: NDArray[np.float32]
) -> Tuple[float, float]:
    """``(Σ a·Πdiag(L), Σ a²·Πdiag(L) / Σ a·Πdiag(L))`` computed independently."""
    det = np.prod(np.asarray(chol[:, _DIAG_3D], dtype=np.float64), axis=1)
    a = np.asarray(amps, dtype=np.float64)
    mass = float(np.sum(a * det))
    return mass, float(np.sum(a * a * det)) / mass


def _authored_window(amps: NDArray[np.float32]) -> Tuple[float, float]:
    """The window ``write_gsplat_arrays`` derives for a splat set: ``[min, p99.9]``."""
    a = np.asarray(amps, dtype=np.float32)
    lo = float(a.min())
    hi = float(np.percentile(a, 99.9))
    return lo, hi if hi > lo else float(a.max())


def _mixture_level(index: int, *, seed: int = 3) -> AdditiveSubLOD:
    """One substitutive level of a sparse, heavy-tailed 3D dataset.

    Level ``index`` (0 = FINEST) holds ``4**-index`` of the finest level's
    splats and the same total mass in each of two populations — the shape a
    mass-conserving merge produces:

    * a dim **background** that carries almost all the mass and merges into
      *wider* representatives at unchanged amplitude;
    * a thin **bright tail** that carries almost no mass and merges into
      representatives ``4**index`` times brighter at unchanged width.

    So the per-level ``p99.9`` (which lives in the tail) climbs steeply while
    the mass-weighted mean amplitude (which the background pins) barely moves —
    exactly the divergence #1691 is about.
    """
    rng = np.random.default_rng(seed + index)
    k = 4**index
    n_bright = max(3, 40 // k)
    n_bg = max(1, 4000 // k) - n_bright
    bg_amp = np.full(n_bg, 0.05, dtype=np.float64)
    bright_amp = np.geomspace(1.0, 10.0, n_bright) * k
    amps = np.concatenate([bg_amp, bright_amp])
    dets = np.concatenate(
        [
            np.full(n_bg, 20000.0 / float(np.sum(bg_amp))),
            np.full(n_bright, 16.0 / float(np.sum(bright_amp))),
        ]
    )
    n = amps.shape[0]
    return AdditiveSubLOD(
        centers=rng.uniform(0.0, 64.0, size=(n, 3)).astype(np.float32),
        amplitudes=amps.astype(np.float32),
        cholesky_factors=_chol_from_dets(dets),
    )


def _mixture_ladder(n_levels: int = 3, *, seed: int = 3) -> List[AdditiveSubLOD]:
    """The ladder's sub-LODs, FINEST FIRST."""
    return [_mixture_level(i, seed=seed) for i in range(n_levels)]


def _simple_level(index: int, *, seed: int = 11) -> AdditiveSubLOD:
    """A mild substitutive level: half the splats and ~1.1x the amplitudes.

    Deliberately gentle, so even a 12-rung ladder keeps every mass-weighted
    amplitude ratio inside the pass's ``_SCALE_BOUND`` guard and the test reads
    the scaling path rather than the fallback.
    """
    rng = np.random.default_rng(seed + index)
    n = max(8, 4096 >> index)
    amps = ((1.1**index) * (0.05 + rng.pareto(2.0, size=n))).astype(np.float32)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, _DIAG_3D] = rng.uniform(0.5, 2.0, size=(n, 3)).astype(np.float32)
    return AdditiveSubLOD(
        centers=rng.uniform(0.0, 64.0, size=(n, 3)).astype(np.float32),
        amplitudes=amps,
        cholesky_factors=chol,
    )


def _simple_ladder(n_levels: int, *, seed: int = 11) -> List[AdditiveSubLOD]:
    """``n_levels`` mild sub-LODs, FINEST FIRST."""
    return [_simple_level(i, seed=seed) for i in range(n_levels)]


def _lod_group_coarsest_first(sublods: List[AdditiveSubLOD]) -> GSplatLodGroup:
    """A ``kind=lod`` group from finest-first levels (on-disk order is reversed)."""
    return GSplatLodGroup(
        children=[GSplatLeaf(additive_sublods=[s]) for s in reversed(sublods)]
    )


def _plain_leaf(n: int, *, seed: int, amp_scale: float = 1.0) -> GSplatLeaf:
    rng = np.random.default_rng(seed)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, _DIAG_3D] = rng.uniform(0.5, 2.0, size=(n, 3)).astype(np.float32)
    return GSplatLeaf(
        additive_sublods=[
            AdditiveSubLOD(
                centers=rng.uniform(0.0, 64.0, size=(n, 3)).astype(np.float32),
                amplitudes=(amp_scale * (rng.pareto(1.5, size=n) + 0.01)).astype(
                    np.float32
                ),
                cholesky_factors=chol,
            )
        ]
    )


def _mild_leaf(index: int, *, seed: int) -> GSplatLeaf:
    """A one-sub-LOD leaf holding :func:`_simple_level` ``index``."""
    return GSplatLeaf(additive_sublods=[_simple_level(index, seed=seed)])


def _uniform_amplitude_leaf(n: int, *, seed: int, amplitude: float) -> GSplatLeaf:
    """A leaf whose amplitudes are all ``amplitude``.

    ``write_gsplat_arrays`` cannot derive a usable window from a constant set —
    ``p99.9 == min``, so it falls back to ``max`` and stamps the DEGENERATE
    ``[x, x]``. With ``amplitude == 0`` it also stamps ``amplitude_mass`` and
    ``amplitude_mass_weighted_mean`` as ``0.0`` (present, not absent), which is
    the reachable stand-in for an "empty" leaf — a 0-splat write raises
    ``ValidationError``, so that case cannot be produced at all.
    """
    rng = np.random.default_rng(seed)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, _DIAG_3D] = rng.uniform(0.5, 2.0, size=(n, 3)).astype(np.float32)
    return GSplatLeaf(
        additive_sublods=[
            AdditiveSubLOD(
                centers=rng.uniform(0.0, 64.0, size=(n, 3)).astype(np.float32),
                amplitudes=np.full(n, amplitude, dtype=np.float32),
                cholesky_factors=chol,
            )
        ]
    )


def _pareto_data(n: int, *, seed: int) -> GSplatData:
    """A flat heavy-tailed 3D set, ready for :func:`build_recipe`."""
    rng = np.random.default_rng(seed)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, _DIAG_3D] = rng.uniform(0.5, 2.0, size=(n, 3)).astype(np.float32)
    return GSplatData(
        centers=rng.uniform(0.0, 64.0, size=(n, 3)).astype(np.float32),
        amplitudes=(0.05 + rng.pareto(2.0, size=n)).astype(np.float32),
        cholesky_factors=chol,
    )


def _constant_amplitude_data(n: int = 4000, *, seed: int = 0) -> GSplatData:
    """A flat set with CONSTANT amplitudes — the ``gsplat import`` shape.

    ``gsplats/interop/_convert.py`` sets ``amplitudes = np.ones(n)`` for EVERY
    imported classical splat file (INRIA PLY, ``.splat``, SPZ, SuperSplat), so
    ``gsplat import`` → ``gsplat lod --recipe levels`` produces exactly this.
    """
    rng = np.random.default_rng(seed)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, _DIAG_3D] = rng.uniform(0.5, 2.0, size=(n, 3)).astype(np.float32)
    return GSplatData(
        centers=rng.uniform(0.0, 64.0, size=(n, 3)).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )


def _leaf_amps(leaf: GSplatLeaf) -> NDArray[np.float32]:
    """Every amplitude in a leaf, concatenated over its additive ladder."""
    return np.concatenate(
        [np.asarray(s.amplitudes, dtype=np.float32) for s in leaf.additive_sublods]
    )


def _write(
    tmp_path: Path, node: GSplatNode, name: str = "t.gsplats.zarr"
) -> zarr.Group:
    """Write a node tree to a ``.gsplats.zarr`` under ``tmp_path`` and reopen it."""
    path = tmp_path / name
    write_gsplats_tree(
        path, node, ordering="none", encoding_mode=EncodingMode.PRECISION
    )
    return zarr.open_group(str(path), mode="r")


def _attrs(group: zarr.Group, path: str) -> Dict[str, object]:
    return dict(group[path].attrs)


def _window(group: zarr.Group, path: str) -> List[float]:
    return [float(v) for v in _attrs(group, path)["amplitude_data_range"]]  # type: ignore[union-attr]


def _mwma(group: zarr.Group, path: str) -> float:
    return float(_attrs(group, path)["amplitude_mass_weighted_mean"])  # type: ignore[arg-type]


def _disk_amps(group: zarr.Group, path: str) -> NDArray[np.float32]:
    """The amplitudes actually stored at ``path`` (PRECISION → exact float32)."""
    return np.asarray(group[f"{path}/amplitudes"][:], dtype=np.float32)


def _child_count(group: zarr.Group, path: str, prefix: str) -> int:
    """How many ``prefix<i>`` child groups the node at ``path`` has."""
    node = group[path] if path else group
    return sum(1 for k in node.group_keys() if str(k).startswith(prefix))


def _clamped(scale: float) -> float:
    """``scale`` as the pass applies it — clamped into the bound, never reset."""
    return min(max(scale, 1.0 / _SCALE_BOUND), _SCALE_BOUND)


# ────────────────────────────────────────────────────────────────────────
# 1. The two statistics themselves
# ────────────────────────────────────────────────────────────────────────


def test_mass_statistics_are_the_closed_form(tmp_path: Path) -> None:
    """``amplitude_mass`` / ``amplitude_mass_weighted_mean`` on a hand-computable set."""
    amps = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    # Π diag(L) = 1, 2 and 4 respectively.
    chol = _chol_from_dets(np.array([1.0, 2.0, 4.0]))
    leaf = GSplatLeaf(
        additive_sublods=[
            AdditiveSubLOD(
                centers=np.array(
                    [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                    dtype=np.float32,
                ),
                amplitudes=amps,
                cholesky_factors=chol,
            )
        ]
    )
    root = _write(tmp_path, leaf)
    attrs = dict(root.attrs)

    # mass = 1·1 + 2·2 + 3·4 = 17 ; self-energy = 1·1 + 4·2 + 9·4 = 45.
    # (The float32 cube-root round-trip in ``_chol_from_dets`` is why this is
    # rel=1e-6 rather than exact.)
    assert float(attrs["amplitude_mass"]) == pytest.approx(17.0, rel=1e-6)
    assert float(attrs["amplitude_mass_weighted_mean"]) == pytest.approx(
        45.0 / 17.0, rel=1e-6
    )
    assert (
        float(attrs["amplitude_mass"]),
        float(attrs["amplitude_mass_weighted_mean"]),
    ) == pytest.approx(_closed_form(amps, chol), rel=1e-9)


@pytest.mark.parametrize(
    "amplitudes, chol_diag, n_splats, expected",
    [
        # Scalar (broadcast) amplitude: a=2 over Π diag(L) = 1, 2, 4.
        # mass = 2·7 = 14 ; self-energy = 4·7 = 28 ; mwma = 2.
        pytest.param(
            2.0,
            np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [4.0, 1.0, 1.0]]),
            3,
            (14.0, 2.0),
            id="scalar-amplitude",
        ),
        # Uniform single-row Cholesky broadcast over 3 splats (Π diag = 2).
        # mass = 2·(1+2+3) = 12 ; self-energy = 2·(1+4+9) = 28.
        pytest.param(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([[2.0, 1.0, 1.0]]),
            3,
            (12.0, 28.0 / 12.0),
            id="uniform-single-row-cholesky",
        ),
        # Empty set — nothing to integrate.
        pytest.param(
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3)),
            0,
            (0.0, 0.0),
            id="empty",
        ),
        # Non-positive mass (all-zero amplitudes) — no ratio to report.
        pytest.param(
            np.zeros((4,), dtype=np.float32),
            np.ones((4, 3)),
            4,
            (0.0, 0.0),
            id="zero-mass",
        ),
        # A negative Cholesky pivot must NOT cancel mass against its
        # neighbours: |Π diag| is taken, matching ``_subset_mass``/
        # ``_mass_score``. Without the abs() this set totals 0.0.
        pytest.param(
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            2,
            (2.0, 1.0),
            id="negative-cholesky-diagonal",
        ),
    ],
)
def test_compute_amplitude_mass_stats_edge_cases(
    amplitudes: object,
    chol_diag: NDArray[np.float64],
    n_splats: int,
    expected: Tuple[float, float],
) -> None:
    result = compute_amplitude_mass_stats(
        amplitudes,  # type: ignore[arg-type]
        np.asarray(chol_diag, dtype=np.float32),
        n_splats,
    )
    assert result == pytest.approx(expected, rel=1e-6, abs=1e-12)


def test_additive_ladder_parent_aggregates_the_statistics(tmp_path: Path) -> None:
    """A ladder's parent carries ``Σ mass`` and the mass-weighted mean of the means."""
    subs = [
        AdditiveSubLOD(
            centers=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            amplitudes=np.array([1.0, 2.0], dtype=np.float32),
            cholesky_factors=_chol_from_dets(np.array([1.0, 2.0])),
        ),
        AdditiveSubLOD(
            centers=np.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
            amplitudes=np.array([3.0, 4.0], dtype=np.float32),
            cholesky_factors=_chol_from_dets(np.array([4.0, 1.0])),
        ),
    ]
    root = _write(tmp_path, GSplatLeaf(additive_sublods=subs))
    all_amps = np.concatenate([np.asarray(s.amplitudes) for s in subs])
    all_chol = np.concatenate([np.asarray(s.cholesky_factors) for s in subs])
    mass, mwma = _closed_form(all_amps.astype(np.float32), all_chol.astype(np.float32))

    assert float(root.attrs["amplitude_mass"]) == pytest.approx(mass, rel=1e-9)
    assert float(root.attrs["amplitude_mass_weighted_mean"]) == pytest.approx(
        mwma, rel=1e-9
    )
    assert int(root.attrs["n_additive_sublods"]) == 2


def test_zero_mass_ladder_parent_still_stamps_the_statistics(tmp_path: Path) -> None:
    """A ladder parent stamps ``0.0`` / ``0.0``; it never OMITS them.

    Omitting made the node read back as statistic-LESS — indistinguishable from
    a legacy store — and one such part drops its whole enclosing structure to
    scale 1.0.
    """
    zero_ladder = GSplatLeaf(
        additive_sublods=[
            _uniform_amplitude_leaf(60, seed=101, amplitude=0.0).additive_sublods[0],
            _uniform_amplitude_leaf(60, seed=102, amplitude=0.0).additive_sublods[0],
        ]
    )
    root = _write(
        tmp_path,
        GSplatLodGroup(
            children=[
                _mild_leaf(3, seed=104),
                GSplatPartition(
                    children=[_mild_leaf(0, seed=103), zero_ladder],
                    max_elements=4096,
                ),
            ]
        ),
    )

    assert int(_attrs(root, "child_1/part_1")["n_additive_sublods"]) == 2  # type: ignore[arg-type]
    assert float(_attrs(root, "child_1/part_1")["amplitude_mass"]) == 0.0  # type: ignore[arg-type]
    assert _mwma(root, "child_1/part_1") == 0.0

    # …so the combine keeps its statistics and the coarse cap is still scaled.
    ref = _window(root, "child_1/part_0")
    scale = _mwma(root, "child_0") / _mwma(root, "child_1/part_0")
    assert scale != pytest.approx(1.0, rel=1e-3)
    assert _window(root, "child_0")[1] == pytest.approx(ref[1] * scale, rel=1e-6)


# ────────────────────────────────────────────────────────────────────────
# 2. A substitutive kind=lod ladder
# ────────────────────────────────────────────────────────────────────────


def test_lod_levels_share_one_window_scaled_by_the_mass_weighted_mean(
    tmp_path: Path,
) -> None:
    sublods = _mixture_ladder()  # finest first
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))

    n = len(sublods)
    # On disk child_0 is the COARSEST, so child_<n-1> is the finest.
    windows = [_window(root, f"child_{i}") for i in range(n)]
    mwmas = [_mwma(root, f"child_{i}") for i in range(n)]
    ref_window, ref_mwma = windows[-1], mwmas[-1]

    # THE reference identity: the finest level keeps EXACTLY the window its own
    # writer derived. Without this, every assertion below is of the relative
    # form ``windows[i] ≈ windows[-1]·ratio`` and stays true no matter which
    # child the pass picked as its reference.
    assert tuple(ref_window) == pytest.approx(
        _authored_window(np.asarray(sublods[0].amplitudes)), rel=1e-9
    )

    # Every level is the finest level's window times its own mass-weighted
    # amplitude ratio — the finest necessarily at scale 1.0.
    for i in range(n):
        scale = mwmas[i] / ref_mwma
        assert windows[i][0] == pytest.approx(ref_window[0] * scale, rel=1e-6)
        assert windows[i][1] == pytest.approx(ref_window[1] * scale, rel=1e-6)

    # And the on-disk statistic agrees with the closed form over the arrays.
    for i, sub in enumerate(reversed(sublods)):
        _, mwma = _closed_form(
            np.asarray(sub.amplitudes, dtype=np.float32),
            np.asarray(sub.cholesky_factors, dtype=np.float32),
        )
        assert mwmas[i] == pytest.approx(mwma, rel=1e-5)


def test_lod_window_tops_span_far_less_than_the_per_level_p999(
    tmp_path: Path,
) -> None:
    """The pre-fix windows were per-level ``p99.9``; measure against those."""
    sublods = _mixture_ladder()
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))

    # What the writer derived per level, i.e. the windows before this fix.
    before = [_authored_window(np.asarray(s.amplitudes))[1] for s in sublods]
    after = [_window(root, f"child_{i}")[1] for i in range(len(sublods))]

    # Pinned, not bounded: a plain VERBATIM implementation (every level on the
    # finest window, spread 1.0) also satisfies ``after < before / 2``, so a
    # loose bound would not test the mass-weighted scaling at all.
    assert max(before) / min(before) == pytest.approx(16.811, rel=1e-3)
    assert max(after) / min(after) == pytest.approx(2.7579, rel=1e-3)


def test_twelve_level_ladder_references_the_child_index_finest_level(
    tmp_path: Path,
) -> None:
    """A >=10-level ladder, ordered by ``child_index`` (rule (a)).

    Every store a Python producer writes carries ``child_index``, so this is the
    path the shipped ladders take — the ``child_<i>``-suffix and sorted-name
    fallbacks are exercised separately below. What it pins is that the reference
    is the true finest level (most splats, last ``child_index``), not whichever
    child a name sort would have put last.
    """
    n = 12
    sublods = _simple_ladder(n)  # finest first
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))

    counts = [int(_attrs(root, f"child_{i}")["n_splats"]) for i in range(n)]  # type: ignore[arg-type]
    # The finest level is the last child AND the one with the most splats.
    assert counts[-1] == max(counts)
    assert counts == sorted(counts)

    # The reference is that level, and it keeps its own authored window.
    assert tuple(_window(root, f"child_{n - 1}")) == pytest.approx(
        _authored_window(np.asarray(sublods[0].amplitudes)), rel=1e-9
    )
    # An alphabetical sort would have made child_9 the reference; it does not
    # keep its own window (it is scaled off the finest).
    ref_mwma = _mwma(root, f"child_{n - 1}")
    for i in range(n):
        assert _window(root, f"child_{i}")[1] == pytest.approx(
            _window(root, f"child_{n - 1}")[1] * (_mwma(root, f"child_{i}") / ref_mwma),
            rel=1e-6,
        )
    assert _window(root, "child_9") != pytest.approx(
        _authored_window(np.asarray(sublods[n - 1 - 9].amplitudes)), rel=1e-3
    )


def test_ladder_without_child_index_falls_back_to_the_numeric_suffix(
    tmp_path: Path,
) -> None:
    """Rule (b): no ``child_index``, so the ``child_<i>`` digits order the levels.

    Unreachable from a Python producer (``core/node/node.py`` always stamps
    ``child_index``), so it is reached here by deleting the attr — the point
    being that ``child_10``/``child_11`` must not sort before ``child_2``.
    """
    n = 12
    sublods = _simple_ladder(n)  # finest first
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))

    # Drop child_index AND restore each level's own authored window, so the
    # re-run starts from the pre-fix state and the donor is observable.
    edits: Dict[str, Dict[str, object]] = {
        f"child_{i}": {
            "child_index": None,
            "amplitude_data_range": list(
                _authored_window(np.asarray(sublods[n - 1 - i].amplitudes))
            ),
        }
        for i in range(n)
    }
    _, reread = _rewrite_and_reopen(root, edits)

    # child_11 donated (it keeps its own window), not the name-sorted child_9.
    assert tuple(_window(reread, f"child_{n - 1}")) == pytest.approx(
        _authored_window(np.asarray(sublods[0].amplitudes)), rel=1e-9
    )
    assert _window(reread, "child_9")[1] != pytest.approx(
        _authored_window(np.asarray(sublods[n - 1 - 9].amplitudes))[1], rel=1e-3
    )


def test_ladder_with_neither_index_nor_child_names_sorts_by_name(
    tmp_path: Path,
) -> None:
    """Rule (c): author-named children with no ``child_index`` — sorted name wins.

    The last resort, assuming alphabetically-last is finest (the convention
    ``finalize/lod_backfill.py`` already uses). Also unreachable from a Python
    producer; reached here by deleting ``child_index`` from a hand-authored
    ladder whose names happen to sort coarsest→finest.
    """
    names = ["level_a", "level_b", "level_c"]  # coarsest → finest
    root = _hand_built_ladder(tmp_path / "sorted.luxar.zarr", names)

    edits: Dict[str, Dict[str, object]] = {}
    for i, name in enumerate(names):
        sub = _simple_level(len(names) - 1 - i, seed=53)
        edits[f"ladder/{name}"] = {
            "child_index": None,
            "amplitude_data_range": list(_authored_window(np.asarray(sub.amplitudes))),
        }
    _, reread = _rewrite_and_reopen(root, edits)

    assert tuple(_window(reread, "ladder/level_c")) == pytest.approx(
        _authored_window(np.asarray(_simple_level(0, seed=53).amplitudes)), rel=1e-9
    )
    assert _window(reread, "ladder/level_a")[1] != pytest.approx(
        _authored_window(np.asarray(_simple_level(2, seed=53).amplitudes))[1], rel=1e-3
    )


def test_levels_carrying_stream_ladders_scale_their_sublods_too(
    tmp_path: Path,
) -> None:
    """The DEFAULT shape: each level is itself an ``additive_<i>`` stream ladder."""
    n_levels, n_sub = 3, 3
    levels: List[List[AdditiveSubLOD]] = [
        [_simple_level(i, seed=20 + 7 * j) for j in range(n_sub)]
        for i in range(n_levels)
    ]  # finest first
    group = GSplatLodGroup(
        children=[GSplatLeaf(additive_sublods=subs) for subs in reversed(levels)]
    )
    root = _write(tmp_path, group)

    for i in range(n_levels):
        level_window = _window(root, f"child_{i}")
        for j in range(n_sub):
            # Sub-LODs are prefix increments of the same content their parent
            # level describes — they take the level's window unscaled.
            assert _window(root, f"child_{i}/additive_{j}") == level_window

    # …and the levels themselves are still scaled relative to one another.
    ref = _window(root, f"child_{n_levels - 1}")
    ref_mwma = _mwma(root, f"child_{n_levels - 1}")
    for i in range(n_levels):
        assert _window(root, f"child_{i}")[1] == pytest.approx(
            ref[1] * (_mwma(root, f"child_{i}") / ref_mwma), rel=1e-6
        )
    assert _window(root, "child_0")[1] != pytest.approx(ref[1], rel=1e-3)


# ────────────────────────────────────────────────────────────────────────
# 3. A kind=partition
# ────────────────────────────────────────────────────────────────────────


def test_partition_parts_share_the_window_verbatim(tmp_path: Path) -> None:
    part = GSplatPartition(
        children=[
            _plain_leaf(120, seed=1, amp_scale=10.0),
            _plain_leaf(120, seed=2, amp_scale=0.1),
            _plain_leaf(120, seed=3, amp_scale=1.0),
        ],
        max_elements=256,
    )
    root = _write(tmp_path, part)
    windows = [_window(root, f"part_{i}") for i in range(3)]
    assert windows[0] == windows[1] == windows[2]


def test_partition_top_is_the_count_weighted_mean_not_the_max(tmp_path: Path) -> None:
    """A part's own top is its own ``p99.9``; the MAX of N of them grows with N."""
    leaves = [
        _plain_leaf(1000, seed=11, amp_scale=1.0),
        _plain_leaf(1000, seed=12, amp_scale=1.0),
        _plain_leaf(40, seed=13, amp_scale=50.0),  # small but very bright
    ]
    root = _write(tmp_path, GSplatPartition(children=leaves, max_elements=4096))

    tops = [_authored_window(_leaf_amps(leaf))[1] for leaf in leaves]
    counts = [int(_leaf_amps(leaf).shape[0]) for leaf in leaves]
    expected_hi = sum(t * c for t, c in zip(tops, counts)) / sum(counts)
    expected_lo = min(_authored_window(_leaf_amps(leaf))[0] for leaf in leaves)

    shared = _window(root, "part_0")
    assert shared[0] == pytest.approx(expected_lo, rel=1e-9)
    assert shared[1] == pytest.approx(expected_hi, rel=1e-6)
    # The old ``max`` rule would have taken the bright 40-splat tile's top and
    # blackened the two big ones; the weighted mean stays near them.
    assert shared[1] < 0.2 * max(tops)


def test_degenerate_part_window_is_excluded_from_the_pooled_top(
    tmp_path: Path,
) -> None:
    """A ``[x, x]`` part carries no window information — pooling it clips the rest."""
    n_const, n_real = 3000, 300
    const = _uniform_amplitude_leaf(n_const, seed=81, amplitude=1.0)
    real = _plain_leaf(n_real, seed=82, amp_scale=1.0)
    root = _write(tmp_path, GSplatPartition(children=[const, real], max_elements=8192))

    real_lo, real_top = _authored_window(_leaf_amps(real))
    shared = _window(root, "part_0")
    assert shared == _window(root, "part_1")

    # The real part's own top survives intact…
    assert shared[1] == pytest.approx(real_top, rel=1e-6)
    # …whereas admitting the degenerate top at its full 3000-splat weight would
    # have collapsed the window and over-brightened the real part several-fold.
    naive = (1.0 * n_const + real_top * n_real) / (n_const + n_real)
    assert naive < 0.5 * shared[1]
    # The degenerate part's own LO is still a real observation of the union.
    assert shared[0] == pytest.approx(min(1.0, real_lo), rel=1e-6)


def test_part_without_n_splats_is_pooled_rather_than_erased(tmp_path: Path) -> None:
    """An ABSENT ``n_splats`` means "weight unknown" (pool at 1), not "weight 0"."""
    leaves = [
        _plain_leaf(100, seed=91, amp_scale=1.0),
        _plain_leaf(100, seed=92, amp_scale=1.0),
        _plain_leaf(100, seed=93, amp_scale=9.0),
    ]
    root = _write(tmp_path, GSplatPartition(children=leaves, max_elements=4096))
    tops = [_authored_window(_leaf_amps(leaf))[1] for leaf in leaves]

    edits: Dict[str, Dict[str, object]] = {
        f"part_{i}": {
            "amplitude_data_range": list(_authored_window(_leaf_amps(leaves[i])))
        }
        for i in range(3)
    }
    edits["part_2"]["n_splats"] = None
    _, reread = _rewrite_and_reopen(root, edits)

    pooled = (tops[0] * 100 + tops[1] * 100 + tops[2] * 1) / 201
    assert _window(reread, "part_0")[1] == pytest.approx(pooled, rel=1e-6)
    # Weighting the attr-less part 0 would have dropped it from the pool.
    assert _window(reread, "part_0")[1] > (tops[0] * 100 + tops[1] * 100) / 200


def test_write_partition_streaming_harmonizes_its_parts(tmp_path: Path) -> None:
    """The streaming (batch-fit merge) writer, the pass's second shipped call site."""
    leaves = [
        _plain_leaf(400, seed=31, amp_scale=8.0),
        _plain_leaf(400, seed=32, amp_scale=0.2),
        _plain_leaf(200, seed=33, amp_scale=1.0),
    ]

    def parts() -> Iterator[GSplatNode]:
        yield from leaves

    out = tmp_path / "streamed.gsplats.zarr"
    n_written = write_partition_streaming(
        out,
        parts,
        max_elements=4096,
        ordering="none",
        encoding_mode=EncodingMode.PRECISION,
    )
    assert n_written == 3

    root = open_group(str(out), mode="r")
    windows = [_window(root, f"part_{i}") for i in range(3)]
    assert windows[0] == windows[1] == windows[2]

    tops = [_authored_window(_leaf_amps(leaf))[1] for leaf in leaves]
    counts = [int(_leaf_amps(leaf).shape[0]) for leaf in leaves]
    assert windows[0][1] == pytest.approx(
        sum(t * c for t, c in zip(tops, counts)) / sum(counts), rel=1e-6
    )
    # Not merely "all equal": each part's own authored top really did differ.
    assert max(tops) / min(tops) > 5.0


def test_write_partition_streaming_with_lod_parts(tmp_path: Path) -> None:
    """The shape ``batch-fit merge --recipe`` writes: each part is a ladder.

    The flat-leaf streaming case above never enters the ``kind=lod`` branch of
    the assignment walk, so the per-level scaling under a streamed part was
    untested.
    """
    n = 3
    ladders = [
        _lod_group_coarsest_first(_mixture_ladder(n, seed=seed)) for seed in (3, 17)
    ]

    def parts() -> Iterator[GSplatNode]:
        yield from ladders

    out = tmp_path / "streamed_lod.gsplats.zarr"
    assert (
        write_partition_streaming(
            out,
            parts,
            max_elements=8192,
            ordering="none",
            encoding_mode=EncodingMode.PRECISION,
        )
        == 2
    )

    root = open_group(str(out), mode="r")
    finest = [_window(root, f"part_{p}/child_{n - 1}") for p in range(2)]
    assert finest[0] == finest[1]

    for p in range(2):
        ref = _window(root, f"part_{p}/child_{n - 1}")
        ref_mwma = _mwma(root, f"part_{p}/child_{n - 1}")
        for i in range(n):
            scale = _clamped(_mwma(root, f"part_{p}/child_{i}") / ref_mwma)
            assert _window(root, f"part_{p}/child_{i}")[1] == pytest.approx(
                ref[1] * scale, rel=1e-6
            )
        # A coarse level really is scaled away from its part's reference.
        assert _window(root, f"part_{p}/child_0")[1] != pytest.approx(ref[1], rel=1e-3)


# ────────────────────────────────────────────────────────────────────────
# 4. `adaptive` — a partition of lod groups; `overview` — a lod over a partition
# ────────────────────────────────────────────────────────────────────────


def test_adaptive_parts_agree_at_the_finest_level_and_scale_within_a_part(
    tmp_path: Path,
) -> None:
    n = 3
    part = GSplatPartition(
        children=[
            _lod_group_coarsest_first(_mixture_ladder(n, seed=seed)) for seed in (3, 17)
        ],
        max_elements=4096,
    )
    root = _write(tmp_path, part)

    finest = [_window(root, f"part_{p}/child_{n - 1}") for p in range(2)]
    assert finest[0] == finest[1]

    for p in range(2):
        ref = _window(root, f"part_{p}/child_{n - 1}")
        ref_mwma = _mwma(root, f"part_{p}/child_{n - 1}")
        for i in range(n):
            scale = _mwma(root, f"part_{p}/child_{i}") / ref_mwma
            assert _window(root, f"part_{p}/child_{i}")[1] == pytest.approx(
                ref[1] * scale, rel=1e-6
            )
        # A coarse level really is scaled away from the reference.
        assert _window(root, f"part_{p}/child_0")[1] != pytest.approx(ref[1], rel=1e-3)


def test_overview_lod_over_a_partition_references_the_combined_parts(
    tmp_path: Path,
) -> None:
    """``overview``: the reference ``mwma`` comes from a COMBINE, not one leaf."""
    parts = [
        _plain_leaf(500, seed=41, amp_scale=1.0),
        _plain_leaf(300, seed=42, amp_scale=3.0),
    ]
    coarse = _plain_leaf(60, seed=43, amp_scale=2.0)
    root = _write(
        tmp_path,
        GSplatLodGroup(
            children=[coarse, GSplatPartition(children=parts, max_elements=4096)]
        ),
    )

    # The finest child is the partition: its parts share the combined window.
    part_windows = [_window(root, f"child_1/part_{i}") for i in range(2)]
    assert part_windows[0] == part_windows[1]

    tops = [_authored_window(_leaf_amps(leaf))[1] for leaf in parts]
    counts = [int(_leaf_amps(leaf).shape[0]) for leaf in parts]
    assert part_windows[0][1] == pytest.approx(
        sum(t * c for t, c in zip(tops, counts)) / sum(counts), rel=1e-6
    )
    assert part_windows[0][0] == pytest.approx(
        min(_authored_window(_leaf_amps(leaf))[0] for leaf in parts), rel=1e-9
    )

    # The coarse cap is scaled by its ratio against the COMBINED parts' mwma.
    masses = [
        float(_attrs(root, f"child_1/part_{i}")["amplitude_mass"]) for i in (0, 1)
    ]  # type: ignore[arg-type]
    mwmas = [_mwma(root, f"child_1/part_{i}") for i in (0, 1)]
    combined_mwma = sum(m * w for m, w in zip(masses, mwmas)) / sum(masses)
    scale = _mwma(root, "child_0") / combined_mwma
    assert 1.0 / _SCALE_BOUND <= scale <= _SCALE_BOUND
    assert _window(root, "child_0")[1] == pytest.approx(
        part_windows[0][1] * scale, rel=1e-6
    )


# ────────────────────────────────────────────────────────────────────────
# 5. Hand-authored structures — the child names are the AUTHOR's
# ────────────────────────────────────────────────────────────────────────


def _hand_built_ladder(out: Path, names: List[str]) -> zarr.Group:
    """A scene with one hand-built ``kind=lod`` group, children named by hand.

    ``Node.add_lod_group`` is public and the child names are the caller's —
    ``examples/partition_of_lod_example.py`` uses ``lod_coarse`` / ``lod_fine``
    — so the pass may not key off ``child_<i>``.
    """
    with LuxarZarrCompiler(out, enable_spatial_index=False) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        ladder = scene.add_lod_group("ladder", display_type="gsplats")
        for i, name in enumerate(names):  # coarsest first
            sub = _simple_level(len(names) - 1 - i, seed=53)
            ladder.add_gsplats(
                name=name,
                centers=np.asarray(sub.centers),
                amplitudes=np.asarray(sub.amplitudes),
                cholesky_factors=np.asarray(sub.cholesky_factors),
                coverage_fraction=float(i),
                partition=False,
            )
    return zarr.open_group(str(out), mode="r")


def test_hand_built_lod_group_with_author_named_children(tmp_path: Path) -> None:
    names = ["lod_coarse", "lod_mid", "lod_fine"]
    root = _hand_built_ladder(tmp_path / "hand.luxar.zarr", names)

    finest_amps = np.asarray(_simple_level(0, seed=53).amplitudes)
    # The FINEST (last-added) child is the reference and keeps its own window.
    assert tuple(_window(root, f"ladder/{names[-1]}")) == pytest.approx(
        _authored_window(finest_amps), rel=1e-9
    )
    ref = _window(root, f"ladder/{names[-1]}")
    ref_mwma = _mwma(root, f"ladder/{names[-1]}")
    for name in names:
        scale = _mwma(root, f"ladder/{name}") / ref_mwma
        assert _window(root, f"ladder/{name}")[1] == pytest.approx(
            ref[1] * scale, rel=1e-6
        )
    # The pass really did something: the coarse level no longer carries the
    # window its own writer derived.
    coarse_amps = np.asarray(_simple_level(len(names) - 1, seed=53).amplitudes)
    assert _window(root, f"ladder/{names[0]}")[1] != pytest.approx(
        _authored_window(coarse_amps)[1], rel=1e-3
    )


def test_lod_group_whose_finest_child_is_points(tmp_path: Path) -> None:
    """``add_points(substitutive_lod=…)``'s shape: gsplats levels, Points finest.

    The finest child carries no ``amplitude_data_range`` at all, so the pass has
    to fall back to the finest child that does — otherwise the whole structure
    is silently left alone.
    """
    out = tmp_path / "composed.luxar.zarr"
    coarse = [_simple_level(i, seed=61) for i in (2, 1, 0)]  # coarsest first
    rng = np.random.default_rng(7)
    with LuxarZarrCompiler(out, enable_spatial_index=False) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        ladder = scene.add_lod_group("composed", display_type="points")
        for i, sub in enumerate(coarse):
            ladder.add_gsplats(
                name=f"level_{i}",
                centers=np.asarray(sub.centers),
                amplitudes=np.asarray(sub.amplitudes),
                cholesky_factors=np.asarray(sub.cholesky_factors),
                coverage_fraction=float(i),
                partition=False,
            )
        ladder.add_points(
            "points_finest",
            rng.uniform(0.0, 64.0, size=(500, 3)).astype(np.float32),
            radii=0.5,
            coverage_fraction=float(len(coarse)),
            partition=False,
        )

    root = zarr.open_group(str(out), mode="r")
    assert "amplitude_data_range" not in _attrs(root, "composed/points_finest")

    # The donor is the finest GSPLATS level, which keeps its own window…
    donor = f"level_{len(coarse) - 1}"
    assert tuple(_window(root, f"composed/{donor}")) == pytest.approx(
        _authored_window(np.asarray(coarse[-1].amplitudes)), rel=1e-9
    )
    # …and its siblings are scaled off it (i.e. the pass did NOT bail out).
    ref = _window(root, f"composed/{donor}")
    ref_mwma = _mwma(root, f"composed/{donor}")
    for i, sub in enumerate(coarse):
        scale = _mwma(root, f"composed/level_{i}") / ref_mwma
        assert _window(root, f"composed/level_{i}")[1] == pytest.approx(
            ref[1] * scale, rel=1e-6
        )
    assert _window(root, "composed/level_0")[1] != pytest.approx(
        _authored_window(np.asarray(coarse[0].amplitudes))[1], rel=1e-3
    )


# ────────────────────────────────────────────────────────────────────────
# 5b. Structures built by the SHIPPED producer (gsplats.lod.recipes)
# ────────────────────────────────────────────────────────────────────────


def test_constant_amplitude_levels_recipe_falls_through_to_a_usable_donor(
    tmp_path: Path,
) -> None:
    """``gsplat import`` → ``gsplat lod --recipe levels``: the finest is ``[1, 1]``.

    The finest level holds the untouched constant amplitudes, so its writer
    window is degenerate and it cannot donate. Treating "finite" as "usable"
    made the pass pick it anyway, refuse every write, and leave the ladder on
    its per-level windows with no log line at all.
    """
    result = build_recipe(
        _constant_amplitude_data(),
        "levels",
        RecipeParams(
            compression_factor=4,
            levels=3,
            additive_ladders=False,
            quality_stamps=False,
            seed=1,
        ),
    )
    out = tmp_path / "const.gsplats.zarr"
    result.save(out, ordering="none", encoding_mode=EncodingMode.PRECISION)
    root = zarr.open_group(str(out), mode="r")

    n = _child_count(root, "", "child_")
    assert n >= 3
    # The finest level really is constant (hence degenerate) on disk.
    finest_amps = _disk_amps(root, f"child_{n - 1}")
    assert float(finest_amps.min()) == float(finest_amps.max())

    # The donor is the next-finest level, and it keeps its own authored window.
    donor = f"child_{n - 2}"
    assert tuple(_window(root, donor)) == pytest.approx(
        _authored_window(_disk_amps(root, donor)), rel=1e-6
    )

    ref, ref_mwma = _window(root, donor), _mwma(root, donor)
    for i in range(n):
        lo, hi = _window(root, f"child_{i}")
        assert hi > lo  # every level ends on a USABLE window
        scale = _clamped(_mwma(root, f"child_{i}") / ref_mwma)
        assert hi == pytest.approx(ref[1] * scale, rel=1e-6)
    # …including the constant finest level, which no longer reads as [1, 1].
    assert _window(root, f"child_{n - 1}")[1] > _window(root, f"child_{n - 1}")[0]


def test_recipe_built_adaptive_structure_is_harmonized(tmp_path: Path) -> None:
    """An ``adaptive`` partition-of-ladders straight out of :func:`build_recipe`."""
    result = build_recipe(
        _pareto_data(800, seed=2),
        "adaptive",
        RecipeParams(
            max_elements=300,
            compression_factor=4,
            levels=2,
            additive_ladders=False,
            quality_stamps=False,
            seed=1,
        ),
    )
    root = _write(tmp_path, result, name="adaptive.gsplats.zarr")

    n_parts = _child_count(root, "", "part_")
    assert n_parts >= 2
    n_levels = _child_count(root, "part_0", "child_")
    assert n_levels >= 2

    finest = f"child_{n_levels - 1}"
    shared = {tuple(_window(root, f"part_{p}/{finest}")) for p in range(n_parts)}
    assert len(shared) == 1  # every part's finest level on ONE window

    for p in range(n_parts):
        ref = _window(root, f"part_{p}/{finest}")
        ref_mwma = _mwma(root, f"part_{p}/{finest}")
        for i in range(n_levels):
            scale = _clamped(_mwma(root, f"part_{p}/child_{i}") / ref_mwma)
            assert _window(root, f"part_{p}/child_{i}")[1] == pytest.approx(
                ref[1] * scale, rel=1e-6
            )


# ────────────────────────────────────────────────────────────────────────
# 6. Guards — no degenerate, inverted, or wildly rescaled window
# ────────────────────────────────────────────────────────────────────────


def _rewrite_and_reopen(
    root: zarr.Group, edits: Dict[str, Dict[str, object]]
) -> Tuple[zarr.Group, zarr.Group]:
    """Apply per-node attr ``edits``, re-run the pass, return ``(edited, reread)``.

    Re-reads through the facade (``use_consolidated=False``): the store's root
    consolidated index still holds the PRE-edit attrs, and a plain
    ``zarr.open_group`` would serve those.
    """
    path = str(root.store_path)
    rw = open_group(path, mode="r+")
    for node_path, attrs in edits.items():
        for key, value in attrs.items():
            if value is None:
                del rw[node_path].attrs[key]
            else:
                rw[node_path].attrs[key] = value
    harmonize_gsplat_amplitude_windows(rw)
    return rw, open_group(path, mode="r")


@pytest.mark.parametrize(
    "factor, expected_scale",
    [
        pytest.param(500.0, _SCALE_BOUND, id="above-the-bound"),
        pytest.param(1.0 / 500.0, 1.0 / _SCALE_BOUND, id="below-the-bound"),
    ],
)
def test_out_of_bound_mass_ratio_is_clamped_to_the_bound(
    tmp_path: Path, factor: float, expected_scale: float
) -> None:
    """A wild ratio is CLAMPED, not reset to 1.0.

    Resetting is the worst of the three answers: at a genuine ratio of 0.02,
    scale 1.0 leaves the level windowed 50x too wide (it renders black), while
    clamping to 0.1 caps the error at 5x.
    """
    sublods = _mixture_ladder()
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))
    ref_before = _window(root, "child_2")

    _, reread = _rewrite_and_reopen(
        root,
        {"child_0": {"amplitude_mass_weighted_mean": factor * _mwma(root, "child_2")}},
    )
    # Neither 500x nor verbatim: exactly the bound.
    assert _window(reread, "child_0") == pytest.approx(
        [ref_before[0] * expected_scale, ref_before[1] * expected_scale], rel=1e-9
    )
    assert _window(reread, "child_0")[1] != pytest.approx(ref_before[1], rel=1e-3)
    # The in-bound sibling is still scaled by its own true ratio.
    assert _window(reread, "child_1")[1] == pytest.approx(
        ref_before[1] * (_mwma(reread, "child_1") / _mwma(reread, "child_2")), rel=1e-6
    )


def test_the_bound_warning_is_one_rolled_up_line_per_structure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Not one line per level: a 40-rung ladder would flood the finalize console."""
    n = 12
    sublods = _simple_ladder(n)
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))
    ref_mwma = _mwma(root, f"child_{n - 1}")

    # Push EVERY coarse level far outside the bound.
    edits: Dict[str, Dict[str, object]] = {
        f"child_{i}": {"amplitude_mass_weighted_mean": (i + 2) * 100.0 * ref_mwma}
        for i in range(n - 1)
    }
    capsys.readouterr()
    _rewrite_and_reopen(root, edits)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "clamped" in ln]

    assert len(lines) == 1
    assert f"{n - 1} LOD level(s)" in lines[0]
    # …and it names the most extreme ratio seen, not the last one.
    assert f"{(n - 1 + 1) * 100.0:.4g}" in lines[0]


@pytest.mark.parametrize("bad_mwma", [0.0, -1.5])
def test_degenerate_or_negative_mwma_never_writes_a_bad_window(
    tmp_path: Path, bad_mwma: float
) -> None:
    """``mwma <= 0`` must not produce ``[0, 0]`` (identity) or ``lo > hi`` (inverted)."""
    sublods = _mixture_ladder()
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))
    ref_before = _window(root, "child_2")

    _, reread = _rewrite_and_reopen(
        root, {"child_0": {"amplitude_mass_weighted_mean": bad_mwma}}
    )
    lo, hi = _window(reread, "child_0")
    assert lo < hi
    assert (lo, hi) == pytest.approx(tuple(ref_before), rel=1e-9)


def test_mass_less_sibling_does_not_disable_scaling_for_the_rest(
    tmp_path: Path,
) -> None:
    """A mass-less part is stamped ``mass = mwma = 0``, not "no stats".

    A 0-splat leaf cannot be written at all (``ValidationError``), so the
    reachable analogue is a leaf whose amplitudes are all zero: mass 0, both
    statistics PRESENT as ``0.0``, and a degenerate ``[0, 0]`` window. Nothing
    is hand-poked here — the store is what the writer produced.
    """
    # Mild (pareto(2.0)) levels, so the surviving ratios stay inside
    # ``_SCALE_BOUND`` and the test reads the scaling path, not the guard.
    parts = [
        _mild_leaf(0, seed=71),
        _uniform_amplitude_leaf(200, seed=72, amplitude=0.0),
    ]
    coarse = _mild_leaf(3, seed=73)
    root = _write(
        tmp_path,
        GSplatLodGroup(
            children=[coarse, GSplatPartition(children=parts, max_elements=4096)]
        ),
    )

    # The writer stamped the statistics as present-and-zero, not absent.
    assert float(_attrs(root, "child_1/part_1")["amplitude_mass"]) == 0.0  # type: ignore[arg-type]
    assert _mwma(root, "child_1/part_1") == 0.0

    ref = _window(root, "child_1/part_0")
    # The mass-less part is harmonized onto the shared window like any other…
    assert _window(root, "child_1/part_1") == ref
    # …and the surviving part alone sets the combined mwma, so the coarse cap is
    # STILL scaled against it — the mass-less part did not poison the structure.
    scale = _mwma(root, "child_0") / _mwma(root, "child_1/part_0")
    assert scale != pytest.approx(1.0, rel=1e-3)
    assert _window(root, "child_0")[1] == pytest.approx(ref[1] * scale, rel=1e-6)


def test_running_the_pass_twice_writes_nothing_the_second_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Idempotence: every window identical, and no console line the second run."""
    sublods = _mixture_ladder()
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))
    before = [_window(root, f"child_{i}") for i in range(len(sublods))]

    capsys.readouterr()
    _, reread = _rewrite_and_reopen(root, {})
    assert "Harmonized amplitude_data_range" not in capsys.readouterr().out
    assert [_window(reread, f"child_{i}") for i in range(len(sublods))] == before


# ────────────────────────────────────────────────────────────────────────
# 7. Legacy fallback — no statistics on disk
# ────────────────────────────────────────────────────────────────────────


def test_legacy_store_without_statistics_shares_the_window_verbatim(
    tmp_path: Path,
) -> None:
    sublods = _mixture_ladder()
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))
    n = len(sublods)

    edits: Dict[str, Dict[str, object]] = {}
    for i in range(n):
        edits[f"child_{i}"] = {
            "amplitude_mass": None,
            "amplitude_mass_weighted_mean": None,
            # Restore a per-level window so the fallback has something to
            # overwrite (each level's own, i.e. the pre-fix state).
            "amplitude_data_range": list(
                _authored_window(np.asarray(sublods[n - 1 - i].amplitudes))
            ),
        }
    _, reread = _rewrite_and_reopen(root, edits)  # must not raise

    windows = [_window(reread, f"child_{i}") for i in range(n)]
    assert windows[0] == windows[1] == windows[2]
    # …and the shared value is the FINEST level's own window, verbatim.
    assert tuple(windows[-1]) == pytest.approx(
        _authored_window(np.asarray(sublods[0].amplitudes)), rel=1e-9
    )


def test_mixed_legacy_sibling_alone_stays_at_scale_one(tmp_path: Path) -> None:
    """Only the level whose statistics are missing falls back to verbatim."""
    sublods = _mixture_ladder()
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))
    ref_before = _window(root, "child_2")

    _, reread = _rewrite_and_reopen(
        root,
        {
            "child_0": {
                "amplitude_mass": None,
                "amplitude_mass_weighted_mean": None,
            }
        },
    )
    # The stats-less level shares the reference window verbatim…
    assert _window(reread, "child_0") == pytest.approx(ref_before, rel=1e-9)
    # …while the level that still has them keeps being scaled.
    scale = _mwma(reread, "child_1") / _mwma(reread, "child_2")
    assert scale != pytest.approx(1.0, rel=1e-3)
    assert _window(reread, "child_1")[1] == pytest.approx(
        ref_before[1] * scale, rel=1e-6
    )


# ────────────────────────────────────────────────────────────────────────
# 8. The scene-compiler path (the demo shape from the issue)
# ────────────────────────────────────────────────────────────────────────


def test_scene_compiler_harmonizes_a_colormapped_substitutive_lod(
    tmp_path: Path,
) -> None:
    sublods = _mixture_ladder()  # finest first — GSplatData's own order
    data = GSplatData(
        substitutive_levels=[
            SubstitutiveLevel(
                additive_sublods=[sub],
                compression_factor=4**i,
                level_index=i,
            )
            for i, sub in enumerate(sublods)
        ]
    )
    out = tmp_path / "s.luxar.zarr"
    with LuxarZarrCompiler(out, enable_spatial_index=False) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_gsplats_from_data("splats", data, colormap="plasma")

    root = zarr.open_group(str(out), mode="r")
    n = len(sublods)
    windows = [_window(root, f"splats/child_{i}") for i in range(n)]
    mwmas = [_mwma(root, f"splats/child_{i}") for i in range(n)]

    # The finest level keeps the window its own writer derived — from the
    # amplitudes AS INSERTED. This ladder's raw amplitudes sit above 1.0, so the
    # scene adder normalises them on the way in (one factor for the whole
    # pyramid; see core/group/gsplats_pipeline/amplitude_norm.py) and the
    # authored window scales with them. Reading the factor back rather than
    # hard-coding the scaled numbers keeps this asserting the writer's rule
    # instead of a literal that silently re-pins whenever the reference moves.
    factor = dict(root["splats/child_0"].attrs).get(
        "amplitude_normalization_factor"
    ) or dict(root["splats"].attrs).get("amplitude_normalization_factor")
    assert factor is not None, "the adder should have normalised this ladder"
    assert tuple(windows[-1]) == pytest.approx(
        _authored_window(np.asarray(sublods[0].amplitudes) * factor), rel=1e-6
    )
    for i in range(n):
        scale = mwmas[i] / mwmas[-1]
        assert windows[i][1] == pytest.approx(windows[-1][1] * scale, rel=1e-6)
    assert windows[0][1] != pytest.approx(windows[-1][1], rel=1e-3)


# ────────────────────────────────────────────────────────────────────────
# 9. The pass never CREATES the attr
# ────────────────────────────────────────────────────────────────────────


def test_pass_never_creates_an_absent_window(tmp_path: Path) -> None:
    sublods = _mixture_ladder()
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))

    _, reread = _rewrite_and_reopen(root, {"child_0": {"amplitude_data_range": None}})

    assert "amplitude_data_range" not in dict(reread["child_0"].attrs)
    # The group wrappers never carried one and must not gain one either.
    assert "amplitude_data_range" not in dict(reread.attrs)
    # …while the siblings that DID carry one are still harmonized.
    assert "amplitude_data_range" in dict(reread["child_1"].attrs)
