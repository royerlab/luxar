"""Finalize-time ``amplitude_data_range`` harmonization (#1691).

Covers the two writer-stamped mass statistics
(:func:`~luxar.io._compiler.gsplat_assembly.compute_amplitude_mass_stats`) and
the pass that consumes them
(:func:`~luxar.io._compiler.finalize.amplitude_window.
harmonize_gsplat_amplitude_windows`) across every gsplat structure shape: a
substitutive ``kind=lod`` ladder (writer-named and hand-authored), levels
carrying their default ``additive_<i>`` stream ladders, a ``kind=partition``
(in-memory and streamed), the ``adaptive`` partition-of-ladders, the
``overview`` lod-over-partition, a ladder whose finest child is Points, the
no-stats legacy fallback, the scale/degeneracy guards, and the scene-compiler
path.
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


def test_ten_level_ladder_references_the_true_finest_level(tmp_path: Path) -> None:
    """A >=10-level ladder: ``child_10`` must not sort before ``child_2``."""
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


def test_out_of_bound_mass_ratio_falls_back_to_the_verbatim_window(
    tmp_path: Path,
) -> None:
    """A ratio outside ``[1/10, 10]`` is a degenerate statistic, not a rescale."""
    sublods = _mixture_ladder()
    root = _write(tmp_path, _lod_group_coarsest_first(sublods))
    ref_before = _window(root, "child_2")

    _, reread = _rewrite_and_reopen(
        root,
        {"child_0": {"amplitude_mass_weighted_mean": 500.0 * _mwma(root, "child_2")}},
    )
    # Not 500x the reference window — shared verbatim instead.
    assert _window(reread, "child_0") == pytest.approx(ref_before, rel=1e-9)
    # The in-bound sibling is still scaled.
    assert _window(reread, "child_1")[1] == pytest.approx(
        ref_before[1] * (_mwma(reread, "child_1") / _mwma(reread, "child_2")), rel=1e-6
    )


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


def test_empty_sibling_does_not_disable_scaling_for_the_rest(tmp_path: Path) -> None:
    """An ``n_splats == 0`` part is stamped ``mass = mwma = 0``, not "no stats"."""
    # Mild (pareto(2.0)) levels, so the surviving ratios stay inside
    # ``_SCALE_BOUND`` and the test reads the scaling path, not the guard.
    parts = [_mild_leaf(0, seed=71), _mild_leaf(2, seed=72)]
    coarse = _mild_leaf(3, seed=73)
    root = _write(
        tmp_path,
        GSplatLodGroup(
            children=[coarse, GSplatPartition(children=parts, max_elements=4096)]
        ),
    )

    # Empty the SECOND part the way the writer stamps a zero-splat leaf.
    _, reread = _rewrite_and_reopen(
        root,
        {
            "child_1/part_1": {
                "amplitude_mass": 0.0,
                "amplitude_mass_weighted_mean": 0.0,
                "n_splats": 0,
            }
        },
    )

    ref = _window(reread, "child_1/part_0")
    # The surviving part alone now sets the combined mwma, and the coarse cap is
    # STILL scaled against it — the empty part did not poison the structure.
    scale = _mwma(reread, "child_0") / _mwma(reread, "child_1/part_0")
    assert scale != pytest.approx(1.0, rel=1e-3)
    assert _window(reread, "child_0")[1] == pytest.approx(ref[1] * scale, rel=1e-6)


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
    # The finest level keeps the window its own writer derived.
    assert tuple(windows[-1]) == pytest.approx(
        _authored_window(np.asarray(sublods[0].amplitudes)), rel=1e-9
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
