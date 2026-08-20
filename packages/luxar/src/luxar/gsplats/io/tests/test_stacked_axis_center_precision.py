"""Regression tests for the centers sigma rail and its snap interaction (#1748).

A stacked axis built with ``combine_as_new_dimension(..., sigma=0.0)`` has a
sigma floored to 1e-7 but an extent of one unit per frame. Under the default
``AUTO`` (and under ``MEMORY``) centers are stored as per-axis uint16
fixed-point, whose step on that axis is ``(n_frames - 1) / 65535`` — thousands
of sigma. Every interior frame then lands off its integer coordinate and stops
matching a slice query, while the two endpoints quantize exactly and so make the
store look fine on inspection.

Two mechanisms cover that, and this module pins the boundary between them:

* The **grid snap**, in the encoder
  (:func:`~luxar.encoding.gridded_axis_step`). A *gridded* axis — one whose
  distinct values all sit on one regular grid, which a stacked/categorical axis
  always is — has its quantization grid widened onto the data's own spacing, so
  every value round-trips bit-exactly at uint16. It costs nothing: ``lo``/``hi``
  are stored per axis regardless, so it is a scale choice, not a dtype change.
* The **sigma rail**, in
  :func:`~luxar.io._compiler.gsplat_assembly.write_gsplat_arrays`. It compares
  HALF the grid step (the worst-case round-trip displacement) against EACH
  splat's own marginal sigma per axis and escalates *the centers only* to float32
  when more than
  :data:`~luxar.io._compiler.gsplat_assembly.MAX_UNREPRESENTABLE_SPLAT_FRACTION`
  of the splats could be displaced past
  :data:`~luxar.io._compiler.gsplat_assembly.MAX_CENTER_DISPLACEMENT_SIGMAS` of
  their own sigma. It is deliberately a population test rather than a minimum:
  every real fit holds a few needle Gaussians, and one of those must not double
  the size of an otherwise ordinary centers array.

The rail is the BACKSTOP, and it is snap-aware: an axis the encoder is going to
store exactly is skipped, so a stacked dataset keeps uint16 centers and stays
silent. What is left for the rail is a degenerate sub-population on a
**non-gridded** axis — a ``sigma=0`` track stack merged into a fit whose time
axis is continuous, which is the real ``luxar gsplat merge`` case — where the
snap declines and the splats really are destroyed.
"""

import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.encoding.decoder import ArrayDecoder
from luxar.gsplats import GSplatData
from luxar.io._compiler.gsplat_assembly import (
    MAX_CENTER_DISPLACEMENT_SIGMAS,
    MAX_UNREPRESENTABLE_SPLAT_FRACTION,
    _center_quantization_offender,
)

LOSSY_MODES = [EncodingMode.AUTO, EncodingMode.MEMORY]
ALL_MODES = LOSSY_MODES + [EncodingMode.PRECISION]

#: Diagonal positions of a packed row-major lower-triangular 4-D Cholesky row.
_DIAG_4D = [0, 2, 5, 9]


def _frame(n_splats: int, seed: int) -> GSplatData:
    """One 3-D timepoint with sane, well-resolved spatial sigmas."""
    rng = np.random.default_rng(seed)
    chol = np.zeros((n_splats, 6), dtype=np.float32)
    # Diagonal at packed positions 0, 2, 5 (row-major lower-triangular).
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 1.5, size=(n_splats, 3))
    return GSplatData(
        centers=(rng.random((n_splats, 3)) * 10.0).astype(np.float32),
        amplitudes=rng.random(n_splats).astype(np.float32) + 0.1,
        cholesky_factors=chol,
    )


def _stacked(n_frames: int = 12, n_splats: int = 5) -> GSplatData:
    """``n_frames`` 3-D fits stacked onto a discrete (sigma=0) time axis.

    GRIDDED: the time column holds only the integers ``0 .. n_frames - 1``, so
    the encoder snaps and the rail must stand down.
    """
    return GSplatData.combine_as_new_dimension(
        [_frame(n_splats, seed=t) for t in range(n_frames)],
        values=[float(t) for t in range(n_frames)],
        sigma=0.0,
    )


def _merged_track_centers(
    n_splats: int, n_tracks: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Centers + Cholesky for a track stack merged onto a CONTINUOUS time axis.

    The ``luxar gsplat merge`` case, and the only thing the sigma rail still has
    to catch: the bulk of the store is an ordinary fit whose time column carries
    a real sigma and thousands of irregular float coordinates — so no regular
    grid describes it and the encoder's snap declines — while the last
    ``n_tracks`` rows are a ``sigma=0`` track stack pinned to integer frames.
    Those tracks are displaced by up to 305 of their own sigma (half of the
    ``4 / 65535`` step against the 1e-7 floor) and vanish.
    """
    rng = np.random.default_rng(seed)
    times = rng.random(n_splats) * 4.0
    times[-n_tracks:] = rng.integers(0, 5, size=n_tracks).astype(np.float64)
    centers = np.column_stack([rng.random((n_splats, 3)) * 10.0, times]).astype(
        np.float32
    )
    chol = np.zeros((n_splats, 10), dtype=np.float32)
    chol[:, _DIAG_4D] = rng.uniform(0.5, 1.5, size=(n_splats, 4))
    chol[:, 9] = 3.0  # a real, well-resolved sigma on the fit's own time axis
    chol[-n_tracks:, 9] = 1e-7  # what combine_as_new_dimension(sigma=0) leaves
    return centers, chol


def _merged_tracks(
    n_splats: int = 20_000, n_tracks: int = 200, seed: int = 1748
) -> GSplatData:
    """:func:`_merged_track_centers` as a saveable dataset (1% degenerate)."""
    centers, chol = _merged_track_centers(n_splats, n_tracks, seed)
    rng = np.random.default_rng(seed + 1)
    return GSplatData(
        centers=centers,
        amplitudes=rng.random(n_splats).astype(np.float32) + 0.1,
        cholesky_factors=chol,
    )


@pytest.mark.parametrize("mode", ALL_MODES)
def test_stacked_axis_round_trips_exactly(mode: EncodingMode) -> None:
    """Every splat lands back on its exact integer frame, in every mode."""
    n_frames, n_splats = 12, 5
    stacked = _stacked(n_frames, n_splats)
    expected = np.repeat(np.arange(n_frames, dtype=np.float64), n_splats)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "stacked.gsplats.zarr"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            stacked.save(path, encoding_mode=mode)
        loaded = GSplatData.load(path)

    assert loaded.ndim == 4
    times = np.asarray(loaded.centers[:, 3], dtype=np.float64)
    np.testing.assert_array_equal(np.sort(times), expected)


@pytest.mark.parametrize("mode", LOSSY_MODES)
def test_stacked_axis_is_not_escalated_because_the_snap_covers_it(
    mode: EncodingMode,
) -> None:
    """The rail↔snap boundary, which is the whole point of the two-tier design.

    100% of a stacked axis's splats trip the rail's per-splat criterion (sigma
    1e-7 against a 1.7e-4 step), so a snap-unaware rail escalates the entire
    centers array to float32 — doubling the centers bytes and warning loudly —
    for an axis the encoder stores EXACTLY in uint16 for free. Both messages
    would even print at once. So: no warning, still ``linear_perchannel_u16``,
    and still bit-exact.
    """
    n_frames, n_splats = 12, 5
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "stacked.gsplats.zarr"
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _stacked(n_frames, n_splats).save(path, encoding_mode=mode)

        root = zarr.open_group(str(path), mode="r")
        assert root["centers"].attrs["encoding"]["name"] == "linear_perchannel_u16"
        times = np.asarray(GSplatData.load(path).centers[:, 3], dtype=np.float64)
        np.testing.assert_array_equal(
            np.sort(times), np.repeat(np.arange(n_frames, dtype=np.float64), n_splats)
        )


def test_grid_ness_alone_decides_whether_the_rail_fires() -> None:
    """Same degenerate population, gridded vs not — only the second escalates.

    A direct measurement that the rail defers to the ENCODER'S predicate rather
    than to some proxy of its own. Both arrays have identical Cholesky factors
    and an identical fraction of unrepresentable splats on axis 3; they differ
    only in whether that axis lies on a regular grid.
    """
    n, n_tracks = 5_000, 500
    rng = np.random.default_rng(4)
    chol = np.zeros((n, 10), dtype=np.float32)
    chol[:, _DIAG_4D] = rng.uniform(0.5, 1.5, size=(n, 4))
    chol[:, 9] = 3.0
    # Every 10th row, so the degenerate splats are spread across ALL five
    # frames. Pinning them to the last rows instead put every one of them at
    # time 4.0 — the axis MAXIMUM, which uint16 reproduces exactly with or
    # without a snap — so the "identical except for grid-ness" claim would have
    # been made about splats that were never at risk on either side.
    chol[::10, 9] = 1e-7
    assert np.count_nonzero(chol[:, 9] < 1.0) == n_tracks

    xyz = (rng.random((n, 3)) * 10.0).astype(np.float32)
    gridded = np.repeat(np.arange(5, dtype=np.float32), n // 5)
    # The same frames, nudged off the grid. The jitter (1e-3) is ~16 full
    # quantization steps (6.1e-5), so it is not small against the step — but it
    # is small against the axis EXTENT, which it changes by 0.025%. The step,
    # and hence the displacement bound the rail measures, is therefore
    # unchanged; the only thing destroyed is grid eligibility.
    jittered = (gridded + rng.random(n).astype(np.float32) * 1e-3).astype(np.float32)

    assert (
        _center_quantization_offender(np.column_stack([xyz, gridded]), chol, 4) is None
    )
    offender = _center_quantization_offender(np.column_stack([xyz, jittered]), chol, 4)
    assert offender is not None
    assert offender[0] == 3


def test_a_gridded_axis_does_not_mask_a_broken_one() -> None:
    """A skipped (gridded) axis must not raise the bar for the axes after it.

    The order inside the per-axis check is load-bearing and nothing else pins
    it: the grid decline has to happen BEFORE the running worst fraction is
    updated. Here axis 3 is gridded and 100% degenerate while axis 4 is
    continuous and 0.2% degenerate. If the gridded axis were allowed to set the
    bar first — by testing grid-ness after the update, or by hoisting the update
    out of the loop — axis 4's 0.2% would lose to it and the rail would report
    nothing, which is issue #1748 all over again on the axis that really is
    broken.
    """
    n, n_bad = 10_000, 20  # 0.2%: over the 0.1% gate
    rng = np.random.default_rng(1748)
    # 5-D packed lower-triangular: diagonal at 0, 2, 5, 9, 14.
    chol = np.zeros((n, 15), dtype=np.float32)
    chol[:, [0, 2, 5, 9, 14]] = rng.uniform(0.5, 1.5, size=(n, 5))
    chol[:, 9] = 1e-7  # axis 3: EVERY splat degenerate...
    chol[:n_bad, 14] = 1e-7  # ...axis 4: a small minority

    gridded = np.repeat(np.arange(5, dtype=np.float32), n // 5)  # axis 3
    continuous = (rng.random(n) * 4.0).astype(np.float32)  # axis 4
    centers = np.column_stack(
        [(rng.random((n, 3)) * 10.0).astype(np.float32), gridded, continuous]
    ).astype(np.float32)

    offender = _center_quantization_offender(centers, chol, 5)
    assert offender is not None
    axis, _step, count, fraction, _sigma_median = offender
    assert axis == 4
    assert count == n_bad
    assert fraction == pytest.approx(n_bad / n)


def test_lut_eligible_centers_are_not_escalated() -> None:
    """The encoder's OTHER exact path: LUT beats the rail, and float32.

    LUT encoding is tried before the dtype encoder and stores the original
    values verbatim at ~1 B/value. Escalating a LUT-eligible array to
    ``PRECISION`` would both quadruple its bytes AND suppress the LUT (the
    encoder skips LUT in ``PRECISION``), while warning that splats can be moved
    clear of their own core — damage that was never going to happen. So the rail
    must ask :meth:`ArrayEncoder.encodes_as_lut` as well as the grid predicate.

    The fixture is deliberately one the rail DOES flag: axis 2's values
    ``{0, 1, 3.7, 12}`` lie on no regular grid (the snap declines), and every
    splat's sigma there is 1e-7.
    """
    n = 20_000
    rng = np.random.default_rng(1748)
    xy = rng.integers(0, 60, size=(n, 2)).astype(np.float32)
    z = np.array([0.0, 1.0, 3.7, 12.0], dtype=np.float32)[rng.integers(0, 4, n)]
    centers = np.column_stack([xy, z]).astype(np.float32)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 1.5, size=(n, 3))
    chol[:, 5] = 1e-7

    # The rail really does see an offender here — the stand-down below is the
    # LUT check, not an absent offender.
    offender = _center_quantization_offender(centers, chol, 3)
    assert offender is not None and offender[0] == 2

    data = GSplatData(
        centers=centers,
        amplitudes=rng.random(n).astype(np.float32) + 0.1,
        cholesky_factors=chol,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "lut.gsplats.zarr"
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            data.save(path, encoding_mode=EncodingMode.AUTO)

        root = zarr.open_group(str(path), mode="r")
        assert root["centers"].attrs["encoding"]["name"] == "lut_uint8"
        # Exact, and a quarter of what the escalation would have written.
        loaded = np.asarray(GSplatData.load(path).centers, dtype=np.float32)
        np.testing.assert_array_equal(
            loaded[np.lexsort(loaded.T)], centers[np.lexsort(centers.T)]
        )


@pytest.mark.parametrize("mode", LOSSY_MODES)
def test_guard_warns_and_stores_centers_as_float32(mode: EncodingMode) -> None:
    """The rail fires, says which axis, and downgrades ONLY the centers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "merged.gsplats.zarr"
        with pytest.warns(UserWarning, match=r"axis 3.*sigma"):
            _merged_tracks().save(path, encoding_mode=mode)

        root = zarr.open_group(str(path), mode="r")
        assert root["centers"].attrs["encoding"]["name"] == "float32"
        # The Cholesky tier is untouched by the escalation: still quantized.
        diag_name = root["cholesky_factors_diag"].attrs["encoding"]["name"]
        assert diag_name != "float32"


def test_ordinary_spatial_data_keeps_uint16_centers() -> None:
    """The rail must not disable the size win for normal 3-D gsplats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "plain.gsplats.zarr"
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _frame(200, seed=7).save(path, encoding_mode=EncodingMode.MEMORY)

        root = zarr.open_group(str(path), mode="r")
        assert root["centers"].attrs["encoding"]["name"] == "linear_perchannel_u16"


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf, np.nan])
def test_non_finite_extent_is_declined_rather_than_escalated(
    bad_value: float,
) -> None:
    """A non-finite extent is declined, not escalated.

    A non-finite extent makes the grid step ``inf``/``nan``. ``inf`` beats every
    sigma, so without an early-out the rail would escalate EVERY such array —
    including one whose only problem is a stray sentinel the value validators
    own and report far better than a size rail can. Two guards cover the two
    arms: an ``inf`` extent exceeds ``COORDINATE_U16_MAX_EXTENT`` and declines
    with the over-wide axes, while ``nan`` slips past that comparison and is
    caught by the per-axis ``not np.isfinite(step)`` check (its own downstream
    comparisons are False anyway).
    """
    n = 100
    rng = np.random.default_rng(17)
    centers = (rng.random((n, 3)) * 10.0).astype(np.float32)
    centers[0, 0] = bad_value
    chol = np.zeros((n, 6), dtype=np.float32)
    # Axis 0 is degenerate too, so a finite step there WOULD offend: only the
    # non-finite check can be what makes this return None.
    chol[:, 0] = 1e-9
    chol[:, [2, 5]] = rng.uniform(0.5, 1.5, size=(n, 2))

    assert _center_quantization_offender(centers, chol, 3) is None


def test_over_wide_axis_is_left_to_the_encoders_own_extent_rail() -> None:
    """An axis at/above 2¹⁶ is the encoder's case, not this rail's.

    The encoder already stores such an array as float32 — exactly — and says
    why ("extent ... ≥ 2¹⁶; uint16 fixed-point cannot resolve a unit step"),
    which is the useful diagnosis. The sigma rail fires on the same data (a
    ~100,000-unit axis has a ~1.5 step, so a σ of 0.3 is over-run for every
    splat) and, unhandled, would pre-empt that message with one pointing at a
    degenerate axis that is not there — and would drop the array's content dedup
    for a verdict that depends on the centers bytes alone.
    """
    n = 2000
    rng = np.random.default_rng(19)
    centers = np.column_stack(
        [rng.random(n) * 100_000.0, rng.random((n, 2)) * 10.0]
    ).astype(np.float32)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 0.3  # far under half of the 1.53 step on axis 0

    assert _center_quantization_offender(centers, chol, 3) is None

    data = GSplatData(
        centers=centers,
        amplitudes=rng.random(n).astype(np.float32) + 0.1,
        cholesky_factors=chol,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "wide.gsplats.zarr"
        with pytest.warns(UserWarning, match=r"extent \d+ ≥ 2¹⁶"):
            data.save(path, encoding_mode=EncodingMode.MEMORY)

        root = zarr.open_group(str(path), mode="r")
        assert root["centers"].attrs["encoding"]["name"] == "float32"


def test_constant_axis_does_not_trip_the_guard() -> None:
    """A zero-extent axis has a zero step, so it can never be misquantized."""
    frame = _frame(50, seed=3)
    # One dataset stacked → the new axis is constant (extent 0) AND sigma≈0.
    stacked = GSplatData.combine_as_new_dimension([frame], values=[4.0], sigma=0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "constant.gsplats.zarr"
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            stacked.save(path, encoding_mode=EncodingMode.MEMORY)

        root = zarr.open_group(str(path), mode="r")
        assert root["centers"].attrs["encoding"]["name"] == "linear_perchannel_u16"
        loaded = GSplatData.load(path)
        np.testing.assert_array_equal(loaded.centers[:, 3], 4.0)


def test_guard_constants_are_the_documented_numbers() -> None:
    """Pin the two constants the READMEs and GSPLATS_ZARR_FORMAT.md quote."""
    assert MAX_CENTER_DISPLACEMENT_SIGMAS == 1.0
    assert MAX_UNREPRESENTABLE_SPLAT_FRACTION == 0.001


def test_per_splat_criterion_is_half_a_step_against_one_sigma() -> None:
    """Measure the criterion instead of restating it.

    The line is the worst-case round-trip displacement — HALF a grid step —
    against the splat's own marginal sigma. So a splat whose sigma is just under
    half the step is unrepresentable and one just over is not, and the same
    array flips between the two answers on nothing but that.

    The offending column is deliberately IRREGULAR (random, with the two
    endpoints pinned so the extent is exactly 10). An evenly spaced column would
    be gridded, the encoder would store it exactly and the rail would correctly
    decline — measuring the snap rather than the criterion.
    """
    n = 1000
    extent = 10.0
    half_step = (extent / 65535.0) / 2.0
    rng = np.random.default_rng(23)
    col = rng.random(n) * extent
    col[0], col[-1] = 0.0, extent
    centers = np.zeros((n, 3), dtype=np.float32)
    centers[:, 0] = col.astype(np.float32)

    def _offender_with(
        sigma_axis0: float,
    ) -> Optional[Tuple[int, float, int, float, float]]:
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, 0] = sigma_axis0
        chol[:, [2, 5]] = 1.0
        return _center_quantization_offender(centers, chol, 3)

    # Every splat sits at the same sigma, so the population gate is 0% or 100%
    # and only the per-splat line decides.
    assert _offender_with(half_step * 1.01) is None
    tripped = _offender_with(half_step * 0.99)
    assert tripped is not None
    axis, step, count, fraction, _sigma_median = tripped
    assert axis == 0
    assert step / 2.0 == pytest.approx(half_step, rel=1e-6)
    assert count == n
    assert fraction == 1.0


def test_one_needle_splat_does_not_escalate_the_whole_array() -> None:
    """A single degenerate splat must not cost 4000 others their uint16 win.

    The direct regression guard for the ``min``-over-splats predicate this rail
    started out with: every real fit has needle Gaussians (an SPZ import decodes
    scales as ``exp(u8/16 - 10)``, floor 4.5e-5), and one of those flipped the
    entire centers array to float32 — measured at +96% on centers, +54% on a
    600k-splat store.
    """
    n = 4000
    rng = np.random.default_rng(11)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 1.5, size=(n, 3))
    # One needle: 1/4000 = 0.025%, four times under the 0.1% population gate.
    chol[0, [0, 2, 5]] = 1e-6
    data = GSplatData(
        centers=(rng.random((n, 3)) * 10.0).astype(np.float32),
        amplitudes=rng.random(n).astype(np.float32) + 0.1,
        cholesky_factors=chol,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "needle.gsplats.zarr"
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            data.save(path, encoding_mode=EncodingMode.MEMORY)

        root = zarr.open_group(str(path), mode="r")
        assert root["centers"].attrs["encoding"]["name"] == "linear_perchannel_u16"


def test_minority_of_unrepresentable_splats_escalates() -> None:
    """A degenerate MINORITY is enough — the finding-B merge case in miniature.

    Merging a small ``sigma=0`` track stack into a big ordinary fit leaves the
    tracks a sub-percent slice of the store, which the rail's original 1% gate
    waved through while every one of those splats was displaced by >1000 sigma.
    0.5% of 100,000 is comfortably inside that old blind spot and comfortably
    over the 0.1% gate that replaced it. The centers here are continuous on
    every axis, so no snap can rescue them.
    """
    n, n_bad = 100_000, 500  # 0.5%
    rng = np.random.default_rng(12)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 1.5, size=(n, 3))
    chol[:n_bad, [0, 2, 5]] = 1e-6
    data = GSplatData(
        centers=(rng.random((n, 3)) * 10.0).astype(np.float32),
        amplitudes=rng.random(n).astype(np.float32) + 0.1,
        cholesky_factors=chol,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "needles.gsplats.zarr"
        with pytest.warns(UserWarning, match=r"0\.50%"):
            data.save(path, encoding_mode=EncodingMode.MEMORY)

        root = zarr.open_group(str(path), mode="r")
        assert root["centers"].attrs["encoding"]["name"] == "float32"


@pytest.mark.parametrize(
    "n_bad,should_trip",
    [(10, False), (11, True)],  # 0.1% is allowed; strictly above it is not
)
def test_unrepresentable_fraction_threshold_is_exact(
    n_bad: int, should_trip: bool
) -> None:
    """Pin :data:`MAX_UNREPRESENTABLE_SPLAT_FRACTION` on either side of the line.

    Continuous centers on every axis, so grid-ness plays no part and the count
    is the only thing that moves.
    """
    n = 10_000  # so the threshold count is exactly 10
    rng = np.random.default_rng(13)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 1.5, size=(n, 3))
    chol[:n_bad, [0, 2, 5]] = 1e-6
    centers = (rng.random((n, 3)) * 10.0).astype(np.float32)

    offender = _center_quantization_offender(centers, chol, 3)
    assert (offender is not None) is should_trip
    if offender is not None:
        axis, _step, count, fraction, sigma_median = offender
        assert axis == 0  # every axis offends equally; the first one wins the tie
        assert count == n_bad
        assert fraction == pytest.approx(n_bad / n)
        assert sigma_median == pytest.approx(1e-6, rel=1e-3)


def test_unpacked_cholesky_is_declined_rather_than_misread() -> None:
    """An (N, d, d) Cholesky would read sigma=0 and escalate unconditionally.

    ``validate_gsplat_inputs`` makes that unreachable from either call site, but
    a silently WRONG answer is the exact failure mode this rail exists to stop,
    so the shape precondition is checked rather than assumed. The centers are
    continuous, so the shape decline is the only thing that can return ``None``.
    """
    n = 100
    centers = (np.random.default_rng(29).random((n, 3)) * 10.0).astype(np.float32)
    unpacked = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    assert _center_quantization_offender(centers, unpacked, 3) is None


def test_escalated_centers_are_not_deduped_onto_a_quantized_sibling(
    tmp_path: Path,
) -> None:
    """An escalated centers array must never become an ``array_ref``.

    The encoder's content-dedup registry is keyed on the centers BYTES, but the
    rail's verdict depends on a SIBLING array (``cholesky_factors``) the
    registry knows nothing about. Two nodes with byte-identical centers and
    different Cholesky therefore used to collapse onto whichever was written
    first: the escalated node was stored as a ``(0, 4)`` ``array_ref`` pointing
    at the OTHER node's ``linear_perchannel_u16`` array, so it read back
    quantized — issue #1748 verbatim, behind a warning claiming it had been
    prevented.
    """
    n, n_tracks = 5_000, 200
    centers, chol_degenerate = _merged_track_centers(n, n_tracks, seed=1748)
    # The same store WITHOUT the degenerate tracks: identical centers bytes, a
    # Cholesky the rail is happy with.
    chol_ok = chol_degenerate.copy()
    chol_ok[-n_tracks:, 9] = 3.0

    dims = Dimensions(
        [
            Dimension("X", display=True),
            Dimension("Y", display=True),
            Dimension("Z", display=True),
            Dimension("Time", display=False, range=(0.0, 4.0)),
        ]
    )
    path = tmp_path / "two_nodes.luxar.zarr"
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        # `a` first, so it is the one that registers the centers bytes.
        scene.add_gsplats(
            "a",
            centers,
            amplitudes=1.0,
            cholesky_factors=chol_ok,
            extend_to_all=[],
        )
        with pytest.warns(UserWarning, match=r"axis 3.*sigma"):
            scene.add_gsplats(
                "b",
                centers,
                amplitudes=1.0,
                cholesky_factors=chol_degenerate,
                extend_to_all=[],
            )

    root = zarr.open_group(str(path), mode="r")
    assert root["a/centers"].attrs["encoding"]["name"] == "linear_perchannel_u16"
    encoding_b = root["b/centers"].attrs["encoding"]
    assert encoding_b["name"] == "float32"
    assert "target" not in encoding_b
    assert root["b/centers"].shape == centers.shape

    # Compared as row multisets: the writer spatially reorders both nodes
    # identically, which is precisely why their bytes collided in the first
    # place, so the permutation is not what is under test here.
    decoder = ArrayDecoder()
    decoded_b = np.asarray(decoder.decode(root["b/centers"], zarr_root=root))
    np.testing.assert_array_equal(
        decoded_b[np.lexsort(decoded_b.T)], centers[np.lexsort(centers.T)]
    )

    # `a` keeps the size win, and its (non-gridded) time column still quantizes
    # off the original values — which is exactly why `b` may not share it.
    decoded_a = np.asarray(decoder.decode(root["a/centers"], zarr_root=root))
    assert not np.array_equal(np.sort(decoded_a[:, 3]), np.sort(centers[:, 3]))
