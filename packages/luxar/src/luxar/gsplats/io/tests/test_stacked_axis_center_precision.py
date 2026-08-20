"""Regression tests for the centers sigma rail (issue #1748).

A stacked axis built with ``combine_as_new_dimension(..., sigma=0.0)`` has a
sigma floored to 1e-7 but an extent of one unit per frame. Under the default
``AUTO`` (and under ``MEMORY``) centers are stored as per-axis uint16
fixed-point, whose step on that axis is ``(n_frames - 1) / 65535`` — thousands
of sigma. Every interior frame then lands off its integer coordinate and stops
matching a slice query, while the two endpoints quantize exactly and so make the
store look fine on inspection.

:func:`~luxar.io._compiler.gsplat_assembly.write_gsplat_arrays` now compares HALF
the grid step (the worst-case round-trip displacement) against EACH splat's own
marginal sigma per axis and escalates *the centers only* to float32 when more
than
:data:`~luxar.io._compiler.gsplat_assembly.MAX_UNREPRESENTABLE_SPLAT_FRACTION` of
the splats could be displaced past
:data:`~luxar.io._compiler.gsplat_assembly.MAX_CENTER_DISPLACEMENT_SIGMAS` of
their own sigma. It is deliberately a population test rather than a minimum:
every real fit holds a few needle Gaussians, and one of those must not double
the size of an otherwise ordinary centers array.
"""

import tempfile
import warnings
from pathlib import Path

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
    """``n_frames`` 3-D fits stacked onto a discrete (sigma=0) time axis."""
    return GSplatData.combine_as_new_dimension(
        [_frame(n_splats, seed=t) for t in range(n_frames)],
        values=[float(t) for t in range(n_frames)],
        sigma=0.0,
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
def test_guard_warns_and_stores_centers_as_float32(mode: EncodingMode) -> None:
    """The rail fires, says which axis, and downgrades ONLY the centers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "stacked.gsplats.zarr"
        with pytest.warns(UserWarning, match=r"axis 3.*sigma"):
            _stacked().save(path, encoding_mode=mode)

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
    """The ``not np.isfinite(step)`` early-out, exercised directly.

    A non-finite extent makes the grid step ``inf``/``nan``. ``inf`` beats every
    sigma, so without the early-out the rail would escalate EVERY such array —
    including one whose only problem is a stray sentinel the value validators
    own and report far better than a size rail can. (``nan`` compares False
    downstream, so only the ``inf`` arm is load-bearing; both are pinned because
    the branch handles them together.)
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
    """
    n = 1000
    extent = 10.0
    half_step = (extent / 65535.0) / 2.0
    centers = np.zeros((n, 3), dtype=np.float32)
    centers[:, 0] = np.linspace(0.0, extent, n, dtype=np.float32)

    def _offender_with(sigma_axis0: float):
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
    over the 0.1% gate that replaced it.
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
    """Pin :data:`MAX_UNREPRESENTABLE_SPLAT_FRACTION` on either side of the line."""
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
    so the shape precondition is checked rather than assumed.
    """
    n = 100
    centers = np.linspace(0, 10, n * 3, dtype=np.float32).reshape(n, 3)
    unpacked = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    assert _center_quantization_offender(centers, unpacked, 3) is None


def test_escalated_centers_are_not_deduped_onto_a_quantized_sibling(
    tmp_path,
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
    n_frames, per_frame = 5, 100
    n = n_frames * per_frame
    rng = np.random.default_rng(1748)
    centers = np.column_stack(
        [
            rng.random((n, 3)) * 10.0,
            np.repeat(np.arange(n_frames, dtype=np.float64), per_frame),
        ]
    ).astype(np.float32)

    # 4-D packed lower-triangular: diagonal at 0, 2, 5, 9.
    chol_ok = np.zeros((n, 10), dtype=np.float32)
    chol_ok[:, [0, 2, 5, 9]] = rng.uniform(0.5, 1.5, size=(n, 4))
    chol_degenerate = chol_ok.copy()
    chol_degenerate[:, 9] = 1e-7  # what combine_as_new_dimension(sigma=0) leaves

    dims = Dimensions(
        [
            Dimension("X", display=True),
            Dimension("Y", display=True),
            Dimension("Z", display=True),
            Dimension("Time", display=False, range=(0, n_frames - 1)),
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

    # `a` keeps the size win, and its time column still quantizes off the
    # integer frames — which is exactly why `b` may not share it.
    decoded_a = np.asarray(decoder.decode(root["a/centers"], zarr_root=root))
    assert not np.array_equal(np.sort(decoded_a[:, 3]), np.sort(centers[:, 3]))
