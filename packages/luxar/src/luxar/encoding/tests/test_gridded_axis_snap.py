"""A gridded coordinate axis must survive uint16 quantization exactly (#1748).

`combine_as_new_dimension(sigma=0)` builds a stacked axis whose splats have an
effective sigma of 1e-7 (the epsilon `trils.py` substitutes to keep the covariance
positive-definite). Ordinary per-axis uint16 quantization moves interior values by
up to half a step -- thousands of sigma -- so every frame but the two endpoints
stops matching a slice query and disappears.

The encoder therefore snaps such an axis's quantization grid onto the data's own
spacing. These tests pin both halves of that: it must fire where it is needed, and
it must NOT fire on ordinary continuous coordinates, which are the whole corpus.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.encoding import EncodingMode
from luxar.encoding._encoders.perchannel import _gridded_step_from_uniques
from luxar.encoding.decoder import ArrayDecoder
from luxar.encoding.encoder import ArrayEncoder
from luxar.encoding.semantic_types import SemanticType


def _encode(tmp_path, data: np.ndarray, mode=EncodingMode.AUTO):
    group = zarr.open_group(str(tmp_path / "g.zarr"), mode="w")
    ArrayEncoder().encode(
        np.asarray(data, dtype=np.float32),
        group,
        "coords",
        semantic_type=SemanticType.COORDINATE,
        mode=mode,
    )
    return group["coords"]


def _roundtrip(tmp_path, data: np.ndarray, mode=EncodingMode.AUTO) -> np.ndarray:
    """Encode then DECODE, which is the only thing worth asserting on.

    Reading the stored array back gives raw uint16 levels, not coordinates. Those
    happen to equal the frame index for a 0..n-1 axis, so comparing them straight
    to the data passes for that one case and would keep passing for a broken
    encoder on any axis that does not start at 0 with unit spacing.
    """
    group = zarr.open_group(str(tmp_path / "g.zarr"), mode="w")
    ArrayEncoder().encode(
        np.asarray(data, dtype=np.float32),
        group,
        "coords",
        semantic_type=SemanticType.COORDINATE,
        mode=mode,
    )
    return np.asarray(ArrayDecoder().decode(group["coords"], group))


def _was_snapped(array, data: np.ndarray) -> bool:
    """An axis was snapped iff its stored upper rail exceeds the data's own max."""
    col_hi = np.atleast_1d(dict(array.attrs.get("encoding", {})).get("col_hi", []))
    if col_hi.size != data.shape[1]:
        return False
    return bool(np.any(col_hi > data.max(axis=0) * (1 + 1e-6) + 1e-9))


def _stacked(n_frames: int = 100, per_frame: int = 50) -> np.ndarray:
    rng = np.random.default_rng(0)
    xyz = rng.random((n_frames * per_frame, 3), dtype=np.float32) * 60.0
    t = np.repeat(np.arange(n_frames, dtype=np.float32), per_frame).reshape(-1, 1)
    return np.hstack([xyz, t]).astype(np.float32)


@pytest.mark.parametrize("mode", [EncodingMode.AUTO, EncodingMode.MEMORY])
@pytest.mark.parametrize("n_frames", [3, 100])
def test_stacked_axis_round_trips_exactly(tmp_path, mode, n_frames) -> None:
    """The defect: interior frames used to land thousands of sigma off.

    Both modes matter — AUTO is the default, so this was never opt-in — and both
    sizes matter, because the failure is structural rather than large-N: the sigma
    floor is a constant while the quantization step scales with the axis.
    """
    data = _stacked(n_frames)
    back = _roundtrip(tmp_path, data, mode)

    stored_t = back[:, 3]
    want_t = data[:, 3]
    assert np.array_equal(stored_t, want_t), (
        "stacked axis was not preserved: max |delta| = %g"
        % np.abs(stored_t - want_t).max()
    )
    # Every splat still sits exactly on its own integer frame.
    assert np.array_equal(stored_t, np.round(stored_t))


def test_the_snap_is_what_makes_it_exact(tmp_path) -> None:
    """Guard against the test above passing for some unrelated reason."""
    data = _stacked()
    assert _was_snapped(_encode(tmp_path, data), data)


def test_long_time_axis_uses_portable_step_and_still_round_trips(tmp_path) -> None:
    authored = (
        np.float64(1000.0) + np.arange(1000, dtype=np.float64) * np.float64(0.1)
    ).astype(np.float32)
    uniq = np.unique(authored.astype(np.float64))
    lo = float(uniq[0])
    extent = float(uniq[-1] - uniq[0])
    offsets = uniq - lo
    coarsest = float(np.diff(uniq).min())
    rung = np.round(offsets / coarsest)
    rung_f64 = rung.astype(np.float64)
    offsets_f64 = offsets.astype(np.float64)
    raw_step = float(rung_f64 @ offsets_f64 / (rung_f64 @ rung_f64))

    result = _gridded_step_from_uniques(uniq, lo, extent, 65_535.0)
    assert result is not None
    step, n_unique = result
    assert step == float(f"{raw_step:.12g}")
    assert step == float(f"{np.nextafter(raw_step, np.inf):.12g}")
    assert n_unique == len(uniq)

    back = _roundtrip(tmp_path, authored[:, None])
    np.testing.assert_array_equal(back[:, 0], authored)


def test_grid_step_is_independent_of_input_float_width() -> None:
    col32 = (
        1_533_291.9996785969 + np.arange(1510, dtype=np.float64) * 248.01550756409404
    ).astype(np.float32)
    lo = float(col32[0])
    extent = float(col32[-1] - col32[0])

    from_float32 = _gridded_step_from_uniques(col32, lo, extent, 65_535.0)
    from_float64 = _gridded_step_from_uniques(
        col32.astype(np.float64), lo, extent, 65_535.0
    )

    assert from_float32 == from_float64 == (248.015507189, len(col32))


def test_ten_thousand_frame_axis_still_snaps_exactly(tmp_path) -> None:
    frames = np.arange(10_000, dtype=np.float32)

    result = _gridded_step_from_uniques(frames, 0.0, 9_999.0, 65_535.0)
    assert result == (1.0, len(frames))
    np.testing.assert_array_equal(_roundtrip(tmp_path, frames[:, None])[:, 0], frames)


@pytest.mark.parametrize(
    "name, make",
    [
        # The whole corpus: continuous positions must be untouched.
        ("continuous", lambda r: r.random((20000, 3), dtype=np.float32) * 100.0),
        # Few distinct values, but not on ANY regular grid — nothing to snap to.
        (
            "irregular",
            lambda r: np.hstack(
                [
                    r.random((5000, 2), dtype=np.float32) * 10.0,
                    np.array([0.0, 1.0, 3.7, 12.0], dtype=np.float32)[
                        r.integers(0, 4, 5000)
                    ].reshape(-1, 1),
                ]
            ),
        ),
        # A 2D scene's constant third axis: zero extent, nothing to snap.
        (
            "constant_axis",
            lambda r: np.hstack(
                [
                    r.random((5000, 2), dtype=np.float32) * 10.0,
                    np.zeros((5000, 1), np.float32),
                ]
            ),
        ),
        # More distinct values than uint16 has levels: no grid can represent it,
        # so there is nothing to snap to. 70000 > 65536.
        (
            "more_distinct_than_levels",
            lambda r: np.hstack(
                [
                    r.random((70000, 2), dtype=np.float32),
                    (np.arange(70000, dtype=np.float32) * 0.5).reshape(-1, 1),
                ]
            ),
        ),
    ],
)
def test_ordinary_coordinates_are_not_snapped(tmp_path, name, make) -> None:
    data = make(np.random.default_rng(0)).astype(np.float32)
    assert not _was_snapped(_encode(tmp_path, data), data), name


@pytest.mark.parametrize("n_frames", [4097, 10_000, 65_536])
def test_large_frame_counts_still_snap(tmp_path, n_frames) -> None:
    """Any stack that FITS uint16 must snap, however many frames it has.

    An earlier version capped this at 4096 distinct values, which silently left a
    10 000-frame timelapse broken even though its grid fits exactly. The only real
    limit is the number of levels the encoding has.
    """
    values = np.arange(n_frames, dtype=np.float32)
    data = np.hstack(
        [
            np.zeros((n_frames, 3), dtype=np.float32),
            values.reshape(-1, 1),
        ]
    )
    back = _roundtrip(tmp_path, data)
    assert np.array_equal(back[:, 3], values), (
        "%d-frame axis not preserved: max |delta| = %g"
        % (n_frames, np.abs(back[:, 3] - values).max())
    )


@pytest.mark.parametrize(
    "name, values",
    [
        # A spatial tile of a stacked dataset holds whatever frames have content
        # inside its box, and a filter can empty a timepoint in one region — so a
        # part's own time column routinely has MISSING rungs. Requiring every gap
        # to be equal rejects this and leaves the part broken by the very defect
        # the snap exists to prevent (measured 458 sigma before this was covered).
        ("missing rungs", [0.0, 1.0, 2.0, 7.0, 8.0, 9.0]),
        # A frame interval that float32 only approximates. The gaps then jitter by
        # more than a relative 1e-6, so any gap-equality test with a usable
        # tolerance rejects it (measured 734 sigma).
        ("0.1 s interval", (np.arange(100) * 0.1).tolist()),
        ("1/3 s interval", (np.arange(60) / 3.0).tolist()),
    ],
)
def test_incomplete_and_fractional_grids_still_snap(tmp_path, name, values) -> None:
    """A grid need not be gapless, nor exactly representable in float32."""
    want = np.asarray(values, dtype=np.float32)
    data = np.hstack(
        [
            np.random.default_rng(0).random((want.size, 3), dtype=np.float32) * 60.0,
            want.reshape(-1, 1),
        ]
    ).astype(np.float32)
    back = _roundtrip(tmp_path, data)
    assert np.array_equal(back[:, 3], want), "%s: max |delta| = %g" % (
        name,
        np.abs(back[:, 3] - want).max(),
    )


def test_one_dimensional_input_is_left_alone(tmp_path) -> None:
    """A 1-D COORDINATE array has no per-axis columns — it must not raise.

    Asserted through a real DECODE, not just on the encoding name. The name
    alone certified a successful write of an unreadable array: the 1-D
    reductions give 0-d ``col_lo``/``col_hi``, which were stored as bare JSON
    scalars, and the decoder rejects anything but equal-length 1-D rails
    ("got shapes () and ()"). A 1-D array is ONE column of N values on both
    sides now.
    """
    values = np.array([1.0, 2.0, 5.0], dtype=np.float32)
    group = zarr.open_group(str(tmp_path / "g.zarr"), mode="w")
    ArrayEncoder().encode(
        values,
        group,
        "coords",
        semantic_type=SemanticType.COORDINATE,
    )
    encoding = dict(group["coords"].attrs["encoding"])
    assert encoding["name"] == "linear_perchannel_u16"
    assert encoding["col_lo"] == [1.0] and encoding["col_hi"] == [5.0]

    back = np.asarray(ArrayDecoder().decode(group["coords"], group))
    assert back.shape == values.shape
    # Ordinary uint16 fixed point over the whole array (nothing is snapped
    # here — the snap needs per-axis columns), so the bound is half a step.
    np.testing.assert_allclose(back, values, atol=(5.0 - 1.0) / 65535.0 / 2.0)


def test_snapping_does_not_grow_the_store(tmp_path) -> None:
    """The point of snapping rather than falling back to float32.

    A float32 fallback would also be exact, but centers are one array with one
    encoding, so it would convert every axis — measured at +78% on a stacked
    store. Snapping keeps uint16, so the dtype must not change.
    """
    data = _stacked()
    array = _encode(tmp_path, data)
    assert array.dtype == np.uint16
