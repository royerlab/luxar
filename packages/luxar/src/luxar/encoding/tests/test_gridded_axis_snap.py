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
    array = _encode(tmp_path, data, mode)
    back = np.asarray(array[:])

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


@pytest.mark.parametrize(
    "name, make",
    [
        # The whole corpus: continuous positions must be untouched.
        ("continuous", lambda r: r.random((20000, 3), dtype=np.float32) * 100.0),
        # Few distinct values but NOT evenly spaced — no grid to snap to.
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
        # Gridded but with more distinct values than the cap allows.
        (
            "too_many_distinct",
            lambda r: np.hstack(
                [
                    r.random((20000, 2), dtype=np.float32),
                    (np.arange(20000, dtype=np.float32) * 0.5).reshape(-1, 1),
                ]
            ),
        ),
    ],
)
def test_ordinary_coordinates_are_not_snapped(tmp_path, name, make) -> None:
    data = make(np.random.default_rng(0)).astype(np.float32)
    assert not _was_snapped(_encode(tmp_path, data), data), name


def test_snapping_does_not_grow_the_store(tmp_path) -> None:
    """The point of snapping rather than falling back to float32.

    A float32 fallback would also be exact, but centers are one array with one
    encoding, so it would convert every axis — measured at +78% on a stacked
    store. Snapping keeps uint16, so the dtype must not change.
    """
    data = _stacked()
    array = _encode(tmp_path, data)
    assert array.dtype == np.uint16
