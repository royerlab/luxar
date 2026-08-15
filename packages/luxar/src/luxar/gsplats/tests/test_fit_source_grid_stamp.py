"""A fit records the volume its splats represent.

Without these stamps a stored ``.gsplats.zarr`` cannot answer "how much did this
compress?": the source grid appears nowhere on disk, and it is not recoverable
from the producing script either, because the fitted grid is derived at run time
from downscale factors and from isotropic resampling of the voxel spacing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from luxar._zarr_compat import read_node_attrs  # noqa: E402
from luxar.gsplats.fit_gsplats import fit_gaussian_splats  # noqa: E402

STAMPS = (
    "source_shape",
    "source_dtype",
    "source_voxels",
    "source_bytes",
    "fitted_shape",
    "fitted_voxels",
    "occupancy",
    "voxels_per_splat",
)


def _sparse_blobs(
    shape=(24, 32, 32), n=20, dtype=np.uint16, pedestal=0, noise=0.0
) -> np.ndarray:
    """A sparse blob field: sparse enough that occupancy is a meaningful number.

    ``pedestal``/``noise`` add the constant offset and read noise a real
    acquisition has. Without them the background is EXACTLY zero, which is the
    one input on which a `> 0` occupancy test cannot go wrong.
    """
    rng = np.random.default_rng(0)
    V = np.zeros(shape, dtype=np.float32)
    hi = [s - 2 for s in shape]
    for _ in range(n):
        idx = rng.integers([2] * len(shape), hi)
        sl = tuple(slice(i - 1, i + 2) for i in idx)
        V[sl] = 3000
    if pedestal or noise:
        V = np.clip(V + pedestal + rng.normal(0, noise, size=shape), 0, None)
    return np.round(V).astype(dtype)


def _fit(V, **kw):
    return fit_gaussian_splats(
        V=V,
        seeds=120,
        n_iters=30,
        verbose=False,
        device="cpu",
        enable_dynamic_ops=False,
        napari_movie=False,
        **kw,
    )


def test_fit_stamps_the_source_grid() -> None:
    V = _sparse_blobs()
    stats = _fit(V).stats
    for key in STAMPS:
        assert key in stats, f"missing stamp {key!r}"
    assert stats["source_shape"] == list(V.shape)
    assert stats["source_dtype"] == str(V.dtype)
    assert stats["source_voxels"] == V.size
    assert stats["source_bytes"] == V.nbytes
    # A uint16 volume must not be recorded as the float the fitter works in:
    # the byte count is the denominator of every compression ratio.
    assert stats["source_dtype"] == "uint16"
    assert 0.0 < stats["occupancy"] < 1.0
    assert stats["voxels_per_splat"] > 0


def test_downscaling_keeps_the_two_grids_apart() -> None:
    """``source_*`` is what was handed in; ``fitted_*`` is what was optimised.

    Collapsing the two would silently overstate compression by the downscale
    factor cubed — the one mistake these stamps exist to prevent.
    """
    V = _sparse_blobs(shape=(24, 32, 32))
    stats = _fit(V, downscale=2).stats
    assert stats["source_shape"] == [24, 32, 32]
    assert stats["source_voxels"] == V.size
    assert stats["fitted_shape"] != stats["source_shape"], (
        "fitted grid was not downscaled, or it was recorded as the source grid"
    )
    assert stats["fitted_voxels"] < stats["source_voxels"]


def test_grids_agree_when_nothing_is_downscaled() -> None:
    stats = _fit(_sparse_blobs()).stats
    assert stats["fitted_shape"] == stats["source_shape"]
    assert stats["fitted_voxels"] == stats["source_voxels"]


def test_occupancy_tracks_how_full_the_volume_is() -> None:
    """A denser volume must report a higher occupancy than a sparse one.

    Pinning an absolute value would only restate the fixture; the property that
    matters is that the number responds to the data.
    """
    sparse = _fit(_sparse_blobs(n=5)).stats["occupancy"]
    dense = _fit(_sparse_blobs(n=60)).stats["occupancy"]
    assert dense > sparse, f"occupancy did not respond to density: {dense} vs {sparse}"


def test_occupancy_measures_signal_not_camera_noise() -> None:
    """A noisy background must not read as occupied volume.

    The stamp is printed as the caveat on the compression ratio, so a number that
    says "58% full" about a stack that is 2% signal is worse than no number: it
    contradicts the ratio it stands beside. A pedestal with read noise is what
    every real acquisition looks like, and a plain "above the subtracted floor"
    test counts about half of it — the floor sits at the background's own level,
    so half the background is above it.

    Default floor suppression, i.e. what a real fit runs. Under ``--floor none``
    the pedestal is not subtracted and does count, which is the honest answer
    there: those voxels are content the fit spent splats representing.
    """
    V = _sparse_blobs(pedestal=100, noise=8.0)
    truth = 20 * 27 / V.size  # 20 blobs of 3x3x3, before overlaps
    occ = _fit(V).stats["occupancy"]
    assert occ < 4 * truth, (
        f"occupancy {occ:.4f} reads noise as signal (truth ~{truth:.4f})"
    )
    assert occ > 0.0


def test_stamps_reach_the_fitting_group_on_disk(tmp_path: Path) -> None:
    """They must survive the save, not just live in the in-memory stats.

    ``split_fitting_info`` whitelists what lands in ``fitting/``; a key missing
    from that list falls through to ``pipeline/`` instead, which is where a
    reader looking for provenance would not think to look.
    """
    V = _sparse_blobs()
    out = tmp_path / "stamped.gsplats.zarr"
    if out.exists():
        shutil.rmtree(out)
    _fit(V).save(out)
    attrs = read_node_attrs(out / "fitting")
    for key in STAMPS:
        assert key in attrs, f"{key!r} did not reach fitting/ on disk"
    assert attrs["source_bytes"] == V.nbytes

    # The point of the exercise: compression is now computable from the artifact.
    stored = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    assert stored > 0
    assert attrs["source_bytes"] / stored > 0

    # That `gsplat info` then SURFACES these is asserted in the CLI package
    # (`cli/tests/test_gsplat_info_source_grid.py`): a domain test may not
    # import `luxar.cli` — the import-linter layer contract forbids it, and it
    # is checked in its own CI step rather than by the test run.


def test_a_caller_that_already_cast_can_name_the_stored_dtype() -> None:
    """``source_dtype`` overrides what ``V`` reports.

    The CLI loads through ``load_volume``, which returns float32 whatever the
    file holds — so on the path that produces essentially every stored dataset
    the fitter never sees the acquisition's element type, and measuring ``V``
    would report the float working copy (2x too many bytes for a 16-bit stack).
    """
    V = _sparse_blobs().astype(np.float32)  # as load_volume would hand it over
    stats = _fit(V, source_dtype="uint16").stats
    assert stats["source_dtype"] == "uint16"
    assert stats["source_bytes"] == stats["source_voxels"] * 2
    # Without the override the same array records its float32 self, which is
    # what makes passing it necessary rather than decorative.
    assert _fit(V).stats["source_bytes"] == stats["source_voxels"] * 4


@pytest.mark.parametrize(
    "explicit", [np.dtype("uint16"), np.uint16, "uint16"], ids=["dtype", "type", "str"]
)
def test_an_explicit_dtype_object_is_normalized_to_its_name(explicit: object) -> None:
    """A dtype OBJECT is what a Python caller naturally passes.

    It used to be stored verbatim, so the whole optimisation completed and then
    died in ``.save()`` with "Object of type UInt16DType is not JSON
    serializable" — after all the compute, which is the worst possible time.
    """
    from luxar.gsplats.fitting.validation import _resolve_source_dtype

    name, itemsize = _resolve_source_dtype(np.zeros((2, 2), dtype=np.float32), explicit)
    assert name == "uint16"
    assert isinstance(name, str)
    assert itemsize == 2


def test_a_blank_explicit_dtype_behaves_as_absent() -> None:
    """``""`` must not be stored: it yields a ``None`` itemsize, which sends
    ``results.py`` to the post-cast float32 ``V.nbytes`` — the 2x-overstated
    source size the stamp exists to prevent."""
    from luxar.gsplats.fitting.validation import _resolve_source_dtype

    V = np.zeros((2, 2), dtype=np.uint16)
    assert _resolve_source_dtype(V, "") == ("uint16", 2)
    assert _resolve_source_dtype(V, "   ") == ("uint16", 2)


def test_an_unrecognizable_dtype_name_is_recorded_without_a_size() -> None:
    """Tolerance, not a crash: the name is kept, the itemsize is left unknown."""
    from luxar.gsplats.fitting.validation import _resolve_source_dtype

    name, itemsize = _resolve_source_dtype(np.zeros(2, dtype=np.float32), "mystery12")
    assert name == "mystery12"
    assert itemsize is None


def test_an_unsizable_dtype_records_no_byte_count() -> None:
    """The recorded dtype and the recorded bytes must describe the SAME type.

    A dtype name numpy cannot size is kept verbatim, so the byte count must not
    quietly come from the float32 working copy instead: paired that way, ``info``
    would report a compression ratio inflated by the cast — the very error the
    stamp exists to prevent. Silence about the source size is what ``info``
    already does for a dataset that never carried one.
    """
    stats = _fit(_sparse_blobs(shape=(8, 8, 8), n=4), source_dtype="mystery12").stats
    assert stats["source_dtype"] == "mystery12"
    assert "source_bytes" not in stats, stats.get("source_bytes")
    # The rest of the stamp is unaffected — only the size is unknowable.
    assert stats["source_shape"] == [8, 8, 8]
    assert stats["source_voxels"] == 512


def test_a_dtype_object_survives_the_save(tmp_path: Path) -> None:
    """End to end: the JSON serialization that used to blow up at the very end."""
    V = _sparse_blobs(shape=(8, 8, 8), n=4).astype(np.float32)
    out = tmp_path / "dtype_object.gsplats.zarr"
    _fit(V, source_dtype=np.dtype("uint16")).save(out)
    attrs = read_node_attrs(out / "fitting")
    assert attrs["source_dtype"] == "uint16"
    assert attrs["source_bytes"] == V.size * 2


def test_load_volume_reports_the_stored_dtype(tmp_path: Path) -> None:
    """The loader is the last place the on-disk element type exists."""
    from luxar.io.volume import load_volume

    src = tmp_path / "vol.npy"
    np.save(src, _sparse_blobs(shape=(8, 8, 8)))
    info: dict = {}
    volume = load_volume(src, info=info)
    assert volume.dtype == np.float32  # unchanged contract
    assert info["source_dtype"] == "uint16"


# The end-to-end `gsplat fit` counterpart of the test above — that a fit driven
# from a uint16 FILE records uint16 and not the loader's float32 — lives in
# `cli/tests/test_gsplat_info_source_grid.py`, for the layering reason given at
# `test_stamps_reach_the_fitting_group_on_disk`.


# That a sub-unity compression ratio prints as itself rather than rounding to
# "0:1" is asserted in `cli/tests/test_gsplat_info_source_grid.py` too — the
# ratio is formatted by the printer, and a domain test may not import it.


def test_a_whitespace_padded_dtype_is_still_sized() -> None:
    """A quoted YAML `source_dtype: "uint16 "` must not lose its byte count."""
    from luxar.gsplats.fitting.validation import _resolve_source_dtype

    V = np.zeros((2, 2), dtype=np.float32)
    assert _resolve_source_dtype(V, "uint16 ") == ("uint16", 2)
    assert _resolve_source_dtype(V, " uint16") == ("uint16", 2)


# ── Declaring the acquisition when the caller preprocessed first ──────────────
#
# Most producers do not hand the fitter the acquisition. They pull one channel
# out of a 5D store, downscale it and normalize it — so the array the fitter
# sees is a working copy, and a ratio measured against it answers a question
# nobody asked.


def test_a_declared_source_grid_overrides_the_array_handed_in() -> None:
    V = _sparse_blobs(shape=(24, 32, 32))  # the 128^3-style working copy
    stats = _fit(V, source_shape=(96, 128, 128), source_dtype="uint16").stats
    assert stats["source_shape"] == [96, 128, 128]
    assert stats["source_voxels"] == 96 * 128 * 128
    assert stats["source_bytes"] == 96 * 128 * 128 * 2
    # The optimiser still reports what it actually saw.
    assert stats["fitted_shape"] == [24, 32, 32]


def test_a_declared_grid_is_marked_as_declared() -> None:
    """A stated denominator must not be indistinguishable from a measured one."""
    V = _sparse_blobs(shape=(24, 32, 32))
    assert _fit(V, source_shape=(48, 64, 64)).stats["source_declared"] is True
    assert "source_declared" not in _fit(V).stats


@pytest.mark.parametrize(
    "declared", [None, (96, 128, 128)], ids=["measured", "declared"]
)
def test_source_bytes_always_agrees_with_the_grid_it_is_quoted_against(
    declared: object,
) -> None:
    """``source_bytes`` must be voxels x itemsize OF THE RECORDED GRID.

    This is the invariant a compression ratio rests on. It is stated as a
    relation rather than as a branch test on purpose: the byte count has more
    than one origin inside the helper, and only the relation distinguishes
    "counted the grid we published" from "counted the array we happened to
    hold" — which for a declared grid are different arrays entirely.
    """
    V = _sparse_blobs(shape=(24, 32, 32))
    stats = _fit(V, source_shape=declared, source_dtype="uint16").stats
    assert stats["source_bytes"] == stats["source_voxels"] * 2
    assert stats["source_voxels"] == int(np.prod(stats["source_shape"]))


def test_an_unsizable_declared_dtype_reports_no_bytes_at_all() -> None:
    """No number beats a number measured against the wrong array."""
    stats = _fit(
        _sparse_blobs(shape=(24, 32, 32)),
        source_shape=(96, 128, 128),
        source_dtype="not-a-dtype",
    ).stats
    assert stats["source_dtype"] == "not-a-dtype"
    assert "source_bytes" not in stats


@pytest.mark.parametrize(
    "bad",
    [(), (0, 8, 8), (-4, 8, 8), ("a", 8, 8), (12.9, 16, 16), "128", 96, (True, 8, 8)],
    ids=[
        "empty",
        "zero",
        "neg",
        "str",
        "fractional",
        "bare-string",
        "bare-int",
        "bool",
    ],
)
def test_a_malformed_declared_grid_is_refused(bad: object) -> None:
    """It becomes the denominator of a published ratio — fail here, not later.

    Two of these would pass a bare ``int(x)`` per element with no complaint, and
    both are silent rather than loud afterwards: ``(12.9, 16, 16)`` truncates to
    a grid the caller never meant, and the bare ``"128"`` iterates into
    ``[1, 2, 8]`` — a denominator wrong by four orders of magnitude, which is
    exactly the plausible-looking figure this validation exists to stop.
    """
    with pytest.raises(ValueError):
        _fit(_sparse_blobs(shape=(8, 8, 8)), source_shape=bad)


def test_the_declared_marker_reaches_the_fitting_group_beside_the_grid(
    tmp_path: Path,
) -> None:
    """The marker is worthless one group away from the number it qualifies.

    ``split_fitting_info`` whitelists ``fitting/``; a key missing from that list
    lands in ``pipeline/`` — reduction/topology provenance — where a reader
    holding ``fitting/source_shape`` has no reason to look, and so cannot tell a
    stated denominator from a measured one.
    """
    out = tmp_path / "declared.gsplats.zarr"
    if out.exists():
        shutil.rmtree(out)
    _fit(_sparse_blobs(shape=(24, 32, 32)), source_shape=(96, 128, 128)).save(out)
    attrs = read_node_attrs(out / "fitting")
    assert attrs["source_shape"] == [96, 128, 128]
    assert attrs["source_declared"] is True

    # And a measured grid must not acquire the marker on the way to disk.
    plain = tmp_path / "measured.gsplats.zarr"
    _fit(_sparse_blobs(shape=(24, 32, 32))).save(plain)
    assert "source_declared" not in (read_node_attrs(plain / "fitting") or {})


def test_an_exact_integral_dimension_is_still_accepted() -> None:
    """Strictness must not reject the dimensions callers legitimately compute.

    A shape read off a numpy array, or divided out exactly, is a fine
    declaration; only a value that is not the integer it claims to be is not.
    """
    from luxar.gsplats.fitting.validation import _explicit_source_shape

    assert _explicit_source_shape(np.array([96, 128, 128])) == [96, 128, 128]
    assert _explicit_source_shape((96.0, 128.0, 128.0)) == [96, 128, 128]
    assert _explicit_source_shape([236 // 2, 16, 16]) == [118, 16, 16]
