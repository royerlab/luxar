"""Direct unit tests for the label CSR serializers in luxar.io._compiler.labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.io._compiler.labels.image_labels import (
    normalize_image_label,
    write_image_labels_csr,
)
from luxar.io._compiler.labels.text_labels import write_labels_csr
from luxar.io.reader import DEFAULT_COMP
from luxar.validation.writing import (
    check_image_label_type,
    validate_image_labels_for_writing,
)


def _group() -> zarr.Group:
    return zarr.group()


def _decode(group: zarr.Group, i: int) -> bytes:
    offsets = group["label_offsets"][:]
    data = group["label_bytes"][:]
    return data[offsets[i] : offsets[i + 1]].tobytes()


def test_write_labels_csr_roundtrip() -> None:
    g = _group()
    labels = ["alpha", "", "gamma"]
    write_labels_csr(g, labels, 3, DEFAULT_COMP)
    assert g.attrs["has_labels"] is True
    assert _decode(g, 0) == b"alpha"
    assert _decode(g, 1) == b""
    assert _decode(g, 2) == b"gamma"


def test_write_labels_csr_applies_sort_order() -> None:
    g = _group()
    labels = ["a", "b", "c"]
    write_labels_csr(g, labels, 3, DEFAULT_COMP, sort_order=np.array([2, 0, 1]))
    assert _decode(g, 0) == b"c"
    assert _decode(g, 1) == b"a"
    assert _decode(g, 2) == b"b"


def test_write_labels_csr_length_mismatch_raises() -> None:
    g = _group()
    try:
        write_labels_csr(g, ["only-one"], 3, DEFAULT_COMP)
    except ValueError as exc:
        assert "must match element count" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on length mismatch")


def test_normalize_image_label_passthrough_and_none() -> None:
    assert normalize_image_label(None) == b""
    assert normalize_image_label(b"\x01\x02") == b"\x01\x02"


def test_write_image_labels_csr_bytes_roundtrip() -> None:
    g = _group()
    blobs = [b"\xff\xd8", b"", b"\x89PNG"]
    write_image_labels_csr(g, blobs, 3, DEFAULT_COMP)
    assert g.attrs["has_image_labels"] is True
    offsets = g["image_label_offsets"][:]
    data = g["image_label_bytes"][:]
    assert data[offsets[0] : offsets[1]].tobytes() == b"\xff\xd8"
    assert offsets[1] == offsets[2]  # empty entry
    assert data[offsets[2] : offsets[3]].tobytes() == b"\x89PNG"


def test_write_image_labels_csr_sparse_dict() -> None:
    g = _group()
    write_image_labels_csr(g, {1: b"\x01\x02\x03"}, 3, DEFAULT_COMP)
    offsets = g["image_label_offsets"][:]
    assert offsets[0] == offsets[1]  # index 0 empty
    assert int(offsets[2] - offsets[1]) == 3  # index 1 has the blob


# ---------------------------------------------------------------------------
# validate_image_labels_for_writing (issue #1491) — the pure length/index
# checks write_image_labels_csr used to run inline, extracted so a caller can
# run them BEFORE any zarr write (see core/group/compositing.py's pre-split
# gates for the second, higher-up caller).
# ---------------------------------------------------------------------------


def test_validate_image_labels_for_writing_length_mismatch_raises() -> None:
    try:
        validate_image_labels_for_writing([b"a", b"b"], 3)
    except ValueError as exc:
        assert str(exc) == ("Image labels length (2) must match element count (3)")
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on length mismatch")


def test_validate_image_labels_for_writing_sparse_out_of_range_raises() -> None:
    try:
        validate_image_labels_for_writing({5: b"a"}, 3)
    except ValueError as exc:
        assert str(exc) == "Image label index 5 out of range [0, 3)"
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on an out-of-range sparse index")


def test_validate_image_labels_for_writing_negative_sparse_index_raises() -> None:
    try:
        validate_image_labels_for_writing({-1: b"a"}, 3)
    except ValueError as exc:
        assert str(exc) == "Image label index -1 out of range [0, 3)"
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on a negative sparse index")


def test_validate_image_labels_for_writing_accepts_valid_dense_and_sparse() -> None:
    # Neither call raises — a dense list of the right length, and a sparse dict
    # whose keys are all in bounds (missing keys just mean "no image").
    validate_image_labels_for_writing([b"a", b"b", b"c"], 3)
    validate_image_labels_for_writing({0: b"a", 2: b"c"}, 3)


class _OneShotImageLabels:
    """A ``Sized`` container whose ``__iter__`` returns the SAME, single-pass
    iterator every call — unlike ``list`` / ``tuple`` / ``ndarray`` /
    ``pandas.Series``, which are all re-iterable. A second full walk over one
    of these sees nothing, since the first walk already exhausted it.
    """

    def __init__(self, items: list) -> None:
        self._iter = iter(items)
        self._n = len(items)

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        return self._iter


def test_write_image_labels_csr_materializes_a_one_shot_iterable_once() -> None:
    """A single-pass ``Sized`` iterable is walked exactly ONCE (#1491).

    Before this fix, ``validate_image_labels_for_writing``'s own type-check
    loop drained a one-shot input, leaving the blob-encoding loop nothing to
    see — a SILENT all-empty CSR (``offsets`` all zero, ``has_image_labels``
    still stamped ``True``) rather than a raised error. That failure class is
    exactly what #1491 is about, so it must not be reintroduced even for an
    unusual input. Materialising once, before validating, means a
    writer-only call (no higher pre-split gate has touched the input yet)
    still round-trips correctly.
    """
    g = _group()
    items = [b"AAA", b"BBB", b"CCC"]
    write_image_labels_csr(g, _OneShotImageLabels(items), 3, DEFAULT_COMP)
    offsets = g["image_label_offsets"][:]
    data = g["image_label_bytes"][:]
    assert list(offsets) == [0, 3, 6, 9]
    assert bytes(data) == b"AAABBBCCC"


def test_validate_image_labels_for_writing_refuses_a_one_shot_iterable() -> None:
    """The PRE-WRITE gate refuses a one-shot dense iterable outright (#1491).

    It cannot both validate every entry's type and leave the sequence intact
    for its caller to encode, and it has no way to return its own materialised
    copy — so draining it would leave the writer nothing to see and stamp an
    all-empty CSR. Refusing the container up front is what keeps every gate
    that calls this (a writer's step-0 sweep, a ``substitutive_lod=``
    wrapper's pre-split gate) from stranding a node.
    """
    with pytest.raises(TypeError, match="must be a re-iterable sequence"):
        validate_image_labels_for_writing(_OneShotImageLabels([b"a", b"b", b"c"]), 3)

    # A re-iterable sequence of the same content is accepted, and repeatedly:
    # the check must not consume anything it looks at.
    labels = [b"a", b"b", b"c"]
    validate_image_labels_for_writing(labels, 3)
    validate_image_labels_for_writing(labels, 3)


def test_validate_image_labels_for_writing_rejects_a_non_integral_sparse_key() -> None:
    """An in-range but non-integral sparse key is refused BEFORE the write.

    ``{1.5: b"x"}`` cleared the bounds check (``0 <= 1.5 < 3``) and then
    raised ``TypeError: list indices must be integers or slices, not float``
    from ``write_image_labels_csr``'s own ``normalized[idx]`` subscript —
    post-write, with the caller's other arrays already on disk. Same for a
    ``np.float64``, which is how it arises in practice (an arithmetic slip on
    an index). ``int`` / ``bool`` / any numpy integer satisfy
    ``operator.index`` and are unaffected.
    """
    for key in (1.5, np.float64(1.0), "0"):
        with pytest.raises(TypeError, match="index must be an integer"):
            validate_image_labels_for_writing({key: b"x"}, 3)

    # The control: a numpy integer key IS a legal index and still round-trips.
    g = _group()
    write_image_labels_csr(g, {np.int64(1): b"\x01\x02"}, 3, DEFAULT_COMP)
    assert list(g["image_label_offsets"][:]) == [0, 0, 2, 2]


def test_write_image_labels_csr_reports_an_already_drained_one_shot_iterable() -> None:
    """If the CALLER already walked a one-shot iterable before handing it over,
    this function's own materialisation sees zero items and its length check
    reports that clearly, instead of writing an all-empty CSR silently. (The
    gates above never reach this state — they refuse a one-shot container
    outright; see the test above.)
    """
    items = [b"AAA", b"BBB", b"CCC"]
    drained = _OneShotImageLabels(items)
    list(drained)  # simulate an earlier gate's full walk over the same object
    g = _group()
    try:
        write_image_labels_csr(g, drained, 3, DEFAULT_COMP)
    except ValueError as exc:
        assert str(exc) == "Image labels length (0) must match element count (3)"
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on an already-drained iterable")


def test_write_image_labels_csr_still_raises_the_same_messages() -> None:
    """The CSR writer's own checks now delegate to the validator — same wording.

    Pins that extracting the checks (#1491) did not change what a DIRECT
    ``write_image_labels_csr`` call raises, for a caller that never goes
    through the higher pre-split gate.
    """
    g = _group()
    try:
        write_image_labels_csr(g, [b"a"], 3, DEFAULT_COMP)
    except ValueError as exc:
        assert str(exc) == "Image labels length (1) must match element count (3)"
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on length mismatch")

    try:
        write_image_labels_csr(g, {7: b"a"}, 3, DEFAULT_COMP)
    except ValueError as exc:
        assert str(exc) == "Image label index 7 out of range [0, 3)"
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on an out-of-range sparse index")


# ---------------------------------------------------------------------------
# check_image_label_type (issue #1491) — the TYPE-dispatch half pulled out of
# normalize_image_label so validate_image_labels_for_writing can run it over
# every entry PRE-write too (a right-length/right-shaped list with one
# mistyped entry stranded exactly like a wrong-length one did).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item",
    [None, b"\x01\x02", bytearray(b"\x01"), Path("/does/not/exist.png"), "not-a-file"],
)
def test_check_image_label_type_accepts_every_documented_type_without_reading(
    item: object,
) -> None:
    """None / bytes / bytearray / Path / str are accepted by TYPE alone.

    None of these can raise from a bad path or a missing file — that read
    only happens in :func:`normalize_image_label`, never here.
    """
    check_image_label_type(item)  # must not raise


@pytest.mark.parametrize(
    "shape",
    [(2, 2), (2, 2, 3), (2, 2, 4)],
)
def test_check_image_label_type_accepts_every_documented_ndarray_shape(
    shape: tuple,
) -> None:
    """An ndarray of (H,W) / (H,W,3) / (H,W,4) is accepted by type AND shape."""
    check_image_label_type(np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize(
    "shape",
    [(5,), (2, 2, 2), (2, 2, 2, 2)],
)
def test_check_image_label_type_rejects_an_unsupported_ndarray_shape(
    shape: tuple,
) -> None:
    """The (H,W[,3|4]) SHAPE check now runs EAGERLY here, not deferred to a PIL
    round-trip — ``ndim``/``shape[2]`` are pure attribute reads, so there is no
    reason to defer them (#1491). Pre-fix this shape reached
    :func:`normalize_image_label`'s own ``fromarray`` call unchecked.
    """
    bad = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported ndarray shape"):
        check_image_label_type(bad)


def test_check_image_label_type_rejects_an_unsupported_type() -> None:
    try:
        check_image_label_type(12345)
    except TypeError as exc:
        assert str(exc) == (
            "Unsupported image label type: int. "
            "Expected bytes, PIL.Image, numpy.ndarray, or file path."
        )
    else:  # pragma: no cover
        raise AssertionError("expected TypeError for an unsupported type")


def test_normalize_image_label_calls_the_same_type_check_first() -> None:
    """normalize_image_label's own dispatch agrees with check_image_label_type.

    Both must reject the same unsupported type with the identical message —
    that identity is the entire point of factoring the dispatch out.
    """
    try:
        normalize_image_label(12345)
    except TypeError as exc:
        assert str(exc) == (
            "Unsupported image label type: int. "
            "Expected bytes, PIL.Image, numpy.ndarray, or file path."
        )
    else:  # pragma: no cover
        raise AssertionError("expected TypeError for an unsupported type")


# ---------------------------------------------------------------------------
# validate_image_labels_for_writing now ALSO type-checks every entry (#1491),
# not just length/index — and does so in a fixed, deliberate order: every
# key's BOUND (sparse) / every entry's LENGTH (dense) first, THEN every
# entry's TYPE. That reorders behaviour for BOTH the sparse dict form and the
# dense sequence form, not only the dict one.
# ---------------------------------------------------------------------------


def test_validate_image_labels_for_writing_rejects_a_mistyped_entry() -> None:
    """A right-length list with one bad-typed entry is refused, not silently kept.

    Pre-#1491 this was refused only from inside normalize_image_label, deep in
    the writer, after every earlier array was already on disk.
    """
    try:
        validate_image_labels_for_writing([b"a", 12345, b"c"], 3)
    except TypeError as exc:
        assert str(exc) == (
            "Unsupported image label type: int. "
            "Expected bytes, PIL.Image, numpy.ndarray, or file path."
        )
    else:  # pragma: no cover
        raise AssertionError("expected TypeError for the mistyped entry")


def test_validate_image_labels_for_writing_sparse_bounds_outrank_types() -> None:
    """Deliberate order divergence from the pre-#1491 interleaved check (#1491).

    Pre-fix, ``write_image_labels_csr`` checked one key's bound and immediately
    ran ``normalize_image_label`` on that key's value before moving to the next
    key, so ``{0: 12345, 5: b"x"}`` against ``n_elements=3`` raised
    ``TypeError: Unsupported image label type: int`` from key 0's value —
    key 5's out-of-range index was never reached. ALL key bounds are now
    checked before ANY value's type, so the same call now raises the index
    error instead: a different exception TYPE and message. See
    ``labels/README.md`` for the write-up; this is the direct pin.
    """
    try:
        validate_image_labels_for_writing({0: 12345, 5: b"x"}, 3)
    except ValueError as exc:
        assert str(exc) == "Image label index 5 out of range [0, 3)"
    else:  # pragma: no cover
        raise AssertionError("expected ValueError naming the out-of-range index")

    # And write_image_labels_csr — a caller with no higher pre-split gate of
    # its own — shows the identical divergence, since it now delegates both
    # checks to this same function.
    g = _group()
    try:
        write_image_labels_csr(g, {0: 12345, 5: b"x"}, 3, DEFAULT_COMP)
    except ValueError as exc:
        assert str(exc) == "Image label index 5 out of range [0, 3)"
    else:  # pragma: no cover
        raise AssertionError("expected ValueError naming the out-of-range index")


def test_validate_image_labels_for_writing_dense_types_outrank_file_reads() -> None:
    """The SAME order divergence (#1491), but for the DENSE sequence form.

    Pre-fix, ``normalize_image_label`` walked the list in order, so a missing
    file at an EARLIER index was read (and raised ``FileNotFoundError``)
    before a mistyped entry at a LATER index was ever inspected. The type
    sweep now runs over every entry before any file is opened, so the same
    list raises ``TypeError`` instead — a different exception TYPE, not just
    a different message. Arguably an improvement: ``TypeError`` IS caught by
    the leaf adders' ``except (ValueError, TypeError)`` funnel, while
    ``FileNotFoundError`` is not.
    """
    dense = [b"ok", "/definitely/does/not/exist/nope.png", 123]
    try:
        validate_image_labels_for_writing(dense, 3)
    except TypeError as exc:
        assert str(exc) == (
            "Unsupported image label type: int. "
            "Expected bytes, PIL.Image, numpy.ndarray, or file path."
        )
    else:  # pragma: no cover
        raise AssertionError("expected TypeError for the mistyped entry")


# ---------------------------------------------------------------------------
# The residual (#1491): what still can only be caught POST-write, because it
# needs an ACTUAL file read or an ACTUAL PIL encode — validated here so the
# boundary between "moved" and "still strands" is explicit rather than
# implied by what the docstrings say. The ``ndarray`` SHAPE check is
# deliberately NOT in this section any more: ``ndim``/``shape[2]`` are pure
# attribute reads (no PIL round-trip needed), so ``check_image_label_type``
# now validates them eagerly too — see the next test.
# ---------------------------------------------------------------------------


def test_validate_image_labels_for_writing_does_not_catch_an_unreadable_path() -> None:
    """A ``Path`` is accepted by TYPE alone — reading it is still write-time only."""
    bad_path = Path("/definitely/does/not/exist/image.png")
    # The pre-write gate is silent: Path is one of the types it accepts.
    validate_image_labels_for_writing([bad_path], 1)
    # Only normalize_image_label (called from write_image_labels_csr) actually
    # opens the file, and that is where the residual strand still lives.
    with pytest.raises(FileNotFoundError):
        normalize_image_label(bad_path)
    g = _group()
    with pytest.raises(FileNotFoundError):
        write_image_labels_csr(g, [bad_path], 1, DEFAULT_COMP)


def test_validate_image_labels_for_writing_now_catches_a_bad_ndarray_shape() -> None:
    """An ``ndarray``'s (H,W[,3|4]) shape is now caught PRE-write too (#1491).

    ``check_image_label_type`` validates ``ndim``/``shape[2]`` eagerly (no PIL
    round-trip needed for that), so this is no longer part of the genuine
    residual above — see ``test_check_image_label_type_rejects_an_unsupported_ndarray_shape``
    for the direct pin on ``check_image_label_type`` itself. This is the
    higher-level strand this closes: a right-length, right-TYPE list with one
    mis-shaped ``ndarray`` entry used to reach ``normalize_image_label``'s own
    ``fromarray`` call only from deep inside the writer (or, on a
    ``substitutive_lod=`` ladder, only after every coarser level was already
    on disk) — the exact same shape of strand a mistyped ``int`` entry had,
    for the same reason.
    """
    bad_shape = np.zeros((2, 2, 2, 2), dtype=np.uint8)  # not (H,W)/(H,W,3)/(H,W,4)
    with pytest.raises(ValueError, match="Unsupported ndarray shape"):
        validate_image_labels_for_writing([bad_shape], 1)
    with pytest.raises(ValueError, match="Unsupported ndarray shape"):
        normalize_image_label(bad_shape)
