"""Direct unit tests for the label CSR serializers in luxar.io._compiler.labels."""

from __future__ import annotations

import numpy as np
import zarr

from luxar.io._compiler.labels.image_labels import (
    normalize_image_label,
    write_image_labels_csr,
)
from luxar.io._compiler.labels.text_labels import write_labels_csr
from luxar.io.reader import DEFAULT_COMP


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
