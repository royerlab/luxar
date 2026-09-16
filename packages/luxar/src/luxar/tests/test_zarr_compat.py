"""Tests for :mod:`luxar._zarr_compat` — the zarr-format seam.

Luxar writes zarr format 3 by default and can still be told to write format 2,
while READING both unconditionally. These tests pin the parts of that
arrangement that would otherwise fail silently: a store that quietly came out in
the wrong format, a compressor that quietly became Blosc when RAW was meant, a
codec that quietly failed to translate between the two formats' spellings, and a
production writer that quietly stopped naming its compressor at all.

Most of the format assertions are parameterized over BOTH formats rather than
pinning the current default. That is the actual contract now — existing stores
stay format 2 and new ones are format 3, so both are live at once — and it also
means the file does not have to be rewritten the next time the default moves.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from luxar import _zarr_compat as zc

PROD_ROOT = Path(__file__).resolve().parents[1]

#: The root metadata document each format writes, for on-disk assertions.
ROOT_DOC = {2: ".zgroup", 3: "zarr.json"}

#: What ``compressor="auto"`` resolves to per format. NOT the same compressor:
#: "auto" means "whatever zarr would have picked", and zarr picked differently
#: in the two formats (Blosc/lz4/5 at 2, zstd at 3). Anything whose bytes matter
#: names its compressor; this is only for fixtures that never did.
AUTO_COMPRESSOR = {2: "blosc", 3: "zstd"}


@pytest.fixture(params=(2, 3), ids=("v2", "v3"))
def write_format(request: pytest.FixtureRequest):
    """Run a test once per write format, restoring the default afterwards.

    Uses the real override rather than monkeypatching the module attribute, so
    every test that takes this fixture is also a test that the override works.
    """
    original = zc.ZARR_FORMAT
    zc.set_zarr_format(request.param)
    try:
        yield request.param
    finally:
        zc.set_zarr_format(original)


@pytest.fixture
def v3_writes():
    """Pin format-3 writes for one test, restoring the default afterwards.

    A bare `set_zarr_format(3)` in a test body leaks into every test that runs
    after it in the same process, which is invisible while the default already is
    3 and confusing under `LUXAR_ZARR_FORMAT=2`.
    """
    original = zc.ZARR_FORMAT
    zc.set_zarr_format(3)
    try:
        yield 3
    finally:
        zc.set_zarr_format(original)


def _array_meta(path: Path, name: str) -> dict:
    """An array's raw metadata document, in whichever format it was written.

    Deliberately read off disk rather than through zarr's API: these tests are
    about what LANDS there, and zarr would normalise away the very differences
    being asserted.
    """
    v2 = path / name / ".zarray"
    if v2.exists():
        return json.loads(v2.read_text())
    return json.loads((path / name / "zarr.json").read_text())


def _only_chunk(path: Path, name: str) -> bytes:
    """The single stored chunk of a one-chunk array, in whichever format.

    The formats key chunks differently — ``codes/0`` at format 2, ``codes/c/0``
    at format 3 — so this collects every non-metadata file under the array and
    insists there is exactly one, rather than naming a layout.
    """
    array_dir = path / name
    chunks = sorted(
        f
        for f in array_dir.rglob("*")
        if f.is_file() and f.name not in (".zarray", ".zattrs", "zarr.json")
    )
    assert len(chunks) == 1, f"expected one chunk under {array_dir}, got {chunks}"
    return chunks[0].read_bytes()


def _shuffle_sensitive_codes() -> np.ndarray:
    """A deterministic uint16 ramp whose bytes compress better byte-shuffled.

    Quantization codes in a real store are Hilbert-ordered, so the high byte is
    nearly constant while the low byte churns — precisely what a byte shuffle
    separates. A flat ``arange`` would compress the same either way and would
    make an assertion on the shuffled bytes vacuous.
    """
    steps = np.random.default_rng(0xC0FFEE).integers(-5, 6, size=4096)
    return (np.cumsum(steps) % 65536).astype(np.uint16)


def _compressor_view(meta: dict) -> dict | None:
    """The array's compressor as a comparable dict, from either format.

    Format 2 records a single ``compressor`` object keyed by ``id``; format 3
    records an ordered ``codecs`` chain whose members are ``{name,
    configuration}`` and which always contains the mandatory ``bytes`` codec.
    Normalised to ``{"name": ..., **configuration}`` so assertions read the same
    either way. ``None`` means stored RAW.
    """
    if "compressor" in meta:  # format 2
        comp = meta["compressor"]
        if comp is None:
            return None
        return {"name": comp["id"], **{k: v for k, v in comp.items() if k != "id"}}
    compressors = [
        c
        for c in meta.get("codecs", [])
        if c.get("name") not in ("bytes", "transpose")
        and c.get("name") not in _FILTER_NAMES
    ]
    if not compressors:
        return None
    first = compressors[0]
    return {"name": first["name"], **first.get("configuration", {})}


#: Array-to-array codecs are FILTERS, not compressors; they must not be mistaken
#: for one when reading a format-3 chain.
_FILTER_NAMES = {"luxar_delta_v1"}


def _filter_names(meta: dict) -> list[str]:
    """The array's filters by name, from either format."""
    if "filters" in meta:  # format 2
        return [f["id"] for f in (meta["filters"] or [])]
    return [c["name"] for c in meta.get("codecs", []) if c["name"] in _FILTER_NAMES]


# ---------------------------------------------------------------------------
# The format we write
# ---------------------------------------------------------------------------


def test_the_default_write_format_is_3() -> None:
    """The declared default. Changing it is a deliberate, reviewed act."""
    assert zc.DEFAULT_ZARR_FORMAT == 3


def test_format_2_remains_producible() -> None:
    """The escape hatch must exist, and must be the only other option.

    Format 2 stays writable for a tool that cannot read 3; nothing else is, so
    a typo'd override fails loudly instead of picking a format nobody supports.
    """
    assert zc.SUPPORTED_ZARR_FORMATS == (2, 3)
    with pytest.raises(ValueError):
        zc.set_zarr_format(1)


def test_open_group_creates_the_selected_format(
    tmp_path: Path, write_format: int
) -> None:
    """A created store must be the requested format on disk, and only that one.

    Asserting the OTHER format's document is absent matters as much as asserting
    its own is present: `mode="a"` on an existing store can write a second root
    beside the first, leaving both documents and a store whose format depends on
    which reader looks.
    """
    p = tmp_path / "s.zarr"
    zc.open_group(p, mode="w")
    doc = p / ROOT_DOC[write_format]
    assert json.loads(doc.read_text())["zarr_format"] == write_format
    other = ROOT_DOC[2 if write_format == 3 else 3]
    assert not (p / other).exists(), f"also wrote {other}"


def test_open_group_ignores_the_ambient_default(
    tmp_path: Path, write_format: int
) -> None:
    """The facade must not depend on ``zarr.config``'s ambient default.

    Setting ``default_zarr_format`` globally would be an easier fix than
    threading ``zarr_format`` through, but it makes correctness depend on import
    order and on nobody else touching the config. The ambient default is set to
    the OPPOSITE format here; the store must still come out as asked.
    """
    opposite = 2 if write_format == 3 else 3
    with zarr.config.set({"default_zarr_format": opposite}):
        p = tmp_path / "cfg.zarr"
        zc.open_group(p, mode="w")
        doc = p / ROOT_DOC[write_format]
        assert doc.exists(), f"ambient default {opposite} won over the facade"
        assert json.loads(doc.read_text())["zarr_format"] == write_format


def test_every_group_constructor_pins_the_format(
    tmp_path: Path, write_format: int
) -> None:
    """All three constructors must pin it, not just `open_group`.

    They are separate code paths and only one of them is exercised by most
    tests, so a constructor that quietly followed zarr's default would go
    unnoticed until something read the store back.
    """
    assert zc.memory_group().metadata.zarr_format == write_format
    store = zc.open_store(tmp_path / "r.zarr", mode="w")
    assert zc.create_root_group(store).metadata.zarr_format == write_format
    p = tmp_path / "og.zarr"
    zc.open_group(p, mode="w")
    assert json.loads((p / ROOT_DOC[write_format]).read_text())["zarr_format"] == (
        write_format
    )


def test_read_is_version_agnostic(tmp_path: Path) -> None:
    """Being on zarr-python 3 is worth it precisely because v3 becomes readable.

    2.18 could not open a format-3 store at all, which is what blocked consuming
    zarr v3 output from other tools. Both formats must open through the facade.
    """
    for fmt in (2, 3):
        p = tmp_path / f"v{fmt}.zarr"
        g = zarr.create_group(store=str(p), zarr_format=fmt)
        g.attrs["marker"] = fmt
        assert zc.open_group(p, mode="r").attrs["marker"] == fmt


# ---------------------------------------------------------------------------
# The compressor hazard
# ---------------------------------------------------------------------------


def test_compressor_none_stores_raw_not_blosc(
    tmp_path: Path, write_format: int
) -> None:
    """``None`` must mean RAW.

    This is the sharp edge of the zarr-3 API: ``compressors`` defaults to
    ``"auto"``, so a writer that means "no compression" and simply omits the
    argument gets Blosc/lz4 instead. Luxar stores its packed label byte-blobs
    raw, and that must survive.
    """
    p = tmp_path / "raw.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(
        g, "blob", data=np.arange(16, dtype=np.uint8), chunks=(16,), compressor=None
    )
    assert _compressor_view(_array_meta(p, "blob")) is None


def test_array_compressor_never_reports_a_compressed_array_as_raw(
    tmp_path: Path, write_format: int
) -> None:
    """``array_compressor(a) is None`` must mean RAW in BOTH formats.

    Tests assert ``is None`` to mean "stored uncompressed" (the packed
    image-label byte blobs are the real case), so anything else answering None
    turns a compression regression into a green test.

    Format 3 got this wrong: the helper skipped every codec without a ``cname``
    on the theory that it was the mandatory ``bytes`` codec, and fell through to
    ``None``. Measured, ``.compressors`` holds ONLY bytes-to-bytes compressors —
    the ``bytes`` codec is in ``.serializer`` — so the skip was reached solely by
    a real non-blosc compressor, and a zstd-compressed array read back as raw
    while the identical format-2 array raised.
    """
    from numcodecs import Zstd

    from luxar.conftest import array_compressor

    p = tmp_path / "mixed.zarr"
    g = zc.open_group(p, mode="w")
    data = np.arange(16, dtype=np.uint16)
    zc.create_array(g, "raw", data=data, chunks=(16,), compressor=None)
    zc.create_array(g, "zstd", data=data, chunks=(16,), compressor=Zstd(level=9))

    opened = zc.open_group(p, mode="r")
    assert array_compressor(opened["raw"]) is None
    # Loud, in whichever way the format expresses it — never a quiet None.
    with pytest.raises((TypeError, AttributeError)):
        array_compressor(opened["zstd"])


def test_compressor_object_is_used_verbatim(tmp_path: Path, write_format: int) -> None:
    """A numcodecs codec must reach `.zarray` unaltered — this is Luxar's zstd-9 policy."""
    p = tmp_path / "z.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(
        g,
        "codes",
        data=np.arange(32, dtype=np.uint16),
        chunks=(16,),
        compressor=Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE),
    )
    c = _compressor_view(_array_meta(p, "codes"))
    assert c is not None
    # Format 2 spells shuffle as numcodecs' integer, format 3 as a name; both
    # must mean BYTE shuffle, which is the half of the policy that is measured.
    shuffle = c["shuffle"]
    assert (c["name"], c["cname"], c["clevel"]) == ("blosc", "zstd", 9)
    assert shuffle in (1, "shuffle"), f"lost byte shuffle: {shuffle!r}"


def test_the_recorded_shuffle_is_the_one_the_chunk_got(
    tmp_path: Path, write_format: int
) -> None:
    """The stored CHUNK must be what the policy's compressor produces.

    The test above reads the recorded configuration, which is exactly what a
    lost shuffle does NOT disturb — and format 3 can lose it. zarr's format-3
    ``BloscCodec`` hands numcodecs the SERIALIZED BYTE buffer rather than the
    typed array, so blosc infers ``typesize=1`` and the byte shuffle degrades to
    a no-op unless zarr can forward ``typesize`` explicitly — which it only does
    for ``numcodecs >= 0.16`` (hence the direct floor in ``pyproject.toml``).
    Below it, ``zarr.json`` still says ``typesize: 2, shuffle: shuffle`` while
    the bytes are the unshuffled ones; nothing raises, the round-trip is fine
    (blosc records its own parameters in the frame header, so a reader is
    unaffected), and the store is simply larger. Measured on a 200k-point scene:
    12.5% of the chunk bytes, more than the delta filter was added to win.

    So this asserts on the BYTES, which is the only place the difference is
    visible, and it does so for both formats — the chunk is byte-identical to
    the numcodecs encoding either way, which is the property that makes a
    format-2 store and a format-3 one cost the same.
    """
    codes = _shuffle_sensitive_codes()
    policy = Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE)
    shuffled = bytes(policy.encode(codes))
    plain = bytes(Blosc(cname="zstd", clevel=9, shuffle=Blosc.NOSHUFFLE).encode(codes))
    # Non-vacuity: on data the shuffle cannot help, storing the unshuffled bytes
    # would satisfy the assertion below by accident.
    assert len(shuffled) < len(plain), "sample data must reward the byte shuffle"

    p = tmp_path / "policy.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(g, "codes", data=codes, chunks=(codes.size,), compressor=policy)
    assert _only_chunk(p, "codes") == shuffled, (
        f"format {write_format} stored {len(_only_chunk(p, 'codes'))} bytes; the "
        f"byte-shuffled policy encoding is {len(shuffled)} and the unshuffled one "
        f"{len(plain)} — a match with the latter means the shuffle was dropped"
    )


def test_compressor_auto_is_the_formats_own_default(
    tmp_path: Path, write_format: int
) -> None:
    """``"auto"`` is the compatibility escape hatch, and it is FORMAT-dependent.

    It means "whatever zarr would have picked", which is not the same compressor
    in the two formats: Blosc/lz4/5 at format 2, zstd at format 3. Test fixtures
    that never named a compressor get that, and if zarr's choice drifts their
    bytes change — this is where that shows up. Anything whose bytes matter
    names its compressor instead.
    """
    p = tmp_path / "a.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(
        g, "x", data=np.arange(8, dtype=np.float32), chunks=(8,), compressor="auto"
    )
    c = _compressor_view(_array_meta(p, "x"))
    assert c is not None, '"auto" must not mean RAW'
    assert c["name"] == AUTO_COMPRESSOR[write_format]


def test_the_compressor_DEFAULT_is_auto_not_raw(
    tmp_path: Path, write_format: int
) -> None:
    """Omitting ``compressor`` must mean "auto", never RAW.

    Distinct from the ``compressor="auto"`` test above, which passes the value
    explicitly and so cannot notice the DEFAULT changing. Mutation testing found
    that gap: flipping the default to ``None`` left the whole suite green while
    silently turning ~79 test fixtures from compressed to RAW — a byte change
    with no failing test anywhere.
    """
    p = tmp_path / "default.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(g, "d", data=np.arange(8, dtype=np.float32), chunks=(8,))
    c = _compressor_view(_array_meta(p, "d"))
    assert c is not None, "the default must not be RAW"
    assert c["name"] == AUTO_COMPRESSOR[write_format]


def test_chunks_true_is_translated_and_never_reaches_zarr(tmp_path: Path) -> None:
    """``chunks=True`` must be translated, not forwarded.

    zarr 3 rejects a bool outright, and `ChunkSpec` defaults to ``True``, so a
    broken translation breaks every resizable array. Mutation testing found this
    unpinned in the facade's own suite: the only coverage lived in a different
    test file, via a helper that only tests exercise.
    """
    g = zc.memory_group()
    a = zc.create_array(
        g, "auto", shape=(64, 3), dtype=np.float32, chunks=True, compressor=None
    )
    assert isinstance(a.chunks, tuple) and all(isinstance(c, int) for c in a.chunks)
    # `False` means one chunk spanning the array, and must not become "auto".
    b = zc.create_array(
        g, "whole", shape=(64, 3), dtype=np.float32, chunks=False, compressor=None
    )
    assert b.chunks == (64, 3)
    # An int is a real chunk size and must survive as one — bool subclasses int,
    # so an equality-based check here would conflate `True` with `1`.
    c = zc.create_array(
        g, "sized", shape=(64,), dtype=np.float32, chunks=8, compressor=None
    )
    assert c.chunks == (8,)


def test_a_filter_lands_where_its_format_records_filters(
    tmp_path: Path, write_format: int
) -> None:
    """``luxar_delta_v1`` must reach the metadata under the SAME name either way.

    The two formats file a filter differently — format 2 in the ``.zarray``
    ``filters`` list, format 3 as an array-to-array member of the ``zarr.json``
    ``codecs`` chain — and the viewer resolves each through a different zarrita
    registry namespace. What must not vary is the NAME, because a single
    TypeScript codec serves both.

    Writers always hand the facade a numcodecs filter, so at format 3 this is
    also the test that the translation happened at all: an untranslated
    numcodecs object does not survive into a format-3 chain.
    """
    from luxar.encoding._encoders.delta_codec import LuxarDelta

    p = tmp_path / "f.zarr"
    g = zc.open_group(p, mode="w")
    codes = np.arange(24, dtype=np.uint16).reshape(8, 3)
    zc.create_array(
        g,
        "d",
        data=codes,
        chunks=(8, 3),
        compressor=None,
        filters=[LuxarDelta(cols=3, bits=16)],
    )
    assert _filter_names(_array_meta(p, "d")) == ["luxar_delta_v1"]
    # ...and it must still decode, which a name alone does not prove.
    assert np.array_equal(zc.open_group(p, mode="r")["d"][:], codes)


def test_a_filter_with_no_format_3_twin_is_refused(tmp_path: Path) -> None:
    """A filter that cannot be translated must fail the WRITE, not be dropped.

    Dropping it would produce codes no reader can invert: a store that looks
    valid and decodes to garbage. numcodecs' own ``Delta`` is the case in point
    — it is a legitimate format-2 filter with no registered format-3 codec.
    """
    from numcodecs import Delta

    original = zc.ZARR_FORMAT
    zc.set_zarr_format(3)
    try:
        g = zc.open_group(tmp_path / "orphan.zarr", mode="w")
        with pytest.raises(ValueError, match="no format-3 codec registered"):
            zc.create_array(
                g,
                "d",
                data=np.arange(16, dtype=np.uint16),
                chunks=(16,),
                compressor=None,
                filters=[Delta(dtype="<u2")],
            )
    finally:
        zc.set_zarr_format(original)


def test_unknown_codec_errors_are_resolved_at_import() -> None:
    expected = getattr(zarr.errors, "UnknownCodecError", KeyError)
    assert zc._UNKNOWN_CODEC_ERRORS == (KeyError, expected)


# ---------------------------------------------------------------------------
# create_array's zarr-2 argument tolerance
# ---------------------------------------------------------------------------


def test_data_and_shape_together_are_accepted(tmp_path: Path) -> None:
    """zarr 3 raises on ``data=`` + ``shape=``; zarr 2 did not, and callers pass both.

    ``luxar.io._compiler.spatial_ordering`` passes data, shape AND dtype for its
    chunk-bounds arrays.
    """
    g = zc.memory_group()
    data = np.arange(12, dtype=np.float32).reshape(4, 3)
    a = zc.create_array(
        g,
        "cb",
        data=data,
        shape=data.shape,
        dtype=np.float32,
        chunks=(4, 3),
        compressor=None,
    )
    assert a.shape == (4, 3)
    assert np.array_equal(a[:], data)


def test_a_contradictory_shape_is_rejected() -> None:
    """Tolerating `data=` + `shape=` must not extend to tolerating a MISMATCH."""
    g = zc.memory_group()
    with pytest.raises(ValueError, match="contradicts"):
        zc.create_array(
            g, "bad", data=np.zeros((4, 3), np.float32), shape=(9, 9), compressor=None
        )


def test_dtype_wins_over_the_supplied_datas_dtype() -> None:
    """An explicit `dtype=` must still narrow the stored type, as it did in zarr 2.

    zarr 3 derives dtype from `data`, so without the cast a caller passing
    float64 data with `dtype=np.float32` would silently store float64 — doubling
    the array on disk.
    """
    g = zc.memory_group()
    a = zc.create_array(
        g,
        "cast",
        data=np.arange(4, dtype=np.float64),
        dtype=np.float32,
        compressor=None,
    )
    assert a.dtype == np.float32


def test_needs_data_or_shape() -> None:
    """Neither `data` nor `shape` is a caller bug, not a zero-length array."""
    g = zc.memory_group()
    with pytest.raises(ValueError, match="needs either"):
        zc.create_array(g, "nope", compressor=None)


# ---------------------------------------------------------------------------
# Errors, closing, consolidation
# ---------------------------------------------------------------------------


def test_is_missing_error_covers_group_not_found(tmp_path: Path) -> None:
    """zarr 3 kept ``GroupNotFoundError`` but dropped ``PathNotFoundError``.

    ``GroupNotFoundError`` subclasses ``FileNotFoundError``, which is what makes
    the single-exception replacement of the old two-name tuple correct.
    """
    p = tmp_path / "arr.zarr"
    zarr.create_array(store=str(p), shape=(2,), dtype=np.float32, zarr_format=2)
    with pytest.raises(FileNotFoundError) as gi:
        zarr.open_group(str(p), mode="r")  # a node, but an array not a group
    assert zc.is_missing_error(gi.value)

    with pytest.raises(FileNotFoundError) as missing:
        zarr.open_group(str(tmp_path / "absent.zarr"), mode="r")
    assert zc.is_missing_error(missing.value)

    assert not zc.is_missing_error(ValueError("unrelated"))


def test_close_is_a_no_op_for_a_local_store(tmp_path: Path) -> None:
    """zarr 3's Group has no ``close()``; the facade must not explode."""
    zc.close(zc.open_group(tmp_path / "c.zarr", mode="w"))


def test_read_raw_bytes_reads_a_plain_file_inside_a_subgroup(tmp_path: Path) -> None:
    """A non-zarr blob written into a group's own directory (an overlay image) is
    reachable through no array or group API — only through the store, resolved
    against the SUBGROUP's prefix rather than the root's."""
    root = zc.open_group(tmp_path / "s.zarr", mode="w")
    root.create_group("overlays").create_group("logo")
    payload = b"\x89PNG\r\n\x1a\nnot-a-zarr-node\x00"
    (tmp_path / "s.zarr" / "overlays" / "logo" / "image.png").write_bytes(payload)

    assert zc.read_raw_bytes(root["overlays/logo"], "image.png") == payload
    # Absent is None rather than an exception or empty bytes — "the file is gone"
    # and "the file is empty" are different facts to a caller.
    assert zc.read_raw_bytes(root["overlays/logo"], "missing.png") is None


def test_list_raw_keys_answers_case_exactly(tmp_path: Path) -> None:
    """The question :func:`read_raw_bytes` cannot answer: is a key spelled
    EXACTLY this one there? An open-by-name goes through the filesystem, and on
    a case-insensitive one ``Zarr.json`` resolves to the node's own
    ``zarr.json`` — so ``luxar.io.optimise`` cannot tell a dangling payload attr
    from a real file that would clobber that document. A listing compared in
    Python is folded by nothing."""
    root = zc.open_group(tmp_path / "s.zarr", mode="w", zarr_format=3)
    logo = root.create_group("overlays").create_group("logo")
    logo_dir = tmp_path / "s.zarr" / "overlays" / "logo"
    (logo_dir / "image.png").write_bytes(b"payload")
    # A name with real UPPERCASE in it, or a listing that lowercased everything
    # it reported would pass every other assertion here: the two spellings the
    # caller distinguishes would both be `logo.png`, and the fold this function
    # exists to defeat would be back, inside the defence.
    (logo_dir / "Logo.PNG").write_bytes(b"payload")

    keys = zc.list_raw_keys(logo)
    assert "image.png" in keys
    assert "missing.png" not in keys
    assert "Logo.PNG" in keys
    assert "logo.png" not in keys
    # The group's OWN document is a key like any other — which is the whole
    # point, since it is what a case-shifted payload name would collide with.
    assert "zarr.json" in keys
    assert "Zarr.json" not in keys
    assert "Image.PNG" not in keys
    # Immediate children only, as bare names: the subgroup shows up, its
    # contents do not, and nothing is reported as a nested path.
    assert zc.list_raw_keys(root) >= {"overlays", "zarr.json"}
    assert not any("/" in key for key in zc.list_raw_keys(root))
    assert "image.png" not in zc.list_raw_keys(root)


def test_list_raw_keys_is_store_agnostic(tmp_path: Path) -> None:
    """Store-agnostic for the same reason :func:`read_raw_bytes` is: it is the
    verdict a copy pass acts on, and ``--verify`` re-reads a ``.zarr.zip``
    output through a ``ZipStore``. A ``MemoryStore`` has no filesystem to fold
    names, a ``ZipStore`` has no directories at all — both must still list."""
    memory = zc.memory_group()
    logo = memory.create_group("logo")
    zc.write_raw_bytes(logo, "image.png", b"payload")
    assert "image.png" in zc.list_raw_keys(logo)
    assert "Image.png" not in zc.list_raw_keys(logo)

    source = tmp_path / "s.zarr"
    root = zc.open_group(source, mode="w")
    zc.write_raw_bytes(root.create_group("logo"), "image.png", b"payload")
    zc.consolidate(root)
    archive = tmp_path / "s.zarr.zip"
    with zipfile.ZipFile(archive, "w") as out:
        for item in sorted(source.rglob("*")):
            if item.is_file():
                out.write(item, item.relative_to(source).as_posix())

    # Constructed OUTSIDE the call so an `open_group` raise still closes it:
    # an open zip handle keeps the file locked on Windows and `tmp_path`
    # teardown then fails with a second, unrelated error.
    store = zarr.storage.ZipStore(archive, mode="r")
    try:
        zipped = zarr.open_group(store=store, mode="r")
        assert "image.png" in zc.list_raw_keys(zipped["logo"])
        assert "missing.png" not in zc.list_raw_keys(zipped["logo"])
    finally:
        store.close()


def test_write_raw_bytes_round_trips_through_the_store(tmp_path: Path) -> None:
    """The write twin has to resolve against the SUBGROUP's prefix too, and
    materialise real bytes: a payload written one directory up (or not at all)
    leaves the group's ``image_file`` attr naming a file nobody can read, which
    no array or group API would ever notice."""
    path = tmp_path / "w.zarr"
    root = zc.open_group(path, mode="w")
    logo = root.create_group("overlays").create_group("logo")
    payload = b"\x89PNG\r\n\x1a\nwritten-through-the-store\x00"

    zc.write_raw_bytes(logo, "image.png", payload)

    assert (path / "overlays" / "logo" / "image.png").read_bytes() == payload
    assert zc.read_raw_bytes(logo, "image.png") == payload


def test_the_write_backstop_covers_the_hasher_document_set() -> None:
    """The backstop's document names are RESTATED here (importing the hasher's
    set would be a cycle — ``hashing`` imports ``_zarr_compat``), so nothing but
    this pins the two together. A name added to one and not the other silently
    weakens the guard on the one primitive that can replace a node's own
    metadata document, and no other test would go red."""
    from luxar.io._compiler.finalize.hashing import _ZARR_METADATA_DOCS

    assert zc._METADATA_DOC_KEYS == {doc.lower() for doc in _ZARR_METADATA_DOCS}


@pytest.mark.parametrize(
    "key", ["zarr.json", ".zgroup", ".zattrs", ".zarray", ".zmetadata", "Zarr.json"]
)
def test_write_raw_bytes_refuses_a_metadata_document_key(
    tmp_path: Path, key: str
) -> None:
    """Writing a node's own metadata document destroys the store — every reader
    resolves the node through it — so the primitive refuses rather than trusting
    each call site's name check. Case-insensitively: ``Zarr.json`` IS
    ``zarr.json`` on macOS, and this is the write side."""
    path = tmp_path / "guard.zarr"
    root = zc.open_group(path, mode="w")
    logo = root.create_group("overlays").create_group("logo")
    logo_dir = path / "overlays" / "logo"
    before = {p.name: p.read_bytes() for p in logo_dir.iterdir()}

    with pytest.raises(ValueError, match=re.escape(repr(key))):
        zc.write_raw_bytes(logo, key, b"clobbered")

    assert {p.name: p.read_bytes() for p in logo_dir.iterdir()} == before


def test_consolidate_indexes_the_arrays_in_either_format(
    tmp_path: Path, write_format: int
) -> None:
    """Consolidated metadata is load-bearing, not an optimisation.

    The viewer's scene loader builds its whole graph from the store's
    ``contents()`` listing, which comes from this document, and has no
    directory-walking fallback. The two formats put it in different places —
    format 2 in a separate ``.zmetadata``, format 3 in a
    ``consolidated_metadata`` member of the root ``zarr.json`` — and zarrita
    reads both, so what must hold either way is that the array is INDEXED.

    Format 3's consolidated metadata is a zarr-python extension rather than part
    of the v3 spec (zarr warns about exactly that). It stays load-bearing here
    regardless, because the viewer's reader implements it.
    """
    p = tmp_path / "cm.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(g, "a", data=np.arange(4, dtype=np.float32), compressor=None)
    zc.consolidate(g)

    assert zc.is_consolidated(p), "the completion sentinel must see it"
    if write_format == 2:
        meta = json.loads((p / ".zmetadata").read_text())
        assert meta["zarr_consolidated_format"] == 1
        assert "a/.zarray" in meta["metadata"]
        suffixes = {k.rsplit("/", 1)[-1] for k in meta["metadata"]}
        assert suffixes <= {".zgroup", ".zattrs", ".zarray"}, suffixes
    else:
        root = json.loads((p / "zarr.json").read_text())
        indexed = root["consolidated_metadata"]["metadata"]
        assert "a" in indexed, sorted(indexed)
        assert indexed["a"]["node_type"] == "array"
        assert not (p / ".zmetadata").exists(), "wrote a v2 sidecar too"


def test_consolidate_is_silent(tmp_path: Path, write_format: int) -> None:
    """Consolidating must not emit a warning, in either format.

    zarr-python warns that format-3 consolidated metadata is a zarr-python
    extension. Unsuppressed it fires on EVERY save, and — because
    ``ZarrUserWarning`` subclasses ``UserWarning`` — any caller that promotes
    warnings to errors cannot save at all: ``LuxarZarrCompiler.finalize``
    catches it and re-raises ``Could not finalize Zarr store``. Several tests
    legitimately assert warning-freedom around a whole compile
    (``warnings.simplefilter("error", UserWarning)``), so this is not a
    hypothetical.

    Asserted as "no warning AND still consolidated", because silencing it by
    not consolidating would produce a store the viewer loads as an empty scene.

    Matched on the warning CATEGORY, not its wording: the suppression itself is
    keyed on the message text, so a reworded zarr warning would slip past it —
    and a test keyed on the same text would slip past in exactly the same way,
    agreeing with the bug instead of catching it.
    """
    p = tmp_path / "quiet.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(g, "a", data=np.arange(4, dtype=np.float32), compressor=None)
    payload = g.create_group("sound")
    payload.attrs.update({"type": "sound", "audio_file": "audio.mp3"})
    zc.write_raw_bytes(payload, "audio.mp3", b"ID3payload")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        zc.consolidate(g)

    assert zc.is_consolidated(p), "must still write the index"
    offending = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not offending, [f"{w.category.__name__}: {w.message}" for w in offending]


def _consolidated_index_holders(store_dir: Path, fmt: int) -> list[str]:
    """Store-relative dirs carrying a consolidated index ( ``"."`` = the root)."""
    out: list[str] = []
    if fmt == 2:
        for doc in sorted(store_dir.rglob(".zmetadata")):
            out.append(str(doc.parent.relative_to(store_dir)))
    else:
        for doc in sorted(store_dir.rglob("zarr.json")):
            if json.loads(doc.read_text()).get("consolidated_metadata") is not None:
                out.append(str(doc.parent.relative_to(store_dir)))
    return out


def test_editing_in_place_leaves_exactly_one_index(
    tmp_path: Path, write_format: int
) -> None:
    """An in-place edit through the facade must not leave a NESTED index behind.

    Format 3 permits a consolidated index on ANY group, and the facade's
    deliberate ``use_consolidated=False`` bypasses only the ROOT one — a nested
    index is still honoured. Re-opening an already-consolidated store with plain
    ``zarr.open_group`` returns nodes built FROM the root index, so
    re-consolidating serializes that stale tree back out beneath the root; reads
    afterwards return the PRE-EDIT attributes while every document on disk is
    correct, and nothing raises.

    Re-opening through the facade carries no index to re-serialize. Pinned as
    "exactly one index, at the root" (the format-2 invariant the rest of the
    codebase assumes) AND "the edit is what reads back", because the count alone
    would still pass if the single remaining index were stale.
    """
    p = tmp_path / "edit.zarr"
    root = zc.open_group(p, mode="w")
    child = root.create_group("part_0").create_group("child_0")
    child.attrs["coverage_fraction"] = 0.25
    child.attrs["keep"] = "me"
    zc.create_array(
        child, "a", data=np.arange(4, dtype=np.uint8), chunks=(4,), compressor=None
    )
    zc.consolidate(root)
    assert _consolidated_index_holders(p, write_format) == ["."]

    # The edit-in-place cycle Luxar's own tools perform (annotate-quality,
    # doctor --fix, the migrate fixtures).
    reopened = zc.open_group(p, mode="r+")
    del reopened["part_0"]["child_0"].attrs["coverage_fraction"]
    reopened["part_0"]["child_0"].attrs["min_pixel_size"] = 100.0
    zc.consolidate(reopened)

    assert _consolidated_index_holders(p, write_format) == ["."], (
        "a nested consolidated index survived; reads will prefer it over the "
        "per-node documents and serve pre-edit attributes"
    )
    after = dict(zc.open_group(p, mode="r")["part_0"]["child_0"].attrs)
    assert "coverage_fraction" not in after, after
    assert after["min_pixel_size"] == 100.0
    assert after["keep"] == "me", "untouched attrs must survive the round trip"


def test_append_to_an_existing_v3_store_does_not_shadow_it(tmp_path: Path) -> None:
    """``mode="a"`` must NOT pin the format against a store that already exists.

    ``"a"`` is create-or-open. Pinning `zarr_format=2` unconditionally does not
    fail on a v3 store — it writes a SECOND, v2 root beside the v3 one, leaving
    `zarr.json` and `.zgroup` side by side. Everything written afterwards lands in
    the v2 view while an auto-detecting reader resolves the v3 one, so the write
    succeeds and readers cannot see it. Worse than an error, and reachable now
    that Luxar can be pointed at foreign v3 stores at all —
    `denoise_workers.py` opens its output with `mode="a"`.
    """
    p = tmp_path / "foreign_v3.zarr"
    zarr.create_group(store=str(p), zarr_format=3).attrs["origin"] = "other-tool"

    g = zc.open_group(p, mode="a")
    zc.create_array(g, "added", data=np.arange(3, dtype=np.float32), compressor=None)

    assert not (p / ".zgroup").exists(), "wrote a v2 shadow root beside the v3 one"
    reader = zarr.open_group(str(p), mode="r")  # auto-detect, as a consumer would
    assert reader.metadata.zarr_format == 3
    assert "added" in reader, "the append is invisible to an auto-detecting reader"
    assert reader.attrs["origin"] == "other-tool", "clobbered the foreign attrs"


def test_document_readers_resolve_a_mixed_store_the_way_zarr_does(
    tmp_path: Path, v3_writes: int
) -> None:
    """With BOTH formats' documents present, the readers must answer format 3.

    zarr resolves such a node as format 3 (it warns, then uses `zarr.json`), so a
    document-level reader that consulted `.zattrs`/`.zarray` first would report a
    pre-migration view of a store that every OPENER — the viewer included — reads
    as v3. Silent, and the shape it hands back is what `batch-fit validate`
    checks tiles against.

    Luxar's writers do not produce this state (see the shadow-root test above);
    an interrupted in-place migration or a half-finished copy does.
    """
    p = tmp_path / "mixed.zarr"
    g = zc.open_group(p, mode="w")
    g.attrs["kind"] = "v3-truth"
    zc.create_array(g, "a", data=np.arange(6, dtype=np.uint8), compressor=None)
    zc.consolidate(g)

    # Drop a stale v2 root beside it, as an interrupted migration would leave.
    (p / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (p / ".zattrs").write_text(json.dumps({"kind": "v2-STALE"}))
    (p / ".zmetadata").write_text(
        json.dumps({"metadata": {".zattrs": {"kind": "v2-STALE"}}})
    )
    (p / "a" / ".zarray").write_text(json.dumps({"shape": [999], "dtype": "|u1"}))

    attrs = zc.read_node_attrs(p)
    assert attrs is not None and attrs["kind"] == "v3-truth", (
        f"read_node_attrs answered the stale v2 view: {attrs!r}"
    )
    meta = zc.read_array_meta(p / "a")
    assert meta is not None and meta["shape"] == [6], (
        f"read_array_meta answered the stale v2 shape: {meta and meta.get('shape')!r}"
    )
    # The consolidated reader answers the same store as the per-node one. It has
    # its own document (`.zmetadata` vs the index inside `zarr.json`), so a
    # v2-first order here would have the two facade readers disagreeing about one
    # store while every opener sees only the v3 view.
    consolidated = zc.read_consolidated_attrs(p)
    assert consolidated["/"]["kind"] == "v3-truth", (
        f"read_consolidated_attrs answered the stale v2 index: {consolidated!r}"
    )
    # ...and each agrees with what an auto-detecting opener sees.
    opened = zarr.open_group(str(p), mode="r")
    assert opened.metadata.zarr_format == 3
    assert opened.attrs["kind"] == attrs["kind"]


def test_a_present_v3_document_is_never_second_guessed_by_a_v2_one(
    tmp_path: Path, v3_writes: int
) -> None:
    """A `zarr.json` that is unusable answers for the node ANYWAY.

    Ordering alone is not enough: a v3 document that says GROUP, or one too
    corrupt to parse, would otherwise fall through to a stale `.zarray`/`.zattrs`
    and hand back a plausible pre-migration view. Both of those ARE the
    corruption `batch-fit validate` is looking for — it reads `read_array_meta`
    for exactly that verdict — so falling back would let a broken tile pass.
    """
    root = tmp_path / "shapes.zarr"
    g = zc.open_group(root, mode="w")
    g.create_group("child")
    (root / "child" / ".zarray").write_text(json.dumps({"shape": [777]}))
    assert zc.read_array_meta(root / "child") is None, (
        "a v3 GROUP document fell through to a stale .zarray"
    )

    corrupt = tmp_path / "corrupt.zarr"
    corrupt.mkdir()
    (corrupt / "zarr.json").write_text('{"zarr_format": 3, "node_type": "arr')
    (corrupt / ".zarray").write_text(json.dumps({"shape": [555]}))
    (corrupt / ".zattrs").write_text(json.dumps({"kind": "v2-STALE"}))
    assert zc.read_array_meta(corrupt) is None, (
        "an unparseable v3 document fell through to a stale .zarray"
    )
    assert zc.read_node_attrs(corrupt) is None, (
        "an unparseable v3 document fell through to stale .zattrs"
    )

    # A single-format v2 node is untouched by any of this.
    v2 = tmp_path / "v2array"
    v2.mkdir()
    (v2 / ".zarray").write_text(json.dumps({"shape": [3]}))
    (v2 / ".zattrs").write_text(json.dumps({"kind": "leaf"}))
    assert (zc.read_array_meta(v2) or {})["shape"] == [3]
    assert zc.read_node_attrs(v2) == {"kind": "leaf"}


def test_a_document_name_demotes_but_never_promotes(tmp_path: Path) -> None:
    """`doc_name` vetoes an unwrap; it never forces one.

    The content signal alone (`zarr_format: 3`) mis-reads a format-2 document
    whose USER attributes carry that key — it would answer `{}` and drop every
    authored value. Naming the document it came from settles that, and
    `read_node_attrs` always can.

    The asymmetry is the point: promoting on the name would answer `{}` for any
    non-v3 body served from a `zarr.json` address, which no real writer produces
    but fakes and misconfigured proxies do — trading a reachable failure for a
    silent one.
    """
    # Demote: a v2 attrs document that merely LOOKS v3 comes back verbatim.
    confusing = {"zarr_format": 3, "kind": "leaf", "mine": 1}
    assert zc.attrs_from_node_doc(confusing) == {}, "the content sniff still guesses"
    assert zc.attrs_from_node_doc(confusing, doc_name=".zattrs") == confusing

    # Never promote: a flat body named `zarr.json` is still returned verbatim.
    flat = {"kind": "leaf"}
    assert zc.attrs_from_node_doc(flat, doc_name="zarr.json") == flat

    # A real v3 record still unwraps, named or not.
    v3 = {"zarr_format": 3, "node_type": "group", "attributes": {"k": "v"}}
    assert zc.attrs_from_node_doc(v3) == {"k": "v"}
    assert zc.attrs_from_node_doc(v3, doc_name="zarr.json") == {"k": "v"}

    # End to end: `read_node_attrs` names the document, so a v2 store whose
    # attrs carry `zarr_format` keeps them.
    d = tmp_path / "confusing.zarr"
    d.mkdir()
    (d / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (d / ".zattrs").write_text(json.dumps(confusing))
    assert zc.read_node_attrs(d) == confusing


def test_is_consolidated_ignores_a_stale_v2_index_beside_a_v3_root(
    tmp_path: Path, v3_writes: int
) -> None:
    """An unfinished v3 save must not read as finished because of a leftover
    `.zmetadata`.

    This is the completion sentinel `batch-fit` uses to tell a written tile from
    an interrupted one, so answering "yes" for a store whose v3 root carries no
    consolidated index would let a half-written tile into a merge.
    """
    p = tmp_path / "unfinished.zarr"
    g = zc.open_group(p, mode="w")  # created, never consolidated
    zc.create_array(g, "a", data=np.arange(3, dtype=np.uint8), compressor=None)
    assert not zc.is_consolidated(p), "a fresh unconsolidated v3 store is not finished"

    (p / ".zmetadata").write_text(json.dumps({"metadata": {}}))
    assert not zc.is_consolidated(p), (
        "a stale v2 index made an unfinished v3 store report as complete"
    )


def test_append_still_pins_the_format_when_there_is_nothing_there(
    tmp_path: Path, write_format: int
) -> None:
    """The other half of the `mode="a"` rule: creating still pins the format.

    Guards against "fixing" the shadowing bug by dropping `zarr_format` from
    `"a"` altogether, which would silently hand every create-or-open writer
    zarr's ambient default instead of Luxar's. Includes the
    pre-made-empty-directory case, since a caller making its output dir first
    must not change the format.
    """
    doc = ROOT_DOC[write_format]

    fresh = tmp_path / "fresh.zarr"
    zc.open_group(fresh, mode="a")
    assert json.loads((fresh / doc).read_text())["zarr_format"] == write_format

    premade = tmp_path / "premade.zarr"
    premade.mkdir()
    zc.open_group(premade, mode="a")
    assert json.loads((premade / doc).read_text())["zarr_format"] == write_format


@pytest.mark.parametrize("mode", ["w", "a"])
def test_creating_a_fresh_zipped_store_still_pins_the_format(
    tmp_path: Path, mode: str, write_format: int
) -> None:
    """A brand-new ``.zarr.zip`` must come out in the selected format.

    This guards an ORDERING dependency inside :func:`open_group`: it constructs the
    store before consulting :func:`_metadata_docs_exist`, and for a zip that helper
    treats "the file exists" as "the store exists". It works today only because
    zarr's ``ZipStore`` is LAZY — constructing one does not create the archive — so
    the helper still sees nothing and the format is pinned. If a future zarr made
    ZipStore eager, the pin would be skipped and a fresh archive would silently
    come out format 3. That is not a hypothesis worth leaving unguarded.
    """
    p = tmp_path / f"fresh_{mode}.zarr.zip"
    g = zc.open_group(p, mode=mode)
    zc.create_array(g, "a", data=np.arange(4, dtype=np.float32), compressor=None)
    zc.close(g)  # ZipStore must be closed to flush the archive

    with zipfile.ZipFile(p) as zf:
        names = zf.namelist()
    doc = ROOT_DOC[write_format]
    assert any(n.endswith(doc) for n in names), (
        f"fresh zipped store is not format {write_format}; archive holds {names[:5]}"
    )
    other = ROOT_DOC[2 if write_format == 3 else 3]
    assert not any(n.endswith(other) for n in names)


def test_open_store_honours_mode(tmp_path: Path) -> None:
    """A declared read must yield a READ-ONLY store, not merely a readable one.

    Handing back a writable store for `mode="r"` makes the argument a decoration:
    a read path that acquired a write by accident would not be caught anywhere.
    """
    assert zc.open_store(tmp_path / "ro.zarr", mode="r").read_only is True
    assert zc.open_store(tmp_path / "rw.zarr", mode="w").read_only is False
    with pytest.raises(Exception):  # noqa: B017 - zarr's own refusal, type is its business
        zarr.create_group(store=zc.open_store(tmp_path / "refuse.zarr", mode="r"))


def test_reads_do_not_trust_stale_consolidated_metadata(
    tmp_path: Path, write_format: int
) -> None:
    """A deleted array must read as ABSENT, not as present-per-the-stale-index.

    zarr 2 only consulted ``.zmetadata`` through the separate
    ``open_consolidated``, so ``open_group`` always saw what was on disk. zarr 3
    reversed that default. The difference silently disables Luxar's detection of a
    partially written store: the writers have crash-safety machinery precisely
    because half-written stores happen, and the readers' "required array is
    missing" guards are the backstop — they have to see the filesystem.
    """
    import shutil

    p = tmp_path / "partial.zarr"
    g = zc.open_group(p, mode="w")
    zc.create_array(g, "keep", data=np.arange(4, dtype=np.float32), compressor=None)
    zc.create_array(g, "doomed", data=np.arange(4, dtype=np.float32), compressor=None)
    zc.consolidate(g)

    shutil.rmtree(p / "doomed")  # simulate a partial/interrupted write
    assert zc.is_consolidated(p), "the stale index must still be present"

    reopened = zc.open_group(p, mode="r")
    assert "doomed" not in reopened, "read answered from the stale consolidated index"
    assert sorted(reopened.array_keys()) == ["keep"]

    # And prove the hazard is real rather than hypothetical: zarr's own default
    # still reports the deleted array. If a future zarr changes that, this line
    # fails and the workaround above can go.
    trusting = zarr.open_group(str(p), mode="r")
    assert "doomed" in trusting


def test_open_store_dispatches_zip_by_suffix(tmp_path: Path) -> None:
    """zarr 2 sniffed ``.zip`` inside ``zarr.open``; zarr 3 needs it explicit."""
    assert isinstance(zc.open_store(tmp_path / "d.zarr"), zarr.storage.LocalStore)
    zs = zc.open_store(tmp_path / "z.zarr.zip", mode="w")
    assert isinstance(zs, zarr.storage.ZipStore)
    zs.close()


def _write_zipped_store(path: Path) -> None:
    g = zc.open_group(path, mode="w")
    g.attrs["marker"] = "original"
    zc.create_array(g, "a", data=np.arange(4, dtype=np.float32), compressor=None)
    zc.close(g)


def test_zipped_store_accepts_read_write_reopen(tmp_path: Path) -> None:
    """``mode="r+"`` must work on an archive, not die inside zipfile.

    ``ZipStore`` speaks only zipfile's r/w/a, while ``open_group`` speaks zarr's —
    which also has ``"r+"``. Passed through unchanged it raised ``ValueError:
    ZipFile requires mode 'r', 'w', 'x', or 'a'`` from a layer that cannot say
    what the caller did wrong.
    """
    p = tmp_path / "rw.zarr.zip"
    _write_zipped_store(p)

    g = zc.open_group(p, mode="r+")
    g.attrs["added"] = 1
    zc.close(g)

    reopened = zc.open_group(p, mode="r")
    assert reopened.attrs["added"] == 1
    assert reopened.attrs["marker"] == "original"
    assert list(reopened.array_keys()) == ["a"]
    zc.close(reopened)


def test_zipped_store_exclusive_create_refuses_without_truncating(
    tmp_path: Path,
) -> None:
    """``mode="w-"`` must refuse an existing archive and leave it INTACT.

    It cannot lean on zarr's own refusal: a ZipStore opened for ``"w"`` truncates
    the archive the first time it is touched, which is before ``open_group`` looks
    for the root it would have declined to overwrite. Exclusive creation would
    destroy exactly the file it exists to protect.
    """
    p = tmp_path / "keep.zarr.zip"
    _write_zipped_store(p)

    with pytest.raises(FileExistsError):
        zc.open_group(p, mode="w-")

    survivor = zc.open_group(p, mode="r")
    assert survivor.attrs["marker"] == "original"
    assert list(survivor.array_keys()) == ["a"]
    zc.close(survivor)

    fresh = tmp_path / "fresh.zarr.zip"
    created = zc.open_group(fresh, mode="w-")
    assert created.metadata.zarr_format == zc.ZARR_FORMAT
    zc.close(created)


def test_open_store_rejects_an_unknown_zip_mode(tmp_path: Path) -> None:
    """An unsupported mode names itself rather than surfacing as a zipfile error."""
    with pytest.raises(ValueError, match="not supported for a zipped store"):
        zc.open_store(tmp_path / "x.zarr.zip", mode="x")


# ---------------------------------------------------------------------------
# The production-explicitness lint
# ---------------------------------------------------------------------------


def _production_files() -> list[Path]:
    return [
        p
        for p in sorted(PROD_ROOT.rglob("*.py"))
        if "tests" not in p.parts and p.name != "_zarr_compat.py"
    ]


def test_production_create_array_calls_pass_a_compressor() -> None:
    """Every PRODUCTION ``create_array`` must name its compressor.

    ``compressor`` defaults to ``"auto"`` so that test fixtures which never named
    one keep zarr 2's bytes. That default must never be what a real writer gets:
    Luxar's stores carry a measured policy (zstd clevel 9, shuffle by dtype
    width) and some arrays are deliberately RAW, and "auto" is neither. Enforced
    by AST because it is exactly the kind of omission that reads fine in review
    and only shows up as a silent change in the bytes on disk.
    """
    offenders: list[str] = []
    for path in _production_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name != "create_array":
                continue
            if "compressor" not in {k.arg for k in node.keywords}:
                offenders.append(f"{path.relative_to(PROD_ROOT)}:{node.lineno}")

    assert not offenders, (
        "production create_array() calls with no explicit compressor "
        f"(they would silently get Blosc/lz4/5): {offenders}"
    )


def test_no_zarr_2_create_dataset_calls_remain() -> None:
    """``Group.create_dataset`` was REMOVED in zarr 3 — no call site may survive.

    Scanned as text over the whole repo rather than just this package, because the
    call sites that actually bit were outside it (the viewer's fixture generators)
    and because a branch merged from before this migration reintroduces them
    invisibly: that is exactly how `io/tests/test_volume_lazy.py` arrived mid-PR.
    ``AttributeError`` at runtime is a poor substitute for failing here.
    """
    repo = PROD_ROOT.parents[3]
    offenders: list[str] = []
    for path in sorted(repo.rglob("*.py")):
        parts = set(path.parts)
        if (
            "node_modules" in parts
            or ".venv" in parts
            or "_zarr_compat.py" == path.name
        ):
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            # `create_resizable_dataset` is Luxar's OWN API and stays.
            if ".create_dataset(" in line and "create_resizable_dataset" not in line:
                offenders.append(f"{path.relative_to(repo)}:{lineno}")

    assert not offenders, (
        "zarr 3 removed Group.create_dataset; use "
        f"luxar._zarr_compat.create_array instead: {offenders}"
    )


#: zarr entry points that CREATE a node, and therefore decide a format. Reading
#: is version-agnostic and needs no pin; creating does. ``consolidate_metadata``
#: is absent on purpose — it follows the format already on the store.
_ZARR_CREATING_CALLS = frozenset(
    {
        "array",
        "create",
        "create_array",
        "create_group",
        "empty",
        "full",
        "group",
        "ones",
        "open",
        "open_array",
        "open_group",
        "save",
        "save_array",
        "zeros",
    }
)


def _is_read_only_call(call: ast.Call) -> bool:
    """Does the call pass a literal ``mode="r"``? Then it creates nothing."""
    return any(
        k.arg == "mode" and isinstance(k.value, ast.Constant) and k.value.value == "r"
        for k in call.keywords
    )


def _zarr_import_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """``(module names bound to zarr, creating names imported FROM zarr)``.

    Matching only the literal identifier ``zarr`` would let two entirely ordinary
    spellings through — ``import zarr as z`` and ``from zarr import open_group`` —
    so the aliases are resolved instead of assumed. Names imported from
    ``luxar._zarr_compat`` deliberately do NOT count: that module IS the fix.
    """
    modules, direct = {"zarr"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "zarr" or alias.name.startswith("zarr."):
                    modules.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "zarr" or mod.startswith("zarr."):
                for alias in node.names:
                    if alias.name in _ZARR_CREATING_CALLS:
                        direct.add(alias.asname or alias.name)
    return modules, direct


def _root_name(node: ast.expr) -> str | None:
    """Leftmost identifier of an attribute chain — ``zarr`` in ``zarr.api.open``."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _getattr_creating_call(func: ast.expr, modules: set[str]) -> bool:
    """Is ``func`` a ``getattr(zarr, "open")``-style lookup of a creating call?

    Reflection is a thin disguise, and closing it costs six lines. Only the
    literal-string form is resolvable — a name computed at runtime is beyond any
    static check, and a lint that pretended otherwise would be worse than one
    with a stated limit.
    """
    if not (isinstance(func, ast.Call) and isinstance(func.func, ast.Name)):
        return False
    if func.func.id != "getattr" or len(func.args) < 2:
        return False
    target, attr = func.args[0], func.args[1]
    return (
        _root_name(target) in modules
        and isinstance(attr, ast.Constant)
        and attr.value in _ZARR_CREATING_CALLS
    )


def _is_zarr_creating_callee(
    func: ast.expr, modules: set[str], direct: set[str]
) -> bool:
    """Does ``func`` name a zarr call that CREATES a node, in any of its spellings?

    Three shapes reach a creating call: an attribute on a module bound to zarr
    (including a chain like ``zarr.api.synchronous.open``), a bare name imported
    from zarr, and a literal ``getattr`` lookup.
    """
    if isinstance(func, ast.Attribute):
        return _root_name(func) in modules and func.attr in _ZARR_CREATING_CALLS
    if isinstance(func, ast.Name):
        return func.id in direct
    return _getattr_creating_call(func, modules)


def _format_unpinned_zarr_writes(source: str) -> list[int]:
    """Line numbers of zarr node-creating calls that pin no format."""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - defensive
        return []
    modules, direct = _zarr_import_names(tree)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_zarr_creating_callee(node.func, modules, direct):
            continue
        if _is_read_only_call(node):
            continue
        if "zarr_format" not in {k.arg for k in node.keywords}:
            hits.append(node.lineno)
    return hits


def test_no_writer_creates_a_store_without_pinning_the_format() -> None:
    """A writer that skips the facade silently emits format 3.

    zarr 3's default is format 3, and the default applies at the node that gets
    CREATED — so a bare ``zarr.open(mode="a")`` root makes every array under it
    v3 no matter what :func:`create_array` is told, and a bare ``zarr.save``
    writes a ``zarr.json`` + ``c/`` array where the rest of Luxar writes
    ``.zarray`` + dot-separated chunks. Neither raises; the store is simply the
    wrong format, which is why this is a lint and not a runtime check.

    Scope is every writer that ships or produces a consumed artifact: the
    package, the repo scripts, the viewer's fixture generators, AND the example
    scripts — those write the `datasets/examples/` scenes that `make run-examples`
    produces and the E2E suite loads, so a v3 store there would surface as a
    baffling viewer failure rather than as a format complaint. Go through
    :mod:`luxar._zarr_compat`, or pass ``zarr_format=ZARR_FORMAT`` explicitly.
    """
    repo_root = PROD_ROOT.parents[3]
    extra_dirs = [
        repo_root / "scripts",
        repo_root / "stats",
        repo_root / "packages" / "luxar" / "examples",
        repo_root / "packages" / "luxar-viewer" / "tests" / "fixtures",
    ]
    files = list(_production_files())
    for d in extra_dirs:
        if d.is_dir():  # absent when the package is tested outside the repo
            files.extend(sorted(d.rglob("*.py")))

    offenders = [
        f"{path.relative_to(repo_root)}:{lineno}"
        for path in files
        for lineno in _format_unpinned_zarr_writes(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "zarr calls that create a node without pinning the format (they would "
        f"silently write format 3): {offenders}"
    )


def test_the_format_lint_can_actually_fail() -> None:
    """Guard the guard: the walk must flag the two real shapes and spare the rest.

    Lines 1-2 are the exact regressions this lint exists for (a bare ``save``,
    and the creating ``mode="a"`` open whose children inherit the format). The
    rest must stay quiet, or the lint would be unusable noise on read paths.
    """
    assert _format_unpinned_zarr_writes(
        "zarr.save(p, a)\n"
        "zarr.open(p, mode='a')\n"
        "zarr.open(p, mode='r')\n"
        "zarr.save(p, a, zarr_format=2)\n"
        "zarr.consolidate_metadata(s)\n"
        "zarr.storage.ZipStore(p, mode='w')\n"
    ) == [1, 2]


def test_the_format_lint_resists_the_obvious_evasions() -> None:
    """Aliasing or from-importing zarr must not slip a creating call past the lint.

    ``import zarr as z`` and ``from zarr import open_group`` are not adversarial
    tricks, they are two ordinary spellings — and matching the literal identifier
    ``zarr`` misses both. Nothing in the repo writes them today, which is exactly
    why this is pinned now rather than after one appears.
    """
    # Aliased module, and a nested module path.
    assert _format_unpinned_zarr_writes("import zarr as z\nz.open(p, mode='w')\n") == [
        2
    ]
    assert _format_unpinned_zarr_writes(
        "import zarr\nzarr.api.synchronous.open(p, mode='w')\n"
    ) == [2]
    # From-imported creating name, bare and aliased.
    assert _format_unpinned_zarr_writes(
        "from zarr import open_group\nopen_group(p, mode='w')\n"
    ) == [2]
    assert _format_unpinned_zarr_writes(
        "from zarr import create_group as cg\ncg(store=s)\n"
    ) == [2]
    # ...and the same spellings stay quiet when they DO pin, or only read.
    assert (
        _format_unpinned_zarr_writes(
            "import zarr as z\nz.open(p, mode='w', zarr_format=2)\nz.open(p, mode='r')\n"
        )
        == []
    )
    # The facade's own names must never be flagged — that module is the fix, and
    # `luxar._zarr_compat` must not be mistaken for the `zarr` package.
    assert (
        _format_unpinned_zarr_writes(
            "from luxar._zarr_compat import open_group, create_root_group\n"
            "open_group(p, mode='w')\n"
            "create_root_group(store)\n"
        )
        == []
    )
    # Reflection is a thin disguise, so the literal-string form is resolved.
    assert _format_unpinned_zarr_writes(
        "import zarr\ngetattr(zarr, 'open')(p, mode='w')\n"
    ) == [2]
    assert _format_unpinned_zarr_writes(
        "import zarr as z\ngetattr(z, 'create_group')(store=s)\n"
    ) == [2]
    # ...but it must not fire on an unrelated object, on a non-creating call, or
    # when the reflective call DOES pin the format.
    assert (
        _format_unpinned_zarr_writes(
            "import zarr\n"
            "getattr(zarr, 'open')(p, mode='w', zarr_format=2)\n"
            "getattr(obj, 'open')(p, mode='w')\n"
            "getattr(zarr, 'consolidate_metadata')(s)\n"
        )
        == []
    )
    # A name computed at runtime is beyond any static check. Pinned so the limit
    # is a stated one rather than a surprise.
    assert (
        _format_unpinned_zarr_writes(
            "import zarr\nname = 'open'\ngetattr(zarr, name)(p, mode='w')\n"
        )
        == []
    )


def test_the_lint_can_actually_fail(tmp_path: Path) -> None:
    """Guard the guard: prove the AST walk detects a missing compressor.

    Without this, a refactor that broke the visitor (e.g. stopped matching
    ``ast.Name`` callees) would leave the lint above passing vacuously forever.
    """
    src = "create_array(g, 'x', data=d)\ncreate_array(g, 'y', compressor=None)\n"
    tree = ast.parse(src)
    missing = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "create_array"
        and "compressor" not in {k.arg for k in n.keywords}
    ]
    assert missing == [1]
