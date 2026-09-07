"""VTK XML PolyData (``.vtp``) reader — inline, appended, raw, base64, zlib or not.

The format ParaView, VTK, PyVista and every ITK/VTK pipeline writes for a
surface. It is XML with the *bulk* data escaping the XML, which is where all three of
its traps live. NumPy + stdlib only (``xml.etree`` + ``base64`` + ``zlib``), matching
the rest of :mod:`luxar.mesh.interop`.

**Trap 1 — ``<AppendedData encoding="raw">`` makes the document invalid XML.** The bytes
after the ``_`` marker are arbitrary binary: they contain ``<``, ``&`` and sequences that
are not UTF-8, so ``ElementTree`` refuses the *whole* file, header included. The stream is
therefore split at the ``<AppendedData`` start tag first; only the leading portion is fed to
a pull parser, which hands back a usable tree even though the closing tags never arrive,
and each appended ``DataArray`` is indexed into the tail by its own ``offset=``.

**Trap 2 — a compressed base64 block is TWO concatenated base64 streams.** VTK encodes
the block header (``nblocks``, block size, last partial block size, then one compressed
size per block — all ``header_type``-wide) *separately* from the compressed payload and
writes the two encodings back to back. A single :func:`base64.b64decode` of the element
text therefore yields garbage **without raising**, which is the single most common way a
hand-written VTP reader misparses. The header's own length is not known up front either
— it depends on ``nblocks`` — so it is read in three steps: decode enough characters for
the first word, decode the full header, then decode the payload as a *second* stream
starting at the character the first one ended on. Uncompressed data has no such split: a
single ``header_type``-wide byte count precedes the payload inside one stream.

**Trap 3 — ``offsets`` are cumulative END offsets.** There is no leading ``0`` in the VTK
XML form (unlike the legacy ``.vtk`` and unlike ``vtkCellArray``'s in-memory
layout), so cell *i* spans ``connectivity[offsets[i - 1] : offsets[i]]`` with an implied
0 for *i* = 0. Reading them as start offsets shifts every polygon by one cell and still
produces a plausible-looking surface.

Per-array widths come from each ``DataArray``'s own ``type=`` attribute, never assumed:
``connectivity`` is ``Int64`` from modern VTK and ``Int32`` from older writers, and
``offsets`` need not match it.

Colour rule (documented because the format does not settle it): only 3- or 4-component
arrays are candidates at all, and one is taken as per-vertex colour when
``<PointData Scalars="...">`` names it, when its ``Name`` is one of the conventional
colour spellings, or when it is ``UInt8`` — VTK's own unsigned-char colour convention. A
``Scalars=`` naming a 1-component field (a segmentation label, a curvature scalar — the
commonest thing it points at in real VTK output) is therefore ignored, not painted on.
A *nameless* float 3-vector is left alone too, because in a VTK surface that is far more
often a displacement or velocity field than a colour. Values
are converted by observed range, the same rule :mod:`._ply_mesh` applies: peak ``<= 1``
means the 0..1 convention and is scaled by 255, anything else is clipped into 0..255.
"""

from __future__ import annotations

import base64
import binascii
import re

# B405/B314 are waived here. `defusedxml` would be a new dependency this package
# deliberately does not have, and the headline exposure it guards — entity-expansion
# DoS (billion laughs / quadratic blowup) — is already contained by libexpat's own
# input-amplification cap, present since expat 2.4 and therefore in every interpreter
# this repo supports (Python 3.12+). A bomb fed to `import_mesh` dies in a fraction of
# a second with "limit on input amplification factor (from DTD and entities)
# breached", which `_parse_header_document` reports as a malformed header.
# ElementTree DOES expand internal entities, so the cap is what does the containing,
# not any abstinence on ElementTree's part.
import xml.etree.ElementTree as ET  # nosec B405
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray

#: VTK ``type=`` names → numpy dtype strings.
_VTK_DTYPES: dict[str, str] = {
    "Int8": "i1",
    "UInt8": "u1",
    "Int16": "i2",
    "UInt16": "u2",
    "Int32": "i4",
    "UInt32": "u4",
    "Int64": "i8",
    "UInt64": "u8",
    "Float32": "f4",
    "Float64": "f8",
}

#: The only two widths ``header_type=`` is allowed to take.
_HEADER_DTYPES: dict[str, str] = {"UInt32": "u4", "UInt64": "u8"}

_BYTE_ORDERS: dict[str, Literal["<", ">"]] = {
    "LittleEndian": "<",
    "BigEndian": ">",
}

#: The compressor this reader can undo. Anything else is refused BY NAME.
_ZLIB_COMPRESSOR = "vtkZLibDataCompressor"

#: ``Name=`` spellings that mean "this is colour", case-insensitively.
_COLOR_NAMES = frozenset(
    {"colors", "colours", "color", "colour", "rgb", "rgba", "vertexcolors"}
)

#: The optional `ns:` allows a PREFIXED root — `<vtk:VTKFile xmlns:vtk="…">` is legal and
#: `read_vtp` reads it (every tag is matched by local name), so a sniffer that missed it
#: would refuse a file the reader can decode.
_VTKFILE_TAG = re.compile(rb"<(?:[\w.-]+:)?VTKFile\b[^>]*>", re.S)
#: Both anchored with a LEFT word boundary. XML attribute order is not semantic, and
#: canonicalization (C14N, lxml) sorts alphabetically — which puts `header_type=` ahead
#: of `type=`. Unanchored, the first match in
#: `<VTKFile byte_order="…" header_type="UInt32" type="PolyData" …>` is the header
#: width, and a perfectly valid file is refused as "type is 'UInt32'". `_` and `t` are
#: both word characters, so `\b` cannot match inside `header_type`.
#:
#: Both quote characters, because `AttValue ::= '"' … '"' | "'" … "'"` — a single-quoted
#: `type='PolyData'` is as valid as the double-quoted spelling, and matching only the
#: latter makes the sniffer disagree with the parser about the very same file. The value
#: is group 2; group 1 is the quote the backreference matches.
_TYPE_ATTR = re.compile(rb'\btype\s*=\s*(["\'])(.*?)\1', re.S)
#: The same optional `ns:` as `_VTKFILE_TAG`, for the same reason: a document whose tags
#: are ALL prefixed (`<vtk:VTKFile>` … `<vtk:AppendedData>`) sniffs as PolyData and every
#: tag is matched by local name, so a literal `<AppendedData` here would never cut the
#: stream and the reader would report "no <AppendedData> section" about a file that has
#: one. A default `xmlns=` needs nothing — the tag spelling stays unprefixed. The `\b`
#: keeps a hypothetical `<AppendedDataFoo>` from matching.
_APPENDED_TAG = re.compile(rb"<(?:[\w.-]+:)?AppendedData\b")
_ENCODING_ATTR = re.compile(r'\bencoding\s*=\s*(["\'])(.*?)\1', re.S)


def sniff_vtk_type(head: bytes) -> str:
    """The ``type=`` of the leading ``<VTKFile>`` tag, or ``""`` when there is none.

    Regex rather than a parse: with ``encoding="raw"`` appended data the document is not
    well-formed XML at all, and the sniffer only ever sees the first block of the file.
    """
    tag = _VTKFILE_TAG.search(head)
    if tag is None:
        return ""
    attr = _TYPE_ATTR.search(tag.group(0))
    return attr.group(2).decode("ascii", errors="replace") if attr else ""


@dataclass(frozen=True)
class _Context:
    """Everything a ``DataArray`` needs in order to decode itself."""

    #: ``path.name``, so every error names the file.
    name: str
    #: ``"<"`` or ``">"``, from ``byte_order=``.
    order: Literal["<", ">"]
    #: ``header_type=``, already byte-ordered.
    header: np.dtype
    #: True when ``compressor="vtkZLibDataCompressor"`` was declared.
    compressed: bool
    #: Raw bytes after the ``_`` marker; empty when there is no appended section.
    appended: bytes
    #: The same bytes as ASCII, for ``encoding="base64"``; empty otherwise.
    appended_text: str


def _localname(tag: str) -> str:
    """``{ns}Piece`` → ``Piece``. VTK XML is namespace-free in practice, not by rule."""
    return tag.rsplit("}", 1)[-1]


def _child(parent: ET.Element, name: str) -> Optional[ET.Element]:
    for element in parent:
        if _localname(element.tag) == name:
            return element
    return None


def _children(parent: ET.Element, name: str) -> list[ET.Element]:
    return [e for e in parent if _localname(e.tag) == name]


def _named_array(group: ET.Element, name: str) -> Optional[ET.Element]:
    for element in group:
        if _localname(element.tag) == "DataArray" and element.get("Name") == name:
            return element
    return None


def _int_attr(
    element: ET.Element, name: str, ctx: _Context, where: str, default: int
) -> int:
    """``int(element.get(name))``, but a named error instead of a bare ``ValueError``.

    Every attribute here is data from the file, so ``NumberOfComponents="x"`` or an
    ``offset=`` a writer left empty must arrive as "which file, which array, which
    attribute" rather than as ``invalid literal for int()`` from somewhere in the stack.
    """
    text = element.get(name)
    if text is None:
        return default
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            f"{ctx.name}: {where} has {name}={text!r}, which is not an integer"
        ) from exc


def _split_appended(raw: bytes, filename: str) -> tuple[bytes, bytes, str]:
    """Cut the byte stream at ``<AppendedData``. Returns ``(header, payload, encoding)``.

    With no appended section the payload and the encoding both come back empty.

    This has to happen before anything else: with ``encoding="raw"`` the payload is
    arbitrary binary sitting inside the document, so feeding the whole file to an XML
    parser fails on the *data*, not on the markup, and the header is lost with it.

    The payload keeps its trailing ``</AppendedData></VTKFile>``. Harmless — every array
    is located by an absolute offset and bounded by its own length header — and trimming
    it would mean scanning binary data for a closing tag that may legitimately occur
    inside it.

    ``encoding=`` is REQUIRED rather than defaulted. Raw bytes and base64 text are not
    reliably distinguishable from the payload itself, and guessing wrong does not fail
    honestly: base64 read as raw yields a byte count in the billions and an error blaming
    the data, on a file that is perfectly valid. Every VTK writer emits the attribute, so
    refusing by name costs nothing real and never mis-diagnoses.
    """
    found = _APPENDED_TAG.search(raw)
    if found is None:
        return raw, b"", ""
    marker = found.start()
    tag_end = raw.find(b">", marker)
    if tag_end < 0:
        raise ValueError(f"{filename}: <AppendedData> tag is never closed")
    tag = raw[marker : tag_end + 1].decode("ascii", errors="replace")
    match = _ENCODING_ATTR.search(tag)
    if match is None:
        raise ValueError(
            f"{filename}: <AppendedData> carries no encoding= attribute, so whether its "
            "payload is raw bytes or base64 text is undecidable — the two cannot be told "
            "apart from the payload, and guessing wrong reports a corrupt file rather "
            "than a valid one. Every VTK writer emits it."
        )
    encoding = match.group(2).lower()
    if encoding not in ("raw", "base64"):
        raise ValueError(
            f"{filename}: <AppendedData encoding={encoding!r}> — expected 'raw' or "
            "'base64'"
        )
    start = raw.find(b"_", tag_end)
    if start < 0:
        raise ValueError(
            f"{filename}: <AppendedData> has no '_' marker, so where its payload begins "
            "is undefined"
        )
    return raw[:marker], raw[start + 1 :], encoding


def _parse_header_document(head: bytes, filename: str, *, cut: bool) -> ET.Element:
    """Parse the pre-``<AppendedData>`` prefix, which is missing its closing tags.

    A *pull* parser rather than :func:`ET.fromstring`, because the prefix is deliberately
    an incomplete document and ``fromstring`` demands a complete one. Nothing has to be
    synthesised to make up the difference: ``ElementTree``'s tree builder appends each
    element to its parent at its own ``start`` event, so the element carried by the FIRST
    ``start`` event *is* the root, and by the time the feed returns it already has every
    element that completed before the cut hanging off it, text included. Synthesising
    closing tags and re-parsing would additionally have to reproduce each tag's source
    spelling — which is not recoverable from ``element.tag`` at all once a namespace is in
    play, and not recoverable from the prefix map either, since two prefixes may legally
    bind the same URI.

    ``cut`` says whether the stream really was truncated at ``<AppendedData``. When it was
    not, the whole file is here and is required to be a COMPLETE document, so the parser
    is closed and an unbalanced tag is reported rather than quietly tolerated.

    The event queue is DRAINED rather than broken out of at the first event, and that is
    load-bearing rather than tidiness. :meth:`XMLPullParser.feed` *catches* the parser's
    ``SyntaxError`` and appends it to the event queue; :meth:`read_events` re-raises it
    only when the iteration reaches that position. Stopping at the first ``start`` event
    therefore buries every markup error after the root tag — and on the appended arm
    nothing else ever surfaces it, because ``close()`` (which would) is deliberately not
    called. The result was a file whose second ``<Piece>`` the parser had rejected being
    imported as a smaller surface, with no error at all.

    ``xml.etree`` is used directly — bandit's B314; the waiver rationale is on the import
    at the top of this module.
    """
    # Typed loosely: the stub's event-tuple generic does not narrow to `Element` for a
    # parser constructed with an `events` tuple.
    parser: Any = ET.XMLPullParser(events=("start",))
    try:
        parser.feed(head)
        root: Optional[ET.Element] = None
        for _event, element in parser.read_events():
            if root is None:
                root = element
        if root is None:
            # No element started at all. `close()` raises ElementTree's own
            # "no element found", which names the position; the raise below is
            # unreachable belt-and-braces.
            parser.close()
            raise ValueError(f"{filename}: malformed VTK XML header — no element found")
        if not cut:
            parser.close()
        return root
    except (ET.ParseError, LookupError) as exc:
        # `LookupError`, not a parse error: expat delegates an unrecognised `encoding=`
        # in the XML declaration to Python's codec registry, and `<?xml version="1.0"
        # encoding="x-mac-roman"?>` on an otherwise perfect file raises "unknown
        # encoding" from there. Uncaught it escapes `import_mesh`'s documented
        # `ValueError` contract and the CLI's error funnel with it.
        raise ValueError(f"{filename}: malformed VTK XML header — {exc}") from exc


def _b64_chars(nbytes: int) -> int:
    """Characters a COMPLETE base64 stream of ``nbytes`` bytes occupies, padding included."""
    return ((nbytes + 2) // 3) * 4


def _b64_take(text: str, start: int, nbytes: int, ctx: _Context, where: str) -> bytes:
    """Decode the first ``nbytes`` bytes of the base64 stream beginning at ``start``.

    Taking a whole number of 4-character groups is what makes this safe to call on a
    stream that continues past the bytes wanted: 4 characters decode to exactly 3 bytes
    regardless of what follows, so a partial window decodes correctly and is trimmed.
    """
    chars = _b64_chars(nbytes)
    chunk = text[start : start + chars]
    if len(chunk) < chars:
        raise ValueError(
            f"{ctx.name}: {where} base64 stream is truncated — needs {chars} characters "
            f"at position {start}, found {len(chunk)}"
        )
    try:
        decoded = base64.b64decode(chunk)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{ctx.name}: {where} is not valid base64 — {exc}") from exc
    if len(decoded) < nbytes:
        raise ValueError(
            f"{ctx.name}: {where} base64 stream decoded to {len(decoded)} bytes, "
            f"expected at least {nbytes}"
        )
    return decoded[:nbytes]


def _words(buf: bytes, offset: int, count: int, ctx: _Context, where: str) -> list[int]:
    """Read ``count`` ``header_type``-wide integers starting at ``offset``."""
    need = count * ctx.header.itemsize
    if offset < 0 or offset + need > len(buf):
        raise ValueError(
            f"{ctx.name}: {where} block header needs {need} bytes at offset {offset}, "
            f"but the data holds {max(0, len(buf) - offset)}"
        )
    return [
        int(v) for v in np.frombuffer(buf, dtype=ctx.header, count=count, offset=offset)
    ]


def _inflate(
    blocks: bytes, start: int, sizes: list[int], ctx: _Context, where: str
) -> bytes:
    """Concatenate the zlib-inflated compressed blocks laid end to end from ``start``."""
    out = bytearray()
    position = start
    for size in sizes:
        if size < 0 or position + size > len(blocks):
            raise ValueError(
                f"{ctx.name}: {where} compressed block runs past the end of the data "
                f"({position + size} > {len(blocks)})"
            )
        try:
            out += zlib.decompress(blocks[position : position + size])
        except zlib.error as exc:
            raise ValueError(
                f"{ctx.name}: {where} zlib block failed to inflate — {exc}. A block "
                "header read at the wrong width or a single-stream base64 decode both "
                "land here."
            ) from exc
        position += size
    return bytes(out)


def _decode_raw(offset: int, ctx: _Context, where: str) -> bytes:
    """Decode one appended ``encoding="raw"`` block starting at ``offset``."""
    buf = ctx.appended
    width = ctx.header.itemsize
    if not ctx.compressed:
        (nbytes,) = _words(buf, offset, 1, ctx, where)
        start = offset + width
        if nbytes < 0 or start + nbytes > len(buf):
            raise ValueError(
                f"{ctx.name}: {where} declares {nbytes} bytes at offset {offset} but "
                f"only {max(0, len(buf) - start)} remain"
            )
        return buf[start : start + nbytes]

    (nblocks,) = _words(buf, offset, 1, ctx, where)
    _guard_nblocks(nblocks, (len(buf) - offset) // width, ctx, where)
    header = _words(buf, offset, 3 + nblocks, ctx, where)
    return _inflate(buf, offset + width * (3 + nblocks), header[3:], ctx, where)


def _decode_base64(text: str, start: int, ctx: _Context, where: str) -> bytes:
    """Decode one base64 block — inline ``format="binary"`` or appended base64.

    **This is trap 2.** Uncompressed, the byte-count header and the payload share ONE
    base64 stream. Compressed, they are two streams written back to back, so the payload
    must be decoded starting at the character the header's own stream ended on. Decoding
    the concatenation as a single stream produces well-formed-looking bytes that inflate
    to nothing, or to garbage — never an exception from base64 itself.
    """
    width = ctx.header.itemsize
    if not ctx.compressed:
        (nbytes,) = _words(_b64_take(text, start, width, ctx, where), 0, 1, ctx, where)
        if nbytes < 0:
            raise ValueError(f"{ctx.name}: {where} declares {nbytes} bytes")
        whole = _b64_take(text, start, width + nbytes, ctx, where)
        return whole[width:]

    (nblocks,) = _words(_b64_take(text, start, width, ctx, where), 0, 1, ctx, where)
    _guard_nblocks(nblocks, (len(text) - start) // width, ctx, where)
    header_bytes = width * (3 + nblocks)
    header = _words(
        _b64_take(text, start, header_bytes, ctx, where), 0, 3 + nblocks, ctx, where
    )
    sizes = header[3:]
    # The second stream begins exactly where the header's stream ended — padding and all.
    payload_start = start + _b64_chars(header_bytes)
    payload = _b64_take(text, payload_start, sum(sizes), ctx, where)
    return _inflate(payload, 0, sizes, ctx, where)


def _guard_nblocks(nblocks: int, ceiling: int, ctx: _Context, where: str) -> None:
    """Refuse an absurd block count before it is used to size an allocation.

    A header read at the wrong width, or at the wrong offset, produces a number in the
    billions. Bounding it by what the remaining data could possibly hold turns that into
    a named error instead of a MemoryError.
    """
    if nblocks < 0 or nblocks > max(ceiling, 0):
        raise ValueError(
            f"{ctx.name}: {where} declares {nblocks} compressed blocks, which the "
            "remaining data cannot hold — the block header was read at the wrong "
            f"width or offset (header_type is {ctx.header.itemsize * 8}-bit)"
        )


def _from_payload(
    payload: bytes, stored: np.dtype, ctx: _Context, where: str
) -> NDArray:
    """``np.frombuffer`` with the length check spelled out as a named error.

    A block header declaring a byte count that is not a whole number of items is exactly
    what a mis-read header or a truncated write looks like, and ``np.frombuffer``'s own
    "buffer size must be a multiple of element size" names neither the file nor the array.
    """
    if len(payload) % stored.itemsize:
        raise ValueError(
            f"{ctx.name}: {where} decoded to {len(payload)} bytes, which is not a "
            f"multiple of its {stored.itemsize}-byte item size — the block header "
            "declares a byte count the data does not match"
        )
    return np.frombuffer(payload, dtype=stored)


def _ascii_values(
    text: str, native: np.dtype, type_name: str, ctx: _Context, where: str
) -> NDArray:
    """Parse a whitespace-separated ``format="ascii"`` payload.

    Numpy's string cast refuses rather than truncates, so ``1.5`` in an ``Int64`` array is
    an error and not a silent 1 — but it raises TWO different exceptions and only one of
    them is a ``ValueError``. A malformed token (a word in a ``Float32`` array, ``1.5`` in
    an integer one) is a ``ValueError``; an integer token OUTSIDE the target dtype's range
    — ``300`` or ``-1`` in a ``UInt8`` colour array a writer forgot to clip, an oversized
    ``Int64`` connectivity index — is an ``OverflowError``, which is an ``ArithmeticError``
    and shares no base with ``ValueError``. Left uncaught it escapes ``import_mesh``'s
    documented contract *and* the CLI's ``except (ValueError, …)`` funnel, so the user
    gets a raw traceback for an ordinary bad file. Both are re-raised naming the file and
    the array, which neither carries.
    """
    tokens = text.split()
    try:
        values: NDArray = np.array(tokens, dtype=native)
    except (ValueError, OverflowError) as exc:
        raise ValueError(
            f"{ctx.name}: {where} has an ascii token that is not a {type_name} "
            f"value — {exc}"
        ) from exc
    return values


def _read_data_array(element: ET.Element, ctx: _Context, where: str) -> NDArray:
    """Decode one ``<DataArray>`` into a 1-D or ``(N, NumberOfComponents)`` array."""
    type_name = element.get("type", "")
    if type_name not in _VTK_DTYPES:
        raise ValueError(
            f"{ctx.name}: {where} has DataArray type {type_name!r}; expected one of "
            f"{', '.join(sorted(_VTK_DTYPES))}"
        )
    native = np.dtype(_VTK_DTYPES[type_name])
    stored = native.newbyteorder(ctx.order) if native.itemsize > 1 else native
    fmt = (element.get("format") or "ascii").lower()

    values: NDArray
    if fmt == "ascii":
        values = _ascii_values(element.text or "", native, type_name, ctx, where)
    elif fmt == "binary":
        # "binary" in VTK XML means base64-in-the-element-text, not raw bytes.
        text = "".join((element.text or "").split())
        values = _from_payload(_decode_base64(text, 0, ctx, where), stored, ctx, where)
    elif fmt == "appended":
        if not ctx.appended:
            raise ValueError(
                f"{ctx.name}: {where} is format='appended' but the file has no "
                "<AppendedData> section"
            )
        offset = _int_attr(element, "offset", ctx, where, 0)
        if offset < 0:
            raise ValueError(
                f"{ctx.name}: {where} has negative appended offset {offset}"
            )
        payload = (
            _decode_raw(offset, ctx, where)
            if not ctx.appended_text
            else _decode_base64(ctx.appended_text, offset, ctx, where)
        )
        values = _from_payload(payload, stored, ctx, where)
    else:
        raise ValueError(
            f"{ctx.name}: {where} has DataArray format {fmt!r}; expected 'ascii', "
            "'binary' or 'appended'"
        )

    ncomp = _int_attr(element, "NumberOfComponents", ctx, where, 1)
    if ncomp < 1:
        # Only `ncomp > 1` reshapes, so 0 or -3 would otherwise fall through as a 1-D
        # array — and the <Points> 1-D rescue below would then import it as if it had
        # said 3.
        raise ValueError(
            f"{ctx.name}: {where} declares NumberOfComponents={ncomp}; a DataArray has "
            "at least one component per tuple"
        )
    if ncomp > 1:
        if values.size % ncomp:
            raise ValueError(
                f"{ctx.name}: {where} decoded to {values.size} values, not a multiple "
                f"of its {ncomp} components"
            )
        values = values.reshape(-1, ncomp)
    return values


def _cell_rows(
    piece: ET.Element, tag: str, ctx: _Context, where: str
) -> list[list[int]]:
    """``<Polys>`` / ``<Strips>`` → one index list per cell.

    **Trap 3.** ``offsets`` holds cumulative END offsets with no leading zero, so the
    starts are the offsets shifted right by one with an implied 0 in front. Treating them
    as starts silently rotates the topology by one cell.
    """
    group = _child(piece, tag)
    if group is None:
        return []
    if not _children(group, "DataArray"):
        # A PRESENT but empty group is zero cells, exactly like an absent one. A surface
        # written entirely as <Strips> legitimately carries an empty <Polys></Polys>, and
        # refusing it for a missing 'connectivity' would reject the whole file.
        return []
    conn_el = _named_array(group, "connectivity")
    off_el = _named_array(group, "offsets")
    if conn_el is None or off_el is None:
        raise ValueError(
            f"{ctx.name}: {where} <{tag}> is missing its 'connectivity' or 'offsets' "
            "DataArray"
        )
    conn = np.asarray(
        _read_data_array(conn_el, ctx, f"{where} {tag}/connectivity"), dtype=np.int64
    ).reshape(-1)
    offsets = np.asarray(
        _read_data_array(off_el, ctx, f"{where} {tag}/offsets"), dtype=np.int64
    ).reshape(-1)
    if offsets.size == 0:
        return []
    if int(offsets[-1]) > conn.size:
        raise ValueError(
            f"{ctx.name}: {where} <{tag}> offsets end at {int(offsets[-1])} but the "
            f"connectivity array holds only {conn.size} indices"
        )
    if bool(np.any(np.diff(offsets) < 0)) or int(offsets[0]) < 0:
        raise ValueError(
            f"{ctx.name}: {where} <{tag}> offsets are not non-decreasing, so they are "
            "not the cumulative end offsets the VTK XML format defines"
        )
    starts = np.concatenate(([0], offsets[:-1]))
    return [conn[a:b].tolist() for a, b in zip(starts, offsets)]


def _strip_triangles(strip: list[int]) -> list[list[int]]:
    """Triangulate one triangle strip, flipping the winding on alternate elements.

    ``i, i+1, i+2`` for even *i* and ``i+1, i, i+2`` for odd — the standard flip, without
    which every other triangle faces backwards and the surface culls into stripes.
    """
    return [
        [strip[i], strip[i + 1], strip[i + 2]]
        if i % 2 == 0
        else [strip[i + 1], strip[i], strip[i + 2]]
        for i in range(len(strip) - 2)
    ]


def _colors_from(values: NDArray, is_uint8: bool) -> NDArray[np.uint8]:
    """Convert a colour array to uint8 by observed range — the ``_ply_mesh`` rule.

    A declared ``UInt8`` is already 0..255. A float array is 0..1 when its peak is at most
    1 and 0..255 otherwise; keying on the range rather than the declared type is what
    makes a ``Float32`` array written by a scanner (0..255) survive alongside ParaView's
    normalized output.
    """
    if is_uint8:
        clipped: NDArray[np.uint8] = np.clip(values, 0, 255).astype(np.uint8)
        return clipped
    peak = float(np.max(values)) if values.size else 0.0
    scaled = values * 255.0 if peak <= 1.0 else values
    out: NDArray[np.uint8] = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    return out


def _point_attributes(
    pdata: Optional[ET.Element], ctx: _Context, where: str, n_points: int
) -> tuple[Optional[NDArray[np.float32]], Optional[NDArray[np.uint8]]]:
    """Pull per-vertex normals and colours out of a ``<PointData>`` block."""
    if pdata is None:
        return None, None
    arrays = [e for e in pdata if _localname(e.tag) == "DataArray"]

    def ncomp(element: ET.Element) -> int:
        return _int_attr(element, "NumberOfComponents", ctx, where, 1)

    # `is not None` on BOTH sides is load-bearing. `<PointData>` need not carry a
    # `Normals=` designation, and a `<DataArray>` need not carry a `Name=` — VTK's own
    # unsigned-char colour arrays routinely have neither. Matching `None == None` adopts
    # the first NAMELESS array as normals, which turns a colour array into garbage
    # shading (or, at 4 components, into a hard error blaming normals for it).
    designated = pdata.get("Normals")
    normals_el = (
        None
        if designated is None
        else next((e for e in arrays if e.get("Name") == designated), None)
    )
    if normals_el is None:
        normals_el = next(
            (e for e in arrays if (e.get("Name") or "").lower() == "normals"), None
        )

    normals: Optional[NDArray[np.float32]] = None
    if normals_el is not None:
        raw = _read_data_array(normals_el, ctx, f"{where} PointData/Normals")
        if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] != n_points:
            raise ValueError(
                f"{ctx.name}: {where} PointData normals decoded to {raw.shape}; "
                f"expected ({n_points}, 3)"
            )
        normals = np.ascontiguousarray(raw, dtype=np.float32)

    scalars = pdata.get("Scalars")
    candidates = [e for e in arrays if e is not normals_el and ncomp(e) in (3, 4)]
    colors_el = (
        None
        if scalars is None
        else next((e for e in candidates if e.get("Name") == scalars), None)
    )
    if colors_el is None:
        colors_el = next(
            (e for e in candidates if (e.get("Name") or "").lower() in _COLOR_NAMES),
            None,
        )
    if colors_el is None:
        # VTK's own convention: unsigned-char 3/4-component point data IS colour. A
        # nameless FLOAT 3-vector is deliberately not taken — in a surface that is far
        # more often a displacement or velocity field.
        colors_el = next((e for e in candidates if e.get("type") == "UInt8"), None)

    colors: Optional[NDArray[np.uint8]] = None
    if colors_el is not None:
        raw = _read_data_array(colors_el, ctx, f"{where} PointData/colors")
        if raw.ndim != 2 or raw.shape[0] != n_points:
            raise ValueError(
                f"{ctx.name}: {where} PointData colours decoded to {raw.shape}; "
                f"expected ({n_points}, 3) or ({n_points}, 4)"
            )
        colors = _colors_from(raw, colors_el.get("type") == "UInt8")

    return normals, colors


def _build_context(
    root: ET.Element, path: Path, appended: bytes, encoding: str
) -> _Context:
    """Validate the ``<VTKFile>`` attributes and freeze them into a :class:`_Context`."""
    compressor = root.get("compressor") or ""
    if compressor and compressor != _ZLIB_COMPRESSOR:
        raise ValueError(
            f"{path.name}: compressor {compressor!r} is not supported — this reader "
            f"undoes {_ZLIB_COMPRESSOR} (zlib) only, and LZ4/LZMA need a codec that is "
            "not in the standard library. Re-save the file with zlib or no compression "
            "(ParaView: Save Data → Data Compressor)."
        )

    order_name = root.get("byte_order") or "LittleEndian"
    if order_name not in _BYTE_ORDERS:
        raise ValueError(
            f"{path.name}: byte_order={order_name!r}; expected 'LittleEndian' or "
            "'BigEndian'"
        )
    header_name = root.get("header_type") or "UInt32"
    if header_name not in _HEADER_DTYPES:
        raise ValueError(
            f"{path.name}: header_type={header_name!r}; expected 'UInt32' or 'UInt64'"
        )
    order = _BYTE_ORDERS[order_name]
    return _Context(
        name=path.name,
        order=order,
        header=np.dtype(_HEADER_DTYPES[header_name]).newbyteorder(order),
        compressed=compressor == _ZLIB_COMPRESSOR,
        appended=appended,
        appended_text=(
            appended.decode("ascii", errors="replace") if encoding == "base64" else ""
        ),
    )


def _read_points(piece: ET.Element, ctx: _Context, where: str) -> NDArray[np.float32]:
    """The ``(N, 3)`` positions of one ``<Piece>``, cross-checked against its own count."""
    group = _child(piece, "Points")
    array = _child(group, "DataArray") if group is not None else None
    if array is None:
        raise ValueError(f"{ctx.name}: {where} has no <Points><DataArray>")
    points = _read_data_array(array, ctx, f"{where} Points")
    if points.ndim == 1:
        if points.size % 3:
            raise ValueError(
                f"{ctx.name}: {where} Points decoded to {points.size} values, not a "
                "multiple of 3"
            )
        points = points.reshape(-1, 3)
    if points.shape[1] != 3:
        raise ValueError(
            f"{ctx.name}: {where} Points has {points.shape[1]} components; a PolyData "
            "point is always 3D"
        )
    declared = piece.get("NumberOfPoints")
    if (
        declared is not None
        and _int_attr(piece, "NumberOfPoints", ctx, where, 0) != points.shape[0]
    ):
        raise ValueError(
            f"{ctx.name}: {where} declares NumberOfPoints={declared} but its Points "
            f"array decoded to {points.shape[0]} — the data block was misread"
        )
    return np.ascontiguousarray(points, dtype=np.float32)


def _stack_if_complete(
    chunks: list[Optional[NDArray]], filename: str, label: str
) -> Optional[NDArray]:
    """Concatenate per-piece attribute arrays, or drop the attribute entirely.

    A multi-piece file where only SOME pieces carry normals (or colours) has no
    per-vertex value for the rest, and inventing one would shade or colour those pieces
    with data the file never gave them. All or nothing.

    Widths are checked by hand: pieces may disagree (RGB in one, RGBA in the next), and
    ``np.concatenate``'s own complaint about mismatched dimensions names neither the
    file nor which attribute went wrong.
    """
    present = [c for c in chunks if c is not None]
    if not present or len(present) != len(chunks):
        return None
    widths = sorted({int(c.shape[1]) for c in present})
    if len(widths) > 1:
        raise ValueError(
            f"{filename}: its <Piece>s carry {label} of different widths "
            f"({', '.join(str(w) for w in widths)} components), which cannot be stacked "
            "into one per-vertex array"
        )
    return np.concatenate(present, axis=0)


def read_vtp(path: Path) -> dict[str, object]:
    """Decode a VTK XML PolyData file into raw component arrays.

    Returns the dict shape :func:`luxar.mesh.interop.mesh_import.import_mesh` assembles a
    ``TriangleMesh`` from. Polygons come back as ``face_rows`` so the shared
    fan-triangulation runs on them, exactly as for PLY and OBJ; triangle strips are
    triangulated here (a strip is not a polygon loop, so a fan would be wrong) and merged
    into the same rows as 3-element entries, which the fan passes through unchanged.

    ``<Verts>`` and ``<Lines>`` are dropped without comment: they carry no surface, and a
    PolyData file that mixes a point cloud or a set of feature edges in beside its
    polygons is ordinary output, not an error.
    """
    raw = path.read_bytes()
    head, appended, encoding = _split_appended(raw, path.name)
    # A shorter head is exactly the case where the stream WAS cut at `<AppendedData`, so
    # the closing tags are legitimately missing; anything else is the whole file and must
    # still be a complete document.
    root = _parse_header_document(head, path.name, cut=len(head) < len(raw))

    if _localname(root.tag) != "VTKFile":
        raise ValueError(
            f"{path.name}: root element is <{_localname(root.tag)}>, not <VTKFile>"
        )
    declared_type = root.get("type") or ""
    if declared_type != "PolyData":
        raise ValueError(
            f"{path.name}: VTKFile type is {declared_type!r}, not 'PolyData' — "
            "`luxar mesh import` reads VTK XML PolyData surfaces (.vtp). Convert with "
            "ParaView's 'Extract Surface' filter first."
        )
    ctx = _build_context(root, path, appended, encoding)

    polydata = _child(root, "PolyData")
    pieces = _children(polydata, "Piece") if polydata is not None else []
    if not pieces:
        raise ValueError(f"{path.name}: <PolyData> declares no <Piece>")

    chunks: list[NDArray[np.float32]] = []
    face_rows: list[list[int]] = []
    normal_chunks: list[Optional[NDArray[np.float32]]] = []
    color_chunks: list[Optional[NDArray[np.uint8]]] = []
    base = 0

    for index, piece in enumerate(pieces):
        where = f"Piece {index}" if len(pieces) > 1 else "Piece"
        points = _read_points(piece, ctx, where)
        n_points = points.shape[0]

        rows = _cell_rows(piece, "Polys", ctx, where)
        for strip in _cell_rows(piece, "Strips", ctx, where):
            rows.extend(_strip_triangles(strip))

        for row in rows:
            for vertex in row:
                if vertex < 0 or vertex >= n_points:
                    raise ValueError(
                        f"{path.name}: {where} references vertex {vertex}, but it "
                        f"declares only {n_points} points"
                    )
        # Pieces are concatenated into one vertex array, so every index shifts by the
        # running base — bounded against its OWN piece above, before the shift hides it.
        face_rows.extend([v + base for v in row] for row in rows)

        normals, colors = _point_attributes(
            _child(piece, "PointData"), ctx, where, n_points
        )
        chunks.append(points)
        normal_chunks.append(normals)
        color_chunks.append(colors)
        base += n_points

    if not face_rows:
        raise ValueError(
            f"{path.name}: PolyData has no <Polys> or <Strips> — it carries only "
            "vertices and/or lines, which are not a surface. Load those with "
            "`scene.add_points(...)` or `scene.add_lines(...)`."
        )

    return {
        "vertices": np.concatenate(chunks, axis=0),
        "face_rows": face_rows,
        "normals": _stack_if_complete(normal_chunks, path.name, "normals"),
        "colors": _stack_if_complete(color_chunks, path.name, "colours"),
    }
