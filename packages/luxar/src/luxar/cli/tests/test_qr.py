"""The hand-rolled QR encoder is verified by DECODING, not by comparison.

Why that distinction earns its own paragraph. The first version of this
encoder was checked against `segno`, another encoder, and disagreed on one
codeword of padding. Reading segno's source suggested segno was the one
deviating from the specification, which was a comfortable conclusion and the
wrong one: a real decoder read segno's symbols and could not read ours at all.
Two bugs were hiding behind a plausible-looking matrix — a Reed-Solomon
generator built with its coefficients reversed, and alignment patterns dropped
wherever a centre fell on the timing row. Both produce a symbol of the right
size with the right finders that no camera will ever read.

So the oracle here is `zxing-cpp`, a decoder. "Does it scan" is the only
property the export actually needs.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from luxar.cli import _qr
from luxar.cli._qr import (
    QrError,
    qr_ascii,
    qr_matrix,
    qr_png_bytes,
)

zxingcpp = pytest.importorskip("zxingcpp", reason="the QR oracle is a test dep")
Image = pytest.importorskip("PIL.Image", reason="reading the PNG under test")


def _decode(png: bytes) -> str | None:
    """Read a QR out of PNG bytes, or None when nothing scans."""
    img = np.array(Image.open(io.BytesIO(png)).convert("L"))
    found = zxingcpp.read_barcodes(img)
    return found[0].text if found else None


def _roundtrip(text: str, *, scale: int = 4, border: int = 4) -> str | None:
    return _decode(qr_png_bytes(qr_matrix(text), scale=scale, border=border))


# The payloads this actually ships: a loopback URL, a LAN URL, and a LAN URL
# carrying a control token.
URLS = [
    "http://127.0.0.1:8000/viewer/control.html",
    "http://192.168.1.42:8000/viewer/control.html",
    "http://192.168.1.42:8000/viewer/control.html?token=7f3a9c2e1b4d8a60",
    "http://10.0.0.7:8765/viewer/control.html?token=" + "a" * 64,
]


@pytest.mark.parametrize("url", URLS)
def test_a_control_panel_url_scans(url: str) -> None:
    assert _roundtrip(url) == url


def test_every_supported_version_scans_at_its_capacity_boundary() -> None:
    """One, half, one-under and exactly-full, for versions 1 to 20.

    The boundary cases are where the padding and block-splitting logic lives,
    and the multi-block versions (7 and up) are where the alignment-pattern
    bug hid.
    """
    for version in sorted(_qr._VERSION_TABLE):
        capacity = _qr._byte_capacity(version)
        for length in sorted({1, capacity // 2, capacity - 1, capacity}):
            if length < 1:
                continue
            text = "".join(chr(0x41 + (i * 7) % 26) for i in range(length))
            matrix = qr_matrix(text)
            assert (len(matrix) - 17) // 4 >= version or length < capacity
            assert _decode(qr_png_bytes(matrix, scale=4)) == text, (
                f"version {version}, {length} bytes"
            )


def test_non_ascii_and_url_punctuation_survive() -> None:
    """Byte mode carries UTF-8, and the query string's punctuation is safe."""
    for text in ("tokens/|?&=#% and unicode: café ångström", "█▀"):
        assert _roundtrip(text) == text


def test_over_capacity_raises_instead_of_truncating() -> None:
    """A silently shortened URL would scan and go to the wrong place."""
    biggest = _qr._byte_capacity(20)
    assert _roundtrip("z" * biggest) == "z" * biggest
    with pytest.raises(QrError, match="does not fit"):
        qr_matrix("z" * (biggest + 1))


def test_the_smallest_version_that_fits_is_chosen() -> None:
    """A needlessly large symbol is harder to scan from a phone."""
    sizes = [len(qr_matrix("h" * n)) for n in (10, 30, 60, 120)]
    assert sizes == sorted(sizes)
    assert len(qr_matrix("h" * 10)) == 21  # version 1


@pytest.mark.parametrize("scale,border", [(1, 0), (2, 1), (4, 4), (10, 8)])
def test_it_still_scans_across_scales_and_quiet_zones(scale: int, border: int) -> None:
    """Except at border 0, which the standard does not permit; assert nothing there."""
    url = URLS[2]
    out = _decode(qr_png_bytes(qr_matrix(url), scale=scale, border=border))
    if border >= 2 and scale >= 2:
        assert out == url
    # A too-small quiet zone or one-pixel modules may legitimately fail to
    # scan; the point of the case is that it does not raise.


def test_the_png_is_a_valid_greyscale_png() -> None:
    png = qr_png_bytes(qr_matrix(URLS[0]), scale=3, border=4)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    img = Image.open(io.BytesIO(png))
    assert img.mode in ("L", "1")
    size = len(qr_matrix(URLS[0])) + 8
    assert img.size == (size * 3, size * 3)


def test_the_terminal_rendering_is_a_lossless_view_of_the_matrix() -> None:
    """Half blocks pack two matrix rows per text row, so the symbol is square.

    One module per character comes out twice as tall as wide and many phone
    cameras will not lock onto it, so this is not cosmetic.
    """
    matrix = qr_matrix(URLS[2])
    art = qr_ascii(matrix, border=2)
    inverse = {glyph: pair for pair, glyph in _qr._HALF.items()}
    rows: list[list[bool]] = []
    for line in art.split("\n"):
        top, bottom = [], []
        for glyph in line:
            t, b = inverse[glyph]
            top.append(t)
            bottom.append(b)
        rows.append(top)
        rows.append(bottom)
    size = len(matrix)
    assert [row[2 : 2 + size] for row in rows[2 : 2 + size]] == matrix
    # Square-ish: half as many text rows as module rows (the quiet zone
    # included), rounded up, since each text row carries two module rows.
    assert len(art.split("\n")) == -(-(size + 4) // 2)


def test_inverting_swaps_dark_and_light_but_keeps_the_shape() -> None:
    matrix = qr_matrix(URLS[0])
    normal = qr_ascii(matrix, border=2)
    inverted = qr_ascii(matrix, border=2, invert=True)
    assert normal != inverted
    assert len(normal) == len(inverted)


def test_padding_codewords_are_the_specified_alternating_pair() -> None:
    """Pin the pad bytes, which the decoder oracle structurally cannot see.

    "Does it scan" is the right oracle for almost everything here, but it has
    one blind spot: padding lives PAST the terminator, so a symbol with wrong
    pad bytes still decodes to the right payload. The error correction is
    computed over those bytes, so a conforming reader is happy and only a
    comparison against the specification catches it — and it was exactly one
    codeword of padding that started the segno confusion described above.

    The standard fills the remaining data capacity with 0b11101100 and
    0b00010001 alternating, beginning with 0b11101100.
    """
    version = 1
    total, ec_per_block, (g1, g2) = _qr._VERSION_TABLE[version]
    capacity = total - ec_per_block * (g1 + g2)

    def expected_pad(payload_len: int) -> list[int]:
        """Codewords the payload itself occupies, then the rest as pad bytes."""
        used_bits = 4 + 8 + 8 * payload_len  # mode + count + data
        used = -(-min(used_bits + 4, capacity * 8) // 8)  # + terminator, rounded up
        return [(0xEC, 0x11)[i % 2] for i in range(capacity - used)]

    codewords = _qr._data_codewords(b"hi", version, capacity)
    assert len(codewords) == capacity
    # 4 bits mode + 8 bits count + 2 payload bytes + 4 bits terminator, padded
    # to a byte boundary, is 4 codewords; the remaining 15 are padding, and 15
    # is ODD -- the alternation does not end on a pair.
    assert codewords[:4] == [0x40, 0x26, 0x86, 0x90]
    assert codewords[4:] == expected_pad(2)
    assert codewords[4:] == [0xEC, 0x11] * 7 + [0xEC]

    # The two boundaries where an off-by-one in the fill loop would hide: a
    # payload that lands exactly on capacity takes no padding, and one byte
    # less takes exactly one pad codeword.
    exact_len = capacity - 2
    assert expected_pad(exact_len) == []
    exact = _qr._data_codewords(b"Z" * exact_len, version, capacity)
    assert len(exact) == capacity
    # The 12-bit header (4 mode + 8 count) is not a whole number of bytes, so
    # payload bytes STRADDLE codeword boundaries: the final codeword is the low
    # nibble of the last "Z" (0x5A -> 1010) followed by the four terminator
    # zeros, which is 0xA0 -- not 0x5A, and not a pad byte.
    assert exact[-1] == 0xA0

    one_short = _qr._data_codewords(b"Z" * (exact_len - 1), version, capacity)
    assert expected_pad(exact_len - 1) == [0xEC]
    assert one_short[-1] == 0xEC
    assert len(one_short) == capacity
