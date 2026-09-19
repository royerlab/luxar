"""A minimal QR encoder, Python stdlib only.

Why hand-rolled. The exported scene folder ships a ``serve.py`` that must run
with nothing installed (see :mod:`luxar.cli.export`), and its whole point is
being handed to someone who has never installed Luxar. A QR of the control-panel
URL is the difference between an operator typing ``192.168.1.42:8000/viewer/
control.html?token=...`` into a tablet and pointing a camera at the screen, so
it has to work in that zero-dependency folder. `segno` and `qrcode` are both
pure Python and both excellent, and neither may be assumed present.

Scope is deliberately the narrow case this needs, not the QR standard:

* **Byte mode only.** URLs are ASCII; alphanumeric mode would be denser but
  cannot carry a lowercase path or a query string.
* **Error correction level L.** A URL shown on a screen or a clean print is
  read from a few centimetres away with no occlusion, so the 7% recovery level
  is right; higher levels only make the symbol denser for no gain here.
* **Versions 1-20.** The smallest version that fits is chosen. Version 20
  holds 858 bytes at level L — an order of magnitude more than the longest
  plausible tokenised LAN URL.
* **Mask pattern chosen by the standard's penalty rules**, because a bad mask
  is the usual reason a home-made QR scans on one phone and not another.

The oracle is a DECODER, not another encoder: ``test_qr.py`` renders each
matrix and asserts `zxing-cpp` reads the payload back, over a spread of sizes
and versions. Agreement with another encoder does not establish that a symbol
scans, and a symbol of the right size with the right finders can still be
unreadable. zxing-cpp is a TEST dependency and is never imported at runtime.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Callable
from typing import TYPE_CHECKING

__all__ = [
    "QrError",
    "qr_matrix",
    "qr_ascii",
    "qr_png_bytes",
]


class QrError(ValueError):
    """The payload does not fit the supported versions."""


#: Total codewords and level-L error-correction blocks per version, 1-20.
#: ``(total_codewords, ec_codewords_per_block, (blocks_group1, blocks_group2))``
#: from the QR specification's table 9; level L only.
_VERSION_TABLE: dict[int, tuple[int, int, tuple[int, int]]] = {
    1: (26, 7, (1, 0)),
    2: (44, 10, (1, 0)),
    3: (70, 15, (1, 0)),
    4: (100, 20, (1, 0)),
    5: (134, 26, (1, 0)),
    6: (172, 18, (2, 0)),
    7: (196, 20, (2, 0)),
    8: (242, 24, (2, 0)),
    9: (292, 30, (2, 0)),
    10: (346, 18, (2, 2)),
    11: (404, 20, (4, 0)),
    12: (466, 24, (2, 2)),
    13: (532, 26, (4, 0)),
    14: (581, 30, (3, 1)),
    15: (655, 22, (5, 1)),
    16: (733, 24, (5, 1)),
    17: (815, 28, (1, 5)),
    18: (901, 30, (5, 1)),
    19: (991, 28, (3, 4)),
    20: (1085, 28, (3, 5)),
}

#: Alignment-pattern centre coordinates per version (specification annex E).
_ALIGNMENT: dict[int, list[int]] = {
    1: [],
    2: [6, 18],
    3: [6, 22],
    4: [6, 26],
    5: [6, 30],
    6: [6, 34],
    7: [6, 22, 38],
    8: [6, 24, 42],
    9: [6, 26, 46],
    10: [6, 28, 50],
    11: [6, 30, 54],
    12: [6, 32, 58],
    13: [6, 34, 62],
    14: [6, 26, 46, 66],
    15: [6, 26, 48, 70],
    16: [6, 26, 50, 74],
    17: [6, 30, 54, 78],
    18: [6, 30, 56, 82],
    19: [6, 30, 58, 86],
    20: [6, 34, 62, 90],
}

#: Version information bit strings for versions 7+ (18 bits, BCH-encoded).
_VERSION_BITS: dict[int, int] = {
    7: 0x07C94,
    8: 0x085BC,
    9: 0x09A99,
    10: 0x0A4D3,
    11: 0x0BBF6,
    12: 0x0C762,
    13: 0x0D847,
    14: 0x0E60D,
    15: 0x0F928,
    16: 0x10B78,
    17: 0x1145D,
    18: 0x12A17,
    19: 0x13532,
    20: 0x149A6,
}


# The symbol under construction. ``_Grid`` holds 0/1 for a placed module and
# ``None`` for one the data walk still has to fill; ``_Mask`` marks every module
# the walk must skip, which is not the same set -- a reserved module can still be
# ``None`` (the format strips are claimed long before their bits are known).
#
# Behind TYPE_CHECKING because this module is COPIED VERBATIM into an exported
# folder and must import on whatever Python the recipient has; stock macOS is
# still 3.9. The __future__ import above defers ANNOTATIONS, but a module-level
# alias is an expression evaluated at import, and `int | None` raises TypeError
# before 3.10.
if TYPE_CHECKING:
    _Grid = list[list[int | None]]
    _Mask = list[list[bool]]


# ---------------------------------------------------------------------------
# GF(256) arithmetic and Reed-Solomon
# ---------------------------------------------------------------------------

_EXP = [0] * 512
_LOG = [0] * 256


def _init_tables() -> None:
    x = 1
    for i in range(255):
        _EXP[i] = x
        _LOG[x] = i
        x <<= 1
        if x & 0x100:  # the QR field's primitive polynomial, x^8+x^4+x^3+x^2+1
            x ^= 0x11D
    for i in range(255, 512):
        _EXP[i] = _EXP[i - 255]


_init_tables()


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _rs_generator(degree: int) -> list[int]:
    """The generator polynomial for ``degree`` error-correction codewords.

    Coefficients HIGHEST POWER FIRST, so ``poly[0]`` is the monic leading 1
    and :func:`_rs_remainder` can index ``gen[i + 1]``. An earlier version
    built the same polynomial reversed, which produced plausible-looking but
    wrong check bytes: the symbol had the right size, the right function
    patterns and the right data modules, and no decoder would read it. Only a
    real decoder catches that, which is why `test_qr.py` decodes rather than
    comparing against another encoder.
    """
    poly = [1]
    for i in range(degree):
        poly.append(0)
        for j in range(len(poly) - 1, 0, -1):
            poly[j] ^= _gf_mul(poly[j - 1], _EXP[i])
    return poly


def _rs_remainder(data: list[int], degree: int) -> list[int]:
    """Reed-Solomon check codewords for ``data``."""
    gen = _rs_generator(degree)
    remainder = [0] * degree
    for byte in data:
        factor = byte ^ remainder[0]
        remainder = remainder[1:] + [0]
        for i in range(degree):
            remainder[i] ^= _gf_mul(gen[i + 1], factor)
    return remainder


# ---------------------------------------------------------------------------
# Data encoding
# ---------------------------------------------------------------------------


def _byte_capacity(version: int) -> int:
    """How many payload bytes fit in ``version`` at level L."""
    total, ec_per_block, (g1, g2) = _VERSION_TABLE[version]
    blocks = g1 + g2
    data_codewords = total - ec_per_block * blocks
    # mode indicator (4 bits) + character count (8 or 16 bits)
    header_bits = 4 + (8 if version < 10 else 16)
    return data_codewords - (header_bits + 7) // 8


def _pick_version(length: int) -> int:
    for version in sorted(_VERSION_TABLE):
        if length <= _byte_capacity(version):
            return version
    raise QrError(
        f"{length} bytes does not fit a version-20 level-L QR "
        f"(max {_byte_capacity(20)}); shorten the URL"
    )


def _data_codewords(data: bytes, version: int, data_codewords: int) -> list[int]:
    """Payload -> exactly ``data_codewords`` bytes: header, data, terminator, padding."""
    bits: list[int] = []

    def put(value: int, width: int) -> None:
        for i in range(width - 1, -1, -1):
            bits.append((value >> i) & 1)

    put(0b0100, 4)  # byte mode
    put(len(data), 8 if version < 10 else 16)
    for byte in data:
        put(byte, 8)

    capacity_bits = data_codewords * 8
    if len(bits) > capacity_bits:  # pragma: no cover - _pick_version prevents it
        raise QrError("payload overflows the chosen version")
    # Terminator, then pad to a byte boundary, then the alternating pad bytes.
    put(0, min(4, capacity_bits - len(bits)))
    while len(bits) % 8:
        bits.append(0)
    codewords = [
        int("".join(str(b) for b in bits[i : i + 8]), 2) for i in range(0, len(bits), 8)
    ]
    pad = 0xEC
    while len(codewords) < data_codewords:
        codewords.append(pad)
        pad = 0x11 if pad == 0xEC else 0xEC
    return codewords[:data_codewords]


def _split_blocks(
    codewords: list[int], version: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Split into the version's blocks and compute each one's EC codewords.

    Group 2 blocks hold one more data codeword each than group 1, which is the
    whole reason the split is not a plain even division.
    """
    _, ec_per_block, (g1, g2) = _VERSION_TABLE[version]
    blocks = g1 + g2
    short_len = len(codewords) // blocks
    data_blocks: list[list[int]] = []
    ec_blocks: list[list[int]] = []
    pos = 0
    for index in range(blocks):
        size = short_len + (1 if index >= g1 else 0)
        block = codewords[pos : pos + size]
        pos += size
        data_blocks.append(block)
        ec_blocks.append(_rs_remainder(block, ec_per_block))
    return data_blocks, ec_blocks


def _interleave(data_blocks: list[list[int]], ec_blocks: list[list[int]]) -> list[int]:
    """Column-major across blocks, all data first and then all error correction."""
    out: list[int] = []
    for i in range(max(len(b) for b in data_blocks)):
        for block in data_blocks:
            if i < len(block):
                out.append(block[i])
    for i in range(len(ec_blocks[0])):
        for block in ec_blocks:
            out.append(block[i])
    return out


def _encode_data(data: bytes, version: int) -> list[int]:
    """Payload -> interleaved data + error-correction codewords."""
    total, ec_per_block, (g1, g2) = _VERSION_TABLE[version]
    data_codewords = total - ec_per_block * (g1 + g2)
    codewords = _data_codewords(data, version, data_codewords)
    return _interleave(*_split_blocks(codewords, version))


# ---------------------------------------------------------------------------
# Matrix construction
# ---------------------------------------------------------------------------

#: The eight standard mask functions, indexed by mask number.
_MASKS: tuple[Callable[[int, int], bool], ...] = (
    lambda r, c: (r + c) % 2 == 0,
    lambda r, c: r % 2 == 0,
    lambda r, c: c % 3 == 0,
    lambda r, c: (r + c) % 3 == 0,
    lambda r, c: (r // 2 + c // 3) % 2 == 0,
    lambda r, c: (r * c) % 2 + (r * c) % 3 == 0,
    lambda r, c: ((r * c) % 2 + (r * c) % 3) % 2 == 0,
    lambda r, c: ((r + c) % 2 + (r * c) % 3) % 2 == 0,
)

#: Format-information bit strings for level L, masks 0-7 (15 bits, BCH).
_FORMAT_BITS = (
    0x77C4, 0x72F3, 0x7DAA, 0x789D, 0x662F, 0x6318, 0x6C41, 0x6976,
)  # fmt: skip


def _place_finders(grid: _Grid, reserved: _Mask, size: int) -> None:
    """The three 7x7 finder patterns, each with its one-module light separator."""
    for top, left in ((0, 0), (0, size - 7), (size - 7, 0)):
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                r, c = top + dr, left + dc
                if not (0 <= r < size and 0 <= c < size):
                    continue
                ring = max(abs(dr - 3), abs(dc - 3))
                grid[r][c] = 1 if ring in (0, 1, 3) else 0
                reserved[r][c] = True


def _place_timing(grid: _Grid, reserved: _Mask, size: int) -> None:
    """The alternating timing patterns along row and column 6."""
    for i in range(size):
        if not reserved[6][i]:
            grid[6][i] = 1 if i % 2 == 0 else 0
            reserved[6][i] = True
        if not reserved[i][6]:
            grid[i][6] = 1 if i % 2 == 0 else 0
            reserved[i][6] = True


def _place_alignment(grid: _Grid, reserved: _Mask, version: int) -> None:
    """The 5x5 alignment patterns at every centre pair except three.

    Exactly THREE centres are omitted -- the ones that would sit on a finder:
    ``(first, first)``, ``(first, last)`` and ``(last, first)``. Every other
    centre is drawn even when it lands on the timing row or column, where the
    alignment pattern takes precedence and the timing pattern simply continues
    either side of it.

    Skipping any centre whose module was already reserved (the obvious
    shortcut) is WRONG from version 7 up, where centres like (6, 22) and
    (22, 6) lie on the timing lines and are real patterns. It is right by
    accident for versions 2-6, whose only non-finder centre is (last, last),
    which is why the bug survived until a multi-alignment version was decoded.
    """
    centres = _ALIGNMENT[version]
    if not centres:
        return
    first, last = centres[0], centres[-1]
    skip = {(first, first), (first, last), (last, first)}
    for r in centres:
        for c in centres:
            if (r, c) in skip:
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    ring = max(abs(dr), abs(dc))
                    grid[r + dr][c + dc] = 1 if ring in (0, 2) else 0
                    reserved[r + dr][c + dc] = True


def _reserve_format_areas(grid: _Grid, reserved: _Mask, size: int) -> None:
    """Reserve both format-information strips and set the always-dark module.

    The format bits themselves depend on the mask, so they are written once the
    mask is chosen; here the modules are only claimed so the data walk skips
    them.
    """
    for i in range(9):
        for r, c in ((8, i), (i, 8)):
            if 0 <= r < size and 0 <= c < size:
                reserved[r][c] = True
    for i in range(8):
        reserved[8][size - 1 - i] = True
        reserved[size - 1 - i][8] = True
    grid[size - 8][8] = 1  # dark module
    reserved[size - 8][8] = True


def _place_version_info(grid: _Grid, reserved: _Mask, size: int, version: int) -> None:
    """The two 18-bit version blocks, which exist only from version 7 up."""
    if version < 7:
        return
    value = _VERSION_BITS[version]
    for i in range(18):
        bit = (value >> i) & 1
        grid[size - 11 + i % 3][i // 3] = bit
        reserved[size - 11 + i % 3][i // 3] = True
        grid[i // 3][size - 11 + i % 3] = bit
        reserved[i // 3][size - 11 + i % 3] = True


def _place_function_patterns(size: int, version: int) -> tuple[_Grid, _Mask]:
    """An empty grid with finders, timing, alignment and format areas reserved.

    Returns ``(grid, reserved)``. ``grid`` holds 0/1 for placed modules and
    ``None`` where data goes; ``reserved`` marks every module the data walk
    must skip. Order matters only in that the finders run first, so the timing
    pass can tell a finder module from a free one.
    """
    grid: _Grid = [[None] * size for _ in range(size)]
    reserved: _Mask = [[False] * size for _ in range(size)]
    _place_finders(grid, reserved, size)
    _place_timing(grid, reserved, size)
    _place_alignment(grid, reserved, version)
    _reserve_format_areas(grid, reserved, size)
    _place_version_info(grid, reserved, size, version)
    return grid, reserved


def _place_data(grid: _Grid, reserved: _Mask, codewords: list[int]) -> None:
    """Walk the two-module-wide serpentine and drop the bit stream in."""
    size = len(grid)
    bits = [(byte >> i) & 1 for byte in codewords for i in range(7, -1, -1)]
    index = 0
    upward = True
    col = size - 1
    while col > 0:
        if col == 6:  # the vertical timing pattern is never a data column
            col -= 1
        rows = range(size - 1, -1, -1) if upward else range(size)
        for row in rows:
            for c in (col, col - 1):
                if reserved[row][c]:
                    continue
                grid[row][c] = bits[index] if index < len(bits) else 0
                index += 1
        upward = not upward
        col -= 2


def _penalty_runs(lines: list[list[bool]]) -> int:
    """Rule 1: a run of five or more same-coloured modules scores 3, +1 each beyond five."""
    score = 0
    for line in lines:
        run, prev = 1, line[0]
        for value in line[1:]:
            if value == prev:
                run += 1
                continue
            if run >= 5:
                score += 3 + (run - 5)
            run, prev = 1, value
        if run >= 5:
            score += 3 + (run - 5)
    return score


def _penalty_blocks(matrix: list[list[bool]]) -> int:
    """Rule 2: every 2x2 block of one colour scores 3."""
    size = len(matrix)
    score = 0
    for r in range(size - 1):
        for c in range(size - 1):
            quad = (
                matrix[r][c],
                matrix[r][c + 1],
                matrix[r + 1][c],
                matrix[r + 1][c + 1],
            )
            if all(quad) or not any(quad):
                score += 3
    return score


def _penalty_finder_lookalike(lines: list[list[bool]], size: int) -> int:
    """Rule 3: a 1:1:3:1:1 finder lookalike with four light modules on one side scores 40."""
    dark_light = [True, False, True, True, True, False, True]
    before = [False] * 4 + dark_light
    after = dark_light + [False] * 4
    score = 0
    for line in lines:
        for i in range(size - 10):
            window = line[i : i + 11]
            if window == before or window == after:
                score += 40
    return score


def _penalty_dark_balance(matrix: list[list[bool]], size: int) -> int:
    """Rule 4: deviation of the dark share from 50%, in 5% steps, scores 10 a step."""
    dark = sum(1 for row in matrix for value in row if value)
    percent = dark * 100 / (size * size)
    return 10 * (int(abs(percent - 50)) // 5)


def _penalty(matrix: list[list[bool]]) -> int:
    """The standard's four penalty rules, summed, used to choose the mask.

    Implemented to the letter because the mask is chosen by comparing these
    scores: a wrong rule picks a different mask, and the symbol then differs
    from every conforming encoder's even though both scan. That makes the rules
    the part of this module least likely to fail loudly, so each is its own
    named function against the clause it implements.
    """
    size = len(matrix)
    # strict: the matrix is square, so the transpose consumes every row.
    lines = [list(row) for row in matrix] + [
        list(col) for col in zip(*matrix, strict=True)
    ]
    return (
        _penalty_runs(lines)
        + _penalty_blocks(matrix)
        + _penalty_finder_lookalike(lines, size)
        + _penalty_dark_balance(matrix, size)
    )


def qr_matrix(data: str) -> list[list[bool]]:
    """Encode ``data`` as a level-L QR matrix of booleans (True = dark).

    No quiet zone: the renderers add it, since the border a terminal wants and
    the border a PNG wants differ.
    """
    payload = data.encode("utf-8")
    version = _pick_version(len(payload))
    codewords = _encode_data(payload, version)
    size = version * 4 + 17

    best: tuple[int, list[list[bool]]] | None = None
    for mask in range(8):
        grid, reserved = _place_function_patterns(size, version)
        _place_data(grid, reserved, codewords)
        rule = _MASKS[mask]
        matrix = [
            [
                bool(grid[r][c]) ^ (rule(r, c) and not reserved[r][c])
                for c in range(size)
            ]
            for r in range(size)
        ]
        # Format information, both copies, after masking.
        fmt = _FORMAT_BITS[mask]
        for i in range(15):
            bit = bool((fmt >> i) & 1)
            if i < 6:
                matrix[8][i] = bit
            elif i == 6:
                matrix[8][7] = bit
            elif i == 7:
                matrix[8][8] = bit
            elif i == 8:
                matrix[7][8] = bit
            else:
                matrix[14 - i][8] = bit
            if i < 8:
                matrix[8][size - 1 - i] = bit
            else:
                matrix[size - 15 + i][8] = bit
        matrix[size - 8][8] = True  # dark module, never masked
        score = _penalty(matrix)
        if best is None or score < best[0]:
            best = (score, matrix)
    assert best is not None
    return best[1]


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

#: Unicode half blocks: two matrix rows per text row, so the symbol comes out
#: roughly square in a terminal whose cells are about twice as tall as wide.
#: A QR printed one module per character is twice as tall as it is wide and
#: many phone cameras refuse to lock onto it.
_HALF = {(False, False): " ", (True, False): "▀", (False, True): "▄", (True, True): "█"}


def qr_ascii(matrix: list[list[bool]], *, border: int = 2, invert: bool = False) -> str:
    """The matrix as half-block text, ready to print to a terminal.

    ``border`` is the quiet zone in modules; the standard asks for 4, but 2 is
    enough against a terminal background and keeps the symbol inside an
    80-column window up to version 6. ``invert`` swaps dark and light, for a
    light-on-dark terminal where an un-inverted symbol scans badly.
    """
    size = len(matrix)
    width = size + 2 * border
    rows = [[False] * width for _ in range(border)]
    rows += [[False] * border + list(row) + [False] * border for row in matrix]
    rows += [[False] * width for _ in range(border)]
    if len(rows) % 2:
        rows.append([False] * width)
    out = []
    # strict: the pad above makes the row count even, so the halves pair exactly.
    for top, bottom in zip(rows[0::2], rows[1::2], strict=True):
        out.append(
            "".join(
                _HALF[(t ^ invert, b ^ invert)]
                for t, b in zip(top, bottom, strict=True)
            )
        )
    return "\n".join(out)


def qr_png_bytes(matrix: list[list[bool]], *, scale: int = 8, border: int = 4) -> bytes:
    """The matrix as a 1-bit greyscale PNG.

    Written by hand from zlib and struct rather than with Pillow, for the same
    reason the encoder is hand-rolled: this runs in the exported folder, where
    nothing is installed. ``border`` defaults to the standard's four-module
    quiet zone, since a PNG may be printed or shown on a light page where the
    zone is what separates the symbol from whatever is beside it.
    """
    size = len(matrix)
    width = (size + 2 * border) * scale
    grid = [[False] * (size + 2 * border) for _ in range(border)]
    grid += [[False] * border + list(row) + [False] * border for row in matrix]
    grid += [[False] * (size + 2 * border) for _ in range(border)]

    raw = bytearray()
    for row in grid:
        line = bytearray()
        for value in row:
            line += bytes([0 if value else 255]) * scale
        for _ in range(scale):
            raw += b"\x00" + line  # filter type 0 (none) per scanline

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, width, 8, 0, 0, 0, 0)  # 8-bit grey
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
