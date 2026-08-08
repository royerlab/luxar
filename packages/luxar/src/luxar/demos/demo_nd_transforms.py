#!/usr/bin/env python3
"""Self-Contained Demo: nD Transform Test Bench

An instrument, not a picture. This demo exists to make Luxar's ``nd_transform``
feature — per-dimension affine / permutation transforms on NON-DISPLAYED
dimensions — obvious to verify with your eyes, tick by tick.

================================================================================
HOW TO READ THE BENCH
================================================================================

Everything is laid out against a ruler along X. One tick = one frame index::

    WORLD FRAME  0    1    2    3    4    5    6    7    8   ...  15
                 |    |    |    |    |    |    |    |    |         |
    cursor  ─────────────────────────────────────┃   (world T = 7)

    IDENTITY     ·    ·    ·    ·    ·    ·    ·   [7]   ·   ...
    OFFSET +5    ·    ·   [2]   ·    ·    ·    ·    ·    ·   ...
                          └────── 5 ticks ──────┘
    REVERSE      ·    ·    ·    ·    ·    ·    ·    ·   [8]  ...

* The bright cyan cursor column marks the WORLD frame T — the value on the
  slider. It is plain untransformed geometry: 16 point columns, column k tagged
  ``Frame = k``, with no ``nd_transform`` anywhere near it.
* Each row is a group carrying ONE ``nd_transform``. Its markers are 3D
  point-font digits that print their own LOCAL frame index — so whichever digit
  the viewer shows IS the local index its inverse-query resolved to.
* Faint grey ghost digits (always on via ``extend_to_all``) mark every slot a
  row could ever light, so a dark row reads as "no local frame maps to this T",
  never as "the row failed to load".

The verification therefore needs no trust in any label::

    read the digit   →  the LOCAL index the viewer chose
    read the cursor  →  the WORLD index you asked for
    count the ticks  →  that gap IS the transform

A bottom-left readout prints the EXPECTED local index for every row at the
current T, generated in Python from the same transform definitions that placed
the data. If the render and the readout disagree, the feature is broken — and
you see it in one glance instead of squinting at a point cloud.

================================================================================
WHAT EACH ROW TESTS
================================================================================

Frame (discrete ordinal → affine ``scale`` / ``offset``)::

  row                   nd_transform                 world=f(local)  signature
  IDENTITY              (none)                       T = k           under the cursor
  OFFSET +5             {"offset": 5}                T = k + 5       5 ticks left
  OFFSET -3             {"offset": -3}               T = k - 3       3 ticks right
  SCALE *2              {"scale": 2}                 T = 2k          half speed, even T
  SCALE *2 OFFSET +2    {"scale": 2, "offset": 2}    T = 2k + 2      even T, 1 tick right
  NESTED *2 THEN +1     parent {"scale": 2}          T = 2k + 2      MUST match the row
                        child  {"offset": 1}                         above
  REVERSE *-1 +15       {"scale": -1, "offset": 15}  T = 15 - k      runs backwards

The nested row is a genuine composition-ORDER test, not a decoration: composing
parent-outward gives ``scale 2, offset 2`` (lights on even T), the other order
gives ``scale 2, offset 1`` (odd T). The two are one tick apart on screen, so a
wrong composition order shows up as a row lighting on the wrong parity.

Channel (categorical → permutation). Markers are colour-coded letters carrying
their own LOCAL channel identity — a red "R" is local channel 0 wherever it
ends up::

  IDENTITY            [0, 1, 2]   every letter lights under its own name
  SWAP RED-GREEN      [1, 0, 2]   R lights under GREEN, G lights under RED
  ROTATE              [2, 0, 1]   R lights under BLUE, G under RED, B under GREEN

================================================================================
FEATURES DEMONSTRATED
================================================================================

- ``nd_transform`` affine (offset, scale, negative scale, scale+offset) on a
  discrete ordinal dimension
- ``nd_transform`` permutation on a categorical dimension
- Hierarchical composition of ``nd_transform`` through nested groups
- A spatial 4x4 ``transform`` and an ``nd_transform`` on the same group (every
  row is positioned in Y by a plain translate)
- ``extend_to_all`` for static furniture (rulers, rails, labels, ghosts) and for
  per-section pinning (Frame rows ignore Channel and vice versa)
- Dimension-gated text overlays (``visible_range``) as a live expected-value
  readout
- Composite group layers: the two halves of the bench hang off ``Frame_Section``
  and ``Channel_Section``, the only two nodes marked ``layer=True``. The Layers
  panel therefore shows one row per half — fanning visibility, range, gamma and
  blend down to every ruler, rail, ghost and marker — instead of one row per
  node (66 of them) for what a reader thinks of as two things.

Usage:
    python demo_nd_transforms.py [--no-serve]

Controls:
    - Press '4' to select Frame, then '[' / ']' to step the world frame
    - Press '5' to select Channel, then '[' / ']' to step the world channel
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

DEMO_META = {
    "key": "nd_transforms",
    "title": "nD Transform Test Bench",
    "description": "A ruler, a world cursor and self-labelling markers that make every nD transform readable tick by tick.",
    "category": "synthetic",
    "geometry": "points+lines",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["nd_transforms_bench"],
}

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from arbol import aprint, asection

from luxar import (
    CameraConfig,
    Dimension,
    Dimensions,
    DimensionsConfig,
    LuxarZarrCompiler,
    ViewerConfig,
    transforms,
)
from luxar.demos import launch_viewer, parse_demo_flags
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# A 5x7 point font
# =============================================================================
#
# Markers have to *say* which local index they are, so the bench needs text in
# world space — screen-space overlays would slide off the rows the moment you
# orbit. A 5x7 bitmap drawn one point per lit cell reads as crisp pixel-art from
# any angle and costs a few thousand points for the whole scene.

_GLYPHS: Dict[str, Tuple[str, ...]] = {
    " ": ("     ", "     ", "     ", "     ", "     ", "     ", "     "),
    "0": (" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "),
    "1": ("  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "),
    "2": (" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"),
    "3": ("#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "),
    "4": ("   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "),
    "5": ("#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "),
    "6": ("  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "),
    "7": ("#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "),
    "8": (" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "),
    "9": (" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "),
    "A": (" ### ", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "B": ("#### ", "#   #", "#   #", "#### ", "#   #", "#   #", "#### "),
    "C": (" ### ", "#   #", "#    ", "#    ", "#    ", "#   #", " ### "),
    "D": ("#### ", "#   #", "#   #", "#   #", "#   #", "#   #", "#### "),
    "E": ("#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#####"),
    "F": ("#####", "#    ", "#    ", "#### ", "#    ", "#    ", "#    "),
    "G": (" ### ", "#   #", "#    ", "#  ##", "#   #", "#   #", " ### "),
    "H": ("#   #", "#   #", "#   #", "#####", "#   #", "#   #", "#   #"),
    "I": (" ### ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "),
    "J": ("  ###", "   # ", "   # ", "   # ", "   # ", "#  # ", " ##  "),
    "K": ("#   #", "#  # ", "# #  ", "##   ", "# #  ", "#  # ", "#   #"),
    "L": ("#    ", "#    ", "#    ", "#    ", "#    ", "#    ", "#####"),
    "M": ("#   #", "## ##", "# # #", "#   #", "#   #", "#   #", "#   #"),
    "N": ("#   #", "##  #", "# # #", "#  ##", "#   #", "#   #", "#   #"),
    "O": (" ### ", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    "P": ("#### ", "#   #", "#   #", "#### ", "#    ", "#    ", "#    "),
    "Q": (" ### ", "#   #", "#   #", "#   #", "# # #", "#  # ", " ## #"),
    "R": ("#### ", "#   #", "#   #", "#### ", "# #  ", "#  # ", "#   #"),
    "S": (" ####", "#    ", "#    ", " ### ", "    #", "    #", "#### "),
    "T": ("#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "  #  "),
    "U": ("#   #", "#   #", "#   #", "#   #", "#   #", "#   #", " ### "),
    "V": ("#   #", "#   #", "#   #", "#   #", "#   #", " # # ", "  #  "),
    "W": ("#   #", "#   #", "#   #", "#   #", "# # #", "## ##", "#   #"),
    "X": ("#   #", "#   #", " # # ", "  #  ", " # # ", "#   #", "#   #"),
    "Y": ("#   #", "#   #", " # # ", "  #  ", "  #  ", "  #  ", "  #  "),
    "Z": ("#####", "    #", "   # ", "  #  ", " #   ", "#    ", "#####"),
    "+": ("     ", "  #  ", "  #  ", "#####", "  #  ", "  #  ", "     "),
    "-": ("     ", "     ", "     ", "#####", "     ", "     ", "     "),
    "=": ("     ", "     ", "#####", "     ", "#####", "     ", "     "),
    "*": ("     ", "#   #", " # # ", "  #  ", " # # ", "#   #", "     "),
    "/": ("    #", "    #", "   # ", "  #  ", " #   ", "#    ", "#    "),
    ".": ("     ", "     ", "     ", "     ", "     ", " ##  ", " ##  "),
    ":": ("     ", " ##  ", " ##  ", "     ", " ##  ", " ##  ", "     "),
    "(": ("  ## ", " #   ", "#    ", "#    ", "#    ", " #   ", "  ## "),
    ")": (" ##  ", "   # ", "    #", "    #", "    #", "   # ", " ##  "),
    "<": ("   # ", "  #  ", " #   ", "#    ", " #   ", "  #  ", "   # "),
    ">": (" #   ", "  #  ", "   # ", "    #", "   # ", "  #  ", " #   "),
    "?": (" ### ", "#   #", "    #", "   # ", "  #  ", "     ", "  #  "),
}

_GLYPH_W = 5  # cells per glyph, horizontally
_GLYPH_H = 7  # cells per glyph, vertically
_ADVANCE = 6  # cells from one glyph's origin to the next
_LINE_ADVANCE = 9  # cells between the top rows of stacked lines


def _span_cells(n_chars: int) -> int:
    """Distance in cells between the first and last lit-cell CENTRES.

    Centring uses the span of cell centres, not the nominal advance width, so a
    centred glyph's drawn pixels straddle ``origin`` exactly. That matters here:
    markers must line up with the cursor column to within a fraction of a tick,
    or the whole "count the ticks" reading gets a half-cell bias.
    """
    return max(0, (n_chars - 1) * _ADVANCE + (_GLYPH_W - 1))


def text_points(
    text: str,
    origin: Tuple[float, float],
    cell: float,
    align: str = "left",
) -> np.ndarray:
    """Rasterize ``text`` into an (M, 2) array of XY point positions.

    Args:
        text: String to render. ``\\n`` starts a new line. Characters missing
            from the font are drawn as ``?``.
        origin: (x, y) anchor. Y is the vertical centre of the whole block.
        cell: World size of one font cell.
        align: ``left``, ``center`` or ``right`` — how the block is anchored
            horizontally at ``origin[0]``.

    Returns:
        (M, 2) float32 array, one point per lit cell.
    """
    lines = text.split("\n")
    # Y of the first line's top row, measured up from the block centre.
    top = origin[1] + ((len(lines) - 1) * _LINE_ADVANCE + _GLYPH_H - 1) * cell / 2.0

    out: List[Tuple[float, float]] = []
    for li, line in enumerate(lines):
        span = _span_cells(len(line))
        if align == "center":
            x0 = origin[0] - span * cell / 2.0
        elif align == "right":
            x0 = origin[0] - span * cell
        else:
            x0 = origin[0]
        y0 = top - li * _LINE_ADVANCE * cell

        for ci, char in enumerate(line):
            rows = _GLYPHS.get(char.upper(), _GLYPHS["?"])
            gx = x0 + ci * _ADVANCE * cell
            for r, row in enumerate(rows):
                for c, pixel in enumerate(row):
                    if pixel != " ":
                        out.append((gx + c * cell, y0 - r * cell))

    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


# =============================================================================
# Bench geometry constants
# =============================================================================

N_FRAMES = 16  # world frame indices 0 .. 15
TICK = 5.2  # world X units per frame index
ROW_DY = 7.0  # world Y units between rows

CELL_MARKER = 0.36  # font cell for the lit / ghost index digits
CELL_LABEL = 0.30  # font cell for row labels
CELL_NOTE = 0.24  # font cell for the right-hand annotations
CELL_RULER = 0.32  # font cell for the ruler numerals

R_MARKER = 0.55 * CELL_MARKER  # radius > half a cell → strokes read as solid
R_GHOST = 0.40 * CELL_MARKER
R_CURSOR = 0.13
SHARPNESS = 0.9  # crisp discs; soft blobs would smear the glyph pixels

# Colours are linear-light. Values above 1.0 are HDR and read as "lit".
C_LIT = (1.9, 1.75, 1.30)  # warm white — the marker the viewer chose
C_GHOST = (0.09, 0.10, 0.14)  # barely-there slot markers
C_CURSOR = (0.15, 1.60, 1.75)  # cyan — the world index you asked for
C_LABEL = (0.44, 0.48, 0.56)
C_RULER = (0.58, 0.63, 0.74)  # brighter than the ghosts — this is the reference
C_RAIL = (0.11, 0.12, 0.16)
C_NOTE = (0.52, 0.39, 0.22)

CHANNELS = ["RED", "GREEN", "BLUE"]
CH_LETTER = ["R", "G", "B"]
CH_LIT = [(2.2, 0.25, 0.20), (0.25, 2.0, 0.30), (0.35, 0.65, 2.4)]
CH_TICK = 13.0  # world X units between channel slots


def frame_x(k: float) -> float:
    """World X of frame index ``k`` (the bench is centred on X = 0)."""
    return (k - (N_FRAMES - 1) / 2.0) * TICK


def channel_x(c: float) -> float:
    """World X of channel slot ``c``."""
    return (c - (len(CHANNELS) - 1) / 2.0) * CH_TICK


X_LEFT = frame_x(0) - TICK * 0.6
X_RIGHT = frame_x(N_FRAMES - 1) + TICK * 0.6
LABEL_X = X_LEFT - 1.8  # row labels are right-aligned to end here
NOTE_X = X_RIGHT + 1.8  # right-hand annotations start here


# =============================================================================
# Row definitions
# =============================================================================


class FrameRow:
    """One transformed row of the Frame section.

    ``scale`` / ``offset`` describe the COMPOSED transform (``world = scale *
    local + offset``). They are used only to predict which slots a row can
    light and to print the expected-value readout — the scene itself is driven
    by ``nd_transform`` / ``parent_nd_transform``, so a mismatch between the
    two shows up on screen instead of being hidden.

    Args:
        label: Two-line row caption, drawn right-aligned left of the rail.
        scale: Composed affine scale.
        offset: Composed affine offset.
        nd_transform: The marker group's own ``nd_transform`` (None = identity).
        parent_nd_transform: When set, the markers live in a CHILD group under a
            parent carrying this transform — the hierarchical composition case.
        note: Short annotation drawn to the right of the rail.
    """

    def __init__(
        self,
        label: str,
        *,
        scale: float,
        offset: float,
        nd_transform: Optional[Dict[str, Any]] = None,
        parent_nd_transform: Optional[Dict[str, Any]] = None,
        note: str = "",
    ) -> None:
        self.label = label
        self.scale = scale
        self.offset = offset
        self.nd_transform = nd_transform
        self.parent_nd_transform = parent_nd_transform
        self.note = note

    @property
    def name(self) -> str:
        """First label line, as a zarr-safe node-name fragment.

        Signed numbers have to survive: ``OFFSET +5`` and ``OFFSET -3`` differ
        only in punctuation, so the sign is transliterated (``p``/``m``) rather
        than collapsed to ``_``, which would make the two node names collide.
        The channel rows use the simpler :func:`_channel_node_name` — their
        labels carry no signed numbers.
        """
        first = self.label.split("\n")[0]
        return "".join(
            ch if ch.isalnum() else {"+": "p", "-": "m", "*": "x"}.get(ch, "_")
            for ch in first
        )

    def local_frames(self) -> List[int]:
        """Local indices whose world image is an in-range integer frame."""
        out: List[int] = []
        for k in range(N_FRAMES):
            world = self.scale * k + self.offset
            if abs(world - round(world)) < 1e-9 and 0 <= round(world) <= N_FRAMES - 1:
                out.append(k)
        return out

    def local_for_world(self, world: int) -> Optional[int]:
        """Local index the viewer should show at ``world``, or None if dark."""
        local = (world - self.offset) / self.scale
        if abs(local - round(local)) > 1e-9:
            return None
        local_i = int(round(local))
        return local_i if 0 <= local_i <= N_FRAMES - 1 else None


FRAME_ROWS: List[FrameRow] = [
    FrameRow(
        "IDENTITY\nLOCAL = T",
        scale=1.0,
        offset=0.0,
        nd_transform=None,
        note="NO ND-TRANSFORM",
    ),
    FrameRow(
        "OFFSET +5\nLOCAL = T-5",
        scale=1.0,
        offset=5.0,
        nd_transform={"Frame": {"offset": 5.0}},
        note="5 TICKS LEFT OF CURSOR",
    ),
    FrameRow(
        "OFFSET -3\nLOCAL = T+3",
        scale=1.0,
        offset=-3.0,
        nd_transform={"Frame": {"offset": -3.0}},
        note="3 TICKS RIGHT OF CURSOR",
    ),
    FrameRow(
        "SCALE *2\nLOCAL = T/2",
        scale=2.0,
        offset=0.0,
        nd_transform={"Frame": {"scale": 2.0}},
        note="HALF SPEED - EVEN T ONLY",
    ),
    FrameRow(
        "SCALE *2 OFFSET +2\nLOCAL = (T-2)/2",
        scale=2.0,
        offset=2.0,
        nd_transform={"Frame": {"scale": 2.0, "offset": 2.0}},
        note="EVEN T ONLY",
    ),
    FrameRow(
        "NESTED *2 THEN +1\nLOCAL = (T-2)/2",
        scale=2.0,
        offset=2.0,
        nd_transform={"Frame": {"offset": 1.0}},
        parent_nd_transform={"Frame": {"scale": 2.0}},
        note="MUST MATCH ROW ABOVE",
    ),
    FrameRow(
        "REVERSE *-1 +15\nLOCAL = 15-T",
        scale=-1.0,
        offset=15.0,
        nd_transform={"Frame": {"scale": -1.0, "offset": 15.0}},
        note="RUNS BACKWARDS",
    ),
]


def _channel_node_name(label: str) -> str:
    """First label line as a zarr-safe node-name fragment, for a channel row.

    Deliberately simpler than :attr:`FrameRow.name`: channel labels contain no
    signed numbers, so collapsing every non-alphanumeric to ``_`` cannot make
    two rows collide (the hyphen in ``SWAP RED-GREEN`` is punctuation, not a
    minus sign, so transliterating it would only make the path uglier).
    """
    return "".join(ch if ch.isalnum() else "_" for ch in label.split("\n")[0])


# (label, permutation, note) — permutation[local_index] = world_index.
CHANNEL_ROWS: List[Tuple[str, Optional[List[int]], str]] = [
    ("IDENTITY\nPERM 0 1 2", None, "EACH LETTER UNDER ITS OWN NAME"),
    ("SWAP RED-GREEN\nPERM 1 0 2", [1, 0, 2], "R UNDER GREEN - G UNDER RED"),
    ("ROTATE\nPERM 2 0 1", [2, 0, 1], "R UNDER BLUE - G UNDER RED"),
]


# =============================================================================
# Small builders
# =============================================================================


def _to_5d(
    xy: np.ndarray,
    *,
    frame: float,
    channel: float,
    z: float = 0.0,
) -> np.ndarray:
    """Lift (M, 2) font points into the scene's (X, Y, Z, Frame, Channel)."""
    out = np.zeros((xy.shape[0], 5), dtype=np.float32)
    out[:, 0] = xy[:, 0]
    out[:, 1] = xy[:, 1]
    out[:, 2] = z
    out[:, 3] = frame
    out[:, 4] = channel
    return out


def _tinted(n: int, color: Sequence[float]) -> np.ndarray:
    """(n, 3) float32 array of a single colour."""
    return np.tile(np.asarray(color, dtype=np.float32), (n, 1))


def _column_points(x: float, y_top: float, y_bottom: float, step: float) -> np.ndarray:
    """A dotted vertical column of (M, 2) points — used for the cursors."""
    n = max(2, int(round(abs(y_top - y_bottom) / step)) + 1)
    ys = np.linspace(y_top, y_bottom, n, dtype=np.float32)
    xs = np.full(n, x, dtype=np.float32)
    return np.column_stack([xs, ys])


def _rail_vertices(y: float, x0: float, x1: float) -> np.ndarray:
    """Two (X, Y, Z, Frame, Channel) vertices for one horizontal rail."""
    verts = np.zeros((2, 5), dtype=np.float32)
    verts[0, 0], verts[1, 0] = x0, x1
    verts[:, 1] = y
    return verts


def _add_static_text(
    group: Any,
    name: str,
    text: str,
    origin: Tuple[float, float],
    cell: float,
    color: Sequence[float],
    align: str,
    counter: List[int],
    radius: Optional[float] = None,
) -> None:
    """Add always-visible world-space text (pinned across both nD dims)."""
    xy = text_points(text, origin, cell, align=align)
    if xy.shape[0] == 0:
        return
    pos = _to_5d(xy, frame=0.0, channel=0.0)
    group.add_points(
        name,
        pos,
        colors=_tinted(pos.shape[0], color),
        radii=radius if radius is not None else 0.55 * cell,
        sharpness=SHARPNESS,
        # Both non-displayed dims extended → the viewer skips this node during
        # slicing entirely, so it is drawn once and stays put. This is what
        # makes the rulers, rails, labels and ghosts "furniture".
        extend_to_all=["Frame", "Channel"],
    )
    counter[0] += pos.shape[0]


# =============================================================================
# Scene construction
# =============================================================================


def _add_frame_ruler(section: Any, y: float, counter: List[int]) -> None:
    """World-frame ruler: always-on numerals over a baseline with tick notches."""
    ruler = section.add_group("World_Frame_Ruler", blending_mode="normal")

    baseline_y = y - 1.9
    ruler.add_lines(
        "Ruler_Baseline",
        _rail_vertices(baseline_y, X_LEFT, X_RIGHT),
        widths=0.09,
        colors=_tinted(2, C_RULER),
        line_type="segments",
        extend_to_all=["Frame", "Channel"],
    )
    counter[0] += 2

    numerals = [
        text_points(str(k), (frame_x(k), y), CELL_RULER, align="center")
        for k in range(N_FRAMES)
    ]
    # Short notches hanging below the baseline, so the numerals sit clearly
    # ABOVE a ruler edge instead of looking like another row of ghosts.
    notches = [
        _column_points(frame_x(k), baseline_y - 0.25, baseline_y - 0.95, 0.35)
        for k in range(N_FRAMES)
    ]
    pos = _to_5d(np.vstack(numerals + notches), frame=0.0, channel=0.0)
    ruler.add_points(
        "Ruler_Numerals",
        pos,
        colors=_tinted(pos.shape[0], C_RULER),
        radii=0.55 * CELL_RULER,
        sharpness=SHARPNESS,
        extend_to_all=["Frame", "Channel"],
    )
    counter[0] += pos.shape[0]

    _add_static_text(
        ruler,
        "Ruler_Caption",
        "WORLD FRAME T\nSLIDER VALUE",
        (LABEL_X, y),
        CELL_LABEL,
        C_LABEL,
        "right",
        counter,
    )


def _add_frame_cursor(
    section: Any,
    y_top: float,
    y_bottom: float,
    readout_y: float,
    counter: List[int],
) -> None:
    """The world cursor: one point column per world frame, tagged Frame = k.

    There is no ``nd_transform`` anywhere on this group — it is the
    untransformed reference the rows are measured against. Exactly one column
    matches the slice at a time, so it reads as a cursor sliding along the
    ruler, with the world index spelled out above it.
    """
    cursor = section.add_group("World_Frame_Cursor", blending_mode="normal")

    columns: List[np.ndarray] = []
    digits: List[np.ndarray] = []
    for k in range(N_FRAMES):
        columns.append(
            _to_5d(
                _column_points(frame_x(k), y_top, y_bottom, 0.85),
                frame=k,
                channel=0.0,
            )
        )
        digits.append(
            _to_5d(
                text_points(
                    str(k), (frame_x(k), readout_y), CELL_RULER * 1.6, align="center"
                ),
                frame=k,
                channel=0.0,
            )
        )

    col_pos = np.vstack(columns)
    dig_pos = np.vstack(digits)
    # Pinned across Channel so the frame cursor does not vanish when you step
    # the channel slider — a partial extend, hence Points (Lines cannot do it).
    cursor.add_points(
        "Frame_Cursor_Column",
        col_pos,
        colors=_tinted(col_pos.shape[0], C_CURSOR),
        radii=R_CURSOR,
        sharpness=SHARPNESS,
        extend_to_all=["Channel"],
    )
    cursor.add_points(
        "Frame_Cursor_Readout",
        dig_pos,
        colors=_tinted(dig_pos.shape[0], C_CURSOR),
        radii=0.55 * CELL_RULER * 1.6,
        sharpness=SHARPNESS,
        extend_to_all=["Channel"],
    )
    counter[0] += col_pos.shape[0] + dig_pos.shape[0]


def _add_frame_row(section: Any, row: FrameRow, y: float, counter: List[int]) -> Any:
    """Rail + label + always-on ghosts + the transformed lit markers.

    Returns the group the markers were written into — the node whose
    ``world_nd_transform`` is the composed transform under test.
    """
    name = row.name
    locals_ = row.local_frames()

    # --- static furniture: no nd_transform, always visible -----------------
    furniture = section.add_group(f"Row_{name}_Static", blending_mode="normal")

    furniture.add_lines(
        f"Rail_{name}",
        _rail_vertices(y, X_LEFT, X_RIGHT),
        widths=0.07,
        colors=_tinted(2, C_RAIL),
        line_type="segments",
        extend_to_all=["Frame", "Channel"],
    )
    counter[0] += 2

    ghost_xy = np.vstack(
        [
            text_points(str(k), (frame_x(k), y), CELL_MARKER, align="center")
            for k in locals_
        ]
    )
    ghost_pos = _to_5d(ghost_xy, frame=0.0, channel=0.0)
    furniture.add_points(
        f"Ghosts_{name}",
        ghost_pos,
        colors=_tinted(ghost_pos.shape[0], C_GHOST),
        radii=R_GHOST,
        sharpness=SHARPNESS,
        extend_to_all=["Frame", "Channel"],
    )
    counter[0] += ghost_pos.shape[0]

    _add_static_text(
        furniture,
        f"Label_{name}",
        row.label,
        (LABEL_X, y),
        CELL_LABEL,
        C_LABEL,
        "right",
        counter,
    )
    if row.note:
        _add_static_text(
            furniture,
            f"Note_{name}",
            row.note,
            (NOTE_X, y),
            CELL_NOTE,
            C_NOTE,
            "left",
            counter,
        )

    # --- the transformed markers -------------------------------------------
    # Y placement comes from a plain 4x4 translate on the SAME group that
    # carries the nd_transform: the two transform kinds are independent and
    # compose side by side.
    if row.parent_nd_transform is not None:
        parent = section.add_group(
            f"Row_{name}_Parent",
            transform=transforms.translate(0.0, y, 0.0),
            nd_transform=row.parent_nd_transform,
            blending_mode="normal",
        )
        target = parent.add_group(
            f"Row_{name}_Child",
            nd_transform=row.nd_transform,
            blending_mode="normal",
        )
    else:
        extra: Dict[str, Any] = (
            {} if row.nd_transform is None else {"nd_transform": row.nd_transform}
        )
        target = section.add_group(
            f"Row_{name}_Markers",
            transform=transforms.translate(0.0, y, 0.0),
            blending_mode="normal",
            **extra,
        )

    # Markers are authored at Y = 0 — the group's translate lifts them onto the
    # row. Each marker's Frame coordinate is its own LOCAL index, and the digit
    # it draws is that same index.
    marker_pos = np.vstack(
        [
            _to_5d(
                text_points(str(k), (frame_x(k), 0.0), CELL_MARKER, align="center"),
                frame=k,
                channel=0.0,
            )
            for k in locals_
        ]
    )
    target.add_points(
        f"Markers_{name}",
        marker_pos,
        colors=_tinted(marker_pos.shape[0], C_LIT),
        radii=R_MARKER,
        sharpness=SHARPNESS,
        extend_to_all=["Channel"],
    )
    counter[0] += marker_pos.shape[0]
    return target


def _add_channel_section(section: Any, y0: float, counter: List[int]) -> Dict[str, Any]:
    """Categorical-permutation half of the bench.

    Returns the marker group of each channel row, keyed by node name, so the
    caller can assert the written permutations against their declarations.
    """
    marker_nodes: Dict[str, Any] = {}
    n_rows = len(CHANNEL_ROWS)
    bottom_y = y0 - (n_rows + 0.6) * ROW_DY
    x_lo = channel_x(0) - CH_TICK * 0.5
    x_hi = channel_x(len(CHANNELS) - 1) + CH_TICK * 0.5

    ruler = section.add_group("World_Channel_Ruler", blending_mode="normal")
    for c, cname in enumerate(CHANNELS):
        _add_static_text(
            ruler,
            f"Channel_Name_{cname}",
            cname,
            (channel_x(c), y0),
            CELL_RULER,
            C_RULER,
            "center",
            counter,
        )
    _add_static_text(
        ruler,
        "Channel_Caption",
        "WORLD CHANNEL\nSLIDER VALUE",
        (LABEL_X, y0),
        CELL_LABEL,
        C_LABEL,
        "right",
        counter,
    )

    # Channel cursor — the same idea as the frame cursor, on the other axis.
    cursor = section.add_group("World_Channel_Cursor", blending_mode="normal")
    cur_pos = np.vstack(
        [
            _to_5d(
                _column_points(channel_x(c), y0 - 2.0, bottom_y, 0.85),
                frame=0.0,
                channel=c,
            )
            for c in range(len(CHANNELS))
        ]
    )
    cursor.add_points(
        "Channel_Cursor",
        cur_pos,
        colors=_tinted(cur_pos.shape[0], C_CURSOR),
        radii=R_CURSOR,
        sharpness=SHARPNESS,
        extend_to_all=["Frame"],
    )
    counter[0] += cur_pos.shape[0]

    for i, (label, perm, note) in enumerate(CHANNEL_ROWS):
        y = y0 - (i + 1) * ROW_DY
        name = _channel_node_name(label)

        furniture = section.add_group(f"Chan_{name}_Static", blending_mode="normal")
        furniture.add_lines(
            f"Chan_Rail_{name}",
            _rail_vertices(y, x_lo, x_hi),
            widths=0.07,
            colors=_tinted(2, C_RAIL),
            line_type="segments",
            extend_to_all=["Frame", "Channel"],
        )
        counter[0] += 2

        ghost_xy = np.vstack(
            [
                text_points(
                    CH_LETTER[c], (channel_x(c), y), CELL_MARKER, align="center"
                )
                for c in range(len(CHANNELS))
            ]
        )
        ghost_pos = _to_5d(ghost_xy, frame=0.0, channel=0.0)
        furniture.add_points(
            f"Chan_Ghosts_{name}",
            ghost_pos,
            colors=_tinted(ghost_pos.shape[0], C_GHOST),
            radii=R_GHOST,
            sharpness=SHARPNESS,
            extend_to_all=["Frame", "Channel"],
        )
        counter[0] += ghost_pos.shape[0]

        _add_static_text(
            furniture,
            f"Chan_Label_{name}",
            label,
            (LABEL_X, y),
            CELL_LABEL,
            C_LABEL,
            "right",
            counter,
        )
        _add_static_text(
            furniture,
            f"Chan_Note_{name}",
            note,
            (x_hi + 1.8, y),
            CELL_NOTE,
            C_NOTE,
            "left",
            counter,
        )

        extra: Dict[str, Any] = (
            {} if perm is None else {"nd_transform": {"Channel": {"permutation": perm}}}
        )
        markers = section.add_group(
            f"Chan_{name}_Markers",
            transform=transforms.translate(0.0, y, 0.0),
            blending_mode="normal",
            **extra,
        )
        marker_nodes[name] = markers
        # Each marker keeps its LOCAL identity — letter AND colour. Whichever
        # one lights tells you which local channel the permutation routed to
        # the slot under the cursor.
        for c in range(len(CHANNELS)):
            pos = _to_5d(
                text_points(
                    CH_LETTER[c], (channel_x(c), 0.0), CELL_MARKER, align="center"
                ),
                frame=0.0,
                channel=c,
            )
            markers.add_points(
                f"Chan_Marker_{name}_{CH_LETTER[c]}",
                pos,
                colors=_tinted(pos.shape[0], CH_LIT[c]),
                radii=R_MARKER,
                sharpness=SHARPNESS,
                extend_to_all=["Frame"],
            )
            counter[0] += pos.shape[0]

    return marker_nodes


_PANEL_STYLE = (
    "color:rgba(190,204,224,0.9);background:rgba(6,8,14,0.78);"
    "border:1px solid rgba(120,140,170,0.22);border-radius:5px;padding:0.9vh 1.2vh"
)

_PRE_STYLE = (
    "margin:0;font:500 1.55vh/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;"
    "color:rgba(214,228,242,0.94);background:rgba(6,8,14,0.78);"
    "border:1px solid rgba(120,140,170,0.22);border-radius:5px;padding:0.9vh 1.2vh"
)


def _readout_html(title: str, rows: List[Tuple[str, str]]) -> str:
    """A monospace expected-value block.

    Uses ``add_html`` with a ``<pre>`` rather than ``add_text``: text overlays
    are rendered with ``white-space: nowrap``, so a ``\\n`` in an ``add_text``
    string collapses onto one line and the column alignment is lost.
    """
    body = "\n".join(f"  {name:<21}{value}" for name, value in rows)
    return f'<pre style="{_PRE_STYLE}">{title}\n\n{body}</pre>'


def _add_overlays(scene: Any) -> None:
    """Title, legend, and the per-slice EXPECTED-value readouts.

    The readouts are computed here from the same transform definitions that
    placed the data, and gated to a single slider value with ``visible_range``.
    Comparing them against the render is the whole point of the bench.
    """
    # One panel, top-right, carrying the title AND the reading instructions.
    # The bench fills the middle of the frame and its row labels + right-hand
    # notes reach far to both sides, so a bare floating title/legend collides
    # with world-space text; a bordered panel matching the readouts below reads
    # cleanly wherever it lands.
    scene.add_html(
        f'<div style="{_PANEL_STYLE};font:500 1.5vh/1.7 ui-monospace,'
        'SFMono-Regular,Menlo,monospace;text-align:right">'
        '<div style="font-size:2.1vh;letter-spacing:0.06em;'
        'color:rgba(238,243,255,0.92);padding-bottom:0.5vh">nD TRANSFORM TEST BENCH</div>'
        "Cyan column = the WORLD index on the slider.<br>"
        "Bright digit = the LOCAL index the viewer resolved to.<br>"
        "The gap between them, in ruler ticks, IS the nd_transform.<br>"
        "Faint grey = every slot the row could light.<br>"
        '<div style="padding-top:0.6vh">'
        "Press <b>4</b> then <b>[</b> / <b>]</b> to step Frame<br>"
        "Press <b>5</b> then <b>[</b> / <b>]</b> to step Channel"
        "</div>"
        "</div>",
        position=(0.985, 0.025),
        anchor="top-right",
        # Without an explicit width the block wraps to a narrow ragged column.
        width=0.31,
    )

    for world in range(N_FRAMES):
        rows = []
        for row in FRAME_ROWS:
            local = row.local_for_world(world)
            rows.append(
                (
                    row.label.split("\n")[0],
                    f"local {local}" if local is not None else "(nothing)",
                )
            )
        scene.add_html(
            _readout_html(f"EXPECTED AT WORLD FRAME T = {world}", rows),
            name=f"expected_frame_{world}",
            position=(0.015, 0.975),
            anchor="bottom-left",
            visible_range={"Frame": float(world)},
        )

    for c, cname in enumerate(CHANNELS):
        rows = []
        for label, perm, _note in CHANNEL_ROWS:
            local = c if perm is None else perm.index(c)
            rows.append(
                (
                    label.split("\n")[0],
                    f"local {local} = {CH_LETTER[local]} ({CHANNELS[local]})",
                )
            )
        scene.add_html(
            _readout_html(f"EXPECTED AT WORLD CHANNEL = {cname}", rows),
            name=f"expected_channel_{c}",
            position=(0.985, 0.975),
            anchor="bottom-right",
            visible_range={"Channel": float(c)},
        )


class Layout:
    """Analytic extents of the whole bench, used to frame the camera.

    The bench must be legible the instant it opens — the viewer's automatic
    framing has no idea the labels and notes are part of the instrument, so the
    extents are derived here from the same constants and strings that place the
    geometry, and the camera distance is solved from them.
    """

    def __init__(self) -> None:
        n_frame_rows = len(FRAME_ROWS)
        self.ruler_y = ROW_DY * 1.15
        self.readout_y = self.ruler_y + 3.8
        self.last_row_y = -(n_frame_rows - 1) * ROW_DY
        self.channel_y0 = self.last_row_y - ROW_DY * 1.7
        self.channel_bottom = self.channel_y0 - (len(CHANNEL_ROWS) + 0.6) * ROW_DY

        label_texts = [r.label for r in FRAME_ROWS] + [
            lbl for lbl, _, _ in CHANNEL_ROWS
        ]
        label_texts += ["WORLD FRAME T\nSLIDER VALUE", "WORLD CHANNEL\nSLIDER VALUE"]
        widest_label = max(
            _span_cells(len(line)) for text in label_texts for line in text.split("\n")
        )
        chan_note_x = channel_x(len(CHANNELS) - 1) + CH_TICK * 0.5 + 1.8
        widest_note = max(
            [
                NOTE_X + _span_cells(len(r.note)) * CELL_NOTE
                for r in FRAME_ROWS
                if r.note
            ]
            + [
                chan_note_x + _span_cells(len(n)) * CELL_NOTE
                for _, _, n in CHANNEL_ROWS
            ]
        )

        self.x_min = LABEL_X - widest_label * CELL_LABEL
        self.x_max = max(widest_note, X_RIGHT)
        self.y_max = self.readout_y + (_GLYPH_H - 1) / 2 * CELL_RULER * 1.6
        self.y_min = self.channel_bottom

    def camera(self, fov_deg: float = 28.0, min_aspect: float = 1.35) -> CameraConfig:
        """A long-lens camera that frames the whole bench with a small margin.

        ``min_aspect`` is the narrowest viewport the framing must survive; the
        solved distance satisfies both the vertical and horizontal fits there,
        so a wider window only adds margin.

        Horizontal strips at the top and bottom of the frame are kept clear of
        world-space geometry, because the screen-space overlays live there: the
        legend panel at the top, and the expected-value readouts plus the
        viewer's transient dimension-navigation panel at the bottom. The bench
        is fitted into the band between them, so an overlay never lands on a
        row's marker or its note.
        """
        height = self.y_max - self.y_min
        width = self.x_max - self.x_min
        top_strip = 0.16
        bottom_strip = 0.20
        band = 1.0 - top_strip - bottom_strip

        cx = (self.x_min + self.x_max) / 2.0
        half_tan = np.tan(np.radians(fov_deg) / 2.0)
        # Fit the content height into the free band, not the whole frame.
        d_vertical = height / band / 2.0 / half_tan
        d_horizontal = width / 2.0 / (half_tan * min_aspect)
        distance = float(max(d_vertical, d_horizontal) * 1.04)

        # Centre the content on the band's midpoint rather than the frame's.
        # A frame spans 2·distance·tan(fov/2) vertically, so the band centre
        # sits this far below the frame centre:
        frame_half_height = distance * half_tan
        band_offset = frame_half_height * (top_strip - bottom_strip)
        cy = (self.y_min + self.y_max) / 2.0 + band_offset
        return CameraConfig(
            position=(cx, cy, distance),
            target=(cx, cy, 0.0),
            fov=fov_deg,
        )


def generate_demo(output_path: Path) -> int:
    """Write the bench scene. Returns the number of elements written."""
    layout = Layout()
    ruler_y = layout.ruler_y
    readout_y = layout.readout_y
    last_row_y = layout.last_row_y
    channel_y0 = layout.channel_y0

    dims = Dimensions(
        [
            Dimension(name="X", unit="tick", display=True),
            Dimension(name="Y", unit="tick", display=True),
            Dimension(name="Z", unit="tick", display=True),
            Dimension(
                name="Frame",
                unit="",
                range=(0, N_FRAMES - 1),
                step=1.0,
                display=False,
                discrete=True,
                description="World frame index — the reference the rows are measured against",
            ),
            Dimension(
                name="Channel",
                unit="",
                categories=CHANNELS,
                display=False,
                description="World channel slot — the target of the permutations",
            ),
        ]
    )

    counter = [0]

    with asection("Writing nD Transform Test Bench"):
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    background_color="#07080c",
                    # Neutral rather than the ACES default: the bench encodes
                    # meaning in exact colours (a red R must stay red) and ACES
                    # deliberately shifts hues. Bloom stays off so the glyph
                    # pixels stay crisp and countable.
                    tone_mapping="Neutral",
                    bloom_enabled=False,
                    auto_rotate=False,
                    # A long lens is nearly orthographic, so ticks line up
                    # honestly instead of fanning out with perspective.
                    camera=layout.camera(),
                    # Open on T = 7: past the +5 row's start, odd (so both *2
                    # rows are dark), and not a symmetric special case.
                    dimensions=DimensionsConfig(
                        current_step=[0.0, 0.0, 0.0, 7.0, 0.0],
                        selected_dimension=3,
                    ),
                ),
            )

            # Two composite layers, not twenty. Every ruler, cursor, rail,
            # ghost, label and marker of a half-bench hangs off one
            # `layer=True` group, so the Layers panel offers exactly the two
            # rows a reader of this bench thinks in — "the Frame half" and
            # "the Channel half" — and fans visibility / range / gamma / blend
            # down to every descendant. Both wrappers are deliberately bare:
            # no `transform` and no `nd_transform`, so the composed
            # `world_nd_transform` self-check below is unaffected by them.
            frame_section = scene.add_group(
                "Frame_Section", layer=True, blending_mode="normal"
            )
            channel_section = scene.add_group(
                "Channel_Section", layer=True, blending_mode="normal"
            )

            with asection("World frame ruler + cursor"):
                _add_frame_ruler(frame_section, ruler_y, counter)
                _add_frame_cursor(
                    frame_section,
                    ruler_y - 2.8,
                    last_row_y - ROW_DY * 0.6,
                    readout_y,
                    counter,
                )

            marker_nodes: Dict[str, Any] = {}
            with asection("Frame rows"):
                for i, row in enumerate(FRAME_ROWS):
                    marker_nodes[row.name] = _add_frame_row(
                        frame_section, row, -i * ROW_DY, counter
                    )
                    lights = [
                        int(row.scale * k + row.offset) for k in row.local_frames()
                    ]
                    aprint(
                        f"{row.label.split(chr(10))[0]:<20} "
                        f"world = {row.scale:g}*local + {row.offset:g}   "
                        f"lights at T = {lights}"
                    )

            with asection("Channel rows"):
                channel_marker_nodes = _add_channel_section(
                    channel_section, channel_y0, counter
                )

            _add_overlays(scene)

            # Self-check: every row's node must actually carry the composed
            # transform this file claims for it. In particular the nested row
            # has to compose parent-outward to the SAME affine as the flat row
            # above it — that is the Python ground truth the on-screen
            # comparison is measured against.
            with asection("Composed world nd_transforms (Python side)"):
                for row in FRAME_ROWS:
                    composed = marker_nodes[row.name].world_nd_transform
                    entry = composed.get("Frame", {})
                    got = (entry.get("scale", 1.0), entry.get("offset", 0.0))
                    aprint(f"{row.label.split(chr(10))[0]:<20} {composed}")
                    if got != (row.scale, row.offset):
                        raise AssertionError(
                            f"{row.name}: composed Frame transform {got} != "
                            f"declared ({row.scale}, {row.offset})"
                        )
                aprint("✓ nested *2-then-+1 composes to the flat *2 +2 row")

                # The channel rows' permutations must also survive the write —
                # the readout's expected values are derived from CHANNEL_ROWS,
                # so a mismatch here would make the bench lie in the one place
                # it is supposed to be authoritative.
                for label, perm, _note in CHANNEL_ROWS:
                    node_name = _channel_node_name(label)
                    composed = channel_marker_nodes[node_name].world_nd_transform
                    got = composed.get("Channel", {}).get("permutation")
                    expected = perm  # None = identity, written as no entry
                    if got != expected:
                        raise AssertionError(
                            f"{node_name}: composed Channel permutation {got} "
                            f"!= declared {expected}"
                        )
                aprint("✓ channel permutations match their declarations")

        aprint(f"Total elements: {counter[0]:,}")
        aprint(f"Written to {output_path}")

    return counter[0]


def main() -> None:
    """Main demo entry point."""
    flags = parse_demo_flags()

    aprint("=" * 72)
    aprint("ND TRANSFORM TEST BENCH")
    aprint("=" * 72)
    aprint("")
    aprint("Every row carries one nd_transform and prints the LOCAL frame index")
    aprint("the viewer resolved to. The cyan cursor marks the WORLD index. The")
    aprint("gap between them, in ruler ticks, is the transform.")
    aprint("")

    output_path = get_demos_output_dir() / "nd_transforms_bench.luxar.zarr"
    total = generate_demo(output_path)

    aprint("")
    aprint(f"Generated {total:,} elements")
    aprint("")
    aprint("What to check in the viewer:")
    aprint("  1. Press '4' to select Frame, then step with '[' and ']'.")
    aprint("  2. IDENTITY's digit always sits under the cyan cursor and equals T.")
    aprint("  3. OFFSET +5 trails by exactly 5 ticks; OFFSET -3 leads by 3.")
    aprint("  4. Both *2 rows light only on even T, one tick apart.")
    aprint("  5. NESTED *2 THEN +1 lands on the SAME tick as SCALE *2 OFFSET +2.")
    aprint("  6. REVERSE walks right-to-left as T increases.")
    aprint("  7. Press '5' for Channel: the lit letter's colour is its LOCAL")
    aprint("     channel; its position is that channel's own slot.")
    aprint("  8. The bottom-left readout is the EXPECTED answer for this T.")
    aprint("")

    if flags["no_serve"]:
        aprint(f"✓ Dataset generated at {output_path}")
        return

    launch_viewer(output_path)


if __name__ == "__main__":
    main()
