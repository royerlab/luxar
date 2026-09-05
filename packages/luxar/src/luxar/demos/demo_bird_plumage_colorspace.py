"""
Bird Plumage Colour Space — the UV axis a human figure has to throw away
========================================================================

Every line in this scene is one spectrophotometer reading of one patch of
plumage on one bird: a reflectance curve from 300 to 700 nm, measured off a
museum specimen or a live bird, from `BirdColorBase
<https://github.com/BirdColorBase/home>`_: 360,432 readings over 2,632 species,
contributed by ten research groups.

The three DISPLAYED axes are human colour vision. Each reading is integrated
through the CIE 1931 observer and plotted in CIELAB, drawn as a spike from the
neutral grey axis out to its own colour — direction is hue, length is chroma,
height is lightness — and painted with the colour it actually is. That much is
a colour-space hedgehog anyone can draw.

The point of the demo is the axis that plot cannot hold. Birds are
**tetrachromats**: they carry a fourth, ultraviolet-sensitive cone, and the
measurements run down to 300 nm precisely because a plumage patch's UV is part
of its colour to the bird looking at it. A human figure has to integrate that
away — the CIE observer is blind below ~380 nm — so two patches that land on
exactly the same CIELAB spike can be as different in UV as red is from green.

So UV becomes the scene's fourth dimension. Press ``1`` then ``[`` / ``]`` (or
drag the slider) to walk the **UV chroma** axis, the standard avian-colour
metric R300–400 / R300–700: the fraction of a patch's total reflectance that
falls in the ultraviolet. Ten deciles, so each stop holds a tenth of the
corpus. The dim ghost of the whole dataset stays behind every stop, so what
lights up is "which parts of human colour space have THIS much UV" — and the
answer is not the part you would guess.

Colour science, and what is approximated
----------------------------------------

* **Observer** — CIE 1931 2°, via the multi-lobe Gaussian fit of Wyman, Sloan &
  Shirley (2013): within ~1% of the tabulated curves and needs no data file
  (the same fit ``demo_galaxy_simulation`` uses for blackbody colour).
* **Illuminant** — a 6500 K Planckian stand-in for daylight, white-balanced so
  a spectrally flat reflector renders exactly neutral. Not literally D65 (which
  is not a Planckian), but the difference is far below the between-specimen
  spread, and the illuminant does not enter the UV axis at all — UV chroma is a
  ratio of reflectances.
* **Range** — the readings stop at 700 nm, where the observer's x̄ is already
  down to ~1% of peak, so the truncated deep red is negligible.
* **Gamut** — a saturated plumage colour can fall outside sRGB; those are
  clipped into gamut, which desaturates the most extreme spikes slightly.
  Colours are stored LINEAR-light, which is what Luxar's zarr holds.

Usage:
    python -m luxar.demos.demo_bird_plumage_colorspace
    python -m luxar.demos.demo_bird_plumage_colorspace --no-serve
    python -m luxar.demos.demo_bird_plumage_colorspace --recompute

Dependencies:
    pip install luxar[demos]   # nothing beyond it: the .xlsx files are read
                               # with the standard library
"""

DEMO_META = {
    "key": "bird_plumage_colorspace",
    "title": "Bird Plumage Colour Space",
    "description": (
        "360k plumage reflectance spectra from 2,632 bird species in CIELAB, "
        "with the ultraviolet axis human colour vision drops as the 4th."
    ),
    "category": "embeddings",
    "geometry": "lines",
    "requirements": {
        "download_mb": 525,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["birdcolorbase"],
    "outputs": ["bird_plumage_colorspace"],
    "citation": {
        "short": "BirdColorBase (Gluckman & Endler, and ten contributing groups)",
        "ref": "BirdColorBase",
        "license": "MIT",
        "url": "https://github.com/BirdColorBase/home",
    },
}

import io
import math
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterator, Optional
from xml.etree import ElementTree as ET

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import (
    add_demo_caption,
    cache_computed,
    cached_download,
    launch_viewer,
    parse_demo_flags,
)
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.demos._lod_policy import hidden_axis_stops, stream_ladder
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Data source
# =============================================================================

#: Pinned to a commit, not to ``main``: the database is edited in place (the
#: Gomez file was revised 9 Apr 2025 and gained four columns), and a cached
#: download keyed on the URL would otherwise mean two machines silently holding
#: two different corpora under the same cache name.
BIRDCOLORBASE_COMMIT = "b655031d9e770ee2bf5a301730c837cba966c44e"
BIRDCOLORBASE_RAW = (
    f"https://raw.githubusercontent.com/BirdColorBase/home/{BIRDCOLORBASE_COMMIT}"
    "/Data_from_each_Research_Group"
)

CACHE_NAME = "birdcolorbase"

#: Bumped whenever the parser changes which ROWS survive, independently of the
#: colour parameters already in the cache key. v2 places cells by column
#: reference instead of by order (see :func:`_iter_rows`), which recovers the
#: 40% of readings a positional reader silently dropped. v3 fixes reflectance
#: scale detection for files spanning more than one reduction block.
_PARSER_VERSION = 3

#: Repository path -> sha256. Some contributing groups split their data across
#: multiple files to keep each workbook openable. Checksums are verified on
#: download, so a truncated 83 MB transfer is caught rather than silently parsed
#: into a short corpus.
SOURCE_FILES: dict[str, str] = {
    "Burns.Schultz/BurnsSchultz_Bananaquit.Buntings.Sparrows_27Dec24.xlsx": (
        "f803bd0573be86b46a66d2b424dd7e193566fb32890a49d084732732936ec91e"
    ),
    "Burns.Schultz/BurnsSchultz_Grosbeaks.Tanagers_27Dec24.xlsx": (
        "87b887d577f1d4a3c24a92a69307ee110b00b5c0198b7531b272e29c7b50749a"
    ),
    "Cardoso.Gomes.Mota_27Dec24.xlsx": (
        "b68fbcf01e48e2bfb0a40b3e931289bafcb4e5e3f56ae5eead54d79857d745b7"
    ),
    "Dale_27Dec24.xlsx": (
        "0c10dcf1d8fc4e4b5a55f7d3acd7217601bc583fde2a8396c6b0889952537d33"
    ),
    "Doutrelant.Fargeveille_27Dec24.xlsx": (
        "bcd5668d933ddcd978c8c85cfca14f5b99ef486cd57381bb7e7aaca337c2f884"
    ),
    "Dunn.Armenta/DunnArmenta_Nonpasserines_27Dec24.zip": (
        "2cc82c824a0580032dff779f96d61dbe8f1319a4d9767e431c7e7b92f7406c11"
    ),
    "Dunn.Armenta/DunnArmenta_SppAtoD_Passeriformes_27Dec24.xlsx": (
        "adf68fb6aa45d0767db547308656b3a445bd723b4ef7259c925f95b4b3969696"
    ),
    "Dunn.Armenta/DunnArmenta_SppEtoO_Passeriformes_27Dec24.xlsx": (
        "82929dae4835a5b74f1b88e6e455c916f625fa33cdaed2b6c13c96e12284fcf4"
    ),
    "Dunn.Armenta/DunnArmenta_SppPtoZ_Passeriformes_27Dec24.xlsx": (
        "13b5da05a448218ec246531c7e1c44fc1107a53b4bd26a4e140f005c30c01aaf"
    ),
    "Dunning.Endler_27Dec24.xlsx": (
        "c96df4714898fadadff28f89fcf979a52b1a42b468c17b9c95fea2187eb03901"
    ),
    "Eaton_27Dec24.xlsx": (
        "8eb0c7e8feb590b19185415518db5a7d13c3331a788421e69406de2d66b0a09e"
    ),
    "Gomez_31Dec24_rev9Apr25.xlsx": (
        "bf8366f0398a5925a99e4790e2bcca8ab3ed208846997ddca61fa4d100858c8c"
    ),
    "Maia_27Dec24.xlsx": (
        "6e44f6990017cbbbf74af837884681cae7f236d6a116d6097a4d4afdd9e86495"
    ),
    "StoddardPrum_31Dec24.xlsx": (
        "a35e46f4a7fbcb90299919d986b251eca5b849ec9ed8da690b3f48a4c7ec9d01"
    ),
}

#: Metadata this demo reads, as ``field -> header names to try in order``.
#: Resolved BY NAME rather than by position: the layouts agree on their column
#: NAMES but not on their offsets — the revised Gomez file carries four extra
#: columns and renamed ``Patch``, so a fixed index would read a museum code as a
#: plumage patch for 68,000 rows without erroring.
META_COLUMNS: dict[str, tuple[str, ...]] = {
    "english": ("English_BirdTree_Jetz",),
    "species": ("JetzSpecies", "OriginalSpecies"),
    "order": ("Order3",),
    "family": ("Family3",),
    "sex": ("Sex",),
    "patch": ("Patch", "Patch recoded 9Apr25", "Patch (original)"),
}

#: The measured band, in nm. Every file bins reflectance identically: 201
#: columns headed 300, 302, ... 700.
WAVELENGTHS = np.arange(300.0, 701.0, 2.0)

#: Upper edge of the ultraviolet band for the UV-chroma ratio. 400 nm is the
#: conventional cut in the avian-plumage literature (Andersson & Prager 2006).
UV_CUTOFF_NM = 400.0

# =============================================================================
# Configuration
# =============================================================================

#: Deciles of UV chroma. Ten equal-population stops: the slider then never lands
#: on a nearly-empty slice, and each stop is a tenth of the corpus, which is a
#: legible share of the frame. Quantile edges rather than fixed widths because
#: UV chroma is strongly right-skewed — fixed widths put three quarters of the
#: readings in the first two bins.
N_UV_BINS = 10

#: A reading is dropped when its spectrum is missing, all-zero, or its
#: integrated lightness is below this. Sub-1% L* readings are instrument floor,
#: not plumage, and they pile up on the neutral axis as a black hairball.
MIN_LIGHTNESS = 1.0

#: Scene units are CIELAB units directly, and the scene is centred on mid-grey
#: (L* = 50) so the spray radiates in every direction rather than out of one
#: corner. A* and b* are used unshifted.
LAB_CENTRE_LIGHTNESS = 50.0

#: Spike width in CIELAB units. Chroma is strongly right-skewed — median 8.4,
#: p99 63.8, max 117.9 — so most spikes are short and the long ones are the
#: story. Widths are per-vertex and taper from the neutral end to the tip, which
#: is what stops the dense achromatic core from reading as one solid ball.
WIDTH_AT_NEUTRAL = 0.05
WIDTH_AT_TIP = 0.32

#: Exposure. Additive blending sums every spike along the ray and half the
#: corpus sits within chroma 8.4 of the neutral axis, so the core is where tens
#: of thousands of them overlap. Authored well below 1 for that reason: at 0.55
#: the core rendered as one flat white wedge several times the size of the
#: measured cloud, swallowing every short spike in it. Judge a change to these
#: on whether the SHORT spikes near the centre keep their hue, not on how bright
#: the frame is — that is the failure mode. ACES (the demo default) takes what
#: rolloff is left.
SPIKE_OPACITY = 0.33
SPIKE_INTENSITY = 0.40

#: The always-visible ghost of the whole corpus, NEUTRAL rather than in its own
#: colours. Two reasons, both learned by rendering it the other way: in colour
#: it is indistinguishable from the selected decile, so the highlight stops
#: reading as a highlight; and a grey silhouette is what a context layer is for
#: — it shows the shape the corpus fills without competing for a hue. A dim
#: blue-grey rather than pure grey so it never reads as a desaturated bird.
GHOST_COLOR = (0.16, 0.18, 0.23)
GHOST_OPACITY = 0.22
GHOST_INTENSITY = 0.20
GHOST_WIDTH = 0.035

#: Opening framing. The subject is FRAMED AS A SPHERE (``d = R / sin(fov/2)``,
#: the rule ``_cinematic_camera`` states), for two reasons: the spray is roughly
#: isotropic about the neutral axis, so a sphere is a fair bound rather than a
#: wasteful one; and a sphere is rotation-invariant, which is what lets the
#: opening pose sit off-axis. That matters here — looking straight down L* the
#: hedgehog reads as a flat rainbow star and the third axis is invisible until
#: you drag.
#:
#: The radius comes from a QUANTILE of chroma, not from the bounding box. The
#: box is set by a handful of near-monochromatic patches at chroma 118 against a
#: median of 8.4, so framing it puts the whole corpus in the middle fifth of the
#: frame. At p99.5 about 1,800 spikes overshoot the edge, which reads as
#: intended rather than as clipping.
CAMERA_CHROMA_QUANTILE = 0.995
CAMERA_FILL = 0.8
#: Direction from the subject to the camera, normalised at use. A gentle
#: three-quarter view from above right: enough parallax to show that lightness
#: is a real axis, not so much that the hue circle stops reading as a circle.
CAMERA_DIRECTION = (0.45, 0.32, 1.0)


# =============================================================================
# xlsx reading (standard library)
# =============================================================================

_SHEET_XML = re.compile(r"xl/worksheets/sheet\d+\.xml$")
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _open_workbook(path: Path) -> zipfile.ZipFile:
    """Open ``path`` as a zipped xlsx, unwrapping the one zipped-xlsx file.

    ``DunnArmenta_Nonpasserines`` ships as a .zip around its .xlsx (an .xlsx is
    itself a zip, so this is a zip in a zip). Read in memory rather than
    unpacked to disk: the member is 260 MB uncompressed and is streamed once.
    """
    if path.suffix.lower() == ".zip":
        outer = zipfile.ZipFile(path)
        members = [n for n in outer.namelist() if n.lower().endswith(".xlsx")]
        if not members:
            raise ValueError(f"{path.name}: zip holds no .xlsx member")
        return zipfile.ZipFile(io.BytesIO(outer.read(members[0])))
    return zipfile.ZipFile(path)


def _shared_strings(book: zipfile.ZipFile) -> list[str]:
    """The workbook's shared-string table (every text cell indexes into it)."""
    try:
        raw = book.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    return [
        "".join(t.text or "" for t in si.iter(_NS + "t")) for si in ET.fromstring(raw)
    ]


_DIGITS = "0123456789"
#: Spreadsheet column letters -> 0-based index, memoised (only ~230 occur).
_COLUMN_INDEX: dict[str, int] = {}


def _column_index(letters: str) -> int:
    """``"A"`` -> 0, ``"Z"`` -> 25, ``"AA"`` -> 26 …"""
    cached = _COLUMN_INDEX.get(letters)
    if cached is None:
        cached = 0
        for character in letters:
            cached = cached * 26 + (ord(character) - 64)
        cached -= 1
        _COLUMN_INDEX[letters] = cached
    return cached


def _iter_rows(book: zipfile.ZipFile, width: int = 0) -> Iterator[list[Optional[str]]]:
    """Yield each worksheet row as a fixed-width list of raw cell strings.

    Streamed with ``iterparse`` and cleared per row: the largest sheet expands
    to 590 MB of XML, which neither fits comfortably in a DOM nor needs to.

    **Cells are placed by their column reference, never by order.** xlsx omits a
    genuinely empty cell entirely, and these files do: the header rows carry all
    225 columns but the DATA rows carry 224, because one metadata cell is blank
    throughout. Reading positionally therefore shifts every value one column
    left — which, taken as "the row is the wrong length, skip it", silently drops
    40% of the corpus and 100% of the Maia and Stoddard-Prum files, and taken as
    "close enough" would read reflectance at 302 nm as reflectance at 300 nm.

    Args:
        book: The opened workbook.
        width: Row width to pad to; 0 means "take it from the first row",
            which is the header and is the only row guaranteed to be complete.
    """
    strings = _shared_strings(book)
    sheets = sorted(n for n in book.namelist() if _SHEET_XML.match(n))
    if len(sheets) != 1:
        raise ValueError(f"expected exactly one worksheet, found {len(sheets)}")
    with book.open(sheets[0]) as handle:
        for _event, element in ET.iterparse(handle, events=("end",)):
            if element.tag != _NS + "row":
                continue
            cells: list[tuple[int, Optional[str]]] = []
            for position, cell in enumerate(element):
                reference = cell.get("r")
                column = (
                    _column_index(reference.rstrip(_DIGITS)) if reference else position
                )
                value = cell.find(_NS + "v")
                if value is None or value.text is None:
                    text = None
                elif cell.get("t") == "s":
                    index = int(value.text)
                    text = strings[index] if index < len(strings) else None
                else:
                    text = value.text
                cells.append((column, text))
            element.clear()

            if not width:
                width = max((column for column, _ in cells), default=-1) + 1
            row: list[Optional[str]] = [None] * width
            for column, text in cells:
                if 0 <= column < width:
                    row[column] = text
            yield row


def _resolve_columns(header: list[Optional[str]]) -> tuple[dict[str, int], np.ndarray]:
    """Locate the metadata and the 201 spectral columns in one file's header.

    Returns ``(meta_index, spectral_index)``. Raises when the wavelength grid is
    not the documented 300–700 nm / 2 nm one, because a file binned differently
    would integrate against the wrong colour-matching weights and produce a
    plausible wrong colour rather than an error.
    """
    lookup = {(name or "").strip(): position for position, name in enumerate(header)}
    meta_index: dict[str, int] = {}
    for field, candidates in META_COLUMNS.items():
        for candidate in candidates:
            if candidate in lookup:
                meta_index[field] = lookup[candidate]
                break

    spectral_index = np.array(
        [lookup[f"{int(nm)}"] for nm in WAVELENGTHS if f"{int(nm)}" in lookup],
        dtype=np.int64,
    )
    if len(spectral_index) != len(WAVELENGTHS):
        raise ValueError(
            f"expected the {len(WAVELENGTHS)} documented 300-700 nm / 2 nm "
            f"columns, found {len(spectral_index)}"
        )
    return meta_index, spectral_index


# =============================================================================
# Colorimetry
# =============================================================================


def _cie_lobe(x: np.ndarray, mu: float, s1: float, s2: float) -> np.ndarray:
    """Asymmetric ("piecewise") Gaussian lobe used by the colour-matching fit."""
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def _observer(lam: np.ndarray) -> np.ndarray:
    """CIE 1931 2° colour-matching functions at ``lam``, as an ``(n, 3)`` array.

    The multi-lobe Gaussian fit of Wyman, Sloan & Shirley (2013) — within about
    1% of the tabulated curves and needing no data file. Below ~380 nm all three
    lobes are already negligible, which IS the point being made by the demo: the
    ultraviolet third of every measured spectrum contributes nothing here.
    """
    x_bar = (
        1.056 * _cie_lobe(lam, 599.8, 37.9, 31.0)
        + 0.362 * _cie_lobe(lam, 442.0, 16.0, 26.7)
        - 0.065 * _cie_lobe(lam, 501.1, 20.4, 26.2)
    )
    y_bar = 0.821 * _cie_lobe(lam, 568.8, 46.9, 40.5) + 0.286 * _cie_lobe(
        lam, 530.9, 16.3, 31.1
    )
    z_bar = 1.217 * _cie_lobe(lam, 437.0, 11.8, 36.0) + 0.681 * _cie_lobe(
        lam, 459.0, 26.0, 13.8
    )
    return np.stack([x_bar, y_bar, z_bar], axis=1)


def _illuminant(lam: np.ndarray, temperature_k: float = 6500.0) -> np.ndarray:
    """Planckian spectral radiance at ``lam`` nm — the daylight stand-in."""
    lam_m = lam * 1e-9
    h, c_light, k_b = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    return (2 * h * c_light**2 / lam_m**5) / np.expm1(
        h * c_light / (lam_m * k_b * temperature_k)
    )


#: XYZ -> LINEAR sRGB (sRGB/Rec.709 primaries, D65 white).
_XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ]
)


class _ColourModel:
    """The observer/illuminant weights, built once and reused per block."""

    def __init__(self) -> None:
        weights = _illuminant(WAVELENGTHS)
        observer = _observer(WAVELENGTHS)
        # Normalise so a perfect (R == 1) reflector has Y == 1, which makes the
        # white point Yn == 1 and L* land on its usual 0-100 scale.
        self.weighted = observer * weights[:, None]
        self.weighted /= self.weighted[:, 1].sum()
        self.white = self.weighted.sum(axis=0)
        # White-balance the primaries so a spectrally flat reflector renders
        # exactly neutral under this illuminant. Without it the 6500 K
        # Planckian's small departure from D65 tints every colour in the scene.
        self.rgb_gain = 1.0 / (self.white @ _XYZ_TO_LINEAR_SRGB.T)
        self.uv_band = WAVELENGTHS < UV_CUTOFF_NM

    def evaluate(
        self, reflectance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(lab, linear_rgb, uv_chroma)`` for an ``(n, 201)`` reflectance block."""
        xyz = reflectance @ self.weighted
        lab = _xyz_to_lab(xyz, self.white)
        rgb = np.clip((xyz @ _XYZ_TO_LINEAR_SRGB.T) * self.rgb_gain, 0.0, 1.0)

        total = reflectance.sum(axis=1)
        uv = np.divide(
            reflectance[:, self.uv_band].sum(axis=1),
            total,
            out=np.zeros(len(reflectance)),
            where=total > 0,
        )
        return lab, rgb.astype(np.float32), uv.astype(np.float32)


def _xyz_to_lab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    """CIELAB from XYZ against ``white`` (the CIE piecewise cube-root)."""
    ratio = xyz / white
    delta = 6.0 / 29.0
    f = np.where(
        ratio > delta**3,
        np.cbrt(np.maximum(ratio, 0.0)),
        ratio / (3 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        [
            116.0 * f[:, 1] - 16.0,
            500.0 * (f[:, 0] - f[:, 1]),
            200.0 * (f[:, 1] - f[:, 2]),
        ],
        axis=1,
    )


# =============================================================================
# Corpus assembly
# =============================================================================


class _Accumulator:
    """Per-reading colour, accumulated across files in fixed-size blocks.

    Spectra are reduced and dropped block by block rather than concatenated: the
    corpus is 360,432 x 201 float64, which is 580 MB nobody needs to hold to
    produce three numbers per row.
    """

    def __init__(self, model: "_ColourModel", block_size: int = 20_000) -> None:
        self.model = model
        self.block_size = block_size
        self.labs: list[np.ndarray] = []
        self.rgbs: list[np.ndarray] = []
        self.uvs: list[np.ndarray] = []
        self.meta: dict[str, list[str]] = {field: [] for field in META_COLUMNS}
        self._spectra: list[list[float]] = []
        self._meta_block: dict[str, list[str]] = {f: [] for f in META_COLUMNS}
        self._percent_scale: Optional[bool] = None

    def begin_file(self) -> None:
        """Reset file-scoped scale detection after the previous file is flushed."""
        if self._spectra:
            raise RuntimeError("cannot begin a file with a pending reduction block")
        self._percent_scale = None

    @property
    def scaled_from_percent(self) -> bool:
        return bool(self._percent_scale)

    @property
    def full(self) -> bool:
        return len(self._spectra) >= self.block_size

    def add(self, spectrum: list[float], meta_row: dict[str, str]) -> None:
        self._spectra.append(spectrum)
        for field in META_COLUMNS:
            self._meta_block[field].append(meta_row.get(field, ""))

    def flush(self) -> tuple[int, int]:
        """Reduce the pending block; return ``(kept, invalid_or_too_dark)``."""
        if not self._spectra:
            return 0, 0
        spectra = np.array(self._spectra, dtype=np.float64)
        # Percent-vs-fraction: every file here stores fractions, but a group
        # storing 0-100 would silently produce L* in the thousands, so the scale
        # is measured rather than assumed. Decide once per file so later blocks
        # cannot use a different unit from the first.
        if self._percent_scale is None:
            self._percent_scale = bool(np.nanpercentile(spectra, 99) > 1.5)
        if self._percent_scale:
            spectra /= 100.0
        # Spectrophotometry undershoots its white reference on dark patches;
        # negative reflectance is noise, not absorption.
        np.clip(spectra, 0.0, 1.0, out=spectra)

        lab, rgb, uv = self.model.evaluate(spectra)
        keep = np.isfinite(lab).all(axis=1) & (lab[:, 0] >= MIN_LIGHTNESS)
        self.labs.append(lab[keep].astype(np.float32))
        self.rgbs.append(rgb[keep])
        self.uvs.append(uv[keep])
        for field, values in self._meta_block.items():
            column = np.array(values, dtype=object)
            self.meta[field].extend(column[keep].tolist())
            values.clear()
        self._spectra.clear()
        kept = int(keep.sum())
        return kept, len(keep) - kept

    def result(self) -> dict[str, np.ndarray]:
        corpus: dict[str, np.ndarray] = {
            "lab": np.concatenate(self.labs),
            "rgb": np.concatenate(self.rgbs),
            "uv_chroma": np.concatenate(self.uvs),
        }
        for field, values in self.meta.items():
            corpus[field] = np.array(values, dtype=object)
        return corpus


def _read_corpus() -> dict[str, np.ndarray]:
    """Download, parse and reduce every BirdColorBase file to per-reading colour.

    Pure compute plus downloads; caching is the caller's job.
    """
    accumulator = _Accumulator(_ColourModel())

    for repo_path, digest in SOURCE_FILES.items():
        name = repo_path.rsplit("/", 1)[-1]
        with asection(f"{name}"):
            accumulator.begin_file()
            local = cached_download(
                f"{BIRDCOLORBASE_RAW}/{repo_path}", CACHE_NAME, sha256=digest
            )
            book = _open_workbook(local)
            rows = _iter_rows(book)
            try:
                header = next(rows)
            except StopIteration:
                aprint("empty workbook, skipped")
                continue
            meta_index, spectral_index = _resolve_columns(header)

            kept = 0
            unparseable = 0
            reduced_out = 0
            for row in rows:
                raw_spectrum = [row[i] for i in spectral_index]
                try:
                    spectrum = [float(v) for v in raw_spectrum]  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    unparseable += 1
                    continue
                accumulator.add(
                    spectrum,
                    {
                        field: (row[position] or "").strip()
                        for field, position in meta_index.items()
                    },
                )
                if accumulator.full:
                    block_kept, block_reduced_out = accumulator.flush()
                    kept += block_kept
                    reduced_out += block_reduced_out
            block_kept, block_reduced_out = accumulator.flush()
            kept += block_kept
            reduced_out += block_reduced_out
            scale_note = (
                "; reflectance scaled from percent"
                if accumulator.scaled_from_percent
                else ""
            )
            aprint(
                f"{kept:,} readings kept; {unparseable:,} unparseable; "
                f"{reduced_out:,} invalid or below L* {MIN_LIGHTNESS:g}{scale_note}"
            )

    return accumulator.result()


def load_corpus(recompute: bool = False) -> dict[str, np.ndarray]:
    """The cached per-reading corpus: CIELAB, linear sRGB, UV chroma, metadata.

    Keyed on the pinned commit and on every parameter that changes the numbers
    (the wavelength grid, the UV cut, the lightness floor), so a retune of any
    of them cannot return the previous corpus.
    """
    key = (
        f"corpus_{BIRDCOLORBASE_COMMIT[:12]}_p{_PARSER_VERSION}"
        f"_uv{int(UV_CUTOFF_NM)}_lmin{MIN_LIGHTNESS:g}_n{len(WAVELENGTHS)}"
    )
    return cache_computed(CACHE_NAME, key, _read_corpus, recompute=recompute)


# =============================================================================
# Geometry
# =============================================================================


def _uv_bins(uv_chroma: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Decile-bin UV chroma, returning ``(bin_index, category_labels)``."""
    edges = np.quantile(uv_chroma, np.linspace(0.0, 1.0, N_UV_BINS + 1))
    # Interior edges only, and `right=False` so the top decile keeps its maximum.
    index = np.searchsorted(edges[1:-1], uv_chroma, side="right").astype(np.int64)
    labels = [
        f"{edges[i] * 100:.1f}–{edges[i + 1] * 100:.1f}% UV" for i in range(N_UV_BINS)
    ]
    return index, labels


def _spikes(
    lab: np.ndarray, rgb: np.ndarray, uv_bin: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One two-vertex spike per reading, for ``line_type="segments"``.

    Each spike runs from the neutral grey of the reading's OWN lightness,
    ``(0, 0, L*)``, out to ``(a*, b*, L*)``: direction is hue, length is chroma,
    height is lightness. Anchoring at its own lightness rather than at a common
    origin is what keeps the plot a colour-space plot — a spike's length is then
    exactly C*ab, the reading's chroma, and nothing else.

    Columns are ``(a*, b*, L* - 50, uv_bin)``: L* is recentred on mid-grey so
    the cloud is centred on the scene origin.
    """
    count = len(lab)
    vertices = np.empty((2 * count, 4), dtype=np.float32)
    lightness = lab[:, 0] - LAB_CENTRE_LIGHTNESS

    vertices[0::2, 0] = 0.0
    vertices[0::2, 1] = 0.0
    vertices[1::2, 0] = lab[:, 1]
    vertices[1::2, 1] = lab[:, 2]
    vertices[0::2, 2] = lightness
    vertices[1::2, 2] = lightness
    vertices[0::2, 3] = uv_bin
    vertices[1::2, 3] = uv_bin

    colors = np.repeat(rgb, 2, axis=0)

    widths = np.empty(2 * count, dtype=np.float32)
    widths[0::2] = WIDTH_AT_NEUTRAL
    widths[1::2] = WIDTH_AT_TIP
    return vertices, colors, widths


_SEX_SYMBOL = {"male": "♂", "m": "♂", "female": "♀", "f": "♀"}
_PATCH_PLACEHOLDERS = {"", "not noted", "not recorded", "?"}


def _labels(corpus: dict[str, np.ndarray]) -> tuple[list[str], list[str]]:
    """Per-reading hover text and the Wikipedia key behind it.

    Returns ``(labels, keys)`` per SPIKE; the caller repeats each for the
    spike's two vertices.

    The key drives a Wikipedia SEARCH rather than a direct article title,
    because these are BirdTree/Jetz common names and their capitalisation and
    hyphenation routinely differ from the article ("Sooty Thicket-fantail" vs
    "Sooty thicket fantail"), so a title guess would 404 on exactly the obscure
    species worth looking up. It falls back to the binomial, which Wikipedia
    resolves just as well, and stays empty when there is neither — the viewer
    reads an empty substitution as "not clickable" rather than searching for
    nothing.
    """
    english = corpus["english"]
    species = corpus["species"]
    patch = corpus["patch"]
    sex = corpus["sex"]
    uv = corpus["uv_chroma"]

    labels: list[str] = []
    keys: list[str] = []
    for i in range(len(english)):
        binomial = species[i].replace("_", " ")
        name = english[i] or binomial or "unidentified"
        parts = [name]
        patch_name = patch[i].strip().casefold()
        if patch_name in _PATCH_PLACEHOLDERS:
            patch_name = ""
        detail = [
            value
            for value in (
                patch_name,
                _SEX_SYMBOL.get(sex[i].strip().casefold(), ""),
            )
            if value
        ]
        if detail:
            parts.append(" ".join(detail))
        parts.append(f"{uv[i] * 100:.0f}% UV")
        labels.append(" — ".join(parts))
        keys.append(english[i] or binomial)
    return labels, keys


# =============================================================================
# Scene
# =============================================================================


def _camera(lab: np.ndarray) -> CameraConfig:
    """An opening pose derived from the loaded data, composed for 63°.

    The target is the neutral mid-grey point the whole construction is measured
    from — ``(0, 0, 0)`` in scene units — NOT the bounding-box centre, which the
    asymmetric chroma outliers drag away from the visual anchor and which
    therefore lands the bright core off to one side.

    The distance follows from framing the subject as a sphere:
    ``sin(fov/2) = R / d`` (see :mod:`luxar.demos._cinematic_camera`), with the
    vertical FOV, so the framing holds at any viewport aspect and at any camera
    direction — which is what lets the pose be off-axis.
    """
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    reach = float(np.quantile(chroma, CAMERA_CHROMA_QUANTILE))
    half_depth = float(
        np.abs(lab[:, 0] - LAB_CENTRE_LIGHTNESS).max()
    )  # L* is bounded, so this needs no quantile
    radius = math.sqrt(2.0 * reach**2 + half_depth**2)

    distance = radius / (CAMERA_FILL * math.sin(math.radians(CINEMATIC_FOV_DEG) / 2.0))
    length = math.sqrt(sum(component**2 for component in CAMERA_DIRECTION))
    position = tuple(component / length * distance for component in CAMERA_DIRECTION)
    return CameraConfig(position=position, target=(0.0, 0.0, 0.0))


def _legend_html(labels: list[str], counts: np.ndarray) -> str:
    """A compact readout of what the UV slider is stepping through."""
    rows = "".join(
        f"<div>{i + 1}. {label} &nbsp;<span style='color:#888'>"
        f"{counts[i]:,}</span></div>"
        for i, label in enumerate(labels)
    )
    return (
        '<div style="font-size:1.2vh;line-height:1.5;background:rgba(0,0,0,0.5);'
        'padding:0.6vh;border-radius:3px">'
        '<div style="font-weight:bold;color:#ccc;margin-bottom:0.3vh">'
        "UV chroma deciles &nbsp;<span style='font-weight:normal;color:#888'>"
        "(R300–400 / R300–700)</span></div>"
        f"{rows}</div>"
    )


def build_scene(output_path: Path, recompute: bool = False) -> int:
    """Build the colour-space scene. Returns the reading count."""
    corpus = load_corpus(recompute=recompute)
    lab, rgb = corpus["lab"], corpus["rgb"]
    aprint(f"✓ {len(lab):,} readings")
    aprint(f"  species: {len(set(corpus['species'].tolist())):,}")
    aprint(f"  orders:  {len(set(corpus['order'].tolist())):,}")

    uv_bin, bin_labels = _uv_bins(corpus["uv_chroma"])
    counts = np.bincount(uv_bin, minlength=N_UV_BINS)
    with asection("UV chroma deciles"):
        for i, label in enumerate(bin_labels):
            aprint(f"{i + 1:2d}. {label:22s} {counts[i]:,}")

    vertices, colors, widths = _spikes(lab, rgb, uv_bin)
    labels, keys = _labels(corpus)
    vertex_labels = [text for text in labels for _ in (0, 1)]
    vertex_keys = [key for key in keys for _ in (0, 1)]

    dims = Dimensions(
        [
            Dimension("a*", unit="CIELAB", display=True),
            Dimension("b*", unit="CIELAB", display=True),
            Dimension("L*", unit="CIELAB", display=True),
            Dimension(
                "UV chroma",
                unit="decile",
                display=False,
                categories=bin_labels,
                description=(
                    "Share of total 300-700 nm reflectance falling below 400 nm "
                    "— the ultraviolet the CIE observer cannot see."
                ),
            ),
        ]
    )
    slices = hidden_axis_stops(vertices, dims.non_displayed)

    with asection("Writing scene"):
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                citation=DEMO_META["citation"],
                dimensions=dims,
                viewer_config=ViewerConfig(cinematic_mode=True, camera=_camera(lab)),
            )

            # The ghost. `extend_to_all` pins one copy of every reading at EVERY
            # UV stop, so scrubbing lights a decile up against the silhouette of
            # the whole corpus instead of against empty space — which is the
            # comparison the demo exists to make. Luminous, not volumetric: this
            # layer is meant to be seen THROUGH.
            scene.add_lines(
                "all readings (context)",
                vertices=vertices,
                widths=GHOST_WIDTH,
                colors=GHOST_COLOR,
                line_type="segments",
                sharpness=0.3,
                opacity=GHOST_OPACITY,
                intensity=GHOST_INTENSITY,
                blending_mode="luminous",
                extend_to_all=["UV chroma"],
                # `slices=1` is the RIGHT value here, not an omission: a
                # broadcast node has one resident selection — the whole node, at
                # every stop — so its first rung wants the plain download budget,
                # exactly as for a node with no hidden axis at all. The sliced
                # share floor below would under-serve first paint by 8x for a
                # layer that never gets culled.
                additive_lod=stream_ladder(len(vertices), geometry="lines", slices=1),
                layer=True,
            )

            # The selected decile. `extend_to_all=[]` is explicit: these live at
            # their own UV coordinate and must be culled off-slice.
            scene.add_lines(
                "selected UV decile",
                vertices=vertices,
                widths=widths,
                colors=colors,
                labels=vertex_labels,
                keys=vertex_keys,
                link=(
                    "https://en.wikipedia.org/wiki/Special:Search?search={hover_key}"
                ),
                line_type="segments",
                sharpness=0.55,
                opacity=SPIKE_OPACITY,
                intensity=SPIKE_INTENSITY,
                blending_mode="additive",
                extend_to_all=[],
                additive_lod=stream_ladder(
                    len(vertices), geometry="lines", slices=slices
                ),
                layer=True,
            )

            scene.add_text(
                "Bird Plumage Colour Space",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.65)",
            )
            scene.add_text(
                "press 1 then [ / ] to walk the ultraviolet axis",
                position=(0.02, 0.085),
                font_size=0.022,
                anchor="top-left",
                color="rgba(255,255,255,0.4)",
            )
            scene.add_html(
                _legend_html(bin_labels, counts),
                position=(0.055, 0.97),
                anchor="bottom-left",
            )
            add_demo_caption(
                scene,
                f"{len(lab):,} plumage reflectance spectra • CIELAB under a "
                "daylight illuminant • 4th axis = UV chroma",
                DEMO_META.get("citation"),
            )

    aprint(f"✓ Wrote {len(lab):,} spikes to {output_path}")
    return len(lab)


# =============================================================================
# Entry point
# =============================================================================


def main() -> None:
    flags = parse_demo_flags()

    aprint("=" * 70)
    aprint("BIRD PLUMAGE COLOUR SPACE")
    aprint("=" * 70)
    aprint("")
    aprint("360,432 plumage reflectance spectra in CIELAB.")
    aprint("The 4th dimension is the ultraviolet human vision throws away.")
    aprint("")

    def build(path: Path) -> int:
        try:
            return build_scene(path, recompute=flags["recompute"])
        except Exception as error:  # pragma: no cover - demo entry point
            aprint(f"\nError: {error}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    if flags["no_serve"]:
        output_path = get_demos_output_dir() / "bird_plumage_colorspace.luxar.zarr"
        count = build(output_path)
        aprint(f"Dataset generated at {output_path} ({count:,} readings)")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_birdcolor_") as tmpdir:
        output_path = Path(tmpdir) / "bird_plumage_colorspace.luxar.zarr"
        count = build(output_path)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("")
        aprint("  - Each spike is one plumage patch, painted its own colour")
        aprint("  - Direction = hue, length = chroma, height = lightness")
        aprint("  - Press 1, then [ / ] to walk the UV chroma deciles")
        aprint("  - The faint ghost is the whole corpus, shown at every stop")
        aprint("  - Hover a spike for the species; click it for Wikipedia")
        aprint("")
        aprint(f"Total readings: {count:,}")
        aprint("")
        aprint("Press Ctrl+C when done.")

        launch_viewer(output_path)


if __name__ == "__main__":
    main()
