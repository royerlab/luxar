"""Physical voxel spacing from TIFF metadata (OME-TIFF and ImageJ hyperstacks).

``tifffile.imread`` returns pixels only. The spacing a microscope recorded lives
in one of two places, and :func:`read_tiff_voxel_metadata` reads both:

* **OME-TIFF** — ``PhysicalSizeX/Y/Z`` (+ ``PhysicalSize?Unit``, default ``µm``
  per the OME schema) on the ``Pixels`` element of the OME-XML description.
* **ImageJ** — the z step in the ``spacing`` key and the unit in ``unit`` of the
  ImageJ description; the in-plane pixel size is the reciprocal of the
  ``XResolution`` / ``YResolution`` tags (pixels per unit).

Both are reported against the axis letters of the first series, which is the
array ``tifffile.imread`` returns, so :func:`voxel_info_for_axes` can line the
spacing up with whichever axes the loader keeps.
"""

from __future__ import annotations

import math

# B405/B314 waived for the same reason as `mesh/interop/_vtp.py`: no new
# `defusedxml` dependency, and libexpat's input-amplification cap (expat >= 2.4,
# every supported interpreter) already contains entity-expansion bombs.
import xml.etree.ElementTree as ET  # nosec B405
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..typing_utils.enums import PhysicalUnit

__all__ = ["TiffVoxelMetadata", "read_tiff_voxel_metadata", "voxel_info_for_axes"]

#: OME's default ``PhysicalSize?Unit`` when the attribute is absent.
_OME_DEFAULT_UNIT = "µm"


@dataclass
class TiffVoxelMetadata:
    """Spacing a TIFF recorded, keyed by lowercase axis letter (``z``/``y``/``x``)."""

    #: Series axis letters, uppercase as tifffile reports them (e.g. ``"ZYX"``).
    axes: str
    #: Shape of the series, aligned with ``axes``.
    shape: tuple
    sizes: Dict[str, float] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)
    #: ``"ome"`` or ``"imagej"`` — which block the spacing came from.
    source: str = ""


#: Spellings of the angstrom: the letter A-ring (U+00C5), the dedicated
#: ANGSTROM SIGN (U+212B), ImageJ's literal escape, and the word.
_ANGSTROM_SPELLINGS = frozenset(
    {"\u00c5", "\u212b", "\\u00C5", "\\u00c5", "angstrom", "Angstrom"}
)


def normalize_unit(raw: Optional[str]) -> Optional[str]:
    """A Luxar unit name for ``raw`` (``µm``/``micron`` → ``um``), else ``raw``.

    ImageJ writes the micro sign as a literal ``\\u00B5`` escape, OME as U+00B5;
    both are folded onto the Greek mu :class:`PhysicalUnit` already accepts. Any
    angstrom spelling becomes ``"angstrom"`` (not a :class:`PhysicalUnit`, but
    convertible — see :data:`MICROMETRES_PER_UNIT`). An unrecognised unit is kept
    verbatim rather than dropped.
    """
    if not raw:
        return None
    text = raw.strip().replace("\\u00B5", "μ").replace("\\u00b5", "μ")
    text = text.replace("µ", "μ")
    if text in _ANGSTROM_SPELLINGS:
        return "angstrom"
    try:
        return PhysicalUnit.validate(text).value
    except ValueError:
        return raw.strip()


#: Micrometres per unit, for every length unit a microscope TIFF plausibly
#: records (keys are :func:`normalize_unit` outputs).
MICROMETRES_PER_UNIT: Dict[str, float] = {
    "pm": 1e-6,
    "angstrom": 1e-4,
    "nm": 1e-3,
    "um": 1.0,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
    "metre": 1e6,
    "km": 1e9,
}


def _positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _ome_spacing(xml_text: str, meta: TiffVoxelMetadata) -> None:
    try:
        root = ET.fromstring(xml_text)  # nosec B314
    except ET.ParseError:
        return
    pixels = next((el for el in root.iter() if el.tag.endswith("Pixels")), None)
    if pixels is None:
        return
    for letter in "XYZ":
        size = _positive(pixels.get(f"PhysicalSize{letter}"))
        if size is None:
            continue
        meta.sizes[letter.lower()] = size
        unit = normalize_unit(
            pixels.get(f"PhysicalSize{letter}Unit", _OME_DEFAULT_UNIT)
        )
        if unit:
            meta.units[letter.lower()] = unit
    if meta.sizes:
        meta.source = "ome"


def _resolution_size(page: Any, tag_name: str) -> Optional[float]:
    tag = page.tags.get(tag_name)
    if tag is None:
        return None
    value = tag.value
    if isinstance(value, tuple) and len(value) == 2 and value[0]:
        return _positive(value[1] / value[0])
    return None


def _imagej_spacing(imagej: Dict[str, Any], page: Any, meta: TiffVoxelMetadata) -> None:
    unit = normalize_unit(imagej.get("unit"))
    if unit is None or unit == PhysicalUnit.PIXEL.value:
        # Uncalibrated: ImageJ still writes a 1-pixel-per-unit resolution, which
        # is not a measurement.
        return
    sizes = {
        "x": _resolution_size(page, "XResolution"),
        "y": _resolution_size(page, "YResolution"),
        "z": _positive(imagej.get("spacing")),
    }
    for letter, size in sizes.items():
        if size is None:
            continue
        meta.sizes[letter] = size
        if unit:
            meta.units[letter] = unit
    if meta.sizes:
        meta.source = "imagej"


def read_tiff_voxel_metadata(path: Any) -> Optional[TiffVoxelMetadata]:
    """Read OME / ImageJ voxel spacing from ``path``; ``None`` when none recorded.

    OME wins when both are present (it carries a unit per axis). Never raises on
    malformed metadata — spacing is a hint, and the pixels load regardless.
    """
    import tifffile

    try:
        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            meta = TiffVoxelMetadata(axes=series.axes, shape=tuple(series.shape))
            if tif.ome_metadata:
                _ome_spacing(tif.ome_metadata, meta)
            if not meta.sizes and tif.imagej_metadata:
                _imagej_spacing(tif.imagej_metadata, tif.pages[0], meta)
    except (OSError, ValueError, IndexError, KeyError):
        return None
    return meta if meta.sizes else None


def _kept_axes(meta: TiffVoxelMetadata, labels: Optional[Sequence[str]]) -> List[str]:
    """Lowercase letters of the axes the loader returns, in order."""
    if labels is not None:
        return [label.strip().lower() for label in labels]
    return [
        letter.lower()
        for letter, size in zip(meta.axes, meta.shape, strict=True)
        if size != 1
    ]


def voxel_info_for_axes(
    meta: TiffVoxelMetadata, labels: Optional[Sequence[str]], ndim: int
) -> Optional[Dict[str, Any]]:
    """``info`` entries for a loaded volume of rank ``ndim``, or ``None``.

    ``voxel_size`` is normalized to ONE unit, micrometres (``voxel_unit="um"``),
    so a mixed-unit file (OME lets each axis carry its own) yields comparable
    numbers; it is present only when every unit is a known length. The values and
    units exactly as recorded are kept in ``source_voxel_size`` /
    ``source_voxel_units`` (a unit is ``None`` where the file named none).

    ``labels`` are the spatial axis names the loader kept (an explicit ``--axes``
    spec); ``None`` means the positional path, whose returned array is the series
    with its size-1 axes squeezed away. Spacing is reported only when EVERY
    returned axis has one — a partial list would be ambiguous about which axis
    it describes.
    """
    axes = _kept_axes(meta, labels)
    if len(axes) != ndim or not all(axis in meta.sizes for axis in axes):
        return None
    sizes = [meta.sizes[axis] for axis in axes]
    units = [meta.units.get(axis) for axis in axes]
    info: Dict[str, Any] = {
        "axes": axes,
        "voxel_size_source": meta.source,
        "source_voxel_size": sizes,
        "source_voxel_units": units,
    }
    factors = [MICROMETRES_PER_UNIT.get(unit or "") for unit in units]
    if all(factor is not None for factor in factors):
        info["voxel_size"] = [
            size * float(factor or 0.0)
            for size, factor in zip(sizes, factors, strict=True)
        ]
        info["voxel_unit"] = "um"
    return info
