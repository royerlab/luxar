"""Voxel spacing and crop regions for ``gsplat fit`` / ``cal`` / ``batch-fit``.

Two questions every fit of a microscopy stack has to answer, decided here once:

* **What is the voxel spacing, and where did it come from?** An explicit
  ``--voxel-size Z,Y,X`` wins; then a ``voxel_size:`` in the fit config; then,
  only when ``--physical`` asks for it, the file's own metadata: the OME-Zarr
  NGFF ``coordinateTransformations`` scale, or the OME-TIFF ``PhysicalSize*`` /
  calibrated ImageJ spacing :func:`luxar.io.volume.load_volume` reports in its
  ``info`` dict (normalized to micrometres). One line names the spacing and its
  source; ``--physical`` with nothing to read warns and the fit stays in voxel
  units. :func:`ngff_spatial_spacing` and :func:`validated_spacing` are shared
  with ``batch-fit --physical`` so the two cannot read one store differently.
* **Which part of the volume, and where does it sit?** ``--region
  z0:z1,y0:y1,x0:x1`` crops at load time (lazily for zarr). The fitted splats are
  then translated by the crop's start, so their centers are in FULL-VOLUME
  coordinates — a crop overlays the whole — and the store records the crop as
  ``fitting/source_region``.

:class:`FitFrame` carries both decisions to the save, where
:func:`apply_fit_frame` applies the offset and stamps ``fitting/`` with
``voxel_size``, ``voxel_unit``, ``voxel_size_source``, ``physical_coordinates``
and ``source_region``.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import typer
from arbol import aprint

__all__ = [
    "FitFrame",
    "VoxelSpacing",
    "apply_fit_frame",
    "discover_spacing",
    "FitFrameDecision",
    "format_region",
    "ngff_spatial_spacing",
    "normalized_region",
    "parse_region",
    "parse_voxel_size",
    "region_rank_guard",
    "resolve_fit_frame",
    "resolve_voxel_spacing",
    "validated_spacing",
]

#: Source labels, as printed and as stamped in ``fitting/voxel_size_source``.
SOURCE_FLAG = "--voxel-size"
SOURCE_CONFIG = "fit config"
SOURCE_NGFF = "OME-Zarr NGFF scale"
_TIFF_SOURCES = {"ome": "OME-TIFF metadata", "imagej": "ImageJ metadata"}

#: Unit stamped for an explicit ``--voxel-size`` (the flag documents micrometres,
#: the unit TIFF metadata is normalized to).
EXPLICIT_UNIT = "um"

#: Largest/smallest spacing ratio above which a stack counts as anisotropic for
#: the "pass --physical" hint.
ANISOTROPY_RATIO = 1.01


@dataclass(frozen=True)
class VoxelSpacing:
    """A per-axis voxel spacing (array axis order) and where it came from."""

    values: Tuple[float, ...]
    unit: Optional[str]
    source: str

    def describe(self, note: str = "") -> str:
        """``0.29×0.108×0.108 µm (from OME-TIFF metadata)``, plus ``; note``."""
        sizes = "×".join(f"{value:g}" for value in self.values)
        unit = f" {display_unit(self.unit)}" if self.unit else ""
        return f"{sizes}{unit} (from {self.source}{'; ' + note if note else ''})"


def display_unit(unit: Optional[str]) -> str:
    """Human spelling of a Luxar unit name (``um`` → ``µm``)."""
    return "µm" if unit == "um" else (unit or "")


def is_anisotropic(values: Sequence[float]) -> bool:
    """Whether the spacing differs between axes by more than 1 %."""
    return bool(values) and max(values) > ANISOTROPY_RATIO * min(values)


def _axis_letters(ndim: int) -> str:
    return {2: "Y,X", 3: "Z,Y,X"}.get(ndim, ",".join(f"d{i}" for i in range(ndim)))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_voxel_size(
    spec: Optional[str], flag: str = "--voxel-size"
) -> Optional[Tuple[float, ...]]:
    """Parse ``Z,Y,X`` (array axis order) into positive finite spacings."""
    if spec is None:
        return None
    try:
        values = tuple(float(part.strip()) for part in spec.split(","))
    except ValueError as exc:
        raise typer.BadParameter(
            f"{flag} must be comma-separated numbers (Z,Y,X), got {spec!r}"
        ) from exc
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise typer.BadParameter(f"{flag} values must be positive, got {spec!r}")
    return values


def _region_bound(text: str, flag: str) -> Optional[int]:
    text = text.strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise typer.BadParameter(f"{flag}: {text!r} is not an integer") from exc
    if value < 0:
        raise typer.BadParameter(
            f"{flag}: negative index {value} — give absolute voxel indices"
        )
    return value


def _parse_span(text: str, flag: str) -> slice:
    parts = text.split(":")
    if len(parts) != 2:
        raise typer.BadParameter(
            f"{flag}: {text!r} must be START:STOP (empty side = axis edge; no step)"
        )
    start, stop = (_region_bound(part, flag) for part in parts)
    if start is not None and stop is not None and stop <= start:
        raise typer.BadParameter(f"{flag}: {text!r} selects nothing")
    return slice(start, stop)


def parse_region(
    spec: Optional[str], flag: str = "--region"
) -> Optional[Tuple[slice, ...]]:
    """Parse ``z0:z1,y0:y1,x0:x1`` into one slice per spatial axis.

    Numpy-style half-open spans in voxel indices of the loaded volume's axes
    (array order). An empty span or a bare ``:`` keeps the whole axis, and either
    side may be omitted (``100:``, ``:64``). Steps and negative indices are
    refused: the crop's start becomes the fitted splats' offset, which only a
    plain non-negative start describes.
    """
    if spec is None:
        return None
    return tuple(
        slice(None) if part.strip() in ("", ":") else _parse_span(part, flag)
        for part in spec.split(",")
    )


def format_region(region: Sequence[slice]) -> str:
    """Inverse of :func:`parse_region` (for forwarding to worker subprocesses)."""
    return ",".join(
        f"{'' if s.start is None else s.start}:{'' if s.stop is None else s.stop}"
        for s in region
    )


def format_spans(spans: Sequence[Tuple[int, int]]) -> str:
    """``z 4:12, y 8:32, x 0:20`` for a normalized 3-D region."""
    letters = _axis_letters(len(spans)).lower().split(",")
    return ", ".join(
        f"{axis} {start}:{stop}"
        for axis, (start, stop) in zip(letters, spans, strict=True)
    )


def _region_rank_message(n_spans: int, n_axes: int, shape: str = "") -> str:
    return (
        f"--region names {n_spans} axes but the volume has {n_axes}{shape}. Give "
        "one span per spatial axis in array order; a one-plane crop of a zarr "
        "store is squeezed away unless --axes keeps it (e.g. --axes z,y,x)."
    )


@contextmanager
def region_rank_guard() -> Iterator[None]:
    """Report a loader's span-count mismatch as a ``--region`` usage error."""
    from luxar.io.volume import RegionRankError

    try:
        yield
    except RegionRankError as exc:
        raise typer.BadParameter(_region_rank_message(exc.n_spans, exc.n_axes)) from exc


def normalized_region(
    region: Sequence[slice], cropped_shape: Sequence[int]
) -> Tuple[Tuple[int, int], ...]:
    """``((start, stop), ...)`` of a crop, from its spans and the loaded shape.

    Raises:
        typer.BadParameter: When the loaded volume does not have one axis per
            span (a span count that disagrees with the volume, or a zarr crop
            one plane thick that the positional loader squeezed away), or when a
            span starts past the end of its axis.
    """
    if len(region) != len(cropped_shape):
        raise typer.BadParameter(
            _region_rank_message(
                len(region), len(cropped_shape), f" ({tuple(cropped_shape)})"
            )
        )
    spans = []
    for span, size in zip(region, cropped_shape, strict=True):
        if size == 0:
            raise typer.BadParameter(
                f"--region span {format_region([span])} selects nothing "
                "(it starts past the end of its axis)"
            )
        start = int(span.start or 0)
        spans.append((start, start + int(size)))
    return tuple(spans)


# ---------------------------------------------------------------------------
# Discovery + selection
# ---------------------------------------------------------------------------


def _normalized_ngff_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    from luxar.typing_utils.enums import PhysicalUnit

    try:
        return PhysicalUnit.validate(unit).value
    except ValueError:
        return unit


def ngff_spatial_spacing(
    ome_info: Any, axes_list: Optional[Sequence[str]]
) -> Optional[Tuple[float, ...]]:
    """The NGFF spacing of the axes the loader returns, or ``None``.

    Without an explicit ``--axes`` the positional loader squeezes size-1 spatial
    axes away, so their spacing is dropped with them.
    """
    discovered = ome_info.voxel_size
    if discovered is not None and axes_list is None:
        discovered = tuple(
            spacing
            for size, spacing in zip(ome_info.spatial_shape, discovered, strict=True)
            if size != 1
        )
    return tuple(float(v) for v in discovered) if discovered is not None else None


def _ngff_spacing(
    path: Path, array_key: Optional[str], axes: Optional[str]
) -> Optional[VoxelSpacing]:
    from luxar.io.ome_zarr import discover_ome_zarr_shape
    from luxar.utils.verbosity import verbosity

    axes_list = [a.strip() for a in axes.split(",")] if axes else None
    try:
        # Discovery narrates (and warns about guessed axes); the load already
        # said what it read, so only the spacing matters here.
        with verbosity("silent"):
            info = discover_ome_zarr_shape(
                path, axes_override=axes_list, array_key=array_key
            )
    except Exception:  # noqa: BLE001 - spacing is a hint; the pixels already loaded
        return None
    values = ngff_spatial_spacing(info, axes_list)
    if values is None:
        return None
    return VoxelSpacing(values, _normalized_ngff_unit(info.unit), SOURCE_NGFF)


def discover_spacing(
    input_path: Path,
    source_info: Mapping[str, Any],
    *,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
) -> Optional[VoxelSpacing]:
    """Voxel spacing the input file records: TIFF (via ``info``) or OME-Zarr NGFF."""
    tiff = source_info.get("voxel_size")
    if tiff:
        source = _TIFF_SOURCES.get(
            str(source_info.get("voxel_size_source")), "TIFF metadata"
        )
        return VoxelSpacing(
            tuple(float(v) for v in tiff), source_info.get("voxel_unit"), source
        )
    from luxar._zarr_compat import is_zarr_path

    if is_zarr_path(input_path):
        return _ngff_spacing(input_path, array_key, axes)
    return None


def validated_spacing(selected: Any, ndim: int, what: str) -> List[float]:
    """``selected`` (scalar or per-axis) as ``ndim`` positive floats.

    Raises:
        typer.BadParameter: On a wrong count or a non-positive value.
    """
    from luxar.gsplats.tiling import resolve_grid_scale

    if not isinstance(selected, (int, float)) and len(selected) != ndim:
        raise typer.BadParameter(
            f"{what} gives {len(selected)} values but the volume is {ndim}-D; "
            f"give one per axis in array order ({_axis_letters(ndim)})"
        )
    try:
        resolve_grid_scale(ndim, voxel_size=selected, output_space="real")
        if isinstance(selected, (int, float)):
            return [float(selected)] * ndim
        return [float(value) for value in selected]
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"{what} is invalid: {exc}") from exc


def require_real_output_space(config: Mapping[str, Any]) -> None:
    """``--physical`` cannot be honoured when the config asks for voxel output."""
    if config.get("output_space", "real") != "real":
        raise typer.BadParameter(
            "--physical requires output_space: real; remove the config override "
            "or omit --physical to keep voxel coordinates."
        )


def _report_spacing(
    spacing: Optional[VoxelSpacing],
    discovered: Optional[VoxelSpacing],
    *,
    physical: bool,
    hint: bool,
    input_name: str,
    ndim: int,
    axes: Optional[str] = None,
) -> None:
    """The one line naming the spacing — or the warning / hint in its place."""
    if spacing is not None:
        overridden = physical and spacing.source in (SOURCE_FLAG, SOURCE_CONFIG)
        note = "takes precedence over file metadata" if overridden else ""
        aprint(f"Voxel size {_axis_letters(ndim)} = {spacing.describe(note)}")
    elif physical and ndim > 3 and axes is None:
        aprint(
            f"⚠ --physical: {input_name} loaded as a {ndim}-D volume, so its "
            "spacing cannot be matched to spatial axes; fitting in voxel units. "
            "For a timelapse pass --axes t,z,y,x (with --timepoint), or give "
            "--voxel-size."
        )
    elif physical:
        aprint(
            f"⚠ --physical: no usable voxel size in {input_name}'s metadata "
            "(OME-Zarr NGFF scale, OME-TIFF PhysicalSize, ImageJ spacing); "
            f"fitting in voxel units. Pass --voxel-size {_axis_letters(ndim)} "
            "to set it."
        )
    elif hint and discovered is not None and is_anisotropic(discovered.values):
        aprint(
            f"Hint: this stack has voxel size {_axis_letters(ndim)} = "
            f"{discovered.describe()}; pass --physical to fit in "
            f"{display_unit(discovered.unit) or 'physical units'}."
        )


def resolve_voxel_spacing(
    *,
    input_path: Path,
    source_info: Mapping[str, Any],
    ndim: int,
    explicit: Optional[Sequence[float]],
    physical: bool,
    config: Mapping[str, Any],
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
    hint: bool = False,
) -> Optional[VoxelSpacing]:
    """Pick the fit's voxel spacing: ``--voxel-size`` > config > file metadata.

    File metadata is used only under ``physical``; with ``hint`` an anisotropic
    spacing the user did not ask for is pointed out instead. Prints exactly one
    line (spacing used, ``--physical`` warning, or hint) or nothing.

    Raises:
        typer.BadParameter: On ``--physical`` with ``output_space: voxel`` in
            the config, or an explicit/configured spacing whose count does not
            match the volume.
    """
    if physical:
        require_real_output_space(config)
    configured = config.get("voxel_size")
    spacing: Optional[VoxelSpacing] = None
    if explicit is not None:
        values = validated_spacing(tuple(explicit), ndim, SOURCE_FLAG)
        spacing = VoxelSpacing(tuple(values), EXPLICIT_UNIT, SOURCE_FLAG)
    elif configured is not None:
        values = validated_spacing(configured, ndim, "the config's voxel_size")
        spacing = VoxelSpacing(tuple(values), None, SOURCE_CONFIG)
    discovered = None
    if spacing is None and (physical or hint):
        discovered = discover_spacing(
            input_path, source_info, array_key=array_key, axes=axes
        )
        if discovered is not None and len(discovered.values) != ndim:
            aprint(
                f"Note: {discovered.source} gives {len(discovered.values)} "
                f"spacings for a {ndim}-D volume; ignoring it."
            )
            discovered = None
        if physical:
            spacing = discovered
    _report_spacing(
        spacing,
        discovered,
        physical=physical,
        hint=hint,
        input_name=input_path.name,
        ndim=ndim,
        axes=axes,
    )
    return spacing


# ---------------------------------------------------------------------------
# Applying the frame at save time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitFrame:
    """The coordinate frame a fit's output is saved in.

    Attributes:
        spacing: Voxel spacing the fit used, or ``None`` (voxel units).
        physical: Whether centers/Cholesky are in physical units (spacing ×
            voxel index) rather than voxel indices.
        region: ``((start, stop), ...)`` of a ``--region`` crop in the source
            volume's voxel indices, or ``None``. Its start is added to the
            centers at save time (× spacing when ``physical``).
        source_shape: Shape of the array the fit was handed (the crop's, for a
            region). Filled into ``fitting/source_shape`` only where the fitter
            recorded none (a content fit), so ``dimension_metadata`` naming
            (and its unit) applies to every tiling mode.
    """

    spacing: Optional[VoxelSpacing] = None
    physical: bool = False
    region: Optional[Tuple[Tuple[int, int], ...]] = None
    source_shape: Optional[Tuple[int, ...]] = None

    def offset(self) -> Optional[np.ndarray]:
        """Translation from crop-local to full-volume coordinates, or ``None``."""
        if not self.region or not any(start for start, _ in self.region):
            return None
        starts = np.array([start for start, _ in self.region], dtype=np.float64)
        if self.physical and self.spacing is not None:
            starts = starts * np.asarray(self.spacing.values, dtype=np.float64)
        return starts

    def stamps(self) -> Dict[str, Any]:
        """``fitting/`` attrs describing this frame."""
        out: Dict[str, Any] = {"physical_coordinates": bool(self.physical)}
        if self.spacing is not None:
            out["voxel_size"] = [float(v) for v in self.spacing.values]
            out["voxel_size_source"] = self.spacing.source
            if self.spacing.unit:
                out["voxel_unit"] = self.spacing.unit
        if self.region is not None:
            out["source_region"] = [[int(a), int(b)] for a, b in self.region]
        return out


def _translate_tree(node: Any, offset: np.ndarray) -> Any:
    """Translate every leaf of a partition/nested tree, split planes included."""
    from luxar.cli.gsplat_ops.transforms.transform import _map_partition_planes
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.tree import map_leaves

    def _leaf(leaf: Any) -> Any:
        moved = GSplatData.from_tree(leaf).translate(offset).tree
        return replace(moved, meta=dict(leaf.meta))

    moved = map_leaves(node, _leaf)
    return _map_partition_planes(moved, np.eye(len(offset)), offset)


def apply_fit_frame(result: Any, frame: Optional[FitFrame]) -> Any:
    """Move ``result`` into full-volume coordinates and stamp its frame.

    ``result`` is a flat ``GSplatData`` (stats on ``.stats``) or a tree node
    (stats on ``meta["fit_stats"]``), as ``save_fit_output`` receives it.
    """
    if frame is None:
        return result
    from luxar.gsplats.gsplat_data import GSplatData

    offset = frame.offset()
    fill = (
        {"source_shape": [int(n) for n in frame.source_shape]}
        if frame.source_shape is not None
        else {}
    )
    if isinstance(result, GSplatData):
        if offset is not None and result.n_splats > 0:
            stats = result.stats
            result = result.translate(offset)
            result.stats = {**stats, **(result.stats or {})}
        result.stats.update(frame.stamps())
        _fill_missing(result.stats, fill)
        return result
    if offset is not None:
        result = _translate_tree(result, offset)
    stats = result.meta.setdefault("fit_stats", {})
    stats.update(frame.stamps())
    _fill_missing(stats, fill)
    return result


def _fill_missing(stats: Dict[str, Any], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        if not stats.get(key):
            stats[key] = value


@dataclass(frozen=True)
class FitFrameDecision:
    """What ``gsplat fit`` does with the resolved spacing and crop.

    Attributes:
        voxel_size: Spacing to hand the fitter as a CLI override (explicit or
            metadata-derived), or ``None`` to leave the config's own value.
        content_physical: Whether a ``--tiling content`` fit should emit physical
            geometry. Only an explicit request does; a config-only ``voxel_size``
            keeps content tiling's legacy voxel output.
        frame: The frame the output is saved in (see :class:`FitFrame`).
    """

    voxel_size: Optional[Tuple[float, ...]]
    content_physical: bool
    frame: FitFrame


def resolve_fit_frame(
    *,
    input_path: Path,
    source_info: Mapping[str, Any],
    volume_shape: Sequence[int],
    explicit: Optional[Sequence[float]],
    handed_unit: Optional[str] = None,
    handed_source: Optional[str] = None,
    physical: bool,
    physical_coordinates: bool,
    preset: Optional[str],
    config_path: Optional[Path],
    content: bool,
    region: Optional[Sequence[slice]],
    worker: bool,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
) -> FitFrameDecision:
    """Resolve ``gsplat fit``'s spacing, content-physical mode and output frame.

    ``worker`` marks an internal tile/box subprocess: it prints no hint and
    applies no crop offset (the parent that merges the tiles does, once).
    """
    from luxar.cli.gsplat_config import load_fit_config

    # The crop is validated BEFORE any spacing line is printed.
    norm_region = (
        normalized_region(region, volume_shape)
        if region is not None and not worker
        else None
    )
    config = load_fit_config(preset=preset, config_path=config_path)
    spacing = resolve_voxel_spacing(
        input_path=input_path,
        source_info=source_info,
        ndim=len(volume_shape),
        explicit=explicit,
        physical=physical,
        config=config,
        array_key=array_key,
        axes=axes,
        hint=not worker,
    )
    if handed_source is not None and spacing is not None:
        spacing = replace(spacing, unit=handed_unit, source=handed_source)
    requested = spacing is not None and spacing.source != SOURCE_CONFIG
    content_physical = physical_coordinates or (content and requested)
    # A content fit without physical geometry drops voxel_size entirely
    # (`fit_planned._content_fit_frame`), so its output is in voxel units.
    used = spacing if (content_physical or not content) else None
    output_space = config.get("output_space", "real")
    if norm_region is not None:
        aprint(
            f"Region: {format_spans(norm_region)} of the input; the splats are "
            "placed at their position in the full volume"
        )
    frame = FitFrame(
        spacing=used,
        physical=used is not None and output_space == "real",
        region=norm_region,
        source_shape=tuple(int(n) for n in volume_shape),
    )
    return FitFrameDecision(
        voxel_size=spacing.values if requested and spacing is not None else None,
        content_physical=content_physical,
        frame=frame,
    )
