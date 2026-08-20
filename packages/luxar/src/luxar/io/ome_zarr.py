"""OME-Zarr shape discovery (domain layer).

Discovers the shape and axis structure of an OME-Zarr / NGFF dataset (T/C/Z/Y/X
layout, voxel size, unit, resolution levels), with fallbacks to a custom
``axes`` attribute and a shape-based heuristic. Reusable domain logic with no
CLI/Typer coupling. Previously lived in ``luxar.cli.gsplat_config``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from luxar.io.volume import _select_zarr_array

__all__ = [
    "CHANNEL_LIKE_AXIS_LABELS",
    "OMEZarrInfo",
    "TIME_AXIS_LABELS",
    "classify_axis_labels",
    "discover_ome_zarr_shape",
]

# The axis-label vocabulary for LABEL-driven axis classification, in ONE place:
# every consumer that has to decide which axes of a store are time /
# channel-like / spatial reads these (see :func:`classify_axis_labels`). Keeping
# a private copy per call site is how they drift — a ``view`` axis classified as
# spatial by one and channel-like by another silently disagrees about the store's
# shape.
#
# DELIBERATE EXCEPTION, do not "unify" it: ``luxar.io.volume._axis_kind`` keeps
# its OWN, narrower vocabulary — no ``view``/``angle``, and it RAISES on an
# unknown label instead of treating it as spatial. That is load-bearing: it backs
# the user-facing ``--axes`` spec, where a typo must be a clean error rather than
# a silently mis-sliced volume, whereas discovery here must stay lenient about
# whatever a store happens to declare.
#
# Not part of the classification rule below (an unrecognised label defaults to
# spatial, so nothing consults this set) — it is documentation of the labels a
# store is expected to use, hence not exported.
SPATIAL_AXIS_LABELS = frozenset({"z", "y", "x", "depth", "height", "width"})
TIME_AXIS_LABELS = frozenset({"time", "t"})
CHANNEL_LIKE_AXIS_LABELS = frozenset(
    # channel + extra non-spatial ("camera"-like) axes: each combination of them
    # becomes its own flat channel task.
    {"channel", "c", "ch", "camera", "cam", "view", "angle"}
)


def classify_axis_labels(
    axes: "List[str] | Tuple[str, ...]",
) -> "Tuple[Optional[int], List[int], List[int]]":
    """Split axis LABELS into ``(time_axis, channel_like_axes, spatial_axes)``.

    Indices into ``axes``; an unrecognised label is treated as spatial (the same
    lenient rule the NGFF parser uses), and a repeated time label keeps the last
    one. Case-insensitive.
    """
    time_axis: Optional[int] = None
    channel_like: List[int] = []
    spatial: List[int] = []
    for i, label in enumerate(axes):
        lowered = str(label).strip().lower()
        if lowered in TIME_AXIS_LABELS:
            time_axis = i
        elif lowered in CHANNEL_LIKE_AXIS_LABELS:
            channel_like.append(i)
        else:
            spatial.append(i)
    return time_axis, channel_like, spatial


@dataclass
class OMEZarrInfo:
    """Metadata about an OME-Zarr dataset's structure."""

    axes: List[str]
    """Axis labels, e.g. ``["t", "c", "z", "y", "x"]``."""

    shape: Tuple[int, ...]
    """Full array shape at highest resolution."""

    n_timepoints: int
    """Size of the T dimension (1 if absent)."""

    n_channels: int
    """Number of flat channel tasks (product of channel-like axes, or 1)."""

    channel_axes: List[str]
    """Axis labels folded into the flat channel task index."""

    channel_shape: Tuple[int, ...]
    """Shape of axes folded into the flat channel task index."""

    spatial_shape: Tuple[int, ...]
    """ZYX (or YX) portion of the shape."""

    spatial_axes: List[str]
    """Spatial axis labels, e.g. ``["z", "y", "x"]``."""

    voxel_size: Optional[Tuple[float, ...]] = None
    """Physical spacing from coordinateTransformations (spatial axes only)."""

    unit: Optional[str] = None
    """Physical unit string (e.g. ``"micrometer"``)."""

    resolution_levels: int = 1
    """Number of multiscale levels."""

    path: Optional[Path] = None
    """Path to the zarr store."""


def _describes(block: Any, ndim: int) -> bool:
    """Is an OWNING group's ``multiscales`` block usable for the selected array?

    The gate applies to the owner's block ONLY (the root's keeps whatever
    behaviour it had). A group can carry a block describing something other than
    the array selected below it — another series, a stale hand-written attribute —
    and adopting that on the strength of its mere existence would silently rewrite
    a T/C decomposition the root already had right, which is a wrong `batch-fit`
    plan rather than an error. An axis count that disagrees with the array is the
    cheap, decisive test: such a block is not metadata about this array.
    """
    if not (isinstance(block, list) and block and isinstance(block[0], dict)):
        return False
    axes = block[0].get("axes")
    return isinstance(axes, list) and len(axes) == ndim


def discover_ome_zarr_shape(
    path: Path,
    axes_override: Optional[List[str]] = None,
    array_key: Optional[str] = None,
) -> OMEZarrInfo:
    """Discover the shape and axis structure of an OME-Zarr dataset.

    Parses ``.zattrs`` ``multiscales`` metadata (NGFF v0.4+). Falls back
    to a custom ``axes`` attribute, then to a shape-based heuristic
    (5D→TCZYX, 4D→CZYX, 3D→ZYX) for non-NGFF zarr stores.

    Accepts both plain ``.zarr`` directories and ``.zarr.zip`` archives —
    zarr's ZipStore handles the latter transparently.

    Args:
        path: Path to the ``.zarr`` store or ``.zarr.zip`` archive.
        axes_override: Explicit axis labels (e.g. ``["time","channel","z","y","x"]``).
            Overrides all auto-detection when provided.
        array_key: Key path to a specific array within the zarr store
            (e.g. ``"h2afva/fused"``).  When provided, skips auto-selection
            and navigates directly to this array.

    Returns:
        :class:`OMEZarrInfo` with discovered metadata.

    Raises:
        ValueError: If the zarr store has no arrays, ``array_key`` is not
            found, or the store is unreadable.
    """
    import zarr

    from luxar._zarr_compat import open_store

    # See the note in `luxar.io.volume`: zarr 3 no longer sniffs a `.zip` suffix,
    # so the ZipStore dispatch has to be explicit or `.zarr.zip` inputs raise.
    store = zarr.open(store=open_store(path, mode="r"), mode="r")

    # Navigate to the group/array. The rule lives in `_select_zarr_array` so this
    # and the two volume-loading entry points cannot drift apart — an
    # `array_key` may be nested (e.g. "h2afva/fused") and may name a group.
    attrs: Dict[str, Any]
    # `multiscales` / `axes` describe the array they sit BESIDE, and that is not
    # always the root: a bioformats2raw store puts the NGFF block on the image
    # group ("0") and leaves only `bioformats2raw.layout` at the root, so reading
    # the root alone finds nothing and falls through to the shape heuristic —
    # which GUESSES the T/C roles and recovers no voxel size. So the owning
    # group's attributes are consulted FIRST, per key and only when usable for
    # the array actually selected (see `_describes`); otherwise the root's stand,
    # which is the plain OME-NGFF case (there the owner IS the root).
    owner_attrs: Dict[str, Any] = {}
    if isinstance(store, zarr.Array):
        arr = store
        attrs = dict(getattr(store, "attrs", {}))
    else:
        arr, _, owner = _select_zarr_array(store, path, array_key)
        attrs = dict(store.attrs)
        if owner is not None and owner is not store:
            owner_attrs = dict(owner.attrs)

    shape = tuple(arr.shape)
    ndim = len(shape)

    # User-supplied axes override: skip all auto-detection
    if axes_override is not None:
        if len(axes_override) != ndim:
            raise ValueError(
                f"--axes has {len(axes_override)} labels but array is {ndim}D "
                f"(shape {shape}). Provide exactly {ndim} comma-separated axis names."
            )
        return _parse_custom_axes_attr(axes_override, shape, path)

    # Try NGFF multiscales metadata
    owner_ms = owner_attrs.get("multiscales")
    multiscales = owner_ms if _describes(owner_ms, ndim) else attrs.get("multiscales")
    if multiscales and isinstance(multiscales, list) and len(multiscales) > 0:
        ms = multiscales[0]
        return _parse_ngff_metadata(ms, shape, path, store)

    # Try custom axes attribute (e.g. Keller-lab zarr.zip files store
    # axes = ['time', 'camera', 'channel', 'z', 'y', 'x'])
    # Same owner-first rule, same usability test in its `axes` form: one label
    # per dimension of the array actually selected, or the root's list stands.
    owner_axes = owner_attrs.get("axes")
    custom_axes = (
        owner_axes
        if isinstance(owner_axes, list) and len(owner_axes) == ndim
        else attrs.get("axes")
    )
    if custom_axes and isinstance(custom_axes, list) and len(custom_axes) == ndim:
        return _parse_custom_axes_attr(custom_axes, shape, path)

    # Fallback: heuristic based on ndim
    return _heuristic_ome_info(shape, ndim, path)


def _parse_ngff_metadata(
    ms: Dict[str, Any],
    shape: Tuple[int, ...],
    path: Path,
    store: Any,
) -> OMEZarrInfo:
    """Parse NGFF v0.4+ multiscales metadata."""
    axes_raw = ms.get("axes", [])
    axes = [a["name"] if isinstance(a, dict) else str(a) for a in axes_raw]

    # Identify T, C, spatial axes
    t_idx: Optional[int] = None
    c_idx: Optional[int] = None
    spatial_indices: List[int] = []
    spatial_axes: List[str] = []

    for i, a in enumerate(axes_raw):
        if isinstance(a, dict):
            atype = a.get("type", "").lower()
            aname = a.get("name", "").lower()
        else:
            atype = ""
            aname = str(a).lower()

        if atype == "time" or aname == "t":
            t_idx = i
        elif atype == "channel" or aname == "c":
            c_idx = i
        elif atype == "space" or aname in ("z", "y", "x"):
            spatial_indices.append(i)
            spatial_axes.append(aname)
        else:
            # Unknown axis — treat as spatial
            spatial_indices.append(i)
            spatial_axes.append(aname)

    n_t = shape[t_idx] if t_idx is not None else 1
    channel_axes = [axes[c_idx]] if c_idx is not None else []
    channel_shape = (shape[c_idx],) if c_idx is not None else ()
    n_c = shape[c_idx] if c_idx is not None else 1
    spatial_shape = tuple(shape[i] for i in spatial_indices)

    # Extract voxel_size from coordinateTransformations
    voxel_size = None
    unit = None
    datasets = ms.get("datasets", [])
    if datasets:
        transforms = datasets[0].get("coordinateTransformations", [])
        for t in transforms:
            if t.get("type") == "scale":
                scale = t.get("scale", [])
                # Extract spatial dimensions only
                if spatial_indices and len(scale) == len(shape):
                    voxel_size = tuple(float(scale[i]) for i in spatial_indices)
                elif len(scale) == len(spatial_indices):
                    voxel_size = tuple(float(s) for s in scale)

    # Extract unit from axes metadata
    for a in axes_raw:
        if isinstance(a, dict) and a.get("type") == "space":
            u = a.get("unit")
            if u:
                unit = u
                break

    # Count resolution levels
    n_levels = len(datasets) if datasets else 1

    return OMEZarrInfo(
        axes=axes,
        shape=shape,
        n_timepoints=n_t,
        n_channels=n_c,
        channel_axes=channel_axes,
        channel_shape=channel_shape,
        spatial_shape=spatial_shape,
        spatial_axes=spatial_axes,
        voxel_size=voxel_size,
        unit=unit,
        resolution_levels=n_levels,
        path=path,
    )


def _parse_custom_axes_attr(
    axes: List[str], shape: Tuple[int, ...], path: Path
) -> OMEZarrInfo:
    """Build OMEZarrInfo from a custom ``axes`` list attribute.

    Recognises common axis name conventions:
      - T: ``time``, ``t``
      - C: ``channel``, ``c``, ``ch``
      - Camera / extra non-spatial dims (``camera``, ``cam``, ``view``,
        ``angle``): folded into the channel count so each combination
        becomes its own fitting task.
      - Spatial: ``z``, ``y``, ``x``, ``depth``, ``height``, ``width``
        (and any unrecognised leftover axes)
    """
    # Shared vocabulary (see classify_axis_labels): an unrecognised label is
    # treated as spatial.
    t_idx, channel_indices, spatial_indices = classify_axis_labels(axes)

    n_t = shape[t_idx] if t_idx is not None else 1
    channel_shape = tuple(shape[i] for i in channel_indices)
    channel_axes = [axes[i] for i in channel_indices]
    n_c = 1
    for size in channel_shape:
        n_c *= size

    spatial_shape = tuple(shape[i] for i in spatial_indices)
    spatial_axes = [axes[i] for i in spatial_indices]

    return OMEZarrInfo(
        axes=axes,
        shape=shape,
        n_timepoints=n_t,
        n_channels=n_c,
        channel_axes=channel_axes,
        channel_shape=channel_shape,
        spatial_shape=spatial_shape,
        spatial_axes=spatial_axes,
        path=path,
    )


def _heuristic_ome_info(shape: Tuple[int, ...], ndim: int, path: Path) -> OMEZarrInfo:
    """Fallback OME info based on shape heuristics."""
    if ndim == 5:
        # Assume TCZYX
        return OMEZarrInfo(
            axes=["t", "c", "z", "y", "x"],
            shape=shape,
            n_timepoints=shape[0],
            n_channels=shape[1],
            channel_axes=["c"],
            channel_shape=(shape[1],),
            spatial_shape=shape[2:],
            spatial_axes=["z", "y", "x"],
            path=path,
        )
    elif ndim == 4:
        # Assume CZYX (could be TZYX — user can override)
        return OMEZarrInfo(
            axes=["c", "z", "y", "x"],
            shape=shape,
            n_timepoints=1,
            n_channels=shape[0],
            channel_axes=["c"],
            channel_shape=(shape[0],),
            spatial_shape=shape[1:],
            spatial_axes=["z", "y", "x"],
            path=path,
        )
    elif ndim == 3:
        return OMEZarrInfo(
            axes=["z", "y", "x"],
            shape=shape,
            n_timepoints=1,
            n_channels=1,
            channel_axes=[],
            channel_shape=(),
            spatial_shape=shape,
            spatial_axes=["z", "y", "x"],
            path=path,
        )
    elif ndim == 2:
        return OMEZarrInfo(
            axes=["y", "x"],
            shape=shape,
            n_timepoints=1,
            n_channels=1,
            channel_axes=[],
            channel_shape=(),
            spatial_shape=shape,
            spatial_axes=["y", "x"],
            path=path,
        )
    else:
        # Generic nD — all spatial
        axes = [f"dim{i}" for i in range(ndim)]
        return OMEZarrInfo(
            axes=axes,
            shape=shape,
            n_timepoints=1,
            n_channels=1,
            channel_axes=[],
            channel_shape=(),
            spatial_shape=shape,
            spatial_axes=axes,
            path=path,
        )
