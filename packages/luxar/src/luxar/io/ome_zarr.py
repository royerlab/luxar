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

import numpy as np

from luxar.io.volume import _find_all_arrays

__all__ = [
    "OMEZarrInfo",
    "discover_ome_zarr_shape",
]


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

    store = zarr.open(str(path), mode="r")

    # Navigate to the group/array
    if isinstance(store, zarr.Array):
        arr = store
        attrs: Dict[str, Any] = dict(getattr(store, "attrs", {}))
    elif isinstance(store, zarr.Group):
        attrs = dict(store.attrs)
        if array_key is not None:
            # User-specified array key (may be nested, e.g. "h2afva/fused")
            try:
                arr = store[array_key]
            except KeyError:
                available = list(store.keys())
                raise ValueError(
                    f"Array key '{array_key}' not found in {path}. "
                    f"Available keys: {available}"
                )
        elif "0" in store:
            # OME-NGFF standard: resolution level "0" is highest resolution
            arr = store["0"]
        else:
            # Find the largest array, searching recursively into sub-groups
            arrays = _find_all_arrays(store)
            if not arrays:
                raise ValueError(f"No arrays found in zarr group: {path}")
            # Pick the array with the most elements
            arr = max(arrays, key=lambda kv: int(np.prod(kv[1].shape)))[1]
    else:
        raise ValueError(f"Unexpected zarr object type: {type(store)}")

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
    multiscales = attrs.get("multiscales")
    if multiscales and isinstance(multiscales, list) and len(multiscales) > 0:
        ms = multiscales[0]
        return _parse_ngff_metadata(ms, shape, path, store)

    # Try custom axes attribute (e.g. Keller-lab zarr.zip files store
    # axes = ['time', 'camera', 'channel', 'z', 'y', 'x'])
    custom_axes = attrs.get("axes")
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
    _SPATIAL = {"z", "y", "x", "depth", "height", "width"}
    _TIME = {"time", "t"}
    _CHANNEL = {"channel", "c", "ch"}
    _CAMERA = {"camera", "cam", "view", "angle"}

    t_idx: Optional[int] = None
    channel_indices: List[int] = []  # channel + camera axes
    spatial_indices: List[int] = []

    for i, ax in enumerate(axes):
        ax_l = ax.lower()
        if ax_l in _TIME:
            t_idx = i
        elif ax_l in _CHANNEL or ax_l in _CAMERA:
            channel_indices.append(i)
        elif ax_l in _SPATIAL:
            spatial_indices.append(i)
        else:
            # Unknown axis — treat as spatial
            spatial_indices.append(i)

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
