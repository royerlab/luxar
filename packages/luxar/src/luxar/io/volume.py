"""Multi-format volume loading (domain layer).

Loads a spatial volume from ``.npy`` / ``.npz`` / ``.zarr`` / ``.zarr.zip`` /
``.tiff`` / imageio-supported files, with OME-Zarr-aware positional slicing and
an explicit ``--axes`` override. This is reusable domain logic (no CLI/Typer
coupling): missing optional readers raise :class:`ImportError` with an install
hint, which the CLI converts to a clean exit. Previously lived in
``luxar.cli.gsplat_config``; moved here so domain code (e.g. the denoise
pipeline) no longer imports upward into the CLI layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from arbol import aprint

__all__ = [
    "decode_flat_channel_index",
    "load_volume",
]


def decode_flat_channel_index(
    channel: int, channel_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Decode a flat channel task index into folded channel-axis coordinates.

    For data with multiple non-spatial, channel-like axes (for example
    ``camera`` and ``channel``), batch planning treats each axis combination as
    one flat channel task. This helper uses row-major order to map the flat
    index back to per-axis coordinates.
    """
    if channel < 0:
        raise ValueError(f"channel index must be non-negative, got {channel}")
    if not channel_shape:
        if channel == 0:
            return ()
        raise ValueError("channel index > 0 is invalid when there are no channel axes")

    total = 1
    for size in channel_shape:
        if size <= 0:
            raise ValueError(
                f"channel axis sizes must be positive, got {channel_shape}"
            )
        total *= size
    if channel >= total:
        raise ValueError(
            f"flat channel index {channel} is out of range for channel_shape={channel_shape} "
            f"(total={total})"
        )

    coords: List[int] = []
    remaining = channel
    for dim_size in reversed(channel_shape):
        coords.insert(0, remaining % dim_size)
        remaining //= dim_size
    return tuple(coords)


def _apply_axes_spec(
    arr: np.ndarray,
    axes: str,
    channel: Optional[int],
    timepoint: Optional[int],
) -> np.ndarray:
    """Collapse a non-canonically-ordered nD array to its spatial volume.

    ``axes`` is a comma-separated label per array dimension (e.g.
    ``"z,c,y,x"`` or ``"t,z,y,x"``). Recognised: time (``t``/``time``),
    channel (``c``/``channel``/``ch``/``camera``/``cam``), spatial
    (``z``/``y``/``x``/``depth``/``height``/``width``). Each time/channel axis is
    indexed (by ``timepoint``/``channel``, default 0) and dropped; the remaining
    spatial axes are kept in their given order. This is the single-volume
    counterpart of ``batch-fit submit --axes`` — it lets ``fit``/``cal`` consume
    data whose axis order isn't the assumed TCZYX/CZYX/ZYX.
    """
    labels = [a.strip().lower() for a in axes.split(",") if a.strip() != ""]
    if len(labels) != arr.ndim:
        raise ValueError(
            f"--axes has {len(labels)} labels but the array is {arr.ndim}D "
            f"(shape {arr.shape}); give one label per dimension."
        )

    def _kind(label: str) -> str:
        if label in ("t", "time"):
            return "t"
        if label in ("c", "channel", "ch", "camera", "cam"):
            return "c"
        if label in ("z", "y", "x", "depth", "height", "width"):
            return "s"
        raise ValueError(
            f"--axes label {label!r} not recognised; use time/t, "
            "channel/c/ch/camera/cam, or z/y/x (depth/height/width)."
        )

    kinds = [_kind(label) for label in labels]
    index: list = [slice(None)] * arr.ndim
    for i, k in enumerate(kinds):
        if k in ("t", "c"):
            which, idx = (
                ("--timepoint", timepoint) if k == "t" else ("--channel", channel)
            )
            idx = 0 if idx is None else int(idx)
            size = arr.shape[i]
            if not (0 <= idx < size):
                raise ValueError(
                    f"{which} index {idx} is out of range for the '{labels[i]}' "
                    f"axis of size {size} (valid 0..{size - 1})."
                )
            index[i] = idx
    return np.asarray(arr[tuple(index)])


def load_volume(
    path: Path,
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
) -> np.ndarray:
    """Load a volume from various file formats.

    Supported formats:
        .npy         — NumPy binary (numpy, base dep)
        .npz         — NumPy compressed (numpy, base dep)
        .zarr        — Zarr array/group, including OME-ZARR 5D (zarr, base dep)
        .tiff / .tif — TIFF image (tifffile, optional: pip install luxar[io])
        other        — Fallback via imageio (optional: pip install luxar[io])

    Args:
        path: Path to the volume file
        channel: Channel index for 4D/5D+ OME-ZARR data. If None, defaults
            to 0 when slicing is needed; for 4D arrays, ``None`` returns
            the array as-is.
        timepoint: Timepoint index for 5D+ OME-ZARR data. If None, defaults
            to 0 when slicing is needed; for 4D arrays, ``None`` returns
            the array as-is.
        array_key: Array key within .npz or .zarr files
        axes: Explicit per-dimension axis labels (e.g. ``"z,c,y,x"``) overriding
            the positional TCZYX/CZYX/ZYX heuristic — for data whose axis order
            differs. Time/channel axes are sliced (by ``timepoint``/``channel``)
            and dropped; spatial axes are kept in the given order.

    Returns:
        Volume as float32 numpy array (>=2D)

    Raises:
        ImportError: If an optional reader (tifffile/imageio) is required but
            not installed. Callers that want a clean CLI exit should catch this.
        ValueError: On an invalid key or a sub-2D volume.
    """
    suffix = path.suffix.lower()

    if suffix == ".npy":
        aprint(f"Loading NumPy array: {path.name}")
        volume = np.load(str(path))

    elif suffix == ".npz":
        aprint(f"Loading NumPy archive: {path.name}")
        with np.load(str(path)) as npz:
            keys = list(npz.keys())
            if array_key:
                if array_key not in keys:
                    raise ValueError(
                        f"Key '{array_key}' not found in {path.name}. "
                        f"Available keys: {keys}"
                    )
                volume = np.array(npz[array_key])
            else:
                volume = np.array(npz[keys[0]])
                if len(keys) > 1:
                    aprint(f"  Using first array '{keys[0]}' (available: {keys})")

    elif suffix == ".zarr" or (suffix == ".zip" and path.stem.endswith(".zarr")):
        # Handles both plain .zarr directories and .zarr.zip archives.
        # zarr natively supports ZipStore so no extraction needed. With an
        # explicit --axes the raw array is loaded and sliced by _apply_axes_spec
        # below (bypassing the positional TCZYX/CZYX heuristic).
        volume = _load_zarr_volume(
            path, channel, timepoint, array_key, raw=axes is not None
        )

    elif suffix in (".tiff", ".tif"):
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError(
                "Loading TIFF files requires tifffile: pip install luxar[io]"
            ) from exc
        aprint(f"Loading TIFF: {path.name}")
        volume = tifffile.imread(str(path))

    else:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError(
                f"Cannot load '{suffix}' files — imageio not installed: "
                "pip install luxar[io]"
            ) from exc
        aprint(f"Loading via imageio: {path.name}")
        volume = iio.imread(str(path))

    # Explicit axis spec (overrides the positional heuristic): slice/drop the
    # time & channel axes and keep the spatial axes in the given order.
    if axes is not None:
        # Pass `volume` as-is (a lazy zarr array for .zarr inputs) so
        # _apply_axes_spec slices the time/channel axes BEFORE materializing —
        # do NOT np.asarray() here or a huge nD movie loads fully into RAM.
        volume = _apply_axes_spec(volume, axes, channel, timepoint)
        # The spec already fixed the shape (time/channel dropped, spatial kept) —
        # do NOT squeeze, or a deliberately-kept size-1 spatial axis (e.g. a
        # single z-plane via --axes z,y,x) would be silently dropped.
        volume = np.asarray(volume, dtype=np.float32)
    else:
        # Post-process: drop incidental size-1 dims from the positional heuristic.
        volume = np.asarray(volume, dtype=np.float32)
        volume = np.squeeze(volume)

    if volume.ndim < 2:
        raise ValueError(
            f"Volume must be at least 2D after squeezing, got {volume.ndim}D "
            f"with shape {volume.shape}"
        )

    aprint(f"  Shape: {volume.shape}, dtype: float32")
    return volume


def _find_all_arrays(group: Any, prefix: str = "") -> list:
    """Recursively find all arrays in a zarr group, returning (key_path, array) pairs."""
    import zarr

    results = []
    for k in group.keys():
        item = group[k]
        key_path = f"{prefix}/{k}" if prefix else k
        if isinstance(item, zarr.Array):
            results.append((key_path, item))
        elif isinstance(item, zarr.Group):
            results.extend(_find_all_arrays(item, key_path))
    return results


def _load_zarr_volume(
    path: Path,
    channel: Optional[int],
    timepoint: Optional[int],
    array_key: Optional[str],
    raw: bool = False,
) -> np.ndarray:
    """Load a volume from a zarr store, handling OME-ZARR conventions.

    With ``raw=True`` the full array is returned WITHOUT the positional
    TCZYX/CZYX slicing — the caller (``load_volume`` with an explicit ``--axes``)
    applies its own axis spec instead.
    """
    import zarr

    aprint(f"Loading Zarr: {path.name}")
    store = zarr.open(str(path), mode="r")

    # Navigate to the target array
    if isinstance(store, zarr.Array):
        arr = store
    elif isinstance(store, zarr.Group):
        if array_key is not None:
            try:
                arr = store[array_key]
            except KeyError:
                available = list(store.keys())
                raise ValueError(
                    f"Array key '{array_key}' not found in {path}. "
                    f"Available keys: {available}"
                )
            aprint(f"  Using array '{array_key}'")
        elif "0" in store:
            # OME-ZARR convention: "0" is highest resolution
            aprint("  Detected OME-ZARR layout (using resolution level '0')")
            arr = store["0"]
        else:
            # Find the largest array in the group, searching recursively
            # into sub-groups (e.g. h2afva/fused, mezzo/fused).
            arrays = _find_all_arrays(store)
            if not arrays:
                raise ValueError(f"No arrays found in zarr group: {path}")
            best_key = max(arrays, key=lambda kv: int(np.prod(kv[1].shape)))[0]
            arr = store[best_key]
            aprint(f"  Using array '{best_key}'")
    else:
        raise ValueError(f"Unexpected zarr object type: {type(store)}")

    shape = arr.shape
    ndim = len(shape)
    aprint(f"  Raw array shape: {shape} ({ndim}D)")

    if raw:
        # Explicit --axes path: hand back the LAZY zarr array (NOT np.array(arr)) so
        # the caller's _apply_axes_spec slices the requested timepoint/channel BEFORE
        # materializing — otherwise a whole nD movie (e.g. a 329-timepoint stack,
        # >1 TiB) would be loaded into RAM just to extract one 3D volume.
        return arr  # type: ignore[no-any-return]

    # Slice the array down to a 2D/3D spatial volume.
    # For nD data where ndim > 5, consume leading dimensions using
    # timepoint and channel indices (defaulting to 0 for each).
    if ndim >= 6:
        # Generic >5D: treat first dim as T, fold all leading non-spatial
        # dimensions before the final 3 spatial axes into one flat channel index.
        t = timepoint if timepoint is not None else 0
        remaining_non_spatial = ndim - 4  # -1 for time, -3 for spatial
        channel_shape = tuple(shape[1 : 1 + remaining_non_spatial])
        if channel is None:
            channel_coords = tuple(0 for _ in channel_shape)
        else:
            channel_coords = decode_flat_channel_index(channel, channel_shape)
        idx = [t, *channel_coords]
        aprint(f"  Slicing {ndim}D: indices {idx} → 3D spatial")
        volume = np.array(arr[tuple(idx)])
    elif ndim == 5:
        t = timepoint if timepoint is not None else 0
        c = channel if channel is not None else 0
        aprint(f"  Slicing 5D (TCZYX): T={t}, C={c}")
        volume = np.array(arr[t, c, :, :, :])
    elif ndim == 4:
        if channel is not None:
            aprint(f"  Slicing 4D (CZYX): C={channel}")
            volume = np.array(arr[channel, :, :, :])
        elif timepoint is not None:
            aprint(f"  Slicing 4D (TZYX): T={timepoint}")
            volume = np.array(arr[timepoint, :, :, :])
        else:
            aprint("  4D array — using as-is (use --channel or --timepoint to slice)")
            volume = np.array(arr)
    else:
        volume = np.array(arr)

    return volume
