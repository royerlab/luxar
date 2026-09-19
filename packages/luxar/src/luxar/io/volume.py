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

from luxar._zarr_compat import is_zarr_path

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


def _axis_kind(label: str, flag: str = "--axes") -> str:
    """Classify one axis label as time, channel or spatial.

    ``flag`` names the option being parsed, so a typo is reported against the
    option the user actually typed: this vocabulary is shared by ``--axes`` and
    ``--target-axes``, which mean different things.
    """
    if label in ("t", "time"):
        return "t"
    if label in ("c", "channel", "ch", "camera", "cam"):
        return "c"
    if label in ("z", "y", "x", "depth", "height", "width"):
        return "s"
    raise ValueError(
        f"{flag} label {label!r} not recognised; use time/t, "
        "channel/c/ch/camera/cam, or z/y/x (depth/height/width)."
    )


def open_volume_lazy(path: Path, array_key: Optional[str] = None) -> Any:
    """Open a volume WITHOUT materialising it, when the format allows.

    A zarr store is returned as its array object, so a caller that only ever
    slices it (a per-timepoint or per-tile volume re-fit) never pays for the
    whole dataset: a 253-timepoint 407x2048x2048 uint16 timelapse is 431 GB while
    one timepoint is 3.4 GB. Formats with no lazy reader (.npy/.npz/.tiff) fall
    back to :func:`load_volume`; those are single volumes by nature, so the
    fallback is not the case this exists for.

    The returned object supports ``.shape`` and numpy basic indexing — the
    contract :func:`~luxar.gsplats.lod.volume_regions.select_sub_volume` needs.

    Array selection within a group is literally the SAME code as
    :func:`load_volume` / :func:`~luxar.io.ome_zarr.discover_ome_zarr_shape` use
    (:func:`_select_zarr_array`) — explicit ``array_key`` (blank counts as
    absent), else the OME-NGFF resolution level ``"0"``, else the largest array
    found recursively. It has to be: the caller re-opens a store some other
    command already read the shape of, and a different choice here would silently
    re-fit against a different (e.g. downsampled) array.
    """
    if is_zarr_path(path):
        import zarr

        from luxar._zarr_compat import open_store

        # This branch explicitly accepts `.zarr.zip`, so the store has to be
        # dispatched explicitly: zarr 2 sniffed the suffix inside
        # `normalize_store_arg`, zarr 3 does not, and a bare
        # `zarr.open(str(path))` on an archive raises GroupNotFoundError.
        node = zarr.open(store=open_store(path, mode="r"), mode="r")
        return _select_zarr_array(node, path, array_key)[0]
    return load_volume(path, array_key=array_key)


class _PinnedAxes:
    """Lazy view of an array-like with some axes fixed to a single index.

    A zarr array MATERIALISES on ``__getitem__``, so pre-slicing away the axes a
    caller does not need (a channel it selected, a time axis it is not walking)
    would defeat the whole point of :func:`open_volume_lazy`. This defers the
    pinned indices into every read instead.

    Only ``.shape`` and numpy basic indexing are provided — the contract
    :func:`~luxar.gsplats.lod.volume_regions.select_sub_volume` needs.
    """

    def __init__(self, base: Any, pins: dict) -> None:
        shape = tuple(int(s) for s in base.shape)
        for axis, index in pins.items():
            if not 0 <= int(axis) < len(shape):
                raise ValueError(f"pinned axis {axis} out of range for {shape}")
            if not 0 <= int(index) < shape[int(axis)]:
                raise ValueError(
                    f"pinned index {index} out of range on axis {axis} "
                    f"(extent {shape[int(axis)]})"
                )
        self._base = base
        self._pins = {int(a): int(i) for a, i in pins.items()}
        self._free = [i for i in range(len(shape)) if i not in self._pins]
        self.shape = tuple(shape[i] for i in self._free)
        self.dtype = getattr(base, "dtype", None)

    def __getitem__(self, key: Any) -> Any:
        idx = key if isinstance(key, tuple) else (key,)
        full: List[Any] = [slice(None)] * (len(self._free) + len(self._pins))
        for axis, index in self._pins.items():
            full[axis] = index
        for position, entry in zip(self._free, idx):
            full[position] = entry
        return self._base[tuple(full)]


def pin_volume_axes(volume: Any, pins: dict) -> Any:
    """Fix ``{axis: index}`` of ``volume``, dropping those axes, without reading.

    Returns ``volume`` unchanged when there is nothing to pin, so the common case
    keeps handing the underlying store straight through.
    """
    return _PinnedAxes(volume, pins) if pins else volume


def volume_axes_from_spec(
    axes: str, ndim: int, *, flag: str = "--target-axes"
) -> tuple:
    """Map each SPLAT center dim to the volume axis that holds it.

    The counterpart of :func:`_apply_axes_spec` for the case where a non-spatial
    axis must be KEPT rather than sliced away: a volume re-fit of a stacked
    timelapse walks the barrier axis itself, one slice per group.

    Luxar's fitted splats order their center columns spatial-first (in the
    array's own spatial order) with stacked time/channel dims LAST, while a
    source array is usually the other way round (``t, z, y, x``). So the mapping
    is the spatial axis positions followed by the stacked ones:
    ``"t,z,y,x"`` gives ``(1, 2, 3, 0)``.
    """
    labels = [a.strip().lower() for a in axes.split(",") if a.strip() != ""]
    if len(labels) != ndim:
        raise ValueError(
            f"{flag} has {len(labels)} labels but the splats are {ndim}D; "
            "give one label per dimension."
        )
    kinds = [_axis_kind(label, flag) for label in labels]
    spatial = [i for i, k in enumerate(kinds) if k == "s"]
    stacked = [i for i, k in enumerate(kinds) if k in ("t", "c")]
    return tuple(spatial + stacked)


def _apply_axes_spec(
    arr: np.ndarray,
    axes: str,
    channel: Optional[int],
    timepoint: Optional[int],
    region: Optional[Tuple[slice, ...]] = None,
) -> np.ndarray:
    """Collapse a non-canonically-ordered nD array to its spatial volume.

    ``axes`` is a comma-separated label per array dimension (e.g.
    ``"z,c,y,x"`` or ``"t,z,y,x"``). Recognised: time (``t``/``time``),
    channel (``c``/``channel``/``ch``/``camera``/``cam``), spatial
    (``z``/``y``/``x``/``depth``/``height``/``width``). Time axes are indexed by
    ``timepoint``; a flat ``channel`` index is decoded across all channel-like axes
    in row-major order. Those axes are dropped and the remaining spatial axes stay
    in their given order. This is the single-volume counterpart of ``batch-fit
    submit --axes`` — it lets ``fit``/``cal`` consume data whose axis order isn't
    the assumed TCZYX/CZYX/ZYX.
    """
    labels = [a.strip().lower() for a in axes.split(",") if a.strip() != ""]
    if len(labels) != arr.ndim:
        raise ValueError(
            f"--axes has {len(labels)} labels but the array is {arr.ndim}D "
            f"(shape {arr.shape}); give one label per dimension."
        )

    kinds = [_axis_kind(label) for label in labels]
    if sum(kind == "t" for kind in kinds) > 1:
        raise ValueError(
            f"--axes {axes!r} names more than one time axis; one --timepoint "
            "cannot index them independently."
        )
    channel_indices = [i for i, kind in enumerate(kinds) if kind == "c"]
    channel_shape = tuple(arr.shape[i] for i in channel_indices)
    flat_channel = 0 if channel is None else int(channel)
    try:
        channel_coords = (
            decode_flat_channel_index(flat_channel, channel_shape)
            if channel_shape
            else ()
        )
    except ValueError as exc:
        channel_axes = ", ".join(f"{labels[i]}={arr.shape[i]}" for i in channel_indices)
        raise ValueError(
            f"--channel index {flat_channel} is invalid for channel-like axes "
            f"{channel_axes}: {exc}"
        ) from exc
    channel_coord_by_axis = dict(zip(channel_indices, channel_coords))
    index: list = [slice(None)] * arr.ndim
    spatial_indices = [i for i, kind in enumerate(kinds) if kind == "s"]
    if region is not None:
        if len(region) != len(spatial_indices):
            raise ValueError(
                f"region has {len(region)} axes but --axes {axes!r} names "
                f"{len(spatial_indices)} spatial axes"
            )
        for axis, span in zip(spatial_indices, region, strict=True):
            index[axis] = span
    for i, k in enumerate(kinds):
        if k in ("t", "c"):
            which, idx = (
                ("--timepoint", timepoint)
                if k == "t"
                else ("--channel", channel_coord_by_axis[i])
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


def _record_source_dtype(info: Optional[dict], volume: Any) -> None:
    """Record the STORED element type of ``volume`` into ``info``, if requested.

    Must be called BEFORE the loader's float32 cast — the last point at which
    the on-disk element type still exists (it is the honest denominator of any
    compression ratio quoted about a fit of this volume). Reads
    ``getattr(volume, "dtype", None)`` rather than ``np.asarray(volume).dtype``
    so a lazy zarr array is not materialized just to be measured.
    """
    if info is None:
        return
    src_dtype = getattr(volume, "dtype", None)
    if src_dtype is not None:
        info["source_dtype"] = str(np.dtype(src_dtype))


def _slice_eager_region(
    volume: Any, path: Path, region: Optional[Tuple[slice, ...]]
) -> Any:
    """Apply ``region`` only when the reader could not do it lazily."""
    if region is None or is_zarr_path(path):
        return volume
    return np.asarray(volume[region], dtype=np.float32)


def _squeezed_region_spans(
    shape: Tuple[int, ...], region: Optional[Tuple[slice, ...]]
) -> Tuple[slice, ...]:
    """Map post-squeeze spatial spans back onto the reader-visible axes."""
    if region is None:
        return (slice(None),) * len(shape)
    surviving_axes = [axis for axis, size in enumerate(shape) if size != 1]
    if len(region) != len(surviving_axes):
        raise ValueError(
            f"region has {len(region)} axes but positional loading keeps "
            f"{len(surviving_axes)} axes after squeezing shape {shape}"
        )
    spans = [slice(None)] * len(shape)
    for axis, span in zip(surviving_axes, region, strict=True):
        spans[axis] = span
    return tuple(spans)


def load_volume(
    path: Path,
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
    info: Optional[dict] = None,
    region: Optional[Tuple[slice, ...]] = None,
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
        channel: Channel index for 4D/5D+ OME-ZARR data. With ``axes``, this is
            a flat row-major index across every channel-like axis. If None,
            defaults to 0 when slicing is needed; for 4D arrays without
            ``axes``, ``None`` returns the array as-is.
        timepoint: Timepoint index for 5D+ OME-ZARR data. If None, defaults
            to 0 when slicing is needed; for 4D arrays, ``None`` returns
            the array as-is.
        array_key: Array key within .npz or .zarr files
        axes: Explicit per-dimension axis labels (e.g. ``"z,c,y,x"``) overriding
            the positional TCZYX/CZYX/ZYX heuristic — for data whose axis order
            differs. The single time axis is sliced by ``timepoint``; the flat
            ``channel`` index is decoded across all channel-like axes. Those
            axes are dropped and spatial axes are kept in the given order.
        info: Optional dict, populated with ``source_dtype`` — the element type
            of the array AS STORED, captured before the float32 cast below.
            This is the only place it is knowable: the returned array is always
            float32, so a consumer that wants to quote a size (e.g. the
            denominator of a compression ratio) would otherwise describe the
            working copy and overstate it by the cast's inflation factor —
            exactly 2x for the 16-bit acquisitions most microscopy produces.
        region: Optional spatial slices applied before materializing a zarr array.
            Non-zarr formats are sliced after loading.

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

    elif is_zarr_path(path):
        # Handles both plain .zarr directories and .zarr.zip archives.
        # zarr natively supports ZipStore so no extraction needed. With an
        # explicit --axes the raw array is loaded and sliced by _apply_axes_spec
        # below (bypassing the positional TCZYX/CZYX heuristic).
        volume = _load_zarr_volume(
            path,
            channel,
            timepoint,
            array_key,
            raw=axes is not None,
            region=None if axes is not None else region,
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

    # Record the STORED element type before the float32 cast below — the last
    # point at which it exists.
    _record_source_dtype(info, volume)

    # Explicit axis spec (overrides the positional heuristic): slice/drop the
    # time & channel axes and keep the spatial axes in the given order.
    if axes is not None:
        # Pass `volume` as-is (a lazy zarr array for .zarr inputs) so
        # _apply_axes_spec slices the time/channel axes BEFORE materializing —
        # do NOT np.asarray() here or a huge nD movie loads fully into RAM.
        volume = _apply_axes_spec(volume, axes, channel, timepoint, region)
        # The spec already fixed the shape (time/channel dropped, spatial kept) —
        # do NOT squeeze, or a deliberately-kept size-1 spatial axis (e.g. a
        # single z-plane via --axes z,y,x) would be silently dropped.
        volume = np.asarray(volume, dtype=np.float32)
    else:
        # Post-process: drop incidental size-1 dims from the positional heuristic.
        # The second asarray is a no-op at runtime (already float32, no copy); it
        # is there because np.squeeze is typed as returning Any.
        volume = np.asarray(volume, dtype=np.float32)
        volume = np.asarray(np.squeeze(volume), dtype=np.float32)
        volume = _slice_eager_region(volume, path, region)

    if volume.ndim < 2:
        raise ValueError(
            f"Volume must be at least 2D after squeezing, got {volume.ndim}D "
            f"with shape {volume.shape}"
        )

    assert isinstance(volume, np.ndarray)
    aprint(f"  Shape: {volume.shape}, dtype: float32")
    return volume


def _find_all_arrays(
    group: Any, prefix: str = "", skip: "frozenset[str]" = frozenset()
) -> List[Tuple[str, Any, Any]]:
    """Recursively find all arrays in a zarr group.

    Returns ``(key_path, array, owner_group)`` triples — the owning group comes
    for free here (this function holds it open), sparing the caller a second
    metadata request to re-open it. Keys are walked SORTED rather than in
    ``keys()`` order, which zarr does not guarantee to be stable; see
    :func:`_pick_largest` for why that matters.

    ``skip`` names subgroup keys the walk must not descend into, AT EVERY DEPTH
    (``{"labels"}`` for an image group: NGFF puts a mask at
    ``<image>/labels/<name>/<level>``, and nothing says an image's levels cannot
    themselves be nested, so a first-level-only filter would still reach a mask
    one group further down).
    """
    import zarr

    results: List[Tuple[str, Any, Any]] = []
    for k in sorted(group.keys()):
        if k in skip:
            continue
        item = group[k]
        key_path = f"{prefix}/{k}" if prefix else k
        if isinstance(item, zarr.Array):
            results.append((key_path, item, group))
        elif isinstance(item, zarr.Group):
            results.extend(_find_all_arrays(item, key_path, skip))
    return results


def _largest_first(candidate: Tuple[str, Any, Any]) -> Tuple[int, str]:
    """Sort key ordering candidates largest-first, ties broken on the key path.

    Split out of :func:`_pick_largest` so a caller that already knows its list is
    non-empty can ``min()`` directly and get a non-``Optional`` result, without a
    second copy of the ordering.
    """
    return (-int(np.prod(candidate[1].shape)), candidate[0])


def _pick_largest(candidates: list) -> Optional[Tuple[str, Any, Any]]:
    """The candidate with the most elements; the LOWEST key path breaks a tie.

    ``None`` for an empty list. The tie-break is what makes selection
    REPRODUCIBLE: zarr gives no GUARANTEE that ``Group.keys()`` yields members in
    a stable order (it depends on the store implementation, and a remote or
    archive store need not enumerate the way a directory does), so a plain
    ``max()`` over two equal-sized candidates could answer differently for the
    same store — and the batch planner reading the shape and a per-tile worker
    re-opening it are different processes. Sorting by ``(-element_count,
    key_path)`` is a total order, so they cannot disagree whatever the
    enumeration does. Defensive rather than a fix for an observed flip.
    """
    if not candidates:
        return None
    best: Tuple[str, Any, Any] = min(candidates, key=_largest_first)
    return best


def _largest_array(group: Any, prefix: str = "") -> Optional[Tuple[str, Any, Any]]:
    """The ``(key_path, array, owner)`` with the most elements under ``group``.

    Searches recursively. ``None`` when there is no array anywhere below it.
    ``prefix`` is prepended to every key path, so a search scoped to a SUBGROUP
    still reports paths relative to the store root.
    """
    return _pick_largest(_find_all_arrays(group, prefix))


def _declared_levels(
    group: Any,
    prefix: str,
    report: bool = True,
    skip_groups: frozenset[str] = frozenset(),
) -> List[Tuple[str, Any, Any]]:
    """The arrays ``group``'s own NGFF ``multiscales`` block names as its levels.

    ``datasets[*]["path"]``, resolved relative to ``group``. A declared path that
    is missing, is not an array, or is not even a legal zarr path (``".."``
    segments, a null byte, a segment longer than a filename — all of which zarr 3
    rejects with something other than ``KeyError``) is skipped — and REPORTED,
    because the survivors of a half-written block can perfectly well be a
    downsampled level, and a degraded selection that says nothing is
    indistinguishable from a correct one.

    ``report=False`` silences that line for a caller merely ASKING whether a
    block names an array (:func:`_declares_array`), which walks candidate groups
    whose blocks are expected not to match: a stale declaration there explains
    why that block lost and is not a degraded selection to warn about.

    A declared path is NORMALISED before it is looked up
    (:func:`~luxar.io.ome_zarr._normalised_dataset_path`, the same rule the
    metadata-MATCHING side uses, so both halves apply the same rule to every
    spelling). ``"./0"`` is a legal NGFF spelling of ``"0"`` and writers do emit
    it, while zarr REFUSES a path holding a ``.`` segment outright — so the raw
    spelling resolved no level at all, the block stopped being evidence about the
    array it declares (:func:`_declares_array`), and a 4D ``TZYX`` image group
    fell onto the ``CZYX`` shape heuristic. Although NGFF requires a string, real
    numeric scalars are deliberately recoverable; other non-string values name
    nothing. Using one rule prevents lookup and metadata selection from
    disagreeing about which declaration names the chosen array.

    ``skip_groups`` excludes a declared candidate when any NON-LEAF path segment
    names one of those groups. The leaf is deliberately exempt: an array itself
    may legitimately have the same name as a reserved subgroup. It is asked of
    the spelling the lookup uses — the NORMALISED one, or the SLASH-STRIPPED
    original when normalisation yields nothing — and neither can weaken the
    exclusion: they never ADD a non-leaf segment or move the leaf (which is
    ``split("/")[-1]`` under every spelling), and only DROP a leading ``"."`` or
    an empty segment (``".//0"`` checks ``{"", "."}`` raw and ``{}`` normalised).
    No caller skips a group named ``"."`` or ``""``, so ``"./labels/0"`` stays
    excluded exactly as ``"labels/0"`` is.

    Every path this REPORTS — the ``Skipping declared level`` diagnostic and the
    returned key alike — is built from the spelling the lookup actually used, so
    one entry can never be described two ways. For a level that resolves that is
    the canonical spelling (``"0/0"``, never ``"0/./0"``), which matters because
    the key is what :func:`_select_zarr_array` hands back as "the array's path
    relative to the store root": :func:`load_volume` PRINTS it as the array in
    use and :func:`~luxar.io.ome_zarr._relative_key` re-expresses it against the
    owner, so a spelling zarr itself refuses would advertise an ``array_key`` the
    very next command rejects. Two stores whose declarations NORMALISE ALIKE
    therefore select, log and match identically, however they spell themselves.
    Only a declaration still holding a ``.`` SEGMENT (``"."``, ``"a/./b"``) falls
    back to the SLASH-STRIPPED original, and for the same reason: normalisation
    yields nothing for it, so that is the spelling the lookup used. Those are not
    normalised alike and do not log alike — ``"a/./b"`` reports ``'0/a/./b'``
    while the POSIX-equivalent ``"./a/./b"`` reports ``'0/./a/./b'`` — but neither
    names a level under any spelling. Everything else is quoted canonically
    whether or not it resolves: a merely MISSING ``"./7"`` reads as ``"7"`` in the
    diagnostic, which is the key that was looked up and the one to go check.

    The reported owner is ``group`` ITSELF, not the level's immediate parent: the
    owner is used as "the node carrying the metadata that describes this array",
    and it is ``group`` that declared it. For a level nested one deeper
    (``datasets[0].path == "res/0"``) the immediate parent declares nothing.

    Both OME-Zarr layouts are read — 0.4's top-level ``multiscales`` and 0.5's
    block nested under ``ome`` — through the one resolver every reader of NGFF
    attributes goes through. Spelling ``attrs["multiscales"]`` here instead would
    make WHICH ARRAY IS SELECTED depend on the OME-Zarr version: a 0.5 image group
    would fall past this branch and could answer with a bigger undeclared sibling
    where the identical 0.4 store answers with its declared level.
    """
    import zarr

    from luxar.io.ome_zarr import _normalised_dataset_path, resolve_ngff_attrs

    block = resolve_ngff_attrs(group.attrs).get("multiscales")
    if not (isinstance(block, list) and block and isinstance(block[0], dict)):
        return []
    datasets = block[0].get("datasets")
    if not isinstance(datasets, list):
        return []
    results: List[Tuple[str, Any, Any]] = []
    for entry in datasets:
        raw = entry.get("path") if isinstance(entry, dict) else None
        # A string retaining a `.` segment cannot resolve, but keep its own
        # spelling so the failed lookup is reported instead of disappearing.
        rel = _normalised_dataset_path(raw) or (
            raw.strip("/") if isinstance(raw, str) else ""
        )
        if not rel:
            continue
        if not skip_groups.isdisjoint(rel.split("/")[:-1]):
            continue
        declared_path = f"{prefix}/{rel}" if prefix else rel
        try:
            item = group[rel]
        except (KeyError, ValueError, OSError, TypeError) as e:
            if report:
                aprint(
                    f"  Skipping declared level '{declared_path}': "
                    f"{type(e).__name__}: {e}"
                )
            continue
        if isinstance(item, zarr.Array):
            results.append((declared_path, item, group))
        elif report:
            # Resolving to a GROUP is a dropped level exactly like a failed
            # lookup, and was the only declaration that RESOLVED and was then
            # discarded silently — the store degrades to the ndim heuristic with
            # no per-level explanation of why its own declaration was not used.
            aprint(
                f"  Skipping declared level '{declared_path}': "
                f"names a {type(item).__name__}, not an array"
            )
    return results


def _declares_array(group: Any, array: Any) -> bool:
    """Does ``group``'s own ``multiscales`` block DECLARE ``array`` as a level?

    The EVIDENCE test for "this block is metadata about that array". Arity is not
    evidence: a permuted axis list has exactly the right length, so a block that
    exists but describes something else (another series, a stale hand-written
    attribute) would be adopted over a root block that had the T/C decomposition
    right — a silently wrong ``batch-fit`` plan rather than an error. A block that
    names the selected array among its own ``datasets[*]["path"]`` entries, on the
    other hand, is talking about it by construction.

    Compared on the arrays' store-relative ``path``, so the same array reached by
    two different lookups matches.
    """
    target = getattr(array, "path", None)
    if not isinstance(target, str):
        return False
    target = target.strip("/")
    return any(
        str(getattr(item, "path", "")).strip("/") == target
        for _, item, _ in _declared_levels(group, "", report=False)
    )


# Subgroup keys an IMAGE-GROUP search must never descend into: NGFF puts an
# image's segmentation masks at `<image>/labels/<name>/<level>`.
_LABELS = frozenset({"labels"})


def _image_group_array(
    group: Any, prefix: str
) -> Optional[Tuple[str, Any, Any, Optional[bool]]]:
    """The full-resolution array of an image GROUP (a bioformats2raw series).

    The candidates are RESTRICTED rather than swept for recursively, because NGFF
    puts an image's segmentation masks at ``<image>/labels/<name>/<level>``: a
    recursive largest-array search reaches those, so a mask as big as (or bigger
    than) level 0 can win — and a fit would then silently run against the mask.
    In order:

    1. the arrays the group's own ``multiscales`` block declares as its levels —
       the principled answer, since that block names its own pyramid;
    2. else the largest DIRECT array child: NGFF requires the levels to be direct
       children of the image group, so this cannot reach ``labels/``;
    3. else the largest array anywhere below, skipping a ``labels`` subgroup AT
       ANY DEPTH — for a store that declares nothing and nests its levels further
       down, which is also exactly the store whose masks are nested further down.

    ``prefix`` is prepended to the reported key path, which therefore stays
    relative to the store ROOT (``"0/0"``, not ``"0"``).

    The fourth element of the result is the ``owner_declares`` fact
    :func:`_select_zarr_array` hands on (see its docstring): ``True`` on branch 1
    by construction, ``False`` on branch 2 (reaching it means ``group`` declared
    no resolvable level at all), and ``None`` on branch 3, whose owner is a
    subgroup this function never inspected the ``multiscales`` of.
    """
    import zarr

    declared = _pick_largest(_declared_levels(group, prefix, skip_groups=_LABELS))
    if declared is not None:
        return declared[0], declared[1], declared[2], True

    direct: List[Tuple[str, Any, Any]] = []
    subgroups: List[Tuple[str, Any]] = []
    for key in sorted(group.keys()):
        if key in _LABELS:
            # `labels/` holds THIS image's segmentation masks, never its levels.
            continue
        item = group[key]
        key_path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, zarr.Array):
            direct.append((key_path, item, group))
        elif isinstance(item, zarr.Group):
            subgroups.append((key_path, item))
    if direct:
        # `direct` is non-empty inside this branch, so the ordering always
        # answers — `_pick_largest`'s empty-list `None` was unreachable here, and
        # handling it read as though a direct array child could vanish.
        best = min(direct, key=_largest_first)
        return best[0], best[1], best[2], False

    nested: List[Tuple[str, Any, Any]] = []
    for key_path, subgroup in subgroups:
        nested.extend(_find_all_arrays(subgroup, key_path, _LABELS))
    found = _pick_largest(nested)
    return None if found is None else (found[0], found[1], found[2], None)


def _declaring_owner(
    root: Any, key_path: str, array: Any, asked: Any = None
) -> Tuple[Any, bool]:
    """The NEAREST ancestor of ``key_path`` whose ``multiscales`` declares ``array``.

    For a path whose parent the caller does NOT already hold open (an explicit
    nested ``array_key``); the search helpers above report an owner directly.
    Walked from the array's immediate parent up to the store ROOT, returning at the
    FIRST declaring ancestor (so the common case reads one group's attributes and
    stops). Falls back to ``(deepest OPENABLE ancestor, False)`` when no ancestor
    declares it — which is definitive, the walk having asked every one of them.
    That is the immediate parent in the ordinary case, but NOT when the parent is
    one of the implicit groups described below: the reported owner is then the
    deepest ancestor that actually opened.

    ``asked`` is the array's immediate parent when the CALLER has already run
    :func:`_declared_levels` on it and found nothing resolvable
    (:func:`_image_group_array`'s branch 2 reports exactly that). The walk then
    starts one level ABOVE it and falls back to it, so the answer is identical
    while the group's ``multiscales`` is resolved once instead of twice — the
    doubling the plumbed-through ``owner_declares`` fact exists to avoid.

    An explicit ``array_key`` must land on the same owner the auto-selection paths
    report for the same array, or two invocations describe one array differently:
    for ``datasets[0].path == "res/0"`` the level's immediate parent (``0/res``)
    declares nothing, while the declaring group (``0``) is what
    :func:`_declared_levels` hands back — so ``--array-key 0/res/0`` would fall
    through to the ndim heuristic and plan a different ``batch-fit`` T×C fan-out
    than the very same store planned without the key.

    An ancestor that will not OPEN is skipped rather than raised through: zarr v2
    permits an array below a group with no metadata document of its own (a
    hand-written store, or one whose intermediate ``.zgroup`` was never written),
    and `root["labels/cells"]` is then a ``KeyError`` even though
    ``root["labels/cells/1"]`` resolves perfectly well. The array is right there,
    so failing the whole read to ask a bookkeeping question about its ancestry
    would be absurd — the walk simply carries on upward. A TRANSIENT ``OSError``
    from a remote store is swallowed by that same ``continue``, so a momentarily
    unreachable ancestor is reported as "does not declare it" rather than raised;
    the array itself is already open, so the read still answers.

    The final iteration is the store ROOT with no lookup at all, so ``deepest`` is
    always set by the time the loop ends — except when ``asked`` skipped it, in
    which case ``asked`` IS the root and seeded ``deepest`` already.
    """
    segments = key_path.strip("/").split("/")[:-1]
    deepest: Any = asked
    start = len(segments) - 1 if asked is not None else len(segments)
    for depth in range(start, -1, -1):
        prefix = "/".join(segments[:depth])
        try:
            group = root[prefix] if prefix else root
        except (KeyError, ValueError, OSError, TypeError):
            continue
        if deepest is None:
            deepest = group
        if _declares_array(group, array):
            return group, True
    return deepest, False


# How many alternative array paths the terminal error lists before eliding. A
# store can hold thousands (every level of every series); the point is to name a
# usable `--array-key`, not to dump an inventory.
_MAX_SUGGESTED_ARRAYS = 8


def _no_array_in_group_error(
    group: Any, node: Any, path: Path, subject: str
) -> ValueError:
    """The error for a group that resolves to no array of its own.

    ONE message for both routes into :func:`_image_group_array` — an explicit
    ``array_key`` naming a group, and the OME-NGFF ``"0"`` key that turns out to be
    an image group. They have to answer the same way: a store whose image group
    ``"0"`` holds nothing but ``labels/`` used to raise this for ``--array-key 0``
    while the very same store, with no key, fell through to the unscoped
    whole-store sweep and silently selected the segmentation MASK.

    Same message, different SUBJECT. ``subject`` is the phrase naming what led
    here, because only one of the two routes involves a key the caller typed: with
    no ``--array-key`` at all, "Array key '0' names a zarr group…" sends the reader
    hunting their command line for a ``0`` they never wrote, when what actually
    picked ``"0"`` was the OME-NGFF convention.

    The condition is unrecoverable without a key, so the message names the keys
    that WOULD work: a store whose ``"0"`` is an empty stub with the real image at
    ``"1"`` otherwise reports only that the root holds ``['0', '1', 'OME']``,
    leaving the reader to guess which of those is an image and how deep its levels
    sit. The paths are swept from the store itself, so they are keys that exist —
    ``labels/`` included, since a mask this rule refuses to select SILENTLY is
    still a legitimate thing to ask for explicitly.

    That sweep is best-effort: it descends every subgroup, and one of them failing
    to open (the metadata-less intermediate group :func:`_declaring_owner`
    tolerates) must not replace the caller's diagnosis with a ``KeyError`` from the
    code that was only trying to be helpful about it.
    """
    elsewhere: Optional[List[str]]
    try:
        elsewhere = [key_path for key_path, _, _ in _find_all_arrays(node)]
    except (KeyError, ValueError, OSError, TypeError):
        # Unswept, which is NOT the same as "there are none" — say neither.
        elsewhere = None
    if elsewhere:
        shown = ", ".join(repr(k) for k in elsewhere[:_MAX_SUGGESTED_ARRAYS])
        if len(elsewhere) > _MAX_SUGGESTED_ARRAYS:
            shown += f", … ({len(elsewhere)} in total)"
        advice = (
            f" Arrays this store does hold: {shown} — pass one of those as an "
            f"array_key (--array-key)."
        )
    elif elsewhere is None:
        advice = " — pass an array_key (--array-key) naming an array."
    else:
        advice = " That store holds no array anywhere; there is nothing to select."
    return ValueError(
        f"{subject} names a zarr group holding no array "
        f"in {path}. That group holds {sorted(group.keys())}; the "
        f"store holds {sorted(node.keys())}.{advice}"
    )


def _resolved_group_selection(
    root: Any, found: Tuple[str, Any, Any, Optional[bool]]
) -> Tuple[Any, str, Any, Optional[bool]]:
    """A SEARCH result with its owner resolved the ONE way.

    :func:`_image_group_array` (and the whole-store sweep) reports the group it
    happened to search when that group's own ``multiscales`` did not declare the
    array, which is not necessarily the group that DECLARES it — an ancestor may.
    Adopting that report verbatim gave the same underlying array two different
    owners depending on how the caller spelled the key: for ``datasets[0].path ==
    "res/0"`` on image group ``"0"``, ``--array-key 0/res`` searched the ``res``
    subgroup and reported ``res`` (which declares nothing), so that spelling alone
    fell through to the ndim heuristic while ``0``, ``0/res/0`` and no key at all
    read the declaration. The no-key SWEEP had the identical defect one route
    over: on a store with no ``"0"`` at the root it reported the level's immediate
    parent, so ``img/res`` won over the ``img`` that declared it and the no-key
    spelling alone lost the axes and voxel size.

    So whenever the search did not itself find a declaration (``declares`` is not
    ``True``), the owner is re-derived with :func:`_declaring_owner` — the same
    rule the explicit-array-key branch uses — and every spelling converges.

    ``declares is False`` is stronger than ``None``: it means the search ALREADY
    ran :func:`_declared_levels` on the reported owner and found nothing there, so
    that group is handed over as ``asked`` and the walk resumes above it rather
    than resolving its block a second time.
    """
    key_path, array, owner, declares = found
    if declares is not True:
        owner, declares = _declaring_owner(
            root, key_path, array, asked=owner if declares is False else None
        )
    return array, key_path, owner, declares


def _keyed_zarr_array(
    node: Any, path: Path, array_key: str, key: str
) -> Tuple[Any, str, Any, Optional[bool]]:
    """The explicit-``array_key`` branch of :func:`_select_zarr_array`.

    ``key`` is ``array_key`` with surrounding slashes stripped; ``array_key`` is
    kept only so the error messages quote what the caller actually passed.
    """
    import zarr

    try:
        selected = node[key]
    except (KeyError, ValueError, OSError, TypeError) as e:
        # Not just KeyError: zarr 3 rejects a path with `..` segments or an
        # embedded null byte with ValueError, and a segment longer than a
        # filename surfaces as OSError(ENAMETOOLONG) from the store. All of
        # those are "that key is not in this store" as far as a caller is
        # concerned, and must report the documented message rather than
        # zarr's internal one — but the underlying error is NAMED in it, not
        # merely chained: an OSError from a remote store is a genuine I/O
        # failure, and "not found" alone sends the reader hunting a typo.
        raise ValueError(
            f"Array key '{array_key}' not found in {path} "
            f"({type(e).__name__}: {e}). "
            f"Available keys: {sorted(node.keys())}"
        ) from e
    if isinstance(selected, zarr.Array):
        owner, declares = _declaring_owner(node, key, selected)
        return selected, key, owner, declares
    # The key names a GROUP — an image group in a bioformats2raw store, or
    # any multiscale group. Descend with the image-group rule rather than
    # handing back a Group whose `.shape` the caller is about to read.
    found = _image_group_array(selected, key)
    if found is None:
        raise _no_array_in_group_error(selected, node, path, f"Array key {array_key!r}")
    return _resolved_group_selection(node, found)


def _auto_zarr_array(node: Any, path: Path) -> Tuple[Any, str, Any, Optional[bool]]:
    """The no-``array_key`` branch of :func:`_select_zarr_array`."""
    import zarr

    if "0" in node:
        level_zero = node["0"]
        if isinstance(level_zero, zarr.Array):
            # OME-NGFF convention: "0" is the highest resolution level.
            return level_zero, "0", node, None
        # bioformats2raw puts an image GROUP at "0" and its pyramid levels one
        # level down ("0/0", "0/1", …). Resolve INSIDE that group only: the store
        # root also holds the other series ("1", "2", …) and an `OME` metadata
        # group, so a whole-store search could hand back level 0 of a DIFFERENT
        # image. Scoping preserves the existing intent — full resolution of the
        # first image.
        found = _image_group_array(level_zero, "0")
        if found is None:
            # TERMINAL, not a fall-through to the whole-store sweep below. That
            # sweep is unscoped — it descends `labels/` at every depth and every
            # other series — so an image group holding only `0/labels/seg/0`
            # silently selected the segmentation MASK, and one holding nothing
            # silently selected a DIFFERENT series, defeating the "first image"
            # intent this branch exists to preserve. The same store already
            # raised for an explicit `--array-key 0`, so answering three
            # different ways depending on the spelling was the real defect.
            # Named as the CONVENTION, not as a key: nothing was passed, so
            # "Array key '0'" would send the reader hunting their command line.
            raise _no_array_in_group_error(
                level_zero, node, path, "The OME-NGFF resolution level '0'"
            )
        return _resolved_group_selection(node, found)

    # Largest array anywhere in the group, searching recursively into
    # sub-groups (e.g. h2afva/fused, mezzo/fused). The sweep reports the array's
    # IMMEDIATE PARENT and inspected nobody's `multiscales`, so the owner goes
    # through the same resolution every other route uses — otherwise the declaring
    # group being one level up (levels at `img/res/0` declared by `img`) makes this
    # spelling, and only this spelling, fall through to the ndim heuristic.
    largest = _largest_array(node)
    if largest is None:
        raise ValueError(f"No arrays found in zarr group: {path}")
    return _resolved_group_selection(node, (largest[0], largest[1], largest[2], None))


def _select_zarr_array(
    node: Any, path: Path, array_key: Optional[str] = None
) -> Tuple[Any, str, Any, Optional[bool]]:
    """Pick the array to read out of an already-opened zarr store.

    ONE copy of the selection rule, shared by :func:`load_volume`,
    :func:`open_volume_lazy` and
    :func:`~luxar.io.ome_zarr.discover_ome_zarr_shape`. They MUST agree: a caller
    routinely re-opens a store another command already read the shape of, and a
    different choice here would silently re-fit against a different (e.g.
    downsampled) array.

    The rule, in order: an explicit ``array_key``; else the OME-NGFF resolution
    level ``"0"``; else the largest array found recursively. A key (or a ``"0"``)
    that names a GROUP resolves through :func:`_image_group_array`, whose search
    is scoped to that image group and skips ``labels/`` at any depth, so an
    image's own segmentation masks cannot be selected as the image. That
    guarantee is the IMAGE-GROUP branch's, and it is TERMINAL: an image group that
    resolves to no array of its own is a ``ValueError``, never a fall-through to
    the whole-store sweep (which is unscoped and would answer with that image's
    own mask, or with a different series). The whole-store fallback is therefore
    reached only when there is no ``"0"`` key at all — the pre-existing
    ``h2afva/fused`` path — and sweeps recursively, ``labels/`` included, as it
    always has. A size tie is broken on the key path so two processes reading the
    same store cannot disagree.

    When the store ROOT is itself an array, ``array_key`` is deliberately IGNORED
    (any value, including a non-string or a key that names nothing): there is
    exactly one array to read, and the three entry points must agree on it — a
    lazy re-open with a stale key must not diverge from the read that produced
    the shape. So the ``Raises`` below only describe the group case.

    Returns ``(array, key_path, owner_group, owner_declares)`` — the chosen array,
    its path relative to the store ROOT (``""`` when the store itself is an
    array), the group carrying the NGFF ``multiscales``/``axes`` attributes that
    may describe the chosen array (``None`` in that same case), and whether that
    owner DECLARES the chosen array as one of its own levels.

    The owner is the array's immediate parent, except for a DECLARED level, where
    it is the group whose ``multiscales`` block declared it. EVERY route that does
    not find the declaration itself resolves that by the ONE rule
    (:func:`_declaring_owner`, reached through :func:`_resolved_group_selection`
    for the two searches and called directly for an explicit key that names an
    array) — the explicit-key branch, the image-group branch and the whole-store
    sweep alike. Anything less than all of them is not a rule: each route left out
    gave the same array a different owner, and so a different set of axes and a
    different voxel size, decided by nothing but how the key was spelled.

    ``owner_declares`` is the SELECTION's own answer to
    :func:`_declares_array`, plumbed through rather than re-derived: resolving a
    declared pyramid's paths a second time doubles the metadata requests a
    multi-level store costs (4 extra ``get``s on a 4-level bioformats2raw store).
    ``None`` means UNDETERMINED — the selection never needed the answer, so a
    caller that does must ask :func:`_declares_array` itself.

    Raises:
        ValueError: If ``array_key`` is not a string, is not found, or names a
            group holding no array; if the OME-NGFF ``"0"`` key is a group holding
            no array; if the store holds no array at all; or if the store is
            neither an array nor a group.
    """
    import zarr

    if isinstance(node, zarr.Array):
        return node, "", None, None
    if not isinstance(node, zarr.Group):
        raise ValueError(f"Unexpected zarr object type: {type(node)}")

    if array_key is not None and not isinstance(array_key, str):
        # A hand-edited manifest can carry a non-string here; the old code caught
        # the TypeError from the lookup below and reported it as a ValueError.
        raise ValueError(
            f"Array key {array_key!r} is not a key path string "
            f"(got {type(array_key).__name__})."
        )
    # A blank or slash-only key means NO key: typer hands back `""` for an
    # omitted `--array-key`, and `node[""]` is the ROOT group, which would then
    # be descended by an unscoped search and could answer with a different image
    # entirely. Normalising here is what keeps the three entry points in
    # agreement (`open_volume_lazy` used to test a plain `if array_key:`).
    if array_key is not None:
        key = array_key.strip("/")
        if key:
            return _keyed_zarr_array(node, path, array_key, key)
    return _auto_zarr_array(node, path)


def _load_zarr_volume(
    path: Path,
    channel: Optional[int],
    timepoint: Optional[int],
    array_key: Optional[str],
    raw: bool = False,
    region: Optional[Tuple[slice, ...]] = None,
) -> np.ndarray:
    """Load a volume from a zarr store, handling OME-ZARR conventions.

    With ``raw=True`` the full array is returned WITHOUT the positional
    TCZYX/CZYX slicing — the caller (``load_volume`` with an explicit ``--axes``)
    applies its own axis spec instead.
    """
    import zarr

    from luxar._zarr_compat import open_store

    aprint(f"Loading Zarr: {path.name}")
    # `open_store`, not a bare `zarr.open(str(path))`: zarr 2 sniffed a `.zip`
    # suffix inside `normalize_store_arg` and handed back a ZipStore, but zarr 3
    # does not — it treats the archive as a LocalStore directory and raises
    # GroupNotFoundError. That would break `luxar gsplat fit data.zarr.zip`, a
    # documented entry point, so the dispatch is explicit here.
    store = zarr.open(store=open_store(path, mode="r"), mode="r")

    # Navigate to the target array. The rule lives in `_select_zarr_array` so
    # this, `open_volume_lazy` and `discover_ome_zarr_shape` cannot drift apart.
    arr, key_path, _, _ = _select_zarr_array(store, path, array_key)
    # `not array_key`, not `array_key is None`: the selector treats a blank key as
    # absent, so the log line has to agree with what it actually did.
    if not array_key and key_path == "0":
        # OME-ZARR convention: "0" is highest resolution
        aprint("  Detected OME-ZARR layout (using resolution level '0')")
    elif key_path:
        # Names the array actually landed on — for a bioformats2raw store that is
        # a level INSIDE the image group ("0/0"), not the group itself.
        aprint(f"  Using array '{key_path}'")

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
        spans = _squeezed_region_spans(tuple(shape[-3:]), region)
        volume = np.array(arr[tuple([*idx, *spans])])
    elif ndim == 5:
        t = timepoint if timepoint is not None else 0
        c = channel if channel is not None else 0
        aprint(f"  Slicing 5D (TCZYX): T={t}, C={c}")
        spans = _squeezed_region_spans(tuple(shape[-3:]), region)
        volume = np.array(arr[(t, c, *spans)])
    elif ndim == 4:
        if channel is not None:
            aprint(f"  Slicing 4D (CZYX): C={channel}")
            spans = _squeezed_region_spans(tuple(shape[1:]), region)
            volume = np.array(arr[(channel, *spans)])
        elif timepoint is not None:
            aprint(f"  Slicing 4D (TZYX): T={timepoint}")
            spans = _squeezed_region_spans(tuple(shape[1:]), region)
            volume = np.array(arr[(timepoint, *spans)])
        else:
            aprint("  4D array — using as-is (use --channel or --timepoint to slice)")
            spans = _squeezed_region_spans(tuple(shape), region)
            volume = np.array(arr[spans])
    else:
        spans = _squeezed_region_spans(tuple(shape), region)
        volume = np.array(arr[spans])

    return volume
