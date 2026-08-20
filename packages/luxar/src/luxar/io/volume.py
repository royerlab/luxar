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

    kinds = [_axis_kind(label) for label in labels]
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


def load_volume(
    path: Path,
    channel: Optional[int] = None,
    timepoint: Optional[int] = None,
    array_key: Optional[str] = None,
    axes: Optional[str] = None,
    info: Optional[dict] = None,
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
        info: Optional dict, populated with ``source_dtype`` — the element type
            of the array AS STORED, captured before the float32 cast below.
            This is the only place it is knowable: the returned array is always
            float32, so a consumer that wants to quote a size (e.g. the
            denominator of a compression ratio) would otherwise describe the
            working copy and overstate it by the cast's inflation factor —
            exactly 2x for the 16-bit acquisitions most microscopy produces.

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

    # Record the STORED element type before the float32 cast below — the last
    # point at which it exists.
    _record_source_dtype(info, volume)

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
        # The second asarray is a no-op at runtime (already float32, no copy); it
        # is there because np.squeeze is typed as returning Any.
        volume = np.asarray(volume, dtype=np.float32)
        volume = np.asarray(np.squeeze(volume), dtype=np.float32)

    if volume.ndim < 2:
        raise ValueError(
            f"Volume must be at least 2D after squeezing, got {volume.ndim}D "
            f"with shape {volume.shape}"
        )

    aprint(f"  Shape: {volume.shape}, dtype: float32")
    return volume


def _find_all_arrays(group: Any, prefix: str = "") -> list:
    """Recursively find all arrays in a zarr group.

    Returns ``(key_path, array, owner_group)`` triples — the owning group comes
    for free here (this function holds it open), sparing the caller a second
    metadata request to re-open it. Keys are walked SORTED rather than in
    ``keys()`` order, which zarr does not stabilise across opens; see
    :func:`_pick_largest` for why that matters.
    """
    import zarr

    results = []
    for k in sorted(group.keys()):
        item = group[k]
        key_path = f"{prefix}/{k}" if prefix else k
        if isinstance(item, zarr.Array):
            results.append((key_path, item, group))
        elif isinstance(item, zarr.Group):
            results.extend(_find_all_arrays(item, key_path))
    return results


def _pick_largest(candidates: list) -> Optional[Tuple[str, Any, Any]]:
    """The candidate with the most elements; the LOWEST key path breaks a tie.

    ``None`` for an empty list. The tie-break is what makes selection
    REPRODUCIBLE: zarr does not stabilise ``Group.keys()`` order across opens, so
    a plain ``max()`` over two equal-sized candidates can answer differently for
    the same store — and the batch planner reading the shape and a per-tile
    worker re-opening it are different processes. Sorting by
    ``(-element_count, key_path)`` is a total order, so they cannot disagree.
    """
    if not candidates:
        return None
    best: Tuple[str, Any, Any] = min(
        candidates, key=lambda kv: (-int(np.prod(kv[1].shape)), kv[0])
    )
    return best


def _largest_array(group: Any, prefix: str = "") -> Optional[Tuple[str, Any, Any]]:
    """The ``(key_path, array, owner)`` with the most elements under ``group``.

    Searches recursively. ``None`` when there is no array anywhere below it.
    ``prefix`` is prepended to every key path, so a search scoped to a SUBGROUP
    still reports paths relative to the store root.
    """
    return _pick_largest(_find_all_arrays(group, prefix))


def _declared_levels(group: Any, prefix: str) -> list:
    """The arrays ``group``'s own NGFF ``multiscales`` block names as its levels.

    ``datasets[*]["path"]``, resolved relative to ``group``. A declared path that
    is missing or is not an array is skipped, so a half-written block degrades to
    the searches in :func:`_image_group_array` instead of raising.
    """
    import zarr

    block = group.attrs.get("multiscales")
    if not (isinstance(block, list) and block and isinstance(block[0], dict)):
        return []
    datasets = block[0].get("datasets")
    if not isinstance(datasets, list):
        return []
    results = []
    for entry in datasets:
        rel = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(rel, str) or not rel.strip("/"):
            continue
        rel = rel.strip("/")
        try:
            item = group[rel]
        except KeyError:
            continue
        if isinstance(item, zarr.Array):
            results.append(
                (f"{prefix}/{rel}" if prefix else rel, item, _owner_group(group, rel))
            )
    return results


def _image_group_array(group: Any, prefix: str) -> Optional[Tuple[str, Any, Any]]:
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
    3. else the largest array anywhere below, skipping a ``labels`` subgroup —
       for a store that declares nothing and nests its levels further down.

    ``prefix`` is prepended to the reported key path, which therefore stays
    relative to the store ROOT (``"0/0"``, not ``"0"``).
    """
    import zarr

    declared = _pick_largest(_declared_levels(group, prefix))
    if declared is not None:
        return declared

    direct: list = []
    subgroups: list = []
    for key in sorted(group.keys()):
        item = group[key]
        key_path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, zarr.Array):
            direct.append((key_path, item, group))
        elif isinstance(item, zarr.Group) and key != "labels":
            # `labels/` holds THIS image's segmentation masks, never its levels.
            subgroups.append((key_path, item))
    if direct:
        return _pick_largest(direct)

    nested: list = []
    for key_path, subgroup in subgroups:
        nested.extend(_find_all_arrays(subgroup, key_path))
    return _pick_largest(nested)


def _owner_group(root: Any, key_path: str) -> Any:
    """The group that immediately CONTAINS the array at ``key_path``.

    Only for a path whose parent the caller does NOT already hold open (an
    explicit nested ``array_key``): every search helper above reports its owner
    directly, which is one metadata request cheaper on a remote store.
    """
    key_path = key_path.strip("/")
    parent = key_path.rsplit("/", 1)[0] if "/" in key_path else ""
    return root[parent] if parent else root


def _select_zarr_array(
    node: Any, path: Path, array_key: Optional[str] = None
) -> Tuple[Any, str, Any]:
    """Pick the array to read out of an already-opened zarr store.

    ONE copy of the selection rule, shared by :func:`load_volume`,
    :func:`open_volume_lazy` and
    :func:`~luxar.io.ome_zarr.discover_ome_zarr_shape`. They MUST agree: a caller
    routinely re-opens a store another command already read the shape of, and a
    different choice here would silently re-fit against a different (e.g.
    downsampled) array.

    The rule, in order: an explicit ``array_key``; else the OME-NGFF resolution
    level ``"0"``; else the largest array found recursively. A key that names a
    GROUP resolves through :func:`_image_group_array` (never a ``labels/``
    sub-image), and a size tie is broken on the key path so two processes reading
    the same store cannot disagree.

    Returns ``(array, key_path, owner_group)`` — the chosen array, its path
    relative to the store ROOT (``""`` when the store itself is an array), and
    the group that immediately contains it (``None`` in that same case). The
    owner is not always the root, and it is the node carrying the NGFF
    ``multiscales``/``axes`` attributes that describe the chosen array.

    Raises:
        ValueError: If ``array_key`` is not a string, is not found, or names a
            group holding no array; if the store holds no array at all; or if the
            store is neither an array nor a group.
    """
    import zarr

    if isinstance(node, zarr.Array):
        return node, "", None
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
    key = array_key.strip("/") if array_key is not None else ""

    if key:
        try:
            selected = node[key]
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Array key '{array_key}' not found in {path}. "
                f"Available keys: {sorted(node.keys())}"
            ) from e
        if isinstance(selected, zarr.Array):
            return selected, key, _owner_group(node, key)
        # The key names a GROUP — an image group in a bioformats2raw store, or
        # any multiscale group. Descend with the image-group rule rather than
        # handing back a Group whose `.shape` the caller is about to read.
        found = _image_group_array(selected, key)
        if found is None:
            raise ValueError(
                f"Array key '{array_key}' names a zarr group holding no array "
                f"in {path}. That group holds {sorted(selected.keys())}; the "
                f"store holds {sorted(node.keys())} — pass an array_key "
                f"(--array-key) naming an array."
            )
        return found[1], found[0], found[2]

    found = None
    if "0" in node:
        level_zero = node["0"]
        if isinstance(level_zero, zarr.Array):
            # OME-NGFF convention: "0" is the highest resolution level.
            return level_zero, "0", node
        # bioformats2raw puts an image GROUP at "0" and its pyramid levels one
        # level down ("0/0", "0/1", …). Resolve INSIDE that group only: the store
        # root also holds the other series ("1", "2", …) and an `OME` metadata
        # group, so a whole-store search could hand back level 0 of a DIFFERENT
        # image. Scoping preserves the existing intent — full resolution of the
        # first image.
        found = _image_group_array(level_zero, "0")

    if found is None:
        # Largest array anywhere in the group, searching recursively into
        # sub-groups (e.g. h2afva/fused, mezzo/fused).
        found = _largest_array(node)
    if found is None:
        raise ValueError(f"No arrays found in zarr group: {path}")
    return found[1], found[0], found[2]


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
    arr, key_path, _ = _select_zarr_array(store, path, array_key)
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
