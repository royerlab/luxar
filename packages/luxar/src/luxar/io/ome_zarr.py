"""OME-Zarr shape discovery (domain layer).

Discovers the shape and axis structure of an OME-Zarr / NGFF dataset (T/C/Z/Y/X
layout, voxel size, unit, resolution levels), with fallbacks to a custom
``axes`` attribute and a shape-based heuristic. Reusable domain logic with no
CLI/Typer coupling. Previously lived in ``luxar.cli.gsplat_config``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from arbol import aprint

from luxar.io.volume import _find_all_arrays

__all__ = [
    "CHANNEL_LIKE_AXIS_LABELS",
    "OMEZarrInfo",
    "TIME_AXIS_LABELS",
    "classify_axis_labels",
    "discover_ome_zarr_shape",
    "ngff_scale_transform",
    "resolve_ngff_attrs",
]


def resolve_ngff_attrs(attrs: Mapping[str, Any]) -> Dict[str, Any]:
    """The mapping that actually carries ``multiscales``, for either OME-Zarr layout.

    OME-Zarr **0.4** puts ``multiscales`` at the top level of a node's attributes.
    OME-Zarr **0.5** nests the whole NGFF block one level down under an ``ome``
    key. Reading only the 0.4 spelling on a 0.5 store fails SILENTLY — no
    metadata is found, so axis roles get guessed from the shape and no physical
    voxel size is recovered — which is why every reader goes through here instead
    of spelling ``attrs["multiscales"]`` itself.

    The layout can NOT be inferred from the store's zarr format version: a zarr
    v3 store written by a 0.4-era tool has the v3 chunk layout with 0.4
    (top-level) attributes. Both spellings are therefore always tried.

    The predicate is deliberately ``multiscales``-specific rather than "any NGFF
    key": a store may carry a top-level 0.4 ``multiscales`` *and* an ``ome`` block
    holding only rendering metadata (``omero``), and selecting that block on the
    strength of ``omero`` alone would throw away the pyramid the caller came for —
    the exact silent mis-read this function exists to prevent, from the other side.

    Presence of the key is not enough either, for the same reason: a store
    carrying a good top-level 0.4 pyramid alongside ``{"ome": {"multiscales":
    []}}`` would have the pyramid discarded and be reported as declaring an empty
    ``multiscales`` — false of the store. The nested block therefore wins only
    when its ``multiscales`` is a NON-EMPTY list, **or** when the top level
    declares no ``multiscales`` at all (in which case the nested one, empty or
    malformed as it may be, is the only thing the store said, and handing it back
    is what lets the caller report "declared but unusable" instead of "nothing
    declared").

    Defensive by design — ``ome`` may be absent, not a mapping, or a mapping with
    no ``multiscales``. In each of those cases the top-level attributes are
    returned, so a caller's own "no multiscales here" handling runs as it did
    before rather than this raising.

    Args:
        attrs: A zarr node's user attributes (e.g. ``dict(group.attrs)``).

    Returns:
        Either the nested ``ome`` block (0.5) or ``attrs`` itself (0.4), as a
        plain dict.
    """
    ome = attrs.get("ome")
    if isinstance(ome, Mapping) and "multiscales" in ome:
        nested = ome.get("multiscales")
        if (isinstance(nested, list) and nested) or "multiscales" not in attrs:
            return dict(ome)
    return dict(attrs)


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

    time_axis: Optional[int] = None
    """Index into ``shape`` of the time axis, or ``None`` when there is none.

    Part of the ``(time_axis, channel_indices, spatial_indices)`` trio below: the
    decomposition discovery ACTUALLY used to derive ``n_timepoints`` /
    ``n_channels`` / ``spatial_shape``.
    """

    channel_indices: Tuple[int, ...] = ()
    """Indices into ``shape`` of the axes folded into the flat channel index.

    In the order they are folded, so ``decode_flat_channel_index(c,
    channel_shape)`` maps position-for-position onto them.
    """

    spatial_indices: Tuple[int, ...] = ()
    """Indices into ``shape`` of the spatial axes, in ``spatial_shape`` order.

    Publishing all three indices closes a standing hazard: a consumer that needs
    to know which axis played which role had to RE-DERIVE it from ``axes`` with a
    second vocabulary, and the vocabularies disagree in both directions. NGFF
    metadata is classified by the ``type`` field, so a channel axis named
    ``stain`` is a channel here but not to any name-driven rule, while an axis
    typed ``view`` falls through to SPATIAL here but is channel-like to
    :func:`classify_axis_labels`. Two vocabularies deciding the same question is
    how a consumer silently plans against a layout discovery never reported —
    read these fields instead of re-classifying the labels.
    """

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

    Parses the node's NGFF ``multiscales`` metadata in BOTH OME-Zarr layouts —
    top-level (0.4) and nested under an ``ome`` key (0.5); see
    :func:`resolve_ngff_attrs`. A block whose ``axes`` count disagrees with the
    SELECTED array's ndim is not metadata about that array (a 5D image beside its
    3D ``labels/…``) and is skipped. Falls back to a custom ``axes`` attribute,
    then to a shape-based heuristic (5D→TCZYX, 4D→CZYX, 3D→ZYX) for non-NGFF
    zarr stores. That last fallback GUESSES the T/C roles and recovers no voxel
    size, so for an ambiguous (≥4D) store it says so on the console — stating
    whether nothing was declared or something was declared but unusable — and
    points at ``axes_override``.

    Accepts both plain ``.zarr`` directories and ``.zarr.zip`` archives —
    zarr's ZipStore handles the latter transparently.

    Args:
        path: Path to the ``.zarr`` store or ``.zarr.zip`` archive.
        axes_override: Explicit axis labels (e.g. ``["time","channel","z","y","x"]``).
            Overrides all auto-detection when provided.
        array_key: Key path to a specific array within the zarr store
            (e.g. ``"h2afva/fused"``).  When provided, skips auto-selection
            and navigates directly to this array, and selects the matching
            ``datasets[]`` entry for the voxel size (so a coarser pyramid level
            reports its own spacing, not level 0's).

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

    # Try NGFF multiscales metadata, in either OME-Zarr layout (0.4 top-level or
    # 0.5 nested under `ome`). `unusable` records WHY a block that IS present
    # could not be read, so the give-up notice below can say so.
    ms, unusable = _usable_multiscales(resolve_ngff_attrs(attrs), ndim)
    if ms is not None:
        return _parse_ngff_metadata(ms, shape, path, store, array_key)

    # Try custom axes attribute (e.g. Keller-lab zarr.zip files store
    # axes = ['time', 'camera', 'channel', 'z', 'y', 'x']).
    custom_axes, custom_problem = _usable_custom_axes(attrs, ndim)
    if custom_axes is not None:
        return _parse_custom_axes_attr(custom_axes, shape, path)
    unusable = unusable or custom_problem

    # Fallback: heuristic based on ndim
    info = _heuristic_ome_info(shape, ndim, path)
    _announce_guessed_axes(info, path, unusable)
    return info


def _usable_multiscales(
    ngff: Mapping[str, Any], ndim: int
) -> "Tuple[Optional[Dict[str, Any]], Optional[str]]":
    """The multiscales block to parse, or why the declared one cannot be used.

    Returns ``(block, None)`` when the store's ``multiscales`` genuinely describes
    an ``ndim``-D array, else ``(None, reason)`` — ``reason`` is ``None`` only
    when the store declared no ``multiscales`` at all, which is what lets the
    give-up notice tell "nothing declared" from "declared but unusable".

    The axes/ndim agreement is not pedantry: a block may describe a DIFFERENT
    array of the store than the one selected (a 5D ``TCZYX`` image whose
    ``labels/…`` companion is 3D), and its axis indices then run off the end of
    this array's shape — a bare ``IndexError`` out of the parser.

    The ``axes`` entry is type-checked before it is measured, like every other
    field this module reads: ``{"axes": null}`` (or a number) is not a list, and
    ``len()`` of it is a ``TypeError`` straight out of discovery rather than the
    fallback this module promises everywhere else.
    """
    multiscales = ngff.get("multiscales")
    if multiscales is None:
        return None, None
    ms = multiscales[0] if isinstance(multiscales, list) and multiscales else None
    if not isinstance(ms, Mapping):
        return None, "its `multiscales` attribute is empty or not a list of blocks"
    axes = ms.get("axes")
    if not isinstance(axes, (list, tuple)):
        return None, "its `multiscales` block declares no `axes` list"
    n_axes = len(axes)
    if n_axes == ndim:
        return dict(ms), None
    return None, (
        f"its `multiscales` block describes {n_axes} axes but the "
        f"selected array is {ndim}-D"
    )


def _usable_custom_axes(
    attrs: Mapping[str, Any], ndim: int
) -> "Tuple[Optional[List[str]], Optional[str]]":
    """The custom ``axes`` label list to use, or why the declared one cannot be.

    Same ``(value, reason)`` contract as :func:`_usable_multiscales`. Read from
    the TOP level deliberately: this is a non-NGFF convention (Keller-lab
    ``.zarr.zip`` files, say), so it never lives inside an ``ome`` block.
    """
    custom_axes = attrs.get("axes")
    if custom_axes is None:
        return None, None
    if not isinstance(custom_axes, list):
        return None, "its `axes` attribute is not a list of labels"
    if len(custom_axes) == ndim:
        return custom_axes, None
    return None, (
        f"its `axes` attribute has {len(custom_axes)} labels for a {ndim}-D array"
    )


def _announce_guessed_axes(
    info: OMEZarrInfo, path: Path, unusable: Optional[str] = None
) -> None:
    """Say out loud that these axis roles were GUESSED, not read off the store.

    The heuristic fallback is silent by construction — it returns a perfectly
    ordinary :class:`OMEZarrInfo` — so a store whose metadata could not be parsed
    is indistinguishable from one that was. On a 4D ``TZYX`` store the guess reads
    the time axis as a channel, which fans a batch plan out over the wrong axis,
    and no voxel size is recovered either way.

    ``unusable`` distinguishes the two very different ways discovery gets here:
    the store declares nothing, or it declares something this reader could not
    apply (see the call site) — "no metadata found" would be a lie in the second
    case, and hides the one detail that makes it fixable.

    Only for ndim >= 4, where T vs C is genuinely ambiguous: 2D/3D are all-spatial
    and have nothing to get wrong.
    """
    if len(info.shape) < 4:
        return
    why = (
        f"OME-Zarr/NGFF metadata is present but unusable ({unusable})"
        if unusable
        else "no OME-Zarr/NGFF metadata found"
    )
    aprint(
        f"⚠️  {Path(path).name}: {why} — axis roles GUESSED from the "
        f"{len(info.shape)}D shape as {','.join(info.axes)}, and no physical "
        f"voxel size recovered. Pass the store's real axis labels via "
        f"`axes_override` (the `--axes` flag on the CLI) if that is wrong."
    )


def ngff_scale_transform(transforms: Any) -> Optional[List[float]]:
    """The ``scale`` vector of a NGFF ``coordinateTransformations`` list, if any.

    The list is SEARCHED for the ``type == "scale"`` entry rather than indexed at
    ``[0]``: the spec allows a ``translation`` (or any other transform) to come
    first, and ``transforms[0]["scale"]`` then raises on a perfectly valid store.

    Never raises. Anything malformed — not a list, no scale entry, a ``scale``
    that is not a sequence, or a component that is not a number (``null``, a
    non-numeric string) — yields ``None``, i.e. "no spacing declared", which is
    how every other malformed-metadata path in this module degrades. Numeric
    STRINGS still convert, since that is how some writers spell a float.
    """
    if not isinstance(transforms, list):
        return None
    for t in transforms:
        if isinstance(t, Mapping) and t.get("type") == "scale":
            scale = t.get("scale")
            if isinstance(scale, (list, tuple)):
                try:
                    return [float(s) for s in scale]
                except (TypeError, ValueError):
                    return None
    return None


def _dataset_path_matches(entry: Mapping[str, Any], wanted: str) -> bool:
    """Whether a ``datasets[]`` entry's ``path`` names the array ``wanted``.

    ``path`` is relative to the multiscales group while ``array_key`` is relative
    to the store ROOT, so ``--array-key labels/cells/1`` must be matched against
    the entry's ``"1"``. The trailing segment is tried as well as the whole key.
    """
    path = str(entry.get("path", "")).strip("/")
    if not path:
        return False
    return path == wanted or path == wanted.rsplit("/", 1)[-1]


def _selected_dataset(
    datasets: Any, array_key: Optional[str]
) -> Optional[Mapping[str, Any]]:
    """The ``datasets[]`` entry describing the selected array, or ``None``.

    ``array_key`` may select a coarser pyramid LEVEL, which has its own entry;
    quoting level 0's spacing for it halves every number. Matched on the entry's
    ``path`` (see :func:`_dataset_path_matches`).

    ``None`` — "unknowable", not "level 0" — when an ``array_key`` was asked for,
    nothing matched, and the pyramid has more than one level: the caller selected
    SOME array and this cannot say which, so a wrong spacing quoted as fact would
    be worse than none. A single-level pyramid has nothing to be wrong about and
    still answers.

    Never raises: a ``datasets`` that is not a list of mappings is metadata this
    cannot read.
    """
    if not isinstance(datasets, list) or not datasets:
        return None
    first = datasets[0] if isinstance(datasets[0], Mapping) else None
    if array_key is None:
        return first
    wanted = str(array_key).strip("/")
    for d in datasets:
        if isinstance(d, Mapping) and _dataset_path_matches(d, wanted):
            return d
    return None if len(datasets) > 1 else first


def _composed_scale(
    ms: Mapping[str, Any], selected: Mapping[str, Any]
) -> Optional[List[float]]:
    """The effective scale vector: the dataset's, times the multiscales entry's.

    Both 0.4 and 0.5 allow an optional ``coordinateTransformations`` on the
    multiscales ENTRY itself, applied on top of the per-dataset one — so the
    effective spacing is the PRODUCT of the two, not the dataset's alone. Two
    vectors of different lengths cannot be composed at all, and a half-composed
    number is not the spacing this promises, so that yields ``None`` rather than
    the per-dataset vector on its own.
    """
    scale = ngff_scale_transform(selected.get("coordinateTransformations"))
    ms_scale = ngff_scale_transform(ms.get("coordinateTransformations"))
    if ms_scale is None:
        return scale
    if scale is None:
        return ms_scale
    if len(ms_scale) != len(scale):
        return None
    return [a * b for a, b in zip(scale, ms_scale)]


def _ngff_voxel_size(
    ms: Mapping[str, Any],
    datasets: Any,
    shape: Tuple[int, ...],
    spatial_indices: List[int],
    array_key: Optional[str],
) -> Optional[Tuple[float, ...]]:
    """Physical spacing of the SELECTED array's spatial axes, or ``None``.

    Two things the naive "read ``datasets[0]``'s scale" gets plausibly wrong —
    which entry describes the selected array (:func:`_selected_dataset`) and the
    multiscales-entry transform stacked on top of it (:func:`_composed_scale`).
    Both degrade to ``None`` rather than to a plausible wrong number, and neither
    raises on malformed metadata.
    """
    selected = _selected_dataset(datasets, array_key)
    if selected is None:
        return None
    scale = _composed_scale(ms, selected)
    if scale is None:
        return None
    # Spatial dimensions only — the scale vector spans every axis.
    if spatial_indices and len(scale) == len(shape):
        return tuple(float(scale[i]) for i in spatial_indices)
    if len(scale) == len(spatial_indices):
        return tuple(float(s) for s in scale)
    return None


def _parse_ngff_metadata(
    ms: Dict[str, Any],
    shape: Tuple[int, ...],
    path: Path,
    store: Any,
    array_key: Optional[str] = None,
) -> OMEZarrInfo:
    """Parse NGFF v0.4+ multiscales metadata.

    Classification is by the NGFF ``type`` field first, name second — which is
    the spec's own rule, and is why the resulting decomposition is published on
    :class:`OMEZarrInfo` (``time_axis`` / ``channel_indices`` /
    ``spatial_indices``) rather than left to be re-derived from the labels.

    A malformed axis record degrades rather than raising, like the rest of this
    module: a missing ``name`` becomes an empty label and a non-string ``type``
    is read as no type (so the axis falls through to spatial).
    """
    axes_raw = ms.get("axes", [])
    axes = [str(a.get("name", "")) if isinstance(a, dict) else str(a) for a in axes_raw]

    # Identify T, C, spatial axes
    t_idx: Optional[int] = None
    c_idx: Optional[int] = None
    spatial_indices: List[int] = []
    spatial_axes: List[str] = []

    for i, a in enumerate(axes_raw):
        if isinstance(a, dict):
            atype = str(a.get("type") or "").lower()
            aname = str(a.get("name") or "").lower()
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

    datasets = ms.get("datasets", [])
    voxel_size = _ngff_voxel_size(ms, datasets, shape, spatial_indices, array_key)

    # Extract unit from axes metadata
    unit = None
    for a in axes_raw:
        if isinstance(a, dict) and a.get("type") == "space":
            u = a.get("unit")
            if u:
                unit = u
                break

    # Count resolution levels. Type-checked, not just truthiness-checked: a
    # `datasets` that is a number is a `len()` TypeError, and this module
    # degrades on malformed metadata rather than raising out of discovery.
    n_levels = len(datasets) if isinstance(datasets, (list, tuple)) and datasets else 1

    return OMEZarrInfo(
        axes=axes,
        shape=shape,
        n_timepoints=n_t,
        n_channels=n_c,
        channel_axes=channel_axes,
        channel_shape=channel_shape,
        spatial_shape=spatial_shape,
        spatial_axes=spatial_axes,
        time_axis=t_idx,
        channel_indices=(c_idx,) if c_idx is not None else (),
        spatial_indices=tuple(spatial_indices),
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
        time_axis=t_idx,
        channel_indices=tuple(channel_indices),
        spatial_indices=tuple(spatial_indices),
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
            time_axis=0,
            channel_indices=(1,),
            spatial_indices=(2, 3, 4),
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
            time_axis=None,
            channel_indices=(0,),
            spatial_indices=(1, 2, 3),
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
            spatial_indices=(0, 1, 2),
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
            spatial_indices=(0, 1),
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
            spatial_indices=tuple(range(ndim)),
            path=path,
        )
