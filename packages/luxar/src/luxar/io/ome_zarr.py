"""OME-Zarr shape discovery (domain layer).

Discovers the shape and axis structure of an OME-Zarr / NGFF dataset (T/C/Z/Y/X
layout, voxel size, unit, resolution levels), with fallbacks to a custom
``axes`` attribute and a shape-based heuristic. Reusable domain logic with no
CLI/Typer coupling. Previously lived in ``luxar.cli.gsplat_config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from arbol import aprint

from luxar.io.volume import _declares_array, _select_zarr_array

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

    axis_units: List[Optional[str]] = field(default_factory=list)
    """Physical unit for each source axis, position-for-position with ``axes``."""

    axis_scales: Optional[Tuple[float, ...]] = None
    """Composed NGFF scale for every source axis, or ``None`` when unavailable."""

    voxel_size: Optional[Tuple[float, ...]] = None
    """Physical spacing from coordinateTransformations (spatial axes only)."""

    unit: Optional[str] = None
    """Physical unit string (e.g. ``"micrometer"``)."""

    resolution_levels: int = 1
    """Number of multiscale levels."""

    path: Optional[Path] = None
    """Path to the zarr store."""

    def __post_init__(self) -> None:
        """Keep positional axis metadata aligned with the discovered axes."""
        if not self.axis_units:
            self.axis_units = [None] * len(self.axes)
        elif len(self.axis_units) != len(self.axes):
            raise ValueError("axis_units must align position-for-position with axes")


# The one reason string for "the store declared `multiscales`, but there is no
# block in it to read". Shared by :func:`_usable_multiscales` (root) and
# :func:`_owner_ngff_attrs` (owning group) so the SAME malformed declaration is
# not reported two different ways depending on which node carries it.
_EMPTY_MULTISCALES_REASON = (
    "its `multiscales` attribute is empty or not a list of blocks"
)


def _relative_key(owner: Any, key_path: str) -> Optional[str]:
    """``key_path`` (store-root-relative) re-expressed relative to ``owner``.

    An owning group's ``datasets[*].path`` entries are relative to ITSELF, so the
    selected array has to be named the same way before
    :func:`_selected_dataset` can match it: the level a bioformats2raw store
    reaches at ``"0/0"`` is ``"0"`` to the image group that declares it.

    ``None`` when the selected array does not live under ``owner`` at all — which
    the selection never produces (an owner is an ancestor by construction) but
    which must not silently become a bogus match if it ever does.
    """
    owner_path = str(getattr(owner, "path", "") or "").strip("/")
    selected = str(key_path or "").strip("/")
    if not owner_path:
        return selected or None
    prefix = f"{owner_path}/"
    return selected[len(prefix) :] if selected.startswith(prefix) else None


def _owner_ngff_attrs(
    owner: Any, arr: Any, owner_attrs: Dict[str, Any], declares: Optional[bool]
) -> "Tuple[Optional[Dict[str, Any]], Optional[str]]":
    """The owning group's attributes, but only when its block is EVIDENCE.

    ``multiscales`` describes the array it sits BESIDE, and that is not always the
    root: a bioformats2raw store puts the NGFF block on the image group (``"0"``)
    and leaves only ``bioformats2raw.layout`` at the root, so reading the root
    alone finds nothing and falls through to the shape heuristic — which GUESSES
    the T/C roles and recovers no voxel size.

    So the owner's block may override the root's, but only on evidence that it
    describes the array actually SELECTED: it has to declare that array as one of
    its own levels (:func:`~luxar.io.volume._declares_array`). Arity is not
    evidence — a permuted axis list has exactly the right length, so a block that
    exists but describes something else (another series, a stale hand-written
    attribute) would be adopted over a root block that had the T/C decomposition
    right, a silently wrong ``batch-fit`` fan-out rather than an error.

    ``declares`` is the SELECTION's own answer, plumbed through rather than
    re-derived (re-resolving a declared pyramid's paths doubles the metadata
    requests a multi-level store costs); ``None`` means it never needed one, so
    ask.

    EVIDENCE is the whole test. There is deliberately no second, SHAPE gate on the
    block's ``datasets``: the parser does not crash on a malformed one (it
    type-checks every entry it reads and only takes ``len()`` for the level
    count), so the gate could only throw away a block that IS about the selected
    array. The one shape that ever reached it with evidence in hand — a MIXED
    ``["0", {"path": "0", …}]`` list, whose dict entry is what resolved to the
    array in the first place — is exactly the one the parser reads completely,
    axes, voxel size and all, and rejecting it dropped a 4D ``TZYX`` store onto
    the ``CZYX`` heuristic: the silently wrong ``batch-fit`` fan-out this module
    exists to prevent, caused by the guard against it.

    Returns ``(attrs, problem)`` — ``attrs`` is ``None`` whenever the root's block
    stands, which is both the plain OME-NGFF case (there the owner IS the root)
    and the safe answer for a group carrying a block about something else. The
    latter carries a ``problem``: the store DID declare NGFF metadata, so letting
    the give-up notice say "no OME-Zarr/NGFF metadata found" would be a lie about
    a store whose one fixable detail is that its ``datasets[*].path`` entries do
    not resolve to the array that was selected.

    A PRESENT but falsy ``multiscales`` is that same lie one step earlier, so it
    is told apart from an absent one and reported with the very reason
    :func:`_usable_multiscales` gives for the identical declaration sitting on the
    ROOT — the two must not describe one malformed store differently depending on
    which node carries it. An explicit ``null`` is the one falsy value that stays
    "nothing declared", again matching :func:`_usable_multiscales`.
    """
    block = resolve_ngff_attrs(owner_attrs).get("multiscales")
    if block is None:
        return None, None
    if not block:
        return None, _EMPTY_MULTISCALES_REASON
    if not (_declares_array(owner, arr) if declares is None else declares):
        return None, (
            "the group owning the selected array declares a `multiscales` "
            "block that does not name it"
        )
    return owner_attrs, None


def discover_ome_zarr_shape(
    path: Path,
    axes_override: Optional[List[str]] = None,
    array_key: Optional[str] = None,
    *,
    announce: bool = True,
) -> OMEZarrInfo:
    """Discover the shape and axis structure of an OME-Zarr dataset.

    Which array is read is ONE rule shared with the two volume-loading entry
    points (:func:`~luxar.io.volume._select_zarr_array`), so a re-fit that
    re-opens a store cannot land on a different (e.g. downsampled) array than the
    command that read its shape.

    Parses that array's NGFF ``multiscales`` metadata in BOTH OME-Zarr layouts —
    top-level (0.4) and nested under an ``ome`` key (0.5); see
    :func:`resolve_ngff_attrs`. The block is read off the GROUP THAT OWNS the
    selected array when that group's block is evidence about it
    (:func:`_owner_ngff_attrs`), off the root otherwise. A block whose ``axes``
    count disagrees with the SELECTED array's ndim is not metadata about that
    array (a 5D image beside its 3D ``labels/…``) and is skipped. Falls back to a
    custom ``axes`` attribute — root-first, owner second, see
    :func:`_usable_custom_axes` — then to a shape-based heuristic (5D→TCZYX,
    4D→CZYX, 3D→ZYX) for non-NGFF zarr stores. That last fallback GUESSES the T/C
    roles and recovers no voxel size, so for an ambiguous (≥4D) store it says so
    on the console — stating whether nothing was declared or something was
    declared but unusable — and points at ``axes_override``.

    Accepts both plain ``.zarr`` directories and ``.zarr.zip`` archives —
    zarr's ZipStore handles the latter transparently.

    Args:
        path: Path to the ``.zarr`` store or ``.zarr.zip`` archive.
        axes_override: Explicit axis labels (e.g. ``["time","channel","z","y","x"]``).
            Overrides axis classification and names while retaining positional
            NGFF units/scales from the selected array when available.
        array_key: Key path to a specific array within the zarr store
            (e.g. ``"h2afva/fused"``).  When provided, skips auto-selection
            and navigates directly to this array (which may itself be a group —
            then its own full-resolution level is taken).

            The ``datasets[]`` entry for the voxel size is matched against the
            array that was SELECTED, whether or not a key was passed — so a
            coarser pyramid level reports its own spacing, not level 0's, and
            that holds for the auto path too (which can perfectly well land on a
            coarser level, e.g. when level 0's declared path does not resolve).
            The match is exact, but on the NORMALISED spellings, so a block
            writing its own levels explicitly relative (``"./0"`` for the array at
            ``"0"``) still names them; see :func:`_selected_dataset`.
        announce: Print the ambiguous-axis warning when discovery falls back to
            shape-based heuristics. Internal metadata probes may disable it.

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
    # `array_key` may be nested (e.g. "h2afva/fused") and may name a group, and a
    # bioformats2raw store's `"0"` IS a group (its levels live at "0/0", "0/1",
    # …), which the rule descends into rather than reading `.shape` off.
    arr, key_path, owner, declares = _select_zarr_array(store, path, array_key)
    attrs: Dict[str, Any] = dict(getattr(store, "attrs", {}))

    # `multiscales` / `axes` describe the array they sit BESIDE, and that is not
    # always the root — see `_owner_ngff_attrs` for the evidence gate that lets an
    # owning group's block override the root's, and `_relative_key` for why the
    # selected array has to be renamed relative to whichever group won.
    owner_attrs: Dict[str, Any] = {}
    owner_ngff: Optional[Dict[str, Any]] = None
    owner_key: Optional[str] = None
    owner_problem: Optional[str] = None
    if owner is not None and owner is not store:
        owner_attrs = dict(owner.attrs)
        owner_key = _relative_key(owner, key_path)
        owner_ngff, owner_problem = _owner_ngff_attrs(owner, arr, owner_attrs, declares)

    shape = tuple(arr.shape)
    ndim = len(shape)

    ms, unusable, ngff_array_key = _usable_ngff_for_selection(
        attrs, owner_ngff, owner_key, key_path or None, ndim
    )

    # User-supplied axes override controls classification and labels, but the
    # selected array's positional NGFF units/scales remain valid metadata.
    if axes_override is not None:
        if len(axes_override) != ndim:
            raise ValueError(
                f"--axes has {len(axes_override)} labels but array is {ndim}D "
                f"(shape {shape}). Provide exactly {ndim} comma-separated axis names."
            )
        info = _parse_custom_axes_attr(axes_override, shape, path)
        if ms is not None:
            declared = _parse_ngff_metadata(ms, shape, path, ngff_array_key)
            info.axis_units = declared.axis_units
            info.axis_scales = declared.axis_scales
            if declared.axis_scales is not None:
                info.voxel_size = tuple(
                    declared.axis_scales[index] for index in info.spatial_indices
                )
            elif info.spatial_indices == declared.spatial_indices:
                info.voxel_size = declared.voxel_size
            spatial_units = [
                info.axis_units[index]
                for index in info.spatial_indices
                if info.axis_units[index]
            ]
            info.unit = spatial_units[0] if spatial_units else None
        return info

    # Try NGFF multiscales metadata, in either OME-Zarr layout (0.4 top-level or
    # 0.5 nested under `ome`). The owning group's block wins when it is evidence
    # about the selected array, the root's otherwise; its dataset paths are
    # relative to whichever group that is, not to the store root. `unusable`
    # records WHY a block that IS present could not be read, so the give-up notice
    # below can say so.
    if ms is not None:
        return _parse_ngff_metadata(ms, shape, path, ngff_array_key)
    unusable = unusable or owner_problem

    # Try custom axes attribute (e.g. Keller-lab zarr.zip files store
    # axes = ['time', 'camera', 'channel', 'z', 'y', 'x']).
    custom_axes, custom_problem = _usable_custom_axes(attrs, ndim, owner_attrs)
    if custom_axes is not None:
        return _parse_custom_axes_attr(custom_axes, shape, path)
    unusable = unusable or custom_problem

    # Fallback: heuristic based on ndim
    info = _heuristic_ome_info(shape, ndim, path)
    if announce:
        _announce_guessed_axes(info, path, unusable)
    return info


def _usable_ngff_for_selection(
    attrs: Dict[str, Any],
    owner_attrs: Optional[Dict[str, Any]],
    owner_key: Optional[str],
    selected_key: Optional[str],
    ndim: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Prefer the owning group's usable block, otherwise retry the root's.

    ``owner_attrs`` is already evidence-gated (:func:`_owner_ngff_attrs`) — it is
    ``None`` whenever the root's block is the one to read, including the plain
    OME-NGFF case where the owner IS the root. So the only question left here is
    USABILITY, and it is asked with one predicate (:func:`_usable_multiscales`)
    for both candidates.

    A block that is present but unusable does not hide the other one: a malformed
    or wrongly-sized owner block falls back to the root's, and only when NEITHER
    can be read is a reason returned for the give-up notice.

    The third element is the key to hand :func:`_parse_ngff_metadata`, i.e. the
    selected array named relative to whichever group's block won.
    """
    if owner_attrs is not None:
        ms, unusable = _usable_multiscales(resolve_ngff_attrs(owner_attrs), ndim)
        if ms is not None:
            return ms, None, owner_key
        root_ms, root_problem = _usable_multiscales(resolve_ngff_attrs(attrs), ndim)
        if root_ms is not None:
            return root_ms, None, selected_key
        return None, unusable or root_problem, selected_key

    ms, unusable = _usable_multiscales(resolve_ngff_attrs(attrs), ndim)
    return ms, unusable, selected_key


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
        return None, _EMPTY_MULTISCALES_REASON
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
    attrs: Mapping[str, Any],
    ndim: int,
    owner_attrs: Optional[Mapping[str, Any]] = None,
) -> "Tuple[Optional[List[str]], Optional[str]]":
    """The custom ``axes`` label list to use, or why the declared one cannot be.

    Same ``(value, reason)`` contract as :func:`_usable_multiscales`. Read from
    the TOP level of each node deliberately: this is a non-NGFF convention
    (Keller-lab ``.zarr.zip`` files, say), so it never lives inside an ``ome``
    block.

    ROOT-FIRST, the opposite way round from ``multiscales``: a bare ``axes`` list
    names nothing, so no EVIDENCE that an owner's list is about the selected array
    is obtainable — and a same-length PERMUTATION of the root's would silently
    rewrite a decomposition the root already had right (a Keller-lab store's
    ``['z','y','x','time','camera','channel']`` on the intermediate group against
    the root's correct ``['time','camera',…]`` moves ``n_timepoints`` 2→4 and
    ``n_channels`` 6→16). The owner is a FALLBACK, which still fixes the
    bioformats2raw case, whose root carries only ``bioformats2raw.layout``.
    """
    axes, problem = _custom_axes_of(attrs, ndim)
    if axes is not None or owner_attrs is None:
        return axes, problem
    owner_axes, owner_problem = _custom_axes_of(owner_attrs, ndim)
    if owner_axes is not None:
        return owner_axes, None
    return None, problem or owner_problem


def _custom_axes_of(
    attrs: Mapping[str, Any], ndim: int
) -> "Tuple[Optional[List[str]], Optional[str]]":
    """One node's custom ``axes`` list, or why it cannot be used."""
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


def _normalised_dataset_path(raw: Any) -> str:
    """A ``datasets[].path`` (or a selected key) reduced to its canonical spelling.

    Surrounding slashes and ONE leading ``./`` are cosmetic — ``"0"``, ``"/0"``
    and ``"./0"`` all name the same child — and NGFF writers do emit the explicitly
    relative form. Treating them as distinct made a multi-level pyramid spelled
    ``"./0"``, ``"./1"`` match no entry at all, so it silently lost its voxel size
    (``len(datasets) > 1`` denies it the single-level fall-back to ``datasets[0]``),
    with no give-up notice either, since the block itself parsed fine.

    Anything still carrying a ``.`` SEGMENT after that — ``"."``, ``"./"``, a
    nested ``"a/./b"`` — normalises to ``""``, i.e. "names nothing": those are not
    a level's name, and zarr refuses such a path anyway. ``""`` never matches
    (:func:`_dataset_path_matches` requires a non-empty path), so an unspellable
    entry stays unmatched rather than becoming a bogus match for some other level.

    NGFF requires ``path`` to be a string, but some producers serialise numeric
    level names as numbers. Those real numeric scalars are intentionally coerced;
    booleans and structured/null values name nothing rather than plausible arrays
    called ``"True"``, ``"None"``, ``"[]"`` or ``"{}"``. Lookup and matching
    share this policy so malformed metadata cannot name different levels in the
    two halves of selection.
    """
    if isinstance(raw, bool) or not isinstance(raw, (str, Real)):
        return ""
    path = str(raw).strip("/")
    if path.startswith("./"):
        path = path[2:].strip("/")
    return "" if "." in path.split("/") else path


def _dataset_path_matches(entry: Mapping[str, Any], wanted: str) -> bool:
    """Whether a ``datasets[]`` entry's ``path`` names the array ``wanted``.

    The caller makes ``wanted`` relative to the group that owns the multiscales
    block, so only an exact match — of the two spellings NORMALISED
    (:func:`_normalised_dataset_path`) — can identify the selected level safely.
    """
    path = _normalised_dataset_path(entry.get("path", ""))
    return bool(path) and path == wanted


def _selected_dataset(
    datasets: Any, array_key: Optional[str]
) -> Optional[Mapping[str, Any]]:
    """The ``datasets[]`` entry describing the selected array, or ``None``.

    ``array_key`` is the SELECTED array named relative to the group whose block
    this is, and discovery supplies it on every call — not only when the user
    passed a key. The selection can land on a coarser pyramid LEVEL with no key at
    all (level 0's declared path may not resolve), and quoting level 0's spacing
    for it halves every number. Matched on the entry's ``path`` (see
    :func:`_dataset_path_matches`). ``None`` is passed only for a store whose
    selected array has no name relative to the owner at all.

    ``None`` — "unknowable", not "level 0" — when a key was matched against and
    nothing matched, unless the unmatched key is flat and the pyramid has one
    level. A nested unmatched key names an array outside the block's owner, so even
    a single-level pyramid cannot describe it. An exact match is the only one that
    can identify a level safely — exact after both spellings are normalised, so a
    block writing its own level as ``"./0"`` still matches the array at ``"0"``
    (:func:`_normalised_dataset_path`).

    Never raises: a ``datasets`` that is not a list of mappings is metadata this
    cannot read.
    """
    if not isinstance(datasets, list) or not datasets:
        return None
    first = datasets[0] if isinstance(datasets[0], Mapping) else None
    if array_key is None:
        return first
    wanted = _normalised_dataset_path(array_key)
    for d in datasets:
        if isinstance(d, Mapping) and _dataset_path_matches(d, wanted):
            return d
    return None if "/" in wanted or len(datasets) > 1 else first


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
    transforms = selected.get("coordinateTransformations")
    ms_transforms = ms.get("coordinateTransformations")
    scale = ngff_scale_transform(transforms)
    ms_scale = ngff_scale_transform(ms_transforms)
    if scale is None and _has_scale_entry(transforms):
        return None
    if ms_scale is None and _has_scale_entry(ms_transforms):
        return None
    if ms_scale is None:
        return scale
    if scale is None:
        return ms_scale
    if len(ms_scale) != len(scale):
        return None
    return [a * b for a, b in zip(scale, ms_scale)]


def _has_scale_entry(transforms: Any) -> bool:
    """Whether a transformations list declares a scale, valid or malformed."""
    return isinstance(transforms, list) and any(
        isinstance(transform, Mapping) and transform.get("type") == "scale"
        for transform in transforms
    )


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
    selected = _selected_dataset(datasets, array_key)
    axis_scale = _composed_scale(ms, selected) if selected is not None else None
    axis_scales = (
        tuple(float(value) for value in axis_scale)
        if axis_scale is not None and len(axis_scale) == len(shape)
        else None
    )
    voxel_size = _ngff_voxel_size(ms, datasets, shape, spatial_indices, array_key)

    axis_units = [
        str(a["unit"]) if isinstance(a, dict) and a.get("unit") else None
        for a in axes_raw
    ]
    spatial_units = [
        axis_units[index] for index in spatial_indices if axis_units[index]
    ]
    unit = spatial_units[0] if spatial_units else None

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
        axis_units=axis_units,
        axis_scales=axis_scales,
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
