"""Load Gaussian splat results from .gsplats.zarr format."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Mapping, Sequence, Union

from arbol import aprint

from luxar._zarr_compat import open_group as zc_open_group
from luxar.gsplats import GSplatData
from luxar.gsplats.io._archive import read_archive_root_attrs, resolve_store_path


def load_gsplats(
    path: str | Path,
    include_stats: bool = False,
) -> GSplatData:
    """Load Gaussian splats from .gsplats.zarr format.

    Supports both uncompressed (.gsplats.zarr) and compressed formats
    (.gsplats.zarr.zip, .gsplats.zarr.tar.gz). Compressed archives are
    automatically extracted to a temporary directory.

    Arrays are automatically decoded from their stored encoding (quantization,
    broadcasting, etc.) to float32.

    Only **matrix-shaped** node trees map to a ``GSplatData`` — a leaf, or a
    ``kind=lod`` group whose children are all leaves (the substitutive × additive
    matrix). A genuinely nested tree (a ``kind=partition`` root, or a lod group
    with non-leaf children) has no flat ``GSplatData`` equivalent and raises
    ``ValueError``; use ``load_default_gsplats`` to materialize the default-rendered
    selection, or consume the node tree directly with ``load_gsplat_node``.

    Args:
        path: Path to .gsplats.zarr directory or compressed archive
        include_stats: Whether to include fitting/provenance metadata in stats

    Returns:
        GSplatData with decoded arrays and optional stats

    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If the format is invalid/incompatible, or the file is a
            non-matrix (partition/nested) tree.
    """
    node, stats = load_gsplat_node(path, include_stats=include_stats)
    return GSplatData.from_tree(node, stats=stats)


def load_default_gsplats(
    path: str | Path,
    include_stats: bool = False,
) -> GSplatData:
    """Load the splats selected by the tree's default rendering semantics.

    Matrix-shaped inputs retain their existing substitutive/additive structure.
    For a partition or nested tree, all partition children, the default (finest)
    child of each substitutive LOD group, and every additive sub-LOD are
    materialized as one flat in-memory dataset. Root stats are deliberately
    retained unchanged when requested because this helper is read-only; a
    writer that changes topology must scrub structure-scoped metadata itself.
    """
    node, stats = load_gsplat_node(path, include_stats=include_stats)
    return GSplatData.from_default_selection(node, stats=stats)


def read_authored_appearance(path: str | Path) -> Dict[str, Any]:
    """Read the authored appearance attrs off a ``.gsplats.zarr`` ROOT.

    A structure-only rebuild (``gsplat lod`` and friends) constructs fresh nodes
    that know nothing about the input's appearance, so without this the authored
    values are silently dropped and the writer's own defaults take their place —
    ``blending_mode`` vanishes and ``opacity``/``gamma``/``intensity``/
    ``absorption`` snap back to their identity. Feed the result to
    ``write_gsplats_tree(root_attrs=...)`` (or ``GSplatData.save(root_attrs=...)``).

    Works on a ``.gsplats.zarr`` directory and on a ``.gsplats.zarr.zip`` /
    ``.gsplats.zarr.tar.gz`` archive alike — both are first-class inputs to the
    rebuild commands, so appearance must survive both (#1604).

    An authored ``colormap`` rides along too since #1600 — the writer stopped
    manufacturing a ``"gray"`` that shadowed it and the viewer composes it
    root→leaf. The one exception is the ``"custom"`` sentinel, which is DROPPED
    (with a warning) rather than carried: it names a sibling ``colormap_lut``
    ARRAY, and this function reads the root's attrs only — for an archive input
    it never opens the store at all. Carrying the bare sentinel would write a
    dangling reference and the viewer would fall back to viridis, which is
    worse than the writer's own default. Re-apply such a palette explicitly
    (e.g. ``gsplat convert --colormap NAME``) after the rebuild.

    Not every dropped attr is fixed by this: the 4x4 ``transform`` is
    deliberately left behind because feeding a stored (column-major) matrix
    back through the writer transposes it a second time. It is documented on
    the key set below.

    Only keys actually present are returned, so an input that authored nothing
    yields ``{}`` and the writer's defaults apply unchanged. Missing/unreadable
    stores yield ``{}`` rather than raising: this is a best-effort carry-over
    alongside the real load, which reports its own errors.

    See :data:`~luxar.core.group.compositing.AUTHORED_APPEARANCE_ATTRS` for the
    key set and https://github.com/royerlab/luxar/issues/1600 for the invariant.
    """
    carried, has_custom_colormap = _read_authored_appearance(path)
    if has_custom_colormap:
        _warn_custom_colormap_loss()
    return carried


def read_rebuild_root_attrs(path: str | Path) -> Dict[str, Any]:
    """Read root attrs a structure-only rebuild can preserve unchanged.

    ``dimension_metadata`` describes center columns, so it survives topology,
    encoding, filtering, and culling rebuilds. Geometric transforms must keep
    using :func:`read_authored_appearance` because scaling or rotating centers
    invalidates the recorded axis descriptors.
    """
    attrs = _read_root_attrs(path)
    carried, has_custom_colormap = _authored_appearance_from_attrs(attrs)
    if has_custom_colormap:
        _warn_custom_colormap_loss()
    if "dimension_metadata" in attrs:
        carried["dimension_metadata"] = attrs["dimension_metadata"]
    return carried


#: What a ``colormap: "custom"`` costs, shared verbatim by the one-input carry
#: and the N-input merge so both explain the same loss the same way.
_CUSTOM_COLORMAP_LOSS = (
    "a custom colormap LUT lives in a sibling `colormap_lut` ARRAY, which the "
    "attrs-only appearance carry cannot reach (an archive input is not "
    "extracted — only the root's metadata is read), so the palette is not "
    "preserved"
)

_CUSTOM_COLORMAP_REMEDY = (
    ". Re-apply it on the result (e.g. `luxar gsplat convert --colormap NAME`)."
)


def _read_authored_appearance(path: str | Path) -> "tuple[Dict[str, Any], bool]":
    """:func:`read_authored_appearance` without the warning, plus the flag.

    Returns ``(carried, root_declares_custom_colormap)``. The flag is what the
    ``"custom"`` strip THREW AWAY, and the N-input merge needs it: for a single
    input the strip is the whole story (the writer's own default takes over),
    but with siblings in play a stripped input is indistinguishable from one
    with no palette opinion at all — so it would silently adopt a sibling's
    palette. :func:`agreed_authored_appearance` uses the flag to refuse
    ``colormap`` outright instead, and to warn ONCE for N such inputs rather
    than once per input.
    """
    return _authored_appearance_from_attrs(_read_root_attrs(path))


def _authored_appearance_from_attrs(
    attrs: "Mapping[str, Any]",
) -> "tuple[Dict[str, Any], bool]":
    from luxar.core.group.compositing import AUTHORED_APPEARANCE_ATTRS

    carried = {k: attrs[k] for k in sorted(AUTHORED_APPEARANCE_ATTRS) if k in attrs}
    has_custom_colormap = carried.get("colormap") == "custom"
    if has_custom_colormap:
        # The palette itself lives in a sibling `colormap_lut` array, which
        # this attrs-only read (and the archive peek in particular) cannot
        # reach. See the docstring: a dangling sentinel renders worse than the
        # writer's own default, so drop it — and hand the fact back to the
        # caller, which is what says so.
        del carried["colormap"]
    return carried, has_custom_colormap


def _warn_custom_colormap_loss() -> None:
    aprint(
        '⚠️  The source root declares colormap: "custom"; '
        + _CUSTOM_COLORMAP_LOSS
        + _CUSTOM_COLORMAP_REMEDY
    )


def _read_root_attrs(path: str | Path) -> Dict[str, Any]:
    """Read root attrs from a directory or archive, best-effort."""
    p = Path(path)
    try:
        if p.is_dir():
            # The facade, not a bare ``zarr.open_group``: every zarr read routes
            # through ``luxar._zarr_compat``, and it opts reads out of
            # consolidated metadata, so a directory store is read from the ROOT
            # NODE'S OWN metadata document — the same one the archive peek below
            # reads, and the same one the loader's own group-open sees. Which
            # document that is depends on the store's format: `.zattrs` at
            # format 2, the `attributes` half of `zarr.json` at format 3, where
            # the node document also carries the consolidated index of the tree
            # (which the facade ignores). Measured on zarr 3.3, the facade and a
            # bare open agree for a ROOT GROUP'S OWN ATTRS even with a stale
            # consolidated index present, so this is the module convention
            # holding rather than a divergence being papered over.
            root = zc_open_group(p, mode="r")
            attrs = dict(root.attrs)
        else:
            # An archive is peeked, not extracted: only the ROOT NODE'S metadata
            # member is read — `.zattrs` at format 2, `zarr.json` at format 3,
            # where the attributes are unwrapped out of the node document — and
            # nothing is written to disk. The two are budgeted differently
            # because a format-3 root document also carries the consolidated
            # index of the whole tree; an over-budget member warns rather than
            # silently answering {} (see `_archive._warn_size_refusal` — which
            # the `except` below swallows if it is promoted to an error).
            # A regular file that is not an archive yields {} from the helper.
            attrs = read_archive_root_attrs(p)
    except Exception:
        return {}
    return attrs


def _distinct(values: "Sequence[Any]") -> "List[Any]":
    """``values`` deduplicated BY EQUALITY, in first-seen order.

    Not a ``set``: an appearance value can be unhashable (``nd_transform`` is a
    dict of dicts), and the agreement rule below is defined by ``==`` anyway.

    ``nan`` is a deliberate NON-exception: ``nan != nan``, so two roots both
    carrying (say) ``opacity=nan`` read as a disagreement and the key is
    dropped. Left as is because (a) it fails in the safe direction — the writer
    then stamps a renderable identity instead of propagating a value that makes
    the splats vanish — and (b) it is unreachable through the writer anyway:
    ``validate_render_attrs`` refuses a non-finite value for every numeric
    appearance attr ("Opacity must be between 0.0 and 1.0, got nan",
    "Absorption must be finite, got nan", and likewise gamma / intensity /
    offset), measured. A nan-aware walk would have to recurse into the floats
    nested inside an ``nd_transform`` dict for no reachable gain.
    """
    seen: List[Any] = []
    for value in values:
        if not any(value == other for other in seen):
            seen.append(value)
    return seen


#: Sentinel for "this input said nothing about this key" — distinct from every
#: value an attr can legally hold (``None`` included).
_NO_VALUE: Any = object()


#: Keys whose ABSENCE from a root is a defined VALUE rather than silence, and
#: what it means. Only ``visible``: the viewer reads a missing ``visible`` as
#: visible (``node.attrs.visible !== false`` in ``ui/layers/layer-state.ts``),
#: so an input that does not carry the key is positively saying "shown", not
#: "no opinion". Without this the no-vote clause would let ONE input's
#: ``visible=false`` carry onto the merged root and open the whole merged
#: dataset hidden. Contrast ``blending_mode``, where absence genuinely means
#: "inherit / no opinion" and the no-vote clause is right.
_ABSENCE_MEANS_VALUE: Dict[str, Any] = {"visible": True}


def _appearance_votes(
    key: str,
    per_input: "Sequence[Mapping[str, Any]]",
    input_has_colors: "Union[Sequence[bool], None]" = None,
) -> "List[tuple[int, Any]]":
    """The values that actually COUNT as an opinion on ``key``, in input order.

    Returned as ``(input index, value)`` pairs rather than bare values so a
    disagreement can name WHICH input dissented — with three inputs where only
    the third differs, a values-only message is byte-identical to the two-input
    one and leaves the user with no idea which store to re-tune.

    Two kinds of non-opinion are filtered out here, which is what makes the
    unanimity rule usable in practice:

    * an input that does not carry the key at all (the no-vote clause) — unless
      the key is in :data:`_ABSENCE_MEANS_VALUE`, where absence is itself a
      value and votes as one;
    * an input whose value is exactly what the WRITER manufactures for that key
      (:data:`~luxar.core.group.compositing.WRITER_STAMPED_APPEARANCE_DEFAULTS`).
      ``colormap="gray"`` is silence only on a colorless input: the writer never
      stamps it on a colored store, where it is necessarily authored and must
      vote like any other palette.

    That second clause is the load-bearing one and it has a real cost, stated
    plainly: a store nobody ever touched is stamped ``opacity=1.0`` /
    ``absorption=1.0`` / ``gamma=1.0`` / ``intensity=1.0`` / ``offset=0.0`` /
    ``layer=true`` (plus ``colormap="gray"`` on a colorless store), and NOTHING
    on disk distinguishes those from an author who deliberately chose the
    identity. So treating them as silence does lose a deliberate choice:
    merging a store where the user deliberately set ``opacity=1.0`` with a
    sibling at ``0.75`` now carries ``0.75``.

    It is still the right trade, because the alternative is not "keep the
    deliberate 1.0" — it is what this code did before: seven of the eleven keys
    disagree on the single commonest merge (a tuned dataset + a freshly fitted
    one), all seven are dropped, and the writer then stamps its defaults back,
    which IS the untouched input's value. The user got the untouched look
    either way; the only difference was seven warning lines announcing a loss
    that had already happened silently. Now the tuned look survives and only a
    genuine 0.75-vs-0.5 disagreement drops and warns.
    """
    from luxar.core.group.compositing import WRITER_STAMPED_APPEARANCE_DEFAULTS

    manufactured = WRITER_STAMPED_APPEARANCE_DEFAULTS.get(key, _NO_VALUE)
    absent = _ABSENCE_MEANS_VALUE.get(key, _NO_VALUE)
    votes: List[tuple[int, Any]] = []
    for index, attrs in enumerate(per_input):
        value = attrs.get(key, absent)
        writer_manufactured = value == manufactured
        if (
            key == "colormap"
            and input_has_colors is not None
            and input_has_colors[index]
        ):
            writer_manufactured = False
        if value is _NO_VALUE or writer_manufactured:
            continue
        votes.append((index, value))
    return votes


def _dropped_outcome(key: str, *, output_has_colors: bool | None = None) -> str:
    """What actually LANDS on the merged root for a key that is not carried.

    "Dropped" never means the output has a hole where the attr would be: the
    writer runs afterwards. Spelling out the real outcome per key keeps the
    warning from implying a neutrality that does not exist — for an
    identity-stamped attr the writer's default is precisely the untouched
    input's value, so a drop is not a tie-break, it is a side.
    """
    from luxar.core.group.compositing import WRITER_STAMPED_APPEARANCE_DEFAULTS

    if key == "colormap":
        # The one CONDITIONAL stamp: `apply_gsplat_group_attrs` manufactures
        # "gray" only for a colorless leaf with no ancestor palette, so on a
        # colored output nothing is written at all.
        if output_has_colors is False:
            stamped = WRITER_STAMPED_APPEARANCE_DEFAULTS[key]
            return (
                f"the writer then stamps its own {stamped!r}, which is what an "
                "untouched input carries"
            )
        return (
            "the merged root gets no palette of its own (the writer's 'gray' "
            "default is stamped only on a colorless store)"
        )
    if key in WRITER_STAMPED_APPEARANCE_DEFAULTS:
        stamped = WRITER_STAMPED_APPEARANCE_DEFAULTS[key]
        return (
            f"the writer then stamps its own {stamped!r}, which is what an "
            "untouched input carries"
        )
    return "the merged root leaves it unset, so the viewer's own default applies"


def _outcome_clause(
    key: str,
    *,
    after: "Union[str, None]",
    output_has_colors: "Union[bool, None]" = None,
) -> str:
    """The " — what actually lands" clause, punctuated to follow ``after``.

    An exclusion reason can itself end in an em-dash clause (the mixed-colors
    one does), and chaining a second one onto it produced a 430-character
    run-on sentence with two dashes in it. Start a new sentence instead when
    the reason already spent the dash — same information, three readable
    sentences in a terminal rather than one.
    """
    outcome = _dropped_outcome(key, output_has_colors=output_has_colors)
    if after and "—" in after:
        return f". {outcome[:1].upper()}{outcome[1:]}."
    return f" — {outcome}."


def _input_label(
    paths: "Sequence[Union[str, Path]]",
    votes: "Sequence[tuple[int, Any]]",
    value: Any,
) -> str:
    """Name the FIRST input that voted ``value``, briefly.

    A disagreement that lists only the differing values is unactionable once
    there are more than two inputs: with three stores where the third dissents,
    the message is byte-identical to the two-input one and the user cannot tell
    which one to re-tune. Matched by ``==`` so it agrees with :func:`_distinct`,
    which defines the values being listed.

    The basename, not the path, once the path is long AND that basename is
    unique among the inputs: these are ``.gsplats.zarr`` stores under a working
    directory, and a full path per value turns a one-line warning into a
    paragraph. A colliding basename would erase the distinction this label
    exists to provide, so those inputs keep their full paths. Short paths (what
    a user actually typed on the command line) are shown verbatim.
    """
    index = next((i for i, voted in votes if voted == value), None)
    if index is None or index >= len(paths):  # pragma: no cover - defensive
        return "an input"
    text = str(paths[index])
    if len(text) <= 40:
        return text
    basename = Path(text).name
    return (
        basename
        if sum(Path(str(path)).name == basename for path in paths) == 1
        else text
    )


def agreed_authored_appearance(
    paths: "Sequence[Union[str, Path]]",
    *,
    exclude: "Union[AbstractSet[str], Mapping[str, str]]" = frozenset(),
    input_has_colors: "Union[Sequence[bool], None]" = None,
    output_has_colors: "Union[bool, None]" = None,
) -> Dict[str, Any]:
    """The authored appearance N inputs UNANIMOUSLY agree on (``gsplat merge``).

    The N-input counterpart of :func:`read_authored_appearance`. A command with
    one input can simply carry that input's look; a command with several has to
    decide what "the" appearance even is, and the only answer that cannot be
    wrong is the one every input already agrees on. So a key is carried onto the
    merged root only when every input that HAS an opinion on it agrees; on any
    disagreement it is dropped and the merged artifact says nothing rather than
    promoting one input's choice over its siblings'. At least one input must
    have an opinion for the key to appear at all.

    What counts as "having an opinion" is :func:`_appearance_votes`, and it is
    the subtle half of this function. Two kinds of non-opinion are filtered out
    before the vote: an input that does not carry the key at all, and an input
    whose value is exactly the one the WRITER manufactures for that key. The
    second is what makes the rule usable — without it the commonest merge of
    all (a tuned dataset + a freshly fitted one) disagrees on seven keys and
    reverts to the untouched look while shouting about it. It also has a real
    cost: an author who deliberately set ``opacity=1.0`` is indistinguishable
    on disk from one who never touched it, and loses to a sibling's ``0.75``.
    That whole trade-off is argued out on :func:`_appearance_votes`; read it
    before changing the rule.

    ``visible`` is the one key where ABSENCE votes, because the format gives
    absence a meaning there (a missing ``visible`` is visible) — see
    :data:`_ABSENCE_MEANS_VALUE`. So ``visible=false`` rides along only when
    EVERY input hides; one hidden input plus one silent one is a disagreement,
    not a unanimous hide.

    ``colormap`` has one merge-specific absence rule supplied through
    ``input_has_colors``: a colored input with no palette relies on per-splat
    RGB, so carrying a sibling's palette would repaint it. In that case the
    palette is excluded and the loss is announced.

    That rule is :func:`~luxar.gsplats.io.save_gsplats.agreed_normalization_stats`
    verbatim — deliberately, since it is the same question about the same merge.
    The ONE divergence is that this one is LOUD: it warns per key it had to drop,
    naming the differing values (each with the input that voted it, so a
    dissenter among N is identifiable) and what lands instead. Normalization
    stats are
    machine-recorded, so a silent drop loses nothing a user chose; appearance is
    hand-authored in the Layers panel, and someone who tuned two datasets and
    merged them must be told which of their choices did not survive rather than
    discovering it by looking at the render (issue #1600 point 4: loud over
    silent wherever something cannot be preserved).

    ``exclude`` names keys the CALLING MODE invalidates, independently of whether
    the inputs agree — see
    :func:`~luxar.cli.gsplat_ops.transforms.merge._mode_invalidated_appearance`
    for the two cases ``gsplat merge`` has (both about ``colormap``, both about
    the merge having MANUFACTURED per-splat RGB). Those are dropped even under
    perfect agreement, and warn — but only when an input actually authored the
    key, an authored value being one that survives the vote filter above, so an
    exclusion nobody would have exercised stays quiet. Pass a ``{key: reason}``
    mapping to have the reason quoted in the warning. Typed as a SET (not a
    ``Collection``) so a bare ``str`` is a type error rather than a silent
    iteration over its characters, which would exclude nothing.

    ``colormap`` is additionally refused, without the caller asking, when ANY
    input root declares the ``"custom"`` sentinel: that palette cannot be
    carried (its LUT is a sibling array), and letting the input fall through as
    "no opinion" would hand the merged root a SIBLING's palette — repainting
    the custom-LUT splats with someone else's ramp.

    ``output_has_colors`` lets the warning describe the writer's conditional
    ``colormap="gray"`` stamp accurately: a colorless merged output gets that
    stamp after a disagreement, while a colored output gets no root palette.

    Returns only the keys carried, so N inputs that authored nothing yield ``{}``
    and the writer's defaults apply unchanged. Feed the result to
    ``write_gsplats_tree(root_attrs=...)`` / ``GSplatData.save(root_attrs=...)``.
    """
    reads = [_read_authored_appearance(p) for p in paths]
    per_input = [attrs for attrs, _ in reads]
    if input_has_colors is not None and len(input_has_colors) != len(paths):
        raise ValueError("input_has_colors must have one entry per input path")
    reasons: Dict[str, str] = dict(exclude) if isinstance(exclude, Mapping) else {}
    excluded = set(exclude)
    # Keys whose exclusion has already been explained in full by a warning of
    # its own, so the per-key line below would only repeat it.
    explained: set[str] = set()

    if input_has_colors is not None and any(
        has_colors and "colormap" not in attrs
        for attrs, has_colors in zip(per_input, input_has_colors)
    ):
        excluded.add("colormap")
        reasons.setdefault(
            "colormap",
            "a colored input authored no palette and therefore relies on its "
            "per-splat RGB; a sibling's palette would repaint it",
        )

    n_custom = sum(1 for _, has_custom in reads if has_custom)
    if n_custom:
        # ONE line for N inputs: the per-input strip used to warn once per
        # input, so merging four custom-LUT stores printed the same sentence
        # four times.
        aprint(
            f"⚠️  {n_custom} of {len(reads)} input roots declare colormap: "
            f'"custom"; {_CUSTOM_COLORMAP_LOSS}. No palette is carried onto the '
            "merged root at all — adopting a sibling's would repaint those "
            "splats with someone else's ramp" + _CUSTOM_COLORMAP_REMEDY
        )
        excluded.add("colormap")
        explained.add("colormap")

    carried: Dict[str, Any] = {}
    for key in sorted({k for attrs in per_input for k in attrs}):
        votes = _appearance_votes(key, per_input, input_has_colors=input_has_colors)
        if not votes:
            # Every input carries only what the writer manufactured, so there is
            # no authored value to preserve OR to lose: not carrying it is a
            # no-op (the writer stamps the same value back) and saying anything
            # would be noise about a choice nobody made. This is also why the
            # canonical `merge ch0 ch1 -o out --channel-colors …` no longer
            # warns about excluding a palette neither input ever set.
            continue
        if key in excluded:
            if key not in explained:
                because = reasons.get(key)
                aprint(
                    f"⚠️  Not carrying authored '{key}' onto the merged root"
                    + (f": {because}" if because else "")
                    + _outcome_clause(
                        key,
                        after=because,
                        output_has_colors=output_has_colors,
                    )
                    + " Set it explicitly on the result if you want it."
                )
            continue
        distinct = _distinct([value for _, value in votes])
        if len(distinct) == 1:
            carried[key] = distinct[0]
        else:
            shown = ", ".join(
                f"{value!r} from {_input_label(paths, votes, value)}"
                for value in distinct
            )
            aprint(
                f"⚠️  Inputs disagree on authored '{key}' ({shown}); not "
                "carrying it rather than picking one — "
                f"{_dropped_outcome(key, output_has_colors=output_has_colors)}. "
                "Set it explicitly on the result if you want it."
            )
    return carried


def read_gsplat_root_stats(root: Any, *, include_stats: bool = True) -> Dict[str, Any]:
    """Validate a standalone gsplat root and return its persisted statistics."""
    format_type = root.attrs.get("format_type")
    if format_type != "gsplats_zarr":
        raise ValueError(
            f"Invalid format_type: {format_type}, expected 'gsplats_zarr'. "
            "If this path is a node-tree subtree, pass its standalone "
            ".gsplats.zarr store root instead."
        )

    from luxar.gsplats.io.save_gsplats import SUPPORTED_FORMAT_VERSIONS

    format_version = root.attrs.get("format_version")
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"Unsupported format_version: {format_version!r} "
            f"(expected one of {SUPPORTED_FORMAT_VERSIONS}). The on-disk "
            f"format is a detached node-tree subtree. Convert legacy "
            f"v1.x / v2.0 files (and old substitutive directories) with "
            f"`luxar gsplat migrate-format <input> <output.gsplats.zarr>`."
        )

    if not include_stats:
        return {}

    stats: Dict[str, Any] = {}
    if "fitting" in root:
        stats.update(dict(root["fitting"].attrs))
    if "pipeline" in root:
        for key, value in root["pipeline"].attrs.items():
            stats.setdefault(key, value)
    if "provenance" in root:
        stats["provenance"] = dict(root["provenance"].attrs)
    stats["format_version"] = format_version
    stats["timestamp"] = root.attrs.get("timestamp")
    stats["luxar_gsplats_version"] = root.attrs.get("luxar_gsplats_version")
    if "description" in root.attrs:
        stats["description"] = root.attrs["description"]
    return stats


def load_gsplat_node(
    path: str | Path,
    include_stats: bool = False,
) -> "tuple[Any, Dict[str, Any]]":
    """Load the raw v3.0 node-tree (a :class:`~luxar.gsplats.tree.GSplatNode`).

    Unlike :func:`load_gsplats`, this does NOT flatten to a ``GSplatData`` and so
    works for **every** shape — including ``kind=partition`` roots and genuinely
    nested trees that have no flat matrix equivalent. Use this to graft a
    standalone ``.gsplats.zarr`` into a scene, or to inspect a partition/nested
    file. Handles ``.zip`` / ``.tar.gz`` archives transparently.

    Args:
        path: Standalone gsplat store or supported compressed archive.
        include_stats: Whether to read the optional root-level statistics.

    Returns:
        ``(node, stats)`` — the tree root and the (optional) root-level stats.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GSplats zarr not found: {path}")

    # Archive resolution is shared with inspect_gsplats_zarr; this caller reads
    # array data, so it wants a resolved DIRECTORY store and does not opt into
    # `flat_zip_in_place` (the inspector's temp-space shortcut — see its
    # docstring). Every archive shape, flat included, resolves by extraction.
    zarr_path, temp_dir = resolve_store_path(path)

    try:
        # Open zarr store
        root = zc_open_group(str(zarr_path), mode="r")

        stats = read_gsplat_root_stats(root, include_stats=include_stats)

        # Read the node-tree subtree rooted at the file.
        from luxar.io._compiler.gsplat_tree import read_gsplat_node

        node = read_gsplat_node(root, root)

        return node, stats

    finally:
        # Cleanup temporary directory if we extracted a compressed archive
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
