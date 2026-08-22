"""Load Gaussian splat results from .gsplats.zarr format."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Sequence, Union

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
    ``ValueError``; consume those via the node tree directly (``read_gsplat_node``).

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
    from luxar.core.group.compositing import AUTHORED_APPEARANCE_ATTRS

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
    carried = {k: attrs[k] for k in sorted(AUTHORED_APPEARANCE_ATTRS) if k in attrs}
    if carried.get("colormap") == "custom":
        # The palette itself lives in a sibling `colormap_lut` array, which
        # this attrs-only read (and the archive peek in particular) cannot
        # reach. See the docstring: a dangling sentinel renders worse than the
        # writer's own default, so drop it and say so.
        del carried["colormap"]
        aprint(
            "⚠️  Source root carries a custom colormap LUT; a structure-only "
            "rebuild cannot carry the LUT array, so the palette is not "
            "preserved. Re-apply it on the result (e.g. "
            "`luxar gsplat convert --colormap NAME`)."
        )
    return carried


def _distinct(values: "Sequence[Any]") -> "List[Any]":
    """``values`` deduplicated BY EQUALITY, in first-seen order.

    Not a ``set``: an appearance value can be unhashable (``nd_transform`` is a
    dict of dicts), and the agreement rule below is defined by ``==`` anyway.
    """
    seen: List[Any] = []
    for value in values:
        if not any(value == other for other in seen):
            seen.append(value)
    return seen


def agreed_authored_appearance(
    paths: "Sequence[Union[str, Path]]",
    *,
    exclude: "Union[Collection[str], Mapping[str, str]]" = (),
) -> Dict[str, Any]:
    """The authored appearance N inputs UNANIMOUSLY agree on (``gsplat merge``).

    The N-input counterpart of :func:`read_authored_appearance`. A command with
    one input can simply carry that input's look; a command with several has to
    decide what "the" appearance even is, and the only answer that cannot be
    wrong is the one every input already agrees on. So a key is carried onto the
    merged root only when every input that HAS an opinion on it agrees; on any
    disagreement it is dropped and the merged artifact says nothing rather than
    promoting one input's choice over its siblings'.

    An input that does not carry a key **casts no vote**: a dataset that was
    never touched in the Layers panel authors nothing at all, and letting that
    silence veto a sibling's authored value would mean a single default-looking
    input erased the whole carry. At least one input must carry the key for it
    to appear.

    Silence is narrower on disk than it sounds, and that is not a bug in the
    rule: the writer STAMPS the identity values, so a store nobody ever tuned
    still has ``opacity=1.0`` / ``absorption=1.0`` / ``gamma=1.0`` /
    ``intensity=1.0`` / ``offset=0.0`` / ``layer=true`` / ``colormap="gray"`` on
    its root, and nothing on disk distinguishes those from someone deliberately
    choosing the identity. So they DO vote, and merging a tuned dataset with an
    untouched one legitimately drops them (with the warning saying so) rather
    than promoting one input's look over the other's. The keys with no stamped
    identity — ``blending_mode``, ``visible``, ``nd_transform``, ``join`` — are
    the ones where genuine silence occurs, and they are exactly the ones the
    no-vote clause rescues.

    That rule is :func:`~luxar.gsplats.io.save_gsplats.agreed_normalization_stats`
    verbatim — deliberately, since it is the same question about the same merge.
    The ONE divergence is that this one is LOUD: it warns per key it had to drop,
    naming the differing values. Normalization stats are machine-recorded, so a
    silent drop loses nothing a user chose; appearance is hand-authored in the
    Layers panel, and someone who tuned two datasets and merged them must be told
    which of their choices did not survive rather than discovering it by looking
    at the render (issue #1600 point 4: loud over silent wherever something
    cannot be preserved).

    ``exclude`` names keys the CALLING MODE invalidates, independently of whether
    the inputs agree — ``gsplat merge --as-dimension`` adds a dimension, so an
    ``nd_transform`` keyed by dimension name no longer describes the output's
    dimension set; ``--channel-colors`` bakes per-splat RGB, so an input's
    ``colormap`` no longer describes what is rendered. Those are dropped even
    under perfect agreement, and warn — but only when an input actually authored
    the key, since an exclusion nobody would have exercised is not news. Pass a
    ``{key: reason}`` mapping to have the reason quoted in the warning.

    Returns only the keys carried, so N inputs that authored nothing yield ``{}``
    and the writer's defaults apply unchanged. Feed the result to
    ``write_gsplats_tree(root_attrs=...)`` / ``GSplatData.save(root_attrs=...)``.
    """
    per_input = [read_authored_appearance(p) for p in paths]
    reasons: Mapping[str, str] = exclude if isinstance(exclude, Mapping) else {}
    excluded = set(exclude)

    carried: Dict[str, Any] = {}
    for key in sorted({k for attrs in per_input for k in attrs}):
        if key in excluded:
            because = reasons.get(key)
            aprint(
                f"⚠️  Not carrying authored '{key}' onto the merged root"
                + (f": {because}. " if because else ". ")
                + "Set it explicitly on the result if you want it."
            )
            continue
        distinct = _distinct([attrs[key] for attrs in per_input if key in attrs])
        if len(distinct) == 1:
            carried[key] = distinct[0]
        else:
            shown = ", ".join(repr(v) for v in distinct)
            aprint(
                f"⚠️  Inputs disagree on authored '{key}' ({shown}); dropping it "
                "from the merged root rather than picking one. Set it explicitly "
                "on the result if you want it."
            )
    return carried


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

        # Validate format
        format_type = root.attrs.get("format_type")
        if format_type != "gsplats_zarr":
            raise ValueError(
                f"Invalid format_type: {format_type}, expected 'gsplats_zarr'"
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

        # Read the node-tree subtree rooted at the file.
        from luxar.io._compiler.gsplat_tree import read_gsplat_node

        node = read_gsplat_node(root, root)

        # Gather root-level stats (fitting / pipeline / provenance / header).
        stats: Dict[str, Any] = {}
        if include_stats:
            if "fitting" in root:
                for key, value in root["fitting"].attrs.items():
                    stats[key] = value
            if "pipeline" in root:
                # Reduction/topology stats (lod_kind, method, coverage_inflation,
                # refine, ...) — split_fitting_info's fourth bucket. setdefault:
                # fitting/header keys keep precedence on any collision.
                for key, value in root["pipeline"].attrs.items():
                    stats.setdefault(key, value)
            if "provenance" in root:
                stats["provenance"] = dict(root["provenance"].attrs)
            stats["format_version"] = format_version
            stats["timestamp"] = root.attrs.get("timestamp")
            stats["luxar_gsplats_version"] = root.attrs.get("luxar_gsplats_version")
            if "description" in root.attrs:
                stats["description"] = root.attrs["description"]

        return node, stats

    finally:
        # Cleanup temporary directory if we extracted a compressed archive
        if temp_dir is not None and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
