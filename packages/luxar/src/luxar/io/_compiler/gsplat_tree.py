"""Root-agnostic Gaussian-splat node-subtree serializer (node-tree format).

This is the **single authoring path** shared by the standalone ``.gsplats.zarr``
writer and the scene gsplat-node embed — there is no parallel writer. It writes a
:class:`luxar.gsplats.tree.GSplatNode` subtree into *any* ``zarr.Group`` (a scene
child group, or the root of a standalone file), reusing the exact leaf machinery
the scene compiler uses:

* :func:`~luxar.io._compiler.gsplat_assembly.apply_gsplat_spatial_ordering`
* :func:`~luxar.io._compiler.gsplat_assembly.write_gsplat_arrays`
* :func:`~luxar.io._compiler.gsplat_assembly.apply_gsplat_group_attrs`

Because leaves go through those same functions, a standalone leaf is byte-identical
to a scene leaf (same arrays, chunking, ordering, ``position_bounds`` + render
attrs) by construction, and they share one chunk-size formula.

On-disk grammar (the node tree):

* **leaf** → ``type=gsplats``; a single splat set writes arrays directly, an
  additive ladder writes ``additive_<i>/`` subgroups + ``n_additive_sublods``.
* **lod group** → ``type=group, kind=lod``; children written **coarsest→finest**
  as ``child_<i>/`` (the in-memory tree is also coarsest-first, so the writer
  writes them straight through with no reversal), each carrying its
  ``coverage_fraction`` selector threshold + LOD provenance.
* **partition group** → ``type=group, kind=partition``; children as ``part_<i>/``.

Every node carries a ``position_bounds`` attr (groups = union of children) so the
viewer can frame a bare-node file on load.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, cast

import numpy as np
import zarr

from ...encoding import ArrayEncoder, EncodingMode
from ...typing_utils.constants import DEFAULT_TRUNCATION_RADIUS
from .colormap import write_colormap_lut_if_needed
from .context import DatasetCtx, OrderingCtx
from .gsplat_assembly import (
    apply_gsplat_group_attrs,
    apply_gsplat_spatial_ordering,
    validate_gsplat_inputs,
    write_gsplat_arrays,
)

if TYPE_CHECKING:
    from luxar.gsplats.tree import (
        GSplatLeaf,
        GSplatNode,
    )

#: Provenance / selector attr keys carried per-node in a leaf's ``meta`` and
#: surfaced verbatim onto the node's zarr ``.zattrs``.
#:
#: ``selector`` (a kind=lod GROUP's meta, never a leaf's in practice) names the
#: UNITS of the children's ``coverage_fraction`` thresholds and must round-trip
#: WITH them: a legacy store's authored diagonal-metric values re-saved under a
#: fresh ``"screen-area"`` stamp would be misread by the viewer. The reader
#: whitelists it to the known modes so a stale pre-v3.2 value can't ride along.
_NODE_META_ATTR_KEYS = (
    "coverage_fraction",
    "selector",
    "compression_factor",
    "parent_method",
    "level_index",
)


# ────────────────────────────────────────────────────────────────────────
# Context helpers
# ────────────────────────────────────────────────────────────────────────


def make_ordering_ctx(ordering: str) -> OrderingCtx:
    """Build an :class:`OrderingCtx` from an ``ordering`` string.

    ``"none"`` disables spatial ordering; ``"morton"`` / ``"hilbert"`` enable it.
    """
    if ordering == "none":
        return OrderingCtx(enable_spatial_index=False, ordering_method="hilbert")
    if ordering not in ("morton", "hilbert"):
        raise ValueError(
            f"ordering must be 'morton', 'hilbert', or 'none'; got {ordering!r}"
        )
    return OrderingCtx(
        enable_spatial_index=True,
        ordering_method=cast(Literal["morton", "hilbert"], ordering),
    )


def make_dataset_ctx(
    encoding_mode: EncodingMode = EncodingMode.AUTO,
    *,
    compressor: Optional[Any] = None,
) -> DatasetCtx:
    """Build a standalone :class:`DatasetCtx` (encoder + mode + compressor).

    float16 is intentionally disabled (``ArrayEncoder`` default): COORDINATE
    centers must stay float32 for TypeScript/WebGL compatibility, and float16
    on coordinates is a precision footgun.
    """
    return DatasetCtx(
        encoder=ArrayEncoder(),
        encoding_mode=encoding_mode,
        compressor=compressor,
    )


# ────────────────────────────────────────────────────────────────────────
# Writer
# ────────────────────────────────────────────────────────────────────────


def _write_single_splat_set(
    group: zarr.Group,
    sublod: Any,  # AdditiveSubLOD
    *,
    dataset_ctx: DatasetCtx,
    ordering_ctx: OrderingCtx,
    lightweight: bool,
    store: zarr.Group,
    attrs: Optional[Dict[str, Any]] = None,
    scene_tone_mapping: Optional[str] = None,
    barrier_dims: Optional[Sequence[int]] = None,
    inherited_colormap: Optional[str] = None,
    warn_on_missing_tone_mapping: bool = True,
) -> Dict[str, Any]:
    """Order + write one splat set's arrays into ``group``; return metadata.

    ``lightweight=True`` writes only the data attrs (used for ``additive_<i>``
    sub-LOD subgroups — no rendering defaults); ``False`` runs the full
    :func:`apply_gsplat_group_attrs` (rendering defaults + ``position_bounds``).

    ``scene_tone_mapping`` (the scene's ``viewer_config.tone_mapping``, or
    ``None`` for a standalone file) is threaded into
    :func:`apply_gsplat_group_attrs` so a scene leaf written through this path
    gets the same colormap-LUT tone handling as the compiler's own writer.

    ``inherited_colormap`` is the palette an in-flight ``kind=lod`` /
    ``kind=partition`` ancestor authored — see
    :func:`~luxar.io._compiler.gsplat_assembly.inherited_gsplat_colormap` for
    why this path hands it down instead of walking the store.
    """
    # Shape normalization only: write_gsplat_leaf guarantees the whole leaf
    # already passed preflight_validate_leaf, so the O(N) value scans are
    # skipped here (each leaf is value-scanned exactly once).
    centers, amplitudes, cholesky, colors, n_splats, n_dims, chol_uniform = (
        validate_gsplat_inputs(
            sublod.centers,
            sublod.amplitudes,
            sublod.cholesky_factors,
            sublod.colors,
            check_values=False,
        )
    )
    truncation_radius = float(
        (attrs or {}).get("truncation_radius", sublod.truncation_radius)
    )
    (
        centers,
        amplitudes,
        cholesky,
        colors,
        label_ids,
        ordering_data,
        centers_encoding_plan,
    ) = apply_gsplat_spatial_ordering(
        centers,
        amplitudes,
        cholesky,
        colors,
        sublod.label_ids,
        n_splats,
        n_dims,
        chol_uniform,
        ordering_ctx,
        coverage_sigma=truncation_radius,
        barrier_dims=barrier_dims,
        dataset_ctx=dataset_ctx,
    )
    metadata = write_gsplat_arrays(
        group,
        centers,
        amplitudes,
        cholesky,
        colors,
        label_ids,
        sublod.label_vocabulary,
        n_splats,
        n_dims,
        chol_uniform,
        ordering_data,
        centers_encoding_plan,
        dataset_ctx,
    )

    if lightweight:
        # Per-additive-sub-LOD attrs — data only, no rendering defaults. (Both
        # the scene and standalone additive ladders are written here; there is
        # no separate compiler multi-LOD writer.)
        group.attrs["type"] = "gsplats"
        group.attrs["n_splats"] = metadata["n_splats"]
        group.attrs["ndim"] = metadata["ndim"]
        group.attrs["has_colors"] = metadata["has_colors"]
        group.attrs["has_label_ids"] = metadata["has_label_ids"]
        if metadata["has_label_ids"]:
            group.attrs["label_vocabulary"] = metadata["label_vocabulary"]
        group.attrs["amplitude_range"] = metadata["amplitude_range"]
        group.attrs["center_bounds"] = metadata["center_bounds"]
        group.attrs["ordering"] = metadata["ordering"]
        if metadata["ordering"] != "none":
            for key in (
                "ordering_min",
                "ordering_max",
                "ordering_bits_per_dim",
                "chunk_size",
                "slice_dims",
                "ordering_dims",
            ):
                if key in metadata:
                    group.attrs[key] = metadata[key]
        else:
            group.attrs["chunk_size"] = min(1024, max(64, metadata["n_splats"]))
        if sublod.stats:
            group.attrs["lod_stats"] = _safe_lod_stats(sublod.stats)
        group.attrs["truncation_radius"] = truncation_radius
    else:
        leaf_attrs = dict(attrs or {})
        leaf_attrs.setdefault("truncation_radius", truncation_radius)
        # Per-additive-sub-LOD stats (e.g. cumulative_psnr_db) — written here on
        # the single-set fast path too, mirroring the lightweight ladder branch,
        # so the reader's unconditional lod_stats read round-trips faithfully.
        if sublod.stats and "lod_stats" not in leaf_attrs:
            leaf_attrs["lod_stats"] = _safe_lod_stats(sublod.stats)
        apply_gsplat_group_attrs(
            group,
            metadata,
            leaf_attrs,
            store,
            scene_tone_mapping=scene_tone_mapping,
            lut_tone_mapping_warned=False,
            inherited_colormap=inherited_colormap,
            warn_on_missing_tone_mapping=warn_on_missing_tone_mapping,
        )
    return metadata


def _validate_ladder_color_and_dim_consistency(sublods: Sequence[Any]) -> None:
    """Reject an additive ladder the viewer could not load.

    Sub-LOD levels must agree on color layout (RGB vs RGBA), color dtype, and
    dimensionality — the viewer strides its progressive level-concat by each of
    these, so a mixed ladder is malformed data there. A single-level ladder
    trivially passes. Runs in the pre-flight gate (before any group is created)
    so a mixed ladder never leaves a partial node.
    """
    color_layouts = {s.colors.shape[1] for s in sublods if s.colors is not None}
    if len(color_layouts) > 1:
        raise ValueError(
            "additive ladder has mixed color layouts across sub-LODs "
            f"(channel counts {sorted(color_layouts)}); all levels must share "
            "RGB vs RGBA. Merge sources via GSplatData.concatenate (which "
            "normalizes the ladder) before writing."
        )
    color_dtypes = {str(s.colors.dtype) for s in sublods if s.colors is not None}
    if len(color_dtypes) > 1:
        raise ValueError(
            "additive ladder has mixed color dtypes across sub-LODs "
            f"({sorted(color_dtypes)}); all levels must share one dtype. "
            "Merge sources via GSplatData.concatenate (which normalizes the "
            "ladder) before writing."
        )
    sub_ndims = {s.centers.shape[1] for s in sublods}
    if len(sub_ndims) > 1:
        raise ValueError(
            "additive ladder has mixed dimensionality across sub-LODs "
            f"(ndims {sorted(sub_ndims)}); all levels must share the dataset "
            "dimensionality."
        )
    label_presence = {sub.label_ids is not None for sub in sublods}
    if len(label_presence) > 1:
        raise ValueError(
            "additive ladder must carry label_ids on every sub-LOD or none"
        )
    vocabularies = [
        sub.label_vocabulary for sub in sublods if sub.label_ids is not None
    ]
    if any(vocabulary != vocabularies[0] for vocabulary in vocabularies[1:]):
        raise ValueError("additive ladder label_vocabulary values must be identical")


def preflight_validate_leaf(leaf: "GSplatLeaf") -> None:
    """Validate the WHOLE leaf before any group is created.

    Runs every additive sub-LOD's arrays AND colors (both covered by
    ``validate_gsplat_inputs``), then the cross-level consistency checks
    (mixed color layout / dtype / ndim), so an invalid *later* level cannot
    be discovered only after earlier levels — and the parent node group — are
    already on disk (a half-written node). This is the ONE value scan per
    leaf: the per-level writes downstream re-run only the cheap shape
    normalization (``check_values=False``). It is NOT fully transactional,
    though: ``transform``/``nd_transform`` and custom-colormap-LUT resolution
    still run post-write (the F7 residual), so a bad ``transform=`` can still
    leave a partial node. Side-effect-free (validators only inspect /
    normalize copies).
    """
    sublods = leaf.additive_sublods
    # Per-sub-LOD arrays + colors FIRST: these give a descriptive ValidationError
    # on a malformed single level (e.g. 1-D colors/centers). The cross-level
    # consistency check runs AFTER, so it only ever compares well-formed levels
    # and never turns a bad shape into a bare IndexError on ``.shape[1]``.
    from luxar.gsplats.gsplat_data import validate_label_channel

    for sub in sublods:
        validate_gsplat_inputs(
            sub.centers, sub.amplitudes, sub.cholesky_factors, sub.colors
        )
        validate_label_channel(
            sub.label_ids, sub.label_vocabulary, sub.centers.shape[0]
        )
    _validate_ladder_color_and_dim_consistency(sublods)


def write_gsplat_leaf(
    group: zarr.Group,
    leaf: "GSplatLeaf",
    *,
    dataset_ctx: DatasetCtx,
    ordering_ctx: OrderingCtx,
    store: zarr.Group,
    attrs: Optional[Dict[str, Any]] = None,
    scene_tone_mapping: Optional[str] = None,
    barrier_dims: Optional[Sequence[int]] = None,
    preflighted: bool = False,
    inherited_colormap: Optional[str] = None,
    warn_on_missing_tone_mapping: bool = True,
) -> Dict[str, Any]:
    """Write a :class:`GSplatLeaf` (single set or additive ladder) into ``group``.

    ``preflighted=True`` promises the caller already ran
    :func:`preflight_validate_leaf` on this exact leaf (the scene compiler does,
    BEFORE creating ``group``) so the leaf is not value-scanned twice.

    ``inherited_colormap`` is an in-flight ancestor's palette; see
    :func:`~luxar.io._compiler.gsplat_assembly.inherited_gsplat_colormap`.
    """
    # Preflight: validate every sub-LOD before writing any additive_<i> group.
    if not preflighted:
        preflight_validate_leaf(leaf)
    sublods = leaf.additive_sublods
    if len(sublods) == 1:
        return _write_single_splat_set(
            group,
            sublods[0],
            dataset_ctx=dataset_ctx,
            ordering_ctx=ordering_ctx,
            lightweight=False,
            store=store,
            attrs=attrs,
            scene_tone_mapping=scene_tone_mapping,
            barrier_dims=barrier_dims,
            inherited_colormap=inherited_colormap,
            warn_on_missing_tone_mapping=warn_on_missing_tone_mapping,
        )

    # Additive ladder → additive_<i>/ subgroups + aggregate parent attrs.
    # (Per-level inputs and cross-level consistency were checked up front by
    # preflight_validate_leaf, so every additive_<i> write below is safe.)
    n_dims: Optional[int] = None
    total = 0
    has_any_colors = False
    has_label_ids = False
    label_vocabulary: Optional[Dict[str, str]] = None
    center_mins: List[List[float]] = []
    center_maxs: List[List[float]] = []
    amp_mins: List[float] = []
    amp_maxs: List[float] = []
    disp_los: List[float] = []
    disp_his: List[float] = []
    total_mass = 0.0
    total_self_energy = 0.0
    for i, sub in enumerate(sublods):
        sub_group = group.require_group(f"additive_{i}")
        meta = _write_single_splat_set(
            sub_group,
            sub,
            dataset_ctx=dataset_ctx,
            ordering_ctx=ordering_ctx,
            lightweight=True,
            store=store,
            attrs=attrs,
            barrier_dims=barrier_dims,
        )
        n_dims = meta["ndim"] if n_dims is None else n_dims
        total += meta["n_splats"]
        has_any_colors = has_any_colors or meta["has_colors"]
        has_label_ids = has_label_ids or meta["has_label_ids"]
        if meta["has_label_ids"]:
            label_vocabulary = meta["label_vocabulary"]
        center_mins.append(meta["center_bounds"]["min"])
        center_maxs.append(meta["center_bounds"]["max"])
        amp_mins.append(meta["amplitude_range"]["min"])
        amp_maxs.append(meta["amplitude_range"]["max"])
        adr = meta.get("amplitude_data_range")
        if adr is not None:
            disp_los.append(adr[0])
            disp_his.append(adr[1])
        # Sub-LODs are DISJOINT increments the viewer concatenates, so the
        # ladder's totals are the sums of theirs (the same reasoning as
        # ``total += meta["n_splats"]`` above). The mass-weighted mean amplitude
        # is therefore a mass-weighted average of the per-level ones, carried
        # here as the summed self-energy ``Σ mass·mwma``.
        sub_mass = float(meta.get("amplitude_mass", 0.0))
        total_mass += sub_mass
        total_self_energy += sub_mass * float(
            meta.get("amplitude_mass_weighted_mean", 0.0)
        )

    assert n_dims is not None
    agg_min = [min(m[d] for m in center_mins) for d in range(n_dims)]
    agg_max = [max(m[d] for m in center_maxs) for d in range(n_dims)]
    agg_meta: Dict[str, Any] = {
        "n_splats": total,
        "ndim": n_dims,
        "has_colors": has_any_colors,
        "has_label_ids": has_label_ids,
        "amplitude_range": {"min": min(amp_mins), "max": max(amp_maxs)},
        "center_bounds": {"min": agg_min, "max": agg_max},
        "ordering": "none",
        "n_additive_sublods": len(sublods),
    }
    if label_vocabulary is not None:
        agg_meta["label_vocabulary"] = label_vocabulary
    if disp_his:
        # Aggregate robust display window across the ladder's sub-LODs, so the
        # colormap-bearing ladder node carries a range (not the [0,1] fallback).
        agg_meta["amplitude_data_range"] = [min(disp_los), max(disp_his)]
    # Stamped UNCONDITIONALLY, ``0.0`` / ``0.0`` for a zero-mass ladder, exactly
    # as a leaf stamps them: the finalize harmonization distinguishes "present
    # and zero" (a mass-less node, scaled locally) from "absent" (a legacy store
    # with no ratio to scale by, which drops its whole structure to scale 1.0).
    # Omitting them here made a zero-mass ladder read back as statistic-LESS and
    # poison its enclosing combine.
    agg_meta["amplitude_mass"] = total_mass
    agg_meta["amplitude_mass_weighted_mean"] = (
        total_self_energy / total_mass if total_mass > 0.0 else 0.0
    )
    parent_attrs = dict(attrs or {})
    apply_gsplat_group_attrs(
        group,
        agg_meta,
        parent_attrs,
        store,
        scene_tone_mapping=scene_tone_mapping,
        lut_tone_mapping_warned=False,
        inherited_colormap=inherited_colormap,
        warn_on_missing_tone_mapping=warn_on_missing_tone_mapping,
    )
    group.attrs["n_additive_sublods"] = len(sublods)
    return agg_meta


def _union_bounds(
    bounds: List[Dict[str, List[float]]],
) -> Optional[Dict[str, List[float]]]:
    """Union a list of ``{"min": [...], "max": [...]}`` position bounds."""
    real = [b for b in bounds if b and b.get("min") and b.get("max")]
    if not real:
        return None
    d = len(real[0]["min"])
    return {
        "min": [min(b["min"][i] for b in real) for i in range(d)],
        "max": [max(b["max"][i] for b in real) for i in range(d)],
    }


def json_safe_value(value: Any) -> tuple[bool, Any]:
    """``(ok, converted)`` — recursively coerce ``value`` to a JSON-attr-safe
    form (numpy scalars → Python scalars; tuples → lists; nested dicts/lists
    filtered element-wise). ``ok`` is False for values with no *strictly*-JSON
    form. Shared by the node-meta projection here and the root ``pipeline/``
    stats bucket in :mod:`luxar.gsplats.io.save_gsplats`.

    Two subtleties this guards:

    * **numpy floats are checked BEFORE the Python-scalar branch.** ``np.float64``
      is a subclass of ``float``, so a ``(bool, int, float, str)`` check would
      accept it *un-coerced* and leak a numpy scalar into ``.zattrs``. numpy /
      bool checks come first so every numpy scalar is coerced to a Python one.
    * **non-finite floats are rejected** (``ok=False``). ``NaN`` / ``±Inf`` are
      not valid JSON; zarr writes them as bare ``NaN`` / ``Infinity`` tokens
      that a strict parser — notably the TypeScript viewer's ``JSON.parse`` —
      refuses, so a non-finite stat must be dropped, not persisted.
    """
    # numpy scalars first (np.float64 is a subclass of float; np.bool_ of int).
    if isinstance(value, np.integer):
        return True, int(value)
    if isinstance(value, np.floating):
        fv = float(value)
        return (True, fv) if math.isfinite(fv) else (False, None)
    if isinstance(value, np.bool_):
        return True, bool(value)
    if value is None or isinstance(value, (bool, int, str)):
        return True, value
    if isinstance(value, float):
        return (True, value) if math.isfinite(value) else (False, None)
    if isinstance(value, (list, tuple)):
        out_list = []
        for item in value:
            ok, conv = json_safe_value(item)
            if not ok:
                return False, None
            out_list.append(conv)
        return True, out_list
    if isinstance(value, dict):
        out_dict = {}
        for k, v in value.items():
            ok, conv = json_safe_value(v)
            if ok:
                out_dict[str(k)] = conv
        return True, out_dict
    return False, None


def _safe_lod_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    """A leaf's ``lod_stats`` attr, filtered exactly like ``level_stats``.

    ``lod_stats`` carries a leaf's whole fit ``stats`` dict, quality metrics
    included, and those go non-finite on ordinary inputs — a constant or
    signal-free volume has no foreground, so ``foreground_psnr_db`` is ``nan``,
    and an exact reconstruction gives ``psnr_db == +inf``. Written raw, zarr
    emits them as bare ``NaN`` / ``Infinity`` tokens, which are not JSON, and
    because the root document carries the consolidated index for the whole tree
    ONE such value costs a strict reader the ENTIRE store rather than that one
    number. :func:`json_safe_value` drops them per key and keeps the rest.
    """
    ok, safe = json_safe_value(dict(stats))
    return safe if ok and isinstance(safe, dict) else {}


def _meta_to_node_attrs(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Project a node's ``meta`` to the exact attr set the reader recovers.

    This is the write-side inverse of :func:`_node_meta_from_attrs`: it emits
    only the keys that round-trip (``_NODE_META_ATTR_KEYS`` + a JSON-safe
    ``level_stats`` derived from ``stats``). Writing exactly this set — rather
    than dumping the whole ``meta`` dict — keeps the write/read contract
    symmetric, so a group's on-disk attrs never carry provenance keys that the
    reader would silently drop (no lossy round-trip).
    """
    if not meta:
        return {}
    out: Dict[str, Any] = {}
    for key in _NODE_META_ATTR_KEYS:
        if key in meta and meta[key] is not None:
            out[key] = meta[key]
    # Per-level (SubstitutiveLevel) stats ride as a JSON-safe ``level_stats``
    # attr. json_safe_value keeps nested dicts (e.g. the L2 refine_stats block)
    # — the old scalar-only filter silently dropped them (lossy round-trip).
    stats = meta.get("stats")
    if isinstance(stats, dict) and stats:
        ok, safe = json_safe_value(stats)
        if ok and safe:
            out["level_stats"] = safe
    return out


def _leaf_child_attrs(node: "GSplatNode") -> Dict[str, Any]:
    """Per-child node attrs (selector + provenance) lifted from a leaf's ``meta``."""
    from luxar.gsplats.tree import GSplatLeaf

    if not isinstance(node, GSplatLeaf):
        return {}
    return _meta_to_node_attrs(node.meta)


def _resolve_group_colormap(
    group: zarr.Group,
    attrs: Dict[str, Any],
    scene_tone_mapping: Optional[str],
    inherited_colormap: Optional[str],
    warn_on_missing_tone_mapping: bool,
) -> Optional[str]:
    """Resolve a WRAPPER's own ``colormap`` and return what its children inherit.

    A ``colormap`` on a ``kind=lod`` / ``kind=partition`` group (the
    ``root_attrs`` channel is the only way one gets there) is resolved to a
    sibling ``colormap_lut`` array exactly as a leaf's is inside
    :func:`~luxar.io._compiler.gsplat_assembly.apply_gsplat_group_attrs`: an
    ndarray LUT or a matplotlib/colorcet name would otherwise be written
    verbatim into the group's attrs — unserializable, or unresolvable by the
    viewer, which only knows the builtins. ``attrs`` is mutated in place (the
    name becomes ``"custom"``); the LUT array lands on ``group`` immediately,
    while the attrs themselves are still written after the children by the
    caller.

    The returned name rides DOWN the recursion so no descendant leaf
    manufactures a ``"gray"`` that would shadow it (#1600) — this writer emits
    a wrapper's attrs only AFTER its children, so there is nothing on disk for
    the store walk to find.
    """
    if attrs.get("colormap") is not None:
        write_colormap_lut_if_needed(
            group,
            attrs,
            scene_tone_mapping,
            False,
            warn_on_missing_tone_mapping=warn_on_missing_tone_mapping,
        )
    return attrs.get("colormap") or inherited_colormap


def write_gsplat_node(
    group: zarr.Group,
    node: "GSplatNode",
    *,
    dataset_ctx: DatasetCtx,
    ordering_ctx: OrderingCtx,
    store: zarr.Group,
    attrs: Optional[Dict[str, Any]] = None,
    scene_tone_mapping: Optional[str] = None,
    barrier_dims: Optional[Sequence[int]] = None,
    under_partition: bool = False,
    inherited_colormap: Optional[str] = None,
    warn_on_missing_tone_mapping: bool = True,
) -> Dict[str, Any]:
    """Recursively write any :class:`GSplatNode` into ``group``.

    Returns the node's metadata dict including a ``position_bounds`` entry
    (groups = union of children). ``attrs`` are extra node attrs (render
    defaults, selector thresholds, provenance) merged onto the node.
    ``scene_tone_mapping`` (the scene's ``viewer_config.tone_mapping``, or
    ``None`` for a standalone file) is threaded to leaf attr stamping.
    ``barrier_dims`` names categorical/barrier center columns (e.g. time) so
    chunk ordering groups by them first; it is the SAME for every leaf in the
    tree and passed straight through. ``None`` → per-leaf auto-detection.

    ``under_partition`` is set by the recursion once a ``kind=partition`` ancestor
    with **more than one part** has been crossed (a one-part partition is not a
    tiling — see the partition branch). It selects which anchor the FALLBACK
    ``coverage_fraction`` derivation uses for a lod group (see the lod branch
    below); callers always leave it at the default.

    ``inherited_colormap`` carries a WRAPPER-authored palette down the
    recursion (the ``root_attrs`` channel is the only way one gets here). It
    cannot be discovered from the store the way the scene compiler discovers
    it, because this writer emits a wrapper's own attrs only AFTER its
    children — see
    :func:`~luxar.io._compiler.gsplat_assembly.inherited_gsplat_colormap`.
    Callers always leave it at the default.
    """
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

    if isinstance(node, GSplatLeaf):
        merged = dict(attrs or {})
        merged.update(_leaf_child_attrs(node))
        return write_gsplat_leaf(
            group,
            node,
            dataset_ctx=dataset_ctx,
            ordering_ctx=ordering_ctx,
            store=store,
            attrs=merged,
            scene_tone_mapping=scene_tone_mapping,
            barrier_dims=barrier_dims,
            inherited_colormap=inherited_colormap,
            warn_on_missing_tone_mapping=warn_on_missing_tone_mapping,
        )

    # Group prologue (kind=lod / kind=partition) — resolve a wrapper-authored
    # palette and work out what the children inherit.
    attrs = dict(attrs or {})
    child_colormap = _resolve_group_colormap(
        group,
        attrs,
        scene_tone_mapping,
        inherited_colormap,
        warn_on_missing_tone_mapping,
    )

    if isinstance(node, GSplatLodGroup):
        from luxar.core.group.lod.group import (
            coverage_fractions,
            partitioned_coverage_fractions,
        )
        from luxar.gsplats.tree import total_splats

        # In-memory children are coarsest→finest, the SAME order as the on-disk
        # child_<i> layout (child_0 = coarsest) — written straight through.
        on_disk = list(node.children)
        n = len(on_disk)
        # Which ANCHOR the fallback derivation uses. A ladder bound to a spatial
        # partition keeps the pre-#1361 fills-screen anchor, because a tile's
        # projected diagonal is intrinsically a fraction of the whole object's —
        # see ``partitioned_coverage_fractions``. Two ways to be bound:
        #   * this group sits UNDER a partition (the ``adaptive`` per-part groups);
        #   * one of its own children IS a partition (the ``overview`` cap↔fine pair).
        # The second test is deliberately ``any(...)``, so it also marks a group
        # where only a NON-finest child is a partition. No producer builds that
        # shape (overview's partition is always the finest child), so it is
        # untested rather than intended; if one ever does, decide explicitly which
        # anchor it wants instead of inheriting this fallback.
        # Keeping the fallback topology-aware is what lets ``gsplat transform``
        # scrub-and-re-derive stay a no-op instead of silently downgrading such a
        # store to the whole-object anchor.
        partition_bound = under_partition or any(
            isinstance(c, GSplatPartition) for c in on_disk
        )
        derive_cov = (
            partitioned_coverage_fractions if partition_bound else coverage_fractions
        )
        # Derive a per-child selector threshold (coarsest→finest) so EVERY child —
        # leaf OR nested Group — is viewer-selectable. ``coverage_fraction``
        # derived as a screen-area fraction by occupancy halving (the group's
        # ``selector`` attr below names the units). An authored
        # coverage_fraction on the child still takes precedence: for a leaf child
        # it is merged over these
        # passed attrs by ``_leaf_child_attrs`` in the leaf writer; for a nested group
        # child it is reapplied from ``node.meta`` by that group's branch. So this
        # only sets the threshold for meta-less (e.g. hand-built) trees. Without it a
        # nested lod-of-Group child carried no threshold and the selector was stuck
        # always-finest.
        #
        # SELECTOR/THRESHOLD CONSISTENCY — the shared all-or-none gate (see
        # ``gsplats.tree.gate_authored_selector``), the same one the scene
        # graft runs, so direct-file and grafted rendering agree: unknown meta
        # selectors are refused, a selector is preserved only when EVERY child
        # carries an authored threshold (validated against that selector's
        # contract; selector-less authored ladders stay legacy "coverage"),
        # and partially-authored ladders are scrubbed and fully re-derived in
        # screen-area units.
        from luxar.gsplats.tree import gate_authored_selector

        on_disk, selector_out = gate_authored_selector(
            on_disk,
            node.meta.get("selector"),
            source="kind=lod group (standalone writer)",
        )
        derived_cov = derive_cov([total_splats(c) for c in on_disk])
        child_bounds: List[Dict[str, List[float]]] = []
        for i, child in enumerate(on_disk):
            child_group = group.require_group(f"child_{i}")
            cmeta = write_gsplat_node(
                child_group,
                child,
                dataset_ctx=dataset_ctx,
                ordering_ctx=ordering_ctx,
                store=store,
                scene_tone_mapping=scene_tone_mapping,
                # ``child_index`` records on-disk insertion order so the viewer
                # restores it (napari-style) instead of zarr's alphabetical
                # enumeration — matching the scene ``Node`` stamp. Needed here
                # because bare-root .gsplats.zarr children are written by this
                # writer, not via ``Node.__init__``; without it a >=10-child
                # ladder/partition would reorder (child_10 before child_2).
                attrs={
                    "coverage_fraction": float(derived_cov[i]),
                    "child_index": i,
                },
                barrier_dims=barrier_dims,
                # A nested ladder inside a partition-bound one is still inside the
                # same tile, so the binding propagates down.
                under_partition=partition_bound,
                inherited_colormap=child_colormap,
                warn_on_missing_tone_mapping=warn_on_missing_tone_mapping,
            )
            if "position_bounds" in cmeta:
                child_bounds.append(cmeta["position_bounds"])
        # Caller attrs (lowest precedence) → node.meta → structural (authoritative,
        # never clobbered by a stray meta key — fixes the meta-clobbers-structural
        # ordering risk).
        for k, v in (attrs or {}).items():
            group.attrs[k] = v
        for k, v in _meta_to_node_attrs(node.meta).items():
            group.attrs[k] = v
        group.attrs["type"] = "group"
        group.attrs["kind"] = "lod"
        # The selector names the UNITS of the children's coverage_fraction
        # thresholds, so it must travel with them. A tree read from an existing
        # store carries its on-disk mode in meta (preserved — with its FULLY
        # authored thresholds — by the consistency gate above); everything
        # else — fresh recipe/backfill trees, the meta-less hand-built case,
        # and partially-authored trees whose ladder the gate just re-derived —
        # is in screen-area units (every live derivation is).
        group.attrs["selector"] = selector_out
        # The viewer's INITIAL level (before the coverage selector runs) — the
        # COARSEST child (child_0). This is purely a progressive-load hint: it
        # makes the scene appear instantly at low detail, then refine. It is
        # deliberately NOT the data-model default (GSplatData.default_substitutive,
        # which is the finest level the .centers accessor returns); defaulting the
        # viewer to the finest would eager-load every lod group at full resolution
        # (e.g. 100 embryos × 256K splats) and render "backwards".
        group.attrs["default_level"] = 0
        group.attrs["display_type"] = "gsplats"
        bounds = _union_bounds(child_bounds)
        meta: Dict[str, Any] = {"n_children": n}
        if bounds is not None:
            group.attrs["position_bounds"] = bounds
            meta["position_bounds"] = bounds
        return meta

    if isinstance(node, GSplatPartition):
        child_bounds = []
        # A one-part partition is not a tiling: its single part covers the whole
        # object, so a ladder underneath it must keep the whole-object anchor
        # rather than the fills-screen one (same rule as ``build_adaptive``, which
        # produces exactly this shape whenever the dataset fits ``max_elements``).
        # Getting it wrong here would make ``gsplat transform``'s scrub-and-
        # re-derive silently re-coarsen such a store. The exclusion only ever ADDS
        # a binding, never drops an outer one: a one-part partition nested inside a
        # real tiling is still inside that one tile, so OR the incoming binding in
        # rather than overwriting it.
        child_under_partition = under_partition or len(node.children) > 1
        for i, child in enumerate(node.children):
            child_group = group.require_group(f"part_{i}")
            cmeta = write_gsplat_node(
                child_group,
                child,
                dataset_ctx=dataset_ctx,
                ordering_ctx=ordering_ctx,
                store=store,
                scene_tone_mapping=scene_tone_mapping,
                # Insertion order for the viewer's sibling sort (see child_<i>
                # above). Order is visually irrelevant for partition parts (all
                # render), but keeping it consistent prevents a >=10-part graft
                # from enumerating part_10 before part_2.
                attrs={"child_index": i},
                barrier_dims=barrier_dims,
                under_partition=child_under_partition,
                inherited_colormap=child_colormap,
                warn_on_missing_tone_mapping=warn_on_missing_tone_mapping,
            )
            if "position_bounds" in cmeta:
                child_bounds.append(cmeta["position_bounds"])
        # Caller attrs → node.meta → structural last (authoritative).
        for k, v in (attrs or {}).items():
            group.attrs[k] = v
        for k, v in _meta_to_node_attrs(node.meta).items():
            group.attrs[k] = v
        group.attrs["type"] = "group"
        group.attrs["kind"] = "partition"
        group.attrs["display_type"] = "gsplats"
        group.attrs["max_elements"] = int(node.max_elements)
        # Split-plane record for exact viewer back-to-front ordering (present
        # only when the parts came from a single BSP split; a streamed grid
        # merge has none and the viewer falls back to centroids).
        if node.bsp_tree is not None:
            group.attrs["bsp_tree"] = node.bsp_tree
        bounds = _union_bounds(child_bounds)
        meta = {"n_children": len(node.children)}
        if bounds is not None:
            group.attrs["position_bounds"] = bounds
            meta["position_bounds"] = bounds
        return meta

    raise TypeError(f"Unknown gsplat node type: {type(node).__name__}")


# ────────────────────────────────────────────────────────────────────────
# Reader (mirror)
# ────────────────────────────────────────────────────────────────────────


def _decode_cholesky(group: zarr.Group, root: zarr.Group, decoder: Any) -> Any:
    """Decode Cholesky factors, recombining the v3.1 split layout.

    Delegates to the shared
    :func:`luxar.gsplats.utils.trils.recombine_cholesky` so the v3.0 fallback,
    the corruption invariant, and the error message stay in one place (the scene
    reader uses the same helper).
    """
    from luxar.gsplats.utils.trils import recombine_cholesky

    def decode(name: str) -> Any:
        return decoder.decode(group[name], root) if name in group else None

    return recombine_cholesky(decode)


def _read_leaf_arrays(group: zarr.Group, root: zarr.Group, decoder: Any) -> Any:
    """Decode one splat set's arrays from ``group`` into an ``AdditiveSubLOD``."""
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, validate_label_channel

    centers = decoder.decode(group["centers"], root)
    amplitudes = decoder.decode(group["amplitudes"], root)
    cholesky = _decode_cholesky(group, root, decoder)
    colors = decoder.decode(group["colors"], root) if "colors" in group else None
    label_ids = (
        decoder.decode(group["label_ids"], root) if "label_ids" in group else None
    )
    vocabulary_raw = group.attrs.get("label_vocabulary")
    label_vocabulary = (
        {int(label_id): name for label_id, name in dict(vocabulary_raw).items()}
        if vocabulary_raw is not None
        else None
    )
    label_vocabulary = validate_label_channel(
        label_ids, label_vocabulary, centers.shape[0]
    )
    stats_raw = group.attrs.get("lod_stats", {})
    stats = dict(stats_raw) if isinstance(stats_raw, dict) else {}
    truncation_radius = float(
        group.attrs.get("truncation_radius", DEFAULT_TRUNCATION_RADIUS)
    )
    return AdditiveSubLOD(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky,
        colors=colors,
        label_ids=label_ids,
        label_vocabulary=label_vocabulary,
        stats=stats,
        truncation_radius=truncation_radius,
    )


def _node_meta_from_attrs(group: zarr.Group) -> Dict[str, Any]:
    """Recover the per-node ``meta`` dict (provenance/selector) from ``.zattrs``."""
    from luxar.typing_utils.constants import LOD_SELECTORS

    out: Dict[str, Any] = {}
    for key in _NODE_META_ATTR_KEYS:
        if key in group.attrs:
            val = group.attrs[key]
            if key == "selector" and val not in LOD_SELECTORS:
                # A stale pre-v3.2 selector spelling (the pixel_size era) —
                # drop it so the writer re-stamps a valid mode alongside the
                # thresholds it re-derives.
                continue
            out[key] = None if (key == "parent_method" and val == "") else val
    if "level_stats" in group.attrs:
        ls = group.attrs["level_stats"]
        out["stats"] = dict(ls) if isinstance(ls, dict) else {}
    return out


def read_gsplat_node(
    group: zarr.Group, root: zarr.Group, decoder: Any = None
) -> "GSplatNode":
    """Reconstruct a :class:`GSplatNode` subtree from a written zarr ``group``."""
    from luxar.encoding import ArrayDecoder
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

    if decoder is None:
        decoder = ArrayDecoder()

    kind = group.attrs.get("kind")

    if kind == "lod":
        n = sum(1 for name in group if str(name).startswith("child_"))
        # On disk child_0 = coarsest; the in-memory tree is also coarsest-first,
        # so children are read straight through with no reversal. The on-disk
        # ``default_level`` is the viewer's coarsest-first progressive-load hint,
        # NOT the data-model default (which is the derived finest child) — so we
        # don't carry it onto the node.
        children = [
            read_gsplat_node(group[f"child_{i}"], root, decoder) for i in range(n)
        ]
        return GSplatLodGroup(
            children=children,
            meta=_node_meta_from_attrs(group),
        )

    if kind == "partition":
        n = sum(1 for name in group if str(name).startswith("part_"))
        children = [
            read_gsplat_node(group[f"part_{i}"], root, decoder) for i in range(n)
        ]
        bsp_tree = group.attrs.get("bsp_tree")
        return GSplatPartition(
            children=children,
            max_elements=int(group.attrs.get("max_elements", 0)),
            meta=_node_meta_from_attrs(group),
            # Restore the split-plane record (if written) so a disk→node→disk
            # round-trip and the scene graft preserve exact viewer ordering.
            bsp_tree=dict(bsp_tree) if bsp_tree is not None else None,
        )

    # Leaf — either a single set or an additive ladder.
    n_additive = int(group.attrs.get("n_additive_sublods", 1))
    if n_additive > 1:
        sublods = [
            _read_leaf_arrays(group[f"additive_{i}"], root, decoder)
            for i in range(n_additive)
        ]
    else:
        sublods = [_read_leaf_arrays(group, root, decoder)]
    return GSplatLeaf(additive_sublods=sublods, meta=_node_meta_from_attrs(group))
