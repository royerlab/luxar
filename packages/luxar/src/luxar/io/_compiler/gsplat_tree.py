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
_NODE_META_ATTR_KEYS = (
    "coverage_fraction",
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
) -> Dict[str, Any]:
    """Order + write one splat set's arrays into ``group``; return metadata.

    ``lightweight=True`` writes only the data attrs (used for ``additive_<i>``
    sub-LOD subgroups — no rendering defaults); ``False`` runs the full
    :func:`apply_gsplat_group_attrs` (rendering defaults + ``position_bounds``).

    ``scene_tone_mapping`` (the scene's ``viewer_config.tone_mapping``, or
    ``None`` for a standalone file) is threaded into
    :func:`apply_gsplat_group_attrs` so a scene leaf written through this path
    gets the same colormap-LUT tone handling as the compiler's own writer.
    """
    centers, amplitudes, cholesky, colors, n_splats, n_dims, chol_uniform = (
        validate_gsplat_inputs(
            sublod.centers, sublod.amplitudes, sublod.cholesky_factors, sublod.colors
        )
    )
    truncation_radius = float(
        (attrs or {}).get("truncation_radius", sublod.truncation_radius)
    )
    centers, amplitudes, cholesky, colors, ordering_data = (
        apply_gsplat_spatial_ordering(
            centers,
            amplitudes,
            cholesky,
            colors,
            n_splats,
            n_dims,
            chol_uniform,
            ordering_ctx,
            coverage_sigma=truncation_radius,
            barrier_dims=barrier_dims,
        )
    )
    metadata = write_gsplat_arrays(
        group,
        centers,
        amplitudes,
        cholesky,
        colors,
        n_splats,
        n_dims,
        chol_uniform,
        ordering_data,
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
            group.attrs["lod_stats"] = dict(sublod.stats)
        group.attrs["truncation_radius"] = truncation_radius
    else:
        leaf_attrs = dict(attrs or {})
        leaf_attrs.setdefault("truncation_radius", truncation_radius)
        # Per-additive-sub-LOD stats (e.g. cumulative_psnr_db) — written here on
        # the single-set fast path too, mirroring the lightweight ladder branch,
        # so the reader's unconditional lod_stats read round-trips faithfully.
        if sublod.stats and "lod_stats" not in leaf_attrs:
            leaf_attrs["lod_stats"] = dict(sublod.stats)
        apply_gsplat_group_attrs(
            group,
            metadata,
            leaf_attrs,
            store,
            scene_tone_mapping=scene_tone_mapping,
            lut_tone_mapping_warned=False,
        )
    return metadata


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
) -> Dict[str, Any]:
    """Write a :class:`GSplatLeaf` (single set or additive ladder) into ``group``."""
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
        )

    # Additive ladder → additive_<i>/ subgroups + aggregate parent attrs.
    n_dims: Optional[int] = None
    total = 0
    has_any_colors = False
    center_mins: List[List[float]] = []
    center_maxs: List[List[float]] = []
    amp_mins: List[float] = []
    amp_maxs: List[float] = []
    disp_los: List[float] = []
    disp_his: List[float] = []
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
        center_mins.append(meta["center_bounds"]["min"])
        center_maxs.append(meta["center_bounds"]["max"])
        amp_mins.append(meta["amplitude_range"]["min"])
        amp_maxs.append(meta["amplitude_range"]["max"])
        adr = meta.get("amplitude_data_range")
        if adr is not None:
            disp_los.append(adr[0])
            disp_his.append(adr[1])

    assert n_dims is not None
    agg_min = [min(m[d] for m in center_mins) for d in range(n_dims)]
    agg_max = [max(m[d] for m in center_maxs) for d in range(n_dims)]
    agg_meta: Dict[str, Any] = {
        "n_splats": total,
        "ndim": n_dims,
        "has_colors": has_any_colors,
        "amplitude_range": {"min": min(amp_mins), "max": max(amp_maxs)},
        "center_bounds": {"min": agg_min, "max": agg_max},
        "ordering": "none",
        "n_additive_sublods": len(sublods),
    }
    if disp_his:
        # Aggregate robust display window across the ladder's sub-LODs, so the
        # colormap-bearing ladder node carries a range (not the [0,1] fallback).
        agg_meta["amplitude_data_range"] = [min(disp_los), max(disp_his)]
    parent_attrs = dict(attrs or {})
    apply_gsplat_group_attrs(
        group,
        agg_meta,
        parent_attrs,
        store,
        scene_tone_mapping=scene_tone_mapping,
        lut_tone_mapping_warned=False,
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
        )

    if isinstance(node, GSplatLodGroup):
        from luxar.core.group.lod.group import coverage_fractions
        from luxar.gsplats.tree import total_splats

        # In-memory children are coarsest→finest, the SAME order as the on-disk
        # child_<i> layout (child_0 = coarsest) — written straight through.
        on_disk = list(node.children)
        n = len(on_disk)
        # Derive a per-child selector threshold (coarsest→finest) so EVERY child —
        # leaf OR nested Group — is viewer-selectable. Viewport-relative
        # ``coverage_fraction`` = ``sqrt(N_i/N_finest)`` (count ratios; the viewer
        # multiplies by the live viewport diagonal). An authored coverage_fraction on
        # the child still takes precedence: for a leaf child it is merged over these
        # passed attrs by ``_leaf_child_attrs`` in the leaf writer; for a nested group
        # child it is reapplied from ``node.meta`` by that group's branch. So this
        # only sets the threshold for meta-less (e.g. hand-built) trees. Without it a
        # nested lod-of-Group child carried no threshold and the selector was stuck
        # always-finest.
        derived_cov = coverage_fractions([total_splats(c) for c in on_disk])
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
        group.attrs["selector"] = "coverage"
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
    from luxar.gsplats.gsplat_data import AdditiveSubLOD

    centers = decoder.decode(group["centers"], root)
    amplitudes = decoder.decode(group["amplitudes"], root)
    cholesky = _decode_cholesky(group, root, decoder)
    colors = decoder.decode(group["colors"], root) if "colors" in group else None
    stats_raw = group.attrs.get("lod_stats", {})
    stats = dict(stats_raw) if isinstance(stats_raw, dict) else {}
    truncation_radius = float(group.attrs.get("truncation_radius", 3.0))
    return AdditiveSubLOD(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky,
        colors=colors,
        stats=stats,
        truncation_radius=truncation_radius,
    )


def _node_meta_from_attrs(group: zarr.Group) -> Dict[str, Any]:
    """Recover the per-node ``meta`` dict (provenance/selector) from ``.zattrs``."""
    out: Dict[str, Any] = {}
    for key in _NODE_META_ATTR_KEYS:
        if key in group.attrs:
            val = group.attrs[key]
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
