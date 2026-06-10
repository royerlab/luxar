"""Root-agnostic Gaussian-splat node-subtree serializer (v3.0).

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

On-disk grammar (the v3.0 node tree):

* **leaf** → ``type=gsplats``; a single splat set writes arrays directly, an
  additive ladder writes ``additive_<i>/`` subgroups + ``n_additive_sublods``.
* **lod group** → ``type=group, kind=lod``; children written **coarsest→finest**
  as ``child_<i>/`` (the in-memory tree is finest-first — this writer reverses),
  each carrying its ``min_pixel_size`` selector threshold + LOD provenance.
* **partition group** → ``type=group, kind=partition``; children as ``part_<i>/``.

Every node carries a ``position_bounds`` attr (groups = union of children) so the
viewer can frame a bare-node file on load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, cast

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
    "min_pixel_size",
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
        )

    # Additive ladder → additive_<i>/ subgroups + aggregate parent attrs.
    n_dims: Optional[int] = None
    total = 0
    has_any_colors = False
    center_mins: List[List[float]] = []
    center_maxs: List[List[float]] = []
    amp_mins: List[float] = []
    amp_maxs: List[float] = []
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
        )
        n_dims = meta["ndim"] if n_dims is None else n_dims
        total += meta["n_splats"]
        has_any_colors = has_any_colors or meta["has_colors"]
        center_mins.append(meta["center_bounds"]["min"])
        center_maxs.append(meta["center_bounds"]["max"])
        amp_mins.append(meta["amplitude_range"]["min"])
        amp_maxs.append(meta["amplitude_range"]["max"])

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


def _leaf_child_attrs(node: "GSplatNode") -> Dict[str, Any]:
    """Per-child node attrs (selector + provenance) lifted from a leaf's ``meta``."""
    from luxar.gsplats.tree import GSplatLeaf

    if not isinstance(node, GSplatLeaf):
        return {}
    out: Dict[str, Any] = {}
    for key in _NODE_META_ATTR_KEYS:
        if key in node.meta and node.meta[key] is not None:
            out[key] = node.meta[key]
    # Per-level (SubstitutiveLevel) stats ride as a JSON-safe ``level_stats`` attr.
    stats = node.meta.get("stats")
    if isinstance(stats, dict) and stats:
        safe = {
            k: v
            for k, v in stats.items()
            if isinstance(v, (int, float, str, bool, list))
        }
        if safe:
            out["level_stats"] = safe
    return out


def write_gsplat_node(
    group: zarr.Group,
    node: "GSplatNode",
    *,
    dataset_ctx: DatasetCtx,
    ordering_ctx: OrderingCtx,
    store: zarr.Group,
    attrs: Optional[Dict[str, Any]] = None,
    scene_tone_mapping: Optional[str] = None,
) -> Dict[str, Any]:
    """Recursively write any :class:`GSplatNode` into ``group``.

    Returns the node's metadata dict including a ``position_bounds`` entry
    (groups = union of children). ``attrs`` are extra node attrs (render
    defaults, selector thresholds, provenance) merged onto the node.
    ``scene_tone_mapping`` (the scene's ``viewer_config.tone_mapping``, or
    ``None`` for a standalone file) is threaded to leaf attr stamping.
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
        )

    if isinstance(node, GSplatLodGroup):
        from luxar.core.group.lod.group import derive_min_pixel_sizes
        from luxar.gsplats.tree import total_splats

        # In-memory children are finest→coarsest; on disk child_0 = coarsest.
        on_disk = list(reversed(node.children))
        n = len(on_disk)
        on_disk_default = (n - 1) - node.default_level
        # Derive a per-child selector threshold (coarsest→finest) from element
        # counts so EVERY child — leaf OR nested Group — is viewer-selectable.
        # A min_pixel_size already authored on the child (leaf meta / group meta)
        # takes precedence over this derived value (applied via setdefault below /
        # _leaf_child_attrs for leaves). Without this, a nested lod-of-Group child
        # carried no threshold and the selector was stuck always-finest.
        derived_mps = derive_min_pixel_sizes([total_splats(c) for c in on_disk])
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
                attrs={"min_pixel_size": float(derived_mps[i])},
            )
            if "position_bounds" in cmeta:
                child_bounds.append(cmeta["position_bounds"])
        # Caller attrs (lowest precedence) → node.meta → structural (authoritative,
        # never clobbered by a stray meta key — fixes the meta-clobbers-structural
        # ordering risk).
        for k, v in (attrs or {}).items():
            group.attrs[k] = v
        for k, v in (node.meta or {}).items():
            group.attrs[k] = v
        group.attrs["type"] = "group"
        group.attrs["kind"] = "lod"
        group.attrs["selector"] = "pixel_size"
        group.attrs["default_level"] = int(on_disk_default)
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
            )
            if "position_bounds" in cmeta:
                child_bounds.append(cmeta["position_bounds"])
        # Caller attrs → node.meta → structural last (authoritative).
        for k, v in (attrs or {}).items():
            group.attrs[k] = v
        for k, v in (node.meta or {}).items():
            group.attrs[k] = v
        group.attrs["type"] = "group"
        group.attrs["kind"] = "partition"
        group.attrs["display_type"] = "gsplats"
        group.attrs["max_elements"] = int(node.max_elements)
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


def _read_leaf_arrays(group: zarr.Group, root: zarr.Group, decoder: Any) -> Any:
    """Decode one splat set's arrays from ``group`` into an ``AdditiveSubLOD``."""
    from luxar.gsplats.gsplat_data import AdditiveSubLOD

    centers = decoder.decode(group["centers"], root)
    amplitudes = decoder.decode(group["amplitudes"], root)
    cholesky = decoder.decode(group["cholesky_factors"], root)
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
        on_disk = [
            read_gsplat_node(group[f"child_{i}"], root, decoder) for i in range(n)
        ]
        # On disk child_0 = coarsest; restore finest-first in memory.
        children = list(reversed(on_disk))
        on_disk_default = int(group.attrs.get("default_level", 0))
        default_level = (n - 1) - on_disk_default
        return GSplatLodGroup(
            children=children,
            default_level=default_level,
            meta=_node_meta_from_attrs(group),
        )

    if kind == "partition":
        n = sum(1 for name in group if str(name).startswith("part_"))
        children = [
            read_gsplat_node(group[f"part_{i}"], root, decoder) for i in range(n)
        ]
        return GSplatPartition(
            children=children,
            max_elements=int(group.attrs.get("max_elements", 0)),
            meta=_node_meta_from_attrs(group),
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
