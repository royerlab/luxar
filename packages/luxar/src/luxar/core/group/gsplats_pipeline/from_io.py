"""`add_gsplats_from_file` and `add_gsplats_from_volume` impls.

Both load (or fit) a GSplatData object and then call
:func:`add_gsplats_from_data_impl` to write it into the scene.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from .from_data import add_gsplats_from_data_impl

if TYPE_CHECKING:
    from ...gsplats import GSplats
    from ...node import Node
    from ..group import Group


def add_gsplats_from_file_impl(
    group: "Group",
    *,
    name: str,
    path: Union[str, Path],
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    **attrs: Any,
) -> Union["GSplats", "Group"]:
    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.tree import is_matrix_shaped

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GSplats file not found: {path}")

    # Read the v3.0 node tree once. A matrix-shaped tree (leaf / additive ladder
    # / kind=lod of leaves) maps to a GSplatData and embeds via the normal data
    # path (which applies dim_order / extend_to_all / fill). A genuinely nested
    # tree (kind=partition root, or lod with non-leaf children) has no flat
    # GSplatData equivalent, so it is GRAFTED node-for-node, reusing the scene's
    # own builders — the same subtree the file already holds.
    node, _stats = load_gsplat_node(path)

    if is_matrix_shaped(node):
        from luxar.gsplats.gsplat_data import GSplatData

        return add_gsplats_from_data_impl(
            group,
            name=name,
            result=GSplatData.from_tree(node),
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            **attrs,
        )

    if dim_order is not None or fill is not None or fill_sigma is not None:
        raise ValueError(
            "dim_order / fill / fill_sigma are not supported when grafting a "
            "partition / nested .gsplats.zarr (the file is already a full node "
            "subtree). Re-author the file in the target scene dims, or embed a "
            "matrix-shaped (leaf / additive / kind=lod) file instead."
        )
    return graft_gsplat_node(
        group, name=name, node=node, parent=parent, extend_to_all=extend_to_all, **attrs
    )


def graft_gsplat_node(
    group: "Group",
    *,
    name: str,
    node: Any,  # luxar.gsplats.tree.GSplatNode
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    **attrs: Any,
) -> Union["GSplats", "Group"]:
    """Graft a pre-built ``GSplatNode`` subtree into the scene, node-for-node.

    Composes the scene's own builders — ``add_gsplats_from_data`` (leaf /
    additive ladder), ``add_lod_group`` (kind=lod), ``add_partition_group``
    (kind=partition) — so a standalone ``.gsplats.zarr`` of any shape (including
    partition / nested) embeds as the identical subtree it holds on disk. The
    per-child ``min_pixel_size`` selector thresholds ride from each child's
    ``meta`` (so a nested lod combo stays selectable). Compositing attrs land on
    a wrapper Group; the rest fall through to children.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

    from ..compositing import COMPOSITING_ATTRS

    if isinstance(node, GSplatLeaf):
        # Matrix-shaped → the normal data path. A graft preserves the file's own
        # coordinates, so no scene-embed transforms are applied here.
        return add_gsplats_from_data_impl(
            group,
            name=name,
            result=GSplatData.from_tree(node),
            parent=parent,
            extend_to_all=extend_to_all,
            **attrs,
        )

    parent_node = parent or group
    wrapper_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
    child_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}
    child_attrs.pop("min_pixel_size", None)

    if isinstance(node, GSplatLodGroup):
        wrapper_attrs.setdefault("display_type", "gsplats")
        n = len(node.children)
        # In-memory children are finest-first; add_lod_group wants coarsest→finest.
        on_disk = list(reversed(node.children))
        on_disk_default = (n - 1) - node.default_level
        wrapper = parent_node.add_lod_group(
            name, default_level=int(on_disk_default), **wrapper_attrs
        )
        for i, child in enumerate(on_disk):
            mps = float((child.meta or {}).get("min_pixel_size", 0.0) or 0.0)
            graft_gsplat_node(
                wrapper,
                name=f"child_{i}",
                node=child,
                extend_to_all=extend_to_all,
                min_pixel_size=mps,
                **child_attrs,
            )
        return wrapper

    if isinstance(node, GSplatPartition):
        wrapper = parent_node.add_partition_group(
            name=name,
            display_type="gsplats",
            max_elements=int(node.max_elements),
            **wrapper_attrs,
        )
        for i, child in enumerate(node.children):
            graft_gsplat_node(
                wrapper,
                name=f"part_{i}",
                node=child,
                extend_to_all=extend_to_all,
                **child_attrs,
            )
        return wrapper

    raise TypeError(f"Cannot graft unknown gsplat node type: {type(node).__name__}")


def add_gsplats_from_volume_impl(
    group: "Group",
    *,
    name: str,
    volume: np.ndarray,
    seeds: Optional[Union[int, float]] = None,
    n_iters: int = 1000,
    device: Optional[str] = None,
    progressive: bool = False,
    max_splats_per_pass: int = 5000,
    psnr_patience: float = 0.5,
    max_passes: Optional[int] = None,
    parent: Optional["Node"] = None,
    extend_to_all: Optional[Union[List[str], str]] = None,
    dim_order: Optional[List[str]] = None,
    fill: Optional[Dict[str, float]] = None,
    fill_sigma: Optional[Dict[str, float]] = None,
    opacity: Optional[float] = None,
    blending_mode: Optional[str] = None,
    **fit_kwargs: Any,
) -> Union["GSplats", "Group"]:
    if progressive:
        from luxar.gsplats import fit_progressive_gaussian_splats

        max_splats = seeds if isinstance(seeds, int) else 50000
        result = fit_progressive_gaussian_splats(
            volume,
            max_splats=max_splats,
            max_splats_per_pass=max_splats_per_pass,
            iters_per_pass=n_iters,
            psnr_patience=psnr_patience,
            max_passes=max_passes,
            device=device,
            **fit_kwargs,
        )
    else:
        from luxar.gsplats import fit_gaussian_splats

        result = fit_gaussian_splats(
            volume,
            seeds=seeds,
            n_iters=n_iters,
            device=device,
            **fit_kwargs,
        )

    scene_attrs: Dict[str, Any] = {}
    if opacity is not None:
        scene_attrs["opacity"] = opacity
    if blending_mode is not None:
        scene_attrs["blending_mode"] = blending_mode

    return add_gsplats_from_data_impl(
        group,
        name=name,
        result=result,
        parent=parent,
        extend_to_all=extend_to_all,
        dim_order=dim_order,
        fill=fill,
        fill_sigma=fill_sigma,
        **scene_attrs,
    )
