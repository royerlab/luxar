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
    from luxar.gsplats.io.load_gsplats import load_gsplats

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GSplats file not found: {path}")

    result = load_gsplats(path)

    return add_gsplats_from_data_impl(
        group,
        name=name,
        result=result,
        parent=parent,
        extend_to_all=extend_to_all,
        dim_order=dim_order,
        fill=fill,
        fill_sigma=fill_sigma,
        **attrs,
    )


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
