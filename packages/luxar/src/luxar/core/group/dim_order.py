"""Dim-order remapping for Group's leaf adders.

Two free functions used by ``add_points`` / ``add_lines`` / ``add_gsplats``
(and the multi-LOD wrappers) to support the ``dim_order=`` convenience
kwarg:

* :func:`apply_dim_order_positions` — reorder + pad positions/vertices/
  centers to the scene's full dimensionality.
* :func:`apply_dim_order_cholesky` — reorder + embed packed Cholesky
  factors so the GSplats writer can consume them.

Both take a ``Scene`` reference and produce pure NumPy arrays. No
``Group`` / ``self`` coupling — the prior ``self._apply_dim_order_*``
methods never used ``self`` for anything beyond dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
from arbol import aprint

if TYPE_CHECKING:
    from ..scene import Scene


def apply_dim_order_positions(
    positions: np.ndarray,
    scene: "Scene",
    dim_order: Optional[List[str]],
    fill: Optional[Dict[str, float]],
    extend_to_all: Optional[Union[List[str], str]],
) -> Tuple[np.ndarray, Optional[Union[List[str], str]]]:
    """Apply dim_order to position data if provided.

    Returns (transformed_positions, possibly_updated_extend_to_all).
    """
    if dim_order is None:
        return positions, extend_to_all

    transformed, unmapped = scene._apply_dim_order(positions, dim_order, fill)
    aprint(f"  🔀 dim_order: mapped {dim_order} → scene dimensions")

    # Auto-extend unmapped dims if user didn't explicitly set extend_to_all
    if extend_to_all is None and unmapped:
        extend_to_all = unmapped
        aprint(f"  📡 Unmapped dims (auto extend_to_all): {unmapped}")
    elif unmapped:
        aprint(f"  📡 Unmapped dims: {unmapped} (extend_to_all set explicitly)")

    return transformed, extend_to_all


def apply_dim_order_cholesky(
    cholesky_factors: np.ndarray,
    d_data: int,
    scene: "Scene",
    dim_order: List[str],
    fill_sigma: Optional[Dict[str, float]],
) -> np.ndarray:
    """Apply dim_order to Cholesky factors: permute and/or embed."""
    from ...gsplats.utils.trils import embed_cholesky_packed

    scene_names = scene._dimensions.names
    scene_ndim = scene._dimensions.ndim

    # Build dim_mapping: src_dim_i → dst_dim_index
    dim_mapping = [scene_names.index(name) for name in dim_order]

    # Validate fill_sigma keys
    fill_sigma_indexed: Optional[Dict[int, float]] = None
    if fill_sigma:
        dim_order_set = set(dim_order)
        for name in fill_sigma:
            if name not in scene_names:
                raise ValueError(
                    f"fill_sigma key '{name}' not found in scene "
                    f"dimensions {scene_names}"
                )
            if name in dim_order_set:
                raise ValueError(
                    f"fill_sigma key '{name}' is already in dim_order — "
                    f"fill_sigma is only for unmapped dimensions"
                )
        fill_sigma_indexed = {
            scene_names.index(name): sigma for name, sigma in fill_sigma.items()
        }

    if cholesky_factors.ndim == 1:
        # Uniform Cholesky: reshape to (1, k), transform, reshape back
        packed = cholesky_factors.reshape(1, -1)
        result = embed_cholesky_packed(
            packed, d_data, scene_ndim, dim_mapping, fill_sigma_indexed
        )
        return result.reshape(-1)
    else:
        return embed_cholesky_packed(
            cholesky_factors, d_data, scene_ndim, dim_mapping, fill_sigma_indexed
        )
