"""Scene-side dim_order remapping.

Free function called by ``Group``'s leaf adders' dim_order helpers
through Scene's own dim_order method stub.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .scene import Scene


def validate_dim_order_spec(
    scene: "Scene",
    dim_order: List[str],
    data_ndim: int,
    fill: Optional[Dict[str, float]] = None,
) -> None:
    """Validate a ``dim_order`` (+ ``fill``) against the scene, without the data.

    The whole validation preamble of :func:`apply_dim_order`, in its original
    order — length vs data columns, duplicate names, names present in the scene,
    then ``fill`` keys — extracted whole so it can also run BEFORE a split path
    creates its wrapper group. ``apply_dim_order`` still calls it first, so every
    message is unchanged; nothing here touches the array beyond its column count.

    Why a split path needs it: under a ``dim_order`` the post-transform width is
    ``scene_ndim`` by construction (``apply_dim_order`` allocates it that way), so
    the scene-dimension count check can never fire downstream. What DOES fire
    downstream — from inside ``child_0``, after the wrapper is on disk — is one of
    these five refusals. Checking them up front is what makes the
    ``lod_group=`` + ``dim_order=`` combination refuse against the caller's own
    node with nothing written (#1446).
    """
    if len(dim_order) != data_ndim:
        raise ValueError(
            f"dim_order has {len(dim_order)} names but data has "
            f"{data_ndim} columns. They must match."
        )

    scene_names = scene._dimensions.names

    # Validate names exist in scene dimensions and are unique
    if len(set(dim_order)) != len(dim_order):
        raise ValueError(f"dim_order has duplicate names: {dim_order}")
    for name in dim_order:
        if name not in scene_names:
            raise ValueError(
                f"dim_order name '{name}' not found in scene dimensions {scene_names}"
            )

    # Validate fill keys are valid dim names and not in dim_order
    for name in fill or {}:
        if name not in scene_names:
            raise ValueError(
                f"fill key '{name}' not found in scene dimensions {scene_names}"
            )
        if name in dim_order:
            raise ValueError(
                f"fill key '{name}' is already in dim_order — cannot "
                f"both map a data column and fill a fixed value"
            )


def apply_dim_order(
    scene: "Scene",
    positions: np.ndarray,
    dim_order: List[str],
    fill: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Reorder and pad position data to match scene dimensions.

    Maps data columns to scene dimensions by name, reordering and
    padding as needed. Returns the transformed array and a list of
    unmapped dimension names (candidates for extend_to_all).
    """
    if fill is None:
        fill = {}

    scene_names = scene._dimensions.names
    scene_ndim = scene._dimensions.ndim

    # Everything that can be judged from the spec alone (see the validator: the
    # split paths run the same function before they create a wrapper group).
    validate_dim_order_spec(scene, dim_order, positions.shape[1], fill)

    # Build the mapping: for each scene dim, which data column (or fill)
    N = positions.shape[0]
    result = np.zeros((N, scene_ndim), dtype=np.float32)
    unmapped: List[str] = []

    dim_order_set = set(dim_order)
    for scene_idx, scene_name in enumerate(scene_names):
        if scene_name in dim_order_set:
            # Find which data column maps to this scene dim
            data_col = dim_order.index(scene_name)
            result[:, scene_idx] = positions[:, data_col]
        else:
            # Unmapped — fill with fixed value
            result[:, scene_idx] = fill.get(scene_name, 0.0)
            unmapped.append(scene_name)

    return result, unmapped
