"""Dim-order remapping for Group's leaf adders.

Two free functions used by ``add_points`` / ``add_lines`` / ``add_gsplats``
(and the multi-LOD wrappers) to support the ``dim_order=`` convenience
kwarg:

* :func:`apply_dim_order_positions` — reorder + pad positions/vertices/
  centers to the scene's full dimensionality.
* :func:`apply_dim_order_cholesky` — reorder + embed packed Cholesky
  factors so the GSplats writer can consume them.
* :func:`warn_if_dim_order_reverses_winding` — report (never repair) a Mesh
  whose face winding disagrees with the scene frame after the remap.

The second exists because reordering coordinate columns silently invalidates
anything that describes a DIRECTION in them: gsplats' Cholesky factors have to be
carried through the map. Mesh needs no such transform — ``normal_dims`` names
SCENE dimensions, so its normals are already expressed in the destination frame —
but its face winding is a direction that Luxar cannot repair without knowing
which frame the caller wound in, so the third function reports instead.

They take a ``Scene`` reference and produce pure NumPy arrays. No
``Group`` / ``self`` coupling — the prior ``self._apply_dim_order_*``
methods never used ``self`` for anything beyond dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

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


def validate_fill_sigma_keys(
    scene: "Scene",
    dim_order: List[str],
    fill_sigma: Optional[Dict[str, float]],
) -> None:
    """Validate ``fill_sigma`` keys against the scene and ``dim_order``.

    The gsplats-only counterpart of ``validate_dim_order_spec``, and extracted
    for the same reason: both refusals are pure spec checks (they never look at
    the Cholesky array), so a split path can run them before it creates a wrapper
    group instead of discovering them from inside ``child_0`` (#1446).
    :func:`apply_dim_order_cholesky` still calls this first, so the messages and
    their order are unchanged.
    """
    if not fill_sigma:
        return
    scene_names = scene._dimensions.names
    dim_order_set = set(dim_order)
    for name in fill_sigma:
        if name not in scene_names:
            raise ValueError(
                f"fill_sigma key '{name}' not found in scene dimensions {scene_names}"
            )
        if name in dim_order_set:
            raise ValueError(
                f"fill_sigma key '{name}' is already in dim_order — "
                f"fill_sigma is only for unmapped dimensions"
            )


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

    validate_fill_sigma_keys(scene, dim_order, fill_sigma)

    fill_sigma_indexed: Optional[Dict[int, float]] = None
    if fill_sigma:
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


def permutation_parity_is_odd(order: Sequence[int]) -> bool:
    """Whether sorting ``order`` ascending takes an odd number of transpositions.

    Counts inversions rather than composing cycles: for three elements that is
    three comparisons and it stays obviously correct. The Python twin of
    ``permutationParityIsOdd`` in
    ``packages/luxar-viewer/src/data/mesh/projection.ts`` — the viewer asks the
    same question of ``displayDims`` that the writer asks of ``dim_order``, and
    both answers have to agree about what "front-facing" means.

    Args:
        order: A sequence of distinct comparable indices.

    Returns:
        ``True`` when the permutation that sorts ``order`` is odd.
    """
    inversions = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[i] > order[j]:
                inversions += 1
    return inversions % 2 == 1


def dim_order_reverses_winding(
    normal_dims: Optional[Sequence[int]],
    scene: "Scene",
    dim_order: Optional[List[str]],
) -> bool:
    """Whether ``dim_order`` reverses handedness on a mesh's winding frame.

    ``dim_order`` renumbers the vertex COLUMNS, and a column permutation with
    ``det = -1`` reflects space: since ``cross(Ra, Rb) = det(R)·R·cross(a, b)``, a
    triangle's geometric normal is negated relative to the pure permutation while
    its stored corner order is untouched. Faces that were counter-clockwise in the
    caller's own column order are clockwise afterwards.

    **This function only reports; it changes nothing.** That restraint is the whole
    point, and the reason is that Luxar cannot know which frame the caller wound
    in. ``normal_dims`` names SCENE dimension indices — the layout AFTER
    ``dim_order`` — so a caller who reads the contract literally winds against the
    scene frame too, and for them the stored winding is already right; silently
    flipping it would break them. A caller who instead wound against their own
    authored column order has a store that violates §3.2, and only they can say
    which they did. So the writer warns, names the consequence, and gives the
    one-line remedy.

    Undecidable cases return ``False`` (no warning) rather than guessing:

    * no ``normal_dims`` — ``sorted(normal_dims)`` is the only declared winding
      frame there is (§3.2), so without it nothing is decidable, and the viewer
      already renders such a mesh ``DoubleSide`` regardless of ``double_sided``;
    * a frame axis that no authored column maps onto (an unmapped scene dimension
      filled with a constant) — it has no preimage, so the restricted map is not a
      permutation at all.

    Args:
        normal_dims: The three SCENE dimension indices the normals describe.
        scene: The scene whose dimension order is being mapped onto.
        dim_order: Scene dimension names, one per authored column.

    Returns:
        ``True`` when the caller's authored column order and the scene frame
        disagree about handedness.
    """
    if dim_order is None or normal_dims is None or len(list(normal_dims)) != 3:
        return False

    scene_names = scene._dimensions.names
    # src authored column -> dst scene index, the same map
    # `apply_dim_order_cholesky` builds for the gsplat factors.
    dim_mapping = [scene_names.index(name) for name in dim_order]

    # Walk the frame in ascending SCENE order and record which authored column
    # each axis came from. If those columns are not themselves ascending, the
    # restricted map is an odd permutation and handedness flips.
    preimage: List[int] = []
    for scene_index in sorted(int(d) for d in normal_dims):
        if scene_index not in dim_mapping:
            return False  # filled dimension: no preimage, nothing to decide
        preimage.append(dim_mapping.index(scene_index))
    return permutation_parity_is_odd(preimage)


def warn_if_dim_order_reverses_winding(
    name: str,
    normal_dims: Optional[Sequence[int]],
    scene: "Scene",
    dim_order: Optional[List[str]],
) -> None:
    """Warn once when ``dim_order`` reverses handedness on the winding frame.

    Warn-only, in the manner of the mesh writer's unwelded-vertices lint (§3.6):
    it catches a likely authoring mistake without refusing a store that may be
    perfectly deliberate. See :func:`dim_order_reverses_winding` for why this
    cannot be fixed automatically.

    Double-sided drawing does not make the mismatch harmless: stored-normal
    shading uses ``gl_FrontFacing`` to choose the normal sign, so reversed winding
    flips the shading gradient even when rasterization coverage is unchanged.
    """
    if not dim_order_reverses_winding(normal_dims, scene, dim_order):
        return
    aprint(
        f"  ⚠️ Mesh '{name}': dim_order={list(dim_order or [])} reverses handedness "
        "on the winding frame. `normal_dims` names SCENE dimensions, so faces are "
        "expected counter-clockwise in the SCENE column order too. If you wound "
        "them in your own authored column order they are now clockwise. A "
        "single-sided open surface can vanish, and stored-normal shading flips "
        "even when double_sided=True. Pass faces[:, [0, 2, 1]], or verify that "
        "the faces were already wound in the scene frame."
    )
