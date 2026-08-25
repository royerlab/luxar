"""Dim-order remapping for Group's leaf adders.

Two free functions used by ``add_points`` / ``add_lines`` / ``add_gsplats``
(and the multi-LOD wrappers) to support the ``dim_order=`` convenience
kwarg:

* :func:`apply_dim_order_positions` — reorder + pad positions/vertices/
  centers to the scene's full dimensionality.
* :func:`apply_dim_order_cholesky` — reorder + embed packed Cholesky
  factors so the GSplats writer can consume them.
* :func:`apply_dim_order_orientation` — carry a Mesh's normals and face
  WINDING through the same map, the Mesh peer of the Cholesky one.

The last two exist because reordering coordinate columns silently invalidates
anything that describes a DIRECTION in them. A type with no such array (Points,
Lines) needs only the first function; GSplats and Mesh each need a second.

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


def apply_dim_order_orientation(
    normals: Optional[np.ndarray],
    normal_dims: Optional[Sequence[int]],
    faces: np.ndarray,
    scene: "Scene",
    dim_order: Optional[List[str]],
) -> Tuple[Optional[np.ndarray], Optional[Sequence[int]], np.ndarray]:
    """Carry a mesh's ORIENTATION data through a ``dim_order`` change of basis.

    The Mesh peer of :func:`apply_dim_order_cholesky`, and it exists for the same
    reason: :func:`apply_dim_order_positions` renumbers the *coordinate columns*,
    and any array that describes a DIRECTION in those columns has to be carried
    through the same map or it silently starts describing different axes.
    GSplats has one such array (the Cholesky factors). Mesh has two, and the
    second one is not obviously an array at all:

    * ``normals`` — three components positionally bound to ``normal_dims``;
    * ``faces`` — an index buffer whose *corner order* encodes surface
      orientation (spec §3.2), which a handedness-reversing permutation inverts.

    Both are no-ops without ``dim_order``. Both are skipped when ``normals`` is
    absent, because ``sorted(normal_dims)`` is the ONLY declared winding frame
    (§3.2: "there is no other signal for which three axes the author wound
    against"), so without it there is nothing to be correct against — and the
    viewer already renders such a mesh ``DoubleSide`` regardless of
    ``double_sided`` (``data/mesh/projection.ts``, the "no stored normals" arm).

    **The normals transform is a relabel plus a sort, not a rotation.** Component
    ``k`` describes authored axis ``normal_dims[k]``, so remapping the LABEL
    through ``dim_mapping`` already preserves the pairing — the component still
    describes the same physical axis, now under its scene number. The components
    are then permuted so the triple comes out ASCENDING, which is load-bearing
    rather than tidy: the viewer uses stored normals only when ``normal_dims``
    equals ``displayDims`` **in order** (``storedNormalsUsable``), and
    ``displayDims`` is always built ascending. Leaving a non-ascending triple
    would be safe but would silently drop the mesh onto the flat-normal fallback.
    Normals are NOT widened the way positions and Cholesky factors are: they stay
    ``(V, 3)`` forever, with the frame carried in the attr.

    **The winding flip is decided on the frame, not on the whole map.** Only the
    three frame axes participate in the handedness of the surface, so the parity
    that matters is that of ``[dim_mapping[d] for d in sorted(normal_dims)]``.
    Swapping exactly two of a triangle's three indices reverses orientation;
    swapping all three is a rotation and changes nothing.

    Args:
        normals: Optional ``(V, 3)`` per-vertex normals, in AUTHORED component order.
        normal_dims: The three AUTHORED column indices those components describe.
        faces: ``(F, 3)`` or flat ``(3F,)`` triangle indices. Returned in the same shape.
        scene: The scene whose dimension order is being mapped onto.
        dim_order: Scene dimension names, one per authored column. ``None`` is a no-op.

    Returns:
        ``(normals, normal_dims, faces)``, transformed when applicable.

    Raises:
        ValidationError: If a ``normal_dims`` entry does not index an authored
            column. Without ``dim_order`` that entry is checked against the scene
            width and would pass; here it is provably meaningless, and admitting
            it would bind a normal component to an axis the caller never supplied.
    """
    if dim_order is None or normals is None or normal_dims is None:
        return normals, normal_dims, faces

    from ...validation import ValidationError

    scene_names = scene._dimensions.names
    # src authored column -> dst scene index. Same map, same construction, as
    # apply_dim_order_cholesky builds for the gsplat factors.
    dim_mapping = [scene_names.index(name) for name in dim_order]

    for entry in normal_dims:
        if not 0 <= int(entry) < len(dim_mapping):
            raise ValidationError(
                f"normal_dims={list(normal_dims)} indexes column {int(entry)}, but "
                f"dim_order names only {len(dim_mapping)} authored columns "
                f"({list(dim_order)}). With dim_order, normal_dims must name the "
                "AUTHORED columns the normal components describe — the ones being "
                "remapped — not the scene dimensions they are being mapped onto.",
                "Give normal_dims the indices of the vertex columns as you authored "
                "them; they are remapped onto the scene's dimension order for you",
            )

    normals_arr = np.asarray(normals)

    # Pair each component with the SCENE index it now describes, then sort so the
    # stored triple is ascending (see the docstring: the viewer wants ordered
    # equality with displayDims, which is ascending).
    paired = sorted(
        (dim_mapping[int(d)], component) for component, d in enumerate(normal_dims)
    )
    new_normal_dims = [scene_index for scene_index, _ in paired]
    component_order = [component for _, component in paired]
    new_normals = normals_arr[:, component_order]

    # Handedness. `sorted(normal_dims)` is the authored winding frame (§3.2); its
    # image under dim_mapping is odd exactly when the change of basis reflects.
    frame_image = [dim_mapping[int(d)] for d in sorted(int(x) for x in normal_dims)]
    new_faces = faces
    if permutation_parity_is_odd(frame_image):
        faces_arr = np.asarray(faces)
        original_shape = faces_arr.shape
        # Swap two of the three — a three-way rotation would leave winding alone.
        new_faces = faces_arr.reshape(-1, 3)[:, [0, 2, 1]].reshape(original_shape)
        aprint(
            "  🔄 dim_order reverses handedness on the winding frame "
            f"{sorted(int(x) for x in normal_dims)} → {new_normal_dims}; "
            "face winding flipped to keep triangles front-facing"
        )

    if new_normal_dims != [int(x) for x in normal_dims]:
        aprint(
            f"  🧭 dim_order: normal_dims {list(normal_dims)} → {new_normal_dims} "
            "(components permuted to match)"
        )

    return new_normals, new_normal_dims, new_faces
