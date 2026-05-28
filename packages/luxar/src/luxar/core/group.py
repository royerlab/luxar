"""luxar.group – Group node with data-adding methods for scene graphs.

A Group can contain child data nodes (Points, Lines, GSplats) and other Groups,
forming a hierarchical scene structure. Groups walk up the parent chain to find
the root Scene for dimension validation and writer access.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
from arbol import aprint

from ..core.gsplats import GSplats
from ..core.lines import Lines
from ..core.node import Node
from ..core.points import Points
from ..typing_utils.aliases import ColorArray, PositionArray

if TYPE_CHECKING:
    from ..core.scene import Scene
    from ..gsplats.gsplat_data import GSplatData
    from ..io.writer import ZarrWriterProtocol

# Default radius used when radii are not provided
DEFAULT_POINT_RADIUS = 0.5


#: Attrs that ride on a kind=lod / kind=split wrapper Group (where the user
#: thinks of the wrapper as "their layer") rather than getting copied onto
#: each internal child. Compositing semantics (opacity, gamma, ...) flow
#: down to the children through Group inheritance at render time, so writing
#: them once on the parent is correct. ``colormap`` and ``truncation_radius``
#: are deliberately NOT compositing: the writer auto-defaults them per leaf,
#: which under nearest-ancestor-wins would shadow a parent's setting.
_COMPOSITING_ATTRS = frozenset(
    {
        "transform",
        "opacity",
        "gamma",
        "intensity",
        "offset",
        "blending_mode",
        "layer",
        "visible",
        "nd_transform",
    }
)


def _slice_optional_array(
    value: Any, indices: np.ndarray, n_elements: int
) -> Any:
    """Slice an array-valued leaf parameter by index; pass non-per-element values through.

    Used by the ``split=`` wrapping path on the leaf adders. Returns
    unchanged when:
      * ``value`` is ``None`` or a scalar (``int`` / ``float`` / ``bool``
        / ``str``) — applies uniformly to every part.
      * ``value`` is a 0-D array.
      * ``value``'s first-axis length doesn't match ``n_elements`` (e.g.
        a 3-vector RGB broadcast, or a length-1 sentinel).
    Slices the first axis when the input is a list of length
    ``n_elements`` (string labels) or an array whose first axis matches.
    """
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, list):
        if len(value) == n_elements:
            return [value[i] for i in indices]
        return value
    arr = value if isinstance(value, np.ndarray) else np.asarray(value)
    if arr.ndim == 0:
        return value
    if arr.shape[0] == n_elements:
        return arr[indices]
    return value


def _position_bounds_from_array(positions: np.ndarray) -> Dict[str, List[float]]:
    """Per-axis min/max of an ``(N, D)`` position array, in the writer's shape.

    Matches what the compiler's ``_compute_position_bounds`` writes onto
    each leaf node, so the split-kind wrapper's ``position_bounds`` is
    the same shape as its children's. Used by the ``split=`` wrapping
    path to compute the parent bbox directly from the source array
    instead of round-tripping through the per-leaf zarr writes.
    """
    if positions.size == 0:
        raise ValueError("Cannot compute position_bounds from empty array")
    return {
        "min": positions.min(axis=0).astype(float).tolist(),
        "max": positions.max(axis=0).astype(float).tolist(),
    }


class Group(Node):
    """A group node that can contain data children (Points, Lines, GSplats).

    Groups provide add_points(), add_lines(), and add_gsplats() methods for
    adding data nodes. They access the root Scene for dimension validation
    and the writer interface.

    Groups are created via add_group() on any Node, Scene, or Group::

        scene = compiler.create_scene(dimensions=dims)
        group = scene.add_group("my_group")
        group.add_points("pts", positions)  # data written under my_group/

    Dimension mapping (``dim_order``) allows adding lower-dimensional data to
    higher-dimensional scenes::

        # 3D splats fitted from a volume → 4D scene with Time dimension
        scene.add_gsplats_from_data(
            "splats", result_3d,
            dim_order=["Z", "Y", "X"],      # maps data cols to scene dims
            fill={"Time": 0.0},               # fixed value for unmapped dim
        )

    Args:
        name: Name of the group
        parent: Parent node in the hierarchy
        writer: Writer interface for progressive writing
        **attrs: Additional attributes (opacity, blending_mode, etc.)
    """

    def _find_scene(self) -> Scene:
        """Walk up the parent chain to find the root Scene.

        Returns:
            The root Scene node

        Raises:
            ValueError: If this group is not attached to a Scene hierarchy
        """
        from ..core.scene import Scene

        node: Optional[Node] = self
        while node is not None:
            if isinstance(node, Scene):
                return node
            node = node.parent
        raise ValueError(
            f"Group '{self.name}' is not attached to a Scene. "
            "Groups must be part of a scene hierarchy to add data nodes."
        )

    # ---------------------------------------------------------- internal helpers

    def _require_scene_writer(self, scene: Scene) -> ZarrWriterProtocol:
        """Return the scene writer or fail with an explicit runtime error.

        Data-adding methods require scenes created by LuxarZarrCompiler. Do not
        rely on ``assert`` here: assertions can be stripped with ``python -O``.
        """
        writer = scene._writer
        if writer is None:
            raise RuntimeError(
                "Scene writer is not initialized. Use LuxarZarrCompiler to create scenes."
            )
        return writer

    def _apply_dim_order_positions(
        self,
        positions: np.ndarray,
        scene: Scene,
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

    def _apply_dim_order_cholesky(
        self,
        cholesky_factors: np.ndarray,
        d_data: int,
        scene: Scene,
        dim_order: List[str],
        fill_sigma: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """Apply dim_order to Cholesky factors: permute and/or embed."""
        from ..gsplats.utils.trils import embed_cholesky_packed

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

    # ---------------------------------------------------------- data methods

    def add_points(
        self,
        name: str,
        positions: Union[
            PositionArray, np.ndarray[Any, Any], Sequence[Sequence[float]]
        ],
        colors: Optional[
            Union[ColorArray, np.ndarray[Any, Any], Sequence[float | int]]
        ] = None,
        radii: Optional[
            Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any], float]
        ] = None,
        sharpness: Optional[
            Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any], float]
        ] = None,
        scalars: Optional[
            Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any], float]
        ] = None,
        labels: Optional[Union[List[str], Sequence[str]]] = None,
        image_labels: Optional[Any] = None,
        parent: Optional[Node] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        grid_shape: Optional[Tuple[int, ...]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        split: Any = None,
        additive_lod: Any = None,
        **attrs: Any,
    ) -> Union[Points, "Group"]:
        """Add a points node.

        Args:
            name: Name of the points node
            positions: Array of shape (N, D) for point positions
            colors: Optional (N, 3) array, RGB tuple, or None
            radii: Optional (N,) array, scalar, or None (default 0.5)
            sharpness: Optional (N,) array, scalar, or None
            scalars: Optional (N,) array or scalar for colormap lookup.
                Requires ``colormap`` in attrs. Mutually exclusive with ``colors``.
            labels: Optional list of strings, one per point. Used for hover tooltips.
                Length must equal the number of points. Empty strings are treated as
                null labels (no tooltip shown on hover).
            image_labels: Optional per-element images for hover thumbnails.
                Accepts List[bytes], List[PIL.Image], List[ndarray], List[Path],
                or Dict[int, Any] for sparse assignment. Prefer pre-encoded
                JPEG/WebP blobs for best compression.
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            grid_shape: Optional grid shape for structured data
            dim_order: Map data columns to scene dimensions by name.
                E.g., ``["Y", "X"]`` for 2D data in a 3D scene.
                Unmapped dims are filled with ``fill`` values and auto-extended.
            fill: Fixed coordinate values for unmapped scene dimensions
                when using ``dim_order``. Defaults to 0.0 for unspecified dims.
            split: Spatial-decomposition control. ``None`` (default) writes
                a single Points node. ``True`` decomposes via midpoint BSP
                with ``max_elements = DEFAULT_MAX_ELEMENTS``.
                ``dict(max_elements=N)`` uses an explicit cap. When the
                decomposition yields more than one part, returns a
                kind=split ``Group`` wrapper carrying ``display_type=
                "points"``; the wrapper's children are ``part_<i>`` Points
                nodes. The wrapper's ``position_bounds`` is the union of
                the children's so picking treats the layer as one entity.
                ``image_labels`` is not supported alongside ``split=``
                (the sparse-dict semantics complicate slicing).
            **attrs: Additional node attributes. Common ones:

                - ``layer`` (bool): Expose this node in the viewer's Layers
                  panel for per-node control. When ``split=`` produces a
                  wrapper, ``layer=True`` lands on the wrapper, not on
                  each leaf part.
                - ``visible`` (bool): Initial visibility when scene loads
                  (default ``True``). Used by the Layers panel to start a
                  layer hidden.
                - ``opacity``, ``intensity``, ``gamma``, ``blending_mode``,
                  ``colormap``: standard rendering attributes.

        Returns:
            The created ``Points`` node, or a kind=split ``Group``
            wrapper when ``split=`` produced more than one part.
        """
        try:
            scene = self._find_scene()

            pos_arr: np.ndarray = (
                positions
                if isinstance(positions, np.ndarray)
                else np.asarray(positions)
            )
            if pos_arr.ndim != 2:
                raise ValueError(
                    f"Positions must have shape (N, D), got shape {pos_arr.shape}"
                )

            # Apply dim_order before validation
            pos_arr, extend_to_all = self._apply_dim_order_positions(
                pos_arr, scene, dim_order, fill, extend_to_all
            )

            n_points = pos_arr.shape[0]
            ndim = pos_arr.shape[1]

            # Split branch — decompose into N children if the user opted in
            # AND the BSP produces more than one part. Single-part outcomes
            # fall through to the regular single-leaf write below.
            if split is not None and pos_arr.shape[1] >= 3:
                from .split import (
                    DEFAULT_MAX_ELEMENTS,
                    midpoint_bsp_partition,
                    sah_bsp_partition,
                )

                if split is True:
                    max_elements = DEFAULT_MAX_ELEMENTS
                    split_rule = "midpoint"
                elif isinstance(split, dict):
                    max_elements = int(
                        split.get("max_elements", DEFAULT_MAX_ELEMENTS)
                    )
                    if max_elements < 1:
                        raise ValueError(
                            f"split max_elements must be >= 1, got {max_elements}"
                        )
                    split_rule = str(split.get("rule", "midpoint"))
                    if split_rule not in ("midpoint", "sah"):
                        raise ValueError(
                            f"split rule must be 'midpoint' or 'sah'; "
                            f"got {split_rule!r}"
                        )
                else:
                    raise TypeError(
                        f"split must be None, True, or dict; got "
                        f"{type(split).__name__}"
                    )

                if image_labels is not None:
                    raise ValueError(
                        "image_labels is not supported alongside split=. "
                        "Decompose the data manually or omit image_labels."
                    )

                if split_rule == "sah":
                    parts = sah_bsp_partition(pos_arr, max_elements)
                else:
                    parts = midpoint_bsp_partition(pos_arr, max_elements)
                if len(parts) > 1:
                    return self._add_points_split_wrapper(
                        name=name,
                        pos_arr=pos_arr,
                        parts=parts,
                        n_points=n_points,
                        colors=colors,
                        radii=radii,
                        sharpness=sharpness,
                        scalars=scalars,
                        labels=labels,
                        parent=parent,
                        extend_to_all=extend_to_all,
                        grid_shape=grid_shape,
                        max_elements=max_elements,
                        additive_lod=additive_lod,
                        **attrs,
                    )
                # 1 part → fall through to additive-LOD / single-leaf write.

            # Additive-LOD branch — multi-level progressive writes via
            # ``_add_points_multi_lod_wrapper``. Fires after the
            # 1-part-split fall-through so a user can pass both
            # ``split=`` and ``additive_lod=`` and get the inner LOD
            # ladder when split doesn't fire.
            if additive_lod is not None:
                from .lod_points import (
                    make_additive_lod_points,
                    resolve_additive_axis_points,
                )

                additive_spec = resolve_additive_axis_points(additive_lod)
                if additive_spec is not None:
                    # Per-element radii needed for salience; broadcast
                    # scalars to per-element array if applicable.
                    if isinstance(radii, np.ndarray) and radii.shape == (n_points,):
                        radii_arr = radii
                    else:
                        radii_arr = None
                    levels = make_additive_lod_points(
                        pos_arr,
                        radii=radii_arr,
                        method=additive_spec["method"],
                        n_lods=additive_spec["n_lods"],
                        counts=additive_spec["counts"],
                        seed=additive_spec["seed"],
                    )
                    if len(levels) > 1:
                        return self._add_points_multi_lod_wrapper(
                            name=name,
                            pos_arr=pos_arr,
                            levels=levels,
                            n_points=n_points,
                            colors=colors,
                            radii=radii,
                            sharpness=sharpness,
                            scalars=scalars,
                            labels=labels,
                            parent=parent,
                            extend_to_all=extend_to_all,
                            grid_shape=grid_shape,
                            method=additive_spec["method"],
                            **attrs,
                        )
                    # 1 level (degenerate) → fall through to single-leaf.

            aprint(f"Adding points node '{name}' with {n_points:,} points in {ndim}D.")

            scene._validate_data_dimensions(pos_arr, name, data_type="positions")

            final_extend_dims = scene._resolve_extend_to_all(
                extend_to_all, pos_arr, "points"
            )
            if final_extend_dims:
                attrs["extend_to_all"] = final_extend_dims
                aprint(f"  📡 Extending visibility across: {final_extend_dims}")

            parent_node = parent or self

            # Colormap / colors mutual exclusivity
            colormap = attrs.get("colormap")
            if colors is not None and colormap is not None:
                raise ValueError(
                    "Cannot specify both 'colors' and 'colormap'. Use one or the other."
                )
            if scalars is not None and colormap is None:
                raise ValueError(
                    "'scalars' requires a 'colormap' attribute to map values to colors."
                )

            if radii is None:
                radii = DEFAULT_POINT_RADIUS
                aprint(f"  📐 Using default radius: {DEFAULT_POINT_RADIUS}")

            writer = self._require_scene_writer(scene)
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = writer.write_points(
                path,
                pos_arr.astype(np.float32),
                colors=cast(Any, colors),
                radii=radii,
                sharpness=sharpness,
                scalars=scalars,
                labels=labels,
                image_labels=image_labels,
                grid_shape=grid_shape,
                **attrs,
            )

            # Sync colormap attr with what the compiler wrote to zarr:
            # - Array colormaps are resolved and stored as "custom"
            # - Non-built-in string names (matplotlib/colorcet) are also
            #   resolved to LUT and stored as "custom"
            if "colormap" in attrs:
                from ..colormaps.builtins import BUILTIN_COLORMAP_NAMES

                cm = attrs["colormap"]
                if not isinstance(cm, str) or (
                    isinstance(cm, str) and cm not in BUILTIN_COLORMAP_NAMES
                ):
                    attrs["colormap"] = "custom"

            # Notify scene that labels exist (for hover overlay auto-injection)
            if labels is not None:
                scene._notify_labels_added()
            if image_labels is not None:
                scene._notify_image_labels_added()

            return Points(
                name,
                metadata=metadata,
                parent=cast(Any, parent_node),
                writer=writer,
                **attrs,
            )
        except (ValueError, TypeError) as e:
            aprint(f"Failed to add points node '{name}': {e}")
            raise ValueError(f"Could not add points '{name}': {e}") from e

    def _add_points_split_wrapper(
        self,
        name: str,
        pos_arr: np.ndarray,
        parts: List[np.ndarray],
        n_points: int,
        colors: Any,
        radii: Any,
        sharpness: Any,
        scalars: Any,
        labels: Any,
        parent: Optional[Node],
        extend_to_all: Optional[Union[List[str], str]],
        grid_shape: Optional[Tuple[int, ...]],
        max_elements: int,
        additive_lod: Any = None,
        **attrs: Any,
    ) -> "Group":
        """Build a kind=split wrapper Group with one Points child per BSP part."""
        wrapper_attrs = {k: v for k, v in attrs.items() if k in _COMPOSITING_ATTRS}
        leaf_attrs = {k: v for k, v in attrs.items() if k not in _COMPOSITING_ATTRS}

        parent_node = parent or self
        wrapper = parent_node.add_split_group(
            name=name,
            display_type="points",
            max_elements=max_elements,
            **wrapper_attrs,
        )

        aprint(
            f"  ✂️  Split '{name}' into {len(parts)} parts via BSP "
            f"(max_elements={max_elements:,}, "
            f"sizes={[int(p.size) for p in parts]})"
        )

        for i, indices in enumerate(parts):
            wrapper.add_points(
                name=f"part_{i}",
                positions=pos_arr[indices],
                colors=_slice_optional_array(colors, indices, n_points),
                radii=_slice_optional_array(radii, indices, n_points),
                sharpness=_slice_optional_array(sharpness, indices, n_points),
                scalars=_slice_optional_array(scalars, indices, n_points),
                labels=_slice_optional_array(labels, indices, n_points),
                # image_labels banned alongside split= (see add_points entry)
                image_labels=None,
                extend_to_all=extend_to_all,
                grid_shape=grid_shape,
                # dim_order / fill already applied to pos_arr upstream — do
                # not re-apply in the per-part recursion.
                dim_order=None,
                fill=None,
                split=None,
                # Inner LOD ladder per spatial part — each part decides
                # its own ladder independently. Allows the Split-of-
                # AdditiveLOD composition from the plan.
                additive_lod=additive_lod,
                **leaf_attrs,
            )

        # Persist the wrapper's position_bounds (per-axis min/max of the
        # full input) so picking / scene-bounds-cache treat the layer as
        # one logical entity. Computed directly from ``pos_arr`` — same
        # result as unioning per-child bboxes, simpler than round-tripping
        # through the children's on-disk attrs.
        wrapper._persist_attr(
            "position_bounds", _position_bounds_from_array(pos_arr)
        )

        return wrapper

    def _add_points_multi_lod_wrapper(
        self,
        name: str,
        pos_arr: np.ndarray,
        levels: List[np.ndarray],
        n_points: int,
        colors: Any,
        radii: Any,
        sharpness: Any,
        scalars: Any,
        labels: Any,
        parent: Optional[Node],
        extend_to_all: Optional[Union[List[str], str]],
        grid_shape: Optional[Tuple[int, ...]],
        method: str,
        **attrs: Any,
    ) -> Points:
        """Write a Points node with multi-additive-LOD subgroups.

        Produces ``<path>/additive_<i>/`` subgroups (one per LOD level),
        each carrying the points assigned to that level. The parent
        points node carries ``n_additive_sublods=N``, a global
        ``position_bounds``, and the standard compositing attrs.

        The returned :class:`Points` node is the parent (the user's
        logical "one node"). The viewer's progressive loader walks the
        subgroups; the user never sees the decomposition.
        """
        from .points import Points

        scene = self._find_scene()
        writer = self._require_scene_writer(scene)
        parent_node = parent or self
        path = f"{parent_node.path}/{name}" if parent_node.path else name

        # Build per-level slice tuples for the writer.
        level_slices: List[Dict[str, Any]] = []
        for level_indices in levels:
            level_slices.append(
                {
                    "positions": pos_arr[level_indices].astype(np.float32),
                    "colors": _slice_optional_array(
                        colors, level_indices, n_points
                    ),
                    "radii": _slice_optional_array(
                        radii, level_indices, n_points
                    ),
                    "sharpness": _slice_optional_array(
                        sharpness, level_indices, n_points
                    ),
                    "scalars": _slice_optional_array(
                        scalars, level_indices, n_points
                    ),
                    "labels": _slice_optional_array(
                        labels, level_indices, n_points
                    ),
                }
            )

        aprint(
            f"  📐 Additive-LOD '{name}': {len(levels)} levels "
            f"(method={method!r}, sizes={[int(L.size) for L in levels]})"
        )

        metadata = writer.write_points_multi_lod(
            path,
            level_slices,
            method=method,
            grid_shape=grid_shape,
            extend_to_all=extend_to_all,
            **attrs,
        )

        return Points(
            name,
            metadata=metadata,
            parent=cast(Any, parent_node),
            writer=writer,
            **attrs,
        )

    def add_lines(
        self,
        name: str,
        vertices: Union[PositionArray, np.ndarray[Any, Any], Sequence[Sequence[float]]],
        widths: Union[
            np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any], float
        ],
        colors: Optional[
            Union[ColorArray, np.ndarray[Any, Any], Sequence[float | int]]
        ] = None,
        sharpness: Optional[
            Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any], float]
        ] = None,
        scalars: Optional[
            Union[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any], float]
        ] = None,
        labels: Optional[Union[List[str], Sequence[str]]] = None,
        image_labels: Optional[Any] = None,
        indices: Optional[np.ndarray[Any, Any]] = None,
        line_type: str = "polyline",
        parent: Optional[Node] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        additive_lod: Any = None,
        split: Any = None,
        **attrs: Any,
    ) -> Union[Lines, "Group"]:
        """Add a lines node.

        Args:
            name: Name of the lines node
            vertices: Array of shape (N, D) for vertex positions
            widths: (N,) array or scalar for line widths
            colors: Optional (N, 3) array, RGB tuple, or None
            sharpness: Optional (N,) array, scalar, or None
            scalars: Optional (N,) array or scalar for colormap lookup.
                Requires ``colormap`` in attrs. Mutually exclusive with ``colors``.
            labels: Optional list of strings, one per vertex. Used for hover tooltips.
            image_labels: Optional per-element images for hover thumbnails.
            indices: Optional vertex indices for indexed line type
            line_type: Connectivity ("segments", "polyline", "loop", "indexed")
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map data columns to scene dimensions by name
            fill: Fixed values for unmapped dimensions when using dim_order
            **attrs: Additional node attributes. Common ones:

                - ``layer`` (bool): Expose this node in the viewer's Layers
                  panel for per-node control.
                - ``visible`` (bool): Initial visibility when scene loads
                  (default ``True``).
                - ``opacity``, ``intensity``, ``gamma``, ``blending_mode``,
                  ``colormap``: standard rendering attributes.

        Returns:
            The created Lines node
        """
        try:
            scene = self._find_scene()

            vert_arr: np.ndarray = (
                vertices if isinstance(vertices, np.ndarray) else np.asarray(vertices)
            )
            if vert_arr.ndim != 2:
                raise ValueError(
                    f"Vertices must have shape (N, D), got shape {vert_arr.shape}"
                )

            # Apply dim_order before validation
            vert_arr, extend_to_all = self._apply_dim_order_positions(
                vert_arr, scene, dim_order, fill, extend_to_all
            )

            n_vertices = vert_arr.shape[0]
            ndim = vert_arr.shape[1]

            # NOTE: compiler-level auto-split heuristic is a separate PR
            # (β); ``add_lines`` honors user-explicit ``split=`` here and
            # will pick up the auto-split path automatically once β
            # merges. Until then, ``split=`` is opt-in via the call site.

            # Split branch — polyline-aware BSP. Whole polylines are
            # atomic; the BSP runs over per-polyline centroids and assigns
            # each polyline atomically to a part. Mirrors add_points but
            # at the polyline granularity.
            if split is not None and vert_arr.shape[1] >= 3:
                from .lod_lines import identify_polylines
                from .split import (
                    DEFAULT_MAX_ELEMENTS,
                    midpoint_bsp_polylines,
                    sah_bsp_partition,
                )

                if split is True:
                    max_elements = DEFAULT_MAX_ELEMENTS
                    split_rule = "midpoint"
                elif isinstance(split, dict):
                    max_elements = int(
                        split.get("max_elements", DEFAULT_MAX_ELEMENTS)
                    )
                    if max_elements < 1:
                        raise ValueError(
                            f"split max_elements must be >= 1, got {max_elements}"
                        )
                    split_rule = str(split.get("rule", "midpoint"))
                    if split_rule not in ("midpoint", "sah"):
                        raise ValueError(
                            f"split rule must be 'midpoint' or 'sah'; "
                            f"got {split_rule!r}"
                        )
                else:
                    raise TypeError(
                        f"split must be None, True, or dict; got "
                        f"{type(split).__name__}"
                    )

                if image_labels is not None:
                    raise ValueError(
                        "image_labels is not supported alongside split=. "
                        "Decompose the data manually or omit image_labels."
                    )

                polyline_indices = identify_polylines(
                    n_vertices, line_type, indices
                )

                if split_rule == "sah":
                    # SAH operates on per-polyline centroids in this
                    # context too — same atomic-polyline guarantee.
                    if not polyline_indices:
                        polyline_parts: List[List[int]] = []
                    else:
                        centroids = np.array(
                            [
                                vert_arr[p, :3].mean(axis=0)
                                if p.size > 0
                                else np.zeros(3)
                                for p in polyline_indices
                            ],
                            dtype=np.float64,
                        )
                        # Cap is per-vertex; SAH gives us per-centroid
                        # parts; we re-aggregate to vertex-count parts.
                        approx_per_poly = max(
                            1,
                            n_vertices // max(1, len(polyline_indices)),
                        )
                        centroid_cap = max(
                            1, max_elements // approx_per_poly
                        )
                        centroid_parts = sah_bsp_partition(
                            centroids, max_elements=centroid_cap
                        )
                        polyline_parts = [
                            idx_arr.tolist() for idx_arr in centroid_parts
                        ]
                else:
                    polyline_parts = midpoint_bsp_polylines(
                        vert_arr, polyline_indices, max_elements
                    )

                if len(polyline_parts) > 1:
                    return self._add_lines_split_wrapper(
                        name=name,
                        vert_arr=vert_arr,
                        polyline_indices=polyline_indices,
                        polyline_parts=polyline_parts,
                        n_vertices=n_vertices,
                        widths=widths,
                        colors=colors,
                        sharpness=sharpness,
                        scalars=scalars,
                        labels=labels,
                        indices=indices,
                        line_type=line_type,
                        parent=parent,
                        extend_to_all=extend_to_all,
                        max_elements=max_elements,
                        **attrs,
                    )
                # 1 part → fall through to single-leaf write.

            # Additive-LOD branch — polyline-level multi-LOD write.
            # Fires before the single-shot write so we don't double-
            # validate. Mirrors the points add path.
            if additive_lod is not None:
                from .lod_lines import (
                    make_additive_lod_lines,
                    resolve_additive_axis_lines,
                )

                additive_spec = resolve_additive_axis_lines(additive_lod)
                if additive_spec is not None:
                    widths_arr = (
                        widths
                        if isinstance(widths, np.ndarray)
                        and widths.shape == (n_vertices,)
                        else None
                    )
                    polyline_levels = make_additive_lod_lines(
                        vert_arr,
                        line_type=line_type,
                        indices=indices,
                        widths=widths_arr,
                        method=additive_spec["method"],
                        n_lods=additive_spec["n_lods"],
                        counts=additive_spec["counts"],
                        seed=additive_spec["seed"],
                    )
                    if len(polyline_levels) > 1:
                        return self._add_lines_multi_lod_wrapper(
                            name=name,
                            vert_arr=vert_arr,
                            polyline_levels=polyline_levels,
                            n_vertices=n_vertices,
                            widths=widths,
                            colors=colors,
                            sharpness=sharpness,
                            scalars=scalars,
                            labels=labels,
                            parent=parent,
                            extend_to_all=extend_to_all,
                            method=additive_spec["method"],
                            **attrs,
                        )
                    # 1 level (degenerate single polyline) → fall through.

            aprint(
                f"Adding lines node '{name}' with {n_vertices:,} vertices in {ndim}D."
            )

            scene._validate_data_dimensions(vert_arr, name, data_type="vertices")

            final_extend_dims = scene._resolve_extend_to_all(
                extend_to_all, vert_arr, "lines"
            )
            if final_extend_dims:
                attrs["extend_to_all"] = final_extend_dims
                aprint(f"  📡 Extending visibility across: {final_extend_dims}")

            # Colormap / colors mutual exclusivity
            colormap = attrs.get("colormap")
            if colors is not None and colormap is not None:
                raise ValueError(
                    "Cannot specify both 'colors' and 'colormap'. Use one or the other."
                )
            if scalars is not None and colormap is None:
                raise ValueError(
                    "'scalars' requires a 'colormap' attribute to map values to colors."
                )

            parent_node = parent or self

            writer = self._require_scene_writer(scene)
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = writer.write_lines(
                path,
                vert_arr.astype(np.float32),
                widths=widths,
                colors=cast(Any, colors),
                sharpness=sharpness,
                scalars=scalars,
                indices=indices,
                line_type=line_type,
                labels=labels,
                image_labels=image_labels,
                **attrs,
            )

            # Sync colormap attr with what the compiler wrote to zarr
            if "colormap" in attrs:
                from ..colormaps.builtins import BUILTIN_COLORMAP_NAMES

                cm = attrs["colormap"]
                if not isinstance(cm, str) or (
                    isinstance(cm, str) and cm not in BUILTIN_COLORMAP_NAMES
                ):
                    attrs["colormap"] = "custom"

            if labels is not None:
                scene._notify_labels_added()
            if image_labels is not None:
                scene._notify_image_labels_added()

            return Lines(
                name,
                metadata=metadata,
                parent=cast(Any, parent_node),
                writer=writer,
                **attrs,
            )
        except (ValueError, TypeError) as e:
            aprint(f"Failed to add lines node '{name}': {e}")
            raise ValueError(f"Could not add lines '{name}': {e}") from e

    def _add_lines_split_wrapper(
        self,
        name: str,
        vert_arr: np.ndarray,
        polyline_indices: List[np.ndarray],
        polyline_parts: List[List[int]],
        n_vertices: int,
        widths: Any,
        colors: Any,
        sharpness: Any,
        scalars: Any,
        labels: Any,
        indices: Optional[np.ndarray],
        line_type: str,
        parent: Optional[Node],
        extend_to_all: Optional[Union[List[str], str]],
        max_elements: int,
        **attrs: Any,
    ) -> "Group":
        """Build a kind=split wrapper Group with one Lines child per BSP part.

        Polylines are atomic — each polyline lands in exactly one part.
        For the ``segments`` / ``indexed`` line types, the resulting
        per-part data is re-emitted with ``line_type='segments'``: the
        original segment topology is preserved by walking pairs within
        each component the BSP grouped together. For ``polyline`` /
        ``loop`` types (where the input is a single polyline), the BSP
        only ever produces one part — the user is already at the single-
        polyline granularity and there's nothing to split. We refuse the
        split in that case with a clear error.
        """
        wrapper_attrs = {k: v for k, v in attrs.items() if k in _COMPOSITING_ATTRS}
        leaf_attrs = {k: v for k, v in attrs.items() if k not in _COMPOSITING_ATTRS}

        parent_node = parent or self
        wrapper = parent_node.add_split_group(
            name=name,
            display_type="lines",
            max_elements=max_elements,
            **wrapper_attrs,
        )

        part_sizes = [
            sum(int(polyline_indices[p].size) for p in part)
            for part in polyline_parts
        ]
        aprint(
            f"  ✂️  Split '{name}' into {len(polyline_parts)} parts via "
            f"polyline-centroid BSP "
            f"(max_elements={max_elements:,}, sizes={part_sizes})"
        )

        # The new line_type per part is either:
        # - ``polyline`` / ``loop`` with one polyline ⇒ keep as-is.
        # - ``segments`` / ``indexed`` ⇒ re-emit as ``segments`` with the
        #   pairs from the polylines that landed in this part.
        for i, polyline_ids in enumerate(polyline_parts):
            # Collect vertices for this part, preserving original order.
            vertex_index_list: List[np.ndarray] = []
            new_segments: List[List[int]] = []
            cursor = 0
            for p in polyline_ids:
                members = polyline_indices[p]
                if members.size == 0:
                    continue
                vertex_index_list.append(members)
                # For segments / indexed, re-emit each pair after
                # remapping into the part-local vertex indexing (which
                # follows the concatenation order).
                if line_type in ("segments", "indexed"):
                    # Walk in pairs along the polyline's original member
                    # ordering. For ``segments`` this is just (0,1),
                    # (2,3), ... For ``indexed`` connected components,
                    # consecutive members aren't necessarily a segment;
                    # we approximate by linking consecutive members,
                    # which is exact for ``segments`` (the only case the
                    # split path actually decomposes — see polyline /
                    # loop guard below).
                    for k in range(0, members.size - 1, 2):
                        new_segments.append([cursor + k, cursor + k + 1])
                cursor += int(members.size)

            if not vertex_index_list:
                continue
            part_vertex_idx = np.concatenate(vertex_index_list)
            part_vertices = vert_arr[part_vertex_idx]
            part_n = int(part_vertex_idx.size)

            # Slice per-vertex parameters into this part.
            part_widths = (
                widths
                if (not isinstance(widths, np.ndarray)) or widths.shape != (n_vertices,)
                else widths[part_vertex_idx]
            )
            part_colors = _slice_optional_array(colors, part_vertex_idx, n_vertices)
            part_sharpness = _slice_optional_array(
                sharpness, part_vertex_idx, n_vertices
            )
            part_scalars = _slice_optional_array(
                scalars, part_vertex_idx, n_vertices
            )
            part_labels = _slice_optional_array(
                labels, part_vertex_idx, n_vertices
            )

            # Choose the per-part line_type. ``polyline`` / ``loop`` with
            # one polyline = one part, so the original type is preserved.
            # For ``segments`` we emit segments. For ``indexed`` (rare
            # — typically pre-merged graphs) we also emit segments with
            # the reconstructed indices below.
            if line_type in ("polyline", "loop"):
                part_line_type = line_type
                part_indices = None
            elif line_type == "segments":
                part_line_type = "segments"
                part_indices = None
            else:  # indexed
                part_line_type = "indexed"
                part_indices = (
                    np.asarray(new_segments, dtype=np.intp).reshape(-1, 2)
                    if new_segments
                    else None
                )
                if part_indices is None:
                    # Single-vertex polylines on indexed → emit as
                    # ``segments`` of zero length (caller asked for
                    # indexed but the part has no edges; degrades
                    # gracefully).
                    part_line_type = "segments"
                    if part_n % 2 != 0:
                        # Round to an even count to satisfy segments
                        # validation; drop the trailing isolated vertex.
                        part_vertices = part_vertices[:-1]
                        part_n -= 1

            wrapper.add_lines(
                name=f"part_{i}",
                vertices=part_vertices,
                widths=part_widths,
                colors=part_colors,
                sharpness=part_sharpness,
                scalars=part_scalars,
                labels=part_labels,
                # image_labels banned alongside split= (see add_lines entry)
                image_labels=None,
                indices=part_indices,
                line_type=part_line_type,
                extend_to_all=extend_to_all,
                dim_order=None,
                fill=None,
                split=None,
                **leaf_attrs,
            )

        wrapper._persist_attr(
            "position_bounds", _position_bounds_from_array(vert_arr)
        )

        return wrapper

    def _add_lines_multi_lod_wrapper(
        self,
        name: str,
        vert_arr: np.ndarray,
        polyline_levels: List[List[np.ndarray]],
        n_vertices: int,
        widths: Any,
        colors: Any,
        sharpness: Any,
        scalars: Any,
        labels: Any,
        parent: Optional[Node],
        extend_to_all: Optional[Union[List[str], str]],
        method: str,
        **attrs: Any,
    ) -> Lines:
        """Write a Lines node with multi-additive-LOD subgroups.

        Each ``additive_<i>/`` subgroup carries a subset of polylines
        (whole polylines, never bisected). Segment indices are local to
        each subgroup. The parent lines node carries
        ``n_additive_sublods``, the global ``position_bounds``, and the
        standard compositing attrs.
        """
        from .lines import Lines

        scene = self._find_scene()
        writer = self._require_scene_writer(scene)
        parent_node = parent or self
        path = f"{parent_node.path}/{name}" if parent_node.path else name

        level_slices: List[Dict[str, Any]] = []
        for level_polylines in polyline_levels:
            # Concatenate vertex indices across all polylines in this
            # level; build local segment indices per polyline.
            level_vertex_indices: List[int] = []
            level_local_segments: List[np.ndarray] = []
            local_offset = 0
            for poly in level_polylines:
                k = poly.size
                level_vertex_indices.extend(int(idx) for idx in poly)
                if k >= 2:
                    # Polyline connectivity: (0,1), (1,2), ..., (k-2, k-1)
                    seg = np.column_stack(
                        [
                            np.arange(k - 1, dtype=np.uint32) + local_offset,
                            np.arange(1, k, dtype=np.uint32) + local_offset,
                        ]
                    )
                    level_local_segments.append(seg)
                local_offset += k
            vertex_index_arr = np.asarray(level_vertex_indices, dtype=np.intp)
            if level_local_segments:
                level_segments = np.concatenate(level_local_segments, axis=0)
            else:
                level_segments = np.empty((0, 2), dtype=np.uint32)

            level_slices.append(
                {
                    "vertices": vert_arr[vertex_index_arr].astype(np.float32),
                    "widths": _slice_optional_array(
                        widths, vertex_index_arr, n_vertices
                    ),
                    "colors": _slice_optional_array(
                        colors, vertex_index_arr, n_vertices
                    ),
                    "sharpness": _slice_optional_array(
                        sharpness, vertex_index_arr, n_vertices
                    ),
                    "scalars": _slice_optional_array(
                        scalars, vertex_index_arr, n_vertices
                    ),
                    "labels": _slice_optional_array(
                        labels, vertex_index_arr, n_vertices
                    ),
                    "segments": level_segments,
                    "n_polylines": len(level_polylines),
                }
            )

        aprint(
            f"  📐 Additive-LOD '{name}': {len(polyline_levels)} levels "
            f"(method={method!r}, polylines_per_level="
            f"{[len(L) for L in polyline_levels]})"
        )

        metadata = writer.write_lines_multi_lod(
            path,
            level_slices,
            method=method,
            extend_to_all=extend_to_all,
            **attrs,
        )

        return Lines(
            name,
            metadata=metadata,
            parent=cast(Any, parent_node),
            writer=writer,
            **attrs,
        )

    def add_gsplats(
        self,
        name: str,
        centers: Union[PositionArray, np.ndarray[Any, Any], Sequence[Sequence[float]]],
        amplitudes: Union[
            np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any], float
        ],
        cholesky_factors: Union[
            np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, Any]
        ],
        colors: Optional[
            Union[ColorArray, np.ndarray[Any, Any], Sequence[float | int]]
        ] = None,
        labels: Optional[Union[List[str], Sequence[str]]] = None,
        image_labels: Optional[Any] = None,
        parent: Optional[Node] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        fill_sigma: Optional[Dict[str, float]] = None,
        split: Any = None,
        **attrs: Any,
    ) -> Union[GSplats, "Group"]:
        """Add a Gaussian splats node.

        Args:
            name: Name of the gsplats node
            centers: Array of shape (N, D) for splat centers
            amplitudes: (N,) array or scalar for intensities
            cholesky_factors: (N, k) packed Cholesky factors, k=D*(D+1)/2
            colors: Optional (N, 3) array, RGB tuple, or None
            labels: Optional list of strings, one per splat. Used for hover tooltips.
            image_labels: Optional per-element images for hover thumbnails.
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map data columns to scene dimensions by name.
                Also reorders and embeds Cholesky factors automatically.
            fill: Fixed coordinate values for unmapped dimensions
            fill_sigma: Standard deviations for unmapped dimensions in the
                Cholesky embedding (default 1.0). Controls splat extent in
                unmapped dims.
            split: Spatial-decomposition control. ``None`` (default) writes
                a single GSplats node. ``True`` decomposes via midpoint BSP
                with ``max_elements = DEFAULT_MAX_ELEMENTS``.
                ``dict(max_elements=N)`` uses an explicit cap. When the
                decomposition yields more than one part, returns a
                kind=split ``Group`` wrapper carrying ``display_type=
                "gsplats"``; the wrapper's children are ``part_<i>``
                GSplats nodes. ``image_labels`` is not supported alongside
                ``split=``.
            **attrs: Additional node attributes. Common ones:

                - ``layer`` (bool): Expose this node in the viewer's Layers
                  panel for per-node control. When ``split=`` produces a
                  wrapper, ``layer=True`` lands on the wrapper, not on
                  each leaf part.
                - ``visible`` (bool): Initial visibility when scene loads
                  (default ``True``).
                - ``opacity``, ``intensity``, ``gamma``, ``blending_mode``,
                  ``colormap``: standard rendering attributes.

        Returns:
            The created ``GSplats`` node, or a kind=split ``Group``
            wrapper when ``split=`` produced more than one part.
        """
        try:
            scene = self._find_scene()

            ctr_arr: np.ndarray = (
                centers if isinstance(centers, np.ndarray) else np.asarray(centers)
            )
            if ctr_arr.ndim != 2:
                raise ValueError(
                    f"Centers must have shape (N, D), got shape {ctr_arr.shape}"
                )

            d_data = ctr_arr.shape[1]

            chol_arr: np.ndarray = (
                cholesky_factors
                if isinstance(cholesky_factors, np.ndarray)
                else np.asarray(cholesky_factors)
            )

            # Apply dim_order: transform both centers and cholesky_factors
            if dim_order is not None:
                ctr_arr, extend_to_all = self._apply_dim_order_positions(
                    ctr_arr, scene, dim_order, fill, extend_to_all
                )
                chol_arr = self._apply_dim_order_cholesky(
                    chol_arr, d_data, scene, dim_order, fill_sigma
                )

            n_splats = ctr_arr.shape[0]
            ndim = ctr_arr.shape[1]

            # Split branch — decompose into N children if the user opted in
            # AND the BSP produces more than one part.
            if split is not None and ctr_arr.shape[1] >= 3:
                from .split import (
                    DEFAULT_MAX_ELEMENTS,
                    midpoint_bsp_partition,
                    sah_bsp_partition,
                )

                if split is True:
                    max_elements = DEFAULT_MAX_ELEMENTS
                    split_rule = "midpoint"
                elif isinstance(split, dict):
                    max_elements = int(
                        split.get("max_elements", DEFAULT_MAX_ELEMENTS)
                    )
                    if max_elements < 1:
                        raise ValueError(
                            f"split max_elements must be >= 1, got {max_elements}"
                        )
                    split_rule = str(split.get("rule", "midpoint"))
                    if split_rule not in ("midpoint", "sah"):
                        raise ValueError(
                            f"split rule must be 'midpoint' or 'sah'; "
                            f"got {split_rule!r}"
                        )
                else:
                    raise TypeError(
                        f"split must be None, True, or dict; got "
                        f"{type(split).__name__}"
                    )

                if image_labels is not None:
                    raise ValueError(
                        "image_labels is not supported alongside split=. "
                        "Decompose the data manually or omit image_labels."
                    )

                if split_rule == "sah":
                    parts = sah_bsp_partition(ctr_arr, max_elements)
                else:
                    parts = midpoint_bsp_partition(ctr_arr, max_elements)
                if len(parts) > 1:
                    return self._add_gsplats_split_wrapper(
                        name=name,
                        ctr_arr=ctr_arr,
                        chol_arr=chol_arr,
                        amplitudes=amplitudes,
                        parts=parts,
                        n_splats=n_splats,
                        colors=colors,
                        labels=labels,
                        parent=parent,
                        extend_to_all=extend_to_all,
                        max_elements=max_elements,
                        **attrs,
                    )
                # 1 part → fall through to single-leaf write.

            aprint(f"Adding gsplats node '{name}' with {n_splats:,} splats in {ndim}D.")

            scene._validate_data_dimensions(ctr_arr, name, data_type="centers")

            final_extend_dims = scene._resolve_extend_to_all(
                extend_to_all, ctr_arr, "splats"
            )
            if final_extend_dims:
                attrs["extend_to_all"] = final_extend_dims
                aprint(f"  📡 Extending visibility across: {final_extend_dims}")

            # Colormap / colors mutual exclusivity
            colormap = attrs.get("colormap")
            if colors is not None and colormap is not None:
                raise ValueError(
                    "Cannot specify both 'colors' and 'colormap'. Use one or the other."
                )

            parent_node = parent or self

            writer = self._require_scene_writer(scene)
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = writer.write_gsplats(
                path,
                ctr_arr.astype(np.float32),
                amplitudes=amplitudes,
                cholesky_factors=chol_arr,
                colors=cast(Any, colors),
                labels=labels,
                image_labels=image_labels,
                **attrs,
            )

            # Sync colormap attr with what the compiler wrote to zarr
            if "colormap" in attrs:
                from ..colormaps.builtins import BUILTIN_COLORMAP_NAMES

                cm = attrs["colormap"]
                if not isinstance(cm, str) or (
                    isinstance(cm, str) and cm not in BUILTIN_COLORMAP_NAMES
                ):
                    attrs["colormap"] = "custom"

            # The compiler sets default "gray" colormap for gsplats without
            # colors/colormap. Propagate that to the Node attrs so the
            # in-memory node matches the zarr state.
            if not metadata.get("has_colors") and "colormap" not in attrs:
                attrs["colormap"] = "gray"

            if labels is not None:
                scene._notify_labels_added()
            if image_labels is not None:
                scene._notify_image_labels_added()

            return GSplats(
                name,
                metadata=metadata,
                parent=cast(Any, parent_node),
                writer=writer,
                **attrs,
            )
        except (ValueError, TypeError) as e:
            aprint(f"Failed to add gsplats node '{name}': {e}")
            raise ValueError(f"Could not add gsplats '{name}': {e}") from e

    def _add_gsplats_split_wrapper(
        self,
        name: str,
        ctr_arr: np.ndarray,
        chol_arr: np.ndarray,
        amplitudes: Any,
        parts: List[np.ndarray],
        n_splats: int,
        colors: Any,
        labels: Any,
        parent: Optional[Node],
        extend_to_all: Optional[Union[List[str], str]],
        max_elements: int,
        **attrs: Any,
    ) -> "Group":
        """Build a kind=split wrapper Group with one GSplats child per BSP part."""
        wrapper_attrs = {k: v for k, v in attrs.items() if k in _COMPOSITING_ATTRS}
        leaf_attrs = {k: v for k, v in attrs.items() if k not in _COMPOSITING_ATTRS}

        parent_node = parent or self
        wrapper = parent_node.add_split_group(
            name=name,
            display_type="gsplats",
            max_elements=max_elements,
            **wrapper_attrs,
        )

        aprint(
            f"  ✂️  Split '{name}' into {len(parts)} parts via BSP "
            f"(max_elements={max_elements:,}, "
            f"sizes={[int(p.size) for p in parts]})"
        )

        for i, indices in enumerate(parts):
            wrapper.add_gsplats(
                name=f"part_{i}",
                centers=ctr_arr[indices],
                amplitudes=_slice_optional_array(amplitudes, indices, n_splats),
                cholesky_factors=_slice_optional_array(
                    chol_arr, indices, n_splats
                ),
                colors=_slice_optional_array(colors, indices, n_splats),
                labels=_slice_optional_array(labels, indices, n_splats),
                image_labels=None,
                extend_to_all=extend_to_all,
                # dim_order / fill / fill_sigma already applied to ctr_arr +
                # chol_arr upstream — do not re-apply in the per-part call.
                dim_order=None,
                fill=None,
                fill_sigma=None,
                split=None,
                **leaf_attrs,
            )

        wrapper._persist_attr(
            "position_bounds", _position_bounds_from_array(ctr_arr)
        )

        return wrapper

    def add_gsplats_from_data(
        self,
        name: str,
        result: GSplatData,
        parent: Optional[Node] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        fill_sigma: Optional[Dict[str, float]] = None,
        lod_group: Any = None,
        additive_lod: Any = None,
        **attrs: Any,
    ) -> Union[GSplats, "Group"]:
        """Add Gaussian splats from a GSplatData object.

        Multi-additive-LOD data (from progressive fitting or
        ``make_additive_lod``) is written with per-sub-LOD subgroups
        directly under the gsplats node (``<node>/additive_<i>/...``)
        for progressive (prefix-sum) loading. Single-LOD data uses the
        flat layout (arrays at the node path).

        ``lod_group`` and ``additive_lod`` control the two LOD axes (see
        ``luxar.core.lod.resolve_substitutive_axis`` /
        ``resolve_additive_axis`` for the full value vocabulary). When
        the resolved data has multiple substitutive levels, this method
        builds a ``kind="lod"`` ``Group`` containing one gsplats child
        per level (in coarsest→finest order, named ``child_<i>``) and
        returns it; otherwise it returns a single :class:`GSplats` node.

        ``min_pixel_size`` may only be passed in ``**attrs`` when the
        result is single-substitutive AND the parent is itself a
        ``kind="lod"`` ``Group`` (the child is a leaf of an enclosing
        LOD group). Passing it on a multi-substitutive path raises
        ``ValueError`` — use ``lod_group=dict(min_pixel_sizes=[...])`` to
        override the auto-derived thresholds.

        Args:
            name: Name of the gsplats (or kind=lod group) node.
            result: GSplatData from ``fit_gaussian_splats`` or similar.
            parent: Parent node (default: this group).
            extend_to_all: Visibility extension across non-displayed dimensions.
            dim_order: Map data columns to scene dimensions by name.
            fill: Fixed coordinate values for unmapped dimensions.
            fill_sigma: Standard deviations for unmapped dims in Cholesky embedding.
            lod_group: Substitutive-axis control. ``None`` (pass-through; drop
                non-default substitutive levels), ``True`` (require stored
                levels), ``False`` (collapse to finest), ``dict(...)`` (compute
                via :func:`make_substitutive_lod`), or ``dict(..., recompute=
                True)``. Optional ``min_pixel_sizes=[...]`` inside the dict
                overrides the auto-derived thresholds.
            additive_lod: Additive-axis control, uniform across substitutive
                levels. Same value vocabulary as ``lod_group``; ``dict(...)``
                routes to :func:`make_additive_lod`.
            **attrs: Additional node attributes.

        Example:
            >>> result = fit_gaussian_splats(volume_3d)
            >>> # Plain flat gsplats node
            >>> scene.add_gsplats_from_data("splats", result,
            ...     dim_order=["Z", "Y", "X"], fill={"Time": 0})
            >>>
            >>> # Auto-build a 3-level kind=lod Group with a 4-step
            >>> # additive ladder per level, computed from a flat input.
            >>> scene.add_gsplats_from_data(
            ...     "multires", flat_result,
            ...     lod_group=dict(compression_factor=4, levels=2),
            ...     additive_lod=dict(n_lods=4),
            ... )
        """
        from luxar.gsplats.gsplat_data import GSplatData

        from .lod import (
            resolve_additive_axis,
            resolve_substitutive_axis,
        )

        if not isinstance(result, GSplatData):
            raise TypeError(f"Expected GSplatData, got {type(result).__name__}")

        # Propagate truncation_radius through attrs (unless caller overrode it)
        if "truncation_radius" not in attrs:
            attrs["truncation_radius"] = result.truncation_radius

        # Resolve the two LOD axes. Substitutive first (it can produce a
        # multi-level result), then additive (uniform across levels).
        result, explicit_min_pixel_sizes = resolve_substitutive_axis(
            result, lod_group
        )
        result = resolve_additive_axis(result, additive_lod)

        # Multi-substitutive → kind=lod Group with one gsplats child per level
        if result.n_substitutive > 1:
            if "min_pixel_size" in attrs:
                raise ValueError(
                    "min_pixel_size must not be passed when the resolved "
                    "result is multi-substitutive: thresholds are derived "
                    "per-child (or set via lod_group=dict(min_pixel_sizes="
                    "[...]))."
                )
            return self._add_gsplats_as_lod_group(
                name=name,
                result=result,
                explicit_min_pixel_sizes=explicit_min_pixel_sizes,
                parent=parent,
                extend_to_all=extend_to_all,
                dim_order=dim_order,
                fill=fill,
                fill_sigma=fill_sigma,
                **attrs,
            )

        # Single-substitutive: flat or multi-additive path
        if result.n_additive_sublods <= 1:
            return self.add_gsplats(
                name=name,
                centers=result.centers,
                amplitudes=result.amplitudes,
                cholesky_factors=result.cholesky_factors,
                colors=result.colors,
                parent=parent,
                extend_to_all=extend_to_all,
                dim_order=dim_order,
                fill=fill,
                fill_sigma=fill_sigma,
                **attrs,
            )

        return self._add_gsplats_multi_lod(
            name=name,
            result=result,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            **attrs,
        )

    def _add_gsplats_as_lod_group(
        self,
        name: str,
        result: GSplatData,
        explicit_min_pixel_sizes: Optional[List[float]],
        parent: Optional[Node] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        fill_sigma: Optional[Dict[str, float]] = None,
        **attrs: Any,
    ) -> "Group":
        """Build a kind=lod ``Group`` with one gsplats child per substitutive level.

        Children are written in coarsest→finest order and named
        ``child_<i>``. Compositing attrs (opacity, gamma, intensity,
        offset, blending_mode, transform, layer, visible, nd_transform)
        land on the kind=lod ``Group`` itself; per-leaf gsplats attrs
        (truncation_radius, extend_to_all, colormap) ride into each child.
        """
        from .lod import derive_min_pixel_sizes

        # Substitutive convention: index 0 = finest, n-1 = coarsest. The
        # LOD group needs coarsest first.
        n_sub = result.n_substitutive
        order = list(range(n_sub - 1, -1, -1))

        # Per-level total splat count (sum across each level's additive
        # ladder) — used both for logging and for auto-deriving
        # min_pixel_sizes when the user didn't supply them.
        splat_counts: list[int] = [
            sum(sub.n_splats for sub in result.substitutive_levels[s].additive_sublods)
            for s in order
        ]

        if explicit_min_pixel_sizes is not None:
            if len(explicit_min_pixel_sizes) != n_sub:
                raise ValueError(
                    f"min_pixel_sizes has {len(explicit_min_pixel_sizes)} "
                    f"entries but the lod_group has {n_sub} substitutive levels"
                )
            min_pixel_sizes = list(explicit_min_pixel_sizes)
        else:
            min_pixel_sizes = derive_min_pixel_sizes(splat_counts)

        # Separate compositing attrs (go on the kind=lod Group) from
        # per-leaf gsplats attrs (go on each child). Anything not in the
        # compositing set falls through to the child level.
        #
        # ``colormap`` is intentionally NOT compositing here: the writer
        # auto-defaults a missing colormap to "gray" per leaf, which
        # under nearest-ancestor-wins would shadow a parent's setting.
        # Keep it on each child so the user's intent survives.
        lod_attrs = {k: v for k, v in attrs.items() if k in _COMPOSITING_ATTRS}
        child_attrs = {k: v for k, v in attrs.items() if k not in _COMPOSITING_ATTRS}
        # Defense in depth: ``add_gsplats_from_data`` rejects min_pixel_size
        # at entry, but if this method is invoked through a different path
        # (e.g. internal recursion) we still need to strip it — the loop
        # below passes a derived value as an explicit kwarg, and a duplicate
        # in ``**child_attrs`` would raise TypeError.
        child_attrs.pop("min_pixel_size", None)

        parent_node = parent or self
        # All children in this branch are gsplats leaves, so the user-facing
        # ``display_type`` is unambiguously "gsplats". Set it here so the
        # on-disk attrs are self-describing.
        lod_attrs.setdefault("display_type", "gsplats")
        lod_group_node: Group = parent_node.add_lod_group(name, **lod_attrs)

        aprint(
            f"Adding multi-resolution gsplats node '{name}' as kind=lod "
            f"Group with {n_sub} substitutive levels: "
            f"{[f'{n:,}' for n in splat_counts]} splats"
        )

        # Add one gsplats child per substitutive level, coarsest first.
        for child_idx, s in enumerate(order):
            child_name = f"child_{child_idx}"
            level_view = result.at_substitutive(s)
            # Recursive dispatch — but explicitly None on both LOD axes so
            # the resolvers no-op and we never re-enter the kind=lod branch.
            lod_group_node.add_gsplats_from_data(
                name=child_name,
                result=level_view,
                extend_to_all=extend_to_all,
                dim_order=dim_order,
                fill=fill,
                fill_sigma=fill_sigma,
                lod_group=None,
                additive_lod=None,
                min_pixel_size=min_pixel_sizes[child_idx],
                **child_attrs,
            )

        return lod_group_node

    def _add_gsplats_multi_lod(
        self,
        name: str,
        result: GSplatData,
        parent: Optional[Node] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        fill_sigma: Optional[Dict[str, float]] = None,
        **attrs: Any,
    ) -> GSplats:
        """Write multi-LOD GSplatData with per-LOD subgroups."""
        try:
            scene = self._find_scene()
            d_data = result.ndim

            # Build per-LOD tuples, applying dim_order to each LOD
            lod_tuples: list[
                tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
            ] = []
            lod_stats_list: list[dict[str, Any]] = []

            for lod in result.additive_sublods:
                ctr_arr = lod.centers.copy()
                chol_arr = lod.cholesky_factors.copy()

                if dim_order is not None:
                    ctr_arr, extend_to_all = self._apply_dim_order_positions(
                        ctr_arr, scene, dim_order, fill, extend_to_all
                    )
                    chol_arr = self._apply_dim_order_cholesky(
                        chol_arr, d_data, scene, dim_order, fill_sigma
                    )

                scene._validate_data_dimensions(ctr_arr, name, data_type="centers")

                lod_tuples.append(
                    (
                        ctr_arr.astype(np.float32),
                        lod.amplitudes,
                        chol_arr,
                        lod.colors,
                    )
                )
                lod_stats_list.append(dict(lod.stats))

            n_splats = result.n_splats
            ndim = lod_tuples[0][0].shape[1]
            aprint(
                f"Adding multi-LOD gsplats node '{name}' with "
                f"{n_splats:,} splats in {ndim}D ({result.n_additive_sublods} LODs)."
            )

            final_extend_dims = scene._resolve_extend_to_all(
                extend_to_all, lod_tuples[0][0], "splats"
            )
            if final_extend_dims:
                attrs["extend_to_all"] = final_extend_dims
                aprint(f"  📡 Extending visibility across: {final_extend_dims}")

            colormap = attrs.get("colormap")
            if any(t[3] is not None for t in lod_tuples) and colormap is not None:
                raise ValueError(
                    "Cannot specify both 'colors' and 'colormap'. Use one or the other."
                )

            parent_node = parent or self
            writer = self._require_scene_writer(scene)
            path = f"{parent_node.path}/{name}" if parent_node.path else name

            metadata = writer.write_gsplats_multi_lod(  # type: ignore[attr-defined]
                path,
                lods=lod_tuples,
                lod_stats=lod_stats_list,
                **attrs,
            )

            if "colormap" in attrs:
                from ..colormaps.builtins import BUILTIN_COLORMAP_NAMES

                cm = attrs["colormap"]
                if not isinstance(cm, str) or (
                    isinstance(cm, str) and cm not in BUILTIN_COLORMAP_NAMES
                ):
                    attrs["colormap"] = "custom"

            if not metadata.get("has_colors") and "colormap" not in attrs:
                attrs["colormap"] = "gray"

            return GSplats(
                name,
                metadata=metadata,
                parent=cast(Any, parent_node),
                writer=writer,
                **attrs,
            )
        except (ValueError, TypeError) as e:
            aprint(f"Failed to add multi-LOD gsplats node '{name}': {e}")
            raise ValueError(f"Could not add gsplats '{name}': {e}") from e

    def add_gsplats_from_file(
        self,
        name: str,
        path: Union[str, Path],
        parent: Optional[Node] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        fill_sigma: Optional[Dict[str, float]] = None,
        **attrs: Any,
    ) -> Union[GSplats, "Group"]:
        """Add Gaussian splats by loading from a .gsplats.zarr file.

        If the source file carries multiple substitutive levels, only the
        default substitutive level's additive ladder is written into the
        scene; other substitutive levels are dropped (see
        ``add_gsplats_from_data`` for the convention).

        Args:
            name: Name of the gsplats node
            path: Path to .gsplats.zarr file
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map data columns to scene dimensions by name
            fill: Fixed coordinate values for unmapped dimensions
            fill_sigma: Standard deviations for unmapped dims in Cholesky embedding
            **attrs: Additional node attributes
        """
        from pathlib import Path

        from luxar.gsplats.io.load_gsplats import load_gsplats

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GSplats file not found: {path}")

        result = load_gsplats(path)

        return self.add_gsplats_from_data(
            name=name,
            result=result,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            **attrs,
        )

    def add_gsplats_from_volume(
        self,
        name: str,
        volume: np.ndarray,
        seeds: Optional[Union[int, float]] = None,
        n_iters: int = 1000,
        device: Optional[str] = None,
        progressive: bool = False,
        max_splats_per_pass: int = 5000,
        psnr_patience: float = 0.5,
        max_passes: Optional[int] = None,
        parent: Optional[Node] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        fill_sigma: Optional[Dict[str, float]] = None,
        opacity: Optional[float] = None,
        blending_mode: Optional[str] = None,
        **fit_kwargs: Any,
    ) -> Union[GSplats, "Group"]:
        """Fit Gaussian splats to a volume and add them in one step.

        Args:
            name: Name of the gsplats node
            volume: Input n-dimensional volume to fit
            seeds: Number of splats (int), compression ratio (float), or None.
                In progressive mode, this is the total max splats budget.
            n_iters: Optimization iterations (default: 1000).
                In progressive mode, this is iterations per pass.
            device: Compute device ("cuda", "mps", "cpu", or None for auto)
            progressive: Use progressive multi-pass fitting (produces multi-LOD)
            max_splats_per_pass: Max splats per progressive pass (default: 5000)
            psnr_patience: Stop progressive fitting if PSNR gain < this (dB)
            max_passes: Max number of progressive passes (None = unlimited)
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map fitted data columns to scene dimensions by name
            fill: Fixed coordinate values for unmapped dimensions
            fill_sigma: Standard deviations for unmapped dims in Cholesky embedding
            opacity: Node opacity (0.0-1.0)
            blending_mode: Blending mode ("normal", "additive", "max",
                "opaque", "luminous")
            **fit_kwargs: Extra kwargs for fitting function
        """
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

        scene_attrs: dict[str, Any] = {}
        if opacity is not None:
            scene_attrs["opacity"] = opacity
        if blending_mode is not None:
            scene_attrs["blending_mode"] = blending_mode

        return self.add_gsplats_from_data(
            name=name,
            result=result,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            **scene_attrs,
        )
