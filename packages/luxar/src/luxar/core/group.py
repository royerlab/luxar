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
    from .lod_group import LODGroup

# Default radius used when radii are not provided
DEFAULT_POINT_RADIUS = 0.5


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
        **attrs: Any,
    ) -> Points:
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
            **attrs: Additional node attributes. Common ones:

                - ``layer`` (bool): Expose this node in the viewer's Layers
                  panel for per-node control.
                - ``visible`` (bool): Initial visibility when scene loads
                  (default ``True``). Used by the Layers panel to start a
                  layer hidden.
                - ``opacity``, ``intensity``, ``gamma``, ``blending_mode``,
                  ``colormap``: standard rendering attributes.

        Returns:
            The created Points node
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
        **attrs: Any,
    ) -> Lines:
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
        **attrs: Any,
    ) -> GSplats:
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
            **attrs: Additional node attributes. Common ones:

                - ``layer`` (bool): Expose this node in the viewer's Layers
                  panel for per-node control.
                - ``visible`` (bool): Initial visibility when scene loads
                  (default ``True``).
                - ``opacity``, ``intensity``, ``gamma``, ``blending_mode``,
                  ``colormap``: standard rendering attributes.

        Returns:
            The created GSplats node
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
    ) -> Union[GSplats, "LODGroup"]:
        """Add Gaussian splats from a GSplatData object.

        Multi-additive-LOD data (from progressive fitting or
        ``make_additive_lod``) is written with per-sub-LOD subgroups
        directly under the gsplats node (``<node>/additive_<i>/...``)
        for progressive (prefix-sum) loading. Single-LOD data uses the
        flat layout (arrays at the node path).

        ``lod_group`` and ``additive_lod`` control the two LOD axes (see
        ``luxar.core.lod_group.resolve_substitutive_axis`` /
        ``resolve_additive_axis`` for the full value vocabulary). When
        the resolved data has multiple substitutive levels, this method
        builds an :class:`LODGroup` containing one gsplats child per
        level (in coarsest→finest order, named ``child_<i>``) and
        returns it; otherwise it returns a single :class:`GSplats` node.

        ``min_pixel_size`` may only be passed in ``**attrs`` when the
        result is single-substitutive AND the parent is an
        :class:`LODGroup` (the child is itself a leaf of an enclosing
        LODGroup). Passing it on a multi-substitutive path raises
        ``ValueError`` — use ``lod_group=dict(min_pixel_sizes=[...])`` to
        override the auto-derived thresholds.

        Args:
            name: Name of the gsplats (or lod_group) node.
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
            >>> # Auto-build a 3-level LODGroup with a 4-step additive ladder
            >>> # per level, computed from a flat input.
            >>> scene.add_gsplats_from_data(
            ...     "multires", flat_result,
            ...     lod_group=dict(compression_factor=4, levels=2),
            ...     additive_lod=dict(n_lods=4),
            ... )
        """
        from luxar.gsplats.gsplat_data import GSplatData

        from .lod_group import (
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

        # Multi-substitutive → LODGroup with one gsplats child per level
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
    ) -> "LODGroup":
        """Build an LODGroup with one gsplats child per substitutive level.

        Children are written in coarsest→finest order and named
        ``child_<i>``. Compositing attrs (opacity, gamma, intensity,
        offset, blending_mode, transform, layer, visible, colormap) land
        on the LODGroup itself; per-leaf gsplats attrs (truncation_radius,
        extend_to_all) ride into each child.
        """
        from .lod_group import derive_min_pixel_sizes

        # Substitutive convention: index 0 = finest, n-1 = coarsest. The
        # LODGroup needs coarsest first.
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

        # Separate compositing attrs (go on the LODGroup) from per-leaf
        # gsplats attrs (go on each child). Anything not in the
        # compositing set falls through to the child level.
        #
        # `colormap` is intentionally NOT compositing here: the writer
        # auto-defaults a missing colormap to "gray" per leaf, which
        # under nearest-ancestor-wins would shadow a parent's setting.
        # Keep it on each child so the user's intent survives.
        COMPOSITING_ATTRS = {
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
        lod_attrs = {k: v for k, v in attrs.items() if k in COMPOSITING_ATTRS}
        child_attrs = {k: v for k, v in attrs.items() if k not in COMPOSITING_ATTRS}
        # Defense in depth: ``add_gsplats_from_data`` rejects min_pixel_size
        # at entry, but if this method is invoked through a different path
        # (e.g. internal recursion) we still need to strip it — the loop
        # below passes a derived value as an explicit kwarg, and a duplicate
        # in ``**child_attrs`` would raise TypeError.
        child_attrs.pop("min_pixel_size", None)

        parent_node = parent or self
        # Note: cast to LODGroup is needed because add_lod_group is defined
        # on Node, which is fine for any parent.
        lod_group_node: LODGroup = parent_node.add_lod_group(name, **lod_attrs)

        aprint(
            f"Adding multi-resolution gsplats node '{name}' as LODGroup "
            f"with {n_sub} substitutive levels: "
            f"{[f'{n:,}' for n in splat_counts]} splats"
        )

        # Add one gsplats child per substitutive level, coarsest first.
        for child_idx, s in enumerate(order):
            child_name = f"child_{child_idx}"
            level_view = result.at_substitutive(s)
            # Recursive dispatch — but explicitly None on both LOD axes so
            # the resolvers no-op and we never re-enter the LODGroup branch.
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
    ) -> Union[GSplats, "LODGroup"]:
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
    ) -> Union[GSplats, "LODGroup"]:
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
