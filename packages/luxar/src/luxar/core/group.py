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
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            grid_shape: Optional grid shape for structured data
            dim_order: Map data columns to scene dimensions by name.
                E.g., ``["Y", "X"]`` for 2D data in a 3D scene.
                Unmapped dims are filled with ``fill`` values and auto-extended.
            fill: Fixed coordinate values for unmapped scene dimensions
                when using ``dim_order``. Defaults to 0.0 for unspecified dims.
            **attrs: Additional node attributes (opacity, blending_mode, etc.)

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

            if radii is None:
                radii = DEFAULT_POINT_RADIUS
                aprint(f"  📐 Using default radius: {DEFAULT_POINT_RADIUS}")

            writer = scene._writer
            assert writer is not None, "Scene writer is not initialized"
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = writer.write_points(
                path,
                pos_arr.astype(np.float32),
                colors=cast(Any, colors),
                radii=radii,
                sharpness=sharpness,
                grid_shape=grid_shape,
                **attrs,
            )

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
            indices: Optional vertex indices for indexed line type
            line_type: Connectivity ("segments", "polyline", "loop", "indexed")
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map data columns to scene dimensions by name
            fill: Fixed values for unmapped dimensions when using dim_order
            **attrs: Additional node attributes

        Returns:
            The created Lines node
        """
        try:
            scene = self._find_scene()

            vert_arr: np.ndarray = (
                vertices
                if isinstance(vertices, np.ndarray)
                else np.asarray(vertices)
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

            parent_node = parent or self

            writer = scene._writer
            assert writer is not None, "Scene writer is not initialized"
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = writer.write_lines(
                path,
                vert_arr.astype(np.float32),
                widths=widths,
                colors=cast(Any, colors),
                sharpness=sharpness,
                indices=indices,
                line_type=line_type,
                **attrs,
            )

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
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map data columns to scene dimensions by name.
                Also reorders and embeds Cholesky factors automatically.
            fill: Fixed coordinate values for unmapped dimensions
            fill_sigma: Standard deviations for unmapped dimensions in the
                Cholesky embedding (default 1.0). Controls splat extent in
                unmapped dims.
            **attrs: Additional node attributes (opacity, blending_mode, etc.)

        Returns:
            The created GSplats node
        """
        try:
            scene = self._find_scene()

            ctr_arr: np.ndarray = (
                centers
                if isinstance(centers, np.ndarray)
                else np.asarray(centers)
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

            parent_node = parent or self

            writer = scene._writer
            assert writer is not None, "Scene writer is not initialized"
            path = f"{parent_node.path}/{name}" if parent_node.path else name
            metadata = writer.write_gsplats(
                path,
                ctr_arr.astype(np.float32),
                amplitudes=amplitudes,
                cholesky_factors=chol_arr,
                colors=cast(Any, colors),
                **attrs,
            )

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
        **attrs: Any,
    ) -> GSplats:
        """Add Gaussian splats from a GSplatData object.

        Args:
            name: Name of the gsplats node
            result: GSplatData from fit_gaussian_splats()
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map data columns to scene dimensions by name
            fill: Fixed coordinate values for unmapped dimensions
            fill_sigma: Standard deviations for unmapped dims in Cholesky embedding
            **attrs: Additional node attributes

        Example:
            >>> result = fit_gaussian_splats(volume_3d)
            >>> # Add 3D splats to a 4D scene with Time dimension
            >>> scene.add_gsplats_from_data("splats", result,
            ...     dim_order=["Z", "Y", "X"], fill={"Time": 0})
        """
        from luxar.gsplats.gsplat_data import GSplatData

        if not isinstance(result, GSplatData):
            raise TypeError(f"Expected GSplatData, got {type(result).__name__}")

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
    ) -> GSplats:
        """Add Gaussian splats by loading from a .gsplats.zarr file.

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
        parent: Optional[Node] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        fill_sigma: Optional[Dict[str, float]] = None,
        opacity: Optional[float] = None,
        blending_mode: Optional[str] = None,
        **fit_kwargs: Any,
    ) -> GSplats:
        """Fit Gaussian splats to a volume and add them in one step.

        Args:
            name: Name of the gsplats node
            volume: Input n-dimensional volume to fit
            seeds: Number of splats (int), compression ratio (float), or None
            n_iters: Optimization iterations (default: 1000)
            device: Compute device ("cuda", "mps", "cpu", or None for auto)
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map fitted data columns to scene dimensions by name
            fill: Fixed coordinate values for unmapped dimensions
            fill_sigma: Standard deviations for unmapped dims in Cholesky embedding
            opacity: Node opacity (0.0-1.0)
            blending_mode: Blending mode ("normal", "additive", "max")
            **fit_kwargs: Extra kwargs for fit_gaussian_splats()
        """
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
