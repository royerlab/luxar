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
)

import numpy as np

from ...typing_utils.aliases import ColorArray, PositionArray
from ..gsplats import GSplats
from ..lines import Lines
from ..node import Node
from ..points import Points

if TYPE_CHECKING:
    from ...gsplats.gsplat_data import GSplatData
    from ...io.writer import ZarrWriterProtocol
    from ..scene import Scene

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
        from ..scene import Scene

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
        from .adders.points import add_points_impl

        return add_points_impl(
            self,
            name=name,
            positions=positions,
            colors=colors,
            radii=radii,
            sharpness=sharpness,
            scalars=scalars,
            labels=labels,
            image_labels=image_labels,
            parent=parent,
            extend_to_all=extend_to_all,
            grid_shape=grid_shape,
            dim_order=dim_order,
            fill=fill,
            split=split,
            additive_lod=additive_lod,
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
        from .adders.lines import add_lines_impl

        return add_lines_impl(
            self,
            name=name,
            vertices=vertices,
            widths=widths,
            colors=colors,
            sharpness=sharpness,
            scalars=scalars,
            labels=labels,
            image_labels=image_labels,
            indices=indices,
            line_type=line_type,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            additive_lod=additive_lod,
            split=split,
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
        from .adders.gsplats import add_gsplats_impl

        return add_gsplats_impl(
            self,
            name=name,
            centers=centers,
            amplitudes=amplitudes,
            cholesky_factors=cholesky_factors,
            colors=colors,
            labels=labels,
            image_labels=image_labels,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            split=split,
            **attrs,
        )


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
        from .gsplats_pipeline.from_data import add_gsplats_from_data_impl

        return add_gsplats_from_data_impl(
            self,
            name=name,
            result=result,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            lod_group=lod_group,
            additive_lod=additive_lod,
            **attrs,
        )

    # --- BEGIN OLD GSPLATS BODY ---
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
        from .gsplats_pipeline.from_io import add_gsplats_from_file_impl

        return add_gsplats_from_file_impl(
            self,
            name=name,
            path=path,
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
        from .gsplats_pipeline.from_io import add_gsplats_from_volume_impl

        return add_gsplats_from_volume_impl(
            self,
            name=name,
            volume=volume,
            seeds=seeds,
            n_iters=n_iters,
            device=device,
            progressive=progressive,
            max_splats_per_pass=max_splats_per_pass,
            psnr_patience=psnr_patience,
            max_passes=max_passes,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            fill_sigma=fill_sigma,
            opacity=opacity,
            blending_mode=blending_mode,
            **fit_kwargs,
        )
