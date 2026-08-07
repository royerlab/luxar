"""luxar.group – Group node with data-adding methods for scene graphs.

A Group can contain child data nodes (Points, Lines, GSplats, Mesh) and other
Groups, forming a hierarchical scene structure. Groups walk up the parent chain
to find the root Scene for dimension validation and writer access.
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
    Union,
)

import numpy as np

from ...typing_utils.aliases import ColorArray, PositionArray
from ..gsplats import GSplats
from ..lines import Lines
from ..mesh import Mesh
from ..node import Node
from ..points import Points

if TYPE_CHECKING:
    from ...gsplats.gsplat_data import GSplatData
    from ...io.writer import ZarrWriterProtocol
    from ..scene import Scene


class Group(Node):
    """A group node that can contain data children (Points, Lines, GSplats, Mesh).

    Groups provide add_points(), add_lines(), add_gsplats(), and add_mesh()
    methods for adding data nodes. They access the root Scene for dimension validation
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
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        partition: Any = None,
        additive_lod: Any = None,
        substitutive_lod: Any = None,
        **attrs: Any,
    ) -> Union[Points, "Group"]:
        """Add a points node.

        Args:
            name: Name of the points node
            positions: Array of shape (N, D) for point positions
            colors: Optional (N, 3) RGB or (N, 4) RGBA array (the alpha
                column is per-point opacity in [0, 1]), RGB tuple, or None
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
            dim_order: Map data columns to scene dimensions by name.
                E.g., ``["Y", "X"]`` for 2D data in a 3D scene.
                Unmapped dims are filled with ``fill`` values and auto-extended.
            fill: Fixed coordinate values for unmapped scene dimensions
                when using ``dim_order``. Defaults to 0.0 for unspecified dims.
            substitutive_lod: Substitutive-LOD control. ``None`` (default) /
                ``False`` write no substitutive ladder. ``True`` / ``dict()``
                synthesise coarse LOD levels as Gaussian splats: each point is
                lifted to an isotropic Gaussian and reduced by the gsplat
                substitutive pipeline (mass-preserving), assembled as a
                ``kind="lod"`` Group whose finest child is the original Points
                node. ``dict(...)`` keys: ``compression_factor`` (``K``),
                ``levels`` (``n_lods``), ``method``, ``truncation_radius``,
                ``device``, ``seed``, ``coverage_fractions``, ``coarsen_dims``,
                ``max_aspect`` (anisotropy cap on the coarse levels, default
                3.0; ``None`` disables).
                Composes with ``additive_lod``, which then describes how
                each level streams in (every level gets a streaming ladder by
                default; pass ``additive_lod=False`` to opt out). Mutually
                exclusive with ``partition``.
                ``scalars``+``colormap``
                points are supported by baking scalars→RGB for the coarse gsplat
                levels (the finest Points child stays scalar-driven; a live
                colormap change re-colours only the finest level). See
                :func:`luxar.core.group.lod.points.resolve_substitutive_axis_points`.
            partition: Spatial-decomposition control. ``None`` (default) writes
                a single Points node. ``True`` decomposes via balanced median
                BSP with ``max_elements = DEFAULT_MAX_ELEMENTS``.
                ``dict(max_elements=N, rule=...)`` uses an explicit cap and
                rule (``"median"`` default, ``"midpoint"``, or ``"sah"``).
                When the decomposition yields more than one part, returns a
                kind=partition ``Group`` wrapper carrying ``display_type=
                "points"``; the wrapper's children are ``part_<i>`` Points
                nodes. The wrapper's ``position_bounds`` is the union of
                the children's so picking treats the layer as one entity.
                ``image_labels`` is not supported alongside ``partition=``
                (the sparse-dict semantics complicate slicing).
            **attrs: Additional node attributes. Common ones:

                - ``layer`` (bool): Expose this node in the viewer's Layers
                  panel for per-node control. When ``partition=`` produces a
                  wrapper, ``layer=True`` lands on the wrapper, not on
                  each leaf part.
                - ``visible`` (bool): Initial visibility when scene loads
                  (default ``True``). Used by the Layers panel to start a
                  layer hidden.
                - ``opacity``, ``intensity``, ``gamma``, ``blending_mode``,
                  ``colormap``: standard rendering attributes.
                - ``absorption`` (float >= 0): absorption coefficient kappa,
                  read by the ``"volumetric"`` blending mode; kappa=0 renders
                  like additive. Defaults to 1.0.

        Returns:
            The created ``Points`` node, or a kind=partition ``Group``
            wrapper when ``partition=`` produced more than one part.
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
            dim_order=dim_order,
            fill=fill,
            partition=partition,
            additive_lod=additive_lod,
            substitutive_lod=substitutive_lod,
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
        substitutive_lod: Any = None,
        partition: Any = None,
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
            indices: Vertex-index pairs for ``line_type="indexed"``, as a flat
                even-element ``(2E,)`` array or an ``(E, 2)`` pair array. Connected
                edges must reference the same vertex row for joint continuity;
                duplicated rows at equal coordinates remain independent endpoints.
            line_type: Connectivity (``"segments"``, ``"polyline"``, ``"loop"``,
                or ``"indexed"``). Use ``polyline`` for one continuous chain and
                ``indexed`` for multiple chains or graph topology with shared joints.
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map data columns to scene dimensions by name
            fill: Fixed values for unmapped dimensions when using dim_order
            substitutive_lod: Substitutive-LOD control (peer of Points'). ``None``
                /``False`` write no substitutive ladder. ``True``/``dict()``
                synthesise coarse LOD levels as Gaussian splats: each segment is
                lifted to isotropic "bead" gaussians and reduced by the gsplat
                substitutive pipeline, assembled as a ``kind="lod"`` Group whose
                finest child is the original Lines node. Same dict vocabulary as
                Points; ``scalars``+``colormap`` are baked for the coarse levels.
                Composes with ``additive_lod`` (which then describes how each
                level streams in; every level is laddered by default, pass
                ``additive_lod=False`` to opt out). Mutually exclusive with
                ``partition``. See
                :func:`luxar.core.group.lod.lines.resolve_substitutive_axis_lines`.
            **attrs: Additional node attributes. Common ones:

                - ``layer`` (bool): Expose this node in the viewer's Layers
                  panel for per-node control.
                - ``visible`` (bool): Initial visibility when scene loads
                  (default ``True``).
                - ``opacity``, ``intensity``, ``gamma``, ``blending_mode``,
                  ``colormap``: standard rendering attributes.
                - ``absorption`` (float >= 0): absorption coefficient kappa,
                  read by the ``"volumetric"`` blending mode; kappa=0 renders
                  like additive. Defaults to 1.0.

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
            substitutive_lod=substitutive_lod,
            partition=partition,
            **attrs,
        )

    def add_mesh(
        self,
        name: str,
        vertices: "np.ndarray",
        faces: "np.ndarray",
        normals: Optional["np.ndarray"] = None,
        normal_dims: Optional[Sequence[int]] = None,
        colors: Optional[Any] = None,
        scalars: Optional[Any] = None,
        *,
        shading: Optional[str] = None,
        double_sided: bool = True,
        labels: Optional[Sequence[str]] = None,
        image_labels: Optional[Any] = None,
        parent: Optional["Node"] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        substitutive_lod: Optional[Union[bool, Dict[str, Any]]] = None,
        **attrs: Any,
    ) -> Union["Mesh", "Group"]:
        """Add a triangle mesh (surface) node to this group.

        The surface geometry type: nD ``vertices`` plus a ``faces`` triangle-index
        array. Unlike Points / Lines / GSplats, a mesh has no per-element size —
        a triangle's extent comes from its own vertices — so it also contributes
        no extent padding to the scene's bounds.

        ``substitutive_lod`` IS supported: it writes a ``kind=lod`` group whose
        coarse children are progressively DECIMATED copies of the surface and whose
        finest child is the original. Its vocabulary is shorter than the sibling
        adders' — no ``truncation_radius`` / ``max_aspect`` / ``device`` / ``seed``,
        because those exist only for geometries that coarsen by lifting to gsplats,
        and ``method`` is ``{'auto', 'cluster'}`` rather than the Gaussian-mixture
        reducers. See :func:`luxar.core.group.lod.mesh.resolve_substitutive_axis_mesh`.

        Not supported for meshes (each raises rather than silently degrading):
        ``additive_lod`` (a prefix of an index buffer is a surface with holes, not
        a coarser surface), ``partition`` (a BSP cut needs boundary vertices
        duplicated per part and the label CSR split to match), and adding one
        under a ``kind=partition`` parent. See ``docs/specs/MESH_NODE_SPEC.md`` §9.

        Args:
            name: Name of the mesh node.
            vertices: Vertex positions of shape ``(V, D)``.
            faces: Triangle vertex indices, ``(F, 3)`` or flat ``(3F,)``. Wound
                counter-clockwise as seen with the authored spatial triple in
                ascending index order.
            normals: Optional per-vertex normals of shape ``(V, 3)``. Requires
                ``normal_dims``.
            normal_dims: The three dimension indices ``normals`` describes.
                Required with ``normals`` and rejected without them — it is not
                inferable, and an implicit "first three dimensions" is wrong for
                any mesh whose leading dimension is not spatial (for a
                ``(t, x, y, z)`` mesh those are ``(t, x, y)``). Passing it through
                ``**attrs`` fails: it is writer-reserved metadata.
            colors: Per-vertex colors ``(V, 3|4)``, a broadcast RGB(A) tuple/list,
                or None. A 4th component is per-vertex opacity.
            scalars: Per-vertex scalars ``(V,)`` or a single value for colormap
                lookup. Requires a ``colormap`` attr.
            shading: ``"smooth"`` or ``"flat"``. Defaults to ``"smooth"`` when
                ``normals`` are given, else ``"flat"``. An explicit value is
                stored as given — ``"flat"`` renders faceted even with normals
                present, and ``"smooth"`` without normals falls back to derived
                flat normals at render time.
            double_sided: Whether back faces render (default ``True``).
            labels: Optional per-vertex strings for hover tooltips.
            image_labels: Optional per-vertex images for hover thumbnails.
            parent: Optional explicit parent node (defaults to this group).
            extend_to_all: Dimension name(s) across which this mesh stays visible.
            dim_order: Names of the dimensions the ``vertices`` columns are in,
                for remapping onto the scene's dimension order. ``faces`` is index
                data addressing vertex rows and is never reordered.
            fill: Fill values for scene dimensions absent from ``dim_order``.
            **attrs: Additional attributes — ``opacity``, ``intensity``,
                ``offset``, ``gamma``, ``colormap``, ``layer``, ``visible``,
                ``transform``, ``nd_transform``, ``blending_mode``. Note
                ``volumetric`` blending is rejected — it has no meaning for an
                opaque surface.

        Returns:
            The created Mesh node — or, with ``substitutive_lod``, the ``kind=lod``
            Group wrapping the ladder (matching ``add_points`` / ``add_lines``).
        """
        from .adders.mesh import add_mesh_impl

        return add_mesh_impl(
            self,
            name=name,
            vertices=vertices,
            faces=faces,
            normals=normals,
            normal_dims=normal_dims,
            colors=colors,
            scalars=scalars,
            shading=shading,
            double_sided=double_sided,
            labels=labels,
            image_labels=image_labels,
            parent=parent,
            extend_to_all=extend_to_all,
            dim_order=dim_order,
            fill=fill,
            substitutive_lod=substitutive_lod,
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
        partition: Any = None,
        **attrs: Any,
    ) -> Union[GSplats, "Group"]:
        """Add a Gaussian splats node.

        Args:
            name: Name of the gsplats node
            centers: Array of shape (N, D) for splat centers
            amplitudes: (N,) array or scalar for intensities
            cholesky_factors: (N, k) packed Cholesky factors, k=D*(D+1)/2
            colors: Optional (N, 3) RGB or (N, 4) RGBA array (the alpha
                column is per-splat opacity in [0, 1]), RGB tuple, or None
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
            partition: Spatial-decomposition control. ``None`` (default) writes
                a single GSplats node. ``True`` decomposes via balanced median
                BSP with ``max_elements = DEFAULT_MAX_ELEMENTS``.
                ``dict(max_elements=N, rule=...)`` uses an explicit cap and
                rule (``"median"`` default, ``"midpoint"``, or ``"sah"``).
                When the decomposition yields more than one part, returns a
                kind=partition ``Group`` wrapper carrying ``display_type=
                "gsplats"``; the wrapper's children are ``part_<i>``
                GSplats nodes. ``image_labels`` is not supported alongside
                ``partition=``.
            **attrs: Additional node attributes. Common ones:

                - ``layer`` (bool): Expose this node in the viewer's Layers
                  panel for per-node control. When ``partition=`` produces a
                  wrapper, ``layer=True`` lands on the wrapper, not on
                  each leaf part.
                - ``visible`` (bool): Initial visibility when scene loads
                  (default ``True``).
                - ``opacity``, ``intensity``, ``gamma``, ``blending_mode``,
                  ``colormap``: standard rendering attributes.
                - ``absorption`` (float >= 0): absorption coefficient kappa,
                  read by the ``"volumetric"`` blending mode; kappa=0 renders
                  like additive. Defaults to 1.0. Like ``layer``, on a
                  ``partition=`` wrapper this lands on the wrapper node, not
                  on each leaf part.

        Returns:
            The created ``GSplats`` node, or a kind=partition ``Group``
            wrapper when ``partition=`` produced more than one part.
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
            partition=partition,
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
        ``luxar.core.group.lod.gsplats.resolve_substitutive_axis_gsplats`` /
        ``resolve_additive_axis_gsplats`` for the full value vocabulary). When
        the resolved data has multiple substitutive levels, this method
        builds a ``kind="lod"`` ``Group`` containing one gsplats child
        per level (in coarsest→finest order, named ``child_<i>``) and
        returns it; otherwise it returns a single :class:`GSplats` node.

        ``coverage_fraction`` may only be passed in ``**attrs`` when the
        result is single-substitutive AND the parent is itself a
        ``kind="lod"`` ``Group`` (the child is a leaf of an enclosing
        LOD group). Passing it on a multi-substitutive path raises
        ``ValueError`` — use ``lod_group=dict(coverage_fractions=[...])`` to
        override the auto-derived thresholds.

        Args:
            name: Name of the gsplats (or kind=lod group) node.
            result: GSplatData from ``fit_gaussian_splats`` or similar.
            parent: Parent node (default: this group).
            extend_to_all: Visibility extension across non-displayed dimensions.
            dim_order: Map data columns to scene dimensions by name.
            fill: Fixed coordinate values for unmapped dimensions.
            fill_sigma: Standard deviations for unmapped dims in Cholesky embedding.
            lod_group: Substitutive-axis control. ``None`` (default; auto-lower
                a multi-substitutive pyramid into a ``kind=lod`` Group),
                ``True`` (require stored levels), ``False`` (collapse to
                finest), ``dict(...)`` (compute
                via :func:`make_substitutive_lod`), or ``dict(..., recompute=
                True)``. Optional ``coverage_fractions=[...]`` inside the dict
                overrides the auto-derived thresholds.
            additive_lod: Additive-axis control, uniform across substitutive
                levels. Same value vocabulary as ``lod_group``; ``dict(...)``
                routes to :func:`make_additive_lod`.
            **attrs: Additional node attributes (same vocabulary as
                :meth:`add_gsplats`, including ``absorption``). On a nested
                ``kind=lod`` tree, compositing attributes (e.g. ``absorption``)
                land on the wrapper node while the rest (e.g. ``colormap``)
                are copied onto each leaf.

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

        If the source file carries multiple substitutive levels, the
        pyramid is auto-lowered into a ``kind=lod`` Group (one gsplats
        child per substitutive level); pass ``lod_group=False`` to
        collapse to the finest level instead (see
        ``add_gsplats_from_data`` for the convention).

        Args:
            name: Name of the gsplats node
            path: Path to .gsplats.zarr file
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map data columns to scene dimensions by name
            fill: Fixed coordinate values for unmapped dimensions
            fill_sigma: Standard deviations for unmapped dims in Cholesky embedding
            **attrs: Additional node attributes (same vocabulary as
                :meth:`add_gsplats`, including ``absorption``). On a nested
                tree, compositing attributes (``blending_mode``, ``absorption``,
                ``opacity``, ...) land on the wrapper node ONLY — the viewer
                resolves them down the ancestry — while the rest (e.g.
                ``colormap``) are copied onto each leaf. Stamping
                ``blending_mode`` on the parts too would SHADOW the wrapper
                (it is nearest-setter-wins), leaving the layer's Blend control
                inert.
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
        absorption: Optional[float] = None,
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
            absorption: Absorption coefficient kappa (>= 0) read by the
                "volumetric" blending mode; kappa=0 renders like additive
            blending_mode: Blending mode ("normal", "additive", "max",
                "opaque", "luminous", "volumetric")
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
            absorption=absorption,
            blending_mode=blending_mode,
            **fit_kwargs,
        )
