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
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from ...typing_utils.aliases import ColorArray, PositionArray
from ..gsplats import GSplats
from ..lines import Lines
from ..mesh import Mesh
from ..node import Node
from ..points import Points
from ..sound import Sound

if TYPE_CHECKING:
    from ...gsplats.gsplat_data import GSplatData
    from ...io.writer import ZarrWriterProtocol
    from ..scene import Scene

TNode = TypeVar("TNode")
_DEFAULT_NORMALIZE_AMPLITUDES = object()
_UNSET_LOD = object()


def _resolve_normalize_amplitudes_default(
    parent: Node, normalize_amplitudes: Any
) -> Any:
    if normalize_amplitudes is not _DEFAULT_NORMALIZE_AMPLITUDES:
        return normalize_amplitudes
    return parent.attrs.get("kind") not in ("lod", "partition")


def _resolve_gsplat_substitutive_lod_alias(
    substitutive_lod: Any, lod_group: Any
) -> Any:
    substitutive_requested = (
        substitutive_lod is not _UNSET_LOD and substitutive_lod is not None
    )
    alias_requested = lod_group is not _UNSET_LOD and lod_group is not None
    if substitutive_requested and alias_requested:
        raise ValueError("Pass only one of substitutive_lod= and lod_group=")
    if substitutive_requested:
        return substitutive_lod
    if alias_requested:
        return lod_group
    return None


def _gsplat_data_from_arrays(
    centers: Any,
    amplitudes: Any,
    cholesky_factors: Any,
    colors: Any,
    label_ids: Any,
    label_vocabulary: Optional[Dict[int, str]],
) -> GSplatData:
    from ...gsplats.gsplat_data import GSplatData
    from ...validation.writing import validate_gsplat_inputs

    centers_array = np.asarray(centers)
    if centers_array.ndim != 2:
        raise ValueError(
            f"Centers must have shape (N, D), got shape {centers_array.shape}"
        )
    amplitudes_value: Any = (
        amplitudes if np.isscalar(amplitudes) else np.asarray(amplitudes)
    )
    cholesky_array = np.asarray(cholesky_factors)
    (
        centers_array,
        amplitudes_value,
        cholesky_array,
        colors_value,
        n_splats,
        _n_dims,
        cholesky_is_uniform,
    ) = validate_gsplat_inputs(
        centers_array,
        amplitudes_value,
        cholesky_array,
        colors,
    )
    amplitudes_array = cast(
        np.ndarray[Any, Any],
        np.full(n_splats, amplitudes_value, dtype=np.asarray(amplitudes_value).dtype)
        if np.isscalar(amplitudes_value)
        else amplitudes_value,
    )
    if cholesky_is_uniform:
        cholesky_array = np.broadcast_to(
            cholesky_array, (n_splats, cholesky_array.shape[1])
        ).copy()
    colors_array = None
    if colors_value is not None:
        colors_array = np.asarray(colors_value)
        if colors_array.ndim == 1:
            colors_array = np.broadcast_to(
                colors_array.astype(np.float32), (n_splats, colors_array.shape[0])
            )
        elif colors_array.shape[0] == 1 and n_splats != 1:
            colors_array = np.broadcast_to(
                colors_array, (n_splats, colors_array.shape[1])
            )
    return GSplatData(
        centers=centers_array,
        amplitudes=amplitudes_array,
        cholesky_factors=cholesky_array,
        colors=colors_array,
        label_ids=None if label_ids is None else np.asarray(label_ids),
        label_vocabulary=label_vocabulary,
    )


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

    def _transactional_add(
        self,
        name: str,
        parent: Optional[Node],
        build: Callable[[], TNode],
    ) -> TNode:
        """Run one add call with store, graph, and compiler-state rollback."""
        parent_node = parent or self
        writer = self._require_scene_writer(self._find_scene())
        path = f"{parent_node.path}/{name}" if parent_node.path else name
        children_before = list(parent_node.children)
        try:
            with writer.transaction(path):
                return build()
        except BaseException:
            parent_node.children[:] = children_before
            raise

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
        keys: Optional[Union[List[str], Sequence[str]]] = None,
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
            keys: Optional list of machine-readable strings, one per point,
                for ``link`` / ``copy`` templates to substitute as
                ``{hover_key}``. Same length rule and the same spatial
                reordering as ``labels`` — a key stays paired with its element
                — but kept separate so the visible label can stay readable
                prose while the URL is built from a bare id.
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map data columns to scene dimensions by name.
                E.g., ``["Y", "X"]`` for 2D data in a 3D scene.
                Unmapped dims are filled with ``fill`` values and auto-extended.
            fill: Fixed coordinate values for unmapped scene dimensions
                when using ``dim_order``. Defaults to 0.0 for unspecified dims.
            substitutive_lod: Substitutive-LOD control. ``None`` (default) /
                ``False`` write no substitutive ladder. ``True`` / ``dict()``
                use ``coarse="gsplats"`` (the default): each point is lifted to
                an isotropic Gaussian and reduced by the gsplat substitutive
                pipeline. ``coarse="points"`` instead writes spatially
                stratified subsamples as Points children, preserving the
                original radius; under effective additive/luminous blending
                their RGB values are scaled to preserve the finest level's
                summed point energy, widening to float32 HDR only when the gain
                is not 1. ``brightness_compensation="auto"`` selects that rule,
                while a numeric value applies that per reduction level (use
                ``1`` to disable it). Both forms assemble a ``kind="lod"`` Group
                whose finest child is the original Points node. ``dict(...)``
                keys: ``compression_factor`` (``K``), ``coarse``,
                ``brightness_compensation``,
                ``levels`` (``n_lods``), ``method``, ``truncation_radius``,
                ``device``, ``seed``, ``coverage_fractions``, ``coarsen_dims``,
                ``max_aspect`` (anisotropy cap on the coarse levels, default
                3.0; ``None`` disables), and ``quality_stamps`` (measure
                per-level quality, default ``True``).
                ``coarse="points"`` accepts ``method="subsample"`` or
                ``method="merge"``. Merge writes moment-matched representatives,
                preserves discrete hidden coordinates, uses quantized colour as
                a soft ordering preference, bakes scalar colormaps, and drops
                identity channels on coarse levels. Under additive/luminous
                blending it conserves per-bin light by reducing radius before
                increasing RGB, so SDR colours remain representable. It refuses
                numeric brightness compensation. ``truncation_radius``,
                ``device``, ``coarsen_dims``, and ``max_aspect`` remain refused.
                Integer ``coarsen_dims`` entries name the scene-ordered position
                columns after ``dim_order`` has been applied.
                For stacked nodes, every discrete hidden coordinate must fit in
                the coarsest point level; otherwise authoring raises.
                Composes with ``additive_lod``, which then describes how
                each level streams in (every level gets a streaming ladder by
                default; pass ``additive_lod=False`` to opt out). When combined
                with an explicit ``partition=``, authors an overview topology:
                global coarse levels above a spatially partitioned finest
                Points branch, selected only once it fills the viewport.
                ``scalars``+``colormap``
                points are supported by baking scalars→RGB where coarse-level
                brightness compensation is required (the finest Points child
                stays scalar-driven; a live colormap change then re-colours only
                the finest level). See
                :func:`luxar.core.group.lod.points.resolve_substitutive_axis_points`.
            partition: Spatial-decomposition control. ``None`` (default) writes
                a single Points node. ``True`` decomposes via balanced median
                BSP with ``max_elements = DEFAULT_MAX_ELEMENTS``.
                ``dict(max_elements=N, rule=...)`` uses an explicit cap and
                rule (``"median"`` default, ``"midpoint"``, or ``"sah"``).
                When the decomposition yields more than one part, returns a
                kind=partition ``Group`` wrapper carrying ``display_type=
                "points"``; the wrapper's children are ``part_<i>`` Points
                nodes. When ``substitutive_lod=`` is also set, that wrapper is
                instead the finest child of a kind=lod ``Group``. The partition
                wrapper's ``position_bounds`` is the union of the children's so
                picking treats the layer as one entity.
                ``image_labels`` is not supported alongside ``partition=``
                (the sparse-dict semantics complicate slicing).
            **attrs: Additional node attributes. Common ones:

                - ``layer`` (bool): Expose this node in the viewer's Layers
                  panel for per-node control. When structural options produce
                  wrappers, ``layer=True`` lands on the outermost wrapper, not
                  on a nested partition wrapper or each leaf part.
                - ``visible`` (bool): Initial visibility when scene loads
                  (default ``True``). Used by the Layers panel to start a
                  layer hidden.
                - ``opacity``, ``intensity``, ``gamma``, ``blending_mode``,
                  ``colormap``: standard rendering attributes. An explicit
                  ``None`` for ``colormap`` or ``coverage_fraction`` means
                  "absent" — identical to omitting the key — so
                  ``colormap=maybe_colormap`` is a safe call form. Every OTHER
                  render attr still refuses a ``None``.
                - ``absorption`` (float >= 0): absorption coefficient kappa,
                  read by the ``"volumetric"`` blending mode; kappa=0 renders
                  like additive. Defaults to 1.0.

        Returns:
            The created ``Points`` node, a kind=partition ``Group`` when
            ``partition=`` produces multiple parts, or a kind=lod ``Group``
            when ``substitutive_lod=`` produces coarse levels (with the
            partition as its finest child when both controls are combined).
        """
        from .adders.points import add_points_impl

        return self._transactional_add(
            name,
            parent,
            lambda: add_points_impl(
                self,
                name=name,
                positions=positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness,
                scalars=scalars,
                labels=labels,
                keys=keys,
                image_labels=image_labels,
                parent=parent,
                extend_to_all=extend_to_all,
                dim_order=dim_order,
                fill=fill,
                partition=partition,
                additive_lod=additive_lod,
                substitutive_lod=substitutive_lod,
                **attrs,
            ),
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
        keys: Optional[Union[List[str], Sequence[str]]] = None,
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
            keys: Optional list of machine-readable strings, one per vertex,
                for ``link`` / ``copy`` templates to substitute as
                ``{hover_key}``. Same length rule and the same spatial
                reordering as ``labels`` — a key stays paired with its element
                — but kept separate so the visible label can stay readable
                prose while the URL is built from a bare id.
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
                finest child is the original Lines node. ``coarse="lines"``
                writes either seeded whole-polyline subsamples or, with
                ``method="merge"``, equal-vertex-count centroid polylines with
                transverse moment-matched widths. Merge preserves hidden
                coordinates, uses orientation and colour as soft ordering
                preferences, bakes scalar colormaps, and drops identity channels
                on coarse levels. Additive/luminous levels conserve per-bin light
                through width up to the transition's pixel-floor cap, then
                through residual HDR colour, materialising a colour channel when
                needed. The
                coarsest level must represent every occupied discrete hidden
                coordinate.
                Each level carries ``level_stats.quality`` unless the spec sets
                ``quality_stamps=False``.
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
                  ``colormap``: standard rendering attributes. An explicit
                  ``None`` for ``colormap`` or ``coverage_fraction`` means
                  "absent" — identical to omitting the key — so
                  ``colormap=maybe_colormap`` is a safe call form. Every OTHER
                  render attr still refuses a ``None``.
                - ``absorption`` (float >= 0): absorption coefficient kappa,
                  read by the ``"volumetric"`` blending mode; kappa=0 renders
                  like additive. Defaults to 1.0.
                - ``join`` (str): join style at degree-2 polyline joints --
                  ``"miter"`` (the default) or ``"none"``. Without join
                  geometry a turn leaves an uncovered wedge on the outside of
                  the bend and a double-covered lens inside; ``"miter"``
                  rotates each quad's end edge onto the shared miter edge so
                  the two tile exactly. Gated in-shader by rendered width and
                  a miter limit, so ``"none"`` is rarely worth authoring. An
                  unrecognised value is rejected rather than silently treated
                  as ``"none"``.

        Returns:
            The created Lines node
        """
        from .adders.lines import add_lines_impl

        return self._transactional_add(
            name,
            parent,
            lambda: add_lines_impl(
                self,
                name=name,
                vertices=vertices,
                widths=widths,
                colors=colors,
                sharpness=sharpness,
                scalars=scalars,
                labels=labels,
                keys=keys,
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
            ),
        )

    def add_sound(
        self,
        name: str,
        clip: Union[bytes, bytearray, str, Path],
        *,
        positions: Optional["np.ndarray"] = None,
        hidden: Optional[Mapping[str, float]] = None,
        spatial: Optional[bool] = None,
        trigger: str = "continuous",
        delay_ms: float = 0.0,
        gain: float = 1.0,
        bus: str = "ambient",
        fade_in_ms: float = 0.0,
        fade_out_ms: float = 0.0,
        distance_model: str = "inverse",
        ref_distance: Optional[float] = None,
        max_distance: Optional[float] = None,
        rolloff: Optional[float] = None,
        cone_inner_deg: Optional[float] = None,
        cone_outer_deg: Optional[float] = None,
        cone_outer_gain: Optional[float] = None,
        orientation: Optional[Sequence[float]] = None,
        attach_to: Optional[str] = None,
        ambisonic: Optional[str] = None,
        license: str = "",
        attribution: str = "",
        source_url: str = "",
        parent: Optional["Node"] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        **attrs: Any,
    ) -> "Sound":
        """Add a sound node — an MP3/AAC clip that plays in the viewer.

        The one node type that is *heard* rather than drawn
        (``docs/guides/specs/SOUND_SPEC.md``). Four placements:

        * **Everywhere** — ``positions=None, hidden=None``: an ambient bed that
          plays whatever the sliders say.
        * **Bound to a hidden-dimension value** — ``hidden={"story": 3}``: sugar
          for one ``(1, ndim)`` row at ``story=3``, extended over every other
          non-displayed dimension, so the slab rule that decides which points
          are visible decides when this clip is live. Non-spatial.
        * **Spatial** — ``positions=[[3, 7.3, -7.4, -0.3]]``: one nD row per
          place the source exists; the clip plays through a panner there and
          gets louder as the camera approaches. ``spatial`` defaults to True.
        * **Attached** — ``attach_to="cluster_hsp70"``: the source follows the
          bounding-box centre of the named node ("the cluster hums" without
          authoring coordinates). Spatial by default; combine with ``hidden=``
          to make it live at one hidden-dimension value only.

        Args:
            name: Node name (no ``/``).
            clip: Encoded MP3 or AAC (``.m4a``) bytes, or a path to such a file.
                Ogg/Opus is refused (Safari cannot decode it); WAV/FLAC are
                refused (wrong size class for a hosted store).
            positions: ``(K, ndim)`` source positions, or ``None``.
            hidden: ``{dimension_name: value}`` binding for a non-spatial clip.
                Mutually exclusive with ``positions``.
            spatial: Route through a panner (needs ``positions`` or
                ``attach_to``). Defaults to ``positions is not None or
                attach_to is not None``.
            trigger: ``"continuous"`` (looped while audible, fades on the slab
                edge), ``"once"`` (plays once each time the node becomes
                audible), ``"on_depart"`` / ``"on_arrive"`` (plays once when a
                story flight leaves / lands on the ``viewer_config.waypoints``
                entry whose ``when`` clause this node's row satisfies — a node
                without rows belongs to every waypoint).
            attach_to: Name of the node whose bounding-box centre the source
                follows. Mutually exclusive with ``positions``.
            ambisonic: ``"foa"`` for a first-order ambisonic FIELD — a
                4-channel AmbiX clip (AAC only) the viewer rotates against the
                camera so the field stays fixed to the world. Non-spatial and
                position-free by nature; ``hidden=`` still decides when it is
                live.
            delay_ms: Delay after the trigger fires, ``>= 0``.
            gain: Per-node linear gain, ``>= 0``.
            bus: ``"ambient"`` (default) / ``"voice"`` / ``"effects"``. The
                voice bus ducks ambient while it plays.
            fade_in_ms, fade_out_ms: Ramp lengths on the audible edge, ``>= 0``.
            distance_model, ref_distance, max_distance, rolloff, cone_inner_deg,
                cone_outer_deg, cone_outer_gain, orientation: ``PannerNode``
                knobs (spatial only). Distances left ``None`` default in the
                viewer from the scene scale (``scale/20`` and ``scale``).
            license, attribution, source_url: REQUIRED provenance for the clip
                (e.g. ``"CC0"``, the author, the URL it came from).
            parent: Parent node (default: this group).
            extend_to_all: As for ``add_points``; with ``hidden=`` it defaults
                to every non-displayed dimension not named there.
            **attrs: ``layer`` / ``visible`` / ``transform`` / ``nd_transform``
                only — a sound has no appearance attrs.

        Returns:
            The created Sound node.
        """
        from .adders.sound import add_sound_impl

        return self._transactional_add(
            name,
            parent,
            lambda: add_sound_impl(
                self,
                name=name,
                clip=clip,
                positions=positions,
                hidden=hidden,
                spatial=spatial,
                trigger=trigger,
                delay_ms=delay_ms,
                gain=gain,
                bus=bus,
                fade_in_ms=fade_in_ms,
                fade_out_ms=fade_out_ms,
                distance_model=distance_model,
                ref_distance=ref_distance,
                max_distance=max_distance,
                rolloff=rolloff,
                cone_inner_deg=cone_inner_deg,
                cone_outer_deg=cone_outer_deg,
                cone_outer_gain=cone_outer_gain,
                orientation=orientation,
                attach_to=attach_to,
                ambisonic=ambisonic,
                license=license,
                attribution=attribution,
                source_url=source_url,
                parent=parent,
                extend_to_all=extend_to_all,
                attrs=attrs,
            ),
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
        uvs: Optional["np.ndarray"] = None,
        texture: Optional[Any] = None,
        *,
        texture_encoding: str = "raw",
        texture_width: Optional[int] = None,
        texture_height: Optional[int] = None,
        texture_channels: Optional[int] = None,
        texture_color_space: str = "srgb",
        texture_ktx2_mode: str = "uastc",
        texture_ktx2_quality: Optional[int] = None,
        texture_ktx2_rdo_l: Optional[float] = None,
        texture_ktx2_zcmp: Optional[int] = None,
        shading: Optional[str] = None,
        double_sided: bool = True,
        labels: Optional[Sequence[str]] = None,
        image_labels: Optional[Any] = None,
        keys: Optional[Union[List[str], Sequence[str]]] = None,
        partition: Any = None,
        parent: Optional["Node"] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        substitutive_lod: Optional[Union[bool, Dict[str, Any]]] = None,
        additive_lod: Optional[Union[bool, Dict[str, Any]]] = None,
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
        and ``method`` is ``{'auto', 'cluster', 'qem'}`` rather than the Gaussian-mixture
        reducers. See :func:`luxar.core.group.lod.mesh.resolve_substitutive_axis_mesh`.

        ``partition`` IS supported too. Returns the ``kind=partition`` wrapper
        :class:`Group` instead of a :class:`Mesh` when the split yields more than
        one part (a single part falls through to a plain leaf), matching
        ``add_points`` / ``add_gsplats``. A mesh may also be added directly to a
        ``kind=partition`` group you built yourself, provided that group declares
        ``display_type='mesh'`` — a partition is homogeneous, so a mismatched
        declaration is refused. ``partition`` and ``substitutive_lod`` cannot be
        combined for Mesh or Lines; Points uses that pair for its overview shape.

        ``additive_lod`` IS supported, as a **reveal ladder and nothing else**: it
        writes ``additive_<i>/`` levels inside the leaf, each holding one concentric
        shell of faces, innermost first, which the viewer draws cumulatively so the
        surface grows outward from its centre as it streams. ``method`` accepts only
        ``"radial"`` — a prefix of an *arbitrarily ordered* index buffer is a surface
        with holes rather than a coarser one, so ``"random"`` / ``"salience"`` and the
        element samplers are refused, as are ``salience_kind`` and ``seed``. Use
        ``substitutive_lod`` to make a surface genuinely coarser. The ladder carries
        no energy stamps by construction (a reveal is a partial surface at FULL
        brightness, so the viewer's ``1/e(k)`` brightness compensation must not reach
        it), it degrades to a plain leaf with a ``UserWarning`` when ``labels`` or
        ``image_labels`` is set (a level re-indexes its own vertices, so there is no
        single index space for a union label CSR), and it cannot yet be combined with
        ``substitutive_lod`` or ``partition`` — each pairing is refused by name, where
        Points and Lines compose both. See
        :func:`luxar.core.group.lod.mesh.resolve_additive_axis_mesh`.

        The viewer half ships too: a mesh node declaring ``n_additive_sublods > 1``
        is loaded by its own progressive loader, which fetches the levels in order
        and commits each grown prefix into the same buffers.

        Not supported for meshes (raises rather than silently degrading):
        ``blending_mode='volumetric'`` — a zero-thickness surface has no path length
        to integrate. See ``docs/specs/MESH_NODE_SPEC.md`` §9.

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
            shading: ``"smooth"``, ``"flat"``, or unlit ``"none"``. Defaults to
                ``"smooth"`` when ``normals`` are given, else ``"flat"``. An
                explicit value is stored as given — ``"flat"`` renders faceted
                even with normals present, ``"smooth"`` without normals falls
                back to derived flat normals at render time, and ``"none"``
                computes no lighting normal.
            double_sided: Whether back faces render (default ``True``).
            labels: Optional per-vertex strings for hover tooltips.
            image_labels: Optional per-vertex images for hover thumbnails. Not
                supported alongside ``partition``.
            keys: Optional list of machine-readable strings, one per vertex,
                for ``link`` / ``copy`` templates to substitute as
                ``{hover_key}``. Same length rule and the same spatial
                reordering as ``labels`` — a key stays paired with its element
                — but kept separate so the visible label can stay readable
                prose while the URL is built from a bare id.
            partition: ``True`` for the default cap, or
                ``{"max_elements": int, "rule": "median"|"midpoint"|"sah"}``, to
                split the surface into spatially-culled parts. ``max_elements``
                counts FACES — the BSP recurses on face centroids, so a triangle
                is the indivisible unit. Faces are assigned whole (never cut) and
                each part gathers and renumbers the vertices its own faces use, so
                vertices on a cut are duplicated between neighbouring parts.
            parent: Optional explicit parent node (defaults to this group).
            extend_to_all: Dimension name(s) across which this mesh stays visible.
            dim_order: Names of the dimensions the ``vertices`` columns are in,
                for remapping onto the scene's dimension order. ``faces`` is index
                data addressing vertex rows and is never reordered. An
                orientation-reversing order flips face handedness relative to the
                scene frame; the writer warns but does not repair it. Reverse the
                corner order with ``faces[:, [0, 2, 1]]`` when needed; see the mesh
                spec §3.7.
            fill: Fill values for scene dimensions absent from ``dim_order``.
            substitutive_lod: ``True`` / ``{...}`` to write a ``kind=lod`` group of
                progressively DECIMATED copies of the surface (see above).
            additive_lod: ``True`` / ``{...}`` to write a reveal ladder of
                ``additive_<i>/`` levels inside the leaf. Keys: ``method``
                (``"radial"`` only), ``n_lods``, ``counts`` (alias ``breakpoints``),
                ``reveal_center``, ``spatial_dims``. Levels hold concentric shells of
                FACES — a triangle is the indivisible unit, as it is for
                ``partition`` — and are cumulative when concatenated, so the surface
                grows outward as it loads.
            **attrs: Additional attributes — ``opacity``, ``intensity``,
                ``offset``, ``gamma``, ``colormap``, ``layer``, ``visible``,
                ``transform``, ``nd_transform``, ``blending_mode``, and the
                mesh-only appearance controls ``ambient``, ``specular``,
                ``alpha_cutoff`` (each in ``[0, 1]``), ``shade_exponent``, and
                ``shininess`` (both strictly positive and finite). Also mesh-only:
                ``material`` (``"luxar"``, the default house shader, or
                ``"physical"`` for three's physically based material lit by the
                viewer's scene environment — ``MESH_PHYSICAL_MATERIALS_SPEC.md``)
                and the knobs it unlocks, ``roughness``, ``metalness``,
                ``clearcoat``, ``clearcoat_roughness``, ``iridescence``, ``sheen``
                (each in ``[0, 1]``) and ``sheen_color`` (``"#rrggbb"``), plus
                the glass family: ``transmission`` (``[0, 1]``), ``ior``
                (``[1, 2.333]``), ``thickness`` (``>= 0``), ``attenuation_color``
                (``"#rrggbb"``), ``attenuation_distance`` (``> 0``),
                ``dispersion`` (``>= 0``) and ``refract_data`` (``bool``). A
                physical knob without ``material="physical"`` is refused; a
                physical mesh refuses the house-shader knobs, ``blending_mode``,
                ``colormap``, ``texture`` and ``shading="none"``; and
                ``thickness`` / ``attenuation_*`` / ``dispersion`` /
                ``refract_data`` are refused without a ``transmission`` above
                zero — none of them means anything in those pairings. Glass
                refracts the background and other meshes; with
                ``refract_data=True`` it also refracts the points, lines and
                splats BEHIND it, while data in front of the glass stays crisp
                on top (the viewer partitions each data fragment by depth
                against the glass; spec §3.4). Also mesh-only,
                and a LOADING knob rather than an appearance one:
                ``slab_tolerance`` (strictly positive and finite, default ``1.0``)
                — the half-width, IN CELLS, of the nD membership slab a
                *continuous* hidden dimension is culled against: a vertex is
                inside when it is within ``slab_tolerance`` cells of the slice.
                A mesh renders a triangle only when all three of its vertices
                fall inside that slab, so on a continuous hidden axis it shows
                "the surface near this slice" rather than a planar cut, and this
                is the only control over how thick "near" is. It has no effect
                on a discrete hidden axis (time, channel), which uses a half-cell
                membership rule instead. Note
                ``volumetric`` blending is rejected — it has no meaning for an
                opaque surface. An explicit ``None`` for ``colormap`` or
                ``coverage_fraction`` means "absent" — identical to omitting the
                key — so ``colormap=maybe_colormap`` is a safe call form. Every
                OTHER render attr still refuses a ``None``.

        Returns:
            The created Mesh node — or, with ``substitutive_lod``, the ``kind=lod``
            Group wrapping the ladder, or with ``partition``, the ``kind=partition``
            Group wrapping the parts (matching ``add_points`` / ``add_lines``). With
            ``additive_lod`` it is still the Mesh node: a reveal ladder lives INSIDE
            the leaf, so the caller's "one node" is unchanged.
        """
        from .adders.mesh import add_mesh_impl

        return self._transactional_add(
            name,
            parent,
            lambda: add_mesh_impl(
                self,
                name=name,
                vertices=vertices,
                faces=faces,
                normals=normals,
                normal_dims=normal_dims,
                colors=colors,
                scalars=scalars,
                uvs=uvs,
                texture=texture,
                texture_encoding=texture_encoding,
                texture_width=texture_width,
                texture_height=texture_height,
                texture_channels=texture_channels,
                texture_color_space=texture_color_space,
                texture_ktx2_mode=texture_ktx2_mode,
                texture_ktx2_quality=texture_ktx2_quality,
                texture_ktx2_rdo_l=texture_ktx2_rdo_l,
                texture_ktx2_zcmp=texture_ktx2_zcmp,
                shading=shading,
                double_sided=double_sided,
                labels=labels,
                keys=keys,
                image_labels=image_labels,
                partition=partition,
                parent=parent,
                extend_to_all=extend_to_all,
                dim_order=dim_order,
                fill=fill,
                substitutive_lod=substitutive_lod,
                additive_lod=additive_lod,
                **attrs,
            ),
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
        label_ids: Optional[np.ndarray[Any, Any]] = None,
        label_vocabulary: Optional[Dict[int, str]] = None,
        labels: Optional[Union[List[str], Sequence[str]]] = None,
        image_labels: Optional[Any] = None,
        keys: Optional[Union[List[str], Sequence[str]]] = None,
        parent: Optional[Node] = None,
        extend_to_all: Optional[Union[List[str], str]] = None,
        dim_order: Optional[List[str]] = None,
        fill: Optional[Dict[str, float]] = None,
        fill_sigma: Optional[Dict[str, float]] = None,
        partition: Any = None,
        substitutive_lod: Any = _UNSET_LOD,
        additive_lod: Any = _UNSET_LOD,
        _source_dtype: Optional[str] = None,
        **attrs: Any,
    ) -> Union[GSplats, "Group"]:
        """Add a Gaussian splats node.

        Args:
            name: Name of the gsplats node
            centers: Array of shape (N, D) for splat centers
            amplitudes: (N,) array or scalar for intensities
            cholesky_factors: (N, k) packed lower-triangular factor L of the
                covariance (Σ = L·Lᵀ), k=D*(D+1)/2. The diagonal is scale-like:
                isotropic std σ uses [σ, 0, σ, 0, 0, σ], not 1/sigma.
            colors: Optional (N, 3) RGB or (N, 4) RGBA array (the alpha
                column is per-splat opacity in [0, 1]), RGB tuple, or None
            label_ids: Optional non-negative integer class id per splat.
            label_vocabulary: Explicit mapping from every stored class id to its name.
            labels: Optional list of strings, one per splat. Used for hover tooltips.
            image_labels: Optional per-element images for hover thumbnails.
            keys: Optional list of machine-readable strings, one per splat,
                for ``link`` / ``copy`` templates to substitute as
                ``{hover_key}``. Same length rule and the same spatial
                reordering as ``labels`` — a key stays paired with its element
                — but kept separate so the visible label can stay readable
                prose while the URL is built from a bare id.
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
            substitutive_lod: Substitutive-LOD control. The value vocabulary is
                the same as :meth:`add_gsplats_from_data`'s historical
                ``lod_group=``. When combined with ``partition=``, every
                substitutive level is partitioned independently.
            additive_lod: Additive-LOD control. The value vocabulary matches
                :meth:`add_gsplats_from_data`. Requesting an additive ladder
                opts this node out of compiler auto-partitioning; explicit
                ``partition=`` beside the ladder remains unsupported. On a
                stacked or hidden-dimension node, an authored ladder is inert
                when one already exists unless ``recompute=True``, and
                ``slice_dims=`` is the only way to give every slice the same
                absolute budget.
            **attrs: Additional node attributes. Common ones:

                - ``layer`` (bool): Expose this node in the viewer's Layers
                  panel for per-node control. When ``partition=`` produces a
                  wrapper, ``layer=True`` lands on the wrapper, not on
                  each leaf part.
                - ``visible`` (bool): Initial visibility when scene loads
                  (default ``True``).
                - ``opacity``, ``intensity``, ``gamma``, ``blending_mode``,
                  ``colormap``: standard rendering attributes. An explicit
                  ``None`` for ``colormap`` or ``coverage_fraction`` means
                  "absent" — identical to omitting the key — so
                  ``colormap=maybe_colormap`` is a safe call form. Every OTHER
                  render attr still refuses a ``None``.
                - ``absorption`` (float >= 0): absorption coefficient kappa,
                  read by the ``"volumetric"`` blending mode; kappa=0 renders
                  like additive. Defaults to 1.0. Like ``layer``, on a
                  ``partition=`` wrapper this lands on the wrapper node, not
                  on each leaf part.

        Returns:
            The created ``GSplats`` node, or a kind=partition ``Group``
            wrapper when ``partition=`` produced more than one part.
        """
        resolved_substitutive_lod = (
            None if substitutive_lod is _UNSET_LOD else substitutive_lod
        )
        resolved_additive_lod = None if additive_lod is _UNSET_LOD else additive_lod
        if (
            resolved_substitutive_lod is not None
            and resolved_substitutive_lod is not False
        ) or (resolved_additive_lod is not None and resolved_additive_lod is not False):
            from ...validation.writing import (
                GSPLATS_RESERVED_ATTRS,
                validate_render_attrs,
            )
            from .compositing import funnel_add_error

            try:
                validate_render_attrs(attrs, reserved_attrs=GSPLATS_RESERVED_ATTRS)
                result = _gsplat_data_from_arrays(
                    centers,
                    amplitudes,
                    cholesky_factors,
                    colors,
                    label_ids,
                    label_vocabulary,
                )
            except (ValueError, TypeError) as error:
                raise ValueError(funnel_add_error("gsplats", name, error)) from error
            return self.add_gsplats_from_data(
                name,
                result,
                parent=parent,
                extend_to_all=extend_to_all,
                dim_order=dim_order,
                fill=fill,
                fill_sigma=fill_sigma,
                substitutive_lod=resolved_substitutive_lod,
                additive_lod=resolved_additive_lod,
                _source_dtype=_source_dtype,
                # The array adder has always preserved caller amplitudes.
                normalize_amplitudes=False,
                partition=partition,
                labels=labels,
                image_labels=image_labels,
                keys=keys,
                **attrs,
            )

        from .adders.gsplats import add_gsplats_impl

        return self._transactional_add(
            name,
            parent,
            lambda: add_gsplats_impl(
                self,
                name=name,
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky_factors,
                colors=colors,
                label_ids=label_ids,
                label_vocabulary=label_vocabulary,
                labels=labels,
                keys=keys,
                image_labels=image_labels,
                parent=parent,
                extend_to_all=extend_to_all,
                dim_order=dim_order,
                fill=fill,
                fill_sigma=fill_sigma,
                partition=partition,
                _source_dtype=_source_dtype,
                **attrs,
            ),
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
        lod_group: Any = _UNSET_LOD,
        additive_lod: Any = None,
        normalize_amplitudes: Any = _DEFAULT_NORMALIZE_AMPLITUDES,
        substitutive_lod: Any = _UNSET_LOD,
        **attrs: Any,
    ) -> Union[GSplats, "Group"]:
        """Add Gaussian splats from a GSplatData object.

        Multi-additive-LOD data (from ``make_additive_lod`` or an
        ``additive_lod=`` spec) is written with per-sub-LOD subgroups
        directly under the gsplats node (``<node>/additive_<i>/...``)
        for progressive (prefix-sum) loading. Single-LOD data uses the
        flat layout (arrays at the node path).

        ``substitutive_lod`` and ``additive_lod`` control the two LOD axes (see
        ``luxar.core.group.lod.gsplats.resolve_substitutive_axis_gsplats`` /
        ``resolve_additive_axis_gsplats`` for the full value vocabulary).
        ``lod_group=`` remains supported as an alias for
        ``substitutive_lod=``; passing both is refused. When
        the resolved data has multiple substitutive levels, this method
        builds a ``kind="lod"`` ``Group`` containing one gsplats child
        per level (in coarsest→finest order, named ``child_<i>``) and
        returns it; otherwise it returns a single :class:`GSplats` node.

        ``labels`` / ``image_labels`` may not be passed when the resolved result
        is multi-substitutive: every level is its own set of merged
        representative splats with its own count, so no single list has a
        per-element correspondence to carry. Pass ``substitutive_lod=False``
        (or its ``lod_group=False`` alias) to label the collapsed finest level,
        or build the ``kind="lod"`` group yourself with :meth:`add_lod_group`
        and give each child its own labels.

        **An explicit ``None`` in ``**attrs`` means "absent".** For ``labels``,
        ``image_labels``, ``partition``, ``colors``, ``truncation_radius``,
        ``colormap`` and ``coverage_fraction``, passing ``None`` is exactly
        equivalent to omitting the key — so the idiomatic
        ``partition=maybe_partition`` / ``colormap=maybe_colormap`` call form is
        safe. Every OTHER attribute (``opacity``, ``blending_mode``, ``layer``,
        ``visible``, ``gamma``, ``intensity``, ``absorption``, ...) rejects a
        ``None`` as an invalid value, so a typo is not silently swallowed.

        **The data's own channels may not be passed as attributes.** ``centers``,
        ``amplitudes``, ``cholesky_factors`` and a non-``None`` ``colors`` each
        raise ``ValueError``: this method supplies all four from ``result``
        itself, so a keyword of the same name collides with the value already
        being passed, and on a multi-child result it could not be split per child
        anyway. Set them on the ``GSplatData`` before calling, or use
        ``colormap=`` for appearance.

        ``coverage_fraction`` may only be passed in ``**attrs`` when the
        result is single-substitutive AND the parent is itself a
        ``kind="lod"`` ``Group`` (the child is a leaf of an enclosing
        LOD group). Passing it on a multi-substitutive path raises
        ``ValueError`` — use
        ``substitutive_lod=dict(coverage_fractions=[...])`` to override the
        auto-derived thresholds. (``coverage_fraction=None`` is "absent" per
        the rule above, so it is accepted on any path.)

        Args:
            name: Name of the gsplats (or kind=lod group) node.
            result: GSplatData from ``fit_gaussian_splats`` or similar.
            parent: Parent node (default: this group).
            extend_to_all: Visibility extension across non-displayed dimensions.
            dim_order: Map data columns to scene dimensions by name.
            fill: Fixed coordinate values for unmapped dimensions.
            fill_sigma: Standard deviations for unmapped dims in Cholesky embedding.
            substitutive_lod: Substitutive-axis control. ``None`` (default;
                auto-lower a multi-substitutive pyramid into a ``kind=lod``
                Group), ``True`` (require stored levels), ``False`` (collapse
                to finest), ``dict(...)`` (compute via
                :func:`make_substitutive_lod`), or ``dict(..., recompute=True)``.
                Optional ``coverage_fractions=[...]`` inside the dict overrides
                the auto-derived thresholds. On a computed ladder, integer
                ``coarsen_dims`` entries name the raw ``result.centers`` columns
                before ``dim_order``; dimension names remain scene names.
            lod_group: Backward-compatible alias for ``substitutive_lod``.
                Passing both is refused.
            additive_lod: Additive-axis control, uniform across substitutive
                levels. Same value vocabulary as ``substitutive_lod``;
                ``dict(...)`` routes to :func:`make_additive_lod`.
            normalize_amplitudes: Scale amplitudes so a robust upper
                reference (the 99.9th percentile) lands at 1.0, applied as ONE
                factor across every substitutive level and additive rung.
                ``True`` / ``"auto"`` (the default outside a ``kind=lod`` or
                ``kind=partition`` group) acts only when that reference exceeds
                1.0, so data already in range is untouched. Children inserted
                into those specialized groups default to ``False`` because
                their exposure must be shared across siblings; pass ``True``
                explicitly to override that rule. A positive number sets an
                explicit target. The factor used is recorded as
                ``amplitude_normalization_factor``.

                On by default for standalone insertion because raw fitted
                amplitudes cannot be corrected at display time. A fit stores
                source units (detector counts),
                and while the colormap window feeds only the LUT index —
                clamped to ``[0, 1]``, so it picks a colour — emitted radiance
                and volumetric optical depth are both LINEAR in the raw stored
                amplitude and nothing windows them. See
                :mod:`luxar.core.group.gsplats_pipeline.amplitude_norm`.
            **attrs: Additional node attributes — the :meth:`add_gsplats`
                vocabulary (including ``absorption``) MINUS the four channels
                this method supplies from ``result``, which are refused; see the
                two rules above for that and for ``None`` handling. On a nested
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
            ...     substitutive_lod=dict(compression_factor=4, levels=2),
            ...     additive_lod=dict(n_lods=4),
            ... )
        """
        from .gsplats_pipeline.from_data import add_gsplats_from_data_impl

        lod_group = _resolve_gsplat_substitutive_lod_alias(substitutive_lod, lod_group)
        normalize_amplitudes = _resolve_normalize_amplitudes_default(
            parent or self, normalize_amplitudes
        )
        return self._transactional_add(
            name,
            parent,
            lambda: add_gsplats_from_data_impl(
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
                normalize_amplitudes=normalize_amplitudes,
                **attrs,
            ),
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
        normalize_amplitudes: Any = _DEFAULT_NORMALIZE_AMPLITUDES,
        partition: Any = None,
        substitutive_lod: Any = _UNSET_LOD,
        lod_group: Any = _UNSET_LOD,
        additive_lod: Any = None,
        flatten: bool = False,
        **attrs: Any,
    ) -> Union[GSplats, "Group"]:
        """Add Gaussian splats by loading from a .gsplats.zarr file.

        If the source file carries multiple substitutive levels, the
        pyramid is auto-lowered into a ``kind=lod`` Group (one gsplats
        child per substitutive level); pass ``substitutive_lod=False`` to
        collapse to the finest level instead. ``lod_group=False`` remains
        the backward-compatible alias.

        Set ``flatten=True`` to materialize the source tree's default finest
        selection as one leaf before applying ``partition=``,
        ``substitutive_lod=``, or ``additive_lod=``. Without it, ``partition=``
        and ``additive_lod=`` retain a nested source tree and apply to each leaf;
        ``substitutive_lod=`` requires flattening that tree first. ``lod_group=``
        is retained as an alias for ``substitutive_lod=``.

        ``labels`` / ``image_labels`` / ``keys`` are accepted only when the file is a single
        leaf with NO additive ladder. Any multi-LEAF result — an auto-lowered
        pyramid, or a grafted multi-part ``kind=lod`` / ``kind=partition``
        subtree — refuses them, because each leaf holds its own set of splats
        (see ``add_gsplats_from_data``). A single LADDERED leaf (``gsplat lod
        --recipe stream``, or the one-part output of ``--recipe tiles`` on a
        small dataset — a plain ``gsplat fit`` writes a FLAT leaf, which labels
        fine) is refused too, because the additive writer has no per-element string
        channel — ``gsplat flatten`` collapses the ladder if you need one.

        The two ``**attrs`` rules of :meth:`add_gsplats_from_data` apply here
        identically, and identically on BOTH of this method's branches (a
        matrix-shaped file and a grafted nested one): an explicit ``None`` for
        ``labels`` / ``image_labels`` / ``keys`` / ``partition`` / ``colors`` /
        ``truncation_radius`` / ``colormap`` / ``coverage_fraction`` means
        "absent", while ``centers`` / ``amplitudes`` / ``cholesky_factors`` and a
        non-``None`` ``colors`` raise ``ValueError`` (they come from the file
        itself). Nothing is written when either rule refuses.

        Args:
            name: Name of the gsplats node
            path: Path to .gsplats.zarr file
            parent: Parent node (default: this group)
            extend_to_all: Visibility extension across non-displayed dimensions
            dim_order: Map data columns to scene dimensions by name
            fill: Fixed coordinate values for unmapped dimensions
            fill_sigma: Standard deviations for unmapped dims in Cholesky embedding
            normalize_amplitudes: Scale amplitudes so a robust upper
                reference (the 99.9th percentile) lands at 1.0, applied as ONE
                factor across every substitutive level and additive rung.
                ``True`` / ``"auto"`` (the default outside a ``kind=lod`` or
                ``kind=partition`` group) acts only when that reference exceeds
                1.0, so data already in range is untouched. Children inserted
                into those specialized groups default to ``False`` because
                their exposure must be shared across siblings; pass ``True``
                explicitly to override that rule. A positive number sets an
                explicit target. The factor used is recorded as
                ``amplitude_normalization_factor``.

                On by default for standalone insertion because raw fitted
                amplitudes cannot be corrected at display time. A fit stores
                source units (detector counts),
                and while the colormap window feeds only the LUT index —
                clamped to ``[0, 1]``, so it picks a colour — emitted radiance
                and volumetric optical depth are both LINEAR in the raw stored
                amplitude and nothing windows them. See
                :mod:`luxar.core.group.gsplats_pipeline.amplitude_norm`.
            partition: Spatial partition control applied to the loaded matrix,
                or to each leaf of a retained nested tree.
            substitutive_lod: Substitutive-LOD control applied after loading;
                nested trees require ``flatten=True``.
            lod_group: Backward-compatible alias for ``substitutive_lod``.
            additive_lod: Additive-LOD control applied to the loaded matrix, or
                independently to each leaf of a retained nested tree.
            flatten: Collapse stored structure to the default finest selection
                before applying the requested structure.
            **attrs: Additional node attributes — the :meth:`add_gsplats`
                vocabulary (including ``absorption``) MINUS the four channels
                the file supplies, which are refused; see the rules above. On a
                nested tree, compositing attributes (``blending_mode``, ``absorption``,
                ``opacity``, ...) land on the wrapper node ONLY — the viewer
                resolves them down the ancestry — while the rest (e.g.
                ``colormap``) are copied onto each leaf. Stamping
                ``blending_mode`` on the parts too would SHADOW the wrapper
                (it is nearest-setter-wins), leaving the layer's Blend control
                inert.
        """
        from .gsplats_pipeline.from_io import add_gsplats_from_file_impl

        lod_group = _resolve_gsplat_substitutive_lod_alias(substitutive_lod, lod_group)
        normalize_amplitudes = _resolve_normalize_amplitudes_default(
            parent or self, normalize_amplitudes
        )
        return self._transactional_add(
            name,
            parent,
            lambda: add_gsplats_from_file_impl(
                self,
                name=name,
                path=path,
                parent=parent,
                extend_to_all=extend_to_all,
                dim_order=dim_order,
                fill=fill,
                fill_sigma=fill_sigma,
                normalize_amplitudes=normalize_amplitudes,
                partition=partition,
                lod_group=lod_group,
                additive_lod=additive_lod,
                flatten=flatten,
                **attrs,
            ),
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
        normalize_amplitudes: Any = _DEFAULT_NORMALIZE_AMPLITUDES,
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
            progressive: Optimize in several passes against residuals, returning one
                flat splat set. To build a streaming ladder, use
                ``add_gsplats(..., additive_lod=...)`` or
                ``add_gsplats_from_file(..., additive_lod=...)``.
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
            normalize_amplitudes: Scale amplitudes so a robust upper
                reference (the 99.9th percentile) lands at 1.0. The default is
                enabled outside a ``kind=lod`` or ``kind=partition`` group and
                disabled for children inserted directly into those groups so
                sibling exposure stays shared. Pass ``True`` to override that
                specialized-group default, ``False`` to preserve raw units, or
                a positive number to set an explicit target. The factor used is
                recorded as ``amplitude_normalization_factor``.
            **fit_kwargs: Extra kwargs for fitting function
        """
        from .gsplats_pipeline.from_io import add_gsplats_from_volume_impl

        normalize_amplitudes = _resolve_normalize_amplitudes_default(
            parent or self, normalize_amplitudes
        )
        return self._transactional_add(
            name,
            parent,
            lambda: add_gsplats_from_volume_impl(
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
                normalize_amplitudes=normalize_amplitudes,
                opacity=opacity,
                absorption=absorption,
                blending_mode=blending_mode,
                **fit_kwargs,
            ),
        )
