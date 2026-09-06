"""Scene root node for Luxar hierarchical scene graphs.

This module provides the Scene class, which serves as the root node of the
scene hierarchy. Data-adding methods (add_points, add_lines, add_gsplats)
are inherited from Group.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from arbol import aprint

from ...io.writer import ZarrWriterProtocol
from ...utils.atomic_copy import atomic_copytree
from ..citation import validate_citation
from ..dimensions import Dimensions
from ..group import Group
from ..overlay import Overlay
from ..viewer_config import ViewerConfig


class Scene(Group):
    """Scene root node representing the top level of a scene hierarchy.

    The Scene class is a Group that must be created through LuxarZarrCompiler
    for progressive writing and memory-efficient handling of large datasets.

    Scene dimensions are REQUIRED and serve as the single source of truth for
    the coordinate system. All data nodes must conform to these dimensions.

    Data-adding methods (add_points, add_lines, add_gsplats, etc.) are
    inherited from Group and work identically on Scene.

    Example:
        >>> from luxar import LuxarZarrCompiler, Dimensions, Dimension
        >>> dims = Dimensions([
        ...     Dimension("X", display=True),
        ...     Dimension("Y", display=True),
        ...     Dimension("Z", display=True),
        ... ])
        >>> with LuxarZarrCompiler('output.luxar.zarr') as compiler:
        ...     scene = compiler.create_scene(dimensions=dims)
        ...     scene.add_points('points', huge_array)  # Written immediately
        ...
        ...     # Groups also support add_points, add_lines, add_gsplats:
        ...     group = scene.add_group("my_group")
        ...     group.add_points('nested_pts', more_data)

    Args:
        writer: Writer interface for progressive writing (required)
        dimensions: Scene-level dimension definitions (REQUIRED)
    """

    def __init__(
        self,
        writer: ZarrWriterProtocol,
        dimensions: Dimensions,
        viewer_config: Optional[ViewerConfig] = None,
        citation: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Initialize a new Luxar scene.

        Args:
            writer: Writer interface for progressive writing (required)
            dimensions: Scene-level dimension definitions (REQUIRED).
                Defines the coordinate system for all data in the scene.
            viewer_config: Optional viewer configuration hints. Stored in
                the zarr file and used by the viewer as scene-specific defaults.
            citation: Optional credit for whoever produced the underlying
                dataset -- ``{"short", "ref"?, "doi"?, "license"?, "url"?}``.
                Written to the store's root attributes so the attribution travels
                with the data rather than only with the page that happens to show
                it. ``None`` means no external dataset to credit.

        Raises:
            ValueError: If writer is None, dimensions is None, the citation is
                malformed, or initialization fails
        """
        try:
            if writer is None:
                raise ValueError(
                    "Writer is required. Use LuxarZarrCompiler to create scenes."
                )

            if dimensions is None:
                raise ValueError(
                    "dimensions is required. Scene dimensions define the coordinate "
                    "system and are the single source of truth for all data in the scene."
                )

            # Create lightweight root node (sets self._writer)
            super().__init__("Scene", writer=writer)

            # Store dimensions (REQUIRED)
            self._dimensions: Dimensions = dimensions

            # Store dimensions in attributes
            writer.write_group("/", scene_dimensions=dimensions.to_dict())

            # Store viewer config if provided
            self._viewer_config: Optional[ViewerConfig] = viewer_config
            if viewer_config is not None:
                self._persist_attr("viewer_config", viewer_config.to_dict())

            # Store the dataset credit if provided. Validated here rather than
            # trusted, because a malformed citation that reaches the store is
            # then baked into every copy of the data.
            self._citation: Optional[dict[str, str]] = validate_citation(citation)
            if self._citation is not None:
                writer.write_group("/", citation=self._citation)

            # Overlay state
            self._overlay_counter: int = 0
            self._overlays: List[Overlay] = []

            # Picking/label state
            self._has_labels: bool = False
            self._has_image_labels: bool = False
            self._suppress_hover_overlay: bool = False

            aprint("✓ Scene initialized successfully with progressive writer")

        except Exception as e:
            aprint(f"Failed to initialize Scene: {e}")
            raise ValueError(f"Could not initialize Scene: {e}") from e

    @property
    def citation(self) -> Optional[dict[str, str]]:
        """Dataset credit stamped into the store root, or None if none is owed."""
        return dict(self._citation) if self._citation is not None else None

    # ---------------------------------------------------------- hierarchy

    def _ensure_no_duplicate_child(self, name: str) -> None:
        """Validate a top-level user node name before any data is written."""
        if name == "overlays":
            raise ValueError(
                "Top-level node name 'overlays' is reserved for screen-space "
                "overlay metadata. Choose a different user node name."
            )
        super()._ensure_no_duplicate_child(name)

    def add_group(self, name: str, **attrs: Any) -> Group:
        """Create and add a child group node to the scene.

        Args:
            name: Name of the group
            **attrs: Additional attributes for the group. Supports:
                opacity: float (0.0-1.0, default 1.0) - Node opacity
                absorption: float (>=0, default 1.0) - Volumetric kappa
                gamma: float (0.1-10.0, default 1.0) - Gamma correction
                blending_mode: str ("normal", "additive", "max", "opaque",
                "luminous", "volumetric"; default "additive")

        Returns:
            The created group node

        Raises:
            ValueError: If group creation fails or rendering attributes are invalid
        """
        try:
            aprint(f"Adding group node '{name}'.")
            return super().add_group(name, **attrs)
        except Exception as e:
            aprint(f"Failed to add group node '{name}': {e}")
            raise ValueError(f"Could not add group '{name}': {e}") from e

    # ---------------------------------------------------------- scene overrides

    def _find_scene(self) -> Scene:
        """Scene is its own root — returns self."""
        return self

    # ---------------------------------------------------------- validation

    def _resolve_extend_to_all(
        self,
        extend_to_all: Optional[Union[List[str], str]],
        positions: np.ndarray,
        data_type: str,
        _stacklevel: int = 3,
    ) -> List[str]:
        # +2 frames vs the original method body: one for this Scene-method
        # delegate, one for the per-leaf adder impl (e.g.
        # `add_points_impl`) that now sits between Group.add_<type> and us.
        from .validation import resolve_extend_to_all

        return resolve_extend_to_all(
            self, extend_to_all, positions, data_type, _stacklevel=_stacklevel + 2
        )

    def _analyze_extend_candidates(self, positions: np.ndarray) -> List[str]:
        from .validation import analyze_extend_candidates

        return analyze_extend_candidates(self, positions)

    def _validate_dimension_count(
        self,
        positions: np.ndarray,
        node_name: str,
        data_type: str = "positions",
    ) -> None:
        # No _stacklevel counterpart: this half only raises, and a raise
        # carries its own traceback (stacklevel matters solely for the
        # per-dimension range warning in _validate_data_dimensions).
        from .validation import validate_dimension_count

        validate_dimension_count(self, positions, node_name, data_type)

    def _validate_data_dimensions(
        self,
        positions: np.ndarray,
        node_name: str,
        data_type: str = "positions",
        _stacklevel: int = 3,
    ) -> None:
        from .validation import validate_data_dimensions

        validate_data_dimensions(
            self, positions, node_name, data_type, _stacklevel=_stacklevel + 2
        )

    # ---------------------------------------------------------- dim_order

    def _apply_dim_order(
        self,
        positions: np.ndarray,
        dim_order: List[str],
        fill: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        from .dim_order import apply_dim_order

        return apply_dim_order(self, positions, dim_order, fill)

    # ---------------------------------------------------------- properties

    def get_store_path(self) -> str:
        """Get the path to the backing Zarr store.

        Returns:
            Path to the Zarr store backing this scene. Archive-backed writers
            return their staging directory before finalization and the archive
            path afterward.
        """
        if self._writer:
            return self._writer.store_path
        raise ValueError("No store path available without writer")

    @property
    def dimensions(self) -> Dimensions:
        """Get scene-level dimensions.

        Returns:
            Dimensions object (always present - required at construction)
        """
        if self._dimensions is None:
            raise RuntimeError(
                "Scene dimensions are not initialized. This indicates a bug in "
                "scene construction. Use LuxarZarrCompiler.create_scene() with "
                "a Dimensions object."
            )
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims: Dimensions) -> None:
        """Set and persist scene-level dimensions.

        Args:
            dims: Dimensions object (REQUIRED - cannot be None)

        Raises:
            ValueError: If dims is None, or if the dimensionality (ndim)
                differs from the current dimensions after geometry has been
                added (existing data arrays would no longer match).
        """
        if dims is None:
            raise ValueError(
                "dimensions cannot be None. Scene dimensions are required and "
                "define the coordinate system for all data in the scene."
            )
        if (
            self._dimensions is not None
            and dims.ndim != self._dimensions.ndim
            and self._has_authored_geometry()
        ):
            raise ValueError(
                f"Cannot change scene dimensionality from {self._dimensions.ndim}D "
                f"to {dims.ndim}D after geometry has been added: existing data "
                "arrays have the old number of coordinate columns and would no "
                "longer match the scene dimensions. Set dimensions before adding "
                "geometry. Same-dimensionality changes (names/units/ranges/display) "
                "are still allowed."
            )
        self._persist_attr("scene_dimensions", dims.to_dict())
        self._dimensions = dims

    def _has_authored_geometry(self) -> bool:
        """Return True if any descendant node is authored geometry (a DataNode).

        Overlays and plain groups are not geometry; only ``DataNode`` instances
        (Points/Lines/GSplats) count.
        """
        from ..datanode import DataNode

        stack = list(self.children)
        while stack:
            node = stack.pop()
            if isinstance(node, DataNode):
                return True
            stack.extend(node.children)
        return False

    @property
    def viewer_config(self) -> Optional[ViewerConfig]:
        """Get viewer configuration hints.

        Returns:
            ViewerConfig if set, None otherwise.
        """
        if self._viewer_config is None and "viewer_config" in self.attrs:
            vc_dict = self.attrs["viewer_config"]
            self._viewer_config = ViewerConfig.from_dict(vc_dict)
        return self._viewer_config

    @viewer_config.setter
    def viewer_config(self, vc: Optional[ViewerConfig]) -> None:
        """Set viewer configuration hints.

        Args:
            vc: ViewerConfig object, or None to clear.
        """
        self._viewer_config = vc
        if vc is not None:
            vc.validate()
            self._persist_attr("viewer_config", vc.to_dict())
        else:
            self._delete_attr("viewer_config")

    # ---------------------------------------------------------- overlays

    @property
    def overlays(self) -> List[Overlay]:
        """Get the list of overlays added to this scene."""
        return list(self._overlays)

    def add_text(
        self,
        text: str,
        position: Tuple[float, float],
        *,
        name: Optional[str] = None,
        font_size: float = 0.025,
        font: str = "sans",
        color: str = "white",
        opacity: float = 1.0,
        anchor: str = "top-left",
        width: Optional[float] = None,
        text_align: str = "left",
        line_height: float = 1.4,
        background: Optional[str] = None,
        padding: float = 0.005,
        stroke_color: Optional[str] = None,
        stroke_width: float = 0.002,
        visible_range: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
        transition: str = "none",
        transition_duration: float = 0.3,
        interactive: bool = False,
        blend_mode: str = "normal",
        hover: bool = False,
    ) -> Overlay:
        """Add a text overlay to the scene.

        Text is rendered as an HTML element over the viewer canvas.
        If ``width`` is set, text wraps within that viewport-relative width.

        Args:
            text: The text content to display. When ``hover`` is True, this
                is a template with ``{hover_label}``, ``{hover_key}``,
                ``{hover_node}``, ``{hover_index}`` placeholders.
            position: (x, y) in normalized screen coords [0, 1]. Origin is top-left.
            name: Optional overlay name. Auto-generated if None.
            font_size: Font size as fraction of viewport height (default 0.025 = 2.5vh).
            font: Font preset ('sans', 'serif', 'mono') or CSS font-family string.
            color: CSS color string (default 'white').
            opacity: Opacity 0.0-1.0 (default 1.0).
            anchor: Anchor point for positioning (default 'top-left').
            width: Optional width as fraction of viewport width. Enables word wrapping.
            text_align: Text alignment: 'left', 'center', 'right', 'justify'.
            line_height: CSS line-height multiplier (default 1.4).
            background: Optional CSS background color for a backing rectangle.
            padding: Padding around text as fraction of viewport height (default 0.005).
            stroke_color: Optional text stroke/outline color.
            stroke_width: Stroke width as fraction of viewport height (default 0.002).
            visible_range: Optional dimension-based visibility filter.
                Maps dimension names to values or (min, max) range tuples.
            transition: Transition type: 'none' or 'fade' (default 'none').
            transition_duration: Transition duration in seconds (default 0.3).
            interactive: If True, overlay captures pointer events (default False).
            blend_mode: CSS mix-blend-mode (default 'normal'). Use 'difference'
                for XOR-style text that inverts the background colors.
            hover: If True, this overlay is a hover tooltip updated by GPU
                picking. The ``text`` is treated as a template (default False).

        Returns:
            Overlay metadata object.

        Example:
            >>> scene.add_text("Scale: 10um", position=(0.05, 0.95), font_size=0.02)
            >>> scene.add_text(
            ...     "This is a paragraph with wrapping.",
            ...     position=(0.02, 0.85),
            ...     width=0.3,
            ...     text_align='left',
            ...     background='rgba(0,0,0,0.6)',
            ... )
        """
        from .overlays.adders import add_text_impl

        return add_text_impl(
            self,
            text=text,
            position=position,
            name=name,
            font_size=font_size,
            font=font,
            color=color,
            opacity=opacity,
            anchor=anchor,
            width=width,
            text_align=text_align,
            line_height=line_height,
            background=background,
            padding=padding,
            stroke_color=stroke_color,
            stroke_width=stroke_width,
            visible_range=visible_range,
            transition=transition,
            transition_duration=transition_duration,
            interactive=interactive,
            blend_mode=blend_mode,
            hover=hover,
        )

    def add_image(
        self,
        image: Any,
        position: Tuple[float, float],
        *,
        name: Optional[str] = None,
        size: Optional[Tuple[float, float]] = None,
        opacity: float = 1.0,
        anchor: str = "top-left",
        blend_mode: str = "normal",
        format: str = "png",
        visible_range: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
        transition: str = "none",
        transition_duration: float = 0.3,
        interactive: bool = False,
    ) -> Overlay:
        """Add an image overlay to the scene.

        The image is stored directly in the zarr directory and rendered as an
        HTML ``<img>`` element over the viewer canvas.

        Args:
            image: Image data. Accepts: file path (str/Path), raw bytes,
                numpy array (HWC uint8 or float 0-1), PIL Image. Pre-encoded
                bytes and path payloads must be PNG, JPEG, or WebP; a recognized
                path extension must match the payload.
            position: (x, y) in normalized screen coords [0, 1].
            name: Optional overlay name. Auto-generated if None.
            size: Optional (width, height) as fractions of viewport dimensions.
            opacity: Opacity 0.0-1.0 (default 1.0).
            anchor: Anchor point for positioning (default 'top-left').
            blend_mode: CSS blend mode: 'normal', 'multiply', 'screen',
                'overlay', 'additive', or 'difference' (default 'normal').
            format: Encoding format for array/PIL inputs: 'png', 'jpeg', or
                'webp' (default 'png'). Pre-encoded bytes and paths retain their
                detected format.
            visible_range: Optional dimension-based visibility filter.
            transition: Transition type: 'none' or 'fade' (default 'none').
            transition_duration: Transition duration in seconds (default 0.3).
            interactive: If True, overlay captures pointer events (default False).

        Returns:
            Overlay metadata object.

        Example:
            >>> scene.add_image('logo.png', position=(0.9, 0.05), anchor='top-right')
            >>> scene.add_image(
            ...     numpy_heatmap,
            ...     position=(0.0, 0.0),
            ...     size=(1.0, 1.0),
            ...     opacity=0.5,
            ...     blend_mode='multiply',
            ... )
        """
        from .overlays.adders import add_image_impl

        return add_image_impl(
            self,
            image=image,
            position=position,
            name=name,
            size=size,
            opacity=opacity,
            anchor=anchor,
            blend_mode=blend_mode,
            format=format,
            visible_range=visible_range,
            transition=transition,
            transition_duration=transition_duration,
            interactive=interactive,
        )

    def add_video(
        self,
        video: Any,
        position: Tuple[float, float],
        *,
        name: Optional[str] = None,
        size: Optional[Tuple[float, Optional[float]]] = None,
        opacity: float = 1.0,
        anchor: str = "top-left",
        blend_mode: str = "normal",
        loop: bool = True,
        autoplay: bool = True,
        muted: bool = True,
        playback_rate: float = 1.0,
        poster: Any = None,
        visible_range: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
        transition: str = "none",
        transition_duration: float = 0.3,
        interactive: bool = False,
    ) -> "Overlay":
        """Add a looping video overlay to the scene.

        The file is stored verbatim inside the zarr directory (like an image
        overlay) and rendered as an HTML ``<video>`` over the canvas. Muted
        autoplay is what browsers allow without a user gesture, so that is the
        default and the only autoplay mode accepted. A hidden video (its
        ``visible_range`` not matching) is paused, so ten story turntables cost
        one decode at a time.

        Transparency: a VP9 WebM with an alpha channel (``yuva420p``) plays
        transparent in Chrome and Firefox; Safari cannot decode it and shows
        the ``poster`` instead — always supply one for a transparent video.

        Args:
            video: Raw bytes or a file path. Must already be WebM or MP4; there
                is no re-encoding. A recognized path suffix must match the payload.
            position: (x, y) in normalized screen coords [0, 1]. Origin is top-left.
            name: Optional overlay name. Auto-generated if None.
            size: Optional (width, height) as fractions of the viewport. A
                ``None`` height keeps the video's own aspect ratio.
            opacity: Opacity 0.0-1.0 (default 1.0).
            anchor: Anchor point for positioning (default 'top-left').
            blend_mode: CSS blend mode (default 'normal').
            loop: Loop playback (default True).
            autoplay: Start on load / when it becomes visible (default True).
            muted: Required True while ``autoplay`` is True (default True).
            playback_rate: Speed multiplier in (0, 16] (default 1.0).
            poster: Optional still shown before play and where the video cannot
                be decoded (PNG/JPEG/WebP bytes, path, array or PIL image).
            visible_range: Optional dimension-based visibility filter.
            transition: 'none' or 'fade' (default 'none').
            transition_duration: Transition duration in seconds (default 0.3).
            interactive: If True, overlay captures pointer events (default False).

        Returns:
            Overlay metadata object.

        Example:
            >>> scene.add_video(
            ...     "turntable.webm", position=(0.06, 0.5), anchor="center-left",
            ...     size=(0.26, None), poster="turntable.png",
            ...     visible_range={"story": 3}, transition="fade",
            ... )
        """
        from .overlays.adders import add_video_impl

        return add_video_impl(
            self,
            video=video,
            position=position,
            name=name,
            size=size,
            opacity=opacity,
            anchor=anchor,
            blend_mode=blend_mode,
            loop=loop,
            autoplay=autoplay,
            muted=muted,
            playback_rate=playback_rate,
            poster=poster,
            visible_range=visible_range,
            transition=transition,
            transition_duration=transition_duration,
            interactive=interactive,
        )

    def add_html(
        self,
        html: str,
        position: Tuple[float, float],
        *,
        name: Optional[str] = None,
        width: Optional[float] = None,
        opacity: float = 1.0,
        anchor: str = "top-left",
        visible_range: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
        transition: str = "none",
        transition_duration: float = 0.3,
        interactive: bool = False,
        blend_mode: str = "normal",
        hover: bool = False,
        hover_image_size: Optional[Tuple[float, float]] = None,
    ) -> Overlay:
        """Add an HTML overlay to the scene.

        HTML is sanitized to a safe subset of tags and attributes before storage.
        Allowed tags include: b, i, em, strong, a, span, div, br, img, ul, ol,
        li, p, h1-h6, sub, sup, code, pre, table elements.
        Inline styles are allowed. Script tags and event handlers are stripped.

        Args:
            html: HTML content string (will be sanitized). When ``hover``
                is True, this is a template with ``{hover_label}``,
                ``{hover_key}``, ``{hover_image_label}``, ``{hover_node}``,
                ``{hover_index}`` placeholders.
            position: (x, y) in normalized screen coords [0, 1].
            name: Optional overlay name. Auto-generated if None.
            width: Optional width as fraction of viewport width.
            opacity: Opacity 0.0-1.0 (default 1.0).
            anchor: Anchor point for positioning (default 'top-left').
            visible_range: Optional dimension-based visibility filter.
            transition: Transition type: 'none' or 'fade' (default 'none').
            transition_duration: Transition duration in seconds (default 0.3).
            interactive: If True, overlay captures pointer events (default False).
            blend_mode: CSS mix-blend-mode (default 'normal'). Use 'difference'
                for XOR-style content that inverts the background colors.
            hover: If True, this overlay is a hover tooltip updated by GPU
                picking. The ``html`` is treated as a template (default False).
            hover_image_size: Optional (width, height) as viewport fractions
                for hover image thumbnails. Controls the size of
                ``{hover_image_label}`` images.

        Returns:
            Overlay metadata object.

        Example:
            >>> scene.add_html(
            ...     '<span style="color:red;font-weight:bold">Warning</span>: '
            ...     '<span style="color:#aaa">low signal region</span>',
            ...     position=(0.5, 0.9),
            ...     interactive=True,
            ... )
        """
        from .overlays.adders import add_html_impl

        return add_html_impl(
            self,
            html=html,
            position=position,
            name=name,
            width=width,
            opacity=opacity,
            anchor=anchor,
            visible_range=visible_range,
            transition=transition,
            transition_duration=transition_duration,
            interactive=interactive,
            blend_mode=blend_mode,
            hover=hover,
            hover_image_size=hover_image_size,
        )

    # ---------------------------------------------------------- overlay internals

    def _next_overlay_name(self, name: Optional[str]) -> str:
        from .overlays.internals import next_overlay_name

        return next_overlay_name(self, name)

    def _write_overlay(
        self,
        name: str,
        overlay_type: str,
        position: Tuple[float, float],
        attrs: Dict[str, Any],
        image_data: Optional[bytes] = None,
        image_filename: Optional[str] = None,
    ) -> Overlay:
        from .overlays.internals import write_overlay

        return write_overlay(
            self, name, overlay_type, position, attrs, image_data, image_filename
        )

    # ---------------------------------------------------------- picking/labels

    def _notify_labels_added(self) -> None:
        """Called by Group.add_* when labels are provided on any child node."""
        self._has_labels = True

    def _notify_image_labels_added(self) -> None:
        """Called by Group.add_* when image_labels are provided on any child node."""
        self._has_image_labels = True

    def _auto_inject_hover_overlay(self) -> None:
        from .overlays.hover_inject import auto_inject_hover_overlay

        auto_inject_hover_overlay(self)

    # ---------------------------------------------------------- export

    def to_zarr(self, path: PathLike) -> None:
        """Finalize and copy the backing Zarr store to ``path``.

        Luxar scenes are written progressively as nodes are added; the scene
        object itself does not keep an in-memory copy of geometry arrays. This
        method therefore exports by finalizing the current backing store and
        copying that on-disk Zarr directory to a new location.

        Calling this method finalizes the associated writer. Do not add more
        nodes to this scene after calling ``to_zarr()``; create a new
        ``LuxarZarrCompiler`` if additional writes are needed.

        Args:
            path: Destination path for the Zarr store. A directory destination
                must not already exist unless it is the current backing store.
                For an archive-backed scene, the only supported destination is
                the selected archive path, which is replaced if it exists.
                To create an archive from a directory-backed scene, use
                ``LuxarZarrCompiler`` or ``luxar optimise`` instead.

        Raises:
            FileExistsError: If a directory destination already exists and is
                not the current backing store.
            ValueError: If the destination is inside the source store, the
                source store is unavailable, or an archive-backed writer is
                asked to publish anywhere except its selected archive path, or
                a directory-backed writer is asked to copy to an archive path.
        """
        writer = self._writer
        if writer is None:
            raise ValueError("Scene has no backing writer; cannot export to Zarr")

        source = Path(self.get_store_path()).resolve()
        final_source = Path(writer.final_store_path).resolve()
        destination = Path(path).expanduser().resolve()

        # The final destination may differ from the live staging directory.
        # Matching it is still an explicit finalize operation, including when
        # an older archive already exists there and will be replaced.
        if destination == final_source:
            aprint(f"Finalizing scene at {final_source}")
            writer.finalize()
            aprint(f"Finalized scene at {final_source}")
            return

        if source != final_source or not source.is_dir():
            raise ValueError(
                "Scene backing writer only supports its final destination: "
                f"{final_source}"
            )

        if destination.name.endswith(".zip"):
            raise ValueError(
                "Scene.to_zarr() cannot copy a directory store to an archive path: "
                f"{destination}. Create the scene with LuxarZarrCompiler({path!r}) "
                "or run luxar optimise."
            )

        if destination.exists():
            raise FileExistsError(
                f"Destination already exists: {destination}. Remove it first or choose "
                "a different path."
            )

        if destination.is_relative_to(source):
            raise ValueError(
                f"Destination {destination} cannot be inside source Zarr store {source}"
            )

        writer.finalize()
        finalized_source = Path(self.get_store_path()).resolve()
        if finalized_source != source:
            raise ValueError(
                "Scene backing writer relocated the store during finalization; "
                f"use the finalized destination instead: {finalized_source}"
            )
        aprint(f"Exporting scene from {source} to {destination}")
        # CL-1: atomic copy — a failure mid-copy leaves no half-written
        # zarr store at `destination`.
        atomic_copytree(source, destination)
        aprint(f"✓ Scene exported to {destination}")
