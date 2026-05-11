"""Scene root node for Luxar hierarchical scene graphs.

This module provides the Scene class, which serves as the root node of the
scene hierarchy. Data-adding methods (add_points, add_lines, add_gsplats)
are inherited from Group.
"""

from __future__ import annotations

import warnings
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from arbol import aprint

from ..core.dimensions import Dimensions
from ..core.group import Group
from ..core.overlay import Overlay
from ..core.viewer_config import ViewerConfig
from ..io.writer import ZarrWriterProtocol
from ..utils.atomic_copy import atomic_copytree


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
        >>> with LuxarZarrCompiler('output.zarr') as compiler:
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
    ) -> None:
        """Initialize a new Luxar scene.

        Args:
            writer: Writer interface for progressive writing (required)
            dimensions: Scene-level dimension definitions (REQUIRED).
                Defines the coordinate system for all data in the scene.
            viewer_config: Optional viewer configuration hints. Stored in
                the zarr file and used by the viewer as scene-specific defaults.

        Raises:
            ValueError: If writer is None, dimensions is None, or initialization fails
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
                writer.write_group("/", viewer_config=viewer_config.to_dict())

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

    # ---------------------------------------------------------- hierarchy

    def add_group(self, name: str, **attrs: Any) -> Group:
        """Create and add a child group node to the scene.

        Args:
            name: Name of the group
            **attrs: Additional attributes for the group. Supports:
                opacity: float (0.0-1.0, default 1.0) - Node opacity
                gamma: float (0.1-10.0, default 1.0) - Gamma correction
                blending_mode: str ("normal", "additive", "max", "opaque",
                    "luminous"; default "additive")

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
        """Resolve extend_to_all parameter into a final list of dimension names.

        Handles all extend_to_all modes:
        - None: No extension, but warn if candidates detected
        - "all": Extend to all non-displayed dimensions
        - List of names: Validate and use explicit list
        - []: Explicitly no extension (silences warning)

        Args:
            extend_to_all: User-specified extend_to_all value
            positions: Position/vertex/center array for candidate analysis
            data_type: Human-readable data type for warning messages
                ("points", "lines", "splats")

        Returns:
            List of dimension names to extend visibility across

        Raises:
            ValueError: If extend_to_all contains unknown dimensions or invalid value
        """
        if extend_to_all is None:
            # Default: No extension, but warn if candidates detected
            candidates = self._analyze_extend_candidates(positions)
            if candidates:
                warnings.warn(
                    f"Dimension(s) {candidates} have single values but defined ranges.\n"
                    f"If these {data_type} should be visible at ALL values of these dimensions, use:\n"
                    f"    extend_to_all={candidates}\n"
                    f"If intentional ({data_type} only at these specific values), use:\n"
                    f"    extend_to_all=[]  # Explicit: no extension\n"
                    f"Set extend_to_all explicitly to silence this warning.",
                    UserWarning,
                    stacklevel=_stacklevel,
                )
            return []
        elif extend_to_all == "all":
            # Extend to all non-displayed dimensions
            return [
                dim.name
                for dim in self._dimensions.dimensions
                if not dim.display and dim.name
            ]
        elif isinstance(extend_to_all, list):
            # Use explicit list (including empty list to silence warning)
            unknown_dims = [
                dim_name
                for dim_name in extend_to_all
                if dim_name not in self._dimensions.names
            ]
            if unknown_dims:
                raise ValueError(
                    f"Unknown dimension(s) in extend_to_all: {unknown_dims}. "
                    f"Valid dimensions: {self._dimensions.names}"
                )
            return extend_to_all
        else:
            raise ValueError(
                f"Invalid extend_to_all value: {extend_to_all}. "
                f"Expected None, list of dimension names, 'all', or []."
            )

    def _analyze_extend_candidates(self, positions: np.ndarray) -> List[str]:
        """Analyze which dimensions might be candidates for extend_to_all.

        A dimension is a candidate if:
        1. It is not displayed (non-spatial dimension)
        2. It has only ONE unique value in the data
        3. It has a defined range that is larger than just that single value

        Args:
            positions: Position array to analyze

        Returns:
            List of dimension names that are candidates for extension
        """
        candidates: List[str] = []
        data_ndim = positions.shape[1]

        for i, dim in enumerate(self._dimensions.dimensions):
            if dim.display:
                continue
            if i >= data_ndim:
                continue

            unique_values = np.unique(positions[:, i])
            if len(unique_values) != 1:
                continue

            if dim.range is not None:
                value = unique_values[0]
                range_min, range_max = dim.range

                if range_max > range_min and (
                    value >= range_min and value <= range_max
                ):
                    if dim.name:
                        candidates.append(dim.name)

        return candidates

    def _validate_data_dimensions(
        self,
        positions: np.ndarray,
        node_name: str,
        data_type: str = "positions",
        _stacklevel: int = 3,
    ) -> None:
        """Validate that data dimensions match scene dimensions.

        Performs two levels of validation:
        1. HARD ERROR: Dimensionality mismatch (data columns != scene dimensions)
        2. WARNING: Values outside declared dimension ranges

        Args:
            positions: Position/vertex/center array to validate (shape N x D)
            node_name: Name of the node being added (for error messages)
            data_type: Type of data ("positions", "vertices", "centers")

        Raises:
            ValueError: If dimensionality doesn't match scene dimensions
        """
        data_ndim = positions.shape[1]
        scene_ndim = self._dimensions.ndim

        if data_ndim != scene_ndim:
            dim_names = self._dimensions.names
            raise ValueError(
                f"Dimension mismatch for '{node_name}': {data_type} array has "
                f"{data_ndim} columns, but scene has {scene_ndim} dimensions "
                f"({dim_names}).\n"
                f"Expected {data_type} shape: (N, {scene_ndim})\n"
                f"Got {data_type} shape: {positions.shape}"
            )

        if positions.shape[0] == 0:
            return

        for i, dim in enumerate(self._dimensions.dimensions):
            if dim.range is not None:
                col = positions[:, i]
                min_val, max_val = float(col.min()), float(col.max())
                range_min, range_max = dim.range

                if min_val < range_min or max_val > range_max:
                    warnings.warn(
                        f"'{node_name}' {data_type}: dimension '{dim.name}' has values "
                        f"[{min_val:.4g}, {max_val:.4g}] outside declared range "
                        f"[{range_min}, {range_max}]. "
                        f"Consider adjusting the dimension range or data values.",
                        UserWarning,
                        stacklevel=_stacklevel,
                    )

    # ---------------------------------------------------------- dim_order

    def _apply_dim_order(
        self,
        positions: np.ndarray,
        dim_order: List[str],
        fill: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Reorder and pad position data to match scene dimensions.

        Maps data columns to scene dimensions by name, reordering and
        padding as needed. Returns the transformed array and a list of
        unmapped dimension names (candidates for extend_to_all).

        Args:
            positions: Data array of shape (N, d_data)
            dim_order: Scene dimension names for each data column.
                len(dim_order) must equal positions.shape[1].
            fill: Fixed values for unmapped dimensions (default 0.0)

        Returns:
            Tuple of (transformed_positions, unmapped_dim_names):
            - transformed_positions: shape (N, scene_ndim)
            - unmapped_dim_names: names of dims not covered by dim_order

        Raises:
            ValueError: If dim_order names are invalid or have wrong length
        """
        if fill is None:
            fill = {}

        scene_names = self._dimensions.names
        scene_ndim = self._dimensions.ndim
        data_ndim = positions.shape[1]

        # Validate dim_order length matches data columns
        if len(dim_order) != data_ndim:
            raise ValueError(
                f"dim_order has {len(dim_order)} names but data has "
                f"{data_ndim} columns. They must match."
            )

        # Validate names exist in scene dimensions and are unique
        if len(set(dim_order)) != len(dim_order):
            raise ValueError(f"dim_order has duplicate names: {dim_order}")
        for name in dim_order:
            if name not in scene_names:
                raise ValueError(
                    f"dim_order name '{name}' not found in scene dimensions "
                    f"{scene_names}"
                )

        # Validate fill keys are valid dim names and not in dim_order
        for name in fill:
            if name not in scene_names:
                raise ValueError(
                    f"fill key '{name}' not found in scene dimensions {scene_names}"
                )
            if name in dim_order:
                raise ValueError(
                    f"fill key '{name}' is already in dim_order — cannot "
                    f"both map a data column and fill a fixed value"
                )

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

    # ---------------------------------------------------------- properties

    def get_store_path(self) -> str:
        """Get the path to the backing Zarr store.

        Returns:
            Path to the Zarr store backing this scene
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
        """Set scene-level dimensions.

        Args:
            dims: Dimensions object (REQUIRED - cannot be None)

        Raises:
            ValueError: If dims is None
        """
        if dims is None:
            raise ValueError(
                "dimensions cannot be None. Scene dimensions are required and "
                "define the coordinate system for all data in the scene."
            )
        self.attrs["scene_dimensions"] = dims.to_dict()
        self._dimensions = dims

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
            self.attrs["viewer_config"] = vc.to_dict()
            if self._writer:
                self._writer.write_group("/", viewer_config=vc.to_dict())
        elif "viewer_config" in self.attrs:
            del self.attrs["viewer_config"]

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
                is a template with ``{hover_label}``, ``{hover_node}``,
                ``{hover_index}`` placeholders.
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
        from ..validation.overlays import (
            validate_anchor,
            validate_font,
            validate_position,
            validate_text_align,
            validate_transition,
            validate_visible_range,
        )
        from ..validation.overlays import (
            validate_blend_mode as validate_overlay_blend_mode,
        )

        name = self._next_overlay_name(name)
        position = validate_position(position)
        validate_anchor(anchor)
        validate_font(font)
        validate_text_align(text_align)
        validate_transition(transition)
        validate_overlay_blend_mode(blend_mode)
        validated_range = validate_visible_range(visible_range, self._dimensions.names)

        attrs: Dict[str, Any] = {
            "type": "overlay_text",
            "text": str(text),
            "position": list(position),
            "font_size": float(font_size),
            "font": font,
            "color": color,
            "opacity": float(opacity),
            "anchor": anchor,
            "text_align": text_align,
            "line_height": float(line_height),
            "padding": float(padding),
            "stroke_width": float(stroke_width),
            "transition": transition,
            "transition_duration": float(transition_duration),
            "interactive": bool(interactive),
            "z_index": len(self._overlays),
        }
        if width is not None:
            attrs["width"] = float(width)
        if background is not None:
            attrs["background"] = background
        if stroke_color is not None:
            attrs["stroke_color"] = stroke_color
        if blend_mode != "normal":
            attrs["blend_mode"] = blend_mode
        if hover:
            attrs["hover"] = True
        if validated_range is not None:
            attrs["visible_range"] = validated_range

        overlay = self._write_overlay(name, "overlay_text", position, attrs)
        aprint(
            f"✓ Text overlay '{name}' added at ({position[0]:.2f}, {position[1]:.2f})"
        )
        return overlay

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
                numpy array (HWC uint8 or float 0-1), PIL Image.
            position: (x, y) in normalized screen coords [0, 1].
            name: Optional overlay name. Auto-generated if None.
            size: Optional (width, height) as fractions of viewport dimensions.
            opacity: Opacity 0.0-1.0 (default 1.0).
            anchor: Anchor point for positioning (default 'top-left').
            blend_mode: CSS blend mode: 'normal', 'multiply', 'screen',
                'overlay', 'additive', or 'difference' (default 'normal').
            format: Image encoding format: 'png', 'jpeg', 'webp' (default 'png').
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
        from ..validation.overlays import (
            validate_anchor,
            validate_image_input,
            validate_position,
            validate_transition,
            validate_visible_range,
        )
        from ..validation.overlays import (
            validate_blend_mode as validate_overlay_blend_mode,
        )

        name = self._next_overlay_name(name)
        position = validate_position(position)
        validate_anchor(anchor)
        validate_overlay_blend_mode(blend_mode)
        validate_transition(transition)
        validated_range = validate_visible_range(visible_range, self._dimensions.names)

        image_bytes, fmt = validate_image_input(image, fmt=format)
        image_filename = f"image.{fmt}"

        attrs: Dict[str, Any] = {
            "type": "overlay_image",
            "position": list(position),
            "image_file": image_filename,
            "opacity": float(opacity),
            "anchor": anchor,
            "blend_mode": blend_mode,
            "transition": transition,
            "transition_duration": float(transition_duration),
            "interactive": bool(interactive),
            "z_index": len(self._overlays),
        }
        if size is not None:
            attrs["size"] = list(size)
        if validated_range is not None:
            attrs["visible_range"] = validated_range

        overlay = self._write_overlay(
            name,
            "overlay_image",
            position,
            attrs,
            image_data=image_bytes,
            image_filename=image_filename,
        )
        aprint(
            f"✓ Image overlay '{name}' added at ({position[0]:.2f}, {position[1]:.2f})"
        )
        return overlay

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
                ``{hover_image_label}``, ``{hover_node}``,
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
        from ..validation.overlays import (
            sanitize_html,
            validate_anchor,
            validate_position,
            validate_transition,
            validate_visible_range,
        )
        from ..validation.overlays import (
            validate_blend_mode as validate_overlay_blend_mode,
        )

        name = self._next_overlay_name(name)
        position = validate_position(position)
        validate_anchor(anchor)
        validate_transition(transition)
        validate_overlay_blend_mode(blend_mode)
        validated_range = validate_visible_range(visible_range, self._dimensions.names)

        sanitized = sanitize_html(html)

        attrs: Dict[str, Any] = {
            "type": "overlay_html",
            "position": list(position),
            "html": sanitized,
            "opacity": float(opacity),
            "anchor": anchor,
            "transition": transition,
            "transition_duration": float(transition_duration),
            "interactive": bool(interactive),
            "z_index": len(self._overlays),
        }
        if width is not None:
            attrs["width"] = float(width)
        if blend_mode != "normal":
            attrs["blend_mode"] = blend_mode
        if hover:
            attrs["hover"] = True
        if hover_image_size is not None:
            attrs["hover_image_size"] = list(hover_image_size)
        if validated_range is not None:
            attrs["visible_range"] = validated_range

        overlay = self._write_overlay(name, "overlay_html", position, attrs)
        aprint(
            f"✓ HTML overlay '{name}' added at ({position[0]:.2f}, {position[1]:.2f})"
        )
        return overlay

    # ---------------------------------------------------------- overlay internals

    def _next_overlay_name(self, name: Optional[str]) -> str:
        """Generate or validate an overlay name."""
        if name is None:
            name = f"overlay_{self._overlay_counter}"
            self._overlay_counter += 1
        else:
            if "/" in name:
                raise ValueError(f"Overlay name cannot contain '/': got '{name}'")
        # Check for duplicate names
        existing_names = {o.name for o in self._overlays}
        if name in existing_names:
            raise ValueError(
                f"Overlay name '{name}' already exists. Use a unique name."
            )
        return name

    def _write_overlay(
        self,
        name: str,
        overlay_type: str,
        position: Tuple[float, float],
        attrs: Dict[str, Any],
        image_data: Optional[bytes] = None,
        image_filename: Optional[str] = None,
    ) -> Overlay:
        """Write overlay metadata (and optional image) to the zarr store.

        Creates an ``overlays/{name}`` group with metadata in ``.zattrs``.
        For image overlays, writes the image file directly to the zarr directory.
        """
        overlay_path = f"overlays/{name}"

        # Write group with all overlay attributes
        if self._writer is not None:
            self._writer.write_group(overlay_path, **attrs)

            # Write raw image file if provided
            if image_data is not None and image_filename is not None:
                store_path = Path(self._writer.store_path)
                image_dir = store_path / "overlays" / name
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / image_filename
                image_path.write_bytes(image_data)

        overlay = Overlay(
            name=name,
            overlay_type=overlay_type,
            position=position,
            attrs=attrs,
        )
        self._overlays.append(overlay)
        return overlay

    # ---------------------------------------------------------- picking/labels

    def _notify_labels_added(self) -> None:
        """Called by Group.add_* when labels are provided on any child node."""
        self._has_labels = True

    def _notify_image_labels_added(self) -> None:
        """Called by Group.add_* when image_labels are provided on any child node."""
        self._has_image_labels = True

    def _auto_inject_hover_overlay(self) -> None:
        """Auto-inject a default hover overlay if labels/image_labels exist.

        Called by the compiler during finalize(). Checks:
        1. At least one node has labels or image_labels
        2. No existing overlay has hover=True
        3. suppress_hover_overlay is False

        When image_labels are present, uses an HTML overlay (for ``<img>`` tag).
        When only text labels exist, uses a text overlay (current behavior).
        """
        if not self._has_labels and not self._has_image_labels:
            return
        if self._suppress_hover_overlay:
            return
        # Check if user already defined a hover overlay
        if any(o.attrs.get("hover") for o in self._overlays):
            return

        has_text = self._has_labels
        has_img = self._has_image_labels

        # Inject separate overlays for image and text so they don't
        # interfere (image loading would cause layout shift in a
        # combined overlay). Both anchor top-right; demos can suppress
        # auto-injection and define custom hover overlays for
        # different layouts.
        z = len(self._overlays)

        if has_img:
            aprint("  Auto-injecting hover image overlay")
            self._write_overlay(
                name="__hover_image",
                overlay_type="overlay_html",
                position=(0.98, 0.02),
                attrs={
                    "type": "overlay_html",
                    "hover": True,
                    "html": "{hover_image_label}",
                    "position": [0.98, 0.02],
                    "anchor": "top-right",
                    "background": "rgba(0,0,0,0.7)",
                    "padding": 0.008,
                    "opacity": 1.0,
                    "transition": "fade",
                    "transition_duration": 0.15,
                    "interactive": False,
                    "z_index": z,
                },
            )
            z += 1

        if has_text:
            aprint("  Auto-injecting hover text overlay")
            self._write_overlay(
                name="__hover_text",
                overlay_type="overlay_text",
                position=(0.98, 0.02),
                attrs={
                    "type": "overlay_text",
                    "hover": True,
                    "text": "{hover_label}",
                    "position": [0.98, 0.02],
                    "anchor": "top-right",
                    "font_size": 0.018,
                    "font": "sans",
                    "color": "white",
                    "background": "rgba(0,0,0,0.7)",
                    "padding": 0.008,
                    "opacity": 1.0,
                    "transition": "fade",
                    "transition_duration": 0.15,
                    "interactive": False,
                    "z_index": z,
                },
            )

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
            path: Destination path for the copied Zarr store. The destination
                must not already exist unless it is the current backing store.

        Raises:
            FileExistsError: If ``path`` already exists and is not the current
                backing store.
            ValueError: If the destination is inside the source store or the
                source store is unavailable.
        """
        writer = self._writer
        if writer is None:
            raise ValueError("Scene has no backing writer; cannot export to Zarr")

        source = Path(self.get_store_path()).resolve()
        destination = Path(path).expanduser().resolve()

        if not source.exists() or not source.is_dir():
            raise ValueError(
                f"Scene backing store is not an existing directory: {source}"
            )

        # Same-location export is useful as an explicit finalize operation.
        if destination == source:
            aprint(f"Finalizing scene at {source}")
            writer.finalize()
            return

        if destination.exists():
            raise FileExistsError(
                f"Destination already exists: {destination}. Remove it first or choose "
                "a different path."
            )

        if destination.is_relative_to(source):
            raise ValueError(
                f"Destination {destination} cannot be inside source Zarr store {source}"
            )

        aprint(f"Exporting scene from {source} to {destination}")
        writer.finalize()
        # CL-1: atomic copy — a failure mid-copy leaves no half-written
        # zarr store at `destination`.
        atomic_copytree(source, destination)
        aprint(f"✓ Scene exported to {destination}")
