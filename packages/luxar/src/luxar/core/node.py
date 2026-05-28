"""luxar.node – Defines the Node class for Luxar scene graph nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from arbol import aprint

from ..typing_utils.aliases import GroupAttrs, SceneHierarchy, TransformMatrix

if TYPE_CHECKING:
    from ..core.group import Group
    from ..io.writer import ZarrWriterProtocol


class Node:
    """A node in the Luxar scene graph.

    This class represents a single node in the hierarchical scene graph structure.
    Nodes are lightweight metadata containers that write data immediately through
    the writer interface without keeping Zarr groups in memory.

    Args:
        name: Name of the node
        parent: Parent node in the hierarchy
        writer: Writer interface for progressive writing
        **attrs: Additional attributes for the node
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Node] = None,
        writer: Optional[ZarrWriterProtocol] = None,
        **attrs: Any,
    ) -> None:
        """Initialize a scene graph node.

        Args:
            name: Name of the node
            parent: Parent node in the scene hierarchy
            writer: Writer interface for progressive data writing
            **attrs: Additional attributes to set on the node
        """
        if name and "/" in name:
            raise ValueError(
                f"Node name cannot contain '/': got '{name}'. "
                f"Use add_group() to create hierarchical structure instead."
            )
        self.name: str = name
        self._writer = writer
        self.parent: Optional[Node] = parent
        self.children: List[Node] = []
        self._metadata: Dict[str, Any] = {}  # Metadata storage
        self._attrs_cache: Dict[str, Any] = {}  # Attributes cache

        # Determine path in hierarchy
        if parent is not None:
            for existing_child in parent.children:
                if existing_child.name == name:
                    raise ValueError(
                        f"Duplicate child name '{name}' under parent '{parent.name}'."
                    )
            parent.children.append(self)
            parent_path = getattr(parent, "path", None)
            self.path = f"{parent_path}/{name}" if parent_path else name
        else:
            self.path = ""

        # Initialize or merge attributes
        if attrs:
            # Validate transform if present
            if "transform" in attrs:
                try:
                    from ..core.transforms import prepare_transform_for_zarr

                    # Use centralized function for consistent handling
                    attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])
                except Exception as e:
                    raise ValueError(f"Invalid transform for node '{name}': {e}") from e

            # Validate nd_transform if present
            if "nd_transform" in attrs:
                try:
                    from ..validation.nd_transforms import validate_nd_transform

                    attrs["nd_transform"] = validate_nd_transform(attrs["nd_transform"])
                except Exception as e:
                    raise ValueError(
                        f"Invalid nd_transform for node '{name}': {e}"
                    ) from e

            # Validate rendering attributes if present
            if "opacity" in attrs:
                from ..validation.types import validate_opacity

                attrs["opacity"] = validate_opacity(attrs["opacity"])

            if "gamma" in attrs:
                from ..validation.types import validate_gamma

                attrs["gamma"] = validate_gamma(attrs["gamma"])

            if "intensity" in attrs:
                from ..validation.types import validate_intensity

                attrs["intensity"] = validate_intensity(attrs["intensity"])

            if "offset" in attrs:
                from ..validation.types import validate_offset

                attrs["offset"] = validate_offset(attrs["offset"])

            if "blending_mode" in attrs:
                from ..validation.types import validate_blending_mode

                attrs["blending_mode"] = validate_blending_mode(attrs["blending_mode"])

            if "layer" in attrs:
                from ..validation.types import validate_layer

                attrs["layer"] = validate_layer(attrs["layer"])

            if "visible" in attrs:
                from ..validation.types import validate_visible

                attrs["visible"] = validate_visible(attrs["visible"])

            if "colormap" in attrs:
                from ..validation.types import validate_colormap

                attrs["colormap"] = validate_colormap(attrs["colormap"])

            # Store attributes
            if self._writer is not None:
                # Write via writer interface and cache
                self._writer.write_group(self.path, **attrs)
                self._attrs_cache.update(attrs)
            else:
                # Metadata-only mode (no writer available)
                self._attrs_cache.update(attrs)

    # --------------------------------------------------------------------- attrs
    @property
    def attrs(self) -> GroupAttrs:
        """Get node attributes.

        Returns:
            Dictionary of node attributes from cache
        """
        return self._attrs_cache

    def _persist_attr(self, key: str, value: Any) -> None:
        """Update an attribute in both the cache and the zarr store.

        After the writer has been finalized (context exit or
        ``Scene.to_zarr``), the on-disk attribute can no longer be updated
        through this writer — the consolidated metadata is sealed. The
        in-memory cache is still updated so getters reflect the new value
        on the live Node, but a warning surfaces the silent persistence gap
        rather than letting the disk and memory drift apart unnoticed.
        Callers that need to update on-disk attrs after finalize should
        re-open the zarr through a fresh writer/loader.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self._attrs_cache[key] = value
        if self._writer is None:
            return
        # Inspect the writer's finalization state without coupling to its
        # concrete class. The `_check_not_finalized` helper raises on the
        # finalized path; sniffing the flag avoids that and lets us emit a
        # clearer warning instead.
        is_finalized = bool(getattr(self._writer, "_is_finalized", False))
        if is_finalized:
            import warnings

            warnings.warn(
                f"Setting Node attribute {key!r} after the writer has been "
                "finalized. The in-memory cache is updated, but the on-disk "
                "zarr attribute is unchanged. Set attributes inside the "
                "LuxarZarrCompiler context (or before Scene.to_zarr) for "
                "persistence.",
                UserWarning,
                stacklevel=3,
            )
            return
        self._writer.write_group(self.path, **{key: value})

    # --------------------------------------------------------------- hierarchy
    def add_group(self, name: str, **attrs: Any) -> "Group":
        """Create and add a child group node.

        The returned Group has add_points(), add_lines(), and add_gsplats()
        methods for adding data children directly.

        Args:
            name: Name of the child group
            **attrs: Additional attributes for the group

        Returns:
            The created Group node

        Raises:
            ValueError: If group creation fails
        """
        from .group import Group

        try:
            aprint(f"Adding child group '{name}' to node '{self.name}'.")

            # Duplicate child check and writing to storage, attr validation,
            # and caching are all handled by Node.__init__.
            attrs["type"] = "group"

            if self._writer is not None:
                child_node = Group(name, parent=self, writer=self._writer, **attrs)
            else:
                child_node = Group(name, parent=self, **attrs)

            aprint(f"✓ Child group '{name}' added successfully.")
            return child_node
        except Exception as e:
            aprint(f"Failed to add child group '{name}' to node '{self.name}': {e}")
            raise ValueError(f"Could not create child group '{name}': {e}") from e

    def add_lod_group(
        self,
        name: str,
        *,
        selector: str = "pixel_size",
        default_level: int = 0,
        **attrs: Any,
    ) -> "Group":
        """Create and add a child kind=lod ``Group`` node.

        A kind=lod ``Group`` picks one of N alternative children at runtime
        based on the projected bbox diagonal in pixels and each child's
        ``min_pixel_size`` threshold. Children are added via the inherited
        ``add_*`` methods on the returned ``Group`` and each must carry a
        ``min_pixel_size`` attribute. Children must be added in strictly
        increasing ``min_pixel_size`` order; the resolved ``display_type``
        of the finest child becomes the group's user-facing geometry type.

        Example::

            lod = scene.add_lod_group("multires")
            lod.add_gsplats_from_data("c", coarse, min_pixel_size=0)
            lod.add_gsplats_from_data("m", medium, min_pixel_size=100)
            lod.add_gsplats_from_data("f", fine, min_pixel_size=500)

        Args:
            name: Name of the lod-kind group.
            selector: Selector mode. Currently only ``"pixel_size"`` is
                supported.
            default_level: Initial active level index for the
                manual-override UI (0-based).
            **attrs: Additional node attributes (transform, layer, etc.).

        Returns:
            The created ``Group`` (with ``kind="lod"`` in its attrs).

        Raises:
            ValueError: If selector mode is unknown or group creation fails.
        """
        if selector != "pixel_size":
            raise ValueError(
                f"selector must be 'pixel_size' (other modes reserved for "
                f"future use), got {selector!r}"
            )
        if default_level < 0:
            raise ValueError(f"default_level must be >= 0, got {default_level}")
        try:
            aprint(f"Adding child kind=lod group '{name}' to node '{self.name}'.")
            child = self.add_group(
                name,
                kind="lod",
                selector=selector,
                default_level=int(default_level),
                **attrs,
            )
            aprint(f"✓ Child kind=lod group '{name}' added successfully.")
            return child
        except Exception as e:
            aprint(
                f"Failed to add child kind=lod group '{name}' to node "
                f"'{self.name}': {e}"
            )
            raise ValueError(
                f"Could not create child kind=lod group '{name}': {e}"
            ) from e

    def add_split_group(
        self,
        name: str,
        *,
        display_type: str,
        max_elements: int,
        **attrs: Any,
    ) -> "Group":
        """Create and add a child kind=split ``Group`` node.

        A kind=split ``Group`` is a compile-time decomposition of a single
        large geometry node into multiple smaller children for per-child
        frustum culling and per-child LOD. The user does not see the
        decomposition — the layers panel presents one logical layer of
        ``display_type``.

        For the common case where you want the splitting to happen
        automatically, use the ``split=`` convenience kwarg on
        ``add_points`` / ``add_lines`` / ``add_gsplats`` instead of
        constructing the wrapper yourself.

        Args:
            name: Name of the split-kind group.
            display_type: Geometry type the user sees this layer as
                (``"points"``, ``"lines"``, or ``"gsplats"``). All children
                must resolve to this same display type — homogeneity is
                mandatory for Split.
            max_elements: Cap that drove the BSP recursion (recorded on the
                group for diagnostics and for future split-aware tools).
            **attrs: Additional node attributes.

        Returns:
            The created ``Group`` (with ``kind="split"`` in its attrs).
        """
        if display_type not in ("points", "lines", "gsplats"):
            raise ValueError(
                "display_type for a split group must be one of "
                f"'points' / 'lines' / 'gsplats', got {display_type!r}"
            )
        if not isinstance(max_elements, int) or max_elements < 1:
            raise ValueError(
                f"max_elements must be an int >= 1, got {max_elements!r}"
            )
        try:
            aprint(
                f"Adding child kind=split group '{name}' to node '{self.name}'."
            )
            child = self.add_group(
                name,
                kind="split",
                display_type=display_type,
                max_elements=max_elements,
                **attrs,
            )
            aprint(f"✓ Child kind=split group '{name}' added successfully.")
            return child
        except Exception as e:
            aprint(
                f"Failed to add child kind=split group '{name}' to node "
                f"'{self.name}': {e}"
            )
            raise ValueError(
                f"Could not create child kind=split group '{name}': {e}"
            ) from e

    # --------------------------------------------------------------- traversal
    def walk(self, depth: int = 0) -> SceneHierarchy:
        """Walk the node hierarchy depth-first.

        Args:
            depth: Current depth in the hierarchy (used for indentation)

        Yields:
            Tuple of (depth, node) for each node in the hierarchy

        Raises:
            ValueError: If traversal encounters an error
        """
        try:
            aprint(f"Walking node hierarchy from '{self.name}' at depth {depth}.")
            # Cast self to NodeProtocol to satisfy type checker
            from typing import cast

            from ..typing_utils.protocols import NodeProtocol

            yield depth, cast(NodeProtocol, self)
            for child in self.children:
                yield from child.walk(depth + 1)
        except Exception as e:
            aprint(f"Failed to walk node hierarchy from '{self.name}': {e}")
            raise ValueError(
                f"Could not traverse hierarchy from '{self.name}': {e}"
            ) from e

    # --------------------------------------------------------------- properties
    @property
    def transform(self) -> Optional[TransformMatrix]:
        """Get the transformation matrix for this node.

        Returns:
            4x4 transformation matrix if set, None otherwise
        """
        if "transform" in self.attrs:
            from ..core.transforms import read_transform_from_zarr

            transform_list = self.attrs["transform"]
            return read_transform_from_zarr(transform_list)
        return None

    @transform.setter
    def transform(
        self, matrix: Optional[Union[TransformMatrix, np.ndarray, list]]
    ) -> None:
        """Set the transformation matrix for this node.

        Changes are persisted to zarr immediately if a writer is available.

        Args:
            matrix: 4x4 transformation matrix, flat list of 16 values, or None to remove

        Raises:
            ValueError: If transform is invalid
        """
        if matrix is None:
            # Remove transform from cache and zarr store
            if "transform" in self._attrs_cache:
                del self._attrs_cache["transform"]
                if self._writer is not None:
                    self._writer.delete_group_attr(self.path, "transform")
        else:
            from ..core.transforms import prepare_transform_for_zarr

            self._persist_attr("transform", prepare_transform_for_zarr(matrix))

    @property
    def world_transform(self) -> TransformMatrix:
        """Get the world transformation matrix by composing all parent transforms.

        Walks up the parent chain, collecting local transforms, and composes
        them in order (root first, this node last).

        Returns:
            4x4 world transformation matrix. Identity if no transforms are set.
        """
        from ..core.transforms import compose, identity

        transforms = []
        node: Optional[Node] = self
        while node is not None:
            if node.transform is not None:
                transforms.append(node.transform)
            node = node.parent
        if not transforms:
            return identity()
        # Reverse so root transform is first (applied first)
        return compose(*reversed(transforms))

    # --------------------------------------------------------- nd_transform
    @property
    def nd_transform(self) -> Optional[Dict[str, Any]]:
        """Get the nD transform for non-displayed dimensions.

        Returns:
            Dict mapping dimension names to per-dim transforms, or None.
            Affine entries: {"scale": float, "offset": float}
            Permutation entries: {"permutation": [int, ...]}
        """
        return self.attrs.get("nd_transform")

    @nd_transform.setter
    def nd_transform(self, value: Optional[Dict[str, Any]]) -> None:
        """Set the nD transform for non-displayed dimensions.

        Args:
            value: Dict mapping dim names to transform entries, or None to remove.

        Raises:
            ValueError: If nd_transform is invalid
        """
        if value is None:
            if "nd_transform" in self._attrs_cache:
                del self._attrs_cache["nd_transform"]
                if self._writer is not None:
                    self._writer.delete_group_attr(self.path, "nd_transform")
        else:
            from ..validation.nd_transforms import validate_nd_transform

            self._persist_attr("nd_transform", validate_nd_transform(value))

    @property
    def world_nd_transform(self) -> Dict[str, Any]:
        """Get the composed world nD transform by walking the parent chain.

        Returns:
            Composed nD transform dict. Empty dict means identity.
        """
        from ..validation.nd_transforms import compose_nd_transforms

        nd_transforms = []
        node: Optional[Node] = self
        while node is not None:
            if node.nd_transform is not None:
                nd_transforms.append(node.nd_transform)
            node = node.parent
        if not nd_transforms:
            return {}
        # Reverse so root is first (applied outermost)
        return compose_nd_transforms(*reversed(nd_transforms))

    @property
    def num_children(self) -> int:
        """Get the number of direct children of this node."""
        return len(self.children)

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children)."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Check if this node is the root (has no parent)."""
        return self.parent is None

    # --------------------------------------------------------------- layer flag
    @property
    def layer(self) -> bool:
        """Whether this node is exposed as a layer in the viewer's Layers panel.

        Returns:
            True if this node should appear in the Layers panel, False otherwise
        """
        return bool(self.attrs.get("layer", False))

    @layer.setter
    def layer(self, value: Any) -> None:
        """Set the layer flag for this node.

        Changes are persisted to zarr immediately if a writer is available.

        Args:
            value: Boolean layer flag. When True, the node appears as an
                entry in the viewer's Layers panel.

        Raises:
            TypeError: If value is not a boolean-compatible value.
        """
        from ..validation.types import validate_layer

        self._persist_attr("layer", validate_layer(value))

    # --------------------------------------------------------- visibility
    @property
    def visible(self) -> bool:
        """Initial visibility in the viewer (default True).

        This controls whether the layer starts visible when the scene
        loads. Authoring-time property only — runtime toggling is done
        from the viewer's Layers panel (eye icon).

        Returns:
            True if the layer should start visible, False to start hidden.
        """
        return bool(self.attrs.get("visible", True))

    @visible.setter
    def visible(self, value: Any) -> None:
        """Set the initial visibility for this node.

        Changes are persisted to zarr immediately if a writer is available.

        Args:
            value: Boolean visibility flag.

        Raises:
            TypeError: If value is not a boolean-compatible value.
        """
        from ..validation.types import validate_visible

        self._persist_attr("visible", validate_visible(value))

    # --------------------------------------------------------------- rendering
    @property
    def opacity(self) -> float:
        """Get the opacity value for this node.

        Returns:
            Opacity value (0.0 to 1.0), defaults to 1.0 if not set
        """
        return float(self.attrs.get("opacity", 1.0))

    @opacity.setter
    def opacity(self, value: Any) -> None:
        """Set the opacity value for this node.

        Changes are persisted to zarr immediately if a writer is available.

        Args:
            value: Opacity value (0.0 to 1.0)

        Raises:
            ValueError: If opacity is not in valid range
            TypeError: If opacity cannot be converted to float
        """
        from ..validation.types import validate_opacity

        self._persist_attr("opacity", validate_opacity(value))

    @property
    def gamma(self) -> float:
        """Get the gamma value for this node.

        Returns:
            Gamma value (0.1 to 10.0), defaults to 1.0 if not set
        """
        return float(self.attrs.get("gamma", 1.0))

    @gamma.setter
    def gamma(self, value: Any) -> None:
        """Set the gamma value for this node.

        Changes are persisted to zarr immediately if a writer is available.

        Args:
            value: Gamma value (0.1 to 10.0)

        Raises:
            ValueError: If gamma is not in valid range
            TypeError: If gamma cannot be converted to float
        """
        from ..validation.types import validate_gamma

        self._persist_attr("gamma", validate_gamma(value))

    @property
    def intensity(self) -> float:
        """Get the intensity value for this node.

        Returns:
            Intensity value (0.0 to 100.0), defaults to 1.0 if not set
        """
        return float(self.attrs.get("intensity", 1.0))

    @intensity.setter
    def intensity(self, value: Any) -> None:
        """Set the intensity value for this node.

        Changes are persisted to zarr immediately if a writer is available.

        Args:
            value: Intensity value (0.0 to 100.0)

        Raises:
            ValueError: If intensity is not in valid range
            TypeError: If intensity cannot be converted to float
        """
        from ..validation.types import validate_intensity

        self._persist_attr("intensity", validate_intensity(value))

    @property
    def offset(self) -> float:
        """Get the offset value for this node.

        Returns:
            Offset value (-10.0 to 10.0), defaults to 0.0 if not set
        """
        return float(self.attrs.get("offset", 0.0))

    @offset.setter
    def offset(self, value: Any) -> None:
        """Set the offset value for this node.

        Changes are persisted to zarr immediately if a writer is available.

        Args:
            value: Offset value (-10.0 to 10.0)

        Raises:
            ValueError: If offset is not in valid range
            TypeError: If offset cannot be converted to float
        """
        from ..validation.types import validate_offset

        self._persist_attr("offset", validate_offset(value))

    @property
    def blending_mode(self) -> str:
        """Get the blending mode for this node.

        Returns:
            Blending mode string, defaults to "additive" if not set.
            Valid modes: "normal", "additive", "max", "opaque", "luminous"
        """
        return str(self.attrs.get("blending_mode", "additive"))

    @blending_mode.setter
    def blending_mode(self, value: Any) -> None:
        """Set the blending mode for this node.

        Changes are persisted to zarr immediately if a writer is available.

        Args:
            value: Blending mode string. Valid modes:
                - "normal": Standard alpha blending (semi-transparent)
                - "additive": Classic additive blending, ignores depth (renders on top)
                - "max": Maximum of source and destination (brightest wins)
                - "opaque": Solid rendering with depth write (closest wins)
                - "luminous": Same as additive visually, but respects depth occlusion

        Raises:
            ValueError: If blending mode is not valid
            TypeError: If blending mode is not a string
        """
        from ..validation.types import validate_blending_mode

        self._persist_attr("blending_mode", validate_blending_mode(value))

    def set_opacity(self, value: Any) -> "Node":
        """Set opacity and return self for chaining.

        Args:
            value: Opacity value (0.0 to 1.0)

        Returns:
            Self for method chaining
        """
        self.opacity = value
        return self

    def set_gamma(self, value: Any) -> "Node":
        """Set gamma and return self for chaining.

        Args:
            value: Gamma value (0.1 to 10.0)

        Returns:
            Self for method chaining
        """
        self.gamma = value
        return self

    def set_intensity(self, value: Any) -> "Node":
        """Set intensity and return self for chaining.

        Args:
            value: Intensity value (0.0 to 100.0)

        Returns:
            Self for method chaining
        """
        self.intensity = value
        return self

    def set_offset(self, value: Any) -> "Node":
        """Set offset and return self for chaining.

        Args:
            value: Offset value (-10.0 to 10.0)

        Returns:
            Self for method chaining
        """
        self.offset = value
        return self

    def set_blending_mode(self, value: Any) -> "Node":
        """Set blending mode and return self for chaining.

        Args:
            value: Blending mode string. Valid modes:
                - "normal": Standard alpha blending (semi-transparent)
                - "additive": Classic additive blending, ignores depth (renders on top)
                - "max": Maximum of source and destination (brightest wins)
                - "opaque": Solid rendering with depth write (closest wins)
                - "luminous": Same as additive visually, but respects depth occlusion

        Returns:
            Self for method chaining
        """
        self.blending_mode = value
        return self

    @property
    def colormap(self) -> Optional[Any]:
        """Get the colormap for this node.

        Returns:
            Colormap name (str) or LUT array, or None if not set.
        """
        return self.attrs.get("colormap")

    @colormap.setter
    def colormap(self, value: Any) -> None:
        """Set the colormap for this node.

        Changes are persisted to zarr immediately if a writer is available.
        Only string colormap names are supported here. Custom array colormaps
        must be set at node creation time via add_points/add_lines/add_gsplats.

        Args:
            value: Colormap name (str).

        Raises:
            TypeError: If value is not a string (numpy arrays cannot be
                persisted via the property setter — use creation-time API).
        """
        if not isinstance(value, str):
            raise TypeError(
                "Colormap property setter only accepts string names. "
                "Custom array colormaps must be set at node creation time "
                "via add_points(..., colormap=array)."
            )
        from ..validation.types import validate_colormap

        self._persist_attr("colormap", validate_colormap(value))

    def set_colormap(self, value: str) -> "Node":
        """Set colormap and return self for chaining.

        Only string colormap names are supported after creation.
        Custom array colormaps must be set at node creation time.

        Args:
            value: Colormap name (str).

        Returns:
            Self for method chaining
        """
        self.colormap = value
        return self

    # --------------------------------------------------------------- equality
    def _root_id(self) -> int:
        """Return id() of the root node in this hierarchy.

        Used to distinguish nodes with the same path in different scenes.
        """
        node: Optional[Node] = self
        while node is not None:
            if node.parent is None:
                return id(node)
            node = node.parent
        return id(self)  # pragma: no cover — unreachable, satisfies type checker

    def __eq__(self, other: object) -> bool:
        """Equality based on path AND root identity in the scene graph.

        Nodes from different scenes with the same path are NOT equal.
        """
        if not isinstance(other, Node):
            return NotImplemented
        # Root nodes (empty path) use identity to avoid all roots comparing equal
        if self.path == "" and other.path == "":
            return self is other
        return self.path == other.path and self._root_id() == other._root_id()

    def __hash__(self) -> int:
        """Hash based on path AND root identity in the scene graph.

        Nodes from different scenes with the same path hash differently.
        """
        # Root nodes use identity hash to avoid all roots hashing identically
        if self.path == "":
            return id(self)
        return hash((self.path, self._root_id()))

    # --------------------------------------------------------------- repr
    def __repr__(self) -> str:  # pragma: no cover
        """String representation of the node.

        Returns:
            Human-readable string representation of the node
        """
        node_type = self.attrs.get("type", "unknown")
        return f"<{self.__class__.__name__} '{self.name}' ({node_type}) with {len(self.children)} children>"
