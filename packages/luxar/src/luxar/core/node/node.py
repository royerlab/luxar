"""luxar.node – Defines the Node class for Luxar scene graph nodes."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
from arbol import aprint

from ...typing_utils.aliases import SceneHierarchy, TransformMatrix
from ...typing_utils.constants import LEGACY_LOD_SELECTOR

if TYPE_CHECKING:
    from ...io.writer import ZarrWriterProtocol
    from ..dimensions import Dimensions
    from ..group import Group


class _WriteThroughAttrs(MutableMapping[str, Any]):
    """Mutable view of a node's cached attrs that persists every mutation."""

    _PROPERTY_KEYS = frozenset(
        {
            "absorption",
            "blending_mode",
            "colormap",
            "gamma",
            "intensity",
            "join",
            "layer",
            "nd_transform",
            "offset",
            "opacity",
            "transform",
            "visible",
        }
    )

    def __init__(self, node: Node) -> None:
        self._node = node

    def __getitem__(self, key: str) -> Any:
        return self._node._attrs_cache[key]

    def __setitem__(self, key: str, value: Any) -> None:
        descriptor = getattr(type(self._node), key, None)
        if (
            key in self._PROPERTY_KEYS
            and isinstance(descriptor, property)
            and descriptor.fset is not None
        ):
            setattr(self._node, key, value)
            return
        self._reject_mesh_only_on_non_mesh(key)
        self._reject_layer_order_inside_specialized_group(key, value)
        self._node._persist_attr(key, value)

    def _reject_layer_order_inside_specialized_group(
        self, key: str, value: Any
    ) -> None:
        """Close the second door onto ``layer_order`` inside a partition / LOD group.

        The leaf adders refuse it at authoring time, but this write-through
        mapping persists straight to the store, so
        ``part.attrs["layer_order"] = 5`` would otherwise reach disk and split a
        partition wrapper across draw-order bands — destroying the exact BSP part
        order the wrapper guarantees. Same shape and same reasoning as
        :meth:`_reject_mesh_only_on_non_mesh`.

        The walk starts at the node's parent so the partition / LOD wrapper
        itself remains a valid authoring surface while parts and levels are
        still refused.
        """
        if key != "layer_order":
            return
        from ..group.compositing import (
            _enclosing_specialized_group,
            layer_order_inside_specialized_group_reason,
        )

        wrapper = _enclosing_specialized_group(self._node.parent)
        if wrapper is None:
            return
        kind = str(wrapper.attrs.get("kind"))
        raise ValueError(
            f"Cannot set layer_order={value!r} on '{self._node.name}', which is "
            f"inside a kind={kind} group. "
            + layer_order_inside_specialized_group_reason(kind)
        )

    def _reject_mesh_only_on_non_mesh(self, key: str) -> None:
        """Close the second door onto mesh-only appearance attrs (#1782).

        The leaf adders refuse the mesh-only appearance keys on a non-mesh node
        (:func:`reject_mesh_only_appearance`), but this write-through mapping
        (#1764) persists straight to the store, so ``node.attrs["specular"] =
        0.5`` on a points/lines/gsplats/group node would otherwise reach disk and
        the viewer would silently ignore it. Node construction fills the cache
        DIRECTLY (see ``__init__``), never through here, so a mesh leaf authored
        WITH these attrs is unaffected; only a post-hoc set on a non-mesh node is
        refused. ``type`` is the geometry token every node carries in its cache
        (``"mesh"`` for a mesh leaf); anything else — including an absent one —
        is not a mesh.
        """
        from ..group.compositing import MESH_ONLY_APPEARANCE_ATTRS

        if (
            key in MESH_ONLY_APPEARANCE_ATTRS
            and self._node._attrs_cache.get("type") != "mesh"
        ):
            node_type = self._node._attrs_cache.get("type") or "non-mesh"
            raise ValueError(
                f"Cannot set mesh-only attribute '{key}' on {node_type} node "
                f"'{self._node.name}'. The viewer applies these attributes only to "
                "mesh leaves, and they do not compose through non-mesh nodes. Remove "
                "them, set them on each mesh leaf (part_<i> / child_<i>), or pass "
                "them to add_mesh(...), which stamps every generated mesh leaf."
            )

    def __delitem__(self, key: str) -> None:
        if key not in self._node._attrs_cache:
            raise KeyError(key)
        self._node._delete_attr(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._node._attrs_cache)

    def __len__(self) -> int:
        return len(self._node._attrs_cache)

    def copy(self) -> Dict[str, Any]:
        """Return a detached dict, matching the former cache-dict API."""
        return self._node._attrs_cache.copy()

    def __repr__(self) -> str:
        """Render as the dict it wraps.

        ``attrs`` used to BE the cache dict, so a print/notebook/log of it
        showed the attributes. Without this it shows a bare object address
        instead, which is a worse debugging experience than what it replaced.
        """
        return repr(self._node._attrs_cache)


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
        # Single chokepoint for node naming (rejects empty/whitespace-only,
        # '/', dot-prefixed zarr-reserved, and control-char names). An empty
        # name is the worst case: it resolves to the zarr ROOT group and
        # would clobber the scene root. ValidationError is a ValueError.
        from ...validation.base import validate_node_name

        validate_node_name(name)
        self.name: str = name
        self._writer = writer
        self.parent: Optional[Node] = parent
        self.children: List[Node] = []
        self._metadata: Dict[str, Any] = {}  # Metadata storage
        self._attrs_cache: Dict[str, Any] = {}  # Attributes cache
        self._attrs = _WriteThroughAttrs(self)

        # Determine path in hierarchy
        if parent is not None:
            parent._ensure_no_duplicate_child(name)
            # Insertion order among siblings. The viewer rebuilds the scene
            # graph from zarr consolidated metadata, whose enumeration is
            # alphabetical — so we record the add order explicitly here and
            # the loader sorts siblings by it. This keeps the layers panel in
            # napari-style addition order rather than alphabetical.
            child_index = len(parent.children)
            parent_path = getattr(parent, "path", None)
            self.path = f"{parent_path}/{name}" if parent_path else name
            # ``setdefault`` so an explicit caller-supplied value wins (and a
            # re-created node keeps its original slot).
            attrs.setdefault("child_index", child_index)
        else:
            self.path = ""

        # Initialize or merge attributes
        if attrs:
            # Validate transform if present
            if "transform" in attrs:
                try:
                    from ..transforms import prepare_transform_for_zarr

                    # Use centralized function for consistent handling
                    attrs["transform"] = prepare_transform_for_zarr(attrs["transform"])
                except Exception as e:
                    raise ValueError(f"Invalid transform for node '{name}': {e}") from e

            # Validate nd_transform if present. The scene dimensions are
            # resolved from the parent chain (or the writer's store) so a GROUP
            # gets the same store-aware check the geometry writers run on a leaf
            # (issue #1418): a key naming a dimension that does not exist — or
            # one that is displayed — is refused here rather than written clean
            # and then ignored by the viewer.
            if "nd_transform" in attrs:
                try:
                    from ...validation.nd_transforms import validate_nd_transform

                    attrs["nd_transform"] = validate_nd_transform(
                        attrs["nd_transform"], self._resolve_scene_dimensions()
                    )
                except Exception as e:
                    raise ValueError(
                        f"Invalid nd_transform for node '{name}': {e}"
                    ) from e

            # Validate rendering attributes if present
            if "opacity" in attrs:
                from ...validation.types import validate_opacity

                attrs["opacity"] = validate_opacity(attrs["opacity"])

            if "gamma" in attrs:
                from ...validation.types import validate_gamma

                attrs["gamma"] = validate_gamma(attrs["gamma"])

            if "intensity" in attrs:
                from ...validation.types import validate_intensity

                attrs["intensity"] = validate_intensity(attrs["intensity"])

            if "offset" in attrs:
                from ...validation.types import validate_offset

                attrs["offset"] = validate_offset(attrs["offset"])

            if "absorption" in attrs:
                from ...validation.types import validate_absorption

                attrs["absorption"] = validate_absorption(attrs["absorption"])

            if "blending_mode" in attrs:
                from ...validation.types import validate_blending_mode

                attrs["blending_mode"] = validate_blending_mode(attrs["blending_mode"])

            if "join" in attrs:
                from ...validation.types import validate_line_join

                attrs["join"] = validate_line_join(attrs["join"])

            if "layer" in attrs:
                from ...validation.types import validate_layer

                attrs["layer"] = validate_layer(attrs["layer"])

            if "visible" in attrs:
                from ...validation.types import validate_visible

                attrs["visible"] = validate_visible(attrs["visible"])

            if "colormap" in attrs:
                from ...validation.types import validate_colormap

                attrs["colormap"] = validate_colormap(attrs["colormap"])

            # Store attributes
            if self._writer is not None:
                # Write via writer interface and cache. ``transform`` /
                # ``nd_transform`` were already normalized above (the cache must
                # hold the column-major form for the ``transform`` getter), so
                # flag the write to keep write_group from transposing them a
                # second time (prepare_transform_attrs is not idempotent).
                # Skipping that pass stays sound because the block above ran the
                # SAME dimension-aware ``nd_transform`` check, against the
                # Dimensions resolved from the parent chain or the writer's own
                # store.
                self._writer.write_group(self.path, _transform_normalized=True, **attrs)
                # Mirror the writer's custom-colormap resolution (ndarray /
                # non-builtin name → ``"custom"`` + a sibling ``colormap_lut``
                # array) into the cache, the same way the leaf adders do for
                # the node they return: ``write_group`` mutates its own
                # ``**attrs`` copy, so without this a Group authored with an
                # ndarray LUT would report the array while zarr holds the
                # sentinel.
                from ..group.compositing import sync_custom_colormap_attr

                sync_custom_colormap_attr(attrs)
                self._attrs_cache.update(attrs)
            else:
                # Metadata-only mode (no writer available)
                self._attrs_cache.update(attrs)

        # Register as a sibling LAST, so a REFUSED node does not linger. Every
        # raise above happens before this line, so a rejected construction
        # leaves no phantom entry in ``parent.children`` and the obvious retry
        # (fix the bad attr, call the same adder again) succeeds instead of
        # failing with "Duplicate child name". ``child_index`` was captured
        # above and nothing in the attrs block appends siblings, so the recorded
        # add order is unchanged. Attr-agnostic: a bad ``opacity`` /
        # ``transform`` stranded a node the same way.
        self._register_with_parent()

    def _register_with_parent(self) -> None:
        """Append this node to ``self.parent``'s child list; no-op when detached.

        Reads ``self.parent`` rather than taking it as an argument, so it cannot
        register a node under a list that is not its own parent's. Split out of
        :meth:`__init__` so the deferred registration costs the constructor no
        extra branch (it is over the C901 ratchet already).
        """
        if self.parent is not None:
            self.parent.children.append(self)

    # ------------------------------------------------------------- dimensions
    def _resolve_scene_dimensions(self) -> Optional["Dimensions"]:
        """Resolve the scene ``Dimensions`` this node's attrs are checked against.

        Two sources, in order:

        1. The root ``Scene`` found by walking the parent chain. The non-raising
           counterpart of ``Group._find_scene`` — being unattached is a legitimate
           state here, not an error. ``Scene.__init__`` calls ``super().__init__``
           BEFORE assigning ``self._dimensions``, so the attribute can be missing
           while the root is still being constructed; hence ``getattr``.
        2. The writer's store. A node can be writer-attached but Scene-DETACHED —
           the ``parent=`` kwarg on the geometry adders is the supported route,
           and a bare ``Group("name", writer=compiler)`` reaches the same state
           (though that one is a footgun: its ``path`` is ``""``, so its children
           land at the zarr root). Either way there IS an authoritative
           ``scene_dimensions`` in the store even though the parent chain is
           empty. What makes structure-only validation sound is having no store
           to check against, not having no Scene — so consult the store before
           giving up, exactly as ``io/_compiler/node_common.prepare_transform_attrs``
           does for the leaf path. Duck-typed via ``getattr`` so lightweight
           test writers that predate the protocol's ``store`` attribute still
           degrade to structure-only validation.

        Returns:
            The scene ``Dimensions``, or None when there is genuinely no Scene
            and no store — in which case the caller validates structure only.
        """
        from ..scene import Scene

        node: Optional[Node] = self
        while node is not None:
            if isinstance(node, Scene):
                dims: Optional["Dimensions"] = getattr(node, "_dimensions", None)
                if dims is not None:
                    return dims
                break
            node = node.parent

        # A store that is not zarr-Group-shaped (a stub/mock writer) degrades to
        # structure-only validation rather than surfacing its own TypeError as
        # "your nd_transform is invalid". No zarr import here on purpose: this
        # module has no zarr dependency, so the shape is probed by use.
        store = getattr(self._writer, "store", None)
        if store is not None:
            try:
                has_dims = "scene_dimensions" in store.attrs
                raw = store.attrs["scene_dimensions"] if has_dims else None
            except (TypeError, AttributeError, KeyError):
                return None
            if raw is not None:
                from ..dimensions import Dimensions

                return Dimensions.from_dict(raw)
        return None

    # --------------------------------------------------------------------- attrs
    @property
    def attrs(self) -> _WriteThroughAttrs:
        """Get node attributes.

        Annotated with the concrete view rather than the ``GroupAttrs`` alias
        so its ``copy()`` — kept for parity with the cache dict this replaced —
        is visible to type checkers. ``_WriteThroughAttrs`` is a
        ``MutableMapping[str, Any]``, so it still satisfies
        ``NodeProtocol.attrs``.

        Returns:
            Mutable mapping backed by the cache and the zarr store.
        """
        return self._attrs

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
        if self._writer is None:
            self._attrs_cache[key] = value
            return
        # Inspect the writer's finalization state without coupling to its
        # concrete class. The `_check_not_finalized` helper raises on the
        # finalized path; sniffing the flag avoids that and lets us emit a
        # clearer warning instead.
        is_finalized = bool(getattr(self._writer, "_is_finalized", False))
        if is_finalized:
            self._attrs_cache[key] = value
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
        # ``transform`` / ``nd_transform`` values reaching this method are
        # already normalized/validated by the setters; flag the write so
        # write_group does not re-transpose them (no-op for other keys).
        self._writer.write_group(self.path, _transform_normalized=True, **{key: value})
        self._attrs_cache[key] = value

    def _delete_attr(self, key: str) -> None:
        """Remove an attribute from both the cache and the zarr store.

        The deletion counterpart of :meth:`_persist_attr`. After the writer
        has been finalized (context exit or ``Scene.to_zarr``), the on-disk
        attribute can no longer be removed through this writer — the
        consolidated metadata is sealed and editing only the raw ``.zattrs``
        would desynchronize it from the consolidated ``.zmetadata``. The
        in-memory cache is still updated so getters reflect the removal on the
        live Node, but a warning surfaces the silent persistence gap instead
        of letting the raw and consolidated views drift apart unnoticed.
        Callers that need to clear on-disk attrs after finalize should re-open
        the zarr through a fresh writer/loader.

        Args:
            key: Attribute key to remove
        """
        if key not in self._attrs_cache:
            return
        if self._writer is None:
            del self._attrs_cache[key]
            return
        # Inspect the writer's finalization state without coupling to its
        # concrete class, mirroring ``_persist_attr``. Sniffing the flag lets
        # us emit a clear warning instead of touching the sealed store.
        is_finalized = bool(getattr(self._writer, "_is_finalized", False))
        if is_finalized:
            del self._attrs_cache[key]
            import warnings

            warnings.warn(
                f"Clearing Node attribute {key!r} after the writer has been "
                "finalized. The in-memory cache is updated, but the on-disk "
                "zarr attribute is unchanged. Clear attributes inside the "
                "LuxarZarrCompiler context (or before Scene.to_zarr) for "
                "persistence.",
                UserWarning,
                stacklevel=3,
            )
            return
        self._writer.delete_group_attr(self.path, key)
        del self._attrs_cache[key]

    # --------------------------------------------------------------- hierarchy
    def _ensure_no_duplicate_child(self, name: str) -> None:
        """Raise if a child with ``name`` already exists under this node.

        Used by ``Node.__init__`` (post-write belt and braces) AND by the
        geometry adders as a fail-fast PRE-write gate: without the pre-check,
        a duplicate ``add_points('a', ...)`` would first overwrite the
        existing node's arrays on disk and only then raise here.

        Raises:
            ValueError: If a child with the same name exists.
        """
        for existing_child in self.children:
            if existing_child.name == name:
                raise ValueError(
                    f"Duplicate child name '{name}' under parent '{self.name}'."
                )

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
        from ..group import Group
        from ..group.compositing import (
            reject_layer_order_inside_specialized_group,
            reject_mesh_only_appearance,
        )

        try:
            aprint(f"Adding child group '{name}' to node '{self.name}'.")

            reject_mesh_only_appearance("group", name, attrs)
            reject_layer_order_inside_specialized_group("group", name, attrs, self)

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
        selector: str = LEGACY_LOD_SELECTOR,
        default_level: int = 0,
        **attrs: Any,
    ) -> "Group":
        """Create and add a child kind=lod ``Group`` node.

        A kind=lod ``Group`` picks one of N alternative children at runtime by
        comparing the group's on-screen size against each child's
        ``coverage_fraction`` threshold; ``selector`` names the UNITS of those
        thresholds. Under ``selector="screen-area"`` (what every auto-derived
        ladder stamps) a threshold is a literal screen-area fraction — the
        node's projected bbox rect area over the viewport area — so a derived
        whole-object ladder reads ``[0, …, 1/4, 1/2]`` (full detail while the
        node occupies at least half the screen; one level coarser per halving
        of occupied area) and a partition tile anchors at ``1.0``
        (fills-screen). The default ``selector="coverage"`` is the legacy
        diagonal metric (projected bbox diagonal over half the fitted screen
        axis, bounded by ``MAX_COVERAGE_FRACTION`` = 4.0), kept for
        hand-authored ladders and existing datasets whose values were tuned in
        those units. Children are added via the inherited ``add_*`` methods on
        the returned ``Group``, each carrying a ``coverage_fraction`` attribute,
        in strictly increasing order (coarsest ``0.0`` first). The resolved
        ``display_type`` of the finest child becomes the group's user-facing
        geometry type.

        Example::

            lod = scene.add_lod_group("multires")
            lod.add_gsplats_from_data("c", coarse, coverage_fraction=0.0)
            lod.add_gsplats_from_data("m", medium, coverage_fraction=0.5)
            lod.add_gsplats_from_data("f", fine, coverage_fraction=1.0)

        Args:
            name: Name of the lod-kind group.
            selector: Units of the children's ``coverage_fraction`` thresholds:
                ``"coverage"`` (legacy diagonal metric, the default for
                hand-built ladders) or ``"screen-area"`` (literal screen-area
                fractions — what derived ladders use).
            default_level: Initial active level index for the
                manual-override UI (0-based).
            **attrs: Additional node attributes (transform, layer, etc.).

        Returns:
            The created ``Group`` (with ``kind="lod"`` in its attrs).

        Raises:
            ValueError: If selector mode is unknown or group creation fails.
        """
        from .specialized_groups import add_lod_group_impl

        return add_lod_group_impl(
            self,
            name,
            selector=selector,
            default_level=default_level,
            **attrs,
        )

    def add_partition_group(
        self,
        name: str,
        *,
        display_type: str,
        max_elements: int,
        **attrs: Any,
    ) -> "Group":
        """Create and add a child kind=partition ``Group`` node.

        A kind=partition ``Group`` is a compile-time decomposition of a single
        large geometry node into multiple smaller children for per-child
        frustum culling and per-child LOD. The user does not see the
        decomposition — the layers panel presents one logical layer of
        ``display_type``.

        For the common case where you want the partitioning to happen
        automatically, use the ``partition=`` convenience kwarg on
        ``add_points`` / ``add_lines`` / ``add_gsplats`` instead of
        constructing the wrapper yourself.

        Args:
            name: Name of the partition-kind group.
            display_type: Geometry type the user sees this layer as
                (``"points"``, ``"lines"``, or ``"gsplats"``). All children
                must resolve to this same display type — homogeneity is
                mandatory for a partition.
            max_elements: Cap that drove the BSP recursion (recorded on the
                group for diagnostics and for future partition-aware tools).
            **attrs: Additional node attributes.

        Returns:
            The created ``Group`` (with ``kind="partition"`` in its attrs).
        """
        from .specialized_groups import add_partition_group_impl

        return add_partition_group_impl(
            self,
            name,
            display_type=display_type,
            max_elements=max_elements,
            **attrs,
        )

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

            from ...typing_utils.protocols import NodeProtocol

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
            from ..transforms import read_transform_from_zarr

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
            self._delete_attr("transform")
        else:
            from ..transforms import prepare_transform_for_zarr

            self._persist_attr("transform", prepare_transform_for_zarr(matrix))

    @property
    def world_transform(self) -> TransformMatrix:
        """Get the world transformation matrix by composing all parent transforms.

        Walks up the parent chain, collecting local transforms, and composes
        them root-outermost: this node's own transform is applied first
        (innermost) and the root transform last (outermost), i.e.
        ``world = root @ ... @ leaf``.

        Returns:
            4x4 world transformation matrix. Identity if no transforms are set.
        """
        from ..transforms import compose, identity

        transforms = []
        node: Optional[Node] = self
        while node is not None:
            if node.transform is not None:
                transforms.append(node.transform)
            node = node.parent
        if not transforms:
            return identity()
        # transforms is leaf→root; compose applies its first arg first, so this
        # applies the leaf (this node) first (innermost) and yields root @ ... @ leaf
        return compose(*transforms)

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

        Validated against the scene dimensions — resolved from the parent chain
        or, failing that, the writer's store — on EVERY node type, leaves
        included (issue #1418): an unknown or displayed dimension name is
        refused rather than persisted and then ignored by the viewer. Only a
        node with neither a Scene nor a store falls back to structure-only
        validation.

        Args:
            value: Dict mapping dim names to transform entries, or None to remove.

        Raises:
            ValueError: If nd_transform is invalid
        """
        if value is None:
            self._delete_attr("nd_transform")
        else:
            from ...validation.nd_transforms import validate_nd_transform

            self._persist_attr(
                "nd_transform",
                validate_nd_transform(value, self._resolve_scene_dimensions()),
            )

    @property
    def world_nd_transform(self) -> Dict[str, Any]:
        """Get the composed world nD transform by walking the parent chain.

        Returns:
            Composed nD transform dict. Empty dict means identity.
        """
        from ...validation.nd_transforms import compose_nd_transforms

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
        from ...validation.types import validate_layer

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
        from ...validation.types import validate_visible

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
        from ...validation.types import validate_opacity

        self._persist_attr("opacity", validate_opacity(value))

    @property
    def absorption(self) -> float:
        """Get the absorption coefficient (volumetric kappa) for this node.

        Only read by the ``volumetric`` blending mode: it scales how strongly
        this node's content attenuates what is behind it (kappa = 0 renders
        exactly like ``additive``). Composes multiplicatively down the scene
        graph with identity 1.0, like opacity.

        Returns:
            Absorption coefficient (>= 0), defaults to 1.0 if not set
        """
        return float(self.attrs.get("absorption", 1.0))

    @absorption.setter
    def absorption(self, value: Any) -> None:
        """Set the absorption coefficient (volumetric kappa) for this node.

        Changes are persisted to zarr immediately if a writer is available.

        Args:
            value: Absorption coefficient (>= 0, finite)

        Raises:
            ValueError: If absorption is negative, NaN, or infinite
            TypeError: If absorption cannot be converted to float
        """
        from ...validation.types import validate_absorption

        self._persist_attr("absorption", validate_absorption(value))

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
        from ...validation.types import validate_gamma

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
        from ...validation.types import validate_intensity

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
        from ...validation.types import validate_offset

        self._persist_attr("offset", validate_offset(value))

    @property
    def blending_mode(self) -> str:
        """Get the blending mode for this node.

        Returns:
            Blending mode string, defaults to "additive" if not set.
            Valid modes: "normal", "additive", "max", "opaque", "luminous",
            "volumetric"
        """
        from ...typing_utils.constants import DEFAULT_BLENDING_MODE

        return str(self.attrs.get("blending_mode", DEFAULT_BLENDING_MODE))

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
                - "volumetric": Emission-absorption — adds light AND absorbs
                  what's behind, scaled by the ``absorption`` (kappa) attr

        Raises:
            ValueError: If blending mode is not valid
            TypeError: If blending mode is not a string
        """
        from ...validation.types import validate_blending_mode

        self._persist_attr("blending_mode", validate_blending_mode(value))

    @property
    def join(self) -> Optional[str]:
        """Get the line join style for this node (issue #790).

        Lines-only: the strategy the line vertex stage uses at a degree-2
        polyline joint. Compositing, so it may equally be set on a wrapper
        Group, from where it flows down to the lines descendants.

        Unlike :attr:`blending_mode` this does NOT substitute a default when
        unset — it returns ``None``. The default belongs to the viewer, where a
        ``?lineJoin=`` override can still win over it; reporting one here would
        invite writing it back and freezing today's default into the file.

        Returns:
            ``"none"`` or ``"miter"`` if set, None otherwise
        """
        value = self.attrs.get("join")
        return str(value) if value is not None else None

    @join.setter
    def join(self, value: Any) -> None:
        """Set the line join style for this node.

        Changes are persisted to zarr immediately if a writer is available.

        Args:
            value: Join style string:
                - "none": leave the two segment quads alone, so a turn leaves an
                  uncovered wedge outside the bend and a double-covered lens inside
                - "miter": rotate each quad's end edge onto the shared miter edge
                  so the two tile — exact, and correct under every blending mode

        Raises:
            ValueError: If the join style is not valid
            TypeError: If the join style is not a string
        """
        from ...validation.types import validate_line_join

        self._persist_attr("join", validate_line_join(value))

    def set_opacity(self, value: Any) -> "Node":
        """Set opacity and return self for chaining.

        Args:
            value: Opacity value (0.0 to 1.0)

        Returns:
            Self for method chaining
        """
        self.opacity = value
        return self

    def set_absorption(self, value: Any) -> "Node":
        """Set absorption (volumetric kappa) and return self for chaining.

        Args:
            value: Absorption coefficient (>= 0, finite)

        Returns:
            Self for method chaining
        """
        self.absorption = value
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
                - "volumetric": Emission-absorption — adds light AND absorbs
                  what's behind, scaled by the ``absorption`` (kappa) attr

        Returns:
            Self for method chaining
        """
        self.blending_mode = value
        return self

    def set_join(self, value: Any) -> "Node":
        """Set the line join style and return self for chaining.

        Args:
            value: Join style string ("none" or "miter")

        Returns:
            Self for method chaining
        """
        self.join = value
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
        from ...validation.types import validate_colormap

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
