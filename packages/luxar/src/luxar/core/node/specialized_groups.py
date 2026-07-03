"""Free-function impls for `Node.add_lod_group` and `Node.add_partition_group`.

These two convenience builders live on `Node` (so any Node can spawn a
kind=lod or kind=partition Group as a child). The bodies are extracted here
to keep node.py readable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arbol import aprint

if TYPE_CHECKING:
    from ..group import Group
    from .node import Node


def add_lod_group_impl(
    node: "Node",
    name: str,
    *,
    selector: str = "coverage",
    default_level: int = 0,
    **attrs: Any,
) -> "Group":
    """Body of :meth:`Node.add_lod_group`."""
    if selector != "coverage":
        raise ValueError(
            f"selector must be 'coverage' (other modes reserved for "
            f"future use), got {selector!r}"
        )
    if default_level < 0:
        raise ValueError(f"default_level must be >= 0, got {default_level}")
    try:
        aprint(f"Adding child kind=lod group '{name}' to node '{node.name}'.")
        child = node.add_group(
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
            f"Failed to add child kind=lod group '{name}' to node '{node.name}': {e}"
        )
        raise ValueError(f"Could not create child kind=lod group '{name}': {e}") from e


def add_partition_group_impl(
    node: "Node",
    name: str,
    *,
    display_type: str,
    max_elements: int,
    **attrs: Any,
) -> "Group":
    """Body of :meth:`Node.add_partition_group`."""
    if display_type not in ("points", "lines", "gsplats"):
        raise ValueError(
            "display_type for a partition group must be one of "
            f"'points' / 'lines' / 'gsplats', got {display_type!r}"
        )
    if not isinstance(max_elements, int) or max_elements < 1:
        raise ValueError(f"max_elements must be an int >= 1, got {max_elements!r}")
    try:
        aprint(f"Adding child kind=partition group '{name}' to node '{node.name}'.")
        child = node.add_group(
            name,
            kind="partition",
            display_type=display_type,
            max_elements=max_elements,
            **attrs,
        )
        aprint(f"✓ Child kind=partition group '{name}' added successfully.")
        return child
    except Exception as e:
        aprint(
            f"Failed to add child kind=partition group '{name}' to "
            f"node '{node.name}': {e}"
        )
        raise ValueError(
            f"Could not create child kind=partition group '{name}': {e}"
        ) from e
