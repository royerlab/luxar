"""Free-function impls for `Node.add_lod_group` and `Node.add_partition_group`.

These two convenience builders live on `Node` (so any Node can spawn a
kind=lod or kind=partition Group as a child). The bodies are extracted here
to keep node.py readable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arbol import aprint

from ...typing_utils.geometry_capabilities import (
    partition_capable_types,
    require_lod_display_type,
    supports_partition,
)

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
    # An explicit display_type= rides in through **attrs (it is an accepted node
    # attr, so validate_render_attrs passes it through). Gate it here: the
    # finalize back-fill only ever sees groups that DIDN'T supply one, so
    # without this the caller-supplied route is the one hole left open.
    if "display_type" in attrs:
        require_lod_display_type(attrs["display_type"], f"kind=lod group {name!r}")
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
    # DO NOT widen to GEOMETRY_TYPES. This asks a CAPABILITY question — which
    # geometry types have a spatial-partition path — not the leaf-vocabulary
    # question. The two happen to agree today (all four types are partitionable
    # since mesh gained `luxar.mesh.split`), but they are different questions and
    # a future type may answer them differently. The answer lives in one table so
    # it cannot drift between here and the LOD guard; a new type opts in by
    # flipping its row, not by editing callers.
    if not supports_partition(display_type):
        valid = " / ".join(repr(t) for t in partition_capable_types())
        raise ValueError(
            f"display_type for a partition group must be one of {valid}, got "
            f"{display_type!r}. A geometry type is excluded here until it has a "
            "spatial-partition path: partitioning splits geometry across parts, "
            "so a type whose elements cannot be divided into independently "
            "drawable pieces would be silently corrupted rather than merely "
            "unsupported."
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
