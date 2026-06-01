"""Opt-in compiler-level auto-partition resolution.

When ``LuxarZarrCompiler(auto_partition_max_elements=N)`` is set and the
user did NOT pass ``partition=`` at a leaf-adder callsite, leaf adders
should synthesize a ``dict(max_elements=N)`` so any large leaf gets
decomposed automatically. The recursion guard uses ``partition=False`` to
disable auto-partition on the per-part recursive calls.

This one-function module is consumed by ``add_points`` / ``add_lines`` /
``add_gsplats`` in ``core/group/group.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..scene import Scene


def resolve_auto_partition(
    scene: "Scene", n_elements: int, user_partition: Any
) -> Any:
    """Resolve the effective ``partition=`` for a leaf-adder call.

    Opt-in compiler-level auto-partition: when
    ``LuxarZarrCompiler(auto_partition_max_elements=N)`` is set and the
    user did NOT pass ``partition=`` at the call site, return a synthetic
    ``dict(max_elements=N)`` once ``n_elements > N``. Below the
    threshold or with a user-explicit ``partition=``, pass the original
    value through unchanged.

    - ``user_partition is None`` → opt into auto-partition (apply if threshold set)
    - ``user_partition is False`` → explicit no-partition bypass (returns ``None``)
    - anything else → user-explicit, pass through

    The ``False`` sentinel is used internally by the partition-wrapper
    recursion (``add_points_partition_wrapper_impl`` /
    ``add_gsplats_partition_wrapper_impl`` in ``core/group/adders/``) so
    per-part recursive ``add_points`` / ``add_gsplats`` calls do
    not re-trigger auto-partition (which would explode the leaf count
    when the compiler threshold is smaller than the user's explicit
    cap).
    """
    if user_partition is False:
        return None
    if user_partition is not None:
        return user_partition
    writer = scene._writer
    threshold = (
        getattr(writer, "auto_partition_max_elements", None) if writer else None
    )
    if threshold is None:
        return user_partition
    if n_elements <= int(threshold):
        return user_partition
    return {"max_elements": int(threshold)}
