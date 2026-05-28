"""Opt-in compiler-level auto-split resolution.

When ``LuxarZarrCompiler(auto_split_max_elements=N)`` is set and the
user did NOT pass ``split=`` at a leaf-adder callsite, leaf adders
should synthesize a ``dict(max_elements=N)`` so any large leaf gets
decomposed automatically. The recursion guard uses ``split=False`` to
disable auto-split on the per-part recursive calls.

This one-function module is consumed by ``add_points`` / ``add_lines`` /
``add_gsplats`` in ``core/group/group.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..scene import Scene


def resolve_auto_split(
    scene: "Scene", n_elements: int, user_split: Any
) -> Any:
    """Resolve the effective ``split=`` for a leaf-adder call.

    Opt-in compiler-level auto-split: when
    ``LuxarZarrCompiler(auto_split_max_elements=N)`` is set and the
    user did NOT pass ``split=`` at the call site, return a synthetic
    ``dict(max_elements=N)`` once ``n_elements > N``. Below the
    threshold or with a user-explicit ``split=``, pass the original
    value through unchanged.

    - ``user_split is None`` → opt into auto-split (apply if threshold is set)
    - ``user_split is False`` → explicit no-split bypass (returns ``None``)
    - anything else → user-explicit, pass through

    The ``False`` sentinel is used internally by the split-wrapper
    recursion (``_add_points_split_wrapper`` / ``_add_gsplats_split_wrapper``)
    so per-part recursive ``add_points`` / ``add_gsplats`` calls do
    not re-trigger auto-split (which would explode the leaf count
    when the compiler threshold is smaller than the user's explicit
    cap).
    """
    if user_split is False:
        return None
    if user_split is not None:
        return user_split
    writer = scene._writer
    threshold = getattr(writer, "auto_split_max_elements", None) if writer else None
    if threshold is None:
        return user_split
    if n_elements <= int(threshold):
        return user_split
    return {"max_elements": int(threshold)}
