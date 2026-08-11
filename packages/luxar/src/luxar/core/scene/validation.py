"""Scene-side validation helpers for data nodes.

Five free functions called by ``Group``'s leaf adders:

* :func:`resolve_extend_to_all` — interpret the ``extend_to_all=`` kwarg.
* :func:`analyze_extend_candidates` — detect non-spatial dims with a single value.
* :func:`validate_array_rank` — the 2-D ``(N, D)`` shape raise on its own.
* :func:`validate_dimension_count` — the hard column-count raise on its own.
* :func:`validate_data_dimensions` — enforce dimensionality + range bounds.

The Scene class keeps method stubs that just forward; this module holds
the bodies so scene.py stays focused on construction + properties + export.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from .scene import Scene


def resolve_extend_to_all(
    scene: "Scene",
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
    """
    if extend_to_all is None:
        # Default: No extension, but warn if candidates detected
        candidates = analyze_extend_candidates(scene, positions)
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
            for dim in scene._dimensions.dimensions
            if not dim.display and dim.name
        ]
    elif isinstance(extend_to_all, list):
        # Use explicit list (including empty list to silence warning)
        unknown_dims = [
            dim_name
            for dim_name in extend_to_all
            if dim_name not in scene._dimensions.names
        ]
        if unknown_dims:
            raise ValueError(
                f"Unknown dimension(s) in extend_to_all: {unknown_dims}. "
                f"Valid dimensions: {scene._dimensions.names}"
            )
        return extend_to_all
    else:
        raise ValueError(
            f"Invalid extend_to_all value: {extend_to_all}. "
            f"Expected None, list of dimension names, 'all', or []."
        )


def analyze_extend_candidates(scene: "Scene", positions: np.ndarray) -> List[str]:
    """Analyze which dimensions might be candidates for extend_to_all.

    A dimension is a candidate if:
    1. It is not displayed (non-spatial dimension)
    2. It has only ONE unique value in the data
    3. It has a defined range that is larger than just that single value
    """
    candidates: List[str] = []
    data_ndim = positions.shape[1]

    for i, dim in enumerate(scene._dimensions.dimensions):
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

            if range_max > range_min and (value >= range_min and value <= range_max):
                if dim.name:
                    candidates.append(dim.name)

    return candidates


def validate_array_rank(positions: np.ndarray, data_type: str = "positions") -> None:
    """Raise unless the data array is 2-D ``(N, D)``.

    Its own function only so the message has ONE home: the split paths need the
    rank check without the count check (under a ``dim_order`` the incoming width
    is legitimately not the scene's), and a hand-rolled copy of this sentence
    there would be held in sync by nothing but discipline.
    """
    if positions.ndim != 2:
        raise ValueError(
            f"{data_type.capitalize()} must have shape (N, D), "
            f"got shape {positions.shape}"
        )


def validate_dimension_count(
    scene: "Scene",
    positions: np.ndarray,
    node_name: str,
    data_type: str = "positions",
) -> None:
    """Raise if the data's column count disagrees with the scene's dimensions.

    This is the HARD half of :func:`validate_data_dimensions`, split out so the
    split-write paths (``partition=`` / ``additive_lod=`` / ``substitutive_lod=``
    / ``lod_group=``) can run it on the CALLER's source array before they create
    any wrapper group, without also re-firing the per-dimension range
    ``UserWarning`` once per part/level. Cheap (a shape comparison) and
    idempotent, so calling it twice on the same array is harmless.

    The rank guard is not redundant: the leaf adders reject a non-2-D array
    before they get here, but ``add_gsplats_from_data`` hands over a
    ``GSplatData``'s centers, which ``AdditiveSubLOD`` accepts at 1-D — without
    the guard that reached ``shape[1]`` as a bare ``IndexError`` that escapes the
    adders' ``except (ValueError, TypeError)`` funnels. It rejects rank >= 3 as
    well as rank 1, which is stricter than before (a ``(2, 3, 3)`` array used to
    reach ``shape[1]`` and pass); nothing authored through the adders can be
    non-2-D by the time it gets here, so the extra strictness only affects the
    ``GSplatData`` door. The wording matches the rank message of every adder that
    can actually reach this — Points, Lines and GSplats all say
    "… must have shape (N, D)". Mesh words its own as "(V, D)", which would NOT
    match, but mesh rank-checks before it calls here, so that never surfaces.
    """
    validate_array_rank(positions, data_type)

    data_ndim = positions.shape[1]
    scene_ndim = scene._dimensions.ndim

    if data_ndim != scene_ndim:
        dim_names = scene._dimensions.names
        raise ValueError(
            f"Dimension mismatch for '{node_name}': {data_type} array has "
            f"{data_ndim} columns, but scene has {scene_ndim} dimensions "
            f"({dim_names}).\n"
            f"Expected {data_type} shape: (N, {scene_ndim})\n"
            f"Got {data_type} shape: {positions.shape}"
        )


def validate_data_dimensions(
    scene: "Scene",
    positions: np.ndarray,
    node_name: str,
    data_type: str = "positions",
    _stacklevel: int = 3,
) -> None:
    """Validate that data dimensions match scene dimensions.

    Performs two levels of validation:
    1. HARD ERROR: Dimensionality mismatch (data columns != scene dimensions)
    2. WARNING: Values outside declared dimension ranges

    The hard error lives in :func:`validate_dimension_count`; it MUST stay the
    first thing this function does so the raised message is identical whether a
    caller runs the count check on its own or the full validation.
    """
    validate_dimension_count(scene, positions, node_name, data_type)

    if positions.shape[0] == 0:
        return

    for i, dim in enumerate(scene._dimensions.dimensions):
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
