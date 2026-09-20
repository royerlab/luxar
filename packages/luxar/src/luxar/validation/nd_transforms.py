"""Validation and composition for nD transforms on non-displayed dimensions.

nD transforms operate per-dimension on non-displayed dimensions:
- Continuous/discrete ordinal: affine {"scale": float, "offset": float}
- Categorical: permutation {"permutation": [int, ...]}

These are separate from the 4x4 spatial transform and are applied
BEFORE slicing in the viewer pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..typing_utils._format_contract import (
    ND_TRANSFORM_AFFINE_KEYS,
    ND_TRANSFORM_PERMUTATION_KEY,
)
from ..typing_utils.aliases import NdTransform, NdTransformEntry


def _classify_dimension(dim: Any) -> str:
    """Classify a dimension into its algebraic domain.

    Args:
        dim: A Dimension object (from core.dimensions)

    Returns:
        One of: "displayed", "categorical", "discrete_ordinal", "continuous"
    """
    if dim.display:
        return "displayed"
    if dim.is_categorical:
        return "categorical"
    if dim.discrete:
        return "discrete_ordinal"
    return "continuous"


def validate_nd_transform(
    value: Any,
    dimensions: Any = None,
) -> NdTransform:
    """Validate an nD transform dictionary.

    Args:
        value: Input to validate. Expected format:
            {"dim_name": {"scale": float, "offset": float}}  for affine
            {"dim_name": {"permutation": [int, ...]}}         for categorical
        dimensions: Optional Dimensions object for context-aware validation.
            When provided, checks dimension names exist and domain compatibility.

    Returns:
        Validated NdTransform dict

    Raises:
        ValueError: If structure or values are invalid
        TypeError: If value is not a dict
    """
    if not isinstance(value, dict):
        raise TypeError(f"nd_transform must be a dict, got {type(value).__name__}")

    result: NdTransform = {}

    # Build dim lookup if dimensions provided
    dim_lookup: Dict[str, Any] = {}
    if dimensions is not None:
        for dim in dimensions.dimensions:
            dim_lookup[dim.name] = dim

    for dim_name, entry in value.items():
        if not isinstance(dim_name, str):
            raise ValueError(
                f"nd_transform keys must be strings, got {type(dim_name).__name__}"
            )

        if not isinstance(entry, dict):
            raise ValueError(
                f"nd_transform['{dim_name}'] must be a dict, got {type(entry).__name__}"
            )

        # Context-aware validation if dimensions available
        if dim_lookup:
            if dim_name not in dim_lookup:
                raise ValueError(
                    f"nd_transform key '{dim_name}' not found in scene dimensions. "
                    f"Available: {list(dim_lookup.keys())}"
                )
            dim = dim_lookup[dim_name]
            domain = _classify_dimension(dim)

            if domain == "displayed":
                raise ValueError(
                    f"nd_transform cannot target displayed dimension '{dim_name}'. "
                    f"Use the 4x4 'transform' attribute for spatial transforms."
                )

            _validate_entry_for_domain(entry, dim_name, domain, dim)
        else:
            _validate_entry_structure(entry, dim_name)

        result[dim_name] = entry

    return result


def _validate_entry_structure(entry: Dict[str, Any], dim_name: str) -> None:
    """Validate entry structure without dimension context."""
    has_affine = any(key in entry for key in ND_TRANSFORM_AFFINE_KEYS)
    has_permutation = ND_TRANSFORM_PERMUTATION_KEY in entry

    if has_affine and has_permutation:
        raise ValueError(
            f"nd_transform['{dim_name}'] cannot have both affine (scale/offset) "
            f"and permutation params. Use one or the other."
        )

    if not has_affine and not has_permutation:
        raise ValueError(
            f"nd_transform['{dim_name}'] must have 'scale'/'offset' "
            f"or 'permutation'. Got keys: {list(entry.keys())}"
        )

    if has_affine:
        _validate_affine_params(entry, dim_name)
    else:
        _validate_permutation_basic(entry["permutation"], dim_name)


def _validate_entry_for_domain(
    entry: Dict[str, Any],
    dim_name: str,
    domain: str,
    dim: Any,
) -> None:
    """Validate entry against dimension domain."""
    has_affine = any(key in entry for key in ND_TRANSFORM_AFFINE_KEYS)
    has_permutation = ND_TRANSFORM_PERMUTATION_KEY in entry

    if has_affine and has_permutation:
        raise ValueError(
            f"nd_transform['{dim_name}'] cannot have both affine (scale/offset) "
            f"and permutation params."
        )

    if not has_affine and not has_permutation:
        raise ValueError(
            f"nd_transform['{dim_name}'] must have 'scale'/'offset' "
            f"or 'permutation'. Got keys: {list(entry.keys())}"
        )

    if domain == "categorical":
        if has_affine:
            raise ValueError(
                f"Categorical dimension '{dim_name}' requires 'permutation', "
                f"not 'scale'/'offset'. Categories: {dim.categories}"
            )
        _validate_permutation(entry["permutation"], dim_name, dim.categories)
    else:
        # continuous or discrete_ordinal
        if has_permutation:
            raise ValueError(
                f"{domain.replace('_', ' ').title()} dimension '{dim_name}' "
                f"requires 'scale'/'offset', not 'permutation'."
            )
        _validate_affine_params(entry, dim_name)


def _validate_affine_params(entry: Dict[str, Any], dim_name: str) -> None:
    """Validate affine transform parameters."""
    for key in ("scale", "offset"):
        if key in entry:
            try:
                float(entry[key])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"nd_transform['{dim_name}'].{key} must be a number, "
                    f"got {type(entry[key]).__name__}"
                ) from e

    scale = entry.get("scale", 1.0)
    if float(scale) == 0.0:
        raise ValueError(
            f"nd_transform['{dim_name}'].scale is 0; "
            f"zero scale is not a supported transform"
        )

    # Reject unknown keys
    valid_keys = set(ND_TRANSFORM_AFFINE_KEYS)
    unknown = set(entry.keys()) - valid_keys
    if unknown:
        raise ValueError(
            f"nd_transform['{dim_name}'] has unknown keys: {unknown}. "
            f"Valid keys for affine: {valid_keys}"
        )


def _validate_permutation_basic(perm: Any, dim_name: str) -> None:
    """Validate permutation structure without category count."""
    if not isinstance(perm, (list, tuple)):
        raise ValueError(
            f"nd_transform['{dim_name}'].permutation must be a list, "
            f"got {type(perm).__name__}"
        )
    for i, v in enumerate(perm):
        if not isinstance(v, int):
            raise ValueError(
                f"nd_transform['{dim_name}'].permutation[{i}] must be int, "
                f"got {type(v).__name__}"
            )
    if sorted(perm) != list(range(len(perm))):
        raise ValueError(
            f"nd_transform['{dim_name}'].permutation must be a valid "
            f"permutation of [0..{len(perm) - 1}], got {perm}"
        )


def _validate_permutation(perm: Any, dim_name: str, categories: List[str]) -> None:
    """Validate permutation against category list."""
    if len(categories) == 0:
        raise ValueError(
            f"Categorical dimension '{dim_name}' must define at least one category"
        )
    _validate_permutation_basic(perm, dim_name)
    if len(perm) != len(categories):
        raise ValueError(
            f"nd_transform['{dim_name}'].permutation length ({len(perm)}) "
            f"must match category count ({len(categories)}). "
            f"Categories: {categories}"
        )


def compose_nd_transforms(*transforms: NdTransform) -> NdTransform:
    """Compose multiple nD transforms along a hierarchical chain.

    Transforms are passed root-first (parent → ... → leaf). The leaf
    (innermost / last argument) is applied first to the data and each
    parent transforms the result in turn, matching scene-graph semantics.

    For affine: result = parent(child(x)) = s_p * (s_c * x + o_c) + o_p
    For permutation: result[i] = parent_perm[child_perm[i]]

    Args:
        *transforms: Variable number of NdTransform dicts, ordered root-first.

    Returns:
        Composed NdTransform dict. Empty dict means identity.
    """
    if not transforms:
        return {}

    # Collect all dimension names across all transforms
    all_dims: set[str] = set()
    for t in transforms:
        all_dims.update(t.keys())

    if not all_dims:
        return {}

    result: NdTransform = {}

    for dim_name in all_dims:
        # Collect entries for this dimension (root-first order)
        entries = [t[dim_name] for t in transforms if dim_name in t]
        if not entries:
            continue

        if len(entries) == 1:
            result[dim_name] = entries[0]
            continue

        # Check type consistency across entries
        types = ["permutation" if "permutation" in e else "affine" for e in entries]
        if len(set(types)) > 1:
            raise ValueError(
                f"Cannot compose nd_transforms for dimension '{dim_name}': "
                f"mixed types across hierarchy ({types}). "
                f"All transforms for a dimension must be the same type."
            )

        # Determine type from first entry
        if "permutation" in entries[0]:
            # Compose permutations: root-first means
            # composed[i] = entries[0](entries[1](...(entries[-1](i))))
            # Start from innermost (last = leaf) and work outward
            perm = list(entries[-1]["permutation"])
            for e in reversed(entries[:-1]):
                parent_perm = e["permutation"]
                perm = [parent_perm[p] for p in perm]
            result[dim_name] = {"permutation": perm}
        else:
            # Compose affines: root-first
            # Start from leaf (last entry), compose outward
            scale = float(entries[-1].get("scale", 1.0))
            offset = float(entries[-1].get("offset", 0.0))
            for e in reversed(entries[:-1]):
                s_p = float(e.get("scale", 1.0))
                o_p = float(e.get("offset", 0.0))
                # parent(child(x)) = s_p * (scale * x + offset) + o_p
                offset = s_p * offset + o_p
                scale = s_p * scale
            entry: NdTransformEntry = {}
            if scale != 1.0:
                entry["scale"] = scale
            if offset != 0.0:
                entry["offset"] = offset
            if entry:  # Skip identity
                result[dim_name] = entry

    return result


def apply_nd_transform_to_bounds(
    bounds: Dict[str, List[float]],
    nd_transform: NdTransform,
    dimensions: Any,
) -> Dict[str, List[float]]:
    """Apply nD transform to position bounds.

    Transforms the min/max values for each non-displayed dimension
    that has an nd_transform entry.

    Args:
        bounds: Position bounds dict with "min" and "max" lists
        nd_transform: nD transform to apply
        dimensions: Dimensions object for dim name → index mapping

    Returns:
        New bounds dict with transformed values
    """
    min_vals = list(bounds["min"])
    max_vals = list(bounds["max"])

    # EN-2: bounds["min"] and bounds["max"] must have the same length —
    # otherwise we'd silently produce mixed transformed/untransformed bounds
    # for the trailing dims, skewing scene extents downstream. Fail fast so
    # corrupt zarr metadata surfaces at the call site rather than as a
    # mysterious geometry artifact later.
    if len(min_vals) != len(max_vals):
        raise ValueError(
            f"Bounds length mismatch: min has {len(min_vals)} entries, "
            f"max has {len(max_vals)} entries. The bounds arrays must have "
            "equal length."
        )

    for i, dim in enumerate(dimensions.dimensions):
        if dim.name not in nd_transform or dim.display:
            continue
        if i >= len(min_vals):
            # The bounds arrays cover fewer dims than the dimensions schema.
            # Silently skipping would give the caller mixed transformed and
            # untransformed bounds; refuse instead.
            raise ValueError(
                f"Dimension index {i} (name={dim.name!r}) is out of range "
                f"for bounds arrays of length {len(min_vals)}."
            )

        entry = nd_transform[dim.name]

        if "scale" in entry or "offset" in entry:
            s = float(entry.get("scale", 1.0))
            o = float(entry.get("offset", 0.0))
            new_min = s * min_vals[i] + o
            new_max = s * max_vals[i] + o
            if s < 0:
                new_min, new_max = new_max, new_min
            min_vals[i] = new_min
            max_vals[i] = new_max
        # Permutations don't change bounds (range stays [0, n_categories-1])

    return {"min": min_vals, "max": max_vals}
