"""Shared category validation for categorical dimensions.

This module provides validation for category lists used in categorical dimensions.
It is intentionally separate to avoid circular imports between core and validation modules.
"""

from __future__ import annotations

from typing import Optional

from ..typing_utils.aliases import CategoryList
from ..typing_utils.constants import MAX_CATEGORY_LABEL_LENGTH, MIN_CATEGORIES


def validate_categories(categories: CategoryList) -> CategoryList:
    """Validate category list for categorical dimensions.

    Categories define the discrete values for a categorical dimension.
    Each category is a string label (e.g., ["DAPI", "GFP", "mCherry"]).

    Args:
        categories: List of category labels, or None for non-categorical

    Returns:
        Validated category list (or None if input is None)

    Raises:
        TypeError: If categories is not a list or None
        ValueError: If categories is invalid (empty, duplicates, too long, etc.)

    Examples:
        >>> validate_categories(["DAPI", "GFP", "mCherry"])
        ['DAPI', 'GFP', 'mCherry']

        >>> validate_categories(None)
        None

        >>> validate_categories(["A", "A"])  # doctest: +SKIP
        ValueError: duplicate category name...
    """
    if categories is None:
        return None

    if not isinstance(categories, list):
        raise TypeError(
            f"categories must be a list or None, got {type(categories).__name__}"
        )

    if len(categories) < MIN_CATEGORIES:
        raise ValueError(
            f"categories must have at least {MIN_CATEGORIES} element, got {len(categories)}"
        )

    # Check each category label
    seen: dict[str, int] = {}
    for i, cat in enumerate(categories):
        if not isinstance(cat, str):
            raise TypeError(
                f"category at index {i} must be a string, got {type(cat).__name__}"
            )
        if len(cat) == 0:
            raise ValueError(f"category at index {i} is empty string")
        if len(cat) > MAX_CATEGORY_LABEL_LENGTH:
            raise ValueError(
                f"category at index {i} exceeds maximum length ({MAX_CATEGORY_LABEL_LENGTH} chars)"
            )
        if cat in seen:
            raise ValueError(
                f"duplicate category name: '{cat}' appears at indices {seen[cat]} and {i}"
            )
        seen[cat] = i

    return categories
