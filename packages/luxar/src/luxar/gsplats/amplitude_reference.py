"""Shared per-splat amplitude reference for scene insertion and exposure."""

from __future__ import annotations

from typing import Optional

import numpy as np

#: Keep equal to the percentile used for the ``amplitude_data_range`` upper bound.
AMPLITUDE_REFERENCE_PERCENTILE = 99.9

#: Float32 rounding can put an already-normalised reference slightly above 1.
#: This tolerance keeps automatic scaling idempotent.
IN_RANGE_TOLERANCE = 1e-6


def sample_reference(values: np.ndarray) -> float:
    """Pooled robust upper reference for a 1-D amplitude sample."""
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, AMPLITUDE_REFERENCE_PERCENTILE))


def automatic_factor(reference: float) -> Optional[float]:
    """Scale a positive out-of-range reference to one, otherwise leave it alone."""
    if not (reference > 0.0) or not np.isfinite(reference):
        return None
    if reference <= 1.0 + IN_RANGE_TOLERANCE:
        return None
    return 1.0 / reference
