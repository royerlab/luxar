"""Shared chunk-bounds constants for the per-geometry ordering modules."""

from __future__ import annotations

# Padding added to barrier/discrete-dimension chunk bounds. This is ONLY a
# float-boundary safety margin — the query "reach" (how far a slice query
# selects around a category) lives entirely in the reader's per-dimension
# tolerance (see luxar-viewer tolerance-computer.ts, discrete = 0.25 × step).
# It used to be 0.5 (half a step); combined with the reader's own half-step
# tolerance that summed to a full step and made a single-category query (e.g.
# one timepoint) pull in the entire neighbouring category. Keep this tiny.
#
# KNOWN LIMIT: the pad is absolute while the reader's reach is step-scaled
# (0.25 x step), so for pathological discrete steps below ~1.3e-3 the pad
# reaches past the neighbour category's quarter-step boundary and the
# over-fetch returns. Step metadata is not plumbed into these bound
# builders; discrete/categorical dims with milli-scale steps are not a
# supported layout (rescale the axis instead).
_BARRIER_BOUND_EPS = 1e-3
