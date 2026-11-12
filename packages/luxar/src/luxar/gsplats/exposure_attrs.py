"""Shared attribute names for splat exposure provenance."""

#: Provenance attrs stamped by ``gsplat convert``'s auto exposure: which rule
#: produced the ``display_gain`` (``projected`` / ``per_splat`` / ``manual``)
#: and what the opening view was predicted to look like. The viewer does not
#: read them.
NORMALIZATION_MODE_ATTR = "amplitude_normalization_mode"
PROJECTED_SUM_P999_ATTR = "projected_sum_p999"
PROJECTED_SUM_TARGET_ATTR = "projected_sum_target"
PROJECTED_SUM_AXIS_ATTR = "projected_sum_axis"
