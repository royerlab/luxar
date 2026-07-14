"""Spatial ordering algorithms for Points, Lines, and GSplats.

Public facade over the private ``io/_ordering/`` subpackage:

- ``curves/``  — Morton & Hilbert space-filling curve encoders
- ``grid``     — float → integer grid quantization
- ``compound`` — the shared lexsort-barrier + curve primitive
- ``points`` / ``gsplats`` / ``lines`` — per-geometry sorts and chunk-bounds
  builders (parallel structure per the three-geometry symmetry)

Everything is re-exported here so the historical public paths
(``luxar.io.ordering.<name>``) keep resolving unchanged.
"""

from __future__ import annotations

from ._ordering.bounds import _BARRIER_BOUND_EPS as _BARRIER_BOUND_EPS  # noqa: F401
from ._ordering.compound import _compound_sort, detect_barrier_dims  # noqa: F401
from ._ordering.curves.hilbert import hilbert_encode_nd  # noqa: F401
from ._ordering.curves.morton import (  # noqa: F401
    morton_encode_128bit,
    morton_encode_nd,
)
from ._ordering.grid import (  # noqa: F401
    compute_auto_resolution,
    normalize_coords_to_grid,
)
from ._ordering.gsplats import (  # noqa: F401
    compute_chunk_bounds_gsplats,
    sort_splats_spatial,
)
from ._ordering.lines import (  # noqa: F401
    compute_segment_chunk_bounds,
    compute_vertex_chunk_bounds,
    convert_to_indexed,
    order_lines_spatial,
    sort_segments_compound,
)
from ._ordering.points import (  # noqa: F401
    compute_chunk_bounds_points,
    sort_points_compound,
)

__all__ = [
    "morton_encode_nd",
    "morton_encode_128bit",
    "hilbert_encode_nd",
    "normalize_coords_to_grid",
    "compute_auto_resolution",
    "detect_barrier_dims",
    "sort_points_compound",
    "sort_splats_spatial",
    "compute_chunk_bounds_points",
    "compute_chunk_bounds_gsplats",
    "convert_to_indexed",
    "sort_segments_compound",
    "order_lines_spatial",
    "compute_vertex_chunk_bounds",
    "compute_segment_chunk_bounds",
]
