"""Encoding package for semantic types and array encoding/decoding.

This package handles:
- Semantic type definitions (COORDINATE, COLOR, POSITIVE_SCALAR, etc.)
- Encoding modes (AUTO, PRECISION, MEMORY, CUSTOM)
- Array encoding/decoding with multiple strategies
- Array reference registry for deduplication
- Encoding metadata specification
"""

# Importing delta_codec registers the ``luxar_delta_v1`` zarr filter with
# numcodecs — required on BOTH the write and the read path (zarr resolves
# ``.zarray`` filters through the numcodecs registry).
from ._encoders.delta_codec import LuxarDelta

# The COORDINATE grid-snap predicate. Exported because the gsplat writer's sigma
# rail must ask the SAME question the encoder will ask ("will this axis be stored
# exactly?") before it escalates centers to float32 — two implementations of that
# test would be two chances to disagree.
from ._encoders.perchannel import COORDINATE_LEVELS, gridded_axis_step
from .decoder import ArrayDecoder
from .encoder import ArrayEncoder
from .modes import EncodingMode
from .registry import ArrayRefMatch, ArrayRefRegistry
from .semantic_types import SemanticType

__all__ = [
    # Core encoding system
    "ArrayEncoder",
    "ArrayDecoder",
    "SemanticType",
    "EncodingMode",
    "ArrayRefRegistry",
    "ArrayRefMatch",
    "LuxarDelta",
    # COORDINATE grid snap (shared with the gsplat centers sigma rail)
    "gridded_axis_step",
    "COORDINATE_LEVELS",
]
