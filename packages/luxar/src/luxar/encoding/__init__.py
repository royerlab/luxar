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
]
