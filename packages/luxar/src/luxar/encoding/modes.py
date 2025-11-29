"""Encoding mode definitions for array encoding.

Encoding modes define the preference for precision vs storage trade-offs.
"""

from enum import Enum


class EncodingMode(str, Enum):
    """Encoding preference modes.

    Modes control the trade-off between precision and storage efficiency:

    - AUTO: Automatically analyze data and select appropriate encoding.
      Balanced approach that considers data range and semantic type.

    - PRECISION: Preserve maximum precision by using float32 for all
      non-index data. Recommended for scientific accuracy.

    - MEMORY: Minimize storage size aggressively using quantization
      (uint8, uint16, float16) where safe. Recommended for large datasets
      and streaming. May introduce quantization error.

    - CUSTOM: User explicitly specifies which encoder to use for each array.
      Provides full control over encoding choices. Requires custom_encoder
      parameter in encode() calls.

    Note: Broadcasting and Array Reference optimizations apply in ALL modes
    as they are lossless. Mode only affects dtype/quantization encoding.
    """

    AUTO = "auto"
    PRECISION = "precision"
    MEMORY = "memory"
    CUSTOM = "custom"
