"""Semantic type definitions for array encoding.

Semantic types define the meaning and constraints of array data, which
informs encoding choices and validation.
"""

from enum import Enum


class SemanticType(str, Enum):
    """Semantic types for array data.

    Each semantic type has specific constraints and valid encodings:

    - COORDINATE: Spatial positions/centers/vertices. Can be negative.
      Valid: float32, float16
    - COLOR: RGB/RGBA color values. Non-negative.
      SDR (0-1): uint8, uint16, float16, float32
      HDR (>1): float16, float32
    - BOUNDED_SCALAR: Scalars with known [min, max] bounds.
      Examples: sharpness [0, 1], opacity [0, 1]
      Valid: uint8, uint16, float16, float32
    - POSITIVE_SCALAR: Non-negative scalars, potentially wide dynamic range.
      Examples: radii, amplitudes, distances
      Valid: uint8, uint16 (linear or log), float16, float32
    - CHOLESKY: Packed lower-triangular Cholesky factors.
      Shape: (N, d*(d+1)/2) for d-dimensional covariance
      Valid: float32, float16
    - INDEX: Non-negative integer indices or counts.
      Valid: uint8, uint16, uint32, uint64
    - UNIT_VECTOR: Normalized vectors with ||v|| = 1.
      Examples: surface normals, directions
      Valid: float32, float16, or specialized encodings (octahedral)
    """

    COORDINATE = "coordinate"
    COLOR = "color"
    BOUNDED_SCALAR = "bounded_scalar"
    POSITIVE_SCALAR = "positive_scalar"
    CHOLESKY = "cholesky"
    INDEX = "index"
    UNIT_VECTOR = "unit_vector"
