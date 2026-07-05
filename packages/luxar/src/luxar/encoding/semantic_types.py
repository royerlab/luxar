"""Semantic type definitions for array encoding.

Semantic types define the meaning and constraints of array data, which
informs encoding choices and validation.
"""

from enum import Enum


class SemanticType(str, Enum):
    """Semantic types for array data.

    Each semantic type has specific constraints and valid encodings:

    - COORDINATE: Spatial positions/centers/vertices. Can be negative.
      PRECISION → float32; AUTO / MEMORY → uint16 per-axis fixed-point via the
      generic per-channel **linear** quantization (``linear_perchannel_u16``),
      decoded back to float32 — visually lossless (sub-unit) and ~2× smaller.
      Coordinates never use uint8 (too coarse) and never float16 (relative
      precision degrades with magnitude). A per-axis extent ≥ 2¹⁶ auto-falls
      back to float32.
    - COLOR: RGB/RGBA color values. Non-negative.
      SDR (0-1): uint8, uint16, float16, float32
      HDR (>1): float16, float32
    - BOUNDED_SCALAR: Scalars with known [min, max] bounds.
      Examples: sharpness [0, 1], opacity [0, 1]
      Valid: uint8, uint16, float16, float32
    - POSITIVE_SCALAR: Non-negative scalars, potentially wide dynamic range.
      Examples: radii, amplitudes, distances
      Valid: uint8, uint16 (linear or log), float16, float32
    - CHOLESKY: Packed lower-triangular Cholesky factors (whole, unsplit).
      Shape: (N, d*(d+1)/2) for d-dimensional covariance
      Valid: float32, float16
    - CHOLESKY_DIAG: Diagonal of a packed Cholesky factor (positive, scale-like),
      shape (N, d). Encoded float32, or with the generic per-channel **log**
      quantization (``log_perchannel_u8``; ``log_perchannel_u16`` when AUTO's
      covariance certificate escalates — see ``encode_cholesky_split``).
    - CHOLESKY_OFFDIAG: Strictly-lower off-diagonal of a Cholesky factor (signed,
      ~zero-centred), shape (N, d*(d-1)/2). Encoded float32, or with the generic
      per-channel **signed-log** quantization (``signed_log_perchannel_u8``,
      escalating to ``signed_log_perchannel_u16`` alongside the diagonal — both
      halves always share one tier).
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
    CHOLESKY_DIAG = "cholesky_diag"
    CHOLESKY_OFFDIAG = "cholesky_offdiag"
    INDEX = "index"
    UNIT_VECTOR = "unit_vector"
