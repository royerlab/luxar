"""Constants used throughout Luxar.

This module centralizes all magic numbers and constants to improve
maintainability and provide clear documentation of their purposes.
"""

from typing import Final

from ._format_contract import SCENE_FORMAT_VERSION

# Version constants (single-sourced from format-contract/contract.yaml)
LUXAR_VERSION_CURRENT: Final[str] = SCENE_FORMAT_VERSION
DEFAULT_ZARR_VERSION: Final[str] = SCENE_FORMAT_VERSION  # Alias for default version

# Rendering constants
OPACITY_MIN: Final[float] = 0.0
OPACITY_MAX: Final[float] = 1.0
DEFAULT_OPACITY: Final[float] = 1.0

GAMMA_MIN: Final[float] = 0.1  # Symmetric: gamma and 1/gamma have equal range
GAMMA_MAX: Final[float] = 10.0  # Symmetric: gamma and 1/gamma have equal range
DEFAULT_GAMMA: Final[float] = 1.0

INTENSITY_MIN: Final[float] = 0.0
INTENSITY_MAX: Final[float] = 100.0
DEFAULT_INTENSITY: Final[float] = 1.0

OFFSET_MIN: Final[float] = -10.0
OFFSET_MAX: Final[float] = 10.0
DEFAULT_OFFSET: Final[float] = 0.0

# Blending modes
DEFAULT_BLENDING_MODE: Final[str] = "additive"

# Absorption (kappa) — the volumetric blending mode's per-node coefficient.
# Multiplicative composition, identity 1.0; no upper bound (physical
# coefficient); read only by the volumetric shader branch.
ABSORPTION_MIN: Final[float] = 0.0
DEFAULT_ABSORPTION: Final[float] = 1.0

# Sharpness constants.
# Sharpness is a normalised [0, 1] knob mapped in the viewer to the
# super-Gaussian falloff exponent beta = 2^(6s - 2): s=0.5 -> beta=2 (a true
# Gaussian), higher s -> harder/crisper edge, lower s -> peakier cusp.
SHARPNESS_MIN: Final[float] = 0.0  # Normalised range floor
SHARPNESS_MAX: Final[float] = 1.0  # Normalised range ceiling
SHARPNESS_DEFAULT: Final[float] = 0.5  # -> beta = 2 (Gaussian)

# HDR color constants
COLOR_SDR_MIN: Final[float] = 0.0  # Standard dynamic range minimum
COLOR_SDR_MAX: Final[float] = 1.0  # Standard dynamic range maximum
COLOR_HDR_TYPICAL_MAX: Final[float] = 10.0  # Typical HDR maximum
COLOR_HDR_THEORETICAL_MAX: Final[float] = float("inf")  # No theoretical limit

# Chunk size constants (SINGLE SOURCE OF TRUTH - bytes, not elements)
# Consumers (io, gsplats.io) convert to element counts based on array dtype
TARGET_CHUNK_BYTES: Final[int] = 65_536  # 64KB target chunk size
MIN_CHUNK_BYTES: Final[int] = 16_384  # 16KB minimum to amortize HTTP overhead
MAX_CHUNK_BYTES: Final[int] = 262_144  # 256KB maximum for responsive streaming

# Memory constants
KB_TO_BYTES: Final[int] = 1024
MB_TO_BYTES: Final[int] = 1024 * 1024
GB_TO_BYTES: Final[int] = 1024 * 1024 * 1024

# Array size constants
MAX_POINTS_RECOMMENDED: Final[int] = 10_000_000  # 10M points
MAX_POINTS_WARNING: Final[int] = 100_000_000  # 100M points

# Compression constants
COMPRESSION_LEVEL_MIN: Final[int] = 0  # No compression
COMPRESSION_LEVEL_DEFAULT: Final[int] = 3
COMPRESSION_LEVEL_MAX: Final[int] = 9  # Maximum compression
DEFAULT_COMPRESSOR: Final[str] = "blosc"  # Default compression algorithm

# Transform matrix constants
TRANSFORM_MATRIX_SIZE: Final[int] = 4  # 4x4 matrices
TRANSFORM_MATRIX_ELEMENTS: Final[int] = 16  # Total elements when flattened

# Dimension constants
MAX_DISPLAYED_DIMENSIONS: Final[int] = 3  # Maximum dimensions shown in viewer
MIN_DISPLAYED_DIMENSIONS: Final[int] = 1  # Minimum dimensions shown in viewer
DEFAULT_DIMENSION_STEP_PERCENT: Final[float] = 0.01  # 1% of range for navigation

# Categorical dimension constants
MIN_CATEGORIES: Final[int] = 1  # Minimum categories for categorical dimensions
MAX_CATEGORY_LABEL_LENGTH: Final[int] = 1024  # Maximum length for category labels
CATEGORICAL_STEP: Final[float] = 1.0  # Step size for categorical dimensions (always 1)

# Decimal precision for display
POSITION_DISPLAY_DECIMALS: Final[int] = 3
RADIUS_DISPLAY_DECIMALS: Final[int] = 3
COLOR_DISPLAY_DECIMALS: Final[int] = 2

# Physical units
PHYSICAL_UNIT_DEFAULT: Final[str] = "au"  # Arbitrary units

# Zarr metadata keys
ZARR_METADATA_FILENAME: Final[str] = ".zmetadata"
ZARR_ATTRS_KEY: Final[str] = ".zattrs"

# Node type identifiers
NODE_TYPE_SCENE: Final[str] = "scene"
NODE_TYPE_GROUP: Final[str] = "group"
NODE_TYPE_POINTS: Final[str] = "points"
NODE_TYPE_LINES: Final[str] = "lines"
NODE_TYPE_GSPLATS: Final[str] = "gsplats"

# Point radius constants
MIN_POINT_RADIUS: Final[float] = 0.001  # Minimum visible radius
MAX_POINT_RADIUS: Final[float] = 1000.0  # Maximum practical radius

#: GSplat truncation radius ``T``, in sigmas: the Mahalanobis distance beyond
#: which a splat's shifted Gaussian is exactly zero. The kernel is
#: ``max(0, exp(-D²/2) - C) / (1 - C)`` with ``C = exp(-T²/2)``, so ``T`` sets
#: both the support and the normalization ``1/(1 - C)``.
#:
#: 2.75 comes from ``da53b17d2`` (2.5σ measured +1.6 dB over the prior default;
#: 2.75 is the quality/speed middle ground). It was applied only to the fitter
#: config at the time, leaving ~20 sites defaulting to 3.0 — this constant is
#: the single source that replaces them.
#:
#: MIRROR: ``GSPLAT_DEFAULT_TRUNCATION_RADIUS`` in
#: ``packages/luxar-viewer/src/config/constants.ts`` must hold the same value.
#: Both sides are pinned by tests that name each other.
#:
#: EXEMPTION: :mod:`luxar.gsplats.lift` keeps 3.0. Its ``T`` is not a render
#: default but a profile-matching parameter — the point/line super-Gaussian
#: sprite and the gsplat kernel coincide exactly at ``T* = sqrt(2 ln 100) =
#: 3.0349``. Measured radial-weighted relative L2 of the lift: 1.96% at T=3.0,
#: 16.91% at T=2.75. See ``lift.py`` for the derivation.
DEFAULT_TRUNCATION_RADIUS: Final[float] = 2.75

# Spatial index grid constants
SPATIAL_INDEX_MAX_CELLS_DISCRETE: Final[int] = (
    10000  # Maximum cells for discrete dimensions (safety cap)
)
SPATIAL_INDEX_MAX_CELLS_CONTINUOUS: Final[int] = (
    10  # Maximum cells per continuous dimension
)
SPATIAL_INDEX_MIN_CELLS: Final[int] = 3  # Minimum cells per continuous dimension
SPATIAL_INDEX_TARGET_POINTS_PER_CELL: Final[int] = (
    10000  # Target points per cell for adaptive grid sizing
)
SPATIAL_INDEX_FALLBACK_CELLS: Final[int] = 2  # Cells for degenerate dimensions
SPATIAL_INDEX_MIN_CELLS_SINGLE_DIM: Final[int] = (
    10  # Minimum cells for single indexed dimension
)
SPATIAL_INDEX_MAX_CELLS_SINGLE_DIM: Final[int] = (
    100  # Maximum cells for single indexed dimension
)
SPATIAL_INDEX_CHUNK_SIZE: Final[int] = 4096  # Chunk size for index datasets
SPATIAL_INDEX_SINGLE_DIM_POINTS_DIVISOR: Final[int] = (
    100  # Divisor for sqrt calculation in single-dimension grids
)
SPATIAL_INDEX_MULTI_DIM_MIN_TARGET: Final[int] = (
    10  # Minimum target cells for multi-dimensional grids
)
SPATIAL_INDEX_MULTI_DIM_MAX_TARGET: Final[int] = (
    1000  # Maximum target cells for multi-dimensional grids
)
SPATIAL_INDEX_MULTI_DIM_POINTS_DIVISOR: Final[int] = (
    500  # Points divisor for multi-dimensional grid sizing
)
SPATIAL_INDEX_MULTI_DIM_MAX_CELLS_PER_DIM: Final[int] = (
    20  # Maximum cells per dimension in multi-dimensional grids
)

# Validation messages
VALIDATION_POSITIVE_REQUIRED: Final[str] = "Value must be positive"
VALIDATION_SHAPE_MISMATCH: Final[str] = (
    "Shape mismatch: expected {expected}, got {actual}"
)
VALIDATION_OUT_OF_RANGE: Final[str] = (
    "Value {value} is outside valid range [{min}, {max}]"
)
