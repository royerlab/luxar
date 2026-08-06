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

# Hard ceiling on a mesh node's vertex count. Unlike the advisory point limits
# above this is a CORRECTNESS bound, not a performance hint, so it is enforced
# (see validate_vertices_for_writing) rather than warned about.
#
# A mesh's pick elementId is the raw `gl_VertexID` — the one geometry type not
# bounded by the element-texture capacity — and the viewer's pick vote key is
# built with a stride of 2^27 per node. Once a vertex ordinal reaches that
# stride, vote keys alias ACROSS nodes and a pick resolves to the wrong node with
# no diagnostic. The largest ordinal is `n_vertices - 1`, so `n_vertices <= 2^27`
# is the exact alias-free bound: every admitted ordinal stays strictly under the
# stride. See docs/specs/MESH_NODE_SPEC.md §6.5.
#
# It also keeps the writer's face-index check sufficient: with vertices capped
# here, `max(faces) < n_vertices` guarantees every admitted index survives the
# `.astype(np.uint32)` cast unchanged (2^27 is far below 2^32).
#
# MIRROR: MAX_MESH_VERTICES in packages/luxar-viewer/src/config/constants.ts must
# hold this value — that is the loader-side gate this one is the fail-fast twin
# of. If you change one, change the other; a viewer test pins that side.
MAX_MESH_VERTICES: Final[int] = 2**27  # 134,217,728 — pick vote-key stride

# Compression constants
COMPRESSION_LEVEL_MIN: Final[int] = 0  # No compression
COMPRESSION_LEVEL_DEFAULT: Final[int] = 3
COMPRESSION_LEVEL_MAX: Final[int] = 9  # Maximum compression
DEFAULT_COMPRESSOR: Final[str] = "blosc"  # Default compression algorithm

# Transform matrix constants
TRANSFORM_MATRIX_SIZE: Final[int] = 4  # 4x4 matrices

# Dimension constants
MAX_DISPLAYED_DIMENSIONS: Final[int] = 3  # Maximum dimensions shown in viewer

# Categorical dimension constants
MIN_CATEGORIES: Final[int] = 1  # Minimum categories for categorical dimensions
MAX_CATEGORY_LABEL_LENGTH: Final[int] = 1024  # Maximum length for category labels
CATEGORICAL_STEP: Final[float] = 1.0  # Step size for categorical dimensions (always 1)

# Node type identifiers. These must cover `node_types` in
# `format-contract/contract.yaml` exactly — `test_named_node_type_constants_match_contract`
# pins the set both ways, so a contract addition with no constant here fails there.
NODE_TYPE_SCENE: Final[str] = "scene"
NODE_TYPE_GROUP: Final[str] = "group"
NODE_TYPE_POINTS: Final[str] = "points"
NODE_TYPE_LINES: Final[str] = "lines"
NODE_TYPE_GSPLATS: Final[str] = "gsplats"
NODE_TYPE_MESH: Final[str] = "mesh"

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
