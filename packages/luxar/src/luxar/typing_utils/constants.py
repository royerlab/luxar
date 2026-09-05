"""Constants used throughout Luxar.

This module centralizes all magic numbers and constants to improve
maintainability and provide clear documentation of their purposes.
"""

import math
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

OFFSET_MIN: Final[float] = -10.0
OFFSET_MAX: Final[float] = 10.0

# Blending modes
DEFAULT_BLENDING_MODE: Final[str] = "additive"
# Viewer-side defaults are applied only after root→leaf attribute composition:
# create-points-node.ts/create-lines-node.ts/create-gsplats-node.ts use additive,
# while create-mesh-node.ts uses opaque. Keep the source-lock test in
# io/tests/_compiler/test_blending_warnings.py in sync with those consumers.
DEFAULT_BLENDING_MODE_BY_GEOMETRY: Final[dict[str, str]] = {
    "points": "additive",
    "lines": "additive",
    "gsplats": "additive",
    "mesh": "opaque",
}

# Line join style (issue #790) — the strategy the LINE vertex stage uses at a
# degree-2 polyline joint. Lines-only: it has no meaning for points, gsplats or
# mesh, none of which build a screen-space quad per element.
#
# "none"  leave the two quads alone, so the turn leaves an uncovered circular
#         sector on the outside of the bend and a double-covered lens inside.
# "miter" rotate each quad's end edge onto the shared miter edge so the two
#         TILE. Coverage becomes a partition, so there is nothing to sum and
#         every blending mode is correct by construction.
#
# The viewer mirror is `packages/luxar-viewer/src/types/line-join.ts`; keep the
# spellings and the default in step with it.
LINE_JOIN_STYLES: Final[frozenset[str]] = frozenset({"none", "miter"})
# Documentation of the shared default, deliberately WITHOUT a reader here: the
# writer must not bake a join style into the file, or an unset node would freeze
# today's default forever and the viewer could never move it. Not dead code.
DEFAULT_LINE_JOIN: Final[str] = "miter"

# Per-element interaction templates (issue #1917).
#
# Browsing contexts a node's `link_target` may name. Restricted to the two
# keywords that imply `noopener`: any OTHER value is a *named* target, which
# hands the opened page a live `window.opener` it can use to cross-origin
# navigate the viewer tab (reverse tabnabbing). The viewer enforces the same
# two, letter-for-letter, in `core/app/interaction/element-actions.ts`; keep them
# in step.
LINK_TARGETS: Final[frozenset[str]] = frozenset({"_blank", "_self"})
DEFAULT_LINK_TARGET: Final[str] = "_blank"

# URL schemes a `link` template may resolve to. An ALLOWLIST, not a denylist:
# the viewer NAVIGATES to this URL rather than rendering it, and `.zattrs` is
# untrusted input, so anything exotic (`javascript:`, `data:`, `blob:`,
# `file:`, `vbscript:`) must be refused rather than enumerated.
LINK_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

# Ceiling on a built URL, matched by the viewer's own per-click check. Well
# above any real link; there to bound what a hostile store can push at the
# browser after per-element substitution.
MAX_LINK_CHARS: Final[int] = 2048

# Ceiling on a built copy string. Larger than the URL cap because a copy
# payload is legitimately prose (a whole record, a citation) rather than an
# address, but still bounded: it reaches the system clipboard, outliving the
# page, and originates in untrusted `.zattrs`.
MAX_COPY_CHARS: Final[int] = 8 * 1024

# LOD selector modes — how the viewer interprets a kind=lod group's per-child
# `coverage_fraction` thresholds (the group's `selector` attr names the units):
#
# "screen-area"  coverage_fraction is a literal SCREEN-AREA fraction: projected
#                bbox rect area / viewport area. The derived whole-object ladder
#                is [0, …, 1/8, 1/4, 1/2] (full detail while the node occupies
#                at least half the screen, one level coarser per halving of
#                occupied area); a partition tile anchors at 1.0 (fills-screen).
#                What every DERIVED ladder stamps.
# "coverage"     legacy diagonal metric: the viewer compares against projected
#                bbox diagonal / (FILL_FACTOR=0.5 × min(viewport.width,
#                viewport.height) — the fitted screen axis; see the FILL_FACTOR
#                doc in scene/lod-group-registry.ts), bounded by
#                MAX_COVERAGE_FRACTION=4.0. Kept for existing datasets and
#                for explicit `coverage_fractions=[...]` lists, whose authored
#                values were tuned in these units.
#
# The viewer mirror is `packages/luxar-viewer/src/types/lod-group.ts`; keep the
# spellings in step with it.
#
# The two spellings are defined FIRST and `LOD_SELECTORS` is built from them, so
# the vocabulary cannot drift from the constants that name its members.
# The selector every derived (auto-computed) ladder stamps.
DERIVED_LOD_SELECTOR: Final[str] = "screen-area"
# The UNITS an explicitly authored `coverage_fractions=[...]` list is in — the
# legacy diagonal metric, whose values were tuned against it — and the
# `add_lod_group` default (a hand-built ladder is authored, not derived).
LEGACY_LOD_SELECTOR: Final[str] = "coverage"
LOD_SELECTORS: Final[frozenset[str]] = frozenset(
    {LEGACY_LOD_SELECTOR, DERIVED_LOD_SELECTOR}
)

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

# HDR color constants
COLOR_SDR_MIN: Final[float] = 0.0  # Standard dynamic range minimum
COLOR_SDR_MAX: Final[float] = 1.0  # Standard dynamic range maximum
COLOR_HDR_TYPICAL_MAX: Final[float] = 10.0  # Typical HDR maximum

# Per-axis COORDINATE extent at/above which uint16 per-axis fixed-point can no
# longer resolve a unit step, so AUTO/MEMORY store float32 instead. Read by the
# encoder that applies it (encoding._encoders.perchannel) AND by the gsplat
# writer's sigma rail (io._compiler.gsplat_assembly), which defers to it rather
# than pre-empting its clearer diagnosis — the two must not drift apart.
COORDINATE_U16_MAX_EXTENT: Final[float] = 65_536.0

# Chunk size constants (SINGLE SOURCE OF TRUTH - bytes, not elements)
# Consumers (io, gsplats.io) convert to element counts based on array dtype
TARGET_CHUNK_BYTES: Final[int] = 65_536  # 64KB target chunk size
MIN_CHUNK_BYTES: Final[int] = 16_384  # 16KB minimum to amortize HTTP overhead
MAX_CHUNK_BYTES: Final[int] = 262_144  # 256KB maximum for responsive streaming

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

# Per-node ceiling, in bytes, on what a mesh node may declare. The viewer's
# whole-node mesh loader refuses a node over this before fetching a single chunk,
# so a store above it does not render — it fails with a LoaderError.
#
# This is the write-time twin of that gate, and the twin is DELIBERATELY WEAKER:
# see validate_mesh_decode_budget for exactly which terms it charges and why it
# must under-count rather than over-count. The viewer's number is the authority;
# this is here so `add_mesh` can refuse the cases that provably exceed it.
#
# MIRROR: MESH_DECODE_BUDGET_BYTES in
# packages/luxar-viewer/src/config/constants.ts must hold this value.
MESH_DECODE_BUDGET_BYTES: Final[int] = 512 * 1024 * 1024  # 536,870,912

# Every decoder-routed array materializes as float32 in the viewer, so the
# loader charges 4 bytes per LOGICAL value regardless of the stored dtype.
# MIRROR: DECODED_BYTES_PER_VALUE in packages/luxar-viewer/src/data/mesh/preflight.ts
MESH_DECODED_BYTES_PER_VALUE: Final[int] = 4

# Container/codec family this project writes, for metadata and docs. The REAL
# default is the width-aware per-dtype policy in luxar.encoding.compression
# (zstd level 9 inside Blosc), which is where a level belongs — the
# COMPRESSION_LEVEL_MIN/DEFAULT/MAX trio that used to sit here named 0/3/9,
# agreeing with neither that policy nor the 1-9 band the old
# typing_utils.config validator enforced, and had no reader either way.
DEFAULT_COMPRESSOR: Final[str] = "blosc"  # Default compression algorithm

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

#: The radius, in world units, that the renderer draws every point with when a
#: points node stores no ``radii`` array. It is therefore also the extent the
#: WRITE side must expand such a chunk's spatial bounds by: for a radii-less
#: node the pad IS the footprint, so the stored bound is exactly the set of
#: query positions from which a point in the chunk can be seen — never tighter
#: (the reader cannot miss a drawn disc) and never arbitrarily looser. That
#: holds at any coordinate magnitude: the float32 bounds array is written with
#: outward rounding, so a pad smaller than half an ULP cannot vanish into the
#: store. (The claim is scoped to this no-radii case: with per-point
#: uint8-encoded radii the encoder rounds, so a stored radius can exceed the one
#: the bounds were computed from by up to one quantum.) It is likewise the radius
#: :func:`luxar.core.group.adders.points.add_points_impl` materializes when the
#: caller supplies none, which is what makes the two agree by construction.
#:
#: MIRROR: ``DEFAULT_POINT_RADIUS`` in
#: ``packages/luxar-viewer/src/config/constants.ts`` must hold the same value —
#: that is the constant every viewer site stands in for a missing radii array
#: with. ``rendering/node-factory/create-points-node.ts`` uses it BOTH ways — it
#: fills the per-point radius array and seeds the scalar footprint the bounding
#: box is padded by; ``rendering/gpu-buffer-pool/points-adapter.ts`` only fills
#: the array, and ``data/scene-loader/commit/commit-points-geometry.ts`` only
#: supplies the scalar footprint. Both sides are pinned by tests that
#: name each other.
DEFAULT_POINT_RADIUS: Final[float] = 0.5

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

#: Lower bound on each Cholesky diagonal (a splat's per-axis width), in voxels.
#:
#: ``sqrt(1/12)`` is the standard deviation of a uniform distribution over one
#: voxel — the width at which a Gaussian stops describing structure and starts
#: describing the sampling grid. Below it a splat is narrower than the data can
#: resolve, and the fit spends capacity on a delta it cannot justify.
#:
#: Passing ``sigma_min_diag=None`` removes the bound entirely; that is a
#: deliberate act, not a default. It used to be reachable by accident —
#: ``ConstraintConfig`` defaulted to ``None`` while the fitter defaulted to this
#: value, so unpacking a default-constructed config switched the floor off while
#: reading as "no change" (audit A3-01).
#:
#: Lives here rather than in ``gsplats.fitting.validation``, where it was
#: defined, because ``validation`` imports ``FitConfig`` from
#: ``gsplats.fitting.config`` — so the config module could not name its own
#: default without a circular import. ``validation`` re-exports it for the
#: existing import sites.
DEFAULT_SIGMA_MIN_DIAG: Final[float] = float(math.sqrt(1.0 / 12.0))


# =============================================================================
# Per-node element-texture capacity (issue #1957)
# =============================================================================
#
# The viewer stores per-element render data in an "element texture" whose width
# is capped at ELEMENT_TEXTURE_MAX_WIDTH texels and rounded DOWN to a whole
# number of elements, so one node holds at most
#
#     floor(width * maxTextureSize / texels_per_element)
#
# elements. Beyond that the viewer CLAMPS: ``clampElementCapacity``
# (``packages/luxar-viewer/src/rendering/element-texture-layout.ts``) drops the
# node's tail with a single console warning and no other signal. Because
# geometry is stored in Hilbert order, the lost tail is one spatially
# CONTIGUOUS lobe, so the symptom is a clean-edged wedge of missing geometry
# rather than a scatter — see #1957, where clamping 2.3% of an ocean-current
# Lines node erased the whole North Atlantic.
#
# maxTextureSize is a GPU property (16384 on modern desktop, 4096 on the
# conservative floor), so the only bound an AUTHOR can rely on is the
# 4096-class one below. Nodes above it must be split with
# ``partition=dict(max_elements=...)``.
#
# MIRROR: ``ELEMENT_TEXTURE_MAX_WIDTH`` and the per-type ``texelsPerElement`` in
# ``packages/luxar-viewer/src/rendering/element-texture-layout.ts``.
ELEMENT_TEXTURE_MAX_WIDTH: Final[int] = 4096
CONSERVATIVE_MAX_TEXTURE_SIZE: Final[int] = 4096

#: Texels each geometry type consumes per element in the element texture.
ELEMENT_TEXELS_PER_ELEMENT: Final[dict[str, int]] = {
    "points": 3,
    "lines": 6,
    "gsplats": 4,
}


def max_elements_per_node(geometry_type: str) -> int:
    """Elements one node of ``geometry_type`` can render on a 4096-class GPU.

    The conservative floor: a node at or below this renders whole on ANY GPU;
    above it, a 4096-class GPU silently drops the tail.

    Args:
        geometry_type: A key of :data:`ELEMENT_TEXELS_PER_ELEMENT`.

    Returns:
        The element capacity — segments for lines, points for points, splats
        for gsplats.

    Raises:
        KeyError: ``geometry_type`` has no element-texture layout.
    """
    texels = ELEMENT_TEXELS_PER_ELEMENT[geometry_type]
    width = (ELEMENT_TEXTURE_MAX_WIDTH // texels) * texels
    return (width * CONSERVATIVE_MAX_TEXTURE_SIZE) // texels


#: 2,793,472 segments — the conservative per-node cap for Lines.
MAX_SEGMENTS_PER_LINES_NODE: Final[int] = max_elements_per_node("lines")
#: 5,591,040 points — the conservative per-node cap for Points.
MAX_POINTS_PER_POINTS_NODE: Final[int] = max_elements_per_node("points")
#: 4,194,304 splats — the conservative per-node cap for GSplats.
MAX_SPLATS_PER_GSPLATS_NODE: Final[int] = max_elements_per_node("gsplats")


#: Largest integer JavaScript represents exactly (``Number.MAX_SAFE_INTEGER``,
#: 2**53 - 1). The bound for any attr whose VALUE the viewer must round-trip
#: exactly rather than merely approximately — today ``layer_order``, whose only
#: property is its order relative to other layers, so a magnitude that collapses
#: two distinct orders into one JS number is a silent wrong answer rather than a
#: rounding nicety.
JS_SAFE_INTEGER_MAX: Final[int] = 2**53 - 1
