"""Type validation functions for runtime type checking.

This module contains basic validation functions that are used for type guards,
property validation, and runtime type checking. For detailed write-time validation
with helpful error messages, see validation/base.py.

These validators are simpler and focus on type conversion and basic checks,
while the _for_writing validators provide comprehensive error messages for
users writing data.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union, cast

import numpy as np

from ..typing_utils.aliases import PositionArray, TransformMatrix
from ..typing_utils.constants import (
    ABSORPTION_MIN,
    GAMMA_MAX,
    GAMMA_MIN,
    INTENSITY_MAX,
    INTENSITY_MIN,
    JS_SAFE_INTEGER_MAX,
    LINE_JOIN_STYLES,
    LINK_SCHEMES,
    LINK_TARGETS,
    MAX_COPY_CHARS,
    MAX_LINK_CHARS,
    OFFSET_MAX,
    OFFSET_MIN,
    OPACITY_MAX,
    OPACITY_MIN,
    SHARPNESS_MAX,
    SHARPNESS_MIN,
)
from ..typing_utils.enums import BlendingMode, NodeType, PhysicalUnit


def validate_positions(positions: Any, ndim: Optional[int] = None) -> PositionArray:
    """Validate and convert positions array to correct type.

    Args:
        positions: Input array to validate
        ndim: Expected number of dimensions (optional). If None, any dimensionality is accepted.

    Returns:
        Validated positions array with shape (N, D)

    Raises:
        ValueError: If positions are invalid shape or type
    """
    if not isinstance(positions, np.ndarray):
        raise ValueError("Positions must be a numpy array")

    if positions.ndim != 2:
        raise ValueError(
            f"Positions must have shape (N, D), got shape {positions.shape}"
        )

    if positions.shape[1] < 1:
        raise ValueError(
            f"Positions must have at least 1 dimension, got {positions.shape[1]}"
        )

    if ndim is not None and positions.shape[1] != ndim:
        raise ValueError(f"Expected {ndim} dimensions, got {positions.shape[1]}")

    return positions.astype(np.float32, copy=False)


def validate_colors(
    colors: Any, n_points: int, channels: tuple[int, ...] = (3, 4)
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Validate and convert colors array to HDR float32 format.

    Colors are RGB ``(n, 3)`` or RGBA ``(n, 4)``. The optional alpha column is
    a per-element opacity in [0, 1] (NOT an HDR emission channel): each
    blending mode consumes it the way it consumes node-level opacity, and the
    volumetric mode maps it into optical depth (see
    VOLUMETRIC_BLENDING_SPEC.md).

    Args:
        colors: Input array to validate
        n_points: Expected number of points
        channels: Accepted channel counts. Geometry types opt into RGBA
            per phase (gsplats today; points/lines keep ``(3,)`` until their
            volumetric phases land).

    Returns:
        Validated colors array in HDR float32 format

    Raises:
        ValueError: If colors are invalid shape or type
    """
    if not isinstance(colors, np.ndarray):
        raise ValueError("Colors must be a numpy array")

    if (
        colors.ndim != 2
        or colors.shape[0] != n_points
        or colors.shape[1] not in channels
    ):
        expected = " or ".join(f"({n_points}, {c})" for c in channels)
        raise ValueError(f"Colors must have shape {expected}")

    # RGB channels must be finite too (HDR allows arbitrarily LARGE emission,
    # but never NaN/Inf). This used to be checked only on the alpha column,
    # letting NaN/Inf RGB slip through this lightweight type-guard while the
    # write validator (base.py::_validate_numeric_finite_values) rejected them
    # — a NaN could then poison downstream ranking (effective_amplitudes) and
    # export before any save. Aligns the two validators.
    rgb = colors[:, :3]
    if rgb.size and not np.all(np.isfinite(rgb)):
        raise ValueError("Colors must be finite (no NaN or Inf)")

    if colors.shape[1] == 4 and np.issubdtype(colors.dtype, np.floating):
        # Alpha is opacity, not emission: finite and within [0, 1]. Integer
        # storage is SDR in its native range (full-scale = opaque), so the
        # bound applies to floats only — matching the write validator
        # (validation/base.py::validate_colors_for_writing), which the two
        # functions otherwise contradicted on a uint8 alpha=255 array.
        alpha = colors[:, 3]
        if alpha.size and (
            not np.all(np.isfinite(alpha)) or np.any(alpha < 0) or np.any(alpha > 1)
        ):
            raise ValueError("Color alpha channel must be finite and within [0, 1]")

    # Support HDR colors - use float32 for full HDR range
    # RGB can be any positive value (0.0 to infinity) for HDR emission
    return colors.astype(np.float32, copy=False)


def validate_radii(radii: Any, n_points: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Validate and convert radii array to correct type.

    Args:
        radii: Input array to validate
        n_points: Expected number of points

    Returns:
        Validated radii array

    Raises:
        ValueError: If radii are invalid shape or type
    """
    if not isinstance(radii, np.ndarray):
        raise ValueError("Radii must be a numpy array")
    if radii.ndim != 1:
        raise ValueError("Radii must have shape (N,)")
    # Allow broadcasting: either n_points or 1 element
    if radii.shape[0] != n_points and radii.shape[0] != 1:
        raise ValueError(
            f"Radii shape {radii.shape} doesn't match positions. "
            f"Expected {n_points} elements or 1 (broadcast), got {radii.shape[0]}"
        )
    # Ensure all radii are positive
    if np.any(radii <= 0):
        raise ValueError("All radii must be positive values")
    return radii.astype(np.float32, copy=False)


def validate_sharpness(
    sharpness: Any, n_points: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Validate and convert sharpness array to correct type.

    Sharpness is a normalised [0, 1] knob mapped in the viewer to the
    super-Gaussian falloff exponent beta = 2^(6s - 2): s=0.5 -> beta=2 (a true
    Gaussian), higher s -> harder/crisper edge, lower s -> peakier cusp. Must be
    float32 values in [SHARPNESS_MIN, SHARPNESS_MAX] = [0, 1] with shape (N,)
    where N is the number of points. The full range is also enforced by
    ``base.validate_sharpness_for_writing``.

    Args:
        sharpness: Input sharpness array to validate
        n_points: Expected number of points

    Returns:
        Validated sharpness array as float32

    Raises:
        ValueError: If sharpness values are invalid
    """
    if not isinstance(sharpness, np.ndarray):
        raise ValueError("Sharpness must be a numpy array")

    if sharpness.ndim != 1:
        raise ValueError("Sharpness must have shape (N,)")

    # Allow broadcasting: either n_points or 1 element
    if sharpness.shape[0] != n_points and sharpness.shape[0] != 1:
        raise ValueError(
            f"Sharpness shape {sharpness.shape} doesn't match positions. "
            f"Expected {n_points} elements or 1 (broadcast), got {sharpness.shape[0]}"
        )

    # Sharpness is a normalised [0, 1] knob. The full range is also enforced by
    # base.validate_sharpness_for_writing().
    if np.any(sharpness < SHARPNESS_MIN) or np.any(sharpness > SHARPNESS_MAX):
        raise ValueError(
            f"All sharpness values must be in [{SHARPNESS_MIN}, {SHARPNESS_MAX}]"
        )

    return sharpness.astype(np.float32, copy=False)


def validate_transform(transform: Any) -> TransformMatrix:
    """Validate and convert transform matrix to correct type.

    Args:
        transform: Input transform matrix

    Returns:
        Validated 4x4 transform matrix

    Raises:
        ValueError: If transform is invalid shape or type
    """
    if not isinstance(transform, np.ndarray):
        raise ValueError("Transform must be a numpy array")

    if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix")

    if np.any(np.isnan(transform)):
        raise ValueError("Transform contains NaN values")
    if np.any(np.isinf(transform)):
        raise ValueError("Transform contains Inf values")

    # Verify affine transform: bottom row must be [0, 0, 0, 1]
    bottom = transform[3, :]
    if not np.allclose(bottom, [0, 0, 0, 1], atol=1e-6):
        raise ValueError(
            f"Invalid affine transform: bottom row must be [0, 0, 0, 1], "
            f"got {bottom.tolist()}"
        )

    return transform.astype(np.float32, copy=False)


# Spelling this validator accepts on top of the enum's own member values.
# ``PhysicalUnit`` has both METER ("m") and METRE ("metre") but no member whose
# value is the full American word, so it is listed once, here, rather than
# hand-copied into a vocabulary tuple.
#
# NB this is a NARROWER set than ``PhysicalUnit.validate`` accepts — that
# classmethod additionally normalises "micron", "um", "nanometer" and friends.
# The divergence predates this refactor and is left as-is; only the duplication
# is removed.
_UNIT_SPELLING_ALIASES: Tuple[str, ...] = ("meter",)


def _accepted_node_types() -> Tuple[str, ...]:
    """The node-type spellings this module accepts, derived from ``NodeType``.

    Derived rather than restated: a member added to ``NodeType`` (itself pinned
    to ``format-contract/contract.yaml`` by ``test_node_type_matches_contract``)
    is accepted here automatically. The previous hand-copied tuple had already
    drifted — it omitted ``mesh``, so this validator rejected a first-class,
    shipped geometry type.
    """
    return tuple(member.value for member in NodeType)


def _accepted_unit_spellings() -> Tuple[str, ...]:
    """The unit spellings this module accepts, derived from ``PhysicalUnit``.

    Same reasoning as :func:`_accepted_node_types`: the enum is the vocabulary,
    plus :data:`_UNIT_SPELLING_ALIASES` for spellings it has no member for.
    """
    return tuple(member.value for member in PhysicalUnit) + _UNIT_SPELLING_ALIASES


def validate_node_type(node_type: str) -> NodeType:
    """Validate node type string.

    Args:
        node_type: Input node type string

    Returns:
        Validated node type

    Raises:
        ValueError: If node type is invalid
    """
    valid_types = _accepted_node_types()
    if node_type not in valid_types:
        raise ValueError(
            f"Invalid node type '{node_type}'. Must be one of {valid_types}"
        )
    return cast(NodeType, node_type)


def validate_physical_unit(unit: str) -> PhysicalUnit:
    """Validate physical unit string.

    Args:
        unit: Input unit string

    Returns:
        Validated physical unit

    Raises:
        ValueError: If unit is invalid
    """
    valid_units = _accepted_unit_spellings()
    if unit not in valid_units:
        raise ValueError(f"Invalid unit '{unit}'. Must be one of {valid_units}")
    return cast(PhysicalUnit, unit)


def validate_opacity(opacity: Any) -> float:
    """Validate and convert opacity value.

    Args:
        opacity: Value to validate as opacity (0.0 to 1.0)

    Returns:
        Valid opacity as float

    Raises:
        ValueError: If opacity is not a valid float between 0 and 1
        TypeError: If opacity cannot be converted to float
    """
    try:
        opacity_float = float(opacity)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Opacity must be convertible to float, got {type(opacity).__name__}"
        ) from e

    if not OPACITY_MIN <= opacity_float <= OPACITY_MAX:
        raise ValueError(
            f"Opacity must be between {OPACITY_MIN} and {OPACITY_MAX}, got {opacity_float}"
        )

    return opacity_float


def validate_appearance_fraction(value: Any, name: str) -> float:
    """Validate a finite mesh appearance fraction in ``[0, 1]``."""
    try:
        result = float(value)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"{name} must be convertible to float, got {type(value).__name__}"
        ) from e
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and between 0.0 and 1.0, got {result}")
    return result


def validate_positive_finite(value: Any, name: str) -> float:
    """Validate a strictly-positive finite mesh appearance exponent."""
    try:
        result = float(value)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"{name} must be convertible to float, got {type(value).__name__}"
        ) from e
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than 0, got {result}")
    return result


#: The texture magnification/minification filters the viewer implements.
#:
#: ``linear`` also enables mipmaps + anisotropy at upload; ``nearest`` is the
#: right choice for a categorical or index-like texture, where interpolating
#: between two class ids invents a third that means nothing.
TEXTURE_FILTERS: Tuple[str, ...] = ("linear", "nearest")

#: The texture wrap modes. Applied per-axis by the viewer, which defaults to
#: repeat-in-u / clamp-in-v so an equirectangular basemap tiles across the
#: dateline seam without bleeding the north pole into the south.
TEXTURE_WRAPS: Tuple[str, ...] = ("repeat", "clamp")

#: The transfer functions a mesh texture may declare.
TEXTURE_COLOR_SPACES: Tuple[str, ...] = ("srgb", "linear")


def _validate_texture_choice(value: Any, name: str, allowed: Tuple[str, ...]) -> str:
    """Validate a texture sampling attr against a closed vocabulary."""
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}, got {value!r}")
    return value


def validate_texture_filter(value: Any, name: str) -> str:
    """Validate a mesh ``texture_filter`` attr."""
    return _validate_texture_choice(value, name, TEXTURE_FILTERS)


def validate_texture_wrap(value: Any, name: str) -> str:
    """Validate a mesh ``texture_wrap`` attr."""
    return _validate_texture_choice(value, name, TEXTURE_WRAPS)


def validate_texture_color_space(value: Any, name: str) -> str:
    """Validate a mesh ``texture_color_space`` declaration."""
    return _validate_texture_choice(value, name, TEXTURE_COLOR_SPACES)


def validate_absorption(absorption: Any) -> float:
    """Validate and convert an absorption coefficient (volumetric kappa).

    Absorption is the volumetric blending mode's per-node coefficient:
    multiplicative composition with identity 1.0, no upper bound (it is a
    physical coefficient), and kappa=0 reproduces additive blending exactly.

    Args:
        absorption: Value to validate as absorption (>= 0, finite)

    Returns:
        Valid absorption as float

    Raises:
        ValueError: If absorption is negative, NaN, or infinite
        TypeError: If absorption cannot be converted to float
    """
    try:
        absorption_float = float(absorption)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Absorption must be convertible to float, got {type(absorption).__name__}"
        ) from e

    if math.isnan(absorption_float) or math.isinf(absorption_float):
        raise ValueError(f"Absorption must be finite, got {absorption_float}")

    if absorption_float < ABSORPTION_MIN:
        raise ValueError(
            f"Absorption must be >= {ABSORPTION_MIN}, got {absorption_float}"
        )

    return absorption_float


def _min_truncation_radius_float32() -> float:
    """Smallest ``T`` whose shifted-Gaussian normalization survives float32.

    The CUDA (``compute_shift_params``), Metal (``shift_c``) and GPU shader
    paths all evaluate ``C = exp(-T^2/2)`` in single precision. Below some ``T``
    that rounds to exactly ``1.0f``, making ``1/(1-C)`` infinite — so the usable
    lower bound is set by float32, not float64 (which only degenerates around
    1.5e-8, four orders of magnitude lower).

    Bisected once at import against the real float32 arithmetic rather than
    derived in closed form: the obvious analytic bound ``sqrt(eps32)`` is a
    little too high and misclassifies a band of radii near 2.4e-4.
    """

    def degenerate(t: float) -> bool:
        c = np.exp(np.float32(-0.5) * np.float32(t) * np.float32(t))
        return bool(np.float32(1.0) - c <= np.float32(0.0))

    lo, hi = 1e-12, 1.0  # degenerate at lo, fine at hi
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if degenerate(mid):
            lo = mid
        else:
            hi = mid
    return hi


#: Lower bound for :func:`validate_truncation_radius` (~2.44e-4). See
#: :func:`_min_truncation_radius_float32` for why float32 sets it.
MIN_TRUNCATION_RADIUS_FLOAT32: float = _min_truncation_radius_float32()

#: Upper bound for :func:`validate_truncation_radius` (~1.84e19): the largest
#: float32 whose SQUARE is still finite in float32. The GPU consumes ``T``
#: twice — as ``uTruncate`` and as ``uTruncateSq = T²`` (the fragment discard
#: threshold) — so bounding ``T`` at float32's max is not enough: everything in
#: ``(sqrt(float32.max), float32.max]`` uploads a finite ``uTruncate`` next to
#: an infinite ``uTruncateSq``. One ulp above this value the float32 square
#: overflows (pinned by test).
MAX_TRUNCATION_RADIUS_FLOAT32: float = float(np.sqrt(np.finfo(np.float32).max))


def validate_truncation_radius(truncation_radius: Any) -> float:
    """Validate and convert a GSplat truncation radius ``T`` (in sigmas).

    ``T`` sets both the support of the shifted Gaussian and its normalization
    ``1 / (1 - exp(-T^2/2))``, so a non-positive or non-finite value poisons
    every derived uniform (``uShiftC``, ``uInvOneMinusC``) and collapses the
    world-space cull box. It arrives unvalidated from dataset attrs, so it is
    checked on write.

    A large ``T`` is merely a wide (and eventually untruncated) Gaussian, which
    is well defined — so the upper bound is not a modelling choice, it is the
    largest float32 whose square is still finite in float32. The GPU consumes
    ``T²`` as its own uniform (``uTruncateSq``, the fragment discard
    threshold), so a ``T`` that itself narrows finitely but whose float32
    square overflows is the same degeneracy as the lower bound seen from the
    other end.

    The lower bound is derived, not a magic number. ``T > 0`` alone is *not*
    sufficient: below some threshold ``exp(-T^2/2)`` rounds to exactly 1.0, so
    ``1 / (1 - C)`` is ``inf`` and every downstream value is poisoned — the
    very failure this validator exists to prevent. The check is done in
    **float32**, the narrowest consumer (the CUDA and Metal kernels and the GPU
    shaders all evaluate the shift in single precision), which saturates around
    ``T = 3e-4`` — four orders of magnitude before float64's ~1.5e-8. The
    viewer's read-time clamp (``MIN_TRUNCATION_RADIUS`` in
    ``rendering/materials/gsplat/math.ts``) bisects this same float32 bound, so
    a value accepted here (e.g. 0.05) renders unmodified and stays consistent
    with the chunk bounds computed from it, instead of being silently raised.

    Args:
        truncation_radius: Value to validate as a truncation radius (> 0, finite,
            and large enough that ``1 / (1 - exp(-T^2/2))`` is finite)

    Returns:
        Valid truncation radius as float

    Raises:
        ValueError: If the value is <= 0, NaN, infinite, or so small that the
            shifted-Gaussian normalization overflows
        TypeError: If the value cannot be converted to float
    """
    try:
        radius_float = float(truncation_radius)
    except (ValueError, TypeError) as e:
        raise TypeError(
            "Truncation radius must be convertible to float, got "
            f"{type(truncation_radius).__name__}"
        ) from e

    if math.isnan(radius_float) or math.isinf(radius_float):
        raise ValueError(f"Truncation radius must be finite, got {radius_float}")

    if radius_float <= 0:
        raise ValueError(f"Truncation radius must be > 0, got {radius_float}")

    # The shifted-Gaussian shift C = exp(-T^2/2) and its renormalization
    # 1/(1-C). A T small enough that C rounds to 1.0 makes that infinite.
    # The bound is the FLOAT32 one (see MIN_TRUNCATION_RADIUS_FLOAT32): a plain
    # float comparison, because doing the float32 arithmetic per call cost ~5x
    # the rest of the validator and made up over half of AdditiveSubLOD
    # construction time.
    if radius_float > MAX_TRUNCATION_RADIUS_FLOAT32:
        raise ValueError(
            f"Truncation radius {radius_float} is too large: its square "
            "overflows float32, so it reaches the GPU render paths as an "
            "infinite uTruncateSq (and beyond float32's max, as an infinite "
            "uTruncate too)"
        )

    if radius_float < MIN_TRUNCATION_RADIUS_FLOAT32:
        raise ValueError(
            f"Truncation radius {radius_float} is too small: exp(-T^2/2) rounds "
            "to 1.0 in float32, so the shifted-Gaussian normalization 1/(1-C) "
            "overflows on the GPU render paths"
        )

    return radius_float


def validate_gamma(gamma: Any) -> float:
    """Validate and convert gamma value.

    Args:
        gamma: Value to validate as gamma (0.1 to 10.0)

    Returns:
        Valid gamma as float

    Raises:
        ValueError: If gamma is not within valid range
        TypeError: If gamma cannot be converted to float
    """
    try:
        gamma_float = float(gamma)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Gamma must be convertible to float, got {type(gamma).__name__}"
        ) from e

    if not GAMMA_MIN <= gamma_float <= GAMMA_MAX:
        raise ValueError(
            f"Gamma must be between {GAMMA_MIN} and {GAMMA_MAX}, got {gamma_float}"
        )

    return gamma_float


def validate_intensity(intensity: Any) -> float:
    """Validate and convert intensity value.

    Args:
        intensity: Value to validate as intensity (0.0 to 100.0)

    Returns:
        Valid intensity as float

    Raises:
        ValueError: If intensity is not within valid range
        TypeError: If intensity cannot be converted to float
    """
    try:
        intensity_float = float(intensity)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Intensity must be convertible to float, got {type(intensity).__name__}"
        ) from e

    if not INTENSITY_MIN <= intensity_float <= INTENSITY_MAX:
        raise ValueError(
            f"Intensity must be between {INTENSITY_MIN} and {INTENSITY_MAX}, "
            f"got {intensity_float}"
        )

    return intensity_float


def validate_offset(offset: Any) -> float:
    """Validate and convert offset value.

    Args:
        offset: Value to validate as offset (-10.0 to 10.0)

    Returns:
        Valid offset as float

    Raises:
        ValueError: If offset is not within valid range
        TypeError: If offset cannot be converted to float
    """
    try:
        offset_float = float(offset)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Offset must be convertible to float, got {type(offset).__name__}"
        ) from e

    if not OFFSET_MIN <= offset_float <= OFFSET_MAX:
        raise ValueError(
            f"Offset must be between {OFFSET_MIN} and {OFFSET_MAX}, got {offset_float}"
        )

    return offset_float


def validate_layer(value: Any) -> bool:
    """Validate and convert layer flag.

    Args:
        value: Value to validate as a boolean layer flag

    Returns:
        Valid layer flag as bool

    Raises:
        TypeError: If value cannot be interpreted as a boolean
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raise TypeError(f"Layer must be a boolean, got {type(value).__name__}")


def validate_link(value: Any) -> str:
    """Validate a node's ``link`` URL template (issue #1917).

    The template is substituted per element in the viewer and the result is
    NAVIGATED to, so the checks here mirror the viewer's own and exist to turn
    an authoring mistake into a failed write rather than a link that silently
    does nothing when clicked.

    Validated on the template with its placeholders left in place: they are
    ``{...}`` runs, which no URL parser objects to, so scheme and shape can be
    settled at write time. What cannot be settled here is the substituted
    result — a label supplies the rest of the URL at click time — which is why
    the viewer re-checks after substitution. Substituted values are
    percent-encoded there, so a placeholder cannot introduce a scheme, a host,
    or a path segment that is not already visible in this template.

    Args:
        value: Value to validate as a URL template

    Returns:
        The template unchanged

    Raises:
        TypeError: If value is not a string
        ValueError: If the template is empty, over-long, has no scheme
            (relative), or names a scheme outside ``LINK_SCHEMES``
    """
    from urllib.parse import urlsplit

    if not isinstance(value, str):
        raise TypeError(f"link must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError("link must not be empty")
    if len(value) > MAX_LINK_CHARS:
        raise ValueError(
            f"link is {len(value)} characters, exceeding the "
            f"{MAX_LINK_CHARS}-character limit"
        )

    try:
        parts = urlsplit(value)
    except ValueError as exc:
        raise ValueError(f"link is not a parseable URL: {value!r} ({exc})") from exc

    scheme = parts.scheme.lower()
    if not scheme:
        # A relative template would resolve against whatever origin the VIEWER
        # is served from, so a third-party store could aim a click at the
        # embedder's own site. Absolute or nothing.
        raise ValueError(
            f"link must be an absolute URL, got the relative {value!r}. "
            f"A relative link would resolve against the viewer's own origin. "
            f"Prefix it with {' or '.join(sorted(s + '://' for s in LINK_SCHEMES))}."
        )
    if scheme not in LINK_SCHEMES:
        raise ValueError(
            f"link scheme {scheme!r} is not allowed "
            f"(permitted: {', '.join(sorted(LINK_SCHEMES))}). "
            f"Got {value!r}."
        )
    if not parts.netloc:
        # Deliberately STRICTER than the viewer here. For a "special" scheme
        # the browser's WHATWG parser collapses `https:///nowhere` into host
        # `nowhere`, so it would navigate happily; `urlsplit` instead reports
        # an empty netloc and a `/nowhere` path. Refusing at authoring time is
        # the right way round — that spelling is a typo far more often than an
        # intent, and failing the write costs nothing, whereas the viewer must
        # faithfully model what the browser will do with a store it did not
        # author. Do not "align" this by dropping the check.
        raise ValueError(f"link has no host: {value!r}")

    # Refuse embedded credentials. `https://good.example@evil.example/`
    # navigates to evil.example while reading as good.example — including in
    # the viewer's own "Copy link address", which is the one place a user might
    # vet the destination before following it. Nothing legitimate needs
    # userinfo in a scene link, and credentials inside a shareable scene file
    # would be a mistake in their own right.
    if parts.username or parts.password:
        raise ValueError(
            f"link must not embed credentials (user:pass@host): {value!r}. "
            f"They disguise the real destination from anyone reading the URL."
        )

    return value


def validate_copy_template(value: Any) -> str:
    """Validate a node's ``copy`` template (issue #1917).

    Plain text destined for the clipboard, not a URL — so there is nothing to
    check beyond the type and a length bound. It is deliberately NOT escaped
    or restricted in content: the whole point is to hand the user the string
    the author chose.

    Args:
        value: Value to validate as a copy template

    Returns:
        The template unchanged

    Raises:
        TypeError: If value is not a string
        ValueError: If the template is empty or over-long
    """
    if not isinstance(value, str):
        raise TypeError(f"copy must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError("copy must not be empty")
    if len(value) > MAX_COPY_CHARS:
        raise ValueError(
            f"copy is {len(value)} characters, exceeding the "
            f"{MAX_COPY_CHARS}-character limit"
        )
    return value


def validate_link_target(value: Any) -> str:
    """Validate a node's ``link_target`` browsing context (issue #1917).

    Only the two keywords that imply ``noopener`` are accepted. Any other
    value — including a near-miss like ``"_blank "`` — is a *named* target,
    which the browser opens with a live ``window.opener`` the destination can
    use to cross-origin navigate the viewer tab.

    Args:
        value: Value to validate as a browsing context

    Returns:
        The target unchanged

    Raises:
        TypeError: If value is not a string
        ValueError: If value is not one of ``LINK_TARGETS``
    """
    if not isinstance(value, str):
        raise TypeError(f"link_target must be a string, got {type(value).__name__}")
    if value not in LINK_TARGETS:
        raise ValueError(
            f"link_target must be one of {', '.join(sorted(LINK_TARGETS))}, "
            f"got {value!r}"
        )
    return value


def validate_visible(value: Any) -> bool:
    """Validate and convert visible flag.

    Args:
        value: Value to validate as a boolean visibility flag

    Returns:
        Valid visible flag as bool

    Raises:
        TypeError: If value cannot be interpreted as a boolean
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raise TypeError(f"Visible must be a boolean, got {type(value).__name__}")


def validate_blending_mode(mode: Any) -> BlendingMode:
    """Validate blending mode string.

    Args:
        mode: Blending mode to validate. Accepted values are ``"normal"``,
            ``"additive"``, ``"max"``, ``"opaque"``, ``"luminous"``, and
            ``"volumetric"``. Volumetric mode uses emission-absorption
            compositing scaled by the node's ``absorption`` (kappa) attribute.

    Returns:
        Valid blending mode

    Raises:
        ValueError: If mode is not a valid blending mode
        TypeError: If mode is not a string
    """
    if not isinstance(mode, str):
        raise TypeError(f"Blending mode must be a string, got {type(mode).__name__}")

    # Derived from the enum so adding a mode cannot desync validation.
    valid_modes = {m.value for m in BlendingMode}
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid blending mode '{mode}'. Must be one of: {', '.join(sorted(valid_modes))}"
        )

    return cast(BlendingMode, mode)


def validate_layer_order(order: Any) -> int:
    """Validate an authored cross-layer draw order.

    ``layer_order`` states where a layer draws relative to the other layers it
    overlaps: higher = nearer the camera = drawn later, the CSS ``z-index`` /
    Illustrator convention. See ``docs/guides/specs/LAYER_ORDER_SPEC.md``.

    Signed, and bounded only by what survives the trip to the viewer: negative
    values are the natural way to push a backdrop behind everything else, and
    sparse values (10/20/30) leave room to insert a layer later without
    renumbering.

    The bound is the JavaScript safe-integer range, and it is not an arbitrary
    clamp. This attr is written to JSON and read by the viewer as a JS
    ``number``, where every integer above 2**53 - 1 shares its representation
    with its neighbours. Since the ONLY property this value has is its order
    relative to other layers, a magnitude past that point can silently collapse
    two orders the author deliberately separated into one band — handing the
    choice between those layers back to the geometry-derived inference the whole
    attribute exists to override. A loud refusal at write time is the only place
    that failure can be caught.

    A ``bool`` is refused even though ``isinstance(True, int)`` is true in
    Python: ``layer_order=True`` almost certainly means the author confused this
    with a flag, and silently banding that layer at order 1 would be a wrong
    answer rather than an error.

    Args:
        order: Draw order to validate. A Python integer whose magnitude is at
            most ``JS_SAFE_INTEGER_MAX``.

    Returns:
        The order as a plain ``int``.

    Raises:
        TypeError: If ``order`` is not an integer (``bool`` included).
        ValueError: If ``order`` is outside the JS safe-integer range.
    """
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, int):
        raise TypeError(
            f"layer_order must be an integer, got {order!r} ({type(order).__name__}). "
            "Higher draws nearer the camera (CSS z-index convention); sparse values "
            "like 10/20/30 leave room to insert a layer later."
        )

    value = int(order)
    if abs(value) > JS_SAFE_INTEGER_MAX:
        raise ValueError(
            f"layer_order={value} exceeds the JavaScript safe-integer range "
            f"(+/-{JS_SAFE_INTEGER_MAX}). The viewer reads this attr as a JS number, "
            "where integers past that magnitude share a representation with their "
            "neighbours — so two orders you separated could silently collapse into one "
            "band and the draw order between those layers would fall back to being "
            "inferred from the geometry. Only the ORDER matters, never the magnitude: "
            "use small sparse values such as 10/20/30."
        )

    return value


def validate_line_join(style: Any) -> str:
    """Validate a line join style string.

    Lines-only (issue #790): the strategy the line vertex stage uses at a
    degree-2 polyline joint. It has no meaning for points, gsplats or mesh,
    none of which build a screen-space quad per element — the adders for those
    three refuse it, so this validator does not need to know the geometry type.

    Args:
        style: Join style to validate. Accepted values are ``"none"`` (leave the
            two quads alone, so the turn leaves an uncovered wedge) and
            ``"miter"`` (rotate each quad's end edge onto the shared miter edge
            so the two tile).

    Returns:
        Valid join style

    Raises:
        ValueError: If style is not a known join style
        TypeError: If style is not a string
    """
    if not isinstance(style, str):
        raise TypeError(f"Line join style must be a string, got {type(style).__name__}")

    if style not in LINE_JOIN_STYLES:
        raise ValueError(
            f"Unknown line join style {style!r}. "
            f"Expected one of {sorted(LINE_JOIN_STYLES)}."
        )

    return style


def validate_colormap(value: Any) -> Union[str, "np.ndarray[Any, Any]"]:
    """Validate a colormap specification.

    Accepts:
        - A string (colormap name, resolved at write time)
        - A numpy array of shape (N, 3) with N >= 2

    Args:
        value: Colormap name or LUT array.

    Returns:
        Validated colormap (string or numpy array).

    Raises:
        TypeError: If value is not a string or numpy array.
        ValueError: If array has wrong shape or values out of range.
    """
    if isinstance(value, str):
        if not value:
            raise ValueError("Colormap name must be non-empty")
        # Fast-path: accept the sentinel used by the writer for resolved
        # custom/array colormaps (round-tripped through attrs).
        if value == "custom":
            return value
        # Fail fast on unknown names: resolve_colormap walks builtin →
        # matplotlib → colorcet and raises ValueError if the name is
        # found nowhere. Catches typos like "viridus" at authoring time
        # rather than producing a silently-empty colormap downstream.
        from ..colormaps.registry import resolve_colormap

        try:
            resolve_colormap(value)
        except (ValueError, ImportError) as e:
            raise ValueError(str(e)) from e
        return value

    if isinstance(value, np.ndarray):
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError(
                f"Colormap array must have shape (N, 3), got {value.shape}"
            )
        if value.shape[0] < 2:
            raise ValueError(
                f"Colormap array must have at least 2 entries, got {value.shape[0]}"
            )
        # Validate dtype: must be float (range [0,1]) or uint8 (range [0,255])
        if np.issubdtype(value.dtype, np.floating):
            if np.any(value < 0) or np.any(value > 1):
                raise ValueError("Float colormap values must be in [0, 1] range")
        elif value.dtype != np.uint8:
            raise TypeError(
                f"Colormap array must be float32/float64 (range [0,1]) "
                f"or uint8, got {value.dtype}"
            )
        return value

    raise TypeError(
        f"Colormap must be a string or numpy array, got {type(value).__name__}"
    )


# Type guards (return bool for conditional type narrowing)
def validate_finite_reveal_coords(coords: Any, what: str) -> None:
    """Refuse non-finite coordinates feeding a ``radial`` reveal ordering.

    A reveal ranks elements by distance, and a NaN/inf coordinate poisons that
    rank in a way that LOOKS like success: the distances come back non-finite,
    they all compare equal under the stable argsort the three orderings use, and
    the ladder is emitted in INPUT order — a streaming node that fills in at
    random instead of growing outward, with nothing to say why.

    This is the DATA-side twin of the ``reveal_centre`` finite check. That one
    guards a value the user typed; this one guards the array, and it is needed
    separately because the three geometries were measured to disagree without it:
    Points and GSplats returned input order silently, while Lines raised a
    ``reveal_centre must be finite`` error naming a knob the caller never passed
    (its default centre is DERIVED from the vertices, so bad data reached the
    knob's validator wearing the knob's name). One shared validator, called by
    both scorers, is what makes the three agree — hence its home here rather than
    in either implementation.

    Args:
        coords: The coordinate array about to be scored, ``(N, d)``.
        what: Caller-facing name of the array for the message (e.g.
            ``"coords"``, ``"vertices"``, ``"centers"``), so the error blames the
            input the caller actually supplied.

    Raises:
        ValueError: If any entry is NaN or infinite.
    """
    arr = np.asarray(coords)
    if arr.size == 0:
        return
    finite = np.isfinite(arr)
    if bool(finite.all()):
        return
    bad_rows = np.flatnonzero(~finite.all(axis=tuple(range(1, arr.ndim))))
    raise ValueError(
        f"{what} must be finite for a radial reveal: a NaN/inf coordinate makes "
        f"every distance non-finite, so the stable argsort leaves the ladder in "
        f"INPUT order instead of revealing outward. "
        f"{bad_rows.size} of {arr.shape[0]} rows are non-finite "
        f"(first at index {int(bad_rows[0])})."
    )


def validate_integral_axis_indices(values: Any, name: str = "spatial_dims") -> None:
    """Refuse a non-integral axis index before it is silently truncated to one.

    ``int(1.9)`` and ``np.asarray([1.9], dtype=np.intp)`` both give ``1`` without
    a word, so a fractional entry measures the reveal over a DIFFERENT column than
    the caller named — and, when ``reveal_centre`` is given too, pairs that centre
    coordinate with the wrong axis. Every other malformed ``spatial_dims``
    (empty, negative, repeated, nested, out of range) is already rejected; this
    was the one that got through wearing a plausible answer.

    Silent on an integer-dtype input (the overwhelmingly common case) and on an
    empty one, which the callers' own "must not be empty" rule reports better.

    Args:
        values: The candidate index sequence, before any int conversion.
        name: Caller-facing name of the argument, for the message.

    Raises:
        ValueError: If any entry is finite-but-fractional, NaN, or infinite.
    """
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer) or arr.size == 0:
        return
    bad = [
        v
        for v in np.asarray(arr, dtype=np.float64).ravel().tolist()
        if not float(v).is_integer()
    ]
    if bad:
        raise ValueError(
            f"{name} must be integer column indices; got non-integral {bad} "
            f"(truncating would silently measure a different axis than the one "
            f"named)"
        )


def is_position_array(obj: Any) -> bool:
    """Check if object is a valid position array."""
    try:
        validate_positions(obj)
        return True
    except (ValueError, TypeError):
        return False


def is_color_array(obj: Any, n_points: int) -> bool:
    """Check if object is a valid color array."""
    try:
        validate_colors(obj, n_points)
        return True
    except (ValueError, TypeError):
        return False


def is_transform_matrix(obj: Any) -> bool:
    """Check if object is a valid transform matrix."""
    try:
        validate_transform(obj)
        return True
    except (ValueError, TypeError):
        return False


def validate_category_indices(
    values: np.ndarray, categories: List[str], context: str = "values"
) -> None:
    """Validate that array values are valid category indices.

    For categorical dimensions, point coordinates should be integer indices
    into the categories list (0-indexed).

    Args:
        values: 1D array of values to validate
        categories: List of category labels
        context: Context string for error messages

    Raises:
        ValueError: If values contain invalid category indices
    """
    if len(categories) == 0:
        raise ValueError("categories list cannot be empty")

    max_index = len(categories) - 1

    # Check for out-of-range values
    min_val = np.min(values)
    max_val = np.max(values)

    if min_val < 0:
        # Find first negative index for error message
        neg_indices = np.where(values < 0)[0]
        first_neg = neg_indices[0]
        raise ValueError(
            f"negative category index {values[first_neg]} at position {first_neg} in {context}"
        )

    if max_val > max_index:
        # Find first out-of-range index for error message
        out_indices = np.where(values > max_index)[0]
        first_out = out_indices[0]
        raise ValueError(
            f"category index {int(values[first_out])} at position {first_out} is out of range "
            f"[0, {max_index}] in {context}. Valid categories: {categories}"
        )

    # Check that values are integers (or very close to integers)
    if not np.allclose(values, np.round(values)):
        non_int_indices = np.where(~np.isclose(values, np.round(values)))[0]
        first_non_int = non_int_indices[0]
        raise ValueError(
            f"non-integer category index {values[first_non_int]} at position {first_non_int} in {context}"
        )
