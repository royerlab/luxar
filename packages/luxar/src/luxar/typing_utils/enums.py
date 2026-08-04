"""luxar.enums – Enumerations for type-safe constants.

This module provides enumerations for various string literals used throughout
the Luxar codebase, ensuring type safety and preventing typos.
"""

from enum import Enum


class BlendingMode(str, Enum):
    """Blending modes for points/lines/gsplats rendering.

    These control how overlapping elements combine their colors:

    * ``NORMAL``: standard alpha blending (semi-transparent).
    * ``ADDITIVE``: classic additive blending; ignores depth.
    * ``MAX``: maximum of source and destination (brightest wins).
    * ``OPAQUE``: solid rendering with depth writes (closest wins).
    * ``LUMINOUS``: additive appearance while respecting depth occlusion.
    * ``VOLUMETRIC``: emission-absorption compositing (Max 1995). It adds
      emitted light and exponentially attenuates what is behind, scaled by
      the node's ``absorption`` (kappa) attribute. Kappa 0 renders exactly
      like ``ADDITIVE``. See
      ``docs/guides/specs/VOLUMETRIC_BLENDING_SPEC.md``.

    Depth behavior:

    * ``ADDITIVE``: ``depthTest=false``, ``depthWrite=false``.
    * ``LUMINOUS``: ``depthTest=true``, ``depthWrite=false``.
    * ``OPAQUE``: ``depthTest=true``, ``depthWrite=true``.
    * ``NORMAL``: ``depthTest=true`` and ``depthWrite=true`` when opacity is
      at least 0.99 for points and lines. GSplats never depth-write in normal
      mode because coverage-alpha fragments would create occlusion halos.
    * ``MAX``: ``depthTest=true``, ``depthWrite=false``.
    * ``VOLUMETRIC``: ``depthTest=true``, ``depthWrite=false``; requires
      back-to-front depth sorting in the viewer.

    ``OPAQUE`` is the only mode that escapes the viewer's sorted (transparent)
    set entirely and the only one that unconditionally writes depth, so a
    backdrop must be ``OPAQUE`` to be reliably composited *under* the
    transparent content drawn in front of it.

    Validation of raw strings lives in
    :func:`luxar.validation.types.validate_blending_mode`, which derives its
    accepted set from this enum.
    """

    NORMAL = "normal"  # Standard alpha blending
    ADDITIVE = "additive"  # Ignores depth entirely (renders on top of everything)
    MAX = "max"  # Maximum of source and destination (brightest wins)
    OPAQUE = "opaque"  # Solid rendering with depth write
    LUMINOUS = "luminous"  # Same visual as additive, but respects depth occlusion
    VOLUMETRIC = (
        "volumetric"  # Emission-absorption: adds light AND absorbs what's behind
    )


class NodeType(str, Enum):
    """Types of nodes in the scene hierarchy.

    The members must stay in step with ``node_types`` in
    ``format-contract/contract.yaml`` (projected as
    :data:`luxar.typing_utils._format_contract.NODE_TYPES`), which is the
    cross-language source of truth. ``test_node_type_matches_contract`` pins the
    two together, so a contract edit that misses this enum fails a test rather
    than drifting.
    """

    SCENE = "scene"
    GROUP = "group"
    POINTS = "points"
    LINES = "lines"
    GSPLATS = "gsplats"
    MESH = "mesh"

    @classmethod
    def validate(cls, value: str) -> "NodeType":
        """Validate and convert string to NodeType.

        Args:
            value: String representation of node type

        Returns:
            NodeType enum value

        Raises:
            ValueError: If value is not a valid node type
        """
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join([f"'{nt.value}'" for nt in cls])
            raise ValueError(f"Invalid node type '{value}'. Must be one of: {valid}")


class PhysicalUnit(str, Enum):
    """Physical units for spatial dimensions.

    Supports metric, imperial, and specialized units.
    """

    # Metric units
    NANOMETER = "nm"
    MICROMETER = "um"
    MILLIMETER = "mm"
    CENTIMETER = "cm"
    METER = "m"
    METRE = "metre"
    KILOMETER = "km"

    # Imperial units
    INCH = "inch"
    FOOT = "foot"

    # Other units
    PIXEL = "px"
    ASTRONOMICAL_UNIT = "au"

    @classmethod
    def validate(cls, value: str) -> "PhysicalUnit":
        """Validate and convert string to PhysicalUnit.

        Args:
            value: String representation of physical unit

        Returns:
            PhysicalUnit enum value

        Raises:
            ValueError: If value is not a valid physical unit
        """
        # Handle common variations
        normalized = value.lower()
        unit_map = {
            "nanometer": cls.NANOMETER,
            "nanometre": cls.NANOMETER,
            "micrometer": cls.MICROMETER,
            "micrometre": cls.MICROMETER,
            "micron": cls.MICROMETER,
            "μm": cls.MICROMETER,
            "millimeter": cls.MILLIMETER,
            "millimetre": cls.MILLIMETER,
            "centimeter": cls.CENTIMETER,
            "centimetre": cls.CENTIMETER,
            "meter": cls.METER,
            "metre": cls.METRE,
            "kilometer": cls.KILOMETER,
            "kilometre": cls.KILOMETER,
            "pixel": cls.PIXEL,
            "pixels": cls.PIXEL,
        }

        if normalized in unit_map:
            return unit_map[normalized]

        try:
            return cls(value)
        except ValueError:
            valid = ", ".join([f"'{unit.value}'" for unit in cls])
            raise ValueError(
                f"Invalid physical unit '{value}'. Must be one of: {valid}"
            )


# Rendering property ranges
class RenderingLimits:
    """Valid ranges for rendering properties."""

    OPACITY_MIN = 0.0
    OPACITY_MAX = 1.0
    ABSORPTION_MIN = 0.0  # No MAX: kappa is an unbounded physical coefficient
    GAMMA_MIN = 0.1  # Symmetric: gamma and 1/gamma have equal range
    GAMMA_MAX = 10.0  # Symmetric: gamma and 1/gamma have equal range
    SHARPNESS_MIN = 0.0  # Normalised [0, 1] knob (mirrors constants.SHARPNESS_MIN)
    SHARPNESS_MAX = 1.0

    # HDR color ranges
    COLOR_SDR_MIN = 0.0  # Standard dynamic range minimum
    COLOR_SDR_MAX = 1.0  # Standard dynamic range maximum
    COLOR_HDR_MAX = 10.0  # HDR maximum (can go higher but this is practical)


# Default values
class Defaults:
    """Default values for various properties."""

    OPACITY = 1.0
    GAMMA = 1.0
    ABSORPTION = 1.0  # kappa identity: volumetric mode's absorption coefficient
    SHARPNESS = 0.5  # Normalised knob -> super-Gaussian beta = 2 (Gaussian)
    # Blending default lives in typing_utils.constants.DEFAULT_BLENDING_MODE
    CHUNK_SIZE = 32768  # Default chunk size in elements
    RADIUS = 0.1

    # Default colors
    COLOR_WHITE = (1.0, 1.0, 1.0)
    COLOR_BLACK = (0.0, 0.0, 0.0)
