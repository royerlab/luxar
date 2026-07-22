"""Validation utilities with helpful error messages.

This module provides validation functions with detailed, user-friendly error
messages that help users understand and fix issues quickly.
"""

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..typing_utils.constants import (
    SHARPNESS_MAX,
    SHARPNESS_MIN,
)


class ValidationError(ValueError):
    """Custom validation error with helpful suggestions."""

    def __init__(self, message: str, suggestion: Optional[str] = None) -> None:
        """Initialize validation error.

        Args:
            message: The error message
            suggestion: Optional suggestion for fixing the error
        """
        full_message = message
        if suggestion:
            full_message += f"\n💡 Suggestion: {suggestion}"
        super().__init__(full_message)


def validate_node_name(name: Any, context: str = "node name") -> str:
    """Validate a scene-graph node name (a single zarr path segment).

    This is the single chokepoint for node naming, shared by ``Node.__init__``
    (groups + all node objects), the ``add_points``/``add_lines``/``add_gsplats``
    adders, and the compiler writers (via ``validate_node_path``). Rules:

    - Must be a non-empty, non-whitespace-only string. An empty segment is the
      worst case: ``zarr.require_group("")`` resolves to the store ROOT group,
      so a node named ``""`` would stamp ``type='points'`` onto the scene root
      and make the whole store unloadable.
    - Must not contain ``/`` — that is the zarr path separator; use
      ``add_group()`` for hierarchy.
    - Must not start with ``.``. Zarr v2 reserves the dot-prefixed keys
      ``.zgroup`` / ``.zattrs`` / ``.zarray`` / ``.zmetadata`` for its own
      metadata objects; a node named ``.zgroup`` dies with a deep ``KeyError``
      inside zarr and ``.zmetadata`` silently collides with consolidated
      metadata. We reject the entire dot-prefixed namespace (stricter than the
      exact reserved set, but safe: it also covers future zarr metadata keys
      and hidden dot-files that most tooling cannot see).
    - Must not contain control characters (``\\x00``–``\\x1f``, ``\\x7f``) —
      filesystem and JSON hazards for DirectoryStore-backed scenes.

    Args:
        name: Candidate node name.
        context: Context for error messages.

    Returns:
        The validated name (unchanged).

    Raises:
        ValidationError: If the name is invalid.
    """
    if not isinstance(name, str):
        raise ValidationError(
            f"{context}: Expected a string, got {type(name).__name__}",
            "Node names must be strings, e.g. scene.add_points('my_points', ...)",
        )

    if not name.strip():
        raise ValidationError(
            f"{context}: Name must not be empty or whitespace-only (got {name!r}). "
            "An empty name resolves to the zarr ROOT group and would overwrite "
            "the scene root, corrupting the store.",
            "Provide a non-empty node name, e.g. 'points'",
        )

    if "/" in name:
        raise ValidationError(
            f"{context}: Name cannot contain '/': got {name!r}",
            "Use add_group() to create hierarchical structure instead",
        )

    if name.startswith("."):
        raise ValidationError(
            f"{context}: Name cannot start with '.': got {name!r}. "
            "Zarr reserves dot-prefixed keys (.zgroup/.zattrs/.zarray/.zmetadata) "
            "for its own metadata.",
            "Rename the node without the leading dot",
        )

    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in name):
        raise ValidationError(
            f"{context}: Name contains control characters: got {name!r}",
            "Remove control characters (newlines, tabs, NUL, ...) from the name",
        )

    return name


def validate_labels_for_writing(
    labels: Any, n_elements: int, context: str = "labels"
) -> None:
    """Validate per-element string labels BEFORE any zarr write.

    The CSR label serializer UTF-8-encodes each entry; a non-string entry used
    to die with a deep ``AttributeError`` after the node's arrays were already
    on disk. This pre-flight validator runs in the writers' fail-fast gate.

    Args:
        labels: Candidate labels. Must be a sequence (not a bare string) of
            ``str`` entries; ``None`` entries are allowed (null label).
        n_elements: Expected number of elements.
        context: Context for error messages.

    Raises:
        ValidationError: If labels are not a sequence of str/None with one
            entry per element.
    """
    # A bare string is a Sequence[str] of characters — always a bug. Numpy
    # string arrays are accepted (their elements are np.str_, a str subclass).
    if isinstance(labels, (str, bytes)) or not isinstance(
        labels, (Sequence, np.ndarray)
    ):
        raise ValidationError(
            f"{context}: Expected a sequence of strings, got {type(labels).__name__}",
            "Pass one label string per element, e.g. labels=['a', 'b', ...]",
        )

    if len(labels) != n_elements:
        raise ValidationError(
            f"{context}: Labels length ({len(labels)}) must match element "
            f"count ({n_elements})",
            f"Provide exactly {n_elements} labels (use '' or None for no label)",
        )

    for i, label in enumerate(labels):
        if label is not None and not isinstance(label, str):
            raise ValidationError(
                f"{context}: Label at index {i} is {type(label).__name__}, "
                f"expected str or None (got {label!r})",
                "Convert labels to strings, e.g. labels=[str(x) for x in values]",
            )


def _validate_numeric_finite_values(array: NDArray[Any], context: str) -> None:
    """Validate that an array has numeric dtype and contains only finite values."""
    if not np.issubdtype(array.dtype, np.number):
        raise ValidationError(
            f"{context}: Expected numeric array, got dtype {array.dtype}",
            "Convert your data to a numeric dtype such as np.float32",
        )

    try:
        finite_mask = np.isfinite(array)
    except TypeError as exc:
        raise ValidationError(
            f"{context}: Expected numeric finite values, got dtype {array.dtype}",
            "Convert your data to a numeric dtype and remove invalid values",
        ) from exc

    if not bool(np.all(finite_mask)):
        invalid_count = int(np.size(array) - np.count_nonzero(finite_mask))
        raise ValidationError(
            f"{context}: Contains {invalid_count} NaN or Inf value(s)",
            "Remove or replace invalid values before writing, e.g. np.nan_to_num(data)",
        )


def validate_positions_for_writing(
    positions: NDArray[Any], context: str = "positions"
) -> Tuple[int, int]:
    """Validate positions array for writing to Zarr.

    Args:
        positions: Positions array to validate
        context: Context for error messages

    Returns:
        Tuple of (n_points, n_dims). NOTE the array itself is NOT
        returned and is NOT dtype-converted: the validator deliberately
        accepts any numeric dtype (float32/float64/int) and verifies
        SHAPE + FINITENESS only. Callers that need float32 storage MUST
        convert via ``ensure_float32`` (or equivalent) AFTER calling
        this function — see the ``test_validate_positions_dtype_not_checked``
        regression-lock test for the pinned contract.

    Raises:
        ValidationError: If positions are invalid (wrong shape, empty,
            zero-dimensional, or containing NaN / ±Inf).
    """
    if not isinstance(positions, np.ndarray):
        raise ValidationError(
            f"{context}: Expected numpy array, got {type(positions).__name__}",
            "Convert your data to a numpy array using np.array(data)",
        )

    if positions.ndim != 2:
        if positions.ndim == 1:
            raise ValidationError(
                f"{context}: Got 1D array with {len(positions)} elements. "
                f"Positions must be a 2D array with shape (n_points, n_dimensions).",
                f"If you have {len(positions)} 1D points, reshape with: positions.reshape(-1, 1)",
            )
        elif positions.ndim == 3:
            raise ValidationError(
                f"{context}: Got 3D array with shape {positions.shape}. "
                "Positions must be a 2D array with shape (n_points, n_dimensions).",
                "If you have multiple time steps, split them into separate nodes (e.g., points_t0, points_t1, ...)",
            )
        else:
            raise ValidationError(
                f"{context}: Expected 2D array, got {positions.ndim}D array with shape {positions.shape}"
            )

    n_points, n_dims = positions.shape

    if n_points == 0:
        raise ValidationError(
            f"{context}: Cannot write empty points (0 points)",
            "Ensure your positions array contains at least one point",
        )

    if n_dims == 0:
        raise ValidationError(
            f"{context}: Points have 0 dimensions",
            "Each point must have at least 1 dimension (e.g., 1D, 2D, 3D)",
        )

    _validate_numeric_finite_values(positions, context)

    if n_dims > 10:
        import warnings

        warnings.warn(
            f"{context}: Writing {n_dims}D points. "
            "Visualization may be limited to the first 3 dimensions.",
            UserWarning,
        )

    return n_points, n_dims


def validate_colors_for_writing(
    colors: NDArray[Any],
    n_points: int,
    context: str = "colors",
    channels: tuple[int, ...] = (3,),
) -> None:
    """Validate colors array for writing.

    Colors are RGB ``(n, 3)`` — or RGBA ``(n, 4)`` for geometry types that
    have opted into per-element alpha (``channels=(3, 4)``; gsplats today,
    points/lines in their volumetric phases). The alpha column is a
    per-element opacity in [0, 1], not an HDR emission channel.

    Args:
        colors: Colors array to validate
        n_points: Expected number of points
        context: Context for error messages
        channels: Accepted channel counts (broadcast rows ``(1, c)`` allowed
            for each accepted ``c``)

    Raises:
        ValidationError: If colors are invalid
    """
    if not isinstance(colors, np.ndarray):
        raise ValidationError(
            f"{context}: Expected numpy array, got {type(colors).__name__}",
            "Convert colors to numpy array: np.array(colors)",
        )

    # Allow broadcast shape (1, c) or full shape (n_points, c) per accepted c
    expected_shape = (n_points, channels[0])
    accepted = tuple((n, c) for c in channels for n in (n_points, 1))

    if colors.shape not in accepted:
        if colors.ndim == 1 and len(colors) in channels:
            raise ValidationError(
                f"{context}: Got single color {colors.shape}. "
                f"Expected shape {expected_shape} for {n_points} points or (1, {colors.shape[0]}) for broadcasting.",
                f"For a single color, use shape (1, {colors.shape[0]}) for broadcasting",
            )
        elif colors.ndim == 2 and colors.shape[1] not in channels:
            channel_desc = " or ".join(
                {3: "3 (RGB)", 4: "4 (RGBA)"}.get(c, str(c)) for c in channels
            )
            raise ValidationError(
                f"{context}: Colors must have {channel_desc} channels, got {colors.shape[1]} channels",
                f"Ensure colors have shape (n_points, c) or (1, c) with c in {channels}",
            )
        elif colors.shape[0] != n_points and colors.shape[0] != 1:
            raise ValidationError(
                f"{context}: Number of colors ({colors.shape[0]}) doesn't match "
                f"number of points ({n_points}) and is not 1 (broadcast)",
                f"Provide exactly {n_points} colors or a (1, c) broadcast row",
            )
        else:
            raise ValidationError(
                f"{context}: Expected shape {expected_shape} or (1, {channels[0]}), got {colors.shape}"
            )

    _validate_numeric_finite_values(colors, context)

    if colors.size == 0:
        if n_points == 0 and colors.ndim == 2 and colors.shape[1] in channels:
            return
        raise ValidationError(
            f"{context}: Empty colors array is only valid when n_points=0 "
            f"and shape is {expected_shape}. Got shape {colors.shape}.",
            f"Provide one broadcast color with shape (1, c), c in {channels}, "
            "or a full colors array",
        )

    # Check for invalid values
    if np.any(colors < 0):
        min_val: float = float(np.min(colors))
        raise ValidationError(
            f"{context}: Colors cannot be negative. Found minimum value: {min_val:.3f}",
            "Ensure all color values are >= 0. Use np.clip(colors, 0, None) to fix",
        )

    if colors.ndim == 2 and colors.shape[1] == 4:
        # Alpha is opacity, not emission: bounded to [0, 1] regardless of the
        # RGB columns' HDR range (finite-ness already checked above). Integer
        # storage is SDR in its native range, so the bound applies to floats.
        alpha = colors[:, 3]
        if np.issubdtype(colors.dtype, np.floating) and np.any(alpha > 1.0):
            raise ValidationError(
                f"{context}: Color alpha channel must be within [0, 1]. "
                f"Found maximum value: {float(np.max(alpha)):.3f}",
                "Alpha is per-element opacity; clip with np.clip(colors[:, 3], 0, 1)",
            )

    # Warn about extreme HDR values for floating-point HDR/SDR colors.
    # Integer color arrays are SDR storage in their native integer range.
    max_val: float = float(np.max(colors))
    if np.issubdtype(colors.dtype, np.floating) and max_val > 10.0:
        import warnings

        warnings.warn(
            f"{context}: HDR colors with maximum value {max_val:.1f} detected. "
            "Values > 10.0 may cause display issues.",
            UserWarning,
        )


def validate_radii_for_writing(
    radii: Union[NDArray[Any], float, int], n_points: int, context: str = "radii"
) -> None:
    """Validate radii (array or broadcast scalar) for writing.

    Scalars are validated against the same finite/positive rules as arrays
    (mirroring :func:`validate_widths_for_writing`); a scalar ``radii=-1.0``
    or ``radii=float('nan')`` used to slip through to the encoder after the
    positions were already written.

    Args:
        radii: Radii array or a scalar broadcast to all points
        n_points: Expected number of points
        context: Context for error messages

    Raises:
        ValidationError: If radii are invalid
    """
    if isinstance(radii, (int, float)):
        value = float(radii)
        if not np.isfinite(value):
            raise ValidationError(
                f"{context}: Radius must be finite. Got {value}",
                "Provide a finite positive radius value",
            )
        # NOTE deliberate scalar/array asymmetry: a broadcast scalar radius of
        # exactly 0.0 is an accepted degenerate-points contract (see
        # test_all_zero_radius_falls_through_to_flat_points), so only NEGATIVE
        # scalars are rejected here while the array path enforces > 0.
        if value < 0:
            raise ValidationError(
                f"{context}: Radius must not be negative. Got {value}",
                "Provide a non-negative radius value",
            )
        return

    if not isinstance(radii, np.ndarray):
        raise ValidationError(
            f"{context}: Expected numpy array, got {type(radii).__name__}",
            "Convert to numpy array: np.array(radii)",
        )

    if radii.ndim != 1:
        if radii.ndim == 2 and radii.shape[1] == 1:
            raise ValidationError(
                f"{context}: Got 2D array with shape {radii.shape}. Radii must be 1D.",
                "Flatten with: radii.ravel() or radii[:, 0]",
            )
        else:
            raise ValidationError(
                f"{context}: Expected 1D array, got {radii.ndim}D array with shape {radii.shape}",
                "Radii must be a 1D array with one value per point",
            )

    # Allow broadcast shape (1,) or full shape (n_points,)
    if len(radii) != n_points and len(radii) != 1:
        raise ValidationError(
            f"{context}: Number of radii ({len(radii)}) doesn't match "
            f"number of points ({n_points}) and is not 1 (broadcast)",
            f"Provide exactly {n_points} radii values or use shape (1,) for broadcasting",
        )

    _validate_numeric_finite_values(radii, context)

    # Check for invalid values
    if np.any(radii <= 0):
        min_val: float = float(np.min(radii))
        raise ValidationError(
            f"{context}: Radii must be positive (> 0). "
            f"Found zero or negative values (minimum: {min_val:.3f}).",
            "Use np.clip(radii, 0.01, None) to ensure positive values",
        )


def validate_widths_for_writing(
    widths: Any, n_vertices: int, context: str = "widths"
) -> None:
    """Validate line widths for writing.

    Accepts a per-vertex 1D array of shape ``(n_vertices,)``, a broadcast
    array of shape ``(1,)``, or a scalar width. The Lines sibling of
    :func:`validate_radii_for_writing` (Points), per the three-geometry
    symmetry rule.

    Args:
        widths: Per-vertex widths array, broadcast ``(1,)`` array, or scalar
        n_vertices: Expected number of vertices
        context: Context for error messages

    Raises:
        ValidationError: If widths are invalid
    """
    if isinstance(widths, np.ndarray):
        if widths.ndim != 1:
            raise ValidationError(
                f"{context}: Expected 1D array, got {widths.ndim}D array "
                f"with shape {widths.shape}",
                "Widths must be a 1D array with one value per vertex",
            )

        # Allow broadcast shape (1,) or full shape (n_vertices,) — mirrors radii
        if widths.shape[0] != n_vertices and widths.shape[0] != 1:
            raise ValidationError(
                f"{context}: Widths shape {widths.shape} doesn't match "
                f"n_vertices {n_vertices} and is not 1 (broadcast)",
                f"Provide exactly {n_vertices} width values or use shape (1,) "
                f"for broadcasting",
            )

        _validate_numeric_finite_values(widths, context)

        if np.any(widths <= 0):
            min_val: float = float(np.min(widths))
            raise ValidationError(
                f"{context}: Widths must be positive (> 0). "
                f"Found minimum value: {min_val:.3f}",
                "Use np.clip(widths, 0.01, None) to ensure positive values",
            )
    elif isinstance(widths, (int, float)):
        if not np.isfinite(widths):
            raise ValidationError(
                f"{context}: Width must be finite. Got {widths}",
                "Provide a finite positive width value",
            )
        if widths <= 0:
            raise ValidationError(
                f"{context}: Width must be positive (> 0). Got {widths}",
                "Provide a positive width value",
            )
    else:
        # Anything else (str, None, ...) used to fall through silently and
        # die deep in the encoder AFTER the vertices were already written.
        raise ValidationError(
            f"{context}: Expected numpy array or scalar, got {type(widths).__name__}",
            "Provide a per-vertex widths array or a single positive width",
        )


def validate_sharpness_for_writing(
    sharpness: Union[NDArray[Any], float, int],
    n_points: int,
    context: str = "sharpness",
) -> None:
    """Validate sharpness (array or broadcast scalar) for writing.

    Scalars are validated against the same [SHARPNESS_MIN, SHARPNESS_MAX]
    bounds as arrays (mirroring :func:`validate_widths_for_writing`); a scalar
    ``sharpness=5.0`` used to be accepted while the equivalent array was
    rejected.

    Args:
        sharpness: Sharpness array or a scalar broadcast to all points
        n_points: Expected number of points
        context: Context for error messages

    Raises:
        ValidationError: If sharpness values are invalid
    """
    if isinstance(sharpness, (int, float)):
        value = float(sharpness)
        if not np.isfinite(value):
            raise ValidationError(
                f"{context}: Sharpness must be finite. Got {value}",
                f"Provide a value between {SHARPNESS_MIN} and {SHARPNESS_MAX}",
            )
        if value < SHARPNESS_MIN or value > SHARPNESS_MAX:
            raise ValidationError(
                f"{context}: Sharpness must be within "
                f"[{SHARPNESS_MIN}, {SHARPNESS_MAX}]. Got {value}",
                f"Use values between {SHARPNESS_MIN} and {SHARPNESS_MAX} "
                f"(0.5 = true Gaussian)",
            )
        return

    if not isinstance(sharpness, np.ndarray):
        raise ValidationError(
            f"{context}: Expected numpy array, got {type(sharpness).__name__}",
            "Convert to numpy array: np.array(sharpness)",
        )

    if sharpness.ndim != 1:
        if sharpness.ndim == 2 and sharpness.shape[1] == 1:
            raise ValidationError(
                f"{context}: Got 2D array with shape {sharpness.shape}. Sharpness must be 1D.",
                "Flatten with: sharpness.ravel() or sharpness[:, 0]",
            )
        else:
            raise ValidationError(
                f"{context}: Expected 1D array, got {sharpness.ndim}D array with shape {sharpness.shape}",
                "Sharpness must be a 1D array with one value per point",
            )

    # Allow broadcast shape (1,) or full shape (n_points,)
    if len(sharpness) != n_points and len(sharpness) != 1:
        raise ValidationError(
            f"{context}: Number of sharpness values ({len(sharpness)}) doesn't match "
            f"number of points ({n_points}) and is not 1 (broadcast)",
            f"Provide exactly {n_points} sharpness values or use shape (1,) for broadcasting",
        )

    _validate_numeric_finite_values(sharpness, context)

    # Sharpness is a normalised [0, 1] knob mapped in the viewer to the
    # super-Gaussian falloff exponent beta = 2^(6s - 2): s=0.5 -> beta=2 (a
    # true Gaussian), higher s -> harder edge, lower s -> peakier cusp.
    if np.any(sharpness < SHARPNESS_MIN):
        min_val: float = float(np.min(sharpness))
        raise ValidationError(
            f"{context}: Sharpness must be >= {SHARPNESS_MIN}. Found minimum value: {min_val:.3f}",
            f"Use values between {SHARPNESS_MIN} and {SHARPNESS_MAX} for valid range",
        )

    if np.any(sharpness > SHARPNESS_MAX):
        max_val: float = float(np.max(sharpness))
        raise ValidationError(
            f"{context}: Sharpness values exceed maximum allowed value ({SHARPNESS_MAX}). "
            f"Found maximum: {max_val:.3f}",
            f"Clip values to valid range: np.clip(sharpness, {SHARPNESS_MIN}, {SHARPNESS_MAX})",
        )


def validate_zarr_attributes(attrs: dict, is_root: bool = False) -> None:
    """Validate that all required Zarr attributes are present.

    Ensures that zarr groups have the required metadata attributes according
    to the Luxar Zarr format specification.

    Args:
        attrs: Dictionary of zarr attributes
        is_root: Whether this is the root scene group

    Raises:
        ValidationError: If required attributes are missing or invalid
    """
    if is_root:
        # Root scene requires additional attributes
        required = {"type", "luxar_version"}
        recommended = {"scene_dimensions"}
    else:
        # Child nodes only require type
        required = {"type"}
        recommended = set()

    # Check for missing required attributes
    missing_required = required - set(attrs.keys())
    if missing_required:
        raise ValidationError(
            f"Missing required zarr attributes: {missing_required}",
            f"Add these attributes: {', '.join(missing_required)}",
        )

    # Check for missing recommended attributes
    missing_recommended = recommended - set(attrs.keys())
    if missing_recommended:
        import warnings

        warnings.warn(
            f"Missing recommended zarr attributes: {missing_recommended}. "
            f"Consider adding these for better compatibility.",
            UserWarning,
        )

    # Validate type attribute
    if "type" in attrs:
        valid_types = {"scene", "group", "points", "lines", "gsplats"}
        if attrs["type"] not in valid_types:
            raise ValidationError(
                f"Invalid node type: '{attrs['type']}'",
                f"Use one of: {', '.join(valid_types)}",
            )

    # Validate version if present. Keep this import local: typing_utils.config
    # itself imports io.reader.DEFAULT_COMP via a lazy/inline path, and io.reader
    # transitively imports core.dimensions which imports validation.category_validation.
    # validation.base is imported early enough that an unconditional top-level
    # import here would risk re-entering this module via that chain.
    if "luxar_version" in attrs:
        from ..typing_utils.config import SUPPORTED_VERSIONS

        if attrs["luxar_version"] not in SUPPORTED_VERSIONS:
            raise ValidationError(
                f"Unsupported Luxar version: '{attrs['luxar_version']}'",
                f"Supported versions: {', '.join(SUPPORTED_VERSIONS)}",
            )
