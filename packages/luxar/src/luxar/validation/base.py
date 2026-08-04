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
    have opted into per-element alpha (``channels=(3, 4)``; all three
    geometry types since their volumetric phases: gsplats, points, lines).
    The alpha column is a per-element opacity in [0, 1], not an HDR
    emission channel.

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
            f"{context}: Expected a 1D numpy array or a Python float, got {type(radii).__name__}",
            "Pass a Python float — e.g. float(radii) — for a single broadcast "
            "value, or a 1D array with one value per point",
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
            f"{context}: Expected a 1D numpy array or a Python float, got {type(widths).__name__}",
            "Pass a Python float — e.g. float(widths) — for a single broadcast "
            "value, or a 1D array with one value per vertex",
        )


def validate_cholesky_for_writing(
    cholesky_factors: NDArray[Any],
    n_dims: int,
    context: str = "cholesky_factors",
) -> None:
    """Validate packed lower-triangular Cholesky factors for writing.

    The GSplats sibling of :func:`validate_radii_for_writing` (Points) and
    :func:`validate_widths_for_writing` (Lines), per the three-geometry
    symmetry rule. The gsplat assembly gate normalizes SHAPE before calling
    (a 1D uniform ``(k,)`` input is reshaped to ``(1, k)``), so this validator
    accepts a 2D array of shape ``(n_splats, k)`` OR the uniform/broadcast
    ``(1, k)`` form, with ``k = n_dims * (n_dims + 1) // 2``.

    Two checks, mirroring the amplitudes/radii/widths validators:

    - Finiteness across the WHOLE packed array (a single NaN/Inf used to pass
      the shape-only gate and die deep in the encoder AFTER centers were
      already on disk, leaving a half-written node).
    - Strictly positive diagonal. The packed diagonal slots are
      ``np.cumsum(np.arange(1, n_dims + 1)) - 1`` (e.g. ``[0, 2, 5]`` for
      ``n_dims=3``); the format invariant is diag > 0 (positive, scale-like).
      A zero/negative diagonal is a degenerate/singular covariance that the
      uint8 log encoder silently clamps to 0 → a sliver splat with no
      diagnostic.

    Args:
        cholesky_factors: Packed lower-triangular factors, shape
            ``(n_splats, k)`` or ``(1, k)``.
        n_dims: Number of spatial dimensions (used to locate the diagonal
            slots; the caller already has it).
        context: Context for error messages.

    Raises:
        ValidationError: If the factors contain NaN/Inf or any non-positive
            diagonal value.
    """
    _validate_numeric_finite_values(cholesky_factors, context)

    diag_indices = np.cumsum(np.arange(1, n_dims + 1)) - 1
    diagonal = cholesky_factors[:, diag_indices]
    if np.any(diagonal <= 0):
        min_val: float = float(np.min(diagonal))
        raise ValidationError(
            f"{context}: Cholesky diagonal must be positive (> 0). "
            f"Found zero or negative values (minimum: {min_val:.3f}).",
            "The diagonal is scale-like; ensure a positive-definite "
            "Cholesky factor (positive diagonal entries)",
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
            f"{context}: Expected a 1D numpy array or a Python float, got {type(sharpness).__name__}",
            "Pass a Python float — e.g. float(sharpness) — for a single broadcast "
            "value, or a 1D array with one value per point",
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


def validate_vertices_for_writing(
    vertices: NDArray[Any], context: str = "vertices"
) -> None:
    """Enforce the mesh vertex-count ceiling before any zarr write.

    Shape and finiteness are NOT re-checked here — the shared coordinate path
    (:func:`validate_positions_for_writing`) owns those for every geometry type.
    This validator's sole job is the ``n_vertices <= MAX_MESH_VERTICES`` cap,
    which is specific to mesh because a mesh's pick ``elementId`` is the raw
    ``gl_VertexID`` rather than an element-texture index (see
    :data:`~luxar.typing_utils.constants.MAX_MESH_VERTICES`).

    The viewer's loader rejects an over-cap store too. Mirroring it at write time
    is the point: without this, the public ``add_mesh`` path could emit a store
    that Luxar's own loader then refuses — a bound that exists only on the read
    side is not a bound, and the failure would surface far from its cause.

    Args:
        vertices: The mesh ``(V, D)`` vertex array.
        context: Context for error messages.

    Raises:
        ValidationError: If the vertex count exceeds the cap.
    """
    from ..typing_utils.constants import MAX_MESH_VERTICES

    n_vertices = int(np.asarray(vertices).shape[0])
    if n_vertices > MAX_MESH_VERTICES:
        raise ValidationError(
            f"{context}: Mesh has {n_vertices:,} vertices, which exceeds the "
            f"maximum of {MAX_MESH_VERTICES:,} (2^27). Above this the viewer's "
            f"per-node pick vote key aliases across nodes, so picking would "
            f"silently resolve to the wrong node.",
            "Split the surface into several mesh nodes, or decimate it below "
            f"{MAX_MESH_VERTICES:,} vertices",
        )


def validate_faces_for_writing(
    faces: Any, n_vertices: int, context: str = "faces"
) -> None:
    """Validate mesh triangle indices for writing.

    Accepts the two documented layouts — an ``(F, 3)`` triangle array or a flat
    ``(3F,)`` array. The closest precedent is the ``line_type='indexed'`` index
    gate in the lines writer, which already encodes each of these traps; this is
    the shared-validator form of the same rules.

    Every check corresponds to a way bad indices fail *silently* rather than
    loudly, since the writer casts with ``.astype(np.uint32)`` and the viewer
    hands the result straight to kernels that index without bounds-checking:

    - **integer dtype** — a float array truncates on cast (``1.7`` → ``1``),
      producing triangles the author never wound.
    - **min >= 0** — a negative index wraps to ~4 billion on the unsigned cast.
    - **max < n_vertices** — an out-of-range index reads past the vertex buffer;
      in the Rust kernel (``panic = "abort"``) that takes down the whole WASM
      module rather than one node.
    - **F >= 1** — a mesh with no faces draws nothing; an indexed draw never
      references a vertex no face names.

    Args:
        faces: Triangle indices, ``(F, 3)`` or flat ``(3F,)``.
        n_vertices: Vertex count the indices must address.
        context: Context for error messages.

    Raises:
        ValidationError: If the faces array is malformed or out of range.
    """
    faces_array = np.asarray(faces)

    if faces_array.ndim > 2 or (faces_array.ndim == 2 and faces_array.shape[1] != 3):
        raise ValidationError(
            f"{context}: Faces must be an (F, 3) triangle array or a flat (3F,) "
            f"array, got shape {faces_array.shape}",
            "Reshape to (F, 3) — one row per triangle, three vertex indices each",
        )

    if faces_array.size == 0:
        raise ValidationError(
            f"{context}: Mesh has no faces (empty array). A mesh with no faces "
            "renders nothing — an indexed draw never references a vertex that no "
            "face names.",
            "Provide at least one triangle, e.g. faces=[[0, 1, 2]]",
        )

    if faces_array.size % 3 != 0:
        raise ValidationError(
            f"{context}: Faces must have an element count divisible by 3 "
            f"(triangles), got {faces_array.size}",
            "Provide three vertex indices per triangle",
        )

    if not np.issubdtype(faces_array.dtype, np.integer):
        raise ValidationError(
            f"{context}: Faces must be an integer array, got dtype "
            f"{faces_array.dtype}. A float index truncates on the uint32 cast "
            f"(1.7 -> 1), which would silently rewrite the topology.",
            "Convert with faces.astype(np.uint32)",
        )

    face_min = int(faces_array.min())
    if face_min < 0:
        raise ValidationError(
            f"{context}: Face index {face_min} < 0. Negative indices wrap to ~4 "
            f"billion on the uint32 cast instead of failing.",
            "Use non-negative vertex indices",
        )

    face_max = int(faces_array.max())
    if face_max >= n_vertices:
        raise ValidationError(
            f"{context}: Face index {face_max} is out of range for {n_vertices} "
            f"vertices (valid indices are 0..{n_vertices - 1}). An out-of-range "
            f"index reads past the vertex buffer at render time.",
            "Check the indices are 0-based and address this mesh's own vertices",
        )


def validate_normals_for_writing(
    normals: NDArray[Any], n_vertices: int, context: str = "normals"
) -> None:
    """Validate per-vertex mesh normals for writing.

    Shape ``(V, 3)`` and finiteness are required. Normals are always 3-component
    even for an nD mesh: they are a display-space quantity, and which three
    dimensions they belong to is recorded separately by ``normal_dims`` (see
    :func:`validate_normal_dims_for_writing`).

    Zero-length normals are **warned about, not rejected**. Degenerate triangles
    legitimately produce them, and the renderer already handles the case: the
    stored-normal fragment path epsilon-guards its ``normalize`` and falls back to
    a screen-space-derivative flat normal. The warning exists so authors fix the
    source rather than lean on that fallback, because the fallback is pointwise —
    on a shared-vertex mesh the interpolated normal near a degenerate vertex
    blends toward its neighbours, so shading there is locally distorted rather
    than cleanly flat.

    Args:
        normals: Per-vertex normals, shape ``(V, 3)``.
        n_vertices: Expected vertex count.
        context: Context for error messages.

    Raises:
        ValidationError: If normals are the wrong shape or non-finite.
    """
    if not isinstance(normals, np.ndarray):
        raise ValidationError(
            f"{context}: Expected numpy array, got {type(normals).__name__}",
            "Convert your data to a numpy array using np.array(normals)",
        )

    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValidationError(
            f"{context}: Expected shape (n_vertices, 3), got {normals.shape}. "
            "Normals are always 3-component — they describe the displayed "
            "dimensions, named by the normal_dims attr.",
            "Provide one 3-component normal per vertex",
        )

    if normals.shape[0] != n_vertices:
        raise ValidationError(
            f"{context}: Normals count ({normals.shape[0]}) doesn't match vertex "
            f"count ({n_vertices})",
            f"Provide exactly {n_vertices} normals (one per vertex)",
        )

    _validate_numeric_finite_values(normals, context)

    zero_count = int(np.count_nonzero(np.all(normals == 0.0, axis=1)))
    if zero_count:
        import warnings

        warnings.warn(
            f"{context}: {zero_count} of {n_vertices} normals are zero-length. "
            "These are usually produced by degenerate (zero-area) triangles. The "
            "renderer falls back to a derived flat normal per fragment, but "
            "shading near a zero normal on a shared-vertex mesh is locally "
            "distorted rather than cleanly flat — recompute the normals to fix "
            "it properly.",
            UserWarning,
            stacklevel=2,
        )


def validate_normal_dims_for_writing(
    normal_dims: Any, ndim: int, context: str = "normal_dims"
) -> None:
    """Validate the companion attr naming which dimensions ``normals`` describe.

    Required whenever ``normals`` is supplied and rejected when it is not — that
    pairing is enforced by the caller, which knows both. This validator checks
    the value itself: exactly 3 entries, integral, distinct, each a valid
    dimension index.

    Normals are stored ``(V, 3)`` because they are only meaningful for the three
    displayed dimensions, so the store must say *which* three. Storing them
    against an implicit "first three dimensions" is the bug this attr exists to
    prevent: for a ``(t, x, y, z)`` mesh the first three dims are ``(t, x, y)``
    and a normal against them is meaningless. The viewer compares this list to
    the active ``displayDims`` and falls back to flat normals when they differ,
    so a wrong-but-well-formed list degrades shading rather than corrupting it —
    but an *ill-formed* one would index out of bounds.

    Args:
        normal_dims: Candidate dimension-index triple.
        ndim: The mesh's dimensionality (indices must be < this).
        context: Context for error messages.

    Raises:
        ValidationError: If the triple is malformed or out of range.
    """
    if isinstance(normal_dims, (str, bytes)) or not isinstance(
        normal_dims, (Sequence, np.ndarray)
    ):
        raise ValidationError(
            f"{context}: Expected a sequence of 3 dimension indices, got "
            f"{type(normal_dims).__name__}",
            "Pass e.g. normal_dims=(0, 1, 2) naming the dimensions the normals "
            "describe",
        )

    # Iterate the entries as GIVEN, never through ``np.asarray``. Coercing first
    # destroys the evidence: ``np.asarray((True, 0, 2))`` is an int64 array, so a
    # bool entry arrives here already indistinguishable from dimension 1 and the
    # per-entry type check below can never see it. (A bool is an int subclass, so
    # only an explicit rejection catches it.) A 1-element ndarray is unwrapped by
    # ``tolist`` for the same reason — its scalars are numpy types, not Python
    # ones, but ``tolist`` yields Python ints and preserves bool-ness.
    raw_dims = (
        normal_dims.tolist()
        if isinstance(normal_dims, np.ndarray)
        else list(normal_dims)
    )
    if len(raw_dims) != 3:
        raise ValidationError(
            f"{context}: Expected exactly 3 dimension indices, got {len(raw_dims)}",
            "Normals are 3-component, so exactly three dimensions name them",
        )

    for i, dim in enumerate(raw_dims):
        if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)):
            raise ValidationError(
                f"{context}: Entry {i} must be an integer dimension index, got "
                f"{dim!r} ({type(dim).__name__})",
                "Use integer dimension indices, e.g. normal_dims=(0, 1, 2)",
            )
        if not 0 <= int(dim) < ndim:
            raise ValidationError(
                f"{context}: Entry {i} is dimension {int(dim)}, out of range for "
                f"a {ndim}D mesh (valid indices are 0..{ndim - 1})",
                f"Name three of this mesh's own {ndim} dimensions",
            )

    int_dims = [int(d) for d in raw_dims]
    if len(set(int_dims)) != 3:
        raise ValidationError(
            f"{context}: Dimension indices must be distinct, got {int_dims}",
            "Name three different dimensions",
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

    # Validate type attribute against the contract vocabulary, not a literal
    # set: a node type added to ``node_types`` but missed here would make every
    # store containing one fail this validator as "invalid", which is the
    # opposite failure from the geometry-leaf checks (#1203) but the same root
    # cause — a hand-copied vocabulary. Iteration order is the contract's, so
    # the hint is deterministic (a set's ``join`` was not).
    if "type" in attrs:
        from ..typing_utils._format_contract import NODE_TYPES

        if attrs["type"] not in NODE_TYPES:
            raise ValidationError(
                f"Invalid node type: '{attrs['type']}'",
                f"Use one of: {', '.join(NODE_TYPES)}",
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
