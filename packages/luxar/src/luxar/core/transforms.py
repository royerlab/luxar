"""luxar.core.transforms – Transform utilities for creating and manipulating 4x4 transformation matrices.

This module provides convenient functions for creating common transformations
used in 3D graphics, including translation, rotation, and scaling matrices.
All matrices are 4x4 homogeneous transformation matrices suitable for use
with the Luxar scene graph.
"""

from __future__ import annotations

from typing import Any, Literal, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from ..typing_utils.aliases import TransformMatrix
from ..validation.types import validate_transform


def identity() -> TransformMatrix:
    """Create a 4x4 identity transformation matrix.

    Returns:
        4x4 identity matrix as float32 array

    Example:
        >>> t = identity()
        >>> print(t)
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]]
    """
    return np.eye(4, dtype=np.float32)


def translate(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> TransformMatrix:
    """Create a translation transformation matrix.

    Args:
        x: Translation along X axis
        y: Translation along Y axis
        z: Translation along Z axis

    Returns:
        4x4 translation matrix as float32 array

    Example:
        >>> t = translate(5, 0, 0)  # Move 5 units along X
        >>> print(t[:, 3])  # Translation column
        [5. 0. 0. 1.]
    """
    matrix = identity()
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix


def scale(
    x: float = 1.0, y: float = 1.0, z: float = 1.0, uniform: Union[float, None] = None
) -> TransformMatrix:
    """Create a scaling transformation matrix.

    Args:
        x: Scale factor for X axis (ignored if uniform is set)
        y: Scale factor for Y axis (ignored if uniform is set)
        z: Scale factor for Z axis (ignored if uniform is set)
        uniform: If provided, scales all axes equally

    Returns:
        4x4 scaling matrix as float32 array

    Example:
        >>> t1 = scale(2, 2, 2)      # Scale 2x on all axes
        >>> t2 = scale(uniform=2)     # Same as above
        >>> t3 = scale(x=2, y=1, z=0.5)  # Non-uniform scaling
    """
    matrix = identity()
    if uniform is not None:
        matrix[0, 0] = uniform
        matrix[1, 1] = uniform
        matrix[2, 2] = uniform
    else:
        matrix[0, 0] = x
        matrix[1, 1] = y
        matrix[2, 2] = z
    return matrix


def rotate_x(degrees: float) -> TransformMatrix:
    """Create a rotation matrix around the X axis.

    Args:
        degrees: Rotation angle in degrees

    Returns:
        4x4 rotation matrix as float32 array

    Example:
        >>> t = rotate_x(90)  # Rotate 90 degrees around X
    """
    radians = np.radians(degrees)
    cos_a = np.cos(radians)
    sin_a = np.sin(radians)

    matrix = identity()
    matrix[1, 1] = cos_a
    matrix[1, 2] = -sin_a
    matrix[2, 1] = sin_a
    matrix[2, 2] = cos_a
    return matrix


def rotate_y(degrees: float) -> TransformMatrix:
    """Create a rotation matrix around the Y axis.

    Args:
        degrees: Rotation angle in degrees

    Returns:
        4x4 rotation matrix as float32 array

    Example:
        >>> t = rotate_y(45)  # Rotate 45 degrees around Y
    """
    radians = np.radians(degrees)
    cos_a = np.cos(radians)
    sin_a = np.sin(radians)

    matrix = identity()
    matrix[0, 0] = cos_a
    matrix[0, 2] = sin_a
    matrix[2, 0] = -sin_a
    matrix[2, 2] = cos_a
    return matrix


def rotate_z(degrees: float) -> TransformMatrix:
    """Create a rotation matrix around the Z axis.

    Args:
        degrees: Rotation angle in degrees

    Returns:
        4x4 rotation matrix as float32 array

    Example:
        >>> t = rotate_z(180)  # Rotate 180 degrees around Z
    """
    radians = np.radians(degrees)
    cos_a = np.cos(radians)
    sin_a = np.sin(radians)

    matrix = identity()
    matrix[0, 0] = cos_a
    matrix[0, 1] = -sin_a
    matrix[1, 0] = sin_a
    matrix[1, 1] = cos_a
    return matrix


def rotate(
    degrees: float,
    axis: Union[
        Literal["x", "y", "z"], Tuple[float, float, float], NDArray[np.float32]
    ],
) -> TransformMatrix:
    """Create a rotation matrix around a specified axis.

    Args:
        degrees: Rotation angle in degrees
        axis: Either 'x', 'y', 'z' for principal axes, or a 3D vector for arbitrary axis

    Returns:
        4x4 rotation matrix as float32 array

    Example:
        >>> t1 = rotate(45, 'z')              # Rotate around Z axis
        >>> t2 = rotate(30, (1, 1, 0))        # Rotate around diagonal axis
        >>> t3 = rotate(90, np.array([0, 0, 1]))  # Rotate around Z using vector
    """
    if isinstance(axis, str):
        if axis.lower() == "x":
            return rotate_x(degrees)
        elif axis.lower() == "y":
            return rotate_y(degrees)
        elif axis.lower() == "z":
            return rotate_z(degrees)
        else:
            raise ValueError(f"Invalid axis string: {axis}. Use 'x', 'y', or 'z'")

    # Arbitrary axis rotation using Rodrigues' formula
    axis_vector = cast(NDArray[np.float32], np.asarray(axis, dtype=np.float32))
    if axis_vector.shape != (3,):
        raise ValueError(f"Axis must be a 3D vector, got shape {axis_vector.shape}")

    # Normalize axis. A zero-length (or numerically near-zero) axis
    # silently produced a NaN-filled rotation matrix before this guard —
    # downstream scenes then rendered as black void. Reject upfront with
    # a clear message naming the offending input.
    axis_norm = float(np.linalg.norm(axis_vector))
    if axis_norm == 0.0:
        raise ValueError(
            f"rotate: axis vector {tuple(axis_vector.tolist())!r} has zero length; "
            f"a rotation axis must be non-degenerate"
        )
    axis_vector = axis_vector / axis_norm

    radians = np.radians(degrees)
    cos_a = np.cos(radians)
    sin_a = np.sin(radians)
    one_minus_cos = 1 - cos_a

    x, y, z = axis_vector

    matrix = np.array(
        [
            [
                cos_a + x * x * one_minus_cos,
                x * y * one_minus_cos - z * sin_a,
                x * z * one_minus_cos + y * sin_a,
                0,
            ],
            [
                y * x * one_minus_cos + z * sin_a,
                cos_a + y * y * one_minus_cos,
                y * z * one_minus_cos - x * sin_a,
                0,
            ],
            [
                z * x * one_minus_cos - y * sin_a,
                z * y * one_minus_cos + x * sin_a,
                cos_a + z * z * one_minus_cos,
                0,
            ],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    return matrix


def compose(*transforms: TransformMatrix) -> TransformMatrix:
    """Compose multiple transformation matrices into a single matrix.

    Matrices are applied in order from left to right. For example,
    compose(T1, T2, T3) creates a matrix that first applies T1,
    then T2, then T3.

    Mathematical Note:
        To apply transforms in user order (T1, then T2, then T3), we compute:
        result = T3 @ T2 @ T1

        This is because matrix multiplication is right-associative when applied
        to vectors: (T3 @ T2 @ T1) @ v = T3 @ (T2 @ (T1 @ v))

        So the rightmost matrix (T1) is applied first to the vector.

    Args:
        *transforms: Variable number of 4x4 transformation matrices

    Returns:
        4x4 composed transformation matrix as float32 array

    Example:
        >>> t1 = translate(5, 0, 0)   # Move right 5 units
        >>> t2 = rotate_z(45)          # Rotate 45 degrees
        >>> t3 = scale(2, 2, 2)        # Scale by 2x
        >>> combined = compose(t1, t2, t3)  # Translate, then rotate, then scale
    """
    if not transforms:
        return identity()

    # To apply transforms in order t1, t2, t3, we compute: t3 @ t2 @ t1
    # Accumulate from right to left by iterating in reverse and right-multiplying
    result = identity()
    for transform in reversed(transforms):
        result = result @ transform  # Right-multiply: result = result @ next_transform

    return result.astype(np.float32)


def inverse(transform: TransformMatrix) -> TransformMatrix:
    """Compute the inverse of a transformation matrix.

    Args:
        transform: 4x4 transformation matrix

    Returns:
        4x4 inverse transformation matrix as float32 array

    Raises:
        ValueError: If matrix is not invertible

    Example:
        >>> t = translate(5, 0, 0)
        >>> t_inv = inverse(t)  # Translates -5, 0, 0
    """
    try:
        return np.linalg.inv(transform).astype(np.float32)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Transform matrix is not invertible: {e}") from e


def look_at(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0, 1, 0),
) -> TransformMatrix:
    """Create a "look at" transformation matrix.

    This creates a transformation that positions an object at 'eye'
    and orients it to look at 'target' with the given 'up' vector.

    Args:
        eye: Position of the viewer
        target: Position to look at
        up: Up direction vector (default: Y-up)

    Returns:
        4x4 transformation matrix as float32 array

    Example:
        >>> t = look_at((10, 5, 10), (0, 0, 0))  # Look at origin from position (10, 5, 10)
    """
    eye_arr = np.array(eye, dtype=np.float32)
    target_arr = np.array(target, dtype=np.float32)
    up_arr = np.array(up, dtype=np.float32)

    # Calculate basis vectors. Both norms below CAN be zero: eye == target
    # collapses `forward` to the zero vector (caller's mistake), and `up`
    # parallel to `forward` makes the cross product vanish. Either case
    # silently produced NaN-filled matrices that propagated downstream
    # (the camera renders nothing, often without any visible error).
    # Raise upfront with the actionable cause.
    forward = target_arr - eye_arr
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm == 0.0:
        raise ValueError(
            f"look_at: eye and target are coincident ({tuple(eye)!r} == "
            f"{tuple(target)!r}); cannot derive a forward direction"
        )
    forward = forward / forward_norm

    right = np.cross(forward, up_arr)
    right_norm = float(np.linalg.norm(right))
    if right_norm == 0.0:
        raise ValueError(
            f"look_at: up vector {tuple(up)!r} is parallel (or antiparallel) "
            f"to the eye→target direction; cannot derive an orthonormal basis"
        )
    right = right / right_norm

    up_final = np.cross(right, forward)

    # Build matrix
    matrix = identity()
    matrix[0, :3] = right
    matrix[1, :3] = up_final
    matrix[2, :3] = -forward  # Negative because we look down -Z in standard OpenGL
    matrix[:3, 3] = eye_arr

    return matrix


def to_list(transform: TransformMatrix) -> list[float]:
    """Convert a 4x4 transformation matrix to a flat list for storage.

    Note: The matrix is transposed before flattening to match THREE.js
    column-major format requirements.

    Args:
        transform: 4x4 transformation matrix (row-major, NumPy format)

    Returns:
        List of 16 float values in column-major order (for THREE.js)

    Example:
        >>> t = translate(1, 2, 3)
        >>> values = to_list(t)  # Returns transposed, flattened matrix
    """
    # Transpose for THREE.js compatibility (row-major to column-major)
    return list(transform.T.ravel().tolist())


def from_list(values: list[float]) -> TransformMatrix:
    """Create a 4x4 transformation matrix from a flat list.

    Note: The values are assumed to be in THREE.js column-major format
    and are transposed back to NumPy row-major format.

    Args:
        values: List of 16 float values in column-major order (THREE.js format)

    Returns:
        4x4 transformation matrix in row-major order (NumPy format)

    Raises:
        ValueError: If values is not a list of 16 numbers

    Example:
        >>> values = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1]
        >>> t = from_list(values)  # Creates translation matrix (5, 0, 0)
    """
    if len(values) != 16:
        raise ValueError(f"Expected 16 values, got {len(values)}")

    # Reshape and transpose from column-major (THREE.js) to row-major (NumPy)
    matrix = np.array(values, dtype=np.float32).reshape(4, 4).T
    return validate_transform(matrix)


def prepare_transform_for_zarr(transform: Any) -> list[float]:
    """Prepare any transform format for Zarr storage (THREE.js compatible).

    This centralizes the logic for converting transforms to the format expected
    by the Zarr storage and THREE.js viewer. The transform is validated and
    converted to a 16-element list in column-major order.

    All inputs are interpreted as **row-major** (NumPy convention):
    - 4x4 numpy array: standard NumPy row-major matrix
    - 16-element list: flattened row-major (same as ``matrix.ravel().tolist()``)
    - 16-element flat numpy array: flattened row-major

    The output is always column-major (THREE.js convention) for zarr storage.

    Args:
        transform: Transform in any supported format:
            - 4x4 numpy array (row-major)
            - 16-element list (row-major, flattened)
            - 16-element numpy array (row-major, flat)

    Returns:
        16-element list in column-major order for THREE.js

    Raises:
        ValueError: If transform is invalid or wrong shape

    See Also:
        read_transform_from_zarr: Reverse operation to read from storage
    """
    # Convert to numpy array
    transform_array = np.array(transform, dtype=np.float32)

    if transform_array.size != 16:
        raise ValueError(f"Transform must have 16 elements, got {transform_array.size}")

    # Reshape to 4x4 if flat — always row-major (NumPy convention)
    # Both lists and flat numpy arrays are treated as row-major.
    # Column-major (THREE.js/zarr) interpretation is only in read_transform_from_zarr().
    if transform_array.ndim == 1:
        transform_matrix = transform_array.reshape(4, 4)
    elif transform_array.shape == (4, 4):
        transform_matrix = transform_array
    else:
        raise ValueError(f"Transform has invalid shape: {transform_array.shape}")

    # Validate the matrix
    validated = validate_transform(transform_matrix)

    # Transpose for THREE.js (column-major order) and convert to list
    zarr_result: list[float] = validated.T.ravel().tolist()
    return zarr_result


def read_transform_from_zarr(transform_list: list[float]) -> TransformMatrix:
    """Read a transform from Zarr storage and convert to NumPy format.

    This is the inverse of prepare_transform_for_zarr(). It converts a
    16-element list in THREE.js column-major format back to a NumPy
    row-major 4x4 matrix.

    Args:
        transform_list: 16-element list in column-major order (THREE.js format)

    Returns:
        4x4 transformation matrix in row-major order (NumPy format)

    Raises:
        ValueError: If transform_list is invalid

    Example:
        >>> # Read transform from zarr attributes
        >>> transform_list = node_attrs['transform']
        >>> matrix = read_transform_from_zarr(transform_list)
        >>> print(matrix.shape)  # (4, 4)

    See Also:
        prepare_transform_for_zarr: Inverse operation to write to storage
    """
    if not isinstance(transform_list, list):
        raise ValueError(f"Expected list, got {type(transform_list).__name__}")

    if len(transform_list) != 16:
        raise ValueError(f"Expected 16 elements, got {len(transform_list)}")

    # Convert from THREE.js column-major to NumPy row-major
    # 1. Reshape to 4x4 (column-major layout)
    # 2. Transpose to get row-major
    matrix = np.array(transform_list, dtype=np.float32).reshape(4, 4).T

    # Validate before returning
    return validate_transform(matrix)


def transform_bounding_box(
    matrix: Any,
    lo: Any,
    hi: Any,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Transform an axis-aligned 3D bounding box by a 4x4 matrix.

    Transforms all **8 corners** of the box and returns the smallest
    axis-aligned box that encloses the transformed corners. This is the
    mathematically correct way to transform an AABB under rotation /
    shear — transforming only the (min, max) corner pair underestimates
    the rotated extent and is the classic bug that makes peripheral
    geometry get clipped/culled.

    Mirrors ``transformBoundingBox`` in the TypeScript viewer
    (``scene/scene-manager/clipping/bounds-math.ts``).

    Args:
        matrix: 4x4 transformation matrix (row-major, NumPy convention).
            Accepts a 4x4 array or a flat 16-element row-major array.
        lo: Lower corner ``[x, y, z]`` of the box.
        hi: Upper corner ``[x, y, z]`` of the box.

    Returns:
        Tuple ``(new_lo, new_hi)`` of the enclosing box, each a length-3
        float64 array.

    Example:
        >>> m = translate(3, 0, 0)
        >>> lo, hi = transform_bounding_box(m, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        >>> lo.tolist(), hi.tolist()
        ([2.5, -0.5, -0.5], [3.5, 0.5, 0.5])
    """
    mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    lo_arr = np.asarray(lo, dtype=np.float64)
    hi_arr = np.asarray(hi, dtype=np.float64)

    # Build the 8 corners (cartesian product of {lo, hi} per axis).
    corners = np.array(
        [
            [x, y, z]
            for x in (lo_arr[0], hi_arr[0])
            for y in (lo_arr[1], hi_arr[1])
            for z in (lo_arr[2], hi_arr[2])
        ],
        dtype=np.float64,
    )

    # Homogeneous transform: (8, 4) @ (4, 4)^T -> (8, 4).
    homogeneous = np.column_stack([corners, np.ones(8)])
    transformed = homogeneous @ mat.T

    # Perspective divide (w == 1 for affine transforms). Guard against a
    # degenerate w so a pathological matrix can't produce NaN/inf bounds.
    w = transformed[:, 3]
    w = np.where(np.abs(w) < 1e-12, 1.0, w)
    points = transformed[:, :3] / w[:, None]

    return points.min(axis=0), points.max(axis=0)


# Convenience function aliases
def translation(*args: Any, **kwargs: Any) -> TransformMatrix:
    """Alias for translate()."""
    return translate(*args, **kwargs)


def scaling(*args: Any, **kwargs: Any) -> TransformMatrix:
    """Alias for scale()."""
    return scale(*args, **kwargs)


def rotation(*args: Any, **kwargs: Any) -> TransformMatrix:
    """Alias for rotate()."""
    return rotate(*args, **kwargs)
