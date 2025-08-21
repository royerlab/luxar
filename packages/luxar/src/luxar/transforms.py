"""luxar.transforms – Transform utilities for creating and manipulating 4x4 transformation matrices.

This module provides convenient functions for creating common transformations
used in 3D graphics, including translation, rotation, and scaling matrices.
All matrices are 4x4 homogeneous transformation matrices suitable for use
with the Luxar scene graph.
"""

from __future__ import annotations

from typing import Any, Literal, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .types import TransformMatrix, validate_transform


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
    axis = np.array(axis, dtype=np.float32)
    if axis.shape != (3,):
        raise ValueError(f"Axis must be a 3D vector, got shape {axis.shape}")

    # Normalize axis
    axis = axis / np.linalg.norm(axis)

    radians = np.radians(degrees)
    cos_a = np.cos(radians)
    sin_a = np.sin(radians)
    one_minus_cos = 1 - cos_a

    x, y, z = axis

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

    Args:
        *transforms: Variable number of 4x4 transformation matrices

    Returns:
        4x4 composed transformation matrix as float32 array

    Example:
        >>> t1 = translate(5, 0, 0)
        >>> t2 = rotate_z(45)
        >>> t3 = scale(2, 2, 2)
        >>> combined = compose(t1, t2, t3)  # Translate, then rotate, then scale
    """
    if not transforms:
        return identity()

    # To apply transforms in order t1, t2, t3, we need t3 @ t2 @ t1
    # So we accumulate from right to left
    result = identity()
    for transform in reversed(transforms):
        result = transform @ result

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

    # Calculate basis vectors
    forward = target_arr - eye_arr
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up_arr)
    right = right / np.linalg.norm(right)

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
