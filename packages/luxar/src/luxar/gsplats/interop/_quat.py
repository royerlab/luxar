"""Quaternion / rotation helpers (w-first convention throughout).

Extracted from ``classical_splats.py`` in the per-concern module split;
``classical_splats`` re-exports ``quat_to_rotmat`` / ``rotmat_to_quat`` so
existing ``from ...classical_splats import`` call sites stay valid.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternions ``(N, 4)`` (w, x, y, z) to rotation matrices ``(N, 3, 3)``.

    Quaternions are re-normalized defensively; zero-norm quaternions decode to
    the identity rotation.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"q must be (N, 4); got {q.shape}")
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.divide(q, norm, out=np.zeros_like(q), where=norm > 0)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    identity = norm[:, 0] == 0
    w = np.where(identity, 1.0, w)

    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices ``(N, 3, 3)`` to unit quaternions ``(N, 4)`` (w, x, y, z).

    Uses Shepperd's method (branch on the largest diagonal combination) for
    numerical stability near 180° rotations. Inputs must be proper rotations
    (``det = +1``); the caller is responsible for reflection correction.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"R must be (N, 3, 3); got {R.shape}")
    n = R.shape[0]
    q = np.empty((n, 4), dtype=np.float64)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    # Candidate squared components (all >= 0 up to rounding); branch on the
    # largest so the divisor 4s below is always well-conditioned.
    qw2 = np.maximum(0.0, 1.0 + trace) / 4.0
    qx2 = np.maximum(0.0, 1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]) / 4.0
    qy2 = np.maximum(0.0, 1.0 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]) / 4.0
    qz2 = np.maximum(0.0, 1.0 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]) / 4.0
    branch = np.argmax(np.stack([qw2, qx2, qy2, qz2], axis=1), axis=1)

    def _fill(
        mask: np.ndarray, sq: np.ndarray, cols: list[Optional[np.ndarray]]
    ) -> None:
        if np.any(mask):
            s = np.sqrt(sq[mask])
            for target, col in enumerate(cols):
                q[mask, target] = s if col is None else col[mask] / (4 * s)

    r = R  # column shorthands (differences/sums of off-diagonal entries)
    wx = r[:, 2, 1] - r[:, 1, 2]
    wy = r[:, 0, 2] - r[:, 2, 0]
    wz = r[:, 1, 0] - r[:, 0, 1]
    xy = r[:, 0, 1] + r[:, 1, 0]
    xz = r[:, 0, 2] + r[:, 2, 0]
    yz = r[:, 1, 2] + r[:, 2, 1]
    _fill(branch == 0, qw2, [None, wx, wy, wz])
    _fill(branch == 1, qx2, [wx, None, xy, xz])
    _fill(branch == 2, qy2, [wy, xy, None, yz])
    _fill(branch == 3, qz2, [wz, xz, yz, None])

    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # Canonical sign: w >= 0.
    q[q[:, 0] < 0] *= -1
    return q


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions to unit length (zero-norm rows become identity)."""
    q = q.astype(np.float32, copy=True)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.divide(q, norm, out=q, where=norm > 0)
    q[norm[:, 0] == 0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q
