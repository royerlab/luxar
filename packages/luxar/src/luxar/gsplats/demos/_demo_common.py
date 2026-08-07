"""Shared numeric helpers for the gsplats research demos.

Most demos in this directory end the same way: overlay the fitted Gaussians on
the input as t-sigma ellipse polygons (2D) or ellipsoid wireframes (3D), and
report reconstruction quality as a PSNR. Those three routines were copy-pasted
into a dozen demo scripts and drifted only in their comments, so they live here
once and the demos import them.

Not a demo itself (no ``demo_`` prefix), so the demo byte-compile smoke test
skips it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ellipse_polygon_from_L",
    "ellipsoid_wireframe_from_L",
    "psnr",
]


def ellipse_polygon_from_L(
    mu_yx: np.ndarray, L: np.ndarray, t: float = 2.0, n_pts: int = 64
) -> np.ndarray:
    """Build a polygon approximating a splat's 2D t-sigma contour.

    The contour is the level set (x-mu)^T Sigma^{-1} (x-mu) = t^2, where
    Sigma = L L^T is the full covariance in voxel units.

    Args:
        mu_yx: Splat center as a length-2 ``(y, x)`` array.
        L: Lower-triangular ``(2, 2)`` Cholesky factor of the covariance.
        t: Contour level in standard deviations (2.0 -> the 2-sigma ellipse).
        n_pts: Number of polygon vertices sampled around the ellipse.

    Returns:
        ``(n_pts, 2)`` float32 array of ``(y, x)`` polygon points.
    """
    Sigma = L @ L.T  # (2,2)
    # Eigen-decompose Sigma for principal axes. Sigma is PSD, so the eigenvalues
    # are non-negative in exact arithmetic -- but for a near-singular Sigma eigh
    # can return a small NEGATIVE value, hence the clip below (without it sqrt
    # yields NaN; see tests/test_demo_common.py).
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 1e-12, None)
    # Radii along principal axes at level t: r_i = t * sqrt(lambda_i)
    radii = t * np.sqrt(evals)  # (2,)
    # Parametric angles
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2, n_pts)
    # Map unit circle -> ellipse in data coords: mu + R diag(r) circle
    pts = (evecs @ (radii[:, None] * circle)).T + mu_yx[None, :]
    return pts.astype(np.float32)


def ellipsoid_wireframe_from_L(
    mu_zyx: np.ndarray, L: np.ndarray, t: float = 2.0, n_pts: int = 32
) -> np.ndarray:
    """Build a wireframe approximating a splat's 3D t-sigma surface.

    The surface is the level set (x-mu)^T Sigma^{-1} (x-mu) = t^2, where
    Sigma = L L^T is the full covariance in voxel units. The wireframe is three
    circles, one in each of the ellipsoid's principal planes.

    Args:
        mu_zyx: Splat center as a length-3 ``(z, y, x)`` array.
        L: Lower-triangular ``(3, 3)`` Cholesky factor of the covariance.
        t: Contour level in standard deviations (2.0 -> the 2-sigma ellipsoid).
        n_pts: Number of vertices sampled per principal-plane circle.

    Returns:
        ``(3 * n_pts, 3)`` float32 array of ``(z, y, x)`` wireframe points.
    """
    Sigma = L @ L.T  # (3,3)
    # Eigen-decompose Sigma for principal axes. Sigma is PSD, so the eigenvalues
    # are non-negative in exact arithmetic -- but for a near-singular Sigma eigh
    # can return a small NEGATIVE value, hence the clip below (without it radii[0]
    # is NaN, poisoning the xy and xz circles; see tests/test_demo_common.py).
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, 1e-12, None)
    # Radii along principal axes at level t: r_i = t * sqrt(lambda_i)
    radii = t * np.sqrt(evals)  # (3,)

    # Create wireframe circles in the three principal planes
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    wireframe_pts = []

    # XY plane (z=0 in principal coords)
    circle_xy = np.zeros((n_pts, 3))
    circle_xy[:, 0] = radii[0] * cos_theta  # x-axis in principal coords
    circle_xy[:, 1] = radii[1] * sin_theta  # y-axis in principal coords
    circle_xy[:, 2] = 0  # z-axis
    # Transform to data coordinates
    pts_xy = (evecs @ circle_xy.T).T + mu_zyx[None, :]
    wireframe_pts.append(pts_xy)

    # XZ plane (y=0 in principal coords)
    circle_xz = np.zeros((n_pts, 3))
    circle_xz[:, 0] = radii[0] * cos_theta
    circle_xz[:, 1] = 0
    circle_xz[:, 2] = radii[2] * sin_theta
    pts_xz = (evecs @ circle_xz.T).T + mu_zyx[None, :]
    wireframe_pts.append(pts_xz)

    # YZ plane (x=0 in principal coords)
    circle_yz = np.zeros((n_pts, 3))
    circle_yz[:, 0] = 0
    circle_yz[:, 1] = radii[1] * cos_theta
    circle_yz[:, 2] = radii[2] * sin_theta
    pts_yz = (evecs @ circle_yz.T).T + mu_zyx[None, :]
    wireframe_pts.append(pts_yz)

    # Combine all wireframes
    return np.vstack(wireframe_pts).astype(np.float32)


def psnr(rendered: np.ndarray, target: np.ndarray) -> float:
    """Peak signal-to-noise ratio (dB) of ``rendered`` against ``target``.

    The peak is the dynamic RANGE of ``target`` (``max - min``), not its max, so
    these numbers are not interchangeable with the inline PSNRs sibling demos
    still compute: the ``max**2`` ones agree only where ``target.min()`` is 0,
    and ``demo_performance_metrics``' fixed peak of 1.0 agrees only where the
    range is exactly 1.0. Those inline versions also damp the MSE with
    ``+1e-12`` instead of guarding it, so they never return ``inf``.

    Returns ``inf`` for a perfect reconstruction (zero MSE) and for a flat
    ``target`` (zero range), where PSNR is undefined.

    Args:
        rendered: Reconstruction to score.
        target: Ground-truth array of the same shape.

    Returns:
        PSNR in decibels, or ``float("inf")``.
    """
    mse = float(np.mean((rendered.astype(np.float64) - target.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    # Promote BEFORE subtracting: the per-demo copies this consolidates all wrote
    # `float(target.max() - target.min())`, which does the subtraction in the input
    # dtype. A signed-integer target spanning more than half its dtype range then
    # overflows to a negative peak, the guard below fires, and a bad reconstruction
    # scores `inf`. This is the one line that is deliberately not a verbatim merge.
    rng = float(target.max()) - float(target.min())
    if rng <= 0.0:
        return float("inf")
    return 10.0 * float(np.log10(rng**2 / mse))
