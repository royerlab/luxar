"""Shared globe-surface construction for the Earth-based demos.

The earthquake and ocean-current demos build a textured planet out of Points
and kept their own copies of the same three primitives. The copies drifted: the
ocean demo learned to dither its lattice and to sample the texture vectorized,
and the earthquake demo did not, so the same planet came out visibly worse in
one of them. These are the shared versions.

The three primitives, and why each is shaped the way it is:

``fibonacci_sphere``
    A golden-angle lattice, because a regular lat/lon grid clusters points at
    the poles and wastes most of its budget there. **Dithered by default**: the
    bare lattice is a regular pattern, and once the rendered point radius
    approaches the point spacing it beats against any equirectangular texture
    into visible moire — long curved "worms" that look like a broken land mask.
    The dither trades that structure for unstructured noise, which the eye
    forgives.

``lonlat_to_xyz``
    Geographic degrees plus a fractional radial ``relief`` to Cartesian, with
    ``y`` as the north-pole axis and longitude increasing eastward. The ``-z``
    keeps the frame right-handed (East x North = outward) so the globe is not
    mirrored — get this wrong and every continent is its own mirror image,
    which is surprisingly easy to miss on a rotating sphere.

``sample_equirect``
    Bilinear equirectangular lookup, vectorized over all points. The
    per-point-in-a-Python-loop version this replaces cost about 90 us a point,
    which is tolerable at 120k points and impossible at 8M.

``surface_point_radius``
    The one piece of arithmetic that decides whether a Points globe reads as a
    surface or as a dot screen: the rendered radius has to be tied to the point
    SPACING, which shrinks as 1/sqrt(n). A radius pinned to a constant while the
    count changes is exactly how a globe ends up stippled.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fibonacci_sphere",
    "lonlat_to_xyz",
    "sample_equirect",
    "surface_point_radius",
]


def fibonacci_sphere(
    n: int, *, jitter: bool = True, seed: int = 1234
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lon, lat)`` degrees for ``n`` points on a Fibonacci sphere.

    With ``jitter`` (the default) each point is dithered by up to half a mean
    angular spacing. The undithered lattice shows strong moire once the rendered
    point radius approaches the spacing; the dither trades that structure for
    unstructured noise, which is far less visible.

    Args:
        n: Number of points (must be >= 1).
        jitter: Dither the lattice by ~1 cell.
        seed: RNG seed for the dither (deterministic output).

    Returns:
        ``(lon, lat)`` float64 arrays of shape ``(n,)``, degrees.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    i = np.arange(n)
    golden = (1.0 + 5.0**0.5) / 2.0
    y = 1.0 - 2.0 * (i + 0.5) / n
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = 2.0 * np.pi * i / golden
    lat = np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(r_xy * np.sin(theta), r_xy * np.cos(theta)))
    if jitter:
        rng = np.random.default_rng(seed)
        cell = np.degrees(np.sqrt(4.0 * np.pi / n))  # mean angular spacing
        lat = np.clip(lat + rng.uniform(-0.5, 0.5, n) * cell, -89.999, 89.999)
        # a degree of longitude shrinks with cos(lat), so scale the dither up
        lon = lon + rng.uniform(-0.5, 0.5, n) * cell / np.maximum(
            np.cos(np.radians(lat)), 1e-2
        )
    return lon, lat


def lonlat_to_xyz(
    lon: np.ndarray,
    lat: np.ndarray,
    relief: np.ndarray | float,
    radius: float,
) -> np.ndarray:
    """Map geographic degrees + fractional ``relief`` to sphere xyz.

    ``y`` is the north-pole axis and longitude increases eastward; the ``-z``
    keeps the frame right-handed (East x North = outward) so the globe is not
    mirrored.

    Args:
        lon: Longitudes in degrees.
        lat: Latitudes in degrees.
        relief: Fractional radial displacement (0 = on the sphere). Scalar or
            per-point.
        radius: Sphere radius in scene units.

    Returns:
        ``(n, 3)`` float32 positions.
    """
    la, lo = np.radians(lat), np.radians(lon)
    r = radius * (1.0 + np.asarray(relief))
    cl = np.cos(la)
    return np.column_stack(
        [r * cl * np.cos(lo), r * np.sin(la), -r * cl * np.sin(lo)]
    ).astype(np.float32)


def sample_equirect(tex: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinearly sample an equirectangular RGB texture at ``lon``/``lat``.

    Vectorized over all points. Longitude wraps; latitude clamps.

    Args:
        tex: ``(h, w, 3)`` texture; row 0 is +90 deg latitude. Integer dtypes
            are treated as 0..255 and rescaled; float dtypes are assumed to be
            already normalized to [0, 1].
        lon: Longitudes in degrees (any range; wrapped).
        lat: Latitudes in degrees, -90..+90.

    Returns:
        ``(n, 3)`` float32 RGB in [0, 1].
    """
    h, w = tex.shape[:2]
    x = np.mod((lon + 180.0) / 360.0 * w, w)
    y = np.clip((90.0 - lat) / 180.0 * h, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = (x0 + 1) % w
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (x - x0)[:, None].astype(np.float32)
    wy = (y - y0)[:, None].astype(np.float32)
    t = tex.astype(np.float32)
    if np.issubdtype(tex.dtype, np.integer):
        t /= 255.0
    c0 = t[y0, x0] * (1.0 - wx) + t[y0, x1] * wx
    c1 = t[y1, x0] * (1.0 - wx) + t[y1, x1] * wx
    return np.clip(c0 * (1.0 - wy) + c1 * wy, 0.0, 1.0).astype(np.float32)


def surface_point_radius(n: int, radius: float, *, overlap: float = 0.75) -> float:
    """Point radius that makes ``n`` points on a sphere read as a solid surface.

    ``n`` points spread over a sphere of radius ``R`` each own a cap of area
    ``4*pi*R^2 / n``, so the mean centre-to-centre spacing is
    ``R * sqrt(4*pi/n)``. A point drawn with radius ``overlap`` times that
    spacing tiles with enough overlap to close the gaps without turning the
    surface into mush.

    The alternative — a constant radius — is what stipples a globe: it is tuned
    once at one point count and then silently becomes wrong when the count
    changes, in the direction of visible gaps if the count goes down and of a
    smeared surface if it goes up.

    Args:
        n: Number of surface points.
        radius: Sphere radius in scene units.
        overlap: Radius as a fraction of the mean spacing. 0.75 closes the gaps
            with a little margin; below ~0.55 the lattice starts to show
            through.

    Returns:
        Point radius in scene units.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    spacing = radius * float(np.sqrt(4.0 * np.pi / n))
    return spacing * overlap
