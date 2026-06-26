"""Content-density field: a coarse per-cell feature map for the planner.

A single cheap (CPU) pass over the volume produces a coarse 3-D grid whose cells
hold the local feature count (the same content metric the calibration uses to
derive its splats-per-feature density — :func:`luxar.gsplats.calibration.count_features`).
The field supports O(1) box-weight queries (summed-area table) and axis marginals,
which the orientation-aware BSP planner consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage as ndi


@dataclass
class ContentField:
    """Coarse feature-density grid over a volume + fast box queries."""

    density: np.ndarray
    """Coarse grid (``ceil(shape/cell)`` per axis); each cell = feature count."""
    cell: int
    """Full-resolution voxels per coarse cell."""
    shape: tuple
    """Original (full-resolution) volume shape."""
    method: str

    def __post_init__(self) -> None:
        # Summed-area table for O(1) box sums (built once).
        self._integral = self.density.cumsum(0).cumsum(1).cumsum(2)

    @property
    def total(self) -> float:
        return float(self.density.sum())

    def _S(self, a: int, b: int, c: int) -> float:
        if a < 0 or b < 0 or c < 0:
            return 0.0
        gz, gy, gx = self.density.shape
        return float(self._integral[min(a, gz - 1), min(b, gy - 1), min(c, gx - 1)])

    def box_weight(self, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> float:
        """Feature weight inside the full-res half-open box, via coarse cells."""
        c = self.cell
        a0, a1 = z0 // c, (z1 - 1) // c
        b0, b1 = y0 // c, (y1 - 1) // c
        d0, d1 = x0 // c, (x1 - 1) // c
        return (
            self._S(a1, b1, d1)
            - self._S(a0 - 1, b1, d1)
            - self._S(a1, b0 - 1, d1)
            - self._S(a1, b1, d0 - 1)
            + self._S(a0 - 1, b0 - 1, d1)
            + self._S(a0 - 1, b1, d0 - 1)
            + self._S(a1, b0 - 1, d0 - 1)
            - self._S(a0 - 1, b0 - 1, d0 - 1)
        )

    def marginal(self, box: tuple, axis: int) -> np.ndarray:
        """Weighted projection of ``box`` onto ``axis`` (over the box's cells)."""
        z0, z1, y0, y1, x0, x1 = box
        c = self.cell
        sub = self.density[
            z0 // c : (z1 + c - 1) // c,
            y0 // c : (y1 + c - 1) // c,
            x0 // c : (x1 + c - 1) // c,
        ]
        return np.asarray(sub.sum(axis=tuple(i for i in range(3) if i != axis)))


def scan_content(
    volume: np.ndarray,
    cell: int = 16,
    method: str = "peaks",
    downsample: int = 2,
    threshold_rel: float = 0.1,
    device: Optional[str] = None,
) -> ContentField:
    """Compute a coarse feature-density field over ``volume`` (CPU, one pass).

    Parameters
    ----------
    volume : np.ndarray
        3-D input volume.
    cell : int, default=16
        Coarse-cell edge (full-res voxels). Sets the planner's spatial resolution.
    method : {"peaks", "edges", "intensity"}, default="peaks"
        Feature detector — matches ``calibration.count_features``.
    downsample : int, default=2
        Detect features on a ``downsample``-strided volume for speed (peaks/edges);
        coordinates are mapped back to full-res before binning.
    threshold_rel : float, default=0.1
        Relative intensity threshold for peak / edge / foreground detection.
    """
    v = np.asarray(volume, dtype=np.float32)
    if v.ndim != 3:
        raise ValueError(f"scan_content expects a 3-D volume, got shape {v.shape}")
    Z, Y, X = v.shape
    gz, gy, gx = (Z + cell - 1) // cell, (Y + cell - 1) // cell, (X + cell - 1) // cell
    dens = np.zeros((gz, gy, gx), np.float64)

    ds = max(1, int(downsample))
    d = v[::ds, ::ds, ::ds]
    thr = float(d.max()) * threshold_rel

    if method == "peaks":
        from luxar.gsplats.seeds.utils import soft_blur_nd

        db = soft_blur_nd(d)
        mx = ndi.maximum_filter(db, size=3)
        coords = np.argwhere((db == mx) & (db > thr))
        if coords.size:
            coords = coords * ds  # back to full-res
            np.add.at(
                dens,
                (coords[:, 0] // cell, coords[:, 1] // cell, coords[:, 2] // cell),
                1.0,
            )
    elif method == "edges":
        from luxar.gsplats.seeds.edges import _compute_nd_sobel_magnitude

        mag = np.asarray(_compute_nd_sobel_magnitude(d, device=device))
        m = float(mag.max())
        if m > 0:
            ez, ey, ex = np.nonzero(mag >= threshold_rel * m)
            if ez.size:
                ez, ey, ex = ez * ds, ey * ds, ex * ds
                np.add.at(dens, (ez // cell, ey // cell, ex // cell), 1.0)
    elif method == "intensity":
        fz, fy, fx = np.nonzero(d > thr)
        if fz.size:
            fz, fy, fx = fz * ds, fy * ds, fx * ds
            np.add.at(dens, (fz // cell, fy // cell, fx // cell), 1.0)
    else:
        raise ValueError(
            f"unknown method {method!r}; use 'peaks', 'edges', or 'intensity'"
        )

    return ContentField(density=dens, cell=cell, shape=(Z, Y, X), method=method)


__all__ = ["ContentField", "scan_content"]
