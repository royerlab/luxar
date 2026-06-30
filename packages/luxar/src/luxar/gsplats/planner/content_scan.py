"""Content-density field: a coarse per-cell feature map for the planner.

A single cheap (CPU) pass over the volume produces a coarse 3-D grid whose cells
hold the local feature count (the same content metric the calibration uses to
derive its splats-per-feature density — :func:`luxar.gsplats.calibration.count_features`).
The field supports O(1) box-weight queries (summed-area table) and axis marginals,
which the orientation-aware BSP planner consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

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
    downsample: int = 1,
    threshold_rel: float = 0.1,
    threshold_abs: Optional[float] = None,
    device: Optional[str] = None,
) -> ContentField:
    """Compute a coarse feature-density field over ``volume`` (CPU, one pass).

    The total feature count is **consistent with ``calibration.count_features``**
    (same detector, same params) so the calibration's splats-per-feature density
    transfers correctly to the planner's per-box counts. This is why the default
    ``downsample=1``: a coarser scan would shrink the feature counts relative to
    the calibrated reference and miscalibrate the budgets. ``downsample>1`` trades
    that consistency for speed and must only be used if the density was
    calibrated at the same downsample.

    Parameters
    ----------
    volume : np.ndarray
        3-D input volume.
    cell : int, default=16
        Coarse-cell edge (full-res voxels). Sets the planner's spatial resolution.
    method : {"peaks", "edges", "intensity"}, default="peaks"
        Feature detector — matches ``calibration.count_features``.
    downsample : int, default=1
        Detect features on a ``downsample``-strided volume (1 = consistent with
        ``count_features``); coordinates are mapped back to full-res before binning.
    threshold_rel : float, default=0.1
        Relative intensity threshold for peak / edge / foreground detection.
    """
    v = np.asarray(volume, dtype=np.float32)
    if v.ndim != 3:
        raise ValueError(f"scan_content expects a 3-D volume, got shape {v.shape}")
    if method not in ("peaks", "edges", "intensity"):
        raise ValueError(
            f"unknown method {method!r}; use 'peaks', 'edges', or 'intensity'"
        )
    Z, Y, X = v.shape
    gz, gy, gx = (Z + cell - 1) // cell, (Y + cell - 1) // cell, (X + cell - 1) // cell
    dens = np.zeros((gz, gy, gx), np.float64)
    ds = max(1, int(downsample))

    # Global threshold (single cheap reduction, no temporary). A *global* level
    # makes per-box counts compose for the budget transfer; it also matches
    # count_features on the densest region (whose max ≈ global max), so the
    # calibration's reference count stays consistent.
    global_max = float(v.max())
    if global_max <= 0:
        return ContentField(density=dens, cell=cell, shape=(Z, Y, X), method=method)

    # Determine the absolute threshold to use, per method, so the detector here is
    # IDENTICAL to calibration.count_features (else per-box counts drift — badly
    # with hot outliers). Prefer an explicit threshold_abs (the exact level the
    # calibration recorded); otherwise reconstruct the same global level.
    if method == "edges":
        from luxar.gsplats.seeds.edges import _compute_nd_sobel_magnitude

        if threshold_abs is not None:
            thr = float(threshold_abs)
        else:
            # max over the SAME haloed slabs / interior the detection pass uses, so
            # the reference level is consistent at slab boundaries (M5).
            edge_max = 0.0
            for z0, z1, a0, a1 in _z_slabs(Z, cell, ds, halo=2 * ds):
                mag = np.asarray(_compute_nd_sobel_magnitude(v[a0:a1:ds, ::ds, ::ds]))
                zoff = (z0 - a0 + ds - 1) // ds
                interior = mag[zoff : zoff + (z1 - z0 + ds - 1) // ds]
                if interior.size:
                    edge_max = max(edge_max, float(interior.max()))
            thr = threshold_rel * edge_max
    elif method == "peaks" and threshold_abs is None:
        # count_local_maxima thresholds the *blurred* field at threshold_rel*blurred_max
        # (NOT raw max) — reconstruct the global blurred max so we match it.
        from luxar.gsplats.seeds.utils import soft_blur_nd

        bmax = 0.0
        for z0 in range(0, Z, _SLAB):
            sl = v[z0 : min(Z, z0 + _SLAB) : ds, ::ds, ::ds]
            if sl.size:
                bmax = max(bmax, float(np.asarray(soft_blur_nd(sl)).max()))
        thr = threshold_rel * bmax
    elif threshold_abs is not None:
        # peaks-with-threshold_abs and intensity-with-threshold_abs both use the
        # exact recorded level.
        thr = float(threshold_abs)
    elif method == "intensity":
        # count_features("intensity") thresholds at the Otsu cut, NOT a relative
        # level — reconstruct the SAME (single source of truth) so per-box counts
        # stay on the calibration's scale.
        from luxar.gsplats.calibration import _otsu_threshold

        thr = _otsu_threshold(v)
    else:  # defensive — peaks-without-abs is handled above
        thr = global_max * threshold_rel
    if thr <= 0:
        return ContentField(density=dens, cell=cell, shape=(Z, Y, X), method=method)

    halo = 2 * ds  # blur(1) + max-filter(1), scaled by downsample

    for z0, z1, a0, a1 in _z_slabs(Z, cell, ds, halo=halo):
        block = v[a0:a1:ds, ::ds, ::ds]
        # Interior = strided block rows whose full-res z lands in [z0, z1). The
        # exclusive upper bound is ceil over the LAST in-range full-res index from
        # a0 — `zoff + zlen` (sum of two independent ceils from different origins)
        # over-counts by one row when ds>1, double-counting features at the slab
        # boundary. (Identical to the old value at the production default ds=1.)
        zoff = (z0 - a0 + ds - 1) // ds  # first interior row within the (strided) block
        z_end = (z1 - 1 - a0) // ds + 1  # exclusive last interior row
        if method == "peaks":
            from luxar.gsplats.seeds.utils import soft_blur_nd

            db = soft_blur_nd(block)
            # match count_local_maxima exactly: blurred field, mode='nearest'
            # maximum_filter, '>=' threshold.
            mask = (db == ndi.maximum_filter(db, size=3, mode="nearest")) & (db >= thr)
        elif method == "edges":
            from luxar.gsplats.seeds.edges import _compute_nd_sobel_magnitude

            mask = np.asarray(_compute_nd_sobel_magnitude(block)) >= thr
        else:  # intensity — strict '>' matches calibration.foreground_mask_otsu
            mask = block > thr
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        interior = (coords[:, 0] >= zoff) & (coords[:, 0] < z_end)
        coords = coords[interior]
        if coords.size == 0:
            continue
        gzc = (coords[:, 0] * ds + a0) // cell
        gyc = (coords[:, 1] * ds) // cell
        gxc = (coords[:, 2] * ds) // cell
        np.add.at(dens, (gzc, gyc, gxc), 1.0)

    return ContentField(density=dens, cell=cell, shape=(Z, Y, X), method=method)


_SLAB = 64  # z-planes per streamed block (bounds peak-scan memory on huge volumes)


def _z_slabs(
    Z: int, cell: int, ds: int, halo: int = 0
) -> "Iterator[Tuple[int, int, int, int]]":
    """Yield ``(z0, z1, a0, a1)`` z-slabs with halo (full-res coords)."""
    for z0 in range(0, Z, _SLAB):
        z1 = min(Z, z0 + _SLAB)
        a0 = max(0, z0 - halo)
        a0 -= a0 % ds  # snap to stride phase so v[a0::ds] stays on the global grid (M4)
        a1 = min(Z, z1 + halo)
        yield z0, z1, a0, a1


__all__ = ["ContentField", "scan_content"]
