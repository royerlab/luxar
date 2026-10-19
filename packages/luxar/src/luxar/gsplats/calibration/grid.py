"""Data-derived default K bounds for the calibration sweep.

The historical default grid, ``[1000, 512000]`` in 10 doublings, ignores the
volume: on a 3 Mvoxel stack whose cells cover 3% of it, its top point fits
~5 splats per foreground voxel and costs most of the sweep's wall-clock while
measuring nothing (the held-out curve peaked two decades lower). When the user
names no bound, :func:`auto_k_bounds` sizes the grid from the number of
FOREGROUND voxels instead: the top is capped at
:data:`FG_VOXELS_PER_SPLAT_AT_K_MAX` foreground voxels per splat, and the grid
keeps the historical span (``k_max / 512``), which puts its bottom far below any
K* we measured. Measurements: ``calibration/README.md`` (Sweep defaults).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

#: Foreground voxels per splat at the top of the data-derived grid. The K*
#: measured on dense 13-point sweeps sat at 20-40 foreground voxels per splat
#: (skimage cells3d nuclei), 42-85 (a dense 37.7 Mvoxel synthetic) and 100 (CTC
#: A549); a top point at 2 leaves more than three doublings above the densest,
#: room for the early stop to see the curve turn.
FG_VOXELS_PER_SPLAT_AT_K_MAX = 2.0

#: Historical ceiling: the data-derived top never exceeds the old default.
K_MAX_CEILING = 512_000

#: Lower clamp on the data-derived top, so a near-empty volume still sweeps a
#: grid wide enough to say so.
K_MAX_FLOOR = 4_000

#: Span of the grid (``k_max / k_min``): the historical 1000 -> 512000.
GRID_SPAN = 512

#: Never start the grid below this.
K_MIN_FLOOR = 50


@dataclass(frozen=True)
class AutoKBounds:
    """The data-derived sweep bounds and what they were derived from."""

    k_min: int
    k_max: int
    n_voxels: int
    n_foreground_voxels: int
    foreground_fraction: float
    floor_subtracted: float
    rule: str

    def to_dict(self) -> Dict[str, Any]:
        """Plain-data form for ``cal.json`` (``sweep.auto_grid``)."""
        return asdict(self)

    def summary(self) -> str:
        """One narration line for the CLI."""
        return (
            f"Auto K grid: {self.n_foreground_voxels:,} foreground voxels "
            f"({self.foreground_fraction * 100:.1f}% of {self.n_voxels:,}) -> "
            f"K in [{self.k_min:,}, {self.k_max:,}] ({self.rule}; pass "
            "--k-min/--k-max or --k-grid to override)"
        )


def _nice(k: float) -> int:
    """Round to 2 significant figures (keeps the narrated grid readable)."""
    if k <= 0:
        return 0
    mag = 10 ** max(int(math.floor(math.log10(k))) - 1, 0)
    return int(round(k / mag) * mag)


def auto_k_bounds(
    V: np.ndarray, *, floor: Any = "auto", n_points: int = 10
) -> AutoKBounds:
    """Sweep bounds sized from the foreground of ``V``.

    The foreground is the lightly smoothed Otsu mask of the floor-subtracted
    volume — the same mask ``cal`` scores its foreground-weighted metric on, and
    the same floor (``floor``) ``cal`` subtracts before fitting.
    ``k_max = clamp(n_fg / FG_VOXELS_PER_SPLAT_AT_K_MAX, K_MAX_FLOOR, 512000)``,
    never above the voxel count, and ``k_min = max(k_max / GRID_SPAN,
    K_MIN_FLOOR)``, never above ``k_max / 2``. ``n_points`` is accepted for the
    caller's symmetry with :func:`~.curve_analysis.build_k_grid`; the bounds do
    not depend on it.
    """
    from luxar.gsplats.fitting.preprocessing import _resolve_floor_result

    from .content import foreground_mask_otsu_smoothed

    Vf = np.asarray(V, dtype=np.float32)
    floor_result = _resolve_floor_result(Vf, floor)
    level = 0.0 if floor_result is None else float(floor_result.level)
    if level and level < float(Vf.max()):
        Vf = np.clip(Vf - level, 0.0, None)
    mask, _ = foreground_mask_otsu_smoothed(Vf)
    n_fg = int(np.count_nonzero(mask))
    raw_max = n_fg / FG_VOXELS_PER_SPLAT_AT_K_MAX
    # Never more splats than voxels (a tiny volume would otherwise get the
    # 4000 floor), and keep at least a doubling between the bounds.
    k_max = max(int(min(max(_nice(raw_max), K_MAX_FLOOR), K_MAX_CEILING, Vf.size)), 2)
    k_min = min(max(_nice(k_max / GRID_SPAN), K_MIN_FLOOR), max(k_max // 2, 1))
    return AutoKBounds(
        k_min=int(k_min),
        k_max=int(k_max),
        n_voxels=int(Vf.size),
        n_foreground_voxels=n_fg,
        foreground_fraction=float(n_fg / max(Vf.size, 1)),
        floor_subtracted=level,
        rule=(
            f"k_max = foreground voxels / {FG_VOXELS_PER_SPLAT_AT_K_MAX:g}, "
            f"clamped to [{K_MAX_FLOOR:,}, {K_MAX_CEILING:,}]; k_min = k_max / "
            f"{GRID_SPAN}"
        ),
    )
