"""Fit-plan schema: spatial boxes + per-box splat budgets.

A :class:`FitPlan` is the output of the content-aware planner and the input to
:func:`luxar.gsplats.planner.fit_planned.fit_planned`. It is a flat list of
axis-aligned fit regions (boxes) over the *input volume*, each carrying a splat
budget. This is the noise-agnostic counterpart of the (tiling) axis: the planner
decides *how to decompose the volume and how many splats each region gets*; it
knows nothing about the calibration's noise regime — only the transferable
splats-per-feature density that the calibration emits.

Distinct from ``GSplatPartition`` (which partitions *fitted splats* post-hoc for
viewer culling). A :class:`FitPlan` partitions the *volume* pre-fit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class PlanBox:
    """One fit region: a half-open box ``[z0:z1, y0:y1, x0:x1]`` + a budget."""

    box: List[int]
    """Flat ``[z0, z1, y0, y1, x0, x1]`` in the input volume's voxel coordinates."""
    n_features: int
    """Feature content of the box (from the planner's content scan)."""
    budget: int
    """Splat seed budget for this box (from ``SplatDensity.predict_k``)."""

    @property
    def dims(self) -> List[int]:
        z0, z1, y0, y1, x0, x1 = self.box
        return [z1 - z0, y1 - y0, x1 - x0]

    @property
    def voxels(self) -> int:
        d = self.dims
        return int(d[0] * d[1] * d[2])


@dataclass
class FitPlan:
    """A content-balanced decomposition of a volume into budgeted fit regions."""

    volume_shape: List[int]
    boxes: List[PlanBox]
    overlap: int
    """Voxel halo added per box face at fit time (for seamless blending)."""
    feature_method: str
    """Content metric used (``peaks`` | ``edges`` | ``intensity``)."""
    min_leaf: int
    max_leaf: int
    density: Dict[str, Any] = field(default_factory=dict)
    """The :class:`SplatDensity` (as dict) used to size budgets, if any."""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_budget(self) -> int:
        return int(sum(b.budget for b in self.boxes))

    @property
    def n_boxes(self) -> int:
        return len(self.boxes)

    def overlap_fraction(self) -> Tuple[float, float]:
        """(median, max) per-box halo overhead fraction.

        ``fit_planned`` pads each box by ``overlap`` on *both* sides of every axis,
        so the fitted volume is ``(L + 2*overlap)`` per axis; the wasted (halo)
        fraction is ``1 - prod(L / (L + 2*overlap))``.
        """
        if not self.boxes:
            return 0.0, 0.0
        fracs = []
        for b in self.boxes:
            core = 1.0
            for L in b.dims:
                core *= L / (L + 2 * self.overlap)
            fracs.append(1.0 - core)
        fracs.sort()
        return fracs[len(fracs) // 2], fracs[-1]

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "FitPlan":
        raw = json.loads(Path(path).read_text())
        boxes = [
            PlanBox(
                box=[int(x) for x in b["box"]],
                n_features=int(b["n_features"]),
                budget=int(b["budget"]),
            )
            for b in raw["boxes"]
        ]
        return cls(
            volume_shape=[int(x) for x in raw["volume_shape"]],
            boxes=boxes,
            overlap=int(raw["overlap"]),
            feature_method=str(raw["feature_method"]),
            min_leaf=int(raw["min_leaf"]),
            max_leaf=int(raw["max_leaf"]),
            density=dict(raw.get("density", {})),
            meta=dict(raw.get("meta", {})),
        )


__all__ = ["FitPlan", "PlanBox"]
