"""Content-aware fit planner.

Decides *how to decompose a volume into fit regions and how many splats each
gets*, driven by a cheap content scan and the calibration's transferable
splats-per-feature density. Noise-agnostic (the calibration owns the K-selection
/ noise axis); this module owns the tiling / scale / budget axis.

Pipeline:
    scan_content(volume)            -> ContentField   (coarse feature density)
    plan_partition(field, density)  -> FitPlan         (boxes + per-box budgets)
    fit_planned(volume, plan)       -> GSplatData      (fit each box, merge)
"""

from .bsp_boxes import plan_partition, plan_volume
from .content_scan import ContentField, scan_content
from .fit_planned import fit_planned
from .fit_planned_parallel import fit_planned_parallel
from .spec import FitPlan, PlanBox

__all__ = [
    "ContentField",
    "FitPlan",
    "PlanBox",
    "fit_planned",
    "fit_planned_parallel",
    "plan_partition",
    "plan_volume",
    "scan_content",
]
