"""Screen a built store's LOD ladders against the scene's OPENING framing.

``luxar restamp-lod`` (:mod:`luxar.io.lod_restamp`) can re-derive any legacy
``kind=lod`` ladder onto the current ``"screen-area"`` selector. This module
answers the question that decides whether doing so is *worth anything on a given
store*: at the framing the viewer actually opens with, does the re-derived ladder
pick a COARSER level than the stored one — i.e. does it defer any bytes at all?

A ladder can only defer detail if the opening shot leaves it room, and four
things make that a per-GROUP measurement rather than a per-demo slogan
(royerlab/luxar#1839):

* **The cut is the group's own anchor, not a flat 1.0.** A derived whole-object
  ladder's finest rung is
  :data:`~luxar.core.group.lod.group.WHOLE_OBJECT_FINEST_ANCHOR` (``0.5``); only
  a partition-bound ladder anchors at
  :data:`~luxar.core.group.lod.group.PARTITION_FINEST_AREA` (``1.0``). A
  whole-object group whose opening occupancy lands anywhere in ``[0.5, 1.0)``
  still picks the finest level on frame one, so re-deriving it changes nothing.
* **Per group, not per demo.** One number per store cannot describe a scene whose
  ladders are per-tile.
* **No inherited baseline.** What is picked TODAY is measured under the ladder
  and selector the store actually carries, and compared against what the
  RE-DERIVED ladder would pick. Against an assumed baseline a genuine win and a
  no-op look identical.
* **Aspect ratio is an input.** Every group is measured at 1:1, 16:9 and 21:9
  (:data:`DEFAULT_ASPECTS`); anything whose answer flips inside that range is
  reported as :data:`VERDICT_FRAGILE` rather than filed as a win or a no-op.

**The two selectors are in different units and are kept that way.** Under
``selector="screen-area"`` the metric is :func:`project_box_area_fraction` — the
group box's inscribed ellipsoid projected to a screen ellipse, sized by its
view-axis half-chord, clipped to the viewport and read in half-extent (rect)
units, saturating at exactly ``1.0``. Both metrics project the group's LOCAL box
through ``P·V·matrixWorld``, as the viewer does; only the off-screen frustum
gate uses the world AABB. Under the legacy ``selector="coverage"`` it is
:func:`legacy_coverage_metric` — the UNCLIPPED projected diagonal in pixels over
``FILL_FACTOR × min(viewportW, viewportH)``, range roughly ``[0, 4]``. Comparing
a stored coverage threshold against an area fraction is the mistake this module
exists to avoid, so ``today`` is always scored in the stamped selector's unit and
``re-derived`` always in area.

This is a **report and only a report**. Nothing here writes to a store and no
verdict is a failure: :func:`screen_stores` raises only on a genuine error (an
unreadable store), and the CLI in ``scripts/check_demo_ladders.py`` keeps its exit
code entirely off the screening pass.

**What is reproduced from the viewer, and where it lives.** Each of the following
mirrors a named TypeScript function; the citation is the contract, and a change
on either side should move both:

* :func:`calculate_camera_distance` —
  ``scene/scene-manager/clipping/bounds-math.ts::calculateCameraDistance``
* :func:`transform_box` — the same file's ``transformBoundingBox``
* :func:`project_bounds_to_display_dims` — the same file's
  ``projectBoundsToDisplayDims``
* :func:`project_box_ndc_rect`, :func:`project_box_area_fraction`,
  :func:`project_box_diagonal_px`, :func:`pick_child_with_hysteresis` —
  ``scene/lod-selector-math.ts``
* the ``FILL_FACTOR`` normalisation and the off-screen frustum gate —
  ``scene/lod-group-registry.ts``
* the fitted pose (face-on down ``-Z``, ``position = target + (0, 0, distance)``,
  ``up = +Y``) — ``scene/scene-manager/camera/camera-framing.ts``

**What is deliberately NOT modelled.** Dynamic near/far clipping: the near and
far planes are set far outside the fitted scene (:func:`_near_far_for`) so only
the four SIDE planes can gate. That costs nothing, because the near-plane hazard
the metrics actually care about is a box crossing the eye plane: a camera
inside the box saturates, while a box beside it is clipped at the near plane
(:func:`project_box_ndc_rect` and :func:`project_box_area_fraction`).
Nor is the viewer's projected-FOOTPRINT pick for GSplat ladders carrying
complete footprint stamps (``lod-group-registry.ts::pickStampedFootprintChild``),
which runs ahead of the occupancy metric there; the screen reports occupancy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import zarr
from arbol import aprint, asection

from .._zarr_compat import close, group_keys, open_group
from ..core.group.lod.group import (
    PARTITION_FINEST_AREA,
    WHOLE_OBJECT_FINEST_ANCHOR,
    coverage_fractions,
    partitioned_coverage_fractions,
)
from ..core.transforms import transform_bounding_box
from ..typing_utils.constants import DERIVED_LOD_SELECTOR, LOD_SELECTORS

# Private imports, deliberately — the same choice (and the same reasoning)
# `lod_restamp` itself documents where it imports `_child_nodes` / `_lod_children`
# from `_compiler.finalize.amplitude_window`. These four express rules this pass
# must MIRROR EXACTLY rather than re-state, and `lod_restamp` is their single
# source of truth: `_is_partition_bound` is the writers' two-clause anchor rule,
# `_count_of` is "which number sizes this child", `_threshold_of` is the stored
# `coverage_fraction` reader, and `_lod_children` / `_child_nodes` are the
# coarsest→finest ordering and the scene-node filter. A local copy of any of them
# would be one more place for the convention to drift, and the whole point of the
# screen is to predict what `restamp-lod` WOULD do.
from ._compiler.finalize.amplitude_window import _child_nodes, _lod_children
from .lod_restamp import (
    _count_of,
    _descending_ladder_refusal,
    _empty_ladder_refusal,
    _is_partition_bound,
    _orphan_ladder_child_refusal,
    _threshold_of,
)

__all__ = [
    "DEFAULT_ASPECTS",
    "DEFAULT_FIT_FOV",
    "DEFAULT_FIT_RATIO",
    "DEFAULT_VIEWPORT_LONG_PX",
    "DEGENERATE_RECT_HALF_EXTENT",
    "FILL_FACTOR",
    "HYSTERESIS_RATIO",
    "VERDICT_ALREADY_CURRENT",
    "VERDICT_FRAGILE",
    "VERDICT_NO_OP",
    "VERDICT_OFF_SCREEN",
    "VERDICT_ORDER",
    "VERDICT_SKIPPED",
    "VERDICT_WIN",
    "W_EPSILON",
    "AspectMeasurement",
    "Box3",
    "GroupScreening",
    "NdcRect",
    "SceneScreening",
    "ScreenReport",
    "calculate_camera_distance",
    "frustum_intersects_box",
    "frustum_planes",
    "legacy_coverage_metric",
    "mat4_from_column_major",
    "perspective_matrix",
    "pick_child_with_hysteresis",
    "print_screen_report",
    "project_bounds_to_display_dims",
    "project_box_area_fraction",
    "project_box_diagonal_px",
    "project_box_ndc_rect",
    "screen_lod_store",
    "screen_stores",
    "transform_box",
    "view_matrix",
]


# --------------------------------------------------------------------------- #
# Constants mirrored from the viewer. Each is the single value its TS twin
# exports; a divergence here silently makes the whole screen wrong, so they are
# named (never inlined) and pinned by the colocated tests.
# --------------------------------------------------------------------------- #

#: ``config.scene.defaultFitRatio`` (``config/sections/scene/data.ts``) — the
#: share of the fitted viewport axis the scene bbox fills on load.
DEFAULT_FIT_RATIO = 0.75

#: ``renderingControls.defaults.fov`` (``config/sections/rendering-controls``).
#: The FITTED DISTANCE always uses this, even under the cinematic preset, which
#: overrides the FOV only after the fit has run — hence the two separate
#: ``fit_fov`` / ``render_fov`` parameters throughout this module.
DEFAULT_FIT_FOV = 47.0

#: ``FILL_FACTOR`` (``scene/lod-group-registry.ts``) — the legacy ``"coverage"``
#: metric's denominator is ``FILL_FACTOR × min(viewportW, viewportH)``.
FILL_FACTOR = 0.5

#: ``DEGENERATE_RECT_HALF_EXTENT`` (``scene/lod-selector-math.ts``) — below this
#: RAW minor semi-axis the area metric ramps to the ellipse's linear span instead.
DEGENERATE_RECT_HALF_EXTENT = 1e-3

#: ``W_EPSILON`` (``scene/lod-selector-math.ts``) — a corner at or below this
#: homogeneous ``w`` means the box crosses the eye plane; only a camera inside saturates.
W_EPSILON = 1e-6

#: ``HYSTERESIS_RATIO`` (``scene/lod-selector-math.ts``).
HYSTERESIS_RATIO = 0.1

#: The three aspect ratios every group is reported at, as ``(label, value)``.
#: A square monitor, the mainstream 16:9, and an ultrawide 21:9 — wide enough
#: that a verdict surviving all three is not an artefact of one window shape.
DEFAULT_ASPECTS: Tuple[Tuple[str, float], ...] = (
    ("1:1", 1.0),
    ("16:9", 16.0 / 9.0),
    ("21:9", 21.0 / 9.0),
)

#: Pixels along the viewport's LONG axis. The area metric is viewport-size
#: independent by construction, but the legacy diagonal metric is not — it is
#: pixels over ``FILL_FACTOR × min(width, height)`` — so a pixel size has to be
#: chosen for it. 1920 is the mainstream desktop long axis; ``--viewport-long``
#: exposes it because a store screened at one size may bucket differently at
#: another, and that is worth being able to check.
DEFAULT_VIEWPORT_LONG_PX = 1920

#: Every tested aspect picks a strictly coarser level under the re-derived
#: ladder — re-deriving genuinely defers detail on the opening shot.
VERDICT_WIN = "win"

#: No tested aspect picks a coarser level. A re-derived ladder may still pick a
#: finer, more expensive level; the per-aspect report marks that as ``FINER``.
VERDICT_NO_OP = "no-op"

#: Coarser at some tested aspects and not at others. Reported as its own bucket
#: rather than rounded into a win or a no-op: the answer depends on the window.
VERDICT_FRAGILE = "fragile"

#: The world box misses the camera frustum at every tested aspect, so the viewer
#: never takes a metric — it holds the coarsest ready level regardless of ladder.
VERDICT_OFF_SCREEN = "off-screen"

#: Already on ``selector="screen-area"``, which is all ``restamp-lod`` looks at
#: before skipping the group — so there is nothing a rewrite would change here,
#: whatever the stored thresholds happen to be.
VERDICT_ALREADY_CURRENT = "already-current"

#: The group could not be decided; :attr:`GroupScreening.reason` says why.
VERDICT_SKIPPED = "skipped"

#: Every bucket, in report order (most-interesting first).
VERDICT_ORDER: Tuple[str, ...] = (
    VERDICT_WIN,
    VERDICT_FRAGILE,
    VERDICT_NO_OP,
    VERDICT_OFF_SCREEN,
    VERDICT_ALREADY_CURRENT,
    VERDICT_SKIPPED,
)

#: Two derived ladders count as the same one within this absolute tolerance.
#: The stored value is a JSON float that came out of the very same derivation,
#: so anything above round-trip noise is a real difference.
_LADDER_EQUAL_ATOL = 1e-9


Vec3 = Tuple[float, float, float]

#: A node's ``position_bounds`` / ``lod_bounds`` as ``(min, max)`` nD arrays.
_BoundsPair = Tuple[List[float], List[float]]


@dataclass(frozen=True)
class Box3:
    """An axis-aligned 3D box, matching the viewer's ``BoundingBox`` shape."""

    min: Vec3
    max: Vec3

    @property
    def center(self) -> Vec3:
        """The box centre — the viewer's ``getBoundingBoxCenter``."""
        return (
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        )

    @property
    def diagonal(self) -> float:
        """Length of the box diagonal (``0`` for a degenerate box)."""
        dx = self.max[0] - self.min[0]
        dy = self.max[1] - self.min[1]
        dz = self.max[2] - self.min[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(frozen=True)
class NdcRect:
    """A box's screen-space AABB in NDC, as :func:`project_box_ndc_rect` returns.

    ``half_w`` / ``half_h`` are the UNCLIPPED ``(max - min) / 2`` spans the
    diagonal metric consumes; the four bounds are raw and unclamped.
    """

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @property
    def half_w(self) -> float:
        """Half the unclipped NDC width."""
        return (self.max_x - self.min_x) * 0.5

    @property
    def half_h(self) -> float:
        """Half the unclipped NDC height."""
        return (self.max_y - self.min_y) * 0.5


# --------------------------------------------------------------------------- #
# Matrices. Internally these are ordinary row-major NumPy 4x4s (``M @ v``); the
# only column-major values in the module are the 16-element ``transform`` attrs
# read off the store, which :func:`mat4_from_column_major` converts on the way in.
# --------------------------------------------------------------------------- #


def mat4_from_column_major(flat: Sequence[float]) -> np.ndarray:
    """A stored 16-element THREE.js (column-major) transform as a math 4x4.

    Args:
        flat: The ``transform`` attr exactly as written to zarr — column-major,
            so ``flat[col * 4 + row]``.

    Returns:
        The same matrix in row-major NumPy convention, ready for ``M @ v``.

    Raises:
        ValueError: ``flat`` does not hold 16 numbers.
    """
    values = np.asarray(flat, dtype=np.float64)
    if values.size != 16:
        raise ValueError(f"a transform must hold 16 numbers, got {values.size}")
    return values.reshape(4, 4).T


def perspective_matrix(
    fov_deg: float, aspect: float, near: float, far: float
) -> np.ndarray:
    """THREE's symmetric perspective projection, as a row-major 4x4.

    Mirrors ``PerspectiveCamera.updateProjectionMatrix`` →
    ``Matrix4.makePerspective`` at ``zoom == 1`` with no view offset, where the
    frustum is symmetric so the skew terms vanish.

    Args:
        fov_deg: VERTICAL field of view in degrees.
        aspect: Viewport width / height.
        near: Near plane distance (> 0).
        far: Far plane distance (> ``near``).

    Returns:
        The projection matrix. Only the X/Y rows and the ``w`` row affect the
        LOD metrics; the Z row exists so the frustum's near/far planes are real.
    """
    tan_half = math.tan(math.radians(fov_deg) * 0.5)
    x = 1.0 / (aspect * tan_half)
    y = 1.0 / tan_half
    c = -(far + near) / (far - near)
    d = -2.0 * far * near / (far - near)
    return np.array(
        [
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, c, d],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


def view_matrix(position: Vec3) -> np.ndarray:
    """The view matrix of the viewer's fitted, face-on camera pose.

    ``fitCameraToBounds`` places the camera at ``target + (0, 0, distance)`` with
    ``up = +Y`` and ``lookAt(target)``. For the default up that pose has an
    IDENTITY rotation (the camera already looks down ``-Z``), so
    ``matrixWorldInverse`` is a pure translation by ``-position``. A scene that
    authors its own ``viewer_config.camera.up`` / ``position`` breaks that and is
    skipped by :func:`screen_lod_store` rather than mis-modelled here.

    Args:
        position: The camera's world position.

    Returns:
        The 4x4 world→view matrix.
    """
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = -position[0]
    matrix[1, 3] = -position[1]
    matrix[2, 3] = -position[2]
    return matrix


def calculate_camera_distance(
    box: Box3,
    fov_deg: float,
    aspect: float,
    fit_ratio: float = DEFAULT_FIT_RATIO,
    target: Optional[Vec3] = None,
) -> float:
    """The ``+Z`` distance that frames ``box`` on load.

    A literal transcription of ``bounds-math.ts::calculateCameraDistance``: the
    larger screen-plane half-extent about ``target`` fills ``fit_ratio`` of the
    fitted viewport axis at the NEAREST ``Z`` face of the box, and the vertical
    and horizontal fits are maxed so both extents are covered.

    Args:
        box: The box to frame (the scene root's projected ``position_bounds``).
        fov_deg: VERTICAL field of view in degrees — always the fit FOV (47 by
            default), never a cinematic override, which lands after the fit.
        aspect: Viewport width / height.
        fit_ratio: Share of the fitted axis the box fills.
        target: Look-at point; the box centre when omitted.

    Returns:
        The camera distance along ``+Z`` from ``target``.
    """
    look_at = box.center if target is None else target
    half_fov = math.radians(fov_deg) * 0.5
    nearest_depth = max(0.0, box.max[2] - look_at[2])
    screen_plane_radius = max(
        abs(box.min[0] - look_at[0]),
        abs(box.max[0] - look_at[0]),
        abs(box.min[1] - look_at[1]),
        abs(box.max[1] - look_at[1]),
    )
    fit_radius = (
        screen_plane_radius
        if screen_plane_radius > 0
        else abs(box.max[2] - box.min[2]) / 2.0
    )
    vertical_fit = nearest_depth + fit_radius / fit_ratio / math.tan(half_fov)
    horizontal_fit = nearest_depth + fit_radius / fit_ratio / (
        math.tan(half_fov) * aspect
    )
    return max(vertical_fit, horizontal_fit)


def _box_corners(box: Box3) -> np.ndarray:
    """The box's eight corners as homogeneous rows, in the viewer's bit order.

    ``i & 1`` selects max-X, ``i & 2`` max-Y, ``i & 4`` max-Z — the same
    enumeration ``projectBoxNdcRect`` uses, so a corner-order bug would show up
    identically on both sides rather than cancelling out.
    """
    corners = np.ones((8, 4), dtype=np.float64)
    for i in range(8):
        corners[i, 0] = box.max[0] if i & 1 else box.min[0]
        corners[i, 1] = box.max[1] if i & 2 else box.min[1]
        corners[i, 2] = box.max[2] if i & 4 else box.min[2]
    return corners


def transform_box(box: Box3, matrix: np.ndarray) -> Box3:
    """Lift a box through a 4x4 via the shared bounds implementation.

    Args:
        box: The box in local space.
        matrix: A row-major 4x4.

    Returns:
        The world-space AABB of the transformed corners.
    """
    lo, hi = transform_bounding_box(matrix, box.min, box.max)
    return Box3(
        (float(lo[0]), float(lo[1]), float(lo[2])),
        (float(hi[0]), float(hi[1]), float(hi[2])),
    )


def project_bounds_to_display_dims(
    min_bounds: Sequence[float],
    max_bounds: Sequence[float],
    display_dims: Sequence[int],
) -> Box3:
    """Fold nD ``position_bounds`` onto X/Y/Z, as the viewer's own projection does.

    ``bounds-math.ts::projectBoundsToDisplayDims``: at most the first three
    ``display_dims`` are consumed, and an axis that is unmapped OR out of range
    for the supplied arrays stays at ``0`` — the viewer's defensive fallback, not
    an error.

    Args:
        min_bounds: Per-dimension minima (length = ndim).
        max_bounds: Per-dimension maxima (length = ndim).
        display_dims: Dimension indices mapped to X, Y, Z.

    Returns:
        The projected 3D box.
    """
    lo = [0.0, 0.0, 0.0]
    hi = [0.0, 0.0, 0.0]
    for axis in range(min(3, len(display_dims))):
        dim = display_dims[axis]
        if dim < len(min_bounds) and dim < len(max_bounds):
            lo[axis] = float(min_bounds[dim])
            hi[axis] = float(max_bounds[dim])
    return Box3((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))


def project_box_ndc_rect(
    box: Box3, proj_view: np.ndarray, near: Optional[float] = None
) -> Optional[NdcRect]:
    """The box's screen-space AABB in NDC, or ``None`` with the camera inside it.

    ``lod-selector-math.ts::projectBoxNdcRect``: the eight corners go through
    ``P·V`` with an EXPLICIT homogeneous ``w`` (not THREE's unguarded
    ``Vector3.project``). When a corner is at ``w <= W_EPSILON`` the eye plane
    cuts the box and the perspective divide of the raw corners is meaningless:
    with the camera INSIDE the box this returns ``None`` (both metrics saturate
    to ``+inf``); with the eye outside it (a box running past the camera) it is
    the rect of the part in front of the near plane
    (:func:`_clipped_ndc_rect`), or an empty rect when nothing is.

    Args:
        box: A world-space box.
        proj_view: The row-major ``projection @ view`` product.

    Returns:
        The NDC rect, or ``None`` when the camera is inside ``box``.
    """
    projected = _box_corners(box) @ proj_view.T
    w = projected[:, 3]
    if bool((w <= W_EPSILON).any()):
        if _eye_inside_box(box, proj_view):
            return None
        return _clipped_ndc_rect(box, proj_view, near) or NdcRect(0.0, 0.0, 0.0, 0.0)
    ndc_x = projected[:, 0] / w
    ndc_y = projected[:, 1] / w
    return NdcRect(
        float(ndc_x.min()), float(ndc_x.max()), float(ndc_y.min()), float(ndc_y.max())
    )


#: The 12 box edges as corner-index pairs (corner bits as :func:`_box_corners`).
_BOX_EDGES = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _eye_inside_box(box: Box3, m: np.ndarray) -> bool:
    """Whether the eye lies inside (or on) ``box``, in the frame of ``m``.

    ``lod-selector-math.ts::eyeInsideBox``. The eye is the point whose clip
    ``x``, ``y`` and ``w`` all vanish (rows 0, 1 and 3 of ``m``); a frame with
    no such point (singular) answers ``True``, keeping the saturation.
    """
    a = np.array([m[0, :3], m[1, :3], m[3, :3]], dtype=np.float64)
    det = float(np.linalg.det(a))
    if not (abs(det) > 0 and math.isfinite(det)):
        return True
    eye = np.linalg.solve(a, -np.array([m[0, 3], m[1, 3], m[3, 3]]))
    return all(box.min[i] <= float(eye[i]) <= box.max[i] for i in range(3))


def _clipped_ndc_rect(
    box: Box3, m: np.ndarray, near_distance: Optional[float] = None
) -> Optional[NdcRect]:
    """The NDC AABB of the part of ``box`` in front of the near plane.

    ``lod-selector-math.ts::clippedNdcRect``: the corners with view depth
    ``w >= near_distance`` (and ``w > 0``), plus every edge's crossing of that plane,
    each divided by its ``w`` — the box as the renderer clips it. Unclamped.
    ``None`` when nothing is in front.
    """
    clip = _box_corners(box) @ m.T
    # With no explicit near distance, the Python mirror's WebGL projection
    # makes z+w the same physical plane. The runtime passes near explicitly.
    near = (
        clip[:, 3] - near_distance
        if near_distance is not None
        else clip[:, 2] + clip[:, 3]
    )
    points = [clip[n] for n in range(8) if near[n] >= 0 and clip[n, 3] > 0]
    for p, q in _BOX_EDGES:
        if (near[p] >= 0) == (near[q] >= 0):
            continue
        t = near[p] / (near[p] - near[q])
        point = clip[p] + t * (clip[q] - clip[p])
        if point[3] > 0:
            points.append(point)
    if not points:
        return None
    ndc_x = [float(pt[0] / pt[3]) for pt in points]
    ndc_y = [float(pt[1] / pt[3]) for pt in points]
    return NdcRect(min(ndc_x), max(ndc_x), min(ndc_y), max(ndc_y))


def _straddling_area_fraction(
    box: Box3, m: np.ndarray, near: Optional[float] = None
) -> float:
    """:func:`project_box_area_fraction` when the eye plane cuts the box.

    ``lod-selector-math.ts::straddlingAreaFraction``: ``+inf`` with the eye
    inside the box, else the viewport-clipped area fraction of the near-clipped
    rect (:func:`_clipped_ndc_rect`).
    """
    if _eye_inside_box(box, m):
        return math.inf
    rect = _clipped_ndc_rect(box, m, near)
    if rect is None:
        return 0.0
    overlap_w = min(rect.max_x, 1.0) - max(rect.min_x, -1.0)
    overlap_h = min(rect.max_y, 1.0) - max(rect.min_y, -1.0)
    if overlap_w < 0 or overlap_h < 0:
        return 0.0
    return overlap_w * overlap_h / 4


@dataclass
class _Ellipse:
    """A projected ellipse in NDC: centre ``(cx, cy)``, shape matrix ``S``.

    The ellipse is ``{c + S^{1/2} u : |u| = 1}``, so ``sqrt(sxx)`` / ``sqrt(syy)``
    are the half-extents of its screen AABB and ``sqrt(det S)`` is the product of
    its semi-axes.
    """

    cx: float
    cy: float
    sxx: float
    sxy: float
    syy: float


#: :func:`_project_near_depth_ellipse`: the eye plane cuts the box.
_ELLIPSE_STRADDLES = "straddles"
#: :func:`_project_near_depth_ellipse`: the box lies wholly behind the eye.
_ELLIPSE_BEHIND = "behind"


def _half_extents_sq(box: Box3) -> Tuple[float, float, float]:
    """The squared half-extents of ``box`` — its inscribed ellipsoid's semi-axes²."""
    return (
        (0.5 * (box.max[0] - box.min[0])) ** 2,
        (0.5 * (box.max[1] - box.min[1])) ** 2,
        (0.5 * (box.max[2] - box.min[2])) ** 2,
    )


def _weighted_row_dot(
    m: np.ndarray, a: int, b: int, h2: Tuple[float, float, float]
) -> float:
    """``Σ h_i² m[a, i] m[b, i]`` over the three spatial columns of rows a and b."""
    return float(sum(h2[i] * m[a, i] * m[b, i] for i in range(3)))


def _project_inscribed_ellipse(box: Box3, m: np.ndarray) -> "_Ellipse | str":
    """The image of the ellipsoid inscribed in ``box`` under the box-to-clip ``m``.

    ``lod-selector-math.ts::projectInscribedEllipse``. The ellipsoid's dual
    quadric ``T diag(hx², hy², hz², −1) Tᵀ`` projects through the x, y and w
    rows of ``m`` to a dual conic; normalised so its ``w,w`` entry is ``−1`` it
    reads ``[[S − ccᵀ, −c], [−cᵀ, −1]]``. ``k = w_c² − Σ(h_i m_{3i})²`` is
    positive iff the eye plane misses the ellipsoid.
    """
    h2 = _half_extents_sq(box)
    centre = np.array([*box.center, 1.0])
    u = float(m[0] @ centre)
    v = float(m[1] @ centre)
    w = float(m[3] @ centre)
    k = w * w - _weighted_row_dot(m, 3, 3, h2)
    if not k > 0:
        return _ELLIPSE_STRADDLES
    if w < 0:
        return _ELLIPSE_BEHIND
    cx = (u * w - _weighted_row_dot(m, 0, 3, h2)) / k
    cy = (v * w - _weighted_row_dot(m, 1, 3, h2)) / k
    return _Ellipse(
        cx=cx,
        cy=cy,
        sxx=(_weighted_row_dot(m, 0, 0, h2) - u * u) / k + cx * cx,
        sxy=(_weighted_row_dot(m, 0, 1, h2) - u * v) / k + cx * cy,
        syy=(_weighted_row_dot(m, 1, 1, h2) - v * v) / k + cy * cy,
    )


def _ellipsoid_view_chord(box: Box3, m: np.ndarray) -> float:
    """The inscribed ellipsoid's half-chord along the camera axis, in ``w`` units.

    ``lod-selector-math.ts::ellipsoidViewChord``. The camera axis in the box
    frame is the cross product of the clip x/y rows — NOT the ``w`` gradient,
    which an anisotropic group scale tilts — so a scaled local box reads the
    same chord as the identical world box.
    """
    h2 = _half_extents_sq(box)
    view_axis = np.cross(m[0, :3], m[1, :3])
    depth_along_axis = 0.0
    inverse_radius_sq = 0.0
    for i in range(3):
        if h2[i] == 0 and view_axis[i] != 0:
            return 0.0
        depth_along_axis += float(m[3, i] * view_axis[i])
        if h2[i] > 0:
            inverse_radius_sq += float(view_axis[i] ** 2) / h2[i]
    if not inverse_radius_sq > 0:
        return 0.0
    return abs(depth_along_axis) / math.sqrt(inverse_radius_sq)


def _project_near_depth_ellipse(box: Box3, m: np.ndarray) -> "_Ellipse | str":
    """The inscribed-ellipsoid image SIZED by its view-axis half-chord.

    ``lod-selector-math.ts::projectNearDepthEllipse``. On the view axis the bare
    ellipsoid silhouette reads at the geometric-mean depth
    ``sqrt((w_c − c)(w_c + c))``; scaling ``S`` by ``(w_c + c)/(w_c − c)`` moves
    it to the near depth ``w_c − c``, so a face-on box on the axis reads EXACTLY
    its near-face rect at any thickness. The centre is not moved. ``w_c`` is the
    midpoint of the corners' ``w`` range. An orthographic projection or a flat
    tilted card has a zero chord and is not resized.

    Returns :data:`_ELLIPSE_BEHIND` when every corner is behind the eye and
    :data:`_ELLIPSE_STRADDLES` when the nearest corner is at/behind
    :data:`W_EPSILON` — the corner-rect rule, kept even when the eye plane
    misses the inscribed ellipsoid (the caller then saturates or near-clips,
    see :func:`_straddling_area_fraction`).
    """
    w = _box_corners(box) @ m[3]
    w_near = float(w.min())
    w_far = float(w.max())
    if not w_far > 0:
        return _ELLIPSE_BEHIND
    if not w_near > W_EPSILON:
        return _ELLIPSE_STRADDLES
    ellipse = _project_inscribed_ellipse(box, m)
    if isinstance(ellipse, str):
        return ellipse
    w_centre = 0.5 * (w_near + w_far)
    chord = _ellipsoid_view_chord(box, m)
    area_scale = (w_centre + chord) / (w_centre - chord)
    ellipse.sxx *= area_scale
    ellipse.sxy *= area_scale
    ellipse.syy *= area_scale
    return ellipse


def _viewport_inside_ellipse(ellipse: _Ellipse, det: float) -> bool:
    """A convex ellipse contains the viewport iff it contains all four corners."""
    if not (det > 0 and math.isfinite(det)):
        return False
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            dx = x - ellipse.cx
            dy = y - ellipse.cy
            inside = (
                ellipse.syy * dx * dx
                - 2 * ellipse.sxy * dx * dy
                + ellipse.sxx * dy * dy
                <= det
            )
            if not inside:
                return False
    return True


def project_box_area_fraction(
    box: Box3, proj_view: np.ndarray, near: Optional[float] = None
) -> float:
    """The ``selector="screen-area"`` metric: the visible viewport area fraction.

    ``lod-selector-math.ts::projectBoxAreaFraction``, ported exactly. The box is
    measured through its INSCRIBED ellipsoid (semi-axes = the box half-extents),
    projected as a dual quadric onto an image-plane ellipse and sized by the
    ellipsoid's view-axis half-chord (:func:`_project_near_depth_ellipse`); the
    metric is ``sqrt(det S)`` of that ellipse, so an orbit at fixed distance
    does not walk the ladder the way the 8-corner rect did. Face-on on the view
    axis, and under an orthographic camera, it is exactly the legacy rect
    product ``halfW × halfH`` of the near face.

    The area is scaled per axis by the VISIBLE fraction of the ellipse's screen
    AABB, and reads exactly ``1.0`` once the ellipse contains the whole viewport;
    a SIGNED negative overlap on either axis means no viewport overlap at all
    and zeroes the metric before the degenerate ramp can see it. That ramp —
    gated on the RAW, pre-clip MINOR semi-axis — lets sub-pixel-thin content (an
    axis-aligned polyline, an edge-on plane) read its clipped LINEAR span instead
    of a vanishing area, decaying continuously to the plain area at the floor.

    Args:
        box: A box in the frame ``proj_view`` maps to clip space — group-local
            bounds with ``P·V·matrixWorld``, or world bounds with ``P·V``.
        proj_view: The row-major box-to-clip matrix.

    When the eye plane cuts the box (no finite near depth to size the ellipse
    at) the metric is ``+inf`` with the camera INSIDE the box, else the
    viewport-clipped area of the near-clipped rect: a box running past the
    camera reads what it covers instead of forcing the finest level.

    Returns:
        A fraction in ``[0, 1]``, or ``+inf`` when the camera is inside the box.
    """
    ellipse = _project_near_depth_ellipse(box, proj_view)
    if isinstance(ellipse, str):
        if ellipse == _ELLIPSE_STRADDLES:
            return _straddling_area_fraction(box, proj_view, near)
        return 0.0
    # A zero-extent axis leaves a diagonal entry at rounding noise, which may be
    # a hair below zero; that axis is simply zero-width.
    ex = math.sqrt(max(0.0, ellipse.sxx))
    ey = math.sqrt(max(0.0, ellipse.syy))
    overlap_w = min(ellipse.cx + ex, 1.0) - max(ellipse.cx - ex, -1.0)
    overlap_h = min(ellipse.cy + ey, 1.0) - max(ellipse.cy - ey, -1.0)
    if overlap_w < 0 or overlap_h < 0:
        return 0.0
    det = ellipse.sxx * ellipse.syy - ellipse.sxy * ellipse.sxy
    if _viewport_inside_ellipse(ellipse, det):
        return 1.0
    visible_w = min(1.0, overlap_w / (2 * ex)) if ex > 0 else 1.0
    visible_h = min(1.0, overlap_h / (2 * ey)) if ey > 0 else 1.0
    area = math.sqrt(max(0.0, det)) * visible_w * visible_h
    half_trace = 0.5 * (ellipse.sxx + ellipse.syy)
    half_diff = 0.5 * (ellipse.sxx - ellipse.syy)
    minor_sq = half_trace - math.sqrt(half_diff * half_diff + ellipse.sxy**2)
    raw_thin = math.sqrt(max(0.0, minor_sq))
    span = max(overlap_w, overlap_h) * 0.5
    degenerate = span * max(0.0, 1.0 - raw_thin / DEGENERATE_RECT_HALF_EXTENT)
    return max(area, degenerate)


def project_box_diagonal_px(
    box: Box3,
    proj_view: np.ndarray,
    width: float,
    height: float,
    near: Optional[float] = None,
) -> float:
    """The box's UNCLIPPED projected screen diagonal in pixels.

    ``lod-selector-math.ts::projectBoxDiagonalPx``. Unclipped is the point: this
    is the legacy metric's raw input and it grows without bound as the box
    overflows the viewport, which is exactly why it cannot be compared against an
    area fraction.

    Args:
        box: A world-space box.
        proj_view: The row-major ``projection @ view`` product.
        width: Viewport width in pixels.
        height: Viewport height in pixels.

    Returns:
        The pixel diagonal, or ``+inf`` with the camera inside the box.
    """
    rect = project_box_ndc_rect(box, proj_view, near)
    if rect is None:
        return math.inf
    return math.hypot(rect.half_w * width, rect.half_h * height)


def legacy_coverage_metric(
    box: Box3, proj_view: np.ndarray, width: float, height: float
) -> float:
    """The dimensionless legacy ``selector="coverage"`` metric.

    ``lod-group-registry.ts``: the projected pixel diagonal over
    ``FILL_FACTOR × min(width, height)``. The denominator is the SHORTER viewport
    axis because that is the axis ``calculateCameraDistance`` fits, on both sides
    of aspect 1. Range is roughly ``[0, 4]`` — a different scale from the area
    metric's ``[0, 1]``, never interchangeable with it.

    Args:
        box: A world-space box.
        proj_view: The row-major ``projection @ view`` product.
        width: Viewport width in pixels.
        height: Viewport height in pixels.

    Returns:
        The coverage metric, or ``+inf`` with the camera inside the box.
    """
    fitted_axis_px = min(width, height)
    return project_box_diagonal_px(box, proj_view, width, height) / (
        FILL_FACTOR * fitted_axis_px
    )


def pick_child_with_hysteresis(
    thresholds: Sequence[float],
    current_idx: int,
    metric: float,
    hysteresis_ratio: float = HYSTERESIS_RATIO,
) -> int:
    """The viewer's level pick, hysteresis included.

    ``lod-selector-math.ts::pickChildWithHysteresis``. The natural pick is the
    finest child whose threshold is ``<= metric``, found by scanning upward and
    BREAKING at the first threshold above it — so a non-monotone ladder stops at
    the first violation rather than skipping past it. Upgrades are immediate; a
    downgrade must clear a margin of ``hysteresis_ratio`` of the gap to the
    adjacent coarser threshold.

    The ``break`` is transcribed because the TS function has it, NOT because a
    store on disk can reach it: ``load-lod-group-node.ts`` STABLE-SORTS the
    registry children ascending by ``coverageFraction`` (with a warning) whenever
    the ladder is not strictly ascending, so the picker never sees a descending
    one. A store carrying such a ladder is refused upstream of here anyway —
    :func:`_preflight` skips it exactly as ``restamp-lod`` does, since a rewrite
    would invert it — so the ``break`` matters only if this function is called
    directly with a hand-built ladder.

    Args:
        thresholds: Per-child ``coverage_fraction`` values, coarsest→finest.
        current_idx: The index in force this frame (``default_level`` on frame
            one), which is what the hysteresis is measured against.
        metric: The scalar in the units the group's ``selector`` names.
        hysteresis_ratio: The downgrade margin as a share of the inter-level gap.

    Returns:
        The chosen index, or ``-1`` for an empty ladder.
    """
    if not thresholds:
        return -1

    natural = 0
    for i, threshold in enumerate(thresholds):
        if threshold <= metric:
            natural = i
        else:
            break

    if natural == current_idx:
        return current_idx
    if natural > current_idx:
        return natural

    current_threshold = thresholds[current_idx]
    prev_threshold = thresholds[current_idx - 1]
    margin = hysteresis_ratio * (current_threshold - prev_threshold)
    if metric < current_threshold - margin:
        return natural
    return current_idx


def frustum_planes(proj_view: np.ndarray) -> np.ndarray:
    """The six normalised frustum planes of ``P·V``.

    ``THREE.Frustum.setFromProjectionMatrix``: right/left/bottom/top/far/near as
    ``row3 ∓ rowN``, each normalised by its normal's length.

    Args:
        proj_view: The row-major ``projection @ view`` product.

    Returns:
        A ``(6, 4)`` array of ``(a, b, c, d)`` planes.
    """
    r0, r1, r2, r3 = proj_view
    raw = np.array([r3 - r0, r3 + r0, r3 + r1, r3 - r1, r3 - r2, r3 + r2])
    lengths = np.linalg.norm(raw[:, :3], axis=1)
    lengths[lengths == 0] = 1.0
    return np.asarray(raw / lengths[:, None], dtype=np.float64)


def frustum_intersects_box(planes: np.ndarray, box: Box3) -> bool:
    """Does ``box`` touch the frustum? ``THREE.Frustum.intersectsBox``.

    The positive-vertex test: for each plane take the box corner furthest along
    the plane normal; if even that corner is behind the plane, the box is wholly
    outside. Conservative near the frustum corners, exactly as the viewer is.

    Args:
        planes: The ``(6, 4)`` array from :func:`frustum_planes`.
        box: A world-space box.

    Returns:
        False only when the box is definitely outside.
    """
    lo = np.array(box.min, dtype=np.float64)
    hi = np.array(box.max, dtype=np.float64)
    normals = planes[:, :3]
    positive = np.where(normals > 0, hi, lo)
    distances = (normals * positive).sum(axis=1) + planes[:, 3]
    return not bool((distances < 0).any())


# --------------------------------------------------------------------------- #
# Reading a store.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _LadderChild:
    """One ladder level, reduced to the plain data the screening math needs."""

    threshold: Optional[float]
    element_count: Optional[int]
    bounds: Optional[_BoundsPair]
    lod_bounds: Optional[_BoundsPair]


@dataclass(frozen=True)
class _LodGroupFacts:
    """Everything read off ONE ``kind=lod`` group, before any geometry is done."""

    path: str
    selector: str
    default_level: int
    partition_bound: bool
    anchor_reason: str
    world_matrix: np.ndarray
    children: List[_LadderChild]
    #: ``lod_restamp._empty_ladder_refusal``'s message when no ladder child
    #: resolves, else ``""``.
    empty_ladder_refusal: str = ""
    #: ``lod_restamp._orphan_ladder_child_refusal``'s message when this group
    #: holds a ``coverage_fraction`` child that does not resolve as a ladder
    #: level, else ``""``. Resolved during the walk because the refusal needs
    #: the zarr group, which nothing downstream of here keeps.
    orphan_refusal: str = ""


def _bounds_of(attrs: Dict[str, Any], key: str) -> Optional[_BoundsPair]:
    """A node's ``position_bounds`` / ``lod_bounds`` as ``(min, max)``, or ``None``.

    The viewer's ``computeEntryWorldBox`` skips a child whose min/max are empty
    or mismatched in length, so a malformed pair reads as absent here too.
    """
    raw = attrs.get(key)
    if not isinstance(raw, dict):
        return None
    lo = raw.get("min")
    hi = raw.get("max")
    if not isinstance(lo, (list, tuple)) or not isinstance(hi, (list, tuple)):
        return None
    if len(lo) == 0 or len(lo) != len(hi):
        return None
    return ([float(v) for v in lo], [float(v) for v in hi])


def _lod_bounds_of(
    attrs: Dict[str, Any], position_bounds: Optional[_BoundsPair]
) -> Optional[_BoundsPair]:
    """Validated robust bounds, mirroring ``readLodBounds`` in the viewer."""
    if position_bounds is None:
        return None
    raw = attrs.get("lod_bounds")
    if not isinstance(raw, dict):
        return None
    lo = raw.get("min")
    hi = raw.get("max")
    if not isinstance(lo, (list, tuple)) or not isinstance(hi, (list, tuple)):
        return None
    if len(lo) == 0 or len(lo) != len(hi) or len(lo) != len(position_bounds[0]):
        return None

    lod_lo: List[float] = []
    lod_hi: List[float] = []
    for value_lo, value_hi, position_lo, position_hi in zip(
        lo, hi, position_bounds[0], position_bounds[1]
    ):
        if (
            not isinstance(value_lo, (int, float))
            or isinstance(value_lo, bool)
            or not isinstance(value_hi, (int, float))
            or isinstance(value_hi, bool)
        ):
            return None
        number_lo = float(value_lo)
        number_hi = float(value_hi)
        if not math.isfinite(number_lo) or not math.isfinite(number_hi):
            return None
        if number_lo > number_hi:
            return None
        if number_lo < position_lo or number_hi > position_hi:
            return None
        lod_lo.append(number_lo)
        lod_hi.append(number_hi)
    return (lod_lo, lod_hi)


def _ladder_child(child: Any, attrs: Dict[str, Any]) -> _LadderChild:
    """Reduce one ladder child, validating robust bounds against complete bounds."""
    position_bounds = _bounds_of(attrs, "position_bounds")
    return _LadderChild(
        threshold=_threshold_of(attrs),
        element_count=_count_of(child, attrs),
        bounds=position_bounds,
        lod_bounds=_lod_bounds_of(attrs, position_bounds),
    )


def _local_matrix(attrs: Dict[str, Any]) -> np.ndarray:
    """A node's own ``transform`` attr as a math 4x4 (identity when absent)."""
    raw = attrs.get("transform")
    if raw is None:
        return np.eye(4, dtype=np.float64)
    try:
        return mat4_from_column_major(raw)
    except (TypeError, ValueError):
        return np.eye(4, dtype=np.float64)


def _selector_of(attrs: Dict[str, Any]) -> str:
    """The group's stored selector, absent meaning the legacy one.

    Only an ABSENT attr defaults — ``lod_restamp._plan_lod`` gates on
    ``selector is not None``, so a present-but-falsy value (``""``, ``0``) is an
    unsupported selector there and must reach ``_preflight``'s refusal here too.
    """
    raw = attrs.get("selector")
    return "coverage" if raw is None else str(raw)


def _anchor_reason(under_partition: bool, children: Sequence[Any]) -> str:
    """Which clause of the writers' two-clause tile-binding rule fired, in words."""
    own_partition = any(
        child_attrs.get("kind") == "partition" for _, _, child_attrs in children
    )
    if under_partition and own_partition:
        return "under a >1-part kind=partition AND holds a kind=partition child"
    if under_partition:
        return "under a >1-part kind=partition ancestor"
    if own_partition:
        return "holds a kind=partition ladder child (the `overview` cap)"
    return "no partition ancestor and no partition ladder child"


def _collect_lod_groups(
    group: "zarr.Group",
    *,
    under_partition: bool,
    world: np.ndarray,
    out: List[_LodGroupFacts],
) -> None:
    """Recurse the store read-only, gathering one record per ``kind=lod`` group.

    The tile binding is threaded exactly as ``lod_restamp._walk`` threads it —
    a real (>1-part) ``kind=partition`` binds its descendants, and a ``kind=lod``
    group resolves its own binding via
    :func:`~luxar.io.lod_restamp._is_partition_bound` and passes THAT down, not
    the ancestral flag. The world matrix accumulates every node's ``transform``
    on the way down, including the ``kind=lod`` group's own: the viewer reads
    ``matrixWorld`` off the lod group's container, so its transform is inside.

    Child GROUPS only (``group_keys()``): a ``zarr.Array`` has no ``.keys()``.
    """
    attrs = dict(group.attrs)
    world = world @ _local_matrix(attrs)
    kind = attrs.get("kind")
    child_under = under_partition
    if kind == "lod":
        children = _lod_children(group)
        child_under = _is_partition_bound(under_partition, children)
        empty_refusal = _empty_ladder_refusal(group, children)
        orphan_refusal = (
            _orphan_ladder_child_refusal(group, children) if children else None
        )
        out.append(
            _LodGroupFacts(
                path=group.path or "/",
                # A falsy-but-present selector ("" or 0) must NOT be coerced to
                # the legacy one: ``_plan_lod`` gates on ``is not None`` and
                # refuses it as unsupported, so ``or`` here would predict a
                # rewrite ``restamp-lod`` never makes.
                selector=_selector_of(attrs),
                default_level=int(attrs.get("default_level") or 0),
                partition_bound=child_under,
                anchor_reason=_anchor_reason(under_partition, children),
                world_matrix=world,
                children=[
                    _ladder_child(child, child_attrs)
                    for _, child, child_attrs in children
                ],
                empty_ladder_refusal=(
                    "" if empty_refusal is None else empty_refusal.detail
                ),
                orphan_refusal=(
                    "" if orphan_refusal is None else orphan_refusal.detail
                ),
            )
        )
    elif kind == "partition":
        child_under = under_partition or len(_child_nodes(group)) > 1
    # Sorted, unlike the restamp walk: this one's output is a REPORT a human
    # diffs between runs, and `group_keys()` yields whatever order the store
    # backend happens to list (`part_1` before `part_0` on one machine, the
    # other way on the next). The traversal itself is order-insensitive.
    for name in sorted(group_keys(group)):
        _collect_lod_groups(
            group[name],
            under_partition=child_under,
            world=world,
            out=out,
        )


def _display_dims(root_attrs: Dict[str, Any]) -> List[int]:
    """The viewer's ``displayed`` axes, from the root's ``scene_dimensions``.

    ``SceneDimsManager`` takes the first three dimensions flagged
    ``display: true``; a store carrying no ``scene_dimensions`` at all falls back
    to ``[0, 1, 2]``, matching ``computeBoundsFromMetadata``'s own default. A
    store that HAS the attr but flags fewer gets a shorter list — also matching
    the viewer, which then projects the unmapped axes to 0 — and
    :func:`screen_lod_store` skips a scene with fewer than two displayed dims,
    which is where ``lod-group-registry.ts::evaluatePerFrame`` itself bails.
    """
    raw = root_attrs.get("scene_dimensions")
    if not isinstance(raw, dict):
        return [0, 1, 2]
    dims = raw.get("dimensions")
    if not isinstance(dims, list):
        return [0, 1, 2]
    displayed: List[int] = []
    for index, entry in enumerate(dims):
        if isinstance(entry, dict) and entry.get("display") is True:
            if len(displayed) < 3:
                displayed.append(index)
    return displayed


def _authored_camera_blockers(root_attrs: Dict[str, Any]) -> List[str]:
    """Authored ``viewer_config.camera`` fields that invalidate the modelled pose.

    The screen reproduces ONE pose: the ``fitCameraToBounds`` fit to the scene
    root bounds, face-on down ``-Z`` with ``up = +Y``. A scene that authors its
    own ``position``, ``target``, ``target_node`` or ``up`` opens somewhere else,
    and screening it against the fitted pose would report confident numbers for a
    framing nobody sees — so the scene is skipped and the fields are named. An
    authored ``fov`` is NOT a blocker: it moves only the projection, which
    :func:`screen_lod_store` picks up as the render FOV.
    """
    config = root_attrs.get("viewer_config")
    if not isinstance(config, dict):
        return []
    camera = config.get("camera")
    if not isinstance(camera, dict):
        return []
    return [
        key
        for key in ("position", "target", "target_node", "up")
        if camera.get(key) is not None
    ]


def _authored_render_fov(root_attrs: Dict[str, Any]) -> Optional[float]:
    """An authored ``viewer_config.camera.fov``, which the FIRST FRAME renders at.

    Only the numeric ``fov`` is honoured. A ``fov_preset`` names a value that
    lives in the viewer's TypeScript preset table, and guessing it here would be
    a second, unverified copy of that table.
    """
    config = root_attrs.get("viewer_config")
    if not isinstance(config, dict):
        return None
    camera = config.get("camera")
    if not isinstance(camera, dict):
        return None
    fov = camera.get("fov")
    if isinstance(fov, (int, float)) and not isinstance(fov, bool):
        return float(fov)
    return None


def _unmodelled_fov_source(root_attrs: Dict[str, Any]) -> str:
    """An authored FOV this module cannot resolve, named — or ``""``.

    ``viewer_config.camera.fov_preset`` and ``viewer_config.cinematic_mode:
    true`` both put the first frame on a FOV that lives in the viewer's OWN
    preset table (``config/sections/camera/data.ts``; cinematic expands to the
    35 mm entry, 63°, in ``config/cinematic-preset.ts``). Copying that table into
    Python would be a second, unverified copy of it, and GUESSING wrong is not
    harmless: at the fit FOV 47 a scene the viewer renders at 63 reads roughly
    twice the area, which is a whole halving of the derived ladder and can flip a
    win into a no-op. So such a scene is skipped and the operator is told to pass
    the FOV explicitly.

    A numeric ``camera.fov`` shadows both — the bridge's ``CINEMATIC_FOV_PAIR``
    rule is that an author-set framing wins whole — and
    :func:`_authored_render_fov` honours it, so it is not a blocker.
    """
    config = root_attrs.get("viewer_config")
    if not isinstance(config, dict):
        return ""
    camera = config.get("camera")
    camera = camera if isinstance(camera, dict) else {}
    if _authored_render_fov(root_attrs) is not None:
        return ""
    if camera.get("fov_preset") is not None:
        return f"viewer_config.camera.fov_preset={camera['fov_preset']!r}"
    if config.get("cinematic_mode") is True:
        return "viewer_config.cinematic_mode=true"
    return ""


# --------------------------------------------------------------------------- #
# The screening itself.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AspectMeasurement:
    """One group measured at one aspect ratio."""

    label: str
    aspect: float
    viewport: Tuple[int, int]
    #: The metric under the STAMPED selector, in that selector's own unit.
    today_metric: float
    #: The screen-area metric the re-derived ladder would be read against.
    area_metric: float
    today_index: int
    rederived_index: int
    today_elements: Optional[int]
    rederived_elements: Optional[int]
    off_screen: bool

    @property
    def coarser(self) -> bool:
        """Would the re-derived ladder pick a strictly coarser level here?"""
        return self.rederived_index < self.today_index


@dataclass(frozen=True)
class GroupScreening:
    """One ``kind=lod`` group's verdict, with the evidence behind it."""

    path: str
    verdict: str
    selector: str
    partition_bound: bool
    anchor_reason: str
    default_level: int
    element_counts: List[Optional[int]]
    stored_thresholds: List[Optional[float]]
    rederived_thresholds: List[float]
    measurements: List[AspectMeasurement] = field(default_factory=list)
    reason: str = ""

    @property
    def anchor(self) -> str:
        """The anchor the re-derivation used, named."""
        return (
            f"fills-screen (tile) {PARTITION_FINEST_AREA:g}"
            if self.partition_bound
            else f"whole-object {WHOLE_OBJECT_FINEST_ANCHOR:g}"
        )


@dataclass(frozen=True)
class SceneScreening:
    """Every LOD group in one store, plus why a store-wide skip happened."""

    path: str
    groups: List[GroupScreening] = field(default_factory=list)
    skipped_reason: str = ""

    @property
    def tally(self) -> Dict[str, int]:
        """Verdict counts for this scene, in :data:`VERDICT_ORDER`.

        A store-wide skip counts as one :data:`VERDICT_SKIPPED` — it prints a
        ``❔`` line like any other skip, and counting it nowhere made two
        unreadable stores read as ``0 skipped`` in the footer. Such a scene has
        no groups, so the two branches never both fire.
        """
        counts = dict.fromkeys(VERDICT_ORDER, 0)
        if self.skipped_reason:
            counts[VERDICT_SKIPPED] += 1
        for group in self.groups:
            counts[group.verdict] += 1
        return counts


@dataclass(frozen=True)
class ScreenReport:
    """The whole screening pass."""

    scenes: List[SceneScreening] = field(default_factory=list)

    @property
    def tally(self) -> Dict[str, int]:
        """Verdict counts across every scene, in :data:`VERDICT_ORDER`."""
        counts = dict.fromkeys(VERDICT_ORDER, 0)
        for scene in self.scenes:
            for verdict, count in scene.tally.items():
                counts[verdict] += count
        return counts


def _viewport_for(aspect: float, long_px: int) -> Tuple[int, int]:
    """Pixel viewport with ``long_px`` on the longer axis at this aspect."""
    if aspect >= 1.0:
        return (long_px, max(1, round(long_px / aspect)))
    return (max(1, round(long_px * aspect)), long_px)


def _near_far_for(distance: float, diagonal: float) -> Tuple[float, float]:
    """Near/far planes placed far outside the fitted scene.

    Dynamic clipping is not modelled (see the module docstring): these exist only
    so the frustum's near/far planes are well-formed, and are deliberately wide
    enough that they never gate. The metrics themselves never read them — the
    near-plane hazard is the homogeneous-``w`` test in
    :func:`project_box_ndc_rect`.
    """
    scale = max(distance, diagonal, 1.0)
    return (scale * 1e-6, scale * 1e6)


def _skip(
    facts: _LodGroupFacts,
    reason: str,
    rederived: List[float],
    *,
    verdict: str = VERDICT_SKIPPED,
) -> GroupScreening:
    """A refusal record carrying whatever was already resolved."""
    return GroupScreening(
        path=facts.path,
        verdict=verdict,
        selector=facts.selector,
        partition_bound=facts.partition_bound,
        anchor_reason=facts.anchor_reason,
        default_level=facts.default_level,
        element_counts=[child.element_count for child in facts.children],
        stored_thresholds=[child.threshold for child in facts.children],
        rederived_thresholds=rederived,
        reason=reason,
    )


def _rederived_ladder(facts: _LodGroupFacts) -> Tuple[List[float], str]:
    """What ``restamp-lod`` would derive for this group, or the reason it cannot.

    The two COUNT guards ``lod_restamp._plan_lod`` applies, for the same reasons:
    the finest child's element count must resolve and must be positive, because
    :func:`~luxar.core.group.lod.group.coverage_fractions` refuses a ladder whose
    finest level is empty. A coarser level with no recorded count is passed as
    ``0`` — the derivations consume only the LENGTH and the finest entry.

    ``_plan_lod``'s other two refusals — an orphan ladder child, a descending
    stored ladder — are applied by :func:`_preflight` through that module's own
    predicates, since neither is a function of the element counts.
    """
    counts = [child.element_count for child in facts.children]
    if not counts:
        return ([], "kind=lod group resolves no ladder children")
    finest = counts[-1]
    if finest is None:
        return ([], "the finest child records no element count")
    if finest <= 0:
        return ([], f"the finest child holds {finest} elements — a broken ladder")
    derive = (
        partitioned_coverage_fractions if facts.partition_bound else coverage_fractions
    )
    return ([float(v) for v in derive([0 if c is None else c for c in counts])], "")


def _group_local_box(
    facts: _LodGroupFacts, display_dims: Sequence[int], *, use_lod_bounds: bool
) -> Optional[Box3]:
    """Union the ladder children's nD bounds onto X/Y/Z — ``computeEntryWorldBox``.

    With ``use_lod_bounds`` a child's optional ``lod_bounds`` sizes the box and
    falls back to its raw ``position_bounds``; children with unusable bounds are
    skipped, and ``None`` comes back when none survive.
    """
    lo: Optional[List[float]] = None
    hi: Optional[List[float]] = None
    for child in facts.children:
        pair = (child.lod_bounds or child.bounds) if use_lod_bounds else child.bounds
        if pair is None:
            continue
        box = project_bounds_to_display_dims(pair[0], pair[1], display_dims)
        if lo is None or hi is None:
            lo = list(box.min)
            hi = list(box.max)
        else:
            lo = [min(a, b) for a, b in zip(lo, box.min)]
            hi = [max(a, b) for a, b in zip(hi, box.max)]
    if lo is None or hi is None:
        return None
    return Box3((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))


def _ladders_match(stored: Sequence[Optional[float]], derived: Sequence[float]) -> bool:
    """Is the stored ladder already exactly the one ``restamp-lod`` would write?"""
    if len(stored) != len(derived):
        return False
    return all(
        value is not None and abs(value - target) <= _LADDER_EQUAL_ATOL
        for value, target in zip(stored, derived)
    )


def _preflight(facts: _LodGroupFacts) -> Tuple[List[float], Optional[GroupScreening]]:
    """Resolve the re-derived ladder, or the skip record saying why there is none.

    Six hygiene states can make a legacy group undecidable, and each is reported
    rather than guessed at. Four of them are ``restamp-lod``'s OWN refusals,
    applied through its own predicates so the two can never drift: a selector
    outside the vocabulary, an orphan ladder child
    (:func:`~luxar.io.lod_restamp._orphan_ladder_child_refusal`), a DESCENDING
    stored ladder (:func:`~luxar.io.lod_restamp._descending_ladder_refusal`), and
    a ladder whose finest element count will not resolve. A group ``restamp-lod``
    refuses is a group it writes NOTHING for, so calling it a win would be a
    false positive. The remaining two are the screen's own: a child with no
    stored ``coverage_fraction`` (so TODAY's pick has no answer), and a
    ``default_level`` pointing outside its own ladder. A group already stamped
    ``screen-area`` keeps the hygiene detail but remains ``already-current``:
    ``restamp-lod`` returns on the selector before reading any of these fields.

    Args:
        facts: The group as read off the store.

    Returns:
        ``(rederived_thresholds, None)`` when the group can be screened, else
        ``([...], skip_record)``.
    """
    stored = [child.threshold for child in facts.children]

    if facts.selector not in LOD_SELECTORS:
        return (
            [],
            _skip(
                facts,
                f"selector={facts.selector!r} is outside {sorted(LOD_SELECTORS)}; "
                "restamp-lod refuses it too (`luxar gsplat migrate-format` first)",
                [],
            ),
        )

    refusal_verdict = (
        VERDICT_ALREADY_CURRENT
        if facts.selector == DERIVED_LOD_SELECTOR
        else VERDICT_SKIPPED
    )
    if facts.empty_ladder_refusal:
        return (
            [],
            _skip(facts, facts.empty_ladder_refusal, [], verdict=refusal_verdict),
        )
    if facts.orphan_refusal:
        return (
            [],
            _skip(facts, facts.orphan_refusal, [], verdict=refusal_verdict),
        )

    descending = _descending_ladder_refusal(facts.path, stored)
    if descending is not None:
        return (
            [],
            _skip(facts, descending.detail, [], verdict=refusal_verdict),
        )

    rederived, why_not = _rederived_ladder(facts)
    if why_not:
        return (
            rederived,
            _skip(facts, why_not, rederived, verdict=refusal_verdict),
        )
    if any(value is None for value in stored):
        return (
            rederived,
            _skip(
                facts,
                "at least one child carries no 'coverage_fraction', so what the "
                "viewer picks TODAY cannot be determined",
                rederived,
                verdict=refusal_verdict,
            ),
        )
    if not 0 <= facts.default_level < len(stored):
        return (
            rederived,
            _skip(
                facts,
                f"default_level={facts.default_level} is outside the "
                f"{len(stored)}-level ladder",
                rederived,
                verdict=refusal_verdict,
            ),
        )
    return (rederived, None)


@dataclass(frozen=True)
class _Geometry:
    """One group's boxes, world matrix and ladders, fixed across the aspect sweep."""

    raw_world: Box3
    raw_local: Box3
    metric_local: Box3
    world_matrix: np.ndarray
    has_lod_bounds: bool
    selector: str
    default_level: int
    stored_thresholds: List[float]
    rederived: List[float]
    counts: List[Optional[int]]


def _measure(
    geometry: _Geometry,
    root_box: Box3,
    label: str,
    aspect: float,
    *,
    viewport_long_px: int,
    fit_fov: float,
    render_fov: float,
) -> AspectMeasurement:
    """Score one group at ONE aspect: fit the camera, take both metrics, pick.

    Args:
        geometry: The group's fixed boxes and ladders.
        root_box: The scene box the camera is fitted to.
        label: How this aspect is named in the report.
        aspect: Viewport width / height.
        viewport_long_px: Pixels on the long viewport axis.
        fit_fov: FOV the fitted DISTANCE uses.
        render_fov: FOV the first frame's projection uses.

    Returns:
        The measurement, with ``off_screen`` set when the frustum gate fired.
    """
    width, height = _viewport_for(aspect, viewport_long_px)
    target = root_box.center
    distance = calculate_camera_distance(
        root_box, fit_fov, aspect, DEFAULT_FIT_RATIO, target
    )
    near, far = _near_far_for(distance, root_box.diagonal)
    proj_view = perspective_matrix(render_fov, aspect, near, far) @ view_matrix(
        (target[0], target[1], target[2] + distance)
    )

    off_screen = not frustum_intersects_box(
        frustum_planes(proj_view), geometry.raw_world
    )
    if off_screen:
        # The viewer never takes a metric here: it holds `coarsestReadyIndex`
        # (`lod-group-registry.ts`), the first child that is READY — and on frame
        # one the only ready child is the eagerly-committed `default_level`
        # (`load-lod-group-node.ts` defers every other level behind an
        # `ensureLoaded` thunk). So it is `default_level`, not index 0. Both
        # ladders therefore agree, and the metrics are reported as 0 rather than
        # as a number nobody uses.
        today_metric = area_metric = 0.0
        today_index = rederived_index = geometry.default_level
    else:
        # Like the viewer, the metrics project the group's LOCAL box through
        # `P·V·matrixWorld` — the box's own 8 corners as oriented on screen. The
        # world AABB (the frustum gate's box) inflates a rotated group.
        box_to_clip = proj_view @ geometry.world_matrix
        area_metric = project_box_area_fraction(
            geometry.metric_local, box_to_clip, near
        )
        if geometry.has_lod_bounds:
            # The thin-rect ramp is not monotone under containment, so robust
            # bounds may only REDUCE the raw-bounds metric.
            area_metric = min(
                area_metric,
                project_box_area_fraction(geometry.raw_local, box_to_clip, near),
            )
        today_metric = (
            area_metric
            if geometry.selector == DERIVED_LOD_SELECTOR
            else legacy_coverage_metric(
                geometry.metric_local, box_to_clip, width, height
            )
        )
        today_index = pick_child_with_hysteresis(
            geometry.stored_thresholds, geometry.default_level, today_metric
        )
        rederived_index = pick_child_with_hysteresis(
            geometry.rederived, geometry.default_level, area_metric
        )

    return AspectMeasurement(
        label=label,
        aspect=aspect,
        viewport=(width, height),
        today_metric=today_metric,
        area_metric=area_metric,
        today_index=today_index,
        rederived_index=rederived_index,
        today_elements=geometry.counts[today_index],
        rederived_elements=geometry.counts[rederived_index],
        off_screen=off_screen,
    )


def _verdict_of(
    facts: _LodGroupFacts,
    measurements: Sequence[AspectMeasurement],
) -> str:
    """Bucket a group from its aspect sweep, most-specific test first.

    ``off-screen`` and ``already-current`` come first because in both the
    coarser/not-coarser comparison is vacuous — no metric was taken, or
    ``restamp-lod`` would write nothing. Only then is the sweep read: coarser
    everywhere is a ``win``, nowhere a ``no-op``, and anything in between is
    ``fragile`` rather than rounded into either.

    ``already-current`` is decided on the SELECTOR ALONE, exactly as
    ``lod_restamp._plan_lod`` decides it — that function returns before it has
    read a single threshold, so a ``screen-area`` group whose ladder is not the
    one ``restamp-lod`` would derive is still left untouched by a real run. The
    stored-vs-derived diff stays in the report as printed evidence (see
    :func:`_print_group`), where it names a store worth looking at without
    claiming a rewrite that will not happen.

    A ``win`` additionally requires that the sweep is non-empty: ``all([])`` is
    True, so an empty ``aspects`` list would otherwise make every group a win
    with no evidence at all.
    """
    if measurements and all(m.off_screen for m in measurements):
        return VERDICT_OFF_SCREEN
    if facts.selector == DERIVED_LOD_SELECTOR:
        return VERDICT_ALREADY_CURRENT
    if measurements and all(m.coarser for m in measurements):
        return VERDICT_WIN
    if not any(m.coarser for m in measurements):
        return VERDICT_NO_OP
    return VERDICT_FRAGILE


def _screen_group(
    facts: _LodGroupFacts,
    *,
    root_box: Box3,
    display_dims: Sequence[int],
    aspects: Sequence[Tuple[str, float]],
    viewport_long_px: int,
    fit_fov: float,
    render_fov: float,
) -> GroupScreening:
    """Measure one group at every requested aspect and bucket the result."""
    stored = [child.threshold for child in facts.children]
    counts = [child.element_count for child in facts.children]

    rederived, refusal = _preflight(facts)
    if refusal is not None:
        return refusal

    raw_local = _group_local_box(facts, display_dims, use_lod_bounds=False)
    if raw_local is None:
        return _skip(
            facts,
            "no ladder child carries usable 'position_bounds', so the group has "
            "no world box to project",
            rederived,
        )
    has_lod_bounds = any(child.lod_bounds is not None for child in facts.children)
    metric_local = raw_local
    if has_lod_bounds:
        # The `use_lod_bounds` pass reads `child.lod_bounds or child.bounds`, so
        # it accepts a SUPERSET of the children the raw pass accepted: `raw_local`
        # being non-None makes this non-None too. The `or raw_local` fallback that
        # used to sit here could never fire.
        metric_local = cast(
            Box3, _group_local_box(facts, display_dims, use_lod_bounds=True)
        )

    geometry = _Geometry(
        raw_world=transform_box(raw_local, facts.world_matrix),
        raw_local=raw_local,
        metric_local=metric_local,
        world_matrix=facts.world_matrix,
        has_lod_bounds=has_lod_bounds,
        selector=facts.selector,
        default_level=facts.default_level,
        # `_preflight` has already refused any ladder carrying a missing
        # threshold, so every entry is a real number; the `is not None` FILTER
        # that used to live here could only ever desynchronise this ladder from
        # `counts`, which is indexed by the same child position.
        stored_thresholds=[float(cast(float, v)) for v in stored],
        rederived=rederived,
        counts=counts,
    )
    measurements = [
        _measure(
            geometry,
            root_box,
            label,
            aspect,
            viewport_long_px=viewport_long_px,
            fit_fov=fit_fov,
            render_fov=render_fov,
        )
        for label, aspect in aspects
    ]

    return GroupScreening(
        path=facts.path,
        verdict=_verdict_of(facts, measurements),
        selector=facts.selector,
        partition_bound=facts.partition_bound,
        anchor_reason=facts.anchor_reason,
        default_level=facts.default_level,
        element_counts=counts,
        stored_thresholds=stored,
        rederived_thresholds=rederived,
        measurements=measurements,
    )


def _require_aspects(aspects: Sequence[Tuple[str, float]]) -> None:
    """Refuse an empty aspect sweep — a verdict with no evidence behind it.

    Every verdict but ``skipped`` is a statement about a measurement, and
    ``all([])`` is True: an empty sweep would make :data:`VERDICT_WIN`'s
    "coarser at every tested aspect" vacuously true for every group in the
    store. :func:`_verdict_of` guards the clause as well, but a caller that
    passed nothing to measure at wants an error, not a report of no-ops.

    Args:
        aspects: The ``(label, width/height)`` pairs to measure at.

    Raises:
        ValueError: ``aspects`` is empty.
    """
    if not aspects:
        raise ValueError(
            "the LOD screen needs at least one aspect ratio to measure at; got "
            "an empty sequence (a verdict over no measurements is evidence-free)"
        )


def screen_lod_store(
    path: "str | Path",
    *,
    aspects: Sequence[Tuple[str, float]] = DEFAULT_ASPECTS,
    viewport_long_px: int = DEFAULT_VIEWPORT_LONG_PX,
    fit_fov: float = DEFAULT_FIT_FOV,
    render_fov: Optional[float] = None,
) -> SceneScreening:
    """Screen every ``kind=lod`` group in ONE built scene. Read-only.

    The scene is framed exactly as the viewer frames it on load — the root's
    ``position_bounds`` projected onto the displayed axes, fitted face-on — and
    every LOD group is then scored twice per aspect: once against the ladder and
    selector on disk, once against the ladder ``restamp-lod`` would derive, read
    as a screen-area fraction.

    Args:
        path: An uncompressed ``.luxar.zarr`` scene directory.
        aspects: ``(label, width/height)`` pairs to measure at.
        viewport_long_px: Pixels on the viewport's long axis. Affects the legacy
            diagonal metric only; the area metric is size-independent.
        fit_fov: VERTICAL FOV the fitted DISTANCE is computed at. Always the
            viewer default (47) unless you are modelling something unusual — the
            cinematic preset changes the FOV only after the fit.
        render_fov: Fallback VERTICAL FOV for the first frame's projection when
            the scene does not author ``viewer_config.camera.fov``. Defaults to
            ``fit_fov``.

    Returns:
        A :class:`SceneScreening`. A store-wide refusal (not a scene, no root
        bounds, an authored camera pose) comes back with an empty ``groups`` and
        a populated :attr:`~SceneScreening.skipped_reason` — never an exception,
        because a screening verdict must not be able to fail a build.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: ``path`` is not a directory, or ``aspects`` is empty.
    """
    _require_aspects(aspects)
    store_path = Path(path)
    if not store_path.exists():
        raise FileNotFoundError(f"no such store: {store_path}")
    if not store_path.is_dir():
        raise ValueError(
            f"the LOD screen reads an uncompressed .zarr DIRECTORY; got "
            f"{store_path} (unpack a .zip store first)"
        )

    root = open_group(store_path, mode="r")
    try:
        root_attrs = dict(root.attrs)
        if root_attrs.get("type") != "scene":
            return SceneScreening(
                path=str(store_path),
                skipped_reason=(
                    "not a compiled scene (no root type='scene'), so there is no "
                    "opening framing to screen against"
                ),
            )

        blockers = _authored_camera_blockers(root_attrs)
        if blockers:
            return SceneScreening(
                path=str(store_path),
                skipped_reason=(
                    "the scene authors viewer_config.camera."
                    + "/".join(blockers)
                    + ", so it does not open at the fitted face-on pose this "
                    "screen models"
                ),
            )

        if render_fov is None:
            unmodelled_fov = _unmodelled_fov_source(root_attrs)
            if unmodelled_fov:
                return SceneScreening(
                    path=str(store_path),
                    skipped_reason=(
                        f"the scene authors {unmodelled_fov}, whose FOV lives in "
                        "the viewer's own preset table — screening it at the fit "
                        f"FOV {fit_fov:g} would misreport every occupancy. Pass "
                        "--screen-render-fov (63 for the cinematic/35mm preset)"
                    ),
                )

        bounds = _bounds_of(root_attrs, "position_bounds")
        if bounds is None:
            return SceneScreening(
                path=str(store_path),
                skipped_reason="the scene root carries no usable position_bounds",
            )

        display_dims = _display_dims(root_attrs)
        if len(display_dims) < 2:
            # `lod-group-registry.ts::evaluatePerFrame` bails at
            # `displayDims.length < 2` BEFORE any group is evaluated, so the
            # selector never runs on such a scene and a verdict for it would
            # describe something the viewer does not do.
            return SceneScreening(
                path=str(store_path),
                skipped_reason=(
                    f"scene_dimensions flags {len(display_dims)} dimension(s) "
                    "display=true; the viewer's LOD selector bails below 2, so "
                    "it never evaluates a ladder in this scene"
                ),
            )

        root_box = project_bounds_to_display_dims(bounds[0], bounds[1], display_dims)
        if root_box.diagonal <= 0:
            return SceneScreening(
                path=str(store_path),
                skipped_reason=(
                    "the projected scene bbox has zero diagonal, so "
                    "fitCameraToBounds bails out before framing anything"
                ),
            )

        facts: List[_LodGroupFacts] = []
        _collect_lod_groups(
            root,
            under_partition=False,
            world=np.eye(4, dtype=np.float64),
            out=facts,
        )

        effective_render_fov = _authored_render_fov(root_attrs) or render_fov or fit_fov
        return SceneScreening(
            path=str(store_path),
            groups=[
                _screen_group(
                    entry,
                    root_box=root_box,
                    display_dims=display_dims,
                    aspects=aspects,
                    viewport_long_px=viewport_long_px,
                    fit_fov=fit_fov,
                    render_fov=effective_render_fov,
                )
                for entry in facts
            ],
        )
    finally:
        close(root)


def screen_stores(
    paths: Sequence["str | Path"],
    *,
    aspects: Sequence[Tuple[str, float]] = DEFAULT_ASPECTS,
    viewport_long_px: int = DEFAULT_VIEWPORT_LONG_PX,
    fit_fov: float = DEFAULT_FIT_FOV,
    render_fov: Optional[float] = None,
) -> ScreenReport:
    """:func:`screen_lod_store` over several stores, collected into one report.

    A store that cannot be OPENED is recorded as a scene-level skip rather than
    raised: one broken demo must not hide the screening of the other forty.

    Args:
        paths: Scene directories to screen.
        aspects: See :func:`screen_lod_store`.
        viewport_long_px: See :func:`screen_lod_store`.
        fit_fov: See :func:`screen_lod_store`.
        render_fov: See :func:`screen_lod_store`.

    Returns:
        The combined :class:`ScreenReport`.

    Raises:
        ValueError: ``aspects`` is empty. Checked up front, OUTSIDE the
            per-store guard below, so a caller error is raised rather than
            recorded once per store as if the stores were at fault.
    """
    _require_aspects(aspects)
    scenes: List[SceneScreening] = []
    for path in paths:
        try:
            scenes.append(
                screen_lod_store(
                    path,
                    aspects=aspects,
                    viewport_long_px=viewport_long_px,
                    fit_fov=fit_fov,
                    render_fov=render_fov,
                )
            )
        except Exception as error:  # noqa: BLE001 - one bad store must not stop the rest
            scenes.append(
                SceneScreening(path=str(path), skipped_reason=f"cannot read ({error})")
            )
    return ScreenReport(scenes=scenes)


# --------------------------------------------------------------------------- #
# Rendering the report.
# --------------------------------------------------------------------------- #

#: One glyph per verdict. Nothing here is ``❌``: no verdict is an error — this
#: pass only reports.
_VERDICT_ICON: Dict[str, str] = {
    VERDICT_WIN: "🏆",
    VERDICT_FRAGILE: "⚠️ ",
    VERDICT_NO_OP: "· ",
    VERDICT_OFF_SCREEN: "🚫",
    VERDICT_ALREADY_CURRENT: "✅",
    VERDICT_SKIPPED: "❔",
}


def _fmt_counts(values: Sequence[Optional[int]]) -> str:
    """Element counts as compact text; an unresolved one renders as ``?``."""
    return "[" + ", ".join("?" if v is None else f"{v:,}" for v in values) + "]"


def _fmt_ladder(values: Sequence[Optional[float]]) -> str:
    """A ladder as compact text; a missing threshold renders as ``?``."""
    return "[" + ", ".join("?" if v is None else f"{v:g}" for v in values) + "]"


def _fmt_metric(value: float) -> str:
    """A metric, keeping ``inf`` legible."""
    return "inf" if math.isinf(value) else f"{value:.4g}"


def print_screen_report(
    report: ScreenReport, *, verdicts: Optional[Sequence[str]] = None
) -> None:
    """Print the screening pass with arbol sections.

    Args:
        report: The report from :func:`screen_stores`.
        verdicts: Show only groups in these buckets (all of them when omitted).
            A scene with nothing left to show is dropped from the output; its
            groups still count in the tally, which is always over everything.
    """
    wanted = set(verdicts) if verdicts else None
    for scene in report.scenes:
        name = Path(scene.path).name
        if scene.skipped_reason:
            if wanted is None or VERDICT_SKIPPED in wanted:
                aprint(f"❔ {name}: {scene.skipped_reason}")
            continue
        shown = [g for g in scene.groups if wanted is None or g.verdict in wanted]
        if not shown:
            continue
        with asection(name):
            for group in shown:
                _print_group(group)

    tally = report.tally
    aprint(
        "LOD screen: "
        + ", ".join(f"{tally[verdict]} {verdict}" for verdict in VERDICT_ORDER)
    )


def _print_group(group: GroupScreening) -> None:
    """Print one group's verdict, ladders and per-aspect measurements."""
    icon = _VERDICT_ICON.get(group.verdict, "· ")
    with asection(f"{icon} {group.path}: {group.verdict}"):
        aprint(
            f"selector={group.selector!r}, anchor {group.anchor} "
            f"({group.anchor_reason}), default_level={group.default_level}"
        )
        aprint(f"elements {_fmt_counts(group.element_counts)}")
        aprint(
            f"stored   {_fmt_ladder(group.stored_thresholds)}  →  "
            f"re-derived {_fmt_ladder(list(group.rederived_thresholds))}"
        )
        if (
            group.verdict == VERDICT_ALREADY_CURRENT
            and group.rederived_thresholds
            and not _ladders_match(group.stored_thresholds, group.rederived_thresholds)
        ):
            # Evidence, NOT a verdict: `restamp-lod` skips a `screen-area` group
            # on the selector alone and never reads its ladder, so this store is
            # worth a look but a rewrite would not touch it.
            aprint(
                "note: the stored ladder is NOT the one restamp-lod would "
                "derive, but the group is already on "
                f"selector={DERIVED_LOD_SELECTOR!r}, so restamp-lod skips it "
                "and writes nothing"
            )
        if group.reason:
            label = "detail" if group.verdict == VERDICT_ALREADY_CURRENT else "skipped"
            aprint(f"{label}: {group.reason}")
        for m in group.measurements:
            if m.off_screen:
                aprint(
                    f"{m.label:>5} {m.viewport[0]}x{m.viewport[1]}: off-screen "
                    "(no metric taken; held at the coarsest ready level)"
                )
                continue
            unit = "area" if group.selector == DERIVED_LOD_SELECTOR else "coverage"
            arrow = (
                "coarser"
                if m.coarser
                else "same"
                if m.rederived_index == m.today_index
                else "FINER"
            )
            aprint(
                f"{m.label:>5} {m.viewport[0]}x{m.viewport[1]}: "
                f"today {unit}={_fmt_metric(m.today_metric)} → L{m.today_index} "
                f"({_fmt_counts([m.today_elements])[1:-1]}) | "
                f"re-derived area={_fmt_metric(m.area_metric)} → "
                f"L{m.rederived_index} ({_fmt_counts([m.rederived_elements])[1:-1]}) "
                f"[{arrow}]"
            )
