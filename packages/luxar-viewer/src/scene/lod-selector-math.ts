/**
 * Selector math for the per-frame LOD-group selector.
 *
 * The camera-geometry half of `lod-group-registry.ts`, extracted so the
 * registry file holds the policy/state machine and this module holds the
 * (individually unit-testable) math:
 *
 * - {@link computeEntryWorldBox} — fold a group's per-child nD raw or robust
 *   bounds into one world-space box via ``displayDims`` + ``matrixWorld``.
 * - {@link projectBoxDiagonalPx} — project a box through its box-to-clip
 *   matrix to a screen-space pixel diagonal (with near-plane saturation). The legacy
 *   ``selector: 'coverage'`` metric.
 * - {@link projectBoxAreaFraction} — project the box's inscribed ellipsoid,
 *   sized by the ellipsoid's view-axis depth, to the viewport AREA its
 *   screen-space ellipse covers, in rect units (same near-plane saturation). The ``selector: 'screen-area'`` metric.
 * - {@link pickChildWithHysteresis} — pick the level for whichever metric the
 *   group's ``selector`` names, with asymmetric downgrade hysteresis.
 *
 * The registry re-exports `projectBoxAreaFraction` / `projectBoxDiagonalPx` /
 * `pickChildWithHysteresis` so existing importers (the selector unit tests)
 * are unchanged.
 *
 * @module scene/lod-selector-math
 */

import * as THREE from 'three';
import type { ViewContext } from './view-context';

import { type BoundingBox, transformBoundingBox } from './scene-manager/clipping/bounds-math';

/** Asymmetric hysteresis on the "downgrade to coarser" direction. */
const HYSTERESIS_RATIO = 0.1;

/**
 * Module-scope scratch for standalone callers' projection × view product.
 * Calls consume it synchronously before the next projection overwrites it.
 */
const PROJ_VIEW_SCRATCH = new THREE.Matrix4();

/**
 * Saturation epsilon for ``projectBoxDiagonalPx``'s near-plane guard. When any
 * bbox corner's homogeneous ``w`` (clip-space, ≈ view-space depth in front of
 * the camera) falls to/below this, the perspective divide is already producing
 * exploding/flipped NDC, so the diagonal is meaningless. We trip *before* ``w``
 * crosses zero (hence 1e-6, not ``transformBoundingBox``'s singular-point
 * ``1e-12``) to eliminate the unstable near-plane regime, not just the literal
 * singularity. With identity matrices ``w == 1 ≫ 1e-6``, so this never fires in
 * the identity-camera unit tests.
 */
const W_EPSILON = 1e-6;

/**
 * Project a :type:`BoundingBox` through its box-to-clip matrix and
 * return the diagonal of the screen-space AABB in pixels.
 * The box and matrix must use the same coordinate frame: the registry passes
 * group-local bounds with projection × view × group-world, while callers that
 * omit the matrix pass world bounds and use projection × matrixWorldInverse.
 *
 * Treats the bbox's 8 corners independently (works for both
 * perspective and orthographic projection without a closed-form
 * radius). NDC → pixels assumes the viewport size matches the
 * renderer canvas.
 *
 * **Near-plane saturation.** Projects with an explicit homogeneous ``w`` (the
 * supplied box-to-clip matrix (or projection × matrixWorldInverse), not THREE's
 * ``Vector3.project`` which divides by ``w`` unguarded). If any corner has
 * ``w <= W_EPSILON`` — i.e. the camera is inside or straddling the box — the
 * group fills the screen, so we return ``+Infinity`` to saturate the selector
 * to its finest level (``pickChildWithHysteresis`` then picks the top index;
 * the value is never fed to finite arithmetic, so no NaN). This is the inverse
 * of the old behaviour, where a corner crossing behind the near plane
 * *collapsed* the diagonal and wrongly dropped to a coarse level on close
 * approach. Orthographic cameras keep ``w == 1`` and so never saturate.
 *
 * Exported for unit testing.
 */
export function projectBoxDiagonalPx(
  box: BoundingBox,
  camera: THREE.Camera,
  viewport: { width: number; height: number },
  boxToClip?: THREE.Matrix4
): number {
  const rect = projectBoxNdcRect(box, camera, boxToClip);
  if (rect === null) return Number.POSITIVE_INFINITY;
  const widthPx = rect.halfW * viewport.width;
  const heightPx = rect.halfH * viewport.height;
  return Math.hypot(widthPx, heightPx);
}

/**
 * Half-extent (as a fraction of its viewport axis) below which a projected
 * rect's thin dimension counts as DEGENERATE for {@link projectBoxAreaFraction}:
 * effectively lower-dimensional content (an axis-aligned straight polyline, a
 * planar dataset viewed edge-on, 1D/2D bounds on a mapped axis) whose
 * area-product would read ~0 no matter how much of the screen it spans —
 * permanently pinning it to the coarsest level. ``1e-3`` ≈ one pixel on a
 * ~1080p viewport: anything rendering thinner than a pixel genuinely reads as
 * a line, and for a line the faithful "portion of the screen occupied" is its
 * linear span, not the vanishing area. The fallback is a continuous RAMP over
 * ``[0, this]`` (see the function doc), not a cliff.
 *
 * Exported so tests pin the ramp against the real constant (and so the
 * ``@link`` references above resolve in TypeDoc).
 */
export const DEGENERATE_RECT_HALF_EXTENT = 1e-3;

/**
 * Project a :type:`BoundingBox` through its box-to-clip matrix and return the
 * fraction of the viewport AREA it **visibly** covers — the metric for
 * ``selector: 'screen-area'``.
 * As with {@link projectBoxDiagonalPx}, bounds are group-local when the
 * registry supplies projection × view × group-world, or world-space when the
 * matrix is omitted and projection × matrixWorldInverse is used; the inscribed
 * ellipsoid below is taken in that same frame.
 *
 * **Orientation-stable: the box's inscribed ellipsoid, not its corners.** The
 * screen rect of a box's 8 corners grows by up to ~1.7x between a face-on and a
 * corner-on view of the same box at the same distance, so a rect metric walks a
 * lod ladder up and down while the camera merely orbits. The box is therefore
 * measured through its INSCRIBED ellipsoid (semi-axes = the box half-extents),
 * projected exactly as a dual quadric onto the image-plane conic, and then
 * SIZED by the ellipsoid's view-axis half-chord (see
 * ``projectNearDepthEllipse``): in perspective the bare ellipse outline
 * sits near the box's middle plane while its screen coverage is set by the near
 * face, so a thick box close to the camera would otherwise read too small. The
 * metric is ``sqrt(det S)`` of the sized ellipse's NDC shape matrix ``S`` (the
 * product of its semi-axes). For a flat box face-on, or under an orthographic
 * camera, it is exactly the legacy rect product ``halfW × halfH`` (the ellipse
 * is inscribed in the face's rect); a thick box face-on in perspective reads
 * its near-face rect exactly on the view axis, so stored thresholds keep their
 * meaning (a screen-filling face still reads 1.0). A tilted flat card gets no
 * depth correction. A cube at a fixed distance has the same half-chord from
 * every direction, so its orbit metric stays constant; an elongated box still
 * changes size when viewed from different directions.
 *
 * **Clipped to the viewport.** The ellipse's area is scaled, per axis, by the
 * visible fraction of its screen-space AABB, so the metric reads the portion of
 * the screen ACTUALLY occupied: a node whose ellipse extends far off-screen but
 * clips only a corner reads that small visible fraction (and picks a coarse
 * level) — which matters while panning across partition tiles. The metric
 * tops out at exactly ``1.0`` (full coverage); the natural pick uses
 * ``threshold <= metric``, so the fills-screen partition threshold (1.0) is
 * satisfied the moment coverage is complete. The scaling is exact for an
 * axis-aligned ellipse (every face-on view) and an approximation for a tilted
 * one. In NDC each axis spans 2, so the value is viewport-size independent by
 * construction.
 *
 * **Degenerate (lower-dimensional) CONTENT ramps to its LINEAR span.** When the
 * ellipse's RAW (pre-clip) minor semi-axis is below
 * {@link DEGENERATE_RECT_HALF_EXTENT} — sub-pixel thin content: a straight
 * polyline, an edge-on plane — the area reads ~0 regardless of how much screen
 * the content spans, which would pin it to the coarsest level forever. The
 * metric is therefore ``max(area, clippedSpan × (1 − rawThin /
 * DEGENERATE_RECT_HALF_EXTENT))``, where ``clippedSpan`` is the larger clipped
 * half-extent of the ellipse's AABB: at zero thickness it reads the full
 * CLIPPED linear span (a full-width line = 1.0), decays CONTINUOUSLY to the
 * area as the thickness reaches the sub-pixel floor, and is exactly the area
 * everywhere above it. A point still reads ~0 → coarsest. The ramp is gated on
 * the RAW thinness so a wide 2D node whose CLIPPED sliver happens to be thin
 * honestly reads its tiny visible area.
 *
 * **Fully off-screen reads exactly 0**, including a box entirely behind the
 * camera: a clipped interval that is INVERTED (no viewport overlap on that
 * axis) zeroes the metric before the degenerate ramp can see it.
 *
 * **+Infinity when the camera plane cuts the box** (the camera is inside it,
 * or the node straddles the eye plane — including a corner poking behind the
 * eye while the inscribed ellipsoid stays in front): there is no finite
 * near-corner depth to size the ellipse at, so the metric saturates to the
 * finest level, like the corner-rect metrics. An ORTHOGRAPHIC projection
 * never degenerates (``w`` stays 1), so no saturation applies — a camera inside
 * a large node reads full coverage naturally because its ellipse spans the
 * viewport. See the v3.4 spec's normative metric rules.
 *
 * Exported for unit testing.
 */
export function projectBoxAreaFraction(
  box: BoundingBox,
  camera: THREE.Camera,
  boxToClip?: THREE.Matrix4
): number {
  // The box-to-clip matrix when the registry supplies one (group-local
  // bounds), else the shared projection × view (world-space bounds).
  const m =
    boxToClip ??
    PROJ_VIEW_SCRATCH.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const ellipse = projectNearDepthEllipse(box, m.elements);
  if (ellipse === ELLIPSE_STRADDLES) return Number.POSITIVE_INFINITY;
  if (ellipse === ELLIPSE_BEHIND) return 0;
  const { cx, cy, sxx, sxy, syy } = ellipse;
  const ex = Math.sqrt(sxx);
  const ey = Math.sqrt(syy);
  // Visible (clipped to NDC [-1, 1]) extent of the ellipse's AABB. Keep the
  // SIGNED overlaps: negative means no overlap on that axis — off-screen.
  const overlapW = Math.min(cx + ex, 1) - Math.max(cx - ex, -1);
  const overlapH = Math.min(cy + ey, 1) - Math.max(cy - ey, -1);
  if (overlapW < 0 || overlapH < 0) return 0;
  const det = sxx * syy - sxy * sxy;
  // The AABB clip undercounts a tilted ellipse even when it contains the
  // entire viewport. A convex ellipse covers the square iff all four corners
  // are inside its conic.
  if (viewportInsideEllipse(ellipse, det)) return 1;
  const visibleW = ex > 0 ? Math.min(1, overlapW / (2 * ex)) : 1;
  const visibleH = ey > 0 ? Math.min(1, overlapH / (2 * ey)) : 1;
  const area = Math.sqrt(Math.max(0, det)) * visibleW * visibleH;
  // Continuous degenerate ramp, gated on the RAW minor semi-axis.
  const halfTrace = 0.5 * (sxx + syy);
  const halfDiff = 0.5 * (sxx - syy);
  const minorSq = halfTrace - Math.sqrt(halfDiff * halfDiff + sxy * sxy);
  const rawThin = Math.sqrt(Math.max(0, minorSq));
  const span = Math.max(overlapW, overlapH) * 0.5;
  const degenerate = span * Math.max(0, 1 - rawThin / DEGENERATE_RECT_HALF_EXTENT);
  return Math.max(area, degenerate);
}

/** A convex ellipse contains the viewport iff it contains all four corners. */
function viewportInsideEllipse(ellipse: typeof ELLIPSE_SCRATCH, det: number): boolean {
  if (!(det > 0) || !Number.isFinite(det)) return false;
  const { cx, cy, sxx, sxy, syy } = ellipse;
  for (let x = -1; x <= 1; x += 2) {
    for (let y = -1; y <= 1; y += 2) {
      const dx = x - cx;
      const dy = y - cy;
      if (!(syy * dx * dx - 2 * sxy * dx * dy + sxx * dy * dy <= det)) return false;
    }
  }
  return true;
}

/** {@link projectNearDepthEllipse}: the camera plane cuts the box (or its ellipsoid). */
const ELLIPSE_STRADDLES = 1;
/** {@link projectNearDepthEllipse}: the box lies wholly behind the camera. */
const ELLIPSE_BEHIND = 2;

/**
 * The box's inscribed-ellipsoid image ({@link projectInscribedEllipse}) SIZED
 * by its view-axis half-chord. The ellipsoid's outline lies well
 * behind the box's near face, while what the box covers on screen is set by
 * that near face, so in perspective the bare ellipse of a thick box close to the
 * camera reads less than the box covers (the hosted zebrafish endoderm read
 * 25% below its corner rect and dropped a whole level at its opening view).
 * The ellipse is therefore resized to the depth of its near view-axis point.
 *
 * ``w`` is clip-space ``w``, i.e. view depth along the camera axis, in the
 * frame of ``e``. The view direction is the cross product of the clip x/y
 * rows in the box frame, so an anisotropic group scale does not change the
 * chord of an otherwise identical world-space box. The projected ellipsoid
 * does NOT sit at the centre depth: on the axis, an ellipsoid with semi-axis
 * ``a`` across and ``c`` along the view at depth ``D`` has silhouette half-width
 * ``f·a / sqrt(D² − c²)``, i.e. it
 * already reads at the geometric-mean depth ``sqrt((w_c − c)(w_c + c))``, where
 * ``w_c`` is centre depth and ``c`` is the ellipsoid's view-axis half-chord.
 * Resizing it to ``w_c − c`` scales ``S`` by ``(w_c + c)/(w_c − c)`` — which
 * makes a box seen face-on on the axis read EXACTLY its near-face rect, at any
 * thickness. (Scaling by ``w_c / (w_c − c)`` instead, the naive "centre to
 * near" ratio, double-counts that depth: it overshoots the rect by 12% for a
 * box whose centre is three half-depths away, and made orbit LOD flips worse
 * than the corner rect.) The ellipse CENTRE is
 * not moved, only its size; everything downstream (the viewport clip overlap,
 * the full-coverage test, the degenerate ramp) reads the sized ellipse. An
 * orthographic projection has constant ``w``, so the factor is exactly 1. A
 * flat box tilted in perspective has zero view-axis chord and gets no scaling.
 *
 * Returns {@link ELLIPSE_BEHIND} when every corner is behind the eye and
 * {@link ELLIPSE_STRADDLES} when the nearest corner is at/behind
 * {@link W_EPSILON} — the eye plane cuts the box, the same saturation rule as
 * the corner-rect metrics — even if it misses the inscribed ellipsoid.
 */
function projectNearDepthEllipse(
  box: BoundingBox,
  e: ArrayLike<number>
): typeof ELLIPSE_SCRATCH | typeof ELLIPSE_STRADDLES | typeof ELLIPSE_BEHIND {
  let wNear = Infinity;
  let wFar = -Infinity;
  for (let i = 0; i < 8; i++) {
    const x = i & 1 ? box.max.x : box.min.x;
    const y = i & 2 ? box.max.y : box.min.y;
    const z = i & 4 ? box.max.z : box.min.z;
    const w = e[3] * x + e[7] * y + e[11] * z + e[15];
    wNear = Math.min(wNear, w);
    wFar = Math.max(wFar, w);
  }
  if (!(wFar > 0)) return ELLIPSE_BEHIND;
  if (!(wNear > W_EPSILON)) return ELLIPSE_STRADDLES;
  const ellipse = projectInscribedEllipse(box, e);
  if (ellipse === ELLIPSE_STRADDLES || ellipse === ELLIPSE_BEHIND) return ellipse;
  const wCenter = 0.5 * (wNear + wFar);
  const chord = ellipsoidViewChord(box, e);
  const areaScale = (wCenter + chord) / (wCenter - chord);
  ellipse.sxx *= areaScale;
  ellipse.sxy *= areaScale;
  ellipse.syy *= areaScale;
  return ellipse;
}

/** Half-chord along the camera axis, measured in clip-space w units. */
function ellipsoidViewChord(box: BoundingBox, e: ArrayLike<number>): number {
  const hx = 0.5 * (box.max.x - box.min.x);
  const hy = 0.5 * (box.max.y - box.min.y);
  const hz = 0.5 * (box.max.z - box.min.z);
  const h2 = HALF_EXTENTS_SQ;
  h2[0] = hx * hx;
  h2[1] = hy * hy;
  h2[2] = hz * hz;
  // The cross product of the clip x/y rows points along the camera axis in
  // box coordinates. The w gradient alone changes direction under scale.
  const v = VIEW_AXIS_SCRATCH;
  v[0] = e[4] * e[9] - e[8] * e[5];
  v[1] = e[8] * e[1] - e[0] * e[9];
  v[2] = e[0] * e[5] - e[4] * e[1];
  let depthAlongAxis = 0;
  let inverseRadiusSq = 0;
  for (let i = 0; i < 3; i++) {
    const g = e[3 + 4 * i];
    if (h2[i] === 0 && v[i] !== 0) return 0;
    depthAlongAxis += g * v[i];
    if (h2[i] > 0) inverseRadiusSq += (v[i] * v[i]) / h2[i];
  }
  return inverseRadiusSq > 0 ? Math.abs(depthAlongAxis) / Math.sqrt(inverseRadiusSq) : 0;
}

/** Reused result object for {@link projectInscribedEllipse} (no per-call allocation). */
const ELLIPSE_SCRATCH = { cx: 0, cy: 0, sxx: 0, sxy: 0, syy: 0 };

/** Squared box half-extents for projection and view-chord math (reused, no allocation). */
const HALF_EXTENTS_SQ = [0, 0, 0];

/** Camera-axis direction in the box frame (reused, no per-call allocation). */
const VIEW_AXIS_SCRATCH = [0, 0, 0];

/** ``Σ h_i² a_i b_i`` over the three spatial columns of projection rows ``a`` and ``b``. */
function weightedRowDot(e: ArrayLike<number>, a: number, b: number, h2: readonly number[]): number {
  return h2[0] * e[a] * e[b] + h2[1] * e[a + 4] * e[b + 4] + h2[2] * e[a + 8] * e[b + 8];
}

/**
 * Project the ellipsoid inscribed in ``box`` through the column-major
 * projection × view ``e`` and return its image ellipse in NDC: centre
 * ``(cx, cy)`` and shape matrix ``S`` (``sxx, sxy, syy``; the ellipse is
 * ``{c + S^{1/2} u : |u| = 1}``).
 *
 * The ellipsoid's dual quadric is ``Q* = T diag(hx², hy², hz², −1) Tᵀ``
 * (``T`` translates to the box centre); its image is the dual conic
 * ``C* = P Q* Pᵀ`` over the x, y and w rows of the projection. Normalised so
 * ``C*₂₂ = −1`` it reads ``[[S − ccᵀ, −c], [−cᵀ, −1]]``. ``−C*₂₂`` is
 * ``w_c² − Σ(h_i p_{3i})²``: positive iff the eye plane misses the ellipsoid.
 * Returns {@link ELLIPSE_STRADDLES} when it does not, {@link ELLIPSE_BEHIND}
 * when the ellipsoid is wholly behind the eye, else the module-scope scratch
 * (consume it before the next call).
 */
function projectInscribedEllipse(
  box: BoundingBox,
  e: ArrayLike<number>
): typeof ELLIPSE_SCRATCH | typeof ELLIPSE_STRADDLES | typeof ELLIPSE_BEHIND {
  const hx = 0.5 * (box.max.x - box.min.x);
  const hy = 0.5 * (box.max.y - box.min.y);
  const hz = 0.5 * (box.max.z - box.min.z);
  const x = 0.5 * (box.max.x + box.min.x);
  const y = 0.5 * (box.max.y + box.min.y);
  const z = 0.5 * (box.max.z + box.min.z);
  const h2 = HALF_EXTENTS_SQ;
  h2[0] = hx * hx;
  h2[1] = hy * hy;
  h2[2] = hz * hz;
  // The projected homogeneous centre (rows 0, 1 and 3 of P).
  const u = e[0] * x + e[4] * y + e[8] * z + e[12];
  const v = e[1] * x + e[5] * y + e[9] * z + e[13];
  const w = e[3] * x + e[7] * y + e[11] * z + e[15];
  const k = w * w - weightedRowDot(e, 3, 3, h2);
  if (!(k > 0)) return ELLIPSE_STRADDLES;
  if (w < 0) return ELLIPSE_BEHIND;
  const cx = (u * w - weightedRowDot(e, 0, 3, h2)) / k;
  const cy = (v * w - weightedRowDot(e, 1, 3, h2)) / k;
  ELLIPSE_SCRATCH.cx = cx;
  ELLIPSE_SCRATCH.cy = cy;
  ELLIPSE_SCRATCH.sxx = (weightedRowDot(e, 0, 0, h2) - u * u) / k + cx * cx;
  ELLIPSE_SCRATCH.sxy = (weightedRowDot(e, 0, 1, h2) - u * v) / k + cx * cy;
  ELLIPSE_SCRATCH.syy = (weightedRowDot(e, 1, 1, h2) - v * v) / k + cy * cy;
  return ELLIPSE_SCRATCH;
}

/** Reused result object for {@link projectBoxNdcRect} (no per-call allocation). */
const NDC_RECT_SCRATCH = { halfW: 0, halfH: 0, minX: 0, maxX: 0, minY: 0, maxY: 0 };

/**
 * Shared 8-corner projection for the two metrics above: the box's screen-space
 * AABB as raw NDC bounds (``minX``/``maxX``/``minY``/``maxY``, unclamped) plus
 * the HALF-NDC spans (``ndcExtent / 2`` per axis — i.e. the fraction of the
 * viewport covered along each axis, unclamped) the diagonal metric consumes.
 * Returns ``null`` when any corner's homogeneous ``w`` falls to/below
 * {@link W_EPSILON} (camera inside / straddling the box — callers saturate to
 * ``+Infinity``). The returned object is a module-scope scratch: consume it
 * before the next call.
 */
function projectBoxNdcRect(
  box: BoundingBox,
  camera: THREE.Camera,
  boxToClip?: THREE.Matrix4
): { halfW: number; halfH: number; minX: number; maxX: number; minY: number; maxY: number } | null {
  // The registry passes projection × view × group-world for group-local bounds.
  // Standalone callers omit it and use projection × matrixWorldInverse with world
  // bounds. Unlike THREE's ``Vector3.project`` this exposes ``w`` so we can
  // guard the near plane. The module scratch has no per-call allocation.
  const m =
    boxToClip ??
    PROJ_VIEW_SCRATCH.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const e = m.elements; // THREE.Matrix4 is column-major flat[16]
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < 8; i++) {
    const x = i & 1 ? box.max.x : box.min.x;
    const y = i & 2 ? box.max.y : box.min.y;
    const z = i & 4 ? box.max.z : box.min.z;
    // Same column-major indexing as ``transformBoundingBox`` (bounds-math.ts):
    // w = m[3]*x + m[7]*y + m[11]*z + m[15].
    const w = e[3] * x + e[7] * y + e[11] * z + e[15];
    if (w <= W_EPSILON) {
      // Camera inside / straddling the bbox near plane → group fills the
      // screen → saturate so the finest child is selected.
      return null;
    }
    const ndcX = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w;
    const ndcY = (e[1] * x + e[5] * y + e[9] * z + e[13]) / w;
    if (ndcX < minX) minX = ndcX;
    if (ndcX > maxX) maxX = ndcX;
    if (ndcY < minY) minY = ndcY;
    if (ndcY > maxY) maxY = ndcY;
  }
  NDC_RECT_SCRATCH.halfW = (maxX - minX) * 0.5;
  NDC_RECT_SCRATCH.halfH = (maxY - minY) * 0.5;
  NDC_RECT_SCRATCH.minX = minX;
  NDC_RECT_SCRATCH.maxX = maxX;
  NDC_RECT_SCRATCH.minY = minY;
  NDC_RECT_SCRATCH.maxY = maxY;
  return NDC_RECT_SCRATCH;
}

/**
 * Pick the desired child index given a scalar view ``metric`` and the
 * current active index. Applies 10% asymmetric hysteresis on the
 * downgrade direction.
 *
 * ``metric`` is whatever scalar the group's ``selector`` names — the viewport
 * AREA fraction under ``'screen-area'`` (what derived ladders stamp), the
 * dimensionless normalised diagonal (projected bbox diagonal ÷ ``FILL_FACTOR ×
 * fittedAxisPx``, where ``fittedAxisPx`` is ``min(viewport.width,
 * viewport.height)``) under the legacy ``'coverage'`` — and ``thresholds`` are
 * the per-child ``coverage_fraction`` values, in whichever units that
 * ``selector`` names. Under ``'screen-area'`` the derived finest anchor is 0.5
 * whole-object / 1.0 partition-bound; under legacy ``'coverage'`` it is 1.0
 * whole-object, and an explicitly authored **or partition-bound** threshold may
 * exceed 1 (up to ``SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR``) to hold a level
 * until later — both ceilings are enforced Python-side (``validate_lod_group``),
 * never here. The "natural" pick is the finest child whose
 * ``coverageFraction`` is less than or equal to ``metric``. Hysteresis only
 * resists dropping back to a coarser level: when downgrading from index
 * ``currentIdx``, the metric must fall below the current threshold by a margin
 * that is ``hysteresisRatio`` (default 10%) of the GAP to the adjacent coarser
 * threshold — i.e. below
 * ``thresholds[currentIdx] - hysteresisRatio * (thresholds[currentIdx] -
 * thresholds[currentIdx - 1])``; otherwise we stay on the current level
 * even though the natural pick is coarser. (At the bottom level
 * ``thresholds[currentIdx - 1]`` is effectively 0, reducing the margin to
 * ``hysteresisRatio * thresholds[currentIdx]`` — the old isolated-threshold
 * form.) Upgrades to a finer level are immediate (no hysteresis).
 *
 * Exported for unit testing.
 */
export function pickChildWithHysteresis(
  thresholds: readonly number[],
  currentIdx: number,
  metric: number,
  hysteresisRatio: number = HYSTERESIS_RATIO
): number {
  if (thresholds.length === 0) return -1;

  // Natural pick: finest child with threshold ≤ metric. Thresholds
  // are monotonic increasing in coarsest→finest order, so scan upward
  // until the threshold exceeds the metric.
  let natural = 0;
  for (let i = 0; i < thresholds.length; i++) {
    if (thresholds[i] <= metric) natural = i;
    else break;
  }

  if (natural === currentIdx) return currentIdx;
  if (natural > currentIdx) return natural; // upgrade: literal threshold wins

  // Downgrade: require the metric to drop below the current threshold by a
  // hysteresis margin that is **spacing-aware** — a fraction
  // (``hysteresisRatio``) of the GAP to the adjacent coarser threshold,
  // rather than of the current threshold in isolation. For the bottom real
  // level (coarser threshold 0) the gap equals the threshold, so this
  // reduces to the original ``currentThreshold * (1 - ratio)`` behaviour.
  // For tightly-spaced levels (e.g. separated only by the
  // ``coverage_fractions`` ×1.1 monotonicity nudge) the band shrinks
  // proportionally, so the deadband never straddles the neighbour — every level still
  // renders on the way down and the selection can't flip-flop across a band
  // wider than the inter-level spacing.
  //
  // The margin guards only the immediate ``currentIdx → currentIdx - 1``
  // boundary, but ``natural`` may be several levels coarser. That is correct: a
  // multi-level drop means the metric fell well past the adjacent band, so the
  // hysteresis (sized to one inter-level gap) cannot suppress it and we snap
  // straight to ``natural`` — no flip-flop, because the metric is nowhere near
  // the band it would need to re-cross to come back up.
  const currentThreshold = thresholds[currentIdx];
  const prevThreshold = thresholds[currentIdx - 1]; // currentIdx >= 1 here
  const margin = hysteresisRatio * (currentThreshold - prevThreshold);
  if (metric < currentThreshold - margin) {
    return natural;
  }
  return currentIdx;
}

/**
 * The neighbour of level ``currentIdx`` worth making resident ahead of a switch,
 * or ``-1``: the level across whichever of ``currentIdx``'s two boundaries lies
 * within a band of ``metric``. Boundary ``b`` is the activation threshold of
 * level ``b`` (``thresholds[b]``, between levels ``b − 1`` and ``b``); its band
 * half-width is ``fraction · min(gap below, gap above)`` — the gap above the
 * finest boundary is taken equal to the gap below — so the band is the same
 * share of a step at every level of a geometrically spaced ladder and, for
 * ``fraction ≤ 0.5``, no two bands overlap. When the metric is inside the bands
 * of both boundaries (possible only for ``fraction > 0.5``), the nearer wins.
 *
 * This is the band the retired coverage cross-fade DREW both levels in; the
 * registry now only LOADS the neighbour there (``LODGroupRegistry.preloadNeighbour``).
 * Returns ``-1`` for ``fraction <= 0``, an out-of-range ``currentIdx`` or a
 * degenerate (zero-width) band. Allocation-free: it runs per group per frame.
 */
export function preloadNeighbourIndex(
  thresholds: readonly number[],
  currentIdx: number,
  metric: number,
  fraction: number
): number {
  const n = thresholds.length;
  if (!(fraction > 0) || currentIdx < 0 || currentIdx >= n) return -1;
  const finerDist = bandDistance(thresholds, currentIdx + 1, metric, fraction);
  const coarserDist = bandDistance(thresholds, currentIdx, metric, fraction);
  if (finerDist < 0 && coarserDist < 0) return -1;
  if (coarserDist < 0 || (finerDist >= 0 && finerDist <= coarserDist)) return currentIdx + 1;
  return currentIdx - 1;
}

/**
 * ``|metric − thresholds[b]|`` when the metric lies inside boundary ``b``'s band
 * (see {@link preloadNeighbourIndex}), else ``-1``. Boundary ``0`` (the coarsest
 * level's floor) and ``b >= thresholds.length`` do not exist.
 */
function bandDistance(
  thresholds: readonly number[],
  b: number,
  metric: number,
  fraction: number
): number {
  const n = thresholds.length;
  if (b < 1 || b >= n) return -1;
  const gapBelow = thresholds[b] - thresholds[b - 1];
  const gapAbove = b + 1 < n ? thresholds[b + 1] - thresholds[b] : gapBelow;
  const half = fraction * Math.min(gapBelow, gapAbove);
  const dist = Math.abs(metric - thresholds[b]);
  return half > 0 && dist < half ? dist : -1;
}

/**
 * Current viewer median-sigma limit in logical CSS pixels. GSplats draw to about
 * 3σ, so 1.5 px corresponds to a typical rendered blob about 9 px across. The
 * #2685 sweep retained this policy.
 */
export const MAX_MEDIAN_FOOTPRINT_PX = 1.5;

/** Pick the coarsest acceptable footprint, resisting only coarser downgrades. */
export function pickChildByFootprintWithHysteresis(
  footprintsPx: readonly number[],
  currentIdx: number,
  maxFootprintPx: number = MAX_MEDIAN_FOOTPRINT_PX,
  hysteresisRatio: number = HYSTERESIS_RATIO
): number {
  if (footprintsPx.length === 0) return -1;
  let natural = footprintsPx.length - 1;
  for (let i = 0; i < footprintsPx.length; i++) {
    if (footprintsPx[i] <= maxFootprintPx) {
      natural = i;
      break;
    }
  }
  if (natural >= currentIdx) return natural;
  return footprintsPx[natural] <= maxFootprintPx * (1 - hysteresisRatio) ? natural : currentIdx;
}

/**
 * Project a node-local radius into logical CSS pixels. The caller divides the
 * accepted radius by `sqrt(lodBias)` because the public bias remains an area
 * factor while this metric is a length.
 *
 * The vertical scale is the projection's own `P[1][1]` (`elements[5]`), the
 * term the shaders size geometry with: `1 / tan(fov/2)` for a plain
 * perspective camera, `2 / frustumHeight` for an orthographic one, and in
 * both cases including `zoom` and `setViewOffset`. Only a
 * perspective or orthographic camera has a footprint; any other returns null.
 *
 * @param view The camera and its world → view matrix (the frame's `ViewContext`).
 */
export function projectWorldRadiusPx(
  radiusWorld: number,
  worldCenter: THREE.Vector3,
  view: Pick<ViewContext, 'camera' | 'viewMatrix'>,
  viewportHeight: number,
  viewCenterScratch: THREE.Vector3 = new THREE.Vector3()
): number | null {
  if (!(radiusWorld > 0) || !Number.isFinite(radiusWorld) || viewportHeight <= 0) return null;
  const { camera, viewMatrix } = view;
  const scale = Math.abs(camera.projectionMatrix.elements[5]);
  if (camera instanceof THREE.PerspectiveCamera) {
    const viewCenter = viewCenterScratch.copy(worldCenter).applyMatrix4(viewMatrix);
    const depth = -viewCenter.z;
    if (!(depth > 0)) return Number.POSITIVE_INFINITY;
    return (radiusWorld * viewportHeight * scale) / (2 * depth);
  }
  if (camera instanceof THREE.OrthographicCamera) {
    return scale > 0 && Number.isFinite(scale) ? (radiusWorld * viewportHeight * scale) / 2 : null;
  }
  return null;
}

/**
 * Minimal structural shape {@link computeEntryWorldBox} reads off a registry
 * entry (``LODGroupEntry`` satisfies it structurally — same pattern as
 * ``lod-freshness.ts``'s ``FreshnessChild``, avoiding an import cycle back
 * into the registry).
 */
export interface WorldBoxSource {
  /** The lod_group's THREE container. World matrix lives here. */
  groupObject: THREE.Object3D;
  /** Per-child raw nD position bounds (unprojected — displayDims can change). */
  children: readonly {
    positionBounds: { min: readonly number[]; max: readonly number[] };
    /** Optional robust bounds used only for LOD metric sizing. */
    lodBounds?: { min: readonly number[]; max: readonly number[] };
  }[];
}

/** Caller-owned output reused across evaluations; ``useLodBounds`` selects robust metric bounds. */
export interface WorldBoxOptions {
  worldBoxScratch: BoundingBox;
  useLodBounds?: boolean;
}

/**
 * Fold an entry's children nD bounds into a single world-space
 * :type:`BoundingBox`, mapping nD axes onto X/Y/Z via the current
 * ``displayDims`` and lifting through the group's ``matrixWorld``. Returns
 * ``null`` when no child has usable bounds (mismatched/empty min-max). Shared
 * by the auto selector and eviction ranking. With ``useLodBounds`` true, each
 * child's optional ``lodBounds`` sizes the LOD metric and falls back to its raw
 * ``positionBounds`` when absent; callers keep the default raw bounds for
 * frustum gating and eviction so visible outliers remain part of the geometry.
 * Children with bogus bounds are skipped. Uses the caller-owned
 * ``localBoxScratch`` (per-entry) and ``matrixScratch`` (per-registry);
 * ``worldBoxScratch`` receives the transformed bounds and is reused by the
 * caller on the next evaluation.
 */
export function computeEntryWorldBox(
  entry: WorldBoxSource,
  displayDims: readonly number[],
  localBoxScratch: BoundingBox,
  matrixScratch: number[],
  options: WorldBoxOptions
): BoundingBox | null {
  const local = localBoxScratch;
  let any = false;
  for (let ci = 0; ci < entry.children.length; ci++) {
    const child = entry.children[ci];
    const pb = options.useLodBounds
      ? (child.lodBounds ?? child.positionBounds)
      : child.positionBounds;
    if (pb.min.length === 0 || pb.max.length === 0 || pb.min.length !== pb.max.length) {
      continue;
    }
    // Project to X/Y/Z. Unmapped axes default to 0 (matches
    // ``projectBoundsToDisplayDims``'s defensive fallback).
    let x0 = 0;
    let x1 = 0;
    let y0 = 0;
    let y1 = 0;
    let z0 = 0;
    let z1 = 0;
    if (displayDims.length > 0) {
      const d0 = displayDims[0];
      if (d0 < pb.min.length) {
        x0 = pb.min[d0];
        x1 = pb.max[d0];
      }
    }
    if (displayDims.length > 1) {
      const d1 = displayDims[1];
      if (d1 < pb.min.length) {
        y0 = pb.min[d1];
        y1 = pb.max[d1];
      }
    }
    if (displayDims.length > 2) {
      const d2 = displayDims[2];
      if (d2 < pb.min.length) {
        z0 = pb.min[d2];
        z1 = pb.max[d2];
      }
    }
    if (!any) {
      local.min.x = x0;
      local.max.x = x1;
      local.min.y = y0;
      local.max.y = y1;
      local.min.z = z0;
      local.max.z = z1;
      any = true;
    } else {
      if (x0 < local.min.x) local.min.x = x0;
      if (x1 > local.max.x) local.max.x = x1;
      if (y0 < local.min.y) local.min.y = y0;
      if (y1 > local.max.y) local.max.y = y1;
      if (z0 < local.min.z) local.min.z = z0;
      if (z1 > local.max.z) local.max.z = z1;
    }
  }
  if (!any) return null;

  // Lift to world space. THREE updates matrix lazily; force a refresh before
  // reading — cheap and idempotent. Copy ``matrixWorld.elements`` into a
  // reusable array instead of allocating one via ``.toArray()`` every frame.
  entry.groupObject.updateWorldMatrix(true, false);
  const elements = entry.groupObject.matrixWorld.elements;
  const m = matrixScratch;
  for (let i = 0; i < 16; i++) m[i] = elements[i];
  return transformBoundingBox(local, m, options.worldBoxScratch);
}
