/**
 * Selector math for the per-frame LOD-group selector.
 *
 * The camera-geometry half of `lod-group-registry.ts`, extracted so the
 * registry file holds the policy/state machine and this module holds the
 * (individually unit-testable) math:
 *
 * - {@link computeEntryWorldBox} — fold a group's per-child nD raw or robust
 *   bounds into one world-space box via ``displayDims`` + ``matrixWorld``.
 * - {@link projectBoxDiagonalPx} — project that box through the camera to a
 *   screen-space pixel diagonal (with near-plane saturation). The legacy
 *   ``selector: 'coverage'`` metric.
 * - {@link projectBoxAreaFraction} — project that box to the fraction of the
 *   viewport AREA its screen-space rect covers (same near-plane saturation).
 *   The ``selector: 'screen-area'`` metric.
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

import { type BoundingBox, transformBoundingBox } from './scene-manager/clipping/bounds-math';

/** Asymmetric hysteresis on the "downgrade to coarser" direction. */
const HYSTERESIS_RATIO = 0.1;

/**
 * Module-scope scratch for ``projectBoxDiagonalPx``'s projection × view
 * product. Single-threaded — ``evaluatePerFrame`` is the only per-frame entry
 * point, so reusing one matrix across all entries within a frame is safe.
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
 * Project a world-space :type:`BoundingBox` through the camera and
 * return the diagonal of the screen-space AABB in pixels.
 *
 * Treats the bbox's 8 corners independently (works for both
 * perspective and orthographic projection without a closed-form
 * radius). NDC → pixels assumes the viewport size matches the
 * renderer canvas.
 *
 * **Near-plane saturation.** Projects with an explicit homogeneous ``w`` (the
 * combined ``projectionMatrix * matrixWorldInverse``, not THREE's
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
  precomputedProjView?: THREE.Matrix4
): number {
  const rect = projectBoxNdcRect(box, camera, precomputedProjView);
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
 * Project a world-space :type:`BoundingBox` through the camera and return the
 * fraction of the viewport AREA its screen-space AABB **visibly** covers — the
 * metric for ``selector: 'screen-area'``.
 *
 * **Clipped to the viewport.** The projected rect is intersected with the
 * viewport before the area is taken, so the metric reads the portion of the
 * screen ACTUALLY occupied: a node whose rect extends far off-screen but
 * clips only a corner reads that small visible fraction (and picks a coarse
 * level) instead of an arbitrarily large unclipped product — which matters
 * while panning across partition tiles. The metric therefore tops out at
 * exactly ``1.0`` (full coverage); the natural pick uses ``threshold <=
 * metric``, so the fills-screen partition threshold (1.0) is satisfied the
 * moment coverage is complete, and stays satisfied while zoomed past it. In
 * NDC each axis spans 2, so the covered fraction is the product of the
 * clipped half-extents — viewport-size independent by construction (the same
 * framing yields the same fraction on any monitor).
 *
 * **Degenerate (lower-dimensional) CONTENT ramps to its LINEAR span.** For a
 * rect whose RAW (pre-clip) thin half-extent is below
 * {@link DEGENERATE_RECT_HALF_EXTENT} — sub-pixel thin content: an
 * axis-aligned straight polyline, an edge-on plane — the area product reads
 * ~0 regardless of how much screen the content spans, which would pin it to
 * the coarsest level forever (the legacy diagonal metric never had this
 * failure mode — a diagonal reads the long extent). The metric is therefore
 * ``max(area, clippedSpan × (1 − rawThin/DEGENERATE_RECT_HALF_EXTENT))``: at
 * zero thickness it reads the full CLIPPED linear span (a full-width line =
 * 1.0, so the halving ladder keeps its meaning for 1D content), decays
 * CONTINUOUSLY to the plain area product as the thickness reaches the
 * sub-pixel floor — no cliff for the hysteresis to oscillate across when an
 * edge-on plane rotates through the boundary — and is exactly the area
 * product everywhere above it. A both-axes-degenerate rect (a point) still
 * reads ~0 → coarsest. The ramp is gated on the RAW thinness so it fires only
 * for intrinsically thin content — a wide 2D node whose CLIPPED sliver
 * happens to be thin (mostly panned off-screen) honestly reads its tiny
 * visible area rather than being inflated to a full linear span.
 *
 * **Fully off-screen rects read exactly 0.** A clipped interval that is
 * INVERTED (no viewport overlap on that axis) zeroes the whole metric before
 * the degenerate ramp can see it — otherwise a zero-thickness clipped axis
 * would be indistinguishable from off-screen and the ramp would return the
 * other axis's span for geometry not on screen at all (the world-space
 * frustum gate catches most of these, but it is conservative near frustum
 * corners, so this function must not rely on it).
 *
 * Same near-plane saturation contract as {@link projectBoxDiagonalPx}: under
 * a PERSPECTIVE camera, bounds reaching the near plane have no meaningful
 * projection (the homogeneous divide degenerates), so the metric saturates to
 * ``+Infinity`` → finest. An ORTHOGRAPHIC projection never degenerates
 * (``w`` stays 1) so no saturation applies — the plain clipped metric is
 * already well-defined, and a camera inside a large node reads full coverage
 * naturally because its rect spans the viewport. Both selectors share this
 * contract by design (see the v3.4 spec's normative metric rules).
 *
 * Exported for unit testing.
 */
export function projectBoxAreaFraction(
  box: BoundingBox,
  camera: THREE.Camera,
  precomputedProjView?: THREE.Matrix4
): number {
  const rect = projectBoxNdcRect(box, camera, precomputedProjView);
  if (rect === null) return Number.POSITIVE_INFINITY;
  // Intersect with the viewport (NDC [-1, 1] per axis). Keep the SIGNED
  // overlaps: a negative value means no viewport overlap on that axis —
  // fully off-screen, metric 0 — and must not be conflated with a genuine
  // zero-thickness visible interval (a line lying inside the viewport), which
  // clamping alone would do.
  const overlapW = Math.min(rect.maxX, 1) - Math.max(rect.minX, -1);
  const overlapH = Math.min(rect.maxY, 1) - Math.max(rect.minY, -1);
  if (overlapW < 0 || overlapH < 0) return 0;
  const halfW = overlapW * 0.5;
  const halfH = overlapH * 0.5;
  const span = Math.max(halfW, halfH);
  const area = halfW * halfH;
  // Continuous degenerate ramp, gated on the RAW (pre-clip) thinness so only
  // intrinsically thin content takes it (see the doc above).
  const rawThin = Math.min(rect.maxX - rect.minX, rect.maxY - rect.minY) * 0.5;
  const degenerate = span * Math.max(0, 1 - rawThin / DEGENERATE_RECT_HALF_EXTENT);
  return Math.max(area, degenerate);
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
  precomputedProjView?: THREE.Matrix4
): { halfW: number; halfH: number; minX: number; maxX: number; minY: number; maxY: number } | null {
  // Combined projection × view. ``evaluatePerFrame`` already builds this product
  // once per frame (``FRUSTUM_MATRIX_SCRATCH``) and passes it in via
  // ``precomputedProjView`` so we don't recompute the 4×4 per group. Standalone
  // callers (unit tests) omit it and we fall back to a module-scope scratch (no
  // per-call allocation). Unlike THREE's ``Vector3.project`` this exposes ``w``
  // so we can guard the near plane. ``evaluatePerFrame`` is the single per-frame
  // entry point, so sharing the scratch is safe.
  const m =
    precomputedProjView ??
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
