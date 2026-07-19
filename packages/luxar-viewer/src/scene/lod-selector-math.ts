/**
 * Selector math for the per-frame LOD-group selector.
 *
 * The camera-geometry half of `lod-group-registry.ts`, extracted so the
 * registry file holds the policy/state machine and this module holds the
 * (individually unit-testable) math:
 *
 * - {@link computeEntryWorldBox} — fold a group's per-child nD
 *   ``positionBounds`` into one world-space box via ``displayDims`` +
 *   ``matrixWorld``.
 * - {@link projectBoxDiagonalPx} — project that box through the camera to a
 *   screen-space pixel diagonal (with near-plane saturation).
 * - {@link pickChildWithHysteresis} — pick the level for a coverage metric,
 *   with asymmetric downgrade hysteresis.
 *
 * The registry re-exports `projectBoxDiagonalPx` / `pickChildWithHysteresis`
 * so existing importers (the selector unit tests) are unchanged.
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
      return Number.POSITIVE_INFINITY;
    }
    const ndcX = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w;
    const ndcY = (e[1] * x + e[5] * y + e[9] * z + e[13]) / w;
    if (ndcX < minX) minX = ndcX;
    if (ndcX > maxX) maxX = ndcX;
    if (ndcY < minY) minY = ndcY;
    if (ndcY > maxY) maxY = ndcY;
  }
  const widthPx = (maxX - minX) * 0.5 * viewport.width;
  const heightPx = (maxY - minY) * 0.5 * viewport.height;
  return Math.hypot(widthPx, heightPx);
}

/**
 * Pick the desired child index given a scalar view ``metric`` and the
 * current active index. Applies 10% asymmetric hysteresis on the
 * downgrade direction.
 *
 * ``metric`` is the dimensionless coverage metric (projected bbox diagonal ÷
 * ``FILL_FACTOR × viewportDiagonal``) and ``thresholds`` are the per-child
 * ``coverage_fraction`` values; both are in the same [0,1]-ish space. The
 * "natural" pick is the finest child whose ``coverageFraction`` is less than or
 * equal to ``metric``. Hysteresis only resists dropping back to a coarser level:
 * when downgrading from index ``currentIdx``, the metric must fall below the
 * current threshold by a margin that is ``hysteresisRatio`` (default 10%) of the
 * GAP to the adjacent coarser threshold — i.e. below
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
  }[];
}

/**
 * Fold an entry's children nD ``positionBounds`` into a single world-space
 * :type:`BoundingBox`, mapping nD axes onto X/Y/Z via the current
 * ``displayDims`` and lifting through the group's ``matrixWorld``. Returns
 * ``null`` when no child has usable bounds (mismatched/empty min-max). Shared
 * by the auto selector (diagonal pick + frustum gate) and the eviction
 * ranking so both reason over identical geometry. Children with bogus bounds
 * are skipped. Uses the caller-owned ``localBoxScratch`` (per-entry) and
 * ``matrixScratch`` (per-registry); ``transformBoundingBox`` allocates the
 * returned box, so it is independent of those scratches and safe to keep past
 * the next call.
 */
export function computeEntryWorldBox(
  entry: WorldBoxSource,
  displayDims: readonly number[],
  localBoxScratch: BoundingBox,
  matrixScratch: number[]
): BoundingBox | null {
  const local = localBoxScratch;
  let any = false;
  for (let ci = 0; ci < entry.children.length; ci++) {
    const pb = entry.children[ci].positionBounds;
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
  return transformBoundingBox(local, m);
}
