/**
 * Pure utility functions for scene management calculations
 *
 * This module contains side-effect-free functions extracted from scene-manager.ts
 * to improve testability and maintainability. These functions handle camera
 * calculations, bounding box operations, and scene analysis without external
 * dependencies.
 */

import { config } from '../../../config';
import { clamp } from '../../../utils/clamp';
import { log, Modules } from '../../../utils/log';

/**
 * 3D bounding box representation
 */
export interface BoundingBox {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
}

/**
 * Camera configuration for scene fitting
 */
export interface CameraConfig {
  fov: number; // Field of view in degrees
  aspect: number; // Aspect ratio
  near: number; // Near clipping plane
  far: number; // Far clipping plane
}

/**
 * Calculates bounding box center point
 *
 * @param box - Bounding box
 * @returns Center point coordinates
 */
export function getBoundingBoxCenter(box: BoundingBox): { x: number; y: number; z: number } {
  return {
    x: (box.min.x + box.max.x) / 2,
    y: (box.min.y + box.max.y) / 2,
    z: (box.min.z + box.max.z) / 2,
  };
}

/**
 * Calculates bounding box size
 *
 * @param box - Bounding box
 * @returns Size in each dimension
 */
export function getBoundingBoxSize(box: BoundingBox): { x: number; y: number; z: number } {
  return {
    x: box.max.x - box.min.x,
    y: box.max.y - box.min.y,
    z: box.max.z - box.min.z,
  };
}

/**
 * Calculates maximum dimension of bounding box
 *
 * @param box - Bounding box
 * @returns Maximum dimension value
 */
export function getBoundingBoxMaxDimension(box: BoundingBox): number {
  const size = getBoundingBoxSize(box);
  return Math.max(size.x, size.y, size.z);
}

/**
 * Calculates diagonal of bounding box (Euclidean distance from min to max corner).
 * Used as the scene scale metric for scale-aware camera controls.
 *
 * @param box - Bounding box
 * @returns Diagonal length
 */
export function getBoundingBoxDiagonal(box: BoundingBox): number {
  const size = getBoundingBoxSize(box);
  return Math.sqrt(size.x * size.x + size.y * size.y + size.z * size.z);
}

/**
 * Calculates the +Z camera distance that fits a bounding box in view.
 * X/Y are the screen plane and Z is depth, matching fitCameraToBounds' fixed
 * face-on pose. The larger X/Y extent fills fitRatio of the shorter viewport
 * axis at the nearest Z face. If both projected extents are zero, half the Z
 * extent is used as a conservative fallback so the camera stays off the geometry.
 *
 * @param box - Bounding box to fit
 * @param camera - Camera configuration
 * @param fitRatio - How much of the view to fill (0-1, default 0.75)
 * @param target - Look-at target; defaults to the bounding-box center
 * @returns Optimal +Z camera distance from the look-at target
 */
export function calculateCameraDistance(
  box: BoundingBox,
  camera: CameraConfig,
  fitRatio: number = config.scene.defaultFitRatio,
  target: { x: number; y: number; z: number } = getBoundingBoxCenter(box)
): number {
  const fovRadians = (camera.fov * Math.PI) / 180;
  const halfFov = fovRadians / 2;
  const nearestDepth = Math.max(0, box.max.z - target.z);
  const screenPlaneRadius = Math.max(
    Math.abs(box.min.x - target.x),
    Math.abs(box.max.x - target.x),
    Math.abs(box.min.y - target.y),
    Math.abs(box.max.y - target.y)
  );
  const fitRadius = screenPlaneRadius > 0 ? screenPlaneRadius : Math.abs(box.max.z - box.min.z) / 2;

  // Fit the larger screen-plane extent against the shorter viewport axis at
  // the nearest face of the box. Depth does not enlarge the silhouette
  // directly, but it brings that face closer and increases its projection.
  const verticalFit = nearestDepth + fitRadius / fitRatio / Math.tan(halfFov);
  const horizontalFit = nearestDepth + fitRadius / fitRatio / (Math.tan(halfFov) * camera.aspect);

  return Math.max(verticalFit, horizontalFit);
}

/**
 * Validates and clamps field of view value
 *
 * @param fov - Field of view in degrees
 * @param min - Minimum FOV (default 10)
 * @param max - Maximum FOV (default 120)
 * @returns Clamped FOV value
 */
export function validateFOV(fov: number, min: number = 10, max: number = 120): number {
  return clamp(fov, min, max);
}

/**
 * Absolute last-resort near-plane floor (degenerate / zero-radius
 * scenes only). Under a perspective projection the depth-precision floor
 * from {@link nearPlaneFloor} dominates for any real scene; under ortho
 * the scale-aware {@link minNearForRadius} does — see both.
 */
export const MIN_NEAR_PLANE = 1e-9;

/**
 * Relative near-plane floor: near is never smaller than this fraction
 * of the (safety-expanded) scene bounding-sphere radius.
 *
 * Why relative and not absolute: the historical absolute floor
 * (0.0001) silently broke tiny scenes — with the 1000x zoom-in
 * headroom, a scene of diagonal ≲ 0.1 world units lets the camera
 * orbit closer to the target than the floor itself, clipping all
 * nearby geometry. A floor proportional to scene scale keeps the same
 * zoom-in depth working at every scale.
 *
 * The factor is chosen for continuity with the historical constant: a
 * typical diagonal-100 scene has expanded radius ~52.5, and
 * 52.5 x 2e-6 ~ 1e-4 — exactly the old absolute floor.
 */
export const MIN_NEAR_RADIUS_FACTOR = 2e-6;

/**
 * Scale-aware near-plane floor for a scene with the given
 * (safety-expanded) bounding-sphere radius.
 *
 * This is the floor that says nothing about depth-buffer precision. Which
 * role it plays depends on the projection:
 *
 *  - **Perspective** — a dominated backstop. {@link nearPlaneFloor} raises
 *    `near` to `far / MAX_NEAR_FAR_RATIO`, which is always larger (`far >= R`
 *    always holds, so `far / 1200 >= 8e-4 · R` versus this `2e-6 · R`). It
 *    surfaces only at `R -> 0`, where it yields `MIN_NEAR_PLANE` and keeps the
 *    callers' degenerate-frustum guard tripping instead of NaN-ing.
 *  - **Orthographic** — the OPERATIVE floor. Ortho opts out of the ratio bound
 *    (linear depth, nothing to gain, real clipping cost), so `nearPlaneFloor`
 *    returns exactly this value.
 *
 * Clipping paths must still go through `nearPlaneFloor` rather than calling
 * this directly, so the projection decision lives in one place. Reading it
 * directly is fine for callers that want the LOWEST near the policy can ever
 * produce — `RenderingControls.updateClippingSliderRanges` does exactly that
 * to pick the near slider's minimum.
 */
export function minNearForRadius(expandedRadius: number): number {
  return Math.max(MIN_NEAR_PLANE, expandedRadius * MIN_NEAR_RADIUS_FACTOR);
}

/**
 * Largest near/far ratio the near-plane floor will allow.
 *
 * The depth buffer is 24-bit fixed point (three allocates
 * `DEPTH_COMPONENT24` for a render target with no explicit
 * `depthType`, and `webgl.renderer.logarithmicDepthBuffer` is off), so
 * depth quantization in world units at eye distance `d` is
 *
 *     Δz(d) = d² · (far − near) / (near · far) · 2⁻²⁴
 *
 * — inversely proportional to `near`. Left unbounded, the
 * inside-the-sphere branch of {@link calculateClippingPlanesFromSphere}
 * pinned `near` to `R · MIN_NEAR_RADIUS_FACTOR` (2e-6 · R), i.e. a
 * ratio near 6e5:1, which puts Δz at ~4e-2 world units on a
 * diagonal-100 scene viewed from 8.5 units — coarse enough to z-fight
 * visibly, and (because `near` tracks camera distance) to POP as the
 * camera orbits. Only depth-writing geometry is affected (mesh, and
 * opaque `normal`-mode geometry — see `rendering/blending-state.ts`),
 * which is exactly where it was reported.
 *
 * Why 1200 specifically. A LARGER C means a smaller floor: less precision,
 * but less clipped. Two hard constraints put a floor under C, and only a soft
 * preference (keep precision) pushes from above — and that preference turns
 * out to be nearly free, so C is set for MARGIN above the binding constraint
 * rather than at it:
 *
 *  - **C > 551, don't clip the zoom target under the scale-multiplier floor.**
 *    At maximum zoom-in the
 *    orbit target sits at `minDistance = 1e-3 · diagonal`
 *    (`controls.scaleMultipliers`) = `1.905e-3 · R`, while `far ≈ 1.05 · R`.
 *    Keeping it in front of the near plane needs `1.05 R / C < 1.905e-3 R`.
 *    NOTE this assumes the orbit target is at the sphere CENTRE. Panned onto a
 *    bbox corner (`0.952 · R`) the general form `(dist + R)/C < 1.905e-3 · R`
 *    tightens, and for the last ~2% of the zoom-in range the floor overtakes
 *    `minDistance`. All four types are inside the fade-suppressed region
 *    throughout that band by construction, so at most the sub-1% line residual
 *    described in the next bullet is lost there, and this does not move C.
 *    Auto-framing installs a separate `distance / ZOOM_IN_FACTOR` limit. The
 *    projected-bounds fit approaches `distance / diagonal = 0.5` for a
 *    view-axis-elongated box at the default FOV, raising this target constraint
 *    to about 1102; C = 1200 clears it there, with a 1.14x near-plane margin at
 *    the most extreme supported zoom. At wide FOVs the ratio falls further and
 *    C does not clear this target constraint, as it did not before this change;
 *    see the `ZOOM_IN_FACTOR` note in `camera-framing.ts`.
 *  - **C ≥ 992, keep everything the floor clips inside the near fade, for all
 *    four geometry types.** Every one of them already suppresses anything
 *    closer than `nearCull = 1e-3 · diagonal` via `perspectiveNearFade`
 *    (`materials/_shared/glsl-lib.ts`), and the fade is under 0.01 below
 *    `1.0589 · nearCull` (the root of `smoothstep(1, 2, x) = 0.01`, solved
 *    rather than eyeballed in `tests/.../clipping/_near-fade-model.ts`). That
 *    single bound is what the tests assert — but what each type DOES with a
 *    sub-0.01 fade differs, and the difference is the difference between
 *    "lossless" and "very nearly", so it is worth knowing before trusting this:
 *
 *    Points and GSplats REJECT the vertex outright below 0.01 (both backends),
 *    and Mesh discards the FRAGMENT at the same 0.01 (per fragment because a
 *    triangle spans depth; a discard rather than a multiply because it writes
 *    depth — #1431). For those three the floor is exactly lossless: what it
 *    clips was not going to be rasterized at all.
 *
 *    Lines are the one partial case. They cull only when BOTH endpoints are
 *    near, clip a half-near segment onto the `nearCull` plane, and then apply
 *    the fade PER-FRAGMENT as a plain multiply with no reject of its own —
 *    their discard is a separate `max(rgb) < 1e-4` test on the COLOR, which
 *    the fade never enters. So a line fragment inside the clipped band is
 *    attenuated to under 1% of its authored contribution but is not
 *    necessarily zero, and the floor can take it. That residual is a property
 *    of the line shader, not of this constant: no value of C removes it while
 *    still bounding the ratio, it predates #1431, and at ≤1% of one fragment
 *    inside a 0.1%-of-diagonal shell it is not what sets C. Closing it would
 *    mean giving the line shader its own fade reject.
 *
 *    The worst case is NOT the camera on the sphere surface — it is just
 *    OUTSIDE it, at the crossover `dist = R · (C+1)/(C-1) ≈ 1.002 · R`,
 *    the last distance at which the floor still beats the surface term and
 *    therefore where the floor sits highest relative to `nearCull`
 *    (fade 0.00755 there versus 0.00725 on the surface). So
 *    `2.002 R / C ≤ 1.0589 · 1.905e-3 · R` ⟹ `C ≥ 992`.
 *
 * 992 binds, and C = 1200 clears it by **20.9%**. The margin is deliberate,
 * and it is cheap: going from the minimum-viable 1000 to 1200 gives away
 * **0.03%** of the total precision gain (Δz 7.05e-5 → 8.47e-5 world units on
 * the reported pose, against 4.10e-2 before the bound), and buys survival of
 * ordinary tuning elsewhere. Measured: a 10% tightening of `nearCull` needs
 * C ≥ 1104, and reducing the fade reject headroom to 1.0 needs C ≥ 1051 —
 * C = 1000 would have become silently LOSSY under either, C = 1200 holds.
 *
 * The margin is also *enforced* rather than trusted: the "clipped band is
 * already shader-rejected" property in `bounds-math.property.test.ts`
 * generates the whole floor-binding region INCLUDING that crossover, a
 * companion test pins the crossover as strictly worse than the surface, and
 * the arithmetic test asserts the constant survives that 10% `nearCull`
 * tightening — so a future change to `nearCull`, to the fade band, or to this
 * constant fails there. (Mesh used to be the FULL exception — it had no near fade
 * in either backend at all, so the floor could clip it at full brightness. Since
 * #1431 it carries the same fade with the same 0.01 reject and the same
 * `1.0589 · nearCull` headroom, which moves it into the exactly-lossless group
 * with points and gsplats and leaves lines as the only ≤1% residual above.)
 *
 * The losslessness argument assumes `nearCull` and this floor are derived from
 * the SAME bounds, which holds on the metadata and per-frame paths. It can
 * diverge in one narrow case: on a metadata-less scene `SceneBoundsCache.ensure`
 * never populates, so the materials keep `_nearCull`'s 0.1 default while
 * `autoAdjustFromBounds` derives its sphere from `Box3.setFromObject`. On a
 * large metadata-less scene the floor can then exceed `nearCull` and clip
 * geometry of any of the four types that the fade would have drawn. Compiled
 * Luxar scenes always carry `position_bounds`, so this is not reachable through
 * the normal loader.
 *
 * Measured payoff at the reported pose (R = 52.5, dist = 8.5, far = 61):
 * the floor rises 1.05e-4 → 0.0508 and depth quantization improves
 * 4.10e-2 → 8.47e-5 world units, i.e. **485x** finer.
 *
 * PERSPECTIVE ONLY — see the `boundNearFarRatio` parameter of
 * {@link nearPlaneFloor}. An orthographic projection maps eye depth
 * LINEARLY to the depth buffer, so its resolution is
 * `(far - near) / 2²⁴` regardless of `near`: the ratio bound buys ortho
 * nothing, while still clipping a slab in front of the eye that ortho
 * (unlike perspective) really does draw — `perspectiveNearFade` returns
 * 1.0 for ortho, so ALL FOUR geometry types render up to `near` there.
 * Measured on a diagonal-100 scene at the deepest legal orbit distance:
 * applying the bound under ortho clips 43.8% of the eye-to-target depth
 * versus 0.105% without it, and changes depth resolution by 0.1%.
 */
export const MAX_NEAR_FAR_RATIO = 1200;

/**
 * The near-plane floor both clipping paths clamp to: the scale-aware
 * backstop, raised to whatever `far / {@link MAX_NEAR_FAR_RATIO}`
 * demands for depth-buffer precision.
 *
 * Scale-invariant by construction — the bound is derived from `far`,
 * which is itself scene-scaled — so it keeps the tiny-scene guarantee
 * `minNearForRadius` was introduced for (#573) without the Z-precision
 * cost: on a diagonal-0.1 scene at maximum zoom-in, `far / 1200` is
 * ~4.4e-5 while the closest reachable orbit distance is ~1.7e-4, so
 * nearby geometry still renders.
 *
 * Always `< far` for `far > 0`, so callers can rely on it never
 * producing an inverted frustum on its own.
 *
 * @param expandedRadius - Safety-expanded bounding-sphere radius.
 * @param far - The far plane the caller is about to apply.
 * @param boundNearFarRatio - Whether to apply the depth-precision ratio
 *   bound. TRUE for a perspective projection (where depth resolution is
 *   `~1/near`); FALSE for an orthographic one, whose depth is linear in
 *   eye space, so the bound would clip a visible near slab for zero
 *   precision gain — see {@link MAX_NEAR_FAR_RATIO}. Callers derive it
 *   from the live camera rather than assuming, because the viewer
 *   swaps projections at runtime (V key).
 */
export function nearPlaneFloor(
  expandedRadius: number,
  far: number,
  boundNearFarRatio: boolean = true
): number {
  const scaleFloor = minNearForRadius(expandedRadius);
  if (!boundNearFarRatio) return scaleFloor;
  return Math.max(scaleFloor, far / MAX_NEAR_FAR_RATIO);
}

/**
 * 3D bounding sphere representation
 */
export interface BoundingSphere {
  center: { x: number; y: number; z: number };
  radius: number;
}

/** Safety expansion factor for bounding sphere (5%) */
export const SPHERE_SAFETY_EXPANSION = 1.05;

/**
 * Converts a bounding box to a bounding sphere (circumscribed sphere).
 */
export function boundingBoxToSphere(box: BoundingBox): BoundingSphere {
  const center = getBoundingBoxCenter(box);
  const size = getBoundingBoxSize(box);
  const radius = 0.5 * Math.sqrt(size.x * size.x + size.y * size.y + size.z * size.z);
  return { center, radius };
}

/**
 * Calculates camera clipping planes based on a bounding sphere.
 *
 * Uses a sphere instead of a bounding box to avoid discontinuities at box
 * edges/corners. The sphere produces smooth near/far values as the camera
 * moves, eliminating the need for exponential smoothing.
 *
 * `near` is the nearest point on the sphere surface, floored by
 * {@link nearPlaneFloor} — which is what keeps the near/far ratio (and
 * therefore depth-buffer precision) bounded once the camera moves INSIDE
 * the sphere, the regime where the bare surface distance goes to zero.
 *
 * @param sphere - Scene bounding sphere
 * @param cameraPosition - Camera position in world coordinates
 * @param boundNearFarRatio - Forwarded to {@link nearPlaneFloor}; pass
 *   false for an orthographic projection. Defaults to true (perspective).
 * @returns Near and far clipping plane distances
 */
export function calculateClippingPlanesFromSphere(
  sphere: BoundingSphere,
  cameraPosition: { x: number; y: number; z: number },
  boundNearFarRatio: boolean = true
): { near: number; far: number } {
  const dx = cameraPosition.x - sphere.center.x;
  const dy = cameraPosition.y - sphere.center.y;
  const dz = cameraPosition.z - sphere.center.z;
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
  const R = sphere.radius * SPHERE_SAFETY_EXPANSION;

  const far = dist + R;
  const minNear = nearPlaneFloor(R, far, boundNearFarRatio);

  if (dist < R) {
    // Inside sphere: the surface distance is meaningless (it would be
    // negative), so the floor IS the near plane. Bounded by the
    // depth-precision ratio rather than collapsing to
    // minNearForRadius — see MAX_NEAR_FAR_RATIO.
    return { near: minNear, far };
  }

  // Outside sphere: nearest point on sphere surface
  const near = Math.max(minNear, dist - R);
  return { near, far };
}

/**
 * Transforms bounding box by a 4x4 matrix
 *
 * @param box - Original bounding box
 * @param matrix - 4x4 transformation matrix (column-major, flat array)
 * @returns Transformed bounding box
 */
export function transformBoundingBox(
  box: BoundingBox,
  matrix: number[],
  target: BoundingBox = {
    min: { x: 0, y: 0, z: 0 },
    max: { x: 0, y: 0, z: 0 },
  }
): BoundingBox {
  const inputMinX = box.min.x;
  const inputMinY = box.min.y;
  const inputMinZ = box.min.z;
  const inputMaxX = box.max.x;
  const inputMaxY = box.max.y;
  const inputMaxZ = box.max.z;
  // Transform each corner, skipping any whose homogeneous w is ~0 to avoid
  // dividing through to ±Infinity/NaN (perspective projection of points on
  // or near the camera plane). The unprojected box is a safe fallback.
  const W_EPSILON = 1e-12;
  let skippedCorners = 0;
  let transformedCorners = 0;
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  for (let cornerIndex = 0; cornerIndex < 8; cornerIndex++) {
    const x = (cornerIndex & 1) === 0 ? inputMinX : inputMaxX;
    const y = (cornerIndex & 2) === 0 ? inputMinY : inputMaxY;
    const z = (cornerIndex & 4) === 0 ? inputMinZ : inputMaxZ;
    const w = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
    if (Math.abs(w) < W_EPSILON) {
      skippedCorners++;
      continue;
    }
    const transformedX = (matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]) / w;
    const transformedY = (matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]) / w;
    const transformedZ = (matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]) / w;
    if (transformedX < minX) minX = transformedX;
    if (transformedY < minY) minY = transformedY;
    if (transformedZ < minZ) minZ = transformedZ;
    if (transformedX > maxX) maxX = transformedX;
    if (transformedY > maxY) maxY = transformedY;
    if (transformedZ > maxZ) maxZ = transformedZ;
    transformedCorners++;
  }

  if (skippedCorners > 0) {
    log.warning(
      Modules.SCENE_MANAGER,
      `transformBoundingBox: skipped ${skippedCorners}/8 corner(s) with |w| < ${W_EPSILON} (degenerate perspective projection)`
    );
  }

  // If every corner was degenerate, fall back to the input box rather than
  // returning (Infinity, -Infinity).
  if (transformedCorners === 0) {
    target.min.x = inputMinX;
    target.min.y = inputMinY;
    target.min.z = inputMinZ;
    target.max.x = inputMaxX;
    target.max.y = inputMaxY;
    target.max.z = inputMaxZ;
    return target;
  }
  target.min.x = minX;
  target.min.y = minY;
  target.min.z = minZ;
  target.max.x = maxX;
  target.max.y = maxY;
  target.max.z = maxZ;
  return target;
}

/**
 * Project nD scene bounds (per-dimension min/max arrays) onto the
 * three displayed dimensions to produce a 3D BoundingBox.
 *
 * Pulled out of `scene-manager.getSceneBoundsFromMetadata` so the
 * mapping is testable without a populated THREE.Scene. The mapping
 * rules are:
 *   - displayDims maps positions [0..2] to the X / Y / Z axes.
 *   - When fewer than three displayDims are supplied, the unmapped
 *     axes default to 0 (a degenerate bounding box on that axis).
 *   - When a displayDim index is out of range for the supplied
 *     bounds arrays, that axis also stays at 0 — same defensive
 *     fallback the inline code uses to avoid OOB reads.
 *
 * @param minBounds - Per-dimension min values (length = ndim).
 * @param maxBounds - Per-dimension max values (length = ndim).
 * @param displayDims - Up to 3 indices into the per-dimension arrays
 *   identifying which dimensions map to X / Y / Z.
 */
export function projectBoundsToDisplayDims(
  minBounds: readonly number[],
  maxBounds: readonly number[],
  displayDims: readonly number[]
): BoundingBox {
  const min3D = { x: 0, y: 0, z: 0 };
  const max3D = { x: 0, y: 0, z: 0 };
  const axes = ['x', 'y', 'z'] as const;

  for (let i = 0; i < Math.min(3, displayDims.length); i++) {
    const dim = displayDims[i];
    if (dim < minBounds.length && dim < maxBounds.length) {
      min3D[axes[i]] = minBounds[dim];
      max3D[axes[i]] = maxBounds[dim];
    }
  }

  return { min: min3D, max: max3D };
}
