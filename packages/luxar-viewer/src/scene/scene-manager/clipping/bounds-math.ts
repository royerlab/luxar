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
 * Calculates optimal camera distance to fit bounding box in view
 *
 * @param box - Bounding box to fit
 * @param camera - Camera configuration
 * @param fitRatio - How much of the view to fill (0-1, default 0.75)
 * @returns Optimal camera distance from center
 */
export function calculateCameraDistance(
  box: BoundingBox,
  camera: CameraConfig,
  fitRatio: number = config.scene.defaultFitRatio
): number {
  const maxDim = getBoundingBoxMaxDimension(box);

  // Calculate distance based on FOV
  const fovRadians = (camera.fov * Math.PI) / 180;
  const halfFov = fovRadians / 2;

  // Consider aspect ratio to ensure object fits in both dimensions
  const verticalFit = maxDim / fitRatio / (2 * Math.tan(halfFov));
  const horizontalFit = maxDim / fitRatio / (2 * Math.tan(halfFov) * camera.aspect);

  // Use the larger distance to ensure complete fit
  return Math.max(verticalFit, horizontalFit) * 1.2; // Add 20% margin
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
 * scenes only). For any real scene the DEPTH-PRECISION floor from
 * {@link nearPlaneFloor} dominates — see the rationale there.
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
 * A BACKSTOP only. It bounds `near` away from zero at any scene scale,
 * but it says nothing about depth-buffer precision — the operative
 * floor is {@link nearPlaneFloor}, which additionally bounds the
 * near/far RATIO. Do not clamp to this directly; call
 * `nearPlaneFloor` so both clipping paths inherit the same bound.
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
 * Why 1000 specifically. A LARGER C means a smaller floor: less
 * precision, but less clipped. Two constraints put a floor under C, and
 * only a soft preference (keep precision) pushes from above — so C wants
 * to be the smallest value satisfying both:
 *
 *  - **C > 551, don't clip the zoom target.** At maximum zoom-in the
 *    orbit target sits at `minDistance = 1e-3 · diagonal`
 *    (`controls.scaleMultipliers`) = `1.905e-3 · R`, while `far ≈ 1.05 · R`.
 *    Keeping it in front of the near plane needs `1.05 R / C < 1.905e-3 R`.
 *  - **C ≥ 992, stay lossless for Points / Lines / GSplats.** All three
 *    already suppress anything closer than `nearCull = 1e-3 · diagonal`
 *    via `perspectiveNearFade` (`materials/_shared/glsl-lib.ts`), though
 *    by two different mechanisms worth knowing before trusting this:
 *    Points and GSplats REJECT the vertex when the fade drops below 0.01
 *    (both backends); Lines instead cull only when BOTH endpoints are
 *    near, clip a half-near segment onto the `nearCull` plane, and apply
 *    the fade PER-FRAGMENT as a multiply. Either way the fade is what
 *    governs, and it is under 0.01 below `1.0582 · nearCull`. The worst
 *    case is the camera ON the sphere surface, where `far = 2R` and the
 *    floor is largest relative to `nearCull`:
 *    `2R / C ≤ 1.0582 · 1.905e-3 · R` ⟹ `C ≥ 992`. At C = 1000 the
 *    frustum cut lands where the fade is 0.007, so the most any of the
 *    three loses is fragments carrying <1% of their intensity.
 *
 * 992 binds. C = 1000 clears it by **0.8%** — deliberately thin, because
 * every extra unit of C is depth precision given away, and the margin is
 * *enforced* rather than trusted: the "clipped band is already
 * shader-rejected" property in `bounds-math.property.test.ts` fails if a
 * future change to `nearCull`, to the fade band, or to this constant eats
 * it. (Mesh has no near fade in either backend — verified in the GLSL and
 * TSL sources — so it is the one type the floor can clip; there is no
 * value of C that avoids that while still bounding the ratio.)
 *
 * Measured payoff at the reported pose (R = 52.5, dist = 8.5, far = 61):
 * the floor rises 1.05e-4 → 0.061 and depth quantization improves
 * 4.10e-2 → 7.05e-5 world units, i.e. **581x** finer.
 *
 * PERSPECTIVE ONLY — see the `boundNearFarRatio` parameter of
 * {@link nearPlaneFloor}. An orthographic projection maps eye depth
 * LINEARLY to the depth buffer, so its resolution is
 * `(far - near) / 2²⁴` regardless of `near`: the ratio bound buys ortho
 * nothing, while still clipping a slab in front of the eye that ortho
 * (unlike perspective) really does draw — `perspectiveNearFade` returns
 * 1.0 for ortho, so ALL FOUR geometry types render up to `near` there.
 * Measured on a diagonal-100 scene at the deepest legal orbit distance:
 * applying the bound under ortho clips 52.7% of the eye-to-target depth
 * versus 0.105% without it, and changes depth resolution by 0.1%.
 */
export const MAX_NEAR_FAR_RATIO = 1000;

/**
 * The near-plane floor both clipping paths clamp to: the scale-aware
 * backstop, raised to whatever `far / {@link MAX_NEAR_FAR_RATIO}`
 * demands for depth-buffer precision.
 *
 * Scale-invariant by construction — the bound is derived from `far`,
 * which is itself scene-scaled — so it keeps the tiny-scene guarantee
 * `minNearForRadius` was introduced for (#573) without the Z-precision
 * cost: on a diagonal-0.1 scene at maximum zoom-in, `far / 1000` is
 * ~5.3e-5 while the closest reachable orbit distance is ~1.7e-4, so
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
export function transformBoundingBox(box: BoundingBox, matrix: number[]): BoundingBox {
  // Get 8 corners of the box
  const corners = [
    { x: box.min.x, y: box.min.y, z: box.min.z },
    { x: box.max.x, y: box.min.y, z: box.min.z },
    { x: box.min.x, y: box.max.y, z: box.min.z },
    { x: box.max.x, y: box.max.y, z: box.min.z },
    { x: box.min.x, y: box.min.y, z: box.max.z },
    { x: box.max.x, y: box.min.y, z: box.max.z },
    { x: box.min.x, y: box.max.y, z: box.max.z },
    { x: box.max.x, y: box.max.y, z: box.max.z },
  ];

  // Transform each corner, skipping any whose homogeneous w is ~0 to avoid
  // dividing through to ±Infinity/NaN (perspective projection of points on
  // or near the camera plane). The unprojected box is a safe fallback.
  const W_EPSILON = 1e-12;
  let skippedCorners = 0;
  const transformedCorners = corners
    .map((corner) => {
      const w = matrix[3] * corner.x + matrix[7] * corner.y + matrix[11] * corner.z + matrix[15];
      if (Math.abs(w) < W_EPSILON) {
        skippedCorners++;
        return null;
      }
      return {
        x: (matrix[0] * corner.x + matrix[4] * corner.y + matrix[8] * corner.z + matrix[12]) / w,
        y: (matrix[1] * corner.x + matrix[5] * corner.y + matrix[9] * corner.z + matrix[13]) / w,
        z: (matrix[2] * corner.x + matrix[6] * corner.y + matrix[10] * corner.z + matrix[14]) / w,
      };
    })
    .filter((p): p is { x: number; y: number; z: number } => p !== null);

  if (skippedCorners > 0) {
    log.warning(
      Modules.SCENE_MANAGER,
      `transformBoundingBox: skipped ${skippedCorners}/8 corner(s) with |w| < ${W_EPSILON} (degenerate perspective projection)`
    );
  }

  // If every corner was degenerate, fall back to the input box rather than
  // returning (Infinity, -Infinity).
  if (transformedCorners.length === 0) {
    return { min: { ...box.min }, max: { ...box.max } };
  }

  // Find new min/max
  let minX = Infinity,
    minY = Infinity,
    minZ = Infinity;
  let maxX = -Infinity,
    maxY = -Infinity,
    maxZ = -Infinity;

  for (const point of transformedCorners) {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    minZ = Math.min(minZ, point.z);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
    maxZ = Math.max(maxZ, point.z);
  }

  return {
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
  };
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
