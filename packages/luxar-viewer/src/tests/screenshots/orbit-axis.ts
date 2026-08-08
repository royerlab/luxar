/**
 * Orbit axis — the pure CLASSIFICATION half of the gallery harness's rock-axis
 * derivation.
 *
 * `generate-gallery.spec.ts` owns the IO (read `camera.up` out of the live page,
 * which is the only place that camera exists); this module owns the *decision*:
 * which signed world axis that vector points along. Splitting them lets the
 * decision be unit-tested against plain vectors with no browser, dataset or GPU
 * — the same split, for the same reason, as `exposure-policy.ts`,
 * `crop-policy.ts` and `frame-similarity.ts`.
 *
 * That split is worth making here specifically because this is the *fix* for
 * #1377, not merely its detector: the orbit hard-sets `cam.up` from the axis
 * this returns on every frame, so getting it wrong ships an animation rolled
 * away from its own poster. The harness is a manual media run and is not on CI,
 * so nothing else would catch a regression here until the next capture.
 */

/** A world axis plus the direction along it. */
export type SignedUpAxis = { axis: 'x' | 'y' | 'z'; sign: 1 | -1 };

/**
 * What an unusable up-vector falls back to: +Y, the historical default the rock
 * axis had before it was derived at all, and three.js's own default camera up.
 */
export const FALLBACK_UP_AXIS: SignedUpAxis = { axis: 'y', sign: 1 };

/**
 * The SIGNED world axis a camera up-vector most nearly points along.
 *
 * The rock axis has always been axis-aligned, so a non-axis-aligned up is
 * snapped to its dominant component rather than rejected. Ties resolve x → z →
 * y, which only matters for vectors no real scene bakes.
 *
 * The DIRECTION of that component is kept, not just which axis it is: a baked up
 * of (0,0,-1) has to stay -Z, because the orbit hard-sets `cam.up` and snapping
 * it to +Z would roll the animation 180° from the still — the exact divergence
 * this derivation exists to remove. Only `cam.up` is affected; the rock geometry
 * is sign-invariant (the out-of-plane offset enters as `compU·U`).
 *
 * @param up The camera's up-vector, or `null`/`undefined` when the page has no
 *   camera to read yet.
 * @returns The dominant signed axis, or {@link FALLBACK_UP_AXIS} when `up` is
 *   missing or degenerate (zero-length, or carrying a NaN/infinite component).
 *   Falling back rather than guessing matters: the `>=` chain would otherwise
 *   answer 'x' for an all-zero vector, which is an arbitrary axis dressed up as
 *   a measurement.
 */
export function dominantSignedAxis(
  up: { x: number; y: number; z: number } | null | undefined
): SignedUpAxis {
  if (!up) return FALLBACK_UP_AXIS;
  const { x, y, z } = up;
  const ax = Math.abs(x);
  const ay = Math.abs(y);
  const az = Math.abs(z);
  const total = ax + ay + az;
  if (!Number.isFinite(total) || !(total > 0)) return FALLBACK_UP_AXIS;
  if (ax >= ay && ax >= az) return { axis: 'x', sign: x < 0 ? -1 : 1 };
  if (az >= ay) return { axis: 'z', sign: z < 0 ? -1 : 1 };
  return { axis: 'y', sign: y < 0 ? -1 : 1 };
}
