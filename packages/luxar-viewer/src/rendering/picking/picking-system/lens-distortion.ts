/**
 * Brown–Conrady lens-distortion math for the picking system.
 *
 * TypeScript port of the GLSL `applyDistortion` function in
 * `rendering/post-processing/mega/shader.glsl.ts:117-128` (and its
 * TSL counterpart in `mega/shader.tsl.ts`). Uses the green-channel
 * distortion (the reference channel — no chromatic offset).
 *
 * Maps from distorted screen space back into undistorted source space.
 * The picking system applies it to mouse coords so they index into the
 * undistorted pick buffer correctly when post-processing distortion is
 * active.
 *
 * Extracted from `PickingSystem` so the parity contract with the
 * shader can be unit-tested directly. The caller supplies a scratch
 * `{ x, y }` so the hot path stays allocation-free.
 *
 * @module rendering/picking/picking-system/lens-distortion
 */

import type * as THREE from 'three';

/** Brown–Conrady lens parameters consumed by {@link applyLensDistortion}. */
export interface LensDistortionParams {
  /** Radial distortion coefficient per axis. (0, 0) is identity. */
  distortion: THREE.Vector2;
  /** Image principal point in normalised UV space ([-1, 1]). */
  principalPoint: THREE.Vector2;
  /** Camera intrinsic focal length per axis. */
  focalLength: THREE.Vector2;
  /** Camera intrinsic skew (typically 0 for axis-aligned sensors). */
  skew: number;
}

/**
 * Mutable 2-vector container used to return UV results without per-call
 * allocations. Callers reuse the same object across many invocations.
 */
export interface UVScratch {
  x: number;
  y: number;
}

/**
 * Apply Brown–Conrady distortion to UV coordinates and write the result
 * into `out`. Returns `out` for ergonomic chaining.
 *
 * Y-convention contract (MED-27): this function IS the canonical
 * distortion map, defined in TOP-DOWN uv space — the picking-system's
 * screen convention (`event.clientY / height`, y=0 at the top). The
 * TSL mega shader consumes it natively (WebGPURenderer's fullscreen
 * triangle delivers top-down uv). The GLSL mega shader runs only under
 * WebGLRenderer, whose fullscreen triangle delivers BOTTOM-UP uv — it
 * therefore applies the exact y-flip conjugation internally by
 * negating the two odd-symmetry intrinsics (principalPoint.y and
 * skew) in its K matrix rather than flipping uv. Radial/focal terms
 * are flip-even, which is why the historical mismatch was invisible
 * with the default (ppy = skew = 0) presets and why textual-identity
 * parity tests could not catch it. The conjugation-identity test in
 * `lens-distortion.test.ts` pins the relationship: do NOT "re-sync"
 * the GLSL matrix verbatim to this formula.
 *
 * Algorithm (must stay in sync with `mega/shader.glsl.ts::applyDistortion`):
 *   1. UV ∈ [0, 1] → normalised n ∈ [-1, 1].
 *   2. Radial distortion: `xd = (1 + k_x · r²) · xn` (likewise y).
 *   3. Apply intrinsic matrix K with skew, principal point, focal length.
 *   4. Map back into UV ∈ [0, 1].
 */
export function applyLensDistortion(
  u: number,
  v: number,
  params: LensDistortionParams,
  out: UVScratch
): UVScratch {
  // UV [0, 1] → normalised [-1, 1]
  const xn = 2.0 * (u - 0.5);
  const yn = 2.0 * (v - 0.5);

  // Brown–Conrady radial distortion: r' = r · (1 + k · r²)
  const r2 = xn * xn + yn * yn;
  const xd = (1.0 + params.distortion.x * r2) * xn;
  const yd = (1.0 + params.distortion.y * r2) * yn;

  // K × distorted point → back to [0, 1] UV
  const fx = params.focalLength.x;
  const fy = params.focalLength.y;

  out.x = (fx * xd + params.skew * fx * yd + params.principalPoint.x) * 0.5 + 0.5;
  out.y = (fy * yd + params.principalPoint.y) * 0.5 + 0.5;
  return out;
}
