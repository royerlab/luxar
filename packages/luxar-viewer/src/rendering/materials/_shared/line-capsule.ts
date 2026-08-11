/**
 * Capsule line primitive — shared constants + CPU reference (issue #1352).
 *
 * The capsule primitive draws a segment as a gaussian-like profile of the
 * 2D POINT-TO-SEGMENT DISTANCE in pixel space, evaluated on interpolated
 * stencil-local coordinates. It exists because the screen-space quad's
 * near-axial behaviour is pathological (direction instability — the #790
 * sliver class) exactly where a distance-based profile is stable by
 * construction: end-on, the segment projects to a point and the profile
 * becomes a radial disc; the ribbon↔disc transition is continuous because
 * point-to-segment distance is continuous in the endpoints.
 *
 * Three deliberate exactness relaxations (Loic, 2026-08-11 — "lines don't
 * have to be physics-exact, they need to behave reasonably: no pathological
 * near-axial drawing; any mathematically/computationally easier profile
 * that roughly looks gaussian"):
 *
 * 1. **Quartic profile, not a gaussian.** `(1 − p²)^n` with `p = d/r` and
 *    the default knob at n = 2 — within ~12% of `exp(−2p²)` everywhere,
 *    exactly 0 at the rim (compact support, no floor constants, no `exp`).
 * 2. **2σ support** (the trim decision): the drawn radius is 2σ of the
 *    equivalent gaussian, i.e. `2/T` of the legacy quad half-width
 *    (T = GAUSSIAN_EQUIVALENT_TRUNCATION ≈ 3.03σ, the 1%-of-peak floor).
 * 3. **Attributes interpolate across the stencil** (colour/alpha/width/
 *    sharpness as varyings evaluated at clamped corner positions) — the
 *    blend stretches marginally into cap regions; sub-quantization.
 *
 * Joints: EVERY end is a round cap; an interior end keeps its HALF of the
 * joint disc — the cap region (beyond the endpoint) is partitioned along
 * the joint's 2D BISECTOR line (normal = normalize(q̂ − m̂) in pixel
 * space, my side negative; the partner's normal is the exact negation, so
 * the two half-discs tile the disc exactly at any bend angle). The cut is
 * confined to the cap region: where the two rod BODIES genuinely overlap
 * (the inner corner of a bend) both legs render, matching the physical
 * union. The foreign-side cap contribution fades (smoothstep,
 * bend-scaled — see `CAPSULE_CUT_FADE_RADIUS_FRACTION`) instead of
 * cutting hard, so the hand-off to the partner's body is C0 — sub-pixel
 * at normal widths, smooth when zoomed in. The partner's far endpoint is
 * near-plane-clipped toward the joint vertex before projecting (a
 * behind-eye projection flips and poisons the cut normal), and a joint
 * vertex behind the near plane keeps the perpendicular butt.
 *
 * @module rendering/materials/_shared/line-capsule
 */

import { GAUSSIAN_EQUIVALENT_TRUNCATION } from './falloff';

/** Drawn capsule radius in σ units of the equivalent gaussian (the 2σ trim). */
export const CAPSULE_SUPPORT_SIGMA = 2.0;

/**
 * Drawn capsule radius as a fraction of the legacy quad half-width
 * (which spans T·σ): 2/T ≈ 0.659. Multiplies `width × scale` in the vertex.
 */
export const CAPSULE_RADIUS_PER_QUAD_HALFWIDTH =
  CAPSULE_SUPPORT_SIGMA / GAUSSIAN_EQUIVALENT_TRUNCATION;

/** Minimum drawn radius in pixels (matches the quad's 1.5 px AA floor). */
export const CAPSULE_MIN_RADIUS_PX = 1.5;

/** Stencil AA apron beyond the profile support, in pixels. */
export const CAPSULE_STENCIL_APRON_PX = 0.5;

/**
 * Foreign-side cap fade length as a fraction of the end radius: my cap
 * region on the PARTNER's side of the joint bisector fades out
 * (smoothstep) over `fraction × max(|n.y|, 0.25) × radius` of axial
 * overhang instead of a hard cut — C0 with the partner's body at its
 * endpoint line (no chevron edge when zoomed) and with my own half-disc
 * at the bisector. The |n.y| bend scaling keeps a near-straight joint's
 * fade (a genuine double-count band — the partner's body already covers
 * there) short, while a real bend fades exactly where the two legs'
 * apparent radii genuinely diverge in 2D; the 0.25 floor keeps the fade
 * from collapsing to a hard seam at shallow projected bends. Sub-pixel
 * at normal widths. The joint stencil reach accounts for the fade band
 * (`(|n.y| + fraction·max(|n.y|, 0.25))·rMax`).
 */
export const CAPSULE_CUT_FADE_RADIUS_FRACTION = 0.25;

/**
 * Sharpness-knob → profile exponent map: `n = 2^(3 − 4s)`. In `(1 − p²)^n`
 * space SMALLER exponents are boxier, so the map runs opposite to the
 * gaussian-family β: s = 0 → n = 8 (spiky), s = 0.5 → n = 2 (the default
 * quartic), s = 1 → n = 0.5 (boxy).
 */
export function capsuleProfileExponent(sharpKnob: number): number {
  return Math.pow(2, 3 - 4 * sharpKnob);
}

/**
 * CPU reference of the capsule radial profile: `max(1 − p², 0)^n` with
 * `p = distance / radius` — what both shader backends evaluate per
 * fragment (the default knob takes the `w·w` fast path, no `pow`).
 */
export function capsuleProfile(p: number, sharpKnob = 0.5): number {
  const w = Math.max(1 - p * p, 0);
  return Math.pow(w, capsuleProfileExponent(sharpKnob));
}
