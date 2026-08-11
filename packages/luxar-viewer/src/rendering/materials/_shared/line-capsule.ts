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
 * Joints: EVERY end is a round cap; a partner-bearing interior end keeps
 * its HALF of the joint disc — the cap region (beyond the endpoint) is
 * partitioned along the joint's 2D BISECTOR line (normal =
 * normalize(q̂ − m̂) in pixel space, my side negative; the partner's normal
 * is the exact negation, so the two half-discs tile the disc exactly at any
 * bend angle). Which ends cut at all is the SHARED joint-code rule
 * (`luxarLineJointCapSuppression`): a free end and a degree-≥3 hub keep the
 * whole round cap — a hub has no single partner to tile against — while a
 * slice-clipped end is butt-cut. The cut is HARD and spans the FULL joint
 * plane (cap and body): exact tiling, zero overlap, zero double-count —
 * the same domain partition the volumetric primitive integrates per ray.
 * (A soft foreign-side fade was shipped briefly and reverted: any nonzero
 * foreign-side contribution double-counts over the partner's body, which
 * reads as a bright wedge at every joint once zoomed. The hard cut's
 * residual cost is a subtle brightness gradient along the cut at EXTREME
 * oblique zoom, where the two legs' apparent radii genuinely diverge —
 * 2D-intrinsic; only 3D resolves it.) The partner's far endpoint is
 * near-plane-clipped toward the joint vertex before projecting (a
 * behind-eye projection flips and poisons the cut normal).
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
 * Congruence gate for the joint DEFICIT rule: when the partner's far
 * radius is within this fraction of my end radius (deficit ≈ 0 — equal
 * widths, no meaningful perspective divergence), the joint packet stays
 * empty and the fragment takes the cheap hard-cut discard instead of
 * evaluating the partner's field. Congruent joints are the overwhelmingly
 * common case (and the fill-heavy benchmarks' only case), so this keeps
 * the deficit rule's cost confined to the joints that actually need it.
 */
export const CAPSULE_JOINT_DEFICIT_GATE = 0.02;

/**
 * Width gate for the joint packet: below this stencil half-width (px) the
 * joint disc is a few pixels and a deficit is invisible, so the vertex
 * stage skips the packet math entirely (hairline scenes pay nothing for
 * the deficit rule — its cost stays confined to visibly wide joints).
 */
export const CAPSULE_JOINT_PACKET_MIN_RADIUS_PX = 4.0;

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
