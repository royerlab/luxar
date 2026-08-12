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
 * slice-clipped end is butt-cut. The cut spans the FULL joint plane (cap
 * and body) and composes by the DEFICIT rule over a 1 px AA RAMP: each
 * leg's fragment evaluates the plane in its own local frame, so a hard
 * step speckles — complementary ramps sum to exactly 1 and anti-alias the
 * cut for free — and on the partner's side each leg renders
 * max(mine − partner, 0), so the additive pair composes to
 * max(mine, partner). For congruent legs the congruence gate
 * (`CAPSULE_JOINT_DEFICIT_GATE`) keeps the packet empty and the cut is an
 * exact zero-double-count partition — the same domain partition the
 * volumetric primitive integrates per ray. Where the partner tapers away
 * or its apparent radius diverges under perspective, the deficit term
 * contributes exactly the light a pure partition would chop (a fat
 * vertex's disc keeps the half a thin neighbour cannot render). The
 * partner's field is rebuilt per fragment from the cut varying's packed
 * radius gradient + projected length and the shared-vertex radius
 * (`vREnd`, #1494), with the far cap closing the rod (#1490); the stencil
 * reserves the full disc when a packet exists, the deficit being bounded
 * by my own profile (#1488). The partner's far endpoint is
 * near-plane-clipped toward the joint vertex before projecting (a
 * behind-eye projection flips and poisons the cut normal). A CPU model of
 * this composition lives at the bottom of this file; the unit sweep
 * asserts the rendered pair tracks max(mine, partner).
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
 * Taper tolerance for the joint packet gate. The gate opens (building a
 * packet so the fragment runs the DEFICIT rule instead of the cheap hard
 * cut) when ANY of four conditions defeats the hard cut's assumption
 * that the partner covers my foreign side: the legs' radii differ by
 * more than this fraction (either direction), my own leg widens away
 * from the joint by more than it, the partner is shorter than twice my
 * joint radius, or the turn is sharper than 120° (#1495, #1501). Long,
 * congruent, gentle joints — the overwhelmingly common case and the
 * fill benchmarks' only case — keep the exact zero-cost partition.
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
 * Hairline cut gate (px): below this apparent radius the whole joint
 * apparatus — partner fetch, bisector construction, fragment partition —
 * is skipped and interior ends keep plain round caps. The artifact this
 * machinery prevents (a double-bright additive overlap at the joint) is
 * SUB-PIXEL for lines this thin (at or below the AA radius floor, where
 * widths are already fattened and energy-compensated), and the gate is
 * self-correcting: zoom in and the radius crosses it, re-enabling the
 * exact partition. Slice-clipped ends (joint code -1) are NOT gated —
 * their perpendicular butt is a correctness rule (nothing may draw past
 * the slice plane), not a visual nicety. Measured on the 10M hairline
 * scenario the full apparatus cost ~20% of the capsule's GPU time
 * (see perf-results/1352-campaign, recovery-round decomposition).
 */
export const CAPSULE_JOINT_CUT_MIN_RADIUS_PX = 1.5;

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

/**
 * ============================================================
 * CPU reference of the JOINT COMPOSITION — mirrors the fragment
 * shaders' joint math exactly (vR's whole-stencil interpolation, the
 * 1 px AA ramp, the deficit rule with the packed gradient/length and
 * the shared-vertex radius). Exists for the numeric composition test:
 * the two legs' rendered sum must track max(mine, partner) — the three
 * #1487-review defects (#1494/#1488/#1490) were all invisible to
 * source-substring pins and all visible to this sweep.
 * ============================================================
 */

/** One leg of a joint, in a shared 2D px frame with the joint at the origin. */
export interface CapsuleJointLeg {
  /** Unit direction from the joint vertex toward the leg's far end. */
  readonly dir: readonly [number, number];
  /** Radius at the joint vertex, px. */
  readonly rJoint: number;
  /** Radius at the far end, px. */
  readonly rFar: number;
  /** Leg length (projected), px. */
  readonly length: number;
}

function legLocal(leg: CapsuleJointLeg, px: number, py: number): [number, number] {
  const [dx, dy] = leg.dir;
  return [px * dx + py * dy, -px * dy + py * dx];
}

/** The leg's own UNCUT field at a point — the composition's ideal term. */
export function capsuleLegField(leg: CapsuleJointLeg, px: number, py: number): number {
  const [x, y] = legLocal(leg, px, py);
  // EXACT per-fragment radius, as the shaders compute it: the endpoint
  // mix clamped to the segment span (never the cap extensions).
  const t = Math.min(Math.max(x / Math.max(leg.length, 1e-4), 0), 1);
  const r = Math.max(leg.rJoint + (leg.rFar - leg.rJoint) * t, 1e-4);
  const ox = Math.max(Math.max(-x, x - leg.length), 0);
  const q = (y * y + ox * ox) / (r * r);
  return capsuleProfile(Math.sqrt(Math.max(q, 0)));
}

/**
 * One leg's RENDERED contribution at a point, joint machinery included —
 * the mirror of the fragment shader (default sharpness knob).
 */
export function capsuleJointRenderLeg(
  leg: CapsuleJointLeg,
  partner: CapsuleJointLeg,
  px: number,
  py: number
): number {
  const [x, y] = legLocal(leg, px, py);
  let profile = capsuleLegField(leg, px, py);
  if (profile <= 0) return 0;

  // Cut normal: n ∝ q̂ − m̂ in MY local frame (m̂ = +x̂ at the joint end).
  const [qx, qy] = legLocal(leg, partner.dir[0], partner.dir[1]);
  const nRaw: [number, number] = [qx - 1, qy];
  const nl = Math.hypot(nRaw[0], nRaw[1]);
  if (nl <= 1e-3) return profile; // hairpin fallback: plain cap
  // No quantisation of the normal (#1502): each leg would snap in its
  // OWN (u, v) basis, so the rounding does not cancel — it injects
  // ~7e-4 rad of disagreement between two planes that must be exact
  // complements, and with the cut spanning the full stencil that error
  // scales with DISTANCE along the rod (±0.35 of peak on long
  // doubled-back polylines). Unsnapped, the two normals are exact
  // negations in exact arithmetic; the residual float noise is what the
  // 1 px AA ramp absorbs.
  const nx = nRaw[0] / nl;
  const ny = nRaw[1] / nl;

  // Packet per the vertex stage (width gate assumed passed; callers use
  // radii above CAPSULE_JOINT_PACKET_MIN_RADIUS_PX).
  const rpFar = partner.rFar;
  // Packet gate (#1495, #1501): either leg tapering (both directions),
  // a short partner, or a near-hairpin turn defeats the hard cut's
  // assumption that the partner covers my foreign side. The angle clause
  // is NOT redundant with the length clause: at a hairpin the bisector
  // tilts toward my axis and splits my rod LENGTHWISE, so a partner in
  // the 2r–3r length band (long enough to pass the length clause,
  // shorter than my leg) refills only part of the cut half — measured to
  // −0.92 of peak without this clause. Each leg's gate stands alone (the
  // partner may be width-gated off), hence the own-widening clause too.
  const hasPacket =
    Math.abs(1 - rpFar / Math.max(leg.rJoint, 1e-4)) > CAPSULE_JOINT_DEFICIT_GATE ||
    leg.rFar > leg.rJoint * (1 + CAPSULE_JOINT_DEFICIT_GATE) ||
    partner.length < 2 * leg.rJoint ||
    qx > 0.5;
  const g = (rpFar - leg.rJoint) / partner.length;

  const side = nx * x + ny * y;
  if (side <= -0.5) return profile;
  const cover = Math.min(Math.max(0.5 - side, 0), 1);
  let def = 0;
  if (hasPacket) {
    // Reflected partner axis: q = m − 2(m·n)n with m = (1, 0).
    const proj = nx;
    const qdx = 1 - 2 * proj * nx;
    const qdy = -2 * proj * ny;
    const xp = x * qdx + y * qdy;
    const yp2 = Math.max(x * x + y * y - xp * xp, 0);
    const rp = Math.max(leg.rJoint + g * Math.min(Math.max(xp, 0), partner.length), 1e-4);
    const op = Math.max(Math.max(-xp, xp - partner.length), 0);
    const qp = (yp2 + op * op) / (rp * rp);
    const partnerP = capsuleProfile(Math.sqrt(Math.max(qp, 0)));
    def = Math.max(profile - partnerP, 0);
  }
  return profile * cover + def * (1 - cover);
}

/**
 * Sweep the joint neighbourhood and return the worst signed deviations of
 * (leg1 + leg2 rendered) − max(field1, field2), in units of peak profile.
 */
export function capsuleJointCompositionError(
  leg1: CapsuleJointLeg,
  leg2: CapsuleJointLeg,
  extent?: number,
  step = 0.5
): { minErr: number; maxErr: number } {
  // The window must cover BOTH rods end to end: a hairpin cut splits a
  // rod lengthwise, so its damage can sit anywhere along the LONGER leg
  // — a joint-sized window measured −0.02 where the true worst was −0.92
  // (#1501's discovery path).
  const auto =
    Math.max(leg1.length, leg2.length) +
    Math.max(leg1.rJoint, leg1.rFar, leg2.rJoint, leg2.rFar) +
    5;
  const ext = extent ?? auto;
  let minErr = 0;
  let maxErr = 0;
  for (let px = -ext; px <= ext; px += step) {
    for (let py = -ext; py <= ext; py += step) {
      const sum =
        capsuleJointRenderLeg(leg1, leg2, px, py) + capsuleJointRenderLeg(leg2, leg1, px, py);
      const ref = Math.max(capsuleLegField(leg1, px, py), capsuleLegField(leg2, px, py));
      const err = sum - ref;
      if (err < minErr) minErr = err;
      if (err > maxErr) maxErr = err;
    }
  }
  return { minErr, maxErr };
}
