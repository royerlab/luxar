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
 * by my own profile, so a half-disc reach there would leave the
 * rasterizer chopping the very light the deficit rule adds back (#1488).
 * The partner's far endpoint is near-plane-clipped toward the joint vertex
 * before projecting (a behind-eye projection flips and poisons the cut
 * normal). A CPU model of this composition — the joint-end STENCIL REACH
 * included, since a shortfall there is invisible to a model that evaluates
 * the profile everywhere — lives at the bottom of this file; the unit sweep
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
 * the shared-vertex radius) AND the vertex stage's joint-end stencil reach,
 * which bounds where any of it can run. Exists for the numeric composition test:
 * the two legs' rendered sum must track max(mine, partner) — the three
 * #1487-review defects (#1494/#1488/#1490) were all invisible to
 * source-substring pins and all visible to this sweep.
 * ============================================================
 */

/**
 * One leg of a joint, in a shared 2D px frame with the joint at the origin.
 *
 * Both radii are the ALREADY-CLAMPED drawn radii the vertex stage works in —
 * `clamp(width · scale · RADIUS_FACTOR, CAPSULE_MIN_RADIUS_PX,
 * uMaxLinePixelWidth)`, applied before any joint math runs. Worth stating
 * because these are what the fragment actually interpolates and measures
 * distance against, which is what lets the packet's WIDTH gate compare them
 * to an absolute pixel threshold at all. (Not a live hazard: an unclamped
 * radius would only move that decision if `uMaxLinePixelWidth` were ≤ 3.5,
 * since the 1.5 px AA floor already yields rMax 2.0, under the 4 px gate.)
 */
export interface CapsuleJointLeg {
  /** Unit direction from the joint vertex toward the leg's far end. */
  readonly dir: readonly [number, number];
  /** Clamped drawn radius at the joint vertex, px. */
  readonly rJoint: number;
  /** Clamped drawn radius at the far end, px. */
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
 * Axial stencil reach beyond a CUT joint end, px — TWO of the vertex stage's
 * four `ext` arms (`ny` = the cut normal's PERPENDICULAR component in the
 * leg's own local frame).
 *
 * Without a packet the fragment renders only the kept half-disc, whose axial
 * extent is |ny|·rMax — a genuinely cheaper stencil than a cap, which is the
 * whole reason joints cost less fill than free ends. With one, the deficit
 * term reaches wherever MY OWN profile does (it renders max(mine − partner,
 * 0) ≤ mine), so the end must reserve the FULL disc or the rasterizer chops
 * the light the deficit rule draws (#1488). Both carry the AA apron.
 *
 * The two arms it does NOT model: the bare-apron BUTT an interior end keeps
 * when there is no partner-far texel, the joint vertex is behind the near
 * plane, or `nLoc.x` fails its sign test; and the free-end/hairpin `rMax`,
 * which the model hardcodes at its own call sites and so cannot be reached
 * through an injected rule. The first two of those are unproducible here (the
 * model has no texels and no near plane), but the `nLoc.x` one merely goes
 * unexercised: with θ the angle between q̂ and m̂, `nl = 2|sin(θ/2)|` and
 * `nLoc.x = −|sin(θ/2)|`, so `nl > 1e-3 ∧ nLoc.x > −1e-3` holds for θ in
 * (0.057°, 0.115°) — a sliver just above the hairpin fallback, which
 * `leg(180, …)` vs `leg(180.08, …)` would enter. No row in the sweep does.
 */
export function capsuleJointStencilReach(rMax: number, ny: number, hasPacket: boolean): number {
  return (hasPacket ? rMax : Math.abs(ny) * rMax) + CAPSULE_STENCIL_APRON_PX;
}

/**
 * A joint-end reach rule, injectable purely so a test can substitute a WRONG
 * one. Under the shipped rule the clip is exactly inert — every sweep row
 * measures identically with and without it — so deleting the clip leaves the
 * suite green and the line reads as dead code to the next reader. The
 * negative control (`line-capsule.test.ts`) feeds the pre-#1488 half-disc
 * rule through here and asserts the sweep CHOPS, which is what makes the
 * clip's presence observable.
 *
 * Exported deliberately, despite having one in-repo caller: the docstrings
 * above `{@link}` it, and a link to a non-exported symbol trips the TypeDoc
 * warning ratchet (86/86 with the export). Do not un-export it as cleanup.
 */
export type CapsuleJointReachRule = (rMax: number, ny: number, hasPacket: boolean) => number;

/**
 * One leg's RENDERED contribution at a point, joint machinery included —
 * the mirror of the fragment shader (default sharpness knob), CLIPPED to the
 * stencil the vertex stage builds for it. The clip is not decoration: a
 * fragment outside the stencil is never rasterized, so a reach shortfall
 * chops the profile to zero instead of dimming it, and a model that
 * evaluated the profile over the whole plane would be structurally blind to
 * the entire #1488 defect class.
 *
 * That load-bearing part is the JOINT-END reach, and only it — the arm the
 * negative control exercises. The other two clips here, the perpendicular /
 * free-far-end box and the hairpin fallback's `rMax`, mirror the vertex stage
 * for completeness and are provably non-binding while the shipped reaches
 * hold (`profile > 0` already implies a point inside both). Deleting either
 * leaves the suite green, by construction rather than by oversight.
 *
 * @param reach - the joint-end reach rule; see {@link CapsuleJointReachRule}.
 */
export function capsuleJointRenderLeg(
  leg: CapsuleJointLeg,
  partner: CapsuleJointLeg,
  px: number,
  py: number,
  reach: CapsuleJointReachRule = capsuleJointStencilReach
): number {
  const [x, y] = legLocal(leg, px, py);
  let profile = capsuleLegField(leg, px, py);
  if (profile <= 0) return 0;

  // Stencil, in the leg's own frame (x from the joint end at 0 toward the
  // far end at `length`). Its half-width and its FREE far end are known
  // before the cut normal is; the joint end's reach needs `ny` and the
  // packet decision, so it is applied once those exist, below.
  const rMax = Math.max(leg.rJoint, leg.rFar) + CAPSULE_STENCIL_APRON_PX;
  if (Math.abs(y) > rMax || x > leg.length + rMax) return 0;

  // Cut normal: n ∝ q̂ − m̂ in MY local frame (m̂ = +x̂ at the joint end).
  const [qx, qy] = legLocal(leg, partner.dir[0], partner.dir[1]);
  const nRaw: [number, number] = [qx - 1, qy];
  const nl = Math.hypot(nRaw[0], nRaw[1]);
  // Hairpin fallback: plain cap, and with it the free end's full-disc reach.
  if (nl <= 1e-3) return x < -rMax ? 0 : profile;
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
  // The WIDTH gate leads, as in every vertex stage: below it the packet
  // math is skipped outright, so a hairline joint neither gets a deficit
  // term NOR the full-disc reach that term needs. Modelling the clauses
  // without it would reserve a disc the shader never builds — optimistic
  // in precisely the hairline regime the AA radius floor makes common.
  const hasPacket =
    rMax > CAPSULE_JOINT_PACKET_MIN_RADIUS_PX &&
    (Math.abs(1 - rpFar / Math.max(leg.rJoint, 1e-4)) > CAPSULE_JOINT_DEFICIT_GATE ||
      leg.rFar > leg.rJoint * (1 + CAPSULE_JOINT_DEFICIT_GATE) ||
      partner.length < 2 * leg.rJoint ||
      qx > 0.5);
  if (x < -reach(rMax, ny, hasPacket)) return 0;
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
 *
 * @param reach - the joint-end reach rule handed to BOTH legs; see
 * {@link CapsuleJointReachRule}.
 */
export function capsuleJointCompositionError(
  leg1: CapsuleJointLeg,
  leg2: CapsuleJointLeg,
  extent?: number,
  step = 0.5,
  reach: CapsuleJointReachRule = capsuleJointStencilReach
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
        capsuleJointRenderLeg(leg1, leg2, px, py, reach) +
        capsuleJointRenderLeg(leg2, leg1, px, py, reach);
      const ref = Math.max(capsuleLegField(leg1, px, py), capsuleLegField(leg2, px, py));
      const err = sum - ref;
      if (err < minErr) minErr = err;
      if (err > maxErr) maxErr = err;
    }
  }
  return { minErr, maxErr };
}
