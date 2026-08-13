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
 * max(mine, partner). For congruent legs at a turn of 120° or less the
 * congruence gate (`CAPSULE_JOINT_DEFICIT_GATE`) keeps the packet empty and
 * the cut is an exact zero-double-count partition — the same domain
 * partition the volumetric primitive integrates per ray; past 120° even
 * congruent legs need a live packet, since the bisector then cuts each rod
 * lengthwise (#1501, and #1495 below the width gate). Where the partner tapers away
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
 * vertex stage skips the packet math, so hairline scenes pay nothing for the
 * deficit rule — EXCEPT where the turn is SHARPER THAN 120° and BOTH of my
 * raw radii reach `CAPSULE_MIN_RADIUS_PX` (#1495).
 *
 * ANGLE: past 120° the bisector cuts my rod LENGTHWISE, so the skipped packet
 * costs real light — short partner (`ql = 0.4 r`), −0.28 of peak at 135°
 * (r = 1.5) to −0.72 at 160° (r = 3.5) against −0.05…−0.12 for a congruent
 * joint at the same angle and radius, a difference that is ≤0.03 up to ~90°.
 * (That subtrahend is the congruent joint's own packet-suppressed deviation,
 * itself part chop.) A WIDENING partner is not part of the case: its far end
 * clears this gate, so its packet already refills the cut. The test reuses the
 * sharp clause's `> 0.5` axis-dot, agreeing by DUPLICATION, not construction —
 * 16 shader literals (4 surfaces × 2 ends × {gate, clause}) against this
 * model's one `sharpTurn`, so no pin covers the clause copies.
 *
 * FLOOR: `capsuleLegWidthScale` dims a sub-floor leg while the deficit cancels
 * the partner's UNSCALED profile, so a pair at different scales OVER-fills
 * (+0.14…+0.16 at 175° with one end sub-floor) — a bead, the artifact class
 * this primitive exists to remove. It closes that half ONLY: `fade` can still
 * disagree between legs (block comment below), so the exactness condition is
 * "the partner's SCALED field <= mine everywhere", NOT "widthScale == 1". The
 * price is real — a drawable sub-floor taper (own raw 1.2 → 3.0) would measure
 * −0.05…−0.07 with a packet against −0.33…−0.79 gated, and −0.002…−0.004
 * against −0.03…−0.10 with a congruent partner and no bead at all. A dim gap
 * over a bright bead is the trade, not a free win.
 *
 * COST is fragment work, not stencil area (the kept half-disc already spans
 * sin(turn/2) >= 0.87 of the disc past 120°): a live packet makes every
 * foreign-side fragment evaluate the partner rod instead of discarding, and
 * past 120° that region runs lengthwise along the rod — 18.3% of a leg's own
 * covered footprint at 121°, 27.5% at 150°, 56.8% at 175° (r = 2, len 12;
 * 15.8/23.4/50.9 at r = 3.5) against 12–15% at a gentle joint. The thin-line
 * fill arm was NOT re-run, and the perf benches are not obviously immune:
 * their walk draws every step independently, so 24.9% of their joints turn
 * past 120°, and whether their drawn radius lands in the band this opens
 * depends on framing (not measured).
 *
 * THREE RESIDUALS, all pinned by the sweep: a joint at or below 120° still
 * hard-cuts (−0.19 at `ql = 0.4 r`, −0.39 at 0.05 r, r = 3.5); the angle
 * threshold is hard, so a projected bend sweeping it pops (−0.175 → 0.000
 * across 119.5° → 120.5°, r = 2); the floor is harder still (raw 1.4999 → 1.5
 * goes −0.21 → 0.000 at 125°, −0.72 → 0.000 at 175°) — a zoom sweeps that one.
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
 * the shared-vertex radius). Exists for the numeric composition test:
 * the two legs' rendered sum must track max(mine, partner) — the three
 * #1487-review defects (#1494/#1488/#1490) were all invisible to
 * source-substring pins and all visible to this sweep.
 *
 * `vFade` is modelled by HALF: its `widthScale` factor
 * (`capsuleLegWidthScale`) is carried, because a leg thinner than the AA
 * floor draws floored-and-dimmed and the deficit cancels the partner's
 * UNSCALED profile — so two legs at different scales no longer compose to
 * max(mine, partner) and the pair can OVER-fill (a bead). That is what the
 * packet gate's floor conjunct is for (#1495), and it is only measurable with
 * the factor present. It is carried as an endpoint-interpolated field, which
 * is the vertex stage's own construction but NOT its span: the varying is
 * stretched across the whole quad, cap extensions included (measured impact
 * in `capsuleLegWidthScale`). `fade`, the near-plane term, is NOT modelled at
 * all: it is evaluated per corner at the CLAMPED span parameter and
 * interpolated across that same over-long quad, so two legs sharing a vertex
 * disagree on it away from the vertex itself and the same over-fill class
 * appears inside the near-fade ramp — review measured +0.146…+0.155 in the
 * thin band #1495 opens and +0.174 above the width gate (i.e. pre-existing);
 * under ortho, where `fade` is 1, it is exactly 0. Neither figure was
 * re-measured here. The
 * exactness condition is therefore "the partner's SCALED field <= mine at
 * every fragment", NOT "widthScale == 1"; the floor conjunct closes the
 * widthScale half only.
 * ============================================================
 */

/**
 * One leg of a joint, in a shared 2D px frame with the joint at the origin.
 *
 * The radii are RAW (pre-clamp) pixel radii — the shader's `rawA`/`rawB`.
 * The model clamps them to `CAPSULE_MIN_RADIUS_PX` for the profile geometry
 * and dims the leg by `capsuleLegWidthScale`, exactly as the vertex stage
 * does, so a sub-floor leg is a representable (floored, dimmed) leg rather
 * than an illegal one.
 *
 * INVARIANT: the two legs of a pair must share `rJoint`. They meet at one
 * vertex whose width is one number, and the deficit rebuild reads MY
 * shared-vertex radius as the partner's starting radius (`vREnd`, #1494) — a
 * pair that disagrees there measures nonsense (−0.83 on a row that is
 * otherwise exact).
 */
export interface CapsuleJointLeg {
  /** Unit direction from the joint vertex toward the leg's far end. */
  readonly dir: readonly [number, number];
  /** RAW radius at the shared joint vertex, px (both legs must agree). */
  readonly rJoint: number;
  /** RAW radius at the far end, px. */
  readonly rFar: number;
  /** Leg length (projected), px. */
  readonly length: number;
}

function legLocal(leg: CapsuleJointLeg, px: number, py: number): [number, number] {
  const [dx, dy] = leg.dir;
  return [px * dx + py * dy, -px * dy + py * dx];
}

/** Axial span parameter, clamped to the segment (never the cap extensions). */
function legSpanT(leg: CapsuleJointLeg, x: number): number {
  return Math.min(Math.max(x / Math.max(leg.length, 1e-4), 0), 1);
}

/**
 * The leg's thin-width energy compensation at an axial position: the
 * `widthScale = min(raw / CAPSULE_MIN_RADIUS_PX, 1)` factor of `vFade`,
 * evaluated at the two ENDPOINTS — the clamped span parameter every quad
 * corner uses — and interpolated between them. 1 for any segment at or above
 * the floor.
 *
 * It is NOT the shader's field exactly, and the gap is one-sided: `widthScale`
 * rides the same INTERPOLATED varying as `fade`, so the rasterizer stretches
 * it across a quad LONGER than the segment (the cap extensions), where this
 * spreads it over the segment span alone. Modelling that stretch was measured
 * on the two sub-floor sweep rows: bead-free stays exactly +0.000 (with the
 * packet gated the sum is a cover-weighted blend of the two legs' scaled
 * fields, hence ≤ their max at ANY widthScale field), the with-packet bead
 * those rows exist to justify measures +0.144/+0.136 against +0.137/+0.164
 * here, and the accepted chop −0.397/−0.417 against −0.41/−0.47. Same class
 * on every row, so the simpler field stays — but do not read a third digit
 * off this model.
 */
export function capsuleLegWidthScale(leg: CapsuleJointLeg, x: number): number {
  // Clamp at the endpoints, THEN interpolate — the vertex stage's order (each
  // corner evaluates min() at tc ∈ {0, 1}). Clamping a mixed radius instead
  // kinks a leg that straddles the floor (raw 1.2 → 3.0) where the shader
  // ramps straight.
  const wsJoint = Math.min(leg.rJoint / CAPSULE_MIN_RADIUS_PX, 1);
  const wsFar = Math.min(leg.rFar / CAPSULE_MIN_RADIUS_PX, 1);
  return wsJoint + (wsFar - wsJoint) * legSpanT(leg, x);
}

/**
 * The leg's own UNCUT field at a point — the composition's ideal term, before
 * `widthScale`. Radii are floored like the shader's `clamp(raw, MIN, max)`.
 */
export function capsuleLegField(leg: CapsuleJointLeg, px: number, py: number): number {
  const [x, y] = legLocal(leg, px, py);
  // EXACT per-fragment radius, as the shaders compute it: the endpoint
  // mix clamped to the segment span (never the cap extensions).
  const t = legSpanT(leg, x);
  const rJoint = Math.max(leg.rJoint, CAPSULE_MIN_RADIUS_PX);
  const rFar = Math.max(leg.rFar, CAPSULE_MIN_RADIUS_PX);
  const r = Math.max(rJoint + (rFar - rJoint) * t, 1e-4);
  const ox = Math.max(Math.max(-x, x - leg.length), 0);
  const q = (y * y + ox * ox) / (r * r);
  return capsuleProfile(Math.sqrt(Math.max(q, 0)));
}

/**
 * One leg's RENDERED contribution at a point, joint machinery and the
 * `widthScale` dimming included — the mirror of the fragment shader (default
 * sharpness knob, `fade` omitted; see the block comment above).
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
  // vFade's thin-width compensation scales the WHOLE contribution, deficit
  // included — while the deficit cancels the partner's UNSCALED profile.
  const ws = capsuleLegWidthScale(leg, x);

  // Cut normal: n ∝ q̂ − m̂ in MY local frame (m̂ = +x̂ at the joint end).
  const [qx, qy] = legLocal(leg, partner.dir[0], partner.dir[1]);
  const nRaw: [number, number] = [qx - 1, qy];
  const nl = Math.hypot(nRaw[0], nRaw[1]);
  if (nl <= 1e-3) return profile * ws; // hairpin fallback: plain cap
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

  // DRAWN radii from here on — every radius the vertex stage feeds the joint
  // (rA, rB, rpFar) is the clamped one.
  const rJoint = Math.max(leg.rJoint, CAPSULE_MIN_RADIUS_PX);
  const rFar = Math.max(leg.rFar, CAPSULE_MIN_RADIUS_PX);
  const rpFar = Math.max(partner.rFar, CAPSULE_MIN_RADIUS_PX);
  // Width gate as the vertex stage applies it: the packet is skipped for a
  // hairline joint UNLESS the turn is sharp AND my own RAW radii both reach
  // the AA floor, so my widthScale is 1 (#1495) — see the constant.
  const rMax = Math.max(rJoint, rFar) + CAPSULE_STENCIL_APRON_PX;
  const sharpTurn = qx > 0.5;
  const packetAllowed =
    rMax > CAPSULE_JOINT_PACKET_MIN_RADIUS_PX ||
    (sharpTurn && Math.min(leg.rJoint, leg.rFar) >= CAPSULE_MIN_RADIUS_PX);
  // Packet gate (#1495, #1501): either leg tapering (both directions),
  // a short partner, or a near-hairpin turn defeats the hard cut's
  // assumption that the partner covers my foreign side. The angle clause
  // is NOT redundant with the length clause: at a hairpin the bisector
  // tilts toward my axis and splits my rod LENGTHWISE, so a partner in
  // the 2r–3r length band (long enough to pass the length clause,
  // shorter than my leg) refills only part of the cut half — measured to
  // −0.92 of peak without this clause. Each leg's gate stands alone (the
  // partner may be width-gated off), hence the own-widening clause too;
  // the row isolating THAT clause lives with the clause-coverage work, not
  // in this file's sweep.
  const hasPacket =
    packetAllowed &&
    (Math.abs(1 - rpFar / Math.max(rJoint, 1e-4)) > CAPSULE_JOINT_DEFICIT_GATE ||
      rFar > rJoint * (1 + CAPSULE_JOINT_DEFICIT_GATE) ||
      partner.length < 2 * rJoint ||
      sharpTurn);
  const g = (rpFar - rJoint) / partner.length;

  const side = nx * x + ny * y;
  if (side <= -0.5) return profile * ws;
  const cover = Math.min(Math.max(0.5 - side, 0), 1);
  let def = 0;
  if (hasPacket) {
    // Reflected partner axis: q = m − 2(m·n)n with m = (1, 0).
    const proj = nx;
    const qdx = 1 - 2 * proj * nx;
    const qdy = -2 * proj * ny;
    const xp = x * qdx + y * qdy;
    const yp2 = Math.max(x * x + y * y - xp * xp, 0);
    const rp = Math.max(rJoint + g * Math.min(Math.max(xp, 0), partner.length), 1e-4);
    const op = Math.max(Math.max(-xp, xp - partner.length), 0);
    const qp = (yp2 + op * op) / (rp * rp);
    const partnerP = capsuleProfile(Math.sqrt(Math.max(qp, 0)));
    def = Math.max(profile - partnerP, 0);
  }
  return (profile * cover + def * (1 - cover)) * ws;
}

/**
 * Sweep the joint neighbourhood and return the worst signed deviations of
 * (leg1 + leg2 rendered) − max(widthScale1·field1, widthScale2·field2), in
 * units of peak profile.
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
    Math.max(leg1.rJoint, leg1.rFar, leg2.rJoint, leg2.rFar, CAPSULE_MIN_RADIUS_PX) +
    5;
  const ext = extent ?? auto;
  let minErr = 0;
  let maxErr = 0;
  for (let px = -ext; px <= ext; px += step) {
    for (let py = -ext; py <= ext; py += step) {
      const sum =
        capsuleJointRenderLeg(leg1, leg2, px, py) + capsuleJointRenderLeg(leg2, leg1, px, py);
      // The ideal each leg is dimmed toward: max over the pair of
      // widthScale·field (a sub-floor leg's own target is dimmed too).
      const ref = Math.max(
        capsuleLegField(leg1, px, py) * capsuleLegWidthScale(leg1, legLocal(leg1, px, py)[0]),
        capsuleLegField(leg2, px, py) * capsuleLegWidthScale(leg2, legLocal(leg2, px, py)[0])
      );
      const err = sum - ref;
      if (err < minErr) minErr = err;
      if (err > maxErr) maxErr = err;
    }
  }
  return { minErr, maxErr };
}
