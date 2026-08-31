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
 * partition the deleted volumetric primitive integrated per ray; past 120° even
 * congruent legs need a live packet, since the bisector then cuts each rod
 * lengthwise (#1501, and #1495 below the width gate). Where the partner tapers away
 * or its apparent radius diverges under perspective, the deficit term
 * contributes exactly the light a pure partition would chop (a fat
 * vertex's disc keeps the half a thin neighbour cannot render). The
 * partner's field is rebuilt per fragment from the cut varying's packed
 * radius gradient + projected length and the shared-vertex radius
 * (the packed `vPack.z` lane, unpacked as `pkR`, #1494), with the far cap
 * closing the rod (#1490); the stencil reserves the full disc when a
 * packet exists, the deficit being bounded by my own profile, so a
 * half-disc reach there would leave the rasterizer chopping the very light
 * the deficit rule adds back (#1488). The partner's far endpoint is
 * near-plane-clipped toward the joint vertex before projecting (a
 * behind-eye projection flips and poisons the cut normal). A CPU model of
 * this composition — the joint-end STENCIL REACH included, since a
 * shortfall there is invisible to a model that evaluates the profile
 * everywhere — lives at the bottom of this file; the unit sweep
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

/** Minimum drawn radius in CSS pixels (scaled by active DPR in the shaders). */
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
 * Width gate for the joint packet: below this stencil half-width (CSS px) the
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
 * All radii and residual measurements in this block are CSS-pixel values;
 * fragment-work percentages vary with the render-target scale.
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
 * the shared-vertex radius) AND the vertex stage's joint-end stencil reach,
 * which bounds where any of it can run. Exists for the numeric composition test:
 * the two legs' rendered sum must track max(mine, partner) — the three
 * #1487-review defects (#1494/#1488/#1490) were all visible to this
 * sweep. All three now ALSO have source locks over the four shader surfaces,
 * which this model cannot provide (it binds only itself): #1494 in
 * `tests/unit/rendering/materials/line/capsule-partner-radius.test.ts`, #1490
 * in `tests/unit/rendering/materials/line/capsule-joint-packet-source-lock.test.ts`,
 * and #1488 over the packet branch of all four vertex surfaces, in
 * `line-capsule.test.ts` itself. Those bind the shader TEXT; this sweep is
 * still the only thing that binds any of the three by VALUE.
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
 * This reference model operates at DPR/render scale 1, so its pixel values are
 * CSS pixels. The model clamps them to `CAPSULE_MIN_RADIUS_PX` for the profile geometry
 * and dims the leg by `capsuleLegWidthScale`, exactly as the vertex stage
 * does, so a sub-floor leg is a representable (floored, dimmed) leg rather
 * than an illegal one. It mirrors the FLOOR of the shaders'
 * `clamp(raw, MIN, uMaxLinePixelWidth)` only, never the ceiling: that is a
 * uniform, so there is no value to fold in here — pass radii already inside
 * whatever cap the scene uses, or the model reads wider than the shader draws.
 *
 * INVARIANT: the two legs of a pair must share `rJoint`. They meet at one
 * vertex whose width is one number, and the deficit rebuild reads MY
 * shared-vertex radius as the partner's starting radius (the packed
 * `vPack.z` lane, unpacked as `pkR`, #1494) — a pair that disagrees there
 * measures nonsense (−0.83 on a row that is otherwise exact).
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
 * `rMax` is the vertex stage's OWN `rMax` — `max(rA, rB) + apron`, the same
 * value the width gate compares and the quad's half-width uses — so both arms
 * add the apron a second time, exactly as the four `ext` writes do
 * (`extA = rMax + APRON` / `abs(nLoc.y) * rMax + APRON`). Passing the bare
 * drawn radius instead would make this model reserve 0.5 px LESS than the
 * shader: faithful-looking, and quietly able to chop what the shader draws.
 *
 * The two arms it does NOT model: the bare-apron BUTT an interior end keeps
 * when there is no partner-far texel, the joint vertex is behind the near
 * plane, the projected partner length degenerates (the shaders' `ql > 1e-4`
 * guard, `shader-glsl-capsule.ts` — this model divides by that length
 * instead: a tiny-but-finite one is modelled as an extreme taper rather than
 * a butt, and a ZERO one goes non-finite WHERE THE PACKET IS LIVE, which the
 * sweep throws on rather than scoring as a vacuous 0 — gated off, the gradient
 * is never read and the row scores finitely), or `nLoc.x` fails its sign test;
 * and the free-end/hairpin `rMax`, which the model hardcodes at its own call
 * sites and so cannot be reached through an injected rule. The first two of
 * those are unproducible here (the model has no texels and no near plane), but
 * the `nLoc.x` one merely goes unexercised: with θ the angle between q̂ and m̂,
 * `nl = 2|sin(θ/2)|` and `nLoc.x = −|sin(θ/2)|`, so the window
 * `nl > 1e-3 ∧ nLoc.x > −1e-3` holds for θ in (0.057°, 0.115°) — a sliver just
 * above the hairpin fallback, which `leg(180, …)` vs `leg(180.08, …)` would
 * enter. No row in the sweep does.
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
 * One leg's RENDERED contribution at a point, joint machinery and the
 * `widthScale` dimming included — the mirror of the fragment shader (default
 * sharpness knob, `fade` omitted; see the block comment above), CLIPPED to the
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
  // vFade's thin-width compensation scales the WHOLE contribution, deficit
  // included — while the deficit cancels the partner's UNSCALED profile.
  const ws = capsuleLegWidthScale(leg, x);

  // DRAWN radii from here on — every radius the vertex stage feeds the joint
  // (rA, rB, rpFar) is the clamped one.
  const rJoint = Math.max(leg.rJoint, CAPSULE_MIN_RADIUS_PX);
  const rFar = Math.max(leg.rFar, CAPSULE_MIN_RADIUS_PX);
  const rpFar = Math.max(partner.rFar, CAPSULE_MIN_RADIUS_PX);

  // Stencil, in the leg's own frame (x from the joint end at 0 toward the
  // far end at `length`). Its half-width and its FREE far end are known
  // before the cut normal is; the joint end's reach needs `ny` and the
  // packet decision, so it is applied once those exist, below. The same
  // `rMax` is what the width gate below compares against, as in the vertex
  // stage — one definition, drawn radii plus the apron.
  const rMax = Math.max(rJoint, rFar) + CAPSULE_STENCIL_APRON_PX;
  if (Math.abs(y) > rMax || x > leg.length + rMax) return 0;

  // Cut normal: n ∝ q̂ − m̂ in MY local frame (m̂ = +x̂ at the joint end).
  const [qx, qy] = legLocal(leg, partner.dir[0], partner.dir[1]);
  const nRaw: [number, number] = [qx - 1, qy];
  const nl = Math.hypot(nRaw[0], nRaw[1]);
  // Hairpin fallback: plain cap, and with it the free end's full-disc reach.
  if (nl <= 1e-3) return x < -rMax ? 0 : profile * ws;
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

  // Width gate as the vertex stage applies it: the packet is skipped for a
  // hairline joint UNLESS the turn is sharp AND my own RAW radii both reach
  // the AA floor, so my widthScale is 1 (#1495) — see the constant.
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
  // in this file's sweep. The gate also decides the joint end's STENCIL: a
  // packet-less end reserves only the kept half-disc, so modelling the
  // clauses without `packetAllowed` would reserve a disc the shader never
  // builds (#1488).
  const hasPacket =
    packetAllowed &&
    (Math.abs(1 - rpFar / Math.max(rJoint, 1e-4)) > CAPSULE_JOINT_DEFICIT_GATE ||
      rFar > rJoint * (1 + CAPSULE_JOINT_DEFICIT_GATE) ||
      partner.length < 2 * rJoint ||
      sharpTurn);
  if (x < -reach(rMax, ny, hasPacket)) return 0;
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
    Math.max(leg1.rJoint, leg1.rFar, leg2.rJoint, leg2.rFar, CAPSULE_MIN_RADIUS_PX) +
    5;
  const ext = extent ?? auto;
  let minErr = 0;
  let maxErr = 0;
  for (let px = -ext; px <= ext; px += step) {
    for (let py = -ext; py <= ext; py += step) {
      const sum =
        capsuleJointRenderLeg(leg1, leg2, px, py, reach) +
        capsuleJointRenderLeg(leg2, leg1, px, py, reach);
      // The ideal each leg is dimmed toward: max over the pair of
      // widthScale·field (a sub-floor leg's own target is dimmed too).
      const ref = Math.max(
        capsuleLegField(leg1, px, py) * capsuleLegWidthScale(leg1, legLocal(leg1, px, py)[0]),
        capsuleLegField(leg2, px, py) * capsuleLegWidthScale(leg2, legLocal(leg2, px, py)[0])
      );
      const err = sum - ref;
      // NaN-FATAL, deliberately. Both comparisons below are FALSE for NaN, so a
      // degenerate row — a ZERO (or overflow-small) `partner.length` makes the
      // packet's radius gradient non-finite, and 671 fragments of one measured
      // sweep NaN — would sail through as a PERFECT {0, 0} score, the one
      // failure mode this sweep must never have, since its whole job is to be
      // the detector. Fail loudly instead. (A tiny but FINITE length is not
      // this case: it is modelled as an extreme taper, where the shaders would
      // butt-cut at `ql <= 1e-4`. Nor is a degenerate leg whose PACKET is gated
      // off — the gradient is never read there, so the row scores finitely. See
      // `capsuleJointStencilReach`.)
      if (!Number.isFinite(err)) {
        throw new Error(
          `capsuleJointCompositionError: non-finite error at (${px}, ${py}) — ` +
            'a degenerate leg (zero projected partner length, where the shaders ' +
            "take the bare-apron butt arm this model doesn't have) would " +
            'otherwise score a vacuous perfect 0'
        );
      }
      if (err < minErr) minErr = err;
      if (err > maxErr) maxErr = err;
    }
  }
  return { minErr, maxErr };
}
