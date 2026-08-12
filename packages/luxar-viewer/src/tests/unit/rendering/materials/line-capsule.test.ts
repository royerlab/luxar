/**
 * Unit pins for the capsule line primitive's shared constants + CPU
 * reference (`_shared/line-capsule.ts`) — the single source both shader
 * backends fold from (#1352).
 */
import { readFileSync } from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { GAUSSIAN_EQUIVALENT_TRUNCATION } from '../../../../rendering/materials/_shared/falloff';
import {
  CAPSULE_JOINT_DEFICIT_GATE,
  CAPSULE_MIN_RADIUS_PX,
  type CapsuleJointLeg,
  capsuleJointCompositionError,
  capsuleLegField,
  CAPSULE_RADIUS_PER_QUAD_HALFWIDTH,
  CAPSULE_SUPPORT_SIGMA,
  capsuleProfile,
  capsuleProfileExponent,
} from '../../../../rendering/materials/_shared/line-capsule';
import {
  CAPSULE_LINE_FRAGMENT_SHADER,
  CAPSULE_LINE_VERTEX_SHADER,
} from '../../../../rendering/materials/line/shader-glsl-capsule';
import {
  CAPSULE_LINE_PICK_FRAGMENT_SHADER,
  CAPSULE_LINE_PICK_VERTEX_SHADER,
} from '../../../../rendering/picking/line/shaders-capsule';

/** Drop `//` and block comments so prose can never satisfy — or break — a match. */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

/**
 * Comment-stripped, whitespace-squashed text: immune to reformatting and
 * line-wrapping, but an operator, threshold, ordering or deletion change
 * still shows up as a broken contiguous substring.
 */
function squash(source: string): string {
  return stripComments(source).replace(/\s+/g, '');
}

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../../..');

function readSource(relativeToSrc: string): string {
  return readFileSync(path.join(SRC_ROOT, relativeToSrc), 'utf8');
}

describe('capsule constants', () => {
  it('the radius factor is the 2σ fraction of the quad half-width', () => {
    expect(CAPSULE_SUPPORT_SIGMA).toBe(2.0);
    expect(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH).toBeCloseTo(2 / GAUSSIAN_EQUIVALENT_TRUNCATION, 12);
    // The shaders inline it at toFixed(7); pin the literal so a constant
    // change is visible here before it silently reshapes every line.
    expect(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH.toFixed(7)).toBe('0.6590102');
  });

  it('joint constants: 1.5 px AA floor', () => {
    expect(CAPSULE_MIN_RADIUS_PX).toBe(1.5); // matches the quad's AA floor
  });

  it('shaders fold the shared literals (no re-derived magic numbers)', () => {
    for (const src of [CAPSULE_LINE_VERTEX_SHADER, CAPSULE_LINE_PICK_VERTEX_SHADER]) {
      expect(src).toContain(CAPSULE_RADIUS_PER_QUAD_HALFWIDTH.toFixed(7)); // radius factor
      expect(src).toContain('1.5'); // AA radius floor
    }
    for (const src of [CAPSULE_LINE_FRAGMENT_SHADER, CAPSULE_LINE_PICK_FRAGMENT_SHADER]) {
      // The quartic default path and the sharpness exponent map.
      expect(src).toContain('w * w');
      expect(src).toContain('exp2(3.0 - 4.0 * vSharp)');
    }
  });

  it('the joint composes by the DEFICIT rule: max(mine, partner), no fade', () => {
    // On the partner's side of the joint bisector the fragment renders
    // max(mine − partner, 0) — exact partition for congruent legs (zero
    // double-count: a soft fade was reverted after live QA showed bright
    // wedges) while a fat vertex's disc keeps the half a thin neighbour
    // cannot render (live QA showed the pure hard cut chopping it).
    for (const src of [CAPSULE_LINE_FRAGMENT_SHADER, CAPSULE_LINE_PICK_FRAGMENT_SHADER]) {
      expect(src).not.toContain('cutFade');
      expect(src).not.toContain('smoothstep');
      expect(src).toContain('luxarPartnerProfile');
      expect(src).toContain('if (profile <= 0.0) discard;');
      // The cut is a 1 px AA RAMP (hard-step boundary pixels flip
      // independently per leg — speckles), and a non-negative gradient
      // packet contributes no deficit (hard cut via the zero blend).
      expect(src).toContain('clamp(0.5 - sideA, 0.0, 1.0)');
      // Packet validity = a positive packed partner length (the joint
      // state rides packed half-pair varyings; normals stay full
      // precision in vCutN per #1502 — see the shader declaration note).
      expect(src).toContain('if (pkA.y > 0.0) {');
      expect(src).toContain('unpackHalf2x16(vPack.x)');
      expect(src).toContain('flat in vec4 vCutN;');
    }
    for (const src of [CAPSULE_LINE_VERTEX_SHADER, CAPSULE_LINE_PICK_VERTEX_SHADER]) {
      // Stencil reach covers the kept half-disc PLUS the partner's taper
      // deficit (the disc half the deficit rule now renders).
      // Full-disc reach when a packet exists (#1488): the deficit term is
      // bounded by my own profile, so my own support bounds its support.
      expect(src).toMatch(/ext[AB] = rMax \+ /);
      // The partner packet carries the far radius from the SAME texels,
      // packed as a radius GRADIENT + projected LENGTH in the cut varying.
      expect(src).toContain('sanitizeNonNegative(far.w, 0.0)');
      expect(src).toMatch(/cut[AB]\.z = \(rpFar[AB] - r[AB]\) \/ ql;/);
      expect(src).toMatch(/cut[AB]\.w = ql;/);
    }
  });

  it('the joint partition is never width-gated (a hairline joint would double)', () => {
    // The drawn radius is FLOORED at the AA minimum, so even a hairline
    // draws a 1.5 px disc at each end. Skipping the cut there to save the
    // partner fetch would leave two full caps stacked on the shared
    // vertex: measured +1.00 of peak (a 2x bead as wide as the line
    // itself) at every bend angle, so there is no width at which the
    // overlap is sub-pixel. Only the deficit PACKET is width-gated.
    const r = CAPSULE_MIN_RADIUS_PX;
    const leg = (deg: number): CapsuleJointLeg => ({
      dir: [Math.cos((deg * Math.PI) / 180), Math.sin((deg * Math.PI) / 180)],
      rJoint: r,
      rFar: r,
      length: 8,
    });
    for (const turn of [0, 30, 90]) {
      const [l1, l2] = [leg(180), leg(-turn)];
      // With the cut: the pair composes to max(mine, partner) exactly.
      const { maxErr } = capsuleJointCompositionError(l1, l2, undefined, 0.25);
      expect(maxErr).toBeLessThan(0.01);
      // Without it (plain caps): the shared vertex renders at 2x.
      const capped = capsuleLegField(l1, 0, 0) + capsuleLegField(l2, 0, 0);
      expect(capped).toBeCloseTo(2.0, 6);
    }
    // So no shader may suppress the cut below a radius threshold.
    for (const src of [CAPSULE_LINE_VERTEX_SHADER, CAPSULE_LINE_PICK_VERTEX_SHADER]) {
      expect(src).not.toMatch(/if \(rMax < /);
      expect(src).not.toContain('partnerJointA');
    }
  });

  it('the cap rule comes from the shared joint-code helper (a hub keeps its cap)', () => {
    // A degree->=3 hub (code -2) and a free end (0) keep the whole round
    // cap; reading `abs(code) > 0.5` instead butt-cuts a hub and notches it.
    for (const src of [CAPSULE_LINE_VERTEX_SHADER, CAPSULE_LINE_PICK_VERTEX_SHADER]) {
      expect(src).toContain('luxarLineJointCapSuppression(lineT4.y)');
      expect(src).toContain('luxarLineJointCapSuppression(lineT4.z)');
    }
  });

  it('no backticks inside the GLSL template literals', () => {
    for (const src of [
      CAPSULE_LINE_VERTEX_SHADER,
      CAPSULE_LINE_FRAGMENT_SHADER,
      CAPSULE_LINE_PICK_VERTEX_SHADER,
      CAPSULE_LINE_PICK_FRAGMENT_SHADER,
    ]) {
      expect(src).not.toContain('`');
    }
  });

  it('the four packet-gate clauses are hand-transcribed into all four shader surfaces', () => {
    // The gate lives in FIVE places: the CPU model in `_shared/line-capsule.ts`
    // (the composition sweep below binds only THAT one) plus four shader
    // surfaces — GLSL visual + pick (already imported above as plain
    // strings) and their TSL twins (`shader-tsl-capsule.ts`,
    // `pick-capsule.tsl.ts`, read from disk below). This reads the TSL pair
    // the same "regression lock" way `join-width-tsl.test.ts` pins TSL
    // builder source: those two modules build a graph at runtime instead of
    // exporting text, and their ONLY other coverage is the checked-in
    // codegen snapshots, which live in the `e2e-tests` job —
    // `.github/workflows/ci.yml` sets that whole job to `if: false`, so a
    // TSL-only deletion of this gate is invisible to CI.
    //
    // Pinning a single clause substring has three holes: (a) swapping the
    // chain's `||` for `&&` keeps every substring intact and evades the pin
    // entirely; (b) a substring that stops before its comparison (e.g.
    // 'abs(1.0 - rpFarA / max(rA, 1e-4))' alone) still matches if the
    // threshold changes to `> 1.0`; (c) nothing pins the gate VALUE, so a
    // shader hardcoding a different threshold still matches a name-only
    // pin. Pinning the WHOLE clause chain per joint end, in order, closes
    // all three at once. Matching runs on comment-stripped,
    // WHITESPACE-SQUASHED text, so the pin survives reformatting and
    // line-wrapping but still catches an operator, threshold, order or
    // deletion change.
    const gate = CAPSULE_JOINT_DEFICIT_GATE.toFixed(2);
    // GLSL inlines the gate as a literal (shader-glsl-capsule.ts:53);
    // building the expected text from the constant means a deliberate gate
    // change flows through instead of turning this test red.
    const glslClauseA = `abs(1.0 - rpFarA / max(rA, 1e-4)) > ${gate} || rB > rA * (1.0 + ${gate}) || ql < 2.0 * rA || dot(qq / ql, u) > 0.5`;
    const glslClauseB = `abs(1.0 - rpFarB / max(rB, 1e-4)) > ${gate} || rA > rB * (1.0 + ${gate}) || ql < 2.0 * rB || dot(qq / ql, u) < -0.5`;
    // TSL references the gate by NAME (never inlined), so pin the name.
    const tslClauseA =
      'abs(float(1.0).sub(rpFarA.div(max(rA, float(1e-4))))).greaterThan(CAPSULE_JOINT_DEFICIT_GATE).or(rB.greaterThan(rA.mul(float(1.0).add(CAPSULE_JOINT_DEFICIT_GATE)))).or(ql.lessThan(rA.mul(2.0))).or(dot(qhat, u).greaterThan(0.5))';
    const tslClauseB =
      'abs(float(1.0).sub(rpFarB.div(max(rB, float(1e-4))))).greaterThan(CAPSULE_JOINT_DEFICIT_GATE).or(rA.greaterThan(rB.mul(float(1.0).add(CAPSULE_JOINT_DEFICIT_GATE)))).or(ql.lessThan(rB.mul(2.0))).or(dot(qhat, u).lessThan(-0.5))';

    const glslVisual = CAPSULE_LINE_VERTEX_SHADER;
    const glslPick = CAPSULE_LINE_PICK_VERTEX_SHADER;
    const tslVisual = readSource('rendering/materials/line/shader-tsl-capsule.ts');
    const tslPick = readSource('rendering/picking/line/pick-capsule.tsl.ts');

    const surfaces = [
      ['GLSL visual', glslVisual, glslClauseA, glslClauseB],
      ['GLSL pick', glslPick, glslClauseA, glslClauseB],
      ['TSL visual', tslVisual, tslClauseA, tslClauseB],
      ['TSL pick', tslPick, tslClauseA, tslClauseB],
    ] as const;

    // A silently empty/short read (bad path, moved file) must not make
    // every match below vacuously pass.
    for (const [label, src] of surfaces) {
      expect(src.length, `${label} source read suspiciously short`).toBeGreaterThan(2000);
    }
    for (const [label, src, clauseA, clauseB] of surfaces) {
      const squashed = squash(src);
      expect(squashed, `${label}: end-A clause chain not found intact`).toContain(squash(clauseA));
      expect(squashed, `${label}: end-B clause chain not found intact`).toContain(squash(clauseB));
    }
  });
});

describe('capsuleProfile (CPU reference)', () => {
  it('default knob: the quartic bump (1 − p²)²', () => {
    expect(capsuleProfileExponent(0.5)).toBe(2);
    expect(capsuleProfile(0)).toBe(1);
    expect(capsuleProfile(0.5)).toBeCloseTo(0.75 * 0.75, 12);
    expect(capsuleProfile(1)).toBe(0); // exact zero at the 2σ rim
    expect(capsuleProfile(1.5)).toBe(0); // compact support beyond
  });

  it('sharpness map runs boxy↔spiky the right way round', () => {
    // Smaller exponents are boxier in w-space: s=1 (boxy) keeps more of
    // the shoulder than s=0 (spiky) at the same radius.
    expect(capsuleProfileExponent(0)).toBe(8);
    expect(capsuleProfileExponent(1)).toBe(0.5);
    expect(capsuleProfile(0.7, 1.0)).toBeGreaterThan(capsuleProfile(0.7, 0.5));
    expect(capsuleProfile(0.7, 0.0)).toBeLessThan(capsuleProfile(0.7, 0.5));
  });

  it('monotone decreasing in p at every knob', () => {
    for (const s of [0, 0.25, 0.5, 0.75, 1]) {
      let prev = Infinity;
      for (let i = 0; i <= 20; i++) {
        const val = capsuleProfile(i / 20, s);
        expect(val).toBeLessThanOrEqual(prev + 1e-12);
        prev = val;
      }
    }
  });
});

describe('joint composition — the rendered pair tracks max(mine, partner)', () => {
  // The numeric sweep the #1487 review asked for: none of #1494 (wrong
  // vertex radius), #1488 (reach shortfall) or #1490 (missing far cap)
  // were visible to source-substring pins; all three blow these bounds.
  const leg = (
    angleDeg: number,
    rJoint: number,
    rFar: number,
    length: number
  ): CapsuleJointLeg => ({
    dir: [Math.cos((angleDeg * Math.PI) / 180), Math.sin((angleDeg * Math.PI) / 180)],
    rJoint,
    rFar,
    length,
  });

  it('congruent legs: the AA ramp costs ≤2.5% of peak and never over-brightens', () => {
    // turn = 120 sits ON the sharp-turn clause's threshold by construction:
    // qx = dot(partner.dir, leg.dir), and leg.dir = (cos 180°, sin 180°)
    // has sin 180° = 1.2246e-16, not exactly 0, so that term survives —
    // qx = 0.49999999999999967, which is 3.3e-16 below the `qx > 0.5` test
    // (not the 2e-16 that plain −cos(120°) alone would suggest). If that
    // ever flips, the packet opens here and the row measures ~0 instead of
    // the AA-ramp cost it exists to bound.
    for (const turn of [20, 60, 120]) {
      const { minErr, maxErr } = capsuleJointCompositionError(
        leg(180, 10, 10, 60),
        leg(-turn, 10, 10, 60)
      );
      expect(minErr, `turn ${turn}° min`).toBeGreaterThan(-0.025);
      expect(maxErr, `turn ${turn}° max`).toBeLessThan(0.005);
      if (turn === 120) {
        // Non-vacuity: this is the file's only bound on the ramp cost, and
        // it sits 3.3e-16 under the model's own `qx > 0.5` threshold — if
        // that threshold ever moves, the packet opens here, minErr goes to
        // ~0, and the −0.025 bound above still PASSES (measured: at a
        // threshold of 0.4999999999999 this row goes −0.01661 → −0.00000
        // and stays green). Assert the ramp cost is actually being
        // measured, not just that it stays under budget.
        expect(minErr, 'turn 120° must still measure the ramp cost').toBeLessThan(-0.005);
      }
    }
  });

  it('tapered own leg (the #1494 rows): bounded now that rEnd is the vertex radius', () => {
    // Review table, θ = 60° from straight, vertex radius 10, partner tapering 10 → 5
    // over 30 px: shipped-before errors reached −0.478 / +0.263. Measured
    // now: every row is ~0.0000/0.0000, deep inside the old ±0.13/+0.06
    // slack; tightened to (−0.01, 0.005).
    for (const [rFar, len] of [
      [4, 20],
      [4, 60],
      [20, 20],
      [20, 60],
    ] as const) {
      const { minErr, maxErr } = capsuleJointCompositionError(
        leg(180, 10, rFar, len),
        leg(-60, 10, 5, 30)
      );
      expect(minErr, `own 10→${rFar}/L${len} min`).toBeGreaterThan(-0.01);
      expect(maxErr, `own 10→${rFar}/L${len} max`).toBeLessThan(0.005);
    }
  });

  it('constant-width and widening cases (#1495): the gate opens for every deficit source', () => {
    // The review's rows: a one-sided gate (thinning partner only) left all
    // of these hard-chopping, down to −0.861 of peak at 160°/ql=2.
    // Angle convention: leg(-t) puts the partner t degrees FROM STRAIGHT
    // (straight continuation of the own leg at 180° is 0°) — the same
    // convention as the congruent rows above. The first cut of these rows
    // used the angle BETWEEN the legs by mistake, which made every row a
    // near-straight joint the old gate already handled — the sweep passed
    // verbatim with the fix reverted (#1497).
    //
    // Per-clause coverage (#1497): deleting one clause from `hasPacket` and
    // re-running the sweep over all three tightened groups (#1494, this
    // group, short-partners), every row that then breaches its bound —
    //   clause A (ratio, whole clause): 'thinning partner 60° ql30' −0.0431,
    //     'own 10→4/L20' −0.0281 (#1494), 'ql 40' −0.0251 (short-partner),
    //     'own 10→4/L60' −0.0154 (#1494), 'widening partner 119° r5' −0.0306
    //   clause A, widening half only (`abs(...)` swapped for one-sided
    //     `max(1 − rpFar/rJoint, 0)`): 'widening partner 119° r5' −0.0306
    //     (identical value — this row's ratio deviation is purely widening)
    //   clause B (own-widen): 'widening own leg 90° ql60' −0.3352,
    //     'widening own leg 30° ql60' −0.1740, and 'widening partner 119°
    //     r5' −0.0329 (its partner leg also widens on ITS OWN side)
    //   clause C (short): 'const partner 120° ql2' −0.3992
    //   clause D (sharp-turn): 'const partner 175° ql20' −0.9216, 170°/ql25
    //     −0.7572, 160°/ql20 −0.4691, 160°/ql60 −0.0189, 160°/ql15 −0.0109
    //
    // The widening half of clause A: dropping to the one-sided
    // `max(1 − rpFar/rJoint, 0)` form (own leg constant leg(180,r,r,6r),
    // partner leg(-119,r,1.05r,8r), 119° from straight) measures r=4
    // −0.0395, r=5 −0.0306, r=6 −0.0257, r=8 −0.0188, r=10 −0.0152 — read
    // as composition-inert in an earlier pass, but it was a REAL
    // widening-half measurement, just taken where the effect happens to be
    // smallest, not an artifact of the scan. It sits at the same ORDER as
    // the fully-congruent AA-ramp cost at the same radius and angle, not
    // uniformly under it — at r=4 the fully-congruent gated cost is
    // −0.0390, so the widening-half number there (−0.0395) is actually
    // slightly WORSE. (This r-scan's own leg is 6r long and its partner
    // 8r; a different length ratio shifts these figures by percent-level
    // amounts, so quote them only together with this geometry.)
    //
    // 'widening partner 119° r5' sits at r = 5, where dropping clause A
    // (whole or widening-half) measures −0.0306 — about 3× this section's
    // bound — while both legs (5.5, 5.75 px) stay clear of the 4 px
    // packet-width gate. What the packet buys there is refilling the 1 px
    // AA-ramp band: GATED, the row measures ~0, while a fully congruent
    // joint at the SAME radius and angle sits at −0.0316 — that is why the
    // group's −0.01 bound is satisfiable at r=5 at all, and why the row
    // goes red without the clause instead of merely chopping.
    //
    // Bounds: gated, every maxErr below is zero to float noise (2e-16 …
    // 6e-14, worst 'hairpin, both legs long' at 5.5e-14) and the worst
    // minErr is −0.0047 ('const partner 120° ql2'); tightened from ±0.03
    // to (−0.01, 0.005) so the sweep trips before this margin is eaten
    // into. Under the LITERAL pre-#1495 gate (one-sided ratio only, no
    // B/C/D) the worst row is 'const partner 175° ql20' at −0.9216;
    // −0.8606 is 'const partner 160° ql2'.
    const cases: Array<[string, CapsuleJointLeg, CapsuleJointLeg]> = [
      ['const partner 120° ql2', leg(180, 10, 10, 60), leg(-120, 10, 10, 2)],
      ['const partner 160° ql2', leg(180, 10, 10, 60), leg(-160, 10, 10, 2)],
      ['const partner 160° ql15', leg(180, 10, 10, 60), leg(-160, 10, 10, 15)],
      ['widening partner 150°', leg(180, 10, 10, 60), leg(-150, 10, 30, 30)],
      // This row looks like an own-widening-clause (B) probe but is not
      // one: at 160° from straight qx = 0.94 > 0.5, so the sharp-turn
      // clause D opens the packet regardless of B (the sweep stays green
      // with B alone dropped — it took the rows below to change that).
      ['widening own leg 160°', leg(180, 10, 20, 60), leg(-160, 10, 10, 60)],
      // Isolate clause B on the leg under test: a gentle bend keeps D
      // shut, and a constant long partner AT the shared joint radius
      // keeps the ratio clause A and the length clause C shut, so only
      // clause B can open the packet on the leg under test (the partner's
      // own ratio clause A happens to be satisfied on its own side too,
      // but that clause is not the one being dropped here).
      ['widening own leg 90° ql60', leg(180, 10, 20, 15), leg(-90, 10, 10, 60)],
      ['widening own leg 30° ql60', leg(180, 10, 30, 15), leg(-30, 10, 10, 60)],
      // Isolate the WIDENING half of the ratio clause A: own leg constant
      // (clause B shut on the leg under test), partner length 40 ⩾ 2·5
      // (clause C shut), 119° from straight → qx = 0.4848 < 0.5, 0.015
      // clear of the clause D threshold. The partner also widens on its
      // own side (rJoint 5 → rFar 5.25), so this row additionally
      // exercises clause B on the partner (see the per-clause table above).
      ['widening partner 119° r5', leg(180, 5, 5, 30), leg(-119, 5, 5.25, 40)],
      ['const partner 160° ql60', leg(180, 10, 10, 60), leg(-160, 10, 10, 60)],
      // The #1501 band: a hairpin partner LONGER than the disc (passes
      // the length clause) but SHORTER than my leg — the bisector splits
      // my rod lengthwise and only the angle clause opens the packet
      // (measured to −0.92 without it). These isolate that clause.
      ['const partner 160° ql20 (#1501)', leg(180, 10, 10, 60), leg(-160, 10, 10, 20)],
      ['const partner 170° ql25 (#1501)', leg(180, 10, 10, 60), leg(-170, 10, 10, 25)],
      ['const partner 175° ql20 (#1501)', leg(180, 10, 10, 60), leg(-175, 10, 10, 20)],
      // BOTH legs long at a near-hairpin (#1502): with the cut spanning
      // the full stencil, any per-leg quantisation of the plane normal
      // accumulates with distance along the rod — a 1/1024 snap measured
      // +0.057 here and ±0.35 at longer legs. Unsnapped it is ~0.
      ['hairpin, both legs long (#1502)', leg(180, 10, 10, 60), leg(-178.6, 10, 10, 60)],
      // Thinning partner at a moderate bend: isolates the taper-ratio
      // clause A (constant own leg, long partner, gentle enough that the
      // sharp-turn clause D stays closed).
      ['thinning partner 60° ql30', leg(180, 10, 10, 60), leg(-60, 10, 4, 30)],
    ];
    for (const [name, a, b] of cases) {
      const { minErr, maxErr } = capsuleJointCompositionError(a, b);
      expect(minErr, `${name} min`).toBeGreaterThan(-0.01);
      expect(maxErr, `${name} max`).toBeLessThan(0.005);
    }
  });

  it('short partners (#1488/#1490): no chopped disc, no double-count', () => {
    // Constant-width own leg isolates these two; before the fixes the
    // chop reached −0.306 and the excess +0.859 at ql = 2 px. Measured
    // now: minErr −0.0009 (ql2) down to −0.0046 (ql40), maxErr zero to
    // float noise (3.3e-16 … 5.3e-16) at every ql; tightened to
    // (−0.01, 0.005).
    for (const ql of [2, 4, 8, 15, 40]) {
      const { minErr, maxErr } = capsuleJointCompositionError(
        leg(180, 10, 10, 60),
        leg(-60, 10, 5, ql)
      );
      expect(minErr, `ql ${ql} min`).toBeGreaterThan(-0.01);
      expect(maxErr, `ql ${ql} max`).toBeLessThan(0.005);
    }
  });
});
