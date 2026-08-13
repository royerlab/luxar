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
  CAPSULE_MIN_RADIUS_PX,
  type CapsuleJointLeg,
  CAPSULE_STENCIL_APRON_PX,
  capsuleJointCompositionError,
  capsuleJointStencilReach,
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

const SRC_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');

/**
 * Drop `//` and block comments so prose can neither satisfy nor break a pin.
 * Not for the `toContain` checks — neither packet-branch comment contains the
 * pinned string, so those red on a plain revert either way. It is the WRITE
 * COUNT that needs this: a comment mentioning `extA =` reads as a second
 * assignment and would fail the branch spuriously.
 */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

/**
 * The comment-stripped BODY of one vertex surface's `needPacket<end>` branch,
 * brace-matched from the branch header.
 *
 * A slice, not a whole-source grep: the half-disc reach appears twice per end
 * (the `Else` arm and the width-gated arm) and is CORRECT in both, so only a
 * scoped assertion can say which arm assigns what — a whole-source
 * `not.toContain` would forbid the right answer everywhere. Fails loudly when
 * the header is missing or the braces never balance: a locator that silently
 * returned an empty slice would go vacuous under exactly the refactor this pin
 * exists to survive.
 */
function packetBranchBody(label: string, source: string, header: string): string {
  const stripped = stripComments(source);
  const headerAt = stripped.indexOf(header);
  if (headerAt < 0) throw new Error(`${label}: packet branch header '${header}' not found`);
  const open = stripped.indexOf('{', headerAt);
  if (open < 0) throw new Error(`${label}: no '{' after '${header}'`);
  let depth = 0;
  for (let i = open; i < stripped.length; i++) {
    if (stripped[i] === '{') depth += 1;
    else if (stripped[i] === '}' && --depth === 0) return stripped.slice(open + 1, i);
  }
  throw new Error(`${label}: unbalanced braces after '${header}'`);
}

/** The two GLSL vertex stages, as compiled source strings. */
const GLSL_VERTEX_SURFACES: ReadonlyArray<readonly [string, string]> = [
  ['visual GLSL', CAPSULE_LINE_VERTEX_SHADER],
  ['pick GLSL', CAPSULE_LINE_PICK_VERTEX_SHADER],
];

/**
 * The two TSL vertex stages, read from disk: their graphs only build against a
 * real GPU backend, so — as in `line/join-width-tsl.test.ts` — the module text
 * is the only thing an always-running unit test can inspect.
 */
const TSL_VERTEX_SURFACES: ReadonlyArray<readonly [string, string]> = [
  ['visual TSL', 'rendering/materials/line/shader-tsl-capsule.ts'],
  ['pick TSL', 'rendering/picking/line/pick-capsule.tsl.ts'],
];

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

  it('the packet branch of all four vertex surfaces assigns the FULL disc once (#1488)', () => {
    // WHAT THIS GUARANTEES, exactly: inside each `needPacket<end>` branch
    // there is exactly ONE write to `ext<end>` and it is the full-disc form.
    // Worth having because the two TSL factories are pinned by nothing else
    // that runs in CI — `.github/workflows/ci.yml` sets `e2e-tests` to
    // `if: false`, so the codegen snapshots and the parity suite are not
    // merge-gating and a TSL-only revert to the half-disc reach would ship
    // green, chopping the deficit rule's light on WebGPU alone.
    //
    // MEASURED SCOPE — every case below was run as a mutant against this test.
    // CAUGHT: the literal half-disc revert; a `min()` wrapper around the
    // full-disc operand (all three spellings tried, including swapped
    // operands, because each one displaces the `extA = rMax + ` prefix); a
    // SECOND write inside the branch, whether plain (`extA = …`, line breaks
    // included), compound (`extA *= …`) or from the TSL assign family
    // (`extA.mulAssign(…)`); a renamed branch header (fails closed, by throw).
    // EVADES: a write placed AFTER the branch; an alias bound inside it that
    // also avoids the literal half-disc spelling (`const eA = extA;
    // eA.assign(nLoc.y.abs().mul(rMax)…)` — with the literal spelling the
    // `not.toContain` still reds it); and any algebraically equivalent single
    // expression (`extA = rMax + APRON - rMax * (1.0 - abs(nLoc.y));`).
    // So: a tripwire on the edits a reverting change actually makes, not a
    // proof. What backs the rule is the negative control over the CPU model
    // further down.
    //
    // Sliced per branch rather than grepped whole-source because the
    // half-disc form is CORRECT in the `Else` and width-gated arms, so only a
    // scoped assertion can say which arm assigns what.
    const assignmentsTo = (body: string, pattern: RegExp): number =>
      (body.match(pattern) ?? []).length;
    for (const [label, source] of GLSL_VERTEX_SURFACES) {
      for (const end of ['A', 'B'] as const) {
        const body = packetBranchBody(`${label} ${end}`, source, `if (needPacket${end}) {`);
        expect(body, `${label} ${end}: full-disc reach`).toContain(`ext${end} = rMax + `);
        expect(body, `${label} ${end}: half-disc reach`).not.toContain('abs(nLoc.y) * rMax');
        // Compound forms included (`*=`, `+=`, …): a bare `=` counter reads
        // `extA *= abs(nLoc.y);` as no write at all. Same rule #1494 landed
        // for `rEnd`.
        expect(
          assignmentsTo(body, new RegExp(`(?<![\\w$])ext${end}\\s*[-+*/]?=(?!=)`, 'g')),
          `${label} ${end}: the packet branch must write ext${end} exactly once`
        ).toBe(1);
      }
    }
    for (const [label, relativeToSrc] of TSL_VERTEX_SURFACES) {
      const source = readFileSync(path.join(SRC_ROOT, relativeToSrc), 'utf8');
      for (const end of ['A', 'B'] as const) {
        const body = packetBranchBody(`${label} ${end}`, source, `If(needPacket${end}, () => {`);
        expect(body, `${label} ${end}: full-disc reach`).toContain(`ext${end}.assign(rMax.add(`);
        expect(body, `${label} ${end}: half-disc reach`).not.toContain('abs(nLoc.y).mul(rMax)');
        // The whole TSL assign family, not just `.assign(`: `.mulAssign(`,
        // `.addAssign(` and friends all write the var in place.
        expect(
          assignmentsTo(body, new RegExp(`(?<![\\w$])ext${end}\\.\\w*[Aa]ssign\\(`, 'g')),
          `${label} ${end}: the packet branch must write ext${end} exactly once`
        ).toBe(1);
      }
    }
  });

  it('the CPU model mirrors that reach rule (#1488)', () => {
    // The rule the model clips its own contribution by. A packet turns the
    // kept half-disc's |ny|·rMax axial extent into the whole disc; without
    // one the shorter reach is correct (and is why a joint costs less fill
    // than a cap). The apron rides on both.
    const rMax = 10;
    expect(capsuleJointStencilReach(rMax, 0.3, true)).toBe(rMax + CAPSULE_STENCIL_APRON_PX);
    expect(capsuleJointStencilReach(rMax, -0.3, true)).toBe(rMax + CAPSULE_STENCIL_APRON_PX);
    expect(capsuleJointStencilReach(rMax, 0.3, false)).toBeCloseTo(
      3 + CAPSULE_STENCIL_APRON_PX,
      12
    );
    // Sign-blind, as `abs(nLoc.y)` is: the two legs of a joint carry exactly
    // negated normals, and both must reserve the same footprint.
    expect(capsuleJointStencilReach(rMax, -0.3, false)).toBe(
      capsuleJointStencilReach(rMax, 0.3, false)
    );
  });

  it('the model gates the packet on WIDTH first, as every vertex stage does', () => {
    // The shaders test `rMax > CAPSULE_JOINT_PACKET_MIN_RADIUS_PX` BEFORE the
    // four clauses, so a hairline joint gets neither a deficit term nor the
    // full-disc reach that term needs.
    //
    // Asserted on the BOOLEAN, through the reach seam, because no sweep bound
    // could stand in for it. On this geometry, at the step 0.5 used below:
    // gated measures −0.051291 and ungated −0.038773 — the faithful model
    // scores WORSE, so any bound tight enough to notice the gate would be
    // rewarding its removal. And the reach itself is not what moves: with the
    // gate in place, forcing the full rule and forcing the half rule both
    // measure −0.051291, bit-identical, since below 4 px the two reaches
    // differ by under 2 px of near-zero profile. That is precisely WHY the
    // shaders gate there — the deficit is sub-pixel — and precisely why only
    // the boolean is worth pinning.
    const hairline = 1.5; // the AA radius floor ⇒ rMax 2.0, under the 4 px gate
    const fat = 10; // ⇒ rMax 10.5, over it
    const packetFlags = (r: number): boolean[] => {
      const seen: boolean[] = [];
      const spy = (rMax: number, ny: number, hasPacket: boolean): number => {
        seen.push(hasPacket);
        return capsuleJointStencilReach(rMax, ny, hasPacket);
      };
      // A partner shorter than 2·rJoint holds the LENGTH clause wide open at
      // both ends, so the width gate is the only thing that can shut it.
      const own: CapsuleJointLeg = { dir: [-1, 0], rJoint: r, rFar: r, length: 8 * r };
      const partner: CapsuleJointLeg = {
        dir: [0.5, -Math.sqrt(3) / 2],
        rJoint: r,
        rFar: r,
        length: r,
      };
      capsuleJointCompositionError(own, partner, undefined, 0.5, spy);
      return seen;
    };
    const thin = packetFlags(hairline);
    expect(thin.length, 'the sweep must actually reach the clip').toBeGreaterThan(0);
    expect(thin.some(Boolean), 'a hairline joint must never build a packet').toBe(false);
    expect(packetFlags(fat).some(Boolean), 'the same joint, fat, must build one').toBe(true);
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
    for (const turn of [20, 60, 120]) {
      const { minErr, maxErr } = capsuleJointCompositionError(
        leg(180, 10, 10, 60),
        leg(-turn, 10, 10, 60)
      );
      expect(minErr, `turn ${turn}° min`).toBeGreaterThan(-0.025);
      expect(maxErr, `turn ${turn}° max`).toBeLessThan(0.005);
    }
  });

  it('tapered own leg (the #1494 rows): bounded now that rEnd is the vertex radius', () => {
    // Review table, θ = 60°, vertex radius 10, partner tapering 10 → 5
    // over 30 px: shipped-before errors reached −0.478 / +0.263.
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
      expect(minErr, `own 10→${rFar}/L${len} min`).toBeGreaterThan(-0.13);
      expect(maxErr, `own 10→${rFar}/L${len} max`).toBeLessThan(0.06);
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
    // verbatim with the fix reverted (#1497). Bounds are ±0.03: gated,
    // every configuration measures within ±0.02 of max(mine, partner);
    // with the one-sided gate these rows chop to −0.86.
    const cases: Array<[string, CapsuleJointLeg, CapsuleJointLeg]> = [
      ['const partner 120° ql2', leg(180, 10, 10, 60), leg(-120, 10, 10, 2)],
      ['const partner 160° ql2', leg(180, 10, 10, 60), leg(-160, 10, 10, 2)],
      ['const partner 160° ql15', leg(180, 10, 10, 60), leg(-160, 10, 10, 15)],
      ['widening partner 150°', leg(180, 10, 10, 60), leg(-150, 10, 30, 30)],
      ['widening own leg 160°', leg(180, 10, 20, 60), leg(-160, 10, 10, 60)],
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
      // clause (constant own leg, long partner, gentle enough that the
      // sharp clause stays closed). The own-widening clause is the one
      // clause this model CANNOT isolate: the partner's symmetric ratio
      // always opens a packet from the other side here, and the clause
      // exists for the case the model lacks — a width-gated-off partner.
      ['thinning partner 60° ql30', leg(180, 10, 10, 60), leg(-60, 10, 4, 30)],
    ];
    for (const [name, a, b] of cases) {
      const { minErr, maxErr } = capsuleJointCompositionError(a, b);
      expect(minErr, `${name} min`).toBeGreaterThan(-0.03);
      expect(maxErr, `${name} max`).toBeLessThan(0.03);
    }
  });

  it('short partners (#1488/#1490): no chopped disc, no double-count', () => {
    // Constant-width own leg isolates these two; before the fixes the
    // chop reached −0.306 and the excess +0.859 at ql = 2 px.
    // Bounds are ±(0.01, 0.005), not the pre-fix ±(0.13, 0.06) these rows
    // were first written with. That slack was the reason the rows only
    // half-caught #1488: with the model now clipped by the stencil, a revert
    // to the half-disc reach measures −0.4096 / −0.3666 / −0.1131 / −0.0325
    // (ql 2/4/8/15), and −0.13 let the ql 8 and ql 15 rows through — the
    // shallow end of the very gradient the reach rule controls. Shipped,
    // these five measure −0.0009 / −0.0014 / −0.0023 / −0.0027 / −0.0046 and
    // their maxima are float dust (3.3e−16 to 5.3e−16, i.e. zero).
    // The floor is tuned to THESE FIVE ROWS, not to the AA ramp in general:
    // the ramp's honest cost grows with turn angle and with the share of the
    // rod the cut crosses, so defect-free configurations in the same family
    // sit BELOW this floor — `leg(180, 10, 10, 60)` vs `leg(-120, 10, 5, 120)`
    // measures −0.0106, and `leg(180, 5, 5, 60)` vs `leg(-120, 5, 5, 30)`
    // measures −0.0318. A new row added to this group may legitimately need
    // its own bound; widen for it specifically rather than reopening this one.
    for (const ql of [2, 4, 8, 15, 40]) {
      const { minErr, maxErr } = capsuleJointCompositionError(
        leg(180, 10, 10, 60),
        leg(-60, 10, 5, ql)
      );
      expect(minErr, `ql ${ql} min`).toBeGreaterThan(-0.01);
      expect(maxErr, `ql ${ql} max`).toBeLessThan(0.005);
    }
  });

  it('NEGATIVE CONTROL: the pre-#1488 half-disc reach makes those rows chop', () => {
    // The rows above are the coverage for #1488 — but only because the model
    // clips each leg by its stencil, and under the SHIPPED rule that clip is
    // exactly inert (every row measures bit-identically with and without it).
    // So deleting the clip leaves the whole file green and it reads as dead
    // code. This test is what makes it observable: feed the pre-fix rule in
    // and require the sweep to CHOP.
    //
    // These bounds are a NEGATIVE control, not a target — nothing should ever
    // be tuned to satisfy them. They are deliberately loose against the
    // measured chop (−0.4096 at ql 2, −0.1131 at ql 8, i.e. 479x and 48.7x
    // the defect-free −0.0009 / −0.0023 the same rows give under the real
    // rule) so they survive a change to the sweep's step or window; as
    // asserted, the chop floor sits 350x (ql 2) and 21.5x (ql 8) above the
    // defect-free measurement. If a wired-up clip is ever removed, both
    // halves collapse to the same number and this reds.
    const halfDiscReach = (rMax: number, ny: number): number =>
      Math.abs(ny) * rMax + CAPSULE_STENCIL_APRON_PX;
    const short = (ql: number): CapsuleJointLeg[] => [leg(180, 10, 10, 60), leg(-60, 10, 5, ql)];
    for (const [ql, chopFloor] of [
      [2, -0.3],
      [8, -0.05],
    ] as const) {
      const [a, b] = short(ql);
      // Both sweeps take the DEFAULT step: passing it explicitly on one side
      // only would let a change to that default silently unweld the pair.
      const chopped = capsuleJointCompositionError(a, b, undefined, undefined, halfDiscReach);
      expect(chopped.minErr, `ql ${ql}: half-disc reach must chop`).toBeLessThan(chopFloor);
      const shipped = capsuleJointCompositionError(a, b);
      expect(shipped.minErr, `ql ${ql}: full-disc reach must not`).toBeGreaterThan(-0.01);
    }
  });
});
