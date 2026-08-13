/**
 * Unit pins for the capsule line primitive's shared constants + CPU
 * reference (`_shared/line-capsule.ts`) — the single source both shader
 * backends fold from (#1352).
 */
import { describe, expect, it } from 'vitest';

import { GAUSSIAN_EQUIVALENT_TRUNCATION } from '../../../../rendering/materials/_shared/falloff';
import {
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
  // The numeric sweep the #1487 review asked for: all three of #1494
  // (wrong vertex radius), #1488 (reach shortfall) and #1490 (missing far
  // cap) blow these bounds. #1494 additionally has a source lock over the
  // four shader surfaces now (`line/capsule-partner-radius.test.ts`);
  // #1488 and #1490 have no comparable one, so this sweep binds them.
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
    for (const ql of [2, 4, 8, 15, 40]) {
      const { minErr, maxErr } = capsuleJointCompositionError(
        leg(180, 10, 10, 60),
        leg(-60, 10, 5, ql)
      );
      expect(minErr, `ql ${ql} min`).toBeGreaterThan(-0.13);
      expect(maxErr, `ql ${ql} max`).toBeLessThan(0.06);
    }
  });
});
