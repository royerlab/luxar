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

describe('cut-normal precision regression lock (#1502)', () => {
  // A per-leg snap of the cut normal (`round(nLoc * 1024.0) / 1024.0`) does
  // not cancel between the two legs of a joint — each builds it in its own
  // local (u, v) basis — and since the cut spans the full stencil the
  // disagreement grows with distance from the joint, banding near-hairpin
  // joints along the whole rod. The composition sweep above only sees a snap
  // re-added to the CPU model; unit CI pins the four SHADER surfaces with
  // nothing at all (the TSL pair only through the codegen snapshots, and
  // `e2e-tests` in `ci.yml` is `if: false`), hence a source-text lock in the
  // spirit of `line/join-width-tsl.test.ts` — whose two small helpers are
  // copied rather than exported, to leave that file alone. Every match runs
  // on COMMENT-STRIPPED text, so the notes at the shader sites can neither
  // satisfy nor break it.
  const HERE = path.dirname(fileURLToPath(import.meta.url));
  const SRC_ROOT = path.resolve(HERE, '../../../..');
  const readSource = (rel: string): string => readFileSync(path.join(SRC_ROOT, rel), 'utf8');
  const stripComments = (source: string): string =>
    source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');

  // [source, its own cut-normal shape, how many construction sites it has]:
  // two per shader (end A and end B), one in the CPU model. Keyed on the
  // locals, never on the varying names — those get renamed as the joint
  // packing evolves (#1540).
  // The 40-char gap is slack for prettier: `vec2(dot(n2, u), dot(n2, v))`
  // sits ~71 chars deep against printWidth 100, so one more nesting level
  // wraps the two `dot()` args and a tighter gap would red on a reformat.
  const SOURCES: Array<[string, RegExp, number]> = [
    ['rendering/materials/_shared/line-capsule.ts', /legLocal\(leg,\s*partner\.dir\[0\]/g, 1],
    [
      'rendering/materials/line/shader-glsl-capsule.ts',
      /dot\(n2,\s*u\)[\s\S]{0,40}dot\(n2,\s*v\)/g,
      2,
    ],
    [
      'rendering/materials/line/shader-tsl-capsule.ts',
      /dot\(n2,\s*u\)[\s\S]{0,40}dot\(n2,\s*v\)/g,
      2,
    ],
    ['rendering/picking/line/shaders-capsule.ts', /dot\(n2,\s*u\)[\s\S]{0,40}dot\(n2,\s*v\)/g, 2],
    ['rendering/picking/line/pick-capsule.tsl.ts', /dot\(n2,\s*u\)[\s\S]{0,40}dot\(n2,\s*v\)/g, 2],
  ];

  // Each site is scanned within ±600 chars: far enough to catch a snap on
  // the next line, far short of the ≥2350-char gap to the other end's site,
  // so the rest of these 263-649 line modules stays out of scope. None of
  // them holds a `floor()` or a `1024` today, but siblings like
  // `picking/picking-system.ts` do (integer texel math, buffer dims), so a
  // whole-file scan would be one texel helper away from a false red.
  const WINDOW = 600;

  it('keeps the cut normal unquantised at every construction site', () => {
    for (const [rel, shape, count] of SOURCES) {
      const stripped = stripComments(readSource(rel));
      expect(stripped.length, `${rel}: source too short — did the read fail?`).toBeGreaterThan(
        1000
      );
      const matches = [...stripped.matchAll(shape)];
      expect(
        matches.length,
        `${rel}: expected ${count} cut-normal construction site(s), found ` +
          `${matches.length} — the shape regex drifted, so this pin now guards ` +
          'the wrong thing (or nothing)'
      ).toBe(count);
      for (const match of matches) {
        const start = Math.max(0, match.index - WINDOW);
        const window = stripped.slice(start, match.index + match[0].length + WINDOW);
        // `\b1024\b` so `10240`/`0.1024` cannot false-red (it still matches
        // the historical `1024.0`, since `.` is a non-word character), and
        // `round(?:Even)?` to catch GLSL ES 3.0's `roundEven()`.
        expect(
          window,
          `${rel}: the cut normal must stay full precision — the two legs ` +
            'build it in different local bases, so any per-leg rounding never ' +
            'cancels and the error grows with distance along the rod (#1502). ' +
            'Out of scope here: an explicit narrowing or packing of the normal ' +
            '(packHalf2x16, an ivec2 cast, fract-based rounding) — a token scan ' +
            'cannot tell a cut-normal lane from an unrelated one.'
        ).not.toMatch(/\b(?:round(?:Even)?|floor|trunc|ceil)\s*\(|\b1024\b/);
      }
    }
  });

  it('keeps every GLSL capsule stage pinned at highp float (#1502)', () => {
    // three.js prepends `precision <capabilities.precision> float;` to the
    // module source, and that precision follows the renderer's
    // `webgl.renderer.precision` config, which can be mediump. Since the
    // module source lands AFTER the prefix, each stage's own
    // `precision highp float;` is what actually pins the cut normal — losing
    // it narrows the normal per leg with nothing in the source looking wrong.
    // Checked per STAGE, on the compiled strings: a whole-file `toContain`
    // cannot tell "the vertex stage lost its highp" from "both are fine".
    // The precision STATEMENT, not the bare token: the quad sibling and
    // `picking/point/shaders.ts` legitimately declare individual varyings
    // mediump/lowp for bandwidth, which is honest work here too. TSL is
    // excluded — three.js manages precision there, so there is no pragma.
    const STAGES: Array<[string, string]> = [
      ['shader-glsl-capsule.ts vertex', CAPSULE_LINE_VERTEX_SHADER],
      ['shader-glsl-capsule.ts fragment', CAPSULE_LINE_FRAGMENT_SHADER],
      ['shaders-capsule.ts (pick) vertex', CAPSULE_LINE_PICK_VERTEX_SHADER],
      ['shaders-capsule.ts (pick) fragment', CAPSULE_LINE_PICK_FRAGMENT_SHADER],
    ];
    for (const [label, src] of STAGES) {
      const stripped = stripComments(src);
      expect(
        stripped,
        `${label}: must still declare "precision highp float;" — losing it ` +
          "lets three.js's prefix (from the renderer's precision config) run " +
          'the cut normal at mediump/lowp, which reintroduces #1502'
      ).toContain('precision highp float;');
      expect(
        stripped,
        `${label}: must not narrow the default float precision — a later ` +
          'precision statement overrides the highp one above it and ' +
          'reintroduces #1502'
      ).not.toMatch(/precision\s+(?:mediump|lowp)\s+float/);
    }
  });
});
