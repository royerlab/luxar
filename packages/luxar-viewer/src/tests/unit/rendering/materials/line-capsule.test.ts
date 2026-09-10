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
  CAPSULE_JOINT_PACKET_MIN_RADIUS_PX,
  CAPSULE_MIN_RADIUS_PX,
  type CapsuleJointLeg,
  CAPSULE_STENCIL_APRON_PX,
  capsuleJointCompositionError,
  capsuleJointStencilReach,
  capsuleLegField,
  capsuleLegWidthScale,
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

/**
 * Drop `//` and block comments so prose can never satisfy — or break — a match.
 * The WRITE-COUNT pin below needs it most: a comment mentioning `extA =` reads
 * as a second assignment and would fail that branch spuriously.
 */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

/**
 * Comment-stripped, whitespace-FREE source — the form every vertex-surface pin
 * below matches on, so a WHITESPACE reformat (prettier breaking a long TSL
 * chain) cannot break them while a dropped disjunct, a flipped comparison or a
 * changed threshold must. Same spirit as the sibling lock in
 * `line/join-width-tsl.test.ts`.
 *
 * Its limits, so no one over-reads a green pin: only COMMENTS are neutralized
 * (a string literal carrying the pinned text would satisfy a pin), the
 * stripper is not string-aware (a quoted block-comment delimiter pair would
 * swallow real code), and a non-whitespace-preserving edit with identical
 * behaviour still breaks a pin — hoisting `vec2 qhat = qq / ql;` in GLSL, or
 * swapping two disjuncts, would each need the pin updated.
 */
function squash(source: string): string {
  return stripComments(source).replace(/\s+/g, '');
}

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../../..');

function readSource(relativeToSrc: string): string {
  return readFileSync(path.join(SRC_ROOT, relativeToSrc), 'utf8');
}

/**
 * The BODY of one vertex surface's `needPacket<end>` branch, brace-matched from
 * the branch header, in BOTH matching forms:
 *
 * - `squashed` — whitespace-FREE ({@link squash}), for the reach-form literals,
 *   so a behaviour-identical prettier rewrap of a TSL assign cannot red the pin
 *   (the same reason the sibling locks match whitespace-free).
 * - `spaced` — whitespace-COLLAPSED, for the write-count regexes. They are
 *   whitespace-insensitive already, and they need the token boundary their
 *   `(?<![\w$])` lookbehind rests on: squashed, `else extA = …` becomes
 *   `elseextA=`, the lookbehind fails, and a second write goes UNCOUNTED.
 *
 * A slice, not a whole-source grep: the half-disc reach appears twice per end
 * (the `Else` arm and the width-gated arm) and is CORRECT in both, so only a
 * scoped assertion can say which arm assigns what — a whole-source
 * `not.toContain` would forbid the right answer everywhere. Fails loudly when
 * the header is missing or the braces never balance: a locator that silently
 * returned an empty slice would go vacuous under exactly the refactor this pin
 * exists to survive.
 *
 * The header lookup runs ONCE, on squashed text (a rewrapped header still
 * matches); a squashed→stripped index map then re-cuts the same span with its
 * spacing intact, so the two forms cannot drift apart.
 */
function packetBranchBody(
  label: string,
  source: string,
  header: string
): { squashed: string; spaced: string } {
  const stripped = stripComments(source);
  const toStripped: number[] = [];
  let squashed = '';
  for (let i = 0; i < stripped.length; i++) {
    if (/\s/.test(stripped[i])) continue;
    squashed += stripped[i];
    toStripped.push(i);
  }
  const headerAt = squashed.indexOf(squash(header));
  if (headerAt < 0) throw new Error(`${label}: packet branch header '${header}' not found`);
  const open = squashed.indexOf('{', headerAt);
  if (open < 0) throw new Error(`${label}: no '{' after '${header}'`);
  let depth = 0;
  for (let i = open; i < squashed.length; i++) {
    if (squashed[i] === '{') depth += 1;
    else if (squashed[i] === '}' && --depth === 0) {
      return {
        squashed: squashed.slice(open + 1, i),
        spaced: stripped.slice(toStripped[open] + 1, toStripped[i]).replace(/\s+/g, ' '),
      };
    }
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

  it('joint constants: 1.5 px AA floor, 4 px packet width gate, 2% deficit gate', () => {
    expect(CAPSULE_MIN_RADIUS_PX).toBe(1.5); // matches the quad's AA floor
    // Pin the VALUE, not just the interpolated spelling. The source pins below
    // fold the constant in, so they stay green at any value; the numeric rows
    // DO breach when it moves (0 → four rows, 8 → the apron row), but as a
    // scatter of composition failures rather than "the gate changed".
    expect(CAPSULE_JOINT_PACKET_MIN_RADIUS_PX).toBe(4.0);
    // Same reasoning for the taper gate, plus a truncation of its own: both
    // GLSL surfaces inline it at toFixed(2) (`G.DEFICIT_GATE`), so a change
    // finer than 0.005 never reaches a shader at all — and the clause-chain
    // pin below, which builds its expected text from the same toFixed(2),
    // cannot see one either. This is the only assertion that can.
    expect(CAPSULE_JOINT_DEFICIT_GATE).toBe(0.02);
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
    // Worth having because the codegen snapshots do not run in CI and the
    // parity suite cannot distinguish this TSL-only width-source regression;
    // a revert to the half-disc reach could otherwise ship green, chopping
    // the deficit rule's light on WebGPU alone.
    //
    // WHAT IS MATCHED AGAINST WHAT. The reach-form literals run on
    // {@link squash}ed text — comment-stripped AND whitespace-free on both
    // sides — so a behaviour-identical prettier rewrap
    // (`extA.assign(\n  rMax.add(APRON)\n);`) cannot red them, while an
    // operand, order or spelling change still must. The write COUNT runs on
    // whitespace-COLLAPSED text instead, and that difference is load-bearing:
    // the regexes are whitespace-insensitive anyway, but squashed they lose the
    // token boundary their `(?<![\w$])` lookbehind needs — `else extA = …`
    // becomes `elseextA=`, the lookbehind fails, and a second write inside an
    // `if`/`else` (the shape a real build-flag edit takes) goes UNCOUNTED while
    // both `toContain` arms stay green by design.
    //
    // MEASURED SCOPE — every case below was run as a mutant against this test.
    // CAUGHT: the literal half-disc revert; a `min()` wrapper around the
    // full-disc operand (all three spellings tried, including swapped
    // operands, because each one displaces the `extA=rMax+` prefix); a
    // SECOND write inside the branch, whether plain (`extA = …`), compound
    // (`extA *= …`), from the TSL assign family (`extA.mulAssign(…)`) or
    // guarded by an inline `if`/`else` in either language; a renamed branch
    // header (fails closed, by throw).
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
        expect(body.squashed, `${label} ${end}: full-disc reach`).toContain(
          squash(`ext${end} = rMax + `)
        );
        expect(body.squashed, `${label} ${end}: half-disc reach`).not.toContain(
          squash('abs(nLoc.y) * rMax')
        );
        // Compound forms included (`*=`, `+=`, …): a bare `=` counter reads
        // `extA *= abs(nLoc.y);` as no write at all. Same rule #1494 landed
        // for `rEnd`.
        expect(
          assignmentsTo(body.spaced, new RegExp(`(?<![\\w$])ext${end}\\s*[-+*/]?=(?!=)`, 'g')),
          `${label} ${end}: the packet branch must write ext${end} exactly once`
        ).toBe(1);
      }
    }
    for (const [label, relativeToSrc] of TSL_VERTEX_SURFACES) {
      const source = readSource(relativeToSrc);
      for (const end of ['A', 'B'] as const) {
        const body = packetBranchBody(`${label} ${end}`, source, `If(needPacket${end}, () => {`);
        expect(body.squashed, `${label} ${end}: full-disc reach`).toContain(
          squash(`ext${end}.assign(rMax.add(`)
        );
        expect(body.squashed, `${label} ${end}: half-disc reach`).not.toContain(
          squash('abs(nLoc.y).mul(rMax)')
        );
        // The whole TSL assign family, not just `.assign(`: `.mulAssign(`,
        // `.addAssign(` and friends all write the var in place.
        expect(
          assignmentsTo(body.spaced, new RegExp(`(?<![\\w$])ext${end}\\.\\w*[Aa]ssign\\(`, 'g')),
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

  it('a gentle hairline joint builds NO packet, so it reserves no full disc (#1488)', () => {
    // The width half of the packet gate, asserted on the MODEL's boolean rather
    // than on shader text: below `CAPSULE_JOINT_PACKET_MIN_RADIUS_PX` (and with
    // #1495's sharp-turn escape shut) not one of the four clauses can fire, so
    // a hairline joint gets neither a deficit term NOR the full-disc reach that
    // term needs. The sibling row 'every surface gates the packet by width OR a
    // floored sharp turn (#1495)' pins the same rule as SOURCE TEXT on the four
    // shader surfaces; this one pins that the CPU model agrees, and it reads the
    // decision where it actually matters, through the reach seam.
    //
    // Why the boolean and not a sweep bound: on this geometry, at the step 0.5
    // used below, gated measures −0.051291 and ungated −0.038773 — the faithful
    // model scores WORSE, so any bound tight enough to notice the gate would be
    // rewarding its removal. And the reach itself is not what moves: with the
    // gate in place, forcing the full rule and forcing the half rule both
    // measure −0.051291, bit-identical, since below 4 px the two reaches differ
    // by under 2 px of near-zero profile. That is precisely WHY the shaders gate
    // there — the deficit is sub-pixel — and precisely why only the boolean is
    // worth pinning. (Both figures re-measured on this build.)
    const hairline = 1.5; // the AA radius floor ⇒ rMax 2.0, under the 4 px gate
    const fat = 10; // ⇒ rMax 10.5, over it
    const packetFlags = (r: number): boolean[] => {
      const seen: boolean[] = [];
      const spy = (rMax: number, ny: number, hasPacket: boolean): number => {
        seen.push(hasPacket);
        return capsuleJointStencilReach(rMax, ny, hasPacket);
      };
      // A partner shorter than 2·rJoint holds the LENGTH clause wide open on
      // the leg under test (the partner leg's own clauses are all shut — its
      // length clause reads 8r < 2r — so its gate needs no width gate at all),
      // so the WIDTH comparison is the only thing that can shut the
      // gate — provided #1495's escape stays shut, which is why the turn is
      // gentle: the legs meet 60° from straight, so both ends measure
      // qx = −0.5 exactly against the `qx > 0.5` sharp test. The escape's other
      // conjunct IS satisfied here (both raw radii sit exactly on the AA floor),
      // so the turn is the only thing holding it: loosen that test past −0.5
      // (`qx > −0.6`) and the packet opens on the hairline pair, reddening this
      // row.
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
    expect(thin.some(Boolean), 'a gentle hairline joint must never build a packet').toBe(false);
    expect(packetFlags(fat).some(Boolean), 'the same joint, fat, must build one').toBe(true);
  });

  it('the joint partition is never width-gated (a hairline joint would double)', () => {
    // The drawn radius is FLOORED at the AA minimum, so even a hairline
    // draws a 1.5 px disc at each end. Skipping the cut there to save the
    // partner fetch would leave two full caps stacked on the shared
    // vertex: measured +1.00 of peak (a 2x bead as wide as the line
    // itself) at every bend angle, so there is no width at which the
    // overlap is sub-pixel. Only the deficit PACKET is width-gated, and even
    // that gate is lifted at a sharp turn on an at-or-above-floor segment
    // (#1495, below).
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
    // exporting text, and their ONLY other direct coverage is the checked-in
    // codegen snapshots, which do not run in CI, so a TSL-only deletion of
    // this gate is invisible to CI.
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
    //
    // This covers the four CLAUSES only. Reaching them also needs the width
    // gate they sit inside, which is pinned separately per surface by 'every
    // surface gates the packet by width OR a floored sharp turn (#1495)'.
    const gate = CAPSULE_JOINT_DEFICIT_GATE.toFixed(2);
    // GLSL inlines the gate as a literal (shader-glsl-capsule.ts:53);
    // building the expected text from the constant means a deliberate gate
    // change flows through instead of turning this test red — the constant's
    // own value is pinned in the joint-constants test above instead.
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
  // The numeric sweep the #1487 review asked for: all three of #1494
  // (wrong vertex radius), #1488 (reach shortfall) and #1490 (missing far
  // cap) blow these bounds. All three additionally have source locks over the
  // four shader surfaces now — #1494 in `line/capsule-partner-radius.test.ts`,
  // #1490 in `line/capsule-joint-packet-source-lock.test.ts`, #1488 in the
  // packet-branch pin above — but those bind TEXT. The sweep still binds all
  // three in the CPU model by VALUE, which no text pin can.
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
    // (−0.01, 0.005) from the pre-fix ±(0.13, 0.06) these rows were first
    // written with. That slack was the reason the rows only half-caught
    // #1488: with the model now clipped by the stencil, a revert to the
    // half-disc reach measures −0.4096 / −0.3666 / −0.1131 / −0.0325
    // (ql 2/4/8/15), and −0.13 let the ql 8 and ql 15 rows through — the
    // shallow end of the very gradient the reach rule controls.
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

  it('the sweep is NaN-FATAL: a degenerate row cannot score a vacuous 0', () => {
    // A zero projected partner length makes the packet's radius gradient
    // non-finite (671 NaN fragments on this geometry), and `<`/`>` are both
    // false for NaN — so without the guard the sweep reports a PERFECT
    // {0, 0} and any future degenerate row passes while measuring nothing.
    expect(() => capsuleJointCompositionError(leg(180, 10, 10, 60), leg(-60, 10, 10, 0))).toThrow(
      /non-finite/
    );
  });

  it('hairline joints keep the packet at a sharp turn (#1495)', () => {
    // Below the packet's width gate (rMax = max(rJoint, rFar) + apron ≤ 4 px)
    // the vertex stage skips the packet block entirely, so NOT ONE of the
    // four clauses can fire — only the sharp-turn exception keeps the
    // lengthwise chop out of a thin polyline's bend. With it every row here
    // measures ~1e-14; with the exception reverted (width gate alone) they
    // measure −0.0175 … −0.7151, so the ±0.005 bounds kill that revert on
    // EVERY row. The widening rows are the weak ones (−0.0175 / −0.0249 at
    // r = 3.5): a partner that widens is wide at its own far end, so its own
    // width gate is already open and its packet refills most of the cut.
    // Radii are RAW (see CapsuleJointLeg) and every one here is >= the AA
    // floor, which is what the exception requires — a thinner leg draws
    // floored and dimmed and deliberately keeps the plain cut (see the
    // floor test below). Hence tapering only appears where r/2 still clears
    // the floor.
    // The step scales with the radius because the wide rows' default 0.5 px
    // is coarse against a 1.5 px disc. It buys accuracy, not detection: the
    // r = 1.5 / 160° row reverts to −0.5084 at step 0.5 vs −0.5253 at 0.075,
    // ~3% apart, and both breach by two orders of magnitude.
    for (const r of [1.5, 2, 3.5]) {
      const step = Math.min(0.125, r / 20);
      const own = leg(180, r, r, 6 * r);
      for (const turn of [135, 160]) {
        const partners: Array<[string, CapsuleJointLeg]> = [
          ['const short', leg(-turn, r, r, 0.4 * r)],
          ['widening', leg(-turn, r, 2 * r, 6 * r)],
          ['congruent long', leg(-turn, r, r, 6 * r)],
        ];
        if (r / 2 >= CAPSULE_MIN_RADIUS_PX) {
          partners.push(['tapering', leg(-turn, r, r / 2, 6 * r)]);
        }
        for (const [kind, partner] of partners) {
          const { minErr, maxErr } = capsuleJointCompositionError(own, partner, undefined, step);
          expect(minErr, `r ${r} turn ${turn}° ${kind} min`).toBeGreaterThan(-0.005);
          expect(maxErr, `r ${r} turn ${turn}° ${kind} max`).toBeLessThan(0.005);
        }
      }
    }
  }, 60_000);

  it('the 120° exception threshold is pinned from BOTH sides (#1495)', () => {
    // The sweep above passes even with the width gate left out of the model
    // entirely (the pre-#1495 state: the short/ratio/sharp clauses build a
    // packet for all four partner kinds), so it does not pin the gate. These
    // rows do, by pinning the residual the exception deliberately leaves:
    // AT or below 120° a hairline joint still hard-cuts, because the
    // exception reuses the sharp clause's own `> 0.5` axis-dot rather than
    // inventing a second, softer threshold. This is the documented trade,
    // NOT a target — the bound is a floor to sit below, not a budget. Nor is
    // it a bound on the residual: at 120° with r = 3.5 the chop grows with a
    // shorter partner, −0.190 at ql = 0.4 r through −0.386 at 0.05 r.
    const shortPartner = (r: number, turn: number) => leg(-turn, r, r, 0.4 * r);
    for (const r of [1.5, 3.5]) {
      const step = Math.min(0.125, r / 20);
      const own = leg(180, r, r, 6 * r);
      // Just BELOW: gated, so the chop stands (−0.179 at r = 1.5, −0.182 at
      // 3.5). Leave the width gate out of the model and it relaxes to −0.060 /
      // −0.027 — so only the r = 3.5 row kills that mutation, the r = 1.5 one
      // still passes at −0.060. Loosening the sharp clause to `> -0.5` takes
      // both to −0.000, which either row kills.
      const below = capsuleJointCompositionError(own, shortPartner(r, 119), undefined, step);
      expect(below.minErr, `r ${r} 119° still hard-cuts`).toBeLessThan(-0.05);
      // Just ABOVE: the exception fires and the pair composes exactly.
      // Tighten the clause to `> 0.707` (135°) and this breaches at −0.19.
      const above = capsuleJointCompositionError(own, shortPartner(r, 121), undefined, step);
      expect(above.minErr, `r ${r} 121° min`).toBeGreaterThan(-0.005);
      expect(above.maxErr, `r ${r} 121° max`).toBeLessThan(0.005);
    }
    // The threshold is HARD: a projected bend sweeping through it pops. Worth
    // seeing in one place, at the tightest spacing that still isolates it.
    const own2 = leg(180, 2, 2, 12);
    const pop = capsuleJointCompositionError(own2, shortPartner(2, 119.5), undefined, 0.1);
    const clean = capsuleJointCompositionError(own2, shortPartner(2, 120.5), undefined, 0.1);
    expect(pop.minErr, 'r 2 119.5°').toBeLessThan(-0.05); // measured −0.175
    expect(clean.minErr, 'r 2 120.5°').toBeGreaterThan(-0.005); // measured −0.000
  });

  it('a sub-floor leg gets NO packet, so the pair cannot bead (#1495)', () => {
    // The reason for the floor conjunct, in the direction that matters. The
    // fragment dims each leg by widthScale = min(raw/1.5, 1) while the deficit
    // cancels the partner's UNSCALED profile, so two legs at different scales
    // no longer compose to max(mine, partner): the pair OVER-fills, and a bead
    // is the artifact class this primitive exists to remove. The model carries
    // that factor, so the bead is visible here.
    expect(capsuleLegWidthScale(leg(180, CAPSULE_MIN_RADIUS_PX, 3, 10), 0)).toBe(1);
    expect(capsuleLegWidthScale(leg(180, 0.4, 1.4, 20), 0)).toBeCloseTo(0.4 / 1.5, 12);
    // Either END below the floor is enough to dim the segment, hence the
    // conjunct's min(rawA, rawB); both partners share rJoint (the pair
    // invariant) and turn 175°, so every other clause would open the packet.
    for (const [name, own, partner] of [
      ['sub-floor FAR end', leg(180, CAPSULE_MIN_RADIUS_PX, 0.4, 9), leg(-175, 1.5, 1.5, 3)],
      ['sub-floor JOINT end', leg(180, 0.4, 1.4, 9), leg(-175, 0.4, 1.4, 0.6)],
    ] as Array<[string, CapsuleJointLeg, CapsuleJointLeg]>) {
      const { maxErr } = capsuleJointCompositionError(own, partner, undefined, 0.075);
      // Bead-free is the whole assertion: allow the packet here and maxErr
      // goes to +0.137 (far end) / +0.164 (joint end).
      expect(maxErr, `${name} must not bead`).toBeLessThan(0.01);
      // The chop these rows keep (−0.41 / −0.47) is the accepted PRICE, not a
      // goal, so it is deliberately NOT asserted — a future fix that removes
      // it without reintroducing the bead should not fail here. It is a real
      // price: for a DRAWABLE sub-floor taper (own raw 1.2 → 3.0) allowing the
      // packet measures −0.05…−0.07 against −0.33…−0.79 gated, and against a
      // congruent partner −0.002…−0.004 against −0.03…−0.10 with no bead at
      // all. We take the dim gap over the bright bead; see the constant.
    }
  });

  it('the raw-floor threshold is a THIRD residual, pinned both sides (#1495)', () => {
    // The exception's floor is as hard as its angle: two legs scaled together
    // are exact at raw 1.5 and chop at 1.4999, a pop a plain zoom sweeps
    // through — and a bigger one than the 120° pop (up to −0.72 vs −0.18).
    // Neither is a bead, so both are accepted; both are documented.
    const pair = (raw: number, turn: number) =>
      capsuleJointCompositionError(
        leg(180, raw, raw, 9),
        leg(-turn, raw, raw, 0.4 * raw),
        undefined,
        0.075
      );
    for (const [turn, expected] of [
      [125, -0.211],
      [160, -0.525],
      [175, -0.72],
    ] as Array<[number, number]>) {
      // At the floor exactly: the exception fires, the pair composes.
      expect(pair(CAPSULE_MIN_RADIUS_PX, turn).minErr, `raw 1.5, ${turn}°`).toBeGreaterThan(-0.005);
      // A hair below: gated off, the chop stands (and the measured value is
      // within a few % of `expected` — quoted so the pop's size is on record).
      const below = pair(1.4999, turn).minErr;
      expect(below, `raw 1.4999, ${turn}° (measured ${expected})`).toBeLessThan(-0.05);
      expect(below, `raw 1.4999, ${turn}° magnitude`).toBeGreaterThan(expected * 1.1);
    }
  });

  it("the gate's rMax is max(rA, rB) PLUS the stencil apron (#1495)", () => {
    // Two rows whose gate decision hinges on the arithmetic itself; without
    // them, building rMax from min() or dropping the apron passes every other
    // row in this file.
    // A widening own leg: max() clears the gate (−0.023), min() does not
    // (−0.060).
    const widening = capsuleJointCompositionError(
      leg(180, 3, 10, 30),
      leg(-60, 3, 3, 1),
      undefined,
      0.15
    );
    expect(widening.minErr, 'rMax from max(rA, rB)').toBeGreaterThan(-0.04);
    // Straddling the gate by less than the apron: with it 4.25 > 4 (−0.029),
    // without it 3.75 < 4 (−0.079). Tightening the gate to 8 px does the same.
    const apron = capsuleJointCompositionError(
      leg(180, 3.75, 3.75, 22),
      leg(-100, 3.75, 3.75, 1.5),
      undefined,
      0.15
    );
    expect(apron.minErr, 'rMax includes the apron').toBeGreaterThan(-0.05);
  });

  it('every surface gates the packet by width OR a floored sharp turn (#1495)', () => {
    // The model above can only prove the exception is RIGHT; these pins prove
    // all four vertex surfaces carry it, at BOTH ends. Three things must hold
    // per site: the width threshold itself stays (the gentle-hairline discard
    // is deliberate), it is OR-ed with the end-appropriate sharp-turn test —
    // the same expression the surface's own `needPacket` sharp clause uses,
    // so a reader sees they are one test — and that disjunct is AND-ed with
    // the AA-floor condition on MY OWN pre-clamp radii (without it the pair
    // over-brightens; see the floor test above). The pick twins must match
    // the visual ones or hover desyncs from pixels.
    for (const [name, src] of [
      ['glsl material', CAPSULE_LINE_VERTEX_SHADER],
      ['glsl pick', CAPSULE_LINE_PICK_VERTEX_SHADER],
    ] as const) {
      const bare = squash(src);
      expect(bare, `${name} end A`).toContain(
        'if(rMax>packetMinRadius||(dot(qq/ql,u)>0.5&&min(rawA,rawB)>=minRadius)){'
      );
      expect(bare, `${name} end B`).toContain(
        'if(rMax>packetMinRadius||(dot(qq/ql,u)<-0.5&&min(rawA,rawB)>=minRadius)){'
      );
    }
    for (const [name, rel] of [
      ['tsl material', 'rendering/materials/line/shader-tsl-capsule.ts'],
      ['tsl pick', 'rendering/picking/line/pick-capsule.tsl.ts'],
    ] as const) {
      const bare = squash(readSource(rel));
      const gate = 'If(rMax.greaterThan(packetMinRadius).or(dot(qhat,u)';
      const floored = '.and(min(rawA,rawB).greaterThanEqual(minRadius)))';
      expect(bare, `${name} end A`).toContain(`${gate}.greaterThan(0.5)${floored}`);
      expect(bare, `${name} end B`).toContain(`${gate}.lessThan(-0.5)${floored}`);
    }
  });
});

describe('cut-normal precision regression lock (#1502)', () => {
  // A per-leg snap of the cut normal (`round(nLoc * 1024.0) / 1024.0`) does
  // not cancel between the two legs of a joint — each builds it in its own
  // local (u, v) basis — and since the cut spans the full stencil the
  // disagreement grows with distance from the joint, banding near-hairpin
  // joints along the whole rod. The composition sweep above only sees a snap
  // re-added to the CPU model; unit CI otherwise pins the TSL pair only through
  // codegen snapshots that do not run in CI, hence a source-text lock in the
  // spirit of `line/join-width-tsl.test.ts` — whose two small helpers are
  // copied rather than exported, to leave that file alone; they now sit at
  // this file's top, shared with the packet-gate lock (#1497). Every match
  // runs on COMMENT-STRIPPED text, so the notes at the shader sites can
  // neither satisfy nor break it.

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
