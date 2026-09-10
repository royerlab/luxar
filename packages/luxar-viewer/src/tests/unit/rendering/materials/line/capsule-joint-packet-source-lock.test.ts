/**
 * Capsule joint DEFICIT-PACKET source lock — the whole packet contract, across
 * all four capsule shader surfaces. #1490 (the bounded partner rod) is the
 * defect that prompted it; #1495 / #1501 (the four packet-gate clauses) share
 * the same live hole and are locked in a sibling describe.
 *
 * Scope note, so the file is not read as a #1490 regression suite: #1490 was
 * purely the FRAGMENT ignoring `cut.w`. The vertex stage already packed `.z`
 * and `.w` and already called the partner before that fix, so reverting #1490
 * reds only the far-cap cases here — the packing, call and gate-clause guards
 * stay green. They lock neighbouring parts of the same contract that are
 * equally unpinned, not #1490 itself.
 *
 * On the partner's side of a joint bisector each capsule leg renders
 * `max(mine − partner, 0)`, rebuilding the partner's field from a packed
 * packet: a radius gradient in `cut.z` and the partner's PROJECTED LENGTH in
 * `cut.w`. #1490 was that the reconstruction ignored `cut.w` on the profile
 * side — an UNBOUNDED rod, tapering (or growing) forever with no far cap — so
 * a short partner was subtracted far past where it actually ends. The fix
 * bounds the rod at both ends:
 *
 *   GLSL  rp = max(rEnd + cut.z * clamp(xp, 0.0, cut.w), 1e-4);
 *         op = max(max(-xp, xp - cut.w), 0.0);
 *   TSL   rp = max(rEnd.add(cut.z.mul(clamp(xp, 0.0, cut.w))), float(1e-4));
 *         op = max(max(xp.negate(), xp.sub(cut.w)), 0.0);
 *
 * Why a string-level lock rather than "just test the behaviour":
 *
 * - The behaviour IS tested — but only through the CPU mirror. Deleting the
 *   far cap from `_shared/line-capsule.ts::capsuleJointRenderLeg` fails SIX
 *   TESTS in `tests/unit/rendering/materials/line-capsule.test.ts`, all under
 *   `joint composition — the rendered pair tracks max(mine, partner)` — among
 *   them `short partners (#1488/#1490)` (first failing row `ql 2 max` 0.715
 *   against a 0.005 bound) and `constant-width and widening cases (#1495)`
 *   (first failing row `const partner 120° ql2 min` −0.399 against −0.01).
 *   Each test stops at its first failing row, so the row count is at least six
 *   and unknown beyond that — the TEST count is the measured figure.
 * - Deleting the far cap's OVERSHOOT half (`xp - cut.w` in `op`) from all four
 *   SHADER sources and leaving the mirror alone reds NOTHING BUT THIS FILE:
 *   measured here, exactly four failures — the `caps the overshoot` case on
 *   each of the four surfaces — with every other file in the viewer unit suite
 *   green. Stated without absolute file/test counts on purpose: those move
 *   whenever an unrelated suite lands (they have gone stale twice already),
 *   while the ASYMMETRY is the claim, and it is what a re-measure should check.
 *   Nothing else ties that shader text to the model, so
 *   the exact regression #1490 describes (a packing refactor applied
 *   identically to all four twins) could land again invisibly. The RADIUS half
 *   is no longer sole: #1563's `line/capsule-partner-radius.test.ts` pins the
 *   whole `rp` base — `clamp(xp, 0.0, cut.w)` included — on all four surfaces,
 *   so removing BOTH halves reds the eight cases here plus four of its.
 *   Neither half was covered anywhere when this file was written.
 * - The TSL↔GLSL parity harness could not see it either: until the fixture
 *   added alongside this file
 *   (`line-capsule-joint-short-partner`) no capsule joint fixture published a
 *   deficit packet with a SHORT partner at all — every one had uniform widths,
 *   and `line-capsule-fold`'s packet carries gradient 0 with ql ≈ 28.8 px, far
 *   outside the joint disc, where the far-cap term is inert.
 * - Mocking the TSL factories and building them records nothing: the vertex
 *   stage is traced inside `Fn(() => {...})`, whose body only runs during a
 *   real node build, which needs a GPU/WebGL backend unavailable under jsdom.
 *   `join-width-tsl.test.ts` documents that dead end at length; this file
 *   follows its pattern (read from disk, strip comments, grep).
 *
 * What each guard catches, per surface:
 *
 * 1. FAR-CAP BOUND ON THE RADIUS — `clamp(xp, 0.0, cut.w)` inside `rp`, and
 *    the unbounded `cut.z * max(xp, 0.0)` provably absent. Without the clamp
 *    the phantom rod keeps tapering: at the short-partner fixture's gradient
 *    of −1.65 px/px it shrinks to the 1e-4 floor 7.7 px along the partner
 *    axis, still INSIDE the 12.65 px joint disc, so the deficit collapses to
 *    the leg's full profile and the joint over-draws.
 * 2. FAR CAP ON THE OVERSHOOT — `xp - cut.w` in `op`, i.e. BOTH caps. The near
 *    cap alone leaves the rod open at its far end.
 * 3. THE PACKET'S LENGTH LANE — the vertex stage must still write `ql` into
 *    the packet's `.w`, PER END. Guards 1 and 2 clamp against `cut.w`; if the
 *    vertex stopped writing it, they would silently bound against 0 and the
 *    whole partner term would vanish.
 * 4. THE PACKET'S GRADIENT LANE — `cut[AB].z`, again per end. Dropping it
 *    leaves a constant-radius phantom rod.
 * 5. THE PARTNER CALL ITSELF — `partnerProfile` / `luxarPartnerProfile` must
 *    still be INVOKED at both ends, on that end's own packet, sign, local
 *    frame and shared-vertex radius. Deleting a call keeps every other
 *    assertion here green: they all inspect the definition.
 * 6. THE PACKET'S TRANSPORT — `cut[AB].zw` into the matching `vPack` lane and
 *    back out into `pk[AB]`. Guards 3-5 pin the write and the read; swapping
 *    the two LANES in between leaves both halves individually correct, and is
 *    invisible to the CPU mirror (it has no packing) and to the parity
 *    fixtures (both backends would swap identically).
 *
 *    Guards 3-6 overlap `line-capsule.test.ts` on the GLSL side, but only
 *    partly, and these are the stronger form. Its `the joint composes by the
 *    DEFICIT rule: max(mine, partner), no fade` case matches `cut[AB].z = ...`
 *    and `cut[AB].w = ql;` with `[AB]` as a CHARACTER CLASS under `toMatch`, so
 *    one match satisfies it: deleting the B-end line leaves it green, and a
 *    mixed `cutA.z = (rpFarB - rB) / ql;` would satisfy it too. Its
 *    `toContain('luxarPartnerProfile')` matches the function DEFINITION, so
 *    deleting both GLSL call sites leaves it green as well, and its
 *    `toContain('unpackHalf2x16(vPack.x)')` survives a lane swap. The guards
 *    here are per-end and, for the call, keyed on the call spelling.
 * 7. THE PACKET GATE CLAUSES (#1495 / #1501) — all four, at BOTH ends of all
 *    four vertex surfaces (sibling describe). Dropping the sharp-turn clause
 *    alone measured −0.92 of peak in the 2r–3r partner band. A clause deletion
 *    is NOT invisible any more: #1548 pins the whole clause CHAIN as one exact
 *    string, per end and per surface, in `line-capsule.test.ts` — which is also
 *    what pins the `||` / `.or(...)` COMPOSITION, deliberately not restated
 *    here. These guards are the per-clause, HOIST-TOLERANT form of that pin
 *    (see the sharp-turn note in the describe below): they are what survives
 *    the cleanup the chain pin will red.
 * 8. THE CPU MIRROR — one brief text cross-check that it carries the same
 *    bounded rod. Deliberately brief: the numeric sweep above already catches
 *    a mirror-only regression BY VALUE, which is strictly better; this exists
 *    only so the four shaders and the model cannot drift apart silently.
 * 9. THE SURFACE LIST ITSELF — the four surfaces are hardcoded, so a fifth
 *    would inherit the whole hazard and escape silently. A completeness test
 *    walks `src/rendering` and asserts the set of modules mentioning a partner
 *    profile is exactly these four (the twin of `join-width-tsl.test.ts`'s
 *    `locks EVERY TSL factory that calls tslLineJoin` case).
 *
 * Overlap with the sibling lock: `line/capsule-partner-radius.test.ts` (#1494 /
 * #1563) came at the same four surfaces from the other side of the same
 * contract, and reaches three of the guards above — its base-radius pin carries
 * guard 1's `clamp(xp, 0.0, cut.w)`, its per-end call pins cover guard 5, and
 * its `src/rendering` scan is guard 9 under a different needle. Guards 2, 3, 4
 * and 6 are unique here. The overlap is left standing rather than pruned: it
 * costs a handful of static string matches, and each file stays a complete
 * statement of the contract it describes.
 *
 * Every match runs on comment-stripped, whitespace-squashed text: prose can
 * never satisfy a pin, and reformatting can never break one — but dropping
 * `cut.w` from an expression always does. (One exception: guard 9 matches on
 * stripped but UNsquashed text, since it only asks whether a module mentions a
 * partner profile at all.)
 *
 * One deliberate divergence from `join-width-tsl.test.ts`: it parses lazily
 * inside test bodies because an `expect()` throwing in a describe callback
 * collapses the file to zero tests. Both surface tables here are built at
 * module scope instead — they contain no `expect()`, only `readFileSync` and
 * string work, and vitest reports a collection-time throw as a loud failure
 * rather than a silent skip.
 */

import { readdirSync, readFileSync } from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { CAPSULE_JOINT_DEFICIT_GATE } from '../../../../../rendering/materials/_shared/line-capsule';
import {
  CAPSULE_LINE_FRAGMENT_SHADER,
  CAPSULE_LINE_VERTEX_SHADER,
} from '../../../../../rendering/materials/line/shader-glsl-capsule';
import {
  CAPSULE_LINE_PICK_FRAGMENT_SHADER,
  CAPSULE_LINE_PICK_VERTEX_SHADER,
} from '../../../../../rendering/picking/line/shaders-capsule';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../../../..');

function readSource(relativeToSrc: string): string {
  return readFileSync(path.join(SRC_ROOT, relativeToSrc), 'utf8');
}

/** Drop `//` and block comments so prose can never satisfy — or break — a grep. */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

/**
 * Remove ALL whitespace, so every needle below is spacing- and
 * line-break-insensitive while staying exact about tokens and operands.
 * Prettier reflows these expressions freely; `cut.w` never moves.
 */
function squash(source: string): string {
  return source.replace(/\s+/g, '');
}

/**
 * The squashed text of the single statement `marker` opens, from the match to
 * the next `;`.
 *
 * Scoping matters: `xp - cut.w` appearing SOMEWHERE in a 600-line file proves
 * nothing about the overshoot term, and `clamp(xp, ...)` reappears in the
 * radius of a neighbouring surface. None of the statements pinned here
 * contain an inner `;`, and each marker matches exactly once per surface
 * today — this takes the FIRST match, so a second `float rp =` elsewhere in a
 * fragment would go unexamined rather than red.
 */
function statement(text: string, marker: RegExp, label: string): string {
  const match = marker.exec(text);
  expect(match, `${label}: no statement matching ${marker}`).not.toBeNull();
  const start = match!.index;
  const end = text.indexOf(';', start);
  expect(end, `${label}: unterminated statement matching ${marker}`).toBeGreaterThan(start);
  return squash(text.slice(start, end + 1));
}

/**
 * The squashed argument text of every `name(...)` CALL in `text`, by
 * paren-matching (the argument lists nest, so a regex cannot delimit them).
 *
 * The GLSL definition `luxarPartnerProfile(vec4 cut, ...)` matches the same
 * `name(` opener as its calls, so callers filter on the leading `vec4(` that
 * only a call has.
 */
function callArguments(text: string, name: string): string[] {
  const opener = new RegExp(`\\b${name}\\(`, 'g');
  const out: string[] = [];
  let match: RegExpExecArray | null;
  while ((match = opener.exec(text)) !== null) {
    let depth = 1;
    let cursor = match.index + match[0].length;
    while (cursor < text.length && depth > 0) {
      if (text[cursor] === '(') depth += 1;
      else if (text[cursor] === ')') depth -= 1;
      cursor += 1;
    }
    expect(depth, `unbalanced parens in a ${name}(...) call`).toBe(0);
    out.push(text.slice(match.index + match[0].length, cursor - 1));
    opener.lastIndex = cursor;
  }
  return out;
}

/**
 * The per-END operands of the two partner-profile calls, with the END each
 * belongs to. Every one of them is a swap hazard in its own right — give end A
 * end B's packet, frame, sign or shared-vertex radius and the deficit rule
 * quietly reconstructs the wrong rod — so all four are pinned, and none is
 * left "loose for reformatting": the arguments are located by paren-matching,
 * so reformatting cannot move them anyway.
 *
 * The FRAME operand is matched by the presence or absence of `vAbLen` rather
 * than by spelling: end B evaluates in the shifted frame (`x - vAbLen` in
 * GLSL, `x.sub(vAbLen)` in TSL), end A in the unshifted one.
 */
const CALL_OPERANDS = [
  { end: 'A', packet: 'vCutN.xy,pkA', sign: ',1.0,', rEnd: 'pkR.x', shiftedFrame: false },
  { end: 'B', packet: 'vCutN.zw,pkB', sign: ',-1.0,', rEnd: 'pkR.y', shiftedFrame: true },
] as const;

/**
 * Assert the two partner-profile calls carry the right per-end operands.
 * Shared by both dialects: the only difference is the callee name and the
 * definition filter, both handled by `callArguments` plus the `vec4(` test.
 */
function expectPartnerCalls(fragment: string, callee: string, label: string): void {
  const calls = callArguments(squash(fragment), callee).filter((args) => args.startsWith('vec4('));
  expect(calls, `${label}: expected two ${callee} call sites`).toHaveLength(2);
  for (const operands of CALL_OPERANDS) {
    const matching = calls.filter((args) => args.includes(operands.packet));
    expect(
      matching,
      `${label}: expected exactly one ${callee} call on packet ${operands.packet}`
    ).toHaveLength(1);
    const args = matching[0];
    expect(
      args.includes(operands.sign),
      `${label}: end ${operands.end} needs sign ${operands.sign}`
    ).toBe(true);
    expect(
      args.includes(operands.rEnd),
      `${label}: end ${operands.end} must start the rod at ${operands.rEnd} (the SHARED-vertex radius, #1494)`
    ).toBe(true);
    expect(
      args.includes('vAbLen'),
      `${label}: end ${operands.end} must evaluate in the ${
        operands.shiftedFrame ? 'shifted (vAbLen)' : 'unshifted'
      } local frame`
    ).toBe(operands.shiftedFrame);
  }
}

/**
 * The packet's TRANSPORT lanes, pinned per end on every surface.
 *
 * The write (`cut[AB].z/.w`) and the read (`clamp(xp, 0, cut.w)`) are each
 * pinned elsewhere in this file, but swapping the two LANES between them —
 * `packHalf2x16(cutB.zw)` into `.x`, or `pkA` from `vPack.y` — leaves both
 * halves individually correct and every other assertion here green. So does
 * the CPU mirror's numeric sweep (the mirror has no packing at all) and the
 * parity fixtures (both backends swapped identically). It is exactly the
 * "one refactor applied to all four twins" failure this file exists to catch.
 *
 * The write side is pinned as an ORDERED PAIR — a per-lane `toContain` is
 * satisfied by a swap, since both spellings survive it. `[^,]*` between the
 * two absorbs TSL's `as unknown as TSLNode` casts.
 */
const PACK_ORDER = /uvec4\(packHalf2x16\(cutA\.zw\)[^,]*,packHalf2x16\(cutB\.zw\)/;
const UNPACK_A = /pkA[^=]*=\(?unpackHalf2x16\(vPack\.x\)/;
const UNPACK_B = /pkB[^=]*=\(?unpackHalf2x16\(vPack\.y\)/;

/** The deficit-gate literal exactly as the GLSL templates interpolate it. */
const GATE = CAPSULE_JOINT_DEFICIT_GATE.toFixed(2);

/** One shader surface: a vertex stage that packs the packet + a fragment that reads it. */
interface Surface {
  readonly label: string;
  /** Comment-stripped text of the stage that writes `cut.w`. */
  readonly vertex: string;
  /** Comment-stripped text of the stage that rebuilds the partner rod. */
  readonly fragment: string;
}

/**
 * The GLSL surfaces are pinned through their EXPORTED strings — what actually
 * ships once the `${...}` constants are interpolated — not the module text.
 */
const GLSL_SURFACES: readonly Surface[] = [
  {
    label: 'visual GLSL (materials/line/shader-glsl-capsule.ts)',
    vertex: stripComments(CAPSULE_LINE_VERTEX_SHADER),
    fragment: stripComments(CAPSULE_LINE_FRAGMENT_SHADER),
  },
  {
    label: 'pick GLSL (picking/line/shaders-capsule.ts)',
    vertex: stripComments(CAPSULE_LINE_PICK_VERTEX_SHADER),
    fragment: stripComments(CAPSULE_LINE_PICK_FRAGMENT_SHADER),
  },
];

/**
 * The TSL surfaces are read from disk: they build node graphs, so there is no
 * shippable string to import and no way to build one under jsdom (see the
 * header). Vertex and fragment live in one module, so both fields hold the
 * whole file — the `statement()` scoping is what keeps each pin precise.
 */
const TSL_MODULES: ReadonlyArray<readonly [string, string]> = [
  [
    'visual TSL (materials/line/shader-tsl-capsule.ts)',
    'rendering/materials/line/shader-tsl-capsule.ts',
  ],
  ['pick TSL (picking/line/pick-capsule.tsl.ts)', 'rendering/picking/line/pick-capsule.tsl.ts'],
];

const TSL_SURFACES: readonly Surface[] = TSL_MODULES.map(([label, relativePath]) => {
  const source = stripComments(readSource(relativePath));
  return { label, vertex: source, fragment: source };
});

describe('capsule joint deficit packet: the packet contract (#1490 far cap)', () => {
  describe.each(GLSL_SURFACES)('$label', ({ vertex, fragment, label }) => {
    it('bounds the partner radius at the packed length (far cap, not an infinite rod)', () => {
      const rp = statement(fragment, /\bfloat\s+rp\s*=/, label);
      expect(rp, `${label}: rp must clamp xp into [0, cut.w]`).toContain('clamp(xp,0.0,cut.w)');
      // The pre-fix shape, spelled out so a revert reds here by name.
      expect(rp, `${label}: rp must not taper past the partner's far end`).not.toContain(
        'cut.z*max(xp,0.0)'
      );
    });

    it('caps the overshoot at BOTH ends of the partner rod', () => {
      // Two semantic assertions and no exact-equality pin: an equality would
      // subsume both (so neither could ever fire) and would red on a
      // behaviour-identical reparenthesisation such as
      // `max(0.0, max(-xp, xp - cut.w))`. The TSL arm below matches.
      const op = statement(fragment, /\bfloat\s+op\s*=/, label);
      expect(op, `${label}: op needs the FAR cap term xp - cut.w`).toContain('xp-cut.w');
      expect(op, `${label}: op must not carry the near cap alone`).not.toContain('max(-xp,0.0)');
    });

    it('writes the projected partner length into the packet .w lane at both ends', () => {
      // Without this the clamps above bound against 0 and the partner term
      // silently disappears instead of over-drawing — a different bug, equally
      // invisible.
      // Asserted as a boolean rather than `toContain` so a failure reports the
      // missing lane instead of dumping the whole shader as "Received".
      const packed = squash(vertex);
      expect(packed.includes('cutA.w=ql;'), `${label}: end A must pack ql into cut.w`).toBe(true);
      expect(packed.includes('cutB.w=ql;'), `${label}: end B must pack ql into cut.w`).toBe(true);
    });

    it('writes the partner radius GRADIENT into the packet .z lane at both ends', () => {
      const packed = squash(vertex);
      expect(
        packed.includes('cutA.z=(rpFarA-rA)/ql;'),
        `${label}: end A must pack the radius gradient (rpFarA - rA)/ql into cut.z`
      ).toBe(true);
      expect(
        packed.includes('cutB.z=(rpFarB-rB)/ql;'),
        `${label}: end B must pack the radius gradient (rpFarB - rB)/ql into cut.z`
      ).toBe(true);
    });

    it('still CALLS luxarPartnerProfile at both ends, on the matching operands', () => {
      // The GLSL definition shares the `luxarPartnerProfile(` opener with its
      // calls, so a plain count is 3 and would survive deleting one call;
      // `callArguments` filters on the leading `vec4(` that only a call has.
      expectPartnerCalls(fragment, 'luxarPartnerProfile', label);
    });

    it('transports each end packet in its own vPack lane', () => {
      expect(PACK_ORDER.test(squash(vertex)), `${label}: cut A then cut B into vPack.xy`).toBe(
        true
      );
      const packed = squash(fragment);
      expect(UNPACK_A.test(packed), `${label}: pkA must unpack vPack.x`).toBe(true);
      expect(UNPACK_B.test(packed), `${label}: pkB must unpack vPack.y`).toBe(true);
    });
  });

  describe.each(TSL_SURFACES)('$label', ({ vertex, fragment, label }) => {
    it('bounds the partner radius at the packed length (far cap, not an infinite rod)', () => {
      const rp = statement(fragment, /\bconst\s+rp\s*:/, label);
      expect(rp, `${label}: rp must clamp xp into [0, cut.w]`).toContain('clamp(xp,0.0,cut.w)');
      expect(rp, `${label}: rp must not taper past the partner's far end`).not.toContain(
        'cut.z.mul(max(xp,0.0))'
      );
    });

    it('caps the overshoot at BOTH ends of the partner rod', () => {
      const op = statement(fragment, /\bconst\s+op\s*:/, label);
      expect(op, `${label}: op needs the FAR cap term xp.sub(cut.w)`).toContain('xp.sub(cut.w)');
      expect(op, `${label}: op must not carry the near cap alone`).not.toContain(
        'max(xp.negate(),0.0)'
      );
    });

    it('writes the projected partner length into the packet .w lane at both ends', () => {
      // Boolean form, as above: the TSL "vertex" text is the whole module.
      const packed = squash(vertex);
      expect(packed.includes('cutA.w.assign(ql)'), `${label}: end A must pack ql into cut.w`).toBe(
        true
      );
      expect(packed.includes('cutB.w.assign(ql)'), `${label}: end B must pack ql into cut.w`).toBe(
        true
      );
    });

    it('writes the partner radius GRADIENT into the packet .z lane at both ends', () => {
      // Dropping the assign leaves the lane at its `vec4(nLoc, 0.0, 0.0)`
      // initialiser, i.e. a CONSTANT-radius phantom rod — a taper that
      // silently disappears on one backend.
      const packed = squash(vertex);
      expect(
        packed.includes('cutA.z.assign(rpFarA.sub(rA).div(ql))'),
        `${label}: end A must pack the radius gradient (rpFarA - rA)/ql into cut.z`
      ).toBe(true);
      expect(
        packed.includes('cutB.z.assign(rpFarB.sub(rB).div(ql))'),
        `${label}: end B must pack the radius gradient (rpFarB - rB)/ql into cut.z`
      ).toBe(true);
    });

    it('still CALLS partnerProfile at both ends, on the matching operands', () => {
      // Every other assertion in this file inspects the partnerProfile
      // DEFINITION, so deleting a call site keeps them all green while the
      // deficit rule stops running. (The arrow-function definition here reads
      // `partnerProfile = (cut: ...`, so it is not a `partnerProfile(` match
      // at all; the GLSL arm needs the `vec4(` filter to say the same thing.)
      expectPartnerCalls(fragment, 'partnerProfile', label);
    });

    it('transports each end packet in its own vPack lane', () => {
      const whole = squash(vertex);
      expect(PACK_ORDER.test(whole), `${label}: cut A then cut B into vPack.xy`).toBe(true);
      expect(UNPACK_A.test(whole), `${label}: pkA must unpack vPack.x`).toBe(true);
      expect(UNPACK_B.test(whole), `${label}: pkB must unpack vPack.y`).toBe(true);
    });
  });

  it('the CPU mirror carries the same bounded rod', () => {
    // Brief on purpose: `line-capsule.test.ts`'s numeric sweep already fails
    // by VALUE on a mirror-only regression (measured: two failing tests — see
    // the header). This is here so that a change made to the four shaders and
    // the model together cannot leave the two silently describing different
    // geometry.
    const mirror = squash(stripComments(readSource('rendering/materials/_shared/line-capsule.ts')));
    expect(mirror, 'the mirror must clamp xp into [0, partner.length]').toContain(
      'Math.min(Math.max(xp,0),partner.length)'
    );
    expect(mirror, 'the mirror needs the FAR cap term xp - partner.length').toContain(
      'Math.max(Math.max(-xp,xp-partner.length),0)'
    );
  });

  it('locks EVERY capsule surface that rebuilds a partner profile, not just the four listed', () => {
    // `GLSL_SURFACES` / `TSL_SURFACES` are hardcoded, so a fifth surface (a
    // future depth-prepass or outline capsule, say) would inherit the whole
    // packet hazard and escape this file silently. Ground truth is every
    // module under `src/rendering` whose comment-stripped text mentions a
    // partner profile under either spelling (`luxarPartnerProfile` in GLSL,
    // `partnerProfile` in TSL) — matched case-insensitively so neither is
    // privileged, and on stripped text so a doc comment elsewhere does not
    // read as a new surface. Modelled on `join-width-tsl.test.ts`'s
    // 'locks EVERY TSL factory that calls tslLineJoin' case.
    const renderingRoot = path.join(SRC_ROOT, 'rendering');
    const found = readdirSync(renderingRoot, { recursive: true })
      .map((entry) => String(entry).split(path.sep).join('/'))
      .filter((entry) => entry.endsWith('.ts'))
      .filter((entry) => /partnerprofile/i.test(stripComments(readSource(`rendering/${entry}`))))
      .map((entry) => `rendering/${entry}`)
      .sort();

    const locked = [
      'rendering/materials/line/shader-glsl-capsule.ts',
      'rendering/picking/line/shaders-capsule.ts',
      ...TSL_MODULES.map(([, relativePath]) => relativePath),
    ].sort();
    expect(found).toEqual(locked);
  });
});

/**
 * The packet GATE — a sibling describe, not a child of the one above, because
 * these clauses are #1495 / #1501 and are green on a #1490 revert. They live
 * here because they belong to the same packet contract. Division of labour with
 * #1548's chain pin in `line-capsule.test.ts`: that one holds the clauses in
 * ORDER, joined by `||` / `.or(...)`, as a single exact string — so it, not
 * this, is what an operator swap reds (measured: `||` → `&&` on the GLSL pair
 * plus `.or(` → `.and(` on the TSL pair reds it and nothing else). These are
 * the per-clause form, scoped to the gate statement and tolerant of the operand
 * hoist below, and they cover the surfaces one at a time so a failure names the
 * clause rather than printing a whole shader.
 */
describe('capsule joint deficit packet: the gate clauses (#1495 / #1501)', () => {
  describe.each(GLSL_SURFACES)('$label', ({ vertex, label }) => {
    it.each([
      {
        end: 'A',
        marker: /\bbool\s+needPacketA\s*=/,
        clauses: [`abs(1.0-rpFarA/max(rA,1e-4))>${GATE}`, `rB>rA*(1.0+${GATE})`, 'ql<2.0*rA'],
        turn: '>0.5',
      },
      {
        end: 'B',
        marker: /\bbool\s+needPacketB\s*=/,
        clauses: [`abs(1.0-rpFarB/max(rB,1e-4))>${GATE}`, `rA>rB*(1.0+${GATE})`, 'ql<2.0*rB'],
        turn: '<-0.5',
      },
    ])('keeps all four packet-gate clauses at end $end', ({ marker, clauses, turn }) => {
      const gate = statement(vertex, marker, label);
      for (const clause of clauses) {
        expect(gate, `${label}: missing packet-gate clause ${clause}`).toContain(clause);
      }
      // The sharp-turn clause is pinned OPERAND-AGNOSTICALLY: "a dot product,
      // compared against ±0.5". #1562 (issue #1495) has since put the same
      // expression in the width gate just above, and the natural cleanup now
      // that it appears twice is to hoist it to a local — exactly what the TSL
      // twin already does (`dot(qhat, u)`). Two `toContain`s rather than one
      // regex: a regex tight enough to bind the operands would red on that
      // hoist, and one loose enough to tolerate it (`dot\([^)]*\)`) cannot
      // cross a nested `)`, so it would ALSO red on `dot(normalize(qq), u)`.
      // Each gate STATEMENT — the scope `statement()` cuts, which stops short
      // of the width gate — holds exactly one `dot(` and one `±0.5`, and
      // deleting the clause removes both.
      expect(gate, `${label}: the sharp-turn clause needs a dot product`).toContain('dot(');
      expect(gate, `${label}: the sharp-turn clause must compare against ${turn}`).toContain(turn);
    });
  });

  describe.each(TSL_SURFACES)('$label', ({ vertex, label }) => {
    it.each([
      {
        end: 'A',
        marker: /\bconst\s+needPacketA\s*:/,
        clauses: [
          'abs(float(1.0).sub(rpFarA.div(max(rA,float(1e-4))))).greaterThan(CAPSULE_JOINT_DEFICIT_GATE)',
          'rB.greaterThan(rA.mul(float(1.0).add(CAPSULE_JOINT_DEFICIT_GATE)))',
          'ql.lessThan(rA.mul(2.0))',
          // Already at the hoisted altitude the GLSL arm is relaxed toward, so
          // this one stays literal.
          'dot(qhat,u).greaterThan(0.5)',
        ],
      },
      {
        end: 'B',
        marker: /\bconst\s+needPacketB\s*:/,
        clauses: [
          'abs(float(1.0).sub(rpFarB.div(max(rB,float(1e-4))))).greaterThan(CAPSULE_JOINT_DEFICIT_GATE)',
          'rA.greaterThan(rB.mul(float(1.0).add(CAPSULE_JOINT_DEFICIT_GATE)))',
          'ql.lessThan(rB.mul(2.0))',
          'dot(qhat,u).lessThan(-0.5)',
        ],
      },
    ])('keeps all four packet-gate clauses at end $end', ({ marker, clauses }) => {
      const gate = statement(vertex, marker, label);
      for (const clause of clauses) {
        expect(gate, `${label}: missing packet-gate clause ${clause}`).toContain(clause);
      }
    });
  });
});
