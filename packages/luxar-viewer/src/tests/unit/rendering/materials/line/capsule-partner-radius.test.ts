/**
 * Capsule joint partner-radius source lock (#1494).
 *
 * The capsule joint DEFICIT rule rebuilds the PARTNER leg's field per fragment
 * (each leg renders max(mine − partner, 0) past the bisector). The base radius
 * of that reconstruction must be the SHARED VERTEX radius — the radius both legs
 * agree on at the joint — not `rPx`, the fragment's OWN per-pixel interpolated
 * radius. Re-introducing that defect in today's CPU model, over the four #1494
 * review rows of `tests/unit/rendering/materials/line-capsule.test.ts` (θ = 60°,
 * vertex radius 10, partner tapering 10 → 5 over 30 px), measures +0.077 of peak
 * over-draw where my own leg tapers and −0.192 where it widens. The larger
 * −0.478 / +0.263 pair recorded in `line-capsule.test.ts` (the 'tapered own leg
 * (the #1494 rows)' test) is the worst seen
 * across those same four rows on a code state that also lacked #1488's reach and
 * #1490's far cap.
 *
 * The two segment-constant endpoint radii already ride the flat varying lane
 * `vPack.z` (`packHalf2x16(vec2(rA, rB))`, unpacked in the fragment as `pkR`) —
 * that is where the fragment's own `rPx` comes from. #1494 added no varying: it
 * routed `pkR.x` (end A) and `pkR.y` (end B) into the reconstruction, which takes
 * that value as its `rEnd` parameter and uses it as the base radius.
 *
 * Why a source-text tripwire rather than a behavioural test:
 *
 * - The numeric sweep in `line-capsule.test.ts` binds the CPU reference model
 *   (`capsuleJointRenderLeg`) only. Nothing bound the four SHADER surfaces, so a
 *   GLSL-only or TSL-only revert — passing `rPx` back into the partner
 *   reconstruction, or packing something other than the endpoint radii into the
 *   lane the fragment reads — stayed fully green.
 * - The TSL builders' only other direct coverage is the checked-in codegen
 *   snapshots (`tests/e2e/tsl-codegen-snapshot.spec.ts`), which do not run in
 *   CI; the hosted `e2e-tests` job runs only the mobile suite.
 * - The TSL fragment cannot be exercised here at all: its body is traced inside
 *   `Fn(() => {...})`, which only runs during a real node build, and that needs a
 *   WebGPU/WebGL backend unavailable under jsdom.
 *
 * The guards, overlapping so that no single DIRECT edit slips through:
 *
 * (a) the vertex stage packs the two ENDPOINT radii, and packs them into the lane
 *     the FRAGMENT actually unpacks as `pkR` — read out of the fragment rather
 *     than hardcoded, so a consistent lane migration stays green while a
 *     one-sided one (either stage moved alone) reds;
 * (b) each fragment call site passes the endpoint radius for ITS end (the WHOLE
 *     call is pinned, so an argument swap is visible);
 * (c) the reconstruction's base radius is that parameter, tapered by the packed
 *     gradient (the trailing `1e-4` is deliberately NOT pinned — it is this shader
 *     family's generic divide-by-zero floor, recurring several times in each
 *     surface's fragment stage, where `luxarPartnerProfile` lives, and in the
 *     vertex stage too; it is not part of the #1494 contract);
 * (d) `pkR` and `rp` are each written EXACTLY once, and `rEnd` NEVER — and
 *     neither `pkR` nor the varying `vPack` is written PER COMPONENT. Every pin
 *     above is satisfied by a later write, because the last one wins:
 *     `pkR = vec2(pkR.y, pkR.x);` after the unpack, a second `rp = ...`, or — the
 *     subtlest — a write to `rEnd` itself at the top of the reconstruction, which
 *     reinstates the exact #1494 defect while the pinned `rp` line and every call
 *     site stay byte-identical. A GLSL function parameter is a writable local and
 *     the TSL `rEnd` is a plain JS binding, so both backends allow it. A
 *     component write is the same edit spelled through a swizzle and needs its own
 *     clause in every pattern: `pkR.x = rPx; pkR.y = rPx;` in the fragment (or
 *     `pkR.x.assign(rPx)` in TSL) hands both ends the per-fragment radius, and
 *     `vPack.z = packHalf2x16(vec2(rA, rA));` after the pinned pack re-parks the
 *     lane — all three measured green before those clauses existed;
 * (e) TSL only: the `partnerProfile` closure body mentions no `rPx`. That closure
 *     is defined in the same scope as the fragment's per-pixel radius, so a
 *     regression could read `rPx` inside it and ignore its `rEnd` parameter while
 *     every call site still looked correct. It has no GLSL twin because `rPx` is a
 *     `main()` local, out of scope in `luxarPartnerProfile`. It is also not the
 *     guard that stops a re-derivation from the varyings (those ARE global inside
 *     the GLSL function, and a TSL re-derivation from `pkR` never spells `rPx`) —
 *     (d)'s no-write-to-`rEnd` rule is;
 * (f) the surface list is complete: every file under `src/rendering` carrying a
 *     partner reconstruction must be one of the four locked here.
 *
 * Scope of the write counts in (d): for the TSL surfaces `rp` and `rEnd` are
 * counted inside the `partnerProfile` closure (the module also holds the vertex
 * stage, where an unrelated local named `rp` is legitimate), and `pkR` over the
 * fragment (it is unpacked outside the closure). For GLSL the counts are
 * STAGE-WIDE, which is exact only because `rp` and `rEnd` exist nowhere else in
 * the fragment source; a legitimate second write — an `#ifdef` variant of the `rp`
 * line, say — means scoping this guard to the function, not deleting it. The
 * `vPack` clause is counted over the VERTEX stage, and forbids only the per-lane
 * form: writing the varying whole is the legitimate spelling, and how many whole
 * writes there are (an early-out pack, the real one) is not this lock's business —
 * (a) already checks that every one of them agrees on the lane.
 *
 * What a text pin does NOT reach: these guards catch the direct edits — an
 * argument swap, a lane move, a re-bind (plain, compound, per-component, or via a
 * TSL assign method), an `rPx` read inside the closure. A determined rewrite of the
 * surrounding arithmetic stays green — four such shapes were checked and do:
 * redefining `rA` / `rB` just above the pinned pack, dividing by a re-derived
 * radius in the UNPINNED `qp` line instead of the pinned `rp` line, renaming the
 * pinned function to `…Core` behind a thin forwarding wrapper that re-derives the
 * radius, and a `#define rEnd …` inside the function body. The numeric sweep in
 * `line-capsule.test.ts` is what covers the model side of that gap.
 *
 * Every match runs on comment-stripped text — so prose can neither satisfy a pin
 * nor move a slice boundary — with whitespace squashed out for the expression
 * pins (reformatting and re-wrapping survive; an argument swap, an operator change
 * or a deletion does not) and whitespace-tolerant patterns for the write counts,
 * which need word boundaries a squash would destroy.
 *
 * These are source pins: renaming `rEnd`, `pkR` or `vPack`, or changing the
 * reconstruction's signature, requires updating them. A red here can be a rename
 * rather than a regression — check the diff before assuming the latter.
 */

import { readdirSync, readFileSync } from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

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

/**
 * The TSL modules build a node graph at runtime and export no shader text, so
 * their sources are read from disk. The GLSL sources are imported instead — the
 * exported strings are what the material actually compiles.
 */
function readSource(relativeToSrc: string): string {
  return readFileSync(path.join(SRC_ROOT, relativeToSrc), 'utf8');
}

/** Drop `//` and block comments so prose can never satisfy — or break — a grep. */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

/** Comment-stripped and whitespace-free: survives reformatting, not an edit. */
function squash(source: string): string {
  return stripComments(source).replace(/\s+/g, '');
}

/**
 * `includes` behind a boolean expectation, not `toContain`.
 *
 * These haystacks are whole squashed modules (5–20 kB); `toContain` dumps the
 * entire haystack as `Received` on failure and buries the message that names the
 * surface and the expected text.
 */
function expectContains(haystack: string, needle: string, message: string): void {
  expect(haystack.includes(needle), message).toBe(true);
}

/** Shortest source of interest here is ~4 kB; anything near this is a bad read. */
const MIN_PLAUSIBLE_SOURCE_CHARS = 2000;

/** A write form, and how many times it may appear in its scope. */
interface WriteRule {
  /** What is being written, for the failure message. */
  readonly what: string;
  /**
   * Matched against COMMENT-STRIPPED text (whitespace intact, tolerated with
   * `\s*`): squashing would glue the declaration's type onto the name
   * (`float rp` → `floatrp`) and cost the `\b` that keeps `vSharp = …` from
   * reading as a write to `rp`.
   *
   * The assignment patterns cover plain `=` and the arithmetic compound forms
   * `+= -= *= /= %=` (never `==`), plus — for TSL nodes, where assignment is a
   * method call — `.assign()` and `.addAssign()` / `.subAssign()` /
   * `.mulAssign()` / `.divAssign()`. Vector targets also admit a SWIZZLE
   * (`pkR.x = …`, `pkR.x.assign(…)`), which is the same write with one more token,
   * so `SWIZZLE` rides along in front of every `=` / assign method.
   */
  readonly pattern: RegExp;
  /** Permitted match count — 1 for a single declaration, 0 for a forbidden form. */
  readonly times: number;
}

interface CapsuleSurface {
  readonly label: string;
  /** Raw text of the stage that packs the endpoint radii. */
  readonly vertex: string;
  /** Raw text of the stage that rebuilds the partner field. */
  readonly fragment: string;
  /** `packHalf2x16` of the two endpoint radii, as written. */
  readonly pack: string;
  /** Reads the lane letter of the fragment's `pkR` unpack (capture group 1). */
  readonly laneRead: RegExp;
  /** The whole end-A partner call, including its radius argument. */
  readonly callEndA: string;
  /** The whole end-B partner call, including its radius argument. */
  readonly callEndB: string;
  /** The reconstruction's base-radius expression, up to the taper term. */
  readonly baseRadius: string;
  /**
   * Write rules counted over `fragment` as given: the fragment stage for GLSL, the
   * whole module for TSL (where both stages share one file). Sound either way —
   * the names counted here (`pkR`) occur only in the fragment.
   */
  readonly fragmentWrites: readonly WriteRule[];
  /**
   * Write rules counted over the stage that packs the lane. Empty for TSL, whose
   * single module is already covered by `fragmentWrites`.
   */
  readonly vertexWrites: readonly WriteRule[];
  /**
   * Write rules counted inside the partner reconstruction only. Empty for GLSL,
   * whose `rp` / `rEnd` exist nowhere else in the stage (see the header note);
   * used for TSL, whose module also carries the vertex stage.
   */
  readonly reconstructionWrites: readonly WriteRule[];
  /** TSL surfaces only: the closure whose body must not read `rPx`. */
  readonly closure?: string;
}

const GLSL_PACK = 'packHalf2x16(vec2(rA, rB))';
const GLSL_CALL_A = 'luxarPartnerProfile(vec4(vCutN.xy, pkA), vec2(x, y), 1.0, pkR.x, vSharp)';
const GLSL_CALL_B =
  'luxarPartnerProfile(vec4(vCutN.zw, pkB), vec2(x - vAbLen, y), -1.0, pkR.y, vSharp)';
/**
 * Through the taper term only: the trailing `, 1e-4)` is the generic
 * divide-by-zero floor this shader family uses throughout both stages, not
 * #1494's contract.
 */
const GLSL_BASE_RADIUS = 'float rp = max(rEnd + cut.z * clamp(xp, 0.0, cut.w)';
/** `vec2 pkR = unpackHalf2x16(vPack.z)` squashed — the lane letter is captured. */
const GLSL_LANE_READ = /pkR=unpackHalf2x16\(vPack\.([xyzw])\)/g;
/**
 * An optional component selector, in any of GLSL's three interchangeable spellings
 * (`.xy` / `.rg` / `.st`), tolerating a break before the dot so prettier splitting a
 * TSL member chain does not read as a missing write. Only vector targets need it —
 * `rp` and `rEnd` are scalars, which GLSL cannot swizzle at all.
 */
const SWIZZLE = String.raw`(?:\s*\.[xyzwrgbastpq]+)?`;

const GLSL_FRAGMENT_WRITES: readonly WriteRule[] = [
  // `pkR.x = rPx;` is the #1494 defect with every pin byte-identical, so the
  // swizzle counts as a write to the pair.
  {
    what: 'pkR (the unpacked endpoint-radius pair, component writes included)',
    pattern: new RegExp(String.raw`\bpkR${SWIZZLE}\s*[-+*/%]?=(?!=)`, 'g'),
    times: 1,
  },
  { what: 'rp (the rebuilt partner radius)', pattern: /\brp\s*[-+*/%]?=(?!=)/g, times: 1 },
  // A GLSL parameter is a writable local, and the varyings are global inside the
  // function: `rEnd = max(mix(pkR.x, pkR.y, …), 1e-4);` on the first line of the
  // reconstruction is the #1494 defect exactly, with every other pin intact.
  // COMPOUND forms count as writes: `rEnd += rSelf - rEnd;` and `rp *= rSelf / rp;`
  // are the same defect spelled to dodge a bare-`=` pattern.
  {
    what: 'rEnd (the shared-vertex radius parameter)',
    pattern: /\brEnd\s*[-+*/%]?=(?!=)/g,
    times: 0,
  },
];

/**
 * The pinned pack writes the varying WHOLE; a per-lane write afterwards
 * (`vPack.z = packHalf2x16(vec2(rA, rA));`) re-parks the radius lane with the
 * pinned `uvec4(...)` untouched, and no whole-varying count can see it.
 */
const GLSL_VERTEX_WRITES: readonly WriteRule[] = [
  {
    what: 'vPack per lane (the pack must write the varying whole)',
    pattern: new RegExp(String.raw`\bvPack\s*\.[xyzwrgbastpq]+\s*[-+*/%]?=(?!=)`, 'g'),
    times: 0,
  },
];

const TSL_CALL_A = 'partnerProfile(vec4(vCutN.xy, pkA), vec2(x, y), 1.0, pkR.x)';
const TSL_CALL_B = 'partnerProfile(vec4(vCutN.zw, pkB), vec2(x.sub(vAbLen), y), -1.0, pkR.y)';
const TSL_BASE_RADIUS = 'const rp: TSLNode = max(rEnd.add(cut.z.mul(clamp(xp, 0.0, cut.w)))';
/**
 * The `as unknown as TSLNode` cast's `(` is OPTIONAL: dropping those casts (nine
 * per module) is a plausible typings cleanup and must not read as a missing lane.
 */
const TSL_LANE_READ = /pkR:TSLNode=\(?unpackHalf2x16\(vPack\.([xyzw])\)/g;
const TSL_FRAGMENT_WRITES: readonly WriteRule[] = [
  { what: 'pkR (the unpacked endpoint-radius pair)', pattern: /\bconst\s+pkR\b/g, times: 1 },
  // A TSL node is re-bound by `.assign()` — or by a compound-assign method, which
  // is an in-repo idiom (`materials/gsplat/shader-tsl.ts` uses `addAssign`) — and
  // no `const` count can see either. `pkR` is a `.toVar()`, so a LANE of it takes
  // `.assign()` too: `pkR.x.assign(rPx)` is the GLSL swizzle write in TSL spelling.
  {
    what: 'pkR via .assign() (per-lane included)',
    pattern: new RegExp(String.raw`\bpkR${SWIZZLE}\s*\.assign\(`, 'g'),
    times: 0,
  },
  {
    what: 'pkR via a compound-assign method (per-lane included)',
    pattern: new RegExp(String.raw`\bpkR${SWIZZLE}\s*\.(?:add|sub|mul|div)Assign\(`, 'g'),
    times: 0,
  },
  // The vertex stage shares this module, so the whole-varying rule lands here.
  {
    what: 'vPack per lane (the pack must write the varying whole)',
    pattern: /\bvPack\s*\.[xyzw]+\s*\.(?:assign|(?:add|sub|mul|div)Assign)\(/g,
    times: 0,
  },
];
/** Counted inside the closure: the module's vertex stage may have its own `rp`. */
const TSL_RECONSTRUCTION_WRITES: readonly WriteRule[] = [
  { what: 'rp (the rebuilt partner radius)', pattern: /\bconst\s+rp\b/g, times: 1 },
  { what: 'rp via .assign()', pattern: /\brp\s*\.assign\(/g, times: 0 },
  // `rEnd` is a plain JS binding, so `rEnd = max(mix(pkR.x, pkR.y, …))` inside the
  // closure reinstates #1494 without ever spelling `rPx` — invisible to (e).
  {
    what: 'rEnd (the shared-vertex radius parameter)',
    pattern: /\brEnd\s*[-+*/%]?=(?!=)/g,
    times: 0,
  },
  { what: 'rEnd via .assign()', pattern: /\brEnd\s*\.assign\(/g, times: 0 },
  // `rp.mulAssign(<re-derived radius>.div(rp))` is the defect with the pinned `rp`
  // line untouched and no `rPx` token anywhere; the plain `.assign(` rule above
  // cannot see it.
  {
    what: 'rp / rEnd via a compound-assign method',
    pattern: /\b(?:rp|rEnd)\s*\.(?:add|sub|mul|div)Assign\(/g,
    times: 0,
  },
];

const TSL_VISUAL_PATH = 'rendering/materials/line/shader-tsl-capsule.ts';
const TSL_PICK_PATH = 'rendering/picking/line/pick-capsule.tsl.ts';
const TSL_VISUAL = readSource(TSL_VISUAL_PATH);
const TSL_PICK = readSource(TSL_PICK_PATH);

const SURFACES: readonly CapsuleSurface[] = [
  {
    label: 'GLSL visual (shader-glsl-capsule.ts)',
    vertex: CAPSULE_LINE_VERTEX_SHADER,
    fragment: CAPSULE_LINE_FRAGMENT_SHADER,
    pack: GLSL_PACK,
    laneRead: GLSL_LANE_READ,
    callEndA: GLSL_CALL_A,
    callEndB: GLSL_CALL_B,
    baseRadius: GLSL_BASE_RADIUS,
    fragmentWrites: GLSL_FRAGMENT_WRITES,
    vertexWrites: GLSL_VERTEX_WRITES,
    reconstructionWrites: [],
  },
  {
    label: 'GLSL pick (shaders-capsule.ts)',
    vertex: CAPSULE_LINE_PICK_VERTEX_SHADER,
    fragment: CAPSULE_LINE_PICK_FRAGMENT_SHADER,
    pack: GLSL_PACK,
    laneRead: GLSL_LANE_READ,
    callEndA: GLSL_CALL_A,
    callEndB: GLSL_CALL_B,
    baseRadius: GLSL_BASE_RADIUS,
    fragmentWrites: GLSL_FRAGMENT_WRITES,
    vertexWrites: GLSL_VERTEX_WRITES,
    reconstructionWrites: [],
  },
  {
    label: 'TSL visual (shader-tsl-capsule.ts)',
    // One module holds both stages. The substring pins are unique enough to stay
    // stage-specific without slicing it; the WRITE COUNTS are not, so `rp` and
    // `rEnd` are counted inside the closure (`reconstructionWrites`).
    vertex: TSL_VISUAL,
    fragment: TSL_VISUAL,
    pack: GLSL_PACK,
    laneRead: TSL_LANE_READ,
    callEndA: TSL_CALL_A,
    callEndB: TSL_CALL_B,
    baseRadius: TSL_BASE_RADIUS,
    fragmentWrites: TSL_FRAGMENT_WRITES,
    vertexWrites: [],
    reconstructionWrites: TSL_RECONSTRUCTION_WRITES,
    closure: TSL_VISUAL,
  },
  {
    label: 'TSL pick (pick-capsule.tsl.ts)',
    vertex: TSL_PICK,
    fragment: TSL_PICK,
    pack: GLSL_PACK,
    laneRead: TSL_LANE_READ,
    callEndA: TSL_CALL_A,
    callEndB: TSL_CALL_B,
    baseRadius: TSL_BASE_RADIUS,
    fragmentWrites: TSL_FRAGMENT_WRITES,
    vertexWrites: [],
    reconstructionWrites: TSL_RECONSTRUCTION_WRITES,
    closure: TSL_PICK,
  },
];

/** `vPack.<letter>` → the `uvec4(...)` argument index that feeds it. */
const LANE_INDEX: Record<string, number> = { x: 0, y: 1, z: 2, w: 3 };

/**
 * Split the `callee(...)` argument list starting at `open` into its TOP-LEVEL
 * arguments.
 *
 * Depth-tracked rather than a `split(',')`: every argument here nests further
 * calls whose own commas would otherwise split it.
 */
function argumentsAt(squashedSource: string, open: number, callee: string): string[] {
  let cursor = open + callee.length + 1;
  let depth = 1;
  let current = '';
  const args: string[] = [];
  while (cursor < squashedSource.length && depth > 0) {
    const character = squashedSource[cursor];
    if (character === '(') depth += 1;
    else if (character === ')') depth -= 1;
    if (depth === 0) break;
    if (character === ',' && depth === 1) {
      args.push(current);
      current = '';
    } else {
      current += character;
    }
    cursor += 1;
  }
  expect(depth, `unbalanced parentheses in a ${callee}(...) call`).toBe(0);
  args.push(current);
  return args;
}

/**
 * The lane lists of EVERY `callee(...)` whose arguments contain `needle`.
 *
 * Every surface writes more than one `uvec4(...)` (an early-out pack, or the
 * varying's zero initializer), so the calls are identified by CONTENT rather than
 * by position — a reordering of the surrounding code cannot pick the wrong one.
 *
 * ALL of them, not the single one: the early-out branch currently packs a
 * placeholder into the radius lane, and making it pack the real endpoint radii is
 * strictly more correct. Every pack site must agree on the lane; how many sites
 * there are is not this lock's business.
 */
function argumentsContaining(squashedSource: string, callee: string, needle: string): string[][] {
  const matches: string[][] = [];
  for (let at = 0; ;) {
    const open = squashedSource.indexOf(`${callee}(`, at);
    if (open < 0) break;
    const args = argumentsAt(squashedSource, open, callee);
    if (args.some((argument) => argument.includes(needle))) matches.push(args);
    at = open + callee.length + 1;
  }
  expect(
    matches.length,
    `expected at least one ${callee}(...) whose arguments contain '${needle}'`
  ).toBeGreaterThan(0);
  return matches;
}

/**
 * The body of the `partnerProfile` arrow function, comment-stripped.
 *
 * Located by the same regex that COUNTS the declaration: an `indexOf` of the
 * literal text fails open on `const  partnerProfile` (two spaces) — `start`
 * becomes −1 and the brace scan below silently slices the vertex `Fn` body
 * instead, passing every assertion while the defect is live.
 *
 * Brace-matched from the arrow, so the slice ends where the closure does and
 * cannot swallow the call sites below it (which legitimately mention `pkR`, and
 * — for a mutation — could mention `rPx`).
 */
function partnerProfileBody(label: string, rawSource: string): string {
  const source = stripComments(rawSource);
  const declarations = source.match(/\bconst\s+partnerProfile\b/g) ?? [];
  expect(
    declarations,
    `${label}: partnerProfile must be declared exactly once — a second ` +
      'declaration could shadow the one whose body is scanned here'
  ).toHaveLength(1);

  const declaration = /\bconst\s+partnerProfile\b/.exec(source);
  expect(declaration, `${label}: no 'const partnerProfile' declaration`).not.toBeNull();
  const start = declaration!.index;
  const arrow = source.indexOf('=>', start);
  expect(arrow, `${label}: no arrow after 'const partnerProfile'`).toBeGreaterThan(start);
  const bodyStart = source.indexOf('{', arrow);
  expect(bodyStart, `${label}: no closure body after the arrow`).toBeGreaterThan(arrow);

  let cursor = bodyStart + 1;
  let depth = 1;
  while (cursor < source.length && depth > 0) {
    const character = source[cursor];
    if (character === '{') depth += 1;
    else if (character === '}') depth -= 1;
    cursor += 1;
  }
  expect(depth, `${label}: unbalanced braces in the partnerProfile closure`).toBe(0);
  return source.slice(bodyStart + 1, cursor - 1);
}

describe('capsule joint partner radius is the SHARED VERTEX radius (#1494)', () => {
  for (const surface of SURFACES) {
    describe(surface.label, () => {
      it('reads a plausible source (guards every pin below against a vacuous pass)', () => {
        for (const [stage, text] of [
          ['vertex', surface.vertex],
          ['fragment', surface.fragment],
        ] as const) {
          expect(
            text.length,
            `${surface.label}: ${stage} source is implausibly short (${text.length} chars) — ` +
              'the substring pins would pass vacuously'
          ).toBeGreaterThan(MIN_PLAUSIBLE_SOURCE_CHARS);
        }
      });

      it('packs the two ENDPOINT radii into the partner-radius lane', () => {
        // (a) A vertex whose pack EXPRESSION is a midpoint, a partner-derived
        // radius or one endpoint twice defeats the fix even though every fragment
        // call site still reads `pkR.x` / `pkR.y`. Redefining `rA` / `rB` upstream
        // of this line is out of reach (see the header's boundary note).
        expectContains(
          squash(surface.vertex),
          squash(surface.pack),
          `${surface.label}: the vertex stage must pack the segment's two endpoint ` +
            `radii as '${surface.pack}'`
        );
      });

      it('packs them into the lane the FRAGMENT unpacks as pkR', () => {
        // Same contract from the other side, and the half that catches a
        // one-sided lane change: the pin above is satisfied by a vertex that
        // parks (rA, rB) in a lane the fragment never reads, and the call-site
        // pins are satisfied by a fragment that unpacks the wrong lane.
        // The expected index is READ OUT of the fragment rather than hardcoded,
        // so migrating the pair to another lane on BOTH sides stays green.
        const lanes = squash(surface.fragment).matchAll(new RegExp(surface.laneRead));
        const letters = [...lanes].map((match) => match[1]);
        expect(
          letters,
          `${surface.label}: the fragment must unpack pkR from exactly one vPack ` +
            `lane, matching ${String(surface.laneRead)}`
        ).toHaveLength(1);
        const index = LANE_INDEX[letters[0]];
        const packed = squash(surface.pack);
        // EVERY pack site must agree with the fragment's lane, not just one.
        for (const packLanes of argumentsContaining(squash(surface.vertex), 'uvec4', packed)) {
          expect(packLanes, `${surface.label}: vPack is a uvec4 of four lanes`).toHaveLength(4);
          expectContains(
            packLanes[index],
            packed,
            `${surface.label}: the fragment reads pkR from vPack.${letters[0]} (lane ` +
              `${index}), so every vertex-stage pack of the endpoint radii must put ` +
              'them in THAT lane'
          );
        }
      });

      it('passes each END its own endpoint radius', () => {
        // (b) The WHOLE call is pinned, not just the `pkR.x` substring: that is
        // what makes an argument swap — or `rPx` in place of the endpoint
        // radius, the #1494 defect itself — visible.
        const fragment = squash(surface.fragment);
        expectContains(
          fragment,
          squash(surface.callEndA),
          `${surface.label}: end-A partner call must be '${surface.callEndA}'`
        );
        expectContains(
          fragment,
          squash(surface.callEndB),
          `${surface.label}: end-B partner call must be '${surface.callEndB}'`
        );
      });

      it('bases the rebuilt partner radius on that parameter', () => {
        // (c) The call sites are worth nothing if the reconstruction ignores
        // the argument it was handed. Pinned through the taper term and no
        // further: the trailing epsilon floor is tuning, not this contract.
        expectContains(
          squash(surface.fragment),
          squash(surface.baseRadius),
          `${surface.label}: the partner radius must start at rEnd and taper by the ` +
            `packed gradient — '${surface.baseRadius}'`
        );
      });

      it('writes pkR and rp once and rEnd never, so nothing can re-bind them', () => {
        // (d) The last write wins, so every pin above survives a later write:
        // `pkR = vec2(pkR.y, pkR.x);` after the unpack silently feeds end A the
        // FAR endpoint's radius; a second `rp = ...` replaces the pinned one
        // wholesale; and a write to `rEnd` on the reconstruction's first line
        // reinstates #1494 with every other pin byte-identical.
        // Comment-stripped, whitespace INTACT (see `WriteRule.pattern`).
        const scopes: Array<[string, string, readonly WriteRule[]]> = [
          ['fragment', stripComments(surface.fragment), surface.fragmentWrites],
        ];
        if (surface.vertexWrites.length > 0) {
          scopes.push(['vertex', stripComments(surface.vertex), surface.vertexWrites]);
        }
        if (surface.reconstructionWrites.length > 0) {
          const closureSource = surface.closure;
          expect(
            closureSource,
            `${surface.label}: reconstruction-scoped rules need a closure source`
          ).toBeDefined();
          scopes.push([
            'partnerProfile closure',
            partnerProfileBody(surface.label, closureSource as string),
            surface.reconstructionWrites,
          ]);
        }
        for (const [scope, text, rules] of scopes) {
          for (const { what, pattern, times } of rules) {
            expect(
              text.match(new RegExp(pattern)) ?? [],
              `${surface.label} (${scope}): ${what} must be written ` +
                `${times === 0 ? 'never' : `exactly ${times}×`} — a later write ` +
                'replaces the pinned one and the last write wins'
            ).toHaveLength(times);
          }
        }
      });

      const closure = surface.closure;
      if (closure) {
        it('never reads the per-fragment radius inside the partner closure', () => {
          // (e) TSL only: the closure sits in the fragment scope, so it can read
          // `rPx` directly and ignore `rEnd` with both call sites intact.
          const body = partnerProfileBody(surface.label, closure);
          expect(
            body.length,
            `${surface.label}: the partnerProfile body slice is implausibly short ` +
              `(${body.length} chars) — a bad slice must not pass vacuously`
          ).toBeGreaterThan(400);
          expect(
            body,
            `${surface.label}: the partnerProfile body must use its rEnd parameter`
          ).toContain('rEnd');
          expect(
            body,
            `${surface.label}: the partnerProfile body must not read the per-fragment ` +
              'radius rPx — the partner radius is the shared-vertex radius (#1494)'
          ).not.toMatch(/\brPx\b/);
        });
      }
    });
  }

  it('locks EVERY surface with a partner reconstruction, not just the four listed', () => {
    // (f) `SURFACES` is hardcoded, so a fifth capsule surface — another pick
    // variant, a depth prepass — would inherit the same hazard and escape this
    // file silently. Ground truth is every module under `src/rendering` that
    // builds a partner profile, by either backend's name for it. The scan runs
    // on comment-stripped text, so a doc comment mentioning either name is not
    // a new surface. This test file lives outside `src/rendering`, so its own
    // pinned strings cannot match. Standalone shader sources count too: a
    // `.glsl` / `.wgsl` (or a `.tsx` factory) carrying the reconstruction would
    // otherwise be a silent hole — none exist today, so the set stays exact.
    const renderingRoot = path.join(SRC_ROOT, 'rendering');
    const found = readdirSync(renderingRoot, { recursive: true })
      .map((entry) => String(entry).split(path.sep).join('/'))
      .filter((entry) => /\.(?:ts|tsx|glsl|wgsl)$/.test(entry))
      .filter((entry) =>
        /\bluxarPartnerProfile\b|\bconst\s+partnerProfile\b/.test(
          stripComments(readSource(`rendering/${entry}`))
        )
      )
      .map((entry) => `rendering/${entry}`)
      .sort();

    expect(found).toEqual(
      [
        'rendering/materials/line/shader-glsl-capsule.ts',
        'rendering/picking/line/shaders-capsule.ts',
        TSL_VISUAL_PATH,
        TSL_PICK_PATH,
      ].sort()
    );
  });
});
