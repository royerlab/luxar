/**
 * TSL line-join width regression lock (#790 / #1345).
 *
 * The line join block writes the two `flat` varyings `vCapSuppressStart` /
 * `vCapSuppressEnd`. A `flat` varying is resolved from ONE provoking vertex —
 * the LAST vertex under WebGL/GLSL and the FIRST under WGSL — so any decision
 * feeding it must be segment-constant. Feeding the join a per-VERTEX clamped
 * pixel width lets the t=0 and t=1 corners of a tapered / foreshortened segment
 * land on opposite sides of the 2 px join gate, and the two backends then pick
 * DIFFERENT answers for the same segment. Every line vertex stage therefore
 * computes two per-END, segment-constant half-widths and hands each
 * `tslLineJoin` call its own.
 *
 * Why a string-level tripwire rather than a behavioural test:
 *
 * - The GLSL half of this contract is already pinned by a source assertion in
 *   `material-glsl.test.ts` ('#790 both vertex stages hand luxarLineJoin each
 *   END its own segment-constant width'). The TSL half was pinned ONLY by the
 *   checked-in codegen snapshots, and `.github/workflows/ci.yml` sets the whole
 *   `e2e-tests` job to `if: false` — so the two TSL factories could regress to a
 *   single shared per-vertex `joinPixelWidth` with every CI check still green.
 * - The TSL/GLSL parity harness cannot see the divergence either: its TSL side
 *   runs `WebGPURenderer({ forceWebGL: true })`, so both sides share WebGL's
 *   provoking rule, and the tapered parity fixture does not straddle the gate.
 * - Mocking `tslLineJoin` and building the factory records ZERO calls: the whole
 *   vertex stage is traced inside `Fn(() => {...})`, whose body only runs during
 *   a real node build, which needs a GPU/WebGL backend unavailable under jsdom.
 *
 * So this reads the two TSL module sources from disk and greps them, in the same
 * documented "regression lock" spirit as `tests/unit/rendering/shader-hot-path.test.ts`.
 *
 * Pinning only the `joinPixelWidth:` operands is not enough: a block-scoped
 * shadow (`const startEndPixelWidth = <per-vertex>;` inside the join `if`)
 * leaves every call-site assertion green while both joins read a per-vertex
 * width. Three guards overlap so no single edit gets through — each width is
 * declared EXACTLY once (name-independent, so a per-vertex local of any name is
 * caught), everything from the width helper down to the per-corner offset
 * select is free of per-vertex identifiers (catches a re-binding that declares
 * nothing), and that region provably spans both join calls. Every match runs on
 * comment-stripped text, so prose can neither satisfy a pin nor move a region
 * boundary.
 */

import { readdirSync, readFileSync } from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, it, expect } from 'vitest';

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
 * Return the raw text of every `callee({ ... })` argument object in `source`.
 *
 * A brace-matching scan rather than a regex: the argument objects nest (each one
 * spreads `...shared` and passes multi-line node expressions), and a greedy or
 * lazy regex either swallows both calls or stops at the first inner `}`.
 */
function extractObjectArguments(source: string, callee: string): string[] {
  const opener = new RegExp(`\\b${callee}\\s*\\(\\s*\\{`, 'g');
  const bodies: string[] = [];
  let match: RegExpExecArray | null;
  while ((match = opener.exec(source)) !== null) {
    const bodyStart = match.index + match[0].length;
    let depth = 1;
    let cursor = bodyStart;
    while (cursor < source.length && depth > 0) {
      const character = source[cursor];
      if (character === '{') depth += 1;
      else if (character === '}') depth -= 1;
      cursor += 1;
    }
    expect(depth, `unbalanced braces in a ${callee}({...}) call`).toBe(0);
    bodies.push(source.slice(bodyStart, cursor - 1));
    opener.lastIndex = cursor;
  }
  return bodies;
}

/**
 * Identifiers that vary per QUAD VERTEX — none may appear anywhere between the
 * per-END width helper and the corner select.
 *
 * A name list can only ever be a BACKSTOP: a per-vertex local named something
 * else walks straight through it. The name-independent guard is the
 * declared-exactly-once rule below; this list adds precise messages for the
 * handles that actually exist today, `t` being the rawest of them (`tEff`,
 * `mvPos` and `width` all derive from it).
 *
 * `clampedPixelWidth` is deliberately included even though no TSL factory
 * currently declares it: it is the GLSL-side name for the per-vertex clamp
 * (guarded in `material-glsl.test.ts`), listed here as forward defence in case a
 * TSL factory ever grows the same local. Matching is on comment-stripped text,
 * so short names like `t` cannot be tripped by prose ("at t = 0").
 */
const PER_VERTEX_IDENTIFIERS = [
  't',
  'tEff',
  'width',
  'aQuadCorner',
  'rawPixelWidth',
  'clampedPixelWidth',
  'mvPos',
] as const;

/** The two per-END widths, each of which must be declared EXACTLY once. */
const PER_END_WIDTHS = ['startEndPixelWidth', 'endEndPixelWidth'] as const;

/** Region markers — the guarded span runs from the first to the third. */
const HELPER_MARKER = 'const endPixelWidthAt';
const WIDTHS_MARKER = 'const startEndPixelWidth';
const REGION_END_MARKER = 'const cornerOffset';

const TSL_VERTEX_STAGES: Array<[string, string]> = [
  ['visual', 'rendering/materials/line/shader-tsl.ts'],
  ['pick', 'rendering/picking/line/pick.tsl.ts'],
];

interface ParsedFactory {
  /** The factory source with every comment removed — ALL matching runs on this. */
  readonly source: string;
  /** Each `tslLineJoin({...})` argument object, paired with the END it serves. */
  readonly joinCalls: ReadonlyArray<{ readonly atEnd: boolean; readonly body: string }>;
  /** The `endPixelWidthAt` definition. */
  readonly helper: string;
  /** Width helper → corner select: the span that must stay segment-constant. */
  readonly joinRegion: string;
}

const PARSE_CACHE = new Map<string, ParsedFactory>();

/**
 * Parse one factory source — lazily, from inside a test body.
 *
 * Deliberately NOT done at collection time: an `expect()` that throws while the
 * describe callback runs collapses the whole file to zero tests, so one shape
 * change would blind every assertion here at once. Memoised so the repeated
 * calls cost one read.
 *
 * Comments are stripped ONCE, up front, and every marker is located in the
 * stripped text. Locating them in the raw source lets a comment move a region
 * boundary: a `// ... const cornerOffset ...` line above the join block shrinks
 * the guarded span to nothing (a shadow inside the block then goes unseen), and
 * a comment mentioning `const endPixelWidthAt` grows it up into the per-vertex
 * prologue and reds spuriously.
 */
function parseFactory(relativeToSrc: string): ParsedFactory {
  const cached = PARSE_CACHE.get(relativeToSrc);
  if (cached) return cached;

  const source = stripComments(readSource(relativeToSrc));
  const joinCalls = extractObjectArguments(source, 'tslLineJoin').map((body) => {
    const atEnd = /\batEnd\s*:\s*(true|false)\b/.exec(body);
    expect(atEnd, `${relativeToSrc}: no atEnd flag in a tslLineJoin call:\n${body}`).not.toBeNull();
    return { atEnd: atEnd![1] === 'true', body };
  });

  const helperStart = source.indexOf(HELPER_MARKER);
  const widthsStart = source.indexOf(WIDTHS_MARKER);
  const regionEnd = source.indexOf(REGION_END_MARKER);
  // Assert every marker was found BEFORE slicing: a rename would otherwise
  // yield an empty (and trivially green) region.
  expect(helperStart, `${relativeToSrc}: '${HELPER_MARKER}' not found`).toBeGreaterThan(-1);
  expect(widthsStart, `${relativeToSrc}: '${WIDTHS_MARKER}' not after the helper`).toBeGreaterThan(
    helperStart
  );
  expect(
    regionEnd,
    `${relativeToSrc}: '${REGION_END_MARKER}' not after the widths`
  ).toBeGreaterThan(widthsStart);

  const parsed: ParsedFactory = {
    source,
    joinCalls,
    helper: source.slice(helperStart, widthsStart),
    joinRegion: source.slice(helperStart, regionEnd),
  };
  PARSE_CACHE.set(relativeToSrc, parsed);
  return parsed;
}

function joinCallFor(parsed: ParsedFactory, atEnd: boolean): string {
  const matches = parsed.joinCalls.filter((call) => call.atEnd === atEnd);
  expect(matches, `expected exactly one join call with atEnd: ${atEnd}`).toHaveLength(1);
  return matches[0].body;
}

describe('TSL line join width wiring (#790 / #1345)', () => {
  for (const [label, relativePath] of TSL_VERTEX_STAGES) {
    describe(`${label} factory (${path.basename(relativePath)})`, () => {
      it('calls tslLineJoin exactly twice, once per segment end', () => {
        const { joinCalls } = parseFactory(relativePath);
        expect(joinCalls).toHaveLength(2);
        // Keyed on `atEnd`, never on emission order: swapping the two calls is
        // behaviourally identical and must stay green.
        expect(joinCalls.map((call) => call.atEnd).sort()).toEqual([false, true]);
      });

      it('gives each end its OWN width, never one shared operand', () => {
        // A single shared `joinPixelWidth` on both calls is exactly the
        // pre-fix shape this lock exists to catch.
        const parsed = parseFactory(relativePath);
        const widthOf = (atEnd: boolean): string => {
          const body = joinCallFor(parsed, atEnd);
          const width = /\bjoinPixelWidth\s*:\s*([A-Za-z_$][\w$]*)\s*[,}]/.exec(body);
          expect(width, `no joinPixelWidth operand in:\n${body}`).not.toBeNull();
          return width![1];
        };
        expect(widthOf(false), 'start-end join width').toBe('startEndPixelWidth');
        expect(widthOf(true), 'end-end join width').toBe('endEndPixelWidth');
        // ...and the operand is passed at the call site only — nothing may
        // smuggle a third one in through the spread `...shared` object.
        expect(parsed.source.match(/\bjoinPixelWidth\s*:/g)).toHaveLength(2);
      });

      it('declares each per-END width exactly once, so nothing can shadow it', () => {
        // The name-INDEPENDENT guard, and the one that matters most: every
        // assertion keyed on `startEndPixelWidth` / `endEndPixelWidth` is
        // satisfied by a second, block-scoped declaration of the same name
        // inside the join `if` that binds a per-vertex value — whatever that
        // value happens to be called.
        const { source } = parseFactory(relativePath);
        for (const name of PER_END_WIDTHS) {
          expect(
            source.match(new RegExp(`\\bconst\\s+${name}\\b`, 'g')) ?? [],
            `${label}: ${name} must be declared exactly once — a second ` +
              'declaration shadows the segment-constant width inside the join block'
          ).toHaveLength(1);
        }
      });

      it('derives both widths from the shared per-END helper at the segment ends', () => {
        const { source, helper } = parseFactory(relativePath);
        expect(source).toMatch(
          /const\s+startEndPixelWidth[^=]*=\s*endPixelWidthAt\(\s*tA\s*,\s*mvStart\.z\s*\)/
        );
        expect(source).toMatch(
          /const\s+endEndPixelWidth[^=]*=\s*endPixelWidthAt\(\s*tB\s*,\s*mvEnd\.z\s*\)/
        );
        // The helper must be a function OF ITS PARAMETERS: the END's own
        // interpolated width at the END's own view depth, clamped into the same
        // [minPixelWidth, maxPW] window the quad expands by. Everything else in
        // the argument list is left loose on purpose — a hoisted ortho flag or
        // an interleaved comment is semantics-preserving, and a helper that
        // reads a per-vertex value instead is caught by the region scan below.
        expect(helper).toMatch(/\(\s*tEnd\s*:[^,]*,\s*mvZ\s*:/);
        expect(helper).toMatch(
          /clamp\(\s*tslLineEndPixelWidth\(\s*[^,]+,\s*mix\(\s*startW\s*,\s*endW\s*,\s*tEnd\s*\)\s*,\s*mvZ\s*,[\s\S]*?\)\s*,\s*minPixelWidth\s*,\s*maxPW\s*,?\s*\)/
        );
      });

      it('routes no per-VERTEX identifier into the join region', () => {
        const { joinRegion, joinCalls } = parseFactory(relativePath);
        // Integrity first: a region that no longer spans both calls guards
        // nothing, and would otherwise pass every scan below vacuously.
        expect(
          joinRegion.match(/\btslLineJoin\s*\(/g) ?? [],
          `${label}: the guarded region must span BOTH tslLineJoin calls`
        ).toHaveLength(2);
        for (const identifier of PER_VERTEX_IDENTIFIERS) {
          // The whole span, so a per-vertex value reaching the join reds here
          // even though every call site still looks right. Complements the
          // declared-once rule above, which cannot see a re-BINDING that
          // declares nothing (`startEndPixelWidth.assign(rawPixelWidth)`).
          expect(
            joinRegion,
            `${label}: nothing between the width helper and the corner select ` +
              `may read per-vertex ${identifier}`
          ).not.toMatch(new RegExp(`\\b${identifier}\\b`));
        }
        // Same check per call, purely for a message that names the offender.
        for (const { atEnd, body } of joinCalls) {
          for (const identifier of PER_VERTEX_IDENTIFIERS) {
            expect(
              body,
              `${label} join call (atEnd: ${atEnd}) must not read per-vertex ${identifier}`
            ).not.toMatch(new RegExp(`\\b${identifier}\\b`));
          }
        }
      });
    });
  }

  it('locks EVERY TSL factory that calls tslLineJoin, not just the two listed', () => {
    // `TSL_VERTEX_STAGES` is hardcoded, so a third line factory (a future
    // depth-prepass or outline stage, say) would inherit the same flat-varying
    // hazard and escape this file silently. Ground truth is every call site
    // under `src/rendering` (the root stays narrow on purpose — this test file
    // contains a literal `tslLineJoin({...})` and would match its own
    // detector); the module that DEFINES `tslLineJoin` is excluded by content,
    // not by path, so moving the definition does not read as a new factory.
    // Both content greps run on comment-stripped text, so a `{@link
    // tslLineJoin()}` in someone's doc comment is not a new factory either.
    const renderingRoot = path.join(SRC_ROOT, 'rendering');
    const callers = readdirSync(renderingRoot, { recursive: true })
      .map((entry) => String(entry).split(path.sep).join('/'))
      .filter((entry) => entry.endsWith('.ts'))
      .map((entry) => ({ entry, source: stripComments(readSource(`rendering/${entry}`)) }))
      .filter(({ source }) => /\btslLineJoin\s*\(/.test(source))
      .filter(({ source }) => !/export\s+function\s+tslLineJoin\b/.test(source))
      .map(({ entry }) => `rendering/${entry}`)
      .sort();

    expect(callers).toEqual(TSL_VERTEX_STAGES.map(([, file]) => file).sort());
  });

  it('keeps the shared end-width helper defined, and parameter-pure, on BOTH backends', () => {
    // The GLSL twin of this lock lives in `material-glsl.test.ts` ('#790 both
    // vertex stages hand luxarLineJoin each END its own segment-constant
    // width'); the two backends compute the width from one definition each, and
    // renaming either without updating the other silently unpins one half.
    const tslHelpers = stripComments(readSource('rendering/materials/_shared/tsl-helpers.ts'));
    const signature = 'export function tslLineEndPixelWidth(';
    expect(tslHelpers).toContain(signature);
    // Stripped like everything else here: `glsl-lib.ts` holds GLSL inside
    // template literals, so a commented-out signature would otherwise satisfy
    // the pin.
    expect(stripComments(readSource('rendering/materials/_shared/glsl-lib.ts'))).toContain(
      'float luxarLineEndPixelWidth('
    );

    // Pinning the call sites is worth nothing if the shared helper reaches
    // around them: reading a vertex attribute inside its body makes the result
    // per-vertex again, everywhere at once.
    const bodyStart = tslHelpers.indexOf(signature);
    const bodyEnd = tslHelpers.indexOf('\n}', bodyStart);
    expect(bodyEnd, 'tslLineEndPixelWidth has no closing brace').toBeGreaterThan(bodyStart);
    const body = tslHelpers.slice(bodyStart, bodyEnd);
    expect(body, 'tslLineEndPixelWidth must not read a vertex attribute').not.toMatch(
      /\battribute\s*\(/
    );
    expect(body, 'tslLineEndPixelWidth must not read aQuadCorner').not.toMatch(/\baQuadCorner\b/);
  });
});
