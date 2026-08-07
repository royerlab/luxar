/**
 * Numeric tests for the screen-space line MITER JOIN decision math (#790).
 *
 * `luxarLineJoin` (`materials/_shared/glsl-lib.ts`) and its TSL twin
 * `tslLineJoin` (`materials/_shared/tsl-helpers.ts`) decide, per END of per
 * segment, whether a joint is mitred, what corner offset the quad expands by,
 * and what endpoint cap the `flat` varyings carry. The whole point of the
 * construction is that the TWO SEGMENTS meeting at a joint reach the SAME
 * answer — a disagreement leaves one rotated edge with nothing to tile against,
 * which rasterises as a flap sticking out of the tube. That two-sided property
 * is what this file pins.
 *
 * Audit acknowledgment, same as `line-cap-math.test.ts`: `evalJoin` below is a
 * verbatim TS re-implementation of the GLSL, so it tests the copy, not the
 * shader. GLSL evaluation needs a WebGL context vitest/jsdom cannot supply. The
 * shader TEXT is tied back to this mirror by source assertions in
 * `materials/line/material-glsl.test.ts`.
 *
 * What the browser specs do and do NOT add, precisely: `line-join-artifacts`
 * renders `test_line_joints.luxar.zarr`, every joint of which is a
 * `line_type="polyline"` end->start joint at constant width on a face-on plane,
 * so it is structurally blind to all three of the properties below — the
 * partner-leg orientation (which only bites at a same-parity joint), the
 * per-END join width (which needs a taper or foreshortening to straddle the
 * gate), and the near-plane conjunction (every endpoint there sits at the same
 * depth, far in front of the near plane, so both operands pass either way). Two
 * fixture pairs in the GLSL/TSL parity harness
 * (`tests/e2e/harnesses/tsl-harness/lines.ts`, asserted in
 * `tests/e2e/tsl-shader-parity.spec.ts`) reach the real shaders with two of the
 * three: `line-join-same-parity-*` with the orientation, and
 * `line-join-near-plane-*` with the conjunction — the latter puts one leg's far
 * endpoint behind nearCull, so both sides must decline and the mitred render
 * must come out pixel-identical to the unmitred one.
 *
 * The per-END width is NOT backstopped there: that harness renders its TSL side
 * through `WebGPURenderer({ forceWebGL: true })`, so both sides compile to GLSL
 * and a per-vertex width would look identical on each. What pins it is the
 * codegen snapshot `tests/__codegen__/line.vertex.glsl.txt` (generated from the
 * TSL graph) together with the source assertion `#790 both vertex stages hand
 * luxarLineJoin each END its own segment-constant width` in
 * `materials/line/material-glsl.test.ts` — plus the decision itself, mirrored
 * below.
 *
 * The projection is factored out: the shader derives `sharedPx` / `farPx` from
 * NDC and the view matrices, and this mirror takes the pixel-space points and
 * the view-space depths directly. Projection is not what the join decides.
 *
 * So is the PARTNER TEXEL CHOICE, and that one is a genuine blind spot rather
 * than an irrelevance: `partnerSharesItsStart` remains the correct predicate
 * for WHICH of the partner's two endpoints to fetch — a different question from
 * the orientation rule above, which moved to `atEnd`. Simplifying the fetch to
 * `atEnd` as well would be a real regression, and this mirror is handed `farPx`
 * already chosen, so it could not fail. What pins it behaviourally is the
 * same-parity fixture, where the two predicates disagree.
 */
import { describe, it, expect } from 'vitest';

type Vec2 = readonly [number, number];

const sub = (a: Vec2, b: Vec2): Vec2 => [a[0] - b[0], a[1] - b[1]];
const add = (a: Vec2, b: Vec2): Vec2 => [a[0] + b[0], a[1] + b[1]];
const mul = (a: Vec2, s: number): Vec2 => [a[0] * s, a[1] * s];
const dot = (a: Vec2, b: Vec2): number => a[0] * b[0] + a[1] * b[1];
const len = (a: Vec2): number => Math.hypot(a[0], a[1]);
/** GLSL `vec2(-v.y, v.x)` — the +R side of a direction. */
const perpOf = (v: Vec2): Vec2 => [-v[1], v[0]];

/** Everything `luxarLineJoin` reads, with the projection already applied. */
interface JoinInput {
  /** This vertex sits at the segment's END (t == 1). */
  readonly atEnd: boolean;
  /** Near-clipping did NOT move this endpoint off its source vertex. */
  readonly reachesVertex: boolean;
  /** texel4.y at the start, texel4.z at the end. */
  readonly jointCode: number;
  /** This segment's own storage slot (self-reference guard). */
  readonly selfSlot: number;
  /** The shared vertex, pixel space. */
  readonly sharedPx: Vec2;
  /** The PARTNER's far endpoint, pixel space (the one the texel fetch reads). */
  readonly farPx: Vec2;
  /** That partner far endpoint's view-space depth (-mvZ). */
  readonly farDepth: number;
  /** THIS segment's far endpoint depth, from the UNCLIPPED endpoints. */
  readonly selfFarDepth: number;
  /** This segment's unit direction, pixel space. */
  readonly lineDir: Vec2;
  readonly pixelLen: number;
  /** THIS END's clamped pixel half-width — segment-constant. */
  readonly joinPixelWidth: number;
  readonly nearCull: number;
  readonly isOrtho: boolean;
  /** uLineJoin: 0 none, 1 miter. */
  readonly lineJoin: number;
}

interface JoinResult {
  /** This corner's offset in pixel space. */
  readonly cornerOffset: Vec2;
  /** The endpoint cap, or -1 for "no partner reached — keep the default". */
  readonly cap: number;
  /** True only on the mitred path (the rotated end edge). */
  readonly mitred: boolean;
  /**
   * `dot(dirIn, dirOut)`, exposed so a test can assert the orientation rule
   * without re-deriving it (the shader has no such output). NaN on the paths
   * that return before any partner leg is oriented.
   */
  readonly turn: number;
}

/** Mirror of `luxarLineJoin` in `materials/_shared/glsl-lib.ts`. */
function evalJoin(i: JoinInput): JoinResult {
  const perpendicular = perpOf(i.lineDir);
  const noJoin: JoinResult = {
    cornerOffset: mul(perpendicular, i.joinPixelWidth),
    cap: -1,
    mitred: false,
    turn: Number.NaN,
  };

  const joinMinHalfWidth = 2.0;
  if (i.lineJoin < 0.5 || i.joinPixelWidth <= joinMinHalfWidth) return noJoin;
  if (!i.reachesVertex) return noJoin;

  const partnerSharesItsStart = i.jointCode > 0.0;
  const partnerSlot = partnerSharesItsStart
    ? Math.trunc(i.jointCode) - 1
    : Math.trunc(-i.jointCode) - 3;
  const namesAPartner = (i.jointCode > 0.5 || i.jointCode < -2.5) && partnerSlot !== i.selfSlot;
  if (!namesAPartner) return noJoin;

  // Oriented on `atEnd`, NOT on `partnerSharesItsStart`: the chain runs through
  // the joint in THIS segment's traversal sense, so the partner leg points away
  // from the shared vertex when this segment arrives and into it when it leaves.
  const partnerDelta = i.atEnd ? sub(i.farPx, i.sharedPx) : sub(i.sharedPx, i.farPx);
  const partnerLen = len(partnerDelta);
  const bothFarInFront = i.isOrtho || (i.farDepth >= i.nearCull && i.selfFarDepth >= i.nearCull);
  if (!bothFarInFront || partnerLen <= 0.0001 || i.pixelLen <= 0.0001) return noJoin;

  const partnerDir = mul(partnerDelta, 1 / partnerLen);
  const dirIn = i.atEnd ? i.lineDir : partnerDir;
  const dirOut = i.atEnd ? partnerDir : i.lineDir;
  const turn = dot(dirIn, dirOut);
  const suppression = Math.min(Math.max(turn, 0), 1);

  const grow = Math.sqrt(2.0 / Math.max(1.0 + turn, 1e-6));
  const axialReach = i.joinPixelWidth * Math.sqrt(Math.max(grow * grow - 1.0, 0.0));
  if (grow > 2.0 || axialReach > 0.5 * Math.min(i.pixelLen, partnerLen)) {
    return {
      cornerOffset: mul(perpendicular, i.joinPixelWidth),
      cap: suppression,
      mitred: false,
      turn,
    };
  }

  const perpIn = perpOf(dirIn);
  const perpOut = perpOf(dirOut);
  return {
    cornerOffset: mul(add(perpIn, perpOut), i.joinPixelWidth / (1.0 + turn)),
    cap: 1.0,
    mitred: true,
    turn,
  };
}

/**
 * The RETIRED orientation rule, written out once so the two-line difference is
 * on the record: the partner leg oriented on which of the PARTNER's endpoints
 * is shared is the partner's own traversal direction, which is the negation of
 * what this joint needs whenever the two segments meet end-to-end or
 * start-to-start.
 *
 * This is documentation, not coverage — it is a second implementation, so no
 * assertion reading it can constrain `evalJoin` (let alone the shader). The
 * coverage is `evalJoin`'s own `turn`, asserted alongside it below.
 */
function preFixTurn(i: JoinInput): number {
  const partnerSharesItsStart = i.jointCode > 0.0;
  const partnerDelta = partnerSharesItsStart ? sub(i.farPx, i.sharedPx) : sub(i.sharedPx, i.farPx);
  const partnerDir = mul(partnerDelta, 1 / len(partnerDelta));
  const dirIn = i.atEnd ? i.lineDir : partnerDir;
  const dirOut = i.atEnd ? partnerDir : i.lineDir;
  return dot(dirIn, dirOut);
}

/**
 * One leg of a joint. `sharedIsEnd` is the AUTHORED orientation — which of the
 * leg's own endpoints happens to be the shared vertex. The geometry is the
 * same either way; only the joint parity changes.
 */
interface Leg {
  readonly sharedIsEnd: boolean;
  readonly farPx: Vec2;
  readonly slot: number;
  /** View-space depth of this leg's far endpoint. */
  readonly farDepth: number;
}

const leg = (sharedIsEnd: boolean, farPx: Vec2, slot: number, farDepth = 100): Leg => ({
  sharedIsEnd,
  farPx,
  slot,
  farDepth,
});

/** Build one side's `JoinInput` from the two legs meeting at `sharedPx`. */
function sideInput(
  self: Leg,
  partner: Leg,
  sharedPx: Vec2,
  joinPixelWidth: number,
  overrides: Partial<JoinInput> = {}
): JoinInput {
  const away = sub(self.farPx, sharedPx);
  const pixelLen = len(away);
  // The leg runs far -> shared when its END is shared, shared -> far otherwise.
  const lineDir = mul(away, (self.sharedIsEnd ? -1 : 1) / pixelLen);
  // +(slot + 1) names the partner's START as the shared vertex, -(slot + 3) its END.
  const jointCode = partner.sharedIsEnd ? -(partner.slot + 3) : partner.slot + 1;
  return {
    atEnd: self.sharedIsEnd,
    reachesVertex: true,
    jointCode,
    selfSlot: self.slot,
    sharedPx,
    farPx: partner.farPx,
    farDepth: partner.farDepth,
    selfFarDepth: self.farDepth,
    lineDir,
    pixelLen,
    joinPixelWidth,
    nearCull: 1.0,
    isOrtho: false,
    lineJoin: 1.0,
    ...overrides,
  };
}

/**
 * The turn `evalJoin` itself computed. Read from the result rather than
 * re-derived, so the orientation rule exists exactly ONCE in this file and an
 * assertion on the turn constrains the same code path the corner offset and
 * the cap come out of.
 */
const turnOf = (i: JoinInput): number => evalJoin(i).turn;

/** The four ways two legs can be authored at one shared vertex. */
const PARITIES: ReadonlyArray<{ name: string; a: boolean; b: boolean }> = [
  { name: 'end->start', a: true, b: false },
  { name: 'start->end', a: false, b: true },
  { name: 'END-END', a: true, b: true },
  { name: 'START-START', a: false, b: false },
];

const SHARED: Vec2 = [100, 100];

describe('line join math (vertex-side, #790)', () => {
  // --- Turn sign at every joint parity ------------------------------------
  // `turn` must describe the ANGLE OF THE JOINT, which is a property of the
  // geometry alone — the authored direction of either polyline cannot change
  // it. Orienting the partner leg on `partnerSharesItsStart` instead of on
  // `atEnd` breaks exactly the two same-parity rows below: it hands back the
  // partner's own traversal direction, so a COLLINEAR joint reads turn = -1,
  // the miter limit rejects it, and clamp(turn, 0, 1) keeps the soft cap —
  // the #780 dimming, at every END-END and START-START joint.

  for (const parity of PARITIES) {
    it(`collinear joint reads turn == 1 from both sides (${parity.name})`, () => {
      // One straight line through the shared vertex; only the authored
      // orientation of each leg changes between parities.
      const a = leg(parity.a, [0, 100], 0);
      const b = leg(parity.b, [200, 100], 1);
      const inA = sideInput(a, b, SHARED, 8);
      const inB = sideInput(b, a, SHARED, 8);

      expect(turnOf(inA)).toBeCloseTo(1, 12);
      expect(turnOf(inB)).toBeCloseTo(1, 12);

      const resA = evalJoin(inA);
      const resB = evalJoin(inB);
      // Both sides agree the joint is mitred, and the miter reduces EXACTLY to
      // the plain perpendicular half-width, each on its own +R offset line.
      expect(resA.mitred).toBe(true);
      expect(resB.mitred).toBe(true);
      expect(resA.cap).toBe(1);
      expect(resB.cap).toBe(1);
      for (const [res, input] of [
        [resA, inA],
        [resB, inB],
      ] as Array<[JoinResult, JoinInput]>) {
        const expected = mul(perpOf(input.lineDir), input.joinPixelWidth);
        expect(res.cornerOffset[0]).toBeCloseTo(expected[0], 10);
        expect(res.cornerOffset[1]).toBeCloseTo(expected[1], 10);
      }
    });
  }

  it('the pre-fix orientation negates the turn at the two same-parity joints', () => {
    // The regression witness. end->start / start->end were always correct,
    // which is why the defect survived: it only shows where a polyline is
    // authored so that two segments meet end-to-end or start-to-start.
    const expectPreFix: Record<string, number> = {
      'end->start': 1,
      'start->end': 1,
      'END-END': -1,
      'START-START': -1,
    };
    for (const parity of PARITIES) {
      const a = leg(parity.a, [0, 100], 0);
      const b = leg(parity.b, [200, 100], 1);
      for (const input of [sideInput(a, b, SHARED, 8), sideInput(b, a, SHARED, 8)]) {
        expect(preFixTurn(input), parity.name).toBeCloseTo(expectPreFix[parity.name], 12);
        // The fixed orientation reads the geometry, not the authoring.
        expect(turnOf(input), parity.name).toBeCloseTo(1, 12);
      }
    }
  });

  // --- Both sides of a BENT joint agree ------------------------------------

  for (const parity of PARITIES) {
    it(`both sides of a 53 deg joint agree, each on its own +R line (${parity.name})`, () => {
      // NOT a right angle, deliberately. At exactly 90 deg the retired
      // orientation is indistinguishable from the current one: turn = -0 = 0
      // either way, and negating dirOut negates perpOut too, which leaves both
      // dot(M, perpOwn) and |M| unchanged. So a 90 deg bend passes pre-fix at
      // every parity and discriminates nothing.
      //
      // Legs: A runs 100 px horizontally into the shared vertex, B leaves it
      // 50 px along (0.6, 0.8), so turn = 0.6 (theta = 53.13 deg) and the
      // pre-fix same-parity reading would be -0.6 — past the miter limit
      // (grow = sqrt(2/0.4) = 2.24 > 2), i.e. NOT mitred at all.
      const R = 10;
      const a = leg(parity.a, [0, 100], 0);
      const b = leg(parity.b, [130, 140], 1);
      const inA = sideInput(a, b, SHARED, R);
      const inB = sideInput(b, a, SHARED, R);

      // Same turn, hence the same miter-limit and overshoot decisions.
      expect(turnOf(inA)).toBeCloseTo(0.6, 12);
      expect(turnOf(inB)).toBeCloseTo(0.6, 12);

      const resA = evalJoin(inA);
      const resB = evalJoin(inB);
      // grow = sqrt(2/1.6) = 1.118, and the axial reach R*sqrt(grow^2 - 1) = 5
      // px stays under the guard's 0.5 * min(pixelLen, partnerLen). That is
      // 25 px from BOTH sides, not 50 and 25: the min picks the shorter (50 px)
      // leg whichever side is asking, which is the whole point of taking a min
      // over the pair. So both sides miter, and a mitred joint is never dimmed.
      expect(resA.mitred).toBe(true);
      expect(resB.mitred).toBe(true);
      expect(resA.cap).toBe(1);
      expect(resB.cap).toBe(1);

      // The miter point lies exactly on each side's OWN +R offset line, which
      // is what keeps vPerpNorm a true perpendicular coordinate on both quads.
      expect(dot(resA.cornerOffset, perpOf(inA.lineDir))).toBeCloseTo(R, 10);
      expect(dot(resB.cornerOffset, perpOf(inB.lineDir))).toBeCloseTo(R, 10);
      // |M| = R / cos(theta/2) = R * sqrt(2/(1 + turn)) = R * sqrt(1.25).
      const miterLen = R * Math.sqrt(1.25);
      expect(len(resA.cornerOffset)).toBeCloseTo(miterLen, 10);
      expect(len(resB.cornerOffset)).toBeCloseTo(miterLen, 10);
    });
  }

  it('a turn past the 120 deg miter limit is declined by both sides', () => {
    // grow = sqrt(2/(1+turn)) > 2 at turn < -0.5. A 150 deg turn (the polyline
    // doubling back) reads turn = -cos(30 deg). Both sides read the same turn,
    // so neither miters and both fall back to the perpendicular.
    const a = leg(true, [0, 100], 0);
    const b = leg(false, [100 - Math.cos(Math.PI / 6) * 100, 150], 1);
    const inA = sideInput(a, b, SHARED, 8);
    const inB = sideInput(b, a, SHARED, 8);
    expect(turnOf(inA)).toBeLessThan(-0.5);
    expect(turnOf(inA)).toBeCloseTo(turnOf(inB), 12);
    expect(evalJoin(inA).mitred).toBe(false);
    expect(evalJoin(inB).mitred).toBe(false);
    // The cap is still DERIVED from the screen-space turn (clamped), not left
    // at the "no partner" sentinel.
    expect(evalJoin(inA).cap).toBe(0);
    expect(evalJoin(inB).cap).toBe(0);
  });

  // --- Each end's width must be segment-constant ---------------------------

  it('per-END widths make both caps single-valued across the quad corners', () => {
    // A tapered (or perspective-foreshortened) segment whose t=0 corner renders
    // at 1.8 px half-width — below the 2 px join gate — and whose t=1 corner
    // renders at 6 px. The two caps are `flat` varyings, so a value that
    // differs between the corners is resolved from one provoking vertex alone —
    // whichever corner that is, three of the four corners are then rasterised
    // with a cap their own end never derived.
    const cornerWidths = [1.8, 6.0] as const; // clamped width at t=0 and at t=1
    const startJoinWidth = cornerWidths[0];
    const endJoinWidth = cornerWidths[1];

    // Middle segment of a polyline: a partner at each end.
    const self = { slot: 1, start: [0, 100] as Vec2, end: [100, 100] as Vec2 };
    const startPartner = leg(true, [0, 0], 0); // bends at the start vertex
    const endPartner = leg(false, [200, 100], 2); // collinear at the end vertex

    /** Mirror of the call site: which width each end's call is given. */
    const capsAtCorner = (corner: 0 | 1, mode: 'per-end' | 'per-vertex') => {
      const wStart = mode === 'per-end' ? startJoinWidth : cornerWidths[corner];
      const wEnd = mode === 'per-end' ? endJoinWidth : cornerWidths[corner];
      const startSide = sideInput(
        leg(false, self.end, self.slot),
        startPartner,
        self.start,
        wStart
      );
      const endSide = sideInput(leg(true, self.start, self.slot), endPartner, self.end, wEnd);
      return { start: evalJoin(startSide).cap, end: evalJoin(endSide).cap };
    };

    const perEnd0 = capsAtCorner(0, 'per-end');
    // The two ends genuinely have DIFFERENT answers, and each end keeps its
    // own across the quad: the 1.8 px start end is below the gate (sentinel),
    // the 6 px end end miters. (`capsAtCorner` is a pure function of the two
    // widths in 'per-end' mode, so comparing its two corners against each
    // other would assert nothing — the corner argument is unread there.)
    expect(perEnd0.start).toBe(-1);
    expect(perEnd0.end).toBe(1);

    // Pre-fix: one per-VERTEX width fed to BOTH calls, so the same END reads a
    // different cap depending on which corner is being processed.
    const perVertex0 = capsAtCorner(0, 'per-vertex');
    const perVertex1 = capsAtCorner(1, 'per-vertex');
    expect(perVertex0.end).not.toBe(perVertex1.end);
    expect(perVertex0.start).not.toBe(perVertex1.start);
  });

  it('the join width also scales the miter point, not just the gate decisions', () => {
    // `joinPixelWidth` is not only a gate operand: the corner offset is
    // M = (perpIn + perpOut) * w / (1 + turn), linear in w. So which width an
    // end is given moves the quad's corner, and the two segments only tile if
    // both sides scale the SAME shared miter point. Nothing else in this file
    // pins that linearity — the cap assertions above are all decisions.
    //
    // The 53 deg joint again: |M| = w * sqrt(2/(1 + 0.6)) = w * sqrt(1.25), and
    // the axial reach w * 0.5 stays under 0.5 * min(100, 50) = 25 px at both
    // widths below, so neither falls back.
    const self = leg(true, [0, 100], 0);
    const partner = leg(false, [130, 140], 1);
    const at = (w: number) => evalJoin(sideInput(self, partner, SHARED, w));
    const m8 = at(8);
    const m16 = at(16);

    expect(m8.mitred).toBe(true);
    expect(m16.mitred).toBe(true);
    expect(len(m8.cornerOffset)).toBeCloseTo(8 * Math.sqrt(1.25), 10);
    // Doubling the width doubles the offset and leaves its direction alone.
    expect(m16.cornerOffset[0]).toBeCloseTo(2 * m8.cornerOffset[0], 10);
    expect(m16.cornerOffset[1]).toBeCloseTo(2 * m8.cornerOffset[1], 10);
  });

  // --- The near-plane guard is two-sided -----------------------------------

  it('a far endpoint behind the near plane makes BOTH sides decline', () => {
    // A runs front -> shared; B runs shared -> behind the near plane. Testing
    // only the FETCHED partner's far endpoint has each side testing a
    // DIFFERENT point, so A declines while B miters alone and B's rotated edge
    // has nothing to tile against.
    const nearCull = 1.0;
    const a = leg(true, [0, 100], 0, 100); // A's far endpoint is well in front
    const b = leg(false, [200, 100], 1, 0.5); // B's far endpoint is behind
    const inA = sideInput(a, b, SHARED, 8, { nearCull });
    const inB = sideInput(b, a, SHARED, 8, { nearCull });

    // The one-sided predicate the two sides used to disagree on:
    expect(inA.farDepth >= nearCull).toBe(false);
    expect(inB.farDepth >= nearCull).toBe(true);
    // The conjunction over BOTH far endpoints is the same pair from either
    // side — {A.far, B.far} — so the branch agrees.
    expect(evalJoin(inA).mitred).toBe(false);
    expect(evalJoin(inB).mitred).toBe(false);
    expect(evalJoin(inA).cap).toBe(-1);
    expect(evalJoin(inB).cap).toBe(-1);
  });

  it('either far endpoint behind is enough to decline; both in front miters', () => {
    const nearCull = 1.0;
    const behind = 0.5;
    const front = 100;
    const cases: Array<[number, number, boolean]> = [
      [front, front, true],
      [behind, front, false],
      [front, behind, false],
      [behind, behind, false],
    ];
    for (const [aDepth, bDepth, mitred] of cases) {
      const a = leg(true, [0, 100], 0, aDepth);
      const b = leg(false, [200, 100], 1, bDepth);
      const inA = sideInput(a, b, SHARED, 8, { nearCull });
      const inB = sideInput(b, a, SHARED, 8, { nearCull });
      expect(evalJoin(inA).mitred, `A with depths ${aDepth}/${bDepth}`).toBe(mitred);
      expect(evalJoin(inB).mitred, `B with depths ${aDepth}/${bDepth}`).toBe(mitred);
    }
  });

  it('ortho keeps the near-plane guard inert', () => {
    // There is no 1/z singularity under ortho, and the depths carry no
    // meaning for the expansion — the guard must not cull a legitimate joint.
    const a = leg(true, [0, 100], 0, 0.5);
    const b = leg(false, [200, 100], 1, 0.5);
    const inA = sideInput(a, b, SHARED, 8, { isOrtho: true });
    const inB = sideInput(b, a, SHARED, 8, { isOrtho: true });
    expect(evalJoin(inA).mitred).toBe(true);
    expect(evalJoin(inB).mitred).toBe(true);
  });

  // --- The gates that come before any partner is reached -------------------

  it('style none, the 2 px width gate, a clipped endpoint and a self-reference all fall back', () => {
    // A BENT base geometry (the 53 deg joint again), not a collinear one: the
    // miter reduces exactly to perp * w at a collinear joint, so the corner
    // offsets below would equal the fall-back value whether or not the gate
    // fired, and every one of them would pass vacuously.
    const a = leg(true, [0, 100], 0);
    const b = leg(false, [130, 140], 1);
    const base = sideInput(a, b, SHARED, 8);
    const plain = mul(perpOf(base.lineDir), base.joinPixelWidth);
    for (const [label, input] of [
      ['style none', { ...base, lineJoin: 0 }],
      ['below the 2 px gate', { ...base, joinPixelWidth: 2.0 }],
      ['near-clipped endpoint', { ...base, reachesVertex: false }],
      ['free-end sentinel', { ...base, jointCode: 0 }],
      ['slice-clipped sentinel', { ...base, jointCode: -1 }],
      ['degree->=3 hub sentinel', { ...base, jointCode: -2 }],
      ['self-reference', { ...base, selfSlot: 1 }],
    ] as Array<[string, JoinInput]>) {
      const res = evalJoin(input);
      expect(res.cap, label).toBe(-1);
      expect(res.mitred, label).toBe(false);
      const expected = mul(perpOf(input.lineDir), input.joinPixelWidth);
      expect(res.cornerOffset[0], label).toBeCloseTo(expected[0], 10);
      expect(res.cornerOffset[1], label).toBeCloseTo(expected[1], 10);
    }
    // The same input WITHOUT any of those gates miters, and to a corner the
    // fall-back never reaches: M = (perpIn + perpOut) * w / (1 + turn) is
    // (-4, 8) here against a plain perpendicular of (0, 8), 4 px apart. That
    // distance is what makes each `cornerOffset` assertion above a real one.
    const ungated = evalJoin(base);
    expect(ungated.mitred).toBe(true);
    expect(len(sub(ungated.cornerOffset, plain))).toBeCloseTo(4, 10);
  });
});
