/**
 * The two-sided near-plane guard on the screen-space line miter (issue #1346).
 *
 * Audit acknowledgment (rendering.md [W3]): {@link joinGate} below is a TS
 * mirror of the decision the shared join block makes — `luxarLineJoin` in
 * `materials/_shared/glsl-lib.ts` and its twin `tslLineJoin` in
 * `materials/_shared/tsl-helpers.ts`. It tests the copy, not the shader. That
 * is by design: GLSL/WGSL evaluation needs a WebGL/WebGPU context vitest's
 * jsdom cannot supply. Divergence between this mirror and the real shaders is
 * caught by the GLSL↔TSL parity fixtures `line-join-nearplane-miter` /
 * `-none` and their `-control-` counterparts (`e2e/tsl-shader-parity.spec.ts`),
 * which render this very scenario on both backends — the control being the pair
 * that proves the join block runs at all under this camera — PLUS the source
 * locks at the bottom of this file, which pin the shader text that feeds the
 * mirror's inputs.
 *
 * The scenario itself lives in
 * `tests/helpers/line-join-near-plane-scenario.ts` — the same module the
 * Playwright fixtures build from, so the unit test and the GPU test are
 * provably the same configuration.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  NEAR_PLANE_CAMERA,
  NEAR_PLANE_A_START,
  NEAR_PLANE_SHARED,
  NEAR_PLANE_B_FAR,
  NEAR_PLANE_CONTROL_B_FAR,
  NEAR_PLANE_JOINT_CODES,
  PERSPECTIVE_LINE_SCALE,
  halfWidthAtDepth,
  mixVec3,
  nearPlaneClipRange,
  projectToPixels,
  projectViewToPixels,
  viewDepth,
  type Vec2,
  type Vec3,
} from '../../helpers/line-join-near-plane-scenario';
import { GLSL_LINE_JOIN } from '../../../rendering/materials/_shared/glsl-lib';
// The rendered-half-width gate, imported rather than re-declared here. GLSL
// cannot import it, so `glsl-lib.ts` keeps a `joinMinHalfWidth` literal of its
// own; the source lock below asserts the two are equal, which is what actually
// stops this mirror drifting from either shader.
import { LINE_JOIN_MIN_HALF_WIDTH } from '../../../rendering/materials/_shared/tsl-helpers';
import { LINE_VERTEX_SHADER } from '../../../rendering/materials/line/shader-glsl';
import { LINE_PICK_VERTEX_SHADER } from '../../../rendering/picking/line/shaders';

const HERE = dirname(fileURLToPath(import.meta.url));
/**
 * The two TSL vertex factories, read as TEXT. Their join call sites cannot be
 * locked by importing anything: the argument object is consumed at graph-build
 * time inside a browser-only traced body, so the mapping is only observable in
 * the source. Same `readFileSync` + `import.meta.url` shape as
 * `unit/styles/data-loading-monitor-css.test.ts`.
 */
const TSL_STAGE_SOURCES = [
  ['visual', resolve(HERE, '../../../rendering/materials/line/shader-tsl.ts')],
  ['pick', resolve(HERE, '../../../rendering/picking/line/pick.tsl.ts')],
] as const;

/** Which near-plane conjunction to evaluate — the regression, expressed. */
type NearPlaneGuard = 'partner-only' | 'both';

/** One side of the joint, as the calling vertex stage assembles it. */
interface JoinSide {
  /** `true` for the segment whose END is the shared vertex (`atEnd`). */
  readonly atEnd: boolean;
  /** Pixel-space unit direction of THIS segment, after near-plane clipping. */
  readonly lineDir: Vec2;
  /** THIS segment's clipped pixel length. */
  readonly pixelLen: number;
  /** Rendered half-width at this vertex. */
  readonly clampedPixelWidth: number;
  /** `tA <= 0` / `tB >= 1` — near-clipping did not move this endpoint. */
  readonly reachesVertex: boolean;
  /** PRE-CLIP view depth of THIS segment's OTHER endpoint. */
  readonly thisFarDepth: number;
  /** Joint code at this endpoint (`texel4.y` / `texel4.z`). */
  readonly jointCode: number;
  /** This segment's own storage slot (the self-reference guard's operand). */
  readonly selfSlot: number;
  /** Shared vertex in pixel space. */
  readonly sharedPx: Vec2;
  /** The partner's far endpoint, as stored (pre-clip), in pixel space. */
  readonly partnerFarPx: Vec2;
  /** That endpoint's pre-clip view depth. */
  readonly partnerFarDepth: number;
}

/** What the join block decided, decomposed so a failure reads clearly. */
interface JoinDecision {
  /** `'perpendicular'` = the `noJoin` fallback; `'miter'` = rotated edge. */
  readonly kind: 'perpendicular' | 'miter';
  /** The corner offset in pixel space — `.xy` of the GLSL return. */
  readonly offset: Vec2;
  /**
   * `.z` of the GLSL return: `-1` is the "no partner reached" SENTINEL, meaning
   * the shader keeps the joint code's implied cap.
   */
  readonly capSentinel: number;
  /** Why the block bailed, for assertion messages. `null` when it mitred. */
  readonly bailedAt:
    | null
    | 'width-gate'
    | 'reaches-vertex'
    | 'names-a-partner'
    | 'near-plane'
    | 'length-epsilon'
    | 'miter-limit-or-overshoot';
  /** `dot(dirIn, dirOut)` where the block got that far, else `null`. */
  readonly turn: number | null;
  /** `sqrt(2/(1+turn))` where computed, else `null`. */
  readonly grow: number | null;
  /** `R * tan(theta/2)` where computed, else `null`. */
  readonly axialReach: number | null;
  /** `0.5 * min(pixelLen, partnerLen)` where computed, else `null`. */
  readonly overshootBound: number | null;
}

const perp = ([x, y]: Vec2, r: number): Vec2 => [-y * r, x * r];
const dot2 = (a: Vec2, b: Vec2): number => a[0] * b[0] + a[1] * b[1];

/**
 * Faithful mirror of `luxarLineJoin`'s decision, in the shader's own order:
 * width gate → `reachesVertex` → slot-bearing/self-reference → the near-plane
 * conjunction → the length epsilons → miter limit → axial-reach overshoot.
 *
 * `nearPlaneGuard` selects which conjunction to evaluate: `'both'` is the
 * shipped two-sided form, `'partner-only'` the pre-#1346 one-sided form. Having
 * both here is what lets the regression be expressed as a behavioural
 * difference between the two sides of one joint instead of as prose.
 */
function joinGate(side: JoinSide, nearPlaneGuard: NearPlaneGuard): JoinDecision {
  const { nearCull } = NEAR_PLANE_CAMERA;
  const fallback = (
    bailedAt: NonNullable<JoinDecision['bailedAt']>,
    extra: Partial<JoinDecision> = {}
  ): JoinDecision => ({
    kind: 'perpendicular',
    offset: perp(side.lineDir, side.clampedPixelWidth),
    capSentinel: -1,
    bailedAt,
    turn: null,
    grow: null,
    axialReach: null,
    overshootBound: null,
    ...extra,
  });

  if (side.clampedPixelWidth <= LINE_JOIN_MIN_HALF_WIDTH) return fallback('width-gate');
  if (!side.reachesVertex) return fallback('reaches-vertex');

  const partnerSharesItsStart = side.jointCode > 0;
  const partnerSlot = partnerSharesItsStart ? side.jointCode - 1 : -side.jointCode - 3;
  const namesAPartner =
    (side.jointCode > 0.5 || side.jointCode < -2.5) && partnerSlot !== side.selfSlot;
  if (!namesAPartner) return fallback('names-a-partner');

  const partnerDelta: Vec2 = partnerSharesItsStart
    ? [side.partnerFarPx[0] - side.sharedPx[0], side.partnerFarPx[1] - side.sharedPx[1]]
    : [side.sharedPx[0] - side.partnerFarPx[0], side.sharedPx[1] - side.partnerFarPx[1]];
  const partnerLen = Math.hypot(partnerDelta[0], partnerDelta[1]);

  // The guard under test. `thisFarInFront` is the #1346 addition; both sides of
  // a joint evaluate the identical conjunction once it is present.
  const thisFarInFront = side.thisFarDepth >= nearCull;
  const partnerInFront = side.partnerFarDepth >= nearCull;
  const inFront = nearPlaneGuard === 'both' ? thisFarInFront && partnerInFront : partnerInFront;
  if (!inFront) return fallback('near-plane');
  if (partnerLen <= 0.0001 || side.pixelLen <= 0.0001) return fallback('length-epsilon');

  // CANONICAL operand order — incoming edge first, outgoing second.
  const partnerDir: Vec2 = [partnerDelta[0] / partnerLen, partnerDelta[1] / partnerLen];
  const dirIn: Vec2 = side.atEnd ? side.lineDir : partnerDir;
  const dirOut: Vec2 = side.atEnd ? partnerDir : side.lineDir;
  const turn = dot2(dirIn, dirOut);

  const grow = Math.sqrt(2 / Math.max(1 + turn, 1e-6));
  const axialReach = side.clampedPixelWidth * Math.sqrt(Math.max(grow * grow - 1, 0));
  const overshootBound = 0.5 * Math.min(side.pixelLen, partnerLen);
  const measured = { turn, grow, axialReach, overshootBound };
  if (grow > 2 || axialReach > overshootBound) {
    // Note the cap here is the DERIVED screen-space suppression, not the
    // sentinel: a partner WAS reached, only the miter was refused.
    return {
      ...fallback('miter-limit-or-overshoot', measured),
      capSentinel: Math.min(Math.max(turn, 0), 1),
    };
  }

  const perpIn: Vec2 = [-dirIn[1], dirIn[0]];
  const perpOut: Vec2 = [-dirOut[1], dirOut[0]];
  const s = side.clampedPixelWidth / (1 + turn);
  return {
    kind: 'miter',
    offset: [(perpIn[0] + perpOut[0]) * s, (perpIn[1] + perpOut[1]) * s],
    capSentinel: 1,
    bailedAt: null,
    ...measured,
  };
}

/**
 * Assemble both sides of the joint for the scenario, with B's far endpoint
 * placed at `bFar` and (for the last test) the shared vertex at `shared`.
 * Everything is derived from the scenario module's projection helpers so
 * nothing is hand-tuned twice.
 */
function buildJoint(bFar: Vec3, shared: Vec3 = NEAR_PLANE_SHARED): { a: JoinSide; b: JoinSide } {
  const aStartDepth = viewDepth(NEAR_PLANE_A_START);
  const sharedDepth = viewDepth(shared);
  const bFarDepth = viewDepth(bFar);

  const toView = (w: Vec3): Vec3 => [w[0], w[1], w[2] - NEAR_PLANE_CAMERA.camZ];

  /** One segment's CLIPPED pixel-space endpoints, direction and length. */
  const segment = (start: Vec3, end: Vec3, startDepth: number, endDepth: number) => {
    const { tA, tB } = nearPlaneClipRange(startDepth, endDepth);
    const mvStart = mixVec3(toView(start), toView(end), tA);
    const mvEnd = mixVec3(toView(start), toView(end), tB);
    const pxStart = projectViewToPixels(mvStart);
    const pxEnd = projectViewToPixels(mvEnd);
    const delta: Vec2 = [pxEnd[0] - pxStart[0], pxEnd[1] - pxStart[1]];
    const pixelLen = Math.hypot(delta[0], delta[1]);
    return {
      tA,
      tB,
      pixelLen,
      lineDir: [delta[0] / pixelLen, delta[1] / pixelLen] as Vec2,
      startHalfWidth: halfWidthAtDepth(-mvStart[2]),
      endHalfWidth: halfWidthAtDepth(-mvEnd[2]),
    };
  };

  const segA = segment(NEAR_PLANE_A_START, shared, aStartDepth, sharedDepth);
  const segB = segment(shared, bFar, sharedDepth, bFarDepth);
  const sharedPx = projectToPixels(shared).px;

  return {
    // A's END is the shared vertex; its partner is B, whose START is shared
    // (code +2), so A fetches B's stored END.
    a: {
      atEnd: true,
      lineDir: segA.lineDir,
      pixelLen: segA.pixelLen,
      clampedPixelWidth: segA.endHalfWidth,
      reachesVertex: segA.tB >= 1,
      thisFarDepth: aStartDepth,
      jointCode: NEAR_PLANE_JOINT_CODES.aEnd,
      selfSlot: 0,
      sharedPx,
      partnerFarPx: projectToPixels(bFar).px,
      partnerFarDepth: bFarDepth,
    },
    // B's START is the shared vertex; its partner is A, whose END is shared
    // (code −3), so B fetches A's stored START.
    b: {
      atEnd: false,
      lineDir: segB.lineDir,
      pixelLen: segB.pixelLen,
      clampedPixelWidth: segB.startHalfWidth,
      reachesVertex: segB.tA <= 0,
      thisFarDepth: bFarDepth,
      jointCode: NEAR_PLANE_JOINT_CODES.bStart,
      selfSlot: 1,
      sharedPx,
      partnerFarPx: projectToPixels(NEAR_PLANE_A_START).px,
      partnerFarDepth: aStartDepth,
    },
  };
}

/** Whitespace-normalised source, so a prettier reflow cannot break a lock. */
const flat = (src: string): string => src.replace(/\s+/g, ' ');

/**
 * Every `tslLineJoin({ ... })` call's argument object, as text, from a
 * whitespace-normalised TSL factory source. Splitting on the call and its
 * closing `});` keeps each object's fields together, so the `atEnd` ↔
 * `thisFarDepth` pairing can be checked PER CALL — a single flat regex over the
 * file would pass while the two depths were swapped between the calls.
 *
 * The bodies contain no nested `});`, which is what makes the naive scan safe;
 * a future nested closure inside one of these calls would need a real matcher.
 */
function tslJoinCallArgs(flatSrc: string): string[] {
  const needle = 'tslLineJoin({';
  const out: string[] = [];
  for (let i = flatSrc.indexOf(needle); i !== -1;) {
    const end = flatSrc.indexOf('});', i);
    if (end === -1) break;
    out.push(flatSrc.slice(i + needle.length, end));
    i = flatSrc.indexOf(needle, end);
  }
  return out;
}

describe('line join near-plane guard is two-sided (#1346)', () => {
  it('NON-VACUITY: the near-plane conjunction is the ONLY thing that can reject this joint', () => {
    const { a, b } = buildJoint(NEAR_PLANE_B_FAR);

    // Both shared-vertex corners clear the rendered-width gate.
    expect(a.clampedPixelWidth).toBeGreaterThan(LINE_JOIN_MIN_HALF_WIDTH);
    expect(b.clampedPixelWidth).toBeGreaterThan(LINE_JOIN_MIN_HALF_WIDTH);
    expect(a.clampedPixelWidth).toBeCloseTo(6.6511, 3);
    expect(b.clampedPixelWidth).toBeCloseTo(6.6511, 3);
    // ...and it really is width * uPerspectiveLineScale / depth at depth 1.
    expect(a.clampedPixelWidth).toBeCloseTo(0.06 * PERSPECTIVE_LINE_SCALE, 6);

    // Both endpoints reach their source vertex: A is entirely in front, and B's
    // near-plane clip lands on its FAR end (tB = 0.625), leaving tA = 0.
    expect(a.reachesVertex).toBe(true);
    expect(b.reachesVertex).toBe(true);

    // Both codes name the OTHER slot, never themselves.
    expect(a.jointCode).toBe(2);
    expect(b.jointCode).toBe(-3);
    expect(a.jointCode - 1).toBe(b.selfSlot);
    expect(-b.jointCode - 3).toBe(a.selfSlot);

    // And with the guard forced open (partner-only, which passes on B's side),
    // the miter limit and the overshoot guard both PASS — on both sides, with
    // identical `turn`, `grow` and axial reach. NOT because of the canonical
    // operand order (`dot` is commutative, so the ordering buys nothing for
    // `turn`): see the CAVEAT in the scenario module — the shared vertex sits at
    // the NDC origin, which makes the wGuard clamp on B's behind-plane endpoint a
    // pure radial scale about `sharedPx`. So nothing downstream of the near-plane
    // test can be what rejects this joint.
    const relaxed = {
      a: joinGate({ ...a, partnerFarDepth: NEAR_PLANE_CAMERA.nearCull }, 'partner-only'),
      b: joinGate(b, 'partner-only'),
    };
    for (const [name, d] of Object.entries(relaxed)) {
      expect(d.kind, `side ${name} must reach the miter`).toBe('miter');
      expect(d.turn!).toBeCloseTo(0.645942, 5);
      expect(d.grow!).toBeCloseTo(1.10232, 5);
      expect(d.grow!).toBeLessThanOrEqual(2);
      expect(d.axialReach!).toBeCloseTo(3.0848, 3);
      expect(d.axialReach!).toBeLessThanOrEqual(d.overshootBound!);
      expect(d.overshootBound!).toBeCloseTo(9.2952, 3);
    }
  });

  it('one-sided guard DISAGREES across the joint: A falls back while B mitres alone', () => {
    const { a, b } = buildJoint(NEAR_PLANE_B_FAR);
    const dA = joinGate(a, 'partner-only');
    const dB = joinGate(b, 'partner-only');

    // The issue's exact failure. A looks at B's far endpoint (depth 0.2 < 0.5)
    // and bails; B looks at A's (depth 1.0) and mitres.
    expect(dA.bailedAt).toBe('near-plane');
    expect(dA.kind).toBe('perpendicular');
    expect(dB.kind).toBe('miter');
    expect(dA.kind).not.toBe(dB.kind);

    // B rotates its start edge onto a miter line A never builds: B's corner
    // moves ~2.9 px away from where its own perpendicular put it, and A's stays
    // put — the asymmetric flap.
    const bPerp = perp(b.lineDir, b.clampedPixelWidth);
    expect(Math.hypot(dB.offset[0] - bPerp[0], dB.offset[1] - bPerp[1])).toBeGreaterThan(2.5);
    expect(dA.offset).toEqual(perp(a.lineDir, a.clampedPixelWidth));

    // The two sides also disagree about the CAP: A returns the "no partner
    // reached" sentinel and defers to the joint code, while B commits an
    // explicit value. (Here both resolve to 1.0 — a slot-bearing code implies
    // full suppression and B's completed miter also sets 1.0 — so the visible
    // divergence in THIS scenario is purely geometric. The cap values
    // themselves diverge whenever B instead bails at the overshoot test, where
    // it would commit clamp(turn, 0, 1) = 0.646 against A's 1.0.)
    expect(dA.capSentinel).toBe(-1);
    expect(dB.capSentinel).toBe(1);
    expect(dA.capSentinel).not.toBe(dB.capSentinel);
  });

  it('two-sided guard AGREES: both sides fall back, both keep the code-implied cap', () => {
    const { a, b } = buildJoint(NEAR_PLANE_B_FAR);
    const dA = joinGate(a, 'both');
    const dB = joinGate(b, 'both');

    expect(dA.bailedAt).toBe('near-plane');
    expect(dB.bailedAt).toBe('near-plane');
    expect(dA.kind).toBe(dB.kind);
    expect(dA.kind).toBe('perpendicular');

    // Each quad keeps the plain perpendicular expansion...
    expect(dA.offset).toEqual(perp(a.lineDir, a.clampedPixelWidth));
    expect(dB.offset).toEqual(perp(b.lineDir, b.clampedPixelWidth));
    // ...and the cap SENTINEL on both sides, so the joint keeps its
    // code-implied cap and the two quads are identical to what a
    // join-style-`none` render produces. That identity is exactly what the
    // `line-join-nearplane-*` parity fixture pair measures on the GPU.
    expect(dA.capSentinel).toBe(-1);
    expect(dB.capSentinel).toBe(-1);
  });

  it("CONTROL: B's far endpoint in front → both sides mitre, and to the SAME corner", () => {
    const { a, b } = buildJoint(NEAR_PLANE_CONTROL_B_FAR);
    const dA = joinGate(a, 'both');
    const dB = joinGate(b, 'both');

    // The fix does not disable the join in general.
    expect(dA.kind).toBe('miter');
    expect(dB.kind).toBe('miter');

    // The strongest thing the mirror can honestly assert about tiling: both
    // sides compute the SAME miter point. `perpIn + perpOut` is symmetric under
    // swapping the two directions, so the canonical operand order is not what
    // buys this either — the two sides see the same UNORDERED direction pair
    // (both endpoints in front here, so no wGuard clamp distorts one side's view
    // of the other) and the shared vertex gives them the same half-width. The
    // two quads therefore meet edge-to-edge at sharedPx ± M, no wedge, no
    // overlap.
    expect(dA.offset[0]).toBeCloseTo(dB.offset[0], 9);
    expect(dA.offset[1]).toBeCloseTo(dB.offset[1], 9);
    expect(dA.turn!).toBeCloseTo(dB.turn!, 9);
    expect(dA.capSentinel).toBe(dB.capSentinel);
    // ...and it is a real rotation, not the collinear no-op R*perp.
    expect(Math.hypot(dA.offset[0], dA.offset[1])).toBeCloseTo(a.clampedPixelWidth * dA.grow!, 4);
    expect(dA.grow!).toBeGreaterThan(1.05);
  });

  it('shared vertex itself behind the near-cull plane: both sides fall back via reachesVertex', () => {
    // Put the SHARED vertex behind nearCull (depth 0.2) with both outer ends in
    // front. Now each segment is near-clipped AT the joint, so neither endpoint
    // reaches its source vertex — a symmetry that predates #1346 and must stay.
    const sharedBehind: Vec3 = [0, 0, 0.8];
    expect(viewDepth(sharedBehind)).toBeCloseTo(0.2, 9);
    const { a, b } = buildJoint(NEAR_PLANE_CONTROL_B_FAR, sharedBehind);

    // Both segments' clip ranges really do move the shared endpoint (A's tB
    // drops to 0.625, B's tA rises to 0.375) — the fallback below is not an
    // accident of the harness.
    expect(a.reachesVertex).toBe(false);
    expect(b.reachesVertex).toBe(false);
    expect(nearPlaneClipRange(viewDepth(NEAR_PLANE_A_START), 0.2).tB).toBeCloseTo(0.625, 6);
    expect(nearPlaneClipRange(0.2, viewDepth(NEAR_PLANE_CONTROL_B_FAR)).tA).toBeCloseTo(0.375, 6);
    // Both FAR endpoints are in front here, so the near-plane conjunction alone
    // would have let this joint through — `reachesVertex` is what stops it.
    expect(a.thisFarDepth).toBeGreaterThanOrEqual(NEAR_PLANE_CAMERA.nearCull);
    expect(b.thisFarDepth).toBeGreaterThanOrEqual(NEAR_PLANE_CAMERA.nearCull);

    for (const guard of ['partner-only', 'both'] as const) {
      const dA = joinGate(a, guard);
      const dB = joinGate(b, guard);
      expect(dA.bailedAt, `guard ${guard}`).toBe('reaches-vertex');
      expect(dB.bailedAt, `guard ${guard}`).toBe('reaches-vertex');
      expect(dA.kind).toBe(dB.kind);
      expect(dA.capSentinel).toBe(dB.capSentinel);
    }
  });

  // --- SOURCE LOCKS ---------------------------------------------------------
  // These are the assertions that fail without the shader change: the mirror
  // above can only be trusted if the real shaders take the same two operands.
  //
  // The TSL twin needs a lock too, and specifically a lock on the MAPPING.
  // `thisFarDepth` being a REQUIRED field on `TSLLineJoinArgs` only forbids
  // OMISSION — a required field says nothing about the value supplied, so
  // swapping `thisFarDepth: endDepth` and `thisFarDepth: startDepth` between the
  // two calls typechecks cleanly and passes every mirror test above (the mirror
  // builds its own sides and never reads the factory). Only the source text
  // distinguishes them, because the argument object is consumed at graph-build
  // time inside a browser-only traced body.

  it("SOURCE LOCK: luxarLineJoin declares thisFarDepth, and its width gate is the imported constant's value", () => {
    const src = flat(GLSL_LINE_JOIN);
    // Non-greedy gaps, not `[^)]*`: the parameter list's own doc comments
    // contain parentheses ("(t == 1)", "(ndcStart / ndcEnd)").
    expect(src).toMatch(
      /vec3 luxarLineJoin\( bool atEnd,.*?bool reachesVertex,.*?float thisFarDepth, .*?float jointCode,/
    );
    // GLSL cannot import, so the block keeps its own `joinMinHalfWidth` literal
    // and the TS mirror above uses `LINE_JOIN_MIN_HALF_WIDTH`. Nothing in the
    // type system ties the two — this assertion is what does, so the pattern is
    // BUILT from the constant rather than hardcoding 2.0. Change the constant
    // without changing the shader (or vice versa) and this fails.
    const gateLiteral = LINE_JOIN_MIN_HALF_WIDTH.toFixed(1).replace('.', '\\.');
    expect(src, 'GLSL joinMinHalfWidth must equal LINE_JOIN_MIN_HALF_WIDTH').toMatch(
      new RegExp(`float joinMinHalfWidth = ${gateLiteral};`)
    );
  });

  it("SOURCE LOCK: the guard ANDs this side's own far-endpoint test with the partner's", () => {
    const src = flat(GLSL_LINE_JOIN);
    expect(src).toMatch(
      /bool thisFarInFront = \(uIsOrtho == 1\) \|\| \(thisFarDepth >= nearCull\);/
    );
    expect(src).toMatch(/bool partnerInFront = \(uIsOrtho == 1\) \|\| \(farPx\.z >= nearCull\);/);
    // Anchored through the NEXT clause: an unanchored prefix would also match an
    // inverted guard such as `if (!(thisFarInFront && partnerInFront) == false
    // || ...)`, which rejects exactly the joints it should admit.
    expect(src).toMatch(
      /if \(!\(thisFarInFront && partnerInFront\) \|\| partnerLen <= 0\.0001 \|\| pixelLen <= 0\.0001\)/
    );
  });

  it('SOURCE LOCK: both GLSL stages pass the OPPOSITE endpoint pre-clip depth per end', () => {
    for (const [name, src] of [
      ['visual', LINE_VERTEX_SHADER],
      ['pick', LINE_PICK_VERTEX_SHADER],
    ] as const) {
      const s = flat(src);
      // The start call (atEnd = false) passes endDepth; the end call passes
      // startDepth. Both are the PRE-clip locals — the stages compute them
      // before the near-plane clip rewrites mvStart/mvEnd.
      expect(s, `${name}: start call must pass endDepth`).toMatch(
        /luxarLineJoin\( false, tA <= 0\.0, endDepth, aStartJointCode,/
      );
      expect(s, `${name}: end call must pass startDepth`).toMatch(
        /luxarLineJoin\( true, tB >= 1\.0, startDepth, aEndJointCode,/
      );
      // ORDERING IS THE LOAD-BEARING PART, and it is the one refactor that
      // silently re-breaks #1346: sliding either declaration BELOW the clip that
      // rewrites mvStart/mvEnd turns `thisFarDepth` into the CLIPPED depth,
      // which for a near-clipped endpoint is exactly `nearCull` — so
      // `thisFarDepth >= nearCull` passes, the guard reverts to one-sided in
      // effect, and B mitres alone again. Nothing runnable can see that: the
      // text locks above still match, the mirror builds its own sides, and the
      // GPU fixture's `miter == none` silently becomes `miter != none` only on a
      // backend nobody snapshots. So pin BOTH declarations before the rewrite.
      const startDeclIdx = s.indexOf('float startDepth = -mvStart.z;');
      const endDeclIdx = s.indexOf('float endDepth = -mvEnd.z;');
      const clipIdx = s.indexOf('mvStart = mvStartClipped;');
      expect(startDeclIdx, `${name}: startDepth declaration not found`).toBeGreaterThan(-1);
      expect(endDeclIdx, `${name}: endDepth declaration not found`).toBeGreaterThan(-1);
      expect(clipIdx, `${name}: segment-clip rewrite not found`).toBeGreaterThan(-1);
      expect(
        clipIdx,
        `${name}: startDepth must be computed BEFORE the near-plane clip rewrites mvStart`
      ).toBeGreaterThan(startDeclIdx);
      expect(
        clipIdx,
        `${name}: endDepth must be computed BEFORE the near-plane clip rewrites mvEnd`
      ).toBeGreaterThan(endDeclIdx);
    }
  });

  it('SOURCE LOCK: both TSL stages map thisFarDepth to the OPPOSITE end', () => {
    for (const [name, filePath] of TSL_STAGE_SOURCES) {
      const calls = tslJoinCallArgs(flat(readFileSync(filePath, 'utf8')));
      expect(calls, `${name}: expected exactly two tslLineJoin call sites`).toHaveLength(2);

      const startCall = calls.find((c) => /\batEnd: false\b/.test(c));
      const endCall = calls.find((c) => /\batEnd: true\b/.test(c));
      expect(startCall, `${name}: no atEnd:false call`).toBeDefined();
      expect(endCall, `${name}: no atEnd:true call`).toBeDefined();

      // The crossover is the whole point: the START call needs the depth of the
      // segment's OTHER (end) endpoint and vice versa. Matched per call object
      // rather than with one flat regex over the file, so a reordered field list
      // still locks the pairing.
      expect(startCall, `${name}: atEnd:false must pass thisFarDepth: endDepth`).toMatch(
        /\bthisFarDepth: endDepth\b/
      );
      expect(endCall, `${name}: atEnd:true must pass thisFarDepth: startDepth`).toMatch(
        /\bthisFarDepth: startDepth\b/
      );
      // And they are the PRE-clip `.toVar()`s. The TEXT is only half of it —
      // ORDERING is the load-bearing part, and it is the one refactor that
      // silently re-breaks #1346: `mvStart`/`mvEnd` are `.toVar()`s that the
      // clip block REASSIGNS, so moving either declaration below
      // `mvStart.assign(mvStartClipped)` makes `thisFarDepth` the CLIPPED depth,
      // which for a near-clipped endpoint is exactly `nearCull` — the guard then
      // passes, reverts to one-sided in effect, and B mitres alone again with
      // every runnable assertion still green. Pin both against the reassignment.
      const src = flat(readFileSync(filePath, 'utf8'));
      const startDeclIdx = src.indexOf('const startDepth: TSLNode = mvStart.z.negate().toVar();');
      const endDeclIdx = src.indexOf('const endDepth: TSLNode = mvEnd.z.negate().toVar();');
      const clipIdx = src.indexOf('mvStart.assign(mvStartClipped);');
      expect(startDeclIdx, `${name}: startDepth must come from the raw mvStart`).toBeGreaterThan(
        -1
      );
      expect(endDeclIdx, `${name}: endDepth must come from the raw mvEnd`).toBeGreaterThan(-1);
      expect(clipIdx, `${name}: segment-clip reassignment not found`).toBeGreaterThan(-1);
      expect(
        clipIdx,
        `${name}: startDepth must be declared BEFORE the clip reassigns mvStart`
      ).toBeGreaterThan(startDeclIdx);
      expect(
        clipIdx,
        `${name}: endDepth must be declared BEFORE the clip reassigns mvEnd`
      ).toBeGreaterThan(endDeclIdx);
    }
  });
});
