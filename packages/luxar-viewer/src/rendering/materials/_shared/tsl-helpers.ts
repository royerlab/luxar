/**
 * Shared TSL helper functions used by the geometry-material factories.
 *
 * `sanitizeNonNegative` reproduces the GLSL function of the same name —
 * guard against NaN/Inf produced by upstream data loaders and fall back to a
 * sensible scalar default. It is a pure TSL builder call, so the same body
 * works under any NodeBuilder (WebGL2 or WebGPU). (`glsl-lib.ts` also defines
 * a `sanitizePositive`; no TSL shader calls it, so there is no TSL mirror.)
 *
 * `proxyIUniform` bridges the THREE `IUniform`-shaped public API
 * (`material.uniforms.uX.value = Y`) directly onto a TSL
 * `UniformNode.value` getter/setter. It replaces the old
 * `uniform(value).onUpdate(() => iuniform.value, 'render')` bridge,
 * which read the host IUniform's value into the node before every
 * render. That callback was structural noise — the IUniform record
 * could *be* the node, just dressed in the IUniform shape. The proxy
 * forwards reads and writes directly to the node so there is no
 * per-frame callback overhead and no two-source-of-truth class.
 *
 * @module rendering/tsl-helpers
 */

import type { IUniform } from 'three';

// Single source of truth for the join's width gate — see glsl-lib.ts. Imported
// rather than redeclared so the GLSL and TSL backends cannot drift apart.
import { LINE_JOIN_MIN_HALF_WIDTH } from './glsl-lib';

export { LINE_JOIN_MIN_HALF_WIDTH };
import {
  If,
  attribute,
  cameraProjectionMatrix,
  clamp,
  dot,
  float,
  int,
  uint,
  ivec2,
  length,
  max,
  min,
  modelViewMatrix,
  smoothstep,
  sqrt,
  vec2,
  vec3,
  vec4,
} from 'three/tsl';

/**
 * Loosely-typed TSL node alias. TSL's typed overloads return many
 * mutually-incompatible inner constructor types; relaxing at helper
 * boundaries lets the runtime TSL builder do the real type checking
 * when it compiles to GLSL/WGSL.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type TSLNode = any;

/**
 * Double-buffered draw-slot → storage-slot index — the TSL twin of
 * `GLSL_SORTED_INDEX` (depth-sorting spec §2.1 tier 3).
 *
 * A permutation must swap ATOMICALLY: a half-applied ordering is not a
 * reordering but a corrupt permutation (elements drawn twice / not at
 * all). Each new ordering therefore streams into the INACTIVE attribute
 * across frames, and `uSortedIndexSlot` flips only once that buffer holds
 * the whole permutation.
 *
 * BOTH attributes are referenced unconditionally, which is required, not
 * merely tidy: the WebGPU backend only uploads graph-referenced
 * attributes, so an unreferenced back buffer would never receive its
 * chunked uploads — and `RenderObject` dereferences a referenced
 * attribute before its undefined guard, so every geometry must carry
 * both. `attachElementStorage` allocates them as two DISTINCT buffers
 * together — never aliased, never added later, since the WebGPU vertex
 * layout is cached from the attribute set at first draw and a set that
 * grows afterwards renders the scene black.
 *
 * `uSortedIndexSlot` must stay a RUNTIME uniform: a compile-time flag
 * would rebuild the graph on every swap. (The lines material treats
 * `uIsOrtho` as compile-time — deliberately NOT the pattern here.)
 *
 * BRANCHLESS on purpose. `.select()` emits an if/else STATEMENT, and the
 * pick factories consume this index at `varying(float(...))` — evaluated
 * OUTSIDE their `Fn()` body, where a statement cannot legally land. That
 * produced a shader which built and code-generated fine but rendered
 * zero pixels (the same vacuous-black failure mode the gsplat harness
 * comment in `tests/e2e/harnesses/tsl-harness/gsplats.ts` warns about).
 * A pure arithmetic mix is position-independent, so the one helper is
 * safe both inside and outside an `Fn()`. The multiplier is a uniform,
 * so there is no per-vertex divergence to save by branching anyway.
 *
 * Returns an INT node (not uint): the downstream `int(...)` wraps and
 * `float(...)` casts accept it unchanged.
 */
export function sortedIndexNode(uSortedIndexSlot: TSLNode): TSLNode {
  const slot: TSLNode = int(uSortedIndexSlot);
  const a: TSLNode = int(attribute<'uint'>('aSortedIndex', 'uint'));
  const b: TSLNode = int(attribute<'uint'>('aSortedIndexB', 'uint'));
  return a.mul(int(1).sub(slot)).add(b.mul(slot));
}

/**
 * Projected-density thinning predicate. Mirrors GLSL `luxarDensityDropped()`
 * (glsl-lib.ts) bit for bit: the storage index is run through the lowbias32
 * integer hash, mapped to [0, 1) and compared against `uDensityDrop`, the
 * fraction of the node's elements to drop. A pure arithmetic expression, so it
 * is safe both inside and outside an `Fn()` body (see the note on
 * `sortedIndexNode`). Zero drops nothing.
 */
export function densityDroppedNode(uDensityDrop: TSLNode, sortedIndex: TSLNode): TSLNode {
  let h: TSLNode = uint(sortedIndex);
  h = h.bitXor(h.shiftRight(uint(16)));
  h = h.mul(uint(0x7feb352d));
  h = h.bitXor(h.shiftRight(uint(15)));
  h = h.mul(uint(0x846ca68b));
  h = h.bitXor(h.shiftRight(uint(16)));
  const unit: TSLNode = float(h).mul(1.0 / 4294967296.0);
  return float(uDensityDrop)
    .greaterThan(0.0)
    .and(unit.lessThan(float(uDensityDrop)));
}

/** Sanitise a non-negative scalar. Mirrors GLSL `sanitizeNonNegative`. */
export function sanitizeNonNegative(value: TSLNode, fallback: TSLNode): TSLNode {
  const isFinite = value.lessThan(1e30).and(value.greaterThan(-1e30));
  const isNonNeg = value.greaterThanEqual(0.0);
  return isFinite.and(isNonNeg).select(value, fallback);
}

/**
 * Per-element opacity sanitizer. Mirrors GLSL `sanitizeAlpha`: NaN/Inf
 * route to the 1.0 opaque identity (corruption stays LOUD), finite
 * values clamp to [0, 1] — alpha is opacity, never HDR (Python pins the
 * range at write; this guards hand-crafted zarr). The clamp keeps the
 * zero boundary CONTINUOUS (a -1e-4 epsilon vanishes like +0.0 renders,
 * instead of jumping to full opacity) and keeps the value
 * mediump-varying-safe.
 */
export function sanitizeAlpha(value: TSLNode): TSLNode {
  const isFinite = value.lessThan(1e30).and(value.greaterThan(-1e30));
  return isFinite.select(value.max(0.0).min(1.0), float(1.0));
}

/**
 * Boolean TSL node: true when `value` is NaN or +/-Inf. Mirrors GLSL
 * `isInvalidFloat`. TSL has no direct `isnan`/`isinf` exposed across
 * backends, so we approximate via the same finite-range test
 * `sanitizeNonNegative` uses: any value outside (-1e30, 1e30) is
 * treated as non-finite. The same pattern is used by the visual
 * point/shader-tsl.ts sharpness-compensation guard.
 */
export function invalidFloatTSL(value: TSLNode): TSLNode {
  return value.lessThan(1e30).and(value.greaterThan(-1e30)).not();
}

/**
 * Wrap a TSL `UniformNode` in an `IUniform`-shaped getter/setter so
 * callers can keep using `material.uniforms.uX.value = Y` while the
 * read/write is routed straight to `node.value`. See the module
 * preamble for why this replaces the older `.onUpdate('render')`
 * bridge.
 *
 * The node argument is typed loosely as `TSLNode` because TSL's
 * typed-overload surface returns many concrete inner classes; the
 * runtime contract is just that `node.value` is a readable & writable
 * property of type `T`.
 */
export function proxyIUniform<T>(node: TSLNode): IUniform<T> {
  return {
    get value(): T {
      return node.value as T;
    },
    set value(v: T) {
      node.value = v;
    },
  } as IUniform<T>;
}

/**
 * TSL twins of GLSL `luxarIsOrthoProjection` / `luxarProjectionSizeScale`
 * (glsl-lib.ts): read from `cameraProjectionMatrix`, which three sets for the
 * camera being drawn with. `element(c)` is column c, so P[3][3] is
 * `element(3).w` and P[1][1] is `element(1).y`.
 */
export function isOrthoProjectionTSL(): TSLNode {
  const P: TSLNode = cameraProjectionMatrix;
  return P.element(int(3)).w.greaterThan(0.5).select(int(1), int(0));
}

/** |P11|: see {@link isOrthoProjectionTSL}. */
export function projectionSizeScaleTSL(): TSLNode {
  const P: TSLNode = cameraProjectionMatrix;
  return P.element(int(1)).y.abs();
}

/**
 * Unified perspective near-plane fade (runtime-uniform variant, for
 * the point/gsplat graphs whose ortho flag is the `uIsOrtho` uniform).
 * Mirrors GLSL `perspectiveNearFade` in glsl-lib.ts: perspective =
 * 0.0 behind the camera, smoothstep across [nearCull, 2*nearCull];
 * ortho = 1.0 always (NDC clipping is the sole cull authority).
 * Callers reject the vertex when the result < 0.01 and multiply the
 * surviving amplitude/alpha/brightness by it.
 */
export function perspectiveNearFadeTSL(
  uIsOrtho: TSLNode,
  viewZ: TSLNode,
  uNearCull: TSLNode
): TSLNode {
  const fade = smoothstep(uNearCull, uNearCull.mul(2.0), viewZ.negate());
  const persp = viewZ.greaterThanEqual(0.0).select(float(0.0), fade);
  return int(uIsOrtho).equal(int(1)).select(float(1.0), persp);
}

/**
 * Compile-time-ortho variant for the line graphs (their ortho flag is
 * the factory `config.isOrtho`, baked into the graph): ortho variants
 * carry NO fade/cull code at all.
 */
export function perspectiveNearFadeStaticTSL(
  isOrtho: boolean,
  viewZ: TSLNode,
  uNearCull: TSLNode
): TSLNode {
  if (isOrtho) return float(1.0);
  return viewZ
    .greaterThanEqual(0.0)
    .select(float(0.0), smoothstep(uNearCull, uNearCull.mul(2.0), viewZ.negate()));
}

/**
 * TSL counterpart of `GLSL_LINE_JOINT_CODE`'s
 * `luxarLineJointCapSuppression` — the endpoint cap multiplier implied by a
 * per-endpoint joint code (`texel4.yz`; see that GLSL block for the encoding
 * and for why a slot-bearing code must suppress rather than keep the cap).
 *
 * Emitted as a node expression rather than a TSL `Fn()` so it composes inside
 * the line factories' single traced vertex body, where a structural branch is
 * deliberately avoided.
 */
export function tslLineJointCapSuppression(jointCode: TSLNode): TSLNode {
  const isFreeEnd = jointCode.greaterThan(-0.5).and(jointCode.lessThan(0.5));
  const isHub = jointCode.lessThan(-1.5).and(jointCode.greaterThan(-2.5));
  return isFreeEnd.or(isHub).select(float(0.0), float(1.0));
}

/**
 * Rendered half-width AT ONE ENDPOINT, in pixels — TSL twin of GLSL's
 * `luxarLineEndPixelWidth`.
 *
 * The rendered width is otherwise PER-VERTEX (it interpolates width and view
 * depth at the vertex's own t), and both cap varyings are `flat`, whose value
 * comes from one provoking vertex. Gating the join on a per-vertex width
 * therefore lets a segment whose ends straddle the gate resolve its cap from
 * whichever corner provokes — and WebGL provokes from the last vertex while
 * WGSL provokes from the first. Evaluated at an END this is segment-constant,
 * and equals the per-vertex value exactly at the corner that consumes it, so
 * the geometry is unchanged.
 *
 * `isOrtho` is the build-time graph variant, so only one branch is emitted.
 * `lineScale` is resY·|P11| (GLSL `luxarLineScale`), the same scale in both
 * projections.
 */
export function tslLineEndPixelWidth(
  isOrtho: boolean,
  widthAtEnd: TSLNode,
  viewZ: TSLNode,
  nearCull: TSLNode,
  lineScale: TSLNode
): TSLNode {
  return isOrtho
    ? widthAtEnd.mul(lineScale)
    : widthAtEnd.mul(lineScale).div(max(viewZ.negate(), nearCull));
}

/** Everything `tslLineJoin` needs from its calling vertex stage. */
export interface TSLLineJoinArgs {
  /** Build-time camera mode — the line graphs' `config.isOrtho`. */
  readonly isOrtho: boolean;
  /** The RGBA32F line texture node (6 texels/segment). */
  readonly uLineTex: TSLNode;
  /** Its width in texels — already `.toVar()`ed by the caller's prologue. */
  readonly lineTexW: TSLNode;
  /** Viewport resolution uniform node. */
  readonly uResolution: TSLNode;
  /** Physical render-target pixels per CSS pixel. */
  readonly uPixelRatio: TSLNode;
  /** Scene-relative near-cull distance (already floored). */
  readonly nearCull: TSLNode;
  /** This segment's own STORAGE slot, as an int node (self-reference guard). */
  readonly selfSlot: TSLNode;
  /**
   * Which end of the segment this call is for — a plain JS boolean, not a
   * node: the caller evaluates BOTH ends unconditionally (see below), so the
   * direction ternaries fold at graph-build time instead of emitting a
   * `select()` that would materialise both arms per vertex.
   */
  readonly atEnd: boolean;
  /** False when near-clipping moved this endpoint off its source vertex. */
  readonly reachesVertex: TSLNode;
  /** texel4.y at the start vertex, texel4.z at the end one. */
  readonly jointCode: TSLNode;
  /** NDC of the shared vertex (ndcStart / ndcEnd). */
  readonly sharedNdc: TSLNode;
  /** This segment's unit direction in pixel space. */
  readonly lineDir: TSLNode;
  readonly pixelLen: TSLNode;
  /**
   * THIS END's clamped pixel half-width — segment-constant, NOT the calling
   * vertex's. Both width-dependent decisions below (the 2 px gate and the
   * axial-overshoot guard) feed a `flat` cap varying, so a per-vertex width
   * would let the t=0 and t=1 corners of a tapered or foreshortened segment
   * answer differently and hand the cap to the provoking vertex alone (WGSL
   * provokes first, WebGL last — so the backends would disagree too).
   */
  readonly joinPixelWidth: TSLNode;
  /**
   * View-space depth (-mvZ) of THIS segment's far endpoint, i.e. the one that
   * is NOT the shared vertex, taken from the UNCLIPPED endpoint depths.
   */
  readonly selfFarDepth: TSLNode;
  /**
   * OUT — a vec2 `.toVar()` the caller pre-set to `perpendicular *
   * joinPixelWidth`. Overwritten with the miter point only where the join
   * actually applies, so every skipped path keeps the shipped expansion.
   */
  readonly cornerOffset: TSLNode;
  /**
   * OUT — a float `.toVar()` the caller pre-set to -1.0 ("no partner reached,
   * keep the code-implied cap"). Set to the screen-space suppression wherever
   * a partner IS reached.
   */
  readonly capValue: TSLNode;
}

/**
 * Screen-space miter join at degree-2 polyline joints (#790) — TSL counterpart
 * of `GLSL_LINE_JOIN`'s `luxarLineJoin`, shared by the visual and picking line
 * factories so a pick footprint keeps matching the visible one.
 *
 * Two segments meeting at a turn of angle theta leave an uncovered circular
 * sector of that angle on the OUTSIDE of the bend and double-cover a lens on
 * the inside. No per-endpoint intensity scalar can close the outer wedge —
 * nothing rasterises there to shade — so this rotates the quad's end edge onto
 * the shared miter edge and the two quads TILE.
 *
 * That tiling is exact in exact arithmetic, not bit-exact in float32. The two
 * sides of a joint evaluate algebraically identical operands in a canonical
 * order, but they reach pixel space differently — the vertex stage scales the
 * NDC difference `ndcEnd.sub(ndcStart)` once, while this helper scales each
 * endpoint (`sharedPx`, `farPx`) and subtracts afterwards — so their miter
 * points agree only to float32 rounding, order 1e-5 px on the
 * `test_line_joins` fixture, which is at most one seam pixel once the
 * rasteriser quantises. `tests/e2e/line-join-artifact.spec.ts` quantifies it
 * and gates the bend bands accordingly. Subtracting in NDC and scaling
 * afterwards on both paths — here and in `GLSL_LINE_JOIN` — would make the two
 * sides bit-exact; that is a known, deliberately deferred change.
 *
 * STRUCTURAL DIFFERENCE FROM THE GLSL TWIN, and it is deliberate: the join
 * STYLE is a build-time graph variant here (the caller simply does not call
 * this when the style is `none`), exactly as the line factories already treat
 * `config.isOrtho`, whereas GLSL keeps `uLineJoin` a runtime uniform so a
 * `?lineJoin=` override never recompiles a program. The parity harness compares
 * pixels, not mechanisms.
 *
 * Emits real `If` blocks rather than `select()`: `select()` evaluates both
 * arms, which would pay the partner texel fetch and projection on every vertex
 * of every thin line and throw away the width gate's entire point.
 */
export function tslLineJoin(args: TSLLineJoinArgs): void {
  const {
    isOrtho,
    uLineTex,
    lineTexW,
    uResolution,
    uPixelRatio,
    nearCull,
    selfSlot,
    atEnd,
    reachesVertex,
    jointCode,
    sharedNdc,
    lineDir,
    pixelLen,
    joinPixelWidth,
    selfFarDepth,
    cornerOffset,
    capValue,
  } = args;

  // Sentinels: 0 free end, -1 slice-clipped, -2 degree->=3 hub. Only a
  // slot-bearing code names a partner; the sign carries which of the partner's
  // endpoints is the shared one (+(slot + 1) its START, -(slot + 3) its END).
  //
  // The self-reference is rejected rather than assumed away: the texel writer
  // already clamps a code naming an unwritten slot and the kernel never emits
  // one, but that guarantee comes from a DIFFERENT module, and a texel writer
  // is not the only producer (the TSL parity harness writes codes by hand). A
  // quad mitered against itself has no partner to tile with, so its rotated
  // end edge rasterises as a flap sticking out of the tube — the exact failure
  // this change exists to remove.
  const partnerSharesItsStart: TSLNode = jointCode.greaterThan(0.0);
  const partnerSlot: TSLNode = partnerSharesItsStart
    .select(int(jointCode).sub(int(1)), int(jointCode.negate()).sub(int(3)))
    .toVar();
  const namesAPartner: TSLNode = jointCode
    .greaterThan(0.5)
    .or(jointCode.lessThan(-2.5))
    .and(partnerSlot.notEqual(selfSlot));

  const gate: TSLNode = joinPixelWidth
    .greaterThan(uPixelRatio.max(float(1.0)).mul(LINE_JOIN_MIN_HALF_WIDTH))
    .and(reachesVertex)
    .and(namesAPartner);

  If(gate, () => {
    // Only the partner's FAR endpoint is needed — the near one is this vertex,
    // already projected. One texel fetch, one projection.
    const partnerBase: TSLNode = partnerSlot.mul(int(6)).toVar();
    const pTexelX: TSLNode = partnerBase.mod(lineTexW).toVar();
    const pTexelY: TSLNode = partnerBase.div(lineTexW).toVar();
    const partnerFar: TSLNode = vec3(
      partnerSharesItsStart.select(
        uLineTex.load(ivec2(pTexelX.add(int(1)), pTexelY)),
        uLineTex.load(ivec2(pTexelX, pTexelY))
      )
    ).toVar();

    // Project into the same pixel space the quad expansion works in, mirroring
    // the main path's wGuard (ortho: w == 1 exactly, the guard is inert).
    const mvFar: TSLNode = modelViewMatrix.mul(vec4(partnerFar, 1.0)).toVar();
    const clipFar: TSLNode = cameraProjectionMatrix.mul(mvFar).toVar();
    const wGuardFar: TSLNode = isOrtho ? float(1.0) : nearCull;
    const farPx: TSLNode = vec2(
      clipFar.xy.div(max(clipFar.w, wGuardFar)).mul(uResolution.mul(0.5))
    ).toVar();
    const sharedPx: TSLNode = vec2(sharedNdc.mul(uResolution.mul(0.5))).toVar();

    // Orient the partner leg on THIS segment's traversal sense — on `atEnd`,
    // NOT on which of the partner's endpoints is the shared one. With
    // dirIn = atEnd ? lineDir : partnerDir (and dirOut its mirror) the chain
    // runs through the joint the way THIS segment traverses it, so the partner
    // leg must point AWAY from the shared vertex when this segment ARRIVES
    // there and INTO it when this segment LEAVES it. Orienting on
    // partnerSharesItsStart is only right for end->start / start->end chains:
    // at an END-END or START-START joint it negates the direction this joint
    // needs, so a collinear joint reads turn = -1, the miter limit rejects it,
    // and clamp(turn, 0, 1) keeps the soft cap (the #780 dimming). `atEnd` is a
    // build-time JS boolean, so this ternary folds at graph-build time;
    // partnerSharesItsStart stays a runtime `.select()` above because it is
    // decoded per-endpoint from the joint code and only chooses the TEXEL.
    const partnerDelta: TSLNode = vec2(atEnd ? farPx.sub(sharedPx) : sharedPx.sub(farPx)).toVar();
    const partnerLen: TSLNode = length(partnerDelta).toVar();
    // BOTH far endpoints must clear the near plane, not just the fetched one:
    // testing only the partner's makes each side test a DIFFERENT point, so for
    // A running front->shared meeting B running shared->behind, A declines while
    // B miters alone and B's rotated edge has nothing to tile against. With the
    // conjunction A tests {A.far, B.far} and B tests {B.far, A.far} — the same
    // pair — so both take the same branch. Inert under ortho (no 1/z).
    const bothFarInFront: TSLNode = isOrtho
      ? float(1.0).greaterThan(0.0)
      : mvFar.z.negate().greaterThanEqual(nearCull).and(selfFarDepth.greaterThanEqual(nearCull));

    // A DEGENERATE partner is the one decline that must not fall back to the
    // code-implied default. The kernel matches endpoints by vertex index and
    // never looks at positions, so a zero-length interior segment still earns
    // this endpoint a slot-bearing code — which means "suppress the cap, a
    // neighbouring quad meets you". Nothing rasterises there, so the joint
    // would get neither a miter nor a cap and the #790 wedge reappears. Keep
    // the cap. Written BEFORE the join block so both backends order the
    // declines identically; the join block already requires a non-degenerate
    // partner, so the two are mutually exclusive.
    If(partnerLen.lessThanEqual(0.0001), () => {
      capValue.assign(float(0.0));
    });

    If(bothFarInFront.and(partnerLen.greaterThan(0.0001)).and(pixelLen.greaterThan(0.0001)), () => {
      // CANONICAL operand order — incoming edge first, outgoing second — so
      // both segments meeting here evaluate the same expression and take the
      // same branch. A branch disagreement leaves one diagonal edge with
      // nothing to tile against, which rasterises as a flap.
      const partnerDir: TSLNode = vec2(partnerDelta.div(partnerLen)).toVar();
      const dirIn: TSLNode = atEnd ? lineDir : partnerDir;
      const dirOut: TSLNode = atEnd ? partnerDir : lineDir;
      const turn: TSLNode = dot(dirIn, dirOut).toVar();

      // The endpoint cap, DERIVED rather than stored: the kernel's old
      // scalar was clamp(-dot(away_a, away_b), 0, 1), which with
      // away_mine = -lineDir and away_partner = +partnerDir is exactly
      // clamp(dot(lineDir, partnerDir), 0, 1) — the same dot the miter limit
      // needs anyway. Deriving it frees texel4.yz to carry the partner code,
      // and it is measured in SCREEN space, so unlike the stored data-space
      // angle it tracks the camera (#795).
      capValue.assign(clamp(turn, 0.0, 1.0));

      // Miter limit |M|/R = sqrt(2/(1 + turn)) <= 2 (theta <= 120 deg), and
      // an overshoot guard on the AXIAL reach R*tan(theta/2), NOT on |M|
      // (which is ~R always): gating on the magnitude would disable the join
      // on every polyline whose segments are shorter than twice the tube
      // radius — exactly the dense-curve case this issue is about. Both
      // tests read operands identical from either side, so the two quads
      // always agree on whether this joint is mitred.
      const grow: TSLNode = sqrt(float(2.0).div(max(turn.add(1.0), float(1e-6)))).toVar();
      const axialReach: TSLNode = joinPixelWidth
        .mul(sqrt(max(grow.mul(grow).sub(1.0), float(0.0))))
        .toVar();
      If(
        grow.lessThanEqual(2.0).and(axialReach.lessThanEqual(min(pixelLen, partnerLen).mul(0.5))),
        () => {
          // Intersection of the two segments' +R offset lines. It reduces to
          // R * perp at a collinear joint, so a straight polyline is
          // unchanged. The miter point lies ON this segment's own +R offset
          // line, which is why the resulting trapezoid keeps vPerpNorm an
          // exact perpendicular coordinate and the super-Gaussian
          // cross-section is untouched. A mitred joint tiles exactly, so
          // nothing may dim it.
          const perpIn: TSLNode = vec2(dirIn.y.negate(), dirIn.x);
          const perpOut: TSLNode = vec2(dirOut.y.negate(), dirOut.x);
          cornerOffset.assign(perpIn.add(perpOut).mul(joinPixelWidth.div(turn.add(1.0))));
          capValue.assign(float(1.0));
        }
      );
    });
  });
}
