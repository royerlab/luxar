/**
 * GSplat picking material TSL factory — NodeMaterial counterpart to
 * `GSPLAT_PICK_SOURCE` in `shaders.ts`.
 *
 * Same 3D→2D covariance projection pipeline as `gsplat.tsl`, but the
 * fragment outputs picking data:
 *   - R: nodeId (uniform)
 *   - G: elementId (instance index) LOW 16 bits
 *   - B: brightness clamped to [0, 1]
 *   - A: the same elementId's HIGH 16 bits (one f32 channel cannot
 *     carry the whole index exactly — see `luxarElementIdParts`)
 * Depth = 1.0 - brightness (brightness-as-depth tie-breaking) — or the
 * real fragment depth when `uSurfaceDepth == 1` (surface/'normal' mode:
 * front-most wins, matching the depth-sorted occluding surface).
 *
 * Picking always uses max-projection mode — no ray integration boost
 * — so the Σ_cam⁻¹ cofactor expansion is omitted vs gsplat.tsl.
 *
 * @module rendering/picking/gsplat/pick.tsl
 */

import * as THREE from 'three';
import {
  Fn,
  uniform,
  attribute,
  varying,
  texture,
  vec2 as _vec2,
  vec3 as _vec3,
  vec4 as _vec4,
  mat3 as _mat3,
  ivec2 as _ivec2,
  float,
  int,
  max,
  min,
  abs,
  sqrt,
  exp,
  clamp,
  smoothstep,
  normalize,
  Discard,
  depth,
  modelViewMatrix,
  cameraProjectionMatrix,
  screenCoordinate,
  screenSize,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { resolveElementTextureWidth, SPLAT_TEXTURE_LAYOUT } from '../../element-texture-layout';
import {
  isOrthoProjectionTSL,
  projectionSizeScaleTSL,
  perspectiveNearFadeTSL,
  invalidFloatTSL,
  type TSLNode,
  sortedIndexNode,
  densityDroppedNode,
} from '../../materials/_shared/tsl-helpers';

const vec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _vec2 as TSLNode;
const vec3: (a?: TSLNode, b?: TSLNode, c?: TSLNode) => TSLNode = _vec3 as TSLNode;
const vec4: (a?: TSLNode, b?: TSLNode, c?: TSLNode, d?: TSLNode) => TSLNode = _vec4 as TSLNode;
const mat3: (a?: TSLNode, b?: TSLNode, c?: TSLNode) => TSLNode = _mat3 as TSLNode;
const ivec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _ivec2 as TSLNode;

/**
 * Pre-created TSL leaf nodes supplied by the wrapper class. See
 * `GSplatTSLNodes` for the rationale (avoids `.onUpdate('render')`
 * callback churn by consuming wrapper-owned `UniformNode`
 * references directly).
 */
export interface GSplatPickTSLNodes {
  /**
   * Splat data texture node (RGBA32F, 4 texels/splat) — shared with
   * the visual material's storage; rebound per node by the commit's
   * material sync.
   */
  readonly uSplatTex: TSLNode;
  readonly uResolution: TSLNode;
  readonly uPixelRatio: TSLNode;
  readonly uFx: TSLNode;
  readonly uFy: TSLNode;
  readonly uTruncate: TSLNode;
  readonly uTruncateSq: TSLNode;
  readonly uIsOrtho: TSLNode;
  /** Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB. */
  readonly uSortedIndexSlot: TSLNode;
  readonly uDensityDrop: TSLNode;
  readonly uNearCull: TSLNode;
  readonly uMaxExtentFactor: TSLNode;
  readonly uCov2DDilation: TSLNode;
  /**
   * Pick depth convention selector: 0 = brightness-as-depth (brightest
   * wins; commutative modes), 1 = real fragment depth (front-most wins;
   * surface/'normal' mode). Mirrors the GLSL `uSurfaceDepth` uniform.
   */
  readonly uSurfaceDepth: TSLNode;
  readonly uNodeId: TSLNode;
  readonly uLabelFilterIndex: TSLNode;
  readonly uShiftC: TSLNode;
  readonly uInvOneMinusC: TSLNode;
}

/**
 * GSplat picking material TSL factory.
 *
 * Consumes pre-created `UniformNode` references; the wrapper
 * (`GSplatPickingTSLMaterial`) owns those nodes and exposes them via
 * `material.uniforms` as `IUniform`-shaped getter/setter proxies.
 */
export function gsplatPickWebGPUFactory(
  nodes: GSplatPickTSLNodes,
  outMaterial?: NodeMaterial
): NodeMaterial {
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  // Draw-slot -> storage-slot mapping; splat data comes from the splat
  // texture (visual-factory parity, shader-tsl.ts).
  const aSortedIndex: TSLNode = sortedIndexNode(nodes.uSortedIndexSlot);
  const densityDropped: TSLNode = densityDroppedNode(nodes.uDensityDrop, aSortedIndex);

  const uSplatTex = nodes.uSplatTex;
  const uResolution = nodes.uResolution;
  const uTruncate = nodes.uTruncate;
  const uNearCull = nodes.uNearCull;
  const uMaxExtentFactor = nodes.uMaxExtentFactor;
  const uCov2DDilation = nodes.uCov2DDilation;
  const dilationPixelRatio = nodes.uPixelRatio.max(float(1.0));
  const cov2DDilation = uCov2DDilation.mul(dilationPixelRatio).mul(dilationPixelRatio);
  const uSurfaceDepth = nodes.uSurfaceDepth;
  const uNodeId = nodes.uNodeId;
  const uLabelFilterIndex = nodes.uLabelFilterIndex;
  const uShiftC = nodes.uShiftC;
  const uInvOneMinusC = nodes.uInvOneMinusC;
  const uTruncateSq = nodes.uTruncateSq;

  // ---- Vertex ----
  //
  // Traced inside a single Fn() body with explicit `.toVar()`
  // statements — same load-bearing structure as the visual gsplat
  // factory (shader-tsl.ts): as a free expression tree, TSL emits a
  // shared subexpression's assignment at its FIRST traversal use, and
  // the eigendecomposition values were first consumed inside the
  // major-axis select branch — read UNINITIALIZED on the near-diagonal
  // (isotropic) path. Inside Fn(), statements emit in trace order.

  const vAmplitude2D: TSLNode = varying(float(0.0));
  const vL2D: TSLNode = varying(vec3(float(0.0), float(0.0), float(0.0)));
  const vCenterScreen: TSLNode = varying(vec2(float(0.0), float(0.0)));
  const vNodeId: TSLNode = varying(uNodeId);
  // Storage slot, NOT instanceIndex (the draw slot): identical under
  // Phase-1 identity ordering, and stays the id the rest of the
  // pipeline addresses splats by once the sort worker permutes draw
  // order (Phase 2+). Mirrors the GLSL pick shader.
  // Storage index split into two 16-bit halves — the TSL twin of
  // `luxarElementIdParts` in glsl-lib.ts. The pick pass carries the index
  // through an RGBA32F buffer and float32 has a 24-bit mantissa, so one
  // channel cannot represent consecutive indices past 16,777,216 while a
  // node's capacity reaches 2^25 on a 32768-texel device. Split in INT
  // space — a float split would already have lost the bit it preserves —
  // and both halves are <= 65535, hence exact. Integer div/sub rather than
  // bit ops so the graph lowers the same way on both backends.
  const elementIdInt: TSLNode = int(aSortedIndex);
  const elementIdHi: TSLNode = elementIdInt.div(int(65536));
  const elementIdLo: TSLNode = elementIdInt.sub(elementIdHi.mul(int(65536)));
  const vElementId: TSLNode = varying(vec2(float(elementIdLo), float(elementIdHi)));

  const vertexBody = Fn(() => {
    // === Splat-texture fetch prologue (visual-factory parity) ===
    // Picking needs texels 0-2 only (center/amplitude/cholesky); color
    // is not fetched. Every value is a `.toVar()` STATEMENT (Fn house
    // rule). Width is a multiple of 4 -> one row per splat.
    const splatBase: TSLNode = int(aSortedIndex).mul(int(4)).toVar();
    // The width is baked as a LITERAL, not read via textureSize(): a
    // compile-time constant lets the shader compiler strength-reduce
    // the per-vertex %/int-div addressing below (measured -7% on the
    // quad's whole GPU pass; a uniform recovered almost none of it).
    // Safe because the width is a per-layout session constant, capped
    // at 4096 on every device (element-texture-layout.ts).
    const splatTexW: TSLNode = int(
      resolveElementTextureWidth(
        SPLAT_TEXTURE_LAYOUT,
        (nodes.uSplatTex as unknown as { value?: { image?: { width?: number } } }).value ?? null
      )
    ).toVar();
    const texelX: TSLNode = splatBase.mod(splatTexW).toVar();
    const texelY: TSLNode = splatBase.div(splatTexW).toVar();
    const splatT0: TSLNode = uSplatTex.load(ivec2(texelX, texelY)).toVar();
    const splatT1: TSLNode = uSplatTex.load(ivec2(texelX.add(int(1)), texelY)).toVar();
    const splatT2: TSLNode = uSplatTex.load(ivec2(texelX.add(int(2)), texelY)).toVar();
    const splatT3: TSLNode = uSplatTex.load(ivec2(texelX.add(int(3)), texelY)).toVar();
    const aCenter: TSLNode = vec3(splatT0).toVar();
    const aAmplitude: TSLNode = splatT0.w.toVar();
    const aCholesky01: TSLNode = splatT1.xy.toVar();
    const aCholesky23: TSLNode = splatT1.zw.toVar();
    const aCholesky45: TSLNode = splatT2.xy.toVar();
    const aLabelIndex: TSLNode = splatT3.z.toVar();

    const centerCam4: TSLNode = modelViewMatrix.mul(vec4(aCenter, 1.0)).toVar();
    const centerCam: TSLNode = vec3(centerCam4).toVar();
    // Clip-space centre through the projection this draw uses (GLSL twin:
    // shader-glsl.ts). The screen centre, the Jacobian, the coverage extent
    // and the ortho branch are all read from it and from P.
    const centerClip: TSLNode = cameraProjectionMatrix.mul(centerCam4).toVar();
    const isOrthoInt: TSLNode = isOrthoProjectionTSL().toVar();
    const invW: TSLNode = float(1.0).div(centerClip.w).toVar();
    const zDepth: TSLNode = centerCam.z.negate().toVar();

    const L00 = aCholesky01.x;
    const L10 = aCholesky01.y;
    const L11 = aCholesky23.x;
    const L20 = aCholesky23.y;
    const L21 = aCholesky45.x;
    const L22 = aCholesky45.y;
    const L3D: TSLNode = mat3(
      vec3(L00, L10, L20),
      vec3(float(0.0), L11, L21),
      vec3(float(0.0), float(0.0), L22)
    );

    const R: TSLNode = mat3(modelViewMatrix);
    const L_cam: TSLNode = R.mul(L3D).toVar();
    const SigmaCam: TSLNode = L_cam.mul(L_cam.transpose()).toVar();

    // Unified near handling (shared helper; subsumes the old
    // standalone behind-camera reject — see shader-tsl.ts).
    // 1e-20 floor = degenerate-smoothstep guard only; uNearCull is
    // scene-bounds-scaled (an absolute 1e-4 faded out tiny-unit scenes
    // entirely).
    const depthFade: TSLNode = perspectiveNearFadeTSL(
      isOrthoInt,
      centerCam.z,
      max(uNearCull, float(1e-20))
    ).toVar();
    const depthFadeReject: TSLNode = depthFade.lessThan(0.01);

    // Coverage fade (both projections; see shader-tsl.ts). Computed
    // UNCONDITIONALLY — the former absolute maxLateralVar > 0.01 gate
    // skipped the fade for splats with spatial sigma < 0.1 world units
    // while the extent clamp still applied.
    const maxLateralVar: TSLNode = max(
      SigmaCam.element(int(0)).element(int(0)),
      max(SigmaCam.element(int(1)).element(int(1)), SigmaCam.element(int(2)).element(int(2)))
    ).toVar();
    const isOrtho: TSLNode = isOrthoInt.equal(int(1)).toVar();
    // 1e-20 floors = pure div-by-zero/sqrt guards, matching the visual
    // shader: maxLateralVar is world-unit² (an absolute 1e-8 floor
    // coverage-culled every splat of a tiny-unit scene); zDepth is
    // bounded by the scene-relative near fade.
    const projectedExtent: TSLNode = uResolution.y
      .mul(0.5)
      .mul(projectionSizeScaleTSL())
      .mul(sqrt(max(maxLateralVar, float(1e-20))))
      .mul(uTruncate)
      .div(isOrtho.select(float(1.0), max(zDepth, float(1e-20))));
    const maxExtent: TSLNode = max(uResolution.x, uResolution.y).mul(uMaxExtentFactor);
    const coverageFade: TSLNode = float(1.0)
      .sub(smoothstep(maxExtent.mul(0.5), maxExtent, projectedExtent))
      .toVar();
    const coverageFadeReject: TSLNode = coverageFade.lessThan(0.01);
    // Use min(depthFade, coverageFade) for amplitude so picking matches
    // GLSL picking (shaders.ts) and visual TSL/GLSL gsplat
    // (shader-tsl.ts, shader-glsl.ts). Multiplication was
    // strictly less than the visual path and made splats near coverage
    // limits harder to pick than they appear.
    const nearFade: TSLNode = min(depthFade, coverageFade).toVar();

    // Projection Jacobian, general form (GLSL twin: shader-glsl.ts), as three
    // vec2 columns (TSL has no mat3x2): J[k] = res/2 * (P[k].xy / w -
    // clip.xy * P[k].w / w^2), P = cameraProjectionMatrix. It reduces to the
    // classic perspective / ortho Jacobians (projection-math.ts).
    const P: TSLNode = cameraProjectionMatrix;
    const halfRes: TSLNode = uResolution.mul(0.5);
    const clipTerm: TSLNode = centerClip.xy.mul(invW.mul(invW));
    const jacobianColumn = (k: number): TSLNode => {
      const Pk: TSLNode = P.element(int(k));
      return halfRes.mul(Pk.xy.mul(invW).sub(clipTerm.mul(Pk.w))).toVar();
    };
    const J0: TSLNode = jacobianColumn(0);
    const J1: TSLNode = jacobianColumn(1);
    const J2: TSLNode = jacobianColumn(2);

    const S00: TSLNode = SigmaCam.element(int(0)).element(int(0));
    const S01: TSLNode = SigmaCam.element(int(0)).element(int(1));
    const S02: TSLNode = SigmaCam.element(int(0)).element(int(2));
    const S10: TSLNode = SigmaCam.element(int(1)).element(int(0));
    const S11: TSLNode = SigmaCam.element(int(1)).element(int(1));
    const S12: TSLNode = SigmaCam.element(int(1)).element(int(2));
    const S20: TSLNode = SigmaCam.element(int(2)).element(int(0));
    const S21: TSLNode = SigmaCam.element(int(2)).element(int(1));
    const S22: TSLNode = SigmaCam.element(int(2)).element(int(2));
    const JS0: TSLNode = J0.mul(S00).add(J1.mul(S01)).add(J2.mul(S02)).toVar();
    const JS1: TSLNode = J0.mul(S10).add(J1.mul(S11)).add(J2.mul(S12)).toVar();
    const JS2: TSLNode = J0.mul(S20).add(J1.mul(S21)).add(J2.mul(S22)).toVar();
    const Sigma2D00: TSLNode = JS0.x.mul(J0.x).add(JS1.x.mul(J1.x)).add(JS2.x.mul(J2.x)).toVar();
    const Sigma2D10: TSLNode = JS0.x.mul(J0.y).add(JS1.x.mul(J1.y)).add(JS2.x.mul(J2.y)).toVar();
    const Sigma2D11: TSLNode = JS0.y.mul(J0.y).add(JS1.y.mul(J1.y)).add(JS2.y.mul(J2.y)).toVar();

    // 2D low-pass dilation — visual/GLSL-pick parity (widen the pickable
    // footprint to match the dilated visual splat). Diagonal only.
    Sigma2D00.addAssign(cov2DDilation);
    Sigma2D11.addAssign(cov2DDilation);

    // 2D Cholesky for the fragment's Mahalanobis solve.
    const s00: TSLNode = max(Sigma2D00, float(1e-8));
    const Lf00: TSLNode = sqrt(s00);
    const invLf00: TSLNode = float(1.0).div(Lf00).toVar();
    const Lf10: TSLNode = Sigma2D10.mul(invLf00).toVar();
    const Lf11: TSLNode = sqrt(max(Sigma2D11.sub(Lf10.mul(Lf10)), float(1e-8)));
    const invLf11: TSLNode = float(1.0).div(Lf11);
    const vL2DVal: TSLNode = vec3(invLf00, Lf10, invLf11);

    // Eigendecomposition for oriented quad — every shared value a Var
    // statement, emitted unconditionally before any consumer branch.
    const trace: TSLNode = Sigma2D00.add(Sigma2D11).toVar();
    const det2: TSLNode = Sigma2D00.mul(Sigma2D11).sub(Sigma2D10.mul(Sigma2D10));
    const disc: TSLNode = max(trace.mul(trace).sub(det2.mul(4.0)), float(0.0));
    const sqrtDisc: TSLNode = sqrt(disc).toVar();
    const lambda1: TSLNode = max(trace.add(sqrtDisc).mul(0.5), float(1e-6)).toVar();
    const lambda2: TSLNode = max(trace.sub(sqrtDisc).mul(0.5), float(1e-6)).toVar();

    const offDiagSig: TSLNode = abs(Sigma2D10).greaterThan(1e-6);
    const majorOff: TSLNode = normalize(vec2(lambda1.sub(Sigma2D11), Sigma2D10));
    const majorDiag: TSLNode = Sigma2D00.greaterThanEqual(Sigma2D11).select(
      vec2(1.0, 0.0),
      vec2(0.0, 1.0)
    );
    const majorAxis: TSLNode = offDiagSig.select(majorOff, majorDiag).toVar();
    const minorAxis: TSLNode = vec2(majorAxis.y.negate(), majorAxis.x).toVar();

    const extent1Raw: TSLNode = uTruncate.mul(sqrt(lambda1)).toVar();
    const extent2Raw: TSLNode = uTruncate.mul(sqrt(lambda2)).toVar();
    const maxExtentPx: TSLNode = max(uResolution.x, uResolution.y).mul(uMaxExtentFactor);
    const largestExtent: TSLNode = max(extent1Raw, extent2Raw).toVar();
    const clampScale: TSLNode = largestExtent
      .greaterThan(maxExtentPx)
      .select(maxExtentPx.div(largestExtent), float(1.0))
      .toVar();
    const extent1: TSLNode = extent1Raw.mul(clampScale);
    const extent2: TSLNode = extent2Raw.mul(clampScale);

    // Screen centre in pixels from the clip-space centre.
    const vCenterScreenVal: TSLNode = centerClip.xy
      .mul(invW)
      .mul(0.5)
      .add(0.5)
      .mul(uResolution)
      .toVar();

    const quadOffset: TSLNode = majorAxis
      .mul(aQuadCorner.x)
      .mul(extent1)
      .add(minorAxis.mul(aQuadCorner.y).mul(extent2));
    const screenPos: TSLNode = vCenterScreenVal.add(quadOffset);
    const ndcXY: TSLNode = screenPos.div(uResolution).mul(2.0).sub(1.0);

    const ndcZ: TSLNode = centerClip.z.div(centerClip.w);

    // Parity with the GLSL `invalidCov2D || isInvalidFloat(aAmplitude)`
    // guard in shaders.ts (GSPLAT_PICK_VERTEX_SHADER) and the
    // visual `invalidCov2D || invalidFloat(aAmplitude)` guard in
    // shader-glsl.ts. Without this, NaN/Inf upstream values
    // propagate through Cholesky / eigendecomposition and can make a
    // splat unpickable in unpredictable ways.
    const invalidAmp: TSLNode = invalidFloatTSL(aAmplitude);
    const invalidCov: TSLNode = invalidFloatTSL(Sigma2D00)
      .or(invalidFloatTSL(Sigma2D10))
      .or(invalidFloatTSL(Sigma2D11));
    const validClipPos: TSLNode = vec4(ndcXY, ndcZ, float(1.0));
    const rejectClipPos: TSLNode = vec4(float(0.0), float(0.0), float(-2.0), float(1.0));
    const labelRejected: TSLNode = int(uLabelFilterIndex)
      .greaterThan(int(0))
      .and(int(aLabelIndex.add(0.5)).notEqual(int(uLabelFilterIndex)));
    const rejected: TSLNode = depthFadeReject
      .or(coverageFadeReject)
      .or(invalidAmp)
      .or(invalidCov)
      .or(labelRejected)
      .or(densityDropped);

    // Pickability always uses max projection — amplitude = aAmplitude · nearFade.
    vAmplitude2D.assign(aAmplitude.mul(nearFade));
    vL2D.assign(vL2DVal);
    vCenterScreen.assign(vCenterScreenVal);

    return rejected.select(rejectClipPos, validClipPos);
  });

  const clipPos: TSLNode = vertexBody();

  // ---- Fragment ----
  //
  // The Mahalanobis chain is computed ONCE and read by BOTH fragment
  // entry points, so where its assignments are EMITTED is load-bearing.
  //
  // `colorNode` and `depthNode` are independent stage entry points, and
  // the order in which three builds them is not part of its API: r184
  // emitted the colour flow first, r185 emits the depth flow first. A
  // value materialised with `.toVar()` is assigned wherever three first
  // BUILDS it, and a branch-scoped assignment is only re-hoisted for a
  // later reader that is itself inside a block
  // (`NodeBuilder.addFlowCodeHierarchy`, gated on
  // `builder.context.nodeBlock !== undefined`) — never for one at the
  // top level of a flow. `depthNode` BRANCHES on `uSurfaceDepth`, which
  // three lowers to a real `if / else`, so if the depth flow builds
  // first a free-standing chain lands inside the `else` arm while every
  // top-level reader (both `Discard` conditions and the output vec4)
  // still reads the variable — unassigned, i.e. 0, which discards every
  // fragment and renders an empty pick buffer.
  //
  // The invariant this enforces by construction: every value shared
  // between the two entry points is declared up front and ASSIGNED in
  // `fragmentPrologue`, a `'void'`-typed `Fn` invoked as the FIRST
  // statement of BOTH entry points. A void `Fn` call is a stack
  // STATEMENT — a non-void one is wrapped in an intent var that three
  // skips, which would leave the call to build at its consumption site,
  // inside the arm again — so the assignments are emitted in trace order
  // in unconditional top-level flow, whichever entry point three builds
  // first.
  //
  // What `.once()` does and does not buy: the prologue's code is emitted
  // where it is FIRST built, and because both entry points call it as
  // their first statement that site is the top level of whichever flow
  // three emits first. `.once()` then lets the second call reuse the
  // already-traced result instead of emitting the chain twice — the
  // "compile once, reference twice" property the `.toVar()`s were there
  // for. Its cache lives on the NodeBuilder (so per material build) and
  // is keyed on shader stage `'any'`, so calling this same prologue from
  // another shader STAGE would silently reuse the first stage's nodes —
  // it is fragment-only for that reason. A cache MISS would merely
  // duplicate the chain, which stays correct.
  //
  // One visible consequence of the r185 flip: with the depth flow
  // emitted first, `gl_FragDepth` is written ABOVE the `Discard`s in
  // source order (the GLSL twins discard first). Still correct — a
  // discarded fragment writes no buffer at all, depth included.
  const mahalSq: TSLNode = float(0.0).toVar('gsplatPickMahalSq');
  const intensity: TSLNode = float(0.0).toVar('gsplatPickIntensity');
  const brightness: TSLNode = float(0.0).toVar('gsplatPickBrightness');

  const fragmentPrologue = Fn(() => {
    // Bottom-left fragcoord reconstruction — same top-left/bottom-left
    // mismatch fix as the visual factory (see shader-tsl.ts fragment).
    // Un-flip with `screenSize` (the bound target's size — the exact term
    // the builder's top-left flip used), not the app-stamped uResolution;
    // see shader-tsl.ts. For picking they currently coincide (uResolution
    // is re-stamped to the pick target dims), but screenSize is exact by
    // construction in every configuration.
    // `.toVar()` on these two for the same reason as the vertex prologue's house rule
    // (every value a STATEMENT): as free expressions they are re-expanded at each of
    // their ~6 downstream uses, which inlines the whole un-flip twice per Mahalanobis
    // term. Materialising them changes no arithmetic, only how often it is written out.
    const fragCoordBL: TSLNode = vec2(
      screenCoordinate.x,
      screenSize.y.sub(screenCoordinate.y)
    ).toVar();
    const d: TSLNode = vec2(fragCoordBL.sub(vCenterScreen)).toVar();
    const y0: TSLNode = d.x.mul(vL2D.x).toVar();
    const y1: TSLNode = d.y.sub(vL2D.y.mul(y0)).mul(vL2D.z).toVar();
    mahalSq.assign(y0.mul(y0).add(y1.mul(y1)));
    intensity.assign(
      vAmplitude2D.mul(uInvOneMinusC).mul(max(exp(mahalSq.mul(-0.5)).sub(uShiftC), float(0.0)))
    );
    brightness.assign(clamp(intensity, 0.0, 1.0));
    // Returned only so `.once()` has a result to cache — a body with no
    // result re-traces on the second call and emits the whole chain
    // twice. The entry points read the vars above, not this value.
    return brightness;
  }, 'void').once();

  const colorNode = Fn(() => {
    fragmentPrologue();
    Discard(mahalSq.greaterThan(uTruncateSq));
    Discard(intensity.lessThan(1e-4));
    return vec4(vNodeId, vElementId.x, brightness, vElementId.y);
  });

  // Pick depth convention — mirrors the GLSL fragment (shaders.ts):
  // The surface modes ('normal'/'opaque') write the REAL fragment depth
  // (`depth` = the builtin fragment depth; the vertex puts the
  // splat-center NDC z in the clip position, so this is the true
  // projected depth) — the front-most splat wins, matching the occluding
  // surface the user sees. Commutative modes (additive/max/luminous)
  // keep brightness-as-depth — the brightest splat wins.
  const depthNode = Fn(() => {
    fragmentPrologue();
    return int(uSurfaceDepth)
      .equal(int(1))
      .select(depth as unknown as TSLNode, float(1.0).sub(brightness));
  });

  const material = outMaterial ?? new NodeMaterial();
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.depthNode = depthNode();
  material.toneMapped = false;
  material.depthTest = true;
  material.depthWrite = true;
  material.transparent = false;
  // The element index's HIGH half rides in alpha, and THREE's NodeMaterial
  // appends `DiffuseColor.w *= material.opacity` to every generated fragment
  // (see the codegen snapshots, and `tsl-opacity-tail.test.ts` for the same
  // tail on the visual materials). NoBlending does not suppress that
  // shader-side multiply, so any opacity other than exactly 1 would scale the
  // high half and decode a WRONG element id — on the TSL path only, since the
  // GLSL twins have no such tail. Pin it so the multiply is provably identity,
  // including when a caller injects `outMaterial`.
  material.opacity = 1;
  // Picking output is an opaque ID buffer; any blending would smear
  // nodeId / elementId across overlapping picks. Matches the GLSL
  // picking material.
  material.blending = THREE.NoBlending;
  return material;
}

/**
 * Snapshot adapter: build a `GSplatPickTSLNodes` set from a flat
 * `IUniform` record. See `buildLineTSLNodesFromUniforms` for the
 * rationale.
 */
export function buildGSplatPickTSLNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>
): GSplatPickTSLNodes {
  return {
    // Splat data texture -- bound from the caller's uniform when present,
    // else a 4x1 RGBA32F placeholder (codegen-only consumers).
    uSplatTex: texture(
      (uniforms.uSplatTex?.value as THREE.Texture | null) ??
        new THREE.DataTexture(new Float32Array(16), 4, 1, THREE.RGBAFormat, THREE.FloatType)
    ),
    uResolution: uniform(
      (uniforms.uResolution?.value as THREE.Vector2 | undefined) ?? new THREE.Vector2(1, 1)
    ),
    uPixelRatio: uniform((uniforms.uPixelRatio?.value as number) ?? 1),
    uFx: uniform((uniforms.uFx?.value as number) ?? 1.0),
    uFy: uniform((uniforms.uFy?.value as number) ?? 1.0),
    uTruncate: uniform((uniforms.uTruncate?.value as number) ?? 1.5),
    // 1.5² — keep the default PAIR consistent (9.0 was half-copied from the
    // visual builder's 3.0/9.0 and sized the quad for 1.5σ while discarding at 3σ).
    uTruncateSq: uniform((uniforms.uTruncateSq?.value as number) ?? 2.25),
    uIsOrtho: uniform((uniforms.uIsOrtho?.value as number) ?? 0),
    uSortedIndexSlot: uniform((uniforms.uSortedIndexSlot?.value as number) ?? 0),
    uDensityDrop: uniform((uniforms.uDensityDrop?.value as number) ?? 0),
    uNearCull: uniform((uniforms.uNearCull?.value as number) ?? 1e-4),
    uMaxExtentFactor: uniform((uniforms.uMaxExtentFactor?.value as number) ?? 1.0),
    // Neutral fallback 0 (harness/snapshot adapter; production sets 0.3).
    uCov2DDilation: uniform((uniforms.uCov2DDilation?.value as number) ?? 0),
    // Default 0 = brightness-as-depth (the commutative-mode convention).
    uSurfaceDepth: uniform((uniforms.uSurfaceDepth?.value as number) ?? 0),
    uNodeId: uniform((uniforms.uNodeId?.value as number) ?? 0),
    uLabelFilterIndex: uniform((uniforms.uLabelFilterIndex?.value as number) ?? 0),
    uShiftC: uniform((uniforms.uShiftC?.value as number) ?? 0.0),
    uInvOneMinusC: uniform((uniforms.uInvOneMinusC?.value as number) ?? 1.0),
  };
}
