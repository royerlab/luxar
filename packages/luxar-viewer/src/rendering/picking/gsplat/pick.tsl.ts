/**
 * GSplat picking material TSL factory — NodeMaterial counterpart to
 * `GSPLAT_PICK_SOURCE` in `shaders.ts`.
 *
 * Same 3D→2D covariance projection pipeline as `gsplat.tsl`, but the
 * fragment outputs picking data:
 *   - R: nodeId (uniform)
 *   - G: elementId (instance index)
 *   - B: brightness clamped to [0, 1]
 *   - A: 1.0
 * Depth = 1.0 - brightness (brightness-as-depth tie-breaking).
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
  instanceIndex,
  vec2 as _vec2,
  vec3 as _vec3,
  vec4 as _vec4,
  mat3 as _mat3,
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
  modelViewMatrix,
  cameraProjectionMatrix,
  screenCoordinate,
  screenSize,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import {
  perspectiveNearFadeTSL,
  invalidFloatTSL,
  type TSLNode,
} from '../../materials/_shared/tsl-helpers';

const vec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _vec2 as TSLNode;
const vec3: (a?: TSLNode, b?: TSLNode, c?: TSLNode) => TSLNode = _vec3 as TSLNode;
const vec4: (a?: TSLNode, b?: TSLNode, c?: TSLNode, d?: TSLNode) => TSLNode = _vec4 as TSLNode;
const mat3: (a?: TSLNode, b?: TSLNode, c?: TSLNode) => TSLNode = _mat3 as TSLNode;

/**
 * Pre-created TSL leaf nodes supplied by the wrapper class. See
 * `GSplatTSLNodes` for the rationale (avoids `.onUpdate('render')`
 * callback churn by consuming wrapper-owned `UniformNode`
 * references directly).
 */
export interface GSplatPickTSLNodes {
  readonly uResolution: TSLNode;
  readonly uFx: TSLNode;
  readonly uFy: TSLNode;
  readonly uTruncate: TSLNode;
  readonly uTruncateSq: TSLNode;
  readonly uIsOrtho: TSLNode;
  readonly uNearCull: TSLNode;
  readonly uMaxExtentFactor: TSLNode;
  readonly uNodeId: TSLNode;
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
  const aCenter: TSLNode = attribute<'vec3'>('aCenter', 'vec3');
  const aCholesky01: TSLNode = attribute<'vec2'>('aCholesky01', 'vec2');
  const aCholesky23: TSLNode = attribute<'vec2'>('aCholesky23', 'vec2');
  const aCholesky45: TSLNode = attribute<'vec2'>('aCholesky45', 'vec2');
  const aAmplitude: TSLNode = attribute<'float'>('aAmplitude', 'float');

  const uResolution = nodes.uResolution;
  const uFx = nodes.uFx;
  const uFy = nodes.uFy;
  const uTruncate = nodes.uTruncate;
  const uIsOrtho = nodes.uIsOrtho;
  const uNearCull = nodes.uNearCull;
  const uMaxExtentFactor = nodes.uMaxExtentFactor;
  const uNodeId = nodes.uNodeId;
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
  const vElementId: TSLNode = varying(float(instanceIndex));

  const vertexBody = Fn(() => {
    const centerCam4: TSLNode = modelViewMatrix.mul(vec4(aCenter, 1.0)).toVar();
    const centerCam: TSLNode = vec3(centerCam4).toVar();
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
    const depthFade: TSLNode = perspectiveNearFadeTSL(
      uIsOrtho,
      centerCam.z,
      max(uNearCull, float(1e-4))
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
    const isOrtho: TSLNode = int(uIsOrtho).equal(int(1)).toVar();
    const projectedExtent: TSLNode = uFx
      .mul(sqrt(max(maxLateralVar, float(1e-8))))
      .mul(uTruncate)
      .div(isOrtho.select(float(1.0), max(zDepth, float(1e-8))));
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

    // Projection Jacobian.
    const invZ: TSLNode = float(1.0)
      .div(max(zDepth, float(1e-8)))
      .toVar();
    const invZ2: TSLNode = invZ.mul(invZ);
    const J0: TSLNode = isOrtho
      .select(vec2(uFx, float(0.0)), vec2(uFx.mul(invZ), float(0.0)))
      .toVar();
    const J1: TSLNode = isOrtho
      .select(vec2(float(0.0), uFy), vec2(float(0.0), uFy.mul(invZ)))
      .toVar();
    const J2: TSLNode = isOrtho
      .select(
        vec2(float(0.0), float(0.0)),
        vec2(uFx.mul(centerCam.x).mul(invZ2), uFy.mul(centerCam.y).mul(invZ2))
      )
      .toVar();

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

    const centerScreenOrtho: TSLNode = vec2(
      uFx.mul(centerCam.x).add(uResolution.x.mul(0.5)),
      uFy.mul(centerCam.y).add(uResolution.y.mul(0.5))
    );
    const centerScreenPersp: TSLNode = vec2(
      uFx.mul(centerCam.x).mul(invZ).add(uResolution.x.mul(0.5)),
      uFy.mul(centerCam.y).mul(invZ).add(uResolution.y.mul(0.5))
    );
    const vCenterScreenVal: TSLNode = isOrtho.select(centerScreenOrtho, centerScreenPersp).toVar();

    const quadOffset: TSLNode = majorAxis
      .mul(aQuadCorner.x)
      .mul(extent1)
      .add(minorAxis.mul(aQuadCorner.y).mul(extent2));
    const screenPos: TSLNode = vCenterScreenVal.add(quadOffset);
    const ndcXY: TSLNode = screenPos.div(uResolution).mul(2.0).sub(1.0);

    const centerClip: TSLNode = cameraProjectionMatrix.mul(centerCam4).toVar();
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
    const rejected: TSLNode = depthFadeReject.or(coverageFadeReject).or(invalidAmp).or(invalidCov);

    // Pickability always uses max projection — amplitude = aAmplitude · nearFade.
    vAmplitude2D.assign(aAmplitude.mul(nearFade));
    vL2D.assign(vL2DVal);
    vCenterScreen.assign(vCenterScreenVal);

    return rejected.select(rejectClipPos, validClipPos);
  });

  const clipPos: TSLNode = vertexBody();

  // ---- Fragment ----
  //
  // Compute the Mahalanobis brightness ONCE, materialised via
  // `.toVar()` so both color and depth fragment outputs reference the
  // same computation instead of each rebuilding the
  // forward-substitution + exp + clamp chain. TSL doesn't expose a
  // multi-output fragment Fn in r184 (separate `colorNode` and
  // `depthNode` are independent stage entry points); the `.toVar()`
  // is the closest available "compile once, reference twice" pattern.
  // Worst case (no common-subexpression elimination by the TSL
  // builder) the cost is equivalent to today's duplicated graph;
  // best case it halves the per-fragment picking cost on
  // splat-heavy scenes.
  // Bottom-left fragcoord reconstruction — same top-left/bottom-left
  // mismatch fix as the visual factory (see shader-tsl.ts fragment).
  // Un-flip with `screenSize` (the bound target's size — the exact term
  // the builder's top-left flip used), not the app-stamped uResolution;
  // see shader-tsl.ts. For picking they currently coincide (uResolution
  // is re-stamped to the pick target dims), but screenSize is exact by
  // construction in every configuration.
  const fragCoordBL: TSLNode = vec2(screenCoordinate.x, screenSize.y.sub(screenCoordinate.y));
  const d: TSLNode = vec2(fragCoordBL.sub(vCenterScreen));
  const y0: TSLNode = d.x.mul(vL2D.x).toVar();
  const y1: TSLNode = d.y.sub(vL2D.y.mul(y0)).mul(vL2D.z).toVar();
  const mahalSq: TSLNode = y0.mul(y0).add(y1.mul(y1)).toVar();
  const intensity: TSLNode = vAmplitude2D
    .mul(uInvOneMinusC)
    .mul(max(exp(mahalSq.mul(-0.5)).sub(uShiftC), float(0.0)))
    .toVar();
  const brightness: TSLNode = clamp(intensity, 0.0, 1.0).toVar();

  const colorNode = Fn(() => {
    Discard(mahalSq.greaterThan(uTruncateSq));
    Discard(intensity.lessThan(1e-4));
    return vec4(vNodeId, vElementId, brightness, 1.0);
  });

  const depthNode = Fn(() => float(1.0).sub(brightness));

  const material = outMaterial ?? new NodeMaterial();
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.depthNode = depthNode();
  material.toneMapped = false;
  material.depthTest = true;
  material.depthWrite = true;
  material.transparent = false;
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
    uResolution: uniform(
      (uniforms.uResolution?.value as THREE.Vector2 | undefined) ?? new THREE.Vector2(1, 1)
    ),
    uFx: uniform((uniforms.uFx?.value as number) ?? 1.0),
    uFy: uniform((uniforms.uFy?.value as number) ?? 1.0),
    uTruncate: uniform((uniforms.uTruncate?.value as number) ?? 1.5),
    // 1.5² — keep the default PAIR consistent (9.0 was half-copied from the
    // visual builder's 3.0/9.0 and sized the quad for 1.5σ while discarding at 3σ).
    uTruncateSq: uniform((uniforms.uTruncateSq?.value as number) ?? 2.25),
    uIsOrtho: uniform((uniforms.uIsOrtho?.value as number) ?? 0),
    uNearCull: uniform((uniforms.uNearCull?.value as number) ?? 1e-4),
    uMaxExtentFactor: uniform((uniforms.uMaxExtentFactor?.value as number) ?? 1.0),
    uNodeId: uniform((uniforms.uNodeId?.value as number) ?? 0),
    uShiftC: uniform((uniforms.uShiftC?.value as number) ?? 0.0),
    uInvOneMinusC: uniform((uniforms.uInvOneMinusC?.value as number) ?? 1.0),
  };
}
