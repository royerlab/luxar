/**
 * GSplat material TSL factory — NodeMaterial counterpart to the
 * GLSL3 shaders in `shader-glsl.ts`.
 *
 * Renders each Gaussian splat as an oriented quad in screen space.
 * Per-vertex `aQuadCorner` (±1) scaled by the 2D covariance eigen-
 * vectors gives an axis-aligned bounding quad of the Gaussian's
 * truncation extent.
 *
 * Per-instance attributes:
 *   - aCenter      (vec3) — 3D world centre
 *   - aCholesky01  (vec2) — [L00, L10]
 *   - aCholesky23  (vec2) — [L11, L20]
 *   - aCholesky45  (vec2) — [L21, L22]
 *   - aAmplitude   (float)
 *   - aColor       (vec3) — replaced by colormap LUT under USE_COLORMAP
 *
 * Vertex pipeline:
 *   1. Project centre to camera space.
 *   2. Reject splats behind camera (gl_Position = -2 NDC).
 *   3. Rotate 3D Cholesky to camera space; Σ_cam = L_cam·L_camᵀ.
 *   4. Apply near-plane fade + coverage fade (perspective only).
 *   5. Project covariance via Jacobian: Σ_2D = J·Σ_cam·Jᵀ.
 *   6. Compute 2D Cholesky for the fragment Mahalanobis math.
 *   7. Sum vs max projection modes: sum integrates along ray
 *      (needs Σ_cam⁻¹ via 3×3 cofactor expansion); max uses peak.
 *   8. Eigendecompose Σ_2D → major/minor axes → oriented quad.
 *   9. Clamp quad to viewport bound.
 *  10. Project to NDC + write gl_Position with proper clip-space z.
 *
 * Fragment pipeline:
 *   - Mahalanobis distance from packed [invL00, L10, invL11].
 *   - Truncation discard.
 *   - Shifted Gaussian intensity: a · scale · max(0, exp(-r²/2) - C).
 *   - GOG (gain/offset/gamma) + opacity multiply on output RGB.
 *
 * @module rendering/materials/gsplat/shader-tsl
 */

import * as THREE from 'three';
import {
  Fn,
  uniform,
  attribute,
  varying,
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
  texture,
  screenCoordinate,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { invalidFloatTSL, type TSLNode } from '../_shared/tsl-helpers';
import { applyBlendingStateToMaterial, getCompleteBlendingState } from '../../blending-state';
import type { BlendingMode } from '../../material-manager';
import { isMaxMode } from '../../blending-state';

// Type-erased constructor aliases. TSL's typed `vec2`/`vec3`/`vec4`/`mat3`
// overloads reject many valid combinations of intermediate `Node<…>`
// results — e.g. `vec2(uFx.mul(invZ), float(0))` doesn't match any
// declared overload, even though the runtime accepts it. Re-exporting
// each as TSLNode-typed sidesteps every overload-mismatch error in
// this file without affecting the generated GLSL/WGSL.
const vec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _vec2 as TSLNode;
const vec3: (a?: TSLNode, b?: TSLNode, c?: TSLNode) => TSLNode = _vec3 as TSLNode;
const vec4: (a?: TSLNode, b?: TSLNode, c?: TSLNode, d?: TSLNode) => TSLNode = _vec4 as TSLNode;
const mat3: (a?: TSLNode, b?: TSLNode, c?: TSLNode) => TSLNode = _mat3 as TSLNode;

export interface GSplatTSLConfig {
  readonly useColormap?: boolean;
  /**
   * When `true` (gamma == 1.0), the per-fragment / pre-LUT gamma
   * `pow()` is skipped — `pow(x, 1) == x`. Mirrors the GLSL3
   * `LUXAR_GAMMA_ONE` define. The material wrapper sets this from the
   * presence of that define and toggles it on `updateGamma`.
   */
  readonly gammaOne?: boolean;
  /**
   * Luxar blending mode. GSplats premultiply intensity into RGB
   * unconditionally so the shader-output style does not need to flip
   * with the mode — only the THREE blending state changes. Defaults
   * to `'additive'` to match the GLSL wrapper class.
   */
  readonly blendingMode?: BlendingMode;
}

/**
 * Pre-created TSL leaf nodes supplied by the wrapper class. The
 * factory consumes them directly in the shader graph instead of
 * snapshotting `IUniform.value` and bridging back via
 * `.onUpdate('render')`. The wrapper owns lifetime; `rebuildGraph()`
 * reuses the same leaves so mutations remain visible after a
 * defines change.
 *
 * The colormap texture node is factory-time bound (TSL
 * `texture(...)` captures the THREE.Texture at call time); the
 * wrapper rebuilds the graph when the texture identity changes
 * (see `setColormapTexture` in the wrapper).
 */
export interface GSplatTSLNodes {
  readonly uResolution: TSLNode;
  readonly uFx: TSLNode;
  readonly uFy: TSLNode;
  readonly uTruncate: TSLNode;
  readonly uTruncateSq: TSLNode;
  readonly uRayIntegralFactor: TSLNode;
  readonly uProjectionMode: TSLNode;
  readonly uIsOrtho: TSLNode;
  readonly uNearCull: TSLNode;
  readonly uMaxExtentFactor: TSLNode;
  readonly uOpacity: TSLNode;
  readonly uInvGamma: TSLNode;
  readonly uIntensity: TSLNode;
  readonly uOffset: TSLNode;
  readonly uShiftC: TSLNode;
  readonly uInvOneMinusC: TSLNode;
  readonly uColormapTex?: TSLNode;
  readonly uScalarMin?: TSLNode;
  readonly uScalarScale?: TSLNode;
}

/**
 * GSplat material TSL factory.
 *
 * Consumes pre-created TSL `UniformNode` references via `nodes`; the
 * wrapper class (`GSplatTSLMaterial`) owns those nodes and exposes
 * them through `material.uniforms` as `IUniform`-shaped
 * getter/setter proxies (see `proxyIUniform` in `tsl-helpers.ts`).
 * Mutations to `material.uniforms.uX.value` land directly on the
 * node with no per-render JS callback.
 *
 * Colormap nodes are required only when `config.useColormap === true`.
 */
export function gsplatWebGPUFactory(
  nodes: GSplatTSLNodes,
  config: GSplatTSLConfig = {},
  outMaterial?: NodeMaterial
): NodeMaterial {
  // Per-vertex / per-instance attributes.
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  const aCenter: TSLNode = attribute<'vec3'>('aCenter', 'vec3');
  const aCholesky01: TSLNode = attribute<'vec2'>('aCholesky01', 'vec2');
  const aCholesky23: TSLNode = attribute<'vec2'>('aCholesky23', 'vec2');
  const aCholesky45: TSLNode = attribute<'vec2'>('aCholesky45', 'vec2');
  const aAmplitude: TSLNode = attribute<'float'>('aAmplitude', 'float');
  const aColor: TSLNode = attribute<'vec3'>('aColor', 'vec3');

  // Uniform leaves come from the wrapper. No per-render callbacks:
  // mutations to `material.uniforms.X.value` already route to
  // `node.value` via `proxyIUniform`.
  const uResolution = nodes.uResolution;
  const uFx = nodes.uFx;
  const uFy = nodes.uFy;
  const uTruncate = nodes.uTruncate;
  const uRayIntegralFactor = nodes.uRayIntegralFactor;
  const uProjectionMode = nodes.uProjectionMode;
  const uIsOrtho = nodes.uIsOrtho;
  const uNearCull = nodes.uNearCull;
  const uMaxExtentFactor = nodes.uMaxExtentFactor;
  const uColormapTex = config.useColormap ? nodes.uColormapTex : null;
  const uScalarMin = config.useColormap ? nodes.uScalarMin : null;
  const uScalarScale = config.useColormap ? nodes.uScalarScale : null;
  const uOpacity = nodes.uOpacity;
  const uInvGamma = nodes.uInvGamma;
  const uIntensity = nodes.uIntensity;
  const uOffset = nodes.uOffset;
  const uShiftC = nodes.uShiftC;
  const uInvOneMinusC = nodes.uInvOneMinusC;
  const uTruncateSq = nodes.uTruncateSq;

  // ---- Vertex computation ----

  // Centre in camera space.
  const centerCam4: TSLNode = modelViewMatrix.mul(vec4(aCenter, 1.0));
  const centerCam: TSLNode = vec3(centerCam4);

  // Behind-camera (camera looks down -Z; +Z means behind).
  const behindCamera: TSLNode = centerCam.z.greaterThanEqual(0.0);
  const zDepth: TSLNode = centerCam.z.negate();

  // 3D Cholesky as a mat3 (column-major).
  // Column 0: [L00, L10, L20], Column 1: [0, L11, L21], Column 2: [0, 0, L22].
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

  // Rotate to camera space: L_cam = R · L3D (R = top-left 3×3 of MV).
  // mat3(mat4) at runtime takes the upper-left 3×3.
  const R: TSLNode = mat3(modelViewMatrix);
  const L_cam: TSLNode = R.mul(L3D);
  // Sigma_cam = L_cam · L_camᵀ. We need explicit row access on the
  // result for downstream cofactor / variance math.
  const SigmaCam: TSLNode = L_cam.mul(L_cam.transpose());

  // Near-plane depth fade (perspective only).
  const depthFade: TSLNode = int(uIsOrtho)
    .equal(int(0))
    .select(smoothstep(uNearCull, uNearCull.mul(2.0), zDepth).toVar(), float(1.0));
  const depthFadeReject: TSLNode = depthFade.lessThan(0.01);

  // Coverage fade — perspective only, gated on the max diagonal
  // variance vs viewport extent.
  const maxLateralVar: TSLNode = max(
    SigmaCam.element(int(0)).element(int(0)),
    max(SigmaCam.element(int(1)).element(int(1)), SigmaCam.element(int(2)).element(int(2)))
  );
  const projectedExtent: TSLNode = uFx
    .mul(sqrt(max(maxLateralVar, float(1e-8))))
    .mul(uTruncate)
    .div(max(zDepth, float(1e-8)));
  const maxExtent: TSLNode = max(uResolution.x, uResolution.y).mul(uMaxExtentFactor);
  const coverageFadeRaw: TSLNode = float(1.0).sub(
    smoothstep(maxExtent.mul(0.5), maxExtent, projectedExtent)
  );
  const coverageFade: TSLNode = int(uIsOrtho)
    .equal(int(0))
    .and(maxLateralVar.greaterThan(0.01))
    .select(coverageFadeRaw.toVar(), float(1.0));
  const coverageFadeReject: TSLNode = coverageFade.lessThan(0.01);

  const nearFade: TSLNode = min(depthFade, coverageFade);

  // Projection Jacobian. mat3x2 in GLSL = three vec2 columns; we
  // represent it as three independent vec2 nodes to sidestep TSL's
  // missing mat3x2 type. JS0/JS1/JS2 form the matrix M = J·Σ_cam.
  const invZ: TSLNode = float(1.0).div(max(zDepth, float(1e-8)));
  const invZ2: TSLNode = invZ.mul(invZ);
  // Perspective Jacobian columns.
  const J0p: TSLNode = vec2(uFx.mul(invZ), float(0.0));
  const J1p: TSLNode = vec2(float(0.0), uFy.mul(invZ));
  const J2p: TSLNode = vec2(uFx.mul(centerCam.x).mul(invZ2), uFy.mul(centerCam.y).mul(invZ2));
  // Orthographic Jacobian (depth-independent).
  const J0o: TSLNode = vec2(uFx, float(0.0));
  const J1o: TSLNode = vec2(float(0.0), uFy);
  const J2o: TSLNode = vec2(float(0.0), float(0.0));
  const isOrtho: TSLNode = int(uIsOrtho).equal(int(1));
  const J0: TSLNode = isOrtho.select(J0o.toVar(), J0p.toVar());
  const J1: TSLNode = isOrtho.select(J1o.toVar(), J1p.toVar());
  const J2: TSLNode = isOrtho.select(J2o.toVar(), J2p.toVar());

  // Σ_2D = J · Σ_cam · Jᵀ.
  const S00: TSLNode = SigmaCam.element(int(0)).element(int(0));
  const S01: TSLNode = SigmaCam.element(int(0)).element(int(1));
  const S02: TSLNode = SigmaCam.element(int(0)).element(int(2));
  const S10: TSLNode = SigmaCam.element(int(1)).element(int(0));
  const S11: TSLNode = SigmaCam.element(int(1)).element(int(1));
  const S12: TSLNode = SigmaCam.element(int(1)).element(int(2));
  const S20: TSLNode = SigmaCam.element(int(2)).element(int(0));
  const S21: TSLNode = SigmaCam.element(int(2)).element(int(1));
  const S22: TSLNode = SigmaCam.element(int(2)).element(int(2));
  const JS0: TSLNode = J0.mul(S00).add(J1.mul(S01)).add(J2.mul(S02));
  const JS1: TSLNode = J0.mul(S10).add(J1.mul(S11)).add(J2.mul(S12));
  const JS2: TSLNode = J0.mul(S20).add(J1.mul(S21)).add(J2.mul(S22));
  const Sigma2D00: TSLNode = JS0.x.mul(J0.x).add(JS1.x.mul(J1.x)).add(JS2.x.mul(J2.x));
  const Sigma2D10: TSLNode = JS0.x.mul(J0.y).add(JS1.x.mul(J1.y)).add(JS2.x.mul(J2.y));
  const Sigma2D11: TSLNode = JS0.y.mul(J0.y).add(JS1.y.mul(J1.y)).add(JS2.y.mul(J2.y));

  // 2D Cholesky factorisation: [invL00, L10, invL11] for fragment-side
  // MUL-instead-of-DIV.
  const s00: TSLNode = max(Sigma2D00, float(1e-6));
  const Lf00: TSLNode = sqrt(s00);
  const invLf00: TSLNode = float(1.0).div(Lf00);
  const Lf10: TSLNode = Sigma2D10.mul(invLf00);
  const Lf11: TSLNode = sqrt(max(Sigma2D11.sub(Lf10.mul(Lf10)), float(1e-6)));
  const invLf11: TSLNode = float(1.0).div(Lf11);
  const vL2DVal: TSLNode = vec3(invLf00, Lf10, invLf11);

  // Sum vs max projection amplitude. The cofactor / ray-integration
  // block is expensive (≈20-30 ops/vertex) and is only used in sum
  // mode. The GLSL path branches at runtime on `uProjectionMode`,
  // which the GPU handles efficiently because the uniform is warp-
  // coherent. TSL's `.select()` does NOT short-circuit — both
  // branches would otherwise materialise — so we JS-conditionally
  // emit only the path that the active blending mode uses, mirroring
  // the GLSL preprocessor's compile-time `if`. The wrapper class
  // calls `rebuildGraph()` whenever the sum/max boundary is crossed.
  const useSumProjection = !isMaxMode(config.blendingMode ?? 'additive');
  void uProjectionMode; // kept as a uniform for runtime telemetry / clone parity, even when not consumed by the graph.
  let vAmplitude2DVal: TSLNode;
  if (useSumProjection) {
    const a = S00;
    const b = S01;
    const c = S02;
    const d = S11;
    const e = S12;
    const f = S22;
    const detSigma: TSLNode = a
      .mul(d.mul(f).sub(e.mul(e)))
      .sub(b.mul(b.mul(f).sub(c.mul(e))))
      .add(c.mul(b.mul(e).sub(c.mul(d))));
    const invDet: TSLNode = float(1.0).div(max(detSigma, float(1e-12)));
    const i00: TSLNode = d.mul(f).sub(e.mul(e)).mul(invDet);
    const i11: TSLNode = a.mul(f).sub(c.mul(c)).mul(invDet);
    const i22: TSLNode = a.mul(d).sub(b.mul(b)).mul(invDet);
    const i01: TSLNode = b.mul(f).sub(c.mul(e)).negate().mul(invDet);
    const i02: TSLNode = b.mul(e).sub(c.mul(d)).mul(invDet);
    const i12: TSLNode = a.mul(e).sub(b.mul(c)).negate().mul(invDet);

    // Ray direction: ortho = (0, 0, -1); perspective = normalize(centerCam).
    const rayDirOrtho: TSLNode = vec3(0.0, 0.0, -1.0);
    const rayDirPersp: TSLNode = normalize(centerCam);
    const rayDir: TSLNode = isOrtho.select(rayDirOrtho.toVar(), rayDirPersp.toVar());
    const prx: TSLNode = i00.mul(rayDir.x).add(i01.mul(rayDir.y)).add(i02.mul(rayDir.z));
    const pry: TSLNode = i01.mul(rayDir.x).add(i11.mul(rayDir.y)).add(i12.mul(rayDir.z));
    const prz: TSLNode = i02.mul(rayDir.x).add(i12.mul(rayDir.y)).add(i22.mul(rayDir.z));
    const quad: TSLNode = max(
      rayDir.x.mul(prx).add(rayDir.y.mul(pry)).add(rayDir.z.mul(prz)),
      float(1e-8)
    );
    const sigmaRay: TSLNode = float(1.0).div(sqrt(quad));
    const rayIntegrationBoost: TSLNode = sigmaRay.mul(uRayIntegralFactor);
    vAmplitude2DVal = aAmplitude.mul(rayIntegrationBoost).mul(nearFade);
  } else {
    vAmplitude2DVal = aAmplitude.mul(nearFade);
  }

  // Eigendecomposition of Σ_2D (symmetric 2×2).
  const trace: TSLNode = Sigma2D00.add(Sigma2D11);
  const det2: TSLNode = Sigma2D00.mul(Sigma2D11).sub(Sigma2D10.mul(Sigma2D10));
  const disc: TSLNode = max(trace.mul(trace).sub(det2.mul(4.0)), float(0.0));
  const sqrtDisc: TSLNode = sqrt(disc);
  const lambda1: TSLNode = max(trace.add(sqrtDisc).mul(0.5), float(1e-6));
  const lambda2: TSLNode = max(trace.sub(sqrtDisc).mul(0.5), float(1e-6));

  // Eigenvector for major axis. Branch on whether the off-diagonal is
  // significant; near-diagonal Σ_2D picks an axis based on which
  // variance is larger.
  const offDiagSig: TSLNode = abs(Sigma2D10).greaterThan(1e-6);
  const majorOff: TSLNode = normalize(vec2(lambda1.sub(Sigma2D11), Sigma2D10));
  const majorDiag: TSLNode = Sigma2D00.greaterThanEqual(Sigma2D11).select(
    vec2(1.0, 0.0).toVar(),
    vec2(0.0, 1.0).toVar()
  );
  const majorAxis: TSLNode = offDiagSig.select(majorOff.toVar(), majorDiag.toVar());
  const minorAxis: TSLNode = vec2(majorAxis.y.negate(), majorAxis.x);

  // Quad extents and clamp.
  const extent1Raw: TSLNode = uTruncate.mul(sqrt(lambda1));
  const extent2Raw: TSLNode = uTruncate.mul(sqrt(lambda2));
  const maxExtentPx: TSLNode = max(uResolution.x, uResolution.y).mul(uMaxExtentFactor);
  const largestExtent: TSLNode = max(extent1Raw, extent2Raw);
  const clampScale: TSLNode = largestExtent
    .greaterThan(maxExtentPx)
    .select(maxExtentPx.div(largestExtent).toVar(), float(1.0));
  const extent1: TSLNode = extent1Raw.mul(clampScale);
  const extent2: TSLNode = extent2Raw.mul(clampScale);

  // Project centre to screen pixels.
  const centerScreenOrtho: TSLNode = vec2(
    uFx.mul(centerCam.x).add(uResolution.x.mul(0.5)),
    uFy.mul(centerCam.y).add(uResolution.y.mul(0.5))
  );
  const centerScreenPersp: TSLNode = vec2(
    uFx.mul(centerCam.x).mul(invZ).add(uResolution.x.mul(0.5)),
    uFy.mul(centerCam.y).mul(invZ).add(uResolution.y.mul(0.5))
  );
  const vCenterScreenVal: TSLNode = isOrtho.select(
    centerScreenOrtho.toVar(),
    centerScreenPersp.toVar()
  );

  // Expand quad in oriented screen space.
  const quadOffset: TSLNode = majorAxis
    .mul(aQuadCorner.x)
    .mul(extent1)
    .add(minorAxis.mul(aQuadCorner.y).mul(extent2));
  const screenPos: TSLNode = vCenterScreenVal.add(quadOffset);
  const ndcXY: TSLNode = screenPos.div(uResolution).mul(2.0).sub(1.0);

  // Clip-space depth from the projection matrix.
  const centerClip: TSLNode = cameraProjectionMatrix.mul(centerCam4);
  const ndcZ: TSLNode = centerClip.z.div(centerClip.w);

  // Final clipPos — with rejects routing to behind-camera (z = -2).
  // Mirror the GLSL `invalidCov2D(Sigma2D) || invalidFloat(aAmplitude)`
  // guard at shader-glsl.ts:183-186 so NaN/Inf upstream values can't
  // propagate through the Cholesky / eigendecomposition and produce
  // garbage splats or backend-specific shader behaviour.
  const invalidAmp: TSLNode = invalidFloatTSL(aAmplitude);
  const invalidCov: TSLNode = invalidFloatTSL(Sigma2D00)
    .or(invalidFloatTSL(Sigma2D10))
    .or(invalidFloatTSL(Sigma2D11));
  const validClipPos: TSLNode = vec4(ndcXY, ndcZ, float(1.0));
  const rejectClipPos: TSLNode = vec4(float(0.0), float(0.0), float(-2.0), float(1.0));
  const rejected: TSLNode = behindCamera
    .or(depthFadeReject)
    .or(coverageFadeReject)
    .or(invalidAmp)
    .or(invalidCov);
  const clipPos: TSLNode = rejected.select(rejectClipPos.toVar(), validClipPos.toVar());

  // Per-instance colour (LUT or attribute). aAmplitude doubles as
  // the colormap scalar — matches the GLSL `(aAmplitude - uScalarMin)`
  // path. In colormap mode the display range (uScalarMin/uScalarScale)
  // and gamma operate on the scalar VALUE before the LUT lookup.
  let perInstanceColor: TSLNode;
  if (config.useColormap && uColormapTex && uScalarMin && uScalarScale) {
    const tt0 = clamp(aAmplitude.sub(uScalarMin).mul(uScalarScale), 0.0, 1.0);
    // gammaOne skips the pre-LUT pow() when gamma == 1.0.
    const tt = config.gammaOne ? tt0 : tt0.pow(uInvGamma); // gamma on the value, pre-LUT
    perInstanceColor = uColormapTex.sample(vec2(tt, 0.5)).rgb;
  } else {
    perInstanceColor = aColor;
  }

  // Varyings.
  const vColor: TSLNode = varying(perInstanceColor);
  const vAmplitude2D: TSLNode = varying(vAmplitude2DVal);
  const vL2D: TSLNode = varying(vL2DVal);
  const vCenterScreen: TSLNode = varying(vCenterScreenVal);

  // ---- Fragment ----

  // Fragment uses `screenCoordinate` (= gl_FragCoord.xy in TSL) to
  // recover pixel position relative to the splat centre.
  const fragmentNode = Fn(() => {
    const d: TSLNode = vec2(screenCoordinate.xy.sub(vCenterScreen));
    // Forward substitution: solve L · y = d.
    const y0: TSLNode = d.x.mul(vL2D.x);
    const y1: TSLNode = d.y.sub(vL2D.y.mul(y0)).mul(vL2D.z);
    const mahalSq: TSLNode = y0.mul(y0).add(y1.mul(y1));
    Discard(mahalSq.greaterThan(uTruncateSq));

    // Shifted Gaussian intensity.
    const intensity: TSLNode = vAmplitude2D
      .mul(uInvOneMinusC)
      .mul(max(exp(mahalSq.mul(-0.5)).sub(uShiftC), float(0.0)));
    Discard(intensity.lessThan(1e-4));

    // GOG. Colormap mode bypasses color GOG — gamma + display-range
    // shaped the scalar VALUE (amplitude) pre-LUT (vertex stage).
    // uIntensity (gain) + uOffset apply in BOTH modes so the layer
    // intensity/offset controls work for a colormapped gsplat too (GLSL parity).
    // Colormap mode still skips the post-LUT gamma (already applied pre-LUT).
    const adjusted: TSLNode = max(vColor.mul(uIntensity).add(uOffset), vec3(0.0));
    Discard(max(adjusted.r, max(adjusted.g, adjusted.b)).lessThan(1e-4));
    // Colormap mode (gamma applied pre-LUT) OR gammaOne both skip the pow().
    const gammaColor: TSLNode =
      config.useColormap || config.gammaOne ? adjusted : adjusted.pow(vec3(uInvGamma));
    const finalColor: TSLNode = gammaColor.mul(intensity).mul(uOpacity);

    return vec4(finalColor, float(1.0));
  });

  const material = outMaterial ?? new NodeMaterial();
  material.vertexNode = clipPos;
  material.colorNode = fragmentNode();
  material.toneMapped = false;

  const blendingMode: BlendingMode = config.blendingMode ?? 'additive';
  const opacityValue = (nodes.uOpacity.value as number | undefined) ?? 1.0;
  const blendingState = getCompleteBlendingState(blendingMode, opacityValue);
  applyBlendingStateToMaterial(material, blendingState);
  return material;
}

/**
 * Snapshot adapter: build a `GSplatTSLNodes` set from a flat
 * `IUniform` record. Used by callers that don't own persistent nodes
 * (the `GSPLAT_SOURCE.webgpu` `buildMaterial` entry point and the TSL
 * parity harness). See `buildLineTSLNodesFromUniforms` for the
 * rationale + lifecycle contract.
 */
export function buildGSplatTSLNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>
): GSplatTSLNodes {
  const nodes: GSplatTSLNodes = {
    uResolution: uniform(
      (uniforms.uResolution?.value as THREE.Vector2 | undefined) ?? new THREE.Vector2(1, 1)
    ),
    uFx: uniform((uniforms.uFx?.value as number) ?? 1.0),
    uFy: uniform((uniforms.uFy?.value as number) ?? 1.0),
    uTruncate: uniform((uniforms.uTruncate?.value as number) ?? 3.0),
    uTruncateSq: uniform((uniforms.uTruncateSq?.value as number) ?? 9.0),
    uRayIntegralFactor: uniform((uniforms.uRayIntegralFactor?.value as number) ?? 1.0),
    uProjectionMode: uniform((uniforms.uProjectionMode?.value as number) ?? 0),
    uIsOrtho: uniform((uniforms.uIsOrtho?.value as number) ?? 0),
    uNearCull: uniform((uniforms.uNearCull?.value as number) ?? 1e-4),
    uMaxExtentFactor: uniform((uniforms.uMaxExtentFactor?.value as number) ?? 1.0),
    uOpacity: uniform((uniforms.uOpacity?.value as number) ?? 1.0),
    uInvGamma: uniform((uniforms.uInvGamma?.value as number) ?? 1.0),
    uIntensity: uniform((uniforms.uIntensity?.value as number) ?? 1.0),
    uOffset: uniform((uniforms.uOffset?.value as number) ?? 0.0),
    uShiftC: uniform((uniforms.uShiftC?.value as number) ?? 0.0),
    uInvOneMinusC: uniform((uniforms.uInvOneMinusC?.value as number) ?? 1.0),
  };
  if (uniforms.uColormapTex) {
    return {
      ...nodes,
      uColormapTex: texture(
        (uniforms.uColormapTex.value as THREE.Texture | null) ?? new THREE.Texture()
      ),
      uScalarMin: uniform((uniforms.uScalarMin?.value as number) ?? 0.0),
      uScalarScale: uniform((uniforms.uScalarScale?.value as number) ?? 1.0),
    };
  }
  return nodes;
}
