/**
 * GSplat material TSL factory — NodeMaterial counterpart to the
 * GLSL3 shaders in `shader-glsl.ts`.
 *
 * Renders each Gaussian splat as an oriented quad in screen space.
 * Per-vertex `aQuadCorner` (±1) scaled by the 2D covariance eigen-
 * vectors gives an axis-aligned bounding quad of the Gaussian's
 * truncation extent.
 *
 * Per-splat data comes from the RGBA32F splat texture (`uSplatTex`,
 * 4 texels/splat — layout in `rendering/element-texture-layout.ts`),
 * fetched in the vertex stage via `textureLoad` and indexed by the
 * ordering attributes (double-buffered pair):
 *   - aSortedIndex (uint) — draw-slot → storage-slot mapping
 *     (identity in Phase 1; the sort worker permutes it in Phase 2+)
 *
 * Vertex pipeline:
 *   0. Fetch center/cholesky/amplitude/color from the splat texture.
 *   1. Project centre to camera space.
 *   2. Reject splats behind camera (gl_Position = -2 NDC).
 *   3. Rotate 3D Cholesky to camera space; Σ_cam = L_cam·L_camᵀ.
 *   4. Apply near-plane fade (perspective only) + coverage fade (both projections).
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
  mix,
  fract,
  Discard,
  modelViewMatrix,
  cameraProjectionMatrix,
  texture,
  screenCoordinate,
  screenSize,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import {
  clipDepthVarying,
  glassPartitionGuardTSL,
  glassPartitionNodesFromUniforms,
  type GlassPartitionTSLNodes,
} from '../_shared/glass-partition-tsl';
import { resolveElementTextureWidth, SPLAT_TEXTURE_LAYOUT } from '../../element-texture-layout';
import {
  invalidFloatTSL,
  perspectiveNearFadeTSL,
  sanitizeAlpha,
  type TSLNode,
  sortedIndexNode,
  densityDroppedNode,
} from '../_shared/tsl-helpers';
import {
  applyBlendingStateToMaterial,
  getCompleteBlendingState,
  getGSplatNormalBlendingState,
  isNormalMode,
  isVolumetricMode,
  usesPeakProjection,
} from '../../blending-state';
import type { BlendingMode } from '../../../types/blending';
import {
  ALPHA_CLAMP,
  VOLUMETRIC_SERIES_TAU_THRESHOLD,
  VOLUMETRIC_TAU_EPS,
  VOLUMETRIC_SERIES_C1,
  VOLUMETRIC_SERIES_C2_DIVISOR,
} from '../_shared/volumetric';

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
const ivec2: (a?: TSLNode, b?: TSLNode) => TSLNode = _ivec2 as TSLNode;

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
   * Fast path: skip the `vColor * uIntensity + uOffset` GOG chain
   * (and its `max(..., vec3(0))` clamp) when the wrapper knows
   * intensity == 1 && offset == 0 — the default and most common
   * configuration. Mirrors the GLSL3 `LUXAR_NO_GOG` define and the
   * line/point factories' `noGOG` flag (three-geometry symmetry).
   * The gain-aware visibility discard keeps reading `uIntensity` —
   * under noGOG uIntensity == 1, so `max(uIntensity, 1.0)` is 1.0
   * and behavior is identical.
   */
  readonly noGOG?: boolean;
  /**
   * Luxar blending mode. GSplats premultiply intensity into RGB in
   * every mode; `normal` ADDITIONALLY emits a clamped coverage alpha
   * (premultiplied alpha-over — mirrors the GLSL
   * `LUXAR_NORMAL_PREMULT` define) and `volumetric` emits the
   * emission–absorption pair `vec4(finalColor·S(τ), 1 − e^(−τ))`
   * (mirrors `LUXAR_VOLUMETRIC`) — both pair with the One /
   * OneMinusSrcAlpha state. All other modes keep the alpha = 1.0
   * output contract. The factory derives both the fragment-output
   * style and the THREE blending state from this one field. Defaults
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
 * The colormap AND splat texture nodes are factory-time bound (TSL
 * `texture(...)` captures the THREE.Texture at call time); the
 * wrapper rebuilds the graph when either texture's identity changes
 * (see `setColormapTexture` / `updateSplatTexture` in the wrapper).
 */
export interface GSplatTSLNodes {
  /**
   * Splat data texture node (RGBA32F, 4 texels/splat). Every gsplat
   * material has one; the pool commit rebinds it per node via
   * `updateSplatTexture`.
   */
  readonly uSplatTex: TSLNode;
  readonly uResolution: TSLNode;
  readonly uPixelRatio: TSLNode;
  readonly uFx: TSLNode;
  readonly uFy: TSLNode;
  readonly uTruncate: TSLNode;
  readonly uTruncateSq: TSLNode;
  readonly uRayIntegralFactor: TSLNode;
  readonly uProjectionMode: TSLNode;
  readonly uIsOrtho: TSLNode;
  /** Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB. */
  readonly uSortedIndexSlot: TSLNode;
  readonly uDensityDrop: TSLNode;
  /** Refraction split (glass-partition-tsl.ts): mode + shared glass depth texture. */
  readonly uGlassPartition: GlassPartitionTSLNodes['uGlassPartition'];
  readonly uGlassDepth: GlassPartitionTSLNodes['uGlassDepth'];
  readonly uNearCull: TSLNode;
  readonly uMaxExtentFactor: TSLNode;
  readonly uCov2DDilation: TSLNode;
  readonly uLabelColorMode: TSLNode;
  readonly uLabelFilterIndex: TSLNode;
  readonly uOpacity: TSLNode;
  /** Absorption coefficient κ — only consumed by the volumetric output branch. */
  readonly uAbsorption: TSLNode;
  /** 1 when colors are RGBA (per-splat opacity present), else 0 — gates the volumetric alpha → optical-depth mapping. */
  readonly uHasElementAlpha: TSLNode;
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
  // Per-vertex / per-instance attributes. Splat data itself lives in
  // the splat texture; `aSortedIndex` maps the draw slot to a storage
  // slot (identity in Phase 1, permuted by the sort worker in Phase 2+).
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  const aSortedIndex: TSLNode = sortedIndexNode(nodes.uSortedIndexSlot);
  const densityDropped: TSLNode = densityDroppedNode(nodes.uDensityDrop, aSortedIndex);

  // Uniform leaves come from the wrapper. No per-render callbacks:
  // mutations to `material.uniforms.X.value` already route to
  // `node.value` via `proxyIUniform`.
  const uSplatTex = nodes.uSplatTex;
  const uResolution = nodes.uResolution;
  const uFx = nodes.uFx;
  const uFy = nodes.uFy;
  const uTruncate = nodes.uTruncate;
  const uRayIntegralFactor = nodes.uRayIntegralFactor;
  const uProjectionMode = nodes.uProjectionMode;
  const uIsOrtho = nodes.uIsOrtho;
  const uNearCull = nodes.uNearCull;
  const uMaxExtentFactor = nodes.uMaxExtentFactor;
  const uCov2DDilation = nodes.uCov2DDilation;
  // Keep the historical framebuffer-pixel low-pass below 1× render scale.
  const dilationPixelRatio = nodes.uPixelRatio.max(float(1.0));
  const cov2DDilation = uCov2DDilation.mul(dilationPixelRatio).mul(dilationPixelRatio);
  const uColormapTex = config.useColormap ? nodes.uColormapTex : null;
  const uScalarMin = config.useColormap ? nodes.uScalarMin : null;
  const uScalarScale = config.useColormap ? nodes.uScalarScale : null;
  const uOpacity = nodes.uOpacity;
  const uAbsorption = nodes.uAbsorption;
  const uHasElementAlpha = nodes.uHasElementAlpha;
  const uInvGamma = nodes.uInvGamma;
  const uIntensity = nodes.uIntensity;
  const uOffset = nodes.uOffset;
  const uShiftC = nodes.uShiftC;
  const uInvOneMinusC = nodes.uInvOneMinusC;
  const uTruncateSq = nodes.uTruncateSq;
  const uLabelColorMode = nodes.uLabelColorMode;
  const uLabelFilterIndex = nodes.uLabelFilterIndex;

  // ---- Vertex computation ----
  //
  // The ENTIRE vertex stage is traced inside a single Fn() body with
  // explicit `.toVar()` statements, mirroring the fragment stage. This
  // is load-bearing, not style: as a free expression tree, TSL
  // materializes a shared subexpression at its FIRST traversal use —
  // and when that first use sits inside a `.select(a.toVar(), …)`
  // branch, the assignment is emitted INSIDE the generated if-block.
  // The eigendecomposition values (Σ2D diagonal, trace, √disc, λ1/λ2)
  // were first consumed by the major-axis branch, so on the
  // near-diagonal path (isotropic splats: off-diagonal == 0) every one
  // of them was read UNINITIALIZED — undefined behaviour that rendered
  // as invisible or garbage splats on the WebGPU backend (both native
  // WGSL and the WebGL2 fallback zero/garbage-fill locals). Inside
  // Fn(), statements emit in trace order, unconditionally.

  // Varyings are declared up front and `.assign()`ed inside the vertex
  // body (the TSL pattern for Fn-traced vertex stages).
  const vColor: TSLNode = varying(vec3(float(0.0), float(0.0), float(0.0)));
  const vAmplitude2D: TSLNode = varying(float(0.0));
  const vClipZW: TSLNode = clipDepthVarying();
  const vAlpha: TSLNode = varying(float(1.0));
  const vL2D: TSLNode = varying(vec3(float(0.0), float(0.0), float(0.0)));
  const vCenterScreen: TSLNode = varying(vec2(float(0.0), float(0.0)));

  const vertexBody = Fn(() => {
    // === Splat-texture fetch prologue ===
    // Four textureLoad reads reconstruct the per-splat values into the
    // exact local names the math below has always used — zero changes
    // downstream of this block. Every value is a `.toVar()` STATEMENT
    // (the Fn house rule; see the block comment above). The texture
    // width is a multiple of 4 (element-texture-layout.ts), so a splat's
    // 4 texels share one row and only x advances.
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
    const aCenter: TSLNode = vec3(splatT0).toVar(); // 3D world centre
    const aAmplitude: TSLNode = splatT0.w.toVar();
    const aCholesky01: TSLNode = splatT1.xy.toVar(); // [L00, L10]
    const aCholesky23: TSLNode = splatT1.zw.toVar(); // [L11, L20]
    const aCholesky45: TSLNode = splatT2.xy.toVar(); // [L21, L22]
    const aColor: TSLNode = vec3(splatT2.z, splatT2.w, splatT3.x).toVar();
    // Per-splat opacity (texel3.y; the writer stamps 1.0 for RGB data).
    const aAlpha: TSLNode = splatT3.y.toVar();
    const aLabelIndex: TSLNode = splatT3.z.toVar();

    // Centre in camera space.
    const centerCam4: TSLNode = modelViewMatrix.mul(vec4(aCenter, 1.0)).toVar();
    const centerCam: TSLNode = vec3(centerCam4).toVar();

    const zDepth: TSLNode = centerCam.z.negate().toVar();

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
    const L_cam: TSLNode = R.mul(L3D).toVar();
    // Sigma_cam = L_cam · L_camᵀ. We need explicit row access on the
    // result for downstream cofactor / variance math.
    const SigmaCam: TSLNode = L_cam.mul(L_cam.transpose()).toVar();

    // Unified near handling (shared perspectiveNearFadeTSL; matches the
    // point + line graphs and the GLSL twin): behind-camera fades to 0
    // (subsumes the old standalone behindCamera reject), the near-plane
    // approach fades across [uNearCull, 2*uNearCull], ortho passes
    // through to NDC clipping.
    // uNearCull is scene-bounds-scaled; the 1e-20 floor only guards
    // uNearCull == 0 (degenerate smoothstep) — an absolute 1e-4 floor
    // overrode the scene-relative value on tiny-unit scenes and faded
    // out the whole scene. GLSL twin: shader-glsl.ts.
    const depthFade: TSLNode = perspectiveNearFadeTSL(
      uIsOrtho,
      centerCam.z,
      max(uNearCull, float(1e-20))
    ).toVar();
    const depthFadeReject: TSLNode = depthFade.lessThan(0.01);

    // Coverage fade — computed UNCONDITIONALLY (GLSL twin updated in
    // lockstep): below maxExtent*0.5 the smoothstep is 0 and the fade
    // is a no-op, so no size gate is needed. The former absolute
    // maxLateralVar > 0.01 gate skipped the fade for splats with
    // spatial sigma < 0.1 world units while the extent clamp still
    // applied (hard-edged clamped rectangles on deep zoom). Applies
    // in BOTH projections: ortho projected size is depth-independent
    // (divisor 1).
    const maxLateralVar: TSLNode = max(
      SigmaCam.element(int(0)).element(int(0)),
      max(SigmaCam.element(int(1)).element(int(1)), SigmaCam.element(int(2)).element(int(2)))
    ).toVar();
    const isOrtho: TSLNode = int(uIsOrtho).equal(int(1)).toVar();
    // 1e-20 floors are pure div-by-zero/sqrt guards, NOT scale floors:
    // maxLateralVar is a WORLD-unit² variance and the old absolute 1e-8
    // floor inflated valid tiny-unit variances up to 1e-4 world units,
    // exploding projectedExtent and coverage-culling every splat;
    // zDepth is bounded below by the scene-relative near fade. GLSL
    // twin: shader-glsl.ts (expressions match exactly).
    const projectedExtent: TSLNode = uFx
      .mul(sqrt(max(maxLateralVar, float(1e-20))))
      .mul(uTruncate)
      .div(isOrtho.select(float(1.0), max(zDepth, float(1e-20))));
    const maxExtent: TSLNode = max(uResolution.x, uResolution.y).mul(uMaxExtentFactor);
    const coverageFade: TSLNode = float(1.0)
      .sub(smoothstep(maxExtent.mul(0.5), maxExtent, projectedExtent))
      .toVar();
    const coverageFadeReject: TSLNode = coverageFade.lessThan(0.01);

    const nearFade: TSLNode = min(depthFade, coverageFade).toVar();

    // Projection Jacobian. mat3x2 in GLSL = three vec2 columns; we
    // represent it as three independent vec2 nodes to sidestep TSL's
    // missing mat3x2 type. JS0/JS1/JS2 form the matrix M = J·Σ_cam.
    // 1e-20 = exact-zero guard only (GLSL twin divides unguarded); the
    // near-fade reject already bounds zDepth at ~uNearCull
    // (scene-relative) — an absolute 1e-8 floor would distort the
    // Jacobian on sub-1e-8-unit scenes.
    const invZ: TSLNode = float(1.0)
      .div(max(zDepth, float(1e-20)))
      .toVar();
    const invZ2: TSLNode = invZ.mul(invZ);
    // Perspective Jacobian columns.
    const J0p: TSLNode = vec2(uFx.mul(invZ), float(0.0));
    const J1p: TSLNode = vec2(float(0.0), uFy.mul(invZ));
    const J2p: TSLNode = vec2(uFx.mul(centerCam.x).mul(invZ2), uFy.mul(centerCam.y).mul(invZ2));
    // Orthographic Jacobian (depth-independent).
    const J0o: TSLNode = vec2(uFx, float(0.0));
    const J1o: TSLNode = vec2(float(0.0), uFy);
    const J2o: TSLNode = vec2(float(0.0), float(0.0));
    const J0: TSLNode = isOrtho.select(J0o, J0p).toVar();
    const J1: TSLNode = isOrtho.select(J1o, J1p).toVar();
    const J2: TSLNode = isOrtho.select(J2o, J2p).toVar();

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
    const JS0: TSLNode = J0.mul(S00).add(J1.mul(S01)).add(J2.mul(S02)).toVar();
    const JS1: TSLNode = J0.mul(S10).add(J1.mul(S11)).add(J2.mul(S12)).toVar();
    const JS2: TSLNode = J0.mul(S20).add(J1.mul(S21)).add(J2.mul(S22)).toVar();
    // Materialized UNCONDITIONALLY — these are exactly the shared
    // values that used to be first-consumed inside the major-axis
    // branch and read uninitialized on the near-diagonal path.
    const Sigma2D00: TSLNode = JS0.x.mul(J0.x).add(JS1.x.mul(J1.x)).add(JS2.x.mul(J2.x)).toVar();
    const Sigma2D10: TSLNode = JS0.x.mul(J0.y).add(JS1.x.mul(J1.y)).add(JS2.x.mul(J2.y)).toVar();
    const Sigma2D11: TSLNode = JS0.y.mul(J0.y).add(JS1.y.mul(J1.y)).add(JS2.y.mul(J2.y)).toVar();

    // 2D low-pass dilation (standard 3DGS anti-aliasing) — GLSL twin in
    // shader-glsl.ts. Widen the diagonal so every splat covers ≥ ~1px, so
    // near-degenerate (edge-on/flat) splats render as a soft ellipse instead
    // of a razor-thin sub-pixel spike. Applied to the shared `.toVar()`s before
    // the Cholesky + eigen extent below so footprint and quad stay consistent.
    // Diagonal only — the off-diagonal would rotate/shear the ellipse.
    //
    // ENERGY COMPENSATION (Mip-Splatting) — GLSL twin carries the derivation:
    // widening without touching the peak creates light, by
    // sqrt(detDilated/detRaw); the compensation below is its reciprocal.
    // Consumed ONLY by the SUM branch's amplitude below, so — per this file's
    // JS-conditional-emission rule (see the projection-mode comment further
    // down) — the determinants are emitted only when that branch is live.
    // detRaw2D must still be captured BEFORE the diagonal dilation.
    const surfaceMode = usesPeakProjection(config.blendingMode ?? 'additive');
    const useSumProjection = !surfaceMode;
    const detRaw2D: TSLNode | null = useSumProjection
      ? Sigma2D00.mul(Sigma2D11).sub(Sigma2D10.mul(Sigma2D10)).toVar()
      : null;
    Sigma2D00.addAssign(cov2DDilation);
    Sigma2D11.addAssign(cov2DDilation);
    let dilationCompensation: TSLNode | null = null;
    if (useSumProjection && detRaw2D) {
      const detDilated2D: TSLNode = Sigma2D00.mul(Sigma2D11).sub(Sigma2D10.mul(Sigma2D10)).toVar();
      dilationCompensation = sqrt(
        max(detRaw2D, float(0.0)).div(max(detDilated2D, float(1e-12)))
      ).toVar();
    }

    // 2D Cholesky factorisation: [invL00, L10, invL11] for fragment-side
    // MUL-instead-of-DIV.
    const s00: TSLNode = max(Sigma2D00, float(1e-6));
    const Lf00: TSLNode = sqrt(s00);
    const invLf00: TSLNode = float(1.0).div(Lf00).toVar();
    const Lf10: TSLNode = Sigma2D10.mul(invLf00).toVar();
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
    // Peak (2D-projected) projection for SURFACE modes (max / normal /
    // opaque — `usesPeakProjection`); sum ray-integral only for emissive
    // (additive/luminous). GLSL twin: applyBlendingMode sets
    // uProjectionMode=1 for every surface mode. Surface compositing
    // wants the projected 2D-Gaussian peak, not the emissive
    // line-integral (whose ~2.4×·sigmaRay boost would over-brighten and,
    // in normal mode, saturate coverage-alpha to opaque and streak at
    // grazing angles). The wrapper calls rebuildGraph() when this
    // boundary flips. (`surfaceMode` / `useSumProjection` are computed above,
    // where the dilation-compensation emission needs the same gate.)
    let vAmplitude2DVal: TSLNode;
    if (useSumProjection) {
      // SCALE-FREE inversion (GLSL twin: shader-glsl.ts): normalize
      // Σ_cam by its mean diagonal variance s = trace/3 before the
      // cofactor inverse. det(Σ) is world-units⁶ and under/overflows
      // float32 on tiny/huge-unit scenes (GPUs flush denormals to
      // zero), which turned the absolute 1e-12 clamp into garbage
      // Σ⁻¹. With Σn = Σ/s the determinant and ray quadratic are O(1)
      // at any scale, so the 1e-12 / 1e-8 floors act as scale-free
      // condition-number guards; sigmaRay = sqrt(s / quadN) restores
      // the world-unit result exactly. 1e-30 on s guards an all-zero
      // covariance only.
      const sTrace: TSLNode = max(
        S00.add(S11)
          .add(S22)
          .mul(1.0 / 3.0),
        float(1e-30)
      ).toVar();
      const invS: TSLNode = float(1.0).div(sTrace).toVar();
      const a = S00.mul(invS).toVar();
      const b = S01.mul(invS).toVar();
      const c = S02.mul(invS).toVar();
      const d = S11.mul(invS).toVar();
      const e = S12.mul(invS).toVar();
      const f = S22.mul(invS).toVar();
      const detSigma: TSLNode = a
        .mul(d.mul(f).sub(e.mul(e)))
        .sub(b.mul(b.mul(f).sub(c.mul(e))))
        .add(c.mul(b.mul(e).sub(c.mul(d))));
      const invDet: TSLNode = float(1.0)
        .div(max(detSigma, float(1e-12)))
        .toVar();
      const i00: TSLNode = d.mul(f).sub(e.mul(e)).mul(invDet).toVar();
      const i11: TSLNode = a.mul(f).sub(c.mul(c)).mul(invDet).toVar();
      const i22: TSLNode = a.mul(d).sub(b.mul(b)).mul(invDet).toVar();
      const i01: TSLNode = b.mul(f).sub(c.mul(e)).negate().mul(invDet).toVar();
      const i02: TSLNode = b.mul(e).sub(c.mul(d)).mul(invDet).toVar();
      const i12: TSLNode = a.mul(e).sub(b.mul(c)).negate().mul(invDet).toVar();

      // Ray direction: ortho = (0, 0, -1); perspective = normalize(centerCam).
      const rayDirOrtho: TSLNode = vec3(0.0, 0.0, -1.0);
      const rayDirPersp: TSLNode = normalize(centerCam);
      const rayDir: TSLNode = isOrtho.select(rayDirOrtho, rayDirPersp).toVar();
      const prx: TSLNode = i00.mul(rayDir.x).add(i01.mul(rayDir.y)).add(i02.mul(rayDir.z));
      const pry: TSLNode = i01.mul(rayDir.x).add(i11.mul(rayDir.y)).add(i12.mul(rayDir.z));
      const prz: TSLNode = i02.mul(rayDir.x).add(i12.mul(rayDir.y)).add(i22.mul(rayDir.z));
      // quad is rᵀ Σn⁻¹ r (normalized space); un-normalize via
      // sqrt(sTrace) — see the scale-free inversion note above.
      const quad: TSLNode = max(
        rayDir.x.mul(prx).add(rayDir.y.mul(pry)).add(rayDir.z.mul(prz)),
        float(1e-8)
      );
      const sigmaRay: TSLNode = sqrt(sTrace).div(sqrt(quad));
      const rayIntegrationBoost: TSLNode = sigmaRay.mul(uRayIntegralFactor);
      // dilationCompensation keeps the screen-integrated light invariant under
      // the 2D low-pass (derivation at its definition). Sum projection only —
      // this branch's quantity IS that integral. The peak branch reports a
      // peak, not an integral, and is left alone.
      vAmplitude2DVal = aAmplitude
        .mul(rayIntegrationBoost)
        .mul(nearFade)
        .mul(dilationCompensation!);
    } else {
      vAmplitude2DVal = aAmplitude.mul(nearFade);
    }

    // Eigendecomposition of Σ_2D (symmetric 2×2). Every shared value
    // is a Var STATEMENT here — emitted unconditionally, before any
    // consumer branch (the whole point of the Fn rewrite).
    const trace: TSLNode = Sigma2D00.add(Sigma2D11).toVar();
    const det2: TSLNode = Sigma2D00.mul(Sigma2D11).sub(Sigma2D10.mul(Sigma2D10));
    const disc: TSLNode = max(trace.mul(trace).sub(det2.mul(4.0)), float(0.0));
    const sqrtDisc: TSLNode = sqrt(disc).toVar();
    const lambda1: TSLNode = max(trace.add(sqrtDisc).mul(0.5), float(1e-6)).toVar();
    const lambda2: TSLNode = max(trace.sub(sqrtDisc).mul(0.5), float(1e-6)).toVar();

    // Eigenvector for major axis. Branch on whether the off-diagonal is
    // significant; near-diagonal Σ_2D picks an axis based on which
    // variance is larger.
    const offDiagSig: TSLNode = abs(Sigma2D10).greaterThan(1e-6);
    const majorOff: TSLNode = normalize(vec2(lambda1.sub(Sigma2D11), Sigma2D10));
    const majorDiag: TSLNode = Sigma2D00.greaterThanEqual(Sigma2D11).select(
      vec2(1.0, 0.0),
      vec2(0.0, 1.0)
    );
    const majorAxis: TSLNode = offDiagSig.select(majorOff, majorDiag).toVar();
    const minorAxis: TSLNode = vec2(majorAxis.y.negate(), majorAxis.x).toVar();

    // Quad extents and clamp.
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

    // Project centre to screen pixels.
    const centerScreenOrtho: TSLNode = vec2(
      uFx.mul(centerCam.x).add(uResolution.x.mul(0.5)),
      uFy.mul(centerCam.y).add(uResolution.y.mul(0.5))
    );
    const centerScreenPersp: TSLNode = vec2(
      uFx.mul(centerCam.x).mul(invZ).add(uResolution.x.mul(0.5)),
      uFy.mul(centerCam.y).mul(invZ).add(uResolution.y.mul(0.5))
    );
    const vCenterScreenVal: TSLNode = isOrtho.select(centerScreenOrtho, centerScreenPersp).toVar();

    // Expand quad in oriented screen space.
    const quadOffset: TSLNode = majorAxis
      .mul(aQuadCorner.x)
      .mul(extent1)
      .add(minorAxis.mul(aQuadCorner.y).mul(extent2));
    const screenPos: TSLNode = vCenterScreenVal.add(quadOffset);
    const ndcXY: TSLNode = screenPos.div(uResolution).mul(2.0).sub(1.0);

    // Clip-space depth from the projection matrix.
    const centerClip: TSLNode = cameraProjectionMatrix.mul(centerCam4).toVar();
    const ndcZ: TSLNode = centerClip.z.div(centerClip.w);

    // Final clipPos — with rejects routing to behind-camera (z = -2).
    // Mirror the GLSL `invalidCov2D(Sigma2D) || invalidFloat(aAmplitude)`
    // guard at shader-glsl.ts so NaN/Inf upstream values can't
    // propagate through the Cholesky / eigendecomposition and produce
    // garbage splats or backend-specific shader behaviour.
    const invalidAmp: TSLNode = invalidFloatTSL(aAmplitude);
    const invalidCov: TSLNode = invalidFloatTSL(Sigma2D00)
      .or(invalidFloatTSL(Sigma2D10))
      .or(invalidFloatTSL(Sigma2D11));
    const validClipPos: TSLNode = vec4(ndcXY, ndcZ, float(1.0));
    const rejectClipPos: TSLNode = vec4(float(0.0), float(0.0), float(-2.0), float(1.0));
    // behindCamera is subsumed by depthFadeReject (perspective fade = 0
    // behind the camera; ortho behind-camera falls to NDC clipping).
    const labelRejected: TSLNode = int(uLabelFilterIndex)
      .greaterThan(int(0))
      .and(int(aLabelIndex.add(0.5)).notEqual(int(uLabelFilterIndex)));
    const rejected: TSLNode = depthFadeReject
      .or(coverageFadeReject)
      .or(invalidAmp)
      .or(invalidCov)
      .or(labelRejected)
      .or(densityDropped);

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

    // Assign varyings (declared outside the Fn; see above).
    const categoricalColor = vec3(
      fract(aLabelIndex.mul(0.61803398875)),
      fract(aLabelIndex.mul(0.38196601125)),
      fract(aLabelIndex.mul(0.75487766625))
    )
      .mul(0.75)
      .add(0.25);
    const useLabelColor = int(uLabelColorMode).equal(int(1)).and(aLabelIndex.greaterThan(0.0));
    vColor.assign(useLabelColor.select(categoricalColor, perInstanceColor));
    // Sanitized like the GLSL twin: NaN/Inf route to the 1.0 opaque
    // identity, finite values clamp to [0, 1] (alpha is load-bearing and
    // feeds optical depth under volumetric).
    vAlpha.assign(sanitizeAlpha(aAlpha));
    vAmplitude2D.assign(vAmplitude2DVal);
    vL2D.assign(vL2DVal);
    vCenterScreen.assign(vCenterScreenVal);

    const clipOut: TSLNode = rejected.select(rejectClipPos, validClipPos).toVar();
    vClipZW.assign(clipOut.zw);
    return clipOut;
  });

  const clipPos: TSLNode = vertexBody();
  void uProjectionMode; // kept as a uniform for runtime telemetry / clone parity, even when not consumed by the graph.

  // ---- Fragment ----

  // Fragment uses `screenCoordinate` (= gl_FragCoord.xy in TSL) to
  // recover pixel position relative to the splat centre.
  const fragmentNode = Fn(() => {
    // Refraction split partition FIRST: a fragment on the wrong side of the glass
    // costs nothing further (glass-partition-tsl.ts; GLSL twin at the top of main()).
    glassPartitionGuardTSL(nodes, vClipZW);
    // `screenCoordinate` is TOP-LEFT-origin on BOTH backends (three
    // normalizes: the WebGL fallback emits `size.y - gl_FragCoord.y`,
    // native WGSL's position builtin is already top-left), but
    // vCenterScreen is BOTTOM-LEFT window coords (it feeds y-up NDC in
    // the vertex). Un-flip back to bottom-left so `d` matches the GLSL
    // reference (`gl_FragCoord.xy - vCenterScreen`) exactly — without
    // this the Gaussian is evaluated around a center MIRRORED about the
    // horizontal midline, so off-center splats discard their whole quad
    // (invisible) and midline-crossing quads show garbage edges. The
    // centered, mirror-symmetric parity fixtures are blind to a y-flip;
    // the off-center parity variant exists to catch exactly this.
    // The un-flip term MUST be `screenSize` (the bound render target's
    // size — exactly what the builder's flip used), NOT the app-stamped
    // uResolution: they differ whenever the target isn't
    // drawing-buffer-sized (e.g. SSAA multiplied), and only screenSize
    // reconstructs gl_FragCoord.y byte-identically to the GLSL twin in
    // every configuration.
    const fragCoordBL: TSLNode = vec2(screenCoordinate.x, screenSize.y.sub(screenCoordinate.y));
    const d: TSLNode = vec2(fragCoordBL.sub(vCenterScreen));
    // Forward substitution: solve L · y = d.
    const y0: TSLNode = d.x.mul(vL2D.x);
    const y1: TSLNode = d.y.sub(vL2D.y.mul(y0)).mul(vL2D.z);
    const mahalSq: TSLNode = y0.mul(y0).add(y1.mul(y1));
    Discard(mahalSq.greaterThan(uTruncateSq));

    // Shifted Gaussian intensity.
    const rawIntensity: TSLNode = vAmplitude2D
      .mul(uInvOneMinusC)
      .mul(max(exp(mahalSq.mul(-0.5)).sub(uShiftC), float(0.0)));
    // Per-splat opacity (color alpha; 1.0 for RGB data): linear factor in
    // every mode, mapped into optical DENSITY w = −ln(1−a) in volumetric
    // (GLSL twin; clamp = ALPHA_CLAMP from ../_shared/volumetric and sits INSIDE the
    // expression — mix evaluates both lanes, so the log argument must be
    // NaN-free even when the gate is 0).
    const volumetricGraph = isVolumetricMode(config.blendingMode ?? 'additive');
    const alphaFactor: TSLNode = volumetricGraph
      ? mix(float(1.0), min(vAlpha, float(ALPHA_CLAMP)).oneMinus().log().negate(), uHasElementAlpha)
      : vAlpha;
    const intensity: TSLNode = rawIntensity.mul(alphaFactor).toVar();
    // Alpha is folded in, so a ~zero-alpha splat discards in every mode.
    // GAIN-AWARE gate (GLSL parity): brightness = intensity * uIntensity
    // * color, so a high gain must relax the visibility floor —
    // max(uIntensity, 1) keeps gain <= 1 exactly at the historical
    // threshold. See shader-glsl.ts for the full rationale.
    Discard(intensity.mul(max(uIntensity, float(1.0))).lessThan(1e-4));

    // GOG. Colormap mode bypasses color GOG — gamma + display-range
    // shaped the scalar VALUE (amplitude) pre-LUT (vertex stage).
    // uIntensity (gain) + uOffset apply in BOTH modes so the layer
    // intensity/offset controls work for a colormapped gsplat too (GLSL parity).
    // Colormap mode still skips the post-LUT gamma (already applied pre-LUT).
    // Fast path: when the wrapper knows intensity==1 && offset==0, the
    // mul/add/clamp chain is identity for non-negative vColor (noGOG —
    // mirrors the line/point factories; the gain-aware discard above
    // keeps reading uIntensity, which is exactly 1 in that regime).
    const adjusted: TSLNode = config.noGOG
      ? vColor
      : max(vColor.mul(uIntensity).add(uOffset), vec3(0.0));
    const maxAdjusted: TSLNode = max(adjusted.r, max(adjusted.g, adjusted.b));
    const volumetric = volumetricGraph;
    // Volumetric optical depth τ = κ·opacity·intensity (pre-GOG density
    // scalar × opacity-as-density; GLSL LUXAR_VOLUMETRIC twin). Built
    // only on the volumetric graph — JS-conditional like the other
    // structural branches.
    const tau: TSLNode | null = volumetric
      ? uAbsorption.mul(uOpacity).mul(intensity).toVar()
      : null;
    if (volumetric && tau) {
      // τ is color-independent — a black splat still absorbs, so the
      // zero-color discard only fires when τ is negligible too.
      Discard(maxAdjusted.lessThan(1e-4).and(tau.lessThan(1e-4)));
    } else {
      Discard(maxAdjusted.lessThan(1e-4));
    }
    // Colormap mode (gamma applied pre-LUT) OR gammaOne both skip the pow().
    const gammaColor: TSLNode =
      config.useColormap || config.gammaOne ? adjusted : adjusted.pow(vec3(uInvGamma));
    const finalColor: TSLNode = gammaColor.mul(intensity).mul(uOpacity);

    if (isNormalMode(config.blendingMode ?? 'additive')) {
      // 'normal': premultiplied alpha-over (GLSL LUXAR_NORMAL_PREMULT
      // twin). RGB carries the full unclamped HDR contribution; alpha
      // carries a CLAMPED coverage term for the One/OneMinusSrcAlpha
      // state below. Never via material.premultipliedAlpha — NodeMaterial
      // would auto-inject a second RGB×alpha on this path.
      const coverage: TSLNode = clamp(intensity.mul(uOpacity), float(0.0), float(1.0));
      return vec4(finalColor, coverage);
    }
    if (volumetric && tau) {
      // 'volumetric': emission–absorption (GLSL LUXAR_VOLUMETRIC twin).
      // RGB carries the self-screened emission (S(τ) = (1−e^(−τ))/τ);
      // alpha = 1 − e^(−τ) for the One/OneMinusSrcAlpha state. Series
      // for τ < 1e-3 keeps S well-conditioned through τ → 0 (κ = 0 ⇒
      // α = 0, S = 1 — bit-identical RGB arithmetic to additive; dst-alpha
      // differs, invisible on the alpha:false canvas). `.select`
      // materializes both sides — fine for this cheap scalar math
      // (unlike the vertex projection branches, which stay
      // JS-conditional).
      // Threshold/coefficients/guard from ./math (the ALPHA_CLAMP
      // single-source pattern — same constants as the GLSL template and
      // the volumetric-math unit test).
      const alpha: TSLNode = float(1.0).sub(exp(tau.negate()));
      const series: TSLNode = float(1.0)
        .sub(tau.mul(VOLUMETRIC_SERIES_C1))
        .add(tau.mul(tau).div(VOLUMETRIC_SERIES_C2_DIVISOR));
      const screen: TSLNode = tau
        .lessThan(VOLUMETRIC_SERIES_TAU_THRESHOLD)
        .select(series, alpha.div(max(tau, VOLUMETRIC_TAU_EPS)));
      return vec4(finalColor.mul(screen), alpha);
    }
    // All other modes keep the alpha=1.0 contract: additive/luminous
    // rely on SrcAlpha being the IDENTITY factor (what makes the shared
    // AdditiveBlending state equal the linear One+One sum — alpha is
    // consumed, not ignored), and max compares premultiplied RGB
    // contributions directly. Mirrors the GLSL twin's #else branch.
    return vec4(finalColor, float(1.0));
  });

  const material = outMaterial ?? new NodeMaterial();
  material.vertexNode = clipPos;
  material.colorNode = fragmentNode();
  material.toneMapped = false;

  const blendingMode: BlendingMode = config.blendingMode ?? 'additive';
  const opacityValue = (nodes.uOpacity.value as number | undefined) ?? 1.0;
  // 'normal' takes the gsplat-specific premultiplied state (symmetric
  // alpha channel — separate alpha-equation state trips gl.getError()
  // under the WebGPU→WebGL2 bridge); 'volumetric' gets the identical
  // One/OneMinusSrcAlpha state via the shared helper's own branch (its
  // fragment emits a real absorption alpha); every other mode keeps the
  // shared state, equivalent because the shader emits alpha = 1 there.
  const blendingState = isNormalMode(blendingMode)
    ? getGSplatNormalBlendingState()
    : getCompleteBlendingState(blendingMode, opacityValue);
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
    // Splat data texture — bound from the caller's uniform when
    // present (harness / material paths), else a 4×1 RGBA32F
    // placeholder so codegen-only consumers still build a valid graph.
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
    // Deliberately NOT `GSPLAT_DEFAULT_TRUNCATION_RADIUS`. This adapter is
    // harness/snapshot only (production materials set the real default in
    // their own constructor), and its values must stay equal to the GLSL twin
    // in `tests/e2e/harnesses/tsl-harness/gsplats.ts` or `tsl-shader-parity`
    // pixel-compares diverge. Note 9.0 is 3.0² — the pair must be changed
    // together, and in the harness too.
    uTruncate: uniform((uniforms.uTruncate?.value as number) ?? 3.0),
    uTruncateSq: uniform((uniforms.uTruncateSq?.value as number) ?? 9.0),
    uRayIntegralFactor: uniform((uniforms.uRayIntegralFactor?.value as number) ?? 1.0),
    uProjectionMode: uniform((uniforms.uProjectionMode?.value as number) ?? 0),
    uIsOrtho: uniform((uniforms.uIsOrtho?.value as number) ?? 0),
    uSortedIndexSlot: uniform((uniforms.uSortedIndexSlot?.value as number) ?? 0),
    uDensityDrop: uniform((uniforms.uDensityDrop?.value as number) ?? 0),
    ...glassPartitionNodesFromUniforms(uniforms),
    uNearCull: uniform((uniforms.uNearCull?.value as number) ?? 1e-4),
    uMaxExtentFactor: uniform((uniforms.uMaxExtentFactor?.value as number) ?? 1.0),
    // Neutral fallback 0 (no dilation) — matches GLSL's missing-uniform default,
    // like uMaxExtentFactor's neutral 1.0 above; this adapter is harness/snapshot
    // only (production materials set the 0.3 default in their own constructor).
    uCov2DDilation: uniform((uniforms.uCov2DDilation?.value as number) ?? 0),
    uOpacity: uniform((uniforms.uOpacity?.value as number) ?? 1.0),
    uAbsorption: uniform((uniforms.uAbsorption?.value as number) ?? 1.0),
    uHasElementAlpha: uniform((uniforms.uHasElementAlpha?.value as number) ?? 0),
    uInvGamma: uniform((uniforms.uInvGamma?.value as number) ?? 1.0),
    uIntensity: uniform((uniforms.uIntensity?.value as number) ?? 1.0),
    uOffset: uniform((uniforms.uOffset?.value as number) ?? 0.0),
    uLabelColorMode: uniform((uniforms.uLabelColorMode?.value as number) ?? 0),
    uLabelFilterIndex: uniform((uniforms.uLabelFilterIndex?.value as number) ?? 0),
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
