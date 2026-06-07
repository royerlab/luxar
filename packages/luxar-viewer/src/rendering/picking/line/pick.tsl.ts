/**
 * Line picking material TSL factory — NodeMaterial counterpart to
 * `LINE_PICK_SOURCE` in `shaders.ts`.
 *
 * Mirrors line.tsl's sprite-expansion math, but the fragment outputs:
 *   - R: nodeId (set via uniform)
 *   - G: elementId (instance index)
 *   - B: brightness (perpendicular falloff × cap × widthScale × widthFade)
 *   - A: 1.0
 *
 * Depth = 1.0 - brightness (brightness-as-depth tie-breaking).
 *
 * Lines use FULL width for picking (no half-radius like points) —
 * thin lines would be impossible to pick otherwise.
 *
 * @module rendering/picking/line/pick.tsl
 */

import * as THREE from 'three';
import {
  Fn,
  If,
  uniform,
  attribute,
  varying,
  instanceIndex,
  vec2,
  vec4,
  float,
  max,
  min,
  clamp,
  mix,
  length,
  step,
  exp,
  modelViewMatrix,
  cameraProjectionMatrix,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { sanitizeNonNegative, type TSLNode } from '../../materials/_shared/tsl-helpers';

/**
 * Pre-created TSL leaf nodes supplied by the wrapper class. Same
 * pattern as `PointPickTSLNodes` / `GSplatPickTSLNodes`: consumers
 * own the `UniformNode`s and the factory references them directly,
 * avoiding the `.onUpdate('render')` callback churn.
 */
export interface LinePickTSLNodes {
  /**
   * uFOV is unused by the shader after the pixel-scale precomputation
   * but is kept here so the wrapper class's uniform table remains
   * structurally identical to the visual material's.
   */
  readonly uFOV: TSLNode;
  readonly uResolution: TSLNode;
  readonly uIsOrtho: TSLNode;
  readonly uNodeId: TSLNode;
  readonly uNearCull: TSLNode;
  readonly uMaxLinePixelWidth: TSLNode;
  /** = resolution.y / tan(fov * 0.5), precomputed by the wrapper. */
  readonly uPerspectiveLineScale: TSLNode;
  /** = 2 * resolution.y / frustumHeight, precomputed by the wrapper. */
  readonly uOrthoLineScale: TSLNode;
}

/**
 * Line picking material TSL factory.
 *
 * Consumes pre-created `UniformNode` references via `nodes`; the
 * wrapper class (`LinePickingTSLMaterial`) owns those nodes and
 * exposes them through `material.uniforms` as `IUniform`-shaped
 * getter/setter proxies. The harness / `shaders.ts` ShaderSource
 * registry constructs the nodes from a plain `uniforms` record via
 * {@link buildLinePickTSLNodesFromUniforms}.
 */
/**
 * Per-build configuration for the line-pick factory.
 */
export interface LinePickTSLConfig {
  /**
   * Camera projection mode at build time — mirrors `LineTSLConfig`.
   * When true, only the ortho pixel-width branch is emitted; when
   * false or undefined, only the perspective branch.
   */
  readonly isOrtho?: boolean;
}

export function linePickWebGPUFactory(
  nodes: LinePickTSLNodes,
  config: LinePickTSLConfig = {},
  outMaterial?: NodeMaterial
): NodeMaterial {
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  const aStartPos: TSLNode = attribute<'vec3'>('aStartPos', 'vec3');
  const aEndPos: TSLNode = attribute<'vec3'>('aEndPos', 'vec3');
  const aStartWidth: TSLNode = attribute<'float'>('aStartWidth', 'float');
  const aEndWidth: TSLNode = attribute<'float'>('aEndWidth', 'float');
  const aStartSharpness: TSLNode = attribute<'float'>('aStartSharpness', 'float');
  const aEndSharpness: TSLNode = attribute<'float'>('aEndSharpness', 'float');
  const aSegmentLength: TSLNode = attribute<'float'>('aSegmentLength', 'float');
  const aStartClipped: TSLNode = attribute<'float'>('aStartClipped', 'float');
  const aEndClipped: TSLNode = attribute<'float'>('aEndClipped', 'float');

  // uFOV intentionally not bound: pixel-width math now consumes the
  // CPU-precomputed uPerspectiveLineScale / uOrthoLineScale instead.
  // uIsOrtho also unbound — projection mode is a JS-level config
  // branch (`config.isOrtho`), not a runtime uniform.
  const uResolution = nodes.uResolution;
  const uNodeId = nodes.uNodeId;
  const uNearCull = nodes.uNearCull;
  const uMaxLinePixelWidth = nodes.uMaxLinePixelWidth;
  const uPerspectiveLineScale = nodes.uPerspectiveLineScale;
  const uOrthoLineScale = nodes.uOrthoLineScale;

  // ---- Vertex computation (mirrors line.tsl exactly) ----

  // Branchless: aQuadCorner.x ∈ {-1, +1} by construction.
  const t: TSLNode = aQuadCorner.x.mul(0.5).add(0.5);
  const startW: TSLNode = sanitizeNonNegative(aStartWidth, float(0.0));
  const endW: TSLNode = sanitizeNonNegative(aEndWidth, float(0.0));
  // Sharpness is a [0, 1] knob -> super-Gaussian exponent beta = 2^(6s - 2)
  // (computed in the fragment). A valid s=0 must NOT be rejected, so clamp a
  // non-negative-sanitised value into [0, 1] with the 0.5 default. Mirrors
  // the visual shaders.
  const startS: TSLNode = clamp(sanitizeNonNegative(aStartSharpness, float(0.5)), 0.0, 1.0);
  const endS: TSLNode = clamp(sanitizeNonNegative(aEndSharpness, float(0.5)), 0.0, 1.0);
  const width: TSLNode = mix(startW, endW, t);
  const vSharpnessVal: TSLNode = mix(startS, endS, t);

  const mvStart: TSLNode = modelViewMatrix.mul(vec4(aStartPos, 1.0));
  const mvEnd: TSLNode = modelViewMatrix.mul(vec4(aEndPos, 1.0));
  const mvPos: TSLNode = mix(mvStart, mvEnd, t);

  const nearCull: TSLNode = max(uNearCull, float(1e-4));
  const startDepth: TSLNode = mvStart.z.negate();
  const endDepth: TSLNode = mvEnd.z.negate();
  const bothBehind: TSLNode = startDepth.lessThan(nearCull).and(endDepth.lessThan(nearCull));

  const clipStart: TSLNode = cameraProjectionMatrix.mul(mvStart);
  const clipEnd: TSLNode = cameraProjectionMatrix.mul(mvEnd);
  // projection is linear, so proj * mix(a,b,t) == mix(proj*a, proj*b, t).
  const clipPosBase: TSLNode = mix(clipStart, clipEnd, t);

  const wStart: TSLNode = max(clipStart.w, float(1e-4));
  const wEnd: TSLNode = max(clipEnd.w, float(1e-4));
  const ndcStart: TSLNode = vec2(clipStart.xy.div(wStart));
  const ndcEnd: TSLNode = vec2(clipEnd.xy.div(wEnd));

  // The +0.5 in (ndc*0.5+0.5)*resolution cancels under subtraction.
  const pixelDir: TSLNode = vec2(ndcEnd.sub(ndcStart).mul(uResolution.mul(0.5)));
  const pixelLen: TSLNode = length(pixelDir);
  // `.toVar()` on the chained branch keeps the sharp edge stable.
  const lineDir: TSLNode = pixelLen
    .greaterThan(0.0001)
    .select(vec2(pixelDir.div(pixelLen)).toVar(), vec2(1.0, 0.0));
  const perpendicular: TSLNode = vec2(lineDir.y.negate(), lineDir.x);

  // Each camera projection mode is a separate graph variant so the
  // unused branch never materialises. Wrapper rebuilds when isOrtho
  // flips. View-space depth (-mvPos.z) matches the visual shader.
  let rawPixelWidth: TSLNode;
  if (config.isOrtho) {
    rawPixelWidth = width.mul(uOrthoLineScale);
  } else {
    const distView: TSLNode = max(mvPos.z.negate(), nearCull);
    rawPixelWidth = width.mul(uPerspectiveLineScale).div(distView);
  }

  const minPixelWidth = float(1.5);
  const maxPW: TSLNode = max(uMaxLinePixelWidth, minPixelWidth.add(1.0));
  const clampedPixelWidth: TSLNode = clamp(rawPixelWidth, minPixelWidth, maxPW);
  const vWidthFadeVal: TSLNode = rawPixelWidth
    .lessThanEqual(maxPW)
    .select(float(1.0), maxPW.div(max(rawPixelWidth, float(1e-4))).toVar());

  // Pathological-segment cull (visual-shader parity): both endpoints
  // inside near-cull margin AND rawPixelWidth blows past the clamp by
  // 2× → degenerate to off-screen. Otherwise picking still rasterizes
  // the half-viewport quad the visual pass already culled.
  const pathological: TSLNode = startDepth
    .lessThan(nearCull.mul(2.0))
    .and(endDepth.lessThan(nearCull.mul(2.0)))
    .and(rawPixelWidth.greaterThan(maxPW.mul(2.0)));

  const pixelOffset: TSLNode = perpendicular.mul(aQuadCorner.y).mul(clampedPixelWidth);
  const ndcOffset: TSLNode = pixelOffset.div(uResolution).mul(2.0);
  const expandedClip: TSLNode = vec4(
    clipPosBase.xy.add(ndcOffset.mul(clipPosBase.w)),
    clipPosBase.z,
    clipPosBase.w
  );

  // Real TSL control flow — see visual `line.tsl` for the rationale
  // (one branch per draw instead of evaluating both via select()).
  const culled: TSLNode = bothBehind.or(pathological);
  const clipPos: TSLNode = Fn(() => {
    const out = vec4(2.0, 2.0, 2.0, 1.0).toVar('clipPos');
    If(culled.not(), () => {
      out.assign(expandedClip);
    });
    return out;
  })();

  // Varyings. Per-segment-constant values (segment length, clipped
  // flags, node id, element id) use `flat` interpolation — matches the
  // GLSL3 `flat` qualifier on the same fields.
  const vSharpness: TSLNode = varying(vSharpnessVal);
  const vPerpNorm: TSLNode = varying(aQuadCorner.y);
  const vT: TSLNode = varying(t);
  const vSegmentLength: TSLNode = varying(aSegmentLength).setInterpolation('flat');
  const vWidthAtT: TSLNode = varying(width);
  const vPixelWidth: TSLNode = varying(rawPixelWidth);
  const vWidthFade: TSLNode = varying(vWidthFadeVal);
  const vClippedStart: TSLNode = varying(aStartClipped).setInterpolation('flat');
  const vClippedEnd: TSLNode = varying(aEndClipped).setInterpolation('flat');
  const vNodeId: TSLNode = varying(uNodeId).setInterpolation('flat');
  const vElementId: TSLNode = varying(float(instanceIndex)).setInterpolation('flat');

  // ---- Fragment: brightness output + brightness-as-depth ----

  // Compute brightness once and materialise it into a fragment-local
  // variable via `.toVar()`. Both `colorNode` and `depthNode` then
  // reference the same variable so the pow/cap/widthScale chain
  // doesn't get inlined twice into the generated WGSL/GLSL. The
  // `.once()` on the underlying `Fn(...)` ensures the function body
  // is emitted exactly once even if multiple call sites elsewhere
  // bind to it.
  const brightnessShared = Fn(() => {
    const p: TSLNode = vPerpNorm.abs();
    // Shifted-truncated super-Gaussian perpendicular cross-section —
    // visual-shader parity. beta = 2^(6s - 2) from the [0, 1] knob.
    // K = ln(100), C = exp(-K).
    const K = 4.6051702;
    const C = 0.01;
    const invOneMinusC = 1.0 / (1.0 - C);
    const beta: TSLNode = float(2.0).pow(vSharpness.mul(6.0).sub(2.0));
    const perpFalloff: TSLNode = exp(p.pow(beta).mul(-K)).sub(C).max(float(0.0)).mul(invOneMinusC);
    const minPW = float(1.5);
    const widthScale: TSLNode = min(vPixelWidth.div(minPW), float(1.0));

    const distFromStart: TSLNode = vT.mul(vSegmentLength);
    const distFromEnd: TSLNode = float(1.0).sub(vT).mul(vSegmentLength);
    const distToNearest: TSLNode = min(distFromStart, distFromEnd);
    const capRamp: TSLNode = vWidthAtT
      .greaterThan(float(1e-4))
      .select(clamp(distToNearest.div(vWidthAtT), 0.0, 1.0).toVar(), float(1.0));
    const baseCap: TSLNode = float(0.5).add(capRamp.mul(0.5));
    const nearestIsStart: TSLNode = step(distFromStart, distFromEnd);
    const nearestClipped: TSLNode = mix(vClippedEnd, vClippedStart, nearestIsStart);
    const capFactor: TSLNode = mix(baseCap, float(1.0), nearestClipped);

    return capFactor.mul(perpFalloff).mul(widthScale).mul(vWidthFade);
  }).once();
  const brightness: TSLNode = brightnessShared().toVar('lineBrightness');

  const colorNode = Fn(() => {
    const p: TSLNode = vPerpNorm.abs();
    Discard(p.greaterThanEqual(1.0));
    Discard(brightness.lessThan(1e-4));
    return vec4(vNodeId, vElementId, brightness, 1.0);
  });

  const depthNode = Fn(() => {
    return float(1.0).sub(clamp(brightness, 0.0, 1.0));
  });

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
 * Build a `LinePickTSLNodes` set from a plain `IUniform` record. Used
 * by the harness and the `LINE_PICK_SOURCE` ShaderSource factory in
 * `shaders.ts` — callers that don't own persistent
 * wrapper-side `UniformNode`s. Symmetric with
 * `buildPointPickTSLNodesFromUniforms` /
 * `buildGSplatPickTSLNodesFromUniforms`.
 */
export function buildLinePickTSLNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>
): LinePickTSLNodes {
  return {
    uFOV: uniform((uniforms.uFOV?.value as number) ?? 1.0),
    uResolution: uniform(
      (uniforms.uResolution?.value as THREE.Vector2 | undefined) ?? new THREE.Vector2(1, 1)
    ),
    uIsOrtho: uniform((uniforms.uIsOrtho?.value as number) ?? 0),
    uNodeId: uniform((uniforms.uNodeId?.value as number) ?? 0),
    uNearCull: uniform((uniforms.uNearCull?.value as number) ?? 1e-4),
    uMaxLinePixelWidth: uniform((uniforms.uMaxLinePixelWidth?.value as number) ?? 1.0),
    uPerspectiveLineScale: uniform((uniforms.uPerspectiveLineScale?.value as number) ?? 1.0),
    uOrthoLineScale: uniform((uniforms.uOrthoLineScale?.value as number) ?? 1.0),
  };
}
