/**
 * Line picking material TSL factory — NodeMaterial counterpart to
 * `LINE_PICK_SOURCE` in `picking-shaders.ts`.
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
 * @module rendering/picking/line-pick.tsl
 */

import * as THREE from 'three';
import {
  Fn,
  uniform,
  attribute,
  varying,
  instanceIndex,
  vec2,
  vec4,
  float,
  int,
  max,
  min,
  clamp,
  mix,
  length,
  step,
  modelViewMatrix,
  cameraProjectionMatrix,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { sanitizeNonNegative, sanitizePositive, type TSLNode } from '../tsl-helpers';

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
 * getter/setter proxies. The harness / picking-shaders ShaderSource
 * registry constructs the nodes from a plain `uniforms` record via
 * {@link buildLinePickTSLNodesFromUniforms}.
 */
export function linePickWebGPUFactory(
  nodes: LinePickTSLNodes,
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
  const uResolution = nodes.uResolution;
  const uIsOrtho = nodes.uIsOrtho;
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
  const startS: TSLNode = sanitizePositive(aStartSharpness, float(2.0));
  const endS: TSLNode = sanitizePositive(aEndSharpness, float(2.0));
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
  // `.toVar()` on the chained branch — see M13 sharp-edge notes.
  const lineDir: TSLNode = pixelLen
    .greaterThan(0.0001)
    .select(vec2(pixelDir.div(pixelLen)).toVar(), vec2(1.0, 0.0));
  const perpendicular: TSLNode = vec2(lineDir.y.negate(), lineDir.x);

  // Both pixel-width branches consume CPU-precomputed scales — no
  // per-vertex tan() or division by uFOV. Visual-shader parity.
  const distView: TSLNode = max(length(mvPos.xyz), nearCull);
  const rawPixelWidthOrtho: TSLNode = width.mul(uOrthoLineScale);
  const rawPixelWidthPersp: TSLNode = width.mul(uPerspectiveLineScale).div(distView);
  const rawPixelWidth: TSLNode = int(uIsOrtho)
    .equal(int(1))
    .select(rawPixelWidthOrtho.toVar(), rawPixelWidthPersp.toVar());

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

  const offscreen: TSLNode = vec4(2.0, 2.0, 2.0, 1.0);
  const culled: TSLNode = bothBehind.or(pathological);
  const clipPos: TSLNode = culled.select(offscreen.toVar(), expandedClip.toVar());

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

  const brightnessNode = () => {
    const p: TSLNode = vPerpNorm.abs();
    const perpFalloff: TSLNode = max(float(1.0).sub(p.mul(p)), float(0.0)).pow(
      max(vSharpness, float(0.0001))
    );
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
  };

  const colorNode = Fn(() => {
    const p: TSLNode = vPerpNorm.abs();
    Discard(p.greaterThanEqual(1.0));
    const brightness: TSLNode = brightnessNode();
    Discard(brightness.lessThan(1e-4));
    return vec4(vNodeId, vElementId, brightness, 1.0);
  });

  const depthNode = Fn(() => {
    const brightness: TSLNode = brightnessNode();
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
 * `picking-shaders.ts` — callers that don't own persistent
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
    uPerspectiveLineScale: uniform(
      (uniforms.uPerspectiveLineScale?.value as number) ?? 1.0
    ),
    uOrthoLineScale: uniform((uniforms.uOrthoLineScale?.value as number) ?? 1.0),
  };
}
