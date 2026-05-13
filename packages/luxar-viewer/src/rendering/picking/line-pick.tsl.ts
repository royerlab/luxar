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
  tan,
  modelViewMatrix,
  cameraProjectionMatrix,
  Discard,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type TSLNode = any;

function sanitizePositive(value: TSLNode, fallback: TSLNode): TSLNode {
  const isFinite = value.lessThan(1e30).and(value.greaterThan(-1e30));
  const isPositive = value.greaterThan(0.0);
  return isFinite.and(isPositive).select(value, fallback);
}

function sanitizeNonNegative(value: TSLNode, fallback: TSLNode): TSLNode {
  const isFinite = value.lessThan(1e30).and(value.greaterThan(-1e30));
  const isNonNeg = value.greaterThanEqual(0.0);
  return isFinite.and(isNonNeg).select(value, fallback);
}

/**
 * Line picking material TSL factory. `uniforms` must include uFOV,
 * uResolution, uIsOrtho, uNodeId, uNearCull, uMaxLinePixelWidth.
 */
export function linePickWebGPUFactory(
  uniforms: Record<string, THREE.IUniform>
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

  const uFOV = uniform((uniforms.uFOV.value as number) ?? 1.0);
  const uResolution = uniform(
    (uniforms.uResolution.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );
  const uIsOrtho = uniform((uniforms.uIsOrtho.value as number) ?? 0);
  const uNodeId = uniform((uniforms.uNodeId.value as number) ?? 0);
  const uNearCull = uniform((uniforms.uNearCull.value as number) ?? 1e-4);
  const uMaxLinePixelWidth = uniform(
    (uniforms.uMaxLinePixelWidth.value as number) ?? 1.0
  );

  // ---- Vertex computation (mirrors line.tsl exactly) ----

  const t: TSLNode = aQuadCorner.x.greaterThan(0.0).select(float(1.0), float(0.0));
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
  const clipPosBase: TSLNode = cameraProjectionMatrix.mul(mvPos);

  const wStart: TSLNode = max(clipStart.w, float(1e-4));
  const wEnd: TSLNode = max(clipEnd.w, float(1e-4));
  const ndcStart: TSLNode = vec2(clipStart.xy.div(wStart));
  const ndcEnd: TSLNode = vec2(clipEnd.xy.div(wEnd));
  const pixelStart: TSLNode = ndcStart.mul(0.5).add(0.5).mul(uResolution);
  const pixelEnd: TSLNode = ndcEnd.mul(0.5).add(0.5).mul(uResolution);

  const pixelDir: TSLNode = vec2(pixelEnd.sub(pixelStart));
  const pixelLen: TSLNode = length(pixelDir);
  // `.toVar()` on the chained branch — see M13 sharp-edge notes.
  const lineDir: TSLNode = pixelLen
    .greaterThan(0.0001)
    .select(vec2(pixelDir.div(pixelLen)).toVar(), vec2(1.0, 0.0));
  const perpendicular: TSLNode = vec2(lineDir.y.negate(), lineDir.x);

  const tanHalfFov: TSLNode = tan(uFOV.mul(0.5));
  const distView: TSLNode = max(length(mvPos.xyz), nearCull);
  const rawPixelWidthOrtho: TSLNode = width.mul(2.0).mul(uResolution.y).div(uFOV);
  const rawPixelWidthPersp: TSLNode = width.mul(uResolution.y).div(distView.mul(tanHalfFov));
  const rawPixelWidth: TSLNode = int(uIsOrtho)
    .equal(int(1))
    .select(rawPixelWidthOrtho.toVar(), rawPixelWidthPersp.toVar());

  const minPixelWidth = float(1.5);
  const maxPW: TSLNode = max(uMaxLinePixelWidth, minPixelWidth.add(1.0));
  const clampedPixelWidth: TSLNode = clamp(rawPixelWidth, minPixelWidth, maxPW);
  const vWidthFadeVal: TSLNode = rawPixelWidth
    .lessThanEqual(maxPW)
    .select(float(1.0), maxPW.div(max(rawPixelWidth, float(1e-4))).toVar());

  const pixelOffset: TSLNode = perpendicular.mul(aQuadCorner.y).mul(clampedPixelWidth);
  const ndcOffset: TSLNode = pixelOffset.div(uResolution).mul(2.0);
  const expandedClip: TSLNode = vec4(
    clipPosBase.xy.add(ndcOffset.mul(clipPosBase.w)),
    clipPosBase.z,
    clipPosBase.w
  );

  const offscreen: TSLNode = vec4(2.0, 2.0, 2.0, 1.0);
  const clipPos: TSLNode = bothBehind.select(offscreen.toVar(), expandedClip.toVar());

  // Varyings.
  const vSharpness: TSLNode = varying(vSharpnessVal);
  const vPerpNorm: TSLNode = varying(aQuadCorner.y);
  const vT: TSLNode = varying(t);
  const vSegmentLength: TSLNode = varying(aSegmentLength);
  const vWidthAtT: TSLNode = varying(width);
  const vPixelWidth: TSLNode = varying(rawPixelWidth);
  const vWidthFade: TSLNode = varying(vWidthFadeVal);
  const vClippedStart: TSLNode = varying(aStartClipped);
  const vClippedEnd: TSLNode = varying(aEndClipped);
  const vNodeId: TSLNode = varying(uNodeId);
  const vElementId: TSLNode = varying(float(instanceIndex));

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

  const material = new NodeMaterial();
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.depthNode = depthNode();
  material.toneMapped = false;
  material.depthTest = true;
  material.depthWrite = true;
  material.transparent = false;
  return material;
}
