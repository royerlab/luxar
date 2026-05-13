/**
 * Point picking material TSL factory — NodeMaterial counterpart to
 * the GLSL3 picking shader in `picking-shaders.ts::POINT_PICK_SOURCE`.
 *
 * Renders one tight sprite per point with output:
 *   - R: nodeId (set as uniform)
 *   - G: elementId (= `gl_InstanceID`)
 *   - B: brightness (Gaussian falloff at the fragment position)
 *   - A: 1.0
 *
 * Depth is set to `1.0 - brightness` (brightness-as-depth) so the
 * picking system's tie-breaking prefers the brightest hit. Matches
 * the GLSL `gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0)` path.
 *
 * The pick sprite is HALF the radius of the visual sprite, so the
 * pick footprint is the bright core only — peripheral falloff
 * regions don't capture hover state.
 *
 * @module rendering/picking/point-pick.tsl
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
  clamp,
  length,
  dot,
  pow,
  Discard,
  modelViewMatrix,
  cameraProjectionMatrix,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { type TSLNode } from '../tsl-helpers';

/**
 * Point picking material TSL factory.
 *
 * `uniforms` must include the full set the GLSL3 picking shader
 * reads (pointSizeFactor, maxPointSize, radiusScale, sharpnessScale,
 * uIsOrtho, uNodeId, uResolution).
 */
export function pointPickWebGPUFactory(
  uniforms: Record<string, THREE.IUniform>
): NodeMaterial {
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  const aCenter: TSLNode = attribute<'vec3'>('aCenter', 'vec3');
  const aRadius: TSLNode = attribute<'float'>('aRadius', 'float');
  const aSharpness: TSLNode = attribute<'float'>('aSharpness', 'float');

  const uPointSizeFactor = uniform((uniforms.pointSizeFactor.value as number) ?? 1.0);
  const uMaxPointSize = uniform((uniforms.maxPointSize.value as number) ?? 1.0);
  const uRadiusScale = uniform((uniforms.radiusScale.value as number) ?? 1.0);
  const uSharpnessScale = uniform((uniforms.sharpnessScale.value as number) ?? 1.0);
  const uIsOrtho = uniform((uniforms.uIsOrtho.value as number) ?? 0);
  const uNodeId = uniform((uniforms.uNodeId.value as number) ?? 0);
  const uResolution = uniform(
    (uniforms.uResolution.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );

  // Per-instance sanitisation. Picking does NOT use the full
  // sanitizePositive helper — the GLSL path uses a simpler
  // `> 0.0 ? x : 2.0` check. Mirror that exactly for parity.
  const normalizedSharpnessRaw: TSLNode = aSharpness.mul(uSharpnessScale);
  const normalizedSharpness: TSLNode = normalizedSharpnessRaw
    .greaterThan(0.0)
    .select(normalizedSharpnessRaw, float(2.0));
  const normalizedRadius: TSLNode = aRadius.mul(uRadiusScale);

  // Vertex transform.
  const mvPos: TSLNode = modelViewMatrix.mul(vec4(aCenter, 1.0));
  const projCenter: TSLNode = cameraProjectionMatrix.mul(mvPos);

  const invDistance: TSLNode = int(uIsOrtho)
    .equal(int(1))
    .select(float(1.0), length(mvPos.xyz).reciprocal());
  const basePointSize: TSLNode = normalizedRadius.mul(uPointSizeFactor).mul(invDistance);

  // Tighter picking: × 0.5 vs the visual material.
  const sharpnessComp: TSLNode = float(1.0).div(
    float(1.0).sub(pow(float(0.01), float(1.0).div(max(normalizedSharpness, float(0.01)))))
  );
  const pickPointSize: TSLNode = max(
    float(1.0),
    clamp(basePointSize.mul(sharpnessComp).mul(0.5), float(1.0), uMaxPointSize)
  );

  const offsetClip: TSLNode = aQuadCorner
    .mul(pickPointSize.div(uResolution))
    .mul(projCenter.w);
  const clipPos: TSLNode = projCenter.add(vec4(offsetClip, 0.0, 0.0));

  // Varyings.
  const vSpriteCoord: TSLNode = varying(aQuadCorner.add(1.0).mul(0.5));
  const vRadius: TSLNode = varying(normalizedRadius);
  const vSharpness: TSLNode = varying(normalizedSharpness);
  // nodeId and elementId are flat in the GLSL path. TSL's `varying()`
  // wraps with per-vertex linear interpolation by default; for a
  // single-instance quad all 4 corners carry the same value, so
  // interpolation is the identity — same numeric result.
  const vNodeId: TSLNode = varying(uNodeId);
  const vElementId: TSLNode = varying(float(instanceIndex));

  // Fragment.
  const colorNode = Fn(() => {
    Discard(vRadius.lessThan(0.0001));

    const centered: TSLNode = vec2(vSpriteCoord.sub(0.5));
    const r2: TSLNode = dot(centered, centered);
    Discard(r2.greaterThan(0.25));

    const normalizedR: TSLNode = r2.mul(4.0).sqrt();
    const falloff: TSLNode = float(1.0).sub(normalizedR).max(float(0.0)).pow(vSharpness);
    const brightness: TSLNode = falloff;
    Discard(brightness.lessThan(1e-4));

    return vec4(vNodeId, vElementId, brightness, 1.0);
  });

  // Depth = 1.0 - brightness (the brightest hit takes precedence).
  // TSL doesn't easily share scope between colorNode and depthNode,
  // so we recompute brightness on the depth path. Same math, same
  // discard-gating happens via colorNode → depth is only written
  // when colorNode also writes.
  const depthNode = Fn(() => {
    const centered: TSLNode = vec2(vSpriteCoord.sub(0.5));
    const r2: TSLNode = dot(centered, centered);
    const normalizedR: TSLNode = r2.mul(4.0).sqrt();
    const falloff: TSLNode = float(1.0).sub(normalizedR).max(float(0.0)).pow(vSharpness);
    return float(1.0).sub(clamp(falloff, 0.0, 1.0));
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
