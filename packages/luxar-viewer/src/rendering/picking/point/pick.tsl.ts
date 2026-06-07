/**
 * Point picking material TSL factory — NodeMaterial counterpart to
 * the GLSL3 picking shader in `shaders.ts::POINT_PICK_SOURCE`.
 *
 * Renders one tight sprite per point with output:
 *   - R: nodeId (set as uniform)
 *   - G: elementId (= `gl_InstanceID`)
 *   - B: brightness (super-Gaussian falloff at the fragment position)
 *   - A: 1.0
 *
 * Depth is set to `1.0 - brightness` (brightness-as-depth) so the
 * picking system's tie-breaking prefers the brightest hit. Matches
 * the GLSL `gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0)` path.
 *
 * The pick sprite is 80% of the radius of the visual sprite
 * (the 0.8 factor below) — slightly tighter than the visible disc so
 * overlapping points still resolve to the one whose core you're over,
 * but forgiving enough that sparse points don't need pixel-perfect aim.
 * Keep in sync with shaders.ts.
 *
 * @module rendering/picking/point/pick.tsl
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
  exp,
  Discard,
  modelViewMatrix,
  cameraProjectionMatrix,
} from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { sanitizeNonNegative, type TSLNode } from '../../materials/_shared/tsl-helpers';

/**
 * Pre-created TSL leaf nodes supplied by the wrapper class. See
 * `LineTSLNodes` / `PointTSLNodes` / `GSplatTSLNodes` for the
 * rationale: avoids `.onUpdate('render')` callback churn by using
 * the wrapper-owned `UniformNode` references directly.
 */
export interface PointPickTSLNodes {
  readonly pointSizeFactor: TSLNode;
  readonly maxPointSize: TSLNode;
  readonly radiusScale: TSLNode;
  readonly uIsOrtho: TSLNode;
  readonly uNodeId: TSLNode;
  readonly uResolution: TSLNode;
}

/**
 * Point picking material TSL factory.
 *
 * Consumes pre-created `UniformNode` references; the wrapper
 * (`PointPickingTSLMaterial`) owns those nodes and exposes them via
 * `material.uniforms` as `IUniform`-shaped getter/setter proxies.
 */
export function pointPickWebGPUFactory(
  nodes: PointPickTSLNodes,
  outMaterial?: NodeMaterial
): NodeMaterial {
  const aQuadCorner: TSLNode = attribute<'vec2'>('aQuadCorner', 'vec2');
  const aCenter: TSLNode = attribute<'vec3'>('aCenter', 'vec3');
  const aRadius: TSLNode = attribute<'float'>('aRadius', 'float');
  const aSharpness: TSLNode = attribute<'float'>('aSharpness', 'float');

  const uPointSizeFactor = nodes.pointSizeFactor;
  const uMaxPointSize = nodes.maxPointSize;
  const uRadiusScale = nodes.radiusScale;
  const uIsOrtho = nodes.uIsOrtho;
  const uNodeId = nodes.uNodeId;
  const uResolution = nodes.uResolution;

  // Per-instance sanitisation — mirrors visual shader-tsl.ts and the GLSL
  // picking shader: sharpness in [0, 1] -> super-Gaussian exponent
  // beta = 2^(6s - 2). sanitizeNonNegative keeps a valid s=0 and routes
  // NaN/Inf/negative to the 0.5 default so the pick footprint can't diverge
  // from the visible footprint.
  const sClamped: TSLNode = clamp(sanitizeNonNegative(aSharpness, float(0.5)), 0.0, 1.0);
  const beta: TSLNode = float(2.0).pow(sClamped.mul(6.0).sub(2.0));
  const normalizedRadius: TSLNode = sanitizeNonNegative(aRadius.mul(uRadiusScale), float(0.0));

  // Vertex transform.
  const mvPos: TSLNode = modelViewMatrix.mul(vec4(aCenter, 1.0));
  const projCenter: TSLNode = cameraProjectionMatrix.mul(mvPos);

  const invDistance: TSLNode = int(uIsOrtho)
    .equal(int(1))
    .select(float(1.0), length(mvPos.xyz).reciprocal());
  const basePointSize: TSLNode = normalizedRadius.mul(uPointSizeFactor).mul(invDistance);

  // Picking footprint: × 0.8 vs the visual material (keep the 0.8 in sync
  // with shaders.ts). No sharpness size compensation — the shifted-truncated
  // super-Gaussian truncates at the sprite edge, so basePointSize IS the
  // visible extent (matches shader-tsl.ts).
  const pickPointSize: TSLNode = max(
    float(1.0),
    clamp(basePointSize.mul(0.8), float(1.0), uMaxPointSize)
  );

  const offsetClip: TSLNode = aQuadCorner.mul(pickPointSize.div(uResolution)).mul(projCenter.w);
  const clipPos: TSLNode = projCenter.add(vec4(offsetClip, 0.0, 0.0));

  // Varyings.
  const vSpriteCoord: TSLNode = varying(aQuadCorner.add(1.0).mul(0.5));
  const vRadius: TSLNode = varying(normalizedRadius);
  const vBeta: TSLNode = varying(beta);
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
    // Shifted-truncated super-Gaussian (matches shader-tsl.ts).
    const K = 4.6051702; // ln(100)
    const C = 0.01; // exp(-K) = floor
    const invOneMinusC = 1.0 / (1.0 - C);
    const falloff: TSLNode = exp(normalizedR.pow(vBeta).mul(-K))
      .sub(C)
      .max(float(0.0))
      .mul(invOneMinusC);
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
    // Shifted-truncated super-Gaussian (matches shader-tsl.ts).
    const K = 4.6051702; // ln(100)
    const C = 0.01; // exp(-K) = floor
    const invOneMinusC = 1.0 / (1.0 - C);
    const falloff: TSLNode = exp(normalizedR.pow(vBeta).mul(-K))
      .sub(C)
      .max(float(0.0))
      .mul(invOneMinusC);
    return float(1.0).sub(clamp(falloff, 0.0, 1.0));
  });

  const material = outMaterial ?? new NodeMaterial();
  material.vertexNode = clipPos;
  material.colorNode = colorNode();
  material.depthNode = depthNode();
  material.toneMapped = false;
  material.depthTest = true;
  material.depthWrite = true;
  material.transparent = false;
  // Picking output is an opaque ID buffer; any blending would
  // smear nodeId / elementId values across overlapping picks and
  // produce nonsense readbacks. Matches the GLSL picking material.
  material.blending = THREE.NoBlending;
  return material;
}

/**
 * Snapshot adapter: build a `PointPickTSLNodes` set from a flat
 * `IUniform` record. Used by callers that don't own persistent nodes.
 * See `buildLineTSLNodesFromUniforms` for the rationale.
 */
export function buildPointPickTSLNodesFromUniforms(
  uniforms: Record<string, THREE.IUniform>
): PointPickTSLNodes {
  return {
    pointSizeFactor: uniform((uniforms.pointSizeFactor?.value as number) ?? 1.0),
    maxPointSize: uniform((uniforms.maxPointSize?.value as number) ?? 1.0),
    radiusScale: uniform((uniforms.radiusScale?.value as number) ?? 1.0),
    uIsOrtho: uniform((uniforms.uIsOrtho?.value as number) ?? 0),
    uNodeId: uniform((uniforms.uNodeId?.value as number) ?? 0),
    uResolution: uniform(
      (uniforms.uResolution?.value as THREE.Vector2 | undefined) ?? new THREE.Vector2(1, 1)
    ),
  };
}
