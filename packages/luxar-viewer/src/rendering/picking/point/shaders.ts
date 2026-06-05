/**
 * GLSL3 picking shader source for points + `ShaderSource` record.
 *
 * Mirrors the visual point shader in `rendering/materials/point/shader-glsl.ts`
 * with three picking-specific additions:
 *   - `uNodeId` uniform + `vNodeId` / `vElementId` varyings, written
 *     into the RGBA32F pick buffer as `(nodeId, elementId, brightness, 1)`.
 *   - 80%-radius truncation so the pick footprint tracks the visible disc
 *     while staying slightly biased toward the bright core.
 *   - Brightness-as-depth so the brightest overlapping fragment wins
 *     the depth test (matters for hover-through-translucent point stacks).
 *
 * Source-of-truth for GLSL3; the WebGPU counterpart lives in `./pick.tsl`
 * and is referenced through the `ShaderSource.webgpu` factory below.
 *
 * @module rendering/picking/point/shaders
 */

import type { ShaderSource } from '../../materials/_shared/shader-source';
import { GLSL_SANITIZE_FUNCTIONS } from '../../materials/_shared/glsl-lib';
import { pointPickWebGPUFactory, buildPointPickTSLNodesFromUniforms } from './pick.tsl';

/**
 * Picking vertex shader for points.
 * Adds uNodeId uniform and vNodeId/vElementId outputs.
 */
export const POINT_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}

    // Per-vertex (4 corners shared across all instances)
    in vec2 aQuadCorner;

    // Per-instance (one per point)
    in vec3 aCenter;
    in float aRadius;
    in float aSharpness;

    uniform float pointSizeFactor;
    uniform float maxPointSize;
    uniform float radiusScale;
    uniform float sharpnessScale;
    uniform int uIsOrtho;
    uniform float uNodeId;
    uniform vec2 uResolution;

    out highp float vRadius;
    out mediump float vSharpness;
    out mediump vec2 vSpriteCoord;
    flat out highp float vNodeId;
    flat out highp float vElementId;

    void main() {
      // Mirror visual point shader sanitization (shader-glsl.ts)
      // so a NaN/Inf sharpness or negative radius can't cause the pick
      // footprint to diverge from the visible footprint.
      float normalizedSharpness = sanitizePositive(aSharpness * sharpnessScale, 2.0);
      vSharpness = normalizedSharpness;

      float normalizedRadius = sanitizeNonNegative(aRadius * radiusScale, 0.0);
      vRadius = normalizedRadius;

      vec4 mvPosition = modelViewMatrix * vec4(aCenter, 1.0);
      vec4 projCenter = projectionMatrix * mvPosition;

      float invDistance = (uIsOrtho == 1) ? 1.0 : inversesqrt(dot(mvPosition.xyz, mvPosition.xyz));
      float basePointSize = normalizedRadius * pointSizeFactor * invDistance;

      // Picking footprint: 80% of the visual radius (the 0.8 factor below).
      // Slightly tighter than the visible disc so dense/overlapping point
      // clouds still resolve to the point whose core you're over, but
      // forgiving enough that hovering a sparse point doesn't require
      // pixel-perfect aim. Keep in sync with pick.tsl.ts.
      // Mirror visual point shader's invalid-result guard so a degenerate
      // vSharpness (e.g. ≪0.01 after clamp) can't poison pointSize.
      float sharpnessCompensationRaw = 1.0 / (1.0 - pow(0.01, 1.0 / max(vSharpness, 0.01)));
      float sharpnessCompensation = isInvalidFloat(sharpnessCompensationRaw) ? 1.0 : sharpnessCompensationRaw;
      float pointSize = basePointSize * sharpnessCompensation * 0.8;
      pointSize = max(1.0, min(pointSize, maxPointSize));

      // Instanced quad expansion (matches shader-glsl.ts approach).
      vec2 offsetClip = aQuadCorner * (pointSize / uResolution) * projCenter.w;
      gl_Position = projCenter + vec4(offsetClip, 0.0, 0.0);

      vSpriteCoord = (aQuadCorner + 1.0) * 0.5;

      vNodeId = uNodeId;
      // Under instanced rendering, gl_InstanceID is the per-point index
      // (the old THREE.Points path used gl_VertexID which was equivalent).
      vElementId = float(gl_InstanceID);
    }
`;

/**
 * Picking fragment shader for points.
 * Outputs vec4(nodeId, elementId, brightness, 1.0) with brightness-as-depth.
 */
export const POINT_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    in highp float vRadius;
    in mediump float vSharpness;
    in mediump vec2 vSpriteCoord;
    flat in highp float vNodeId;
    flat in highp float vElementId;

    out vec4 fragColor;

    void main() {
      if (vRadius < 0.0001) discard;

      vec2 centered = vSpriteCoord - 0.5;
      float r2 = dot(centered, centered);

      // Full circle discard (gl_PointSize is already halved in vertex shader for tighter picking)
      if (r2 > 0.25) discard;

      float normalizedR = sqrt(4.0 * r2);
      float falloff = pow(max(1.0 - normalizedR, 0.0), vSharpness);

      float brightness = falloff;
      if (brightness < 1e-4) discard;

      fragColor = vec4(vNodeId, vElementId, brightness, 1.0);
      gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0);
    }
`;

export const POINT_PICK_SOURCE: ShaderSource = {
  name: 'point-pick',
  webgl: { vertex: POINT_PICK_VERTEX_SHADER, fragment: POINT_PICK_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    return pointPickWebGPUFactory(buildPointPickTSLNodesFromUniforms(u));
  },
};
