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
import {
  GLSL_SANITIZE_FUNCTIONS,
  GLSL_NEAR_FADE_FUNCTIONS,
} from '../../materials/_shared/glsl-lib';
import { pointPickWebGPUFactory, buildPointPickTSLNodesFromUniforms } from './pick.tsl';

/**
 * Picking vertex shader for points.
 * Adds uNodeId uniform and vNodeId/vElementId outputs.
 */
export const POINT_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}

    // Per-vertex (4 corners shared across all instances)
    in vec2 aQuadCorner;

    // Per-instance (one per point)
    in vec3 aCenter;
    in float aRadius;
    in float aSharpness;

    uniform float pointSizeFactor;
    uniform float maxPointSize;
    uniform float radiusScale;
    uniform int uIsOrtho;
    uniform float uNearCull;
    uniform float uNodeId;
    uniform vec2 uResolution;

    out highp float vRadius;
    out mediump float vBeta;
    out mediump float vNearFade;
    out mediump vec2 vSpriteCoord;
    flat out highp float vNodeId;
    flat out highp float vElementId;

    void main() {
      // Mirror visual point shader sanitization (shader-glsl.ts) so the
      // pick footprint can't diverge from the visible footprint: sharpness
      // in [0, 1] -> super-Gaussian exponent beta = 2^(6s - 2).
      float s = clamp(sanitizeNonNegative(aSharpness, 0.5), 0.0, 1.0);
      vBeta = exp2(6.0 * s - 2.0);

      float normalizedRadius = sanitizeNonNegative(aRadius * radiusScale, 0.0);
      vRadius = normalizedRadius;

      vec4 mvPosition = modelViewMatrix * vec4(aCenter, 1.0);

      // Unified near handling — keep in sync with the visual point
      // shader and the line/gsplat pick guards: pickability must track
      // what is actually visible (behind-camera fade 0; smooth
      // [nearCull, 2*nearCull] fade; ortho = 1, NDC clip authority).
      vNearFade = perspectiveNearFade(uIsOrtho, mvPosition.z, max(uNearCull, 1e-4));
      if (vNearFade < 0.01) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0); // off-screen → no fragments
        return;
      }

      vec4 projCenter = projectionMatrix * mvPosition;

      // View-space depth, matching the visual point shader (B9a) so the
      // pick footprint stays congruent with the visible sprite.
      float invDistance = (uIsOrtho == 1) ? 1.0 : 1.0 / max(-mvPosition.z, 1e-4);
      float basePointSize = normalizedRadius * pointSizeFactor * invDistance;

      // Picking footprint: 80% of the visual radius (the 0.8 factor below).
      // Slightly tighter than the visible disc so dense/overlapping point
      // clouds still resolve to the point whose core you're over, but
      // forgiving enough that hovering a sparse point doesn't require
      // pixel-perfect aim. Keep in sync with pick.tsl.ts.
      // No sharpness size compensation: the shifted-truncated super-Gaussian
      // truncates at the sprite edge, so basePointSize IS the visible extent
      // (matches shader-glsl.ts).
      float pointSize = basePointSize * 0.8;
      pointSize = max(1.0, min(pointSize, maxPointSize));

      // Instanced quad expansion (matches shader-glsl.ts approach, including
      // the behind-camera guard above).
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
    in mediump float vBeta;
    in mediump float vNearFade;
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
      // Shifted-truncated super-Gaussian (matches shader-glsl.ts) so the pick
      // brightness tie-break tracks the visible falloff. K=ln(100), C=exp(-K).
      const float K = 4.6051702;
      const float C = 0.01;
      const float INV_ONE_MINUS_C = 1.0 / (1.0 - C);
      float falloff = max(exp(-K * pow(normalizedR, vBeta)) - C, 0.0) * INV_ONE_MINUS_C;

      // nearFade folded into brightness (matches gsplat pick).
      float brightness = falloff * vNearFade;
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
