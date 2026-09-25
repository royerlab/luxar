/**
 * GLSL3 picking shader source for points + `ShaderSource` record.
 *
 * Mirrors the visual point shader in `rendering/materials/point/shader-glsl.ts`
 * with three picking-specific additions:
 *   - `uNodeId` uniform + `vNodeId` / `vElementId` varyings, written
 *     into the RGBA32F pick buffer as `(nodeId, elementId-low16, brightness, elementId-high16)`.
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
  GLSL_PROJECTION_FUNCTIONS,
  GLSL_SORTED_INDEX,
} from '../../materials/_shared/glsl-lib';
import { FALLOFF_FLOOR, FALLOFF_K } from '../../materials/_shared/falloff';
import { requireTslMaterials } from '../../tsl/slot';

/**
 * Picking vertex shader for points.
 * Adds uNodeId uniform and vNodeId/vElementId outputs.
 */
export const POINT_PICK_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}
    ${GLSL_PROJECTION_FUNCTIONS}

    // Per-vertex (4 corners shared across all instances)
    in vec2 aQuadCorner;

    // Draw-slot -> storage-slot mapping (identity in Phase 1; permuted
    // by the sort worker in Phase 2+). Also the pick ELEMENT id: the
    // pick buffer must report the storage slot -- the id the rest of
    // the pipeline (loaders, selection) addresses points by -- not the
    // transient draw slot.
    ${GLSL_SORTED_INDEX}

    // Point data texture: RGBA32F, 3 texels/point (see
    // rendering/element-texture-layout.ts). Picking needs texels 0-1
    // only (center/radius/sharpness) -- color and scalar are not fetched.
    uniform highp sampler2D uPointTex;

    uniform float pointSizeFactor;
    uniform float maxPointSize;
    uniform float radiusScale;
    uniform int uIsOrtho;
    uniform float uNearCull;
    uniform float uPixelRatio;
    uniform float uNodeId;
    uniform vec2 uResolution;

    out highp float vRadius;
    out mediump float vBeta;
    out mediump float vNearFade;
    out mediump float vPickSize;  // RAW pick sprite size (pre-clamp) — sizeScale² parity with the visual shader
    out mediump vec2 vSpriteCoord;
    flat out highp float vNodeId;
    flat out highp vec2 vElementId;

    void main() {
      // === Point-texture fetch prologue (visual-shader parity) ===
      // Width is a multiple of 3, so a point's texels share one row.
      // Projected-density thinning: a point the visual pass dropped must not be
      // pickable either (same hash, same uniform value — see density-drop.ts).
      if (luxarDensityDropped()) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        return;
      }
      int pointBase = int(luxarSortedIndex()) * 3;
      int pointTexW = LUXAR_POINT_TEX_W;
      ivec2 texel0 = ivec2(pointBase % pointTexW, pointBase / pointTexW);
      vec4 pointT0 = texelFetch(uPointTex, texel0, 0);
      vec4 pointT1 = texelFetch(uPointTex, ivec2(texel0.x + 1, texel0.y), 0);
      vec3 aCenter = pointT0.xyz;
      float aRadius = pointT0.w;
      float aSharpness = pointT1.w;

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
      // 1e-20 floor = degenerate-smoothstep guard only; uNearCull is
      // scene-bounds-scaled (see the visual point shader).
      vNearFade = perspectiveNearFade(luxarIsOrthoProjection(), mvPosition.z, max(uNearCull, 1e-20));
      if (vNearFade < 0.01) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0); // off-screen → no fragments
        return;
      }

      vec4 projCenter = projectionMatrix * mvPosition;

      // View-space depth, matching the visual point shader (B9a) so the
      // pick footprint stays congruent with the visible sprite. 1e-20 =
      // pure INF guard (near-fade reject bounds surviving depths at the
      // scene-relative ~uNearCull; the size clamp bounds the output).
      float invDistance = (luxarIsOrthoProjection() == 1) ? 1.0 : 1.0 / max(-mvPosition.z, 1e-20);
      float sizeFactor = 2.0 * uResolution.y * luxarProjectionSizeScale();
      float basePointSize = normalizedRadius * sizeFactor * invDistance;

      // Picking footprint: 80% of the visual radius (the 0.8 factor below).
      // Slightly tighter than the visible disc so dense/overlapping point
      // clouds still resolve to the point whose core you're over, but
      // forgiving enough that hovering a sparse point doesn't require
      // pixel-perfect aim. Keep in sync with pick.tsl.ts.
      // No sharpness size compensation: the shifted-truncated super-Gaussian
      // truncates at the sprite edge, so basePointSize IS the visible extent
      // (matches shader-glsl.ts).
      float pointSize = basePointSize * 0.8;
      vPickSize = pointSize; // raw, pre-clamp — fragment applies sizeScale²
      pointSize = clamp(pointSize, 1.5 * max(uPixelRatio, 1.0), maxPointSize); // CSS-pixel floor tracks the VISUAL sprite floor; below 1× keep the historical framebuffer floor

      // Instanced quad expansion (matches shader-glsl.ts approach, including
      // the behind-camera guard above).
      vec2 offsetClip = aQuadCorner * (pointSize / uResolution) * projCenter.w;
      gl_Position = projCenter + vec4(offsetClip, 0.0, 0.0);

      vSpriteCoord = (aQuadCorner + 1.0) * 0.5;

      vNodeId = uNodeId;
      // Storage slot, NOT gl_InstanceID (the draw slot): identical
      // under Phase-1 identity ordering, and stays correct once the
      // sort worker permutes draw order (Phase 2+).
      vElementId = luxarElementIdParts();
    }
`;

/**
 * Picking fragment shader for points.
 * Outputs vec4(nodeId, elementId-low16, brightness, elementId-high16) with brightness-as-depth.
 */
export const POINT_PICK_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    uniform float uPixelRatio;

    in highp float vRadius;
    in mediump float vBeta;
    in mediump float vNearFade;
    in mediump float vPickSize; // raw pick sprite size (sub-pixel compensation)
    in mediump vec2 vSpriteCoord;
    flat in highp float vNodeId;
    flat in highp vec2 vElementId;

    out vec4 fragColor;

    void main() {
      // Exact-zero only — see the visual shader's twin comment.
      if (vRadius <= 0.0) discard;

      vec2 centered = vSpriteCoord - 0.5;
      float r2 = dot(centered, centered);

      // Full circle discard (gl_PointSize is already halved in vertex shader for tighter picking)
      if (r2 > 0.25) discard;

      float normalizedR = sqrt(4.0 * r2);
      // Shifted-truncated super-Gaussian (matches shader-glsl.ts) so the pick
      // brightness tie-break tracks the visible falloff. K=ln(100), C=exp(-K).
      const float K = ${FALLOFF_K};
      const float C = ${FALLOFF_FLOOR};
      const float INV_ONE_MINUS_C = 1.0 / (1.0 - C);
      float falloff = max(exp(-K * pow(normalizedR, vBeta)) - C, 0.0) * INV_ONE_MINUS_C;

      // nearFade folded into brightness (matches gsplat pick).
      // Sub-pixel compensation (sizeScale², matching the VISUAL point and
      // the line pick's widthScale): pick salience must track visual
      // salience, or a sub-pixel (visually dimmed) point wins the
      // brightness-as-depth tie-break over a visually brighter neighbor.
      mediump float pickSizeScale = min(vPickSize / (1.5 * max(uPixelRatio, 1.0)), 1.0);
      float brightness = falloff * vNearFade * pickSizeScale * pickSizeScale;
      if (brightness < 1e-4) discard;

      fragColor = vec4(vNodeId, vElementId.x, brightness, vElementId.y);
      gl_FragDepth = 1.0 - clamp(brightness, 0.0, 1.0);
    }
`;

export const POINT_PICK_SOURCE: ShaderSource = {
  name: 'point-pick',
  webgl: { vertex: POINT_PICK_VERTEX_SHADER, fragment: POINT_PICK_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    const { pointPickWebGPUFactory, buildPointPickTSLNodesFromUniforms } =
      requireTslMaterials().factories.pickPoint;
    return pointPickWebGPUFactory(buildPointPickTSLNodesFromUniforms(u));
  },
};
