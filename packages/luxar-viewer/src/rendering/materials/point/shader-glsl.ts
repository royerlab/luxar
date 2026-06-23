/**
 * Shared vertex shader for point rendering.
 *
 * Used by both PointMaterial (main rendering) and PointPickingMaterial (GPU picking).
 * Renders one screen-space-sized soft sprite per point using instanced
 * quad expansion (matching the line + gsplat rendering pattern).
 *
 * Per-vertex attribute (4 entries, shared by ALL points):
 *   - aQuadCorner (vec2, ±1)  — unit-quad corner, vertex shader
 *     expands to a sprite of size `pointSize` pixels in screen space.
 *
 * Per-instance attributes (one entry per point, supplied by the
 * gpu-buffer-pool points adapter):
 *   - aCenter    (vec3) — world-space centre position
 *   - aRadius    (float)
 *   - aSharpness (float)
 *   - aColor     (vec3) — always present (instead of Three.js'
 *     vertexColors=true auto-injected `color` attribute)
 *   - aScalar    (float, USE_COLORMAP only)
 */
import { GLSL_SANITIZE_FUNCTIONS } from '../_shared/glsl-lib';
import type { ShaderSource } from '../_shared/shader-source';
import { pointWebGPUFactory } from './shader-tsl';

export const POINT_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}

    // Per-vertex (4 corners): -1..1 normalised quad coordinates.
    in vec2 aQuadCorner;

    // Per-instance (one per point) — all read once per instance, cached
    // by the GPU across the 4 quad corners of the same instance.
    in vec3 aCenter;
    in float aRadius;
    in float aSharpness;
    in vec3 aColor;
    #ifdef USE_COLORMAP
    in float aScalar;                  // Per-point scalar for colormap lookup
    uniform sampler2D uColormapTex;    // 256x1 LUT texture
    uniform float uScalarMin;          // Scalar range minimum (display-range window)
    uniform float uScalarScale;        // 1.0 / (max - min)
    uniform mediump float invGamma;    // Gamma applied to the VALUE, pre-LUT (see note below)
    #endif

    uniform float pointSizeFactor; // Pre-computed: 2.0 * resolution.y / tanHalfFov (or 4.0 * resolution.y / frustumHeight for ortho)
    uniform float maxPointSize;    // Pre-computed: resolution.y * 0.5
    uniform float radiusScale;
    uniform int uIsOrtho;          // 0 = perspective, 1 = orthographic
    uniform vec2 uResolution;      // Physical framebuffer size in pixels

    out mediump vec3 vColor;
    out mediump float vBeta;       // Super-Gaussian exponent beta (per-instance)
    out highp float vRadius;       // Pass radius to fragment for zero-check (needs precision)
    out mediump vec2 vSpriteCoord; // [0, 1] sprite UV, replaces gl_PointCoord

    void main() {
      // Pass vertex color — either from attribute or colormap LUT.
      // In colormap mode the display range (uScalarMin/uScalarScale) and
      // gamma operate on the scalar VALUE before the LUT lookup, not on
      // the resulting color. Direct-color mode keeps GOG on the color
      // (fragment shader). See the fragment-shader note.
      #ifdef USE_COLORMAP
      float t = clamp((aScalar - uScalarMin) * uScalarScale, 0.0, 1.0);
      #ifndef LUXAR_GAMMA_ONE
      t = pow(t, invGamma);              // gamma on the value, pre-LUT
      #endif
      vColor = texture(uColormapTex, vec2(t, 0.5)).rgb;
      #else
      vColor = aColor;
      #endif

      // Sharpness is authored in [0, 1] and maps to the super-Gaussian
      // exponent beta = 2^(6s - 2): s=0.5 -> beta=2 (a true Gaussian, the
      // gsplat member), higher s -> harder edge, lower s -> peakier cusp.
      // sanitizeNonNegative keeps a valid s=0 (-> beta=0.25) and routes
      // NaN/Inf/negative to the 0.5 default; clamp guards the [0, 1] range.
      float s = clamp(sanitizeNonNegative(aSharpness, 0.5), 0.0, 1.0);
      vBeta = exp2(6.0 * s - 2.0);

      // Apply radius scale for dtype normalization (e.g., uint8 needs 1/255 scale)
      float normalizedRadius = sanitizeNonNegative(aRadius * radiusScale, 0.0);
      vRadius = normalizedRadius; // Pass to fragment shader

      // Transform per-instance centre from world space to view + clip space.
      vec4 mvPosition = modelViewMatrix * vec4(aCenter, 1.0);

      // Reject points behind the camera (perspective only; camera looks down -Z,
      // so mvPosition.z >= 0 is behind the near plane). The quad expansion below
      // multiplies by projCenter.w, which is <= 0 for such points and would
      // produce a degenerate/flipped sprite. Mirrors the gsplat shader's guard.
      // Ortho keeps projCenter.w == 1, so it is excluded.
      if (uIsOrtho == 0 && mvPosition.z >= 0.0) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0); // off-screen → no fragments
        return;
      }

      vec4 projCenter = projectionMatrix * mvPosition;

      // OPTIMIZED world-space point sizing:
      // - inversesqrt is a native GPU instruction (faster than sqrt + divide)
      // - pointSizeFactor pre-computed in JS: 2.0 * resolution.y / tanHalfFov
      float invDistance = (uIsOrtho == 1) ? 1.0 : inversesqrt(dot(mvPosition.xyz, mvPosition.xyz));
      float basePointSize = normalizedRadius * pointSizeFactor * invDistance;

      // The shifted-truncated super-Gaussian falloff (fragment shader)
      // truncates to zero exactly at the sprite edge (rho = 1), so the
      // sprite size already IS the visible extent — no sharpness-dependent
      // size compensation is needed (the old polynomial kernel required it).
      float pointSize = basePointSize;

      // Clamp: minimum 1.0 (avoids degenerate quads) and maxPointSize cap.
      // Zero-radius filtering happens in fragment shader.
      pointSize = max(1.0, min(pointSize, maxPointSize));

      // Expand the unit quad to a screen-space sprite. aQuadCorner is
      // in [-1, 1] per axis, so aQuadCorner * (pointSize / uResolution)
      // is the half-extent in NDC space. Multiply by projCenter.w to
      // convert NDC delta to clip-space delta (compensating for the
      // upcoming perspective divide).
      vec2 offsetClip = aQuadCorner * (pointSize / uResolution) * projCenter.w;
      gl_Position = projCenter + vec4(offsetClip, 0.0, 0.0);

      // Sprite UV in [0, 1]² — fragment shader uses this in place of
      // gl_PointCoord (which is unavailable under THREE.Mesh).
      vSpriteCoord = (aQuadCorner + 1.0) * 0.5;
    }
  `;

/**
 * Fragment shader for standard point rendering.
 *
 * Computes the shifted-truncated super-Gaussian falloff, GOG color
 * adjustment, and alpha output.
 * The picking system uses a different fragment shader (see picking/point-picking-material.ts).
 */
export const POINT_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    uniform mediump float opacity;
    uniform mediump float invGamma; // Pre-computed 1/gamma for performance
    uniform mediump float uIntensity; // Per-node linear color multiplier (gain)
    uniform mediump float uOffset; // Per-node additive brightness shift (black level)

    in mediump vec3 vColor;
    in mediump float vBeta; // Super-Gaussian exponent beta (per-instance)
    in highp float vRadius; // Radius from vertex shader (needs precision for zero-check)
    in mediump vec2 vSpriteCoord; // [0,1] sprite UV (replaces gl_PointCoord)

    out vec4 fragColor;

    void main() {
      // Discard zero-radius points (from nD slicing where points don't intersect hyperplane)
      if (vRadius < 0.0001) {
        discard;
      }

      // OPTIMIZATION: Use dot product for squared distance calculation
      vec2 centered = vSpriteCoord - 0.5;
      float r2 = dot(centered, centered);

      // OPTIMIZATION: Compare squared distances to avoid sqrt in discard check
      if (r2 > 0.25) {
        discard;
      }

      // OPTIMIZATION: sqrt(4.0 * r2) combines sqrt and multiply into one operation
      // normalizedR is in 0-1 range (gl_PointCoord is 0-1, centered is -0.5 to 0.5)
      mediump float normalizedR = sqrt(4.0 * r2);

      // Shifted-truncated super-Gaussian falloff:
      //   falloff(rho) = max(exp(-K * rho^beta) - C, 0) / (1 - C)
      // shifted by C and renormalised so falloff(0)=1 and falloff(1)=0
      // (C0-continuous truncation at the sprite edge, no hard ring).
      // beta=2 reproduces the gsplat Gaussian shape. K = ln(1/floor) with
      // floor = 0.01 encodes the 1% iso-contour sizing convention.
      const mediump float K = 4.6051702;          // ln(100)
      const mediump float C = 0.01;               // exp(-K) = floor
      const mediump float INV_ONE_MINUS_C = 1.0 / (1.0 - C);
      mediump float falloff = max(exp(-K * pow(normalizedR, vBeta)) - C, 0.0) * INV_ONE_MINUS_C;

      // Per-node GOG (Gain-Offset-Gamma) color adjustment.
      //
      // Colormap (LUT) mode: gamma + display-range already shaped the
      // scalar VALUE before the LUT lookup (vertex shader), so the mapped
      // color passes through untouched — gamma must NOT warp LUT colors.
      // Direct-color mode: GOG operates on the color, as intended.
      #ifdef USE_COLORMAP
      mediump vec3 adjusted = max(vColor, vec3(0.0));
      #else
      mediump vec3 adjusted = vColor * uIntensity + uOffset;
      adjusted = max(adjusted, vec3(0.0));
      #endif

      // Early discard for zero-contribution fragments after offset
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;

      // LUXAR_GAMMA_ONE (gamma == 1.0) skips the per-fragment pow() —
      // pow(x, 1) == x — same fast path the colormap branch already takes.
      #if defined(USE_COLORMAP) || defined(LUXAR_GAMMA_ONE)
      mediump vec3 finalColor = adjusted;
      #else
      mediump vec3 finalColor = pow(adjusted, vec3(invGamma));
      #endif

      // Calculate alpha (intensity) for additive blending
      mediump float alpha = falloff * opacity;

      // max-mode RGB premultiplication.
      //
      // In max blending the framebuffer uses CustomBlending +
      // MaxEquation + OneFactor/OneFactor. With that state, source RGB
      // is NOT multiplied by alpha at composite time, so a soft point
      // with alpha=0.1 still contributes its full bright RGB → max
      // captures a flat colored disk instead of the intended soft
      // contribution. The fix is to premultiply RGB in the shader so
      // the framebuffer max sees contribution-weighted colour. The
      // LUXAR_MAX_RGB_CONTRIBUTION define is set by
      // PointMaterial.applyBlendingMode('max').
      #ifdef LUXAR_MAX_RGB_CONTRIBUTION
      fragColor = vec4(finalColor * alpha, alpha);
      #else
      // Output final color with alpha for AdditiveBlending (SrcAlpha, One)
      fragColor = vec4(finalColor, alpha);
      #endif
    }
  `;

export const POINT_SOURCE: ShaderSource = {
  name: 'point',
  webgl: { vertex: POINT_VERTEX_SHADER, fragment: POINT_FRAGMENT_SHADER },
  // TSL NodeMaterial that owns vertexNode (sprite expansion)
  // and colorNode (super-Gaussian falloff + GOG). Default config — no
  // toggles. Consumers needing USE_COLORMAP / LUXAR_MAX_RGB_CONTRIBUTION
  // call `pointWebGPUFactory(uniforms, { ...flags })` directly.
  webgpu: (uniforms: Record<string, unknown>) =>
    pointWebGPUFactory(uniforms as Record<string, import('three').IUniform>),
};
