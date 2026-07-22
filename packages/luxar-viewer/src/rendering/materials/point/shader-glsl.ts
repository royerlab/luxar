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
 * Per-point data comes from the RGBA32F point texture (`uPointTex`,
 * 3 texels/point — layout in `rendering/point-geometry.ts` /
 * `rendering/element-texture-layout.ts`), fetched in the vertex stage
 * via `texelFetch` and indexed by the only per-instance attribute:
 *   - aSortedIndex (uint) — draw-slot → storage-slot mapping
 *     (identity in Phase 1; the sort worker permutes it in Phase 2+)
 */
import { GLSL_SANITIZE_FUNCTIONS, GLSL_NEAR_FADE_FUNCTIONS } from '../_shared/glsl-lib';
import type { ShaderSource } from '../_shared/shader-source';
import { pointWebGPUFactory, buildPointTSLNodesFromUniforms } from './shader-tsl';

export const POINT_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}

    // Per-vertex (4 corners): -1..1 normalised quad coordinates.
    in vec2 aQuadCorner;

    // Draw-slot → storage-slot mapping. Identity in Phase 1; the sort
    // worker permutes it (Phase 2+) so draw order tracks view depth
    // without rewriting point data. Uint32Array attribute → bound via
    // vertexAttribIPointer, matching this uint declaration.
    in uint aSortedIndex;

    // Point data texture: RGBA32F, 3 texels/point (see
    // rendering/element-texture-layout.ts for the texel layout).
    uniform highp sampler2D uPointTex;

    #ifdef USE_COLORMAP
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
    uniform float uNearCull;       // Near-fade start distance (world units)

    out mediump vec3 vColor;
    out mediump float vBeta;       // Super-Gaussian exponent beta (per-instance)
    out highp float vRadius;       // Pass radius to fragment for zero-check (needs precision)
    out mediump vec2 vSpriteCoord; // [0, 1] sprite UV, replaces gl_PointCoord
    out mediump float vPointSize;  // RAW projected size (pre-clamp) for sub-pixel compensation
    out mediump float vNearFade;   // Perspective near fade (1.0 under ortho)

    void main() {
      // === Point-texture fetch prologue ===
      // texelFetch reads reconstruct the per-point values into the exact
      // local names the math below has always used — zero changes
      // downstream of this block. The width is a multiple of 3
      // (element-texture-layout.ts), so a point's 3 texels share one row
      // and only x advances. texel2 is fetched only under USE_COLORMAP
      // (the scalar slot); texel2.y (per-point alpha) is reserved for
      // volumetric Phase 3 and not read here.
      int pointBase = int(aSortedIndex) * 3;
      int pointTexW = textureSize(uPointTex, 0).x;
      ivec2 texel0 = ivec2(pointBase % pointTexW, pointBase / pointTexW);
      vec4 pointT0 = texelFetch(uPointTex, texel0, 0);
      vec4 pointT1 = texelFetch(uPointTex, ivec2(texel0.x + 1, texel0.y), 0);
      vec3 aCenter = pointT0.xyz;      // world-space centre
      float aRadius = pointT0.w;
      vec3 aColor = pointT1.rgb;
      float aSharpness = pointT1.w;
      #ifdef USE_COLORMAP
      vec4 pointT2 = texelFetch(uPointTex, ivec2(texel0.x + 2, texel0.y), 0);
      float aScalar = pointT2.x;       // per-point scalar for colormap lookup
      #endif

      // Pass vertex color — either from attribute or colormap LUT.
      // In colormap mode the display range (uScalarMin/uScalarScale) and
      // gamma shape the scalar VALUE before the LUT lookup, not the
      // resulting color. Intensity/offset apply POST-LUT to the mapped
      // color (fragment shader, matching the gsplat shader) so the layer
      // gain/offset controls work on colormapped nodes too.
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

      // Unified near handling (matches line + gsplat shaders): behind-
      // camera vertices fade to 0 (the quad expansion multiplies by
      // projCenter.w, which is <= 0 there and would flip the sprite),
      // near-plane approach fades smoothly across [nearCull, 2*nearCull]
      // instead of drawing a full-brightness maxPointSize sprite until
      // z crosses 0. Ortho: fade = 1, NDC clipping is the authority.
      vNearFade = perspectiveNearFade(uIsOrtho, mvPosition.z, max(uNearCull, 1e-4));
      if (vNearFade < 0.01) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0); // off-screen → no fragments
        return;
      }

      vec4 projCenter = projectionMatrix * mvPosition;

      // World-space point sizing from VIEW-SPACE DEPTH (-mvPosition.z),
      // matching the line + gsplat shaders: screen-space size scales
      // with view-z, not Euclidean distance from the camera position,
      // so identical points render the same size across the field of
      // view (Euclidean shrank edge-of-screen points by cos(theta)).
      // The 1e-4 floor mirrors the line shader's nearCull floor — the
      // behind-camera reject above only guarantees z < 0, not z << 0.
      // pointSizeFactor pre-computed in JS: 2.0 * resolution.y / tanHalfFov
      float invDistance = (uIsOrtho == 1) ? 1.0 : 1.0 / max(-mvPosition.z, 1e-4);
      float basePointSize = normalizedRadius * pointSizeFactor * invDistance;

      // The shifted-truncated super-Gaussian falloff (fragment shader)
      // truncates to zero exactly at the sprite edge (rho = 1), so the
      // sprite size already IS the visible extent — no sharpness-dependent
      // size compensation is needed (the old polynomial kernel required it).
      //
      // Minimum sprite size 1.5px, matching the LINE shader: quads
      // thinner than ~1.5px cause rasterization gaps (flicker).
      // Sub-pixel points keep their visual weight via the fragment's
      // sizeScale^2 energy compensation (vPointSize carries the raw,
      // pre-clamp size). Zero-radius filtering happens in the fragment.
      vPointSize = basePointSize;
      float pointSize = clamp(basePointSize, 1.5, maxPointSize);

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
    in mediump float vPointSize; // Raw pre-clamp sprite size (sub-pixel compensation)
    in mediump float vNearFade; // Perspective near fade (1.0 under ortho)

    out vec4 fragColor;

    void main() {
      // Discard zero-radius points (from nD slicing where points don't intersect hyperplane)
      // Exact-zero only — an absolute epsilon here discarded valid
      // sub-1e-4-unit radii (tiny-unit scenes rendered black). Boundary
      // dust from nD slicing is filtered scale-relatively upstream
      // (data/points/projection.ts).
      if (vRadius <= 0.0) {
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

      // Per-node GOG (Gain-Offset-Gamma) color adjustment. uIntensity (gain)
      // and uOffset apply in BOTH modes so the layer intensity/offset controls
      // work for a colormapped point too (matching the gsplat shader).
      // Colormap (LUT) mode: gamma + the display-range window already shaped
      // the scalar VALUE before the LUT lookup (vertex shader), so only
      // gain/offset apply post-LUT (no extra gamma). Direct-color mode: full
      // GOG on the raw color.
      mediump vec3 adjusted = max(vColor * uIntensity + uOffset, vec3(0.0));

      // Early discard for zero-contribution fragments after offset
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;

      // LUXAR_GAMMA_ONE (gamma == 1.0) skips the per-fragment pow() —
      // pow(x, 1) == x — same fast path the colormap branch already takes.
      #if defined(USE_COLORMAP) || defined(LUXAR_GAMMA_ONE)
      mediump vec3 finalColor = adjusted;
      #else
      mediump vec3 finalColor = pow(adjusted, vec3(invGamma));
      #endif

      // Sub-pixel intensity compensation (mirrors the line shader's
      // widthScale, SQUARED because both sprite dimensions clamp:
      // energy ∝ area ∝ size²). Points at or above the 1.5px floor
      // are unaffected (sizeScale = 1).
      mediump float sizeScale = min(vPointSize / 1.5, 1.0);

      // Calculate alpha (intensity) for additive blending
      mediump float alpha = falloff * opacity * sizeScale * sizeScale * vNearFade;

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
    pointWebGPUFactory(
      buildPointTSLNodesFromUniforms(uniforms as Record<string, import('three').IUniform>)
    ),
};
