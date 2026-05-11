/**
 * Shared vertex shader for line rendering.
 *
 * Used by both LineMaterial (main rendering) and LinePickingMaterial (GPU picking).
 * Contains screen-space expansion, cap factor calculation, and colormap support.
 *
 * Shader contracts:
 *   - Cap factor is evaluated in the fragment shader. The vertex shader
 *     passes `vT`, `vSegmentLength`, `vWidthAtT`, `vClippedStart`, and
 *     `vClippedEnd`; the fragment evaluates the documented "0.5 at
 *     endpoints, 1.0 in body" profile per fragment.
 *   - Near-plane / behind-camera safety rejects segments where both
 *     endpoints are behind/near the camera (clipPos.w → 0/negative
 *     produces invalid NDC and a full-screen quad). When `uNearCull` is
 *     set, segments closer than that view-space depth are degenerated.
 *   - Max pixel width clamp. `pixelWidth` is clamped to
 *     `uMaxLinePixelWidth` (default `resY * 0.5`); when the clamp
 *     engages, intensity fades proportionally so a single very-near
 *     segment doesn't paint the screen.
 *   - Width and sharpness sanitised against negative/NaN/Inf.
 *
 * Shader defines:
 *   - `USE_COLORMAP` — enables the per-vertex scalar attribute +
 *     colormap LUT path. Set in `LineMaterial` when the geometry
 *     binds `aStartScalar`/`aEndScalar`.
 *   - `LUXAR_MAX_RGB_CONTRIBUTION` — fragment-side define that
 *     premultiplies `rgb *= alpha` before output so the
 *     `MaxEquation` + `OneFactor`/`OneFactor` blend captures
 *     contribution-weighted colour (a bright-but-thin fragment loses
 *     against a dim-but-thick fragment), not flat full-bright. Set by
 *     `LineMaterial.applyBlendingMode('max')` and cleared when
 *     switching back. Without this define, max-mode line rendering
 *     looks "stranded" — every visible fragment paints at full
 *     intensity regardless of opacity, intensity, or fade.
 */
import { GLSL_SANITIZE_FUNCTIONS } from './glsl-lib';

export const LINE_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}

    // Static geometry attribute (per quad vertex)
    in vec2 aQuadCorner;  // (-1,-1), (1,-1), (-1,1), (1,1)

    // Instanced attributes (per segment)
    in vec3 aStartPos;
    in vec3 aEndPos;
    in vec3 aStartColor;
    in vec3 aEndColor;
    #ifdef USE_COLORMAP
    in float aStartScalar;
    in float aEndScalar;
    #endif
    in float aStartWidth;
    in float aEndWidth;
    in float aStartSharpness;
    in float aEndSharpness;
    in float aSegmentLength;
    in float aStartClipped;
    in float aEndClipped;

    // Uniforms
    uniform float uFOV;
    uniform vec2 uResolution;
    uniform int uIsOrtho;  // 0 = perspective, 1 = orthographic
    uniform float uNearCull;          // near-plane safety distance (view-space, +z toward camera)
    uniform float uMaxLinePixelWidth; // clamp for screen-space width

    // Colormap uniforms (only active when USE_COLORMAP is defined)
    #ifdef USE_COLORMAP
    uniform sampler2D uColormapTex;
    uniform float uScalarMin;
    uniform float uScalarScale;
    #endif

    // Varyings to fragment shader (smooth interpolation needed)
    out vec3 vColor;
    out float vSharpness;
    out float vPerpNorm;     // Signed: -1 at bottom edge, +1 at top edge
    out float vT;            // interpolated 0..1 along segment for fragment-side cap math
    out float vSegmentLength; // world-space segment length (per-segment, but flat over quad)
    out float vWidthAtT;      // interpolated world-space width (or half-width)
    out float vPixelWidth;   // Raw line width in pixels (for anti-aliasing)
    out float vWidthFade;    // in [0..1], fades intensity when pixel-width clamped
    flat out float vClippedStart; // flat: same value across all 4 quad vertices
    flat out float vClippedEnd;

    void main() {
      // Position along segment: 0 = start, 1 = end
      float t = aQuadCorner.x > 0.0 ? 1.0 : 0.0;
      vT = t;
      vSegmentLength = aSegmentLength;
      vClippedStart = aStartClipped;
      vClippedEnd = aEndClipped;

      // Interpolate attributes along segment
      vec3 worldPos = mix(aStartPos, aEndPos, t);
      #ifdef USE_COLORMAP
      float s = mix(aStartScalar, aEndScalar, t);
      float st = clamp((s - uScalarMin) * uScalarScale, 0.0, 1.0);
      vColor = texture(uColormapTex, vec2(st, 0.5)).rgb;
      #else
      vColor = mix(aStartColor, aEndColor, t);
      #endif

      // sanitise width/sharpness against negative/NaN/Inf so a
      // malformed input can't poison gl_Position via pow() or screen-
      // space expansion. Sanitize helpers from glsl-lib.
      float startW = sanitizeNonNegative(aStartWidth, 0.0);
      float endW = sanitizeNonNegative(aEndWidth, 0.0);
      float startS = sanitizePositive(aStartSharpness, 2.0);
      float endS = sanitizePositive(aEndSharpness, 2.0);

      float width = mix(startW, endW, t);
      vSharpness = mix(startS, endS, t);
      vWidthAtT = width;

      // Project to clip space (pre-multiply modelViewMatrix once per endpoint)
      vec4 mvStart = modelViewMatrix * vec4(aStartPos, 1.0);
      vec4 mvEnd = modelViewMatrix * vec4(aEndPos, 1.0);
      vec4 mvPos = mix(mvStart, mvEnd, t);

      // near-plane / behind-camera safety. Three.js view space has
      // -z pointing into the scene, so a positive viewDepth means the
      // point is in front of the camera. Reject segments where BOTH
      // endpoints fail the near-cull (degenerate the quad to clip).
      // When only ONE endpoint is behind, we keep the full quad: the
      // shader will produce extreme NDC for that endpoint, but the
      // pixel-width clamp and vWidthFade keep the visible footprint
      // bounded. Matches the GSplat near-fade pattern.
      float nearCull = max(uNearCull, 1e-4);
      float startDepth = -mvStart.z;
      float endDepth = -mvEnd.z;
      bool bothBehind = (startDepth < nearCull) && (endDepth < nearCull);
      if (bothBehind) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0); // off-screen (NDC > 1) → no fragments
        vColor = vec3(0.0);
        vPerpNorm = 0.0;
        vPixelWidth = 0.0;
        vWidthFade = 0.0;
        return;
      }

      vec4 clipStart = projectionMatrix * mvStart;
      vec4 clipEnd = projectionMatrix * mvEnd;
      vec4 clipPos = projectionMatrix * mvPos;

      // Convert clip-space endpoints to pixel coordinates for correct aspect ratio handling
      // Guard against tiny clipStart.w / clipEnd.w (near-plane crossing) so 1/w doesn't blow up
      float wStart = max(clipStart.w, 1e-4);
      float wEnd = max(clipEnd.w, 1e-4);
      vec2 ndcStart = clipStart.xy / wStart;
      vec2 ndcEnd = clipEnd.xy / wEnd;
      vec2 pixelStart = (ndcStart * 0.5 + 0.5) * uResolution;
      vec2 pixelEnd = (ndcEnd * 0.5 + 0.5) * uResolution;

      // Compute line direction and perpendicular in pixel space (aspect-ratio correct)
      vec2 pixelDir = pixelEnd - pixelStart;
      float pixelLen = length(pixelDir);

      // Handle degenerate segments (zero length in screen space)
      vec2 lineDir = pixelLen > 0.0001 ? pixelDir / pixelLen : vec2(1.0, 0.0);
      vec2 perpendicular = vec2(-lineDir.y, lineDir.x);  // Unit vector in pixel space

      // World-space to pixel conversion
      float rawPixelWidth;
      if (uIsOrtho == 1) {
        // Orthographic: constant screen size regardless of distance
        // uFOV stores frustumHeight in ortho mode
        // Factor of 2 matches the perspective formula (which has implicit 2x from 1/tanHalfFov)
        rawPixelWidth = width * 2.0 * uResolution.y / uFOV;
      } else {
        float dist = max(length(mvPos.xyz), nearCull); // clamp dist to avoid 1/near-zero blow-up
        float tanHalfFov = tan(uFOV * 0.5);
        rawPixelWidth = width * uResolution.y / (dist * tanHalfFov);
      }

      // Enforce minimum pixel width to prevent sub-pixel rendering artifacts
      // Lines thinner than ~1.5 pixels cause severe aliasing due to rasterization gaps
      float minPixelWidth = 1.5;
      // clamp to a maximum pixel width so a near-camera segment
      // can't paint the entire screen. Default uMaxLinePixelWidth is
      // resolution.y * 0.5 (set by JS).
      float maxPW = max(uMaxLinePixelWidth, minPixelWidth + 1.0);
      // E.1: degenerate (extremely close to camera AND extreme pixel
      // width) segments expand into a half-viewport quad that the GPU
      // still rasterizes pixel-by-pixel. The pixel-width clamp + fade
      // keeps the visible footprint bounded but doesn't avoid the
      // shading cost — discard the segment entirely when both endpoints
      // are within the near cull margin AND rawPixelWidth blows past
      // the clamp by 2× (a clear pathological case, not a normal
      // close-up).
      if (
        startDepth < nearCull * 2.0 &&
        endDepth < nearCull * 2.0 &&
        rawPixelWidth > maxPW * 2.0
      ) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vColor = vec3(0.0);
        vPerpNorm = 0.0;
        vPixelWidth = 0.0;
        vWidthFade = 0.0;
        return;
      }
      float clampedPixelWidth = clamp(rawPixelWidth, minPixelWidth, maxPW);
      // Fade intensity in proportion to the clamp so the giant quad
      // doesn't overcontribute. fade=1 when not clamped, →0 as the
      // raw width grows past the clamp by a factor.
      vWidthFade = (rawPixelWidth <= maxPW) ? 1.0 : (maxPW / max(rawPixelWidth, 1e-4));

      // Pass raw pixel width to fragment shader for intensity scaling
      // This allows thin lines to render at minimum width but with reduced intensity
      vPixelWidth = rawPixelWidth;

      // Perpendicular position: -1 at bottom edge, +1 at top edge
      // GPU interpolates this across the quad, giving 0 at centerline
      vPerpNorm = aQuadCorner.y;

      // Expand quad by perpendicular offset in pixel space, then convert to clip space
      // pixelOffset is in pixels, convert to NDC then to clip space
      vec2 pixelOffset = perpendicular * aQuadCorner.y * clampedPixelWidth;
      vec2 ndcOffset = pixelOffset / uResolution * 2.0;
      clipPos.xy += ndcOffset * clipPos.w;

      gl_Position = clipPos;
    }
  `;

/**
 * Fragment shader for standard line rendering.
 *
 * Computes parabolic falloff from semicircle kernel convolution plus
 * fragment-side cap factor so the segment body reaches the documented
 * full intensity. The picking system uses a different fragment shader
 * (see picking/line-picking-material.ts).
 */
export const LINE_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    uniform float uOpacity;
    uniform float uInvGamma; // Pre-computed 1/gamma for performance
    uniform float uIntensity; // Per-node linear color multiplier (gain)
    uniform float uOffset; // Per-node additive brightness shift (black level)

    in vec3 vColor;
    in float vSharpness;
    in float vPerpNorm;     // Interpolated: 0 at centerline, ±1 at edges
    in float vT;            // interpolated 0..1 along segment
    in float vSegmentLength; // world-space segment length
    in float vWidthAtT;     // interpolated world-space width
    in float vPixelWidth;   // Raw line width in pixels (before minimum clamping)
    in float vWidthFade;    // max-pixel-width clamp fade
    flat in float vClippedStart;
    flat in float vClippedEnd;

    out vec4 fragColor;

    void main() {
      // Compute distance from centerline (0 to 1)
      float p = abs(vPerpNorm);

      // Discard pixels clearly outside the line width
      if (p >= 1.0) discard;

      // Parabolic falloff from semicircle kernel convolution
      // Base: (1 - p²) where p = distance from centerline
      // With per-vertex sharpness: (1 - p²)^sharpness
      float perpFalloff = pow(max(1.0 - p * p, 0.0), max(vSharpness, 0.0001));

      // Anti-aliasing: smooth falloff at edges
      // The AA region is ~1 pixel wide in the rendered quad
      // Since we enforce minimum 1.5px width, use that as reference
      float minPixelWidth = 1.5;
      float renderedWidth = max(vPixelWidth, minPixelWidth);
      float aaWidth = 1.0 / renderedWidth;  // ~1 pixel in normalized coords
      float edgeAA = 1.0 - smoothstep(1.0 - aaWidth, 1.0, p);

      // Intensity scaling for sub-pixel lines
      // When a line is rendered wider than intended, reduce intensity proportionally
      // This preserves the visual "weight" of thin lines
      float widthScale = min(vPixelWidth / minPixelWidth, 1.0);

      // cap factor in fragment. With the 4-vertex quad, vertex-side
      // computation produced 0.5 everywhere. Compute it here so the
      // segment body reaches the documented 1.0.
      // - distance to nearest endpoint along the segment (world units)
      // - if that endpoint was clipped, use full intensity (1.0)
      float distFromStart = vT * vSegmentLength;
      float distFromEnd = (1.0 - vT) * vSegmentLength;
      float distToNearest = min(distFromStart, distFromEnd);
      float capRamp = vWidthAtT > 1e-4
        ? clamp(distToNearest / vWidthAtT, 0.0, 1.0)
        : 1.0;
      float baseCap = 0.5 + 0.5 * capRamp;

      // Override if the nearest endpoint was clipped (the "real"
      // endpoint is outside the slice — full intensity is correct).
      // step(distFromStart, distFromEnd) is 1 when distFromEnd >= distFromStart,
      // i.e. the START is the nearest endpoint.
      float nearestIsStart = step(distFromStart, distFromEnd);
      float nearestClipped = mix(vClippedEnd, vClippedStart, nearestIsStart);
      float capFactor = mix(baseCap, 1.0, nearestClipped);

      // Apply cap factor for correct joint intensity
      float intensity = capFactor * perpFalloff * edgeAA * widthScale * vWidthFade;

      // Per-node GOG (Gain-Offset-Gamma) color adjustment
      vec3 adjusted = vColor * uIntensity + uOffset;
      adjusted = max(adjusted, vec3(0.0));

      // Early discard for zero-contribution fragments after offset
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;

      vec3 gammaColor = pow(adjusted, vec3(uInvGamma));

      // max-mode RGB premultiplication. With CustomBlending +
      // MaxEquation + OneFactor/OneFactor the source RGB isn't
      // multiplied by alpha at composite time, so a soft line in max
      // mode would render as a flat full-bright quad. Premultiply by
      // intensity*opacity here so the framebuffer max captures
      // contribution-weighted colour. Other modes keep alpha-weighted
      // output.
      #ifdef LUXAR_MAX_RGB_CONTRIBUTION
      float a = intensity * uOpacity;
      fragColor = vec4(gammaColor * a, a);
      #else
      vec3 finalColor = gammaColor;
      fragColor = vec4(finalColor, intensity * uOpacity);
      #endif
    }
  `;
