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
import { GLSL_SANITIZE_FUNCTIONS, GLSL_NEAR_FADE_FUNCTIONS } from '../_shared/glsl-lib';
import { lineWebGPUFactory, buildLineTSLNodesFromUniforms } from './shader-tsl';
import type { ShaderSource } from '../_shared/shader-source';

export const LINE_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    ${GLSL_SANITIZE_FUNCTIONS}
    ${GLSL_NEAR_FADE_FUNCTIONS}

    // Static geometry attribute (per quad vertex)
    in vec2 aQuadCorner;  // (-1,-1), (1,-1), (-1,1), (1,1)

    // Instanced attributes (per segment)
    in vec3 aStartPos;
    in vec3 aEndPos;
    // Per-vertex colours are only read in the non-colormap branch. Under
    // USE_COLORMAP the colour comes from the LUT, so these attributes are
    // omitted entirely — a line already carries many instanced attributes,
    // and declaring two unused vec3 attributes alongside the scalar pair
    // can push the active-attribute count past GL_MAX_VERTEX_ATTRIBS (16)
    // once THREE injects position/normal/uv.
    #ifndef USE_COLORMAP
    in vec3 aStartColor;
    in vec3 aEndColor;
    #else
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
    uniform vec2 uResolution;
    uniform int uIsOrtho;  // 0 = perspective, 1 = orthographic
    uniform float uNearCull;          // near-plane safety distance (view-space, +z toward camera)
    uniform float uMaxLinePixelWidth; // clamp for screen-space width
    // Precomputed CPU-side line-width scales — kill the per-vertex tan()
    // and one division. See updateCameraParams in line-material.ts.
    uniform float uPerspectiveLineScale; // = resolution.y / tan(fov * 0.5)
    uniform float uOrthoLineScale;       // = 2 * resolution.y / frustumHeight

    // Colormap uniforms (only active when USE_COLORMAP is defined)
    #ifdef USE_COLORMAP
    uniform sampler2D uColormapTex;
    uniform float uScalarMin;       // display-range window minimum
    uniform float uScalarScale;     // 1.0 / (max - min)
    uniform float uInvGamma;        // gamma applied to the VALUE, pre-LUT
    #endif

    // Varyings to fragment shader
    out vec3 vColor;
    out float vSharpness;
    out float vPerpNorm;     // Signed: -1 at bottom edge, +1 at top edge
    out float vT;            // interpolated 0..1 along segment for fragment-side cap math
    flat out float vSegmentLength; // per-segment constant — same on all 4 quad verts
    out float vWidthAtT;      // interpolated world-space width (or half-width)
    out float vPixelWidth;   // Raw line width in pixels (for anti-aliasing)
    out float vWidthFade;    // in [0..1], fades intensity when pixel-width clamped
    out float vViewZ;        // View-space z (fragment computes the near fade)
    flat out float vClippedStart; // flat: same value across all 4 quad vertices
    flat out float vClippedEnd;

    void main() {
      // Position along segment: 0 = start, 1 = end. Branchless because
      // aQuadCorner.x ∈ {-1, +1} by construction.
      float t = aQuadCorner.x * 0.5 + 0.5;
      vT = t;
      vSegmentLength = aSegmentLength;
      vClippedStart = aStartClipped;
      vClippedEnd = aEndClipped;

      // Project endpoints to view space first — the bothBehind near-cull
      // test reads view-space depth, and culling BEFORE the colormap
      // sample / width sanitisation / colour interpolation skips that
      // wasted work for off-screen segments. The pathological-cull
      // (further down) reads rawPixelWidth which depends on width,
      // so we still have to do the cheap parts of width sanitisation
      // and interpolation; the colormap branch and the full mix /
      // pow chain are the real wins.
      vec4 mvStart = modelViewMatrix * vec4(aStartPos, 1.0);
      vec4 mvEnd = modelViewMatrix * vec4(aEndPos, 1.0);

      // near-plane / behind-camera safety — PERSPECTIVE ONLY. Three.js
      // view space has -z pointing into the scene, so a positive
      // viewDepth means the point is in front of the camera. Reject
      // segments where BOTH endpoints fail the near-cull (degenerate
      // the quad to clip). When only ONE endpoint is behind, we keep
      // the full quad: the shader will produce extreme NDC for that
      // endpoint, but the pixel-width clamp and vWidthFade keep the
      // visible footprint bounded. Under ORTHO there is no 1/z
      // singularity and NDC near/far clipping is the sole cull
      // authority — the previous ungated cull WRONGLY hid in-frustum
      // lines in the near slab (< uNearCull from the camera plane)
      // where points/gsplats still drew. Matches the unified
      // perspectiveNearFade semantics (point + gsplat shaders).
      // uNearCull is scene-bounds-scaled (diagonal * 0.001); the 1e-20
      // floor only guards uNearCull == 0 (degenerate smoothstep /
      // division). An absolute 1e-4 floor overrode the scene-relative
      // value on tiny-unit scenes — every segment sat inside the
      // "both behind" margin and was culled.
      float nearCull = max(uNearCull, 1e-20);
      float startDepth = -mvStart.z;
      float endDepth = -mvEnd.z;
      bool bothBehind =
        (uIsOrtho == 0) && (startDepth < nearCull) && (endDepth < nearCull);
      if (bothBehind) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0); // off-screen → no fragments
        // Defensive: zero the remaining varyings the fragment-stage
        // can read. The rasterizer drops this segment entirely so the
        // values don't actually matter, but uninitialised out-vars can
        // trip driver validators on some platforms.
        vColor = vec3(0.0);
        vSharpness = 0.5;
        vWidthAtT = 0.0;
        vPerpNorm = 0.0;
        vPixelWidth = 0.0;
        vWidthFade = 0.0;
        vViewZ = 0.0;
        return;
      }

      vec4 mvPos = mix(mvStart, mvEnd, t);
      // View-space z travels to the FRAGMENT, which computes the near
      // fade per-fragment. Interpolating the FADE itself would be wrong
      // on long segments: fade(lerp(z)) != lerp(fade(z)) — one endpoint
      // at the camera plane would dim fragments far outside the
      // [nearCull, 2*nearCull] band (mid-segment at ~50%).
      vViewZ = mvPos.z;

      // === Below here only runs when the segment passed the cheap cull. ===

      // Interpolate attributes along segment.
      // Colormap mode: display range (uScalarMin/uScalarScale) and gamma
      // shape the scalar VALUE before the LUT lookup, not the resulting
      // color; intensity/offset apply POST-LUT to the mapped color
      // (fragment shader, matching the gsplat shader) so the layer
      // gain/offset controls work on colormapped nodes too. The gamma
      // fast path (LUXAR_GAMMA_ONE) skips the pow() when gamma == 1.0.
      #ifdef USE_COLORMAP
      float s = mix(aStartScalar, aEndScalar, t);
      float st = clamp((s - uScalarMin) * uScalarScale, 0.0, 1.0);
      #ifndef LUXAR_GAMMA_ONE
      st = pow(st, uInvGamma);          // gamma on the value, pre-LUT
      #endif
      vColor = texture(uColormapTex, vec2(st, 0.5)).rgb;
      #else
      vColor = mix(aStartColor, aEndColor, t);
      #endif

      // sanitise width/sharpness against negative/NaN/Inf so a
      // malformed input can't poison gl_Position via pow() or screen-
      // space expansion. Sanitize helpers from glsl-lib.
      float startW = sanitizeNonNegative(aStartWidth, 0.0);
      float endW = sanitizeNonNegative(aEndWidth, 0.0);
      // Sharpness is authored in [0, 1] and maps (in the fragment) to the
      // super-Gaussian exponent beta = 2^(6s - 2): s=0.5 -> beta=2 (a true
      // Gaussian, the default). sanitizeNonNegative keeps a valid s=0
      // (-> beta=0.25) and routes NaN/Inf/negative to the 0.5 default;
      // clamp guards the [0, 1] range. (NOT sanitizePositive — that would
      // wrongly reject s=0.)
      float startS = clamp(sanitizeNonNegative(aStartSharpness, 0.5), 0.0, 1.0);
      float endS = clamp(sanitizeNonNegative(aEndSharpness, 0.5), 0.0, 1.0);

      float width = mix(startW, endW, t);
      // Pass the interpolated [0, 1] sharpness KNOB to the fragment; beta is
      // computed there from the interpolated value.
      vSharpness = mix(startS, endS, t);
      vWidthAtT = width;

      vec4 clipStart = projectionMatrix * mvStart;
      vec4 clipEnd = projectionMatrix * mvEnd;
      // projection is linear, so proj * mix(a,b,t) == mix(proj*a, proj*b, t).
      vec4 clipPos = mix(clipStart, clipEnd, t);

      // Convert clip-space endpoints to pixel coordinates for correct aspect ratio handling
      // Guard against tiny clipStart.w / clipEnd.w (near-plane crossing) so 1/w
      // doesn't blow up. The guard is the SCENE-RELATIVE nearCull (w == -viewZ
      // under perspective), not an absolute epsilon: the old 1e-4 clamped VALID
      // w on tiny-unit scenes (w ~ 1e-6), collapsing every endpoint's NDC and
      // scrambling quad directions. nearCull scales with the scene, so in-front
      // endpoints are never clamped at any scale while behind/crossing endpoints
      // still get a bounded NDC (clip.xy scales with the scene too, keeping the
      // ratio finite — a raw 1e-20 floor could overflow float32 in the
      // pixel-length math below). Ortho: w == 1 exactly, guard 1.0 is inert.
      float wGuard = (uIsOrtho == 1) ? 1.0 : nearCull;
      float wStart = max(clipStart.w, wGuard);
      float wEnd = max(clipEnd.w, wGuard);
      vec2 ndcStart = clipStart.xy / wStart;
      vec2 ndcEnd = clipEnd.xy / wEnd;

      // Compute line direction in pixel space (aspect-ratio correct).
      // The +0.5 in (ndc*0.5+0.5)*resolution cancels under subtraction.
      vec2 pixelDir = (ndcEnd - ndcStart) * (0.5 * uResolution);
      float pixelLen = length(pixelDir);

      // Handle degenerate segments (zero length in screen space)
      vec2 lineDir = pixelLen > 0.0001 ? pixelDir / pixelLen : vec2(1.0, 0.0);
      vec2 perpendicular = vec2(-lineDir.y, lineDir.x);  // Unit vector in pixel space

      // World-space to pixel conversion. Both branches consume a scale
      // precomputed on the CPU once per camera/resolution change so
      // the shader avoids per-vertex tan() and FOV divisions.
      float rawPixelWidth;
      if (uIsOrtho == 1) {
        // Orthographic: constant screen size regardless of distance.
        rawPixelWidth = width * uOrthoLineScale;
      } else {
        // View-space depth instead of Euclidean distance — drops a
        // sqrt per vertex and is more projection-correct (screen-space
        // size scales with view-z, not distance from camera position).
        // Off-axis segments will be slightly different in apparent
        // width vs the old length(mvPos.xyz) form; this is the
        // intended correctness improvement.
        float dist = max(-mvPos.z, nearCull);
        rawPixelWidth = width * uPerspectiveLineScale / dist;
      }

      // Enforce minimum pixel width to prevent sub-pixel rendering artifacts
      // Lines thinner than ~1.5 pixels cause severe aliasing due to rasterization gaps
      float minPixelWidth = 1.5;
      // clamp to a maximum pixel width so a near-camera segment
      // can't paint the entire screen. Default uMaxLinePixelWidth is
      // resolution.y * 0.5 (set by JS).
      float maxPW = max(uMaxLinePixelWidth, minPixelWidth + 1.0);
      // Degenerate (extremely close to camera AND extreme pixel
      // width) segments expand into a half-viewport quad that the GPU
      // still rasterizes pixel-by-pixel. The pixel-width clamp + fade
      // keeps the visible footprint bounded but doesn't avoid the
      // shading cost — discard the segment entirely when both endpoints
      // are within the near cull margin AND rawPixelWidth blows past
      // the clamp by 2× (a clear pathological case, not a normal
      // close-up).
      // PERSPECTIVE ONLY: under ortho rawPixelWidth is depth-independent
      // (width * uOrthoLineScale), so a depth gate here would make a
      // legitimately wide line vanish only while inside the 2*nearCull
      // slab and pop back one unit deeper — depth-dependent visibility
      // with no physical rationale in a depth-independent projection.
      if (
        uIsOrtho == 0 &&
        startDepth < nearCull * 2.0 &&
        endDepth < nearCull * 2.0 &&
        rawPixelWidth > maxPW * 2.0
      ) {
        gl_Position = vec4(0.0, 0.0, -2.0, 1.0);
        vColor = vec3(0.0);
        vPerpNorm = 0.0;
        vPixelWidth = 0.0;
        vWidthFade = 0.0;
        vViewZ = 0.0;
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
 * Computes the shifted-truncated super-Gaussian perpendicular
 * cross-section (beta = 2^(6s - 2), beta=2 is a truncated Gaussian) plus
 * fragment-side cap factor so the segment body reaches the documented
 * full intensity. The picking system uses a different fragment shader
 * (see picking/line-picking-material.ts).
 */
export const LINE_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;
    ${GLSL_NEAR_FADE_FUNCTIONS}

    uniform int uIsOrtho;   // shared with the vertex stage
    uniform float uNearCull;
    uniform float uOpacity;
    uniform float uInvGamma; // Pre-computed 1/gamma for performance
    uniform float uIntensity; // Per-node linear color multiplier (gain)
    uniform float uOffset; // Per-node additive brightness shift (black level)

    in vec3 vColor;
    in float vSharpness;
    in float vPerpNorm;     // Interpolated: 0 at centerline, ±1 at edges
    in float vT;            // interpolated 0..1 along segment
    flat in float vSegmentLength; // per-segment constant
    in float vWidthAtT;     // interpolated world-space width
    in float vPixelWidth;   // Raw line width in pixels (before minimum clamping)
    in float vWidthFade;    // max-pixel-width clamp fade
    in float vViewZ; // View-space z (near fade computed here per-fragment)
    flat in float vClippedStart;
    flat in float vClippedEnd;

    out vec4 fragColor;

    void main() {
      // Compute distance from centerline (0 to 1)
      float p = abs(vPerpNorm);

      // Discard pixels clearly outside the line width
      if (p >= 1.0) discard;

      // Shifted-truncated super-Gaussian perpendicular cross-section:
      //   perpFalloff(p) = max(exp(-K * p^beta) - C, 0) / (1 - C)
      // where p = distance from centerline in [0, 1] and the per-vertex
      // sharpness KNOB s in [0, 1] maps to beta = 2^(6s - 2): s=0.5 ->
      // beta=2 (a truncated Gaussian, the default), s=1 -> beta=16 (hard
      // edge), s=0 -> beta=0.25 (cusp). Shifted by C and renormalised so
      // perpFalloff(0)=1 and perpFalloff(1)=0 (C0-continuous truncation at
      // the line edge, no hard ring). K = ln(1/floor), floor = 0.01.
      const float K = 4.6051702;          // ln(100)
      const float C = 0.01;               // exp(-K) = floor
      const float INV_ONE_MINUS_C = 1.0 / (1.0 - C);
      float beta = exp2(6.0 * vSharpness - 2.0);
      float perpFalloff = max(exp(-K * pow(p, beta)) - C, 0.0) * INV_ONE_MINUS_C;

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
      // distToNearest / vWidthAtT is a scale-free ratio (both world
      // units), so the guard is a pure div-by-zero threshold at 1e-20 —
      // an absolute 1e-4 skipped the cap ramp for valid sub-1e-4-unit
      // widths (tiny-unit scenes). clamp() bounds the quotient.
      float capRamp = vWidthAtT > 1e-20
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
      // Per-fragment near fade from the interpolated view depth (see
      // the vertex stage note on why the fade itself must not be the
      // varying).
      // 1e-20 floor = degenerate-smoothstep guard only; uNearCull is
      // scene-relative (see the vertex-stage nearCull note).
      float nearFade = perspectiveNearFade(uIsOrtho, vViewZ, max(uNearCull, 1e-20));
      float intensity = capFactor * perpFalloff * edgeAA * widthScale * vWidthFade * nearFade;

      // Per-node GOG (Gain-Offset-Gamma) color adjustment. uIntensity (gain)
      // and uOffset apply in BOTH modes so the layer intensity/offset
      // controls work for a colormapped line too (matching the gsplat
      // shader). Colormap (LUT) mode: gamma + the display-range window
      // already shaped the scalar VALUE before the LUT lookup (vertex
      // shader), so only gain/offset apply post-LUT (no extra gamma).
      // When the wrapper knows intensity==1 && offset==0 (the default),
      // the mul/add/clamp chain is identity for the common non-negative
      // vColor range; the wrapper stamps LUXAR_NO_GOG to skip it.
      #ifdef LUXAR_NO_GOG
      vec3 adjusted = vColor;
      #else
      vec3 adjusted = vColor * uIntensity + uOffset;
      adjusted = max(adjusted, vec3(0.0));
      #endif

      // Early discard for zero-contribution fragments after offset
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;

      // Gamma fast path: when gamma==1 (the default) the pow() is
      // identity. The wrapper class stamps LUXAR_GAMMA_ONE on the
      // material defines whenever gamma transitions to/from 1.0, so
      // this skips three per-fragment pow() calls in the common case.
      // Colormap mode also skips the color pow() — gamma is on the value.
      #if defined(USE_COLORMAP) || defined(LUXAR_GAMMA_ONE)
      vec3 gammaColor = adjusted;
      #else
      vec3 gammaColor = pow(adjusted, vec3(uInvGamma));
      #endif

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

export const LINE_SOURCE: ShaderSource = {
  name: 'line',
  webgl: { vertex: LINE_VERTEX_SHADER, fragment: LINE_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) => {
    const u = uniforms as Record<string, import('three').IUniform>;
    // Read `uIsOrtho` from the uniform record at build time so the
    // projection-mode graph variant matches the camera the caller set
    // up. Live ortho/perspective flips on a long-lived material go
    // through `LineTSLMaterial.updateCameraParams`, which calls
    // `rebuildGraph()` itself — this short-lived ShaderSource path
    // just needs the right variant at construction.
    const isOrtho = ((u.uIsOrtho?.value as number) ?? 0) === 1;
    return lineWebGPUFactory(buildLineTSLNodesFromUniforms(u, {}), { isOrtho });
  },
};
