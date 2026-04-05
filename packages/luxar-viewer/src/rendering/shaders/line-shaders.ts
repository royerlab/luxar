/**
 * Shared vertex shader for line rendering.
 *
 * Used by both LineMaterial (main rendering) and LinePickingMaterial (GPU picking).
 * Contains screen-space expansion, cap factor calculation, and colormap support.
 */
export const LINE_VERTEX_SHADER = /* glsl */ `
    precision highp float;

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

    // Colormap uniforms (only active when USE_COLORMAP is defined)
    #ifdef USE_COLORMAP
    uniform sampler2D uColormapTex;
    uniform float uScalarMin;
    uniform float uScalarScale;
    #endif

    // Varyings to fragment shader (smooth interpolation needed)
    out vec3 vColor;
    out float vSharpness;
    out float vPerpNorm;  // Signed: -1 at bottom edge, +1 at top edge
    out float vCapFactor; // 0.5 at true endpoints, 1.0 in body
    out float vPixelWidth; // Line width in pixels (for anti-aliasing)

    void main() {
      // Position along segment: 0 = start, 1 = end
      float t = aQuadCorner.x > 0.0 ? 1.0 : 0.0;

      // Interpolate attributes along segment
      vec3 worldPos = mix(aStartPos, aEndPos, t);
      #ifdef USE_COLORMAP
      float s = mix(aStartScalar, aEndScalar, t);
      float st = clamp((s - uScalarMin) * uScalarScale, 0.0, 1.0);
      vColor = texture(uColormapTex, vec2(st, 0.5)).rgb;
      #else
      vColor = mix(aStartColor, aEndColor, t);
      #endif
      float width = mix(aStartWidth, aEndWidth, t);
      vSharpness = mix(aStartSharpness, aEndSharpness, t);

      // Project to clip space (pre-multiply modelViewMatrix once per endpoint)
      vec4 mvStart = modelViewMatrix * vec4(aStartPos, 1.0);
      vec4 mvEnd = modelViewMatrix * vec4(aEndPos, 1.0);
      vec4 mvPos = mix(mvStart, mvEnd, t);
      vec4 clipStart = projectionMatrix * mvStart;
      vec4 clipEnd = projectionMatrix * mvEnd;
      vec4 clipPos = projectionMatrix * mvPos;

      // Convert clip-space endpoints to pixel coordinates for correct aspect ratio handling
      // NDC to pixels: ndc * resolution / 2 (NDC range -1 to +1, pixels range 0 to resolution)
      vec2 ndcStart = clipStart.xy / clipStart.w;
      vec2 ndcEnd = clipEnd.xy / clipEnd.w;
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
        rawPixelWidth = width * uResolution.y / uFOV;
      } else {
        float dist = length(mvPos.xyz);
        float tanHalfFov = tan(uFOV * 0.5);
        rawPixelWidth = width * uResolution.y / (dist * tanHalfFov);
      }

      // Enforce minimum pixel width to prevent sub-pixel rendering artifacts
      // Lines thinner than ~1.5 pixels cause severe aliasing due to rasterization gaps
      float minPixelWidth = 1.5;
      float pixelWidth = max(rawPixelWidth, minPixelWidth);

      // Pass raw pixel width to fragment shader for intensity scaling
      // This allows thin lines to render at minimum width but with reduced intensity
      vPixelWidth = rawPixelWidth;

      // Perpendicular position: -1 at bottom edge, +1 at top edge
      // GPU interpolates this across the quad, giving 0 at centerline
      vPerpNorm = aQuadCorner.y;

      // Expand quad by perpendicular offset in pixel space, then convert to clip space
      // pixelOffset is in pixels, convert to NDC then to clip space
      vec2 pixelOffset = perpendicular * aQuadCorner.y * pixelWidth;
      vec2 ndcOffset = pixelOffset / uResolution * 2.0;
      clipPos.xy += ndcOffset * clipPos.w;

      // Cap factor calculation with clipping awareness
      // Normal cap factor: 0.5 at true endpoints, 1.0 in body
      // Clipped endpoints: force 1.0 (the "real" endpoint is outside the slice)
      float distFromStart = t * aSegmentLength;
      float distFromEnd = (1.0 - t) * aSegmentLength;

      // Base cap factor from distance to nearest endpoint
      float distToNearest = min(distFromStart, distFromEnd);
      float baseCap = (distToNearest >= width) ? 1.0 : 0.5 + 0.5 * (distToNearest / width);

      // Override if the nearest endpoint was clipped
      float nearestIsStart = step(distFromEnd, distFromStart);  // 1 if closer to start
      float nearestClipped = mix(aEndClipped, aStartClipped, nearestIsStart);

      // If nearest endpoint was clipped, use full intensity (1.0)
      vCapFactor = mix(baseCap, 1.0, nearestClipped);

      gl_Position = clipPos;
    }
  `;

/**
 * Fragment shader for standard line rendering.
 *
 * Computes parabolic falloff from semicircle kernel convolution,
 * with cap factor for correct joint intensity.
 * The picking system uses a different fragment shader (see picking-materials.ts).
 */
export const LINE_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    uniform float uOpacity;
    uniform float uInvGamma; // Pre-computed 1/gamma for performance
    uniform float uIntensity; // Per-node linear color multiplier (gain)
    uniform float uOffset; // Per-node additive brightness shift (black level)

    in vec3 vColor;
    in float vSharpness;
    in float vPerpNorm;  // Interpolated: 0 at centerline, ±1 at edges
    in float vCapFactor; // 0.5 at endpoints, 1.0 in body
    in float vPixelWidth; // Raw line width in pixels (before minimum clamping)

    out vec4 fragColor;

    void main() {
      // Compute distance from centerline (0 to 1)
      float p = abs(vPerpNorm);

      // Discard pixels clearly outside the line width
      if (p >= 1.0) discard;

      // Parabolic falloff from semicircle kernel convolution
      // Base: (1 - p²) where p = distance from centerline
      // With per-vertex sharpness: (1 - p²)^sharpness
      float perpFalloff = pow(1.0 - p * p, vSharpness);

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

      // Apply cap factor for correct joint intensity
      float intensity = vCapFactor * perpFalloff * edgeAA * widthScale;

      // Per-node GOG (Gain-Offset-Gamma) color adjustment
      vec3 adjusted = vColor * uIntensity + uOffset;
      adjusted = max(adjusted, vec3(0.0));

      // Early discard for zero-contribution fragments after offset
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;

      vec3 gammaColor = pow(adjusted, vec3(uInvGamma));

      // Output color with alpha for AdditiveBlending (SrcAlpha, One)
      vec3 finalColor = gammaColor;
      fragColor = vec4(finalColor, intensity * uOpacity);
    }
  `;
