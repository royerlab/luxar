/**
 * Shared vertex shader for point rendering.
 *
 * Used by both PointMaterial (main rendering) and PointPickingMaterial (GPU picking).
 * Contains world-space sizing, sharpness compensation, nD slicing, and projection logic.
 */
export const POINT_VERTEX_SHADER = /* glsl */ `
    precision highp float;

    in float radius;
    in float sharpness;
    // NOTE: "in vec3 color" is auto-injected by THREE.js when vertexColors=true (see constructor).
    // In colormap mode, vertexColors=false so "color" is not available — use scalar + LUT instead.
    #ifdef USE_COLORMAP
    in float scalar;              // Per-point scalar for colormap lookup
    uniform sampler2D uColormapTex;   // 256x1 LUT texture
    uniform float uScalarMin;         // Scalar range minimum
    uniform float uScalarScale;       // 1.0 / (max - min)
    #endif
    uniform float pointSizeFactor; // Pre-computed: 2.0 * resolution.y / tanHalfFov (or 4.0 * resolution.y / frustumHeight for ortho)
    uniform float maxPointSize;    // Pre-computed: resolution.y * 0.5
    uniform float radiusScale;
    uniform float sharpnessScale;
    uniform int uIsOrtho;          // 0 = perspective, 1 = orthographic

    out mediump vec3 vColor;
    out mediump float vSharpness;
    out highp float vRadius; // Pass radius to fragment for zero-check (needs precision)

    void main() {
      // Pass vertex color — either from attribute or colormap LUT
      #ifdef USE_COLORMAP
      float t = clamp((scalar - uScalarMin) * uScalarScale, 0.0, 1.0);
      vColor = texture(uColormapTex, vec2(t, 0.5)).rgb;
      #else
      vColor = color;
      #endif

      // Apply sharpness scale for dtype normalization and use 2.0 as default
      float normalizedSharpness = sharpness * sharpnessScale;
      vSharpness = normalizedSharpness > 0.0 ? normalizedSharpness : 2.0;

      // Transform vertex position from world space to view space
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mvPosition;

      // Apply radius scale for dtype normalization (e.g., uint8 needs 1/255 scale)
      float normalizedRadius = radius * radiusScale;
      vRadius = normalizedRadius; // Pass to fragment shader

      // OPTIMIZED world-space point sizing:
      // - inversesqrt is a native GPU instruction (faster than sqrt + divide)
      // - pointSizeFactor pre-computed in JS: 2.0 * resolution.y / tanHalfFov
      float invDistance = (uIsOrtho == 1) ? 1.0 : inversesqrt(dot(mvPosition.xyz, mvPosition.xyz));
      float basePointSize = normalizedRadius * pointSizeFactor * invDistance;

      // Sharpness compensation based on visibility threshold
      // For falloff function f(r) = (1-r)^s, the visible radius where intensity drops to 1% is:
      // r_vis = 1 - 0.01^(1/s)
      // We need to scale the point size by 1/r_vis to maintain consistent visible size
      // Exact formula: compensation = 1 / (1 - 0.01^(1/s)), guarded against s=0
      float sharpnessCompensation = 1.0 / (1.0 - pow(0.01, 1.0 / max(vSharpness, 0.01)));
      float pointSize = basePointSize * sharpnessCompensation;

      // Clamp to hardware limits, with minimum of 1.0 to avoid undefined behavior
      // Zero-radius filtering happens in fragment shader
      gl_PointSize = max(1.0, min(pointSize, maxPointSize));
    }
  `;

/**
 * Fragment shader for standard point rendering.
 *
 * Computes Gaussian falloff, GOG color adjustment, and alpha output.
 * The picking system uses a different fragment shader (see picking-materials.ts).
 */
export const POINT_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;

    uniform mediump float opacity;
    uniform mediump float invGamma; // Pre-computed 1/gamma for performance
    uniform mediump float uIntensity; // Per-node linear color multiplier (gain)
    uniform mediump float uOffset; // Per-node additive brightness shift (black level)

    in mediump vec3 vColor;
    in mediump float vSharpness;
    in highp float vRadius; // Radius from vertex shader (needs precision for zero-check)

    out vec4 fragColor;

    void main() {
      // Discard zero-radius points (from nD slicing where points don't intersect hyperplane)
      if (vRadius < 0.0001) {
        discard;
      }

      // OPTIMIZATION: Use dot product for squared distance calculation
      vec2 centered = gl_PointCoord - 0.5;
      float r2 = dot(centered, centered);

      // OPTIMIZATION: Compare squared distances to avoid sqrt in discard check
      if (r2 > 0.25) {
        discard;
      }

      // OPTIMIZATION: sqrt(4.0 * r2) combines sqrt and multiply into one operation
      // normalizedR is in 0-1 range (gl_PointCoord is 0-1, centered is -0.5 to 0.5)
      mediump float normalizedR = sqrt(4.0 * r2);

      // Simple power function for falloff - modern GPUs optimize pow() well
      mediump float falloff = pow(max(1.0 - normalizedR, 0.0), vSharpness);

      // Per-node GOG (Gain-Offset-Gamma) color adjustment
      mediump vec3 adjusted = vColor * uIntensity + uOffset;
      adjusted = max(adjusted, vec3(0.0));

      // Early discard for zero-contribution fragments after offset
      if (max(adjusted.r, max(adjusted.g, adjusted.b)) < 1e-4) discard;

      mediump vec3 finalColor = pow(adjusted, vec3(invGamma));

      // Calculate alpha (intensity) for additive blending
      mediump float alpha = falloff * opacity;

      // Output final color with alpha for AdditiveBlending (SrcAlpha, One)
      fragColor = vec4(finalColor, alpha);
    }
  `;
