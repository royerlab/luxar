/**
 * Shared GLSL helper snippets injected into the Points / Lines / GSplats
 * shaders.
 *
 * Keeping the sanitize helpers in one place lets us:
 *   - change the sanitize policy once (e.g., to also reject denormals)
 *     without hunting through three shader source strings;
 *   - test the GLSL-string content with a single regex assertion;
 *   - make the intent explicit at the call site.
 *
 * Helpers:
 *   - `isInvalidFloat(v)`         — true if v is NaN or +/-Inf.
 *   - `sanitizePositive(v, fb)`   — returns v if v > 0 and finite, else fb.
 *   - `sanitizeNonNegative(v, fb)` — returns v if v ≥ 0 and finite, else fb.
 *
 * Inject by prepending `GLSL_SANITIZE_FUNCTIONS` to a shader source
 * string before the main() block:
 *
 * ```ts
 * const vertexShader = `
 *   uniform vec2 uResolution;
 *   ${GLSL_SANITIZE_FUNCTIONS}
 *   void main() { ... }
 * `;
 * ```
 *
 * @module rendering/shaders/glsl-lib
 */

export const GLSL_SANITIZE_FUNCTIONS = `
bool isInvalidFloat(float v) {
  return isnan(v) || isinf(v);
}

float sanitizePositive(float v, float fallback) {
  return (isInvalidFloat(v) || v <= 0.0) ? fallback : v;
}

float sanitizeNonNegative(float v, float fallback) {
  return (isInvalidFloat(v) || v < 0.0) ? fallback : v;
}
`;

/**
 * Unified perspective near-plane fade, shared by the point / line /
 * gsplat vertex shaders (visual + pick).
 *
 * Perspective: 0.0 behind the camera (viewZ >= 0 — camera looks down
 * -Z, and the quad expansion math flips/degenerates for such
 * vertices), smoothstep fade across [nearCull, 2*nearCull], else 1.0.
 * Ortho: always 1.0 — there is no 1/z singularity, and NDC near/far
 * clipping is the sole cull authority (an explicit vertex-level cull
 * under ortho WRONGLY hid in-frustum content in the near slab).
 *
 * Callers reject the vertex when the result < 0.01 and multiply the
 * surviving amplitude/alpha/brightness by it (no hard pop).
 */
export const GLSL_NEAR_FADE_FUNCTIONS = `
float perspectiveNearFade(int isOrtho, float viewZ, float nearCull) {
  if (isOrtho == 1) return 1.0;
  if (viewZ >= 0.0) return 0.0;
  return smoothstep(nearCull, nearCull * 2.0, -viewZ);
}
`;
