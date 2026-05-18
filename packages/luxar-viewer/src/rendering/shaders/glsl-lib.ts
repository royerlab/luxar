/**
 * Shared GLSL helper snippets injected into the Points / Lines / GSplats
 * shaders.
 *
 * E.2 of viewer-code-review-rerun action plan: previously each shader
 * inlined its own `isnan(v) || isinf(v) || v < 0.0 ? fallback : v`
 * pattern. Keeping the helpers in one place lets us:
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
 *   uniform float uFOV;
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
