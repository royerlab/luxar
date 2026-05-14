/**
 * Shader-source registry type.
 *
 * Every shader in Luxar is exported as a `ShaderSource` value rather
 * than as a pair of inline strings on its material. The registry
 * carries both backends side-by-side:
 *
 * - `webgpu` (TSL / NodeMaterial factory) — primary backend, used
 *   under the default `WebGPURenderer`.
 * - `webgl` (GLSL3 strings) — kept as a runnable reference: drives
 *   the legacy `THREE.WebGLRenderer` path behind
 *   `VITE_LUXAR_USE_LEGACY_WEBGL=1` and the TSL↔GLSL parity harness.
 *
 * Both fields are optional in the type so future shaders can ship
 * WebGPU-only if no reference is needed, but the invariant is that
 * at least one MUST be present. `buildMaterial` throws if a consumer
 * asks for a backend the source doesn't supply.
 */

/**
 * GLSL3 vertex + fragment pair for the WebGL2 path.
 */
export interface WebGLShaderSources {
  readonly vertex: string;
  readonly fragment: string;
}

/**
 * A single shader for one material / pass.
 *
 * Carries the TSL `webgpu` factory (primary) and the GLSL3 `webgl`
 * strings (reference / fallback). Both are optional but at least one
 * must be present — enforced by `buildMaterial` at construction
 * time, not by the type system, so shader authors can author either
 * backend first.
 *
 * The `webgpu` field deliberately uses `unknown` rather than
 * importing `NodeMaterial` types so this module stays
 * three-version-agnostic.
 */
export interface ShaderSource {
  /** Stable identifier for diagnostics (`point`, `bloom-threshold`, …). */
  readonly name: string;
  /**
   * GLSL3 strings for `THREE.ShaderMaterial`. Optional — present
   * today on every Luxar shader as the GLSL reference, but a future
   * shader could ship without it if no reference is needed.
   */
  readonly webgl?: WebGLShaderSources;
  /**
   * TSL / NodeMaterial factory for the WebGPU backend. Returns the
   * renderer-specific material; the shape is intentionally
   * unspecified so this module stays three-version-agnostic.
   */
  readonly webgpu?: (uniforms: Record<string, unknown>) => unknown;
}

/**
 * Narrow `source.webgl` from `WebGLShaderSources | undefined` to
 * `WebGLShaderSources`, throwing with a diagnostic message if the
 * source has no GLSL strings.
 *
 * Use this at construction sites that intrinsically require the
 * GLSL backend (the GLSL `ShaderMaterial` wrappers in
 * `picking-material.ts`, the TSL↔GLSL parity harness). Production
 * shaders all ship `webgl` today, so the throw never fires in
 * practice — it's the runtime invariant that the optional type
 * doesn't express on its own.
 */
export function requireWebGLSources(source: ShaderSource): WebGLShaderSources {
  if (!source.webgl) {
    throw new Error(
      `ShaderSource '${source.name}' has no GLSL fallback strings, but ` +
        "this code path requires them. Either provide a 'webgl' field on " +
        'the shader source or dispatch this material through the WebGPU path.'
    );
  }
  return source.webgl;
}
