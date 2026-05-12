/**
 * Shader-source registry type.
 *
 * Every shader in Luxar is exported as a `ShaderSource` value rather
 * than as a pair of inline strings on its material. This gives the
 * eventual WebGPU port a single seam: porting GLSL → TSL is one PR
 * per shader, each adding a `webgpu` factory alongside the existing
 * `webgl` strings.
 *
 * Today the `webgpu` field is unused; consumers branch on the
 * renderer's `api` (from `RendererCapabilities`) when both are
 * available.
 */

/**
 * GLSL3 vertex + fragment pair for the WebGL2 path.
 */
export interface WebGLShaderSources {
  readonly vertex: string;
  readonly fragment: string;
}

/**
 * A single shader for one material / pass. Holds the WebGL source
 * today and a slot for the WebGPU NodeMaterial factory tomorrow.
 *
 * The `webgpu` field deliberately uses `unknown` rather than
 * importing `NodeMaterial` types so this module stays
 * three-version-agnostic.
 */
export interface ShaderSource {
  /** Stable identifier for diagnostics (`point`, `bloom-threshold`, …). */
  readonly name: string;
  /** GLSL3 strings for `THREE.ShaderMaterial`. */
  readonly webgl: WebGLShaderSources;
  /**
   * Future TSL / NodeMaterial factory. Implementations land alongside
   * the WebGPU port. Returns the renderer-specific material; the
   * shape is intentionally unspecified for now.
   */
  readonly webgpu?: (uniforms: Record<string, unknown>) => unknown;
}
