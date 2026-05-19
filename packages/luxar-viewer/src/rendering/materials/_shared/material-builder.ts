/**
 * `buildMaterial` — central branching helper for the dual-stack
 * renderer (`THREE.WebGLRenderer` + `WebGPURenderer`).
 *
 * Each material wrapper (FxaaPass, BloomChain, PointMaterial, …)
 * delegates inner construction here. The helper branches on
 * `RendererCapabilities.apiSurface`:
 *
 * - WebGL2 path returns a configured `THREE.ShaderMaterial` from
 *   `source.webgl.{vertex,fragment}`.
 * - WebGPU path calls `source.webgpu(uniforms)` to get a
 *   `NodeMaterial`.
 *
 * Both `webgl` and `webgpu` fields on `ShaderSource` are optional in
 * the type so future single-backend shaders can opt out cleanly, but
 * the helper throws when the active backend's source is absent.
 * Under WebGPU specifically we do NOT silently fall back to
 * ShaderMaterial — `WebGPURenderer` cannot dispatch `ShaderMaterial`
 * even when running on its internal WebGL2 backend (see
 * `BROWSER_SUPPORT_POLICY.md`), so the silent fallback would render
 * blank quads. Throwing surfaces the gap at construction time
 * instead.
 *
 * See `MATERIAL_WRAPPER_DESIGN.md` for the design rationale.
 *
 * @module rendering/materials/_shared/material-builder
 */

import * as THREE from 'three';

import type { ShaderSource } from './shader-source';
import type { RendererCapabilities } from '../../renderer-capabilities';

/**
 * Subset of `THREE.ShaderMaterialParameters` the buildMaterial
 * helper threads through. Other properties (uniforms, vertexShader,
 * fragmentShader) are taken from the `ShaderSource`.
 */
export interface BuildMaterialConfig {
  /** Uniform table (passed verbatim to ShaderMaterial / TSL factory). */
  readonly uniforms?: Record<string, THREE.IUniform>;
  /** GLSL preprocessor defines for the WebGL path. */
  readonly defines?: Record<string, string | number | boolean>;
  readonly blending?: THREE.Blending;
  readonly depthTest?: boolean;
  readonly depthWrite?: boolean;
  readonly transparent?: boolean;
  readonly toneMapped?: boolean;
  readonly side?: THREE.Side;
}

/**
 * Build a `THREE.Material` for the active backend.
 *
 * Returns a `THREE.ShaderMaterial` under WebGL2 and a
 * `NodeMaterial` under WebGPU. Throws if the active backend's
 * source is missing — does **not** silently fall through to the
 * other backend (under WebGPU, `ShaderMaterial` would not render at
 * all; under WebGL2, a TSL `NodeMaterial` wouldn't dispatch).
 */
export function buildMaterial(
  source: ShaderSource,
  config: BuildMaterialConfig,
  caps: RendererCapabilities
): THREE.Material {
  if (caps.apiSurface === 'webgpu') {
    if (!source.webgpu) {
      // The active renderer is WebGPURenderer (or WebGPURenderer
      // running on its internal WebGL2 backend, which still dispatches
      // NodeMaterial — see BROWSER_SUPPORT_POLICY.md). A
      // ShaderMaterial fallback would render blank quads, so refuse
      // explicitly with a fix-it-here error.
      throw new Error(
        `buildMaterial: ShaderSource '${source.name}' has no 'webgpu' TSL ` +
          'factory but the active renderer dispatches via the WebGPU ' +
          "path (caps.apiSurface='webgpu'). WebGPURenderer cannot " +
          'dispatch ShaderMaterial even in its WebGL2 fallback mode; ' +
          "add a 'webgpu' factory to the ShaderSource or switch to the " +
          'legacy WebGLRenderer path (default; remove ?renderer=webgpu / ' +
          'VITE_LUXAR_USE_WEBGPU=1).'
      );
    }
    // TSL / NodeMaterial path. Cast through unknown — the factory's
    // return type is intentionally erased on `ShaderSource.webgpu`
    // (see shader-source.ts) so the type module doesn't depend on
    // three/webgpu's NodeMaterial type at compile time.
    return source.webgpu(config.uniforms ?? {}) as THREE.Material;
  }

  // WebGL2 path.
  if (!source.webgl) {
    // Symmetric guard to the WebGPU branch above: triggering this
    // means the shader shipped a `webgpu` factory but no `webgl`
    // reference, AND the active renderer is WebGLRenderer. Surface
    // the gap explicitly instead of `Cannot read properties of
    // undefined (reading 'vertex')`.
    throw new Error(
      `buildMaterial: ShaderSource '${source.name}' has no 'webgl' ` +
        'reference but the active renderer dispatches via the WebGL ' +
        `path (caps.apiSurface='${caps.apiSurface}'). Add a 'webgl' source ` +
        'or opt into the WebGPU renderer (?renderer=webgpu or ' +
        'VITE_LUXAR_USE_WEBGPU=1).'
    );
  }
  return new THREE.ShaderMaterial({
    vertexShader: source.webgl.vertex,
    fragmentShader: source.webgl.fragment,
    glslVersion: THREE.GLSL3,
    uniforms: config.uniforms,
    // Avoid passing `defines: undefined` through to Three's material
    // parameter validation while preserving numeric/boolean define values.
    defines: (config.defines ?? {}) as Record<string, unknown>,
    blending: config.blending ?? THREE.NormalBlending,
    depthTest: config.depthTest ?? true,
    depthWrite: config.depthWrite ?? true,
    transparent: config.transparent ?? false,
    toneMapped: config.toneMapped ?? false,
    side: config.side ?? THREE.FrontSide,
  });
}
