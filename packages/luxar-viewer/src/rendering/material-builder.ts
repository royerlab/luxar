/**
 * `buildMaterial` — central branching helper for the WebGPU migration.
 *
 * Each material wrapper (FxaaPass, BloomChain, PointMaterial, …)
 * delegates inner construction here. The helper branches on
 * `RendererCapabilities.api`:
 *
 * - WebGL2 path returns a configured `THREE.ShaderMaterial` from
 *   `source.webgl.{vertex,fragment}`.
 * - WebGPU path calls `source.webgpu(uniforms)` to get a
 *   `NodeMaterial`. If the source has no `webgpu` factory yet, the
 *   helper falls back to the WebGL path — this lets the migration
 *   land one shader at a time while keeping the rest of the pipeline
 *   green under both renderers.
 *
 * See `MATERIAL_WRAPPER_DESIGN.md` for the design rationale.
 *
 * @module rendering/material-builder
 */

import * as THREE from 'three';

import type { ShaderSource } from './shaders/shader-source';
import type { RendererCapabilities } from './renderer-capabilities';

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
 * Returns a `THREE.ShaderMaterial` under WebGL2. Under WebGPU, calls
 * `source.webgpu(...)` if present and returns the resulting
 * `NodeMaterial`; otherwise falls back to ShaderMaterial. Three.js
 * runtime accepts both — `WebGPURenderer({ forceWebGL: true })` can
 * still render `ShaderMaterial`s (the `ShaderMaterial`-not-supported
 * restriction only applies to native WebGPU dispatch).
 */
export function buildMaterial(
  source: ShaderSource,
  config: BuildMaterialConfig,
  caps: RendererCapabilities
): THREE.Material {
  if (caps.api === 'webgpu' && source.webgpu) {
    // TSL / NodeMaterial path. Cast through unknown — the factory's
    // return type is intentionally erased on `ShaderSource.webgpu`
    // (see shader-source.ts) so the type module doesn't depend on
    // three/webgpu's NodeMaterial type at compile time.
    return source.webgpu(config.uniforms ?? {}) as THREE.Material;
  }

  // WebGL2 (or WebGPU fallback when no TSL factory exists yet).
  if (!source.webgl) {
    // Both backends are optional in the type (see shader-source.ts);
    // the invariant is that at least one must be present for the
    // active code path. Triggering this means a shader shipped a
    // `webgpu` factory but no `webgl` reference, AND the renderer
    // is dispatching via the WebGL path. Surface it explicitly so
    // the gap is easy to diagnose instead of a confusing
    // `Cannot read properties of undefined (reading 'vertex')`.
    throw new Error(
      `buildMaterial: ShaderSource '${source.name}' has no WebGL fallback ` +
        'but the active renderer dispatches via the WebGL path ' +
        `(caps.api='${caps.api}'). Add a 'webgl' source or run with the ` +
        'default WebGPU renderer.'
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
