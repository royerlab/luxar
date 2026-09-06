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
 * **The config's render state is authoritative on both backends.**
 * `blending` / `depthTest` / `depthWrite` / `transparent` /
 * `toneMapped` / `side` are resolved ONCE by `resolveRenderState` and
 * then applied to whichever material the branch produced, so a TSL
 * factory's internally-set values are overridden by whatever the host
 * passed — and by the same resolved defaults when the host passed
 * nothing. The WebGPU branch used to thread `config.uniforms` alone
 * and drop the rest, which silently disabled the bloom upsample pass's
 * `AdditiveBlending`: each upsample overwrote its destination mip
 * instead of accumulating into it, so WebGPU bloom rendered as a flat
 * dim wash (#2563). Resolving in one place is what makes the two
 * backends provably agree for the same config.
 *
 * `defines` is deliberately NOT threaded on the WebGPU branch:
 * `NodeMaterial` has no `defines` field at all — a TSL factory takes
 * its compile-time flags through its own arguments/config instead — so
 * the key is WebGL-only by construction, not by oversight.
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
 * The six render-state properties, with every default already
 * applied. Non-optional on purpose: the branches consume this object
 * rather than re-deriving `config.x ?? default` each, so a default
 * cannot drift between WebGL2 and WebGPU.
 */
interface ResolvedRenderState {
  blending: THREE.Blending;
  depthTest: boolean;
  depthWrite: boolean;
  transparent: boolean;
  toneMapped: boolean;
  side: THREE.Side;
}

/**
 * Resolve the config's render state against the builder's defaults.
 *
 * These are the WebGL branch's historical defaults verbatim, so
 * sharing them changes nothing there. The one worth flagging is
 * `toneMapped: false`, the OPPOSITE of Three's own material default
 * (`true`): a caller here is a pass whose pixels either have not been
 * tone-mapped yet (the bloom mips, which the mega-shader grades) or
 * already have been (FXAA's LDR input), so injecting the renderer's
 * tone-mapping would be wrong in both directions.
 */
function resolveRenderState(config: BuildMaterialConfig): ResolvedRenderState {
  return {
    blending: config.blending ?? THREE.NormalBlending,
    depthTest: config.depthTest ?? true,
    depthWrite: config.depthWrite ?? true,
    transparent: config.transparent ?? false,
    toneMapped: config.toneMapped ?? false,
    side: config.side ?? THREE.FrontSide,
  };
}

/**
 * Build a `THREE.Material` for the active backend.
 *
 * Returns a `THREE.ShaderMaterial` under WebGL2 and a
 * `NodeMaterial` under WebGPU. Throws if the active backend's
 * source is missing — does **not** silently fall through to the
 * other backend (under WebGPU, `ShaderMaterial` would not render at
 * all; under WebGL2, a TSL `NodeMaterial` wouldn't dispatch).
 *
 * The config's render state (`blending`, `depthTest`, `depthWrite`,
 * `transparent`, `toneMapped`, `side`) is applied on BOTH branches and
 * wins over anything a TSL factory set on itself; omitted fields take
 * the same resolved defaults either way. `defines` is WebGL-only —
 * `NodeMaterial` has no such field (see the module docblock).
 */
export function buildMaterial(
  source: ShaderSource,
  config: BuildMaterialConfig,
  caps: RendererCapabilities
): THREE.Material {
  const state = resolveRenderState(config);

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
          'default WebGLRenderer path (remove ?renderer=webgpu / ' +
          'VITE_LUXAR_USE_WEBGPU=1).'
      );
    }
    // TSL / NodeMaterial path. Cast through unknown — the factory's
    // return type is intentionally erased on `ShaderSource.webgpu`
    // (see shader-source.ts) so the type module doesn't depend on
    // three/webgpu's NodeMaterial type at compile time.
    const material = source.webgpu(config.uniforms ?? {}) as THREE.Material;
    // Apply the SAME resolved state the WebGL branch passes to its
    // ShaderMaterial constructor. Assigned field by field (rather than
    // `Object.assign`) so each write is type-checked against
    // `THREE.Material`, which declares all six.
    material.blending = state.blending;
    material.depthTest = state.depthTest;
    material.depthWrite = state.depthWrite;
    material.transparent = state.transparent;
    material.toneMapped = state.toneMapped;
    material.side = state.side;
    return material;
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
    ...state,
  });
}
