/**
 * Shared registry-entry contract for the TSL ↔ GLSL parity harness.
 *
 * Each shader-family module (`post-processing.ts`, `points.ts`,
 * `lines.ts`, `gsplats.ts`) exports a partial
 * `Record<string, RegistryEntry>`; `index.ts` merges them into the
 * single `SHADER_REGISTRY` the Playwright specs drive by name.
 *
 * @module tests/e2e/harnesses/tsl-harness/types
 */

import type * as THREE from 'three';
import type { ShaderSource } from '../../../../rendering/materials/_shared/shader-source';

/**
 * Shape of an entry in the shader registry exposed to Playwright.
 * Adding a new shader to {@link SHADER_REGISTRY} is sufficient to make
 * it usable from the spec.
 */
export interface RegistryEntry {
  readonly source: ShaderSource;
  /** Default uniforms for this shader's parity test. */
  readonly buildUniforms: () => Record<string, THREE.IUniform>;
  /**
   * GLSL3 `defines` to set on the `THREE.ShaderMaterial`. Needed for
   * shaders like `mega` that use `#define` gates for feature toggles
   * + an `LUXAR_TONE_MAPPING_MODE` numeric. Optional; defaults to
   * empty (no defines).
   */
  readonly buildDefines?: () => Record<string, string>;
  /**
   * Override for the TSL material constructor. When provided, the
   * harness calls this directly instead of `source.webgpu(uniforms)`.
   * Used to pass shader-specific factory configs (e.g.
   * `megaWebGPUFactory(uniforms, { toneMappingMode: 1 })`).
   */
  readonly buildTSLMaterial?: (uniforms: Record<string, THREE.IUniform>) => THREE.Material;
  /**
   * Override the mesh built around the material. Defaults to a
   * fullscreen `THREE.Mesh(PlaneGeometry(2, 2), material)` rendered
   * with an OrthographicCamera. Override for point-sprite tests
   * that need `THREE.Points(...)`.
   */
  readonly buildMesh?: (material: THREE.Material) => THREE.Object3D;
  /**
   * Override the camera. Defaults to an `OrthographicCamera` at (0,0,1)
   * looking at the origin (see `buildDefaultCamera`). Override for cases
   * that need a `PerspectiveCamera` — e.g. the behind-camera guard, which
   * is perspective-only (`uIsOrtho == 0`) and a no-op under the default
   * ortho camera.
   */
  readonly buildCamera?: () => THREE.Camera;
  /**
   * Set on the GLSL3 `ShaderMaterial`. Needed by shaders like
   * `point` that read the auto-injected `in vec3 color` attribute —
   * Three only emits the attribute declaration when this is true.
   * TSL reads the same attribute via `attribute<'vec3'>('color',
   * 'vec3')` and doesn't need a parallel flag.
   */
  readonly vertexColors?: boolean;
  /**
   * Enable depthTest/depthWrite on the GLSL3 `ShaderMaterial` (the
   * harness default is depth-off, fine for the single-primitive parity
   * cases). Required by scenarios where two fragments COMPETE through
   * the depth test — e.g. the surface-pick depth variants, whose whole
   * point is which of two overlapping splats wins. The TSL side needs
   * no flag: the pick factories already set depthTest/depthWrite true
   * on their NodeMaterial.
   */
  readonly depthCompete?: boolean;
}
