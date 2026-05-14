/**
 * FXAA post-pass.
 *
 * Standard "FXAA Quality" preset — single-pass edge anti-aliasing on
 * tone-mapped LDR pixels. Runs after the mega-shader and writes to
 * the renderer's current target (typically the canvas backbuffer).
 *
 * @module rendering/post-processing/fxaa-pass
 */

import * as THREE from 'three';
import { FXAA_SOURCE } from './fxaa-shaders';
import { buildMaterial } from '../material-builder';
import { createFullscreenTriangleGeometry } from './fullscreen-geometry';
import type { Renderer, RendererCapabilities } from '../renderer-capabilities';

/**
 * Runs FXAA on an LDR input texture, writing to the renderer's
 * current target. Owns one `THREE.Material` (built via the
 * backend-aware `buildMaterial` helper) and one fullscreen triangle.
 */
export class FxaaPass {
  private material: THREE.Material & { uniforms: Record<string, THREE.IUniform> };
  private uniforms: { uInput: THREE.IUniform; uResolution: THREE.IUniform<THREE.Vector2> };
  private mesh: THREE.Mesh;
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;

  constructor(width: number, height: number, caps: RendererCapabilities) {
    // Uniforms are held by reference so the FxaaPass's existing
    // `setSize` / `render` setters keep working under both backends:
    // mutating `.value` flows through to whichever material type
    // buildMaterial returned.
    this.uniforms = {
      uInput: { value: null as THREE.Texture | null },
      uResolution: { value: new THREE.Vector2(width, height) },
    };
    this.material = buildMaterial(
      FXAA_SOURCE,
      {
        uniforms: this.uniforms,
        depthTest: false,
        depthWrite: false,
        // Bypass renderer-level tone mapping injection — FXAA
        // reads already-tone-mapped LDR values and writes them
        // through.
        toneMapped: false,
      },
      caps
    ) as THREE.Material & { uniforms: Record<string, THREE.IUniform> };

    // Fullscreen triangle with matching uv attribute (see
    // `createFullscreenTriangleGeometry` for the uv contract).
    const geo = createFullscreenTriangleGeometry();
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
    this.scene = new THREE.Scene();
    this.scene.add(this.mesh);
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  setSize(width: number, height: number): void {
    this.uniforms.uResolution.value.set(width, height);
  }

  /**
   * Render FXAA from `inputTexture` to the renderer's current target
   * (use `renderer.setRenderTarget(null)` for the backbuffer).
   */
  render(renderer: Renderer, inputTexture: THREE.Texture): void {
    this.uniforms.uInput.value = inputTexture;
    renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.material.dispose();
    this.mesh.geometry.dispose();
  }
}
