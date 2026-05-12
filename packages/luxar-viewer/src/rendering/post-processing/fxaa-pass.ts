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

/**
 * Runs FXAA on an LDR input texture, writing to the renderer's
 * current target. Owns one ShaderMaterial + one fullscreen triangle.
 */
export class FxaaPass {
  private material: THREE.ShaderMaterial;
  private mesh: THREE.Mesh;
  private scene: THREE.Scene;
  private camera: THREE.OrthographicCamera;

  constructor(width: number, height: number) {
    this.material = new THREE.ShaderMaterial({
      vertexShader: FXAA_SOURCE.webgl.vertex,
      fragmentShader: FXAA_SOURCE.webgl.fragment,
      glslVersion: THREE.GLSL3,
      depthTest: false,
      depthWrite: false,
      // Bypass renderer-level tone mapping injection — FXAA reads
      // already-tone-mapped LDR values and writes them through.
      toneMapped: false,
      uniforms: {
        uInput: { value: null as THREE.Texture | null },
        uResolution: { value: new THREE.Vector2(width, height) },
      },
    });

    // Fullscreen triangle (NDC positions {-1,-1}, {3,-1}, {-1,3}).
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)
    );
    this.mesh = new THREE.Mesh(geo, this.material);
    this.mesh.frustumCulled = false;
    this.scene = new THREE.Scene();
    this.scene.add(this.mesh);
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  setSize(width: number, height: number): void {
    this.material.uniforms.uResolution.value.set(width, height);
  }

  /**
   * Render FXAA from `inputTexture` to the renderer's current target
   * (use `renderer.setRenderTarget(null)` for the backbuffer).
   */
  render(renderer: THREE.WebGLRenderer, inputTexture: THREE.Texture): void {
    this.material.uniforms.uInput.value = inputTexture;
    renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.material.dispose();
    this.mesh.geometry.dispose();
  }
}
