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
import { FXAA_SOURCE } from './shaders';
import { buildMaterial } from '../../materials/_shared/material-builder';
import { FullscreenPass } from '../fullscreen/pass';
import type { Renderer, RendererCapabilities } from '../../renderer-capabilities';

/**
 * Runs FXAA on an LDR input texture, writing to the renderer's
 * current target. Owns one `THREE.Material` (built via the
 * backend-aware `buildMaterial` helper) and one {@link FullscreenPass}.
 */
export class FxaaPass {
  private material: THREE.Material & { uniforms: Record<string, THREE.IUniform> };
  private uniforms: { uInput: THREE.IUniform; uResolution: THREE.IUniform<THREE.Vector2> };
  private pass: FullscreenPass;

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

    this.pass = new FullscreenPass(this.material, caps);
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
    this.pass.render(renderer);
  }

  dispose(): void {
    this.material.dispose();
    this.pass.dispose();
  }
}
