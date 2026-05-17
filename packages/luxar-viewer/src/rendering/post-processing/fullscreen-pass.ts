/**
 * Reusable fullscreen-triangle render pass.
 *
 * Owns the small boilerplate that every post-processing stage repeats:
 * the shared caps-aware fullscreen-triangle geometry, the mesh that
 * wraps it, a single-mesh `Scene`, and the unit `OrthographicCamera`
 * that {@link THREE.WebGLRenderer.render} / {@link WebGPURenderer.render}
 * need as their second argument.
 *
 * Centralising this here means:
 *
 * 1. The caps-aware {@link createFullscreenTriangleGeometry} call lives
 *    in exactly one place, so future Y-orientation tweaks don't have
 *    to be applied per pass.
 * 2. The `frustumCulled = false` invariant (without it the renderer
 *    sometimes culls the triangle because its bounding sphere is small
 *    relative to NDC) is held in one constructor.
 * 3. The render scene/camera pair is shared, so consumers stop carrying
 *    `this.scene` / `this.camera` / `this.mesh` triples on their own.
 *
 * The bloom chain reuses one `FullscreenPass` across its three
 * materials (threshold, downsample, upsample) by calling
 * {@link FullscreenPass.setMaterial} between draws — mirroring what the
 * pre-extraction code already did against the bare `mesh.material`.
 *
 * @module rendering/post-processing/fullscreen-pass
 */

import * as THREE from 'three';

import { createFullscreenTriangleGeometry } from './fullscreen-geometry';
import type { Renderer, RendererCapabilities } from '../renderer-capabilities';

/**
 * A scene+camera+mesh triple wrapping a fullscreen-triangle render.
 * The geometry is built once per instance via the caps-aware factory;
 * the consumer supplies the material and may swap it via
 * {@link setMaterial} between renders.
 */
export class FullscreenPass {
  private readonly geometry: THREE.BufferGeometry;
  private readonly mesh: THREE.Mesh;
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.OrthographicCamera;

  /**
   * @param material - Initial material the fullscreen triangle is drawn
   *   with. Caller retains ownership; this class will not dispose it.
   * @param caps - RendererCapabilities. Used by the geometry factory to
   *   emit framebuffer-Y-correct UVs.
   */
  constructor(material: THREE.Material, caps: RendererCapabilities) {
    this.geometry = createFullscreenTriangleGeometry(caps);
    this.mesh = new THREE.Mesh(this.geometry, material);
    // Vertex (-1,3) lives outside the canonical [-1,1] cube, so Three's
    // automatic frustum check can incorrectly classify the mesh as out
    // of view depending on render-target size and DPR. Disable the
    // check — we're always drawing this.
    this.mesh.frustumCulled = false;
    this.scene = new THREE.Scene();
    this.scene.add(this.mesh);
    // Identity ortho camera: clip space is the same as NDC for the
    // fullscreen triangle, so the camera adds no transform beyond
    // satisfying `renderer.render(scene, camera)`'s signature.
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  /**
   * Swap the active material. Used by passes that draw the same
   * geometry with multiple shaders (e.g. {@link BloomChain}'s
   * threshold / downsample / upsample chain).
   */
  setMaterial(material: THREE.Material): void {
    this.mesh.material = material;
  }

  /**
   * Render the fullscreen triangle to the renderer's currently-bound
   * render target. Callers are responsible for `setRenderTarget` and
   * any clear/autoClear management around this call.
   */
  render(renderer: Renderer): void {
    renderer.render(this.scene, this.camera);
  }

  /**
   * Disposes the owned fullscreen-triangle geometry. The material is
   * owned by the caller and is NOT disposed here.
   */
  dispose(): void {
    this.geometry.dispose();
  }
}
