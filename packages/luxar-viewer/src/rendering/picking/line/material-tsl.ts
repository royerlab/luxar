/**
 * TSL / NodeMaterial counterpart to `LinePickingMaterial`.
 *
 * Mirrors the GLSL `LinePickingMaterial` one-for-one — same
 * constructor signature, same `updateCameraParams` / `dispose`
 * surface, same `CameraAwareMaterial` contract.
 *
 * **Uniform plumbing.** This class owns one `UniformNode` per shader
 * input. The public `uniforms` record exposes each node as an
 * `IUniform`-shaped getter/setter proxy (see `proxyIUniform` in
 * `tsl-helpers.ts`). Mutations to `material.uniforms.uX.value`
 * therefore land directly on `node.value` — no per-render
 * `.onUpdate` callback bridge.
 *
 * @module rendering/picking/line/material-tsl
 */

import * as THREE from 'three';
import { uniform } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { linePickWebGPUFactory, type LinePickTSLNodes } from './pick.tsl';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import { proxyIUniform, type TSLNode } from '../../materials/_shared/tsl-helpers';
import type { LinePickingMaterialConfig } from './material';

export class LinePickingTSLMaterial extends NodeMaterial implements CameraAwareMaterial {
  uniforms: Record<string, THREE.IUniform>;

  private tslNodes: {
    uResolution: TSLNode;
    uIsOrtho: TSLNode;
    uNearCull: TSLNode;
    uMaxLinePixelWidth: TSLNode;
    uNodeId: TSLNode;
    uPerspectiveLineScale: TSLNode;
    uOrthoLineScale: TSLNode;
  };

  constructor(config: LinePickingMaterialConfig) {
    super();

    this.tslNodes = {
      uResolution: uniform(new THREE.Vector2(1, 1)),
      uIsOrtho: uniform(0),
      uNearCull: uniform(0.05),
      uMaxLinePixelWidth: uniform(540),
      uNodeId: uniform(config.nodeId),
      uPerspectiveLineScale: uniform(1.0),
      uOrthoLineScale: uniform(1.0),
    };

    this.uniforms = {
      uResolution: proxyIUniform(this.tslNodes.uResolution),
      uIsOrtho: proxyIUniform(this.tslNodes.uIsOrtho),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      uMaxLinePixelWidth: proxyIUniform(this.tslNodes.uMaxLinePixelWidth),
      uNodeId: proxyIUniform(this.tslNodes.uNodeId),
      uPerspectiveLineScale: proxyIUniform(this.tslNodes.uPerspectiveLineScale),
      uOrthoLineScale: proxyIUniform(this.tslNodes.uOrthoLineScale),
    };

    this.toneMapped = false;
    this.side = THREE.DoubleSide;
    // Picking is opaque so the transparent-and-DoubleSide two-pass
    // guard never trips, but setting `forceSinglePass` explicitly
    // matches the visual material and documents intent.
    this.forceSinglePass = true;

    linePickWebGPUFactory(this.tslNodes as LinePickTSLNodes, this._currentConfig(), this);
  }

  /**
   * Build the per-rebuild factory config from current uniforms.
   */
  private _currentConfig(): { isOrtho: boolean } {
    return {
      isOrtho: (this.tslNodes.uIsOrtho.value as number) === 1,
    };
  }

  /**
   * Clone this picking material. Mirrors the GLSL wrapper's explicit
   * clone (the inherited `Material.clone()` calls the constructor with
   * no config and would throw; `NodeMaterial.copy` would alias the
   * source's node graph instead of this instance's own uniform nodes).
   * The pick graph is JS-specialized on the projection mode (see
   * `updateCameraParams`), so after copying `uIsOrtho` the clone's
   * graph is rebuilt when the mode differs from the constructor
   * default (perspective) — same rebuild-on-flip rule as the source.
   */
  clone(): this {
    const cloned = new LinePickingTSLMaterial({
      nodeId: this.uniforms.uNodeId.value as number,
    });
    (cloned.uniforms.uResolution.value as THREE.Vector2).copy(
      this.uniforms.uResolution.value as THREE.Vector2
    );
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uMaxLinePixelWidth.value = this.uniforms.uMaxLinePixelWidth.value;
    cloned.uniforms.uPerspectiveLineScale.value = this.uniforms.uPerspectiveLineScale.value;
    cloned.uniforms.uOrthoLineScale.value = this.uniforms.uOrthoLineScale.value;
    if (cloned._currentConfig().isOrtho) {
      linePickWebGPUFactory(cloned.tslNodes as LinePickTSLNodes, cloned._currentConfig(), cloned);
      cloned.needsUpdate = true;
    }
    return cloned as this;
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    const prevIsOrtho = (this.tslNodes.uIsOrtho.value as number) === 1;
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    // Accept ANY defined value, including 0 — matching the point/gsplat
    // wrappers (the shader floors at 1e-20). The old `> 0` gate silently
    // KEPT a stale value on zero-diagonal scenes (or, with LRU-cached
    // materials, the previous dataset's nearCull), re-creating the
    // cross-geometry near-fade divergence B9c fixed.
    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
    this.uniforms.uMaxLinePixelWidth.value = Math.max(2, resolution.y * 0.5);
    // Precomputed pixel-width scales — see LineMaterial.updateCameraParams.
    const safeFov = Math.max(fov, 1e-4);
    if (isOrtho) {
      this.uniforms.uOrthoLineScale.value = (2.0 * resolution.y) / safeFov;
    } else {
      this.uniforms.uPerspectiveLineScale.value =
        resolution.y / Math.max(Math.tan(safeFov * 0.5), 1e-4);
    }
    // Rebuild on projection-mode flip so the unused branch drops.
    if (isOrtho !== prevIsOrtho) {
      linePickWebGPUFactory(this.tslNodes as LinePickTSLNodes, this._currentConfig(), this);
      this.needsUpdate = true;
    }
  }
}
