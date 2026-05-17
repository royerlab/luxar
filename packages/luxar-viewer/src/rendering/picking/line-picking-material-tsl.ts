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
 * @module rendering/picking/line-picking-material-tsl
 */

import * as THREE from 'three';
import { uniform } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { linePickWebGPUFactory, type LinePickTSLNodes } from './line-pick.tsl';
import type { CameraAwareMaterial } from '../camera-aware-material';
import { proxyIUniform, type TSLNode } from '../tsl-helpers';
import type { LinePickingMaterialConfig } from './line-picking-material';

export class LinePickingTSLMaterial extends NodeMaterial implements CameraAwareMaterial {
  uniforms: Record<string, THREE.IUniform>;

  private tslNodes: {
    uFOV: TSLNode;
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
      uFOV: uniform((60 * Math.PI) / 180),
      uResolution: uniform(new THREE.Vector2(1, 1)),
      uIsOrtho: uniform(0),
      uNearCull: uniform(0.05),
      uMaxLinePixelWidth: uniform(540),
      uNodeId: uniform(config.nodeId),
      uPerspectiveLineScale: uniform(1.0),
      uOrthoLineScale: uniform(1.0),
    };

    this.uniforms = {
      uFOV: proxyIUniform(this.tslNodes.uFOV),
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

    linePickWebGPUFactory(this.tslNodes as LinePickTSLNodes, this);
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.uniforms.uFOV.value = fov;
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    if (nearCull !== undefined && nearCull > 0) {
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
  }
}
