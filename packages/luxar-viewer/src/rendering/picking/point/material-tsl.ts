/**
 * TSL / NodeMaterial counterpart to `PointPickingMaterial`.
 *
 * Mirrors the GLSL `PointPickingMaterial` one-for-one — same
 * constructor signature, same `updateCameraParams` /
 * `updateRadiusScale` / `dispose` surface,
 * same `CameraAwareMaterial` contract. Per-mesh lifetime (the
 * NodeFactory creates one per points node and registers it with
 * `materialManager` for camera updates).
 *
 * **Uniform plumbing.** This class owns one `UniformNode` per shader
 * input. The public `uniforms` record exposes each node as an
 * `IUniform`-shaped getter/setter proxy (see `proxyIUniform` in
 * `tsl-helpers.ts`). Mutations to `material.uniforms.uX.value`
 * therefore land directly on `node.value` — no per-render
 * `.onUpdate` callback bridge. Three-geometry symmetry with the
 * core PointTSLMaterial.
 *
 * @module rendering/picking/point/material-tsl
 */

import * as THREE from 'three';
import { uniform } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { pointPickWebGPUFactory, type PointPickTSLNodes } from './pick.tsl';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import {
  computePointSizeFactor,
  computeMaxPointSize,
} from '../../materials/_shared/camera-uniforms';
import { proxyIUniform, type TSLNode } from '../../materials/_shared/tsl-helpers';
import type { PointPickingMaterialConfig } from './material';

export class PointPickingTSLMaterial extends NodeMaterial implements CameraAwareMaterial {
  uniforms: Record<string, THREE.IUniform>;

  private tslNodes: {
    pointSizeFactor: TSLNode;
    maxPointSize: TSLNode;
    radiusScale: TSLNode;
    uIsOrtho: TSLNode;
    uNearCull: TSLNode;
    uNodeId: TSLNode;
    uResolution: TSLNode;
  };

  constructor(config: PointPickingMaterialConfig) {
    super();

    const defaultFov = (60 * Math.PI) / 180;
    const defaultResolutionY = 1080;
    const defaultTanHalfFov = Math.tan(defaultFov / 2);

    this.tslNodes = {
      pointSizeFactor: uniform((2.0 * defaultResolutionY) / defaultTanHalfFov),
      maxPointSize: uniform(defaultResolutionY * 0.5),
      radiusScale: uniform(config.radiusScale ?? 1.0),
      uIsOrtho: uniform(0),
      uNearCull: uniform(0.1),
      uNodeId: uniform(config.nodeId),
      uResolution: uniform(new THREE.Vector2(1920, defaultResolutionY)),
    };

    this.uniforms = {
      pointSizeFactor: proxyIUniform(this.tslNodes.pointSizeFactor),
      maxPointSize: proxyIUniform(this.tslNodes.maxPointSize),
      radiusScale: proxyIUniform(this.tslNodes.radiusScale),
      uIsOrtho: proxyIUniform(this.tslNodes.uIsOrtho),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      uNodeId: proxyIUniform(this.tslNodes.uNodeId),
      uResolution: proxyIUniform(this.tslNodes.uResolution),
    };

    this.toneMapped = false;

    pointPickWebGPUFactory(this.tslNodes as PointPickTSLNodes, this);
  }

  /**
   * Clone this picking material. Mirrors the GLSL wrapper's explicit
   * clone (the inherited `Material.clone()` calls the constructor with
   * no config and would throw; `NodeMaterial.copy` would alias the
   * source's node graph instead of this instance's own uniform nodes).
   */
  clone(): this {
    const cloned = new PointPickingTSLMaterial({
      nodeId: this.uniforms.uNodeId.value as number,
      radiusScale: this.uniforms.radiusScale.value as number,
    });
    cloned.uniforms.pointSizeFactor.value = this.uniforms.pointSizeFactor.value;
    cloned.uniforms.maxPointSize.value = this.uniforms.maxPointSize.value;
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    (cloned.uniforms.uResolution.value as THREE.Vector2).copy(
      this.uniforms.uResolution.value as THREE.Vector2
    );
    return cloned as this;
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;
    if (nearCull !== undefined) this.uniforms.uNearCull.value = nearCull;
    this.uniforms.pointSizeFactor.value = computePointSizeFactor(fov, resolution.y, isOrtho);
    this.uniforms.maxPointSize.value = computeMaxPointSize(resolution.y);
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
  }

  updateRadiusScale(scale: number): void {
    this.uniforms.radiusScale.value = scale;
  }
}
