/**
 * TSL / NodeMaterial counterpart to `GSplatPickingMaterial`.
 *
 * Mirrors the GLSL `GSplatPickingMaterial` one-for-one — same
 * constructor signature, same `updateCameraParams` / `dispose`
 * surface, same `CameraAwareMaterial` contract. Always uses the
 * 1.5σ truncation + max-projection mode the GLSL path defaults to.
 *
 * **Uniform plumbing.** This class owns one `UniformNode` per shader
 * input. The public `uniforms` record exposes each node as an
 * `IUniform`-shaped getter/setter proxy (see `proxyIUniform` in
 * `tsl-helpers.ts`). Mutations to `material.uniforms.uX.value`
 * therefore land directly on `node.value` — no per-render
 * `.onUpdate` callback bridge.
 *
 * @module rendering/picking/gsplat-picking-material-tsl
 */

import * as THREE from 'three';
import { uniform } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { gsplatPickWebGPUFactory, type GSplatPickTSLNodes } from './gsplat-pick.tsl';
import type { CameraAwareMaterial } from '../materials/_shared/camera-aware-material';
import { computeFocalLength } from '../materials/_shared/camera-uniforms';
import { proxyIUniform, type TSLNode } from '../materials/_shared/tsl-helpers';
import type { GSplatPickingMaterialConfig } from './gsplat-picking-material';

export class GSplatPickingTSLMaterial extends NodeMaterial implements CameraAwareMaterial {
  uniforms: Record<string, THREE.IUniform>;

  private tslNodes: {
    uResolution: TSLNode;
    uFx: TSLNode;
    uFy: TSLNode;
    uTruncate: TSLNode;
    uTruncateSq: TSLNode;
    uShiftC: TSLNode;
    uInvOneMinusC: TSLNode;
    uIsOrtho: TSLNode;
    uNearCull: TSLNode;
    uMaxExtentFactor: TSLNode;
    uNodeId: TSLNode;
  };

  constructor(config: GSplatPickingMaterialConfig) {
    super();

    // Tighter truncation: 1.5σ (matches GLSL picking).
    const truncate = 1.5;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    this.tslNodes = {
      uResolution: uniform(new THREE.Vector2(1, 1)),
      uFx: uniform(500),
      uFy: uniform(500),
      uTruncate: uniform(truncate),
      uTruncateSq: uniform(truncate * truncate),
      uShiftC: uniform(shiftC),
      uInvOneMinusC: uniform(invOneMinusC),
      uIsOrtho: uniform(0),
      uNearCull: uniform(0.1),
      uMaxExtentFactor: uniform(0.33),
      uNodeId: uniform(config.nodeId),
    };

    this.uniforms = {
      uResolution: proxyIUniform(this.tslNodes.uResolution),
      uFx: proxyIUniform(this.tslNodes.uFx),
      uFy: proxyIUniform(this.tslNodes.uFy),
      uTruncate: proxyIUniform(this.tslNodes.uTruncate),
      uTruncateSq: proxyIUniform(this.tslNodes.uTruncateSq),
      uShiftC: proxyIUniform(this.tslNodes.uShiftC),
      uInvOneMinusC: proxyIUniform(this.tslNodes.uInvOneMinusC),
      uIsOrtho: proxyIUniform(this.tslNodes.uIsOrtho),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      uMaxExtentFactor: proxyIUniform(this.tslNodes.uMaxExtentFactor),
      uNodeId: proxyIUniform(this.tslNodes.uNodeId),
    };

    this.toneMapped = false;
    this.side = THREE.DoubleSide;

    gsplatPickWebGPUFactory(this.tslNodes as GSplatPickTSLNodes, this);
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;

    const fy = computeFocalLength(fov, resolution.y, isOrtho);
    this.uniforms.uFx.value = fy;
    this.uniforms.uFy.value = fy;

    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
  }
}
