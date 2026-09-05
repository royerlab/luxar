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
 * @module rendering/picking/gsplat/material-tsl
 */

import * as THREE from 'three';
import { texture, uniform } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { gsplatPickWebGPUFactory } from './pick.tsl';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import { computeFocalLength } from '../../materials/_shared/camera-uniforms';
import { proxyIUniform, type TSLNode } from '../../materials/_shared/tsl-helpers';
import { getPlaceholderElementTexture } from '../../element-texture-layout';
import type { GSplatPickingMaterialConfig, SurfacePickAwareMaterial } from './material';
import { GSPLAT_COV2D_DILATION_DEFAULT } from '../../materials/gsplat/math';

export class GSplatPickingTSLMaterial
  extends NodeMaterial
  implements CameraAwareMaterial, SurfacePickAwareMaterial
{
  uniforms: Record<string, THREE.IUniform>;

  private tslNodes: {
    uSplatTex: TSLNode;
    uResolution: TSLNode;
    uPixelRatio: TSLNode;
    uFx: TSLNode;
    uFy: TSLNode;
    uTruncate: TSLNode;
    uTruncateSq: TSLNode;
    uShiftC: TSLNode;
    uInvOneMinusC: TSLNode;
    uIsOrtho: TSLNode;
    uNearCull: TSLNode;
    uMaxExtentFactor: TSLNode;
    uCov2DDilation: TSLNode;
    uSurfaceDepth: TSLNode;
    uNodeId: TSLNode;
    uSortedIndexSlot: TSLNode;
    uDensityDrop: TSLNode;
    uLabelFilterIndex: TSLNode;
  };

  constructor(config: GSplatPickingMaterialConfig) {
    super();

    // Tighter truncation: 1.5σ (matches GLSL picking).
    const truncate = 1.5;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    this.tslNodes = {
      // Splat data texture node (placeholder until the commit sync
      // rebinds the pool texture; identity change -> factory re-run).
      uSplatTex: texture(getPlaceholderElementTexture()),
      uResolution: uniform(new THREE.Vector2(1, 1)),
      uPixelRatio: uniform(1),
      uFx: uniform(500),
      uFy: uniform(500),
      uTruncate: uniform(truncate),
      uTruncateSq: uniform(truncate * truncate),
      uShiftC: uniform(shiftC),
      uInvOneMinusC: uniform(invOneMinusC),
      uIsOrtho: uniform(0),
      uNearCull: uniform(0.1),
      uMaxExtentFactor: uniform(0.33),
      uCov2DDilation: uniform(GSPLAT_COV2D_DILATION_DEFAULT),
      // 0 = brightness-as-depth (brightest wins; commutative modes),
      // 1 = real fragment depth (front-most wins; surface/'normal'
      // mode). Synced per pick render by PickingSystem via
      // setSurfacePickDepth() — mirrors the GLSL wrapper.
      uSurfaceDepth: uniform(0),
      uNodeId: uniform(config.nodeId),
      // Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB.
      // The pick pass MUST track the visual one — it emits `vElementId`
      // from the same index, so reading the other buffer resolves hovers
      // against a stale permutation. Pushed by the depth-sort
      // coordinator's `syncSortedIndexSlot`, which finds it through
      // `uniforms` below.
      uSortedIndexSlot: uniform(0),
      uDensityDrop: uniform(0),
      uLabelFilterIndex: uniform(0),
    };

    this.uniforms = {
      uSplatTex: proxyIUniform(this.tslNodes.uSplatTex),
      uResolution: proxyIUniform(this.tslNodes.uResolution),
      uPixelRatio: proxyIUniform(this.tslNodes.uPixelRatio),
      uFx: proxyIUniform(this.tslNodes.uFx),
      uFy: proxyIUniform(this.tslNodes.uFy),
      uTruncate: proxyIUniform(this.tslNodes.uTruncate),
      uTruncateSq: proxyIUniform(this.tslNodes.uTruncateSq),
      uShiftC: proxyIUniform(this.tslNodes.uShiftC),
      uInvOneMinusC: proxyIUniform(this.tslNodes.uInvOneMinusC),
      uIsOrtho: proxyIUniform(this.tslNodes.uIsOrtho),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      uMaxExtentFactor: proxyIUniform(this.tslNodes.uMaxExtentFactor),
      uCov2DDilation: proxyIUniform(this.tslNodes.uCov2DDilation),
      uSurfaceDepth: proxyIUniform(this.tslNodes.uSurfaceDepth),
      uNodeId: proxyIUniform(this.tslNodes.uNodeId),
      uSortedIndexSlot: proxyIUniform(this.tslNodes.uSortedIndexSlot),
      uDensityDrop: proxyIUniform(this.tslNodes.uDensityDrop),
      uLabelFilterIndex: proxyIUniform(this.tslNodes.uLabelFilterIndex),
    };

    this.toneMapped = false;
    this.side = THREE.DoubleSide;

    gsplatPickWebGPUFactory(this.tslNodes, this);
  }

  /**
   * Rebind the splat data texture. Same node-identity lifecycle as
   * the visual TSL wrapper: fresh texture node + factory re-run on an
   * identity change, no-op otherwise.
   */
  updateSplatTexture(tex: THREE.DataTexture | null): void {
    const current = (this.uniforms.uSplatTex?.value as THREE.Texture | null | undefined) ?? null;
    const next = tex ?? getPlaceholderElementTexture();
    if (current === next) return;
    this.tslNodes.uSplatTex = texture(next);
    this.uniforms.uSplatTex = proxyIUniform(this.tslNodes.uSplatTex);
    gsplatPickWebGPUFactory(this.tslNodes, this);
    this.needsUpdate = true;
  }

  /**
   * Select the pick depth convention. `true` = surface ('normal')
   * blending: write the real fragment depth so the FRONT-MOST splat
   * wins — matching the depth-sorted occluding surface the user sees.
   * `false` (default) = brightness-as-depth so the BRIGHTEST splat wins
   * — correct for the commutative modes (additive/max/luminous).
   * Mirrors the GLSL wrapper's method one-for-one; synced per pick
   * render by `PickingSystem.renderPickBuffer()`.
   */
  setSurfacePickDepth(on: boolean): void {
    this.uniforms.uSurfaceDepth.value = on ? 1 : 0;
  }

  updateLabelFilter(filterIndex: number): void {
    this.uniforms.uLabelFilterIndex.value = Math.max(0, Math.floor(filterIndex));
  }

  /**
   * Clone this picking material. Mirrors the GLSL wrapper's explicit
   * clone (the inherited `Material.clone()` calls the constructor with
   * no config and would throw; `NodeMaterial.copy` would alias the
   * source's node graph instead of this instance's own uniform nodes).
   */
  clone(): this {
    const cloned = new GSplatPickingTSLMaterial({
      nodeId: this.uniforms.uNodeId.value as number,
    });
    const splatTex = this.uniforms.uSplatTex?.value as THREE.DataTexture | null | undefined;
    if (splatTex) cloned.updateSplatTexture(splatTex);
    (cloned.uniforms.uResolution.value as THREE.Vector2).copy(
      this.uniforms.uResolution.value as THREE.Vector2
    );
    cloned.uniforms.uPixelRatio.value = this.uniforms.uPixelRatio.value;
    cloned.uniforms.uFx.value = this.uniforms.uFx.value;
    cloned.uniforms.uFy.value = this.uniforms.uFy.value;
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uMaxExtentFactor.value = this.uniforms.uMaxExtentFactor.value;
    cloned.uniforms.uCov2DDilation.value = this.uniforms.uCov2DDilation.value;
    cloned.uniforms.uSurfaceDepth.value = this.uniforms.uSurfaceDepth.value;
    cloned.uniforms.uLabelFilterIndex.value = this.uniforms.uLabelFilterIndex.value;
    // The active ordering slot must ride along: a clone taken while the
    // geometry draws from slot 1 would otherwise read the stale buffer
    // until the coordinator's next per-frame re-assert.
    cloned.uniforms.uSortedIndexSlot.value = this.uniforms.uSortedIndexSlot.value;
    cloned.uniforms.uDensityDrop.value = this.uniforms.uDensityDrop.value;
    return cloned as this;
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number,
    pixelRatio: number = 1
  ): void {
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
    this.uniforms.uPixelRatio.value = pixelRatio;
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;

    const fy = computeFocalLength(fov, resolution.y, isOrtho);
    this.uniforms.uFx.value = fy;
    this.uniforms.uFy.value = fy;

    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
  }
}
