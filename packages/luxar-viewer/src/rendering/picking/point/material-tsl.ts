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
import { texture, uniform } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { pointPickWebGPUFactory } from './pick.tsl';
import {
  getPlaceholderElementTexture,
  POINT_TEXTURE_LAYOUT,
  resolveElementTextureWidth,
} from '../../element-texture-layout';
import { applySharedPickGraph } from '../_shared/shared-pick-graph-tsl';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import { computeMaxPointSize } from '../../materials/_shared/camera-uniforms';
import { proxyIUniform, type TSLNode } from '../../materials/_shared/tsl-helpers';
import {
  createPickVisibilityTSLNodes,
  proxyPickVisibilityUniforms,
} from '../_shared/visibility-tsl';
import { copyPickVisibilityUniforms } from '../_shared/visibility-uniforms';
import type { SurfacePickAwareMaterial } from '../_shared/surface-pick';
import type { PointPickingMaterialConfig } from './material';

export class PointPickingTSLMaterial
  extends NodeMaterial
  implements CameraAwareMaterial, SurfacePickAwareMaterial
{
  uniforms: Record<string, THREE.IUniform>;

  private tslNodes: {
    uPointTex: TSLNode;
    maxPointSize: TSLNode;
    radiusScale: TSLNode;
    uNearCull: TSLNode;
    uPixelRatio: TSLNode;
    uNodeId: TSLNode;
    uResolution: TSLNode;
    uSortedIndexSlot: TSLNode;
    uDensityDrop: TSLNode;
    uIntensity: TSLNode;
    uOpacity: TSLNode;
    uHasElementAlpha: TSLNode;
    uVolumetric: TSLNode;
    uSurfaceDepth: TSLNode;
  };

  constructor(config: PointPickingMaterialConfig) {
    super();

    const defaultResolutionY = 1080;

    this.tslNodes = {
      // Point data texture node (placeholder until the commit sync
      // rebinds the pool texture; identity change -> factory re-run).
      uPointTex: texture(getPlaceholderElementTexture()),
      maxPointSize: uniform(defaultResolutionY * 0.5),
      radiusScale: uniform(config.radiusScale ?? 1.0),
      uNearCull: uniform(0.1),
      uPixelRatio: uniform(1),
      uNodeId: uniform(config.nodeId),
      uResolution: uniform(new THREE.Vector2(1920, defaultResolutionY)),
      // Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB.
      // The pick pass MUST track the visual one — it emits `vElementId`
      // from the same index, so reading the other buffer resolves hovers
      // against a stale permutation. Pushed by the depth-sort
      // coordinator's `syncSortedIndexSlot`, which finds it through
      // `uniforms` below.
      uSortedIndexSlot: uniform(0),
      uDensityDrop: uniform(0),
      // Visual-pass weight inputs, neutral until the first pick render
      // syncs the node's own (picking-system/visibility-sync.ts).
      ...createPickVisibilityTSLNodes(),
      // 0 = brightness-as-depth, 1 = real fragment depth (front-most wins;
      // opaque/normal) — mirrors the GLSL wrapper.
      uSurfaceDepth: uniform(0),
    };

    this.uniforms = {
      uPointTex: proxyIUniform(this.tslNodes.uPointTex),
      maxPointSize: proxyIUniform(this.tslNodes.maxPointSize),
      radiusScale: proxyIUniform(this.tslNodes.radiusScale),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      uPixelRatio: proxyIUniform(this.tslNodes.uPixelRatio),
      uNodeId: proxyIUniform(this.tslNodes.uNodeId),
      uResolution: proxyIUniform(this.tslNodes.uResolution),
      uSortedIndexSlot: proxyIUniform(this.tslNodes.uSortedIndexSlot),
      uDensityDrop: proxyIUniform(this.tslNodes.uDensityDrop),
      ...proxyPickVisibilityUniforms(this.tslNodes),
      uSurfaceDepth: proxyIUniform(this.tslNodes.uSurfaceDepth),
    };

    this.toneMapped = false;

    this.rebuildGraph();
  }

  /**
   * Point this material at the SHARED graph of its configuration
   * (`../_shared/shared-pick-graph-tsl.ts`): one node build per configuration,
   * keyed on the baked element-texture width, instead of one per material.
   */
  private rebuildGraph(): void {
    const key = {
      elementTextureWidth: resolveElementTextureWidth(
        POINT_TEXTURE_LAYOUT,
        this.tslNodes.uPointTex.value as { image?: { width?: number } } | null
      ),
    };
    applySharedPickGraph(this, 'point-pick', key, this.tslNodes, (inputs, scratch) => {
      pointPickWebGPUFactory(inputs, scratch);
    });
  }

  /**
   * Rebind the point data texture. Same node-identity lifecycle as
   * the visual TSL wrapper: fresh texture node + factory re-run on an
   * identity change, no-op otherwise. Mirrors
   * `GSplatPickingTSLMaterial.updateSplatTexture`.
   */
  updatePointTexture(tex: THREE.DataTexture | null): void {
    const current = (this.uniforms.uPointTex?.value as THREE.Texture | null | undefined) ?? null;
    const next = tex ?? getPlaceholderElementTexture();
    if (current === next) return;
    this.tslNodes.uPointTex = texture(next);
    this.uniforms.uPointTex = proxyIUniform(this.tslNodes.uPointTex);
    this.rebuildGraph();
    this.needsUpdate = true;
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
    const pointTex = this.uniforms.uPointTex?.value as THREE.DataTexture | null | undefined;
    if (pointTex) cloned.updatePointTexture(pointTex);
    cloned.uniforms.maxPointSize.value = this.uniforms.maxPointSize.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uPixelRatio.value = this.uniforms.uPixelRatio.value;
    (cloned.uniforms.uResolution.value as THREE.Vector2).copy(
      this.uniforms.uResolution.value as THREE.Vector2
    );
    // The active ordering slot must ride along: a clone taken while the
    // geometry draws from slot 1 would otherwise read the stale buffer
    // until the coordinator's next per-frame re-assert.
    cloned.uniforms.uSortedIndexSlot.value = this.uniforms.uSortedIndexSlot.value;
    cloned.uniforms.uDensityDrop.value = this.uniforms.uDensityDrop.value;
    copyPickVisibilityUniforms(this.uniforms, cloned.uniforms);
    cloned.uniforms.uSurfaceDepth.value = this.uniforms.uSurfaceDepth.value;
    return cloned as this;
  }

  /**
   * Select the pick depth convention (`SurfacePickAwareMaterial`): `true`
   * under the depth-ordered surface modes (`opaque` / `normal`) writes the
   * real projected depth so the FRONT-MOST element wins, as the user sees
   * it; `false` (default) keeps brightness-as-depth so the BRIGHTEST wins,
   * right for the commutative modes. Synced per pick render by
   * `PickingSystem.renderPickBuffer()`.
   */
  setSurfacePickDepth(on: boolean): void {
    this.uniforms.uSurfaceDepth.value = on ? 1 : 0;
  }

  updateCameraParams(resolution: THREE.Vector2, nearCull?: number, pixelRatio: number = 1): void {
    if (nearCull !== undefined) this.uniforms.uNearCull.value = nearCull;
    this.uniforms.maxPointSize.value = computeMaxPointSize(resolution.y);
    this.uniforms.uPixelRatio.value = pixelRatio;
    (this.uniforms.uResolution.value as THREE.Vector2).copy(resolution);
  }

  updateRadiusScale(scale: number): void {
    this.uniforms.radiusScale.value = scale;
  }
}
