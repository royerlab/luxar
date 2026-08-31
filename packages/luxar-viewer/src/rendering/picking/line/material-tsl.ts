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
import { uniform, texture } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { linePickWebGPUFactory, type LinePickTSLConfig } from './pick.tsl';
import { capsuleLinePickWebGPUFactory } from './pick-capsule.tsl';
import type { LineJoinStyle } from '../../../types/line-join';
import { resolveLinePrimitive, type LinePrimitive } from '../../../types/line-primitive';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import { proxyIUniform, type TSLNode } from '../../materials/_shared/tsl-helpers';
import { getPlaceholderElementTexture } from '../../element-texture-layout';
import type { LinePickingMaterialConfig } from './material';

export class LinePickingTSLMaterial extends NodeMaterial implements CameraAwareMaterial {
  uniforms: Record<string, THREE.IUniform>;

  private tslNodes: {
    uLineTex: TSLNode;
    uResolution: TSLNode;
    uPixelRatio: TSLNode;
    uIsOrtho: TSLNode;
    uNearCull: TSLNode;
    uMaxLinePixelWidth: TSLNode;
    uNodeId: TSLNode;
    uPerspectiveLineScale: TSLNode;
    uOrthoLineScale: TSLNode;
    uSortedIndexSlot: TSLNode;
  };

  constructor(config: LinePickingMaterialConfig) {
    super();

    this.tslNodes = {
      // Line data texture node. Starts on the shared placeholder; the
      // commit's material sync rebinds the acquired pool entry's
      // texture via `updateLineTexture` (fresh node + graph rebuild —
      // TSL texture() captures the Texture at build time).
      uLineTex: texture(getPlaceholderElementTexture()),
      uResolution: uniform(new THREE.Vector2(1, 1)),
      uPixelRatio: uniform(1),
      uIsOrtho: uniform(0),
      // 0.1 matches the visual line material ctor default (pre-first-broadcast only).
      uNearCull: uniform(0.1),
      uMaxLinePixelWidth: uniform(540),
      uNodeId: uniform(config.nodeId),
      uPerspectiveLineScale: uniform(1.0),
      uOrthoLineScale: uniform(1.0),
      // Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB.
      // The pick pass MUST track the visual one — it emits `vElementId`
      // from the same index, so reading the other buffer resolves hovers
      // against a stale permutation. Pushed by the depth-sort
      // coordinator's `syncSortedIndexSlot`, which finds it through
      // `uniforms` below.
      uSortedIndexSlot: uniform(0),
    };

    // Join style — a BUILD-time graph variant (see pick.tsl.ts), so it is
    // stashed on userData before the first factory call and re-read by every
    // rebuild. Stored unresolved so the ?lineJoin= session override still wins
    // at build time. MUST match the visual material's value: the two build the
    // same screen-space quad.
    this.userData.lineJoin = config.join;
    // Line primitive (#1352) — likewise a build-time variant, dispatching
    // between the pick factories on every rebuild. UNLIKE lineJoin above
    // this is stamped RESOLVED (exactly like the visual TSL material):
    // rebuilds and clone() must never re-run a per-node policy decision
    // made at first build.
    this.userData.linePrimitive = resolveLinePrimitive(config.primitive);

    this.uniforms = {
      // WARNING: a direct `uniforms.uLineTex.value = tex` write does NOT
      // rebind — `updateLineTexture()` is the only rebind chokepoint.
      uLineTex: proxyIUniform(this.tslNodes.uLineTex),
      uResolution: proxyIUniform(this.tslNodes.uResolution),
      uPixelRatio: proxyIUniform(this.tslNodes.uPixelRatio),
      uIsOrtho: proxyIUniform(this.tslNodes.uIsOrtho),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      uMaxLinePixelWidth: proxyIUniform(this.tslNodes.uMaxLinePixelWidth),
      uNodeId: proxyIUniform(this.tslNodes.uNodeId),
      uPerspectiveLineScale: proxyIUniform(this.tslNodes.uPerspectiveLineScale),
      uOrthoLineScale: proxyIUniform(this.tslNodes.uOrthoLineScale),
      uSortedIndexSlot: proxyIUniform(this.tslNodes.uSortedIndexSlot),
    };

    this.toneMapped = false;
    this.side = THREE.DoubleSide;
    // Picking is opaque so the transparent-and-DoubleSide two-pass
    // guard never trips, but setting `forceSinglePass` explicitly
    // matches the visual material and documents intent.
    this.forceSinglePass = true;

    this._rebuild();
  }

  /**
   * Build the per-rebuild factory config from current uniforms.
   */
  private _currentConfig(): LinePickTSLConfig {
    return {
      isOrtho: (this.tslNodes.uIsOrtho.value as number) === 1,
      join: this.userData.lineJoin as LineJoinStyle | undefined,
    };
  }

  /**
   * (Re)build the pick graph, dispatching on the line primitive (#1352).
   * Every build site — constructor, clone, projection-mode flip, texture
   * rebind — funnels through here so the two factories can never drift.
   */
  private _rebuild(): void {
    const primitive = resolveLinePrimitive(
      this.userData.linePrimitive as LinePrimitive | undefined
    );
    if (primitive === 'capsule') {
      capsuleLinePickWebGPUFactory(this.tslNodes, this._currentConfig(), this);
    } else {
      linePickWebGPUFactory(this.tslNodes, this._currentConfig(), this);
    }
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
      // Graph variants — must ride the CONSTRUCTOR, not a post-hoc copy.
      join: this.userData.lineJoin as LineJoinStyle | undefined,
      primitive: this.userData.linePrimitive as LinePrimitive | undefined,
    });
    // Rebind the line data texture (no-op when still on the placeholder).
    const lineTex = this.uniforms.uLineTex?.value as THREE.DataTexture | null | undefined;
    if (lineTex) cloned.updateLineTexture(lineTex);
    (cloned.uniforms.uResolution.value as THREE.Vector2).copy(
      this.uniforms.uResolution.value as THREE.Vector2
    );
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uPixelRatio.value = this.uniforms.uPixelRatio.value;
    cloned.uniforms.uMaxLinePixelWidth.value = this.uniforms.uMaxLinePixelWidth.value;
    cloned.uniforms.uPerspectiveLineScale.value = this.uniforms.uPerspectiveLineScale.value;
    cloned.uniforms.uOrthoLineScale.value = this.uniforms.uOrthoLineScale.value;
    // The active ordering slot must ride along: a clone taken while the
    // geometry draws from slot 1 would otherwise read the stale buffer
    // until the coordinator's next per-frame re-assert.
    cloned.uniforms.uSortedIndexSlot.value = this.uniforms.uSortedIndexSlot.value;
    if (cloned._currentConfig().isOrtho) {
      cloned._rebuild();
      cloned.needsUpdate = true;
    }
    return cloned as this;
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number,
    pixelRatio: number = 1
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
    this.uniforms.uPixelRatio.value = pixelRatio;
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
      this._rebuild();
      this.needsUpdate = true;
    }
  }

  /**
   * Rebind the line data texture. TSL `texture()` captures the
   * THREE.Texture at factory time, so an identity change needs a fresh
   * node + graph rebuild. Mirrors `PointPickingTSLMaterial.
   * updatePointTexture`. No-op when unchanged (the common per-commit
   * case).
   */
  updateLineTexture(tex: THREE.DataTexture | null): void {
    const current = (this.uniforms.uLineTex?.value as THREE.Texture | null | undefined) ?? null;
    const next = tex ?? getPlaceholderElementTexture();
    if (current === next) return;
    this.tslNodes.uLineTex = texture(next);
    this.uniforms.uLineTex = proxyIUniform(this.tslNodes.uLineTex);
    this._rebuild();
    this.needsUpdate = true;
  }
}
