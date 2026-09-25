/**
 * TSL / NodeMaterial counterpart to `MeshPickingMaterial`.
 *
 * Mirrors the GLSL wrapper one-for-one — same constructor signature, same
 * `setPickMode` / `setPickSide` / `updateOpacityUniform` / `updateAlphaCutoff`
 * surface, same `MeshPickAwareMaterial` contract, and the same half-consumed
 * `CameraAwareMaterial` one (resolution/isOrtho ignored, the near-fade start
 * taken).
 *
 * **Uniform plumbing.** This class owns one `UniformNode` per shader input. The
 * public `uniforms` record exposes each node as an `IUniform`-shaped
 * getter/setter proxy (`proxyIUniform` in `tsl-helpers.ts`), so mutations to
 * `material.uniforms.uX.value` land directly on `node.value` — no per-render
 * `.onUpdate` callback bridge.
 *
 * **No rebuild path.** Every mode-dependent behaviour here is a runtime uniform
 * (§6.5), so unlike the line/point pick wrappers there is no config-flip that
 * rebuilds the graph: the factory runs exactly once, in the constructor. That is
 * the whole reason the cutout and the depth convention are uniforms rather than
 * build flags — a layers-panel mode switch is a uniform write, not a recompile.
 *
 * @module rendering/picking/mesh/material-tsl
 */

import * as THREE from 'three';
import { texture, uniform } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { meshPickWebGPUFactory } from './pick.tsl';
import { proxyIUniform, type TSLNode } from '../../materials/_shared/tsl-helpers';
import { MESH_DEFAULTS, clampAppearanceFraction } from '../../materials/mesh/appearance';
import { resolveMeshPickModeState, type MeshPickAwareMaterial } from './pick-mode';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import type { BlendingMode } from '../../../types/blending';
import type { MeshPickingMaterialConfig } from './material';

export class MeshPickingTSLMaterial
  extends NodeMaterial
  implements CameraAwareMaterial, MeshPickAwareMaterial
{
  uniforms: Record<string, THREE.IUniform>;

  private tslNodes: {
    uNodeId: TSLNode;
    uOpacity: TSLNode;
    uAlphaCutoff: TSLNode;
    uAlphaCutout: TSLNode;
    uSurfaceDepth: TSLNode;
    uNearCull: TSLNode;
    uBaseColorTex?: TSLNode;
  };

  constructor(config: MeshPickingMaterialConfig) {
    super();

    this.tslNodes = {
      uNodeId: uniform(config.nodeId),
      uOpacity: uniform(clampAppearanceFraction(config.opacity, 1.0)),
      uAlphaCutoff: uniform(clampAppearanceFraction(config.alphaCutoff, MESH_DEFAULTS.alphaCutoff)),
      // Both start ON because the mesh default blending mode is `opaque` (§6.3),
      // which is both a cutout mode and a depth-ordered surface mode. The picking
      // system re-derives them from the visual material every pick render, so this
      // governs only the window before the first one.
      uAlphaCutout: uniform(1),
      uSurfaceDepth: uniform(1),
      // 0.1 matches the GLSL twin's default and is overridden per scene by
      // `updateCameraParams`.
      uNearCull: uniform(0.1),
      // Bound only when the node has a texture, matching the GLSL twin's define:
      // `texture()` captures its Texture, so the real image is installed by
      // `updateBaseColorTexture` below rather than written through a proxy.
      ...(config.baseColorTexture ? { uBaseColorTex: texture(config.baseColorTexture) } : {}),
    };

    this.uniforms = {
      uNodeId: proxyIUniform(this.tslNodes.uNodeId),
      uOpacity: proxyIUniform(this.tslNodes.uOpacity),
      uAlphaCutoff: proxyIUniform(this.tslNodes.uAlphaCutoff),
      uAlphaCutout: proxyIUniform(this.tslNodes.uAlphaCutout),
      uSurfaceDepth: proxyIUniform(this.tslNodes.uSurfaceDepth),
      uNearCull: proxyIUniform(this.tslNodes.uNearCull),
      // A plain value holder, NOT a proxy: the graph reads the captured `texture()`
      // node, so writing this would change nothing. `updateBaseColorTexture` rebuilds
      // the node instead, and this exists so callers can READ the bound texture
      // (`clone()` does, and so does `applyMeshTexture`'s idempotence check).
      ...(config.baseColorTexture ? { uBaseColorTex: { value: config.baseColorTexture } } : {}),
    };

    this.toneMapped = false;
    // Overwritten per epoch by setPickSide() from the visual material; FrontSide is
    // the safe start (never rasterizes a face the visual culls). This is the one
    // pick wrapper that does NOT pin DoubleSide — see `./material.ts`.
    this.side = THREE.FrontSide;
    this.forceSinglePass = true;

    meshPickWebGPUFactory(this.tslNodes, this);
  }

  /**
   * Install the real base-colour texture once the data has arrived.
   *
   * Rebuilds the graph rather than writing a uniform, and that is not optional
   * here: `texture()` captures its `THREE.Texture` at construction, so the value
   * held in `uniforms.uBaseColorTex` is only a record of what is bound — the graph
   * reads the captured node. Writing the uniform alone would leave the pick pass
   * sampling the blank placeholder, which has alpha 0 everywhere and would make the
   * whole mesh unpickable.
   *
   * A no-op when the node has no texture (the sampler is not in this variant's
   * graph) and when the texture is unchanged, so the steady state after the first
   * commit costs nothing.
   */
  updateBaseColorTexture(tex: THREE.Texture | null): void {
    if (!this.tslNodes.uBaseColorTex || !tex) return;
    if ((this.uniforms.uBaseColorTex?.value as THREE.Texture | null) === tex) return;
    this.tslNodes.uBaseColorTex = texture(tex);
    this.uniforms.uBaseColorTex = { value: tex };
    meshPickWebGPUFactory(this.tslNodes, this);
    this.needsUpdate = true;
  }

  /**
   * @see MeshPickingMaterial.updateCameraParams — `_resolution` / `_isOrtho`
   * ignored, `nearCull` consumed. It is a runtime uniform, so there is nothing
   * to rebuild (this wrapper has no rebuild path at all).
   */
  updateCameraParams(
    _resolution: THREE.Vector2,
    _isOrtho: boolean = false,
    nearCull?: number,
    _pixelRatio?: number
  ): void {
    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
  }

  /** @see MeshPickingMaterial.setPickMode */
  setPickMode(mode: BlendingMode): void {
    const { cutout, surfaceDepth } = resolveMeshPickModeState(mode);
    this.uniforms.uAlphaCutout.value = cutout ? 1 : 0;
    this.uniforms.uSurfaceDepth.value = surfaceDepth ? 1 : 0;
  }

  /** @see MeshPickingMaterial.setPickSide */
  setPickSide(side: THREE.Side): void {
    if (this.side !== side) {
      this.side = side;
      this.needsUpdate = true;
    }
  }

  /** @see MeshPickingMaterial.updateOpacityUniform */
  updateOpacityUniform(opacity: number): void {
    this.uniforms.uOpacity.value = clampAppearanceFraction(opacity, 1.0);
  }

  /** @see MeshPickingMaterial.updateAlphaCutoff */
  updateAlphaCutoff(cutoff: number): void {
    this.uniforms.uAlphaCutoff.value = clampAppearanceFraction(cutoff, MESH_DEFAULTS.alphaCutoff);
  }

  /**
   * Clone this picking material. Mirrors the GLSL wrapper's explicit clone: the
   * inherited `Material.clone()` calls the constructor with no config, and
   * `NodeMaterial.copy` would alias the source's node graph instead of giving the
   * clone its own uniform nodes.
   *
   * No graph rebuild is needed after copying the values — every one of them is a
   * runtime uniform.
   */
  clone(): this {
    const cloned = new MeshPickingTSLMaterial({
      nodeId: this.uniforms.uNodeId.value as number,
      opacity: this.uniforms.uOpacity.value as number,
      alphaCutoff: this.uniforms.uAlphaCutoff.value as number,
    });
    cloned.uniforms.uAlphaCutout.value = this.uniforms.uAlphaCutout.value;
    cloned.uniforms.uSurfaceDepth.value = this.uniforms.uSurfaceDepth.value;
    // Camera state too — see the GLSL twin: the constructor defaults would fade
    // against the wrong near plane, and would fade at all under ortho.
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    // The epoch's culling must ride along: a clone taken on an undecidable frame
    // would otherwise revert to FrontSide and drop half the pickable surface until
    // the next commit re-applied it.
    cloned.side = this.side;
    return cloned as this;
  }
}
