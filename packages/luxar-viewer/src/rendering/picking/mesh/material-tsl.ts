/**
 * TSL / NodeMaterial counterpart to `MeshPickingMaterial`.
 *
 * Mirrors the GLSL wrapper one-for-one — same constructor signature, same
 * `setPickMode` / `setPickSide` / `updateOpacityUniform` / `updateAlphaCutoff`
 * surface, same `MeshPickAwareMaterial` contract, and deliberately NOT a
 * `CameraAwareMaterial` (a mesh has no screen-space footprint to size).
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
import { uniform } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { meshPickWebGPUFactory } from './pick.tsl';
import { proxyIUniform, type TSLNode } from '../../materials/_shared/tsl-helpers';
import { MESH_DEFAULTS, clampAppearanceFraction } from '../../materials/mesh/appearance';
import { resolveMeshPickModeState, type MeshPickAwareMaterial } from './pick-mode';
import type { BlendingMode } from '../../../types/blending';
import type { MeshPickingMaterialConfig } from './material';

export class MeshPickingTSLMaterial extends NodeMaterial implements MeshPickAwareMaterial {
  uniforms: Record<string, THREE.IUniform>;

  private tslNodes: {
    uNodeId: TSLNode;
    uOpacity: TSLNode;
    uAlphaCutoff: TSLNode;
    uAlphaCutout: TSLNode;
    uSurfaceDepth: TSLNode;
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
    };

    this.uniforms = {
      uNodeId: proxyIUniform(this.tslNodes.uNodeId),
      uOpacity: proxyIUniform(this.tslNodes.uOpacity),
      uAlphaCutoff: proxyIUniform(this.tslNodes.uAlphaCutoff),
      uAlphaCutout: proxyIUniform(this.tslNodes.uAlphaCutout),
      uSurfaceDepth: proxyIUniform(this.tslNodes.uSurfaceDepth),
    };

    this.toneMapped = false;
    // Overwritten per epoch by setPickSide() from the visual material; FrontSide is
    // the safe start (never rasterizes a face the visual culls). This is the one
    // pick wrapper that does NOT pin DoubleSide — see `./material.ts`.
    this.side = THREE.FrontSide;
    this.forceSinglePass = true;

    meshPickWebGPUFactory(this.tslNodes, this);
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
    // The epoch's culling must ride along: a clone taken on an undecidable frame
    // would otherwise revert to FrontSide and drop half the pickable surface until
    // the next commit re-applied it.
    cloned.side = this.side;
    return cloned as this;
  }
}
