/**
 * `PhysicalMeshMaterial` — the WebGL wrapper around three's `MeshPhysicalMaterial`.
 *
 * Deliberately THIN. The house mesh material is a `ShaderMaterial` Luxar writes and
 * pins with codegen snapshots; this one is three's, and three's to pin. What the
 * wrapper adds is exactly the surface the rest of the viewer already assumes of a
 * leaf material — `updateOpacity` / `updateIntensity` / `updateOffset` /
 * `updateGamma` (the `LuxarMaterial` contract the Layers panel, exposure and the
 * LOD cross-fade gate on) and an `applyBlendingMode` that is a documented no-op —
 * each implemented ONCE in `./config.ts` and shared with the WebGPU twin.
 *
 * What it deliberately lacks, and why (spec `MESH_PHYSICAL_MATERIALS_SPEC.md` §3.2):
 * - no `updateCameraParams`: the near fade is a house-shader feature, so this
 *   material is filed as static by `MaterialManager.register` and takes no camera
 *   broadcast;
 * - no `updateShading` / `updateBaseColorTexture` / `updateColormapTexture`: the
 *   per-epoch shading variant, textures and colormaps are house-shader features
 *   (the last two are refused at authoring for a physical mesh), and the commit
 *   path already no-ops when the methods are absent;
 * - no `userData.blendingMode` unless opaque — see `derivePhysicalCompositing`.
 *
 * @module rendering/materials/mesh-physical/material-glsl
 */

import * as THREE from 'three';
import type { BlendingMode } from '../../../types/blending';
import {
  applyPhysicalMeshConfig,
  physicalGetOpacity,
  physicalUpdateGamma,
  physicalUpdateIntensity,
  physicalUpdateOffset,
  physicalUpdateOpacity,
  physicalUpdateRefractData,
  setPhysicalKnob,
  type PhysicalMeshKnobKey,
  type PhysicalMeshMaterialConfig,
} from './config';

/**
 * Three's `MeshPhysicalMaterial` behind the Luxar leaf-material surface (WebGL).
 *
 * Constructed by `MaterialManager.getMeshPhysicalMaterial`; configured entirely by
 * `applyPhysicalMeshConfig`, which the WebGPU twin shares.
 */
export class PhysicalMeshMaterial extends THREE.MeshPhysicalMaterial {
  /** Build and configure from Luxar attrs (see `PhysicalMeshMaterialConfig`). */
  constructor(config: PhysicalMeshMaterialConfig = {}) {
    super();
    applyPhysicalMeshConfig(this, config);
  }

  /** Luxar `opacity`; re-derives translucency, depth write and the cutout. */
  updateOpacity(opacity: number): void {
    physicalUpdateOpacity(this, opacity);
  }

  /** Current node opacity (the LOD cross-fade snapshots this as its fade base). */
  getOpacity(): number {
    return physicalGetOpacity(this);
  }

  /** Luxar `intensity` — a gain on the vertex colour. */
  updateIntensity(intensity: number): void {
    physicalUpdateIntensity(this, intensity);
  }

  /** Luxar `offset` — emitted radiance; negative clamps to 0. */
  updateOffset(offset: number): void {
    physicalUpdateOffset(this, offset);
  }

  /** Luxar `gamma` — recorded only; a physical material has no gamma term. */
  updateGamma(gamma: number): void {
    physicalUpdateGamma(this, gamma);
  }

  /** One live physical knob from the Layers panel (clamped; rebuilds on a zero crossing). */
  updatePhysicalKnob(key: PhysicalMeshKnobKey, value: number): void {
    setPhysicalKnob(this, key, value);
  }

  /** Luxar `refract_data`, live: glass draws after (true) or before (false) the data. */
  updateRefractData(refractData: boolean): void {
    physicalUpdateRefractData(this, refractData);
  }

  /**
   * Deliberate no-op. A physical mesh has no Luxar blending mode (the attr is refused
   * at authoring, and an inherited one is ignored with a notice at node creation);
   * translucency is read off the data by `derivePhysicalCompositing`. The method
   * exists so the Layers panel's generic fallback — which would write the HOUSE
   * shader's blend state for the mesh default `opaque` over this material on a
   * reset — is never reached.
   */
  applyBlendingMode(_mode: BlendingMode): void {
    // Intentionally empty; see the doc comment.
  }
}
