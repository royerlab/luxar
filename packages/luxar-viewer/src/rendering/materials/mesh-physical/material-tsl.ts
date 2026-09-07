/**
 * `PhysicalMeshTSLMaterial` — the WebGPU twin of `./material-glsl.ts`, around three's
 * `MeshPhysicalNodeMaterial`.
 *
 * Same surface, same helpers, same reasoning — see the GLSL twin's module doc. The
 * only difference is the base class, and that both `MeshPhysicalMaterial` and
 * `MeshPhysicalNodeMaterial` expose identical property names is what lets
 * `./config.ts` be the single implementation. This file lives in the lazy
 * `three/webgpu` cone (the `-tsl` suffix is what the ESLint boundary keys on) and is
 * reached only through `rendering/tsl/registry.ts`.
 *
 * @module rendering/materials/mesh-physical/material-tsl
 */

import { MeshPhysicalNodeMaterial } from 'three/webgpu';
import type { BlendingMode } from '../../../types/blending';
import type { MeshShadingMode } from '../mesh/appearance';
import {
  applyPhysicalMeshConfig,
  physicalGetOpacity,
  physicalUpdateGamma,
  physicalUpdateIntensity,
  physicalUpdateOffset,
  physicalUpdateOpacity,
  setPhysicalKnob,
  type PhysicalMeshKnobKey,
  type PhysicalMeshMaterialConfig,
} from './config';

/**
 * Three's `MeshPhysicalNodeMaterial` behind the Luxar leaf-material surface (WebGPU).
 *
 * Constructed by `MaterialManager.getMeshPhysicalMaterial` through the TSL registry;
 * configured entirely by `applyPhysicalMeshConfig`, which the WebGL twin shares.
 */
export class PhysicalMeshTSLMaterial extends MeshPhysicalNodeMaterial {
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

  /** Switch between stored normals and derivative normals for the active display frame. */
  updateShading(mode: MeshShadingMode): void {
    const flatShading = mode !== 'smooth';
    if (this.flatShading === flatShading) return;
    this.flatShading = flatShading;
    this.needsUpdate = true;
  }

  /** One live physical knob from the Layers panel (clamped; rebuilds on a zero crossing). */
  updatePhysicalKnob(key: PhysicalMeshKnobKey, value: number): void {
    setPhysicalKnob(this, key, value);
  }

  /** Deliberate no-op — see the GLSL twin. */
  applyBlendingMode(_mode: BlendingMode): void {
    // Intentionally empty; see `PhysicalMeshMaterial.applyBlendingMode`.
  }
}
