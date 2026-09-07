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

import { MeshPhysicalNodeMaterial, PhysicalLightingModel, type NodeBuilder } from 'three/webgpu';
import { diffuseColor, property } from 'three/tsl';
import type { BlendingMode } from '../../../types/blending';
import type { MeshShadingMode } from '../mesh/appearance';
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
 * Three's physical lighting model with the transmitted alpha pinned to 1 — the WebGPU
 * counterpart of the GLSL twin's `onBeforeCompile` patch (see `TRANSMISSION_ALPHA_MIX_LINE`
 * in `./config.ts` for why Luxar glass must not read the framebuffer alpha).
 *
 * Three's `start()` ends its transmission block with
 * `diffuseColor.a.mulAssign( mix( 1, backdrop.a, transmission ) )`, the only place the
 * sampled alpha reaches the fragment. It runs inside the `LightsNode` stack, so the three
 * statements here are emitted in order: save `diffuseColor.a` into a PROPERTY node,
 * let three run, put it back. A `toVar()` would not do — a var is declared where it is
 * first referenced, which would be AFTER the multiply.
 */
class LuxarPhysicalLightingModel extends PhysicalLightingModel {
  start(builder: NodeBuilder): void {
    if (this.transmission !== true) {
      super.start(builder);
      return;
    }
    const saved = property('float', 'LuxarTransmittedAlpha');
    saved.assign(diffuseColor.a);
    super.start(builder);
    diffuseColor.a.assign(saved);
  }
}

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

  /** The same six feature flags three passes, into the alpha-pinning subclass. */
  setupLightingModel(): PhysicalLightingModel {
    return new LuxarPhysicalLightingModel(
      this.useClearcoat,
      this.useSheen,
      this.useIridescence,
      this.useAnisotropy,
      this.useTransmission,
      this.useDispersion
    );
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

  /** Luxar `refract_data`, live: glass draws after (true) or before (false) the data. */
  updateRefractData(refractData: boolean): void {
    physicalUpdateRefractData(this, refractData);
  }

  /** Deliberate no-op — see the GLSL twin. */
  applyBlendingMode(_mode: BlendingMode): void {
    // Intentionally empty; see `PhysicalMeshMaterial.applyBlendingMode`.
  }
}
