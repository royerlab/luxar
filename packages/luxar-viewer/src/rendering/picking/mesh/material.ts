/**
 * Mesh Picking Material (GLSL backend) for GPU object picking.
 *
 * Renders the mesh to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (the VERTEX ordinal) low 16 bits,
 *   B = brightness (the coverage), A = the same ordinal's high 16 bits
 *
 * Shader source-of-truth lives in `./shaders.ts`.
 *
 * Two deliberate divergences from the sibling pick wrappers, both following the
 * visual mesh material rather than the pick convention:
 *
 * 1. **A `CameraAwareMaterial` for only half the usual reason.** A mesh has no
 *    screen-space footprint to size, so `resolution` and `isOrtho` are ignored (the
 *    near fade's ortho test reads three's `isOrthographic`); what it does consume
 *    is `uNearCull`, because the pick pass has to
 *    reproduce the visual near fade or a fading surface would stay fully pickable
 *    (#1431). Registering it therefore routes it into the camera broadcast, exactly
 *    as the visual mesh material.
 * 2. **`side` is synced from the visual material**, not pinned to `DoubleSide`.
 *    The other three pick materials can hardcode `DoubleSide` because their quads
 *    are view-facing; a mesh's back faces may be culled on screen, and a pick
 *    material that rasterizes them anyway makes an invisible interior face both
 *    pickable and depth-occluding. See {@link MeshPickingMaterial.setPickSide}.
 *
 * @module rendering/picking/mesh/material
 */

import * as THREE from 'three';
import { MESH_PICK_SOURCE } from './shaders';
import { requireWebGLSources } from '../../materials/_shared/shader-source';
import { MESH_DEFAULTS, clampAppearanceFraction } from '../../materials/mesh/appearance';
import { resolveMeshPickModeState, type MeshPickAwareMaterial } from './pick-mode';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import type { BlendingMode } from '../../../types/blending';

// Module-load assertion: the GLSL wrapper requires the GLSL source.
const MESH_PICK_GLSL = requireWebGLSources(MESH_PICK_SOURCE);

export interface MeshPickingMaterialConfig {
  nodeId: number;
  /** Node opacity — the second half of the coverage term. Defaults to 1. */
  opacity?: number;
  /** `opaque`-mode cutout threshold. Defaults to {@link MESH_DEFAULTS}. */
  alphaCutoff?: number;
  /**
   * The visual material's base-colour texture, when it has one.
   *
   * Supplied so the pick pass can multiply the texture's ALPHA into coverage —
   * the visual shader does, so an RGBA basemap's cutout holes exist on screen, and
   * a pick pass that ignored the texture would leave them pickable and
   * depth-occluding. The exact visual/pick divergence `syncMeshPickAppearance`
   * exists to prevent.
   *
   * Build-time, not a runtime setter, because it decides whether the program
   * declares a sampler and reads `uv` at all — and `has_texture` is a per-node
   * constant, unlike the blending mode.
   */
  baseColorTexture?: THREE.Texture | null;
}

export class MeshPickingMaterial
  extends THREE.ShaderMaterial
  implements CameraAwareMaterial, MeshPickAwareMaterial
{
  constructor(config: MeshPickingMaterialConfig) {
    super({
      uniforms: {
        uNodeId: { value: config.nodeId },
        uOpacity: { value: clampAppearanceFraction(config.opacity, 1.0) },
        uAlphaCutoff: {
          value: clampAppearanceFraction(config.alphaCutoff, MESH_DEFAULTS.alphaCutoff),
        },
        // Both default to the ON state because the mesh default blending mode is
        // `opaque` (§6.3), which is both a cutout mode and a depth-ordered surface
        // mode. `PickingSystem.renderPickBuffer` re-derives them from the visual
        // material's mode every pick render, so this only governs the window before
        // the first one.
        uAlphaCutout: { value: 1 },
        uSurfaceDepth: { value: 1 },
        uNearCull: { value: 0.1 }, // Default; overridden per-scene by updateCameraParams
        ...(config.baseColorTexture ? { uBaseColorTex: { value: config.baseColorTexture } } : {}),
      },
      vertexShader: MESH_PICK_GLSL.vertex,
      fragmentShader: MESH_PICK_GLSL.fragment,
      defines: config.baseColorTexture ? { LUXAR_MESH_PICK_BASE_COLOR_TEX: '' } : {},
      glslVersion: THREE.GLSL3,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      blending: THREE.NoBlending,
      toneMapped: false,
      // Overwritten per epoch by setPickSide() from the visual material; FrontSide
      // is the safe start (never rasterizes a face the visual culls).
      side: THREE.FrontSide,
      // Picking is opaque, so the transparent-and-DoubleSide two-pass guard never
      // trips — but say so explicitly, matching the sibling wrappers.
      forceSinglePass: true,
    });
  }

  /**
   * Update the camera-dependent uniforms.
   *
   * `_resolution` / `_isOrtho` are accepted and IGNORED — a mesh has no
   * screen-space footprint to size, and the near fade's ortho test reads three's
   * `isOrthographic`. Only `nearCull` is consumed, and it must be kept identical to the visual material's or pick coverage would stop matching
   * visible coverage near the camera. Mirrors `MeshMaterial.updateCameraParams`.
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

  /**
   * Install the real base-colour texture once the data has arrived.
   *
   * A plain uniform write here, unlike the TSL twin: the sampler is declared by the
   * constructor's define, so only the bound image changes and no recompile is
   * needed. Present under the same NAME as the TSL method so `applyMeshTexture` has
   * one call site rather than a backend branch.
   *
   * A no-op when the node has no texture — the uniform only exists in that variant.
   */
  updateBaseColorTexture(tex: THREE.Texture | null): void {
    if (!this.uniforms.uBaseColorTex || !tex) return;
    this.uniforms.uBaseColorTex.value = tex;
  }

  /**
   * Apply the blending mode's two pick-pass consequences (§6.5).
   *
   * Takes the whole mode rather than a pair of booleans on purpose — see
   * `./pick-mode.ts` for why the pair must not be settable independently.
   */
  setPickMode(mode: BlendingMode): void {
    const { cutout, surfaceDepth } = resolveMeshPickModeState(mode);
    this.uniforms.uAlphaCutout.value = cutout ? 1 : 0;
    this.uniforms.uSurfaceDepth.value = surfaceDepth ? 1 : 0;
  }

  /**
   * Match the visual material's face culling.
   *
   * Load-bearing rather than cosmetic: `side` is a property of the current
   * `displayDims` epoch (an undecidable frame forces `DoubleSide`, an odd-parity
   * reflection stays single-sided — see `applyMeshSide`), so it changes at runtime.
   * If the pick pass kept `DoubleSide` while the visual culls back faces, the
   * interior faces of a sliced closed isosurface would rasterize into the pick
   * buffer at true surface depth: pickable where nothing is drawn, AND occluding
   * picks of the nodes actually visible through the opening.
   *
   * Guarded on change — a `side` write is a program-invalidating event on the GLSL
   * backend and a pipeline rebuild on WebGPU, so it must not fire per frame.
   */
  setPickSide(side: THREE.Side): void {
    if (this.side !== side) {
      this.side = side;
      this.needsUpdate = true;
    }
  }

  /** Rebind the node opacity — the second half of the coverage term. */
  updateOpacityUniform(opacity: number): void {
    this.uniforms.uOpacity.value = clampAppearanceFraction(opacity, 1.0);
  }

  /** Rebind the cutout threshold. */
  updateAlphaCutoff(cutoff: number): void {
    this.uniforms.uAlphaCutoff.value = clampAppearanceFraction(cutoff, MESH_DEFAULTS.alphaCutoff);
  }

  /**
   * Clone this picking material. The inherited `Material.clone()` calls the
   * constructor with no config (so `config.nodeId` would be `undefined`), so —
   * mirroring the sibling pick wrappers — construct with the same nodeId and copy
   * the runtime-tuned values across explicitly.
   */
  clone(): this {
    const cloned = new MeshPickingMaterial({
      nodeId: this.uniforms.uNodeId.value as number,
      opacity: this.uniforms.uOpacity.value as number,
      alphaCutoff: this.uniforms.uAlphaCutoff.value as number,
      // Must go through the CONSTRUCTOR, not a post-hoc uniform write: the texture
      // decides whether the program declares a sampler, so a clone that copied only
      // the uniform would build the untextured variant and silently make every
      // cutout hole pickable again.
      baseColorTexture: (this.uniforms.uBaseColorTex?.value as THREE.Texture | null) ?? null,
    });
    cloned.uniforms.uAlphaCutout.value = this.uniforms.uAlphaCutout.value;
    cloned.uniforms.uSurfaceDepth.value = this.uniforms.uSurfaceDepth.value;
    // Camera state too: a clone left at the perspective/0.1 defaults would fade its
    // pick coverage against the wrong near plane — and under ortho, where the fade
    // is the identity, would fade at all.
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    // The epoch's culling must ride along: a clone taken on an undecidable frame
    // would otherwise revert to FrontSide and drop half the pickable surface until
    // the next commit re-applied it.
    cloned.side = this.side;
    return cloned as this;
  }
}
