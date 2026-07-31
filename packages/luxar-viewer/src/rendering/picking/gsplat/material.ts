/**
 * GSplat Picking Material for GPU object picking.
 *
 * Renders Gaussian splats to an RGBA32F pick buffer encoding:
 *   R = nodeId, G = elementId (gl_InstanceID), B = brightness, A = 1.0
 *
 * Shader source-of-truth lives in `./shaders.ts`.
 */

import * as THREE from 'three';
import type { CameraAwareMaterial } from '../../materials/_shared/camera-aware-material';
import { computeFocalLength } from '../../materials/_shared/camera-uniforms';
import { GSPLAT_PICK_SOURCE } from './shaders';
import { requireWebGLSources } from '../../materials/_shared/shader-source';
import { GSPLAT_COV2D_DILATION_DEFAULT } from '../../materials/gsplat/math';

// Module-load assertion: the GLSL wrapper requires the GLSL source.
const GSPLAT_PICK_GLSL = requireWebGLSources(GSPLAT_PICK_SOURCE);

/**
 * Pick materials with a switchable depth convention (both gsplat pick
 * wrappers implement it; points/lines don't — their brightness-as-depth
 * is unconditional). The picking system's per-render mode sync detects
 * the capability via {@link isSurfacePickAwareMaterial}, mirroring the
 * `CameraAwareMaterial` guard idiom.
 */
export interface SurfacePickAwareMaterial {
  /**
   * Select the pick depth convention: `true` = real projected depth
   * (front-most wins; the depth-sorted `normal` surface mode), `false`
   * = brightness-as-depth (brightest wins; commutative modes).
   */
  setSurfacePickDepth(on: boolean): void;
}

/** Type guard for {@link SurfacePickAwareMaterial}. */
export function isSurfacePickAwareMaterial(
  material: unknown
): material is SurfacePickAwareMaterial {
  return (
    typeof material === 'object' &&
    material !== null &&
    'setSurfacePickDepth' in material &&
    typeof (material as Record<string, unknown>).setSurfacePickDepth === 'function'
  );
}

export interface GSplatPickingMaterialConfig {
  nodeId: number;
}

export class GSplatPickingMaterial
  extends THREE.ShaderMaterial
  implements CameraAwareMaterial, SurfacePickAwareMaterial
{
  constructor(config: GSplatPickingMaterialConfig) {
    // Tighter truncation: 1.5σ instead of 3.0σ
    const truncate = 1.5;
    const shiftC = Math.exp(-0.5 * truncate * truncate);
    const invOneMinusC = 1.0 / (1.0 - shiftC);

    // Picking always uses max-projection mode; the sum-projection
    // ray-integral factor and the uProjectionMode selector aren't
    // referenced in the picking shader body, so we don't bind them
    // here either (kept the GLSL and TSL paths symmetric — see the
    // matching omission in gsplat-picking-material-tsl.ts).

    super({
      uniforms: {
        // Splat data texture — rebound by the commit's material sync
        // (shared with the visual material's pool-owned storage).
        uSplatTex: { value: null },
        uResolution: { value: new THREE.Vector2(1, 1) },
        uFx: { value: 500 },
        uFy: { value: 500 },
        uTruncate: { value: truncate },
        uTruncateSq: { value: truncate * truncate },
        uShiftC: { value: shiftC },
        uInvOneMinusC: { value: invOneMinusC },
        uIsOrtho: { value: 0 },
        // Active ordering buffer: 0 = aSortedIndex, 1 = aSortedIndexB.
        // Flipped by the depth-sort coordinator once the inactive buffer
        // holds a whole permutation (runtime uniform: never a define — a
        // flip must not recompile the program).
        uSortedIndexSlot: { value: 0 },
        uNearCull: { value: 0.1 },
        uMaxExtentFactor: { value: 0.33 },
        uCov2DDilation: { value: GSPLAT_COV2D_DILATION_DEFAULT },
        // 0 = brightness-as-depth (brightest wins; commutative modes),
        // 1 = real projected depth (front-most wins; surface/'normal'
        // mode). Synced per pick render by PickingSystem from the main
        // material's blending mode via setSurfacePickDepth().
        uSurfaceDepth: { value: 0 },
        uNodeId: { value: config.nodeId },
      },
      vertexShader: GSPLAT_PICK_GLSL.vertex,
      fragmentShader: GSPLAT_PICK_GLSL.fragment,
      glslVersion: THREE.GLSL3,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      blending: THREE.NoBlending,
      toneMapped: false,
      side: THREE.DoubleSide,
    });
  }

  /** Rebind the splat data texture (plain uniform update). */
  updateSplatTexture(texture: THREE.DataTexture | null): void {
    this.uniforms.uSplatTex.value = texture;
  }

  /**
   * Select the pick depth convention. `true` = surface ('normal')
   * blending: write the real projected depth so the FRONT-MOST splat
   * wins — matching the depth-sorted occluding surface the user sees.
   * `false` (default) = brightness-as-depth so the BRIGHTEST splat wins
   * — correct for the commutative modes (additive/max/luminous).
   * Synced per pick render by `PickingSystem.renderPickBuffer()`.
   */
  setSurfacePickDepth(on: boolean): void {
    this.uniforms.uSurfaceDepth.value = on ? 1 : 0;
  }

  /**
   * Clone this picking material. The inherited `Material.clone()` calls
   * the constructor with no config (throws on `config.nodeId`), so —
   * mirroring the visual `GSplatMaterial.clone()` pattern — construct
   * with the same nodeId and copy the runtime-tuned uniform values
   * (camera params, dilation, surface-pick depth) across explicitly.
   */
  clone(): this {
    const cloned = new GSplatPickingMaterial({ nodeId: this.uniforms.uNodeId.value });
    cloned.uniforms.uSplatTex.value = this.uniforms.uSplatTex.value;
    cloned.uniforms.uResolution.value.copy(this.uniforms.uResolution.value);
    cloned.uniforms.uFx.value = this.uniforms.uFx.value;
    cloned.uniforms.uFy.value = this.uniforms.uFy.value;
    cloned.uniforms.uIsOrtho.value = this.uniforms.uIsOrtho.value;
    cloned.uniforms.uNearCull.value = this.uniforms.uNearCull.value;
    cloned.uniforms.uMaxExtentFactor.value = this.uniforms.uMaxExtentFactor.value;
    cloned.uniforms.uCov2DDilation.value = this.uniforms.uCov2DDilation.value;
    cloned.uniforms.uSurfaceDepth.value = this.uniforms.uSurfaceDepth.value;
    // The active ordering slot must ride along: a clone taken while the
    // geometry draws from slot 1 would otherwise read the stale buffer
    // until the coordinator's next per-frame re-assert.
    cloned.uniforms.uSortedIndexSlot.value = this.uniforms.uSortedIndexSlot.value;
    return cloned as this;
  }

  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho: boolean = false,
    nearCull?: number
  ): void {
    this.uniforms.uResolution.value.copy(resolution);
    this.uniforms.uIsOrtho.value = isOrtho ? 1 : 0;

    const fy = computeFocalLength(fov, resolution.y, isOrtho);
    this.uniforms.uFx.value = fy;
    this.uniforms.uFy.value = fy;

    if (nearCull !== undefined) {
      this.uniforms.uNearCull.value = nearCull;
    }
  }
}
