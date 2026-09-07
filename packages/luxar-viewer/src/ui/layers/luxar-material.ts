/**
 * LuxarMaterial — the layer-control surface shared by all Luxar leaf materials.
 *
 * The layers panel drives materials exclusively through this interface: the
 * `update*` methods (intensity/offset/gamma/opacity + optional colormap
 * texture/scalar-range) plus the optional `applyBlendingMode`. The two routing
 * helpers ({@link isColormapActive}, {@link applyColorAdjustments}) encode how
 * gamma + display range are pushed differently in colormap (LUT) vs
 * direct-color mode. Extracted from `layers-panel.ts` (which re-exports the
 * public names for existing importers) so the material contract is importable
 * without pulling in the panel's DOM machinery.
 */

import * as THREE from 'three';
import type { BlendingMode } from '../../rendering';
import type { PhysicalMeshKnobKey } from '../../rendering/materials/mesh-physical/config';
import { computeDisplayRange } from './layer-state';

/**
 * Type guard target: does this material have our update* methods?
 *
 * Deliberately does NOT extend `CameraAwareMaterial`, and that is a correction
 * rather than a relaxation. Two independent reasons:
 *
 * 1. **The panel never calls `updateCameraParams`.** Camera uniforms are broadcast by
 *    `MaterialManager`, not from here, so requiring the method described a
 *    dependency this interface does not have.
 * 2. **`isLuxarMaterial` never checked for it.** The guard tests `updateIntensity` +
 *    `updateGamma` only, and `layer-apply.ts` casts to `LuxarMaterial` on the
 *    strength of that — so the type was already over-claiming relative to the check
 *    that produces it.
 *
 * Mesh is what surfaced this: it is a real leaf material with the full layer-control
 * surface, but it draws actual geometry and therefore has no screen-space extent to
 * recompute, so it has no `updateCameraParams` at all. The tempting fix was an empty
 * one on the material; that would be a lie, and would also cost a per-frame call per
 * node if it ever joined the broadcast registry. Materials that ARE camera-aware
 * still declare it via `CameraAwareMaterial` and are detected with
 * `isCameraAwareMaterial` where it matters (the picking system does exactly this).
 */
export interface LuxarMaterial extends THREE.Material {
  updateIntensity(v: number): void;
  updateOffset(v: number): void;
  updateGamma(v: number): void;
  updateOpacity(v: number): void;
  /**
   * Update the absorption coefficient κ (volumetric blending mode).
   * All three geometry-material families implement it (gsplats phase 1,
   * points phase 3, lines phase 4); optional only for exotic/legacy
   * materials (VOLUMETRIC_BLENDING_SPEC.md).
   */
  updateAbsorption?(v: number): void;
  /**
   * The §6.2 mesh appearance knobs: wrapped diffuse, specular, and the
   * `opaque`-mode cutout threshold.
   *
   * Optional, and — unlike `updateAbsorption` above — genuinely so rather than for
   * legacy reasons: **only the mesh materials implement them**, because mesh is the
   * only geometry type that shades. The other three are emissive per-element sprites
   * with no surface orientation, so there is no shade term for a floor to lift.
   *
   * That makes the optional-chained call site in `applyMeshAppearance` the type gate:
   * a points material simply has no `updateAmbient`.
   */
  updateAmbient?(v: number): void;
  updateShadeExponent?(v: number): void;
  updateSpecular?(v: number): void;
  updateShininess?(v: number): void;
  updateAlphaCutoff?(v: number): void;
  /**
   * One physically based knob (`material="physical"` meshes only — the two wrappers
   * in `rendering/materials/mesh-physical/`). Same optional-chained gate as the
   * house knobs above: a house or emissive material simply lacks it.
   */
  updatePhysicalKnob?(key: PhysicalMeshKnobKey, value: number): void;
  /**
   * Luxar `refract_data` (physical glass only, spec §3.4 Phase 3): draw the glass
   * after the emissive data so it refracts it. Same optional-chained gate.
   */
  updateRefractData?(refractData: boolean): void;
  updateColormapTexture?(texture: THREE.DataTexture | null): void;
  updateScalarRange?(min: number, max: number): void;
  updateLabelStyle?(colorByLabel: boolean, filterIndex: number): void;
  /**
   * Apply a blending mode to this material in-place.
   *
   * All four geometry materials implement it today (the optionality is
   * kept for exotic/legacy materials the generic fallback still covers).
   * Call this instead of writing `mat.blending`/`mat.blendEquation`
   * directly so type-specific factors, defines (LUXAR_VOLUMETRIC /
   * LUXAR_MAX_RGB_CONTRIBUTION), and uniforms stay in sync.
   */
  applyBlendingMode?(mode: BlendingMode): void;
}

/**
 * Whether a material is currently rendering in colormap (LUT) mode —
 * the `USE_COLORMAP` shader define is the source of truth (set/cleared
 * by `updateColormapTexture`). In this mode the display range drives the
 * LUT value window and gamma warps the value pre-lookup, so neither
 * should be applied to the output color (see the material shaders).
 *
 * Used by `layer-apply.ts` to pick the colormap-vs-direct route, and by the
 * unit tests covering it.
 */
export function isColormapActive(mat: LuxarMaterial): boolean {
  const defines = (mat as unknown as { defines?: Record<string, unknown> | null }).defines;
  return !!defines && 'USE_COLORMAP' in defines;
}

/**
 * Push gamma + the display-range adjustment to a leaf material, routed by
 * whether it renders through a colormap LUT:
 *
 * - **Colormap (LUT) mode**: the display range defines the value window
 *   mapped into the LUT (`uScalarMin`/`uScalarScale`) and gamma warps that
 *   value before the lookup — both operate on the scalar, not the color.
 *   The composed display window is recovered from the gain/offset pair and
 *   pushed via `updateScalarRange`. The shader does NOT bypass the color
 *   GOG (it always multiplies `vColor * uIntensity + uOffset` post-LUT), so
 *   we actively RESET it to identity (`updateIntensity(1)`/`updateOffset(0)`)
 *   here — otherwise the authored gain would double-apply, both shaping the
 *   LUT window and multiplying the mapped color (#936).
 * - **Direct-color mode**: GOG operates on the color (`intensity`/`offset`).
 *
 * Gamma is pushed in both modes (the shader applies it pre-LUT in colormap
 * mode, on the color otherwise). Opacity and blending are handled by the
 * caller. See the material shaders' `USE_COLORMAP` path.
 *
 * `scalarWindow`, when supplied, REPLACES the window this function would
 * otherwise recover from `intensity`/`offset` — and only on the colormap route,
 * which is the only one that has a scalar window at all. `layer-apply.ts` uses
 * it to hand each leaf the layer's window re-expressed in that leaf's OWN data
 * range (`remapWindowToLeafRange`), so a `kind=lod` / `kind=partition` layer
 * stops rendering every level on the finest level's window (#1753). The
 * gain/offset pair still drives the direct-color branch and the #936 identity
 * reset, so a caller that passes nothing gets exactly the previous behaviour.
 *
 * Applied by `layer-apply.ts` when committing a layer's effective appearance.
 */
export function applyColorAdjustments(
  mat: LuxarMaterial,
  gamma: number,
  intensity: number,
  offset: number,
  scalarWindow?: { min: number; max: number }
): void {
  mat.updateGamma(gamma);
  if (isColormapActive(mat) && mat.updateScalarRange) {
    const { min, max } = scalarWindow ?? computeDisplayRange(intensity, offset);
    mat.updateScalarRange(min, max);
    // The window now lives in the LUT lookup; clear any previously-stamped
    // post-LUT color gain so it does not double-apply (#936).
    mat.updateIntensity(1);
    mat.updateOffset(0);
  } else {
    mat.updateIntensity(intensity);
    mat.updateOffset(offset);
  }
}

/** True when a material exposes the {@link LuxarMaterial} update surface. */
export function isLuxarMaterial(m: THREE.Material): m is LuxarMaterial {
  return (
    typeof (m as LuxarMaterial).updateIntensity === 'function' &&
    typeof (m as LuxarMaterial).updateGamma === 'function'
  );
}
