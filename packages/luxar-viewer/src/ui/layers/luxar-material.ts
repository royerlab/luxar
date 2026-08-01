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
import type { CameraAwareMaterial } from '../../rendering';
import type { BlendingMode } from '../../rendering';
import { computeDisplayRange } from './layer-state';

// Type guard target: does this material have our update* methods?
export interface LuxarMaterial extends THREE.Material, CameraAwareMaterial {
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
  updateColormapTexture?(texture: THREE.DataTexture | null): void;
  updateScalarRange?(min: number, max: number): void;
  /**
   * Apply a blending mode to this material in-place.
   *
   * All three geometry materials implement it today (the optionality is
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
 * @internal Exported for unit testing the colormap-vs-direct routing.
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
 * @internal Exported for unit testing.
 */
export function applyColorAdjustments(
  mat: LuxarMaterial,
  gamma: number,
  intensity: number,
  offset: number
): void {
  mat.updateGamma(gamma);
  if (isColormapActive(mat) && mat.updateScalarRange) {
    const { min, max } = computeDisplayRange(intensity, offset);
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
