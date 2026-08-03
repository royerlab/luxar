/**
 * Shared helpers for colormap LUT + scalar-range plumbing on
 * Point/Line/GSplat materials.
 *
 * The actual uniform / define writes live on the materials themselves
 * (see `ColormapAwareMaterial`). These helpers wrap the material's
 * setters with two extra concerns the materials don't care about:
 *
 *   - **transition tracking** — `applyColormapTextureToMaterial`
 *     returns `{ wasEnabled, nowEnabled }` so callers (PointMaterial
 *     toggling `vertexColors`, `material-manager` deciding whether
 *     to rebuild) can detect an on/off flip.
 *   - **userData.scalarRange bookkeeping** — package-level metadata
 *     that lets later code inspect the colormap range without
 *     reaching into uniforms.
 *
 * No `material.uniforms.X.value =` writes happen in this file. That
 * keeps the WebGPU-port surface confined to the material classes.
 */

import * as THREE from 'three';
import {
  isColormapAwareMaterial,
  type ColormapAwareMaterial,
} from './materials/_shared/colormap-aware-material';
import type { GeometryTypeName } from '../types/format-contract';

/**
 * Apply a colormap texture (or `null` to disable) to a material.
 *
 * Delegates the uniform / define mutation to the material's
 * `setColormapTexture` setter; reports the enabled-state transition
 * so callers can apply side effects on a flip.
 *
 * Materials that don't implement `ColormapAwareMaterial` are
 * silently ignored — callers can pass any `THREE.ShaderMaterial`
 * and the helper falls through. The enabled-state check still uses
 * `material.defines.USE_COLORMAP` because that's the canonical
 * source of truth (see `setColormapTexture`).
 *
 * @param material - The shader material to mutate.
 * @param texture - The LUT texture to bind, or `null` to disable.
 * @returns `{ wasEnabled, nowEnabled }` so callers can detect a transition.
 * @public
 */
export function applyColormapTextureToMaterial(
  material: THREE.Material,
  texture: THREE.DataTexture | null
): { wasEnabled: boolean; nowEnabled: boolean } {
  const wasEnabled = !!material.defines && 'USE_COLORMAP' in material.defines;
  const nowEnabled = !!texture;

  if (isColormapAwareMaterial(material)) {
    (material as unknown as ColormapAwareMaterial).setColormapTexture(texture);
  }

  if (!nowEnabled) {
    delete material.userData.scalarRange;
  }

  return { wasEnabled, nowEnabled };
}

/**
 * Apply a scalar-range update to a material's colormap uniforms.
 * Records `[min, max]` on `material.userData.scalarRange` so later
 * code can inspect the range without touching uniforms.
 *
 * Materials that don't implement `ColormapAwareMaterial` are
 * silently ignored. The `userData.scalarRange` is still recorded
 * — it's the public "what was the last range" record regardless
 * of whether the material actually rendered with a colormap.
 *
 * @param material - The shader material to mutate.
 * @param min - Lower bound of the scalar range.
 * @param max - Upper bound of the scalar range.
 * @public
 */
export function applyScalarRangeToMaterial(
  material: THREE.Material,
  min: number,
  max: number
): void {
  if (isColormapAwareMaterial(material)) {
    (material as unknown as ColormapAwareMaterial).setScalarRange(min, max);
  }
  material.userData.scalarRange = [min, max];
}

/**
 * Whether scalar-colormap support for a node type is end-to-end wired
 * (loader → projection → geometry storage → shader scalar source →
 * material define).
 *
 * - GSplats use the always-present amplitude (texel0.w of the splat
 *   texture) as the scalar source, so colormap mode is supported
 *   unconditionally.
 * - Points carry their scalar in texel2.x of the point texture, which
 *   exists in the FIXED 3-texel layout whether or not the dataset has
 *   scalars (0.0 identity fill) — so presence is knowable only from the
 *   `userData.hasScalars` stamp the texel writers set from
 *   `data.scalars !== undefined` (`createPointsGeometry` and the
 *   gpu-buffer-pool points adapter). This is the texture-storage analog
 *   of the old `hasAttribute('aScalar')` probe, driven by the same
 *   signal that used to bind the attribute.
 * - Lines carry their per-endpoint scalars in texel5.xy of the line
 *   texture — same fixed-layout situation as points, so presence rides
 *   the identical `userData.hasScalars` stamp (`stampLinePresenceFlags`
 *   in line-geometry.ts, called by every texel-write path).
 *
 * Use this to fail-closed: if `false`, the caller should NOT enable
 * `USE_COLORMAP` and should log a warning so the user understands why
 * a metadata-authored colormap did not take effect.
 *
 * @param nodeType - Any {@link GeometryTypeName}. Exhaustive: a new geometry
 *   type must state its own scalar-presence rule below rather than inheriting
 *   the fail-closed tail, which would silently suppress its colormaps.
 * @param geometry - Optional buffer geometry; required for `points` and
 *   `lines`. When `undefined` for those types, returns `false` (fail-closed).
 * @returns `true` only when the geometry carries the per-node-type
 *   scalar data.
 * @public
 */
export function supportsScalarColormap(
  nodeType: GeometryTypeName,
  geometry?: THREE.BufferGeometry
): boolean {
  if (nodeType === 'gsplats') return true;
  if (nodeType === 'points' || nodeType === 'lines') {
    // Scalar presence stamp — see the doc block above.
    return geometry ? geometry.userData?.hasScalars === true : false;
  }
  // Exhaustiveness guard: with every GeometryTypeName handled above, `nodeType`
  // is `never` here. Adding a geometry type breaks this assignment, forcing an
  // explicit decision instead of a silent fail-closed `false`.
  const unhandled: never = nodeType;
  return unhandled;
}
