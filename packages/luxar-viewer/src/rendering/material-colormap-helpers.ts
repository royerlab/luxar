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
} from './colormap-aware-material';

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
  material: THREE.ShaderMaterial,
  texture: THREE.DataTexture | null
): { wasEnabled: boolean; nowEnabled: boolean } {
  const wasEnabled = 'USE_COLORMAP' in material.defines;
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
  material: THREE.ShaderMaterial,
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
 * (loader → projection → geometry → shader attribute → material define).
 *
 * - GSplats use the always-present `aAmplitude` attribute as the scalar
 *   source, so colormap mode is supported unconditionally.
 * - Points need a `scalar` attribute on the geometry; this returns
 *   `false` whenever `geometry` lacks `scalar`.
 * - Lines need both `aStartScalar` and `aEndScalar` instanced attributes.
 *
 * Use this to fail-closed: if `false`, the caller should NOT enable
 * `USE_COLORMAP` and should log a warning so the user understands why
 * a metadata-authored colormap did not take effect.
 *
 * @param nodeType - `'points' | 'lines' | 'gsplats'`.
 * @param geometry - Optional buffer geometry; required for `points` and
 *   `lines`. When `undefined` for those types, returns `false` (fail-closed).
 * @returns `true` only when the geometry has the per-node-type scalar
 *   attributes wired up.
 * @public
 */
export function supportsScalarColormap(
  nodeType: 'points' | 'lines' | 'gsplats',
  geometry?: THREE.BufferGeometry
): boolean {
  if (nodeType === 'gsplats') return true;
  if (nodeType === 'points') {
    return geometry ? geometry.hasAttribute('scalar') : false;
  }
  if (nodeType === 'lines') {
    if (!geometry) return false;
    return geometry.hasAttribute('aStartScalar') && geometry.hasAttribute('aEndScalar');
  }
  return false;
}
