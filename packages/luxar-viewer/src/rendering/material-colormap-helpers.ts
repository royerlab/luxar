/**
 * Shared helpers for colormap LUT + scalar-range uniform updates.
 * Point/Line/GSplat materials all bind the same `uColormapTex` /
 * `uScalarMin` / `uScalarScale` uniform contract and the same
 * `USE_COLORMAP` define toggle. PointMaterial additionally needs to
 * toggle `vertexColors` (its shader uses the color attribute when
 * colormap is OFF, scalar+LUT when ON); Lines and GSplats don't.
 *
 * These helpers do the uniform/define plumbing and return whether the
 * enabled-state changed so the caller can decide what other side
 * effects to apply (vertexColors, needsUpdate).
 */

import * as THREE from 'three';

/**
 * Apply a colormap texture (or null to disable) to a material's
 * USE_COLORMAP define + uColormapTex uniform contract. Returns the
 * before/after enabled state so the caller can apply material-
 * specific side effects on a transition.
 */
export function applyColormapTextureToMaterial(
  material: THREE.ShaderMaterial,
  texture: THREE.DataTexture | null
): { wasEnabled: boolean; nowEnabled: boolean } {
  const wasEnabled = 'USE_COLORMAP' in material.defines;
  const nowEnabled = !!texture;

  if (nowEnabled) {
    material.defines.USE_COLORMAP = '';
    if (!material.uniforms.uColormapTex) {
      material.uniforms.uColormapTex = { value: texture };
      material.uniforms.uScalarMin = { value: 0.0 };
      material.uniforms.uScalarScale = { value: 1.0 };
    } else {
      material.uniforms.uColormapTex.value = texture;
    }
  } else {
    delete material.defines.USE_COLORMAP;
  }

  return { wasEnabled, nowEnabled };
}

/**
 * Apply a scalar-range update to the `uScalarMin` / `uScalarScale`
 * uniform contract (used in the colormap-mode shader path). Stores
 * the original `[min, max]` on `userData.scalarRange` for later
 * inspection. No-op for materials whose colormap mode is disabled
 * (uniforms aren't allocated until `applyColormapTextureToMaterial`
 * runs the first time).
 */
export function applyScalarRangeToMaterial(
  material: THREE.ShaderMaterial,
  min: number,
  max: number
): void {
  if (material.uniforms.uScalarMin) {
    material.uniforms.uScalarMin.value = min;
  }
  if (material.uniforms.uScalarScale) {
    material.uniforms.uScalarScale.value = 1.0 / Math.max(1e-10, max - min);
  }
  material.userData.scalarRange = [min, max];
}
