/**
 * Marker interface for materials that own colormap-LUT uniforms.
 *
 * Implemented by `PointMaterial`, `LineMaterial`, `GSplatMaterial`.
 * Their picking counterparts deliberately do NOT implement it
 * (picking doesn't sample colormaps).
 *
 * The two setters are the public surface for the colormap-helper
 * functions in `material-colormap-helpers.ts`; nobody else should
 * reach into `material.uniforms.uColormapTex` / `.uScalarMin` /
 * `.uScalarScale` directly. This separation is part of the WebGPU
 * prep: under `NodeMaterial` the `.uniforms` shape goes away, so
 * confining writes to setter methods today means the WebGPU port
 * rewrites the setters' bodies and nothing outside changes.
 */
import * as THREE from 'three';

export interface ColormapAwareMaterial {
  /**
   * Bind a colormap LUT texture (or `null` to disable colormap mode).
   *
   * When `texture` is non-null, the `USE_COLORMAP` define is added
   * and the uniforms (`uColormapTex`, `uScalarMin`, `uScalarScale`)
   * are allocated with neutral defaults if they don't exist yet.
   * When `texture` is null, the define is removed and the uniform
   * values are cleared so a subsequent `clone()` doesn't resurrect
   * a stale colormap.
   *
   * No-op when called on a material whose shader path doesn't
   * include `USE_COLORMAP`-guarded uniforms.
   */
  setColormapTexture(texture: THREE.DataTexture | null): void;

  /**
   * Update the scalar range `[min, max]` the colormap maps over.
   * Encoded as `uScalarMin = min` and `uScalarScale = 1/max(1e-10, max-min)`
   * for the shader. No-op when colormap mode is not active (uniforms
   * are absent until `setColormapTexture` has been called once).
   */
  setScalarRange(min: number, max: number): void;
}

/** Type guard. */
export function isColormapAwareMaterial(material: unknown): material is ColormapAwareMaterial {
  if (typeof material !== 'object' || material === null) return false;
  const m = material as Record<string, unknown>;
  return typeof m.setColormapTexture === 'function' && typeof m.setScalarRange === 'function';
}
