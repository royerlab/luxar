/**
 * `uDensityDrop` — the per-node projected-density thinning uniform.
 *
 * Every Luxar leaf material (points / lines / gsplats, GLSL and TSL, visual
 * and picking) declares `uDensityDrop`: the fraction of the node's elements
 * the vertex stage drops, chosen per storage index by `luxarDensityDropped()`
 * / `densityDroppedNode()`. The value is written by the density guard
 * (`scene/density-guard.ts`) and mirrored onto the picking material by the
 * pick pass, so a thinned element can neither be seen nor picked.
 *
 * Duck-typed on `material.uniforms` so one helper serves `ShaderMaterial`
 * uniform records and the TSL materials' `proxyIUniform` records alike.
 *
 * @module rendering/materials/_shared/density-drop
 */

export const DENSITY_DROP_UNIFORM = 'uDensityDrop';

interface DensityDropUniforms {
  uniforms?: Record<string, { value: unknown } | undefined>;
}

/** Fraction dropped on `material`, or 0 when it has no such uniform. */
export function getDensityDrop(material: unknown): number {
  const u = (material as DensityDropUniforms | null | undefined)?.uniforms?.[DENSITY_DROP_UNIFORM];
  const v = u?.value;
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/**
 * Set the dropped fraction (clamped to [0, 1)) on `material`. Returns true
 * when the material carries the uniform and the value changed.
 */
export function setDensityDrop(material: unknown, drop: number): boolean {
  const u = (material as DensityDropUniforms | null | undefined)?.uniforms?.[DENSITY_DROP_UNIFORM];
  if (!u) return false;
  const next = Math.min(0.999999, Math.max(0, drop));
  if (u.value === next) return false;
  u.value = next;
  return true;
}
