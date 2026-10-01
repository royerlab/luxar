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
 * `uDensityAlphaExp` is its alpha-over (`normal`) companion on the VISUAL
 * materials: a thinned alpha-over node has no linear brightness knob, so each
 * surviving element's alpha is raised to `1 − (1 − α)^uDensityAlphaExp` with
 * the exponent `1/keep` — `keep·N` survivors then transmit exactly what `N`
 * elements did. The shaders treat any exponent ≤ 1 (including an unset 0) as
 * the identity, so an unthinned node is bit-identical.
 *
 * @module rendering/materials/_shared/density-drop
 */

export const DENSITY_DROP_UNIFORM = 'uDensityDrop';

export const DENSITY_ALPHA_EXP_UNIFORM = 'uDensityAlphaExp';

interface DensityDropUniforms {
  uniforms?: Record<string, { value: unknown } | undefined>;
}

/** Whether `material` carries the projected-density thinning uniform. */
export function hasDensityDrop(material: unknown): boolean {
  return Boolean(
    (material as DensityDropUniforms | null | undefined)?.uniforms?.[DENSITY_DROP_UNIFORM]
  );
}

/** Fraction dropped on `material`, or 0 when it has no such uniform. */
export function getDensityDrop(material: unknown): number {
  const u = (material as DensityDropUniforms | null | undefined)?.uniforms?.[DENSITY_DROP_UNIFORM];
  const v = u?.value;
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The alpha-over compensation exponent on `material`, or 1 (identity) when unset. */
export function getDensityAlphaExp(material: unknown): number {
  const u = (material as DensityDropUniforms | null | undefined)?.uniforms?.[
    DENSITY_ALPHA_EXP_UNIFORM
  ];
  const v = u?.value;
  return typeof v === 'number' && Number.isFinite(v) && v > 1 ? v : 1;
}

/**
 * Set the alpha-over compensation exponent (floored at the identity 1) on
 * `material`. Returns true when the material carries the uniform and the
 * value changed.
 */
export function setDensityAlphaExp(material: unknown, exponent: number): boolean {
  const u = (material as DensityDropUniforms | null | undefined)?.uniforms?.[
    DENSITY_ALPHA_EXP_UNIFORM
  ];
  if (!u) return false;
  const next = Number.isFinite(exponent) && exponent > 1 ? exponent : 1;
  if (u.value === next) return false;
  u.value = next;
  return true;
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
