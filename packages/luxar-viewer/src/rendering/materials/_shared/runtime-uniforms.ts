/**
 * The RUNTIME uniforms a visual material's `clone()` must carry over: the
 * values written after construction (camera broadcast, depth-sort
 * coordinator, density guard, gamma/texture updates) that a fresh material
 * built from the source's config would otherwise reset to the constructor
 * defaults. Config-derived state (opacity, gain, blending, colormap, …) is
 * re-derived by the constructor from the config each clone passes, and
 * texture rebinds go through each wrapper's own rebind chokepoint (which
 * re-stamps the baked texture width), so neither is listed here.
 *
 * ONE list per geometry type, shared by the GLSL and TSL twins so the two
 * clones cannot drift; a name a twin does not declare (e.g. `uLineJoin`, a
 * runtime uniform only in GLSL; `uIsOrtho`, the TSL line graph variant) is
 * skipped for it. Why each rides along:
 *
 *   - `uResolution`, `uPixelRatio`, `uNearCull`, `maxPointSize`,
 *     `uMaxLinePixelWidth`, `uIsOrtho`: the camera broadcast — a clone
 *     otherwise renders with stale camera state until the next broadcast.
 *   - `uSortedIndexSlot`: the active ordering buffer — a clone taken while
 *     the geometry draws from slot 1 would read the stale buffer until the
 *     coordinator's next per-frame re-assert.
 *   - `uDensityDrop`, `uDensityAlphaExp`: the density guard's thinning — it
 *     is re-asserted only on the guard's next visit, which an off-screen
 *     node never gets.
 *   - `uInvGamma`, `uProjectionMode`, `radiusScale`, `uLineJoin`: derived
 *     state a later update set (gamma, blending's projection, the commit's
 *     dtype scaling, the join override).
 *
 * @module rendering/materials/_shared/runtime-uniforms
 */

const ORDERING_AND_DENSITY = ['uSortedIndexSlot', 'uDensityDrop', 'uDensityAlphaExp'] as const;

export const GSPLAT_RUNTIME_UNIFORMS = [
  'uResolution',
  'uPixelRatio',
  'uNearCull',
  'uProjectionMode',
  'uInvGamma',
  ...ORDERING_AND_DENSITY,
] as const;

export const POINT_RUNTIME_UNIFORMS = [
  'maxPointSize',
  'uInvGamma',
  'radiusScale',
  'uNearCull',
  'uPixelRatio',
  'uResolution',
  ...ORDERING_AND_DENSITY,
] as const;

export const LINE_RUNTIME_UNIFORMS = [
  'uResolution',
  'uIsOrtho',
  'uNearCull',
  'uPixelRatio',
  'uMaxLinePixelWidth',
  'uLineJoin',
  'uInvGamma',
  ...ORDERING_AND_DENSITY,
] as const;

export const MESH_RUNTIME_UNIFORMS = ['uInvGamma', 'uNearCull'] as const;

interface UniformRecord {
  uniforms: Record<string, { value: unknown } | undefined>;
}

/**
 * Copy the named uniform values from `from` onto `to` (a fresh clone). A
 * vector/matrix/color value is copied INTO the clone's own instance (its
 * uniform must not alias the source's); textures and scalars are assigned.
 * Names either material lacks are skipped.
 */
export function copyRuntimeUniforms(
  from: UniformRecord,
  to: UniformRecord,
  names: readonly string[]
): void {
  for (const name of names) {
    const src = from.uniforms[name];
    const dst = to.uniforms[name];
    if (!src || !dst) continue;
    const value = src.value as { copy?: (v: unknown) => unknown; isTexture?: boolean } | null;
    const target = dst.value as { copy?: (v: unknown) => unknown } | null;
    if (
      value &&
      typeof value === 'object' &&
      value.isTexture !== true &&
      typeof target?.copy === 'function'
    ) {
      target.copy(value);
    } else {
      dst.value = src.value;
    }
  }
}
