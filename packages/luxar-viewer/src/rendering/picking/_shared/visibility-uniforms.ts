/**
 * The four pick visibility-weight uniforms (`./visibility-glsl.ts`,
 * `./visibility-tsl.ts`) — names, neutral defaults and the clone copy, shared
 * by the GLSL and TSL pick wrappers of points, lines and gsplats.
 *
 * Kept free of `three/tsl` so the GLSL wrappers can import it.
 *
 * @module rendering/picking/_shared/visibility-uniforms
 */

/** Neutral defaults: an opaque, unit-gain, non-volumetric node with RGB colours. */
export const PICK_VISIBILITY_DEFAULTS = {
  uIntensity: 1,
  uOpacity: 1,
  uHasElementAlpha: 0,
  uVolumetric: 0,
} as const;

export type PickVisibilityUniformName = keyof typeof PICK_VISIBILITY_DEFAULTS;

export const PICK_VISIBILITY_UNIFORM_NAMES = Object.keys(
  PICK_VISIBILITY_DEFAULTS
) as readonly PickVisibilityUniformName[];

/** Fresh `{ value }` records for a GLSL `ShaderMaterial` uniform block. */
export function pickVisibilityUniforms(): Record<PickVisibilityUniformName, { value: number }> {
  return {
    uIntensity: { value: PICK_VISIBILITY_DEFAULTS.uIntensity },
    uOpacity: { value: PICK_VISIBILITY_DEFAULTS.uOpacity },
    uHasElementAlpha: { value: PICK_VISIBILITY_DEFAULTS.uHasElementAlpha },
    uVolumetric: { value: PICK_VISIBILITY_DEFAULTS.uVolumetric },
  };
}

/** Copy the four values between two uniform records (a clone's). */
export function copyPickVisibilityUniforms(
  from: Record<string, { value: unknown }>,
  to: Record<string, { value: unknown }>
): void {
  for (const name of PICK_VISIBILITY_UNIFORM_NAMES) to[name].value = from[name].value;
}
