/**
 * The shader-side near fade, modelled in TypeScript for the tests that pin
 * `MAX_NEAR_FAR_RATIO`'s losslessness derivation.
 *
 * That derivation is a claim about the SHADERS ("everything the near-plane
 * floor clips was already suppressed by the point / line / gsplat near fade"),
 * so the arithmetic testing it has to model `perspectiveNearFade`. Keeping the
 * model in one place means `bounds-math.test.ts` and
 * `bounds-math.property.test.ts` cannot drift apart — and
 * `bounds-math.test.ts` grep-locks the model against the real GLSL sources, so
 * neither can drift away from the shaders either.
 *
 * Not a `.test.ts` file: vitest's default include only collects those, so this
 * carries no tests of its own.
 */

/**
 * The fade value below which the point and gsplat vertex shaders discard the
 * vertex (lines instead multiply the fade in per-fragment, reaching ~0 over
 * the same band).
 */
export const NEAR_FADE_REJECT = 0.01;

/** GLSL `smoothstep`, as used by `perspectiveNearFade`. */
export function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/**
 * `x / nearCull` at which `smoothstep(nearCull, 2*nearCull, x)` reaches
 * `reject` — the top of the band inside which the three emissive geometry
 * types are already suppressed.
 *
 * Solved numerically rather than written down as a literal, so retuning the
 * reject threshold moves the derived constraint instead of quietly
 * invalidating it. `smoothstep` is monotone on [0, 1], so bisection converges.
 */
export function fadeRejectHeadroom(reject: number = NEAR_FADE_REJECT): number {
  let lo = 0;
  let hi = 1;
  for (let i = 0; i < 200; i++) {
    const t = (lo + hi) / 2;
    if (smoothstep(0, 1, t) < reject) lo = t;
    else hi = t;
  }
  return 1 + (lo + hi) / 2;
}
