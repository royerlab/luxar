/**
 * The TSL half of the erf approximation.
 *
 * Split out of `erf.ts` so that file stays free of `three/tsl`. The CPU
 * `erfRef` it exports is used by `materials/gsplat/math.ts` — a pure
 * number-in/number-out helper on the GLSL path — and a single `three/tsl`
 * import there was enough to pull the TSL/WebGPU cone into the eager bundle for
 * every WebGL session (issue #1679). Nothing about the maths changed; see
 * `erf.ts` for the polynomial, its coefficients, and the GLSL twin.
 *
 * @module rendering/materials/_shared/erf-tsl
 */

import { abs, float, min, sign } from 'three/tsl';

import { ERF_POLY_CLAMP, ERF_POLY_COEFFS } from './erf';
import type { TSLNode } from './tsl-helpers';

/**
 * TSL twin of `luxarErf`, built from the SAME coefficient values (the
 * code generator owns literal formatting — see the module header).
 * The sign is applied by multiplying with `sign(x)` — genuinely
 * branchless in the GENERATED code (a `select` would lower to an
 * `if`/`else` that duplicates the whole polynomial across both arms),
 * and legal both inside and outside an `Fn()` body. `sign(0) = 0`
 * matches the polynomial (`P(0) = 0`), and NaN falls under the same
 * caveat as the GLSL form.
 */
export function erfPolyTSL(x: TSLNode): TSLNode {
  const ax = min(abs(x), float(ERF_POLY_CLAMP));
  const t = ax.mul(ax);
  let p: TSLNode = float(ERF_POLY_COEFFS[ERF_POLY_COEFFS.length - 1]);
  for (let k = ERF_POLY_COEFFS.length - 2; k >= 0; k--) {
    p = p.mul(t).add(float(ERF_POLY_COEFFS[k]));
  }
  p = p.mul(ax);
  return p.mul(sign(x));
}
