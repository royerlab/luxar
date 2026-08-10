/**
 * Sharpness radial LUT for the volumetric line primitive's sum lanes
 * (issue #1352, PR-4).
 *
 * ## What it tabulates
 *
 * The sum-family lanes factor the ray integral as `I = a · radial(D) · F`
 * with `F` the (β = 2) axial-window machinery and `radial` the
 * line-of-sight-integrated cross-section of the tube at normalized
 * perpendicular distance `q = D / (T·σ_eff)`. Before this LUT the radial
 * factor was the analytic Gaussian marginal — correct only at the default
 * sharpness knob 0.5 (β = 2). This module tabulates the general-β radial:
 *
 *   S(q, s) = max(A_β(q) − A_β(1), 0) / (A_β(0) − A_β(1)),
 *   A_β(q)  = ∫ exp(−K · (q² + u²)^{β/2}) du   over u ∈ (−∞, ∞),
 *   β       = 2^(6s − 2)                        (the shaders' knob map),
 *
 * i.e. the *untruncated* Abel transform of the repo profile
 * `exp(−K·rad^β)`, shifted to hit exactly 0 at q = 1 and normalized to
 * exactly 1 at q = 0. Two properties are load-bearing:
 *
 * - **The β = 2 row is the analytic radial IDENTICALLY.** The Abel
 *   transform of a Gaussian is a Gaussian of the same σ, so
 *   S(q, 0.5) = (exp(−K·q²) − C) / (1 − C) — byte-for-byte the expression
 *   the sum lanes used before. The fragment therefore ALWAYS samples the
 *   LUT: there is no analytic/LUT seam anywhere on the knob axis. This
 *   is literal only because the knob grid CONTAINS s = 0.5 (see the
 *   height constant): the default knob reads that row exactly instead of
 *   interpolating its neighbours.
 * - **Compact support at q = 1.** The shift pins every row to 0 exactly at
 *   the stencil's truncation radius, so the vertex-stage stadium (radius
 *   T·σ_eff) covers the profile at every sharpness, and ClampToEdge
 *   returns exactly 0 for any q ≥ 1 the stencil's AA apron lets through.
 *
 * The AXIAL window deliberately stays the β = 2 erf machinery — the
 * accepted trade recorded in the #1352 plan. The separable model is exact
 * for an infinite rod (mid-segment, any β); the error is confined to the
 * END CAPS, where the erf keeps a Gaussian-shaped rolloff instead of the
 * profile's own end shape. Measured against the capsule-consistent 3D
 * density, the worst cap-point deviation is ~0.5× (β = 0.25) to ~0.77×
 * (β = 16) of the mid-segment center intensity: a high-β line's ends
 * fade like Gaussian ends instead of extending bluntly ~T·σ past the
 * endpoint. Revisited at the G1 visual gate.
 *
 * ## Texture
 *
 * 128 × 65 (q linear × knob linear — the knob axis is log-β by
 * construction), R16 float, linear filtering, clamp-to-edge, sampled at
 * texel centers: u = (q·(W−1) + 0.5)/W, v = (s·(H−1) + 0.5)/H. Half-float
 * linear filtering is CORE in WebGL2 (unlike 32F, which needs
 * OES_texture_float_linear) and r16float is filterable in WebGPU, so no
 * runtime capability fallback is needed on either supported backend.
 * Built lazily ONCE per session (~2×10⁶ integrand evaluations, ~90 ms
 * measured in node) and cached — the colormap-textures pattern.
 *
 * This module is the SINGLE SOURCE for the LUT: the CPU reference
 * (`lineRadialProfile`), the texture builder, and the dimensions the two
 * shader backends fold into their UV math all live here, so the unit
 * tests hold the texture against the same quadrature the shaders sample.
 *
 * @module rendering/materials/_shared/line-integral-lut
 */

import * as THREE from 'three';
import { FALLOFF_K } from './falloff';

/** LUT width: the normalized-distance axis q ∈ [0, 1], sampled linearly. */
export const LINE_RADIAL_LUT_WIDTH = 128;

/**
 * LUT height: the sharpness-KNOB axis s ∈ [0, 1], sampled linearly.
 * DELIBERATELY ODD: the grid is `i/(H−1)`, and (H−1) must be even so the
 * DEFAULT knob s = 0.5 lands exactly on a texel center (row 32) — that
 * row is the analytic β = 2 radial, and sampling it exactly (not the
 * average of the β ≈ 1.94 / 2.07 neighbours a 64-row grid interpolates)
 * is what makes the no-seam contract literal. Bonus: 0.25 and 0.75 land
 * on rows 16/48. Pinned by the filtered-at-s=0.5 unit test.
 */
export const LINE_RADIAL_LUT_HEIGHT = 65;

/**
 * The shaders' sharpness-knob → super-Gaussian exponent map
 * (`beta = exp2(6·s − 2)`: s = 0 → 0.25, s = 0.5 → 2, s = 1 → 16).
 * Exported so the LUT builder and the peak lanes can never disagree.
 */
export function lineSharpnessKnobToBeta(sharpKnob: number): number {
  return Math.pow(2, 6 * sharpKnob - 2);
}

/**
 * Tanh-sinh (double-exponential) quadrature parameters for the Abel
 * integral. DE handles both hard parts of this integrand family at once:
 * the u → 0 derivative CUSP at β < 1 (a plain compactified trapezoid
 * stalls at ~5e-3 there — measured) and the u ~ 10³ tails at small β.
 * n = 256 over x ∈ [−6, 6] is converged far past the texture's own
 * half-float storage precision (~4.9e-4): worst self-convergence vs a
 * 4096-node run is 3.2e-5 at β = 16 and ≤ 5e-15 everywhere else —
 * pinned against the analytic β = 2 row in the unit tests. (512 nodes
 * buys 7e-9 at β = 16 for double the build time; not worth it under
 * 16-bit storage.)
 */
const ABEL_QUADRATURE_NODES = 256;
const ABEL_QUADRATURE_XMAX = 6;

/**
 * The untruncated Abel transform A_β(q) = 2·∫₀^∞ exp(−K·(q²+u²)^{β/2}) du,
 * via the tanh-sinh substitution u = exp(π/2·sinh(x)): the u = 0 endpoint
 * maps to x → −∞ with double-exponentially vanishing weights (neutralizing
 * the β < 1 cusp), and the profile's decay beats the Jacobian's growth at
 * x → +∞ for every β > 0.
 */
export function lineRadialAbel(q: number, beta: number): number {
  const q2 = q * q;
  const n = ABEL_QUADRATURE_NODES;
  const h = (2 * ABEL_QUADRATURE_XMAX) / n;
  let sum = 0;
  for (let k = 0; k <= n; k++) {
    const x = -ABEL_QUADRATURE_XMAX + k * h;
    const u = Math.exp((Math.PI / 2) * Math.sinh(x));
    const w = (Math.PI / 2) * Math.cosh(x) * u;
    const v = Math.exp(-FALLOFF_K * Math.pow(q2 + u * u, beta / 2)) * w;
    sum += k === 0 || k === n ? 0.5 * v : v;
  }
  return 2 * sum * h;
}

/**
 * The shifted + normalized radial profile S(q, s) — the CPU reference the
 * texture is built from and the unit tests validate against. Clamped to 0
 * beyond the q = 1 support (matching the shader's stencil truncation).
 */
export function lineRadialProfile(q: number, sharpKnob: number): number {
  const beta = lineSharpnessKnobToBeta(sharpKnob);
  const a0 = lineRadialAbel(0, beta);
  const a1 = lineRadialAbel(1, beta);
  return Math.max(lineRadialAbel(q, beta) - a1, 0) / (a0 - a1);
}

/**
 * Build the raw LUT rows: `height` sharpness rows of `width` q samples,
 * both axes spanning [0, 1] inclusive (texel i ↦ i/(N−1), so the shader's
 * texel-center UV map lands q = 0 / q = 1 / s = 0 / s = 1 exactly on the
 * first/last texel). Row-major, matching THREE.DataTexture layout.
 *
 * Parameterized dimensions so the tests can hold the shipped knob axis
 * against a denser rebuild (linear-interpolation adequacy).
 */
export function buildLineRadialLUTData(
  width: number = LINE_RADIAL_LUT_WIDTH,
  height: number = LINE_RADIAL_LUT_HEIGHT
): Float32Array {
  const data = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    const beta = lineSharpnessKnobToBeta(row / (height - 1));
    const a0 = lineRadialAbel(0, beta);
    const a1 = lineRadialAbel(1, beta);
    const invRange = 1 / (a0 - a1);
    for (let col = 0; col < width; col++) {
      const q = col / (width - 1);
      data[row * width + col] = Math.max(lineRadialAbel(q, beta) - a1, 0) * invRange;
    }
  }
  return data;
}

let lutTexture: THREE.DataTexture | null = null;

/**
 * Lazy singleton R16F LUT texture. Read-only after creation, shared by
 * every volumetric line material on both backends (the same one-instance
 * pattern as the colormap textures). Never disposed — it is one ~16 KB
 * texture for the whole session.
 */
export function getLineRadialLUTTexture(): THREE.DataTexture {
  if (lutTexture) return lutTexture;
  const f32 = buildLineRadialLUTData();
  const half = new Uint16Array(f32.length);
  for (let i = 0; i < f32.length; i++) half[i] = THREE.DataUtils.toHalfFloat(f32[i]);
  const texture = new THREE.DataTexture(
    half,
    LINE_RADIAL_LUT_WIDTH,
    LINE_RADIAL_LUT_HEIGHT,
    THREE.RedFormat,
    THREE.HalfFloatType
  );
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  lutTexture = texture;
  return texture;
}
