/**
 * Shared constants for the Points/Lines shifted super-Gaussian sprite falloff.
 *
 * Lives in `_shared/` (not under `point/` or `line/`) because the profile is
 * geometry-agnostic: points evaluate it radially and lines perpendicular to the
 * segment, and the GLSL/TSL twins of BOTH geometries — plus both picking
 * shaders — must emit identical literals (the `tsl-shader-parity` harness and
 * the codegen snapshots depend on it). Same rationale as `volumetric.ts`.
 *
 * The profile is
 *
 *     falloff(rho) = max(0, exp(-K * rho^beta) - C) / (1 - C)
 *
 * on the normalized sprite coordinate `rho` in [0, 1], where `C` is the value
 * the untruncated kernel would have at the sprite edge — so the shifted profile
 * reaches exactly 0 there, with no discontinuity.
 *
 * @module rendering/materials/_shared/falloff
 */

/**
 * Sprite-edge iso-contour: the fraction of peak intensity at `rho = 1`, which
 * the shift subtracts away. 1% is the historical choice — small enough that the
 * visible sprite edge is imperceptible, large enough to keep `K` modest.
 */
export const FALLOFF_FLOOR = 0.01;

/**
 * Falloff steepness `K = ln(1 / FALLOFF_FLOOR) = ln(100)`.
 *
 * IMPORTANT — the `toFixed(7)` is load-bearing, not cosmetic. Raw
 * `Math.log(100)` is `4.605170185988092`; interpolating that into the GLSL
 * shader source (and emitting it via TSL codegen) would rewrite the literal in
 * every `src/tests/__codegen__/*.fragment.glsl.txt` snapshot. Rounding to 7
 * decimals reproduces the exact `4.6051702` those snapshots and the
 * hand-written GLSL have always contained.
 *
 * 7 is not arbitrary: it is float32's significant-digit count, so
 * `Math.fround(FALLOFF_K) === Math.fround(Math.log(100))` — the rounding is a
 * no-op on the GPU. Both facts are pinned in `falloff.test.ts`.
 */
export const FALLOFF_K = Number(Math.log(1 / FALLOFF_FLOOR).toFixed(7));

/**
 * Renormalization so the shifted profile still peaks at exactly 1.0.
 * Serializes as `1.0101010101010102`, matching the codegen snapshots.
 */
export const INV_ONE_MINUS_FALLOFF_FLOOR = 1 / (1 - FALLOFF_FLOOR);

/**
 * The Gaussian truncation radius `T` (in sigmas) at which a truncated Gaussian
 * coincides *exactly* with this super-Gaussian at `beta = 2`: setting
 * `exp(-T²/2) = exp(-K)` gives `T = sqrt(2K) = 3.0349`.
 *
 * This is NOT the gsplat render default (`GSPLAT_DEFAULT_TRUNCATION_RADIUS`,
 * 2.75) and is not expected to equal it — points/lines size their sprite to the
 * 1% iso-contour, gsplats truncate at a chosen sigma cutoff, and the ~10% gap
 * between the two is deliberate. It IS the value `luxar.gsplats.lift` is
 * calibrated near (it uses 3.0); see `LIFT_TRUNCATION_RADIUS` there.
 */
export const GAUSSIAN_EQUIVALENT_TRUNCATION = Math.sqrt(2 * FALLOFF_K);
