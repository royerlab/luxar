/**
 * Shared mathematical helpers for the Point material pair.
 *
 * Lives outside both `material-glsl.ts` and `material-tsl.ts` so the
 * two backends compute identical numeric values from identical inputs —
 * the parity test (`tsl-shader-parity.spec.ts`) depends on this.
 * Mirrors `materials/gsplat/math.ts`.
 *
 * @module rendering/materials/point/math
 */

/**
 * Through-thickness of the Gaussian-profile point ball per unit radius —
 * the chord factor of the volumetric ray integral (the ISOTROPIC special
 * case of the gsplat ray integral, VOLUMETRIC_BLENDING_SPEC.md §3.1).
 *
 * The fragment's radial profile is `exp(−K·ρ²)` at sharpness 0.5
 * (β = 2, a true Gaussian) with ρ = r/R and K = ln(100), i.e. an
 * isotropic 3D Gaussian with σ = R/√(2K). Its line integral along a ray
 * at normalized offset ρ is
 *
 *     mass(ρ) = falloff(ρ) · σ·√(2π) = falloff(ρ) · R · √(π/K)
 *
 * so the shader computes `rayMass = falloff · vRadius · POINT_CHORD_SCALE`.
 * Other β values reshape the transverse profile while the R-proportional
 * thickness stays — any residual profile-dependent factor folds into the
 * user's κ knob. (Note the profile truncates at ρ = 1 ⇔ T = √(2K) ≈ 3.03σ,
 * nearly the gsplat T = 3 truncation, so point and gsplat κ scales agree.)
 */
export const POINT_CHORD_SCALE = Math.sqrt(Math.PI / Math.log(100.0));
