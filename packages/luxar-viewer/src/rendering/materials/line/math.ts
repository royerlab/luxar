/**
 * Shared mathematical helpers for the Line material pair.
 *
 * Lives outside both `material-glsl.ts` and `material-tsl.ts` so the
 * two backends compute identical numeric values from identical inputs —
 * the parity test (`tsl-shader-parity.spec.ts`) depends on this.
 * Mirrors `materials/point/math.ts` and `materials/gsplat/math.ts`.
 *
 * @module rendering/materials/line/math
 */

/**
 * Through-thickness of the Gaussian-profile line ribbon per unit width —
 * the chord factor of the volumetric ray integral (the TRANSVERSE
 * special case of the gsplat ray integral, VOLUMETRIC_BLENDING_SPEC.md
 * §3.1 / §7 Phase 4).
 *
 * The fragment's perpendicular cross-section is `exp(−K·p²)` at
 * sharpness 0.5 (β = 2, a true Gaussian) with p = distance from the
 * centerline / width and K = ln(100). Locally the line is a Gaussian
 * tube: the cross-section perpendicular to the axis is the same profile
 * in 2D radial coordinates, so a ray crossing the tube at normalized
 * perpendicular offset p integrates the depth direction q to
 *
 *     mass(p) = ∫ exp(−K·(p²+q²)) dq · width
 *             = perpFalloff(p) · width · √(π/K)
 *
 * and the shader computes `rayMass = perpFalloff · vWidthAtT · LINE_CHORD_SCALE`.
 * Other β values reshape the transverse profile while the
 * width-proportional thickness stays — any residual profile-dependent
 * factor folds into the user's κ knob. The value is identical to
 * `POINT_CHORD_SCALE` (the same K, the same Gaussian chord), and both
 * scale by the NOMINAL size attribute (radius / width) rather than the
 * 2× visible sprite extent — the same fold into κ for both, which is
 * what keeps the point/line/gsplat κ scales aligned.
 */
export const LINE_CHORD_SCALE = Math.sqrt(Math.PI / Math.log(100.0));
