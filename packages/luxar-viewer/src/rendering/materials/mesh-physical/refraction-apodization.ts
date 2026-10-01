/**
 * Edge apodization of the glass refraction shift (both backends).
 *
 * Three's transmission samples the scene behind a glass at the screen projection of
 * `position + refract(…)·thickness` and never checks that the projection is still on
 * screen. Where the shift carries it past an edge, the texture sampler clamps to the
 * border texel, and the border is smeared inward as streaks. A camera inside a
 * refracting shell is the worst case: every pixel is glass, and on WebGL a double-sided
 * shell is refracted twice (three pre-renders its back faces into the very texture
 * those faces then sample), so the overshoot compounds.
 *
 * The fix shapes the SHIFT FIELD rather than the sample. Per fragment, with the
 * unshifted point at NDC `b` and the shift `s` (both from the same ray three will
 * trace), each axis has `room = 1 − |b|` to its nearest border and the shift is
 * soft-limited against it: the identity up to {@link REFRACTION_SHIFT_KNEE}·room, then
 * a tanh roll-off that approaches {@link REFRACTION_SHIFT_CEILING}·room. The vector
 * keeps its direction — the smaller of the two axis factors scales both — so the field
 * is:
 *
 * - **zero on every screen border** (room 0 there);
 * - **continuous** (and C¹ through the knee), so no seam appears where it engages;
 * - **the identity** wherever the shift stays within the knee of the room, which is
 *   every pixel of an ordinary glass away from the edges — no change in the centre;
 * - **well-conditioned**: the ceiling is what keeps edge detail. Were the shift allowed
 *   to approach the WHOLE room, a large shift would saturate there, the sample point
 *   `b + s` would sit on the border for a whole band of pixels, and that band would show
 *   the border column smeared across it — the very streaks being removed, only on
 *   screen instead of clamped. Capped at a fraction `c` of the room, the sample point
 *   still advances at least `(1 − c)` as fast as the pixel, so the edge band is at most
 *   stretched by `1 / (1 − c)` (2× here), never collapsed.
 *
 * The shaped shift is applied by shortening the ray, not by rewriting the sample
 * coordinate, so it goes through three's own code path on both backends (three's TSL
 * refraction function is private). A point a fraction `λ` along the world ray projects
 * to the fraction `φ = λw₁ / ((1−λ)w₀ + λw₁)` of the screen shift (the
 * perspective-correct interpolation of the clip `w`s at its two ends), so the wanted
 * fraction `κ` needs exactly `λ = κw₀ / (κw₀ + (1−κ)w₁)`. Three receives `thickness·λ`;
 * `attenuationDistance·λ` keeps the Beer–Lambert exponent (path length over distance)
 * exactly what it was, so a tinted glass does not lighten near the edges.
 *
 * With dispersion, three traces three IORs; the scale is taken from the most refracted
 * one (`ior + halfSpread`), which bounds the other two.
 *
 * The TypeScript functions here are the reference the shader text and the TSL twin
 * (`./refraction-apodization-tsl.ts`) mirror; the unit tests hold all of them to it.
 *
 * @module rendering/materials/mesh-physical/refraction-apodization
 */

import { log, Modules } from '../../../utils/log';

/**
 * Fraction of the room to the border a shift may use before it is shaped. Below it the
 * field is untouched; above it, the tanh roll-off takes over.
 */
export const REFRACTION_SHIFT_KNEE = 0.25;

/**
 * Fraction of the room a shaped shift approaches and never reaches. Bounds how much the
 * edge band can be stretched, `1 / (1 − ceiling)`: half the room is at most 2×.
 */
export const REFRACTION_SHIFT_CEILING = 0.5;

/**
 * Floor on the ray scale. Keeps the attenuation compensation finite (an infinite
 * attenuation distance times zero would be NaN) at the border pixel itself, where the
 * exact scale is 0; the residual shift is a millionth of the unshaped one.
 */
export const REFRACTION_MIN_RAY_SCALE = 1e-6;

/** Three's dispersion spread: the outer IORs are `ior ± (ior − 1)·0.025·dispersion`. */
export const DISPERSION_HALF_SPREAD_PER_UNIT = 0.025;

/**
 * Factor by which ONE axis of the shift is scaled: 1 up to the knee, then the soft
 * limit that keeps `|shift|·factor < ceiling·room`. 0 when there is no room at all.
 * @param shift - Absolute shift along the axis, in NDC units (≥ 0).
 * @param room - Distance from the unshifted point to the nearest border on that axis, NDC.
 */
export function apodizeShiftAxis(shift: number, room: number): number {
  const knee = REFRACTION_SHIFT_KNEE * room;
  if (shift <= knee) return 1;
  const span = (REFRACTION_SHIFT_CEILING - REFRACTION_SHIFT_KNEE) * room;
  if (span <= 0) return 0;
  return (knee + span * Math.tanh((shift - knee) / span)) / shift;
}

/**
 * The fraction `κ` of the screen shift to keep: the stricter of the two axis factors,
 * so the shaped shift points the same way as the unshaped one.
 * @param b - Unshifted NDC position `[x, y]` (inside `[-1, 1]` for a visible fragment).
 * @param s - Unshaped NDC shift `[x, y]`.
 */
export function apodizedShiftFraction(b: readonly number[], s: readonly number[]): number {
  const roomX = Math.max(1 - Math.abs(b[0]), 0);
  const roomY = Math.max(1 - Math.abs(b[1]), 0);
  return Math.min(apodizeShiftAxis(Math.abs(s[0]), roomX), apodizeShiftAxis(Math.abs(s[1]), roomY));
}

/**
 * The world-ray scale `λ` that moves the projected sample by the fraction `kappa` of
 * the screen shift, given the clip `w` at the ray's start (`w0`) and end (`w1`).
 * Floored at {@link REFRACTION_MIN_RAY_SCALE}; a ray that ends behind the camera
 * (`w1 ≤ 0`, where the projection is meaningless) gets the floor too.
 */
export function rayScaleForShiftFraction(kappa: number, w0: number, w1: number): number {
  if (!(w0 > 0) || !(w1 > 0)) return REFRACTION_MIN_RAY_SCALE;
  const lambda = (kappa * w0) / (kappa * w0 + (1 - kappa) * w1);
  return Math.max(lambda, REFRACTION_MIN_RAY_SCALE);
}

/**
 * GLSL twin of the reference above, defined at global scope right after three's
 * `transmission_pars_fragment` (it calls that chunk's `getVolumeTransmissionRay`).
 * Returns the ray scale `λ` for one fragment.
 */
export const REFRACTION_APODIZATION_GLSL_FUNCTIONS = /* glsl */ `
#ifdef USE_TRANSMISSION
	// Luxar: refraction-shift edge apodization (rendering/materials/mesh-physical/refraction-apodization.ts).
	float luxarApodizeShiftAxis( const in float shift, const in float room ) {
		float knee = ${REFRACTION_SHIFT_KNEE.toFixed(4)} * room;
		if ( shift <= knee ) return 1.0;
		float span = ${(REFRACTION_SHIFT_CEILING - REFRACTION_SHIFT_KNEE).toFixed(4)} * room;
		if ( span <= 0.0 ) return 0.0;
		return ( knee + span * tanh( ( shift - knee ) / span ) ) / shift;
	}

	float luxarRefractionRayScale( const in vec3 n, const in vec3 v, const in vec3 position, const in float ior,
		const in float dispersion, const in float thickness, const in mat4 modelMatrix, const in mat4 viewMatrix,
		const in mat4 projMatrix ) {
		float iorMax = ior + ( ior - 1.0 ) * ${DISPERSION_HALF_SPREAD_PER_UNIT.toFixed(4)} * dispersion;
		vec3 ray = getVolumeTransmissionRay( n, v, thickness, iorMax, modelMatrix );
		vec4 c0 = projMatrix * viewMatrix * vec4( position, 1.0 );
		vec4 c1 = projMatrix * viewMatrix * vec4( position + ray, 1.0 );
		if ( c0.w <= 0.0 || c1.w <= 0.0 ) return ${REFRACTION_MIN_RAY_SCALE.toExponential()};
		vec2 b = c0.xy / c0.w;
		vec2 s = abs( c1.xy / c1.w - b );
		vec2 room = max( 1.0 - abs( b ), 0.0 );
		float kappa = min( luxarApodizeShiftAxis( s.x, room.x ), luxarApodizeShiftAxis( s.y, room.y ) );
		float lambda = kappa * c0.w / ( kappa * c0.w + ( 1.0 - kappa ) * c1.w );
		return max( lambda, ${REFRACTION_MIN_RAY_SCALE.toExponential()} );
	}
#endif
`;

/** Three's include the helpers follow (still unresolved when `onBeforeCompile` runs). */
export const TRANSMISSION_PARS_INCLUDE = '#include <transmission_pars_fragment>';

/**
 * The line of three's `transmission_fragment` after which the scale is applied: `n`,
 * `v` and `pos` are the very values three hands `getIBLVolumeRefraction` next.
 */
export const TRANSMISSION_NORMAL_LINE =
  'vec3 n = transformNormalByInverseViewMatrix( normal, viewMatrix );';

/** Inserted after {@link TRANSMISSION_NORMAL_LINE}: shorten the ray, keep the absorption. */
export const REFRACTION_APODIZATION_GLSL_APPLY = /* glsl */ `
	#ifdef USE_DISPERSION
		float luxarDispersion = material.dispersion;
	#else
		float luxarDispersion = 0.0;
	#endif
	float luxarRayScale = luxarRefractionRayScale( n, v, pos, material.ior, luxarDispersion, material.thickness,
		modelMatrix, viewMatrix, projectionMatrix );
	material.thickness *= luxarRayScale;
	material.attenuationDistance *= luxarRayScale;
`;

let warnedChunkDrift = false;

/**
 * Patch a WebGL physical fragment shader whose transmission chunk is already EXPANDED
 * (by `pinTransmittedAlphaGlsl`, which runs first) so its refraction shift is
 * edge-apodized.
 *
 * Pure over its input. A shader without the expanded chunk (no transmission, or a pin
 * that stood down) is returned untouched. If a three upgrade moves the parameters
 * include, the shader is left as it was and one warning names the consequence; the
 * unit test on three's real chunks turns either anchor's drift into a red build.
 *
 * @param fragmentShader - The fragment shader after the transmitted-alpha pin.
 */
export function apodizeRefractionShiftGlsl(fragmentShader: string): string {
  // No expanded chunk: no transmission, or the alpha pin stood down (and said so).
  if (!fragmentShader.includes(TRANSMISSION_NORMAL_LINE)) return fragmentShader;
  if (!fragmentShader.includes(TRANSMISSION_PARS_INCLUDE)) {
    if (!warnedChunkDrift) {
      warnedChunkDrift = true;
      log.warning(
        Modules.RENDERER,
        "three's transmission chunks no longer match the refraction edge apodization's " +
          'anchors; glass will clamp-streak at the screen edges again. Update ' +
          'TRANSMISSION_NORMAL_LINE / TRANSMISSION_PARS_INCLUDE.'
      );
    }
    return fragmentShader;
  }
  return fragmentShader
    .replace(
      TRANSMISSION_PARS_INCLUDE,
      TRANSMISSION_PARS_INCLUDE + REFRACTION_APODIZATION_GLSL_FUNCTIONS
    )
    .replace(
      TRANSMISSION_NORMAL_LINE,
      TRANSMISSION_NORMAL_LINE + REFRACTION_APODIZATION_GLSL_APPLY
    );
}
