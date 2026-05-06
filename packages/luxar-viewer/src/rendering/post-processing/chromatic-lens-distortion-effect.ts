/**
 * ChromaticLensDistortionEffect - Physically accurate lens distortion with chromatic aberration
 *
 * Combines lens distortion and chromatic aberration into a single effect by applying
 * wavelength-dependent distortion coefficients. This simulates real optical dispersion
 * where different wavelengths refract differently through the lens.
 *
 * Physical basis:
 * - Refractive index varies with wavelength (Abbe dispersion)
 * - Shorter wavelengths (blue ~450nm) refract more than longer wavelengths (red ~650nm)
 * - This causes different focal lengths and distortion amounts per color channel
 *
 * Effect characteristics:
 * - Color fringing follows lens geometry (stronger at edges)
 * - Works correctly with both barrel and pincushion distortion
 * - More efficient than separate lens distortion + chromatic aberration passes
 *
 * Based on:
 * - Brown-Conrady distortion model (computer vision standard)
 * - Camera intrinsic matrix parameterization
 * - Original pmndrs LensDistortionEffect (ported from three-lens-distortion)
 */

import { BlendFunction, Effect } from 'postprocessing';
import * as THREE from 'three';

/**
 * Fragment shader for chromatic lens distortion
 * Applies wavelength-dependent distortion to simulate optical dispersion
 */
const fragmentShader = /* glsl */ `
  uniform vec2 distortion;
  uniform vec2 principalPoint;
  uniform vec2 focalLength;
  uniform float skew;
  uniform float dispersion;  // Chromatic dispersion strength

  // Check if UV is within valid texture bounds [0, 1]
  float border(const in vec2 uv) {
    return float(uv.s >= 0.0 && uv.s <= 1.0 && uv.t >= 0.0 && uv.t <= 1.0);
  }

  // Apply lens distortion with given distortion coefficients
  // Returns distorted UV coordinates
  vec2 applyDistortion(vec2 uv, vec2 distortionCoeff) {
    // Convert UV from [0,1] to normalized [-1,1] coordinates
    vec2 xn = 2.0 * (uv - 0.5);

    // Apply Brown-Conrady radial distortion model: r' = r * (1 + k * r²)
    // This is the first-order radial distortion (sufficient for most lenses)
    float r2 = dot(xn, xn);
    vec3 xDistorted = vec3((1.0 + distortionCoeff * r2) * xn, 1.0);

    // Build camera intrinsic matrix K:
    // | fx   s*fx  cx |
    // | 0    fy    cy |
    // | 0    0     1  |
    mat3 kk = mat3(
      vec3(focalLength.x, 0.0, 0.0),
      vec3(skew * focalLength.x, focalLength.y, 0.0),
      vec3(principalPoint.x, principalPoint.y, 1.0)
    );

    // Apply camera matrix and convert back to [0,1] UV space
    return (kk * xDistorted).xy * 0.5 + 0.5;
  }

  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
    // Calculate wavelength-dependent distortion coefficients
    // Physics: shorter wavelengths (blue) refract more than longer wavelengths (red)
    // dispersion parameter controls the strength of chromatic effect

    vec2 distortionR = distortion * (1.0 - dispersion);  // Red: least distortion
    vec2 distortionG = distortion;                        // Green: middle (reference)
    vec2 distortionB = distortion * (1.0 + dispersion);  // Blue: most distortion

    // Apply distortion and sample each channel independently
    vec2 uvR = applyDistortion(uv, distortionR);
    vec2 uvG = applyDistortion(uv, distortionG);
    vec2 uvB = applyDistortion(uv, distortionB);

    // Sample texture at distorted coordinates for each channel
    float r = texture(inputBuffer, uvR).r * border(uvR);
    float g = texture(inputBuffer, uvG).g * border(uvG);
    float b = texture(inputBuffer, uvB).b * border(uvB);

    // Combine channels with boundary masking
    // Use the middle channel (green) for alpha to avoid edge artifacts
    float alpha = border(uvG);

    outputColor = vec4(r, g, b, alpha);
  }
`;

/**
 * Options for configuring the ChromaticLensDistortionEffect
 */
export interface ChromaticLensDistortionEffectOptions {
  /**
   * Blend function for compositing the effect
   * @default BlendFunction.NORMAL
   */
  blendFunction?: BlendFunction;

  /**
   * Radial distortion coefficients [x, y]
   * Negative = Barrel distortion (fish-eye, wide angle lenses)
   * Positive = Pincushion distortion (telephoto lenses)
   * @default [0, 0]
   */
  distortion?: THREE.Vector2;

  /**
   * Principal point offset [x, y]
   * Shifts the optical center away from image center
   * Range: [-1, 1] in normalized coordinates
   * @default [0, 0]
   */
  principalPoint?: THREE.Vector2;

  /**
   * Focal length scale [x, y]
   * < 1.0 = Wide angle effect
   * > 1.0 = Telephoto effect
   * @default [1, 1]
   */
  focalLength?: THREE.Vector2;

  /**
   * Skew factor in radians
   * Corrects for non-square pixels (rare in modern cameras)
   * @default 0
   */
  skew?: number;

  /**
   * Chromatic dispersion strength
   * Controls how much distortion varies between wavelengths
   * 0.0 = No chromatic effect (pure lens distortion)
   * 0.05 = Subtle, realistic chromatic aberration
   * 0.2 = Noticeable color fringing
   * 0.5 = Strong stylized effect
   * @default 0.0
   */
  dispersion?: number;
}

/**
 * ChromaticLensDistortionEffect
 *
 * Physically accurate lens distortion with wavelength-dependent chromatic aberration.
 * Combines two effects into one by applying slightly different distortion to each
 * color channel, simulating real optical dispersion.
 *
 * Features:
 * - Wavelength-dependent distortion (blue refracts more than red)
 * - Full camera intrinsic matrix (distortion, principal point, focal length, skew)
 * - Efficient single-pass implementation
 * - Boundary masking for out-of-bounds pixels
 *
 * @example
 * ```typescript
 * // Subtle wide-angle lens with realistic chromatic aberration
 * const effect = new ChromaticLensDistortionEffect({
 *   distortion: new THREE.Vector2(-0.05, -0.05),  // Barrel distortion
 *   dispersion: 0.03,                              // Subtle chromatic fringing
 * });
 *
 * // Strong telephoto with pronounced chromatic effect
 * const effect = new ChromaticLensDistortionEffect({
 *   distortion: new THREE.Vector2(0.08, 0.08),    // Pincushion distortion
 *   focalLength: new THREE.Vector2(1.2, 1.2),     // Telephoto compression
 *   dispersion: 0.15,                              // Strong color separation
 * });
 * ```
 */
export class ChromaticLensDistortionEffect extends Effect {
  constructor(options: ChromaticLensDistortionEffectOptions = {}) {
    super('ChromaticLensDistortionEffect', fragmentShader, {
      blendFunction: options.blendFunction ?? BlendFunction.NORMAL,
      uniforms: new Map<string, THREE.Uniform>([
        ['distortion', new THREE.Uniform(options.distortion ?? new THREE.Vector2(0, 0))],
        ['principalPoint', new THREE.Uniform(options.principalPoint ?? new THREE.Vector2(0, 0))],
        ['focalLength', new THREE.Uniform(options.focalLength ?? new THREE.Vector2(1, 1))],
        ['skew', new THREE.Uniform(options.skew ?? 0)],
        ['dispersion', new THREE.Uniform(options.dispersion ?? 0.0)],
      ]),
    });
  }

  /**
   * Radial distortion coefficients
   * Negative = Barrel (wide angle), Positive = Pincushion (telephoto)
   */
  get distortion(): THREE.Vector2 {
    return this.uniforms.get('distortion')!.value;
  }

  set distortion(value: THREE.Vector2) {
    this.uniforms.get('distortion')!.value = value;
  }

  /**
   * Principal point offset (optical center)
   */
  get principalPoint(): THREE.Vector2 {
    return this.uniforms.get('principalPoint')!.value;
  }

  set principalPoint(value: THREE.Vector2) {
    this.uniforms.get('principalPoint')!.value = value;
  }

  /**
   * Focal length scale factor
   */
  get focalLength(): THREE.Vector2 {
    return this.uniforms.get('focalLength')!.value;
  }

  set focalLength(value: THREE.Vector2) {
    this.uniforms.get('focalLength')!.value = value;
  }

  /**
   * Skew factor (radians)
   */
  get skew(): number {
    return this.uniforms.get('skew')!.value;
  }

  set skew(value: number) {
    this.uniforms.get('skew')!.value = value;
  }

  /**
   * Chromatic dispersion strength
   * Controls wavelength-dependent distortion variation
   */
  get dispersion(): number {
    return this.uniforms.get('dispersion')!.value;
  }

  set dispersion(value: number) {
    // Clamp to reasonable range to prevent extreme artifacts
    this.uniforms.get('dispersion')!.value = Math.max(0, Math.min(1, value));
  }
}

/**
 * Type guard for ChromaticLensDistortionEffect
 */
export function isChromaticLensDistortionEffect(
  effect: unknown
): effect is ChromaticLensDistortionEffect {
  return (
    effect !== null &&
    typeof effect === 'object' &&
    'distortion' in effect &&
    'principalPoint' in effect &&
    'focalLength' in effect &&
    'skew' in effect &&
    'dispersion' in effect
  );
}
