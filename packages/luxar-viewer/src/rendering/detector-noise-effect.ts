/**
 * DetectorNoiseEffect - Physics-based detector noise simulation
 *
 * Simulates realistic detector noise using three components:
 * 1. Shot noise (Poisson) - Signal-dependent noise from photon statistics
 * 2. Readout noise (Gaussian, temporal) - Signal-independent electronic noise, varies per frame
 * 3. Fixed Pattern Noise (Gaussian, static) - Per-pixel offset that stays constant across frames
 *
 * Combined model:
 *   I_observed = Poisson(I_true / gain) × gain + Gaussian_temporal(0, σ_read²) + FPN(pixel)
 *
 * Uses efficient GPU-friendly approximations:
 * - Bob Jenkins hash function for deterministic randomness
 * - Clamped logistic distribution for Gaussian approximation (faster than Box-Muller)
 * - Anscombe transform for Poisson approximation
 */

import { BlendFunction, Effect, EffectAttribute } from 'postprocessing';
import * as THREE from 'three';

/**
 * Fragment shader for physics-based detector noise
 */
const fragmentShader = /* glsl */ `
  uniform float time;
  uniform float readoutSigma;
  uniform float photonGain;
  uniform float fpnSigma;

  // ============================================
  // Bob Jenkins hash function
  // Fast, high-quality deterministic PRNG
  // ============================================

  uint bobJenkinsHash(uint a) {
    a = (a + 0x7ed55d16u) + (a << 12u);
    a = (a ^ 0xc761c23cu) ^ (a >> 19u);
    a = (a + 0x165667b1u) + (a << 5u);
    a = (a + 0xd3a2646cu) ^ (a << 9u);
    a = (a + 0xfd7046c5u) + (a << 3u);
    a = (a ^ 0xb55a4f09u) ^ (a >> 16u);
    return a;
  }

  // Hash functions for 2D, 3D inputs returning uint
  uint rnguint2(vec2 x) {
    uint a = bobJenkinsHash(floatBitsToUint(x.x));
    uint b = bobJenkinsHash(floatBitsToUint(x.y));
    return bobJenkinsHash(a ^ b);
  }

  uint rnguint3(vec3 x) {
    uint a = rnguint2(x.xy);
    uint b = bobJenkinsHash(floatBitsToUint(x.z));
    return bobJenkinsHash(a ^ b);
  }

  // Convert uint hash to float in [0, 1)
  float rngfloat2(vec2 x) {
    return float(rnguint2(x)) / 4294967296.0;
  }

  float rngfloat3(vec3 x) {
    return float(rnguint3(x)) / 4294967296.0;
  }

  // ============================================
  // Clamped logistic distribution
  // Approximates Gaussian (faster than Box-Muller)
  // logistic(x) = log(x / (1-x)) clamped to [-L, L]
  // ============================================

  const float LOGISTIC_CLAMP = 4.0; // ~3.5 sigma equivalent
  const float LOGISTIC_SCALE = 0.5513; // Scale factor for unit variance

  float clampedLogistic(float u) {
    // Avoid singularities at 0 and 1
    float f = clamp(u, 0.0001, 0.9999);
    // Logistic quantile function: log(f / (1-f))
    float logit = log(f / (1.0 - f));
    // Clamp and scale for approximately unit variance
    return clamp(logit, -LOGISTIC_CLAMP, LOGISTIC_CLAMP) * LOGISTIC_SCALE;
  }

  // Generate 3 independent Gaussian-like values from a 3D seed (temporal noise)
  vec3 normal3_temporal(vec3 seed) {
    return vec3(
      clampedLogistic(rngfloat3(seed)),
      clampedLogistic(rngfloat3(seed + vec3(13.37, 7.31, 19.93))),
      clampedLogistic(rngfloat3(seed + vec3(31.17, 41.23, 53.59)))
    );
  }

  // Generate 3 independent Gaussian-like values from a 2D seed (fixed pattern noise)
  // Uses only UV coordinates, so pattern is constant across frames
  vec3 normal3_fixed(vec2 seed) {
    return vec3(
      clampedLogistic(rngfloat2(seed)),
      clampedLogistic(rngfloat2(seed + vec2(13.37, 7.31))),
      clampedLogistic(rngfloat2(seed + vec2(31.17, 41.23)))
    );
  }

  // ============================================
  // Anscombe transform for Poisson approximation
  // Variance-stabilizing transformation:
  // Y = 2 * sqrt(X + 3/8) transforms Poisson(lambda) to ~N(2*sqrt(lambda), 1)
  // ============================================

  float anscombeForward(float x) {
    return 2.0 * sqrt(max(x + 0.375, 0.0));
  }

  float anscombeInverse(float y) {
    // Inverse: X = (Y/2)^2 - 3/8
    float x = (y * 0.5) * (y * 0.5) - 0.375;
    return max(x, 0.0);
  }

  // Poisson-like noise using Anscombe transform
  vec3 poissonNoise(vec3 seed, vec3 lambda) {
    // Forward Anscombe: transform to approximately Gaussian
    vec3 y = vec3(
      anscombeForward(lambda.r),
      anscombeForward(lambda.g),
      anscombeForward(lambda.b)
    );

    // Add unit Gaussian noise
    vec3 noise = normal3_temporal(seed);
    y += noise;

    // Inverse Anscombe: transform back to Poisson-like
    return vec3(
      anscombeInverse(y.r),
      anscombeInverse(y.g),
      anscombeInverse(y.b)
    );
  }

  // ============================================
  // Main effect function
  // ============================================

  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
    // Seeds for random number generation
    vec3 temporalSeed = vec3(uv * 1000.0, time);  // Changes each frame
    vec2 fixedSeed = uv * 1000.0;                  // Constant per pixel

    // Original intensity (assuming linear color space)
    vec3 intensity = inputColor.rgb;

    // === 1. Shot noise (Poisson) ===
    // Scale by photon gain to convert to "photon counts"
    // Higher gain = more visible shot noise (fewer effective photons)
    vec3 photonCount = intensity / max(photonGain, 0.0001);
    vec3 noisyPhotons = poissonNoise(temporalSeed, photonCount);
    vec3 afterShot = noisyPhotons * photonGain;

    // === 2. Readout noise (Gaussian, temporal) ===
    // Signal-independent electronic noise, varies each frame
    vec3 readoutNoise = normal3_temporal(temporalSeed + vec3(100.0)) * readoutSigma;

    // === 3. Fixed Pattern Noise (Gaussian, static) ===
    // Per-pixel offset that stays constant across frames
    // Models dark current non-uniformity, gain variations, etc.
    vec3 fpn = normal3_fixed(fixedSeed) * fpnSigma;

    // Combine all noise components
    vec3 finalColor = afterShot + readoutNoise + fpn;

    // Ensure non-negative (physical constraint)
    finalColor = max(finalColor, vec3(0.0));

    outputColor = vec4(finalColor, inputColor.a);
  }
`;

/**
 * Options for configuring the DetectorNoiseEffect
 */
export interface DetectorNoiseEffectOptions {
  /**
   * Blend function for compositing the effect
   * Default: BlendFunction.NORMAL (replaces original)
   */
  blendFunction?: BlendFunction;

  /**
   * Standard deviation of readout noise (signal-independent Gaussian, temporal)
   * This noise varies frame-to-frame.
   * Range: 0.0 - 0.1 typical
   * Default: 0.01
   */
  readoutSigma?: number;

  /**
   * Photon gain - converts intensity to effective photon count
   * Higher values = more visible shot noise (fewer effective photons)
   * Range: 0.0001 - 0.1 typical
   * Default: 0.01
   */
  photonGain?: number;

  /**
   * Standard deviation of Fixed Pattern Noise (static per-pixel offset)
   * Models dark current non-uniformity, pixel gain variations, etc.
   * This noise is constant across frames - each pixel has its own fixed offset.
   * Range: 0.0 - 0.05 typical
   * Default: 0.005
   */
  fpnSigma?: number;
}

/**
 * DetectorNoiseEffect - Physics-based camera/detector noise simulation
 *
 * This effect simulates realistic noise patterns found in scientific imaging:
 * - Shot noise (Poisson): Signal-dependent noise from photon statistics
 * - Readout noise (Gaussian): Signal-independent temporal noise from electronics
 * - Fixed Pattern Noise: Static per-pixel offsets from detector non-uniformities
 *
 * @example
 * ```typescript
 * const noiseEffect = new DetectorNoiseEffect({
 *   readoutSigma: 0.02,  // Moderate readout noise
 *   photonGain: 0.005,   // Low-light conditions (visible shot noise)
 *   fpnSigma: 0.01,      // Some fixed pattern noise
 * });
 * ```
 */
export class DetectorNoiseEffect extends Effect {
  private _time: number = 0;

  constructor(options: DetectorNoiseEffectOptions = {}) {
    super('DetectorNoiseEffect', fragmentShader, {
      blendFunction: options.blendFunction ?? BlendFunction.NORMAL,
      attributes: EffectAttribute.NONE,
      uniforms: new Map<string, THREE.Uniform>([
        ['time', new THREE.Uniform(0.0)],
        ['readoutSigma', new THREE.Uniform(options.readoutSigma ?? 0.01)],
        ['photonGain', new THREE.Uniform(options.photonGain ?? 0.01)],
        ['fpnSigma', new THREE.Uniform(options.fpnSigma ?? 0.005)],
      ]),
    });
  }

  /**
   * Get/set readout noise sigma (Gaussian noise standard deviation, temporal)
   */
  get readoutSigma(): number {
    return this.uniforms.get('readoutSigma')!.value;
  }

  set readoutSigma(value: number) {
    this.uniforms.get('readoutSigma')!.value = Math.max(0, value);
  }

  /**
   * Get/set photon gain (controls shot noise visibility)
   */
  get photonGain(): number {
    return this.uniforms.get('photonGain')!.value;
  }

  set photonGain(value: number) {
    this.uniforms.get('photonGain')!.value = Math.max(0.0001, value);
  }

  /**
   * Get/set Fixed Pattern Noise sigma (static per-pixel offset)
   */
  get fpnSigma(): number {
    return this.uniforms.get('fpnSigma')!.value;
  }

  set fpnSigma(value: number) {
    this.uniforms.get('fpnSigma')!.value = Math.max(0, value);
  }

  /**
   * Update the effect (called each frame by the EffectPass)
   * Always advances time for temporal noise components.
   */
  update(
    _renderer: THREE.WebGLRenderer,
    _inputBuffer: THREE.WebGLRenderTarget,
    deltaTime?: number
  ): void {
    if (deltaTime !== undefined) {
      this._time += deltaTime;
      this.uniforms.get('time')!.value = this._time;
    }
  }
}

/**
 * Type guard for DetectorNoiseEffect
 */
export function isDetectorNoiseEffect(effect: unknown): effect is DetectorNoiseEffect {
  return (
    effect !== null &&
    typeof effect === 'object' &&
    'readoutSigma' in effect &&
    'photonGain' in effect &&
    'fpnSigma' in effect
  );
}
