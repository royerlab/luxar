/**
 * Robust Vignette Effect
 *
 * A drop-in replacement for pmndrs VignetteEffect that handles additive blending artifacts.
 * Identical parameters and visual output, but won't produce artifacts when
 * the input has accumulated alpha values from additive blending.
 *
 * The issue: Additive blending (THREE.AdditiveBlending) accumulates alpha values.
 * With many overlapping lines/points, alpha can exceed 1.0 and even overflow
 * to Infinity in Float16 framebuffers. When this problematic alpha is preserved
 * through effects, it causes rendering artifacts.
 *
 * The fix: Force alpha to 1.0 in output. Since vignette is a screen-space
 * post-processing effect, the output should always be fully opaque.
 */

import { Effect, BlendFunction } from 'postprocessing';
import * as THREE from 'three';

const fragmentShader = /* glsl */ `
  uniform float darkness;
  uniform float offset;

  void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {
    const vec2 center = vec2(0.5);
    vec3 color = inputColor.rgb;

    // Calculate vignette darkening factor
    // Use squared distance from center for smooth falloff
    vec2 coord = (uv - center) / offset; // offset controls size (smaller = tighter vignette)
    float dist2 = dot(coord, coord);

    // Create smooth darkening that goes to 0 at edges
    // darkness controls intensity, offset controls size
    float vignetteFactor = 1.0 - smoothstep(0.0, 1.5, dist2) * darkness;

    // Apply multiplicative darkening (preserves color ratios, doesn't shift toward gray)
    color *= vignetteFactor;

    // FIX: Force alpha to 1.0
    // The input alpha from HDR additive blending can contain problematic values
    // (NaN, Infinity, or values outside [0,1]) that cause artifacts when preserved.
    // Since vignette is a screen-space effect, alpha should always be fully opaque.
    outputColor = vec4(color, 1.0);
  }
`;

/**
 * Options for configuring the RobustVignetteEffect
 */
export interface RobustVignetteEffectOptions {
  /**
   * The blend function of this effect.
   * @default BlendFunction.NORMAL
   */
  blendFunction?: BlendFunction;

  /**
   * The vignette darkness. Higher values = darker edges.
   * @default 0.5
   */
  darkness?: number;

  /**
   * The vignette offset. Controls where the darkening starts.
   * @default 0.5
   */
  offset?: number;
}

/**
 * Robust Vignette Effect
 *
 * Identical to pmndrs VignetteEffect but handles HDR overflow gracefully.
 * Use this instead of VignetteEffect when working with high HDR values
 * or additive blending that could produce Infinity values.
 */
export class RobustVignetteEffect extends Effect {
  /**
   * Constructs a new Robust Vignette effect.
   *
   * @param options - The options for this effect
   */
  constructor({
    blendFunction = BlendFunction.NORMAL,
    darkness = 0.5,
    offset = 0.5,
  }: RobustVignetteEffectOptions = {}) {
    super('RobustVignetteEffect', fragmentShader, {
      blendFunction,
      uniforms: new Map<string, THREE.Uniform>([
        ['darkness', new THREE.Uniform(darkness)],
        ['offset', new THREE.Uniform(offset)],
      ]),
    });
  }

  /**
   * The vignette darkness.
   */
  get darkness(): number {
    return this.uniforms.get('darkness')!.value;
  }

  set darkness(value: number) {
    this.uniforms.get('darkness')!.value = value;
  }

  /**
   * The vignette offset.
   */
  get offset(): number {
    return this.uniforms.get('offset')!.value;
  }

  set offset(value: number) {
    this.uniforms.get('offset')!.value = value;
  }
}

/**
 * Type guard for RobustVignetteEffect
 */
export function isRobustVignetteEffect(effect: unknown): effect is RobustVignetteEffect {
  return effect instanceof RobustVignetteEffect;
}
