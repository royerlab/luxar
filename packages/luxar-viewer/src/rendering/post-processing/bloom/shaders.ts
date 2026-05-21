/**
 * Bloom pyramid shader sources.
 *
 * Three fullscreen passes share a unit-triangle vertex shader:
 *   - threshold  + 2× downsample → mip[0]
 *   - plain        2× downsample → mip[i+1]
 *   - 4-tap tent   upsample      → mip[i] (additive blend)
 *
 * Source-of-truth lives here so the eventual WebGPU/TSL port has one
 * place to add NodeMaterial factories without touching `bloom-chain.ts`.
 *
 * @module rendering/post-processing/bloom-shaders
 */

import type { ShaderSource } from '../../materials/_shared/shader-source';
import {
  bloomThresholdWebGPUFactory,
  bloomDownsampleWebGPUFactory,
  bloomUpsampleWebGPUFactory,
} from './bloom.tsl';

/**
 * Shared fullscreen-triangle vertex shader for all three bloom passes.
 * The host supplies a unit triangle in NDC via the geometry's
 * `position` attribute, which THREE's ShaderMaterial auto-declares —
 * do NOT redeclare it here.
 */
export const BLOOM_VERTEX_SHADER = /* glsl */ `
  out vec2 vUv;
  void main() {
    // Caps-aware uv attribute encodes WebGL2/WebGPU Y-orientation
    // correction. See createFullscreenTriangleGeometry.
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

/**
 * Threshold + 2× box downsample. Extracts bright pixels above the
 * threshold with a smooth knee to avoid banding at the cutoff.
 *
 * Uses Rec.709 relative luminance for the brightness test. An
 * earlier version used max(r,g,b), which overstated saturated-channel
 * pixels — e.g. pure red would bloom even at low intensity.
 */
export const BLOOM_THRESHOLD_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uInput;
  uniform vec2 uTexelSize;
  uniform float uThreshold;
  uniform float uSmoothing;

  // Soft-knee: smoothstep around the threshold on relative luma, then
  // multiply by the source color (preserves chroma; only the brightness
  // gate is luma-based).
  vec3 thresholdKnee(vec3 color) {
    float l = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float soft = smoothstep(uThreshold, uThreshold + uSmoothing, l);
    return color * soft;
  }

  void main() {
    // 2x2 box downsample
    vec2 d = uTexelSize * 0.5;
    vec3 s0 = texture(uInput, vUv + d * vec2(-1.0, -1.0)).rgb;
    vec3 s1 = texture(uInput, vUv + d * vec2( 1.0, -1.0)).rgb;
    vec3 s2 = texture(uInput, vUv + d * vec2(-1.0,  1.0)).rgb;
    vec3 s3 = texture(uInput, vUv + d * vec2( 1.0,  1.0)).rgb;
    vec3 avg = (s0 + s1 + s2 + s3) * 0.25;
    fragColor = vec4(thresholdKnee(avg), 1.0);
  }
`;

/** Plain 2× box downsample (no threshold). */
export const BLOOM_DOWNSAMPLE_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uInput;
  uniform vec2 uTexelSize;

  void main() {
    vec2 d = uTexelSize * 0.5;
    vec3 s0 = texture(uInput, vUv + d * vec2(-1.0, -1.0)).rgb;
    vec3 s1 = texture(uInput, vUv + d * vec2( 1.0, -1.0)).rgb;
    vec3 s2 = texture(uInput, vUv + d * vec2(-1.0,  1.0)).rgb;
    vec3 s3 = texture(uInput, vUv + d * vec2( 1.0,  1.0)).rgb;
    fragColor = vec4((s0 + s1 + s2 + s3) * 0.25, 1.0);
  }
`;

/**
 * 4-tap tent upsample. Samples the smaller mip with a unit-radius
 * tent and additively blends into the larger mip (achieved by
 * blending with `THREE.AdditiveBlending` on the material).
 */
export const BLOOM_UPSAMPLE_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  in vec2 vUv;
  out vec4 fragColor;

  uniform sampler2D uInput;
  uniform vec2 uTexelSize;
  uniform float uRadius;

  void main() {
    vec2 r = uTexelSize * uRadius;
    vec3 s0 = texture(uInput, vUv + r * vec2(-1.0,  0.0)).rgb;
    vec3 s1 = texture(uInput, vUv + r * vec2( 1.0,  0.0)).rgb;
    vec3 s2 = texture(uInput, vUv + r * vec2( 0.0, -1.0)).rgb;
    vec3 s3 = texture(uInput, vUv + r * vec2( 0.0,  1.0)).rgb;
    vec3 c  = texture(uInput, vUv).rgb;
    fragColor = vec4(c * 0.5 + (s0 + s1 + s2 + s3) * 0.125, 1.0);
  }
`;

export const BLOOM_THRESHOLD_SOURCE: ShaderSource = {
  name: 'bloom-threshold',
  webgl: { vertex: BLOOM_VERTEX_SHADER, fragment: BLOOM_THRESHOLD_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) =>
    bloomThresholdWebGPUFactory(uniforms as Record<string, import('three').IUniform>),
};

export const BLOOM_DOWNSAMPLE_SOURCE: ShaderSource = {
  name: 'bloom-downsample',
  webgl: { vertex: BLOOM_VERTEX_SHADER, fragment: BLOOM_DOWNSAMPLE_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) =>
    bloomDownsampleWebGPUFactory(uniforms as Record<string, import('three').IUniform>),
};

export const BLOOM_UPSAMPLE_SOURCE: ShaderSource = {
  name: 'bloom-upsample',
  webgl: { vertex: BLOOM_VERTEX_SHADER, fragment: BLOOM_UPSAMPLE_FRAGMENT_SHADER },
  webgpu: (uniforms: Record<string, unknown>) =>
    bloomUpsampleWebGPUFactory(uniforms as Record<string, import('three').IUniform>),
};
