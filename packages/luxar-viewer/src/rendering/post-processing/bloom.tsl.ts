/**
 * Bloom pyramid TSL factories — NodeMaterial counterparts to the
 * three GLSL3 shaders in `bloom-shaders.ts`.
 *
 * Shared pattern: each is a fullscreen pass that reads `uInput`
 * with a small 4-tap kernel, writes a vec4 RGB output. Differences:
 *
 * - threshold: 2x2 box downsample + soft-knee threshold gate.
 * - downsample: 2x2 box downsample (no threshold).
 * - upsample: 4-tap tent filter + center sample, blended additively
 *   onto the destination mip via NodeMaterial.blending.
 *
 * @module rendering/post-processing/bloom.tsl
 */

import * as THREE from 'three';
import { Fn, uniform, vec2, vec3, vec4, texture, screenUV, float, smoothstep, dot } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';

// Rec.709 relative luma weights (same as the GLSL `thresholdKnee`).
const LUMA_REC709 = vec3(0.2126, 0.7152, 0.0722);

/**
 * Threshold + 2x2 box downsample TSL factory.
 */
export function bloomThresholdWebGPUFactory(
  uniforms: Record<string, THREE.IUniform>
): NodeMaterial {
  const uInput = texture((uniforms.uInput.value as THREE.Texture | null) ?? new THREE.Texture());
  const uTexelSize = uniform(
    (uniforms.uTexelSize.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );
  const uThreshold = uniform(float((uniforms.uThreshold.value as number) ?? 0.0));
  const uSmoothing = uniform(float((uniforms.uSmoothing.value as number) ?? 0.0));

  const fragmentNode = Fn(() => {
    const d = uTexelSize.mul(0.5);
    const uv = screenUV;
    const s0 = uInput.sample(uv.add(d.mul(vec2(-1.0, -1.0)))).rgb;
    const s1 = uInput.sample(uv.add(d.mul(vec2(1.0, -1.0)))).rgb;
    const s2 = uInput.sample(uv.add(d.mul(vec2(-1.0, 1.0)))).rgb;
    const s3 = uInput.sample(uv.add(d.mul(vec2(1.0, 1.0)))).rgb;
    const avg = s0.add(s1).add(s2).add(s3).mul(0.25);

    // Soft-knee threshold: smoothstep around uThreshold on luma,
    // then multiply by color. Preserves chroma; only the brightness
    // gate is luma-based.
    const l = dot(avg, LUMA_REC709);
    const soft = smoothstep(uThreshold, uThreshold.add(uSmoothing), l);
    return vec4(avg.mul(soft), 1.0);
  });

  const m = new NodeMaterial();
  m.fragmentNode = fragmentNode();
  m.toneMapped = false;
  m.depthTest = false;
  m.depthWrite = false;
  m.transparent = false;
  return m;
}

/**
 * Plain 2x2 box downsample TSL factory (no threshold).
 */
export function bloomDownsampleWebGPUFactory(
  uniforms: Record<string, THREE.IUniform>
): NodeMaterial {
  const uInput = texture((uniforms.uInput.value as THREE.Texture | null) ?? new THREE.Texture());
  const uTexelSize = uniform(
    (uniforms.uTexelSize.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );

  const fragmentNode = Fn(() => {
    const d = uTexelSize.mul(0.5);
    const uv = screenUV;
    const s0 = uInput.sample(uv.add(d.mul(vec2(-1.0, -1.0)))).rgb;
    const s1 = uInput.sample(uv.add(d.mul(vec2(1.0, -1.0)))).rgb;
    const s2 = uInput.sample(uv.add(d.mul(vec2(-1.0, 1.0)))).rgb;
    const s3 = uInput.sample(uv.add(d.mul(vec2(1.0, 1.0)))).rgb;
    return vec4(s0.add(s1).add(s2).add(s3).mul(0.25), 1.0);
  });

  const m = new NodeMaterial();
  m.fragmentNode = fragmentNode();
  m.toneMapped = false;
  m.depthTest = false;
  m.depthWrite = false;
  m.transparent = false;
  return m;
}

/**
 * 4-tap tent upsample TSL factory. Blending mode (`AdditiveBlending`
 * for accumulation onto the previous mip) is set by the host on the
 * returned material.
 */
export function bloomUpsampleWebGPUFactory(
  uniforms: Record<string, THREE.IUniform>
): NodeMaterial {
  const uInput = texture((uniforms.uInput.value as THREE.Texture | null) ?? new THREE.Texture());
  const uTexelSize = uniform(
    (uniforms.uTexelSize.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );
  const uRadius = uniform(float((uniforms.uRadius.value as number) ?? 1.0));

  const fragmentNode = Fn(() => {
    const r = uTexelSize.mul(uRadius);
    const uv = screenUV;
    const s0 = uInput.sample(uv.add(r.mul(vec2(-1.0, 0.0)))).rgb;
    const s1 = uInput.sample(uv.add(r.mul(vec2(1.0, 0.0)))).rgb;
    const s2 = uInput.sample(uv.add(r.mul(vec2(0.0, -1.0)))).rgb;
    const s3 = uInput.sample(uv.add(r.mul(vec2(0.0, 1.0)))).rgb;
    const c = uInput.sample(uv).rgb;
    // Center × 0.5 + 4 taps × 0.125 = tent filter normalized to 1.0
    return vec4(c.mul(0.5).add(s0.add(s1).add(s2).add(s3).mul(0.125)), 1.0);
  });

  const m = new NodeMaterial();
  m.fragmentNode = fragmentNode();
  m.toneMapped = false;
  m.depthTest = false;
  m.depthWrite = false;
  m.transparent = false;
  // Blending set by host (AdditiveBlending on the upsample material
  // so subsequent renders accumulate onto the larger mip).
  return m;
}
