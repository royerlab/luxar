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
import { Fn, uniform, uv, vec2, vec3, vec4, texture, smoothstep, dot } from 'three/tsl';
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
  // Use the JS-value uniform() overload directly. Wrapping the value
  // in `float()` first builds a VarNode<'float', ConstNode> — i.e. a
  // *const* that gets inlined at compile time, so the uniform never
  // updates when the host writes uniforms.X.value. Tested with the
  // tsl-shader-parity harness (`bloom-threshold` parity test).
  const uThreshold = uniform((uniforms.uThreshold.value as number) ?? 0.0);
  const uSmoothing = uniform((uniforms.uSmoothing.value as number) ?? 0.0);

  const fragmentNode = Fn(() => {
    const d = uTexelSize.mul(0.5);
    // Read the geometry's `uv` attribute rather than `screenUV` —
    // screenUV uses WebGPU-flipped Y under `forceWebGL`, which would
    // sample the texture upside-down relative to the GLSL3 path
    // (where vUv = position.xy * 0.5 + 0.5 is bottom-up). The
    // fullscreen-pass mesh provides matching geometry uv.
    const coord = uv();
    const s0 = uInput.sample(coord.add(d.mul(vec2(-1.0, -1.0)))).rgb;
    const s1 = uInput.sample(coord.add(d.mul(vec2(1.0, -1.0)))).rgb;
    const s2 = uInput.sample(coord.add(d.mul(vec2(-1.0, 1.0)))).rgb;
    const s3 = uInput.sample(coord.add(d.mul(vec2(1.0, 1.0)))).rgb;
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
    // Read the geometry's `uv` attribute rather than `screenUV` —
    // screenUV uses WebGPU-flipped Y under `forceWebGL`, which would
    // sample the texture upside-down relative to the GLSL3 path
    // (where vUv = position.xy * 0.5 + 0.5 is bottom-up). The
    // fullscreen-pass mesh provides matching geometry uv.
    const coord = uv();
    const s0 = uInput.sample(coord.add(d.mul(vec2(-1.0, -1.0)))).rgb;
    const s1 = uInput.sample(coord.add(d.mul(vec2(1.0, -1.0)))).rgb;
    const s2 = uInput.sample(coord.add(d.mul(vec2(-1.0, 1.0)))).rgb;
    const s3 = uInput.sample(coord.add(d.mul(vec2(1.0, 1.0)))).rgb;
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
  const uRadius = uniform((uniforms.uRadius.value as number) ?? 1.0);

  const fragmentNode = Fn(() => {
    const r = uTexelSize.mul(uRadius);
    // Read the geometry's `uv` attribute rather than `screenUV` —
    // screenUV uses WebGPU-flipped Y under `forceWebGL`, which would
    // sample the texture upside-down relative to the GLSL3 path
    // (where vUv = position.xy * 0.5 + 0.5 is bottom-up). The
    // fullscreen-pass mesh provides matching geometry uv.
    const coord = uv();
    const s0 = uInput.sample(coord.add(r.mul(vec2(-1.0, 0.0)))).rgb;
    const s1 = uInput.sample(coord.add(r.mul(vec2(1.0, 0.0)))).rgb;
    const s2 = uInput.sample(coord.add(r.mul(vec2(0.0, -1.0)))).rgb;
    const s3 = uInput.sample(coord.add(r.mul(vec2(0.0, 1.0)))).rgb;
    const c = uInput.sample(coord).rgb;
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
