/**
 * Bloom pyramid TSL factories — NodeMaterial counterparts to the
 * three GLSL3 shaders in `shaders.ts`.
 *
 * Shared pattern: each is a fullscreen pass that reads `uInput`
 * with a small 4-tap kernel, writes a vec4 RGB output. Differences:
 *
 * - threshold: 2x2 box downsample + soft-knee threshold gate.
 * - downsample: 2x2 box downsample (no threshold).
 * - upsample: 4-tap tent filter + center sample, blended additively
 *   onto the destination mip via NodeMaterial.blending.
 *
 * @module rendering/post-processing/bloom/bloom.tsl
 */

import * as THREE from 'three';
import { Fn, uniform, uv, vec2, vec3, vec4, texture, smoothstep, dot } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { bindLiveTexture } from '../../materials/_shared/live-texture-tsl';

// Rec.709 relative luma weights (same as the GLSL `thresholdKnee`).
const LUMA_REC709 = vec3(0.2126, 0.7152, 0.0722);

/**
 * Threshold + 2x2 box downsample TSL factory.
 */
export function bloomThresholdWebGPUFactory(
  uniforms: Record<string, THREE.IUniform>
): NodeMaterial {
  // Live-bound texture node — see `bindLiveTexture` for why the swap
  // lives in `updateBefore`. The `fallback` identity is captured
  // outside the closure so the TextureNode's initial binding has a
  // stable reference while `uniforms.uInput.value` is still null.
  const fallback = new THREE.Texture();
  const uInput = bindLiveTexture(
    texture((uniforms.uInput.value as THREE.Texture | null) ?? fallback),
    uniforms.uInput,
    fallback
  );
  const uTexelSize = uniform(
    (uniforms.uTexelSize.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );
  // Primitive uniforms must use `.onUpdate(() => iuniform.value,
  // 'render')` so host setters (BloomChain.setThreshold, the
  // construction-time `uSmoothing` value) reach the GPU on the next
  // render. Without it, the TSL `uniform(number)` overload captures
  // the JS value at factory-build time and silently ignores
  // subsequent `iuniform.value = …` writes. The Vector2 input is
  // mutated in place (`.set(...)`), which the node reads back through
  // the same object reference, so it doesn't need the same wiring.
  const uThreshold = uniform((uniforms.uThreshold.value as number) ?? 0.0).onUpdate(
    () => (uniforms.uThreshold.value as number) ?? 0.0,
    'render'
  );
  const uSmoothing = uniform((uniforms.uSmoothing.value as number) ?? 0.0).onUpdate(
    () => (uniforms.uSmoothing.value as number) ?? 0.0,
    'render'
  );

  const fragmentNode = Fn(() => {
    const d = uTexelSize.mul(0.5);
    // Read the geometry's caps-aware `uv` attribute. The fullscreen-
    // triangle factory encodes WebGL2/WebGPU framebuffer-Y correction
    // there so we never have to branch on the renderer backend here.
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
  // Live-bound texture node — see `bindLiveTexture`. The downsample
  // chain reads a different mip per pass, so the TextureNode must
  // re-resolve `uniforms.uInput.value` each render rather than
  // capture the placeholder at build time.
  const fallback = new THREE.Texture();
  const uInput = bindLiveTexture(
    texture((uniforms.uInput.value as THREE.Texture | null) ?? fallback),
    uniforms.uInput,
    fallback
  );
  const uTexelSize = uniform(
    (uniforms.uTexelSize.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );

  const fragmentNode = Fn(() => {
    const d = uTexelSize.mul(0.5);
    // Read the geometry's caps-aware `uv` attribute. The fullscreen-
    // triangle factory encodes WebGL2/WebGPU framebuffer-Y correction
    // there so we never have to branch on the renderer backend here.
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
export function bloomUpsampleWebGPUFactory(uniforms: Record<string, THREE.IUniform>): NodeMaterial {
  // Live-bound texture node — see `bindLiveTexture`. The upsample
  // chain reads a different mip per pass.
  const fallback = new THREE.Texture();
  const uInput = bindLiveTexture(
    texture((uniforms.uInput.value as THREE.Texture | null) ?? fallback),
    uniforms.uInput,
    fallback
  );
  const uTexelSize = uniform(
    (uniforms.uTexelSize.value as THREE.Vector2) ?? new THREE.Vector2(1, 1)
  );
  // `.onUpdate('render')` so `BloomChain.setRadius` (which writes
  // `uniforms.uRadius.value`) reaches the GPU on subsequent renders.
  // See the threshold-pass comment above for the rationale.
  const uRadius = uniform((uniforms.uRadius.value as number) ?? 1.0).onUpdate(
    () => (uniforms.uRadius.value as number) ?? 1.0,
    'render'
  );

  const fragmentNode = Fn(() => {
    const r = uTexelSize.mul(uRadius);
    // Read the geometry's caps-aware `uv` attribute. The fullscreen-
    // triangle factory encodes WebGL2/WebGPU framebuffer-Y correction
    // there so we never have to branch on the renderer backend here.
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
