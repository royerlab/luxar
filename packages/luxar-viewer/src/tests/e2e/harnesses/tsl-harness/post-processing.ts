/**
 * Post-processing shader family for the TSL ↔ GLSL parity harness:
 * the const-rgb renderer-setup diagnostic, FXAA, the bloom threshold
 * pass, and the mega-shader variants (default/bloom/detector-noise/
 * vignette/ACES). 8 registry entries.
 *
 * @module tests/e2e/harnesses/tsl-harness/post-processing
 */

import * as THREE from 'three';
import { vec4 } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { FXAA_SOURCE } from '../../../../rendering/post-processing/fxaa/shaders';
import { BLOOM_THRESHOLD_SOURCE } from '../../../../rendering/post-processing/bloom/shaders';
import { MEGA_SOURCE } from '../../../../rendering/post-processing/mega/shader.glsl';
import { megaWebGPUFactory } from '../../../../rendering/post-processing/mega/shader.tsl';
import type { ShaderSource } from '../../../../rendering/materials/_shared/shader-source';
import type { RegistryEntry } from './types';

/**
 * Sized 8×8 test texture: gradient horizontally, ramped vertically,
 * with a single bright pixel near the centre to exercise the FXAA
 * edge-detection path. Deterministic across both backends.
 */
function buildTestTexture(): THREE.DataTexture {
  const w = 8;
  const h = 8;
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      data[i] = Math.floor((x / (w - 1)) * 255);
      data[i + 1] = Math.floor((y / (h - 1)) * 255);
      data[i + 2] = 128;
      data[i + 3] = 255;
    }
  }
  // High-contrast pixel for FXAA to bite on.
  const cx = 4;
  const cy = 4;
  const ci = (cy * w + cx) * 4;
  data[ci] = 255;
  data[ci + 1] = 255;
  data[ci + 2] = 255;

  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Trivial diagnostic shader: outputs a constant RGB. Used to verify
 * that the harness's two backends produce pixel-identical results
 * for the simplest possible fragment. If this fails, the divergence
 * is in the renderer-level setup (color space, output transform),
 * not in a per-shader port.
 */
const CONST_SHADER: ShaderSource = {
  name: 'const-rgb',
  webgl: {
    vertex: /* glsl */ `
      void main() {
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragment: /* glsl */ `
      precision highp float;
      out vec4 fragColor;
      void main() {
        fragColor = vec4(0.5, 0.25, 0.75, 1.0);
      }
    `,
  },
  webgpu: () => {
    const m = new NodeMaterial();
    m.fragmentNode = vec4(0.5, 0.25, 0.75, 1.0);
    m.toneMapped = false;
    m.depthTest = false;
    m.depthWrite = false;
    m.transparent = false;
    return m;
  },
};

export const POST_PROCESSING_SHADERS: Record<string, RegistryEntry> = {
  'const-rgb': {
    source: CONST_SHADER,
    buildUniforms: () => ({}),
  },
  fxaa: {
    source: FXAA_SOURCE,
    buildUniforms: () => ({
      uInput: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
    }),
  },
  'bloom-threshold': {
    source: BLOOM_THRESHOLD_SOURCE,
    buildUniforms: () => ({
      uInput: { value: buildTestTexture() },
      uTexelSize: { value: new THREE.Vector2(1 / 8, 1 / 8) },
      uThreshold: { value: 0.5 },
      uSmoothing: { value: 0.5 },
    }),
  },
  // Mega-shader: default configuration only (no bloom, no lens
  // distortion, no vignette, no detector noise, mode=Linear). The
  // tone-mapping mode is pinned to Linear (mode=1) because that's
  // the simplest path through THREE's toneMapping chunk and
  // matches TSL's linearToneMapping output.
  mega: {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      // THREE's tone-mapping chunk reads this; pin to 1.0 so the
      // GLSL3 ShaderMaterial doesn't double-multiply our exposure.
      toneMappingExposure: { value: 1.0 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '1' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, { toneMappingMode: 1 }) as unknown as THREE.Material,
  },
  // Mega + bloom enabled: validates the `USE_BLOOM` JS-side conditional
  // branch in the TSL factory matches the GLSL `#ifdef USE_BLOOM` path.
  'mega-bloom': {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      toneMappingExposure: { value: 1.0 },
      // A second texture for bloom — uniform contents differ from the
      // HDR scene so the test fails if the shader reads the wrong one.
      uBloomTexture: { value: buildTestTexture() },
      uBloomIntensity: { value: 0.5 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '1', USE_BLOOM: '' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, {
        toneMappingMode: 1,
        useBloom: true,
      }) as unknown as THREE.Material,
  },
  // Mega + detector noise: validates the Bob Jenkins hash +
  // Anscombe Poisson + clampedLogistic Gaussian port. The noise is
  // deterministic per (uv, time) so both backends should agree
  // bit-for-bit modulo float-precision rounding.
  'mega-detector-noise': {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      toneMappingExposure: { value: 1.0 },
      uTime: { value: 0.123 }, // fixed value → deterministic
      uReadoutSigma: { value: 0.02 },
      uPhotonGain: { value: 0.05 },
      uFpnSigma: { value: 0.01 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '1', USE_DETECTOR_NOISE: '' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, {
        toneMappingMode: 1,
        useDetectorNoise: true,
      }) as unknown as THREE.Material,
  },
  // Mega + vignette: validates the `USE_VIGNETTE` JS-side branch.
  'mega-vignette': {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      toneMappingExposure: { value: 1.0 },
      uVignetteDarkness: { value: 0.7 },
      uVignetteOffset: { value: 0.5 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '1', USE_VIGNETTE: '' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, {
        toneMappingMode: 1,
        useVignette: true,
      }) as unknown as THREE.Material,
  },
  // Mega + ACES tone-mapping (mode 4) — the PRODUCTION DEFAULT. The other
  // mega cases pin Linear (mode 1), so without this entry the ACES port
  // between shader.glsl.ts and shader.tsl.ts (the path users actually see)
  // is never parity-checked. ACES is non-linear, so this also guards the
  // RRT/ODT matrix + curve port, not just the mode switch.
  'mega-aces': {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      toneMappingExposure: { value: 1.0 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '4' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, { toneMappingMode: 4 }) as unknown as THREE.Material,
  },
};
