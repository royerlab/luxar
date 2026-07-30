/**
 * Regression guard for the WebGPU "background isn't black" bug.
 *
 * `PostProcessingManager` MUST keep the renderer's `outputColorSpace`
 * at `THREE.LinearSRGBColorSpace`, not `THREE.SRGBColorSpace`. The
 * mega-shader is the canonical linear→sRGB converter for the
 * pipeline; if `outputColorSpace` is set to sRGB instead, Three's
 * `WebGPURenderer` runs an unconditional Output Color Transform pass
 * (gated on `currentColorSpace !== workingColorSpace` per the
 * `needsFrameBufferTarget` getter) that applies linearToSRGB a
 * SECOND time, double-encoding every pixel.
 *
 * Visible symptom: any non-black content (or an authored tinted
 * background) is rendered much brighter under WebGPU dispatch —
 * e.g. a `0x111111` background comes out as sRGB-of-sRGB ≈ 0x4A4A4A —
 * while WebGL stays correct (its chunk-injection path no-ops on our
 * GLSL3 `out vec4 fragColor` declaration, so even with the wrong
 * setting no doubling happens on that backend). Note: the default
 * background is now pure black (`0x000000`), for which the double
 * encode is a fixed point (sRGB(sRGB(0)) == 0), so an EMPTY scene no
 * longer shows the symptom — these wiring assertions are the guard.
 *
 * The fix lives in `post-processing-manager.ts`'s constructor; these
 * tests pin the wiring so a future "let's set this to sRGB to match
 * Three's default" edit gets caught before it ships.
 *
 * @module tests/unit/rendering/post-processing/output-color-space-wiring
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { PostProcessingManager } from '../../../../rendering/post-processing/post-processing-manager';
import type { Renderer, RendererCapabilities } from '../../../../rendering/renderer-capabilities';

function mockCaps(apiSurface: 'webgl2' | 'webgpu' = 'webgl2'): RendererCapabilities {
  return {
    apiSurface,
    framebufferYDown: apiSurface === 'webgpu',
    hdr: {
      p3Gamut: false,
      rec2020Gamut: false,
      hdr: false,
      deepColor: false,
      floatTextures: true,
      colorDepth: { red: 8, green: 8, blue: 8 },
      recommendedColorSpace: 'srgb',
    },
    maxTextureSize: 4096,
    maxMSAASamples: 4,
    pointSizeRange: [1, 1024],
    readBackbufferPixels: () => Promise.resolve({ pixels: new Uint8Array(), width: 0, height: 0 }),
  };
}

/**
 * Minimal renderer shape the PostProcessingManager constructor needs.
 * No GL context — we only test the wiring assignments, not render
 * dispatch (the latter is exercised by the post-processing-pipeline
 * E2E spec).
 */
function makeMockRenderer(): Renderer {
  return {
    outputColorSpace: THREE.SRGBColorSpace, // start at the WRONG value
    toneMapping: THREE.ACESFilmicToneMapping, // start at the WRONG value
    getPixelRatio: () => 1,
    // Other methods aren't reached during construction.
  } as unknown as Renderer;
}

describe('PostProcessingManager → outputColorSpace wiring', () => {
  it('pins renderer.outputColorSpace to LinearSRGB so no auto sRGB-encode pass runs', () => {
    const renderer = makeMockRenderer();
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();

    new PostProcessingManager(renderer, mockCaps('webgl2'), scene, camera, {
      width: 64,
      height: 64,
    });

    // The whole point of the fix: keep currentColorSpace ===
    // workingColorSpace (= LinearSRGB) so Three's
    // `needsFrameBufferTarget` getter returns false and no Output
    // Color Transform quad-pass kicks in to double-encode.
    expect(renderer.outputColorSpace).toBe(THREE.LinearSRGBColorSpace);
    expect(renderer.outputColorSpace).not.toBe(THREE.SRGBColorSpace);
  });

  it('pins renderer.toneMapping to NoToneMapping so Three does not auto-tone-map', () => {
    // The mega-shader owns tone mapping; the renderer must stay out
    // of that branch too. Three's `needsFrameBufferTarget` is OR-
    // gated on both color-space AND tone-mapping divergence, so any
    // non-NoToneMapping setting would also trigger the auto pass.
    const renderer = makeMockRenderer();
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();

    new PostProcessingManager(renderer, mockCaps('webgl2'), scene, camera, {
      width: 64,
      height: 64,
    });

    expect(renderer.toneMapping).toBe(THREE.NoToneMapping);
  });

  it('applies the same wiring on the WebGPU dispatch path', () => {
    // Symmetric guard: the fix is specifically about the WebGPU
    // backend (where the bug manifested), so cover that branch
    // explicitly even though the constructor code is shared.
    const renderer = makeMockRenderer();
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();

    new PostProcessingManager(renderer, mockCaps('webgpu'), scene, camera, {
      width: 64,
      height: 64,
    });

    expect(renderer.outputColorSpace).toBe(THREE.LinearSRGBColorSpace);
    expect(renderer.toneMapping).toBe(THREE.NoToneMapping);
  });
});

describe('PostProcessingManager → Three needsFrameBufferTarget gate', () => {
  it('Three.js `needsFrameBufferTarget` evaluates to false after construction', () => {
    // Direct check of the gate logic from
    // three/src/renderers/common/Renderer.js:2416-2423. If either
    // half of the OR flips to true, the auto Output Color Transform
    // pass starts running and our manual linearToSRGB gets doubled.
    const renderer = makeMockRenderer();
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera();

    new PostProcessingManager(renderer, mockCaps('webgpu'), scene, camera, {
      width: 64,
      height: 64,
    });

    const useToneMapping = renderer.toneMapping !== THREE.NoToneMapping;
    // Three's working color space is LinearSRGB; matching keeps the
    // gate at false.
    const useColorSpace = renderer.outputColorSpace !== THREE.ColorManagement.workingColorSpace;

    expect(useToneMapping || useColorSpace).toBe(false);
  });
});
