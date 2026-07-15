/** Unit tests for the FXAA fullscreen pass state contract. */

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { FxaaPass } from '../../../../rendering/post-processing/fxaa/pass';
import type { RendererCapabilities } from '../../../../rendering/renderer-capabilities';

function mockCaps(): RendererCapabilities {
  return {
    apiSurface: 'webgl2',
    framebufferYDown: false,
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

type FullscreenPassInternals = {
  mesh: THREE.Mesh;
};
type FxaaInternals = {
  material: THREE.ShaderMaterial;
  uniforms: {
    uInput: THREE.IUniform;
    uResolution: THREE.IUniform<THREE.Vector2>;
  };
  pass: FullscreenPassInternals;
};

describe('FxaaPass', () => {
  it('creates a toneMapped=false GLSL3 fullscreen pass with initial resolution', () => {
    const pass = new FxaaPass(320, 180, mockCaps());
    const internals = pass as unknown as FxaaInternals;

    expect(internals.material.toneMapped).toBe(false);
    expect(internals.material.glslVersion).toBe(THREE.GLSL3);
    expect(internals.uniforms.uResolution.value.toArray()).toEqual([320, 180]);
    expect(internals.pass.mesh.frustumCulled).toBe(false);
    expect(internals.pass.mesh.geometry.getAttribute('position').count).toBe(3);

    pass.dispose();
  });

  it('updates resolution without reallocating the material', () => {
    const pass = new FxaaPass(320, 180, mockCaps());
    const internals = pass as unknown as FxaaInternals;
    const materialBefore = internals.material;

    pass.setSize(640, 480);

    expect((pass as unknown as FxaaInternals).material).toBe(materialBefore);
    expect(internals.uniforms.uResolution.value.toArray()).toEqual([640, 480]);

    pass.dispose();
  });

  it('binds the input texture and renders to the current renderer target', () => {
    const pass = new FxaaPass(320, 180, mockCaps());
    const internals = pass as unknown as FxaaInternals;
    const texture = new THREE.Texture();
    const renderer = { render: vi.fn() } as unknown as THREE.WebGLRenderer;

    pass.render(renderer, texture);

    expect(internals.uniforms.uInput.value).toBe(texture);
    expect(renderer.render).toHaveBeenCalledTimes(1);

    texture.dispose();
    pass.dispose();
  });
});
