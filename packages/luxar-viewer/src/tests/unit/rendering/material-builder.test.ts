/**
 * Unit tests for `buildMaterial`.
 *
 * Focused on the backend-dispatch contract: pick `webgpu` factory
 * under `caps.apiSurface === 'webgpu'`, pick `webgl` source
 * otherwise, and throw — never silently fall back — when the active
 * backend's source is missing. The WebGPU-side throw is the
 * regression guard for the previous-fallback behaviour, which would
 * have silently rendered blank quads under WebGPURenderer.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { buildMaterial } from '../../../rendering/materials/_shared/material-builder';
import type { ShaderSource } from '../../../rendering/materials/_shared/shader-source';
import type { RendererCapabilities } from '../../../rendering/renderer-capabilities';

function makeCaps(apiSurface: 'webgl2' | 'webgpu'): RendererCapabilities {
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

const TRIVIAL_GLSL = {
  vertex: 'void main() { gl_Position = vec4(0.0); }',
  fragment: 'out vec4 fragColor; void main() { fragColor = vec4(1.0); }',
};

describe('buildMaterial backend dispatch', () => {
  it('returns a ShaderMaterial under WebGL2 when source.webgl is present', () => {
    const source: ShaderSource = {
      name: 'test-glsl-only',
      webgl: TRIVIAL_GLSL,
    };
    const mat = buildMaterial(source, {}, makeCaps('webgl2'));
    expect(mat).toBeInstanceOf(THREE.ShaderMaterial);
    mat.dispose();
  });

  it('calls source.webgpu under WebGPU and does NOT touch source.webgl', () => {
    let webgpuCalled = false;
    const source: ShaderSource = {
      name: 'test-tsl',
      webgl: TRIVIAL_GLSL,
      webgpu: () => {
        webgpuCalled = true;
        const m = new THREE.ShaderMaterial();
        return m as unknown as ReturnType<NonNullable<ShaderSource['webgpu']>>;
      },
    };
    buildMaterial(source, {}, makeCaps('webgpu'));
    expect(webgpuCalled).toBe(true);
  });

  it('throws under WebGPU when source.webgpu is missing (no silent ShaderMaterial fallback)', () => {
    // Pre-fix behaviour: silently rendered ShaderMaterial under
    // WebGPURenderer, which can't dispatch ShaderMaterial — produced
    // blank quads.
    const source: ShaderSource = {
      name: 'test-glsl-only-but-webgpu',
      webgl: TRIVIAL_GLSL,
    };
    expect(() => buildMaterial(source, {}, makeCaps('webgpu'))).toThrowError(
      /no 'webgpu' TSL factory.*caps\.apiSurface='webgpu'/s
    );
  });

  it('throws under WebGL2 when source.webgl is missing', () => {
    const source: ShaderSource = {
      name: 'test-tsl-only',
      webgpu: () =>
        new THREE.ShaderMaterial() as unknown as ReturnType<NonNullable<ShaderSource['webgpu']>>,
    };
    expect(() => buildMaterial(source, {}, makeCaps('webgl2'))).toThrowError(
      /no 'webgl' reference.*caps\.apiSurface='webgl2'/s
    );
  });
});
