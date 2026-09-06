/**
 * Unit tests for `buildMaterial`.
 *
 * Focused on the backend-dispatch contract: pick `webgpu` factory
 * under `caps.apiSurface === 'webgpu'`, pick `webgl` source
 * otherwise, and throw — never silently fall back — when the active
 * backend's source is missing. The WebGPU-side throw is the
 * regression guard for the previous-fallback behaviour, which would
 * have silently rendered blank quads under WebGPURenderer.
 *
 * The second describe block pins the render-state contract: the
 * config's `blending` / `depthTest` / `depthWrite` / `transparent` /
 * `toneMapped` / `side` reach the material on BOTH backends, with the
 * same defaults, and override whatever a TSL factory set on itself.
 * The WebGPU branch used to drop all six (#2563).
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
      filterableFloatTextures: false,
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

describe('buildMaterial render-state contract', () => {
  /**
   * A `webgpu` factory standing in for a TSL one. `NodeMaterial` is in
   * the lazy `three/webgpu` cone (#1679), so these tests use a plain
   * material as the stand-in — `buildMaterial` only ever assigns the
   * six `THREE.Material` render-state fields on the returned object.
   * `overrides` lets a test pre-set contrary values, the way the real
   * bloom / FXAA factories set `toneMapped` / `depthTest` / … on
   * themselves before returning.
   */
  function makeFakeTslSource(overrides?: (m: THREE.Material) => void): ShaderSource {
    return {
      name: 'test-render-state',
      webgl: TRIVIAL_GLSL,
      webgpu: () => {
        const m = new THREE.MeshBasicMaterial();
        overrides?.(m);
        return m as unknown as ReturnType<NonNullable<ShaderSource['webgpu']>>;
      },
    };
  }

  const EXPLICIT_CONFIG = {
    blending: THREE.AdditiveBlending,
    depthTest: false,
    depthWrite: false,
    transparent: true,
    toneMapped: true,
    side: THREE.DoubleSide,
  } as const;

  it('applies an explicitly-requested render state on BOTH backends', () => {
    // Pre-fix, the WebGPU branch passed only `config.uniforms` to the
    // TSL factory and dropped all six of these.
    const source = makeFakeTslSource();
    const glMat = buildMaterial(source, EXPLICIT_CONFIG, makeCaps('webgl2'));
    const gpuMat = buildMaterial(source, EXPLICIT_CONFIG, makeCaps('webgpu'));

    for (const mat of [glMat, gpuMat]) {
      expect(mat.blending).toBe(THREE.AdditiveBlending);
      expect(mat.depthTest).toBe(false);
      expect(mat.depthWrite).toBe(false);
      expect(mat.transparent).toBe(true);
      expect(mat.toneMapped).toBe(true);
      expect(mat.side).toBe(THREE.DoubleSide);
    }

    glMat.dispose();
    gpuMat.dispose();
  });

  it('resolves the SAME defaults on both backends when the config omits them', () => {
    // The stand-in factory returns Three's raw defaults, which are NOT
    // the builder's — `MeshBasicMaterial.toneMapped` is `true` — so an
    // equal-and-documented assertion proves the builder wrote them
    // rather than inheriting them.
    const source = makeFakeTslSource();
    const glMat = buildMaterial(source, {}, makeCaps('webgl2'));
    const gpuMat = buildMaterial(source, {}, makeCaps('webgpu'));

    const stateOf = (m: THREE.Material) => ({
      blending: m.blending,
      depthTest: m.depthTest,
      depthWrite: m.depthWrite,
      transparent: m.transparent,
      toneMapped: m.toneMapped,
      side: m.side,
    });
    const documentedDefaults = {
      blending: THREE.NormalBlending,
      depthTest: true,
      depthWrite: true,
      transparent: false,
      toneMapped: false,
      side: THREE.FrontSide,
    };

    expect(stateOf(gpuMat)).toEqual(stateOf(glMat));
    expect(stateOf(glMat)).toEqual(documentedDefaults);

    glMat.dispose();
    gpuMat.dispose();
  });

  it("config wins over a TSL factory's internally-set render state", () => {
    const source = makeFakeTslSource((m) => {
      m.blending = THREE.NoBlending;
      m.side = THREE.BackSide;
      m.toneMapped = true;
    });
    const mat = buildMaterial(
      source,
      { blending: THREE.AdditiveBlending, side: THREE.FrontSide, toneMapped: false },
      makeCaps('webgpu')
    );

    expect(mat.blending).toBe(THREE.AdditiveBlending);
    expect(mat.side).toBe(THREE.FrontSide);
    expect(mat.toneMapped).toBe(false);

    mat.dispose();
  });
});
