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
import {
  buildMaterial,
  type BuildMaterialConfig,
} from '../../../rendering/materials/_shared/material-builder';
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

  /** All six render-state fields, every one explicit. */
  type FullRenderState = Required<Omit<BuildMaterialConfig, 'uniforms' | 'defines'>>;

  const stateOf = (m: THREE.Material): FullRenderState => ({
    blending: m.blending,
    depthTest: m.depthTest,
    depthWrite: m.depthWrite,
    transparent: m.transparent,
    toneMapped: m.toneMapped,
    side: m.side,
  });

  /** The defaults `resolveRenderState` documents. */
  const RESOLVED_DEFAULTS: FullRenderState = {
    blending: THREE.NormalBlending,
    depthTest: true,
    depthWrite: true,
    transparent: false,
    toneMapped: false,
    side: THREE.FrontSide,
  };

  /** Every field differs from `RESOLVED_DEFAULTS`, so a DROPPED assignment shows up. */
  const ALL_NON_DEFAULT: FullRenderState = {
    blending: THREE.AdditiveBlending,
    depthTest: false,
    depthWrite: false,
    transparent: true,
    toneMapped: true,
    side: THREE.DoubleSide,
  };

  /**
   * The same six fields with the `depthTest`/`depthWrite` and
   * `transparent`/`toneMapped` pairs CROSSED. Every config anywhere
   * else in the repo carries each pair equal (`{}` → true/true and
   * false/false; the bloom and FXAA passes → false/false and
   * false/false), which leaves a copy-paste transposition inside the
   * six-line assignment block invisible to the whole suite. Crossing
   * them is the only thing that pins the field-to-field mapping.
   *
   * `NoBlending` (0) rather than a second additive-family preset: it is
   * distinct from the resolved default `NormalBlending` (1), from
   * `ALL_NON_DEFAULT`'s `AdditiveBlending` (2) and from this arm's
   * `side` (`BackSide`, 1), so a `blending ← side` half-swap still
   * fails. `SubtractiveBlending` / `MultiplyBlending` would be poor
   * canonical choices for a config test: three refuses both on EVERY
   * backend without `material.premultipliedAlpha`, which
   * `BuildMaterialConfig` cannot express.
   */
  const PAIRS_CROSSED: FullRenderState = {
    blending: THREE.NoBlending,
    depthTest: false,
    depthWrite: true,
    transparent: true,
    toneMapped: false,
    side: THREE.BackSide,
  };

  it('applies an explicitly-requested render state on BOTH backends', () => {
    // Pre-fix, the WebGPU branch passed only `config.uniforms` to the
    // TSL factory and dropped all six of these.
    const source = makeFakeTslSource();
    const glMat = buildMaterial(source, ALL_NON_DEFAULT, makeCaps('webgl2'));
    const gpuMat = buildMaterial(source, ALL_NON_DEFAULT, makeCaps('webgpu'));

    expect(stateOf(glMat)).toEqual(ALL_NON_DEFAULT);
    expect(stateOf(gpuMat)).toEqual(ALL_NON_DEFAULT);

    glMat.dispose();
    gpuMat.dispose();
  });

  it('resolves the SAME defaults on both backends when the config omits them', () => {
    // `toneMapped` is the one discriminating field here: it is the
    // only one of the six where the stand-in factory's raw default
    // (`MeshBasicMaterial.toneMapped === true`) differs from the
    // builder's resolved default, so this arm alone would still pass
    // with the other five WebGPU assignments deleted. The third arm
    // below is what pins those, via a factory that pre-sets every
    // field contrary to the config.
    const source = makeFakeTslSource();
    const glMat = buildMaterial(source, {}, makeCaps('webgl2'));
    const gpuMat = buildMaterial(source, {}, makeCaps('webgpu'));

    expect(stateOf(gpuMat)).toEqual(stateOf(glMat));
    expect(stateOf(glMat)).toEqual(RESOLVED_DEFAULTS);

    glMat.dispose();
    gpuMat.dispose();
  });

  it("config wins over a TSL factory's internally-set values, with the boolean pairs crossed", () => {
    // The stand-in pre-sets every field to the OPPOSITE of what the
    // config asks for, the way the real bloom / FXAA factories set
    // their own `toneMapped` / `depthTest` / … before returning. So
    // this arm catches a dropped assignment on the WebGPU branch as
    // well as a transposed one (see `PAIRS_CROSSED`).
    const source = makeFakeTslSource((m) => {
      m.blending = THREE.AdditiveBlending;
      m.depthTest = true;
      m.depthWrite = false;
      m.transparent = false;
      m.toneMapped = true;
      m.side = THREE.FrontSide;
    });
    const glMat = buildMaterial(source, PAIRS_CROSSED, makeCaps('webgl2'));
    const gpuMat = buildMaterial(source, PAIRS_CROSSED, makeCaps('webgpu'));

    expect(stateOf(glMat)).toEqual(PAIRS_CROSSED);
    expect(stateOf(gpuMat)).toEqual(PAIRS_CROSSED);

    glMat.dispose();
    gpuMat.dispose();
  });
});
