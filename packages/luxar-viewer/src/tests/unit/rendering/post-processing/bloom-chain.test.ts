/**
 * Unit tests for BloomChain resource/state management.
 *
 * Rendering itself is covered by browser E2E; these tests pin the
 * no-context contracts that can regress in ordinary TypeScript changes.
 */

import { beforeAll, describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { BloomChain } from '../../../../rendering/post-processing/bloom/chain';
import { loadTslMaterials } from '../../../../rendering/tsl/load';
import type { RendererCapabilities } from '../../../../rendering/renderer-capabilities';

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

describe('BloomChain', () => {
  it('allocates a half-resolution output texture with clamped levels', () => {
    const chain = new BloomChain({ width: 128, height: 64, levels: 4, caps: mockCaps() });

    expect(chain.outputSize).toEqual({ width: 64, height: 32 });
    expect(chain.outputTexture).toBeInstanceOf(THREE.Texture);
    expect(chain.outputTexture.type).toBe(THREE.HalfFloatType);

    const mips = (chain as unknown as { mips: Array<{ width: number; height: number }> }).mips;
    expect(mips.map((m) => [m.width, m.height])).toEqual([
      [64, 32],
      [32, 16],
      [16, 8],
      [8, 4],
    ]);

    chain.dispose();
  });

  it('never allocates useless sub-4px mips but keeps one output mip', () => {
    const chain = new BloomChain({ width: 4, height: 2, levels: 12, caps: mockCaps() });

    expect(chain.outputSize).toEqual({ width: 2, height: 1 });
    const mips = (chain as unknown as { mips: Array<unknown> }).mips;
    expect(mips).toHaveLength(1);

    chain.dispose();
  });

  it('updates threshold and radius uniforms', () => {
    const chain = new BloomChain({
      width: 64,
      height: 64,
      threshold: 0.2,
      radius: 1.0,
      caps: mockCaps(),
    });
    const priv = chain as unknown as {
      thresholdUniforms: { uThreshold: THREE.IUniform<number> };
      upsampleUniforms: { uRadius: THREE.IUniform<number> };
    };

    chain.setThreshold(0.35);
    chain.setRadius(2.5);

    expect(priv.thresholdUniforms.uThreshold.value).toBe(0.35);
    expect(priv.upsampleUniforms.uRadius.value).toBe(2.5);

    chain.dispose();
  });

  it('resizes and rebuilds the output texture identity', () => {
    const chain = new BloomChain({ width: 64, height: 64, levels: 3, caps: mockCaps() });
    const before = chain.outputTexture;

    chain.setSize(128, 96);

    expect(chain.outputSize).toEqual({ width: 64, height: 48 });
    expect(chain.outputTexture).not.toBe(before);

    chain.dispose();
  });
});

describe('BloomChain pass blending — WebGL2 / WebGPU parity', () => {
  // The upsample pass accumulates mip[i+1] into mip[i]; the additive
  // blend IS the accumulation, so a NormalBlending upsample overwrites
  // its destination and the pyramid collapses to a flat dim wash.
  // `buildMaterial` used to drop `config.blending` on the WebGPU
  // branch, leaving the TSL material on NormalBlending / transparent =
  // false — which three's `WebGPUPipelineUtils` reads as "no blend
  // state at all" (#2563). The threshold and downsample passes must
  // stay NormalBlending: they write a fresh mip each time.
  //
  // The WebGPU arm builds real TSL NodeMaterials, so the lazily-loaded
  // registry has to be installed first — `requireTslMaterials()` in
  // `bloom/shaders.ts` throws otherwise. That is also why this block
  // must stay ABOVE the `bloom TSL factories` one below: that block
  // calls `vi.resetModules()` and mocks `three/tsl`, and this one needs
  // the real module through `loadTslMaterials()`. Do not reorder them.
  beforeAll(async () => {
    await loadTslMaterials();
  });

  interface BloomMaterials {
    thresholdMat: THREE.Material;
    downsampleMat: THREE.Material;
    upsampleMat: THREE.Material;
  }

  function expectBloomBlending(caps: RendererCapabilities): void {
    const chain = new BloomChain({ width: 64, height: 64, levels: 3, caps });
    const mats = chain as unknown as BloomMaterials;

    expect(mats.upsampleMat.blending).toBe(THREE.AdditiveBlending);
    expect(mats.thresholdMat.blending).toBe(THREE.NormalBlending);
    expect(mats.downsampleMat.blending).toBe(THREE.NormalBlending);

    for (const mat of [mats.thresholdMat, mats.downsampleMat, mats.upsampleMat]) {
      expect(mat.depthTest).toBe(false);
      expect(mat.depthWrite).toBe(false);
      expect(mat.toneMapped).toBe(false);
      // `transparent` is the other half of three's predicate: a
      // NormalBlending pass only gets overwrite semantics while it is
      // false, so flipping it would start alpha-blending the threshold
      // and downsample passes against their previous mip contents.
      expect(mat.transparent).toBe(false);
      expect(mat.side).toBe(THREE.FrontSide);
    }

    chain.dispose();
  }

  it('gives the upsample pass AdditiveBlending under WebGL2', () => {
    expectBloomBlending(mockCaps('webgl2'));
  });

  it('gives the upsample pass AdditiveBlending under WebGPU', () => {
    expectBloomBlending(mockCaps('webgpu'));
  });
});

describe('bloom TSL factories — primitive uniform propagation', () => {
  // Pins the `.onUpdate('render')` wiring on uThreshold / uSmoothing /
  // uRadius. Before the fix, those were bare `uniform(number)` nodes
  // that captured the JS value at factory-build time, so
  // `BloomChain.setThreshold` / `.setRadius` writes to
  // `IUniform.value` silently dropped under WebGPU. The deterministic
  // side-effect of `.onUpdate(cb, 'render')` on a TSL UniformNode is:
  //   1. `node.updateType === 'render'`
  //   2. `node.update(frame)` runs `cb`; if it returns a non-undefined
  //      value, that value is assigned to `node.value`.
  // Driving `node.update()` simulates one render tick and lets us
  // assert the IUniform → TSL-node read path without a renderer.
  //
  // Inner UniformNodes are closed over by the Fn body and not
  // reachable from the returned NodeMaterial, so we hook them at the
  // TSL `uniform()` factory instead.

  type RenderUpdateNode = {
    updateType: string;
    update?: (frame: unknown) => void;
    value: unknown;
  };

  async function loadFactoriesWithUniformSpy(): Promise<{
    bloomThresholdWebGPUFactory: typeof import('../../../../rendering/post-processing/bloom/bloom.tsl').bloomThresholdWebGPUFactory;
    bloomUpsampleWebGPUFactory: typeof import('../../../../rendering/post-processing/bloom/bloom.tsl').bloomUpsampleWebGPUFactory;
    capturedNodes: RenderUpdateNode[];
  }> {
    // Re-import the bloom TSL module under a vi.doMock so a single
    // shared `uniform` wrapper observes every node the factories
    // create. Numeric-valued uniforms are the only ones we care
    // about for this test (Vector2 has its own contract).
    vi.resetModules();
    const capturedNodes: RenderUpdateNode[] = [];
    const tsl = (await import('three/tsl')) as typeof import('three/tsl');
    const realUniform = tsl.uniform;
    vi.doMock('three/tsl', () => ({
      ...tsl,
      uniform: ((value: unknown, ...rest: unknown[]) => {
        const node = (realUniform as unknown as (v: unknown, ...rest: unknown[]) => unknown)(
          value,
          ...rest
        );
        if (typeof value === 'number') {
          capturedNodes.push(node as RenderUpdateNode);
        }
        return node;
      }) as unknown as typeof tsl.uniform,
    }));
    const mod = await import('../../../../rendering/post-processing/bloom/bloom.tsl');
    return {
      bloomThresholdWebGPUFactory: mod.bloomThresholdWebGPUFactory,
      bloomUpsampleWebGPUFactory: mod.bloomUpsampleWebGPUFactory,
      capturedNodes,
    };
  }

  it('threshold factory wires uThreshold / uSmoothing to .onUpdate("render")', async () => {
    const { bloomThresholdWebGPUFactory, capturedNodes } = await loadFactoriesWithUniformSpy();
    const uniforms: Record<string, THREE.IUniform> = {
      uInput: { value: new THREE.Texture() },
      uTexelSize: { value: new THREE.Vector2(1 / 64, 1 / 64) },
      uThreshold: { value: 0.2 },
      uSmoothing: { value: 0.05 },
    };
    const mat = bloomThresholdWebGPUFactory(uniforms);

    // Both numeric uniforms must have been registered for render-tick updates.
    const renderUpdates = capturedNodes.filter((n) => n.updateType === 'render');
    expect(renderUpdates).toHaveLength(2);

    // Mutate the host IUniform — fix means the next render tick must
    // read this new value into the TSL node.
    uniforms.uThreshold.value = 0.42;
    uniforms.uSmoothing.value = 0.13;
    for (const n of renderUpdates) n.update?.({});

    const numericValues = renderUpdates.map((n) => n.value);
    expect(numericValues).toContain(0.42);
    expect(numericValues).toContain(0.13);

    mat.dispose();
    (uniforms.uInput.value as THREE.Texture).dispose();
    vi.doUnmock('three/tsl');
  });

  it('upsample factory wires uRadius to .onUpdate("render")', async () => {
    const { bloomUpsampleWebGPUFactory, capturedNodes } = await loadFactoriesWithUniformSpy();
    const uniforms: Record<string, THREE.IUniform> = {
      uInput: { value: new THREE.Texture() },
      uTexelSize: { value: new THREE.Vector2(1 / 64, 1 / 64) },
      uRadius: { value: 1.0 },
    };
    const mat = bloomUpsampleWebGPUFactory(uniforms);

    const renderUpdates = capturedNodes.filter((n) => n.updateType === 'render');
    expect(renderUpdates).toHaveLength(1);

    uniforms.uRadius.value = 3.5;
    for (const n of renderUpdates) n.update?.({});

    expect(renderUpdates[0].value).toBe(3.5);

    mat.dispose();
    (uniforms.uInput.value as THREE.Texture).dispose();
    vi.doUnmock('three/tsl');
  });
});
