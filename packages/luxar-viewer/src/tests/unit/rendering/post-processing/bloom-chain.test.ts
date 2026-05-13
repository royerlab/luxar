/**
 * Unit tests for BloomChain resource/state management.
 *
 * Rendering itself is covered by browser E2E; these tests pin the
 * no-context contracts that can regress in ordinary TypeScript changes.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { BloomChain } from '../../../../rendering/post-processing/bloom-chain';
import type { RendererCapabilities } from '../../../../rendering/renderer-capabilities';

function mockCaps(): RendererCapabilities {
  return {
    api: 'webgl2',
    hdr: {
      p3Gamut: false,
      rec2020Gamut: false,
      hdr: false,
      deepColor: false,
      floatTextures: true,
      colorDepth: { red: 8, green: 8, blue: 8 },
      recommendedColorSpace: 'srgb',
    },
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
