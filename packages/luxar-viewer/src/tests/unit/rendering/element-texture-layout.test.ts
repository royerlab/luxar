/**
 * Unit tests for the per-node element capacity bound in
 * `rendering/element-texture-layout.ts` — the texture-dimension bound, and the
 * mobile byte bound that keeps a 16384-class phone from allocating a 1 GB
 * element texture for one node.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  SPLAT_TEXTURE_LAYOUT,
  POINT_TEXTURE_LAYOUT,
  clampSplatCapacity,
  configureElementTextureLayout,
  getMaxElementCapacityPerNode,
  resetElementTextureLayoutForTests,
} from '../../../rendering/element-texture-layout';

// Drive the device class through the real `?input=` override rather than a
// module mock: the layout module may already be in the module graph via the
// shared test setup, and a mock registered here would not reach that instance.
import {
  resetInputProfileForTests,
  setInputProfileOverride,
} from '../../../utils/input-capabilities';

const MIB = 1024 * 1024;

describe('getMaxElementCapacityPerNode', () => {
  beforeEach(() => {
    resetInputProfileForTests();
    resetElementTextureLayoutForTests();
  });
  afterEach(() => {
    resetInputProfileForTests();
    resetElementTextureLayoutForTests();
  });

  it('is the texture-dimension bound on a laptop/desktop (unchanged)', () => {
    // 4096-class floor: 4096 × 4096 / 4 texels per splat.
    expect(getMaxElementCapacityPerNode(SPLAT_TEXTURE_LAYOUT)).toBe((4096 * 4096) / 4);
    configureElementTextureLayout(16384);
    expect(getMaxElementCapacityPerNode(SPLAT_TEXTURE_LAYOUT)).toBe((4096 * 16384) / 4);
  });

  it('on a phone/tablet the 256 MiB byte bound binds on a large-texture device', () => {
    setInputProfileOverride('touch');
    configureElementTextureLayout(16384);
    // 256 MiB / (4 texels × 16 B) = 4,194,304 splats — a quarter of the
    // 16.7 M the texture dimensions alone would permit (1.07 GB).
    expect(getMaxElementCapacityPerNode(SPLAT_TEXTURE_LAYOUT)).toBe((256 * MIB) / (4 * 16));
    // Points: 3 texels per element.
    expect(getMaxElementCapacityPerNode(POINT_TEXTURE_LAYOUT)).toBe(
      Math.floor((256 * MIB) / (3 * 16))
    );
  });

  it('on a phone/tablet with a 4096-class texture the two bounds agree (no change)', () => {
    setInputProfileOverride('touch');
    expect(getMaxElementCapacityPerNode(SPLAT_TEXTURE_LAYOUT)).toBe((4096 * 4096) / 4);
  });

  it('clampSplatCapacity applies the tighter bound and reports the device-class limit', () => {
    setInputProfileOverride('touch');
    configureElementTextureLayout(16384);
    const errors: string[] = [];
    const spy = vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
      errors.push(args.map(String).join(' '));
    });
    try {
      expect(clampSplatCapacity(10_000_000)).toBe((256 * MIB) / (4 * 16));
      expect(errors.join('\n')).toMatch(/256 MiB per element texture on this device class/);
    } finally {
      spy.mockRestore();
    }
  });
});
