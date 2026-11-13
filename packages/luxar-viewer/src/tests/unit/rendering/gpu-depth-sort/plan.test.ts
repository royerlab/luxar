import { describe, expect, it } from 'vitest';

import {
  BLOCK_ELEMENTS,
  PARAMS_BYTES,
  RADIX,
  RADIX_BITS,
  RADIX_PASSES,
  SCAN_BLOCK_ELEMENTS,
  packParams,
  planDispatch,
  planGrowth,
  planScratch,
  scanLevels,
  viewZRow,
} from '../../../../rendering/gpu-depth-sort/plan';

const LIMITS = {
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxComputeWorkgroupsPerDimension: 65535,
  minUniformBufferOffsetAlignment: 256,
};

describe('GPU depth sort plan', () => {
  it('covers the 16-bit key in whole radix passes', () => {
    expect(RADIX).toBe(1 << RADIX_BITS);
    expect(RADIX_PASSES * RADIX_BITS).toBe(16);
  });

  it('scans in levels until one workgroup holds a level', () => {
    expect(scanLevels(1)).toEqual([1]);
    expect(scanLevels(SCAN_BLOCK_ELEMENTS)).toEqual([SCAN_BLOCK_ELEMENTS]);
    expect(scanLevels(SCAN_BLOCK_ELEMENTS + 1)).toEqual([SCAN_BLOCK_ELEMENTS + 1, 2]);
    expect(scanLevels(SCAN_BLOCK_ELEMENTS ** 2 + 1)).toEqual([
      SCAN_BLOCK_ELEMENTS ** 2 + 1,
      SCAN_BLOCK_ELEMENTS + 1,
      2,
    ]);
  });

  it('dispatches one block per BLOCK_ELEMENTS elements', () => {
    expect(planDispatch(1)).toEqual({ count: 1, blocks: 1, scanLevels: [RADIX] });
    expect(planDispatch(BLOCK_ELEMENTS).blocks).toBe(1);
    expect(planDispatch(BLOCK_ELEMENTS + 1)).toEqual({
      count: BLOCK_ELEMENTS + 1,
      blocks: 2,
      scanLevels: [2 * RADIX],
    });
    const tenMillion = planDispatch(10_000_000);
    expect(tenMillion.blocks).toBe(Math.ceil(10_000_000 / BLOCK_ELEMENTS));
    expect(tenMillion.scanLevels).toEqual(scanLevels(RADIX * tenMillion.blocks));
  });

  it('sizes the scratch for whole blocks', () => {
    const plan = planScratch(BLOCK_ELEMENTS + 1, LIMITS);
    if ('unsupported' in plan) throw new Error(plan.unsupported);
    expect(plan.capacity).toBe(2 * BLOCK_ELEMENTS);
    expect(plan.elementBytes).toBe(2 * BLOCK_ELEMENTS * 4);
    expect(plan.histogramBytes).toBe(2 * RADIX * 4);
    expect(plan.levelTotalsBytes).toEqual([4]);
    expect(plan.totalBytes).toBe(4 * plan.elementBytes + plan.histogramBytes + 4);
  });

  it('refuses what one binding or one dispatch dimension cannot hold', () => {
    expect(planScratch(0, LIMITS)).toHaveProperty('unsupported');
    expect(planScratch(1.5, LIMITS)).toHaveProperty('unsupported');
    expect(
      planScratch(1_000_000, { ...LIMITS, maxStorageBufferBindingSize: 1_000_000 })
    ).toHaveProperty('unsupported');
    expect(
      planScratch(1_000_000, { ...LIMITS, maxComputeWorkgroupsPerDimension: 100 })
    ).toHaveProperty('unsupported');
    // 33.5M elements fit the 128 MiB default binding; one more block does not.
    const atLimit = (128 * 1024 * 1024) / 4;
    expect(planScratch(atLimit, LIMITS)).not.toHaveProperty('unsupported');
    expect(planScratch(atLimit + 1, LIMITS)).toHaveProperty('unsupported');
  });

  it('grows by doubling, or to the exact count when doubling does not fit', () => {
    expect(planGrowth(5000, 4096, LIMITS)?.capacity).toBe(8192);
    expect(planGrowth(100_000, 4096, LIMITS)?.capacity).toBe(
      Math.ceil(100_000 / BLOCK_ELEMENTS) * BLOCK_ELEMENTS
    );
    const tight = { ...LIMITS, maxStorageBufferBindingSize: 3 * BLOCK_ELEMENTS * 4 };
    expect(planGrowth(3 * BLOCK_ELEMENTS, 2 * BLOCK_ELEMENTS, tight)?.capacity).toBe(
      3 * BLOCK_ELEMENTS
    );
    expect(planGrowth(4 * BLOCK_ELEMENTS, 0, tight)).toBeNull();
  });

  it('packs the params uniform in the WGSL struct layout', () => {
    const bytes = packParams({
      viewZ: [0.5, -1, 2, 3],
      count: 7,
      blocks: 1,
      textureWidth: 4092,
      texelsPerElement: 6,
      centerMode: 'midpoint01',
    });
    expect(bytes.byteLength).toBe(PARAMS_BYTES);
    expect(Array.from(new Float32Array(bytes, 0, 4))).toEqual([0.5, -1, 2, 3]);
    expect(Array.from(new Uint32Array(bytes, 16, 8))).toEqual([7, 1, 4092, 6, 1, 0, 0, 0]);
    const texel0 = packParams({
      viewZ: [0, 0, 0, 0],
      count: 1,
      blocks: 1,
      textureWidth: 4096,
      texelsPerElement: 4,
      centerMode: 'texel0',
    });
    expect(new Uint32Array(texel0, 16, 8)[4]).toBe(0);
  });

  it('reads the view-z row of a column-major matrix', () => {
    const m = Array.from({ length: 16 }, (_, i) => i);
    expect(viewZRow(m)).toEqual([2, 6, 10, 14]);
  });
});
