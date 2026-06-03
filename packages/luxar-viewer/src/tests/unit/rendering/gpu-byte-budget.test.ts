import { afterEach, describe, expect, it } from 'vitest';
import {
  initGpuByteBudget,
  getGpuByteBudget,
  reduceGpuByteBudgetForContextLoss,
} from '../../../rendering/gpu-byte-budget';

/** Temporarily set navigator.deviceMemory (GB) for a test. */
function withDeviceMemory(gb: number | undefined, fn: () => void): void {
  const had = Object.prototype.hasOwnProperty.call(navigator, 'deviceMemory');
  const prev = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  Object.defineProperty(navigator, 'deviceMemory', { value: gb, configurable: true });
  try {
    fn();
  } finally {
    if (had) Object.defineProperty(navigator, 'deviceMemory', { value: prev, configurable: true });
    else delete (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  }
}

const MB = 1_000_000;

afterEach(() => {
  // Reset to a known state so tests don't leak the singleton budget.
  initGpuByteBudget(512 * MB);
});

describe('gpu-byte-budget', () => {
  it('honors an explicit override and skips the heuristic', () => {
    withDeviceMemory(8, () => {
      initGpuByteBudget(1536 * MB);
      expect(getGpuByteBudget()).toBe(1536 * MB);
    });
  });

  it('auto-sizes from deviceMemory at 25%, clamped to the [512MB, 2GB] range', () => {
    withDeviceMemory(8, () => {
      initGpuByteBudget(); // 8 GB × 0.25 = 2 GB (== ceiling)
      expect(getGpuByteBudget()).toBe(2_000 * MB);
    });
    withDeviceMemory(4, () => {
      initGpuByteBudget(); // 4 GB × 0.25 = 1 GB
      expect(getGpuByteBudget()).toBe(1_000 * MB);
    });
    withDeviceMemory(32, () => {
      initGpuByteBudget(); // clamps to 2 GB ceiling
      expect(getGpuByteBudget()).toBe(2_000 * MB);
    });
    withDeviceMemory(1, () => {
      initGpuByteBudget(); // 0.25 GB → clamps up to 512 MB floor
      expect(getGpuByteBudget()).toBe(512 * MB);
    });
  });

  it('treats 0 as "disable" (verbatim, not auto-sized)', () => {
    withDeviceMemory(8, () => {
      initGpuByteBudget(0); // explicit disable must NOT auto-size to 2 GB
      expect(getGpuByteBudget()).toBe(0);
    });
  });

  it('falls back to the floor when deviceMemory is unavailable', () => {
    withDeviceMemory(undefined, () => {
      initGpuByteBudget();
      expect(getGpuByteBudget()).toBe(512 * MB);
    });
  });

  it('halves the budget on context loss, down to a 256MB floor', () => {
    initGpuByteBudget(2_000 * MB);
    expect(reduceGpuByteBudgetForContextLoss()).toBe(1_000 * MB);
    expect(reduceGpuByteBudgetForContextLoss()).toBe(500 * MB);
    expect(getGpuByteBudget()).toBe(500 * MB);
    // Keeps halving but never below the 256 MB floor.
    reduceGpuByteBudgetForContextLoss(); // 250M → clamped to 256M floor
    expect(getGpuByteBudget()).toBe(256 * MB);
    reduceGpuByteBudgetForContextLoss(); // already at floor → unchanged
    expect(getGpuByteBudget()).toBe(256 * MB);
  });
});
