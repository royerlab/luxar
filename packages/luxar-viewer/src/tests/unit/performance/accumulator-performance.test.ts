/**
 * Performance regression tests for Data Accumulators
 *
 * Verifies that accumulators actually eliminate allocations and reuse buffers
 * as intended. These tests measure allocation counts, not just correctness.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { LoadedPointsDataAccumulator } from '../../../data/accumulators/points';

describe('Accumulator Performance Regression Tests', () => {
  let accumulator: LoadedPointsDataAccumulator;

  beforeEach(() => {
    accumulator = new LoadedPointsDataAccumulator(1000, 3, 10000);
  });

  describe('Zero-Allocation Operation', () => {
    it('should reuse buffers across multiple getData calls (no allocations)', () => {
      // Fill once
      accumulator.fill(0, {
        positions: new Float32Array(Array(3000).fill(1)),
        colors: new Uint8Array(Array(3000).fill(128)),
        radii: new Float32Array(Array(1000).fill(0.5)),
        sharpness: new Float32Array(Array(1000).fill(2.0)),
      });

      // Get data multiple times
      const stats1 = accumulator.getStats();
      const data1 = accumulator.getData(1000);
      const stats2 = accumulator.getStats();

      // No new allocations between getData calls
      expect(stats2.allocations).toBe(stats1.allocations);

      // Get again
      const data2 = accumulator.getData(900);
      const stats3 = accumulator.getStats();

      // Still no allocations
      expect(stats3.allocations).toBe(stats1.allocations);

      // Buffers are views (same underlying data)
      expect(data1.positions.buffer).toBe(data2.positions.buffer);
    });

    it('should return subarrays (zero-copy views, not copies)', () => {
      accumulator.fill(0, {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]),
        colors: new Uint8Array([255, 128, 0, 128, 255, 0]),
        radii: new Float32Array([0.5, 0.6]),
        sharpness: new Float32Array([2.0, 2.5]),
      });

      const data = accumulator.getData(2);

      // Modify returned data
      data.positions[0] = 999;

      // Verify it's a view (modifies underlying buffer)
      const data2 = accumulator.getData(2);
      expect(data2.positions[0]).toBe(999); // Change visible in next getData!
    });

    // performance.md O3 / Phase E19: previously one `it` bundled two
    // independent contracts of the no-alloc-on-read invariant:
    //   (a) within-capacity getData() calls do NOT increment
    //       allocations (3 reads after a single fill)
    //   (b) growing beyond capacity via ensureCapacity() increments
    //       allocations by EXACTLY 1 and growthEvents by 1
    // A regression that grew on every getData() but stayed quiet on
    // ensureCapacity (or vice-versa) would surface as a generic
    // "should only allocate on growth..." failure that doesn't name
    // the broken contract. Split into two focused tests.

    it('repeated getData() calls within capacity do NOT increment the allocation counter', () => {
      const initialAllocs = accumulator.getStats().allocations;

      accumulator.fill(0, {
        positions: new Float32Array(Array(2400).fill(1)), // 800 points * 3
        colors: new Float32Array(Array(2400).fill(0.5)),
        radii: new Float32Array(Array(800).fill(0.5)),
        sharpness: new Float32Array(Array(800).fill(2.0)),
      });

      accumulator.getData(800);
      accumulator.getData(800);
      accumulator.getData(800);

      expect(accumulator.getStats().allocations).toBe(initialAllocs);
    });

    it('ensureCapacity() growth increments allocations by exactly 1 and growthEvents by 1', () => {
      const initialAllocs = accumulator.getStats().allocations;
      const initialGrowth = accumulator.getStats().growthEvents;

      accumulator.ensureCapacity(2000);

      expect(accumulator.getStats().allocations).toBe(initialAllocs + 1);
      expect(accumulator.getStats().growthEvents).toBe(initialGrowth + 1);
    });
  });

  describe('Memory Efficiency', () => {
    it('should track accurate memory usage', () => {
      const capacity = 1000;
      const acc = new LoadedPointsDataAccumulator(capacity, 3, 10000);

      const stats = acc.getStats();

      // Memory = capacity * (pos:12 + color:12 + radii:4 + sharpness:4) = 32 bytes/point
      const expectedMB = (capacity * 32) / 1024 / 1024;
      expect(stats.memoryMB).toBeCloseTo(expectedMB, 4);
    });

    it('should have stable memory after multiple fill cycles', () => {
      // performance.md C2[P2][P7] fix: prior `.toBe(initialMem)` did exact
      // float equality on a derived value `(capacity * 32) / 1024 / 1024`,
      // a latent bug-magnet against future refactors. The actual invariant
      // is "capacity unchanged" — assert that directly + memoryMB via
      // toBeCloseTo so a numerically-identical-but-not-bit-identical
      // refactor (e.g. helper that re-orders ops) doesn't fail spuriously.
      const initialStats = accumulator.getStats();
      const initialMem = initialStats.memoryMB;
      const initialCapacity = initialStats.capacity;

      // Fill and get 10 times
      for (let i = 0; i < 10; i++) {
        accumulator.fill(0, {
          positions: new Float32Array(Array(2400).fill(1)),
          colors: new Float32Array(Array(2400).fill(0.5)),
        });
        accumulator.getData(800);
      }

      const finalStats = accumulator.getStats();
      // Primary invariant: capacity did not grow.
      expect(finalStats.capacity).toBe(initialCapacity);
      // Secondary: memoryMB unchanged within Float64 tolerance.
      expect(finalStats.memoryMB).toBeCloseTo(initialMem, 7);
    });
  });

  describe('Type Preservation', () => {
    it('should preserve Uint8Array types without conversion', () => {
      const uint8Colors = new Uint8Array([255, 128, 0, 128, 255, 0]);

      accumulator.fill(0, {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]),
        colors: uint8Colors,
      });

      const data = accumulator.getData(2);

      // No conversion - still Uint8Array
      expect(data.colors).toBeInstanceOf(Uint8Array);
      expect(data.colors![0]).toBe(255); // Not converted to 1.0
      expect(data.metadata.dtypes!.colors).toBe('uint8');
    });

    it('should preserve Uint16Array types without conversion', () => {
      const uint16Colors = new Uint16Array([65535, 32768, 0]);

      accumulator.fill(0, {
        positions: new Float32Array([1, 2, 3]),
        colors: uint16Colors,
      });

      const data = accumulator.getData(1);

      // No conversion - still Uint16Array
      expect(data.colors).toBeInstanceOf(Uint16Array);
      expect(data.colors![0]).toBe(65535); // Not converted to 1.0
      expect(data.metadata.dtypes!.colors).toBe('uint16');
    });
  });

  describe('Growth Strategy Validation', () => {
    it('should grow by exactly 1.5x until reaching needed capacity', () => {
      const acc = new LoadedPointsDataAccumulator(100, 3, 10000);

      // Grow to 5000
      acc.ensureCapacity(5000);

      const finalCapacity = acc.getStats().capacity;

      // Should be: 100 * 1.5 * 1.5 * 1.5 ... until >= 5000
      // 100 → 150 → 225 → 337 → 505 → 757 → 1135 → 1702 → 2553 → 3829 → 5743
      expect(finalCapacity).toBeGreaterThanOrEqual(5000);
      expect(finalCapacity).toBeLessThan(5000 * 1.5); // Not excessive
    });

    it('should track growth events correctly', () => {
      const acc = new LoadedPointsDataAccumulator(100, 3, 10000);

      acc.ensureCapacity(200); // 1 growth
      acc.ensureCapacity(300); // 1 more growth
      acc.ensureCapacity(250); // No growth (already sufficient)

      expect(acc.getStats().growthEvents).toBe(2);
    });
  });

  describe('Attribute Presence Tracking', () => {
    it('should return undefined for unfilled optional attributes', () => {
      accumulator.fill(0, {
        positions: new Float32Array([1, 2, 3]),
        // No colors, radii, or sharpness
      });

      const data = accumulator.getData(1);

      expect(data.positions).toBeDefined();
      expect(data.colors).toBeUndefined();
      expect(data.radii).toBeUndefined();
      expect(data.sharpness).toBeUndefined();
    });

    it('should correctly track which attributes have been filled', () => {
      // Fill positions and colors only
      accumulator.fill(0, {
        positions: new Float32Array([1, 2, 3]),
        colors: new Uint8Array([255, 0, 0]),
      });

      const data1 = accumulator.getData(1);
      expect(data1.colors).toBeDefined();
      expect(data1.radii).toBeUndefined();

      // Now fill radii
      accumulator.fill(0, {
        positions: new Float32Array([1, 2, 3]),
        colors: new Uint8Array([255, 0, 0]),
        radii: new Float32Array([0.5]),
      });

      const data2 = accumulator.getData(1);
      expect(data2.colors).toBeDefined();
      expect(data2.radii).toBeDefined(); // Now present!
      expect(data2.sharpness).toBeUndefined(); // Still absent
    });
  });
});
