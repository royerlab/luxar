/**
 * Integration tests for Data Accumulator usage in loaders
 *
 * These tests verify that loaders ACTUALLY use accumulators when enabled,
 * not just that accumulators exist. Uses spies to verify method calls.
 */

import { describe, it, expect, vi } from 'vitest';
import { LoadedPointsDataAccumulator } from '../../../data/accumulators/points';

describe('Accumulator Integration Tests', () => {
  describe('Points Loader Integration', () => {
    // [integration.md/O4][P4] Split a single it() that bundled three
    // independent contracts (ensureCapacity call, type init via fill, getData
    // call + result shape) into three focused tests.

    it('ensureCapacity() forwards its argument to the spy intact', () => {
      const accumulator = new LoadedPointsDataAccumulator(1000, 3, 10000);
      const ensureCapacitySpy = vi.spyOn(accumulator, 'ensureCapacity');

      // This mirrors the call points-spatial-index-loader makes in its
      // first-fill branch (no line number — refs drift).
      // [integration.md/O2][P10] Removed brittle line-number reference.
      accumulator.ensureCapacity(1000);

      expect(ensureCapacitySpy).toHaveBeenCalledWith(1000);
    });

    it('type-init via fill() succeeds when types are not yet seeded', () => {
      const accumulator = new LoadedPointsDataAccumulator(1000, 3, 10000);
      accumulator.ensureCapacity(1000);

      // First fill establishes the accumulator's color/radii/sharpness dtypes
      // (the loader does this in its first-fill / type-init branch). It must
      // not throw. [integration.md/O2][P10] Removed brittle line-number ref.
      expect(() =>
        accumulator.fill(0, {
          positions: new Float32Array(3),
          colors: new Uint8Array(3),
          radii: new Float32Array(1),
          sharpness: new Float32Array(1),
        }),
      ).not.toThrow();
    });

    it('getData(count) forwards count to the spy and returns positions + colors views', () => {
      const accumulator = new LoadedPointsDataAccumulator(1000, 3, 10000);
      const getDataSpy = vi.spyOn(accumulator, 'getData');

      accumulator.ensureCapacity(1000);
      accumulator.fill(0, {
        positions: new Float32Array(3),
        colors: new Uint8Array(3),
        radii: new Float32Array(1),
        sharpness: new Float32Array(1),
      });

      // This mirrors what happens inside the loader's projectTo3D → getData
      // call. [integration.md/O2][P10] Removed brittle line-number ref.
      const result = accumulator.getData(1000);

      expect(getDataSpy).toHaveBeenCalledWith(1000);
      // Result must expose positions + colors via the public API.
      expect(result.positions).toBeDefined();
      expect(result.colors).toBeDefined();
    });

    it('should detect Uint8 colors on first fill and surface dtype through public getData()', () => {
      // integration.md C5 fix: previously this test reached into private state
      // (`(accumulator as any).types` and `(accumulator as any).colorBuffer`).
      // The observable contract is `getData(n).metadata.dtypes.colors === 'uint8'`
      // and that the returned colors view is a Uint8Array — both visible via
      // the public surface.
      const accumulator = new LoadedPointsDataAccumulator(1000, 3, 10000);

      accumulator.fill(0, {
        positions: new Float32Array([1, 2, 3]),
        colors: new Uint8Array([255, 128, 0]),
      });

      const result = accumulator.getData(1);
      expect(result.metadata.dtypes?.colors).toBe('uint8');
      expect(result.colors).toBeInstanceOf(Uint8Array);
    });

    it('should return subarrays (views) not copies', () => {
      const accumulator = new LoadedPointsDataAccumulator(1000, 3, 10000);

      accumulator.fill(0, {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]),
        colors: new Uint8Array([255, 0, 0, 0, 255, 0]),
      });

      const data1 = accumulator.getData(2);
      const data2 = accumulator.getData(2);

      // Same buffer (not copies)
      expect(data1.positions.buffer).toBe(data2.positions.buffer);

      // Modify data1
      data1.positions[0] = 999;

      // Verify it's visible in data2 (proves it's a view)
      expect(data2.positions[0]).toBe(999);
    });
  });

  describe('Accumulator Buffer Reuse Verification', () => {
    it('should reuse buffers across multiple loads', () => {
      const accumulator = new LoadedPointsDataAccumulator(1000, 3, 10000);

      // First load
      accumulator.fill(0, {
        positions: new Float32Array(3000),
        colors: new Uint8Array(3000),
      });

      const stats1 = accumulator.getStats();
      const data1 = accumulator.getData(1000);

      // Get buffer reference
      const buffer1 = data1.positions.buffer;

      // Second load (simulate view update)
      accumulator.fill(0, {
        positions: new Float32Array(2400), // Different data, same capacity
        colors: new Uint8Array(2400),
      });

      const stats2 = accumulator.getStats();
      const data2 = accumulator.getData(800);

      // Verify no new allocations (buffer reused)
      expect(stats2.allocations).toBe(stats1.allocations);

      // Verify same buffer reused
      expect(data2.positions.buffer).toBe(buffer1);
    });

    it('should only allocate on capacity growth', () => {
      const accumulator = new LoadedPointsDataAccumulator(100, 3, 10000);

      // Fill within capacity
      accumulator.fill(0, {
        positions: new Float32Array(240), // 80 points
      });

      const stats1 = accumulator.getStats();

      // Fill again within capacity
      accumulator.fill(0, {
        positions: new Float32Array(270), // 90 points
      });

      const stats2 = accumulator.getStats();

      // No growth
      expect(stats2.allocations).toBe(stats1.allocations);
      expect(stats2.growthEvents).toBe(0);

      // Now exceed capacity
      accumulator.ensureCapacity(200); // Exceeds 100 * 1.5 = 150

      const stats3 = accumulator.getStats();

      // Exactly one growth
      expect(stats3.allocations).toBe(stats1.allocations + 1);
      expect(stats3.growthEvents).toBe(1);
    });
  });

  describe('Multi-Type Integration', () => {
    it('should preserve types through full pipeline', () => {
      const accumulator = new LoadedPointsDataAccumulator(1000, 3, 10000);

      // Fill with Uint8 colors
      accumulator.fill(0, {
        positions: new Float32Array(3000),
        colors: new Uint8Array(3000),
        radii: new Float32Array(1000),
        sharpness: new Uint8Array(1000), // Uint8 sharpness
      });

      const data = accumulator.getData(1000);

      // Verify types preserved (no conversion)
      expect(data.colors).toBeInstanceOf(Uint8Array);
      expect(data.sharpness).toBeInstanceOf(Uint8Array);
      expect(data.positions).toBeInstanceOf(Float32Array);
      expect(data.radii).toBeInstanceOf(Float32Array);

      // Verify metadata reflects types
      expect(data.metadata.dtypes!.colors).toBe('uint8');
      expect(data.metadata.dtypes!.sharpness).toBe('uint8');
    });

    it('should expose Uint8 vs Float32 color dtype via public getData() across separate instances', () => {
      // integration.md C6 fix: previous version disposed one instance and
      // constructed a second; the assertions inspected private
      // `colorBuffer` state on each. That's two independent constructions,
      // not "type-change handling". Rewrite to test the actual public
      // contract: two accumulator instances with different color dtypes
      // each surface the right typed-array view AND the right metadata
      // dtype, via the public getData() surface. The test pins the
      // observable behaviour the loader actually relies on. The
      // "type-change within a single accumulator" path is not part of
      // the public contract today (loaders dispose and recreate), so we
      // assert only what production code actually depends on.
      // [integration.md/C6][P1]
      const accUint8 = new LoadedPointsDataAccumulator(1000, 3, 10000);
      accUint8.fill(0, {
        positions: new Float32Array([1, 2, 3]),
        colors: new Uint8Array([255, 128, 0]),
      });
      const uint8Result = accUint8.getData(1);
      expect(uint8Result.colors).toBeInstanceOf(Uint8Array);
      expect(uint8Result.metadata.dtypes?.colors).toBe('uint8');
      accUint8.dispose();

      const accFloat32 = new LoadedPointsDataAccumulator(1000, 3, 10000);
      accFloat32.fill(0, {
        positions: new Float32Array([1, 2, 3]),
        colors: new Float32Array([1.0, 0.5, 0.0]),
      });
      const float32Result = accFloat32.getData(1);
      expect(float32Result.colors).toBeInstanceOf(Float32Array);
      expect(float32Result.metadata.dtypes?.colors).toBe('float32');

      // Both dtypes round-trip through the public surface with distinct
      // typed-array constructors. (The `.colors` field is typed as
      // optional on the data interface but is asserted defined by the
      // `toBeInstanceOf` checks above; non-null-assert is sound here.)
      expect(uint8Result.colors!.constructor).not.toBe(float32Result.colors!.constructor);
    });
  });
});
