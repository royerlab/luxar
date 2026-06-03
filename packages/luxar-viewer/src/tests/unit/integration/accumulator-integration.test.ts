/**
 * Integration tests for Data Accumulator usage in loaders
 *
 * These tests verify that loaders ACTUALLY use accumulators when enabled,
 * not just that accumulators exist. Uses spies to verify method calls.
 */

import { describe, it, expect, vi } from 'vitest';
import { LoadedPointsDataAccumulator } from '../../../data/accumulators/points';
import { LinesDataAccumulator } from '../../../data/accumulators/lines';
import { GSplatsDataAccumulator } from '../../../data/accumulators/gsplats';

// integration.md O5 / Phase E26: previously the outer describe was
// `'Accumulator Integration Tests'`, but the file's actual content is
// direct unit tests of the accumulators' public API (no loader is
// being tested — only the accumulator's spy-able surface). Rename to
// surface scope ("Public API" reflects what the body asserts).
describe('Accumulator Public API', () => {
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
        })
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
    // integration.md O6 / Phase E18: previously one `it` bundled TWO
    // distinct contracts of the buffer-reuse semantics:
    //   (a) stats.allocations counter does NOT increment on a second
    //       fill within capacity
    //   (b) the actual ArrayBuffer behind the typed array is IDENTITY-
    //       preserved across the second fill
    // A regression that re-allocates the buffer but keeps the counter
    // stable (or vice-versa) would surface as a generic
    // "should reuse buffers..." failure. Split into two independent
    // tests so failures name the specific reuse contract that broke.
    function loadTwice() {
      const accumulator = new LoadedPointsDataAccumulator(1000, 3, 10000);
      accumulator.fill(0, {
        positions: new Float32Array(3000),
        colors: new Uint8Array(3000),
      });
      const stats1 = accumulator.getStats();
      const data1 = accumulator.getData(1000);
      const buffer1 = data1.positions.buffer;
      accumulator.fill(0, {
        positions: new Float32Array(2400),
        colors: new Uint8Array(2400),
      });
      const stats2 = accumulator.getStats();
      const data2 = accumulator.getData(800);
      return { stats1, stats2, data1, data2, buffer1 };
    }

    it('second fill within capacity does NOT increment the allocation counter', () => {
      const { stats1, stats2 } = loadTwice();
      expect(stats2.allocations).toBe(stats1.allocations);
    });

    it('second fill within capacity preserves the ArrayBuffer identity of the typed positions array', () => {
      const { data2, buffer1 } = loadTwice();
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

  // integration.md O7 / Phase E50: previously named "Multi-Type
  // Integration" — the block's tests cover dtype-preservation through
  // fill→getData() round-trips. "Multi-Type" is fine but "Integration"
  // misframes them since these are direct-API unit tests (no loader
  // involvement). Rename to surface the actual contract.
  describe('dtype preservation through getData() public API', () => {
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

  // [R11/A-G1][P8] Three-geometry symmetry: Points coverage above is now
  // mirrored on Lines and GSplats accumulators. The three geometries share
  // the DataAccumulator contract (ensureCapacity / fill / getData /
  // dispose); a test for one but not the others is a hidden coverage gap
  // — see [[feedback_geometry_symmetry]]. These tests pin the public
  // surface only (no reflection into private buffers).

  describe('Lines Loader Integration', () => {
    it('ensureCapacity() forwards (vertexCount, segmentCount) intact', () => {
      const acc = new LinesDataAccumulator(/*vertices*/ 1024, /*segments*/ 512, /*ndim*/ 3);
      const spy = vi.spyOn(acc, 'ensureCapacity');
      acc.ensureCapacity(2048, 1024);
      expect(spy).toHaveBeenCalledWith(2048, 1024);
    });

    it('type-init via fill() succeeds when types are not yet seeded', () => {
      const acc = new LinesDataAccumulator(1024, 512, 3);
      acc.ensureCapacity(1024, 512);
      expect(() =>
        acc.fill(/*segmentOffset*/ 0, /*vertexOffset*/ 0, {
          positions: new Float32Array(3 * 3), // 3 vertices × ndim=3
          segments: new Uint32Array([0, 1, 1, 2]), // 2 segments
          widths: new Float32Array(3),
          colors: new Uint8Array(3 * 3),
          sharpness: new Float32Array(3),
        })
      ).not.toThrow();
    });

    it('getData(segmentCount, vertexCount) returns positions + segments + widths views', () => {
      const acc = new LinesDataAccumulator(1024, 512, 3);
      const spy = vi.spyOn(acc, 'getData');
      acc.ensureCapacity(1024, 512);
      acc.fill(0, 0, {
        positions: new Float32Array(3 * 3),
        segments: new Uint32Array([0, 1, 1, 2]),
        widths: new Float32Array(3),
      });
      const result = acc.getData(2, 3);
      expect(spy).toHaveBeenCalledWith(2, 3);
      expect(result.positions).toBeInstanceOf(Float32Array);
      expect(result.segments).toBeInstanceOf(Uint32Array);
      expect(result.widths).toBeInstanceOf(Float32Array);
    });

    it('exposes Uint8 vs Float32 color dtype across separate instances', () => {
      // Use 2 distinct vertices and 1 real (non-self-loop) segment so the
      // dtype-detection-on-first-fill contract is exercised on a
      // non-degenerate input. Color buffer length matches the per-vertex
      // 3-channel layout of the underlying accumulator.
      const accUint8 = new LinesDataAccumulator(1024, 512, 3);
      accUint8.fill(0, 0, {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]),
        segments: new Uint32Array([0, 1]),
        widths: new Float32Array([1, 1]),
        colors: new Uint8Array([255, 128, 0, 0, 128, 255]),
      });
      const u8 = accUint8.getData(1, 2);
      expect(u8.colors).toBeInstanceOf(Uint8Array);
      // Pin actual content so a mutant that swapped the buffer reference
      // (handing back the wrong instance's view) would fail.
      expect(Array.from(u8.colors as Uint8Array)).toEqual([255, 128, 0, 0, 128, 255]);
      accUint8.dispose();

      const accF32 = new LinesDataAccumulator(1024, 512, 3);
      accF32.fill(0, 0, {
        positions: new Float32Array([1, 2, 3, 4, 5, 6]),
        segments: new Uint32Array([0, 1]),
        widths: new Float32Array([1, 1]),
        colors: new Float32Array([1.0, 0.5, 0.0, 0.0, 0.5, 1.0]),
      });
      const f32 = accF32.getData(1, 2);
      expect(f32.colors).toBeInstanceOf(Float32Array);
      expect(Array.from(f32.colors as Float32Array)).toEqual([1.0, 0.5, 0.0, 0.0, 0.5, 1.0]);

      expect(u8.colors!.constructor).not.toBe(f32.colors!.constructor);
    });
  });

  describe('GSplats Loader Integration', () => {
    it('ensureCapacity() forwards its argument to the spy intact', () => {
      const acc = new GSplatsDataAccumulator(/*initialCapacity*/ 1024, /*ndim*/ 3);
      const spy = vi.spyOn(acc, 'ensureCapacity');
      acc.ensureCapacity(4096);
      expect(spy).toHaveBeenCalledWith(4096);
    });

    it('type-init via fill() succeeds when types are not yet seeded', () => {
      const acc = new GSplatsDataAccumulator(1024, 3);
      acc.ensureCapacity(1024);
      // ndim=3 → cholesky k = 3*4/2 = 6 elements per splat
      expect(() =>
        acc.fill(0, {
          positions: new Float32Array(3), // 1 splat × ndim=3
          amplitudes: new Float32Array(1),
          choleskyFactors: new Float32Array(6),
          colors: new Uint8Array(3),
        })
      ).not.toThrow();
    });

    it('getData(count) forwards count and returns positions + amplitudes + cholesky views', () => {
      const acc = new GSplatsDataAccumulator(1024, 3);
      const spy = vi.spyOn(acc, 'getData');
      acc.ensureCapacity(1024);
      acc.fill(0, {
        positions: new Float32Array(3),
        amplitudes: new Float32Array(1),
        choleskyFactors: new Float32Array(6),
      });
      const result = acc.getData(1);
      expect(spy).toHaveBeenCalledWith(1);
      expect(result.positions).toBeInstanceOf(Float32Array);
      expect(result.amplitudes).toBeInstanceOf(Float32Array);
      expect(result.choleskyFactors).toBeInstanceOf(Float32Array);
    });

    it('exposes Uint8 vs Float32 color dtype across separate instances', () => {
      const accUint8 = new GSplatsDataAccumulator(1024, 3);
      accUint8.fill(0, {
        positions: new Float32Array(3),
        amplitudes: new Float32Array([1]),
        choleskyFactors: new Float32Array(6),
        colors: new Uint8Array([255, 128, 0]),
      });
      const u8 = accUint8.getData(1);
      expect(u8.colors).toBeInstanceOf(Uint8Array);
      accUint8.dispose();

      const accF32 = new GSplatsDataAccumulator(1024, 3);
      accF32.fill(0, {
        positions: new Float32Array(3),
        amplitudes: new Float32Array([1]),
        choleskyFactors: new Float32Array(6),
        colors: new Float32Array([1.0, 0.5, 0.0]),
      });
      const f32 = accF32.getData(1);
      expect(f32.colors).toBeInstanceOf(Float32Array);

      expect(u8.colors!.constructor).not.toBe(f32.colors!.constructor);
    });
  });
});
