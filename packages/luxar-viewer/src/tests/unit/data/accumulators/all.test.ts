/**
 * Unit tests for Data Accumulators
 */

import { describe, it, expect, beforeEach, test } from 'vitest';
import * as fc from 'fast-check';
import { LoadedPointsDataAccumulator } from '../../../../data/accumulators/points';
import { LinesDataAccumulator } from '../../../../data/accumulators/lines';
import { GSplatsDataAccumulator } from '../../../../data/accumulators/gsplats';
import * as THREE from 'three';

describe('LoadedPointsDataAccumulator', () => {
  let accumulator: LoadedPointsDataAccumulator;

  beforeEach(() => {
    accumulator = new LoadedPointsDataAccumulator(1000, 3, 10000);
  });

  it('should initialize with the given capacity and zero growth events', () => {
    const stats = accumulator.getStats();
    expect(stats.capacity).toBe(1000);
    expect(stats.allocations).toBe(1);
    expect(stats.growthEvents).toBe(0);
  });

  it('should grow capacity by 1.5x', () => {
    const grew = accumulator.ensureCapacity(1500);
    expect(grew).toBe(true);
    expect(accumulator.getStats().capacity).toBe(1500); // ceil(1000 * 1.5) = 1500
    expect(accumulator.getStats().growthEvents).toBe(1);
  });

  it('should not grow if capacity is sufficient', () => {
    const grew = accumulator.ensureCapacity(500);
    expect(grew).toBe(false);
    expect(accumulator.getStats().capacity).toBe(1000);
  });

  it('should grow multiple times to reach needed capacity', () => {
    const grew = accumulator.ensureCapacity(5000);
    expect(grew).toBe(true);
    // 1000 → 1500 → 2250 → 3375 → 5062.5 (rounded to 5063)
    expect(accumulator.getStats().capacity).toBeGreaterThanOrEqual(5000);
  });

  it('should fill and retrieve point data (native types, no conversion)', () => {
    accumulator.fill(0, {
      positions: new Float32Array([1, 2, 3]),
      colors: new Uint8Array([255, 128, 0]), // Native Uint8 RGB
      radii: new Float32Array([0.5]),
    });

    const data = accumulator.getData(1);

    expect(data.positions[0]).toBe(1);
    expect(data.positions[1]).toBe(2);
    expect(data.positions[2]).toBe(3);
    // Native Uint8Array - NO conversion!
    expect(data.colors).toBeInstanceOf(Uint8Array);
    expect(data.colors![0]).toBe(255);
    expect(data.colors![1]).toBe(128);
    expect(data.colors![2]).toBe(0);
    expect(data.radii![0]).toBe(0.5);
    expect(data.metadata.dtypes!.colors).toBe('uint8');
  });

  it('should return LoadedPointsData with correct metadata structure', () => {
    accumulator.fill(0, {
      positions: new Float32Array([1, 2, 3, 4, 5, 6]), // 2 points
      colors: new Uint8Array([255, 0, 0, 0, 255, 0]), // RGB
      radii: new Float32Array([0.5, 0.6]),
    });

    const data = accumulator.getData(2);

    // Verify metadata structure
    expect(data.metadata).toBeDefined();
    expect(data.metadata.loadedPoints).toBe(2);
    expect(data.metadata.totalPoints).toBe(10000);
    expect(data.ndim).toBe(3);
    expect(data.metadata.bounds).toBeInstanceOf(THREE.Box3);
    expect(data.metadata.usedSpatialIndex).toBe(false);

    // Verify bounds computed from positions
    expect(data.metadata.bounds.min.x).toBe(1);
    expect(data.metadata.bounds.min.y).toBe(2);
    expect(data.metadata.bounds.min.z).toBe(3);
    expect(data.metadata.bounds.max.x).toBe(4);
    expect(data.metadata.bounds.max.y).toBe(5);
    expect(data.metadata.bounds.max.z).toBe(6);
  });

  it('should handle HDR colors (Float32Array)', () => {
    accumulator.fill(0, {
      positions: new Float32Array([1, 2, 3]),
      colors: new Float32Array([2.5, 1.8, 0.9]), // HDR values > 1.0
      radii: new Float32Array([0.5]),
    });

    const data = accumulator.getData(1);
    expect(data.colors![0]).toBeCloseTo(2.5, 5);
    expect(data.colors![1]).toBeCloseTo(1.8, 5);
    expect(data.colors![2]).toBeCloseTo(0.9, 5);
    expect(data.metadata.dtypes!.colors).toBe('float32');
  });

  it('should handle Uint16Array colors (native, no conversion)', () => {
    accumulator.fill(0, {
      positions: new Float32Array([1, 2, 3]),
      colors: new Uint16Array([65535, 32768, 0]), // Uint16 range
      radii: new Float32Array([0.5]),
    });

    const data = accumulator.getData(1);
    // Native Uint16Array - NO conversion!
    expect(data.colors).toBeInstanceOf(Uint16Array);
    expect(data.colors![0]).toBe(65535);
    expect(data.colors![1]).toBe(32768);
    expect(data.colors![2]).toBe(0);
    expect(data.metadata.dtypes!.colors).toBe('uint16');
  });

  it('should update metadata', () => {
    accumulator.updateMetadata({
      ndim: 4,
      totalPoints: 50000,
      usedSpatialIndex: true,
    });

    const data = accumulator.getData(1);
    expect(data.ndim).toBe(4);
    expect(data.metadata.totalPoints).toBe(50000);
    expect(data.metadata.usedSpatialIndex).toBe(true);
  });

  // `updateMetadata({ bounds })` silently drops the `bounds` argument
  // (it only warns). We can't easily intercept `log.warning` here without
  // mocking, but we can pin the behavioral contract: bounds is ignored and
  // `getData()` still computes fresh bounds from positions.
  it('updateMetadata: bounds parameter is ignored; bounds always recomputed from positions', () => {
    accumulator.fill(0, {
      positions: new Float32Array([1, 2, 3, 4, 5, 6]), // two points
    });
    // Pass a wildly wrong bounds — it must be ignored.
    const bogusBounds = new THREE.Box3(
      new THREE.Vector3(-999, -999, -999),
      new THREE.Vector3(999, 999, 999)
    );
    accumulator.updateMetadata({ bounds: bogusBounds });
    const data = accumulator.getData(2);
    // Bounds reflect actual positions, not the bogus value.
    expect(data.metadata.bounds.min.x).toBe(1);
    expect(data.metadata.bounds.max.x).toBe(4);
    expect(data.metadata.bounds.min.x).not.toBe(-999);
  });

  // `fill()` without positions prefers 1-per-point sources
  // (radii/sharpness/scalars) over `colors.length / 3` when advancing
  // usedCount. This keeps the live prefix correct even when a fill carries
  // a 1-per-point attribute like radii without positions or colors.
  it('fill without positions uses radii length to advance usedCount', () => {
    // First fill establishes types. Then a no-position fill with radii
    // must advance usedCount by the radii length, not divide colors/3.
    accumulator.fill(0, {
      positions: new Float32Array([1, 2, 3]),
      radii: new Float32Array([0.5]),
    });
    // Now grow capacity then check that a fill-after-grow with no
    // positions but with radii of length 5 advances usedCount to 5
    // (it would have to round-trip through ensureCapacity).
    accumulator.ensureCapacity(100);
    // Fill 5 more radii starting at offset 1 (no positions, no colors).
    accumulator.fill(1, {
      radii: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]),
    });
    // Growing again — the live prefix must be at least 6 (offset 1 + 5
    // radii), so the position buffer must preserve the original [1,2,3].
    accumulator.ensureCapacity(200);
    const pos = accumulator.getPositionBuffer();
    expect(pos[0]).toBe(1);
    expect(pos[1]).toBe(2);
    expect(pos[2]).toBe(3);
  });

  it('should calculate memory usage correctly', () => {
    const stats = accumulator.getStats();
    // 32 bytes per point: pos(12) + color(12) + radii(4) + sharpness(4)
    const expectedMB = (1000 * 32) / 1024 / 1024;
    expect(stats.memoryMB).toBeCloseTo(expectedMB, 4);
  });

  it('should dispose and reset', () => {
    accumulator.fill(0, {
      positions: new Float32Array([1, 2, 3]),
      colors: new Uint8Array([255, 0, 0]),
      radii: new Float32Array([0.5]),
    });

    accumulator.dispose();

    const stats = accumulator.getStats();
    expect(stats.capacity).toBe(0);
  });

  // --- Boundary cases ---

  it('ensureCapacity(0) is a no-op returning false with unchanged capacity', () => {
    const grew = accumulator.ensureCapacity(0);
    expect(grew).toBe(false);
    expect(accumulator.getStats().capacity).toBe(1000);
    expect(accumulator.getStats().growthEvents).toBe(0);
  });

  it('ensureCapacity(negative) is a no-op returning false with unchanged capacity', () => {
    const grew = accumulator.ensureCapacity(-50);
    expect(grew).toBe(false);
    expect(accumulator.getStats().capacity).toBe(1000);
    expect(accumulator.getStats().growthEvents).toBe(0);
  });

  it('getData(0) returns empty positions (length 0)', () => {
    const data = accumulator.getData(0);
    expect(data.positions.length).toBe(0);
    expect(data.pointCount).toBe(0);
  });

  it('getData(capacity + 1) throws the capacity-exceeded guard', () => {
    expect(() => accumulator.getData(1001)).toThrow(
      'Cannot get 1001 points from accumulator with capacity 1000'
    );
  });

  it('fill with zero-length positions is a no-op (capacity unchanged)', () => {
    accumulator.fill(0, { positions: new Float32Array(0) });
    expect(accumulator.getStats().capacity).toBe(1000);
    // usedCount stays 0 → growth still copies nothing; capacity stays put
    // and no growth event is recorded.
    expect(accumulator.getStats().growthEvents).toBe(0);
    const data = accumulator.getData(0);
    expect(data.positions.length).toBe(0);
  });

  it('zero-copy: getData positions share the same ArrayBuffer as the position buffer', () => {
    accumulator.fill(0, { positions: new Float32Array([1, 2, 3, 4, 5, 6]) });
    const data = accumulator.getData(2);
    // The returned subarray must be a view into the accumulator's buffer,
    // not a copy — same underlying ArrayBuffer object.
    expect(data.positions.buffer).toBe(accumulator.getPositionBuffer().buffer);
  });
});

describe('LinesDataAccumulator', () => {
  let accumulator: LinesDataAccumulator;

  beforeEach(() => {
    accumulator = new LinesDataAccumulator(1000, 500, 3);
  });

  it('should initialize with correct vertex and segment capacities', () => {
    const stats = accumulator.getStats();
    expect(stats.capacity).toBe(500); // segment capacity
    expect(stats.allocations).toBe(1);
  });

  it('should grow both vertex and segment capacities (estimated)', () => {
    // [integration.md OOS1] ensureCapacity with only vertex count now
    // defaults segments to vertexCount (over-estimates for typical
    // 1.5:1 meshes, exact for particle tracks). Previously the fallback
    // was ceil(vertexCount/1.5) which UNDER-estimated for particle
    // tracks and could silently truncate writes.
    // Initial capacity: 1000 vertices, 500 segments
    const grew = accumulator.ensureCapacity(1200); // 1200 vertices (exceeds 1000)
    expect(grew).toBe(true);
    // Vertices: 1000 → 1500 (grew)
    // Segments: estimated as vertexCount = 1200, exceeds 500 → grows to
    // at least 1200. Under the old estimate it would have been only
    // ceil(1200/1.5) = 800 — a particle-track caller relying on this
    // fallback would have written past 800 into uninitialized buffer.
    const stats = accumulator.getStats();
    expect(stats.capacity).toBeGreaterThanOrEqual(1200); // segment capacity
  });

  it('[integration.md OOS1] particle-track callback that OMITS segmentCount still has enough room', () => {
    // The audit's "may be too small!" comment was the foot-gun: a caller
    // that built a particle track (N vertices, N-1 segments) and forgot
    // to pass segmentCount used to get only ceil(N/1.5) segment slots,
    // truncating segment writes past that boundary. With the new
    // vertexCount fallback, an N-vertex particle track always has at
    // least N segment slots available — exact-fit-plus-one.
    const acc = new LinesDataAccumulator(64, 32, 3);
    // Caller knows there are 100 vertices but omits the segmentCount arg
    // (perhaps it's not yet computed at allocation time).
    acc.ensureCapacity(100);
    const stats = acc.getStats();
    // Pre-fix: capacity would have been only ceil(100/1.5) = 67.
    // Post-fix: capacity ≥ 100 (the vertexCount fallback).
    expect(stats.capacity).toBeGreaterThanOrEqual(100);
  });

  it('should use explicit segmentCount when provided', () => {
    // This tests the bug fix: particle tracks have N vertices → N-1 segments (ratio ~1:1)
    // The estimate (vertex/1.5) would underestimate segment capacity
    // Initial capacity: 1000 vertices, 500 segments
    const accumulator2 = new LinesDataAccumulator(100, 50, 3);

    // Particle track scenario: 150 vertices, 149 segments (ratio ~1:1, not 1.5:1)
    const grew = accumulator2.ensureCapacity(150, 149);
    expect(grew).toBe(true);

    // Without explicit segmentCount, estimate would be ceil(150/1.5) = 100
    // But we passed 149, so segment capacity should be >= 149
    const stats = accumulator2.getStats();
    expect(stats.capacity).toBeGreaterThanOrEqual(149); // segment capacity
  });

  it('should handle flat LoadedLinesData structure', () => {
    accumulator.fill(0, 0, {
      positions: new Float32Array([0, 0, 0, 1, 1, 1]), // 2 vertices, 3D
      segments: new Uint32Array([0, 1]), // 1 segment
      widths: new Float32Array([0.1, 0.1]), // PER-VERTEX widths
      colors: new Float32Array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), // RGB Float32
    });

    const data = accumulator.getData(1, 2); // 1 segment, 2 vertices
    expect(data.segmentCount).toBe(1);
    expect(data.vertexCount).toBe(2);
    expect(data.ndim).toBe(3);

    // Verify flat structure
    expect(data.positions.length).toBe(6); // 2 vertices * 3D
    expect(data.segments.length).toBe(2); // 1 segment * 2 indices
    expect(data.widths.length).toBe(2); // PER-VERTEX!
    expect(data.colors).not.toBeNull();
    expect(data.colors!.length).toBe(6); // 2 vertices * RGB
  });

  it('should handle nullable colors and sharpness', () => {
    accumulator.fill(0, 0, {
      positions: new Float32Array([0, 0, 0, 1, 1, 1]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      // No colors or sharpness
    });

    const data = accumulator.getData(1, 2);
    expect(data.colors).toBeNull(); // Not set, should be null
  });

  it('should track colors/sharpness presence', () => {
    // First fill without colors
    accumulator.fill(0, 0, {
      positions: new Float32Array([0, 0, 0]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1]),
    });

    let data = accumulator.getData(1, 1);
    expect(data.colors).toBeNull();

    // Second fill WITH colors
    accumulator.fill(1, 1, {
      positions: new Float32Array([1, 1, 1]),
      segments: new Uint32Array([1, 2]),
      widths: new Float32Array([0.2]),
      colors: new Float32Array([1.0, 0.0, 0.0]),
    });

    data = accumulator.getData(2, 2);
    expect(data.colors).not.toBeNull(); // Now has colors
  });

  it('should use per-vertex widths, not per-segment', () => {
    accumulator.fill(0, 0, {
      positions: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]), // 3 vertices
      segments: new Uint32Array([0, 1, 1, 2]), // 2 segments
      widths: new Float32Array([0.1, 0.2, 0.3]), // 3 widths (PER-VERTEX!)
      colors: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    });

    const data = accumulator.getData(2, 3); // 2 segments, 3 vertices
    expect(data.widths.length).toBe(3); // PER-VERTEX! Not 2 (per-segment)
    expect(data.widths[0]).toBeCloseTo(0.1, 5);
    expect(data.widths[1]).toBeCloseTo(0.2, 5);
    expect(data.widths[2]).toBeCloseTo(0.3, 5);
  });

  it('should dispose and reset', () => {
    accumulator.fill(0, 0, {
      positions: new Float32Array([0, 0, 0]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1]),
    });

    accumulator.dispose();

    const stats = accumulator.getStats();
    expect(stats.capacity).toBe(0);
  });

  // --- Boundary cases (parallel to Points / GSplats) ---

  it('ensureCapacity(0) is a no-op returning false with unchanged segment capacity', () => {
    const grew = accumulator.ensureCapacity(0);
    expect(grew).toBe(false);
    expect(accumulator.getStats().capacity).toBe(500); // segment capacity
    expect(accumulator.getStats().growthEvents).toBe(0);
  });

  it('ensureCapacity(negative) is a no-op returning false with unchanged segment capacity', () => {
    const grew = accumulator.ensureCapacity(-50);
    expect(grew).toBe(false);
    expect(accumulator.getStats().capacity).toBe(500);
    expect(accumulator.getStats().growthEvents).toBe(0);
  });

  it('fill with zero-length arrays is a no-op (capacities unchanged)', () => {
    accumulator.fill(0, 0, {
      positions: new Float32Array(0),
      segments: new Uint32Array(0),
      widths: new Float32Array(0),
    });
    const stats = accumulator.getStats();
    expect(stats.capacity).toBe(500); // segment capacity
    expect(stats.growthEvents).toBe(0);
  });

  it('zero-copy: getData positions share the same ArrayBuffer as the vertex buffer', () => {
    accumulator.fill(0, 0, {
      positions: new Float32Array([0, 0, 0, 1, 1, 1]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
    });
    const data = accumulator.getData(1, 2);
    expect(data.positions.buffer).toBe(accumulator.getVertexBuffer().buffer);
  });
});

describe('GSplatsDataAccumulator', () => {
  let accumulator: GSplatsDataAccumulator;

  beforeEach(() => {
    accumulator = new GSplatsDataAccumulator(1000, 3);
  });

  it('should initialize with correct capacity and cholesky size', () => {
    const stats = accumulator.getStats();
    expect(stats.capacity).toBe(1000);
    expect(stats.allocations).toBe(1);
    // 3D: cholesky size = (3 * 4) / 2 = 6
  });

  it('should grow capacity by 1.5x', () => {
    const grew = accumulator.ensureCapacity(1500);
    expect(grew).toBe(true);
    expect(accumulator.getStats().capacity).toBe(1500); // ceil(1000 * 1.5) = 1500
  });

  it('should use camelCase choleskyFactors', () => {
    accumulator.fill(0, {
      positions: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]), // 3D: 6 elements
      colors: new Float32Array([1.0, 0.0, 0.0]), // RGB Float32
    });

    const data = accumulator.getData(1);
    expect(data.choleskyFactors).toBeDefined(); // CORRECT: camelCase!
    expect(data.choleskyFactors.length).toBe(6); // 3D cholesky
    expect(data.splatCount).toBe(1);
    expect(data.ndim).toBe(3);
  });

  it('should handle nullable colors', () => {
    accumulator.fill(0, {
      positions: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      // No colors
    });

    const data = accumulator.getData(1);
    expect(data.colors).toBeNull();
  });

  it('should track colors presence', () => {
    // Fill with colors
    accumulator.fill(0, {
      positions: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: new Float32Array([1.0, 0.0, 0.0]),
    });

    const data = accumulator.getData(1);
    expect(data.colors).not.toBeNull();
    expect(data.colors!.length).toBe(3); // RGB
  });

  it('should handle 4D cholesky factors', () => {
    const accumulator4D = new GSplatsDataAccumulator(100, 4);
    // 4D: cholesky size = (4 * 5) / 2 = 10

    accumulator4D.fill(0, {
      positions: new Float32Array([0, 0, 0, 0]), // 4D center
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1]), // 10 elements
      colors: new Float32Array([1.0, 0.0, 0.0]),
    });

    const data = accumulator4D.getData(1);
    expect(data.choleskyFactors.length).toBe(10); // 4D cholesky
    expect(data.ndim).toBe(4);
  });

  it('should calculate memory usage correctly', () => {
    const stats = accumulator.getStats();
    // 3D: ndim(3)*4 + amp(4) + cholesky(6)*4 + color(3)*4 = 12 + 4 + 24 + 12 = 52 bytes per splat
    const expectedMB = (1000 * 52) / 1024 / 1024;
    expect(stats.memoryMB).toBeCloseTo(expectedMB, 4);
  });

  it('should dispose and reset', () => {
    accumulator.fill(0, {
      positions: new Float32Array([0, 0, 0]),
      amplitudes: new Float32Array([1.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
      colors: new Float32Array([1.0, 0.0, 0.0]),
    });

    accumulator.dispose();

    const stats = accumulator.getStats();
    expect(stats.capacity).toBe(0);
  });

  // --- Boundary cases (parallel to Points / Lines) ---

  it('ensureCapacity(0) is a no-op returning false with unchanged capacity', () => {
    const grew = accumulator.ensureCapacity(0);
    expect(grew).toBe(false);
    expect(accumulator.getStats().capacity).toBe(1000);
    expect(accumulator.getStats().growthEvents).toBe(0);
  });

  it('ensureCapacity(negative) is a no-op returning false with unchanged capacity', () => {
    const grew = accumulator.ensureCapacity(-50);
    expect(grew).toBe(false);
    expect(accumulator.getStats().capacity).toBe(1000);
    expect(accumulator.getStats().growthEvents).toBe(0);
  });

  it('fill with zero-length arrays is a no-op (capacity unchanged)', () => {
    accumulator.fill(0, {
      positions: new Float32Array(0),
      amplitudes: new Float32Array(0),
      choleskyFactors: new Float32Array(0),
    });
    const stats = accumulator.getStats();
    expect(stats.capacity).toBe(1000);
    expect(stats.growthEvents).toBe(0);
  });

  it('zero-copy: getData positions share the same ArrayBuffer as the center buffer', () => {
    accumulator.fill(0, {
      positions: new Float32Array([0, 0, 0, 1, 1, 1]),
      amplitudes: new Float32Array([1.0, 2.0]),
      choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]),
    });
    const data = accumulator.getData(2);
    expect(data.positions.buffer).toBe(accumulator.getCenterBuffer().buffer);
    // Cholesky subarray is likewise a view into the accumulator buffer.
    expect(data.choleskyFactors.buffer).toBe(accumulator.getCholeskyBuffer().buffer);
  });

  // Cholesky factor sizing: choleskySize === ndim*(ndim+1)/2 and the
  // backing buffer is sized capacity*choleskySize. Verified across a span
  // of dimensionalities (1D/2D in addition to the 3D/4D cases above, plus 5D).
  describe('cholesky factor sizing across dimensions', () => {
    for (const ndim of [1, 2, 5]) {
      it(`ndim=${ndim}: choleskySize === ndim*(ndim+1)/2 and buffer is capacity*choleskySize`, () => {
        const cap = 100;
        const expectedCholeskySize = (ndim * (ndim + 1)) / 2;
        const acc = new GSplatsDataAccumulator(cap, ndim);
        try {
          // Buffer length reflects capacity * choleskySize.
          expect(acc.getCholeskyBuffer().length).toBe(cap * expectedCholeskySize);
          // getData() returns choleskyFactors of length count * choleskySize.
          const cf = new Float32Array(expectedCholeskySize);
          acc.fill(0, {
            positions: new Float32Array(ndim),
            amplitudes: new Float32Array([1.0]),
            choleskyFactors: cf,
          });
          const data = acc.getData(1);
          expect(data.choleskyFactors.length).toBe(expectedCholeskySize);
          expect(data.choleskyFactors.length).toBe((ndim * (ndim + 1)) / 2);
        } finally {
          acc.dispose();
        }
      });
    }
  });
});

describe('accumulator dispose-state guards', () => {
  it('LoadedPointsDataAccumulator: isDisposed() flips on dispose()', () => {
    const acc = new LoadedPointsDataAccumulator(64, 3, 100);
    expect(acc.isDisposed()).toBe(false);
    acc.dispose();
    expect(acc.isDisposed()).toBe(true);
  });

  it('LoadedPointsDataAccumulator: fill() throws after dispose()', () => {
    const acc = new LoadedPointsDataAccumulator(64, 3, 100);
    acc.dispose();
    expect(() => acc.fill(0, { positions: new Float32Array([0, 0, 0]) })).toThrow(
      /called after dispose/
    );
  });

  it('LoadedPointsDataAccumulator: ensureCapacity() throws after dispose()', () => {
    const acc = new LoadedPointsDataAccumulator(64, 3, 100);
    acc.dispose();
    expect(() => acc.ensureCapacity(128)).toThrow(/called after dispose/);
  });

  it('LinesDataAccumulator: isDisposed() flips on dispose()', () => {
    const acc = new LinesDataAccumulator(64, 32, 3);
    expect(acc.isDisposed()).toBe(false);
    acc.dispose();
    expect(acc.isDisposed()).toBe(true);
  });

  it('LinesDataAccumulator: fill() throws after dispose()', () => {
    const acc = new LinesDataAccumulator(64, 32, 3);
    acc.dispose();
    expect(() =>
      acc.fill(0, 0, {
        positions: new Float32Array([0, 0, 0]),
        segments: new Uint32Array([0, 0]),
        widths: new Float32Array([0.1]),
      })
    ).toThrow(/called after dispose/);
  });

  it('GSplatsDataAccumulator: isDisposed() flips on dispose() and fill() throws', () => {
    const acc = new GSplatsDataAccumulator(64, 3);
    expect(acc.isDisposed()).toBe(false);
    acc.dispose();
    expect(acc.isDisposed()).toBe(true);
    expect(() =>
      acc.fill(0, {
        positions: new Float32Array([0, 0, 0]),
        amplitudes: new Float32Array([1]),
        choleskyFactors: new Float32Array([1, 0, 1, 0, 0, 1]),
        colors: new Float32Array([1, 0, 0]),
      })
    ).toThrow(/called after dispose/);
  });

  it('read-only getters still return empty buffers post-dispose (no throw)', () => {
    // Read getters are non-throwing so disposal assertions can inspect
    // cleared buffers. Only mutating calls (fill/ensureCapacity) throw.
    const acc = new LoadedPointsDataAccumulator(64, 3, 100);
    acc.dispose();
    expect(acc.getPositionBuffer().length).toBe(0);
    expect(acc.getColorBuffer().length).toBe(0);
    expect(acc.getRadiiBuffer().length).toBe(0);
    expect(acc.getSharpnessBuffer().length).toBe(0);
    expect(acc.getScalarBuffer().length).toBe(0);
  });
});

describe('lazy scalar buffer allocation', () => {
  it('LoadedPointsDataAccumulator: scalar buffer stays empty when no scalars fill', () => {
    const acc = new LoadedPointsDataAccumulator(1024, 3, 100);
    // Multiple fills with positions / colors / radii / sharpness — no scalars.
    for (let i = 0; i < 3; i++) {
      acc.fill(i, {
        positions: new Float32Array([i, i, i]),
        colors: new Uint8Array([255, 0, 0]),
      });
    }
    const data = acc.getData(3);
    // No `scalars` field on the output → accumulator never marked has_scalars.
    expect(data.scalars).toBeUndefined();
    // Direct buffer access without writing keeps the buffer at length 0 too,
    // EXCEPT getScalarBuffer auto-allocates on first access. Test the
    // unaccessed-via-fill path: getData with no scalars → undefined scalars
    // (which means the buffer truly hasn't been used). Buffer may still be
    // length 0 if no one called getScalarBuffer().
  });

  it('LoadedPointsDataAccumulator: scalar buffer allocates to capacity on first scalar fill', () => {
    const acc = new LoadedPointsDataAccumulator(64, 3, 100);
    // First fill has no scalars — buffer stays empty.
    acc.fill(0, { positions: new Float32Array([0, 0, 0]) });
    // Second fill carries scalars — buffer must allocate to current capacity.
    acc.fill(1, {
      positions: new Float32Array([1, 1, 1]),
      scalars: new Float32Array([0.5]),
    });
    const data = acc.getData(2);
    expect(data.scalars).toBeInstanceOf(Float32Array);
    expect(data.scalars!.length).toBe(2);
    expect(data.scalars![1]).toBeCloseTo(0.5, 5);
  });

  // [integration.md OOS4] dtypes.scalars must self-report the actual
  // on-disk dtype. Pre-fix, every non-Uint8 scalar surfaced as
  // 'float32', silently collapsing Float16 → float32 even though the
  // accumulator's internal type tracker (`PointsAccumulatorTypes.scalar`)
  // distinguished them.
  describe('LoadedPointsDataAccumulator: dtypes.scalars self-report', () => {
    it("Float32Array scalars → dtypes.scalars === 'float32'", () => {
      const acc = new LoadedPointsDataAccumulator(64, 3, 100);
      acc.fill(0, {
        positions: new Float32Array([0, 0, 0]),
        scalars: new Float32Array([0.5]),
      });
      const data = acc.getData(1);
      expect(data.metadata.dtypes!.scalars).toBe('float32');
    });

    it("Uint8Array scalars → dtypes.scalars === 'uint8'", () => {
      const acc = new LoadedPointsDataAccumulator(64, 3, 100);
      acc.fill(0, {
        positions: new Float32Array([0, 0, 0]),
        scalars: new Uint8Array([200]),
      });
      const data = acc.getData(1);
      expect(data.metadata.dtypes!.scalars).toBe('uint8');
    });

    it("Float16Array scalars → dtypes.scalars === 'float16' (no longer silently 'float32')", () => {
      // Skip on environments without Float16Array (currently Node < 22.something).
      // The accumulator's own initializeTypes branch is guarded the same way.
      // Cast through `unknown` because globalThis.Float16Array (when
      // present) has its own constructor type that does not overlap
      // with Float32Array — TypeScript rejects the direct cast.
      const F16 = (globalThis as unknown as { Float16Array?: Float16ArrayConstructor })
        .Float16Array;
      if (typeof F16 === 'undefined') {
        return;
      }

      const acc = new LoadedPointsDataAccumulator(64, 3, 100);
      acc.fill(0, {
        positions: new Float32Array([0, 0, 0]),
        scalars: new F16([0.25]),
      });
      const data = acc.getData(1);
      // The bug was: this asserted 'float32' (the silent collapse).
      // The fix surfaces the true on-disk dtype.
      expect(data.metadata.dtypes!.scalars).toBe('float16');
    });
  });

  it('LinesDataAccumulator: scalar buffer stays empty when no scalars fill', () => {
    const acc = new LinesDataAccumulator(64, 32, 3);
    acc.fill(0, 0, {
      positions: new Float32Array([0, 0, 0]),
      segments: new Uint32Array([0, 0]),
      widths: new Float32Array([0.1]),
    });
    const data = acc.getData(1, 1);
    expect(data.scalars).toBeUndefined();
  });

  it('LinesDataAccumulator: scalar buffer allocates on first scalar fill', () => {
    const acc = new LinesDataAccumulator(64, 32, 3);
    acc.fill(0, 0, {
      positions: new Float32Array([0, 0, 0]),
      segments: new Uint32Array([0, 0]),
      widths: new Float32Array([0.1]),
    });
    // Later fill brings scalars — allocation kicks in.
    acc.fill(0, 0, {
      positions: new Float32Array([1, 1, 1]),
      segments: new Uint32Array([0, 0]),
      widths: new Float32Array([0.1]),
      scalars: new Float32Array([0.5]),
    });
    const data = acc.getData(1, 1);
    expect(data.scalars).toBeInstanceOf(Float32Array);
    expect(data.scalars![0]).toBeCloseTo(0.5, 5);
  });
});

describe('accumulator growth uses usedCount subarray copy', () => {
  it('LoadedPointsDataAccumulator: growing after partial fill preserves the live prefix', () => {
    const acc = new LoadedPointsDataAccumulator(1024, 3, 100);
    // Fill 800 positions: 800 points × 3 floats = 2400 elements.
    const positions = new Float32Array(800 * 3);
    for (let i = 0; i < 800 * 3; i++) positions[i] = i + 1;
    acc.fill(0, { positions });
    // Force growth to 1500 (rounds up via 1.5× to 1536).
    const grew = acc.ensureCapacity(1500);
    expect(grew).toBe(true);
    // The 800 filled positions must survive the growth.
    const buf = acc.getPositionBuffer();
    expect(buf.length).toBeGreaterThanOrEqual(1500 * 3);
    expect(buf[0]).toBe(1);
    expect(buf[800 * 3 - 1]).toBe(800 * 3);
  });

  it('LinesDataAccumulator: vertex and segment usedCounts both track', () => {
    const acc = new LinesDataAccumulator(1024, 512, 3);
    // Fill 100 vertices and 50 segments
    const positions = new Float32Array(100 * 3);
    const segments = new Uint32Array(50 * 2);
    const widths = new Float32Array(100);
    for (let i = 0; i < 100 * 3; i++) positions[i] = i + 1;
    for (let i = 0; i < 50 * 2; i++) segments[i] = i;
    acc.fill(0, 0, { positions, segments, widths });
    const grew = acc.ensureCapacity(1500, 600);
    expect(grew).toBe(true);
    // Filled vertex positions must survive.
    expect(acc.getVertexBuffer()[0]).toBe(1);
    expect(acc.getVertexBuffer()[100 * 3 - 1]).toBe(100 * 3);
    // Filled segment indices must survive.
    expect(acc.getSegmentBuffer()[0]).toBe(0);
    expect(acc.getSegmentBuffer()[50 * 2 - 1]).toBe(50 * 2 - 1);
  });

  it('GSplatsDataAccumulator: growing after partial fill preserves the live prefix', () => {
    const acc = new GSplatsDataAccumulator(1024, 3);
    const positions = new Float32Array(300 * 3);
    const amplitudes = new Float32Array(300);
    const choleskyFactors = new Float32Array(300 * 6); // 3D → 6 cholesky elements
    for (let i = 0; i < 300; i++) {
      positions[i * 3] = i;
      amplitudes[i] = i + 0.5;
    }
    acc.fill(0, { positions, amplitudes, choleskyFactors });
    const grew = acc.ensureCapacity(1500);
    expect(grew).toBe(true);
    const out = acc.getData(300);
    expect(out.positions[0]).toBe(0);
    expect(out.positions[(300 - 1) * 3]).toBe(299);
    expect(out.amplitudes[299]).toBeCloseTo(299.5, 5);
  });
});

// ensureCapacity — algebraic invariants under arbitrary growth requests.
// Pins:
//   * post-condition: capacity >= n on success
//   * monotone growth: capacity never shrinks
//   * idempotent at-or-below: ensureCapacity(<= currentCapacity) is a no-op
//   * prefix preservation: filled prefix survives any number of growths
describe('LoadedPointsDataAccumulator — ensureCapacity property invariants', () => {
  test('capacity >= n after ensureCapacity(n) for any reasonable n', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 50000 }), (n) => {
        const acc = new LoadedPointsDataAccumulator(100, 3, 100000);
        try {
          acc.ensureCapacity(n);
          expect(acc.getStats().capacity).toBeGreaterThanOrEqual(n);
        } finally {
          acc.dispose();
        }
      }),
      { numRuns: 25, seed: 0x5eed }
    );
  });

  test('monotone: capacity never shrinks across a sequence of ensureCapacity calls', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 1, max: 20000 }), { minLength: 1, maxLength: 10 }),
        (requests) => {
          const acc = new LoadedPointsDataAccumulator(100, 3, 100000);
          try {
            let prevCap = acc.getStats().capacity;
            for (const r of requests) {
              acc.ensureCapacity(r);
              const newCap = acc.getStats().capacity;
              expect(newCap).toBeGreaterThanOrEqual(prevCap);
              prevCap = newCap;
            }
          } finally {
            acc.dispose();
          }
        }
      ),
      { numRuns: 15, seed: 0x5eed }
    );
  });

  test('idempotent at-or-below: ensureCapacity(<= currentCapacity) returns false and does not change capacity', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 100, max: 10000 }),
        fc.integer({ min: 1, max: 100 }),
        (initialCap, request) => {
          const acc = new LoadedPointsDataAccumulator(initialCap, 3, 100000);
          try {
            // request <= initialCap (we'll force it within bounds).
            const safeRequest = Math.min(request, initialCap);
            const grew = acc.ensureCapacity(safeRequest);
            expect(grew).toBe(false);
            expect(acc.getStats().capacity).toBe(initialCap);
          } finally {
            acc.dispose();
          }
        }
      ),
      { numRuns: 15, seed: 0x5eed }
    );
  });

  test('growth monotonicity: for an increasing target sequence, capacity is non-decreasing and always >= the requested target', () => {
    fc.assert(
      fc.property(
        // A set of distinct positive targets; sorting ascending gives an
        // increasing request sequence.
        fc.uniqueArray(fc.integer({ min: 1, max: 100000 }), {
          minLength: 1,
          maxLength: 12,
        }),
        (rawTargets) => {
          const targets = [...rawTargets].sort((a, b) => a - b);
          const acc = new LoadedPointsDataAccumulator(50, 3, 1000000);
          try {
            let prevCap = acc.getStats().capacity;
            for (const target of targets) {
              acc.ensureCapacity(target);
              const cap = acc.getStats().capacity;
              // Non-decreasing capacity across the increasing sequence.
              expect(cap).toBeGreaterThanOrEqual(prevCap);
              // Always satisfies the request.
              expect(cap).toBeGreaterThanOrEqual(target);
              prevCap = cap;
            }
          } finally {
            acc.dispose();
          }
        }
      ),
      { numRuns: 100, seed: 0x5eed }
    );
  });

  test('prefix preservation: filled positions survive any growth', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 10, max: 500 }),
        fc.integer({ min: 600, max: 5000 }),
        (fillCount, growTo) => {
          const acc = new LoadedPointsDataAccumulator(1024, 3, 100000);
          try {
            // Fill `fillCount` positions with monotone values.
            const positions = new Float32Array(fillCount * 3);
            for (let i = 0; i < positions.length; i++) positions[i] = i + 1;
            acc.fill(0, { positions });
            acc.ensureCapacity(growTo);
            // The first `fillCount` positions must be exactly what we wrote.
            const buf = acc.getPositionBuffer();
            for (let i = 0; i < positions.length; i++) {
              expect(buf[i]).toBe(i + 1);
            }
          } finally {
            acc.dispose();
          }
        }
      ),
      { numRuns: 12, seed: 0x5eed }
    );
  });
});

describe('LoadedPointsDataAccumulator — RGBA color layout (configureColorComponents)', () => {
  // The color layout (3 = RGB, 4 = RGBA) is a property of the dataset,
  // declared by the loader BEFORE the first color fill so every color
  // buffer (initial, type-repinned, grown) is sized at the right stride.
  // Mirrors LoadedGSplatsDataAccumulator.configureColorComponents.

  function rgbaColors(count: number): Float32Array {
    const colors = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      colors[i * 4] = 0.1 + i;
      colors[i * 4 + 1] = 0.2 + i;
      colors[i * 4 + 2] = 0.3 + i;
      colors[i * 4 + 3] = 0.5 / (i + 1); // distinct per-point alpha
    }
    return colors;
  }

  function positions(count: number): Float32Array {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < pos.length; i++) pos[i] = i;
    return pos;
  }

  it('configureColorComponents(4) before fills → getData carries colorComponents=4 and a count×4 colors view', () => {
    const acc = new LoadedPointsDataAccumulator(8, 3, 100);
    acc.configureColorComponents(4);
    acc.fill(0, { positions: positions(3), colors: rgbaColors(3) });
    const data = acc.getData(3);
    expect(data.colorComponents).toBe(4);
    expect(data.colors).toBeInstanceOf(Float32Array);
    expect(data.colors!.length).toBe(3 * 4);
    // Each point's own RGBA tuple lands stride-aligned (a 3-strided buffer
    // would smear point 1+ and drop the alphas).
    for (let i = 0; i < 3; i++) {
      expect(data.colors![i * 4]).toBeCloseTo(0.1 + i, 6);
      expect(data.colors![i * 4 + 2]).toBeCloseTo(0.3 + i, 6);
      expect(data.colors![i * 4 + 3]).toBeCloseTo(0.5 / (i + 1), 6);
    }
  });

  it('growth past capacity preserves the RGBA stride and the filled tuples', () => {
    const acc = new LoadedPointsDataAccumulator(4, 3, 100);
    acc.configureColorComponents(4);
    acc.fill(0, { positions: positions(4), colors: rgbaColors(4) });
    // Grow beyond the initial capacity — the color copy must move the
    // live prefix at stride 4 (a stride-3 liveColor would truncate it).
    expect(acc.ensureCapacity(10)).toBe(true);
    const data = acc.getData(4);
    expect(data.colorComponents).toBe(4);
    expect(data.colors!.length).toBe(4 * 4);
    for (let i = 0; i < 4; i++) {
      expect(data.colors![i * 4]).toBeCloseTo(0.1 + i, 6);
      expect(data.colors![i * 4 + 3]).toBeCloseTo(0.5 / (i + 1), 6);
    }
  });

  it('throws when the layout changes AFTER colors were already written', () => {
    const acc = new LoadedPointsDataAccumulator(8, 3, 100);
    acc.fill(0, {
      positions: positions(2),
      colors: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), // RGB
    });
    expect(() => acc.configureColorComponents(4)).toThrow(
      /color layout changed to 4 components after colors were already written with 3/
    );
  });

  it('re-declaring the SAME layout is a no-op (even after fills)', () => {
    const acc = new LoadedPointsDataAccumulator(8, 3, 100);
    acc.fill(0, {
      positions: positions(1),
      colors: new Float32Array([0.1, 0.2, 0.3]), // RGB
    });
    const allocationsBefore = acc.getStats().allocations;
    expect(() => acc.configureColorComponents(3)).not.toThrow();
    expect(acc.getStats().allocations).toBe(allocationsBefore); // no re-size
    // Same for RGBA: a repeat declaration after the first is a no-op too.
    const acc4 = new LoadedPointsDataAccumulator(8, 3, 100);
    acc4.configureColorComponents(4);
    acc4.fill(0, { positions: positions(1), colors: rgbaColors(1) });
    expect(() => acc4.configureColorComponents(4)).not.toThrow();
    expect(acc4.getData(1).colorComponents).toBe(4);
  });
});

describe('LinesDataAccumulator — RGBA color layout (configureColorComponents)', () => {
  // The lines twin of the points suite above: the color layout (3 = RGB,
  // 4 = RGBA — alpha = per-vertex opacity, volumetric phase 4) is a
  // property of the dataset, declared by the loader BEFORE the first
  // color fill so every color buffer is sized at the right stride.

  function rgbaVertexColors(count: number): Float32Array {
    const colors = new Float32Array(count * 4);
    for (let i = 0; i < count; i++) {
      colors[i * 4] = 0.1 + i;
      colors[i * 4 + 1] = 0.2 + i;
      colors[i * 4 + 2] = 0.3 + i;
      colors[i * 4 + 3] = 0.5 / (i + 1); // distinct per-vertex alpha
    }
    return colors;
  }

  function vertexPositions(count: number, ndim = 3): Float32Array {
    const pos = new Float32Array(count * ndim);
    for (let i = 0; i < pos.length; i++) pos[i] = i;
    return pos;
  }

  it('configureColorComponents(4) before fills → getData carries colorComponents=4 and a count×4 colors view', () => {
    const acc = new LinesDataAccumulator(8, 4, 3);
    acc.configureColorComponents(4);
    acc.fill(0, 0, {
      positions: vertexPositions(3),
      segments: new Uint32Array([0, 1, 1, 2]),
      colors: rgbaVertexColors(3),
    });
    const data = acc.getData(2, 3);
    expect(data.colorComponents).toBe(4);
    expect(data.colors).toBeInstanceOf(Float32Array);
    expect(data.colors!.length).toBe(3 * 4);
    for (let i = 0; i < 3; i++) {
      expect(data.colors![i * 4]).toBeCloseTo(0.1 + i, 6);
      expect(data.colors![i * 4 + 2]).toBeCloseTo(0.3 + i, 6);
      expect(data.colors![i * 4 + 3]).toBeCloseTo(0.5 / (i + 1), 6);
    }
  });

  it('growth past capacity preserves the RGBA stride and the filled tuples', () => {
    const acc = new LinesDataAccumulator(4, 2, 3);
    acc.configureColorComponents(4);
    acc.fill(0, 0, {
      positions: vertexPositions(4),
      segments: new Uint32Array([0, 1, 2, 3]),
      colors: rgbaVertexColors(4),
    });
    // Grow beyond the initial vertex capacity — the color copy must move
    // the live prefix at stride 4 (a stride-3 liveColor would truncate it).
    expect(acc.ensureCapacity(10, 5)).toBe(true);
    const data = acc.getData(2, 4);
    expect(data.colorComponents).toBe(4);
    expect(data.colors!.length).toBe(4 * 4);
    for (let i = 0; i < 4; i++) {
      expect(data.colors![i * 4]).toBeCloseTo(0.1 + i, 6);
      expect(data.colors![i * 4 + 3]).toBeCloseTo(0.5 / (i + 1), 6);
    }
  });

  it('RGB fills emit colorComponents=3 (the layout is explicit, never inferred wrong)', () => {
    const acc = new LinesDataAccumulator(8, 4, 3);
    acc.fill(0, 0, {
      positions: vertexPositions(2),
      segments: new Uint32Array([0, 1]),
      colors: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), // RGB
    });
    const data = acc.getData(1, 2);
    expect(data.colorComponents).toBe(3);
    expect(data.colors!.length).toBe(2 * 3);
  });

  it('throws when the layout changes AFTER colors were already written', () => {
    const acc = new LinesDataAccumulator(8, 4, 3);
    acc.fill(0, 0, {
      positions: vertexPositions(2),
      segments: new Uint32Array([0, 1]),
      colors: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), // RGB
    });
    expect(() => acc.configureColorComponents(4)).toThrow(
      /color layout changed to 4 components after colors were already written with 3/
    );
  });

  it('Uint8 RGBA colors keep dtype AND stride through type re-pinning', () => {
    const acc = new LinesDataAccumulator(8, 4, 3);
    acc.configureColorComponents(4);
    acc.fill(0, 0, {
      positions: vertexPositions(2),
      segments: new Uint32Array([0, 1]),
      colors: new Uint8Array([10, 20, 30, 217, 40, 50, 60, 128]),
    });
    const data = acc.getData(1, 2);
    expect(data.colorComponents).toBe(4);
    expect(data.colors).toBeInstanceOf(Uint8Array);
    expect(Array.from(data.colors!)).toEqual([10, 20, 30, 217, 40, 50, 60, 128]);
  });
});
