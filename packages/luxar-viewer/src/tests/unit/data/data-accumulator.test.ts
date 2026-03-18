/**
 * Unit tests for Data Accumulators
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  LoadedPointsDataAccumulator,
  LinesDataAccumulator,
  GSplatsDataAccumulator,
} from '../../../data/data-accumulator';
import * as THREE from 'three';

describe('LoadedPointsDataAccumulator', () => {
  let accumulator: LoadedPointsDataAccumulator;

  beforeEach(() => {
    accumulator = new LoadedPointsDataAccumulator(1000, 3, 10000);
  });

  it('should initialize with correct capacity', () => {
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
    accumulator.setHDRMode(true);
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
    // ensureCapacity with only vertex count estimates segments
    // Initial capacity: 1000 vertices, 500 segments
    const grew = accumulator.ensureCapacity(1200); // 1200 vertices (exceeds 1000)
    expect(grew).toBe(true);
    // Vertices: 1000 → 1500 (grew)
    // Segments: estimated as ceil(1200/1.5) = 800, exceeds 500 → grows
    const stats = accumulator.getStats();
    expect(stats.capacity).toBeGreaterThanOrEqual(800); // segment capacity
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
});
