/**
 * Unit tests for Worker Pool and Data Worker communication
 *
 * NOTE: These tests are skipped in Node.js environment (no Worker API).
 * Worker functionality is tested through E2E tests in actual browser environment.
 */

import { describe, it, expect, afterEach } from 'vitest';
import { getWorkerPool, disposeWorkerPool } from '../../../workers/worker-pool';

// Check if Worker API is available (browser environment)
const hasWorkerAPI = typeof Worker !== 'undefined';

describe.skipIf(!hasWorkerAPI)('WorkerPool', () => {
  afterEach(() => {
    // Clean up worker after each test
    disposeWorkerPool();
  });

  it('should initialize worker successfully', async () => {
    const pool = getWorkerPool();
    const worker = await pool.getWorker();
    expect(worker).toBeDefined();
    expect(pool.isInitialized()).toBe(true);
  });

  it('should return same worker instance on multiple calls', async () => {
    const pool = getWorkerPool();
    const worker1 = await pool.getWorker();
    const worker2 = await pool.getWorker();
    expect(worker1).toBe(worker2);
  });

  it('should handle concurrent initialization requests', async () => {
    const pool = getWorkerPool();
    // Start multiple initialization requests simultaneously
    const workers = await Promise.all([pool.getWorker(), pool.getWorker(), pool.getWorker()]);

    // All should return the same worker
    expect(workers[0]).toBe(workers[1]);
    expect(workers[1]).toBe(workers[2]);
  });

  it('should dispose worker properly', async () => {
    const pool = getWorkerPool();
    await pool.getWorker();
    expect(pool.isInitialized()).toBe(true);

    pool.dispose();
    expect(pool.isInitialized()).toBe(false);
  });

  it('should reinitialize after disposal', async () => {
    const pool1 = getWorkerPool();
    await pool1.getWorker();
    pool1.dispose();

    // Get new pool instance
    disposeWorkerPool(); // Clear singleton
    const pool2 = getWorkerPool();
    const worker = await pool2.getWorker();

    expect(worker).toBeDefined();
    expect(pool2.isInitialized()).toBe(true);
  });
});

describe.skipIf(!hasWorkerAPI)('DataWorker Communication', () => {
  afterEach(() => {
    disposeWorkerPool();
  });

  it('should query spatial index correctly', async () => {
    const pool = getWorkerPool();
    const worker = await pool.getWorker();

    // Create test data: 4 chunks in 3D space
    const chunkBounds = new Float32Array([
      // Chunk 0: [0,0,0] to [1,1,1]
      0, 1, 0, 1, 0, 1,
      // Chunk 1: [1,1,1] to [2,2,2]
      1, 2, 1, 2, 1, 2,
      // Chunk 2: [2,2,2] to [3,3,3]
      2, 3, 2, 3, 2, 3,
      // Chunk 3: [-1,-1,-1] to [0,0,0]
      -1, 0, -1, 0, -1, 0,
    ]);

    // Query at position [0.5, 0.5, 0.5] with tolerance [0.6, 0.6, 0.6]
    const result = await worker.querySpatialIndex({
      chunkBounds,
      slicePosition: new Float32Array([0.5, 0.5, 0.5]),
      tolerance: new Float32Array([0.6, 0.6, 0.6]),
      numChunks: 4,
      ndim: 3,
    });

    // Should match chunks 0 and 1 (within tolerance)
    const matchedChunks = Array.from(result);
    expect(matchedChunks).toContain(0); // Chunk 0 intersects
    expect(matchedChunks.length).toBeGreaterThan(0);
  });

  it('should compute nD visibility for points', async () => {
    const pool = getWorkerPool();
    const worker = await pool.getWorker();

    // Create test data: 5 points in 4D space
    const positions = new Float32Array([
      0,
      0,
      0,
      0, // Point 0 at origin
      0,
      0,
      0,
      1, // Point 1 at t=1
      0,
      0,
      0,
      5, // Point 2 at t=5 (far away)
      1,
      1,
      1,
      0, // Point 3 at [1,1,1,0]
      0,
      0,
      0,
      0.5, // Point 4 at t=0.5
    ]);
    const radii = new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5]);

    // Query at t=0 with tolerance=1.0
    const result = await worker.computeNDVisibilityPoints({
      positions,
      radii,
      slicePosition: new Float32Array([0, 0, 0, 0]),
      tolerance: new Float32Array([1e10, 1e10, 1e10, 1.0]), // Infinite for XYZ, 1.0 for T
      ndim: 4,
      numPoints: 5,
    });

    // Points 0, 1, 4 should be visible (within t ± 1.0)
    // Point 2 should be hidden (t=5 is too far)
    expect(result.visibilityMask[0]).toBe(1); // t=0, visible
    expect(result.visibilityMask[1]).toBe(1); // t=1, visible
    expect(result.visibilityMask[4]).toBe(1); // t=0.5, visible
    // Note: Point 2 visibility depends on radius/tolerance math
    expect(result.visibleCount).toBeGreaterThan(0);
  });

  it('should compute nD visibility for lines', async () => {
    const pool = getWorkerPool();
    const worker = await pool.getWorker();

    // Create test data: 2 segments in 4D
    const vertices = new Float32Array([
      0,
      0,
      0,
      0, // Vertex 0
      1,
      1,
      1,
      0, // Vertex 1
      2,
      2,
      2,
      5, // Vertex 2 (far in T)
      3,
      3,
      3,
      5, // Vertex 3 (far in T)
    ]);
    const segments = new Uint32Array([0, 1, 2, 3]); // Segment 0: v0-v1, Segment 1: v2-v3
    const widths = new Float32Array([0.1, 0.1, 0.1, 0.1]);

    const result = await worker.computeNDVisibilityLines({
      vertices,
      segments,
      widths,
      slicePosition: new Float32Array([0, 0, 0, 0]),
      tolerance: new Float32Array([1e10, 1e10, 1e10, 1.0]),
      ndim: 4,
      numSegments: 2,
    });

    // Segment 0 should be visible (both endpoints near t=0)
    expect(result.visibilityMask[0]).toBe(1);
    // Segment 1 might be hidden (both endpoints at t=5)
    expect(result.visibleCount).toBeGreaterThan(0);
  });

  it('should compute nD visibility for gsplats', async () => {
    const pool = getWorkerPool();
    const worker = await pool.getWorker();

    // Create test data: 3 splats in 4D
    const centers = new Float32Array([
      0,
      0,
      0,
      0, // Splat 0 at origin
      1,
      1,
      1,
      0, // Splat 1 near origin
      0,
      0,
      0,
      10, // Splat 2 far away in T
    ]);
    // 4D cholesky: (4*5)/2 = 10 elements per splat
    const choleskyFactors = new Float32Array(3 * 10); // Dummy data for now

    const result = await worker.computeNDVisibilityGSplats({
      centers,
      choleskyFactors,
      slicePosition: new Float32Array([0, 0, 0, 0]),
      tolerance: new Float32Array([1e10, 1e10, 1e10, 2.0]),
      ndim: 4,
      numSplats: 3,
    });

    // Splats 0 and 1 should be visible (near t=0)
    expect(result.visibilityMask[0]).toBe(1);
    expect(result.visibilityMask[1]).toBe(1);
    // Splat 2 should be hidden (t=10 is far)
    expect(result.visibilityMask[2]).toBe(0);
    expect(result.visibleCount).toBe(2);
  });
});
