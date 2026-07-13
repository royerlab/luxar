/**
 * Unit tests for Worker Pool and Data Worker communication
 *
 * NOTE: These tests are skipped in Node.js environment (no Worker API).
 * Worker functionality is covered by:
 * - E2E tests: worker-wasm-integration.spec.ts (verifies pool creation, WASM loading, fallback)
 * - Integration tests: worker-integration.test.ts (verifies query interface and fallback logic)
 */

import { describe, it, expect, afterEach } from 'vitest';
import { getWorkerPool, disposeWorkerPool } from '../../../../workers/worker-pool';

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

  // [R11/A-C1][P2] The prior `if (workerCount <= 1) === else !==` branching made
  // both arms vacuously pass — a getWorker() that always returned a fresh stub
  // would satisfy the !== branch, and one that always returned the same stub
  // would satisfy the === branch. Replaced by an unconditional contract:
  // every call returns a valid worker; the set of distinct workers across K
  // calls is constrained by the configured pool size; with a single-worker
  // pool, all calls return the same instance.
  it('should distribute getWorker() calls within the configured pool size', async () => {
    const pool = getWorkerPool();
    const calls = await Promise.all(Array.from({ length: 6 }, () => pool.getWorker()));
    const { workerCount } = pool.getStats();

    expect(workerCount).toBeGreaterThanOrEqual(1);
    for (const w of calls) {
      expect(w).toBeDefined();
      expect(typeof (w as { projectLinesTo3D?: unknown }).projectLinesTo3D).toBe('function');
    }
    const distinct = new Set(calls);
    expect(distinct.size).toBeGreaterThanOrEqual(1);
    expect(distinct.size).toBeLessThanOrEqual(workerCount);
    if (workerCount === 1) {
      expect(distinct.size).toBe(1);
    }
  });

  // [R11/A-C2][P2] Same vacuous-branching pattern; replaced with the same
  // unconditional contract plus an explicit `isInitialized` post-condition
  // verifying the pool initialised exactly once under contention.
  it('should handle concurrent initialization requests', async () => {
    const pool = getWorkerPool();
    // Start multiple initialization requests simultaneously
    const workers = await Promise.all([pool.getWorker(), pool.getWorker(), pool.getWorker()]);

    const { workerCount } = pool.getStats();

    for (const w of workers) {
      expect(w).toBeDefined();
      expect(typeof (w as { projectLinesTo3D?: unknown }).projectLinesTo3D).toBe('function');
    }
    const distinct = new Set(workers);
    expect(distinct.size).toBeGreaterThanOrEqual(1);
    expect(distinct.size).toBeLessThanOrEqual(workerCount);
    expect(pool.isInitialized()).toBe(true);
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
});
