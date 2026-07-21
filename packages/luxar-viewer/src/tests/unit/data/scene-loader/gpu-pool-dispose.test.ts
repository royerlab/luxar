/**
 * SceneLoader.dispose() must dispose its GPU buffer pool.
 *
 * Without this, dataset switches retained active+pooled
 * InstancedBufferGeometry references in the pool — at million-element
 * scale that's hundreds of MB of GPU memory leaked per dataset switch.
 */
import { describe, it, expect } from 'vitest';
import { GPUBufferPool } from '../../../../rendering/gpu-buffer-pool';

describe('GPUBufferPool.dispose', () => {
  it('clears active buffers + pooled buckets', () => {
    const pool = new GPUBufferPool(20, 300);
    // Allocate one points geometry so there's something to dispose.
    pool.acquirePointsGeometry('p1', 2);

    expect(pool.getStats().activeBuffers).toBeGreaterThan(0);
    pool.dispose();
    // After dispose, no active or pooled buffers.
    const stats = pool.getStats();
    expect(stats.activeBuffers).toBe(0);
    expect(stats.pooledBuffers).toBe(0);
  });

  it('is idempotent (safe to call twice)', () => {
    const pool = new GPUBufferPool(20, 300);
    expect(() => {
      pool.dispose();
      pool.dispose();
    }).not.toThrow();
  });
});
