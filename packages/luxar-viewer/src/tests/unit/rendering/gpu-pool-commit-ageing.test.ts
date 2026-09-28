/**
 * The GPU buffer pool ages pooled (released, not yet disposed) buffers in
 * ATOMIC COMMITS, not rendered frames (#2939).
 *
 * The only production caller of `beginCommit()` is the atomic commit
 * (`data/scene-loader/update-view/atomic-commit.ts`), and buffers are only
 * acquired and released on commits, so counting commits is the policy. These
 * tests pin that unit so the names and docs cannot drift back to "frames":
 *
 *  - a released buffer survives `evictionCommits` commits and is disposed on
 *    the first sweep after it has gone strictly MORE than that many commits
 *    unused (default 300: 299 or 300 keep it, 301 disposes it);
 *  - pool activity with NO commit (what an orbiting or idle view produces:
 *    render frames, sweeps, other nodes' acquire/release) never ages it.
 */

import { describe, it, expect } from 'vitest';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';

/** Default-policy pool with the byte pass disabled so only LRU ageing acts. */
function makePool(): GPUBufferPool {
  return new GPUBufferPool(/* maxPoolSize */ 20, /* evictionCommits */ 300, 5, () => 0);
}

/**
 * Release a small points buffer into the free list and return a probe that
 * reports whether the pool has since disposed it.
 */
function releaseVictim(pool: GPUBufferPool): () => boolean {
  const geometry = pool.acquirePointsGeometry('/victim', 100);
  let disposed = false;
  geometry.addEventListener('dispose', () => {
    disposed = true;
  });
  pool.releasePointsGeometry('/victim');
  return () => disposed;
}

/**
 * One commit's worth of pool traffic: advance the commit counter, then
 * acquire and release a node too large to adopt the victim. The release
 * runs the LRU sweep that would dispose an aged victim.
 */
function commitWithTraffic(pool: GPUBufferPool): void {
  pool.beginCommit();
  pool.acquirePointsGeometry('/other', 5000);
  pool.releasePointsGeometry('/other');
}

describe('GPUBufferPool ageing is counted in commits (#2939)', () => {
  it('keeps a released buffer through 299 commits', () => {
    const pool = makePool();
    const isDisposed = releaseVictim(pool);
    for (let i = 0; i < 299; i++) commitWithTraffic(pool);
    expect(pool.commitCount).toBe(299);
    expect(isDisposed()).toBe(false);
  });

  it('keeps it at exactly evictionCommits (300) — the threshold is strict', () => {
    const pool = makePool();
    const isDisposed = releaseVictim(pool);
    for (let i = 0; i < 300; i++) commitWithTraffic(pool);
    expect(isDisposed()).toBe(false);
  });

  it('disposes it on the sweep after 301 commits', () => {
    const pool = makePool();
    const isDisposed = releaseVictim(pool);
    for (let i = 0; i < 301; i++) commitWithTraffic(pool);
    expect(isDisposed()).toBe(true);
    // The traffic node's own buffer was released THIS commit, so it is fresh.
    expect(pool.getStats().pooledBuffers).toBe(1);
  });

  it('does not age a pooled buffer across render frames without a commit', () => {
    // An orbiting or idle view renders frames but runs no atomic commit.
    // Whatever the pool sees in that time — sweeps and other nodes'
    // acquire/release — must leave the pooled buffer's age at zero.
    const pool = makePool();
    const isDisposed = releaseVictim(pool);
    for (let frame = 0; frame < 1000; frame++) {
      pool.evictUnused();
      pool.acquirePointsGeometry('/other', 5000);
      pool.releasePointsGeometry('/other');
    }
    expect(pool.commitCount).toBe(0);
    expect(isDisposed()).toBe(false);

    // The age starts counting only once commits resume.
    for (let i = 0; i < 301; i++) commitWithTraffic(pool);
    expect(isDisposed()).toBe(true);
  });
});
