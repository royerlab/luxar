/**
 * Property test: the GPU buffer pool's INCREMENTAL byte totals (B9b, #2944)
 * always equal a from-scratch recomputation.
 *
 * `getResidentBytes()` used to walk every active and pooled buffer on each
 * query; it now reads totals kept by `ActiveBufferMap` / `FreeBucketMap`
 * (`rendering/gpu-buffer-pool/byte-tracked-maps.ts`). This drives the pool
 * through 10,000 seeded random operations covering every path that moves a
 * buffer — acquire (fresh, in-place reuse, best-fit adoption, grow), release,
 * commit ageing, LRU and byte-budget sweeps (with the budget changing under
 * it), out-of-band disposal of active and pooled geometries, dispose
 * listeners that THROW mid-sweep, and whole-pool disposal — and after every
 * operation compares the incremental totals against a walk over the pool's
 * actual contents, and against `getStats()` (which still walks).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { GPUBufferPool, estimateGeometryBytes } from '../../../rendering/gpu-buffer-pool';
import type { PooledGeometryType } from '../../../types/data-monitor-types';

/** Tiny deterministic PRNG (mulberry32) — reproducible from a seed. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const TYPES: readonly PooledGeometryType[] = ['points', 'lines', 'gsplats'];
const NODES_PER_TYPE = 10;

/** From-scratch totals: walk what the pool actually holds. */
function recompute(pool: GPUBufferPool): { active: number; pooled: number } {
  let active = 0;
  for (const buffer of pool.activeBuffers.values()) {
    active += estimateGeometryBytes(buffer.geometry);
  }
  let pooled = 0;
  for (const buckets of [
    pool.points.pointBuffers,
    pool.lines.lineBuffers,
    pool.gsplats.gsplatBuffers,
  ]) {
    for (const buffers of buckets.values()) {
      for (const buffer of buffers) pooled += estimateGeometryBytes(buffer.geometry);
    }
  }
  return { active, pooled };
}

function acquire(
  pool: GPUBufferPool,
  type: PooledGeometryType,
  nodeId: string,
  count: number
): THREE.BufferGeometry {
  if (type === 'points') return pool.acquirePointsGeometry(nodeId, count);
  if (type === 'lines') return pool.acquireLinesGeometry(nodeId, count);
  return pool.acquireGSplatsGeometry(nodeId, count);
}

function release(pool: GPUBufferPool, type: PooledGeometryType, nodeId: string): void {
  if (type === 'points') pool.releasePointsGeometry(nodeId);
  else if (type === 'lines') pool.releaseLinesGeometry(nodeId);
  else pool.releaseGSplatsGeometry(nodeId);
}

/** Every pooled (free-bucket) buffer, in a stable order. */
function pooledGeometries(pool: GPUBufferPool): THREE.BufferGeometry[] {
  const out: THREE.BufferGeometry[] = [];
  for (const buckets of [
    pool.points.pointBuffers,
    pool.lines.lineBuffers,
    pool.gsplats.gsplatBuffers,
  ]) {
    for (const buffers of buckets.values()) {
      for (const buffer of buffers) out.push(buffer.geometry);
    }
  }
  return out;
}

/** Operation mix, by cumulative probability. */
type Op =
  | 'acquire'
  | 'release'
  | 'commit'
  | 'sweep'
  | 'budget'
  | 'disposeActive'
  | 'disposePooled'
  | 'throwingListener'
  | 'disposePool';
const OP_WEIGHTS: ReadonlyArray<[Op, number]> = [
  ['acquire', 0.38],
  ['release', 0.28],
  ['commit', 0.12],
  ['sweep', 0.07],
  ['budget', 0.05],
  ['disposeActive', 0.03],
  ['disposePooled', 0.03],
  ['throwingListener', 0.035],
  ['disposePool', 0.005],
];

function pickOp(r: number): Op {
  let acc = 0;
  for (const [op, weight] of OP_WEIGHTS) {
    acc += weight;
    if (r < acc) return op;
  }
  return 'acquire';
}

/** Budgets straddling the pool's working set, plus 0 (byte pass disabled). */
const BUDGETS = [0, 150_000, 400_000, 1_000_000, 4_000_000, 64_000_000];

function runOps(seed: number, operations: number): { applied: Record<Op, number>; threw: number } {
  const random = rng(seed);
  let budget = 1_000_000;
  // Small count limit and short age so the LRU and mustEvict paths fire often.
  const pool = new GPUBufferPool(/* maxPoolSize */ 6, /* evictionCommits */ 4, 2, () => budget);
  const applied = Object.fromEntries(OP_WEIGHTS.map(([op]) => [op, 0])) as Record<Op, number>;
  let threw = 0;

  const pick = <T>(items: readonly T[]): T => items[Math.floor(random() * items.length)];

  for (let step = 0; step < operations; step++) {
    const op = pickOp(random());
    const type = pick(TYPES);
    const nodeId = `/${type}/${Math.floor(random() * NODES_PER_TYPE)}`;
    applied[op]++;
    try {
      switch (op) {
        case 'acquire':
          // Wide count range: in-place reuse, best-fit adoption and grow.
          acquire(pool, type, nodeId, 1 + Math.floor(random() ** 2 * 6000));
          break;
        case 'release':
          release(pool, type, nodeId);
          break;
        case 'commit':
          pool.beginCommit();
          break;
        case 'sweep':
          pool.evictUnused(random() < 0.5);
          break;
        case 'budget':
          budget = pick(BUDGETS);
          break;
        case 'disposeActive': {
          const active = [...pool.activeBuffers.values()];
          if (active.length > 0) pick(active).geometry.dispose();
          break;
        }
        case 'disposePooled': {
          const pooled = pooledGeometries(pool);
          if (pooled.length > 0) pick(pooled).dispose();
          break;
        }
        case 'throwingListener': {
          // A dispose listener that throws aborts whichever sweep disposes
          // this geometry — the totals must still be exact afterwards.
          const active = [...pool.activeBuffers.values()];
          if (active.length > 0) {
            pick(active).geometry.addEventListener('dispose', () => {
              throw new Error('synthetic dispose failure');
            });
          }
          break;
        }
        case 'disposePool':
          pool.dispose();
          break;
      }
    } catch (error) {
      if (!(error instanceof Error) || error.message !== 'synthetic dispose failure') throw error;
      threw++;
    }

    const truth = recompute(pool);
    const stats = pool.getStats();
    const resident = pool.getResidentBytes();
    if (resident !== truth.active + truth.pooled || stats.totalBytes !== resident) {
      throw new Error(
        `seed ${seed} step ${step} (${op} ${nodeId}): resident ${resident}, ` +
          `from scratch active ${truth.active} + pooled ${truth.pooled}, ` +
          `getStats ${stats.activeBytes} + ${stats.pooledBytes}`
      );
    }
    expect(stats.activeBytes).toBe(truth.active);
    expect(stats.pooledBytes).toBe(truth.pooled);
  }
  return { applied, threw };
}

describe('GPUBufferPool incremental byte totals (B9b)', () => {
  it('equal a from-scratch recomputation after each of 10,000 random operations', () => {
    const { applied, threw } = runOps(0x2944, 10_000);
    // The run must actually have exercised the rare paths, or it proves little.
    for (const [op] of OP_WEIGHTS) expect(applied[op], op).toBeGreaterThan(0);
    expect(threw).toBeGreaterThan(0);
  });

  it('hold across several independent seeds', () => {
    for (const seed of [1, 7, 12345]) runOps(seed, 2_000);
  });
});
