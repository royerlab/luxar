/**
 * Direct tests for the pure `selectBuffersToEvict` eviction selector.
 *
 * Previously lived inside `gpu-pool-byte-budget.test.ts`. Moved here
 * alongside the extracted module so the policy can be unit-tested
 * without pulling in the GPU pool, its byte-counter helpers, or
 * THREE.js stubs.
 */

import { describe, it, expect } from 'vitest';
import { selectBuffersToEvict } from '../../../../rendering/gpu-buffer-pool/eviction-policy';
import type { PooledBufferRef } from '../../../../rendering/gpu-buffer-pool/pool-stats';

describe('selectBuffersToEvict (pure function)', () => {
  function ref(bytes: number, id?: number): PooledBufferRef<number> {
    return { bytes, payload: id };
  }

  it('returns empty array when under budget', () => {
    const targets = selectBuffersToEvict([ref(100), ref(200)], 500);
    expect(targets).toEqual([]);
  });

  it('returns empty array for empty input', () => {
    expect(selectBuffersToEvict([], 100)).toEqual([]);
  });

  it('evicts largest-first until under budget', () => {
    // total = 100 + 200 + 300 + 400 = 1000, budget = 500.
    // Largest-first: drop 400 (running 600), drop 300 (running 300 ≤ 500).
    const targets = selectBuffersToEvict([ref(100, 1), ref(200, 2), ref(300, 3), ref(400, 4)], 500);
    const payloads = targets.map((t) => t.payload).sort();
    expect(payloads).toEqual([3, 4]);
  });

  it('all-same-size: stable order (input order preserved)', () => {
    const targets = selectBuffersToEvict([ref(100, 1), ref(100, 2), ref(100, 3), ref(100, 4)], 150);
    // total = 400, budget 150 → need to drop 250 bytes → 3 entries.
    expect(targets.length).toBe(3);
    // First three by input order (stable sort).
    expect(targets.map((t) => t.payload)).toEqual([1, 2, 3]);
  });

  it('single-buffer-over-budget pathological case', () => {
    // One huge buffer dwarfs everything.
    const targets = selectBuffersToEvict([ref(10), ref(20), ref(10_000)], 100);
    // Evicting the huge one alone (10) is enough.
    expect(targets.length).toBe(1);
    expect(targets[0].bytes).toBe(10_000);
  });

  it('uses precomputed total when provided', () => {
    const targets = selectBuffersToEvict(
      [ref(100), ref(200)],
      150,
      300 // explicit total bypasses reduce
    );
    expect(targets.length).toBeGreaterThan(0);
  });
});
