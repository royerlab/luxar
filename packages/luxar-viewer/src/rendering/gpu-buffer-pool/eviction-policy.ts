/**
 * Pure eviction-policy helpers for the GPU buffer pool.
 *
 * Kept separate from the side-effecting eviction loop on `GPUBufferPool`
 * so the policy can be unit-tested without instantiating a pool, and so
 * per-type adapters (carved out in a later refactor step) can call into
 * a single source of truth.
 */

import type { PooledBufferRef } from './pool-stats';

/**
 * Pure selector for byte-budget eviction.
 *
 * Given an array of pooled-buffer refs and a target budget, returns
 * the subset that should be evicted to bring total bytes ≤ maxBytes.
 * Strategy: sort largest-first and walk until the running total drops
 * under budget. JS sort is stable in modern engines (V8, JSC,
 * SpiderMonkey since 2019) so equal-size buffers retain their input
 * order — important for deterministic test output.
 */
export function selectBuffersToEvict<R extends PooledBufferRef>(
  refs: R[],
  maxBytes: number,
  precomputedTotal?: number
): R[] {
  const total =
    precomputedTotal !== undefined ? precomputedTotal : refs.reduce((sum, r) => sum + r.bytes, 0);
  if (total <= maxBytes) return [];
  const sorted = refs.slice().sort((a, b) => b.bytes - a.bytes);
  const targets: R[] = [];
  let running = total;
  for (const ref of sorted) {
    if (running <= maxBytes) break;
    targets.push(ref);
    running -= ref.bytes;
  }
  return targets;
}
