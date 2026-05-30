/**
 * Direct unit tests for the pure helpers in worker-pool/selection/ and
 * worker-pool/stats.ts (workers.md G11, G12, G13 + C3 follow-up).
 *
 * These functions accept a plain `WorkerInstance[]` and return a value
 * — no class instance, no globals, no async, no DOM. They are also
 * the building blocks the load-balancing tests in
 * `load-balancing.test.ts` cover via the WorkerPool wrapper (C3); the
 * audit asked for direct unit tests rather than `as any` mutation on a
 * real pool. This file is that direct coverage.
 */

import { describe, expect, it } from 'vitest';
import fc from 'fast-check';
import { selectLeastBusy } from '../../../../../workers/worker-pool/selection/least-busy';
import { nextRoundRobin } from '../../../../../workers/worker-pool/selection/round-robin';
import {
  computeStats,
  computeQueueDepth,
} from '../../../../../workers/worker-pool/stats';
import type { WorkerInstance } from '../../../../../workers/worker-pool/types';

// Fake WorkerInstance: helpers only read `.activeQueries`; the api /
// worker fields are returned by `selectLeastBusy` so we want unique
// references to assert the selection identity.
function makeInstance(activeQueries: number, label: string): WorkerInstance {
  return {
    worker: { _label: label } as unknown as Worker,
    api: { _label: label } as unknown as WorkerInstance['api'],
    activeQueries,
    wasmFallback: false,
  };
}

describe('selectLeastBusy (G12, H4)', () => {
  it('returns the strictly minimum worker', () => {
    const w0 = makeInstance(3, 'A');
    const w1 = makeInstance(1, 'B'); // least
    const w2 = makeInstance(2, 'C');
    const handle = selectLeastBusy([w0, w1, w2]);
    expect(handle.worker).toBe(w1.worker);
    expect(handle.api).toBe(w1.api);
  });

  it('breaks ties at index 0 (deterministic, H4)', () => {
    const w0 = makeInstance(2, 'A');
    const w1 = makeInstance(2, 'B');
    const w2 = makeInstance(2, 'C');
    const handle = selectLeastBusy([w0, w1, w2]);
    expect(handle.worker).toBe(w0.worker);
  });

  it('single-worker pool always picks that worker', () => {
    const w0 = makeInstance(99, 'only');
    const handle = selectLeastBusy([w0]);
    expect(handle.worker).toBe(w0.worker);
  });

  it('markQueryStart increments the selected worker only', () => {
    const w0 = makeInstance(5, 'A');
    const w1 = makeInstance(0, 'B'); // selected
    const handle = selectLeastBusy([w0, w1]);
    handle.markQueryStart();
    expect(w0.activeQueries).toBe(5);
    expect(w1.activeQueries).toBe(1);
  });

  it('markQueryEnd decrements the selected worker only', () => {
    const w0 = makeInstance(5, 'A');
    const w1 = makeInstance(3, 'B'); // selected
    const handle = selectLeastBusy([w0, w1]);
    handle.markQueryEnd();
    expect(w0.activeQueries).toBe(5);
    expect(w1.activeQueries).toBe(2);
  });

  it('markQueryEnd clamps to 0 (G12, M4 — Math.max(0, …) is load-bearing)', () => {
    const w0 = makeInstance(0, 'A');
    const handle = selectLeastBusy([w0]);
    // Two markEnds on a never-started counter MUST stay at 0; no negative leak.
    handle.markQueryEnd();
    handle.markQueryEnd();
    expect(w0.activeQueries).toBe(0);
  });

  it('markQueryStart / markQueryEnd are paired identity (round-trip is no-op)', () => {
    const w0 = makeInstance(7, 'A');
    const handle = selectLeastBusy([w0]);
    handle.markQueryStart();
    handle.markQueryStart();
    handle.markQueryEnd();
    handle.markQueryEnd();
    expect(w0.activeQueries).toBe(7);
  });

  it('selectLeastBusy throws a clear error for empty input (CRIT-6)', () => {
    // Before the fix this threw an opaque
    // `TypeError: Cannot read properties of undefined (reading 'activeQueries')`
    // from `workers[0].activeQueries`. The guard turns it into an explicit,
    // greppable diagnostic so direct callers (not just WorkerPool) see why.
    expect(() => selectLeastBusy([])).toThrow(/selectLeastBusy: workers array is empty/);
  });
});

describe('nextRoundRobin (G11, H3)', () => {
  it('advances cursor modulo workers.length', () => {
    const w0 = makeInstance(0, 'A');
    const w1 = makeInstance(0, 'B');
    const w2 = makeInstance(0, 'C');
    const workers = [w0, w1, w2];

    expect(nextRoundRobin(workers, 0).instance).toBe(w0);
    expect(nextRoundRobin(workers, 0).nextIndex).toBe(1);
    expect(nextRoundRobin(workers, 1).instance).toBe(w1);
    expect(nextRoundRobin(workers, 1).nextIndex).toBe(2);
    expect(nextRoundRobin(workers, 2).instance).toBe(w2);
    // Wrap-around — modular arithmetic invariant.
    expect(nextRoundRobin(workers, 2).nextIndex).toBe(0);
  });

  it('single-worker pool always returns same instance + nextIndex=0 (H3)', () => {
    const w0 = makeInstance(0, 'only');
    const r1 = nextRoundRobin([w0], 0);
    expect(r1.instance).toBe(w0);
    expect(r1.nextIndex).toBe(0);
    // Repeated calls at cursor=0 are stable.
    expect(nextRoundRobin([w0], 0).nextIndex).toBe(0);
  });

  it('two-worker pool alternates indices', () => {
    const w0 = makeInstance(0, 'A');
    const w1 = makeInstance(0, 'B');
    const workers = [w0, w1];
    let cursor = 0;
    const cursors: number[] = [cursor];
    for (let i = 0; i < 5; i++) {
      cursor = nextRoundRobin(workers, cursor).nextIndex;
      cursors.push(cursor);
    }
    expect(cursors).toEqual([0, 1, 0, 1, 0, 1]);
  });

  it('does not mutate input workers array', () => {
    const w0 = makeInstance(3, 'A');
    const w1 = makeInstance(4, 'B');
    nextRoundRobin([w0, w1], 0);
    expect(w0.activeQueries).toBe(3);
    expect(w1.activeQueries).toBe(4);
  });

  // workers.md [H3][P12] fast-check property test: nextRoundRobin's
  // contract for valid in-range cursors. nextIndex must equal
  // (cursor + 1) mod length and instance must be workers[cursor].
  // (The function assumes cursor is in [0, length); the wrap is in
  // nextIndex, which is the caller-stored value for the NEXT call.)
  it('[property] nextIndex = (cursor + 1) mod length for valid in-range cursor', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 32 }).chain((poolSize) =>
          fc.tuple(fc.constant(poolSize), fc.integer({ min: 0, max: poolSize - 1 }))
        ),
        ([poolSize, cursor]) => {
          const workers = Array.from({ length: poolSize }, (_, i) => makeInstance(0, `w${i}`));
          const result = nextRoundRobin(workers, cursor);
          const expectedNext = (cursor + 1) % poolSize;
          return result.nextIndex === expectedNext && result.instance === workers[cursor];
        }
      ),
      { numRuns: 200 }
    );
  });
});

describe('computeStats (G13, H5)', () => {
  it('returns workerCount, activeQueries, totalActive, peakActive on a non-empty pool', () => {
    const workers = [makeInstance(3, 'A'), makeInstance(5, 'B'), makeInstance(0, 'C')];
    const s = computeStats(workers);
    expect(s.workerCount).toBe(3);
    expect(s.activeQueries).toEqual([3, 5, 0]);
    expect(s.totalActive).toBe(8);
    expect(s.peakActive).toBe(5);
  });

  it('handles empty workers[] (G13 — boundary)', () => {
    // workers.md G13: `computeStats` on `workers[]=[]` boundary.
    const s = computeStats([]);
    expect(s.workerCount).toBe(0);
    expect(s.activeQueries).toEqual([]);
    expect(s.totalActive).toBe(0);
    expect(s.peakActive).toBe(0);
  });

  it('peakActive is the per-worker max (single hot worker detection)', () => {
    // Two cool + one hot.
    const workers = [makeInstance(1, 'A'), makeInstance(1, 'B'), makeInstance(42, 'HOT')];
    const s = computeStats(workers);
    expect(s.peakActive).toBe(42);
  });

  it('cross-check invariant: totalActive === sum(activeQueries) === computeQueueDepth (H5)', () => {
    const workers = [makeInstance(2, 'A'), makeInstance(3, 'B'), makeInstance(7, 'C')];
    const s = computeStats(workers);
    const qd = computeQueueDepth(workers);
    const sum = s.activeQueries.reduce((a, b) => a + b, 0);
    expect(s.totalActive).toBe(sum);
    expect(s.totalActive).toBe(qd);
  });
});

describe('computeQueueDepth (G13, H5)', () => {
  it('sums activeQueries across workers', () => {
    const workers = [makeInstance(3, 'A'), makeInstance(5, 'B'), makeInstance(0, 'C')];
    expect(computeQueueDepth(workers)).toBe(8);
  });

  it('returns 0 on empty workers[]', () => {
    expect(computeQueueDepth([])).toBe(0);
  });

  it('returns 0 when every worker is idle', () => {
    expect(
      computeQueueDepth([makeInstance(0, 'A'), makeInstance(0, 'B'), makeInstance(0, 'C')])
    ).toBe(0);
  });
});
