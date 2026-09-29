/**
 * The residency ceiling under concurrent refinement steps (B9c).
 *
 * `processLoadersBounded` runs up to MAX_CONCURRENT_REFINEMENT_STEPS loader
 * steps at once. Each step is admitted against the budget BEFORE any of the
 * others has recorded its real post-pass bytes, and the next-rung estimate is
 * the MEAN rung — an under-estimate on a geometric ladder. The serial loop's
 * contract was "may cross the ceiling by one rung"; admitting four at once on
 * that estimate let the scene cross it by four, which is the #2426-class tab
 * OOM. These tests drive the real loop, the real wrapper gate and the real
 * budget with a fake ladder whose next rung is three times everything before.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { runProgressiveRefinement } from '../../../../../data/scene-loader/progressive/refinement';
import {
  admitRefinementCandidate,
  makeRefinementProgressCallbacks,
  recordRefinementResidency,
} from '../../../../../data/scene-loader/progressive/refinement-wrapper';
import { RefinementResidencyBudget } from '../../../../../data/scene-loader/progressive/residency-budget';

/** A ladder whose every rung is `growth` x everything before it. */
class GeometricLadder {
  bytes = 100;
  rungs = 1;
  constructor(
    readonly total = 4,
    readonly growth = 3
  ) {}
  get hasMoreLODs(): boolean {
    return this.rungs < this.total;
  }
  get loadedLODCount(): number {
    return this.rungs;
  }
  get totalLODCount(): number {
    return this.total;
  }
  ladderResidency() {
    return {
      residentBytes: this.bytes,
      loadedRungs: this.rungs,
      elementCount: 0,
      bytesPerElement: 0,
    };
  }
}

interface RunResult {
  peak: number;
  maxInFlight: number;
  final: number;
}

async function runLadders(budgetBytes: number, count: number, growth: number): Promise<RunResult> {
  const loaders = new Map<string, GeometricLadder>();
  for (let i = 0; i < count; i++) loaders.set(`/n${i}`, new GeometricLadder(4, growth));
  const budget = new RefinementResidencyBudget(
    budgetBytes,
    [...loaders].map(([path, loader]) => [path, loader.ladderResidency()] as const)
  );
  const total = (): number => [...loaders.values()].reduce((sum, l) => sum + l.bytes, 0);
  let peak = total();
  let inFlight = 0;
  let maxInFlight = 0;
  const queue = { takePending: () => null, hasPending: () => false } as never;
  await runProgressiveRefinement({
    loaders,
    viewStateQueue: queue,
    processLoader: async (path, loader) => {
      const admission = await admitRefinementCandidate(path, loader, budget);
      if (!admission.admitted) return false;
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      try {
        await new Promise((resolve) => setTimeout(resolve, 2));
        loader.bytes += loader.bytes * growth;
        loader.rungs++;
        peak = Math.max(peak, total());
        return true;
      } finally {
        inFlight--;
        recordRefinementResidency(path, loader, budget);
      }
    },
    ...makeRefinementProgressCallbacks('Test', loaders, budget),
    updateVisibleCountsInMonitor: () => undefined,
    releaseLock: () => undefined,
    retriggerUpdate: () => undefined,
  });
  return { peak, maxInFlight, final: total() };
}

describe('residency ceiling with concurrent refinement steps', () => {
  let warn: ReturnType<typeof vi.spyOn>;
  beforeEach(async () => {
    const { log } = await import('../../../../../utils/log');
    warn = vi.spyOn(log, 'warning').mockImplementation(() => undefined);
  });
  afterEach(() => warn.mockRestore());

  it.fails('crosses the ceiling by at most one rung, as the serial loop did', async () => {
    // Serial: /n0 and /n1 each climb to 400 and the scene stops at exactly
    // 1000. Admitting all four on the mean estimate climbed to 1600.
    const { peak } = await runLadders(1000, 4, 3);
    expect(peak).toBeLessThanOrEqual(1000 + 300);
  });

  it('still runs steps concurrently while the ceiling is far away', async () => {
    // A doubling ladder with ample headroom: the gate must not serialise the
    // loop, which would give back the whole B9c win.
    const { maxInFlight, final } = await runLadders(1_000_000, 4, 1);
    expect(maxInFlight).toBe(4);
    expect(final).toBe(4 * 800);
  });

  it.fails('defers, rather than declines, a step that only fails to fit beside the in-flight ones', () => {
    const budget = new RefinementResidencyBudget(
      1000,
      new Map([
        ['/a', { residentBytes: 100, loadedRungs: 1, elementCount: 0, bytesPerElement: 0 }],
        ['/b', { residentBytes: 100, loadedRungs: 1, elementCount: 0, bytesPerElement: 0 }],
        ['/c', { residentBytes: 100, loadedRungs: 1, elementCount: 0, bytesPerElement: 0 }],
        ['/d', { residentBytes: 100, loadedRungs: 1, elementCount: 0, bytesPerElement: 0 }],
      ])
    );
    const r = { residentBytes: 100, loadedRungs: 1, elementCount: 0, bytesPerElement: 0 };
    expect(budget.admit('/a', r).admitted).toBe(true);
    expect(budget.admit('/b', r).admitted).toBe(true);
    // Alone, /c fits (600 + 100 <= 1000). Beside two unmeasured steps it must
    // wait, and waiting is not a refusal: it is not retired for the run.
    expect(budget.admit('/c', r).admitted).toBe(false);
    expect(budget.isDeclined('/c')).toBe(false);
    // Once the in-flight steps have measured, /c is judged alone again.
    budget.record('/a', r);
    budget.record('/b', r);
    expect(budget.admit('/c', r).admitted).toBe(true);
  });
});
