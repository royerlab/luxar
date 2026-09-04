import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { computePerfSnapshot } from '../../../../../core/app/debug/perf-snapshot';
import {
  markLoad,
  noteRefinementComplete,
  resetLoadTimeline,
} from '../../../../../profiling/load-timeline';

describe('computePerfSnapshot', () => {
  beforeEach(() => resetLoadTimeline());
  afterEach(() => resetLoadTimeline());

  it('is readable before init: timeline only, runtime not ready, isSettled unknown', () => {
    markLoad('loadStart');
    const snap = computePerfSnapshot();
    expect(snap.perfReady).toBe(true);
    expect(snap.runtimeReady).toBe(false);
    expect(snap.isSettled).toBeNull();
    expect(snap.rendererInfo).toBeNull();
    expect(snap.adaptiveDpr).toBeNull();
    expect(snap.workers).toBeNull();
    expect(snap.cache).toBeNull();
    expect(snap.timeline.milestones.loadStart).toBeDefined();
    expect(snap.settle).toEqual({
      updateInProgress: null,
      loadPassInProgress: null,
      lodLevelLoading: null,
      refinementComplete: false,
    });
  });

  it('exposes the cache tier stats (L2 write queue included) from the hook', () => {
    const stats = { l2WriteQueue: { pending: 0, inFlight: 0, dropped: 3 } };
    const snap = computePerfSnapshot({ isUpdateInProgress: () => false, cache: () => stats });
    expect(snap.cache).toBe(stats);
  });

  it('isSettled requires no update, no load pass, no lazy level, and refinement complete', () => {
    const hooks = {
      isUpdateInProgress: () => false,
      isAnyLoadPassInProgress: () => false,
      isAnyLodLevelLoading: () => false,
    };
    markLoad('loadStart');
    expect(computePerfSnapshot(hooks).isSettled).toBe(false); // refinement not complete
    noteRefinementComplete();
    expect(computePerfSnapshot(hooks).isSettled).toBe(true);
    expect(computePerfSnapshot({ ...hooks, isUpdateInProgress: () => true }).isSettled).toBe(false);
    expect(computePerfSnapshot({ ...hooks, isAnyLoadPassInProgress: () => true }).isSettled).toBe(
      false
    );
    expect(computePerfSnapshot({ ...hooks, isAnyLodLevelLoading: () => true }).isSettled).toBe(
      false
    );
  });

  it('a throwing hook reads as null instead of breaking the snapshot', () => {
    const boom = () => {
      throw new Error('renderer gone');
    };
    const snap = computePerfSnapshot({
      isUpdateInProgress: () => false,
      isAnyLoadPassInProgress: boom,
      rendererInfo: boom,
      workers: () => ({ workerCount: 3 }),
    });
    expect(snap.runtimeReady).toBe(true);
    expect(snap.settle.loadPassInProgress).toBeNull();
    expect(snap.rendererInfo).toBeNull();
    expect(snap.workers).toEqual({ workerCount: 3 });
    // An unknown load-pass state does not block settling; an unknown lazy-level
    // state (hook absent) does not either — only a positive "busy" does.
    noteRefinementComplete();
    expect(computePerfSnapshot({ isUpdateInProgress: () => false }).isSettled).toBe(true);
  });
});
