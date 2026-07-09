/**
 * Unit tests for `queueNext` in scene-loader/update-view/queue-next.ts.
 *
 * The helper has three outcomes:
 *   1. Pending state → yield via requestAnimationFrame, release the
 *      lock INSIDE the rAF callback, then re-enter updateView.
 *   2. No pending + at least one progressive loader (points, lines, OR
 *      gsplats) with hasMoreLODs:true → kick the refinement loop; lock
 *      stays HELD
 *      (refinement holds the lock so slider events queue and naturally
 *      cancel refinement).
 *   3. No pending + no refinement needed → release the lock.
 *
 * The load-bearing detail in outcome 1 is the ORDER inside the rAF
 * callback: lock release MUST happen BEFORE the re-entry, so that
 * slider events that fire during the yield queue (because lock was
 * held) AND the re-entry sees a released lock to acquire.
 *
 * Outcome 1 also has a fallback path for non-browser environments
 * (Vitest in Node) — if `globalThis.requestAnimationFrame` is missing,
 * the same release-then-reenter sequence runs synchronously.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { queueNext } from '../../../../../data/scene-loader/update-view/queue-next';
import { ViewStateQueue } from '../../../../../data/scene-loader/view-state/view-state-queue';
import type { GSplatsDataLoader } from '../../../../../types/gsplats';
import type { LinesDataLoader } from '../../../../../types/lines';
import type { DataLoader } from '../../../../../data/data-loader-types';
import type { QueueNextCtx } from '../../../../../data/scene-loader/update-view/queue-next';

// ============================================================================
// Local fixtures
// ============================================================================

function makeGSplatsLoader(hasMoreLODs: boolean): GSplatsDataLoader {
  return { hasMoreLODs } as unknown as GSplatsDataLoader;
}

function makePointsLoader(hasMoreLODs: boolean): DataLoader {
  return { hasMoreLODs } as unknown as DataLoader;
}

function makeLinesLoader(hasMoreLODs: boolean): LinesDataLoader {
  return { hasMoreLODs } as unknown as LinesDataLoader;
}

function makeCtx(overrides: Partial<QueueNextCtx> = {}): QueueNextCtx & {
  spies: {
    updateView: ReturnType<typeof vi.fn>;
    setUpdateInProgress: ReturnType<typeof vi.fn>;
    scheduleGSplatsRefinement: ReturnType<typeof vi.fn>;
    resolvePassWaiters: ReturnType<typeof vi.fn>;
  };
} {
  const updateView = vi.fn().mockResolvedValue(undefined);
  const setUpdateInProgress = vi.fn();
  const scheduleGSplatsRefinement = vi.fn().mockResolvedValue(undefined);
  const resolvePassWaiters = vi.fn();

  const ctx: QueueNextCtx = {
    viewStateQueue: new ViewStateQueue(),
    pointsLoaders: new Map(),
    linesLoaders: new Map(),
    gsplatLoaders: new Map(),
    updateView,
    setUpdateInProgress,
    scheduleGSplatsRefinement,
    resolvePassWaiters,
    ...overrides,
  };
  return Object.assign(ctx, {
    spies: { updateView, setUpdateInProgress, scheduleGSplatsRefinement, resolvePassWaiters },
  });
}

// Capture rAF callbacks so we can manually fire them (synchronously
// in tests) and inspect what they do.
let rafCallbacks: FrameRequestCallback[] = [];
function installFakeRaf(): void {
  rafCallbacks = [];
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
    rafCallbacks.push(cb);
    return rafCallbacks.length as unknown as number;
  }) as typeof globalThis.requestAnimationFrame;
}
function uninstallRaf(): void {
  // jsdom (the vitest env) provides rAF; restore by deleting our override
  // before each restore tests sets it up again.
  delete (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame;
}

beforeEach(() => {
  installFakeRaf();
});
afterEach(() => {
  uninstallRaf();
});

// ============================================================================
// Tests
// ============================================================================

describe('queueNext — pending state + rAF available', () => {
  it('calls requestAnimationFrame; release-then-reenter happens INSIDE the rAF callback', () => {
    const ctx = makeCtx();
    ctx.viewStateQueue.setPending({ displayDims: [0, 1, 2] });

    queueNext(ctx);

    // rAF was scheduled, but neither callback has run yet — releaseLock + re-entry
    // are deferred until the next frame.
    expect(rafCallbacks.length).toBe(1);
    expect(ctx.spies.setUpdateInProgress).not.toHaveBeenCalled();
    expect(ctx.spies.updateView).not.toHaveBeenCalled();

    // Now manually fire the frame.
    rafCallbacks[0](performance.now());

    // Inside the callback the lock release happens FIRST, then the re-entry.
    const releaseIdx = ctx.spies.setUpdateInProgress.mock.invocationCallOrder[0];
    const updateIdx = ctx.spies.updateView.mock.invocationCallOrder[0];
    expect(releaseIdx).toBeLessThan(updateIdx);
    expect(ctx.spies.setUpdateInProgress).toHaveBeenCalledWith(false);
    expect(ctx.spies.updateView).toHaveBeenCalledWith({ displayDims: [0, 1, 2] });

    // No refinement scheduling on this branch.
    expect(ctx.spies.scheduleGSplatsRefinement).not.toHaveBeenCalled();
    // Waiters are NOT resolved on the pending branch — they carry over to
    // the re-entered (winning) pass, whose own queueNext resolves them.
    expect(ctx.spies.resolvePassWaiters).not.toHaveBeenCalled();
  });
});

describe('queueNext — pending state + no rAF (Node fallback)', () => {
  it('releases lock then re-enters synchronously when requestAnimationFrame is missing', () => {
    uninstallRaf();
    const ctx = makeCtx();
    ctx.viewStateQueue.setPending({ slicePosition: [1, 2, 3, 4] });

    queueNext(ctx);

    // Synchronous fallback — both ran without any rAF dance.
    expect(ctx.spies.setUpdateInProgress).toHaveBeenCalledWith(false);
    expect(ctx.spies.updateView).toHaveBeenCalledWith({ slicePosition: [1, 2, 3, 4] });
    // Order still matters: release before re-entry.
    const releaseIdx = ctx.spies.setUpdateInProgress.mock.invocationCallOrder[0];
    const updateIdx = ctx.spies.updateView.mock.invocationCallOrder[0];
    expect(releaseIdx).toBeLessThan(updateIdx);
    // No rAF scheduled.
    expect(rafCallbacks.length).toBe(0);
  });
});

describe('queueNext — no pending + at least one loader with hasMoreLODs:true', () => {
  it('kicks scheduleGSplatsRefinement and KEEPS the lock held', () => {
    const ctx = makeCtx({
      gsplatLoaders: new Map<string, GSplatsDataLoader>([
        ['/g0', makeGSplatsLoader(false)],
        ['/g1', makeGSplatsLoader(true)],
      ]),
    });

    queueNext(ctx);

    expect(ctx.spies.scheduleGSplatsRefinement).toHaveBeenCalledTimes(1);
    // Key invariant: lock stays held so slider events queue (refinement
    // is responsible for releasing it on completion).
    expect(ctx.spies.setUpdateInProgress).not.toHaveBeenCalled();
    expect(ctx.spies.updateView).not.toHaveBeenCalled();
    // Waiters resolve at refinement ENTRY: the main pass (first commit) is
    // done — the pacing gate needs first-commit latency, not full-ladder.
    expect(ctx.spies.resolvePassWaiters).toHaveBeenCalledTimes(1);
  });
});

describe('queueNext — no pending + no refinement needed', () => {
  it('releases the lock when every loader is fully refined', () => {
    const ctx = makeCtx({
      gsplatLoaders: new Map<string, GSplatsDataLoader>([
        ['/g0', makeGSplatsLoader(false)],
        ['/g1', makeGSplatsLoader(false)],
      ]),
    });

    queueNext(ctx);

    expect(ctx.spies.setUpdateInProgress).toHaveBeenCalledWith(false);
    expect(ctx.spies.scheduleGSplatsRefinement).not.toHaveBeenCalled();
    expect(ctx.spies.updateView).not.toHaveBeenCalled();
    // Idle branch: the winning pass committed and nothing follows — waiters settle.
    expect(ctx.spies.resolvePassWaiters).toHaveBeenCalledTimes(1);
  });

  it('treats hasMoreLODs === undefined as NOT needing refinement (strict === true)', () => {
    // The guard is `l.hasMoreLODs === true`, so an undefined value (e.g.
    // a loader that never set the flag) must NOT schedule refinement —
    // the lock is released instead.
    const loaderWithUndefined = {} as unknown as GSplatsDataLoader; // hasMoreLODs absent → undefined
    const ctx = makeCtx({
      gsplatLoaders: new Map<string, GSplatsDataLoader>([['/g0', loaderWithUndefined]]),
    });

    queueNext(ctx);

    expect(ctx.spies.scheduleGSplatsRefinement).not.toHaveBeenCalled();
    expect(ctx.spies.setUpdateInProgress).toHaveBeenCalledWith(false);
    expect(ctx.spies.updateView).not.toHaveBeenCalled();
  });

  it('releases the lock when every loader map is empty', () => {
    const ctx = makeCtx({ gsplatLoaders: new Map() });

    queueNext(ctx);

    expect(ctx.spies.setUpdateInProgress).toHaveBeenCalledWith(false);
    expect(ctx.spies.scheduleGSplatsRefinement).not.toHaveBeenCalled();
  });
});

describe('queueNext — refinement rejection recovers the lock (no permanent freeze)', () => {
  it('releases the lock and drains a queued state when scheduleGSplatsRefinement rejects', async () => {
    // Regression: the fire-and-forget .catch only logged, so an error
    // escaping the refinement orchestrator glue (outside the loops' own
    // finally blocks) left _updateInProgress held forever — every future
    // updateView queued into a pending slot nothing drained.
    const ctx = makeCtx({
      gsplatLoaders: new Map<string, GSplatsDataLoader>([['/g', makeGSplatsLoader(true)]]),
    });
    ctx.spies.scheduleGSplatsRefinement.mockRejectedValue(new Error('glue died'));

    queueNext(ctx);
    // Simulate a user scrub arriving while the (doomed) refinement holds the lock.
    ctx.viewStateQueue.setPending({ slicePosition: [9, 9, 9, 9] });
    // Let the rejection propagate through the .catch handler + the drain microtask.
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(ctx.spies.setUpdateInProgress).toHaveBeenCalledWith(false); // lock recovered
    expect(ctx.spies.updateView).toHaveBeenCalledWith({ slicePosition: [9, 9, 9, 9] }); // queued state drained
  });

  it('rejection with nothing queued just releases the lock (drain is a no-op)', async () => {
    const ctx = makeCtx({
      gsplatLoaders: new Map<string, GSplatsDataLoader>([['/g', makeGSplatsLoader(true)]]),
    });
    ctx.spies.scheduleGSplatsRefinement.mockRejectedValue(new Error('glue died'));

    queueNext(ctx);
    await Promise.resolve();
    await Promise.resolve();

    expect(ctx.spies.setUpdateInProgress).toHaveBeenCalledWith(false);
    expect(ctx.spies.updateView).not.toHaveBeenCalled();
    // Belt-and-braces settle in the rejection catch: with nothing queued no
    // re-entry will resolve parked waiters, so the catch must (the entry-time
    // resolve already ran once before refinement was scheduled).
    expect(ctx.spies.resolvePassWaiters).toHaveBeenCalledTimes(2);
  });
});

describe('queueNext — points/lines progressive loaders gate refinement too', () => {
  // Regression: needsRefinement previously checked only gsplatLoaders, so a
  // points- or lines-only scene with additive ladders never refined after any
  // view change (stuck at partial LODs). The gate must mirror the post-load
  // kick in lifecycle/load-scene.ts, which checks all three types.
  it('kicks refinement when only a POINTS loader has more LODs', () => {
    const ctx = makeCtx({
      pointsLoaders: new Map<string, DataLoader>([['/p', makePointsLoader(true)]]),
    });

    queueNext(ctx);

    expect(ctx.spies.scheduleGSplatsRefinement).toHaveBeenCalledTimes(1);
    expect(ctx.spies.setUpdateInProgress).not.toHaveBeenCalled(); // lock held
  });

  it('kicks refinement when only a LINES loader has more LODs', () => {
    const ctx = makeCtx({
      linesLoaders: new Map<string, LinesDataLoader>([['/l', makeLinesLoader(true)]]),
    });

    queueNext(ctx);

    expect(ctx.spies.scheduleGSplatsRefinement).toHaveBeenCalledTimes(1);
    expect(ctx.spies.setUpdateInProgress).not.toHaveBeenCalled(); // lock held
  });

  it('releases the lock when points/lines/gsplats loaders all report no more LODs', () => {
    const ctx = makeCtx({
      pointsLoaders: new Map<string, DataLoader>([['/p', makePointsLoader(false)]]),
      linesLoaders: new Map<string, LinesDataLoader>([['/l', makeLinesLoader(false)]]),
      gsplatLoaders: new Map<string, GSplatsDataLoader>([['/g', makeGSplatsLoader(false)]]),
    });

    queueNext(ctx);

    expect(ctx.spies.scheduleGSplatsRefinement).not.toHaveBeenCalled();
    expect(ctx.spies.setUpdateInProgress).toHaveBeenCalledWith(false);
  });

  it('non-progressive points loaders (no hasMoreLODs) do not schedule refinement', () => {
    // Plain PointsSpatialIndexLoader has no hasMoreLODs property at all.
    const ctx = makeCtx({
      pointsLoaders: new Map<string, DataLoader>([['/p', {} as unknown as DataLoader]]),
    });

    queueNext(ctx);

    expect(ctx.spies.scheduleGSplatsRefinement).not.toHaveBeenCalled();
    expect(ctx.spies.setUpdateInProgress).toHaveBeenCalledWith(false);
  });
});

describe('queueNext — branch ordering: pending wins over refinement', () => {
  it('chooses the pending branch even when refinement WOULD be needed', () => {
    const ctx = makeCtx({
      gsplatLoaders: new Map<string, GSplatsDataLoader>([['/g', makeGSplatsLoader(true)]]),
    });
    ctx.viewStateQueue.setPending({ displayDims: [0, 1, 2] });

    queueNext(ctx);

    // rAF callback queued, but neither has run yet — and refinement was NOT
    // touched because the pending branch returns early.
    expect(rafCallbacks.length).toBe(1);
    expect(ctx.spies.scheduleGSplatsRefinement).not.toHaveBeenCalled();

    // After the rAF fires, the pending re-entry runs (still no refinement).
    rafCallbacks[0](performance.now());
    expect(ctx.spies.updateView).toHaveBeenCalledWith({ displayDims: [0, 1, 2] });
    expect(ctx.spies.scheduleGSplatsRefinement).not.toHaveBeenCalled();
  });
});
