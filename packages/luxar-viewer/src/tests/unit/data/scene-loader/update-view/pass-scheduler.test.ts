// @vitest-environment jsdom
/**
 * Unit tests for `PassScheduler` (scene-loader/update-view/pass-scheduler.ts),
 * driven through a stub host — no SceneLoader, no loaders.
 *
 * The SceneLoader-level suites (`scene-loader.test.ts`: queued-call pacing,
 * B5 drag guarantee, A8 frame yield, A10 retry pre-emption, resync stash) are
 * the behaviour spec; these pin the scheduler's own contract: its explicit
 * phases, what runs after a pass (pending → yield; ladders left → refinement,
 * lock held; else release), the release-then-re-enter order inside the frame,
 * the one release-and-drain path, retry pre-emption and teardown.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  PassScheduler,
  type PassSchedulerHost,
} from '../../../../../data/scene-loader/update-view/pass-scheduler';
import { ViewStateQueue } from '../../../../../data/scene-loader/view-state/view-state-queue';
import { log, Modules } from '../../../../../utils/log';

function makeHost(overrides: Partial<PassSchedulerHost> = {}) {
  const base = {
    queue: new ViewStateQueue(),
    isDisposed: vi.fn(() => false),
    isFaulted: vi.fn(() => false),
    matchesRunningPass: vi.fn(() => false),
    runningDisplayDims: vi.fn(() => [0, 1, 2]),
    viewVersion: vi.fn(() => 1),
    runPass: vi.fn().mockResolvedValue(undefined),
    anyHasMoreLODs: vi.fn(() => false),
    runRefinement: vi.fn().mockResolvedValue(undefined),
  };
  // Overrides replace stub members; the mock-typed shape is kept for assertions.
  return { ...base, ...overrides } as typeof base;
}

let frames: FrameRequestCallback[] = [];
beforeEach(() => {
  frames = [];
  vi.spyOn(document, 'hidden', 'get').mockReturnValue(false);
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
    frames.push(cb);
    return frames.length;
  }) as typeof globalThis.requestAnimationFrame;
});
afterEach(() => {
  delete (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame;
  vi.restoreAllMocks();
});

/** Run one view pass of `state` to its end with the given outcome. */
function runPass(
  passes: PassScheduler,
  state: object,
  outcome = { discarded: false, faulted: false }
) {
  const pass = passes.beginPass(state);
  return passes.endPass(pass, outcome);
}

describe('PassScheduler — phases', () => {
  it('reports idle → pass → yielding → idle across a queued follow-up', () => {
    const host = makeHost();
    const passes = new PassScheduler(host);
    expect(passes.phase).toBe('idle');
    const pass = passes.beginPass({});
    expect(passes.phase).toBe('pass');
    void passes.request({ slicePosition: [1] }, {});
    void passes.endPass(pass, { discarded: false, faulted: false });
    expect(passes.phase).toBe('yielding');
    frames[0](0);
    expect(passes.phase).toBe('idle');
    expect(host.runPass).toHaveBeenCalledWith({ slicePosition: [1] });
  });

  it('reports refining while a run holds the lock, retrying while a retry does', async () => {
    const passes = new PassScheduler(makeHost());
    passes.locked = true;
    passes.beginRefinement(new AbortController());
    expect(passes.phase).toBe('refining');
    expect(passes.isLoadPassInProgress).toBe(false);
    passes.endRefinement(false);
    passes.releaseAndDrain();
    expect(await passes.acquireForRetry()).toBe(true);
    expect(passes.phase).toBe('retrying');
    passes.releaseRetry();
    expect(passes.phase).toBe('idle');
  });
});

describe('PassScheduler — after a pass', () => {
  it('pending state: yields a frame, releases the lock, THEN re-enters', () => {
    const host = makeHost();
    const passes = new PassScheduler(host);
    host.queue.setPending({ displayDims: [0, 1, 2] });

    void runPass(passes, {});
    expect(frames).toHaveLength(1);
    expect(host.runPass).not.toHaveBeenCalled();
    let lockedAtReentry: boolean | undefined;
    host.runPass.mockImplementation(async () => {
      lockedAtReentry = passes.locked;
    });

    frames[0](0);
    expect(lockedAtReentry).toBe(false);
    expect(host.runPass).toHaveBeenCalledWith({ displayDims: [0, 1, 2] });
    expect(host.runRefinement).not.toHaveBeenCalled();
  });

  it('re-enters with the NEWEST pending state when one is queued during the yield (A8)', () => {
    const host = makeHost();
    const passes = new PassScheduler(host);
    host.queue.setPending({ slicePosition: [1] });
    void runPass(passes, {});
    // A view arriving in the yield goes through the ordinary request path.
    void passes.request({ slicePosition: [2] }, {});
    frames[0](0);
    expect(host.runPass).toHaveBeenCalledTimes(1);
    expect(host.runPass).toHaveBeenCalledWith({ slicePosition: [2] });
    expect(host.queue.hasPending()).toBe(false);
  });

  it('a request during the yield neither aborts nor holds the finished pass (A8)', () => {
    const passes = new PassScheduler(makeHost());
    const pass = passes.beginPass({});
    void passes.request({ slicePosition: [1] }, {});
    pass.controller.abort();
    void passes.endPass(pass, { discarded: false, faulted: false });
    expect(passes.phase).toBe('yielding');
    const vi_ = vi.spyOn(globalThis, 'setTimeout');
    void passes.request({ slicePosition: [2] }, {});
    // No B5 hold timer is armed against the dead controller.
    expect(vi_).not.toHaveBeenCalled();
  });

  it('runs synchronously when requestAnimationFrame is missing', () => {
    delete (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame;
    const host = makeHost();
    const passes = new PassScheduler(host);
    host.queue.setPending({ slicePosition: [1, 2, 3, 4] });
    void runPass(passes, {});
    expect(passes.locked).toBe(false);
    expect(host.runPass).toHaveBeenCalledWith({ slicePosition: [1, 2, 3, 4] });
  });

  it('logs a rejected re-entry instead of leaving an unhandled rejection', async () => {
    const failure = new Error('synthetic re-entry failure');
    const host = makeHost({ runPass: vi.fn().mockRejectedValue(failure) });
    const errorLog = vi.spyOn(log, 'error').mockImplementation(() => {});
    const passes = new PassScheduler(host);
    host.queue.setPending({ displayDims: [0, 1, 2] });
    void runPass(passes, {});
    frames[0](0);
    await Promise.resolve();
    expect(errorLog).toHaveBeenCalledWith(
      Modules.SCENE_LOADER,
      'Queued updateView re-entry failed: synthetic re-entry failure',
      failure
    );
  });

  it('no pending + ladders left: starts refinement and KEEPS the lock, waiters settle at entry', async () => {
    const host = makeHost({ anyHasMoreLODs: vi.fn(() => true) });
    const passes = new PassScheduler(host);
    passes.locked = true;
    let settled = false;
    void passes.request({ slicePosition: [1] }, {})!.then(() => {
      settled = true;
    });
    // The queued state runs; its pass commits with nothing pending behind it.
    void runPass(passes, host.queue.takePending()!);
    await Promise.resolve();
    expect(host.runRefinement).toHaveBeenCalledTimes(1);
    expect(passes.locked).toBe(true);
    expect(settled).toBe(true);
  });

  it('no pending, no ladders: releases the lock', () => {
    const host = makeHost();
    const passes = new PassScheduler(host);
    void runPass(passes, {});
    expect(passes.locked).toBe(false);
    expect(host.runRefinement).not.toHaveBeenCalled();
  });

  it('pending wins over refinement', () => {
    const host = makeHost({ anyHasMoreLODs: vi.fn(() => true) });
    const passes = new PassScheduler(host);
    host.queue.setPending({ displayDims: [0, 1, 2] });
    void runPass(passes, {});
    expect(host.runRefinement).not.toHaveBeenCalled();
    frames[0](0);
    expect(host.runPass).toHaveBeenCalledWith({ displayDims: [0, 1, 2] });
  });

  it('an archive-faulted pass abandons the queued state and settles every waiter', async () => {
    const host = makeHost();
    const passes = new PassScheduler(host);
    const pass = passes.beginPass({});
    const queued = passes.request({ slicePosition: [3] }, {});
    void passes.endPass(pass, { discarded: true, faulted: true });
    await queued;
    expect(host.queue.hasPending()).toBe(false);
    expect(passes.locked).toBe(false);
    expect(frames).toHaveLength(0);
  });

  it('a superseded direct pass waits for the pass that supersedes it', async () => {
    const host = makeHost();
    const passes = new PassScheduler(host);
    const first = passes.beginPass({ slicePosition: [1] });
    const queued = passes.request({ slicePosition: [2] }, {});
    expect(first.controller.signal.aborted).toBe(true);
    const wait = passes.endPass(first, { discarded: false, faulted: false });
    expect(wait).toBeDefined();
    let settled = false;
    void wait!.then(() => {
      settled = true;
    });
    frames[0](0);
    const second = passes.beginPass(host.runPass.mock.calls[0][0] as object);
    await Promise.resolve();
    expect(settled).toBe(false);
    void passes.endPass(second, { discarded: false, faulted: false });
    await Promise.all([wait, queued]);
    expect(settled).toBe(true);
    expect(passes.allRequestsCompleted).toBe(true);
  });
});

describe('PassScheduler — resync stash', () => {
  it('folds a resync parked mid-pass into a targeted follow-up pass', () => {
    const host = makeHost();
    const passes = new PassScheduler(host);
    const pass = passes.beginPass({});
    void passes.request({}, { resyncPaths: new Set(['/p/part_1']) });
    void passes.endPass(pass, { discarded: false, faulted: false });
    frames[0](0);
    expect(host.runPass).toHaveBeenCalledWith({}, { resyncPaths: new Set(['/p/part_1']) });
    expect(passes.queuedResyncPaths).toBeNull();
  });

  it('a real view queued after the resync sweeps everything (stash dropped)', () => {
    const host = makeHost();
    const passes = new PassScheduler(host);
    const pass = passes.beginPass({});
    void passes.request({}, { resyncPaths: new Set(['/p/part_1']) });
    void passes.endPass(pass, { discarded: false, faulted: false });
    void passes.request({ slicePosition: [5] }, {});
    frames[0](0);
    expect(host.runPass).toHaveBeenCalledTimes(1);
    expect(host.runPass).toHaveBeenCalledWith({ slicePosition: [5] });
  });

  it('a resync parked during a retry runs once the retry releases', async () => {
    // A layer re-shown while a retry holds the lock (`requestReprocess` with
    // resync paths) must not wait for some later view pass to be folded in.
    const host = makeHost();
    const passes = new PassScheduler(host);
    expect(await passes.acquireForRetry()).toBe(true);
    void passes.request({}, { resyncPaths: new Set(['/p/part_1']) });
    passes.releaseRetry();
    expect(passes.drainPending()).toBe(true);
    await Promise.resolve();
    expect(host.runPass).toHaveBeenCalledWith({}, { resyncPaths: new Set(['/p/part_1']) });
  });

  it('a retry that reloads the whole view drops the parked resync (a superset)', async () => {
    const host = makeHost();
    const passes = new PassScheduler(host);
    expect(await passes.acquireForRetry()).toBe(true);
    void passes.request({}, { resyncPaths: new Set(['/p/part_1']) });
    passes.releaseRetry(true);
    expect(passes.drainPending()).toBe(false);
    // The reload's own full pass: nothing parked rides its end.
    void runPass(passes, {});
    expect(frames).toHaveLength(0);
    expect(host.runPass).not.toHaveBeenCalled();
  });
});

describe('PassScheduler — refinement start, release and failure', () => {
  it('an orchestrator rejection releases the lock and drains a queued state', async () => {
    const host = makeHost({ runRefinement: vi.fn().mockRejectedValue(new Error('glue died')) });
    vi.spyOn(log, 'error').mockImplementation(() => {});
    const passes = new PassScheduler(host);
    passes.startRefinement('Refinement failed');
    expect(passes.locked).toBe(true);
    host.queue.setPending({ slicePosition: [9, 9, 9, 9] });
    for (let i = 0; i < 4; i++) await Promise.resolve();
    expect(passes.locked).toBe(false);
    expect(host.runPass).toHaveBeenCalledWith({ slicePosition: [9, 9, 9, 9] });
  });

  it('with nothing queued, a rejection settles the parked waiters', async () => {
    const host = makeHost({ runRefinement: vi.fn().mockRejectedValue(new Error('glue died')) });
    vi.spyOn(log, 'error').mockImplementation(() => {});
    const passes = new PassScheduler(host);
    void runPass(passes, {});
    passes.startRefinement('Refinement failed');
    const resolve = vi.spyOn(passes, 'resolveCompletedWaiters');
    for (let i = 0; i < 4; i++) await Promise.resolve();
    expect(passes.locked).toBe(false);
    expect(resolve).toHaveBeenCalledTimes(1);
    expect(host.runPass).not.toHaveBeenCalled();
  });

  it.each([
    ['an archive-faulted', { isFaulted: vi.fn(() => true) }],
    ['a disposed', { isDisposed: vi.fn(() => true) }],
  ])('a start on %s loader runs nothing and releases the lock', (_label, overrides) => {
    const host = makeHost(overrides);
    const passes = new PassScheduler(host);
    passes.locked = true;
    passes.startRefinement('Post-load progressive refinement failed');
    expect(host.runRefinement).not.toHaveBeenCalled();
    expect(passes.locked).toBe(false);
  });

  it('the cancellation hand-off re-enters the newest state after one frame', () => {
    const host = makeHost();
    const passes = new PassScheduler(host);
    passes.locked = true;
    passes.beginRefinement(new AbortController());
    passes.handOff({ slicePosition: [1] });
    void passes.request({ slicePosition: [2] }, {});
    passes.endRefinement(true);
    frames[0](0);
    expect(host.runPass).toHaveBeenCalledTimes(1);
    expect(host.runPass).toHaveBeenCalledWith({ slicePosition: [2] });
  });
});

describe('PassScheduler — retry pre-emption (A10)', () => {
  it('a retry during a drain aborts it and inherits the lock as the run unwinds', async () => {
    const passes = new PassScheduler(makeHost());
    const controller = new AbortController();
    passes.locked = true;
    passes.beginRefinement(controller);
    const owned = passes.acquireForRetry();
    expect(controller.signal.aborted).toBe(true);
    expect(passes.retryPreemptPending).toBe(true);
    // The run's last phase does not release a lock a retry is waiting for.
    passes.releaseRefinementLock();
    expect(passes.locked).toBe(true);
    passes.endRefinement(false);
    await expect(owned).resolves.toBe(true);
    expect(passes.phase).toBe('retrying');
  });

  it('a drain cancelled into a view pass hands the retry nothing', async () => {
    const passes = new PassScheduler(makeHost());
    passes.locked = true;
    passes.beginRefinement(new AbortController());
    const owned = passes.acquireForRetry();
    passes.endRefinement(true);
    await expect(owned).resolves.toBe(false);
  });

  it('a view pass, or a view queued behind the drain, defers the retry', async () => {
    const host = makeHost();
    const passes = new PassScheduler(host);
    passes.beginPass({});
    await expect(passes.acquireForRetry()).resolves.toBe(false);

    const drain = new PassScheduler(host);
    drain.locked = true;
    drain.beginRefinement(new AbortController());
    host.queue.setPending({ slicePosition: [1] });
    await expect(drain.acquireForRetry()).resolves.toBe(false);
  });
});

describe('PassScheduler — kick and teardown', () => {
  it('a busy kick schedules ONE re-check, then starts refinement once idle', async () => {
    vi.useFakeTimers();
    try {
      const host = makeHost({ anyHasMoreLODs: vi.fn(() => true) });
      const passes = new PassScheduler(host);
      passes.locked = true;
      passes.kickRefinementIfIdle();
      passes.kickRefinementIfIdle();
      expect(passes.kickPending).toBe(true);
      expect(vi.getTimerCount()).toBe(1);
      passes.locked = false;
      await vi.runOnlyPendingTimersAsync();
      expect(host.runRefinement).toHaveBeenCalledTimes(1);
      expect(passes.locked).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it('dispose releases the lock, the slot, the kick timer and every waiter', async () => {
    const host = makeHost({ anyHasMoreLODs: vi.fn(() => true) });
    const passes = new PassScheduler(host);
    passes.beginPass({});
    const queued = passes.request({ slicePosition: [1] }, {});
    passes.kickRefinementIfIdle();
    expect(passes.kickPending).toBe(true);
    passes.dispose();
    await queued;
    expect(passes.locked).toBe(false);
    expect(passes.kickPending).toBe(false);
    expect(host.queue.hasPending()).toBe(false);
  });
});
