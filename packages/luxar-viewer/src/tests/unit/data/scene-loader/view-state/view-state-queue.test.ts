/**
 * Direct tests for ViewStateQueue — the "update" phase of the
 * scene-loader split. End-to-end behaviour (retry-lock + queued
 * updateView drain, predictive prefetch) is also covered by
 * scene-loader-initial-failure.test.ts and scene-loader-lifecycle.test.ts;
 * these tests guard the queue in isolation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ViewStateQueue } from '../../../../../data/scene-loader/view-state/view-state-queue';
import type { ViewState } from '../../../../../data/data-loader-types';

const dispatchSpy = vi.fn();
vi.mock('../../../../../data/scene-loader/view-state/predicted-view-state', () => ({
  dispatchPredictivePrefetch: (...args: unknown[]) => dispatchSpy(...args),
}));

const baseViewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 4],
  tolerance: [0, 0, 0, 1],
};

describe('ViewStateQueue — pending state slot', () => {
  let queue: ViewStateQueue;
  beforeEach(() => {
    queue = new ViewStateQueue();
  });

  it('starts empty', () => {
    expect(queue.hasPending()).toBe(false);
    expect(queue.takePending()).toBeNull();
  });

  it('setPending makes hasPending true', () => {
    queue.setPending({ slicePosition: [1, 2, 3, 4] });
    expect(queue.hasPending()).toBe(true);
  });

  it('takePending returns the queued state and clears the slot', () => {
    const state = { slicePosition: [1, 2, 3, 4] };
    queue.setPending(state);
    expect(queue.takePending()).toBe(state);
    expect(queue.hasPending()).toBe(false);
    expect(queue.takePending()).toBeNull();
  });

  it('setPending overwrites the previously-queued state (latest-wins)', () => {
    queue.setPending({ slicePosition: [1, 1, 1, 1] });
    queue.setPending({ slicePosition: [9, 9, 9, 9] });
    const taken = queue.takePending();
    expect(taken?.slicePosition).toEqual([9, 9, 9, 9]);
  });

  it('re-entrancy: takePending called twice with one queued state — only the first returns it', () => {
    const state = { slicePosition: [5, 6, 7, 8] };
    queue.setPending(state);
    // First take consumes the slot.
    expect(queue.takePending()).toBe(state);
    // Second take sees an empty slot.
    expect(queue.takePending()).toBeNull();
    expect(queue.hasPending()).toBe(false);
  });
});

describe('ViewStateQueue.drain', () => {
  let queue: ViewStateQueue;
  beforeEach(() => {
    queue = new ViewStateQueue();
  });

  it('does nothing when no state is queued', async () => {
    const trigger = vi.fn().mockResolvedValue(undefined);
    queue.drain(trigger);
    await Promise.resolve();
    expect(trigger).not.toHaveBeenCalled();
  });

  it('fires triggerUpdate with the queued state on a microtask', async () => {
    const trigger = vi.fn().mockResolvedValue(undefined);
    const state = { slicePosition: [1, 2, 3, 4] };
    queue.setPending(state);
    queue.drain(trigger);
    // Drain runs on a microtask — has not fired yet.
    expect(trigger).not.toHaveBeenCalled();
    await Promise.resolve();
    expect(trigger).toHaveBeenCalledWith(state);
  });

  it('clears the pending slot before firing the trigger', async () => {
    const trigger = vi.fn().mockResolvedValue(undefined);
    queue.setPending({ slicePosition: [1, 2, 3, 4] });
    expect(queue.hasPending()).toBe(true);
    queue.drain(trigger);
    expect(queue.hasPending()).toBe(false);
  });

  it('re-entrancy: drain called twice with one queued state — only the first fires the trigger', async () => {
    const trigger = vi.fn().mockResolvedValue(undefined);
    const state = { slicePosition: [1, 2, 3, 4] };
    queue.setPending(state);

    // First drain consumes the slot (synchronously clears it).
    queue.drain(trigger);
    // Second drain sees an empty slot → no-op (doesn't schedule a second fire).
    queue.drain(trigger);

    await Promise.resolve();
    // Only the first drain's microtask fired the trigger.
    expect(trigger).toHaveBeenCalledTimes(1);
    expect(trigger).toHaveBeenCalledWith(state);
  });

  it('swallows trigger rejection (logs warning) without throwing', async () => {
    const trigger = vi.fn().mockRejectedValue(new Error('synthetic drain failure'));
    queue.setPending({ slicePosition: [1, 2, 3, 4] });
    queue.drain(trigger);
    // Wait two microtasks: the drain microtask + the catch microtask.
    await Promise.resolve();
    await Promise.resolve();
    expect(trigger).toHaveBeenCalled();
    // No unhandled rejection — the test wouldn't reach this line if the
    // catch chain were broken.
  });
});

describe('ViewStateQueue.dispatchPrefetch', () => {
  let queue: ViewStateQueue;
  beforeEach(() => {
    queue = new ViewStateQueue();
    dispatchSpy.mockReset();
  });

  it('snapshots the current view-state on first call but does not fire prefetch', async () => {
    const loader = {};
    queue.dispatchPrefetch('/p', baseViewState, loader);
    await new Promise((r) => queueMicrotask(() => r(null)));
    expect(dispatchSpy).not.toHaveBeenCalled();
  });

  it('fires prefetch on the second call, with prev + current snapshots', async () => {
    const loader = {};
    const first: ViewState = { ...baseViewState, slicePosition: [0, 0, 0, 4] };
    const second: ViewState = { ...baseViewState, slicePosition: [0, 0, 0, 5] };
    queue.dispatchPrefetch('/p', first, loader);
    queue.dispatchPrefetch('/p', second, loader);
    await new Promise((r) => queueMicrotask(() => r(null)));
    expect(dispatchSpy).toHaveBeenCalledTimes(1);
    const [prev, current, loaders] = dispatchSpy.mock.calls[0];
    expect(prev.slicePosition).toEqual([0, 0, 0, 4]);
    expect(current.slicePosition).toEqual([0, 0, 0, 5]);
    expect(loaders).toEqual([loader]);
  });

  it('snapshots immutably — mutating source arrays after dispatch does not affect saved snapshot', async () => {
    const loader = {};
    const mutable = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 4],
      tolerance: [0, 0, 0, 1],
    };
    queue.dispatchPrefetch('/p', mutable, loader);

    // Mutate the source.
    mutable.slicePosition[3] = 999;

    // Second call — the saved prev should retain the pre-mutation values.
    queue.dispatchPrefetch('/p', { ...baseViewState, slicePosition: [0, 0, 0, 5] }, loader);
    await new Promise((r) => queueMicrotask(() => r(null)));
    expect(dispatchSpy).toHaveBeenCalledTimes(1);
    const [prev] = dispatchSpy.mock.calls[0];
    expect(prev.slicePosition).toEqual([0, 0, 0, 4]);
  });

  it('keys snapshots per path — different paths track independently', async () => {
    const loader = {};
    queue.dispatchPrefetch('/a', baseViewState, loader);
    queue.dispatchPrefetch('/b', baseViewState, loader);
    // No prev for either path yet — no dispatches.
    await new Promise((r) => queueMicrotask(() => r(null)));
    expect(dispatchSpy).not.toHaveBeenCalled();

    queue.dispatchPrefetch('/a', { ...baseViewState, slicePosition: [0, 0, 0, 5] }, loader);
    await new Promise((r) => queueMicrotask(() => r(null)));
    expect(dispatchSpy).toHaveBeenCalledTimes(1);

    queue.dispatchPrefetch('/b', { ...baseViewState, slicePosition: [0, 0, 0, 6] }, loader);
    await new Promise((r) => queueMicrotask(() => r(null)));
    expect(dispatchSpy).toHaveBeenCalledTimes(2);
  });

  it('forgetPath drops the saved snapshot for one path only', async () => {
    const loader = {};
    queue.dispatchPrefetch('/a', baseViewState, loader);
    queue.dispatchPrefetch('/b', baseViewState, loader);

    queue.forgetPath('/a');

    // Re-call /a — should be first-call again (no prev), no dispatch.
    queue.dispatchPrefetch('/a', { ...baseViewState, slicePosition: [0, 0, 0, 5] }, loader);
    // Re-call /b — should fire (prev exists).
    queue.dispatchPrefetch('/b', { ...baseViewState, slicePosition: [0, 0, 0, 5] }, loader);

    await new Promise((r) => queueMicrotask(() => r(null)));
    expect(dispatchSpy).toHaveBeenCalledTimes(1);
  });

  it('clearPrev drops all saved snapshots', async () => {
    const loader = {};
    queue.dispatchPrefetch('/a', baseViewState, loader);
    queue.dispatchPrefetch('/b', baseViewState, loader);

    queue.clearPrev();

    // Both paths now first-call again — neither dispatches.
    queue.dispatchPrefetch('/a', { ...baseViewState, slicePosition: [0, 0, 0, 5] }, loader);
    queue.dispatchPrefetch('/b', { ...baseViewState, slicePosition: [0, 0, 0, 6] }, loader);
    await new Promise((r) => queueMicrotask(() => r(null)));
    expect(dispatchSpy).not.toHaveBeenCalled();
  });
});
