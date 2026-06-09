/**
 * Unit tests for `runLoaderUpdates` error taxonomy (Guard G2).
 *
 * A per-update AbortSignal can cause an in-flight `updateView` to throw an
 * abort error (zarrita throws a DOMException named 'AbortError'; the worker
 * pool throws a 'WorkerAbortError'). Those mean "this update was superseded",
 * NOT "this loader failed". The helper must therefore:
 *   - return `{ staged: null }` (so the atomic commit leaves geometry untouched),
 *   - NOT record a failure in `failedLoaders`, and
 *   - NOT drop the path's predictive-prefetch baseline (`forgetPath`).
 * Genuine errors keep the existing failure-recording behavior.
 */

import { describe, it, expect, vi } from 'vitest';
import { runLoaderUpdates } from '../../../../../data/scene-loader/loaders/run-loader-updates';
import { ViewStateQueue } from '../../../../../data/scene-loader/view-state/view-state-queue';
import type { FailedLoaderInfo } from '../../../../../data/scene-loader/loaders/loader-registry';

function makeCtx() {
  const viewStateQueue = new ViewStateQueue();
  const forgetPath = vi.spyOn(viewStateQueue, 'forgetPath');
  const failedLoaders = new Map<string, FailedLoaderInfo>();
  return { profiler: null, viewStateQueue, failedLoaders, forgetPath };
}

/** An abort error shaped like the ones zarrita / the worker pool throw. */
function abortError(name: 'AbortError' | 'WorkerAbortError'): Error {
  const e = new Error(`${name}: superseded`);
  e.name = name;
  return e;
}

describe('runLoaderUpdates — abort taxonomy (G2)', () => {
  it.each(['AbortError', 'WorkerAbortError'] as const)(
    'treats %s as superseded: staged=null, no failure, no forgetPath',
    async (name) => {
      const ctx = makeCtx();
      const loaders = new Map<string, object>([['/scene/points', {}]]);

      const results = await runLoaderUpdates(
        loaders,
        'Points',
        () => {
          throw abortError(name);
        },
        ctx
      );

      expect(results).toHaveLength(1);
      expect(results[0].staged).toBeNull();
      expect(ctx.failedLoaders.size).toBe(0);
      expect(ctx.forgetPath).not.toHaveBeenCalled();
    }
  );

  it('still records genuine (non-abort) errors as failures and forgets the path', async () => {
    const ctx = makeCtx();
    const loaders = new Map<string, object>([['/scene/points', {}]]);

    const results = await runLoaderUpdates(
      loaders,
      'Points',
      () => {
        throw new Error('real load failure');
      },
      ctx
    );

    expect(results[0].staged).toBeNull();
    expect(ctx.failedLoaders.has('/scene/points')).toBe(true);
    expect(ctx.failedLoaders.get('/scene/points')?.retryCount).toBe(0);
    expect(ctx.forgetPath).toHaveBeenCalledWith('/scene/points');
  });

  it('passes through staged results unchanged on success', async () => {
    const ctx = makeCtx();
    const loaders = new Map<string, object>([['/scene/points', {}]]);

    const results = await runLoaderUpdates(
      loaders,
      'Points',
      () => Promise.resolve({ path: '/scene/points' }),
      ctx
    );

    expect(results[0].staged).toEqual({ path: '/scene/points' });
    expect(ctx.failedLoaders.size).toBe(0);
  });
});
