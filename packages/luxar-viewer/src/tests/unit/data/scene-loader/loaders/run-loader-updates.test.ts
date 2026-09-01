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
import * as THREE from 'three';
import {
  filterPartitionVisibleLoaders,
  isPartitionPathVisible,
  runLoaderUpdates,
} from '../../../../../data/scene-loader/loaders/run-loader-updates';
import { ViewStateQueue } from '../../../../../data/scene-loader/view-state/view-state-queue';
import { LoaderRegistry } from '../../../../../data/scene-loader/loaders/loader-registry';
import { ArchiveFaultError } from '../../../../../cache/chunk-source';
import { log } from '../../../../../utils/log';

function makeCtx() {
  const viewStateQueue = new ViewStateQueue();
  const forgetPath = vi.spyOn(viewStateQueue, 'forgetPath');
  // A real registry rather than a bare Map: failure recording is now routed
  // through `recordFailure`, which also persists the classified error kind.
  const registry = new LoaderRegistry();
  const onArchiveFault = vi.fn();
  return {
    profiler: null,
    viewStateQueue,
    registry,
    onArchiveFault,
    forgetPath,
    failedLoaders: registry.failedLoaders,
  };
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
      const rollbackToPassStart = vi.fn();
      const loaders = new Map<string, object>([['/scene/points', { rollbackToPassStart }]]);

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
      expect(rollbackToPassStart).not.toHaveBeenCalled();
    }
  );

  it('still records genuine (non-abort) errors as failures and forgets the path', async () => {
    const ctx = makeCtx();
    const rollbackToPassStart = vi.fn().mockReturnValue(2);
    const loaders = new Map<string, object>([['/scene/points', { rollbackToPassStart }]]);
    const errorSpy = vi.spyOn(log, 'error').mockImplementation(() => {});

    try {
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
      expect(rollbackToPassStart).toHaveBeenCalledOnce();
      expect(errorSpy.mock.calls[0][1]).toContain('(unwound 2 level(s))');
    } finally {
      errorSpy.mockRestore();
    }
  });

  it('keeps the original failure path when rollback itself throws', async () => {
    const ctx = makeCtx();
    const rollbackToPassStart = vi.fn(() => {
      throw new Error('rollback failed');
    });
    const loaders = new Map<string, object>([['/scene/points', { rollbackToPassStart }]]);
    const errorSpy = vi.spyOn(log, 'error').mockImplementation(() => {});

    try {
      const results = await runLoaderUpdates(
        loaders,
        'Points',
        () => {
          throw new Error('real load failure');
        },
        ctx
      );

      expect(results[0].staged).toBeNull();
      expect(rollbackToPassStart).toHaveBeenCalledOnce();
      expect(ctx.failedLoaders.has('/scene/points')).toBe(true);
      expect(ctx.forgetPath).toHaveBeenCalledWith('/scene/points');
      expect(errorSpy.mock.calls[0][1]).toContain('real load failure');
      expect(errorSpy.mock.calls[0][1]).not.toContain('rollback failed');
    } finally {
      errorSpy.mockRestore();
    }
  });

  it('omits the unwind suffix when a non-progressive loader discards no levels', async () => {
    const ctx = makeCtx();
    const errorSpy = vi.spyOn(log, 'error').mockImplementation(() => {});

    try {
      await runLoaderUpdates(
        new Map<string, object>([['/scene/plain', {}]]),
        'Points',
        () => {
          throw new Error('plain load failure');
        },
        ctx
      );

      expect(errorSpy).toHaveBeenCalledOnce();
      expect(errorSpy.mock.calls[0][1]).not.toContain('unwound');
    } finally {
      errorSpy.mockRestore();
    }
  });

  it('hoists wrapped archive faults once without recording per-node failures', async () => {
    const ctx = makeCtx();
    const rollbackA = vi.fn().mockReturnValue(1);
    const rollbackB = vi.fn().mockReturnValue(1);
    const loaders = new Map<string, object>([
      ['/scene/points-a', { rollbackToPassStart: rollbackA }],
      ['/scene/points-b', { rollbackToPassStart: rollbackB }],
    ]);
    const fault = new ArchiveFaultError(
      'The archive URL has expired. Refresh the page with a new URL.',
      'https://example.test/scene.zip'
    );

    const results = await runLoaderUpdates(
      loaders,
      'Points',
      () => {
        throw new Error('loader wrapper', { cause: fault });
      },
      ctx
    );

    expect(results.map(({ staged }) => staged)).toEqual([null, null]);
    expect(ctx.onArchiveFault).toHaveBeenCalledOnce();
    expect(ctx.onArchiveFault).toHaveBeenCalledWith(fault);
    expect(ctx.failedLoaders.size).toBe(0);
    expect(ctx.forgetPath).not.toHaveBeenCalled();
    expect(rollbackA).toHaveBeenCalledOnce();
    expect(rollbackB).toHaveBeenCalledOnce();
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

  it('does not invoke loaders excluded by the visibility gate', async () => {
    const ctx = makeCtx();
    const loaders = new Map<string, object>([
      ['/scene/visible', {}],
      ['/scene/culled', {}],
    ]);
    const update = vi.fn((path: string) => Promise.resolve({ path }));

    const results = await runLoaderUpdates(loaders, 'Points', update, {
      ...ctx,
      shouldUpdatePath: (path) => path !== '/scene/culled',
    });

    expect(update).toHaveBeenCalledOnce();
    expect(update).toHaveBeenCalledWith('/scene/visible', {}, expect.anything());
    expect(results.map(({ staged }) => staged)).toEqual([{ path: '/scene/visible' }, null]);
    expect(ctx.forgetPath).toHaveBeenCalledOnce();
    expect(ctx.forgetPath).toHaveBeenCalledWith('/scene/culled');
  });

  it('finds a culled partition marker anywhere in the loader ancestor chain', () => {
    const root = new THREE.Group();
    const part = new THREE.Group();
    const nestedLod = new THREE.Group();
    const leaf = new THREE.Group();
    leaf.name = '/partition/part_4/level_2';
    root.add(part);
    part.add(nestedLod);
    nestedLod.add(leaf);

    expect(isPartitionPathVisible(root, leaf.name)).toBe(true);
    part.userData.partitionFrustumVisible = false;
    expect(isPartitionPathVisible(root, leaf.name)).toBe(false);
  });

  it('removes culled partition loaders from progressive refinement maps', () => {
    const root = new THREE.Group();
    const visible = new THREE.Group();
    visible.name = '/partition/part_0';
    const culled = new THREE.Group();
    culled.name = '/partition/part_1';
    culled.userData.partitionFrustumVisible = false;
    root.add(visible, culled);
    const visibleLoader = {};
    const culledLoader = {};

    const filtered = filterPartitionVisibleLoaders(
      root,
      new Map([
        [visible.name, visibleLoader],
        [culled.name, culledLoader],
      ])
    );

    expect([...filtered]).toEqual([[visible.name, visibleLoader]]);
  });
});
