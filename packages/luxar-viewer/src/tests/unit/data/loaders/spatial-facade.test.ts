/**
 * Direct unit tests for the shared spatial-facade helpers
 * (`data/loaders/spatial-facade.ts`). The three geometry loaders exercise
 * these transitively; this file pins the helpers' own contracts in
 * isolation — signal/probe publication + cleanup (including on throw) and
 * the load template's close-out / store / error branches.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  loadSliceWithCache,
  recordLoadMetrics,
  runWithActiveSignal,
  runWithResidencyProbe,
  makeInitialLoaderMetrics,
  type SpatialFacadeCtx,
} from '../../../../data/loaders';
import { SliceCache } from '../../../../cache/slice-cache';
import type { MonitorEvent } from '../../../../types/data-monitor-types';

function makeCtx(overrides: Partial<SpatialFacadeCtx> = {}): {
  ctx: SpatialFacadeCtx;
  events: MonitorEvent[];
} {
  const events: MonitorEvent[] = [];
  let nextId = 0;
  const ctx: SpatialFacadeCtx = {
    metrics: makeInitialLoaderMetrics('point-spatial-index', '/node'),
    activeQueries: new Map(),
    loader: 'point-spatial-index',
    path: '/node',
    sliceCache: null,
    nextQueryId: () => nextId++,
    accumulatorMemoryMB: () => 2,
    emit: (e) => events.push(e),
    ...overrides,
  };
  return { ctx, events };
}

/** A hidden-dim view (displayDims < slicePosition) so the S-cache engages. */
const hiddenDimView = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 5],
  tolerance: [0, 0, 0, 0.25],
};

describe('runWithActiveSignal', () => {
  it('publishes the signal for the load and clears it in finally', async () => {
    const seen: Array<AbortSignal | null> = [];
    const setSignal = (s: AbortSignal | null) => seen.push(s);
    const controller = new AbortController();

    const result = await runWithActiveSignal(setSignal, controller.signal, async () => {
      expect(seen).toEqual([controller.signal]);
      return 42;
    });

    expect(result).toBe(42);
    expect(seen).toEqual([controller.signal, null]);
  });

  it('publishes null when no signal is supplied', async () => {
    const seen: Array<AbortSignal | null> = [];
    await runWithActiveSignal(
      (s) => seen.push(s),
      undefined,
      async () => 1
    );
    expect(seen).toEqual([null, null]);
  });

  it('clears the signal even when the load throws', async () => {
    const seen: Array<AbortSignal | null> = [];
    const controller = new AbortController();
    await expect(
      runWithActiveSignal(
        (s) => seen.push(s),
        controller.signal,
        async () => {
          throw new Error('boom');
        }
      )
    ).rejects.toThrow('boom');
    expect(seen[seen.length - 1]).toBeNull();
  });
});

describe('runWithResidencyProbe', () => {
  it('attaches a probe for the load, reports allResident, and clears it', async () => {
    const seen: unknown[] = [];
    const { data, allResident } = await runWithResidencyProbe(
      (p) => seen.push(p),
      async () => 'payload'
    );
    expect(data).toBe('payload');
    // A load that touches no chunks counts as resident.
    expect(allResident).toBe(true);
    expect(seen).toHaveLength(2);
    expect(seen[0]).not.toBeNull();
    expect(seen[1]).toBeNull();
  });

  it('reports allResident=false when the probe records a miss', async () => {
    let probe: { record: (hit: boolean) => void } | null = null;
    const { allResident } = await runWithResidencyProbe(
      (p) => {
        if (p) probe = p;
      },
      async () => {
        probe!.record(false); // one chunk missed the cache
        return 'x';
      }
    );
    expect(allResident).toBe(false);
  });

  it('clears the probe even when the load throws', async () => {
    const seen: unknown[] = [];
    await expect(
      runWithResidencyProbe(
        (p) => seen.push(p),
        async () => {
          throw new Error('boom');
        }
      )
    ).rejects.toThrow('boom');
    expect(seen[seen.length - 1]).toBeNull();
  });
});

describe('loadSliceWithCache', () => {
  it('closes out the query and stores the result on success', async () => {
    const sliceCache = new SliceCache({ maxSize: 1024 * 1024 });
    const { ctx } = makeCtx({ sliceCache });
    ctx.metrics.queries = 1; // internal registered the query

    const payload = { data: new Float32Array([1, 2, 3]) };
    const result = await loadSliceWithCache(ctx, hiddenDimView, async (queryId, startTime) => {
      expect(queryId).toBe('/node-' + startTime + '-0');
      ctx.activeQueries.set(queryId, {
        id: queryId,
        loader: ctx.loader,
        path: ctx.path,
        startTime,
        status: 'loading',
      });
      return payload;
    });

    expect(result).toBe(payload);
    expect(ctx.activeQueries.size).toBe(0); // closed out
    expect(sliceCache.getStats().count).toBe(1); // stored
  });

  it('a same-view revisit restores the cached clone without calling the internal', async () => {
    const sliceCache = new SliceCache({ maxSize: 1024 * 1024 });
    const { ctx } = makeCtx({ sliceCache });
    ctx.metrics.queries = 1;

    await loadSliceWithCache(ctx, hiddenDimView, async () => ({
      data: new Float32Array([7]),
    }));
    const internal = vi.fn();
    const restored = await loadSliceWithCache(ctx, hiddenDimView, internal);

    expect(internal).not.toHaveBeenCalled();
    expect((restored as { data: Float32Array }).data[0]).toBe(7);
  });

  it('on failure: closes out the query, counts the error, emits, and rethrows', async () => {
    const { ctx, events } = makeCtx();
    ctx.metrics.queries = 1;

    await expect(
      loadSliceWithCache(ctx, hiddenDimView, async (queryId, startTime) => {
        ctx.activeQueries.set(queryId, {
          id: queryId,
          loader: ctx.loader,
          path: ctx.path,
          startTime,
          status: 'loading',
        });
        throw new Error('Load failed');
      })
    ).rejects.toThrow('Load failed');

    expect(ctx.activeQueries.size).toBe(0); // never leaks the entry
    expect(ctx.metrics.errors).toBe(1);
    expect(events).toEqual([
      expect.objectContaining({
        type: 'error',
        loader: 'point-spatial-index',
        data: expect.objectContaining({
          path: '/node',
          error: expect.stringContaining('Load failed'),
        }),
      }),
    ]);
  });

  it('an abort error closes out the query but neither counts nor emits', async () => {
    const { ctx, events } = makeCtx();
    ctx.metrics.queries = 1;

    await expect(
      loadSliceWithCache(ctx, hiddenDimView, async () => {
        throw new DOMException('aborted', 'AbortError');
      })
    ).rejects.toThrow();

    expect(ctx.metrics.errors).toBe(0);
    expect(events).toEqual([]);
  });

  it('queryIds are unique across same-millisecond loads (nextQueryId suffix)', async () => {
    const { ctx } = makeCtx();
    const ids: string[] = [];
    const now = vi.spyOn(Date, 'now').mockReturnValue(12345);
    try {
      await loadSliceWithCache(ctx, hiddenDimView, async (queryId) => {
        ids.push(queryId);
        return { a: 1 };
      });
      await loadSliceWithCache(
        ctx,
        { ...hiddenDimView, slicePosition: [0, 0, 0, 6] },
        async (queryId) => {
          ids.push(queryId);
          return { a: 2 };
        }
      );
    } finally {
      now.mockRestore();
    }
    expect(new Set(ids).size).toBe(2);
  });
});

describe('recordLoadMetrics', () => {
  it('updates counters, memory footprint, and emits a load event', () => {
    const { ctx, events } = makeCtx();
    ctx.activeQueries.set('q', {
      id: 'q',
      loader: ctx.loader,
      path: ctx.path,
      startTime: Date.now() - 25,
      status: 'loading',
    });

    recordLoadMetrics(ctx, 'positions', 500, new Float32Array(1500));

    expect(ctx.metrics.loads).toBe(1);
    expect(ctx.metrics.elementsLoaded).toBe(500);
    expect(ctx.metrics.bytesLoaded).toBe(1500 * 4);
    expect(ctx.metrics.memoryUsed).toBe(2 * 1024 * 1024); // accumulatorMemoryMB() = 2
    expect(events).toEqual([
      expect.objectContaining({
        type: 'load',
        data: expect.objectContaining({ arrayName: 'positions', elements: 500 }),
      }),
    ]);
  });
});
