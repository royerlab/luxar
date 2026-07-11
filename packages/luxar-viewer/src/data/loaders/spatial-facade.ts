/**
 * Shared facade-level orchestration for the three spatial-index loaders
 * (Points / Lines / GSplats). Each helper here used to exist as three
 * byte-identical private methods; hoisting them keeps the facades symmetric
 * and unable to drift.
 *
 * The loaders build ONE {@link SpatialFacadeCtx} in their constructor
 * (`this`-bound accessors over their private monitor state) and pass it to
 * every helper.
 *
 * @module data/loaders/spatial-facade
 */

import { restoreLadder, storeLadder, type SliceViewLike } from './progressive/slice-cache-helper';
import {
  computeLoadLatency,
  finishQueryTracking,
  recordLoadEvent,
  type LoaderMetricsCounters,
  type QueryMetricsCounters,
} from './loader-metrics';
import { isAbortError } from './abort-error';
import { ResidencyAccumulator } from '../../cache/residency-probe';
import type { SliceCache } from '../../cache/slice-cache';
import type { LoaderType, MonitorEvent, QueryInfo } from '../../types/data-monitor-types';

/** View-state shape the facade helpers need: the S-cache key fields plus the
 *  per-pass playback/prefetch riders (see `data-loader-types.ts`). */
export type FacadeViewState = SliceViewLike & {
  frameBudgetMs?: number | null;
  prefetch?: boolean;
};

/** Mutable metrics slice the facade helpers update. `LoaderMetrics` is
 *  structurally assignable, so loaders pass `this.metrics` unchanged. */
export type FacadeMetrics = LoaderMetricsCounters &
  QueryMetricsCounters & { errors: number; memoryUsed: number };

/**
 * Per-loader context for the shared facade helpers. Built once in each
 * loader's constructor; every field is a stable reference or a `this`-bound
 * accessor, so the ctx never goes stale.
 */
export interface SpatialFacadeCtx {
  /** The loader's live metrics record (mutated in place). */
  metrics: FacadeMetrics;
  /** The loader's active-query map (mutated in place). */
  activeQueries: Map<string, QueryInfo>;
  /** Loader type literal for emitted events. */
  loader: LoaderType;
  /** Node path (S-cache namespace + event payload). */
  path: string;
  /** Shared S-cache; null for progressive sub-LOD instances. */
  sliceCache: SliceCache | null;
  /** Monotonic per-loader counter for same-millisecond queryId uniqueness. */
  nextQueryId(): number;
  /** Current accumulator allocation in MB (0 when absent). */
  accumulatorMemoryMB(): number;
  /** Emit a monitor event through the loader's listener set. */
  emit(event: MonitorEvent): void;
}

/**
 * The shared `loadX` facade template: S-cache restore → internal load →
 * query close-out → S-cache store, with the abort-aware error branch.
 *
 * - A cached slice is a 1-element ladder under the progressive loaders' key
 *   contract; a hit returns the SAME payload object on every same-view call,
 *   so the downstream reference-identity commit check turns same-slice
 *   revisits into no-ops.
 * - The store clones (helper responsibility — the decoded arrays alias the
 *   loader's reused accumulator). Aborted loads throw and never reach it.
 * - A superseded scrub aborts the in-flight read on purpose; runLoaderUpdates
 *   classifies it as 'superseded' (not a failure), so the error branch does
 *   not inflate the error counter or flood the monitor's error stream for
 *   abort errors — but it ALWAYS finishes query tracking (removes the active
 *   query, records timing).
 *
 * `loadInternal` receives the generated `queryId`/`startTime` so it can
 * register the active query itself (its registration must precede the array
 * loads — `recordLoadMetrics` reads the first active query's startTime).
 */
export async function loadSliceWithCache<TData extends object>(
  ctx: SpatialFacadeCtx,
  viewState: FacadeViewState,
  loadInternal: (queryId: string, startTime: number) => Promise<TData>
): Promise<TData> {
  const cached = restoreLadder<TData>(ctx.sliceCache, ctx.path, viewState, 1);
  if (cached) return cached[0];

  const startTime = Date.now();
  const queryId = `${ctx.path}-${startTime}-${ctx.nextQueryId()}`;

  try {
    const result = await loadInternal(queryId, startTime);
    finishQueryTracking(ctx.activeQueries, ctx.metrics, queryId, startTime, 'complete');
    storeLadder(ctx.sliceCache, ctx.path, viewState, [result], {
      scan: viewState.frameBudgetMs != null,
      pin: viewState.prefetch === true,
    });
    return result;
  } catch (err) {
    finishQueryTracking(ctx.activeQueries, ctx.metrics, queryId, startTime, 'error');
    if (!isAbortError(err)) {
      ctx.metrics.errors++;
      ctx.emit({
        type: 'error',
        loader: ctx.loader,
        timestamp: Date.now(),
        data: {
          path: ctx.path,
          error: String(err),
        },
      });
    }
    throw err;
  }
}

/**
 * Update metrics + emit a 'load' event after one attribute-array load.
 * Load latency is measured from the FIRST active query's startTime (the
 * demand query that triggered this load), so the active-query registration
 * must precede the array loads.
 */
export function recordLoadMetrics(
  ctx: SpatialFacadeCtx,
  arrayName: string,
  elements: number,
  output: ArrayBufferView
): void {
  const queryStart = ctx.activeQueries.values().next().value?.startTime;
  const loadTime = computeLoadLatency(queryStart);
  const bytes = output.byteLength;

  recordLoadEvent(ctx.metrics, elements, bytes, loadTime);

  // Resident memory = current accumulator allocation (MB → bytes). Assignment
  // (not +=): memoryUsed is a live footprint that grows/shrinks with the pool,
  // unlike the cumulative bytesLoaded counter updated above.
  ctx.metrics.memoryUsed = Math.round(ctx.accumulatorMemoryMB() * 1024 * 1024);

  ctx.emit({
    type: 'load',
    loader: ctx.loader,
    timestamp: Date.now(),
    data: {
      path: ctx.path,
      arrayName,
      elements,
      memory: bytes,
      latency: loadTime,
    },
  });
}

/**
 * Run one demand load with the per-update abort signal PUBLISHED for the
 * L0-proxy chokepoint (via the loader's `setSignal` field setter), cleared in
 * `finally` so a later cache hit / prefetch isn't seen as abortable. The
 * shared body of the three loaders' `updateView`.
 */
export async function runWithActiveSignal<TData>(
  setSignal: (signal: AbortSignal | null) => void,
  signal: AbortSignal | undefined,
  load: () => Promise<TData>
): Promise<TData> {
  setSignal(signal ?? null);
  try {
    return await load();
  } finally {
    setSignal(null);
  }
}

/**
 * Run one demand load with a cache-residency probe attached, reporting
 * whether the load was served entirely from cache. Drives the progressive
 * loader's per-frame decision to keep loading the next LOD level (resident)
 * or stop and let the refinement loop continue after a miss. A load that
 * touches no chunks counts as resident (`allResident: true`). The shared
 * body of the three loaders' `updateViewWithResidency`.
 */
export async function runWithResidencyProbe<TData>(
  setProbe: (probe: ResidencyAccumulator | null) => void,
  load: () => Promise<TData>
): Promise<{ data: TData; allResident: boolean }> {
  const probe = new ResidencyAccumulator();
  setProbe(probe);
  try {
    const data = await load();
    return { data, allResident: probe.allResident };
  } finally {
    setProbe(null);
  }
}
