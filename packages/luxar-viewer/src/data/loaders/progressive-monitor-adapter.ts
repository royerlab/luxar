/**
 * Shared `LoaderMonitor` surface for the progressive (multi-additive-LOD)
 * loaders. A progressive node wraps N inner per-LOD spatial-index loaders;
 * to the data monitor it must look like a SINGLE loader keyed by the parent
 * node path.
 *
 * The critical job here is **re-pathing**: each inner loader emits monitor
 * events stamped with its own `additive_<i>` sub-path. If those reached the
 * monitor unchanged, `updateMetricsFromEvent` would record a separate metrics
 * entry per sub-path — double-counting throughput/memory in `getGlobalStats`
 * (which sums every entry) and showing stale per-LOD rows in the loader list.
 * Re-stamping events with the parent path makes the inner LODs an invisible
 * implementation detail, so the node behaves exactly like a non-progressive
 * loader at the monitor boundary.
 *
 * Kept as a standalone adapter (rather than duplicated inline) so the three
 * progressive loaders stay byte-for-byte symmetric and the surface is
 * unit-testable in isolation.
 *
 * @module data/loaders/progressive-monitor-adapter
 */

import type {
  LoaderMetrics,
  LoaderMonitor,
  MonitorEvent,
  MonitorEventListener,
  QueryInfo,
} from '../../types/data-monitor-types';
import { aggregateLoaderMetrics } from './aggregate-loader-metrics';

export class ProgressiveMonitorAdapter {
  /** Maps a caller's listener → the re-pathing wrapper registered on inner loaders. */
  private readonly wrapped = new Map<MonitorEventListener, MonitorEventListener>();

  /**
   * @param getLoaders live accessor for the inner per-LOD loaders (a getter so
   *   it reflects post-`dispose()` clearing, where the array is emptied).
   * @param path the parent node path reported as this aggregate's identity.
   */
  constructor(
    private readonly getLoaders: () => LoaderMonitor[],
    private readonly path: string
  ) {}

  addEventListener(listener: MonitorEventListener): void {
    // Idempotent: re-registering the same listener would orphan the previous
    // re-path wrapper on the inner loaders (double-fire), so guard against it.
    if (this.wrapped.has(listener)) return;
    const repath: MonitorEventListener = (event: MonitorEvent) =>
      listener({ ...event, data: { ...event.data, path: this.path } });
    this.wrapped.set(listener, repath);
    for (const loader of this.getLoaders()) loader.addEventListener(repath);
  }

  removeEventListener(listener: MonitorEventListener): void {
    const repath = this.wrapped.get(listener);
    if (!repath) return;
    for (const loader of this.getLoaders()) loader.removeEventListener(repath);
    this.wrapped.delete(listener);
  }

  getActiveQueries(): QueryInfo[] {
    // Re-path to the parent node for the same single-identity reason as
    // events — so a consumer never sees an internal `additive_<i>` sub-path.
    return this.getLoaders().flatMap((loader) =>
      loader.getActiveQueries().map((q) => ({ ...q, path: this.path }))
    );
  }

  getMetrics(): LoaderMetrics {
    return aggregateLoaderMetrics(
      this.getLoaders().map((loader) => loader.getMetrics()),
      this.path
    );
  }
}
