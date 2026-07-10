/**
 * Unit tests for ProgressiveMonitorAdapter.
 *
 * Regression guard: the adapter must RE-PATH inner per-LOD events to the
 * parent node path. Without that, the monitor records a metrics entry per
 * `additive_<i>` sub-path and `getGlobalStats` double-counts the node. These
 * tests pin the re-pathing, the fan-out, and the metrics aggregation.
 */

import { describe, it, expect, vi } from 'vitest';
import { ProgressiveMonitorAdapter } from '../../../../data/loaders/progressive-monitor-adapter';
import type {
  LoaderMetrics,
  LoaderMonitor,
  MonitorEvent,
  MonitorEventListener,
  QueryInfo,
} from '../../../../types/data-monitor-types';

/** A stub inner loader that captures the listener registered on it and lets a
 *  test fire an event through that listener. */
function makeInnerLoader(path: string, metricsOver: Partial<LoaderMetrics> = {}) {
  let listener: MonitorEventListener | null = null;
  const stub = {
    addEventListener: vi.fn((l: MonitorEventListener) => {
      listener = l;
    }),
    removeEventListener: vi.fn(() => {
      listener = null;
    }),
    getActiveQueries: vi.fn((): QueryInfo[] => []),
    getMetrics: vi.fn(
      (): LoaderMetrics => ({
        type: 'point-spatial-index',
        path,
        queries: 0,
        loads: 0,
        evictions: 0,
        errors: 0,
        elementsLoaded: 0,
        bytesLoaded: 0,
        visibleElements: 0,
        avgQueryTime: 0,
        avgLoadTime: 0,
        memoryUsed: 0,
        memoryLimit: 0,
        ...metricsOver,
      })
    ),
    /** Test helper: emit an event as this inner loader would, with ITS path. */
    emit(event: Partial<MonitorEvent> = {}): void {
      listener?.({
        type: 'query',
        loader: 'point-spatial-index',
        timestamp: 1,
        data: { path },
        ...event,
      } as MonitorEvent);
    },
  };
  return stub;
}

describe('ProgressiveMonitorAdapter', () => {
  it('re-paths inner-LOD events to the parent path (double-count regression)', () => {
    const a = makeInnerLoader('/points/additive_0');
    const b = makeInnerLoader('/points/additive_1');
    const adapter = new ProgressiveMonitorAdapter(
      () => [a, b] as unknown as LoaderMonitor[],
      '/points'
    );

    const received: MonitorEvent[] = [];
    adapter.addEventListener((e) => received.push(e));

    // Both inner loaders got a (wrapped) listener registered.
    expect(a.addEventListener).toHaveBeenCalledTimes(1);
    expect(b.addEventListener).toHaveBeenCalledTimes(1);

    // When an inner loader emits with its sub-path, the caller sees the PARENT path.
    a.emit({ data: { path: '/points/additive_0', latency: 5 } });
    b.emit({ data: { path: '/points/additive_1' } });

    expect(received).toHaveLength(2);
    expect(received[0].data.path).toBe('/points');
    expect(received[1].data.path).toBe('/points');
    // Non-path event fields are preserved.
    expect(received[0].data.latency).toBe(5);
    expect(received[0].type).toBe('query');
  });

  it('removeEventListener detaches the wrapped listener from every inner loader', () => {
    const a = makeInnerLoader('/p/additive_0');
    const adapter = new ProgressiveMonitorAdapter(() => [a] as unknown as LoaderMonitor[], '/p');
    const listener = vi.fn();

    adapter.addEventListener(listener);
    adapter.removeEventListener(listener);
    expect(a.removeEventListener).toHaveBeenCalledTimes(1);

    // After removal, an inner emit no longer reaches the caller.
    a.emit();
    expect(listener).not.toHaveBeenCalled();
  });

  it('is idempotent: adding the same listener twice does not double-register', () => {
    const a = makeInnerLoader('/p/additive_0');
    const adapter = new ProgressiveMonitorAdapter(() => [a] as unknown as LoaderMonitor[], '/p');
    const received: MonitorEvent[] = [];
    const listener: MonitorEventListener = (e) => received.push(e);

    adapter.addEventListener(listener);
    adapter.addEventListener(listener); // duplicate — must be ignored
    expect(a.addEventListener).toHaveBeenCalledTimes(1);

    a.emit();
    expect(received).toHaveLength(1); // fires once, not twice
  });

  it('removeEventListener is a no-op for an unknown listener', () => {
    const a = makeInnerLoader('/p/additive_0');
    const adapter = new ProgressiveMonitorAdapter(() => [a] as unknown as LoaderMonitor[], '/p');
    expect(() => adapter.removeEventListener(vi.fn())).not.toThrow();
    expect(a.removeEventListener).not.toHaveBeenCalled();
  });

  it('getMetrics aggregates inner metrics under the parent path', () => {
    const a = makeInnerLoader('/p/additive_0', { queries: 2, elementsLoaded: 100 });
    const b = makeInnerLoader('/p/additive_1', { queries: 3, elementsLoaded: 50 });
    const adapter = new ProgressiveMonitorAdapter(() => [a, b] as unknown as LoaderMonitor[], '/p');
    const m = adapter.getMetrics();
    expect(m.path).toBe('/p');
    expect(m.queries).toBe(5);
    expect(m.elementsLoaded).toBe(150);
  });

  it('getActiveQueries merges inner lists and re-paths them to the parent', () => {
    const a = makeInnerLoader('/p/additive_0');
    const b = makeInnerLoader('/p/additive_1');
    a.getActiveQueries.mockReturnValue([{ id: 'x', path: '/p/additive_0' } as QueryInfo]);
    b.getActiveQueries.mockReturnValue([
      { id: 'y', path: '/p/additive_1' } as QueryInfo,
      { id: 'z', path: '/p/additive_1' } as QueryInfo,
    ]);
    const adapter = new ProgressiveMonitorAdapter(() => [a, b] as unknown as LoaderMonitor[], '/p');
    const queries = adapter.getActiveQueries();
    expect(queries).toHaveLength(3);
    expect(queries.every((q) => q.path === '/p')).toBe(true);
    // Order is preserved (a's queries first, then b's) and ids are intact.
    expect(queries.map((q) => q.id)).toEqual(['x', 'y', 'z']);
  });

  it('reflects a live (post-dispose) empty loader set without throwing', () => {
    let loaders = [makeInnerLoader('/p/additive_0')];
    const adapter = new ProgressiveMonitorAdapter(
      () => loaders as unknown as LoaderMonitor[],
      '/p'
    );
    loaders = []; // simulate dispose() clearing lodLoaders
    expect(adapter.getActiveQueries()).toEqual([]);
    expect(adapter.getMetrics().path).toBe('/p');
    expect(() => adapter.addEventListener(vi.fn())).not.toThrow();
  });
});
