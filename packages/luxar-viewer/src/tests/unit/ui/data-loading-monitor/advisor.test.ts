/**
 * Unit tests for ui/data-loading-monitor/advisor.ts.
 *
 * The advisor is a pure analyzer: it ingests MonitorEvent /
 * LoaderMetrics / MemoryMetrics and emits Recommendation entries
 * keyed by id. Tests run the real implementation — no mocking — and
 * inspect the resulting recommendation map.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { LoadingAdvisor } from '../../../../ui/data-loading-monitor/advisor';
import type { LoaderMetrics, MonitorEvent } from '../../../../types/data-monitor-types';
import type { MemoryMetrics } from '../../../../ui/data-loading-monitor/templates';

const baseMetrics: LoaderMetrics = {
  type: 'point-spatial-index',
  path: '/test',
  queries: 0,
  loads: 0,
  evictions: 0,
  errors: 0,
  elementsLoaded: 0,
  bytesLoaded: 0,
  visiblePoints: 0,
  avgQueryTime: 0,
  avgLoadTime: 0,
  memoryUsed: 0,
  memoryLimit: 1_000_000,
};

function makeEvent(overrides: Partial<MonitorEvent> = {}): MonitorEvent {
  return {
    type: 'query',
    loader: 'point-spatial-index',
    timestamp: Date.now(),
    data: {},
    ...overrides,
  } as MonitorEvent;
}

describe('LoadingAdvisor', () => {
  let advisor: LoadingAdvisor;

  beforeEach(() => {
    advisor = new LoadingAdvisor();
  });

  describe('analyzeEvent', () => {
    it('records events without producing recommendations for benign inputs', () => {
      // 20 fast loads — no slow-load, no error-rate, no nothing.
      for (let i = 0; i < 20; i++) {
        advisor.analyzeEvent(makeEvent({ type: 'load', data: { latency: 50 } }));
      }
      expect(advisor.getRecommendations()).toEqual([]);
    });

    it('emits a slow-query recommendation when latency > highQueryTime', () => {
      advisor.analyzeEvent(makeEvent({ type: 'query', data: { latency: 250 } }));

      const recs = advisor.getRecommendations();
      const slow = recs.find((r) => r.id === 'slow-query');
      expect(slow).toBeDefined();
      expect(slow?.severity).toBe('warning');
      expect(slow?.value).toBe(250);
    });

    it('does not emit slow-query when latency is below threshold', () => {
      advisor.analyzeEvent(makeEvent({ type: 'query', data: { latency: 50 } }));
      expect(advisor.getRecommendations()).toEqual([]);
    });

    it('emits a slow-load recommendation when latency > highLoadTime', () => {
      advisor.analyzeEvent(makeEvent({ type: 'load', data: { latency: 1000 } }));

      const slow = advisor.getRecommendations().find((r) => r.id === 'slow-load');
      expect(slow).toBeDefined();
      expect(slow?.value).toBe(1000);
    });
  });

  describe('analyzeMetrics', () => {
    it('emits high-avg-query when avgQueryTime > 100ms (warning)', () => {
      advisor.analyzeMetrics({ ...baseMetrics, avgQueryTime: 150 });

      const rec = advisor.getRecommendations().find((r) => r.id === 'high-avg-query');
      expect(rec).toBeDefined();
      expect(rec?.severity).toBe('warning');
    });

    it('escalates high-avg-query to error when avgQueryTime > 200ms', () => {
      advisor.analyzeMetrics({ ...baseMetrics, avgQueryTime: 250 });
      const rec = advisor.getRecommendations().find((r) => r.id === 'high-avg-query');
      expect(rec?.severity).toBe('error');
    });

    it('emits high-memory recommendation when usage > 80 %', () => {
      advisor.analyzeMetrics({
        ...baseMetrics,
        memoryUsed: 850_000,
        memoryLimit: 1_000_000,
      });
      const rec = advisor.getRecommendations().find((r) => r.id === 'high-memory');
      expect(rec).toBeDefined();
      expect(rec?.severity).toBe('warning');
    });

    it('escalates high-memory to error when usage > 90 %', () => {
      advisor.analyzeMetrics({
        ...baseMetrics,
        memoryUsed: 950_000,
        memoryLimit: 1_000_000,
      });
      const rec = advisor.getRecommendations().find((r) => r.id === 'high-memory');
      expect(rec?.severity).toBe('error');
    });

    it('does NOT emit high-memory when memoryLimit is 0 (no per-loader cap)', () => {
      // Spatial-index loaders populate memoryUsed (resident bytes) but leave
      // memoryLimit at 0. Without the limit>0 guard, memoryUsed/0 = Infinity
      // would fire a spurious warning on every load.
      advisor.analyzeMetrics({
        ...baseMetrics,
        memoryUsed: 64 * 1024 * 1024,
        memoryLimit: 0,
      });
      expect(advisor.getRecommendations().find((r) => r.id === 'high-memory')).toBeUndefined();
    });

    it('emits low-efficiency when spatial index efficiency is below threshold', () => {
      advisor.analyzeMetrics({
        ...baseMetrics,
        spatialIndex: {
          gridShape: [1],
          gridOrigin: [0],
          cellSize: [1],
          occupiedCells: 1,
          totalCells: 1,
          avgCellsPerQuery: 1,
          avgPointsPerCell: 1,
          queryEfficiency: 0.1,
          rangesInCache: 0,
        },
      });
      const rec = advisor.getRecommendations().find((r) => r.id === 'low-efficiency');
      expect(rec).toBeDefined();
      expect(rec?.severity).toBe('info');
    });

    it('does not emit low-efficiency when efficiency is high', () => {
      advisor.analyzeMetrics({
        ...baseMetrics,
        spatialIndex: {
          gridShape: [1],
          gridOrigin: [0],
          cellSize: [1],
          occupiedCells: 1,
          totalCells: 1,
          avgCellsPerQuery: 1,
          avgPointsPerCell: 1,
          queryEfficiency: 0.9,
          rangesInCache: 0,
        },
      });
      expect(advisor.getRecommendations().find((r) => r.id === 'low-efficiency')).toBeUndefined();
    });
  });

  describe('error rate', () => {
    it('emits high-errors when > 5 % of recent events fail', () => {
      // 20 events: 2 errors (10 %).
      for (let i = 0; i < 18; i++) {
        advisor.analyzeEvent(makeEvent({ type: 'load', data: {} }));
      }
      advisor.analyzeEvent(makeEvent({ type: 'error', data: {} }));
      advisor.analyzeEvent(makeEvent({ type: 'error', data: {} }));

      const rec = advisor.getRecommendations().find((r) => r.id === 'high-errors');
      expect(rec).toBeDefined();
      expect(rec?.severity).toBe('error');
    });

    it('does not emit when no errors are present', () => {
      for (let i = 0; i < 20; i++) {
        advisor.analyzeEvent(makeEvent({ type: 'load', data: {} }));
      }
      expect(advisor.getRecommendations().find((r) => r.id === 'high-errors')).toBeUndefined();
    });
  });

  describe('memory pressure (eviction frequency)', () => {
    it('emits memory-pressure when > 5 evictions in 10s', () => {
      const now = Date.now();
      for (let i = 0; i < 6; i++) {
        advisor.analyzeEvent({
          type: 'evict',
          loader: 'point-spatial-index',
          timestamp: now - i * 100,
          data: {},
        });
      }
      const rec = advisor.getRecommendations().find((r) => r.id === 'memory-pressure');
      expect(rec).toBeDefined();
      expect(rec?.severity).toBe('warning');
    });

    it('does not trip on a single eviction', () => {
      advisor.analyzeEvent({
        type: 'evict',
        loader: 'point-spatial-index',
        timestamp: Date.now(),
        data: {},
      });
      expect(advisor.getRecommendations().find((r) => r.id === 'memory-pressure')).toBeUndefined();
    });
  });

  describe('analyzeMemoryMetrics', () => {
    function makeMemoryMetrics(overrides: Partial<MemoryMetrics> = {}): MemoryMetrics {
      const baseTypeStats = {
        allocations: 0,
        reuses: 0,
        evictions: 0,
        activeBuffers: 0,
        pooledBuffers: 0,
      };
      return {
        gpuPool: {
          allocations: 0,
          reuses: 0,
          evictions: 0,
          capacityGrowths: 0,
          activeBuffers: 0,
          pooledBuffers: 0,
          byType: {
            points: { ...baseTypeStats },
            lines: { ...baseTypeStats },
            gsplats: { ...baseTypeStats },
          },
        },
        accumulators: {
          points: null,
          lines: null,
          gsplats: null,
        },
        ...overrides,
      };
    }

    it('emits low-gpu-reuse-overall when overall reuse < 50 % and total > 10', () => {
      const metrics = makeMemoryMetrics();
      metrics.gpuPool!.allocations = 12;
      metrics.gpuPool!.reuses = 4; // 4/16 = 0.25 → warning (0.25 > 0.2 error boundary)
      advisor.analyzeMemoryMetrics(metrics);

      const rec = advisor.getRecommendations().find((r) => r.id === 'low-gpu-reuse-overall');
      expect(rec).toBeDefined();
      expect(rec?.severity).toBe('warning');
    });

    it('escalates low-gpu-reuse to error when rate < 20 %', () => {
      const metrics = makeMemoryMetrics();
      metrics.gpuPool!.allocations = 100;
      metrics.gpuPool!.reuses = 5; // 5/105 ≈ 4.7 %
      advisor.analyzeMemoryMetrics(metrics);

      const rec = advisor.getRecommendations().find((r) => r.id === 'low-gpu-reuse-overall');
      expect(rec?.severity).toBe('error');
    });

    it('emits per-type low-reuse warnings only when typeTotal > 5', () => {
      const metrics = makeMemoryMetrics();
      metrics.gpuPool!.allocations = 10;
      metrics.gpuPool!.reuses = 5; // 5/15 → 33 %, no overall warn
      metrics.gpuPool!.byType.points.allocations = 10;
      metrics.gpuPool!.byType.points.reuses = 1; // 1/11 → < 20 %
      metrics.gpuPool!.byType.lines.allocations = 2;
      metrics.gpuPool!.byType.lines.reuses = 0; // total = 2, below threshold
      advisor.analyzeMemoryMetrics(metrics);

      const recs = advisor.getRecommendations();
      expect(recs.find((r) => r.id === 'low-gpu-reuse-points')).toBeDefined();
      expect(recs.find((r) => r.id === 'low-gpu-reuse-lines')).toBeUndefined();
    });

    it('skips low-reuse warning when total ≤ 10 (or per-type ≤ 5)', () => {
      const metrics = makeMemoryMetrics();
      metrics.gpuPool!.allocations = 5;
      metrics.gpuPool!.reuses = 1;
      advisor.analyzeMemoryMetrics(metrics);
      expect(
        advisor.getRecommendations().find((r) => r.id === 'low-gpu-reuse-overall')
      ).toBeUndefined();
    });

    it('emits excessive-growth-points when > 5 growth events', () => {
      const metrics = makeMemoryMetrics({
        accumulators: {
          points: { capacity: 1024, allocations: 1, growthEvents: 7, memoryMB: 1 },
          lines: null,
          gsplats: null,
        },
      });
      advisor.analyzeMemoryMetrics(metrics);

      const rec = advisor.getRecommendations().find((r) => r.id === 'excessive-growth-points');
      expect(rec).toBeDefined();
      expect(rec?.severity).toBe('warning');
    });

    it('escalates excessive-growth to error when > 10 growth events', () => {
      const metrics = makeMemoryMetrics({
        accumulators: {
          points: null,
          lines: { capacity: 1024, allocations: 1, growthEvents: 15, memoryMB: 1 },
          gsplats: null,
        },
      });
      advisor.analyzeMemoryMetrics(metrics);

      const rec = advisor.getRecommendations().find((r) => r.id === 'excessive-growth-lines');
      expect(rec?.severity).toBe('error');
    });
  });

  describe('getRecommendations + lifecycle', () => {
    it('returns recommendations sorted by severity (error > warning > info)', () => {
      // Trigger one of each severity.
      advisor.analyzeEvent(makeEvent({ type: 'query', data: { latency: 250 } })); // slow-query (warning)
      advisor.analyzeMetrics({ ...baseMetrics, avgQueryTime: 250 }); // high-avg-query (error)
      advisor.analyzeMetrics({
        ...baseMetrics,
        avgQueryTime: 0,
        spatialIndex: {
          gridShape: [1],
          gridOrigin: [0],
          cellSize: [1],
          occupiedCells: 1,
          totalCells: 1,
          avgCellsPerQuery: 1,
          avgPointsPerCell: 1,
          queryEfficiency: 0.1,
          rangesInCache: 0,
        },
      }); // low-efficiency (info)

      const recs = advisor.getRecommendations();
      const order = recs.map((r) => r.severity);
      expect(order).toEqual(
        [...order].sort((a, b) => {
          const rank = { error: 0, warning: 1, info: 2 } as const;
          return rank[a] - rank[b];
        })
      );
    });

    it('hasWarnings returns true when any warning/error is present', () => {
      advisor.analyzeEvent(makeEvent({ type: 'query', data: { latency: 250 } }));
      expect(advisor.hasWarnings()).toBe(true);
    });

    it('hasWarnings returns false when only info-severity recs are present', () => {
      advisor.analyzeMetrics({
        ...baseMetrics,
        spatialIndex: {
          gridShape: [1],
          gridOrigin: [0],
          cellSize: [1],
          occupiedCells: 1,
          totalCells: 1,
          avgCellsPerQuery: 1,
          avgPointsPerCell: 1,
          queryEfficiency: 0.1,
          rangesInCache: 0,
        },
      });
      expect(advisor.hasWarnings()).toBe(false);
    });

    it('clear() drops all recommendations and history', () => {
      advisor.analyzeEvent(makeEvent({ type: 'query', data: { latency: 250 } }));
      expect(advisor.getRecommendations()).not.toEqual([]);
      advisor.clear();
      expect(advisor.getRecommendations()).toEqual([]);
    });

    it('dispose() clears state (idempotent)', () => {
      advisor.analyzeEvent(makeEvent({ type: 'query', data: { latency: 250 } }));
      advisor.dispose();
      expect(advisor.getRecommendations()).toEqual([]);
      // Second dispose call must not throw.
      advisor.dispose();
      expect(advisor.getRecommendations()).toEqual([]);
    });

    it('updateRecommendations does not throw + leaves non-global recs intact', () => {
      advisor.analyzeEvent(makeEvent({ type: 'query', data: { latency: 250 } }));
      advisor.updateRecommendations({});
      // slow-query is non-global → preserved.
      expect(advisor.getRecommendations().find((r) => r.id === 'slow-query')).toBeDefined();
    });
  });
});
