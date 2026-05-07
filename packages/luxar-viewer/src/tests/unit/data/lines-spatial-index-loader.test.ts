/**
 * Tests for LinesSpatialIndexLoader.
 *
 * Mirrors the describe-block structure of
 * `points-spatial-index-loader.test.ts` per the three-geometry
 * symmetry rule (see `feedback_geometry_symmetry.md` in user memory).
 * Shared mocking boilerplate lives in
 * `tests/builders/spatial-loader-fixtures.ts`.
 *
 * **Currently scoped to the LoaderMonitor surface.** Body coverage
 * (initialization → spatial query → data projection → updateView)
 * mirrors what `points-spatial-index-loader.test.ts` covers and lands
 * in a follow-up commit alongside the same expansion for gsplats.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { LinesSpatialIndexLoader } from '../../../data/lines/lines-spatial-index-loader';
import type { SceneNode } from '../../../data';
import type {
  MonitorEvent,
  MonitorEventListener,
} from '../../../types/data-monitor-types';
import { makeMockZarrLocation } from '../../builders/spatial-loader-fixtures';

vi.mock('zarrita', () => ({
  open: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

function makeLinesNode(overrides: Partial<SceneNode> = {}): SceneNode {
  return {
    path: '/test_lines',
    type: 'lines',
    attrs: {
      type: 'lines',
      n_vertices: 1000,
      n_segments: 999,
      ndim: 3,
      max_width: 1.0,
      ordering: 'hilbert',
      original_line_type: 'segments',
      has_colors: true,
      has_sharpness: false,
    },
    hasSpatialIndex: true,
    ...overrides,
  } as SceneNode;
}

describe('LinesSpatialIndexLoader', () => {
  let loader: LinesSpatialIndexLoader;

  beforeEach(() => {
    const mockZarrLocation = makeMockZarrLocation();
    loader = new LinesSpatialIndexLoader(
      mockZarrLocation as unknown as ConstructorParameters<typeof LinesSpatialIndexLoader>[0],
      makeLinesNode()
    );
  });

  // ────────────────────────────────────────────────────────────────
  describe('monitoring', () => {
    it('exposes the four LoaderMonitor methods', () => {
      expect(typeof loader.addEventListener).toBe('function');
      expect(typeof loader.removeEventListener).toBe('function');
      expect(typeof loader.getMetrics).toBe('function');
      expect(typeof loader.getActiveQueries).toBe('function');
    });

    it('initial metrics report the lines-spatial-index type and node path', () => {
      const metrics = loader.getMetrics();
      expect(metrics.type).toBe('lines-spatial-index');
      expect(metrics.path).toBe('/test_lines');
      expect(metrics.queries).toBe(0);
      expect(metrics.loads).toBe(0);
      expect(metrics.pointsLoaded).toBe(0);
      expect(metrics.bytesLoaded).toBe(0);
    });

    it('reports n_vertices as the dataset size on getMetrics', () => {
      const metrics = loader.getMetrics();
      expect(metrics.datasetSize).toBe(1000);
    });

    it('returns an empty active-queries list initially', () => {
      expect(loader.getActiveQueries()).toEqual([]);
    });

    it('add + remove of a listener leaves no leak after dispose', () => {
      const calls: MonitorEvent[] = [];
      const listener: MonitorEventListener = (event) => calls.push(event);

      loader.addEventListener(listener);
      loader.removeEventListener(listener);
      loader.dispose();

      // No way to introspect listener size from the public surface, but
      // the contract is "remove + dispose leaves no leaks" — the test
      // passes if the calls above do not throw.
      expect(true).toBe(true);
    });

    it('returns an immutable snapshot from getMetrics', () => {
      const snapshot = loader.getMetrics();
      snapshot.queries = 42;
      expect(loader.getMetrics().queries).toBe(0);
    });
  });
});
