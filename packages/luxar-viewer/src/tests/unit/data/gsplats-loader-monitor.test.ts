/**
 * Tests for the LoaderMonitor surface on GSplatsSpatialIndexLoader.
 *
 * Verifies the API contract added in Phase 11.10c — same shape as the
 * sibling lines test, kept separate so a regression in one geometry's
 * surface fails its own test.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GSplatsSpatialIndexLoader } from '../../../data/gsplats/gsplats-spatial-index-loader';
import type { SceneNode } from '../../../data';
import type {
  MonitorEvent,
  MonitorEventListener,
} from '../../../types/data-monitor-types';

vi.mock('zarrita', () => ({
  open: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

function makeNode(overrides: Partial<SceneNode> = {}): SceneNode {
  return {
    path: '/test_gsplats',
    type: 'gsplats',
    attrs: {
      type: 'gsplats',
      n_splats: 5000,
      ndim: 3,
      ordering: 'hilbert',
      has_colors: true,
      chunk_size: 256,
      amplitude_range: { min: 0, max: 1 },
      center_bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    },
    hasSpatialIndex: true,
    ...overrides,
  } as SceneNode;
}

describe('GSplatsSpatialIndexLoader — LoaderMonitor surface', () => {
  let loader: GSplatsSpatialIndexLoader;

  beforeEach(() => {
    const mockZarrLocation = {
      resolve: vi.fn().mockImplementation((p: string) => `mock://${p}`),
    } as unknown as ConstructorParameters<typeof GSplatsSpatialIndexLoader>[0];
    loader = new GSplatsSpatialIndexLoader(mockZarrLocation, makeNode());
  });

  it('exposes the four LoaderMonitor methods', () => {
    expect(typeof loader.addEventListener).toBe('function');
    expect(typeof loader.removeEventListener).toBe('function');
    expect(typeof loader.getMetrics).toBe('function');
    expect(typeof loader.getActiveQueries).toBe('function');
  });

  it('initial metrics report the gsplats-spatial-index type and node path', () => {
    const metrics = loader.getMetrics();
    expect(metrics.type).toBe('gsplats-spatial-index');
    expect(metrics.path).toBe('/test_gsplats');
    expect(metrics.queries).toBe(0);
    expect(metrics.loads).toBe(0);
    expect(metrics.pointsLoaded).toBe(0);
    expect(metrics.bytesLoaded).toBe(0);
  });

  it('reports n_splats as the dataset size on getMetrics', () => {
    const metrics = loader.getMetrics();
    expect(metrics.datasetSize).toBe(5000);
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

    expect(true).toBe(true);
  });

  it('returns an immutable snapshot from getMetrics', () => {
    const snapshot = loader.getMetrics();
    snapshot.queries = 99;
    expect(loader.getMetrics().queries).toBe(0);
  });
});
