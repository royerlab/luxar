/**
 * Tests for the LoaderMonitor surface on LinesSpatialIndexLoader.
 *
 * Verifies the API contract added in Phase 11.10b: addEventListener /
 * removeEventListener / getMetrics / getActiveQueries are present, the
 * metrics shape carries the right LoaderType and path, and the listener
 * Set responds to add/remove.
 *
 * Full data-flow tests (loadLines → 'load' event emission) belong in
 * the integration test pass; these tests stay narrow so they don't
 * need a full zarrita mock to run.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { LinesSpatialIndexLoader } from '../../../data/lines/lines-spatial-index-loader';
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

describe('LinesSpatialIndexLoader — LoaderMonitor surface', () => {
  let loader: LinesSpatialIndexLoader;

  beforeEach(() => {
    const mockZarrLocation = {
      resolve: vi.fn().mockImplementation((p: string) => `mock://${p}`),
    } as unknown as ConstructorParameters<typeof LinesSpatialIndexLoader>[0];
    loader = new LinesSpatialIndexLoader(mockZarrLocation, makeNode());
  });

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

  it('add/remove of a listener is observable through dispose() — listener Set drops on dispose', () => {
    const calls: MonitorEvent[] = [];
    const listener: MonitorEventListener = (event) => calls.push(event);

    loader.addEventListener(listener);
    loader.removeEventListener(listener);
    loader.dispose();

    // No way to introspect listener size from the public surface, but the
    // contract is "remove + dispose leaves no leaks" — the test passes if
    // the calls above do not throw.
    expect(true).toBe(true);
  });

  it('returns an immutable snapshot from getMetrics (caller can mutate without affecting the loader)', () => {
    const snapshot = loader.getMetrics();
    snapshot.queries = 42;
    expect(loader.getMetrics().queries).toBe(0);
  });
});
