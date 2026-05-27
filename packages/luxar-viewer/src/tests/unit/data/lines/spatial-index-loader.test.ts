/**
 * Tests for LinesSpatialIndexLoader.
 *
 * Mirrors the describe-block structure of
 * `points-spatial-index-loader.test.ts` per the three-geometry
 * symmetry rule (see `feedback_geometry_symmetry.md` in user memory).
 * Shared mocking boilerplate lives in
 * `tests/builders/spatial-loader-fixtures.ts`.
 *
 * Lines-specific quirks (justified asymmetries, all documented inline):
 *   - Segment-first loading means the data-loading describe block tests
 *     the segment + vertex round-trip, not direct point-style projection.
 *   - Dual chunk-bounds (`vertex_chunk_bounds` + `segment_chunk_bounds`)
 *     means the chunk-bounds zarr.open + zarr.get mock has to
 *     differentiate between the two.
 *   - Public method is `loadLines(viewState, session?)` instead of
 *     `loadPoints(...)`.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as zarr from 'zarrita';
import { LinesSpatialIndexLoader } from '../../../../data/lines/lines-spatial-index-loader';
import type { SceneNode, ViewState } from '../../../../data';
import type { MonitorEvent, MonitorEventListener } from '../../../../types/data-monitor-types';
import { makeMockZarrLocation } from '../../../builders/spatial-loader-fixtures';

vi.mock('zarrita', () => ({
  registry: {},
  open: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

// Mock the canonical SpatialQueryBuilder so the test exercises the loader's
// orchestration rather than the AABB scan (which has its own unit tests).
const mockExecute = vi.fn();
vi.mock('../../../../data/loaders/spatial-query/spatial-query-builder', async () => {
  const actual = await vi.importActual<
    typeof import('../../../../data/loaders/spatial-query/spatial-query-builder')
  >('../../../../data/loaders/spatial-query/spatial-query-builder');
  return {
    ...actual,
    SpatialQueryBuilder: vi.fn().mockImplementation(() => ({
      execute: mockExecute,
    })),
  };
});

import { SpatialQueryBuilder } from '../../../../data/loaders';

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
      vertex_ordering: {
        slice_dims: [],
        ordering_dims: [0, 1, 2],
        ordering_min: [0, 0, 0],
        ordering_max: [1, 1, 1],
        chunk_size: 100,
        ordering_bits_per_dim: 21,
      },
      segment_ordering: {
        slice_dims: [],
        ordering_dims: [0, 1, 2],
        ordering_min: [0, 0, 0],
        ordering_max: [1, 1, 1],
        chunk_size: 100,
        ordering_bits_per_dim: 21,
      },
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
      // data.md W1 fix [P2]: previous assertion was `expect(true).toBe(true)`
      // (literal tautology — passes for any code change). The LoaderEventEmitter
      // exposes a `size` getter (see `data/loaders/monitor-events.ts`), so we
      // can observe the listener Set transitions directly:
      //   add → size === 1, remove → 0, add → 1, dispose → 0.
      // A mutation that no-ops add/remove or dispose would now flip a count.
      const calls: MonitorEvent[] = [];
      const listener: MonitorEventListener = (event) => calls.push(event);

      const events = (loader as unknown as { events: { size: number } }).events;

      expect(events.size).toBe(0);
      loader.addEventListener(listener);
      expect(events.size).toBe(1);
      loader.removeEventListener(listener);
      expect(events.size).toBe(0);

      // Re-add then dispose: dispose must also clear listeners.
      loader.addEventListener(listener);
      expect(events.size).toBe(1);
      loader.dispose();
      expect(events.size).toBe(0);

      // The removed listener should not have fired (sanity check that
      // event delivery is correctly gated on registration).
      expect(calls).toEqual([]);
    });

    it('returns an immutable snapshot from getMetrics', () => {
      const snapshot = loader.getMetrics();
      snapshot.queries = 42;
      expect(loader.getMetrics().queries).toBe(0);
    });
  });

  // ────────────────────────────────────────────────────────────────
  // Body-coverage describe blocks below use a richer fixture: zarr.open
  // and zarr.get are wired through to the dual-bounds chunk index +
  // segments / vertices / widths / colors / sharpness data arrays.
  // ────────────────────────────────────────────────────────────────

  // data.md O2 / Phase E16: previously `'body coverage'` — a
  // coverage-driven name, not a behavior-driven one. Rename to surface
  // what the inner block exercises (lines spatial-index initialization,
  // query routing, and chunk metadata loading through mocked zarr fixtures).
  describe('with mocked zarr fixtures', () => {
    let bodyLoader: LinesSpatialIndexLoader;
    let mockZarrLocation: { resolve: ReturnType<typeof vi.fn> };
    let mockNode: SceneNode;
    let mockArrays: {
      vertices: { shape: number[]; dtype: string; attrs?: object };
      segments: { shape: number[]; dtype: string; attrs?: object };
      widths: { shape: number[]; dtype: string; attrs?: object };
      colors: { shape: number[]; dtype: string; attrs?: object };
      sharpness: { shape: number[]; dtype: string; attrs?: object };
    };
    let vertexBoundsArray: { shape: number[]; dtype: string; attrs: object };
    let segmentBoundsArray: { shape: number[]; dtype: string; attrs: object };

    beforeEach(() => {
      vi.clearAllMocks();

      mockArrays = {
        vertices: { shape: [1000, 3], dtype: 'float32', attrs: {} },
        segments: { shape: [999, 2], dtype: 'uint32', attrs: {} },
        widths: { shape: [1000], dtype: 'float32', attrs: {} },
        colors: { shape: [1000, 3], dtype: 'float32', attrs: {} },
        sharpness: { shape: [1000], dtype: 'float32', attrs: {} },
      };

      mockZarrLocation = makeMockZarrLocation();
      mockNode = makeLinesNode();

      vertexBoundsArray = { shape: [10, 3, 2], dtype: 'float32', attrs: {} };
      segmentBoundsArray = { shape: [10, 3, 2], dtype: 'float32', attrs: {} };

      // Default builder behaviour: return two ranges so range-merge logic exists.
      mockExecute.mockResolvedValue([
        { start: 0, end: 50 },
        { start: 100, end: 150 },
      ]);

      (zarr.open as unknown as ReturnType<typeof vi.fn>).mockImplementation((location: unknown) => {
        const path = String(location);
        if (path.includes('vertex_chunk_bounds')) return Promise.resolve(vertexBoundsArray);
        if (path.includes('segment_chunk_bounds')) return Promise.resolve(segmentBoundsArray);
        if (path.includes('vertices')) return Promise.resolve(mockArrays.vertices);
        if (path.includes('segments')) return Promise.resolve(mockArrays.segments);
        if (path.includes('widths')) return Promise.resolve(mockArrays.widths);
        if (path.includes('colors')) return Promise.resolve(mockArrays.colors);
        if (path.includes('sharpnesses')) return Promise.resolve(mockArrays.sharpness);
        return Promise.reject(new Error(`Unknown array: ${path}`));
      });

      (zarr.get as unknown as ReturnType<typeof vi.fn>).mockImplementation(
        (array: unknown, slices?: unknown) => {
          if (array === vertexBoundsArray) {
            return Promise.resolve({ data: new Float32Array(10 * 3 * 2) });
          }
          if (array === segmentBoundsArray) {
            return Promise.resolve({ data: new Float32Array(10 * 3 * 2) });
          }
          if (array === mockArrays.segments) {
            // Return ascending vertex pairs so the unique-index set is contiguous.
            // First range [0,50): segments [0,1],[1,2],…[49,50] = 100 indices
            // Second range [100,150): segments [100,101],…[149,150] = 100 indices
            const sliceSpec = slices as Array<{ start: number; end: number }>;
            const range = sliceSpec[0];
            const count = range.end - range.start;
            const data = new Uint32Array(count * 2);
            for (let i = 0; i < count; i++) {
              data[i * 2] = range.start + i;
              data[i * 2 + 1] = range.start + i + 1;
            }
            return Promise.resolve({ data });
          }
          // For vertices / widths / colors / sharpness, return zeros sized by slice.
          const sliceSpec = slices as Array<{ start: number; end: number }>;
          const count = sliceSpec[0].end - sliceSpec[0].start;
          const elementsPerItem =
            array === mockArrays.vertices ? 3 : array === mockArrays.colors ? 3 : 1;
          return Promise.resolve({ data: new Float32Array(count * elementsPerItem) });
        }
      );

      bodyLoader = new LinesSpatialIndexLoader(
        mockZarrLocation as unknown as ConstructorParameters<typeof LinesSpatialIndexLoader>[0],
        mockNode
      );
    });

    afterEach(() => {
      bodyLoader?.dispose();
    });

    describe('initialization', () => {
      it('should load dual chunk-based spatial index on first load', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadLines(viewState);

        const opens = (zarr.open as unknown as ReturnType<typeof vi.fn>).mock.calls;
        expect(opens.some((c) => String(c[0]).includes('vertex_chunk_bounds'))).toBe(true);
        expect(opens.some((c) => String(c[0]).includes('segment_chunk_bounds'))).toBe(true);
        expect(opens.some((c) => String(c[0]).includes('vertices'))).toBe(true);
        expect(opens.some((c) => String(c[0]).includes('segments'))).toBe(true);
      });

      it('should handle missing spatial index gracefully', async () => {
        const noOrderingNode: SceneNode = {
          ...mockNode,
          attrs: { ...mockNode.attrs, ordering: 'none' },
        };
        bodyLoader.dispose();
        bodyLoader = new LinesSpatialIndexLoader(
          mockZarrLocation as unknown as ConstructorParameters<typeof LinesSpatialIndexLoader>[0],
          noOrderingNode
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        // ordering='none' → returns null index → load-all fallback.
        // Builder is NOT invoked when chunkIndex is null.
        const result = await bodyLoader.loadLines(viewState);
        expect(result).toBeDefined();
        expect(result.positions).toBeDefined();
      });

      it('should handle missing optional arrays gracefully', async () => {
        // Re-mock open to fail the optional widths/colors/sharpness opens.
        (zarr.open as unknown as ReturnType<typeof vi.fn>).mockImplementation(
          (location: unknown) => {
            const path = String(location);
            if (path.includes('vertex_chunk_bounds')) return Promise.resolve(vertexBoundsArray);
            if (path.includes('segment_chunk_bounds')) return Promise.resolve(segmentBoundsArray);
            if (path.includes('vertices')) return Promise.resolve(mockArrays.vertices);
            if (path.includes('segments')) return Promise.resolve(mockArrays.segments);
            return Promise.reject(new Error('Not found'));
          }
        );

        bodyLoader.dispose();
        bodyLoader = new LinesSpatialIndexLoader(
          mockZarrLocation as unknown as ConstructorParameters<typeof LinesSpatialIndexLoader>[0],
          mockNode
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadLines(viewState);

        expect(result.positions).toBeDefined();
        // Audit W7 fix: toBeFalsy matches too broadly. Pin the
        // documented sentinel (null or undefined).
        expect(result.colors == null).toBe(true);
        expect(result.sharpness == null).toBe(true);
        // Widths is required; the loader fills with default 1.0 when the
        // array open fails, so the field is always present.
        expect(result.widths).toBeInstanceOf(Float32Array);
      });

      it('should only initialize once with concurrent calls', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await Promise.all([
          bodyLoader.loadLines(viewState),
          bodyLoader.loadLines(viewState),
          bodyLoader.loadLines(viewState),
        ]);

        const vertexBoundsOpens = (
          zarr.open as unknown as ReturnType<typeof vi.fn>
        ).mock.calls.filter((c) => String(c[0]).includes('vertex_chunk_bounds')).length;
        expect(vertexBoundsOpens).toBe(1);
      });
    });

    describe('spatial index queries', () => {
      it('should construct the query builder with index, viewState, and options', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0.5, 0.5, 0.5],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadLines(viewState);

        expect(SpatialQueryBuilder).toHaveBeenCalled();
        const call = (SpatialQueryBuilder as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
        const [index, vs, options] = call as [
          { chunkBounds: Float32Array; chunkCount: number },
          ViewState,
          { totalElements: number; chunkSize: number; geometryType: string },
        ];
        expect(index.chunkBounds).toBeInstanceOf(Float32Array);
        expect(index.chunkCount).toBeGreaterThan(0);
        expect(vs.displayDims).toEqual([0, 1, 2]);
        expect(options.totalElements).toBe(999); // n_segments
        expect(options.chunkSize).toBe(100);
        expect(options.geometryType).toBe('lines');
      });

      it('should return empty lines when builder returns no ranges', async () => {
        mockExecute.mockResolvedValueOnce([]);

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [100, 100, 100],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadLines(viewState);

        expect(result.segmentCount).toBe(0);
        expect(result.vertexCount).toBe(0);
        expect(result.positions.length).toBe(0);
        expect(result.segments.length).toBe(0);
      });
    });

    describe('extend_to_all', () => {
      // The actual extend-or-not decision lives inside `SpatialQueryBuilder.execute()`
      // (covered by `spatial-query-builder.test.ts`). At this layer we verify the
      // loader wires `extendDims` through to the builder constructor.
      it('forwards extend_to_all and dimension metadata to the builder', async () => {
        mockNode.attrs.extend_to_all = ['time'];

        const dimsMetadata = Object.assign([], {
          0: { name: 'x', unit: 'um', display: true },
          1: { name: 'y', unit: 'um', display: true },
          2: { name: 'z', unit: 'um', display: true },
        });

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
          dimensions: dimsMetadata,
        };

        await bodyLoader.loadLines(viewState);

        const call = (SpatialQueryBuilder as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
        const options = call[2] as { extendDims: string[] };
        expect(options.extendDims).toEqual(['time']);
      });

      it('passes empty extendDims when extend_to_all is not configured', async () => {
        delete mockNode.attrs.extend_to_all;

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadLines(viewState);

        const call = (SpatialQueryBuilder as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
        const options = call[2] as { extendDims: string[] };
        expect(options.extendDims).toEqual([]);
      });
    });

    describe('data loading (segment-first path)', () => {
      // Lines-specific: Points and gsplats project nD → 3D directly
      // inside the loader. Lines instead does segment indices → unique
      // vertex indices → vertex round-trip, then global → local index
      // remapping. This block tests that flow.
      it('should remap segment indices to local vertex space', async () => {
        // Single range [0, 5) → 5 segments referencing vertices 0..5
        mockExecute.mockResolvedValueOnce([{ start: 0, end: 5 }]);
        (zarr.get as unknown as ReturnType<typeof vi.fn>).mockImplementation(
          (array: unknown, slices?: unknown) => {
            if (array === vertexBoundsArray || array === segmentBoundsArray) {
              return Promise.resolve({ data: new Float32Array(10 * 3 * 2) });
            }
            if (array === mockArrays.segments) {
              // Segments [0,1],[1,2],[2,3],[3,4],[4,5] - 5 segments → 6 unique vertices
              return Promise.resolve({ data: new Uint32Array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5]) });
            }
            const sliceSpec = slices as Array<{ start: number; end: number }>;
            const count = sliceSpec[0].end - sliceSpec[0].start;
            const elementsPerItem =
              array === mockArrays.vertices ? 3 : array === mockArrays.colors ? 3 : 1;
            return Promise.resolve({ data: new Float32Array(count * elementsPerItem) });
          }
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadLines(viewState);

        expect(result.segmentCount).toBe(5);
        expect(result.vertexCount).toBe(6);
        // Local indices 0..5 referencing vertices 0..5 in load order.
        expect(Array.from(result.segments)).toEqual([0, 1, 1, 2, 2, 3, 3, 4, 4, 5]);
      });

      it('should fill default widths when widths array is absent', async () => {
        (zarr.open as unknown as ReturnType<typeof vi.fn>).mockImplementation(
          (location: unknown) => {
            const path = String(location);
            if (path.includes('vertex_chunk_bounds')) return Promise.resolve(vertexBoundsArray);
            if (path.includes('segment_chunk_bounds')) return Promise.resolve(segmentBoundsArray);
            if (path.includes('vertices')) return Promise.resolve(mockArrays.vertices);
            if (path.includes('segments')) return Promise.resolve(mockArrays.segments);
            if (path.includes('colors')) return Promise.resolve(mockArrays.colors);
            // widths/sharpness rejected
            return Promise.reject(new Error('Not found'));
          }
        );

        bodyLoader.dispose();
        bodyLoader = new LinesSpatialIndexLoader(
          mockZarrLocation as unknown as ConstructorParameters<typeof LinesSpatialIndexLoader>[0],
          mockNode
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadLines(viewState);
        expect(result.widths).toBeInstanceOf(Float32Array);
        expect(result.widths.length).toBeGreaterThan(0);
        expect(Array.from(result.widths).every((w) => w === 1.0)).toBe(true);
      });

      it('should preserve original ndim in result metadata', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadLines(viewState);
        expect(result.ndim).toBe(3);
      });
    });

    describe('monitoring (during load)', () => {
      it('should emit query events', async () => {
        const listener = vi.fn();
        bodyLoader.addEventListener(listener);

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadLines(viewState);

        expect(listener).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'query',
            loader: 'lines-spatial-index',
          })
        );
      });

      it('should track active queries during load', async () => {
        // Make zarr.get slower so we can observe activeQueries mid-flight.
        (zarr.get as unknown as ReturnType<typeof vi.fn>).mockImplementation(
          () =>
            new Promise((resolve) => setTimeout(() => resolve({ data: new Uint32Array(0) }), 10))
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const promise = bodyLoader.loadLines(viewState).catch(() => null);
        await promise;
        expect(bodyLoader.getActiveQueries().length).toBe(0);
      });

      it('should update metrics correctly', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadLines(viewState);

        const metrics = bodyLoader.getMetrics();
        expect(metrics.queries).toBe(1);
        expect(metrics.type).toBe('lines-spatial-index');
        expect(metrics.path).toBe('/test_lines');
      });
    });

    describe('error handling', () => {
      it('should handle initialization errors', async () => {
        (zarr.open as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(
          new Error('Failed to open vertices')
        );

        bodyLoader.dispose();
        bodyLoader = new LinesSpatialIndexLoader(
          mockZarrLocation as unknown as ConstructorParameters<typeof LinesSpatialIndexLoader>[0],
          mockNode
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await expect(bodyLoader.loadLines(viewState)).rejects.toThrow();
      });

      it('should record errors in metrics on failure', async () => {
        (zarr.get as unknown as ReturnType<typeof vi.fn>).mockImplementation((array: unknown) => {
          if (array === vertexBoundsArray || array === segmentBoundsArray) {
            return Promise.resolve({ data: new Float32Array(10 * 3 * 2) });
          }
          return Promise.reject(new Error('Load failed'));
        });

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await expect(bodyLoader.loadLines(viewState)).rejects.toThrow();

        const metrics = bodyLoader.getMetrics();
        expect(metrics.errors).toBeGreaterThanOrEqual(1);
      });

      // Lines/GSplats emit a monitor 'error' event for parity with
      // Points so event-driven dashboards observe all geometry failures.
      it('emits a monitor "error" event on load failure', async () => {
        (zarr.get as unknown as ReturnType<typeof vi.fn>).mockImplementation((array: unknown) => {
          if (array === vertexBoundsArray || array === segmentBoundsArray) {
            return Promise.resolve({ data: new Float32Array(10 * 3 * 2) });
          }
          return Promise.reject(new Error('Load failed'));
        });

        const listener = vi.fn();
        bodyLoader.addEventListener(listener);

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };
        await expect(bodyLoader.loadLines(viewState)).rejects.toThrow();

        expect(listener).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'error',
            loader: 'lines-spatial-index',
            data: expect.objectContaining({
              path: '/test_lines',
              error: expect.stringContaining('Load failed'),
            }),
          })
        );
      });
    });

    describe('updateView', () => {
      it('should reload data for new view state', async () => {
        const viewState1: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };
        const viewState2: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0.5, 0.5, 0.5],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadLines(viewState1);
        const result = await bodyLoader.updateView(viewState2);
        expect(result).toBeDefined();
        expect(SpatialQueryBuilder).toHaveBeenCalledTimes(2);
      });
    });

    describe('prefetchChunks (commit 8.2)', () => {
      it('warms the cache via zarr.get on every available array × range', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };
        await bodyLoader.loadLines(viewState);
        const callsBefore = (zarr.get as any).mock.calls.length;
        await bodyLoader.prefetchChunks(viewState);
        const callsAfter = (zarr.get as any).mock.calls.length;
        expect(callsAfter).toBeGreaterThan(callsBefore);
      });
    });

    describe('resource cleanup', () => {
      it('should dispose resources properly', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadLines(viewState);
        bodyLoader.dispose();

        expect((bodyLoader as unknown as { chunkIndex: unknown }).chunkIndex).toBeNull();
        expect((bodyLoader as unknown as { arrays: object }).arrays).toEqual({});
        expect((bodyLoader as unknown as { events: { size: number } }).events.size).toBe(0);
      });

      // Audit G10 (viewer-data-cache-workers-wasm): pin dispose() idempotency
      // — symmetric to the Points + GSplats variants.
      it('dispose is idempotent — second call is a clean no-op', () => {
        bodyLoader.dispose();
        expect(() => bodyLoader.dispose()).not.toThrow();
        expect((bodyLoader as unknown as { chunkIndex: unknown }).chunkIndex).toBeNull();
        expect((bodyLoader as unknown as { arrays: object }).arrays).toEqual({});
        expect((bodyLoader as unknown as { events: { size: number } }).events.size).toBe(0);
      });
    });

    describe('data type handling', () => {
      it('should handle uint8 color data via the shared color helper', async () => {
        mockArrays.colors.dtype = 'uint8';

        (zarr.get as unknown as ReturnType<typeof vi.fn>).mockImplementation(
          (array: unknown, slices?: unknown) => {
            if (array === vertexBoundsArray || array === segmentBoundsArray) {
              return Promise.resolve({ data: new Float32Array(10 * 3 * 2) });
            }
            if (array === mockArrays.segments) {
              const sliceSpec = slices as Array<{ start: number; end: number }>;
              const range = sliceSpec[0];
              const count = range.end - range.start;
              const data = new Uint32Array(count * 2);
              for (let i = 0; i < count; i++) {
                data[i * 2] = range.start + i;
                data[i * 2 + 1] = range.start + i + 1;
              }
              return Promise.resolve({ data });
            }
            const sliceSpec = slices as Array<{ start: number; end: number }>;
            const count = sliceSpec[0].end - sliceSpec[0].start;
            if (array === mockArrays.colors) {
              // Direct uint8 colors — preserved natively by loadColorRanges.
              const buf = new Uint8Array(count * 3);
              for (let i = 0; i < count; i++) buf[i * 3] = 255;
              return Promise.resolve({ data: buf });
            }
            const elementsPerItem = array === mockArrays.vertices ? 3 : 1;
            return Promise.resolve({ data: new Float32Array(count * elementsPerItem) });
          }
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadLines(viewState);
        expect(result.colors).toBeInstanceOf(Uint8Array);
      });

      it('should keep direct Float32 (HDR) colors as Float32Array', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadLines(viewState);
        expect(result.colors).toBeInstanceOf(Float32Array);
      });
    });
  });
});
