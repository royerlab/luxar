/**
 * Tests for GSplatsSpatialIndexLoader.
 *
 * Mirrors the describe-block structure of
 * `points-spatial-index-loader.test.ts` per the three-geometry
 * symmetry rule (see `feedback_geometry_symmetry.md` in user memory).
 * Shared mocking boilerplate lives in
 * `tests/builders/spatial-loader-fixtures.ts`.
 *
 * GSplats-specific quirks (justified asymmetries, all documented inline):
 *   - Single chunk-bounds (`chunk_bounds`, no segment/vertex split).
 *   - Direct per-splat loading (no segments → vertices remap step).
 *   - Public method is `loadGSplats(viewState, session?)` instead of
 *     `loadPoints(...)`.
 *   - Per-splat data: centers (ndim), amplitudes (1), cholesky_factors
 *     (k = ndim·(ndim+1)/2 packed lower-triangular), colors (optional 3).
 *   - `prefetchChunks(viewState)` — gsplats-only LOD-warming API; tested
 *     in its own block since points and lines have no equivalent.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as zarr from 'zarrita';
import { GSplatsSpatialIndexLoader } from '../../../../data/gsplats/gsplats-spatial-index-loader';
import type { SceneNode, ViewState } from '../../../../data';
import type { MonitorEvent, MonitorEventListener } from '../../../../types/data-monitor-types';
import { makeMockZarrLocation } from '../../../builders/spatial-loader-fixtures';

vi.mock('zarrita', () => ({
  registry: {},
  open: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

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

function makeGSplatsNode(overrides: Partial<SceneNode> = {}): SceneNode {
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

describe('GSplatsSpatialIndexLoader', () => {
  let loader: GSplatsSpatialIndexLoader;

  beforeEach(() => {
    const mockZarrLocation = makeMockZarrLocation();
    loader = new GSplatsSpatialIndexLoader(
      mockZarrLocation as unknown as ConstructorParameters<typeof GSplatsSpatialIndexLoader>[0],
      makeGSplatsNode()
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
      // data.md W2 fix [P2]: parallel to W1 fix in lines-spatial-index-loader.
      // Replaces `expect(true).toBe(true)` with observable Set-size transitions
      // through the private `events: LoaderEventEmitter` whose `size` getter
      // is part of the emitter's documented test-only surface
      // (data/loaders/monitor-events.ts).
      const calls: MonitorEvent[] = [];
      const listener: MonitorEventListener = (event) => calls.push(event);

      const events = (loader as unknown as { events: { size: number } }).events;

      expect(events.size).toBe(0);
      loader.addEventListener(listener);
      expect(events.size).toBe(1);
      loader.removeEventListener(listener);
      expect(events.size).toBe(0);

      loader.addEventListener(listener);
      expect(events.size).toBe(1);
      loader.dispose();
      expect(events.size).toBe(0);

      expect(calls).toEqual([]);
    });

    it('returns an immutable snapshot from getMetrics', () => {
      const snapshot = loader.getMetrics();
      snapshot.queries = 99;
      expect(loader.getMetrics().queries).toBe(0);
    });
  });

  // ────────────────────────────────────────────────────────────────
  // Body-coverage describe blocks below use a richer fixture: zarr.open
  // and zarr.get are wired through to the chunk index + centers /
  // amplitudes / cholesky_factors / colors data arrays.
  // ────────────────────────────────────────────────────────────────

  // data.md O2 / Phase E16: previously `'body coverage'` — a
  // coverage-driven name, not a behavior-driven one. Rename to surface
  // what the inner block exercises (gsplats spatial-index initialization,
  // query routing, and chunk metadata loading through mocked zarr fixtures).
  describe('with mocked zarr fixtures', () => {
    let bodyLoader: GSplatsSpatialIndexLoader;
    let mockZarrLocation: { resolve: ReturnType<typeof vi.fn> };
    let mockNode: SceneNode;
    let mockArrays: {
      centers: { shape: number[]; dtype: string; attrs?: object };
      amplitudes: { shape: number[]; dtype: string; attrs?: object };
      cholesky_factors: { shape: number[]; dtype: string; attrs?: object };
      colors: { shape: number[]; dtype: string; attrs?: object };
    };
    let chunkBoundsArray: { shape: number[]; dtype: string; attrs: object };

    beforeEach(() => {
      vi.clearAllMocks();

      // 3D dataset → cholesky packed = 3·4/2 = 6
      mockArrays = {
        centers: { shape: [5000, 3], dtype: 'float32', attrs: {} },
        amplitudes: { shape: [5000], dtype: 'float32', attrs: {} },
        cholesky_factors: { shape: [5000, 6], dtype: 'float32', attrs: {} },
        colors: { shape: [5000, 3], dtype: 'float32', attrs: {} },
      };

      mockZarrLocation = makeMockZarrLocation();
      mockNode = makeGSplatsNode();

      // 5000 splats / 256 chunk_size = 20 chunks → bounds shape [20, 3, 2]
      chunkBoundsArray = { shape: [20, 3, 2], dtype: 'float32', attrs: {} };

      mockExecute.mockResolvedValue([
        { start: 0, end: 100 },
        { start: 200, end: 300 },
      ]);

      (zarr.open as unknown as ReturnType<typeof vi.fn>).mockImplementation((location: unknown) => {
        const path = String(location);
        if (path.includes('chunk_bounds')) return Promise.resolve(chunkBoundsArray);
        if (path.includes('centers')) return Promise.resolve(mockArrays.centers);
        if (path.includes('amplitudes')) return Promise.resolve(mockArrays.amplitudes);
        if (path.includes('cholesky_factors')) return Promise.resolve(mockArrays.cholesky_factors);
        if (path.includes('colors')) return Promise.resolve(mockArrays.colors);
        return Promise.reject(new Error(`Unknown array: ${path}`));
      });

      (zarr.get as unknown as ReturnType<typeof vi.fn>).mockImplementation(
        (array: unknown, slices?: unknown) => {
          if (array === chunkBoundsArray) {
            return Promise.resolve({ data: new Float32Array(20 * 3 * 2) });
          }
          const sliceSpec = slices as Array<{ start: number; end: number }>;
          const count = sliceSpec[0].end - sliceSpec[0].start;
          let elementsPerItem = 1;
          if (array === mockArrays.centers) elementsPerItem = 3;
          else if (array === mockArrays.cholesky_factors) elementsPerItem = 6;
          else if (array === mockArrays.colors) elementsPerItem = 3;
          return Promise.resolve({ data: new Float32Array(count * elementsPerItem) });
        }
      );

      bodyLoader = new GSplatsSpatialIndexLoader(
        mockZarrLocation as unknown as ConstructorParameters<typeof GSplatsSpatialIndexLoader>[0],
        mockNode
      );
    });

    afterEach(() => {
      bodyLoader?.dispose();
    });

    describe('initialization', () => {
      it('should load chunk-based spatial index on first load', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadGSplats(viewState);

        const opens = (zarr.open as unknown as ReturnType<typeof vi.fn>).mock.calls;
        expect(opens.some((c) => String(c[0]).includes('chunk_bounds'))).toBe(true);
        expect(opens.some((c) => String(c[0]).includes('centers'))).toBe(true);
        expect(opens.some((c) => String(c[0]).includes('amplitudes'))).toBe(true);
        expect(opens.some((c) => String(c[0]).includes('cholesky_factors'))).toBe(true);
      });

      it('should handle missing spatial index gracefully', async () => {
        const noOrderingNode: SceneNode = {
          ...mockNode,
          attrs: { ...mockNode.attrs, ordering: 'none' },
        };
        bodyLoader.dispose();
        bodyLoader = new GSplatsSpatialIndexLoader(
          mockZarrLocation as unknown as ConstructorParameters<typeof GSplatsSpatialIndexLoader>[0],
          noOrderingNode
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadGSplats(viewState);
        // [data.md/W][P2] Strengthen: pin concrete typed-array shape on the
        // ordering='none' fallback path — production must still produce a
        // Float32Array result (load-all path with no index).
        expect(result.positions).toBeInstanceOf(Float32Array);
        expect(result.amplitudes).toBeInstanceOf(Float32Array);
        expect(result.choleskyFactors).toBeInstanceOf(Float32Array);
        // splatCount tracks positions length / ndim (3D in this fixture).
        expect(result.positions.length % 3).toBe(0);
        expect(result.splatCount).toBe(result.positions.length / 3);
      });

      it('should handle missing colors array gracefully', async () => {
        (zarr.open as unknown as ReturnType<typeof vi.fn>).mockImplementation(
          (location: unknown) => {
            const path = String(location);
            if (path.includes('chunk_bounds')) return Promise.resolve(chunkBoundsArray);
            if (path.includes('centers')) return Promise.resolve(mockArrays.centers);
            if (path.includes('amplitudes')) return Promise.resolve(mockArrays.amplitudes);
            if (path.includes('cholesky_factors'))
              return Promise.resolve(mockArrays.cholesky_factors);
            // colors rejected
            return Promise.reject(new Error('Not found'));
          }
        );

        bodyLoader.dispose();
        bodyLoader = new GSplatsSpatialIndexLoader(
          mockZarrLocation as unknown as ConstructorParameters<typeof GSplatsSpatialIndexLoader>[0],
          mockNode
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadGSplats(viewState);
        // [data.md/W][P2] Strengthen typed-array shape assertions.
        expect(result.positions).toBeInstanceOf(Float32Array);
        expect(result.amplitudes).toBeInstanceOf(Float32Array);
        expect(result.choleskyFactors).toBeInstanceOf(Float32Array);
        expect(result.colors).toBeFalsy();
      });

      it('should only initialize once with concurrent calls', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await Promise.all([
          bodyLoader.loadGSplats(viewState),
          bodyLoader.loadGSplats(viewState),
          bodyLoader.loadGSplats(viewState),
        ]);

        const chunkBoundsOpens = (
          zarr.open as unknown as ReturnType<typeof vi.fn>
        ).mock.calls.filter((c) => String(c[0]).includes('chunk_bounds')).length;
        expect(chunkBoundsOpens).toBe(1);
      });
    });

    describe('spatial index queries', () => {
      it('should construct the query builder with index, viewState, and options', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0.5, 0.5, 0.5],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadGSplats(viewState);

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
        expect(options.totalElements).toBe(5000); // n_splats
        expect(options.chunkSize).toBe(256);
        expect(options.geometryType).toBe('gsplats');
      });

      it('should return empty gsplats when builder returns no ranges', async () => {
        mockExecute.mockResolvedValueOnce([]);

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [100, 100, 100],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadGSplats(viewState);
        expect(result.splatCount).toBe(0);
        expect(result.positions.length).toBe(0);
      });
    });

    describe('extend_to_all', () => {
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

        await bodyLoader.loadGSplats(viewState);

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

        await bodyLoader.loadGSplats(viewState);

        const call = (SpatialQueryBuilder as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
        const options = call[2] as { extendDims: string[] };
        expect(options.extendDims).toEqual([]);
      });
    });

    describe('data loading (direct per-splat path)', () => {
      it('should produce arrays sized to the loaded splat count', async () => {
        mockExecute.mockResolvedValueOnce([{ start: 0, end: 4 }]);

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadGSplats(viewState);

        expect(result.splatCount).toBe(4);
        expect(result.positions.length).toBe(4 * 3); // ndim=3
        expect(result.amplitudes.length).toBe(4);
        expect(result.choleskyFactors.length).toBe(4 * 6); // packed 3D = 6
      });

      it('should preserve original ndim in result metadata', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadGSplats(viewState);
        expect(result.ndim).toBe(3);
      });

      it('should size cholesky for higher-dim datasets', async () => {
        // 4D → cholesky packed = 4·5/2 = 10
        const fourDNode: SceneNode = {
          ...mockNode,
          attrs: { ...mockNode.attrs, ndim: 4, n_splats: 100, chunk_size: 50 },
        };
        mockArrays.centers.shape = [100, 4];
        mockArrays.cholesky_factors.shape = [100, 10];
        mockArrays.amplitudes.shape = [100];
        mockArrays.colors.shape = [100, 3];
        chunkBoundsArray.shape = [2, 4, 2]; // 2 chunks at 50 splats each

        mockExecute.mockResolvedValueOnce([{ start: 0, end: 5 }]);
        (zarr.get as unknown as ReturnType<typeof vi.fn>).mockImplementation(
          (array: unknown, slices?: unknown) => {
            if (array === chunkBoundsArray) {
              return Promise.resolve({ data: new Float32Array(2 * 4 * 2) });
            }
            const sliceSpec = slices as Array<{ start: number; end: number }>;
            const count = sliceSpec[0].end - sliceSpec[0].start;
            let elementsPerItem = 1;
            if (array === mockArrays.centers) elementsPerItem = 4;
            else if (array === mockArrays.cholesky_factors) elementsPerItem = 10;
            else if (array === mockArrays.colors) elementsPerItem = 3;
            return Promise.resolve({ data: new Float32Array(count * elementsPerItem) });
          }
        );

        bodyLoader.dispose();
        bodyLoader = new GSplatsSpatialIndexLoader(
          mockZarrLocation as unknown as ConstructorParameters<typeof GSplatsSpatialIndexLoader>[0],
          fourDNode
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, 0],
          tolerance: [0, 0, 0, 0],
        };

        const result = await bodyLoader.loadGSplats(viewState);
        expect(result.splatCount).toBe(5);
        expect(result.positions.length).toBe(5 * 4);
        expect(result.choleskyFactors.length).toBe(5 * 10);
        expect(result.ndim).toBe(4);
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

        await bodyLoader.loadGSplats(viewState);

        expect(listener).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'query',
            loader: 'gsplats-spatial-index',
          })
        );
      });

      it('should clear active queries after completion', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadGSplats(viewState);
        expect(bodyLoader.getActiveQueries().length).toBe(0);
      });

      it('should update metrics correctly', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadGSplats(viewState);

        const metrics = bodyLoader.getMetrics();
        expect(metrics.queries).toBe(1);
        expect(metrics.type).toBe('gsplats-spatial-index');
        expect(metrics.path).toBe('/test_gsplats');
      });
    });

    describe('error handling', () => {
      it('should handle initialization errors', async () => {
        (zarr.open as unknown as ReturnType<typeof vi.fn>).mockRejectedValue(
          new Error('Failed to open centers')
        );

        bodyLoader.dispose();
        bodyLoader = new GSplatsSpatialIndexLoader(
          mockZarrLocation as unknown as ConstructorParameters<typeof GSplatsSpatialIndexLoader>[0],
          mockNode
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await expect(bodyLoader.loadGSplats(viewState)).rejects.toThrow();
      });

      it('should record errors in metrics on failure', async () => {
        (zarr.get as unknown as ReturnType<typeof vi.fn>).mockImplementation((array: unknown) => {
          if (array === chunkBoundsArray) {
            return Promise.resolve({ data: new Float32Array(20 * 3 * 2) });
          }
          return Promise.reject(new Error('Load failed'));
        });

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await expect(bodyLoader.loadGSplats(viewState)).rejects.toThrow();
        const metrics = bodyLoader.getMetrics();
        expect(metrics.errors).toBeGreaterThanOrEqual(1);
      });

      // Points emits a monitor 'error' event on load
      // failure; errors should be surfaced through the same observable
      // path as Points.
      // Event-driven dashboards (timelines/advisors) couldn't see
      // gsplats failures the way they saw points failures.
      it('emits a monitor "error" event on load failure', async () => {
        (zarr.get as unknown as ReturnType<typeof vi.fn>).mockImplementation((array: unknown) => {
          if (array === chunkBoundsArray) {
            return Promise.resolve({ data: new Float32Array(20 * 3 * 2) });
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
        await expect(bodyLoader.loadGSplats(viewState)).rejects.toThrow();

        expect(listener).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'error',
            loader: 'gsplats-spatial-index',
            data: expect.objectContaining({
              path: '/test_gsplats',
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

        await bodyLoader.loadGSplats(viewState1);
        const result = await bodyLoader.updateView(viewState2);
        // [data.md/W][P2] Strengthen: updateView returns a fresh result with
        // concrete typed-array fields, not just any truthy value.
        expect(result.positions).toBeInstanceOf(Float32Array);
        expect(result.amplitudes).toBeInstanceOf(Float32Array);
        expect(result.choleskyFactors).toBeInstanceOf(Float32Array);
        expect(SpatialQueryBuilder).toHaveBeenCalledTimes(2);
      });
    });

    describe('resource cleanup', () => {
      it('should dispose resources properly', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.loadGSplats(viewState);
        bodyLoader.dispose();

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
            if (array === chunkBoundsArray) {
              return Promise.resolve({ data: new Float32Array(20 * 3 * 2) });
            }
            const sliceSpec = slices as Array<{ start: number; end: number }>;
            const count = sliceSpec[0].end - sliceSpec[0].start;
            if (array === mockArrays.colors) {
              const buf = new Uint8Array(count * 3);
              for (let i = 0; i < count; i++) buf[i * 3] = 255;
              return Promise.resolve({ data: buf });
            }
            let elementsPerItem = 1;
            if (array === mockArrays.centers) elementsPerItem = 3;
            else if (array === mockArrays.cholesky_factors) elementsPerItem = 6;
            return Promise.resolve({ data: new Float32Array(count * elementsPerItem) });
          }
        );

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadGSplats(viewState);
        expect(result.colors).toBeInstanceOf(Uint8Array);
      });

      it('should keep direct Float32 (HDR) colors as Float32Array', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const result = await bodyLoader.loadGSplats(viewState);
        expect(result.colors).toBeInstanceOf(Float32Array);
      });
    });

    // ────────────────────────────────────────────────────────────────
    // GSplats-only API: prefetchChunks. Points and lines have no
    // multi-LOD pathway, so this block has no parallel in the other
    // two test files (justified asymmetry).
    // ────────────────────────────────────────────────────────────────
    describe('prefetchChunks (gsplats-only)', () => {
      it('warms the cache with zarr.get on every array × range', async () => {
        mockExecute.mockResolvedValueOnce([{ start: 0, end: 50 }]);

        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        await bodyLoader.prefetchChunks(viewState);

        // Required arrays + colors = 4 arrays. With 1 range each → 4 get() calls.
        // Plus 1 chunk_bounds get() during initialize().
        const getCalls = (zarr.get as unknown as ReturnType<typeof vi.fn>).mock.calls;
        expect(getCalls.length).toBeGreaterThanOrEqual(4);
      });

      it('skips fetches when the spatial query returns no ranges', async () => {
        mockExecute.mockResolvedValueOnce([]);
        // Drain the chunk_bounds open + get from initialization first.
        await bodyLoader.prefetchChunks({
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        });

        const callsBefore = (zarr.get as unknown as ReturnType<typeof vi.fn>).mock.calls.length;

        // Second prefetch with no ranges → no additional fetches.
        mockExecute.mockResolvedValueOnce([]);
        await bodyLoader.prefetchChunks({
          displayDims: [0, 1, 2],
          slicePosition: [100, 100, 100],
          tolerance: [0, 0, 0],
        });

        const callsAfter = (zarr.get as unknown as ReturnType<typeof vi.fn>).mock.calls.length;
        expect(callsAfter).toBe(callsBefore);
      });
    });
  });
});
