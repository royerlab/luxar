/**
 * Tests for LinesSpatialIndexLoader.
 *
 * Mirrors the describe-block structure of
 * `tests/unit/data/points/spatial-index-loader.test.ts` per the three-geometry
 * symmetry rule (Points / Lines / GSplats keep mirrored coverage).
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
import { SliceCache } from '../../../../cache/slice-cache';
import { measureLodBytes } from '../../../../data/loaders/progressive/slice-cache-helper';

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
      expect(metrics.elementsLoaded).toBe(0);
      expect(metrics.bytesLoaded).toBe(0);
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

    // ────────────────────────────────────────────────────────────────
    // Plain-leaf S-cache: a plain leaf caches its decoded slice as a
    // 1-element ladder under the progressive loaders' key contract
    // (restoreLadder/storeLadder). Symmetric block across the three
    // spatial-index loader test files.
    describe('plain-leaf S-cache', () => {
      const hiddenDimView: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 5],
        tolerance: [0, 0, 0, 0.25],
      };
      let sliceCache: SliceCache;
      let cachedLoader: LinesSpatialIndexLoader;

      beforeEach(() => {
        sliceCache = new SliceCache({ maxSize: 8 * 1024 * 1024 });
        cachedLoader = new LinesSpatialIndexLoader(
          mockZarrLocation as unknown as ConstructorParameters<typeof LinesSpatialIndexLoader>[0],
          mockNode,
          undefined,
          undefined,
          undefined,
          undefined,
          sliceCache
        );
      });

      afterEach(() => {
        cachedLoader?.dispose();
      });

      const gets = () => (zarr.get as unknown as ReturnType<typeof vi.fn>).mock.calls.length;

      it('same-view revisits restore from the S-cache: same reference, zero new reads', async () => {
        const first = await cachedLoader.loadLines(hiddenDimView);
        const readsAfterFirst = gets();

        const second = await cachedLoader.loadLines(hiddenDimView);
        const third = await cachedLoader.loadLines(hiddenDimView);

        // Hits return the SAME cached payload object (feeds the downstream
        // reference-identity no-op commit) and touch zarr not at all.
        expect(third).toBe(second);
        expect(second.vertexCount).toBe(first.vertexCount);
        expect(gets()).toBe(readsAfterFirst);
      });

      it('a different slicePosition is a miss: loads fresh and stores a second entry', async () => {
        await cachedLoader.loadLines(hiddenDimView);
        const readsAfterFirst = gets();

        await cachedLoader.loadLines({ ...hiddenDimView, slicePosition: [0, 0, 0, 6] });

        expect(gets()).toBeGreaterThan(readsAfterFirst);
        expect(sliceCache.getStats().count).toBe(2);
      });

      it('stores nothing when every dimension is displayed (single-slice view)', async () => {
        await cachedLoader.loadLines({
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        });
        expect(sliceCache.getStats().count).toBe(0);
      });

      it('a failed (e.g. aborted) load stores nothing', async () => {
        mockExecute.mockRejectedValueOnce(new DOMException('aborted', 'AbortError'));
        await expect(cachedLoader.loadLines(hiddenDimView)).rejects.toThrow();
        expect(sliceCache.getStats().count).toBe(0);
      });

      it('the cached snapshot is a deep clone: post-store accumulator reuse cannot corrupt it', async () => {
        const first = await cachedLoader.loadLines(hiddenDimView);
        // Simulate the loader's next pass overwriting the reused accumulator
        // buffers that `first`'s arrays alias.
        (first.positions as Float32Array).fill(999);

        const second = await cachedLoader.loadLines(hiddenDimView);
        expect(second.positions[0]).toBe(0);
      });

      it('an empty slice is cached via the wrapper: revisits skip the query entirely', async () => {
        // Zero visible ranges → the internal returns empty data and the
        // WRAPPER stores it (same contract as Points/GSplats: an empty slice
        // is a valid, ~0-byte result that revisits should skip).
        mockExecute.mockResolvedValue([]);

        const first = await cachedLoader.loadLines(hiddenDimView);
        expect(first.vertexCount).toBe(0);
        expect(sliceCache.getStats().count).toBe(1);
        const readsAfterFirst = gets();

        const second = await cachedLoader.loadLines(hiddenDimView);
        const third = await cachedLoader.loadLines(hiddenDimView);

        expect(second.vertexCount).toBe(0);
        expect(third).toBe(second);
        expect(gets()).toBe(readsAfterFirst);
      });
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

      it('does not probe optional arrays the store listing rules out', async () => {
        // The consolidated listing saw vertices/segments/widths under this node and
        // nothing else: colors / sharpnesses must not cost a 404 round trip, while
        // widths (listed) is still opened.
        const listedNode = makeLinesNode({
          arrays: new Set([
            'vertices',
            'segments',
            'widths',
            'vertex_chunk_bounds',
            'segment_chunk_bounds',
          ]),
        });
        const listedLoader = new LinesSpatialIndexLoader(
          makeMockZarrLocation() as unknown as ConstructorParameters<
            typeof LinesSpatialIndexLoader
          >[0],
          listedNode
        );
        await listedLoader.loadLines({
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        });
        const opens = (zarr.open as unknown as ReturnType<typeof vi.fn>).mock.calls.map((c) =>
          String(c[0])
        );
        expect(opens.some((p) => p.includes('widths'))).toBe(true);
        expect(opens.some((p) => p.includes('colors'))).toBe(false);
        expect(opens.some((p) => p.includes('sharpnesses'))).toBe(false);
        listedLoader.dispose();
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

    // ────────────────────────────────────────────────────────────────
    // Dataset-switch race: dispose() lands while loadLines' vertex
    // reads are in flight. Pre-fix this dereferenced the nulled
    // `_accumulator` after the awaits and the scene-loader retry path
    // logged it as a FAILURE. The load must instead bail as a
    // cancellation (DOMException name 'AbortError'). Symmetric block
    // in the gsplats loader tests.
    describe('dispose during in-flight load (dataset-switch race)', () => {
      it('rejects with AbortError, not a TypeError', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        // Gate the VERTEX-side reads; bounds + segment reads stay fast
        // so the load reaches the accumulator awaits.
        let releaseReads!: () => void;
        const gate = new Promise<void>((resolve) => {
          releaseReads = resolve;
        });
        const baseGet = zarr.get as unknown as ReturnType<typeof vi.fn>;
        const realImpl = baseGet.getMockImplementation() as (
          array: unknown,
          slices?: unknown
        ) => Promise<unknown>;
        baseGet.mockImplementation(async (array: unknown, slices?: unknown) => {
          const result = await realImpl(array, slices);
          if (
            array !== vertexBoundsArray &&
            array !== segmentBoundsArray &&
            array !== mockArrays.segments
          ) {
            await gate;
          }
          return result;
        });

        const pending = bodyLoader.loadLines(viewState);
        await new Promise((r) => setTimeout(r, 10));
        bodyLoader.dispose();
        releaseReads();

        await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
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

    // ────────────────────────────────────────────────────────────────
    // Issue #1424: the on-disk VERTEX range bounds (index space A) are the
    // loader's half of the slot → on-disk map picking resolves per-vertex
    // labels through. Gated on the node declaring a label CSR, exactly like
    // the Points / GSplats twins.
    describe('vertexRangeBounds publication (picking element IDs)', () => {
      const viewState: ViewState = {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [0, 0, 0],
      };

      /** A copy of the body fixture's node with one label flag flipped on. */
      function makeLabelledNode(flag: 'has_labels' | 'has_image_labels' | 'has_keys'): SceneNode {
        const base = makeLinesNode();
        return { ...base, attrs: { ...base.attrs, [flag]: true } } as SceneNode;
      }

      function makeLabelledLoader(flag: 'has_labels' | 'has_image_labels' | 'has_keys') {
        return new LinesSpatialIndexLoader(
          mockZarrLocation as unknown as ConstructorParameters<typeof LinesSpatialIndexLoader>[0],
          makeLabelledNode(flag)
        );
      }

      it('omits vertexRangeBounds for a node with no label CSR', async () => {
        const result = await bodyLoader.loadLines(viewState);
        expect(result.vertexRangeBounds).toBeUndefined();
      });

      it('publishes the merged on-disk vertex ranges for a has_labels node', async () => {
        // The body fixture's spatial query returns segment ranges [0,50) and
        // [100,150), and each segment r references vertices (r, r+1) — so the
        // loaded vertex set is [0,51) ∪ [100,151), a genuinely multi-range,
        // NON-zero-anchored space. This is exactly the shape that makes the
        // raw pick slot the wrong answer.
        const labelled = makeLabelledLoader('has_labels');
        try {
          const result = await labelled.loadLines(viewState);
          // FLAT `[start0, end0, start1, end1]` pairs in a `Uint32Array` — the
          // shape the slice cache measures and deep-copies (see
          // `LoadedLinesData.vertexRangeBounds`).
          expect(result.vertexRangeBounds).toBeInstanceOf(Uint32Array);
          expect(Array.from(result.vertexRangeBounds!)).toEqual([0, 51, 100, 151]);
          // The ranges describe the loaded vertex arrays exactly.
          expect(result.vertexCount).toBe(102);
        } finally {
          labelled.dispose();
        }
      });

      it('publishes them for a has_image_labels node too', async () => {
        const labelled = makeLabelledLoader('has_image_labels');
        try {
          const result = await labelled.loadLines(viewState);
          expect(result.vertexRangeBounds).toHaveLength(4); // 2 ranges x 2 bounds
        } finally {
          labelled.dispose();
        }
      });

      it('publishes them for a has_keys node too', async () => {
        const labelled = makeLabelledLoader('has_keys');
        try {
          const result = await labelled.loadLines(viewState);
          expect(result.vertexRangeBounds).toHaveLength(4);
        } finally {
          labelled.dispose();
        }
      });

      it('publishes bounds the S-cache measures and deep-copies (a flat typed array)', async () => {
        // `LoadedLinesData.vertexRangeBounds` is a FLAT `Uint32Array` of
        // `[start, end)` pairs rather than an `ElementIdRange[]` object array
        // ENTIRELY because of the SliceCache helpers: they walk own TYPED-ARRAY
        // properties only, so an object array would be billed 0 bytes by
        // `measureLodBytes` (the LRU holding roughly twice the bytes its budget
        // believes for a fragmented labelled slice) and carried into the stored
        // snapshot BY REFERENCE by `cloneLodSnapshot`. Both halves are pinned
        // here against a payload from the REAL loader, so a return to an object
        // array fails this test — the docblocks in `types/lines.ts` and
        // `data/lines/lines-spatial-index-loader.ts` rest on that.
        const sliceCache = new SliceCache({ maxSize: 8 * 1024 * 1024 });
        const labelled = new LinesSpatialIndexLoader(
          mockZarrLocation as unknown as ConstructorParameters<typeof LinesSpatialIndexLoader>[0],
          makeLabelledNode('has_labels'),
          undefined,
          undefined,
          undefined,
          undefined,
          sliceCache
        );
        // A hidden (non-displayed) dimension is what makes a plain leaf
        // cacheable at all — see the plain-leaf S-cache block above.
        const cachedView: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, 5],
          tolerance: [0, 0, 0, 0.25],
        };
        try {
          const first = await labelled.loadLines(cachedView);
          const bounds = first.vertexRangeBounds!;
          expect(ArrayBuffer.isView(bounds)).toBe(true);

          // (a) MEASURED. The label flag is the only difference from the
          // unlabelled fixture loader, so the byte delta between the two
          // payloads is exactly the bounds array (2 ranges x 2 uint32).
          const unlabelled = await bodyLoader.loadLines(cachedView);
          expect(unlabelled.vertexRangeBounds).toBeUndefined();
          expect(bounds.byteLength).toBe(16);
          expect(measureLodBytes([first]) - measureLodBytes([unlabelled])).toBe(bounds.byteLength);
          // …and those are the bytes the LRU charged for the stored snapshot.
          expect(sliceCache.getStats().size).toBe(measureLodBytes([first]));

          // (b) DEEP-COPIED. The revisit is served from the stored clone, so
          // its bounds must be a DISTINCT buffer with equal contents; an object
          // array would have been carried across by reference instead.
          const revisit = await labelled.loadLines(cachedView);
          const cloned = revisit.vertexRangeBounds!;
          expect(cloned).not.toBe(bounds);
          expect(cloned.buffer).not.toBe(bounds.buffer);
          expect(Array.from(cloned)).toEqual(Array.from(bounds));
        } finally {
          labelled.dispose();
        }
      });

      it('publishes them on the accumulator-disabled FALLBACK path as well', async () => {
        // Both return sites must stamp the field: the accumulator path returns
        // a fresh `getData()` literal, the fallback path its own object literal.
        // Dropping either leaves picking silently on the raw-slot fallback for
        // half the configurations.
        const labelled = makeLabelledLoader('has_labels');
        try {
          // First load initializes (and creates the accumulator); then drop it
          // so the second load takes the allocating fallback branch.
          const viaAccumulator = await labelled.loadLines(viewState);
          expect(viaAccumulator.vertexRangeBounds).toHaveLength(4);
          expect(labelled.getAccumulatorStats()).not.toBeNull();

          (labelled as unknown as { _accumulator: unknown })._accumulator = null;
          // The poke is a private-field reach-in, so PROVE it disarmed the
          // branch instead of trusting the field name: `getAccumulatorStats()`
          // reads that same field and returns null only while it is null. A
          // rename would leave the real accumulator in place (non-null here),
          // and a lazy re-create during the load would show up as non-null
          // after it — either way the load below would silently take the
          // accumulator return site and this test would cover nothing.
          expect(labelled.getAccumulatorStats()).toBeNull();
          const viaFallback = await labelled.loadLines(viewState);
          expect(labelled.getAccumulatorStats()).toBeNull();
          // Positive evidence the fallback ALLOCATED: the accumulator path
          // hands back subarrays of its pooled buffers (identical across
          // repeat loads), the fallback a freshly allocated one.
          expect(viaFallback.positions.buffer).not.toBe(viaAccumulator.positions.buffer);
          expect(viaFallback.vertexRangeBounds).toBeInstanceOf(Uint32Array);
          expect(Array.from(viaFallback.vertexRangeBounds!)).toEqual([0, 51, 100, 151]);
        } finally {
          labelled.dispose();
        }
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
        // Regression (×3 symmetric): visibleElements (segments — the queried
        // unit) is written at query time. This loader shipped for months
        // never setting it (monitor showed a permanent 0).
        expect(metrics.visibleElements).toBeGreaterThan(0);
        // Resident memory is populated from the accumulator after a load
        // (was a perpetual 0 before — never written). Matches the MB→bytes
        // conversion done in recordLoadMetrics.
        const accMB = bodyLoader.getAccumulatorStats()?.memoryMB ?? 0;
        expect(accMB).toBeGreaterThan(0);
        expect(metrics.memoryUsed).toBe(Math.round(accMB * 1024 * 1024));
        // Chunk-index telemetry (segment side of the dual index) is attached
        // for the advisor (×3 symmetric).
        expect(metrics.spatialIndex).toBeDefined();
        expect(metrics.spatialIndex!.totalCells).toBeGreaterThan(0);
      });

      it('recreates a progressive-parent-released accumulator on the next query', async () => {
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        const first = await bodyLoader.loadLines(viewState);
        expect(bodyLoader.getAccumulatorStats()).not.toBeNull();

        bodyLoader.releaseAccumulator();
        expect(bodyLoader.getAccumulatorStats()).toBeNull();
        expect(bodyLoader.getMetrics().memoryUsed).toBe(0);

        const second = await bodyLoader.loadLines(viewState);
        expect(bodyLoader.getAccumulatorStats()).not.toBeNull();
        expect(second.vertexCount).toBe(first.vertexCount);
        expect(second.segmentCount).toBe(first.segmentCount);
        expect(second.positions).toEqual(first.positions);
        expect(Array.from(second.segments)).toEqual(Array.from(first.segments));
        expect(second.colors).toBeInstanceOf(Float32Array);
        expect(second.colors).toEqual(first.colors);
      });

      it('should fold completed loads into avgQueryTime (wrapper close-out)', async () => {
        // The wrapper's shared finishQueryTracking stamps the rolling mean
        // after the load completes — mirror of the Points/GSplats suites.
        // With real timers the elapsed may round to 0ms, so pin the
        // type/range rather than a concrete duration.
        const viewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        // Make elapsed time observable: every Date.now() call advances 5ms, so
        // if the wrapper's close-out were deleted, avgQueryTime would stay 0
        // and the strict > 0 assertion below would fail.
        let t = 1_000_000;
        const nowSpy = vi.spyOn(Date, 'now').mockImplementation(() => (t += 5));
        try {
          await bodyLoader.loadLines(viewState);
        } finally {
          nowSpy.mockRestore();
        }

        const metrics = bodyLoader.getMetrics();
        expect(metrics.queries).toBe(1);
        expect(metrics.avgQueryTime).toBeGreaterThan(0);
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
        expect(SpatialQueryBuilder).toHaveBeenCalledTimes(3);
      });

      it('skips fetches when the spatial query returns no ranges', async () => {
        // Mirror of the Points/GSplats prefetch zero-range tests: an
        // out-of-slice prefetch position must not issue any zarr reads.
        await bodyLoader.loadLines({
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        });

        const callsBefore = (zarr.get as any).mock.calls.length;

        mockExecute.mockResolvedValueOnce([]).mockResolvedValueOnce([]);
        await bodyLoader.prefetchChunks({
          displayDims: [0, 1, 2],
          slicePosition: [100, 100, 100],
          tolerance: [0, 0, 0],
        });

        const callsAfter = (zarr.get as any).mock.calls.length;
        expect(callsAfter).toBe(callsBefore);
      });
    });

    describe('resource cleanup', () => {
      it('dispose clears the active-query map (mid-flight leak guard)', async () => {
        // Regression (×3 symmetric): points dispose() historically omitted
        // activeQueries.clear(), so a dispose mid-flight leaked the tracked
        // query entry (lines/gsplats always cleared it).
        const baseViewState: ViewState = {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        };

        // Pre-initialize (chunk-bounds open + get) with the fast mock.
        await bodyLoader.loadLines(baseViewState);
        expect(bodyLoader.getActiveQueries().length).toBe(0);

        // Gate data-array reads so a query is observably mid-flight.
        let release!: () => void;
        const gate = new Promise<void>((resolve) => {
          release = resolve;
        });
        (zarr.get as any).mockImplementation(() =>
          gate.then(() => ({ data: new Float32Array(100) }))
        );

        const loadPromise = bodyLoader
          .loadLines({ ...baseViewState, slicePosition: [0.5, 0.5, 0.5] })
          .catch(() => null); // dispose mid-flight may fail the load — expected

        for (let i = 0; i < 10; i++) await Promise.resolve();
        expect(bodyLoader.getActiveQueries().length).toBeGreaterThanOrEqual(1);

        bodyLoader.dispose();
        expect(bodyLoader.getActiveQueries().length).toBe(0);

        release();
        await loadPromise;
      });
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

      it('should handle uint16 color data via the shared color helper', async () => {
        // Mirror of the Points suite's uint16 case: direct (unencoded)
        // Uint16 colors must be preserved natively end-to-end through the
        // loader, not just by the shared helper's own unit tests.
        mockArrays.colors.dtype = 'uint16';

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
              const buf = new Uint16Array(count * 3);
              for (let i = 0; i < count; i++) buf[i * 3] = 65535;
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
        expect(result.colors).toBeInstanceOf(Uint16Array);
        expect((result.colors as Uint16Array)[0]).toBe(65535);
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
