/**
 * Differential characterization test: L0-cache wiring across all three
 * spatial-index loaders (points / lines / gsplats).
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The three loaders share an identical constructor signature and an
 * identical `wrapWithCache(...)` block repeated once per opened zarr array
 * — 21 call sites in total, and (measured 2026-09) covered by ZERO tests.
 * `this.l0Cache` was falsy in every existing loader test, so the whole
 * wrapping path, including the twelve `() => this._activeProbe` /
 * `() => this._activeSignal` thunks, never executed.
 *
 * The wrapper's own behaviour is NOT what is untested:
 * `cache/decompressed-chunk-cache/cached-zarr-array.ts` sits at 97.9% lines
 * / 100% functions from its own unit test. What is untested is each
 * loader's *wiring* of it. So the seam here is the call site, and the real
 * `wrapWithCache` still runs (partial mock over `importOriginal`) — the spy
 * only records how it was invoked.
 *
 * WHAT IT PINS (as properties, not golden lists)
 * ----------------------------------------------
 *   1. No successfully-opened array bypasses the cache. This is the
 *      invariant with teeth: adding a new zarr array to a loader and
 *      forgetting to wrap it fails here, and it keeps holding after the
 *      duplicated blocks are extracted into one shared helper.
 *   2. Every wrap is keyed `<node.path>/<arrayName>` — the cache is shared
 *      process-wide, so an unprefixed key would collide across nodes.
 *   3. Probe and signal are passed as LIVE accessors, not snapshots.
 *      Replacing `() => this._activeProbe` with `this._activeProbe` reads
 *      as a harmless simplification and would permanently pin the wrapper
 *      to `null`; these assertions mutate the loader's state after wrapping
 *      and require the thunk to observe the change.
 *   4. Degenerate control arm: with no L0 cache, nothing is wrapped and the
 *      loader still loads.
 *
 * This is the acceptance criterion for extracting the shared
 * `load<Geom>Internal` / array-open body, written before the extraction so the
 * extraction has something to be checked against.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as zarr from 'zarrita';

import { PointsSpatialIndexLoader } from '../../../../data/points/points-spatial-index-loader';
import { LinesSpatialIndexLoader } from '../../../../data/lines/lines-spatial-index-loader';
import { GSplatsSpatialIndexLoader } from '../../../../data/gsplats/gsplats-spatial-index-loader';
import type { SceneNode } from '../../../../data';
import { makeMockZarrLocation } from '../../../builders/spatial-loader-fixtures';

// ── zarrita mock ────────────────────────────────────────────────────────
// `NotFoundError` is a real class here on purpose. The loaders funnel every
// optional-array miss through `isNotFoundError`, whose first line is
// `error instanceof zarrita.NotFoundError`; with the bare `{open, get, ...}`
// mock the sibling test files use, that expression throws
// "Right-hand side of 'instanceof' is not callable" instead of answering.
// `vi.hoisted` because both of these are referenced from `vi.mock` factories,
// which are hoisted above every top-level binding in the file.
const { MockNotFoundError, wrapSpy } = vi.hoisted(() => {
  class MockNotFoundError extends Error {}
  return { MockNotFoundError, wrapSpy: vi.fn() };
});

vi.mock('zarrita', () => ({
  registry: {},
  open: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start: number, end: number) => ({ start, end })),
  NotFoundError: MockNotFoundError,
}));

// ── wrapWithCache spy over the REAL implementation ──────────────────────
vi.mock('../../../../cache/decompressed-chunk-cache/cached-zarr-array', async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import('../../../../cache/decompressed-chunk-cache/cached-zarr-array')
    >();
  return {
    ...actual,
    wrapWithCache: (...args: Parameters<typeof actual.wrapWithCache>) => {
      wrapSpy(...args);
      return actual.wrapWithCache(...args);
    },
  };
});

// The canonical query builder is mocked so these tests exercise loader
// orchestration rather than the AABB scan (which owns its own unit tests).
const mockExecute = vi.fn();
vi.mock('../../../../data/loaders/spatial-query/spatial-query-builder', async () => {
  const actual = await vi.importActual<
    typeof import('../../../../data/loaders/spatial-query/spatial-query-builder')
  >('../../../../data/loaders/spatial-query/spatial-query-builder');
  return {
    ...actual,
    SpatialQueryBuilder: vi.fn().mockImplementation(() => ({ execute: mockExecute })),
  };
});

/** Minimal stand-in for a DecompressedChunkCache; identity is all we assert. */
function makeFakeL0Cache() {
  return {
    __brand: 'fake-l0-cache',
    makeKey: vi.fn((path: string, coords: number[]) => `${path}:${coords.join(',')}`),
    get: vi.fn(() => undefined),
    set: vi.fn(),
    has: vi.fn(() => false),
  };
}

/**
 * A mock zarr array. `__name` rides through the `wrapWithCache` Proxy
 * (which forwards unknown property reads to the target), so the `zarr.get`
 * mock can still identify an array after it has been wrapped — identity
 * comparison against the raw object no longer works once a Proxy is in
 * play, which is precisely the situation these tests create.
 */
function makeArray(name: string, shape: number[], attrs: object = {}) {
  return { __name: name, shape, dtype: 'float32', attrs };
}

type Geometry = {
  name: string;
  /** Arrays `zarr.open` will resolve; anything else 404s. */
  arrays: Record<string, ReturnType<typeof makeArray>>;
  node: SceneNode;
  /** Construct with the 7-arg signature all three loaders share. */
  construct: (
    location: unknown,
    node: SceneNode,
    l0Cache: unknown
  ) => { dispose: () => void } & Record<string, unknown>;
  /** Drive one load through the public entry point. */
  load: (loader: any) => Promise<unknown>;
};

const VIEW_STATE = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0],
  tolerance: [0, 0, 0],
};

function pointsGeometry(): Geometry {
  return {
    name: 'points',
    arrays: {
      positions: makeArray('positions', [1000, 3]),
      colors: makeArray('colors', [1000, 3]),
      radii: makeArray('radii', [1000]),
      sharpnesses: makeArray('sharpnesses', [1000]),
    },
    node: {
      path: '/test_points',
      type: 'points',
      attrs: {
        n_points: 1000,
        max_radius: 0.5,
        ordering: 'none',
        ndim: 3,
      },
      hasSpatialIndex: false,
    } as unknown as SceneNode,
    construct: (location, node, l0Cache) =>
      new PointsSpatialIndexLoader(
        location as never,
        node,
        undefined,
        undefined,
        l0Cache as never
      ) as never,
    load: (loader) => loader.loadPoints(VIEW_STATE),
  };
}

function linesGeometry(): Geometry {
  const ordering = {
    slice_dims: [],
    ordering_dims: [0, 1, 2],
    ordering_min: [0, 0, 0],
    ordering_max: [1, 1, 1],
    chunk_size: 100,
    ordering_bits_per_dim: 21,
  };
  return {
    name: 'lines',
    arrays: {
      vertices: makeArray('vertices', [1000, 3]),
      widths: makeArray('widths', [1000]),
      segments: makeArray('segments', [999, 2]),
      colors: makeArray('colors', [1000, 3]),
    },
    node: {
      path: '/test_lines',
      type: 'lines',
      attrs: {
        type: 'lines',
        n_vertices: 1000,
        n_segments: 999,
        ndim: 3,
        max_width: 1.0,
        ordering: 'none',
        original_line_type: 'segments',
        has_colors: true,
        has_sharpness: false,
        vertex_ordering: ordering,
        segment_ordering: ordering,
      },
      hasSpatialIndex: false,
    } as unknown as SceneNode,
    construct: (location, node, l0Cache) =>
      new LinesSpatialIndexLoader(
        location as never,
        node,
        undefined,
        undefined,
        l0Cache as never
      ) as never,
    load: (loader) => loader.loadLines(VIEW_STATE),
  };
}

function gsplatsGeometry(): Geometry {
  return {
    name: 'gsplats',
    arrays: {
      centers: makeArray('centers', [1000, 3]),
      amplitudes: makeArray('amplitudes', [1000]),
      cholesky_factors: makeArray('cholesky_factors', [1000, 6]),
      colors: makeArray('colors', [1000, 3]),
    },
    node: {
      path: '/test_gsplats',
      type: 'gsplats',
      attrs: {
        n_splats: 1000,
        ndim: 3,
        ordering: 'none',
        has_colors: true,
      },
      hasSpatialIndex: false,
    } as unknown as SceneNode,
    construct: (location, node, l0Cache) =>
      new GSplatsSpatialIndexLoader(
        location as never,
        node,
        undefined,
        undefined,
        l0Cache as never
      ) as never,
    load: (loader) => loader.loadGSplats(VIEW_STATE),
  };
}

const GEOMETRIES = [pointsGeometry, linesGeometry, gsplatsGeometry];

describe('L0 cache wiring (differential across the three spatial-index loaders)', () => {
  let opened: string[];

  beforeEach(() => {
    vi.clearAllMocks();
    opened = [];
    mockExecute.mockResolvedValue([{ start: 0, end: 100 }]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  /** Wire `zarr.open` / `zarr.get` for one geometry's array set. */
  function wireZarr(geometry: Geometry) {
    (zarr.open as never as ReturnType<typeof vi.fn>).mockImplementation((location: unknown) => {
      const path = String(location);
      const hit = Object.keys(geometry.arrays).find((name) => path.endsWith(`/${name}`));
      if (!hit) {
        return Promise.reject(new MockNotFoundError(`404 Not Found: ${path}`));
      }
      opened.push(hit);
      return Promise.resolve(geometry.arrays[hit]);
    });

    (zarr.get as never as ReturnType<typeof vi.fn>).mockImplementation(
      (
        array: { __name?: string; shape: number[] },
        slices?: Array<{ start: number; end: number }>
      ) => {
        const perElement = array.shape.length === 2 ? array.shape[1] : 1;
        const count = slices?.[0] ? slices[0].end - slices[0].start : array.shape[0];
        return Promise.resolve({ data: new Float32Array(Math.max(0, count) * perElement) });
      }
    );
  }

  describe.each(GEOMETRIES.map((g) => [g().name, g] as const))('%s', (_name, makeGeometry) => {
    it('routes every successfully-opened array through the L0 cache', async () => {
      const geometry = makeGeometry();
      wireZarr(geometry);
      const cache = makeFakeL0Cache();
      const loader = geometry.construct(makeMockZarrLocation(), geometry.node, cache);

      await geometry.load(loader).catch(() => undefined);

      // Sanity: the fixture must actually have opened something, or the
      // invariant below would hold vacuously.
      expect(opened.length).toBeGreaterThan(0);

      const wrappedPaths = wrapSpy.mock.calls.map((call) => call[2] as string);
      const wrappedNames = new Set(wrappedPaths.map((p) => p.split('/').pop()));

      // THE invariant: nothing that was opened escapes the cache.
      expect([...new Set(opened)].sort()).toEqual([...wrappedNames].sort());

      loader.dispose();
    });

    it('keys every wrap under the node path so sibling nodes cannot collide', async () => {
      const geometry = makeGeometry();
      wireZarr(geometry);
      const cache = makeFakeL0Cache();
      const loader = geometry.construct(makeMockZarrLocation(), geometry.node, cache);

      await geometry.load(loader).catch(() => undefined);

      expect(wrapSpy).toHaveBeenCalled();
      for (const call of wrapSpy.mock.calls) {
        expect(call[1]).toBe(cache); // the one cache instance, not a copy
        expect(call[2]).toMatch(new RegExp(`^${geometry.node.path}/[a-z_]+$`));
      }

      loader.dispose();
    });

    it('hands the wrapper live probe/signal accessors, not snapshots', async () => {
      const geometry = makeGeometry();
      wireZarr(geometry);
      const cache = makeFakeL0Cache();
      const loader = geometry.construct(makeMockZarrLocation(), geometry.node, cache);

      await geometry.load(loader).catch(() => undefined);
      expect(wrapSpy).toHaveBeenCalled();

      const [, , , getProbe, getSignal] = wrapSpy.mock.calls[0];
      expect(typeof getProbe).toBe('function');
      expect(typeof getSignal).toBe('function');

      // Between loads both are null...
      expect(getProbe()).toBeNull();
      expect(getSignal()).toBeNull();

      // ...and a later state change must be VISIBLE through the same
      // accessor. A snapshot (`this._activeProbe` instead of
      // `() => this._activeProbe`) would still return null here.
      const probe = { record: vi.fn() };
      const controller = new AbortController();
      (loader as unknown as Record<string, unknown>)._activeProbe = probe;
      (loader as unknown as Record<string, unknown>)._activeSignal = controller.signal;

      expect(getProbe()).toBe(probe);
      expect(getSignal()).toBe(controller.signal);

      // Every wrapped array must share that liveness, not just the first.
      for (const call of wrapSpy.mock.calls) {
        expect(call[3]()).toBe(probe);
        expect(call[4]()).toBe(controller.signal);
      }

      loader.dispose();
    });

    it('wraps nothing when no L0 cache is configured, and still loads', async () => {
      const geometry = makeGeometry();
      wireZarr(geometry);
      const loader = geometry.construct(makeMockZarrLocation(), geometry.node, undefined);

      await expect(geometry.load(loader)).resolves.toBeDefined();

      expect(opened.length).toBeGreaterThan(0);
      expect(wrapSpy).not.toHaveBeenCalled();

      loader.dispose();
    });
  });
});
