/**
 * Characterization tests for `PointsSpatialIndexLoader`'s derivation of
 * `spatial_extend_dims` from the scene's root attributes, and for the effect
 * that derivation has on the spatial query it issues.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * This is the switch that decides, per non-displayed dimension, whether
 * points extend through it as hyperspheres (spatial) or must match exactly
 * (categorical) — i.e. whether nD slicing shows neighbouring slices' points
 * or only the current one. It was entirely uncovered (measured 2026-09): the
 * sibling tests construct the loader without a `zarrStore`, so
 * `loadSpatialExtendDimsFromSceneDimensions` returned `null` on its first
 * line every time and the derivation below it never ran.
 *
 * WHAT IT PINS
 * ------------
 * The assertions go through an OBSERVABLE consequence, not the private
 * field: a derived config routes the query through
 * `calculateSpatialQueryTolerance` instead of `fallbackQueryTolerance`, so
 * the `tolerance` array handed to `SpatialQueryBuilder` is what gets
 * checked. In particular the `dim.spatial ?? false` default is pinned
 * differentially — an unset `spatial` flag must behave exactly like
 * `spatial: false`, never like `true`. Flipping that default would silently
 * widen every categorical axis into a hypersphere and pull in points from
 * adjacent timepoints/channels.
 *
 * The three `null` paths (no store, no `scene_dimensions`, root open
 * throwing) are pinned too: all three must degrade to the uniform fallback
 * WITHOUT throwing, because a scene whose root attrs cannot be read still
 * has to render.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as zarr from 'zarrita';

import { PointsSpatialIndexLoader } from '../../../../data/points/points-spatial-index-loader';
import type { SceneNode, ViewState } from '../../../../data';
import { makeMockZarrLocation } from '../../../builders/spatial-loader-fixtures';

const { MockNotFoundError } = vi.hoisted(() => {
  class MockNotFoundError extends Error {}
  return { MockNotFoundError };
});

vi.mock('zarrita', () => ({
  registry: {},
  open: vi.fn(),
  get: vi.fn(),
  slice: vi.fn((start: number, end: number) => ({ start, end })),
  NotFoundError: MockNotFoundError,
}));

const mockExecute = vi.fn();
const builderCtor = vi.fn();
vi.mock('../../../../data/loaders/spatial-query/spatial-query-builder', async () => {
  const actual = await vi.importActual<
    typeof import('../../../../data/loaders/spatial-query/spatial-query-builder')
  >('../../../../data/loaders/spatial-query/spatial-query-builder');
  return {
    ...actual,
    SpatialQueryBuilder: vi.fn().mockImplementation((...args: unknown[]) => {
      builderCtor(...args);
      return { execute: mockExecute };
    }),
  };
});

/** 4D node: three displayed spatial dims + one non-displayed dim under test. */
const NDIM = 4;

function makeNode(): SceneNode {
  return {
    path: '/test_points',
    type: 'points',
    attrs: {
      n_points: 1000,
      max_radius: 0.5,
      ordering: 'hilbert',
      ordering_dims: [0, 1, 2],
      slice_dims: [3],
      ordering_bits_per_dim: 21,
      chunk_size: 100,
      ndim: NDIM,
    },
    hasSpatialIndex: true,
  } as unknown as SceneNode;
}

const VIEW_STATE: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 5],
  tolerance: [0, 0, 0, 0.1],
};

/** A marker object standing in for the scene's root `zarr.Readable` store. */
const ROOT_STORE = { __brand: 'root-store' };

const CHUNK_BOUNDS = { __name: 'chunk_bounds', shape: [100, NDIM, 2], dtype: 'float32', attrs: {} };

/**
 * Wire `zarr.open` so the root-group open resolves to `rootAttrs` and the
 * data-array opens resolve to plain descriptors.
 *
 * @param rootAttrs - attrs for the root group, or the string 'throw' to make
 *   the root open reject (exercising the catch path).
 */
function wireZarr(rootAttrs: object | 'throw') {
  (zarr.open as never as ReturnType<typeof vi.fn>).mockImplementation(
    (location: unknown, opts?: { kind?: string }) => {
      if (location === ROOT_STORE || opts?.kind === 'group') {
        if (rootAttrs === 'throw') {
          return Promise.reject(new Error('root group unreadable'));
        }
        return Promise.resolve({ attrs: rootAttrs });
      }
      const path = String(location);
      if (path.includes('chunk_bounds')) return Promise.resolve(CHUNK_BOUNDS);
      if (path.includes('positions'))
        return Promise.resolve({
          __name: 'positions',
          shape: [1000, NDIM],
          dtype: 'float32',
          attrs: {},
        });
      return Promise.reject(new MockNotFoundError(`404 Not Found: ${path}`));
    }
  );

  (zarr.get as never as ReturnType<typeof vi.fn>).mockImplementation(
    (array: { shape: number[] }, slices?: Array<{ start: number; end: number }>) => {
      if (array === CHUNK_BOUNDS) {
        return Promise.resolve({ data: new Float32Array(100 * NDIM * 2) });
      }
      const perElement = array.shape.length === 2 ? array.shape[1] : 1;
      const count = slices?.[0] ? slices[0].end - slices[0].start : array.shape[0];
      return Promise.resolve({ data: new Float32Array(Math.max(0, count) * perElement) });
    }
  );
}

/** Root attrs describing three displayed dims plus one non-displayed dim. */
function sceneDimensions(fourth: { display: boolean; spatial?: boolean }) {
  return {
    scene_dimensions: {
      dimensions: [
        { name: 'z', display: true },
        { name: 'y', display: true },
        { name: 'x', display: true },
        { name: 't', ...fourth },
      ],
    },
  };
}

/** Run one load and return the `tolerance` the loader handed the builder. */
async function toleranceFor(rootAttrs: object | 'throw', withStore = true): Promise<number[]> {
  wireZarr(rootAttrs);
  // Scope the capture to THIS load. `toleranceFor` is called several times
  // per test, so reading `mock.calls[0]` would re-read the first case's
  // arguments for every subsequent case and make them all trivially agree.
  builderCtor.mockClear();
  const loader = new PointsSpatialIndexLoader(
    makeMockZarrLocation() as never,
    makeNode(),
    undefined,
    withStore ? (ROOT_STORE as never) : undefined
  );
  try {
    await loader.loadPoints(VIEW_STATE);
  } catch {
    // Loading may fail past the query (optional arrays are absent); the
    // builder args are captured before that point, which is what we assert.
  }
  expect(builderCtor).toHaveBeenCalled();
  const options = builderCtor.mock.calls[0][2] as { tolerance: number[] };
  loader.dispose();
  return options.tolerance;
}

describe('PointsSpatialIndexLoader — spatial_extend_dims derivation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockExecute.mockResolvedValue([{ start: 0, end: 100 }]);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('a non-displayed dimension`s `spatial` flag', () => {
    it('spatial:true and spatial:false produce different reach on that axis', async () => {
      const spatial = await toleranceFor(sceneDimensions({ display: false, spatial: true }));
      const categorical = await toleranceFor(sceneDimensions({ display: false, spatial: false }));

      expect(spatial).toHaveLength(NDIM);
      expect(categorical).toHaveLength(NDIM);
      // The three displayed dims are unaffected by the flag...
      expect(spatial.slice(0, 3)).toEqual(categorical.slice(0, 3));
      // ...and the flag is what moves the non-displayed one.
      expect(spatial[3]).not.toEqual(categorical[3]);
    });

    it('an UNSET spatial flag defaults to categorical, not spatial', async () => {
      // `dim.spatial ?? false`. If this default ever flips, a categorical
      // axis (time, channel) silently becomes a hypersphere and the viewer
      // pulls in points from adjacent slices.
      const unset = await toleranceFor(sceneDimensions({ display: false }));
      const explicitFalse = await toleranceFor(sceneDimensions({ display: false, spatial: false }));
      const explicitTrue = await toleranceFor(sceneDimensions({ display: false, spatial: true }));

      expect(unset).toEqual(explicitFalse);
      expect(unset).not.toEqual(explicitTrue);
    });

    it('a displayed dimension is spatial regardless of its own spatial flag', async () => {
      // Displayed dims are the viewing plane, so the derivation forces
      // `true` and never consults `dim.spatial` for them.
      const flagged = await toleranceFor(sceneDimensions({ display: true, spatial: false }));
      const unflagged = await toleranceFor(sceneDimensions({ display: true, spatial: true }));
      expect(flagged).toEqual(unflagged);
    });
  });

  describe('degrading to the uniform fallback', () => {
    it('falls back to the PERMISSIVE (spatial) reach, not the categorical one', async () => {
      // Measured, and worth stating plainly: the fallback is not neutral. It
      // reproduces the `spatial: true` reach on every non-displayed dim
      // (maxRadius, not the quarter-cell). So a scene whose root attrs cannot
      // be read queries a categorical axis as though it were spatial —
      // over-fetching rather than under-fetching, which is the safe direction
      // but is a real behavioural difference, not a no-op.
      const spatial = await toleranceFor(sceneDimensions({ display: false, spatial: true }));
      const categorical = await toleranceFor(sceneDimensions({ display: false, spatial: false }));
      const noStore = await toleranceFor(sceneDimensions({ display: false, spatial: true }), false);

      expect(noStore).toHaveLength(NDIM);
      expect(noStore).toEqual(spatial);
      expect(noStore).not.toEqual(categorical);
    });

    it('falls back when root attrs carry no scene_dimensions', async () => {
      const tolerance = await toleranceFor({ some_other_attr: 1 });
      expect(tolerance).toHaveLength(NDIM);
    });

    it('falls back — without throwing — when the root group cannot be read', async () => {
      const tolerance = await toleranceFor('throw');
      expect(tolerance).toHaveLength(NDIM);
    });

    it('all three failure modes produce the SAME fallback tolerance', async () => {
      // One fallback, not three subtly different ones.
      const noStore = await toleranceFor(sceneDimensions({ display: false, spatial: true }), false);
      const noDims = await toleranceFor({ some_other_attr: 1 });
      const threw = await toleranceFor('throw');
      expect(noDims).toEqual(noStore);
      expect(threw).toEqual(noStore);
    });
  });
});
