/**
 * GSplats `centers` / `amplitudes` array-ref resolution, against a REAL store.
 *
 * Regression coverage for issue #2490. When the DESI demo's two Points layers
 * started sharing coarse-level content, the encoder deduplicated the coarse
 * gsplat `centers` across them — an `array_ref` whose stored shape is `(0, 3)`
 * with the real extent in `encoding.original_shape`, and whose TARGET is
 * per-channel quantized. Nothing in the suite covered that combination:
 *
 *   - `array-decoder/decoder.test.ts` covers refs through
 *     `ArrayDecoder.decode` (whole-array, Points `colors`, direct target).
 *   - the gsplats range path is a different route entirely —
 *     `loadArrayRanges` -> `RangeLoader.loadRangesResolvingRef` ->
 *     `resolveArrayRef` -> `loadPerChannel` against the target's own attrs.
 *
 * A ref resolved against the placeholder's metadata could instead decode
 * corrupt values without throwing. This test catches that quiet failure mode
 * by comparing the decoded values with the target layer.
 *
 * The assertions on the STORE (not just on the loaded data) are load-bearing:
 * without them a future encoder change that stops deduplicating would leave
 * this file passing while covering nothing.
 */

import { describe, it, expect } from 'vitest';
import { FileSystemStore } from '@zarrita/storage';
import * as path from 'path';
import { fileURLToPath } from 'url';
import * as zarr from '../../../../data/zarr';
import { GSplatsSpatialIndexLoader } from '../../../../data/gsplats/gsplats-spatial-index-loader';
import { ArrayDecoder, type ArrayMetadata } from '../../../../data/array-decoder/decoder';
import type { SceneNode, ViewState } from '../../../../data';
import type { LoadedGSplatsData } from '../../../../types/gsplats';

const FIXTURES_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../../../../tests/fixtures'
);
const FIXTURE = 'test_gsplats_centers_array_ref.luxar.zarr';

/** The layer written FIRST — owns the stored centers/amplitudes. */
const TARGET_LAYER = 'materialised';
/** The layer written SECOND — its centers/amplitudes are array_refs. */
const REF_LAYER = 'deduplicated';

async function openStore(): Promise<zarr.Location<zarr.Readable>> {
  const store = await zarr.openStore(new FileSystemStore(path.join(FIXTURES_DIR, FIXTURE)));
  return zarr.root(store);
}

async function openArray(
  layer: string,
  name: string
): Promise<zarr.Array<zarr.DataType, zarr.Readable>> {
  const root = await openStore();
  return zarr.open(root.resolve(`${layer}/${name}`), { kind: 'array' });
}

/** Load a whole gsplats leaf through the production spatial-index loader. */
async function loadLayer(layer: string): Promise<LoadedGSplatsData> {
  const root = await openStore();
  const group = await zarr.open(root.resolve(layer), { kind: 'group' });
  const node = {
    path: `/${layer}`,
    type: 'gsplats',
    attrs: group.attrs as Record<string, unknown>,
    hasSpatialIndex: true,
    children: [],
  } as unknown as SceneNode;

  // Pass `zarrStore` to mirror the production loader-factory wiring. The
  // loader can also fall back to the store carried by its zarr location.
  const loader = new GSplatsSpatialIndexLoader(root.resolve(layer), node, undefined, root.store);
  await loader.initialize();
  const viewState = {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0],
    tolerance: [0, 0, 0],
  } as unknown as ViewState;
  return loader.loadGSplats(viewState);
}

/**
 * Loaded once per layer, but LAZILY rather than in a `beforeAll`: a hook that
 * throws skips every test in the file, so the assertion that would have named
 * the defect never runs and the report says "8 skipped". Loading inside the
 * tests keeps a failure attributable to the thing it broke.
 */
const loaded = new Map<string, Promise<LoadedGSplatsData>>();
function layerData(layer: string): Promise<LoadedGSplatsData> {
  const hit = loaded.get(layer);
  if (hit) return hit;
  const pending = loadLayer(layer);
  loaded.set(layer, pending);
  return pending;
}

describe('GSplats array_ref resolution (issue #2490)', () => {
  describe('the fixture really is the case under test', () => {
    it('the ref layer stores centers as a (0, 3) array_ref into the target layer', async () => {
      const array = await openArray(REF_LAYER, 'centers');
      const attrs = array.attrs as unknown as ArrayMetadata;
      expect(ArrayDecoder.isArrayRef(attrs)).toBe(true);
      expect(attrs.encoding?.target).toBe(`${TARGET_LAYER}/centers`);
      // The physical placeholder carries no rows; the real extent is in
      // `original_shape`. A store diff that reads the former as the row count
      // reports 500 lost splats — the #2490 misdiagnosis in one line.
      expect(array.shape).toEqual([0, 3]);
      expect(attrs.encoding?.original_shape).toEqual([500, 3]);
    });

    it('the ref layer stores amplitudes as an array_ref too', async () => {
      const array = await openArray(REF_LAYER, 'amplitudes');
      const attrs = array.attrs as unknown as ArrayMetadata;
      expect(ArrayDecoder.isArrayRef(attrs)).toBe(true);
      expect(attrs.encoding?.target).toBe(`${TARGET_LAYER}/amplitudes`);
    });

    it('the ref TARGET is per-channel quantized, so the perchannel decode is covered', async () => {
      const array = await openArray(TARGET_LAYER, 'centers');
      const attrs = array.attrs as unknown as ArrayMetadata;
      expect(attrs.encoding?.name).toBe('linear_perchannel_u16');
    });

    it('colors do NOT deduplicate, so the two layers stay distinguishable', async () => {
      const array = await openArray(REF_LAYER, 'colors');
      const attrs = array.attrs as unknown as ArrayMetadata;
      expect(ArrayDecoder.isArrayRef(attrs)).toBe(false);
    });
  });

  describe('the ref layer loads the same data as the target layer', () => {
    it('loads a non-empty centers buffer', async () => {
      const target = await layerData(TARGET_LAYER);
      const ref = await layerData(REF_LAYER);
      // Guards against the vacuous pass: an equality check between two empty
      // buffers is what a broken ref path would sail through.
      expect(target.positions.length).toBe(500 * 3);
      expect(ref.positions.length).toBe(target.positions.length);
      expect(ref.positions.some((v) => v !== 0)).toBe(true);
    });

    it('decodes centers bit-identically through the ref', async () => {
      const target = await layerData(TARGET_LAYER);
      const ref = await layerData(REF_LAYER);
      expect(Array.from(ref.positions)).toEqual(Array.from(target.positions));
    });

    it('decodes amplitudes bit-identically through the ref', async () => {
      const target = await layerData(TARGET_LAYER);
      const ref = await layerData(REF_LAYER);
      expect(ref.amplitudes.length).toBe(500);
      expect(ref.amplitudes.some((v) => v !== 0)).toBe(true);
      expect(Array.from(ref.amplitudes)).toEqual(Array.from(target.amplitudes));
    });

    it('keeps the non-deduplicated attributes distinct', async () => {
      const target = await layerData(TARGET_LAYER);
      const ref = await layerData(REF_LAYER);
      expect(ref.colors).not.toBeNull();
      expect(Array.from(ref.colors as Uint8Array)).not.toEqual(
        Array.from(target.colors as Uint8Array)
      );
    });
  });
});
