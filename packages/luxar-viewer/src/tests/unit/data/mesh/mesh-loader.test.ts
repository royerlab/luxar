/**
 * The whole-node mesh loader, against an in-memory zarr store that RECORDS every
 * key it is asked for.
 *
 * That recording is the point of this file. The central claim of the two-stage
 * admission gate is that Stage 1 rejects a hostile declaration *before any chunk
 * is fetched* — and "before any chunk is fetched" is not something the preflight
 * can prove about itself. Only a store that reports what was requested can, so
 * these tests assert on the request log directly rather than on a proxy for it.
 *
 * The store is hand-built (raw, uncompressed zarr v2) rather than a Python
 * fixture, because most of what needs testing here is stores the writer would
 * never produce: signed indices, over-budget declarations, lying presence flags.
 */

import { describe, it, expect } from 'vitest';
import * as zarr from '../../../../data/zarr';
import { ArrayRefRegistry } from '../../../../data/array-decoder/decoder';
import { MeshLoader } from '../../../../data/mesh/mesh-loader';
import { LoaderError } from '../../../../data/scene-loader/nodes/load-leaf-error-dispatch';
import { MESH_DECODE_BUDGET_BYTES } from '../../../../config/constants';
import type { MeshMetadata, MeshViewState } from '../../../../types/mesh';

// ---------------------------------------------------------------------------
// A minimal in-memory zarr v2 store with a request log
// ---------------------------------------------------------------------------

/** Declaration for one array in the synthetic store. */
interface ArraySpec {
  shape: number[];
  chunks?: number[];
  /** numpy typestring, e.g. `'<f4'`, `'|u1'`, `'<i8'` */
  dtype: string;
  /** Values, written as ONE raw chunk. Omit to declare an array with no data. */
  data?: ArrayLike<number> | ArrayLike<bigint>;
  attrs?: Record<string, unknown>;
}

// Structurally satisfies `zarr.Readable` (a `get(key)` returning the bytes or
// undefined) without an `implements` clause: that alias resolves to a type
// without statically known members, which `implements` rejects (TS2422). The
// loader takes a `zarr.Readable` parameter, so assignability is checked there —
// where it actually matters — rather than asserted here.
class RecordingStore {
  /** Every key `get` was called with, in order. */
  readonly requested: string[] = [];

  constructor(private readonly entries: Map<string, Uint8Array>) {}

  get(key: string): Promise<Uint8Array | undefined> {
    this.requested.push(key);
    return Promise.resolve(this.entries.get(key));
  }

  /**
   * Keys that are NOT zarr metadata — i.e. chunk fetches.
   *
   * Deliberately a NEGATIVE classification: anything whose basename is not a
   * known metadata filename counts as a chunk. Matching chunk keys positively
   * (`/\d+(\.\d+)*$/`) would be tidier but could silently fail to notice a fetch
   * spelled some other way, and a "no chunks were fetched" assertion that can
   * miss a fetch is worth nothing.
   *
   * `zarr.json` is on the list because zarrita probes for zarr **v3** metadata
   * before falling back to v2 — so a v2 store legitimately sees a `zarr.json`
   * request per node it opens, with no chunk behind it.
   */
  chunkRequests(): string[] {
    const META = new Set(['.zarray', '.zattrs', '.zgroup', '.zmetadata', 'zarr.json']);
    return this.requested.filter((k) => !META.has(k.slice(k.lastIndexOf('/') + 1)));
  }
}

/**
 * Typed-array constructors, keyed by numpy typestring.
 *
 * The signature takes a LENGTH, not the values: chunks are allocated at the full
 * chunk shape (edge chunks padded), so the buffer is sized from `chunks` and the
 * data is `set` into it afterwards.
 */
const TYPED: Record<string, new (length: number) => ArrayBufferView> = {
  '|u1': Uint8Array as never,
  '|i1': Int8Array as never,
  '<u2': Uint16Array as never,
  '<i2': Int16Array as never,
  '<u4': Uint32Array as never,
  '<i4': Int32Array as never,
  '<u8': BigUint64Array as never,
  '<i8': BigInt64Array as never,
  '<f4': Float32Array as never,
  '<f8': Float64Array as never,
};

/** Build a store holding a single mesh node at `/mesh`. */
function buildStore(attrs: MeshMetadata, arrays: Record<string, ArraySpec>): RecordingStore {
  const entries = new Map<string, Uint8Array>();
  const enc = (o: unknown) => new TextEncoder().encode(JSON.stringify(o));

  entries.set('/.zgroup', enc({ zarr_format: 2 }));
  entries.set('/mesh/.zgroup', enc({ zarr_format: 2 }));
  entries.set('/mesh/.zattrs', enc(attrs));

  for (const [name, spec] of Object.entries(arrays)) {
    const chunks = spec.chunks ?? spec.shape;
    entries.set(
      `/mesh/${name}/.zarray`,
      enc({
        zarr_format: 2,
        shape: spec.shape,
        chunks,
        dtype: spec.dtype,
        compressor: null,
        filters: null,
        fill_value: 0,
        order: 'C',
      })
    );
    entries.set(`/mesh/${name}/.zattrs`, enc(spec.attrs ?? {}));
    if (spec.data) {
      const Ctor = TYPED[spec.dtype];
      // Chunks are allocated at the full chunk shape, edge chunks padded, so the
      // buffer must be chunk-sized even when the data is shorter.
      const chunkElems = chunks.reduce((a, b) => a * b, 1);
      const buf = new Ctor(chunkElems) as unknown as { set(v: unknown, o: number): void };
      buf.set(spec.data as never, 0);
      const view = buf as unknown as ArrayBufferView;
      const key = `/mesh/${name}/${chunks.map(() => 0).join('.')}`;
      entries.set(key, new Uint8Array(view.buffer, view.byteOffset, view.byteLength).slice());
    }
  }
  return new RecordingStore(entries);
}

function meshAttrs(overrides: Partial<MeshMetadata> = {}): MeshMetadata {
  return {
    type: 'mesh',
    n_vertices: 4,
    n_faces: 4,
    ndim: 3,
    has_normals: false,
    has_colors: false,
    has_scalars: false,
    shading: 'flat',
    double_sided: true,
    ordering: 'none',
    ...overrides,
  };
}

/** A welded tetrahedron: 4 vertices, 4 faces, every vertex shared by 3 faces. */
const TET_VERTICES = [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10];
const TET_FACES = [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3];

function tetArrays(overrides: Record<string, ArraySpec> = {}): Record<string, ArraySpec> {
  return {
    vertices: { shape: [4, 3], dtype: '<f4', data: TET_VERTICES },
    faces: { shape: [4, 3], dtype: '<u4', data: TET_FACES },
    ...overrides,
  };
}

function makeLoader(store: RecordingStore, attrs: MeshMetadata): MeshLoader {
  return new MeshLoader('/mesh', attrs, zarr.root(store).resolve('mesh'), {
    zarrStore: store,
    arrayRefRegistry: new ArrayRefRegistry(),
  });
}

const VIEW: MeshViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0],
  tolerance: [1e10, 1e10, 1e10],
} as MeshViewState;

// ---------------------------------------------------------------------------

describe('MeshLoader — the happy path', () => {
  it('loads a tetrahedron whole, in float32 / uint32', async () => {
    const store = buildStore(meshAttrs(), tetArrays());
    const data = await makeLoader(store, meshAttrs()).loadMesh(VIEW);

    expect(data.vertexCount).toBe(4);
    expect(data.faceCount).toBe(4);
    expect(data.ndim).toBe(3);
    expect(Array.from(data.vertices)).toEqual(TET_VERTICES);
    expect(data.faces).toBeInstanceOf(Uint32Array);
    expect(Array.from(data.faces)).toEqual(TET_FACES);
    expect(data.normals).toBeNull();
    expect(data.colors).toBeNull();
    expect(data.scalars).toBeUndefined();
  });

  it('widens a narrowed faces dtype, which is what the writer actually emits', async () => {
    // The INDEX encoder picks the smallest unsigned dtype that fits, so a small
    // mesh's faces land as uint8 on disk (verified against a written store).
    const store = buildStore(
      meshAttrs(),
      tetArrays({ faces: { shape: [4, 3], dtype: '|u1', data: TET_FACES } })
    );
    const data = await makeLoader(store, meshAttrs()).loadMesh(VIEW);
    expect(data.faces).toBeInstanceOf(Uint32Array);
    expect(Array.from(data.faces)).toEqual(TET_FACES);
  });

  it('loads normals, colors and scalars when present', async () => {
    const attrs = meshAttrs({
      has_normals: true,
      normal_dims: [0, 1, 2],
      has_colors: true,
      has_scalars: true,
      shading: 'smooth',
    });
    const store = buildStore(
      attrs,
      tetArrays({
        normals: { shape: [4, 3], dtype: '<f4', data: new Array(12).fill(0).map((_, i) => i) },
        colors: { shape: [4, 3], dtype: '|u1', data: [255, 0, 0, 0, 255, 0, 0, 0, 255, 9, 9, 9] },
        scalars: { shape: [4], dtype: '<f4', data: [0, 0.25, 0.5, 1] },
      })
    );
    const data = await makeLoader(store, attrs).loadMesh(VIEW);

    expect(data.normals?.length).toBe(12);
    expect(data.colorComponents).toBe(3);
    // Native dtype preserved — the GPU normalizes uint8 to [0,1] for free, at a
    // third of float32's bytes. Widening here would defeat that.
    expect(data.colors).toBeInstanceOf(Uint8Array);
    expect(Array.from(data.colors!.slice(0, 3))).toEqual([255, 0, 0]);
    expect(Array.from(data.scalars!)).toEqual([0, 0.25, 0.5, 1]);
  });

  it('reports RGBA colours as 4 components', async () => {
    const attrs = meshAttrs({ has_colors: true });
    const store = buildStore(
      attrs,
      tetArrays({
        colors: { shape: [4, 4], dtype: '|u1', data: new Array(16).fill(128) },
      })
    );
    const data = await makeLoader(store, attrs).loadMesh(VIEW);
    expect(data.colorComponents).toBe(4);
    expect(data.colors!.length).toBe(16);
  });

  it('expands a BROADCAST uniform colour to one colour per vertex', async () => {
    // A uniform colour is stored broadcast: a single (1, d) row with the logical
    // vertex count recorded in `encoding.n_elements` (NOT `original_shape`), which
    // is exactly what the Python encoder writes. This goes through a REAL broadcast
    // store — the whole point is that admission AND decode both honour the
    // convention, expanding the one row to V copies.
    const attrs = meshAttrs({ has_colors: true });
    const store = buildStore(
      attrs,
      tetArrays({
        colors: {
          shape: [1, 3],
          dtype: '|u1',
          data: [255, 0, 0],
          attrs: { encoding: { name: 'broadcasted', n_elements: 4, original_dtype: 'uint8' } },
        },
      })
    );
    const data = await makeLoader(store, attrs).loadMesh(VIEW);
    expect(data.colorComponents).toBe(3);
    expect(data.colors!.length).toBe(4 * 3);
    // Native uint8 preserved through the broadcast decode — losing the
    // `original_dtype` restoration would widen to Float32 and render 255× too
    // bright (the GPU expects normalized uint8, not raw 0-255 floats).
    expect(data.colors).toBeInstanceOf(Uint8Array);
    // Every vertex is the same red — the broadcast row replicated across V.
    expect(Array.from(data.colors!)).toEqual([255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0]);
  });
});

describe('MeshLoader — whole-node residency', () => {
  it('fetches once and serves every later updateView from cache', async () => {
    const store = buildStore(meshAttrs(), tetArrays());
    const loader = makeLoader(store, meshAttrs());

    const first = await loader.loadMesh(VIEW);
    const chunksAfterFirst = store.chunkRequests().length;
    expect(chunksAfterFirst).toBeGreaterThan(0);

    const second = await loader.updateView({ ...VIEW, slicePosition: [5, 5, 5] });
    const third = await loader.updateView({ ...VIEW, displayDims: [2, 1, 0] });

    // Same object, not merely equal: a view change has no subset to fetch, so
    // re-fetching would be pure duplicate work.
    expect(second).toBe(first);
    expect(third).toBe(first);
    expect(store.chunkRequests().length).toBe(chunksAfterFirst);
  });

  it('collapses concurrent first loads onto ONE fetch', async () => {
    // The scene-loader legitimately calls updateView while an initial loadMesh is
    // still in flight (a slice scrub during load). Without the in-flight latch
    // each call starts its own full-mesh fetch.
    const store = buildStore(meshAttrs(), tetArrays());
    const loader = makeLoader(store, meshAttrs());

    const [a, b, c] = await Promise.all([
      loader.loadMesh(VIEW),
      loader.updateView(VIEW),
      loader.updateView(VIEW),
    ]);
    expect(b).toBe(a);
    expect(c).toBe(a);
    // Exactly one read per data array (vertices + faces), not three.
    expect(store.chunkRequests()).toEqual(['/mesh/vertices/0.0', '/mesh/faces/0.0']);
  });

  it('allows a fresh attempt after a failure rather than caching the rejection', async () => {
    const store = buildStore(meshAttrs(), {
      vertices: { shape: [4, 3], dtype: '<f4', data: TET_VERTICES },
      // Declares 4 faces but the attrs say 4 — shape is fine; the VALUES are not.
      faces: { shape: [4, 3], dtype: '<i4', data: [0, 1, 2, 0, 1, -1, 0, 2, 3, 1, 2, 3] },
    });
    const loader = makeLoader(store, meshAttrs());
    await expect(loader.loadMesh(VIEW)).rejects.toThrow(LoaderError);
    // A second attempt must re-run, not re-await the rejected promise forever.
    await expect(loader.loadMesh(VIEW)).rejects.toThrow(LoaderError);
    expect(store.chunkRequests().length).toBeGreaterThan(2);
  });

  it('drops its cached data on dispose', async () => {
    const store = buildStore(meshAttrs(), tetArrays());
    const loader = makeLoader(store, meshAttrs());
    await loader.loadMesh(VIEW);
    const before = store.chunkRequests().length;
    loader.dispose();
    await loader.loadMesh(VIEW);
    expect(store.chunkRequests().length).toBeGreaterThan(before);
  });
});

describe('MeshLoader — Stage 1 rejects BEFORE any chunk is fetched', () => {
  /** Assert the load fails and the store was never asked for a chunk. */
  async function expectRejectedWithoutFetching(
    store: RecordingStore,
    attrs: MeshMetadata,
    pattern: RegExp
  ): Promise<void> {
    const loader = makeLoader(store, attrs);
    await expect(loader.loadMesh(VIEW)).rejects.toThrow(pattern);
    // THE claim of the two-stage design. An oversized or malformed declaration
    // must cost no allocation, so not one chunk key may be requested — only the
    // `.zarray` / `.zattrs` metadata the preflight legitimately reads.
    expect(store.chunkRequests()).toEqual([]);
    expect(store.requested.length).toBeGreaterThan(0);
  }

  it('n_vertices over the cap', async () => {
    const n = 2 ** 30;
    const attrs = meshAttrs({ n_vertices: n });
    await expectRejectedWithoutFetching(
      buildStore(attrs, {
        vertices: { shape: [n, 3], dtype: '<f4', chunks: [1024, 3] },
        faces: { shape: [4, 3], dtype: '<u4' },
      }),
      attrs,
      /exceeds the maximum/
    );
  });

  it('a declared footprint over the byte budget', async () => {
    const n = 40_000_000;
    const attrs = meshAttrs({ n_vertices: n, ndim: 4 });
    await expectRejectedWithoutFetching(
      buildStore(attrs, {
        vertices: { shape: [n, 4], dtype: '<f4', chunks: [65536, 4] },
        faces: { shape: [4, 3], dtype: '<u4' },
      }),
      attrs,
      /over the .* per-node budget/
    );
  });

  it('a single chunk over the byte budget, on an otherwise tiny array', async () => {
    const attrs = meshAttrs({ n_faces: 100 });
    await expectRejectedWithoutFetching(
      buildStore(attrs, {
        vertices: { shape: [4, 3], dtype: '<f4' },
        faces: { shape: [100, 3], dtype: '<u4', chunks: [268_435_456, 3] },
      }),
      attrs,
      /declares a single chunk of/
    );
    expect(268_435_456 * 3 * 4).toBeGreaterThan(MESH_DECODE_BUDGET_BYTES);
  });

  it('a vertices width that disagrees with ndim', async () => {
    const attrs = meshAttrs({ ndim: 4 });
    await expectRejectedWithoutFetching(
      buildStore(attrs, tetArrays()),
      attrs,
      /vertices declares shape/
    );
  });

  it('a float faces dtype', async () => {
    const attrs = meshAttrs();
    await expectRejectedWithoutFetching(
      buildStore(attrs, tetArrays({ faces: { shape: [4, 3], dtype: '<f4', data: TET_FACES } })),
      attrs,
      /an integer dtype is required/
    );
  });

  it('an undersized colors array', async () => {
    const attrs = meshAttrs({ has_colors: true });
    await expectRejectedWithoutFetching(
      buildStore(attrs, tetArrays({ colors: { shape: [2, 3], dtype: '|u1' } })),
      attrs,
      /colors declares shape/
    );
  });

  it('has_normals set with no normals array in the store', async () => {
    const attrs = meshAttrs({ has_normals: true, normal_dims: [0, 1, 2] });
    await expectRejectedWithoutFetching(
      buildStore(attrs, tetArrays()),
      attrs,
      /has_normals is set but the array is missing/
    );
  });

  it('malformed normal_dims', async () => {
    const attrs = meshAttrs({ has_normals: true, normal_dims: [0, 1, 1] });
    await expectRejectedWithoutFetching(
      buildStore(attrs, tetArrays({ normals: { shape: [4, 3], dtype: '<f4' } })),
      attrs,
      /three DISTINCT dimensions/
    );
  });

  it('a missing faces array — a mesh without one is not a mesh', async () => {
    const attrs = meshAttrs();
    const store = buildStore(attrs, {
      vertices: { shape: [4, 3], dtype: '<f4', data: TET_VERTICES },
    });
    await expect(makeLoader(store, attrs).loadMesh(VIEW)).rejects.toThrow(LoaderError);
    expect(store.chunkRequests()).toEqual([]);
  });
});

describe('MeshLoader — Stage 2 rejects on the materialized values', () => {
  it('rejects a signed store’s -1 face index', async () => {
    const store = buildStore(meshAttrs(), {
      vertices: { shape: [4, 3], dtype: '<f4', data: TET_VERTICES },
      faces: { shape: [4, 3], dtype: '<i4', data: [0, 1, 2, 0, 1, -1, 0, 2, 3, 1, 2, 3] },
    });
    await expect(makeLoader(store, meshAttrs()).loadMesh(VIEW)).rejects.toThrow(/face index -1/);
  });

  it('rejects a 64-bit store’s 2^32 + 1 face index', async () => {
    const bad = BigInt(2 ** 32) + 1n;
    const store = buildStore(meshAttrs(), {
      vertices: { shape: [4, 3], dtype: '<f4', data: TET_VERTICES },
      faces: {
        shape: [4, 3],
        dtype: '<i8',
        data: [0n, 1n, 2n, 0n, 1n, 3n, 0n, 2n, 3n, 1n, 2n, bad],
      },
    });
    await expect(makeLoader(store, meshAttrs()).loadMesh(VIEW)).rejects.toThrow(
      /face index 4294967297/
    );
  });

  it('rejects an out-of-range index that Stage 1 could not see', async () => {
    // Shape and dtype are both correct here, so only a value check catches it.
    const store = buildStore(meshAttrs(), {
      vertices: { shape: [4, 3], dtype: '<f4', data: TET_VERTICES },
      faces: { shape: [4, 3], dtype: '<u4', data: [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 99] },
    });
    await expect(makeLoader(store, meshAttrs()).loadMesh(VIEW)).rejects.toThrow(
      /face index 99 .* \[0, 4\)/
    );
  });

  it('fails as a Validation-kind LoaderError, so retry treats it as deterministic', async () => {
    const store = buildStore(meshAttrs(), {
      vertices: { shape: [4, 3], dtype: '<f4', data: TET_VERTICES },
      faces: { shape: [4, 3], dtype: '<u4', data: [0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 99] },
    });
    let thrown: unknown;
    try {
      await makeLoader(store, meshAttrs()).loadMesh(VIEW);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(LoaderError);
    // A malformed store re-fetched on every reconnect would be pure noise; the
    // kind is what stops that.
    expect((thrown as LoaderError).kind).toBe('Validation');
    expect((thrown as LoaderError).path).toBe('/mesh');
  });
});
