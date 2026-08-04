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

/**
 * Build the raw key→bytes entries for a single mesh node at `/mesh`.
 *
 * Extracted from `buildStore` so both the synchronous `RecordingStore` and the
 * read-gating `GatingStore` can be built from the SAME store contents — the two
 * differ only in when a chunk read resolves, never in what is on disk.
 */
function buildEntries(
  attrs: MeshMetadata,
  arrays: Record<string, ArraySpec>
): Map<string, Uint8Array> {
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
  return entries;
}

/** Build a store holding a single mesh node at `/mesh`. */
function buildStore(attrs: MeshMetadata, arrays: Record<string, ArraySpec>): RecordingStore {
  return new RecordingStore(buildEntries(attrs, arrays));
}

// ---------------------------------------------------------------------------
// A store that can HOLD chunk reads, to open a window between two fetches
// ---------------------------------------------------------------------------

/**
 * Structurally a `zarr.Readable`, like {@link RecordingStore}, but it does not
 * resolve chunk reads synchronously. Metadata (`.zarray`/`.zattrs`/…) resolves
 * immediately so a load can walk the tree, but every CHUNK read is parked and
 * only resolves on an explicit `releaseOldest` / `releaseAll`.
 *
 * That control is what lets a test complete fetch A while fetch B is still in
 * flight — impossible with the synchronous `RecordingStore`, whose reads all
 * settle on the next microtask drain. It is the only way to reproduce the
 * dispose+reload latch race, where a stale fetch must settle in the gap after a
 * replacement fetch has already started.
 */
class GatingStore {
  /** Every key `get` was called with, in order. */
  readonly requested: string[] = [];

  /** Chunk reads awaiting release, oldest first. */
  private readonly held: { key: string; resolve: (v: Uint8Array | undefined) => void }[] = [];

  private static readonly META = new Set([
    '.zarray',
    '.zattrs',
    '.zgroup',
    '.zmetadata',
    'zarr.json',
  ]);

  constructor(private readonly entries: Map<string, Uint8Array>) {}

  private static isMeta(key: string): boolean {
    return GatingStore.META.has(key.slice(key.lastIndexOf('/') + 1));
  }

  get(key: string): Promise<Uint8Array | undefined> {
    this.requested.push(key);
    // Metadata resolves eagerly so the loader can open arrays; chunk reads park.
    if (GatingStore.isMeta(key)) return Promise.resolve(this.entries.get(key));
    return new Promise<Uint8Array | undefined>((resolve) => {
      this.held.push({ key, resolve });
    });
  }

  /** Resolve the oldest held read whose key matches, with its stored bytes. */
  releaseOldest(key: string): void {
    const idx = this.held.findIndex((h) => h.key === key);
    if (idx === -1) return;
    const [h] = this.held.splice(idx, 1);
    h.resolve(this.entries.get(h.key));
  }

  /** Resolve every held read, so no awaiting promise hangs at teardown. */
  releaseAll(): void {
    const pending = this.held.splice(0);
    for (const h of pending) h.resolve(this.entries.get(h.key));
  }

  /** Non-metadata (chunk) requests — see {@link RecordingStore.chunkRequests}. */
  chunkRequests(): string[] {
    return this.requested.filter((k) => !GatingStore.isMeta(k));
  }
}

/** Build a read-gating store holding a single mesh node at `/mesh`. */
function buildGatingStore(attrs: MeshMetadata, arrays: Record<string, ArraySpec>): GatingStore {
  return new GatingStore(buildEntries(attrs, arrays));
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

function makeGatingLoader(store: GatingStore, attrs: MeshMetadata): MeshLoader {
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

  it('loads a BROADCAST uniform colour end to end', async () => {
    // The regression the review caught, exercised through the real loader rather
    // than hand-built handles: `add_mesh(..., colors=(1, 0, 0))` writes ONE stored
    // row with `n_elements: V` and no `original_shape`. Stage 1 has to accept it and
    // the decoder has to expand it, or a first-class API produces a store the viewer
    // refuses.
    const attrs = meshAttrs({ has_colors: true });
    const store = buildStore(
      attrs,
      tetArrays({
        colors: {
          shape: [1, 3],
          dtype: '<f4',
          data: [1, 0, 0],
          attrs: { encoding: { name: 'broadcasted', n_elements: 4, original_dtype: 'float32' } },
        },
      })
    );
    const data = await makeLoader(store, attrs).loadMesh(VIEW);
    expect(data.colorComponents).toBe(3);
    // Expanded to one entry per vertex, all the same colour.
    expect(data.colors!.length).toBe(4 * 3);
    expect(Array.from(data.colors!)).toEqual([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]);
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

  it('does not repopulate its cache from a fetch that settles after dispose', async () => {
    // Without the generation token the in-flight completion writes `this.data` back
    // onto a torn-down loader, pinning a whole mesh nothing will ever read. The
    // awaiting caller still gets its data — view-independent, so not wrong — but the
    // loader must not retain it.
    const store = buildStore(meshAttrs(), tetArrays());
    const loader = makeLoader(store, meshAttrs());
    const inFlight = loader.loadMesh(VIEW);
    loader.dispose();
    await expect(inFlight).resolves.toBeDefined();

    // If the cache had been repopulated, this second load would be served from it
    // and issue no further chunk reads.
    const before = store.chunkRequests().length;
    await loader.loadMesh(VIEW);
    expect(store.chunkRequests().length).toBeGreaterThan(before);
  });

  it("a stale fetch settling after dispose+reload does not erase the replacement load's latch", async () => {
    // The `inFlight` latch is shared across concurrent callers, and its
    // `.finally()` must clear it only if it still points at ITS OWN fetch. The
    // dangerous interleaving is a dispose+reload straddling a slow fetch:
    //
    //   1. load A starts               → inFlight = A
    //   2. dispose()                   → generation bumped, latch cleared
    //   3. load B starts               → inFlight = B
    //   4. A settles (stale gen)       → A's .finally() runs
    //
    // An UNCONDITIONAL `this.inFlight = null` at step 4 erases B's latch even
    // though A is not the current fetch. The loader is then left with a pending
    // fetch (B) and NO latch, so the next `updateView` — seeing data === null and
    // inFlight === null — starts a THIRD concurrent whole-mesh fetch, the exact
    // duplicate-work the whole-node design exists to prevent. The identity guard
    // (`if (this.inFlight === pending)`) makes step 4 a no-op, leaving B latched.
    //
    // A gating store is required: only by parking A's reads can A be made to
    // settle in the window after B has already started.
    const store = buildGatingStore(meshAttrs(), tetArrays());
    const loader = makeGatingLoader(store, meshAttrs());
    // setTimeout(0) drains all pending microtasks WITHOUT resolving held reads —
    // those resolve only on an explicit release below.
    const flush = () => new Promise((r) => setTimeout(r, 0));

    // 1. A starts and parks at its held vertices-chunk read.
    const original = loader.loadMesh(VIEW);
    await flush();

    // 2. Dispose bumps the generation and clears the latch.
    loader.dispose();

    // 3. B starts and parks at its own held vertices-chunk read; inFlight = B.
    const replacement = loader.loadMesh(VIEW);
    await flush();

    // 4. Complete ONLY A: release its vertices chunk (it then requests faces),
    //    then its faces chunk, so A's fetch resolves and its `.finally()` runs
    //    while B is still parked.
    store.releaseOldest('/mesh/vertices/0.0');
    await flush();
    store.releaseOldest('/mesh/faces/0.0');
    await flush();

    // 5. Inject an updateView. On the fix it joins the still-pending replacement
    //    load (latch intact) and issues no new chunk read; on the bug the erased
    //    latch lets it start a third whole-mesh fetch.
    const before = store.chunkRequests().length;
    const injected = loader.updateView(VIEW);
    await flush();
    expect(store.chunkRequests().length).toBe(before);

    // Cleanup: drain every parked read so no awaited promise hangs at teardown.
    // Releasing a read lets its loader request the NEXT chunk, which re-parks, so
    // pump release+flush a few times until all fetches have settled.
    const settled = Promise.allSettled([original, replacement, injected]);
    for (let i = 0; i < 6; i++) {
      store.releaseAll();
      await flush();
    }
    await settled;
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
      /account for .* over the/
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
      /vertices describes/
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
      /colors describes/
    );
  });

  it('has_normals set with no normals array in the store', async () => {
    const attrs = meshAttrs({ has_normals: true, normal_dims: [0, 1, 2] });
    await expectRejectedWithoutFetching(
      buildStore(attrs, tetArrays()),
      attrs,
      /has_normals is set but its array\(s\) are missing/
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
