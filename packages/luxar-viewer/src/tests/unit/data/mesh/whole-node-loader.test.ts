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
import { MeshWholeNodeLoader } from '../../../../data/mesh/mesh-whole-node-loader';
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
/**
 * Zarr metadata filenames; anything else under an array path is a chunk fetch.
 *
 * `zarr.json` is included because zarrita probes for zarr **v3** metadata before
 * falling back to v2, so a v2 store legitimately sees one request per node opened.
 */
const META_KEYS = new Set(['.zarray', '.zattrs', '.zgroup', '.zmetadata', 'zarr.json']);

class RecordingStore {
  /** Every key `get` was called with, in order. */
  readonly requested: string[] = [];

  constructor(private readonly entries: Map<string, Uint8Array>) {}

  /**
   * Per-ordinal control over `faces` CHUNK reads, so a test can order two
   * concurrent loads deterministically instead of hoping microtask timing
   * cooperates. `failFacesRead` rejects the nth such read; `parkFacesRead` returns a
   * promise that never settles, holding that load in flight indefinitely.
   */
  failFacesRead: number | null = null;
  parkFacesRead: number | null = null;
  private facesReads = 0;

  /** Reject any `get` for a key containing this substring, with this error. */
  rejectKeyContaining: { needle: string; error: Error } | null = null;

  get(key: string): Promise<Uint8Array | undefined> {
    this.requested.push(key);
    // A zarr v2 chunk key's last segment is DOT-SEPARATED coords (`0.0`), so "has no
    // dot" does not identify one — the metadata allowlist is the reliable
    // discriminator, exactly as `chunkRequests()` uses below.
    const base = key.slice(key.lastIndexOf('/') + 1);
    const isFacesChunk = key.includes('/faces/') && !META_KEYS.has(base);
    if (isFacesChunk) {
      this.facesReads += 1;
      if (this.facesReads === this.failFacesRead) {
        return Promise.reject(new Error('simulated read failure'));
      }
      if (this.facesReads === this.parkFacesRead) {
        return new Promise(() => {}); // never settles
      }
    }
    if (this.rejectKeyContaining && key.includes(this.rejectKeyContaining.needle)) {
      return Promise.reject(this.rejectKeyContaining.error);
    }
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
    return this.requested.filter((k) => !META_KEYS.has(k.slice(k.lastIndexOf('/') + 1)));
  }

  /**
   * Add an array at an arbitrary path — used for `array_ref` targets, which live
   * OUTSIDE the mesh node (a deduplicated array is shared across nodes).
   */
  addArray(arrayPath: string, spec: ArraySpec): void {
    writeArray(this.entries, arrayPath, spec);
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
 * Write one array's metadata (+ optional single raw chunk) into an entry map.
 *
 * Shared by `buildStore` and {@link RecordingStore.addArray} so an `array_ref` target
 * outside the mesh node is built exactly like a node-local array.
 */
function writeArray(entries: Map<string, Uint8Array>, arrayPath: string, spec: ArraySpec): void {
  const enc = (o: unknown) => new TextEncoder().encode(JSON.stringify(o));
  const chunks = spec.chunks ?? spec.shape;
  entries.set(
    `${arrayPath}/.zarray`,
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
  entries.set(`${arrayPath}/.zattrs`, enc(spec.attrs ?? {}));
  if (spec.data) {
    const Ctor = TYPED[spec.dtype];
    // Chunks are allocated at the full chunk shape, edge chunks padded, so the
    // buffer must be chunk-sized even when the data is shorter.
    const chunkElems = chunks.reduce((a, b) => a * b, 1);
    const buf = new Ctor(chunkElems) as unknown as { set(v: unknown, o: number): void };
    buf.set(spec.data as never, 0);
    const view = buf as unknown as ArrayBufferView;
    entries.set(
      `${arrayPath}/${chunks.map(() => 0).join('.')}`,
      new Uint8Array(view.buffer, view.byteOffset, view.byteLength).slice()
    );
  }
}

/** Build a store holding a single mesh node at `/mesh`. */
function buildStore(attrs: MeshMetadata, arrays: Record<string, ArraySpec>): RecordingStore {
  const entries = new Map<string, Uint8Array>();
  const enc = (o: unknown) => new TextEncoder().encode(JSON.stringify(o));

  entries.set('/.zgroup', enc({ zarr_format: 2 }));
  entries.set('/mesh/.zgroup', enc({ zarr_format: 2 }));
  entries.set('/mesh/.zattrs', enc(attrs));

  for (const [name, spec] of Object.entries(arrays)) {
    writeArray(entries, `/mesh/${name}`, spec);
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

function makeLoader(store: RecordingStore, attrs: MeshMetadata): MeshWholeNodeLoader {
  return new MeshWholeNodeLoader('/mesh', attrs, zarr.root(store).resolve('mesh'), {
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

describe('MeshWholeNodeLoader — the happy path', () => {
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

  it('allocates the reusable projection buffer, sized for the node', async () => {
    // The loader owns this buffer because `updateView` hands back this same object for
    // the node's whole life and drops it on dispose — so it inherits exactly the right
    // lifetime with no cache to invalidate. Without it `projectMeshTo3D` allocates a fresh
    // `vertexCount * 3` array on every slice move and the geometry re-uploads the whole
    // vertex buffer each time (#1245). Mirrors the Points accumulator's target buffers.
    const store = buildStore(meshAttrs(), tetArrays());
    const data = await makeLoader(store, meshAttrs()).loadMesh(VIEW);

    expect(data.projection).toBeDefined();
    expect(data.projection!.position).toHaveLength(4 * 3);
    // Nothing extracted yet, so no epoch is recorded.
    expect(data.projection!.displayDimsKey).toBeNull();
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

describe('MeshWholeNodeLoader — whole-node residency', () => {
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

  it('a stale completion must not erase the REPLACEMENT load\u2019s in-flight latch', async () => {
    // The race #1240 named. `dispose()` nulls `inFlight` while the old promise may
    // still be pending; if that promise's cleanup clears the latch unconditionally it
    // wipes the NEW load's latch when it settles — and from then until the new fetch
    // publishes, every updateView (a slice scrub, precisely what the latch exists
    // for) starts another whole-mesh fetch.
    //
    // Driven by ORDINAL rather than by timing, because microtask ordering will not
    // reliably keep the replacement in flight past the original's completion: read #1
    // of `faces` fails fast (settling the original), and read #2 parks forever
    // (holding the replacement in flight). Promise IDENTITY is then the observable —
    // a third call must JOIN the replacement rather than start its own fetch.
    const store = buildStore(meshAttrs(), tetArrays());
    store.failFacesRead = 1;
    store.parkFacesRead = 2;
    const loader = makeLoader(store, meshAttrs());

    const first = loader.loadMesh(VIEW);
    loader.dispose(); // mid-flight, before `first` settles
    const replacement = loader.loadMesh(VIEW);

    await expect(first).rejects.toThrow(); // the stale completion runs its cleanup
    const third = loader.updateView(VIEW);

    // Same promise: the latch survived the stale settle. With an unconditional
    // clear, `third` is a brand-new fetch instead.
    expect(third).toBe(replacement);
    // And no third `faces` read was issued.
    expect(store.requested.filter((k) => k.endsWith('/faces/0.0'))).toHaveLength(2);
    void replacement.catch(() => {}); // parked forever; keep it from surfacing
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

describe('MeshWholeNodeLoader — Stage 1 rejects BEFORE any chunk is fetched', () => {
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

  it('classifies a TRANSIENT open failure as Network, not Validation', async () => {
    // A hardcoded 'Validation' would have the failure record treat a flaky network as
    // deterministic and never retry it — the kind is persisted precisely so retry
    // policy can tell the two apart. Matches how the sibling node loaders route raw
    // errors through `classifyLoaderError`.
    const attrs = meshAttrs();
    const store = buildStore(attrs, tetArrays());
    store.rejectKeyContaining = {
      needle: '/faces/',
      error: new Error('network request failed'),
    };
    let thrown: unknown;
    try {
      await makeLoader(store, attrs).loadMesh(VIEW);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(LoaderError);
    expect((thrown as LoaderError).kind).toBe('Network');
  });

  it('an array_ref TARGET is budgeted, not the (0, k) stub that points at it', async () => {
    // #1247. `ArrayDecoder` resolves `encoding.target` against the store root and
    // reads THAT array in full, but the referring array is a stub the Python encoder
    // writes at `(0, k)` — so budgeting the handle the loader opened charges ~48 bytes
    // for a read that pulls gigabytes. Reproduced in the issue as a 44-second
    // materialization of a ~3 GB fill-value chunk: exactly the exhaustion this gate
    // exists to refuse, and the case its own docstring calls "too late after decode".
    const huge = 268_435_456;
    const attrs = meshAttrs();
    const store = buildStore(attrs, {
      vertices: {
        shape: [0, 3],
        dtype: '<f4',
        attrs: {
          encoding: { name: 'array_ref', target: 'evil', original_shape: [4, 3] },
        },
      },
      faces: { shape: [4, 3], dtype: '<u4', data: TET_FACES },
    });
    // The target lives outside the mesh node, as a deduplicated array really does.
    store.addArray('/evil', { shape: [huge, 3], chunks: [huge, 3], dtype: '<f4' });

    await expect(makeLoader(store, attrs).loadMesh(VIEW)).rejects.toThrow(
      /over the .* per-node budget/
    );
    // The whole point: the target's chunk was never requested.
    expect(store.chunkRequests()).toEqual([]);
  });

  it('keeps a transient failure opening an OPTIONAL array retryable (#1254)', async () => {
    // The asymmetry this closes: the required `vertices`/`faces` open classifies its
    // cause, but the optional opens swallowed everything — so a network blip on
    // `colors` became a flag-with-no-array `Validation` rejection, which the failure
    // record treats as deterministic and never retries.
    const attrs = meshAttrs({ has_colors: true });
    const store = buildStore(attrs, tetArrays());
    // No `/mesh/colors` in the store at all; the open fails with a network-shaped error.
    store.rejectKeyContaining = {
      needle: 'colors',
      error: new Error('fetch failed: network unreachable'),
    };

    let thrown: unknown;
    try {
      await makeLoader(store, attrs).loadMesh(VIEW);
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toBeInstanceOf(LoaderError);
    expect((thrown as LoaderError).kind).toBe('Network');
  });

  it('still reports a lying presence flag as deterministic Validation', async () => {
    // The other half: a genuinely absent array must NOT become retryable. The slot
    // stays empty and the preflight's flag-with-no-array message is what surfaces.
    const attrs = meshAttrs({ has_colors: true });
    const store = buildStore(attrs, tetArrays()); // has_colors, but no colors array

    let thrown: unknown;
    try {
      await makeLoader(store, attrs).loadMesh(VIEW);
    } catch (error) {
      thrown = error;
    }
    expect(thrown).toBeInstanceOf(LoaderError);
    expect((thrown as LoaderError).kind).toBe('Validation');
  });

  it('budgets an array_ref target that is itself broadcast-encoded (#1253)', async () => {
    // An ACCOUNTING hole, not (today) a live exhaustion path — the distinction is worth
    // stating precisely, because the fix is justified either way but the severity is not.
    //
    // `ArrayDecoder.decodeArrayRef` recursively decodes the target with the TARGET's own
    // attrs, so a `broadcasted` target declares ITS `n_elements` however modest the
    // stub's `original_shape` is. Both the stored and per-chunk terms are tiny here (the
    // target stores ONE row) and `broadcasted` is a known-budgetable encoding, so
    // endpoint-only accounting admitted the node — verified by mutation: dropping the
    // chain walk makes this load RESOLVE.
    //
    // What stops it becoming a 540 MB allocation (45M x 3 x 4 bytes) is that the
    // broadcast branch reads `expectedElements ?? enc.n_elements`, and every mesh decode
    // site passes an expectation derived from `n_vertices`. So the target's number is
    // currently ignored at decode time, and Stage 2's length check would object after.
    // That makes this defence-in-depth plus a truthfulness fix: the gate's own docstring
    // promises the budget covers "what they decode to", and it did not. It also removes
    // the standing trap that `expectedElements` is an OPTIONAL parameter — a future
    // decode site that omits it would make the hole load-bearing.
    const attrs = meshAttrs({ has_colors: true });
    const store = buildStore(attrs, {
      ...tetArrays(),
      colors: {
        shape: [0, 3],
        dtype: '<f4',
        attrs: { encoding: { name: 'array_ref', target: 'uniform', original_shape: [4, 3] } },
      },
    });
    store.addArray('/uniform', {
      shape: [1, 3],
      chunks: [1, 3],
      dtype: '<f4',
      data: [1, 0, 0],
      attrs: { encoding: { name: 'broadcasted', n_elements: 45_000_000 } },
    });

    await expect(makeLoader(store, attrs).loadMesh(VIEW)).rejects.toThrow(
      /over the .* per-node budget/
    );
    // And refused from metadata alone: not one chunk of the 540 MB was read.
    expect(store.chunkRequests()).toEqual([]);
  });

  it('still admits an array_ref to a broadcast target that FITS', async () => {
    // The acceptance half — otherwise the check above could be "reject every
    // broadcast target". A uniform colour shared by dedup across nodes is real writer
    // output, and `add_mesh(..., colors=(1, 0, 0))` is a first-class API.
    const attrs = meshAttrs({ has_colors: true });
    const store = buildStore(attrs, {
      ...tetArrays(),
      colors: {
        shape: [0, 3],
        dtype: '<f4',
        attrs: { encoding: { name: 'array_ref', target: 'uniform', original_shape: [4, 3] } },
      },
    });
    store.addArray('/uniform', {
      shape: [1, 3],
      chunks: [1, 3],
      dtype: '<f4',
      data: [1, 0, 0],
      attrs: { encoding: { name: 'broadcasted', n_elements: 4 } },
    });

    const data = await makeLoader(store, attrs).loadMesh(VIEW);
    expect(data.colors).not.toBeNull();
    expect(data.colorComponents).toBe(3);
  });

  it('follows a legitimate array_ref — dedup is a real writer behaviour', async () => {
    // Rejecting array_ref outright is not free: `normals`/`colors`/`scalars` go
    // through write_colors/write_scalars with dedup ON, so two meshes sharing a colour
    // array legitimately produce a ref the viewer must still load.
    const attrs = meshAttrs({ has_colors: true });
    const store = buildStore(attrs, {
      ...tetArrays(),
      colors: {
        shape: [0, 3],
        dtype: '<f4',
        attrs: {
          encoding: { name: 'array_ref', target: 'shared_colors', original_shape: [4, 3] },
        },
      },
    });
    store.addArray('/shared_colors', {
      shape: [4, 3],
      dtype: '<f4',
      data: [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    });

    const data = await makeLoader(store, attrs).loadMesh(VIEW);
    expect(data.colors!.length).toBe(12);
    expect(Array.from(data.colors!.slice(0, 3))).toEqual([1, 0, 0]);
  });

  it('refuses a CYCLIC array_ref chain instead of hanging', async () => {
    const attrs = meshAttrs();
    const store = buildStore(attrs, {
      vertices: {
        shape: [0, 3],
        dtype: '<f4',
        attrs: { encoding: { name: 'array_ref', target: 'loop_a', original_shape: [4, 3] } },
      },
      faces: { shape: [4, 3], dtype: '<u4', data: TET_FACES },
    });
    store.addArray('/loop_a', {
      shape: [0, 3],
      dtype: '<f4',
      attrs: { encoding: { name: 'array_ref', target: 'loop_a', original_shape: [4, 3] } },
    });

    await expect(makeLoader(store, attrs).loadMesh(VIEW)).rejects.toThrow(/cyclic/);
    expect(store.chunkRequests()).toEqual([]);
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

describe('MeshWholeNodeLoader — Stage 2 rejects on the materialized values', () => {
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
