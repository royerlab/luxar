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
import { preflightMesh } from '../../../../data/mesh/preflight';
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

  /**
   * The `AbortSignal` (if any) each request carried, keyed by request key.
   * zarrita forwards `GetOptions.signal` to `store.get(key, { signal })`, so
   * this is where "the abort signal reaches the reads" is actually observable.
   */
  readonly signals = new Map<string, AbortSignal | undefined>();

  constructor(private readonly entries: Map<string, Uint8Array>) {}

  /**
   * Per-ordinal control over `faces` CHUNK reads, so a test can order two
   * concurrent loads deterministically instead of hoping microtask timing
   * cooperates. `failFacesRead` rejects the nth such read; `parkFacesRead` returns a
   * promise that never settles, holding that load in flight indefinitely.
   */
  failFacesRead: number | null = null;
  parkFacesRead: number | null = null;
  /**
   * Set when a parked faces read is issued: calling it settles that read with
   * the real bytes, so a test can hold a load in flight past a `dispose()` and
   * then let it complete — the only way to exercise the publish guard
   * deterministically now that `dispose()` aborts reads still on the wire.
   */
  releaseParkedFacesRead: (() => void) | null = null;
  private facesReads = 0;

  /** Reject any `get` for a key containing this substring, with this error. */
  rejectKeyContaining: { needle: string; error: Error } | null = null;

  /**
   * Parks EVERY request for this exact key, queuing each one's settler in
   * request order — the `initialize()` counterpart of `parkFacesRead`,
   * generalized so more than one metadata open can be held in flight
   * SIMULTANEOUSLY. That is exactly what the `initInFlight` ownership race
   * needs: an ORIGINAL attempt still open when a REPLACEMENT attempt starts
   * its own, so the two can be released in either order — and, since the
   * race only matters when the STALE attempt does NOT publish `handles`
   * (only a successful `doInitialize()` does that), released as either a
   * success or a failure.
   */
  parkKeyAlways: string | null = null;
  /** Queued settlers for `parkKeyAlways` hits, oldest first. */
  readonly parkedKeyResolvers: Array<(error?: Error) => void> = [];

  /** Release the OLDEST still-pending `parkKeyAlways` request, if any. */
  releaseNextParkedKeyRequest(error?: Error): void {
    const settle = this.parkedKeyResolvers.shift();
    if (settle) settle(error);
  }

  get(key: string, opts?: { signal?: AbortSignal }): Promise<Uint8Array | undefined> {
    this.requested.push(key);
    this.signals.set(key, opts?.signal);
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
        // Parked until the test releases it (or forever, if it never does).
        return new Promise((resolve) => {
          this.releaseParkedFacesRead = () => resolve(this.entries.get(key));
        });
      }
    }
    if (this.parkKeyAlways !== null && key === this.parkKeyAlways) {
      return new Promise((resolve, reject) => {
        this.parkedKeyResolvers.push((error) => {
          if (error) reject(error);
          else resolve(this.entries.get(key));
        });
      });
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
    has_uvs: false,
    has_texture: false,
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
    // reliably keep the replacement in flight past the original's completion:
    // `dispose()` aborts the original's reads, so the original settles (with an
    // AbortError) before it ever reaches `faces`, and the replacement's own faces
    // read — the FIRST one issued — parks forever, holding it in flight. Promise
    // IDENTITY is then the observable — a third call must JOIN the replacement
    // rather than start its own fetch.
    const store = buildStore(meshAttrs(), tetArrays());
    store.parkFacesRead = 1;
    const loader = makeLoader(store, meshAttrs());

    const first = loader.loadMesh(VIEW);
    loader.dispose(); // mid-flight, before `first` settles — aborts its reads
    const replacement = loader.loadMesh(VIEW);

    await expect(first).rejects.toThrow(); // the stale completion runs its cleanup
    const third = loader.updateView(VIEW);

    // Same promise: the latch survived the stale settle. With an unconditional
    // clear, `third` is a brand-new fetch instead.
    expect(third).toBe(replacement);
    // Only the replacement ever reached `faces` (the aborted original settled
    // at its vertices read), and no third read was issued for it.
    expect(store.requested.filter((k) => k.endsWith('/faces/0.0'))).toHaveLength(1);
    void replacement.catch(() => {}); // parked forever; keep it from surfacing
  });

  it('does not repopulate its cache from a fetch that settles after dispose', async () => {
    // Without the generation token the in-flight completion writes `this.data` back
    // onto a torn-down loader, pinning a whole mesh nothing will ever read. The
    // awaiting caller still gets its data — view-independent, so not wrong — but the
    // loader must not retain it.
    //
    // `dispose()` also aborts reads still on the wire (pinned below), so reaching
    // the publish guard requires a read that is ISSUED before the dispose and
    // SETTLES after it: the faces read is parked, disposed over, then released.
    const store = buildStore(meshAttrs(), tetArrays());
    store.parkFacesRead = 1;
    const loader = makeLoader(store, meshAttrs());
    const inFlight = loader.loadMesh(VIEW);
    await new Promise<void>((resolve) => {
      const poll = (): void => {
        if (store.releaseParkedFacesRead) resolve();
        else setTimeout(poll, 0);
      };
      poll();
    });
    loader.dispose();
    store.releaseParkedFacesRead!();
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

  it('releases a progressive-parent-folded payload without resetting initialization', async () => {
    const store = buildStore(meshAttrs(), tetArrays());
    const loader = makeLoader(store, meshAttrs());
    const first = await loader.loadMesh(VIEW);
    const before = store.chunkRequests().length;

    loader.releaseData();

    expect(loader.getMetrics().memoryUsed).toBe(0);
    const second = await loader.loadMesh(VIEW);
    expect(store.chunkRequests().length).toBeGreaterThan(before);
    expect(second).not.toBe(first);
    expect(first.vertexCount).toBe(4);
  });

  it('every chunk read carries an abort signal, not just the faces one', async () => {
    // A signal that reaches only one of the arrays is cancellation theatre: the
    // decoder-routed reads (vertices, normals, scalars) and the shared colour
    // path are where the bytes are, so those are the reads that must be
    // stoppable. Observable at the store, because zarrita forwards
    // `GetOptions.signal` into `store.get`.
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
        normals: { shape: [4, 3], dtype: '<f4', data: new Array(12).fill(0) },
        colors: { shape: [4, 3], dtype: '|u1', data: new Array(12).fill(128) },
        scalars: { shape: [4], dtype: '<f4', data: [0, 0.25, 0.5, 1] },
      })
    );
    await makeLoader(store, attrs).loadMesh(VIEW);

    const chunkKeys = store.chunkRequests();
    // All five data arrays were actually read…
    for (const name of ['vertices', 'faces', 'normals', 'colors', 'scalars']) {
      expect(chunkKeys.some((k) => k.includes(`/${name}/`))).toBe(true);
    }
    // …and every one of those reads carried a signal.
    for (const key of chunkKeys) {
      expect(store.signals.get(key), `chunk read ${key} carried no abort signal`).toBeDefined();
    }
  });

  it('dispose() trips the signal governing the in-flight reads', async () => {
    // The generation token stops the post-dispose PUBLISH; this pins that the
    // TRANSFER is stopped too — a near-budget mesh otherwise fetches and decodes
    // half a gigabyte for a loader nothing will ever read. The fake store cannot
    // observe a mid-read cancellation (it ignores the signal), so the assertion
    // is on the signal the read was handed.
    const store = buildStore(meshAttrs(), tetArrays());
    store.parkFacesRead = 1;
    const loader = makeLoader(store, meshAttrs());
    const inFlight = loader.loadMesh(VIEW);

    // Wait until the parked faces read has been issued.
    await new Promise<void>((resolve) => {
      const poll = (): void => {
        if (store.requested.some((k) => k.endsWith('/faces/0.0'))) resolve();
        else setTimeout(poll, 0);
      };
      poll();
    });
    const signal = store.signals.get('/mesh/faces/0.0');
    expect(signal).toBeDefined();
    expect(signal!.aborted).toBe(false);

    loader.dispose();
    expect(signal!.aborted).toBe(true);
    void inFlight.catch(() => {}); // parked forever; keep it from surfacing
  });
});

describe('MeshWholeNodeLoader — runPreflight()', () => {
  // `runPreflight()` is `MeshProgressiveLoader.assertWithinByteBudget`'s door into
  // charging a mesh reveal ladder's levels against ONE byte budget (#1517) — before
  // this file, the only caller of `initialize()` was `fetch()`/`load()`. These pin
  // the two things that make it safe to add as a second public entry point: it
  // reports the exact number the ladder sums (not a re-derivation that could drift
  // from it), fetches no chunk data, and does not cost a later `loadMesh()` a
  // second metadata round trip.

  it('reports the same accountedBytes as calling preflightMesh directly on the same handles', async () => {
    const attrs = meshAttrs();
    const store = buildStore(attrs, tetArrays());
    const loc = zarr.root(store).resolve('mesh');
    const vertices = await zarr.open(loc.resolve('vertices'), { kind: 'array' });
    const faces = await zarr.open(loc.resolve('faces'), { kind: 'array' });
    const direct = await preflightMesh('/mesh', attrs, { vertices, faces }, zarr.root(store));

    const pre = await makeLoader(store, attrs).runPreflight();

    // The exact quantity `MeshProgressiveLoader.assertWithinByteBudget` sums
    // across every level — if `runPreflight()` ever computed this
    // independently instead of returning `preflightMesh`'s own result, the two
    // could silently drift apart.
    expect(pre.accountedBytes).toBe(direct.accountedBytes);
    expect(pre.nVertices).toBe(direct.nVertices);
    expect(pre.nFaces).toBe(direct.nFaces);
  });

  it('fetches metadata only — no chunk is read', async () => {
    const store = buildStore(meshAttrs(), tetArrays());
    const result = await makeLoader(store, meshAttrs()).runPreflight();

    expect(result.nVertices).toBe(4);
    expect(result.nFaces).toBe(4);
    // The same property `nofetch-invariant.test.ts` pins for the loadMesh() entry
    // point: only `.zarray`/`.zattrs` metadata, never a chunk.
    expect(store.chunkRequests()).toEqual([]);
    expect(store.requested.length).toBeGreaterThan(0);
  });

  it('costs loadMesh() no duplicate metadata read, and still returns correct data', async () => {
    const store = buildStore(meshAttrs(), tetArrays());
    const loader = makeLoader(store, meshAttrs());

    await loader.runPreflight();
    const metaRequestsAfterPreflight = store.requested.length;

    const data = await loader.loadMesh(VIEW);

    // Only the chunk reads are new — `initialize()` is idempotent, so `loadMesh()`
    // reuses the handles/preflight `runPreflight()` already opened rather than
    // re-opening `.zarray`/`.zattrs`.
    const metaRequestsAfterLoad = store.requested.filter((k) =>
      META_KEYS.has(k.slice(k.lastIndexOf('/') + 1))
    ).length;
    expect(metaRequestsAfterLoad).toBe(metaRequestsAfterPreflight);
    expect(store.chunkRequests().length).toBeGreaterThan(0);

    expect(data.vertexCount).toBe(4);
    expect(data.faceCount).toBe(4);
    expect(Array.from(data.vertices)).toEqual(TET_VERTICES);
    expect(Array.from(data.faces)).toEqual(TET_FACES);
  });

  it('a CONCURRENT runPreflight() + loadMesh() reads each metadata key exactly once', async () => {
    // Without a single-flight latch on `initialize()`, a concurrent call from each
    // entry point opens every `.zarray`/`.zattrs` TWICE and runs `preflightMesh`
    // twice — the exact "async initialization race" the repo's CLAUDE.md calls
    // out, now reachable because `runPreflight()` makes `initialize()` public.
    const store = buildStore(meshAttrs(), tetArrays());
    const loader = makeLoader(store, meshAttrs());

    const [pre, data] = await Promise.all([loader.runPreflight(), loader.loadMesh(VIEW)]);

    const metaKeys = store.requested.filter((k) => META_KEYS.has(k.slice(k.lastIndexOf('/') + 1)));
    const counts = new Map<string, number>();
    for (const key of metaKeys) counts.set(key, (counts.get(key) ?? 0) + 1);
    for (const [key, count] of counts) {
      expect(count, `metadata key ${key} was requested ${count} times, expected 1`).toBe(1);
    }
    expect(pre.nVertices).toBe(4);
    expect(data.vertexCount).toBe(4);
  });

  it('clears initInFlight on FAILURE too, so a transient error is retried rather than cached forever', async () => {
    // `initialize()`'s `try { await mine; } finally { if (this.initInFlight ===
    // mine) this.initInFlight = null; }` clears the latch on EVERY settle, not
    // just success. Without the `finally`, a failed first attempt's rejected
    // promise stays latched forever: `this.handles` never gets set (so the
    // `if (this.handles) return;` short-circuit never applies either), and
    // every later call replays the SAME rejection instead of genuinely
    // retrying — even once the underlying blip has cleared.
    const store = buildStore(meshAttrs(), tetArrays());
    store.rejectKeyContaining = { needle: 'vertices', error: new Error('network request failed') };
    const loader = makeLoader(store, meshAttrs());

    await expect(loader.runPreflight()).rejects.toThrow(LoaderError);

    // The blip clears.
    store.rejectKeyContaining = null;

    // A later call must re-attempt `initialize()` from scratch — not re-await
    // the cached rejected promise — and this time it succeeds.
    const pre = await loader.runPreflight();
    expect(pre.nVertices).toBe(4);
  });

  it('a stale metadata FAILURE must not erase a REPLACEMENT initialize()’s in-flight latch', async () => {
    // The `initInFlight` counterpart of the `load()`/`inFlight` race pinned
    // above ("a stale completion must not erase the REPLACEMENT load's
    // in-flight latch"), one layer up. Metadata opens carry no abort signal,
    // so `dispose()` mid-flight cannot make the ORIGINAL attempt settle — it
    // genuinely keeps running. The race needs its eventual settlement to be a
    // FAILURE (not a success): only a successful `doInitialize()` publishes
    // `handles`, and `handles` is what SHORT-CIRCUITS every later call before
    // `initInFlight` is ever consulted — so a successful stale completion
    // would mask the ownership question entirely. If the failing stale
    // attempt's `finally` clears `initInFlight` UNCONDITIONALLY rather than
    // only when it still owns the latch, it erases the REPLACEMENT attempt's
    // latch while THAT one is still genuinely pending — and a third caller
    // arriving in that window sees a falsely-empty latch and starts a
    // redundant FOURTH metadata round trip instead of joining the
    // replacement.
    const store = buildStore(meshAttrs(), tetArrays());
    store.parkKeyAlways = '/mesh/vertices/.zarray';
    const loader = makeLoader(store, meshAttrs());

    const first = loader.runPreflight(); // opens vertices — parks (queued #1)
    await new Promise<void>((resolve) => {
      const poll = (): void => {
        if (store.parkedKeyResolvers.length >= 1) resolve();
        else setTimeout(poll, 0);
      };
      poll();
    });

    loader.dispose(); // mid-flight — nulls `initInFlight`; the original open keeps running

    const replacement = loader.runPreflight(); // fresh initialize() attempt — parks again (#2)
    await new Promise<void>((resolve) => {
      const poll = (): void => {
        if (store.parkedKeyResolvers.length >= 2) resolve();
        else setTimeout(poll, 0);
      };
      poll();
    });

    // The ORIGINAL's parked open (queued first) FAILS while the
    // REPLACEMENT's own open is STILL parked — the exact ordering the
    // ownership guard exists for.
    store.releaseNextParkedKeyRequest(new Error('simulated transient network failure'));
    await expect(first).rejects.toThrow();

    // A third call made now must JOIN the still-in-flight replacement — no
    // NEW metadata round trip — rather than see a wrongly-nulled latch and
    // start one.
    const beforeThird = store.parkedKeyResolvers.length;
    const third = loader.runPreflight();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(store.parkedKeyResolvers.length).toBe(beforeThird);

    store.releaseNextParkedKeyRequest(); // the replacement's own open, successfully
    await expect(replacement).resolves.toBeDefined();
    await expect(third).resolves.toBeDefined();
  });
});

describe('MeshWholeNodeLoader — dispose() races doInitialize()', () => {
  it('does not resurrect handles/preflight — a later loadMesh() re-opens metadata and returns correct data', async () => {
    // `doInitialize()` used to publish `this.preflight` then `this.handles`
    // unconditionally, with no generation guard (unlike `load()`'s own publish).
    // `runPreflight()` makes this reachable with NO `load()` in flight at all —
    // the ladder's aggregate byte-budget gate (#1517) preflights every level,
    // including during a dataset switch — so a `dispose()` racing a bare
    // `runPreflight()` could leave the loader holding non-null `handles`, making
    // `dispose()`'s own "a subsequent loadMesh re-initializes" docstring false.
    const store = buildStore(meshAttrs(), tetArrays());
    store.parkKeyAlways = '/mesh/vertices/.zarray';
    const loader = makeLoader(store, meshAttrs());

    const stalePreflight = loader.runPreflight(); // opens vertices — parks
    await new Promise<void>((resolve) => {
      const poll = (): void => {
        if (store.parkedKeyResolvers.length >= 1) resolve();
        else setTimeout(poll, 0);
      };
      poll();
    });

    loader.dispose(); // mid-flight — bumps generation; the parked open keeps running
    store.releaseNextParkedKeyRequest(); // let the stale doInitialize() finish successfully
    store.parkKeyAlways = null; // don't park the loader's NEXT (post-dispose) attempt too

    // The stale attempt's own `runPreflight()` call must not observe a resurrected
    // `this.preflight` — with the generation guard skipping the publish, `this.preflight`
    // stays null and `runPreflight()`'s own null-check rejects it.
    await expect(stalePreflight).rejects.toThrow(LoaderError);

    // A fresh `loadMesh()` must re-open metadata from scratch — not see stale non-null
    // `handles` and skip straight to a (missing) chunk fetch — and still load correctly.
    const before = store.requested.length;
    const data = await loader.loadMesh(VIEW);
    expect(store.requested.length).toBeGreaterThan(before);
    expect(data.vertexCount).toBe(4);
    expect(Array.from(data.vertices)).toEqual(TET_VERTICES);
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

  it('refuses an array_ref target that is itself broadcast-encoded (#1253)', async () => {
    // Two fixes landed for this hole independently, and the STRICTER one wins.
    //
    // The question is whether an `array_ref` whose target is `broadcasted` can be
    // *budgeted* or must be *refused*. It cannot be budgeted, and the reason is the same
    // fact that downgraded this issue's severity: the broadcast branch reads
    // `expectedElements ?? enc.n_elements`, so what gets allocated is driven by the
    // CALLER's expectation, not by anything in the target's metadata. A number the
    // target cannot be held to is not a bound. The stored and per-chunk terms say
    // nothing either — the target physically stores ONE row.
    //
    // So the endpoint gate refuses outright, and it costs nothing legitimate: the
    // encoder broadcast-encodes a uniform array at priority 1, BEFORE dedup is
    // consulted, so the writer never emits an `array_ref` pointing at a `broadcasted`
    // target. What the chain walk adds on top is depth — every hop is checked, not just
    // the last — since a `broadcasted` array is equally unboundable in the middle of a
    // chain as at its end.
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
      /array_ref to a 'broadcasted' target/
    );
    // And refused from metadata alone: not one chunk of the 540 MB was read.
    expect(store.chunkRequests()).toEqual([]);
  });

  it('refuses a broadcast array_ref target however SMALL it declares itself', async () => {
    // The size-independence half, and the point of it: `n_elements: 4` here versus 45
    // million above, same refusal. A gate that let the small one through would be
    // budgeting on a number the target is not held to, which is the whole objection.
    //
    // This is deliberately NOT "reject every broadcast array" — that would break real
    // writer output. `add_mesh(..., colors=(1, 3))` produces a `broadcasted` colours
    // array directly, and it loads fine (verified against bytes from the real Python
    // writer). The refusal is specifically an `array_ref` POINTING AT a broadcast
    // target, which the writer never emits because broadcast encoding is chosen at
    // priority 1, before dedup can turn anything into a ref.
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

    await expect(makeLoader(store, attrs).loadMesh(VIEW)).rejects.toThrow(
      /array_ref to a 'broadcasted' target/
    );
    expect(store.chunkRequests()).toEqual([]);
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

  it('refuses an array_ref whose TARGET is broadcast-encoded, before any chunk', async () => {
    // #1253 — the dual of the huge-physical-target case above. The budget resolves the
    // ref to the target it fetches and charges the target's STORED and per-chunk terms,
    // but a `broadcasted` target stores ONE (1, K) row — a few bytes — and the decoder
    // expands it to one row per logical element at fetch time. So a wide row behind a
    // small stub sails under the ceiling on the stored/chunk terms, then allocates
    // gigabytes on decode: exactly the "too late after decode" exhaustion the gate
    // exists to prevent. Here the row is wide enough that the broadcast would allocate
    // ~4.8 GB, yet the stored (~100 MB) and chunk (~100 MB) terms both fit the budget.
    const wide = 100_000_000;
    const attrs = meshAttrs();
    const store = buildStore(attrs, {
      vertices: {
        shape: [0, 3],
        dtype: '<f4',
        attrs: { encoding: { name: 'array_ref', target: 'evil', original_shape: [4, 3] } },
      },
      faces: { shape: [4, 3], dtype: '<u4', data: TET_FACES },
    });
    store.addArray('/evil', {
      shape: [1, wide],
      chunks: [1, wide],
      dtype: '|u1',
      attrs: { encoding: { name: 'broadcasted', n_elements: 4 } },
    });

    await expect(makeLoader(store, attrs).loadMesh(VIEW)).rejects.toThrow(
      /array_ref to a 'broadcasted' target/
    );
    // Rejected on metadata alone — neither the stub nor the target's chunk was fetched.
    expect(store.chunkRequests()).toEqual([]);
    // The stored and per-chunk terms really do fit, which is what makes the extra check
    // load-bearing rather than redundant with the byte budget.
    expect(wide * 1).toBeLessThan(MESH_DECODE_BUDGET_BYTES);
  });

  it('refuses an array_ref to a row-mode LUT target that decodes far past the ceiling', async () => {
    // #1253, the second attrs-driven expansion. The writer legitimately dedups to a
    // LUT-encoded array (two meshes sharing an identical scalars/colors array), so an
    // `array_ref` -> `lut_*` target cannot be refused outright — it must be budgeted.
    // But a row-mode LUT allocates `stored_indices x k` floats, k taken from the
    // TARGET's `original_shape`. Charging the stub's tiny logical count (as before)
    // let a small index array behind a wide `k` sail under the ceiling and then
    // allocate gigabytes — the same bypass as the broadcast case, one encoding over.
    const n = 100_000;
    const k = 10_000; // 100k x 10k x 4 B ~= 4 GB decoded
    const attrs = meshAttrs();
    const store = buildStore(attrs, {
      vertices: {
        shape: [0, 3],
        dtype: '<f4',
        attrs: { encoding: { name: 'array_ref', target: 'evil', original_shape: [4, 3] } },
      },
      faces: { shape: [4, 3], dtype: '<u4', data: TET_FACES },
    });
    store.addArray('/evil', {
      shape: [n],
      chunks: [n],
      dtype: '|u1',
      // A faithful, decodable LUT: one k-wide code vector, every index 0 — so without
      // the target-decode charge this is admitted and `decodeLUT` really does allocate
      // n x k floats (~4 GB), not merely throw on missing metadata.
      attrs: {
        encoding: {
          name: 'lut_uint8',
          lut_mode: 'row',
          lut: new Array(k).fill(0),
          original_shape: [n, k],
          original_dtype: 'float32',
        },
      },
    });

    await expect(makeLoader(store, attrs).loadMesh(VIEW)).rejects.toThrow(
      /account for .* over the/
    );
    // Rejected on the resolved target's decode size, on metadata alone — no chunk read.
    expect(store.chunkRequests()).toEqual([]);
    // Stored (~100 KB) and per-chunk (~100 KB) both fit; only the k-expanded decode
    // term catches it, which is what makes charging the TARGET's decode load-bearing.
    expect(n * 1).toBeLessThan(MESH_DECODE_BUDGET_BYTES);
  });

  it('a genuinely MISSING array_ref target stays a deterministic Validation rejection', async () => {
    // The other side of #1254's fix: a target that simply is not in the store is a real,
    // permanent defect, so it is `Validation` (retry policy skips it) and reports it as
    // not found — distinct from a transient open failure, which stays retryable below.
    const attrs = meshAttrs();
    const store = buildStore(attrs, {
      vertices: {
        shape: [0, 3],
        dtype: '<f4',
        attrs: { encoding: { name: 'array_ref', target: 'absent', original_shape: [4, 3] } },
      },
      faces: { shape: [4, 3], dtype: '<u4', data: TET_FACES },
    });
    let thrown: unknown;
    try {
      await makeLoader(store, attrs).loadMesh(VIEW);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(LoaderError);
    expect((thrown as LoaderError).kind).toBe('Validation');
    expect((thrown as LoaderError).message).toMatch(/was not found in the store/);
    expect(store.chunkRequests()).toEqual([]);
  });

  it('classifies a TRANSIENT open of an OPTIONAL array as Network, not Validation', async () => {
    // #1254, first site. The optional-array open loop used to swallow EVERY error and
    // leave the slot empty, after which the preflight reported the flag-with-no-array as
    // `Validation` — which the retry policy treats as deterministic and never re-fetches.
    // A network blip fetching `normals`' metadata must instead surface its retryable kind.
    const attrs = meshAttrs({ has_normals: true, normal_dims: [0, 1, 2] });
    const store = buildStore(
      attrs,
      tetArrays({ normals: { shape: [4, 3], dtype: '<f4', data: new Array(12).fill(0) } })
    );
    store.rejectKeyContaining = { needle: '/normals/', error: new Error('network request failed') };
    let thrown: unknown;
    try {
      await makeLoader(store, attrs).loadMesh(VIEW);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(LoaderError);
    expect((thrown as LoaderError).kind).toBe('Network');
  });

  it('classifies a TRANSIENT open of an array_ref TARGET as Network, not Validation', async () => {
    // #1254, second site. `resolveRefTarget` used to convert EVERY `zarr.open` failure on
    // the target into a `Validation` rejection, so a timeout opening the target's metadata
    // was persisted as deterministic and never retried. Only a genuine not-found should be
    // deterministic; a transient must surface its retryable kind.
    const attrs = meshAttrs();
    const store = buildStore(attrs, {
      vertices: {
        shape: [0, 3],
        dtype: '<f4',
        attrs: { encoding: { name: 'array_ref', target: 'shared', original_shape: [4, 3] } },
      },
      faces: { shape: [4, 3], dtype: '<u4', data: TET_FACES },
    });
    store.addArray('/shared', {
      shape: [4, 3],
      dtype: '<f4',
      data: [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    });
    store.rejectKeyContaining = { needle: '/shared/', error: new Error('network request failed') };
    let thrown: unknown;
    try {
      await makeLoader(store, attrs).loadMesh(VIEW);
    } catch (e) {
      thrown = e;
    }
    expect(thrown).toBeInstanceOf(LoaderError);
    expect((thrown as LoaderError).kind).toBe('Network');
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

describe('MeshWholeNodeLoader — an ALREADY-aborted update', () => {
  // The entry-point half of the abort story. That the signal reaches every payload
  // read is pinned above ('every chunk read carries an abort signal, not just the
  // faces one'), which asserts the same five arrays through the `signals` map. What
  // is NOT covered there is a signal that is already aborted when `updateView` is
  // called — a superseded update whose replacement landed first — where the
  // requirement is that it rejects and leaves NOTHING cached, so the next live
  // update refetches rather than serving a phantom.
  const FULL_ATTRS = meshAttrs({
    has_normals: true,
    normal_dims: [0, 1, 2],
    has_colors: true,
    has_scalars: true,
  });
  const fullArrays = () =>
    tetArrays({
      normals: { shape: [4, 3], dtype: '<f4', data: new Array(12).fill(0.5) },
      colors: { shape: [4, 3], dtype: '|u1', data: new Array(12).fill(128) },
      scalars: { shape: [4], dtype: '<f4', data: [1, 2, 3, 4] },
    });

  it('rejects a pre-aborted updateView with AbortError and caches nothing', async () => {
    const store = buildStore(FULL_ATTRS, fullArrays());
    const loader = makeLoader(store, FULL_ATTRS);
    const controller = new AbortController();
    controller.abort();

    await expect(loader.updateView(VIEW, undefined, controller.signal)).rejects.toMatchObject({
      name: 'AbortError',
    });

    // The failed fetch must not have latched: a later un-aborted update loads fine.
    const data = await loader.updateView(VIEW);
    expect(data.vertexCount).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// The LoaderMonitor surface
//
// A mesh node used to reach `connectLoaderToMonitor` and fail its duck-typed
// four-method check silently, so the data-loading monitor showed mesh in its
// scene-graph tree and nowhere else. These assert the shape the check needs AND
// that what it reports is a whole-node loader's honest telemetry rather than a
// spatial-index loader's shape filled with zeros.
// ---------------------------------------------------------------------------

describe('MeshWholeNodeLoader — monitor telemetry', () => {
  it('carries the whole four-method LoaderMonitor shape', () => {
    // The exact predicate `connect-loader-to-monitor.ts` applies. Asserted as a
    // set rather than per method because the check is all-or-nothing: three of
    // four leaves the loader silently unwired, which is the bug being fixed.
    const loader = makeLoader(
      buildStore(meshAttrs(), tetArrays()),
      meshAttrs()
    ) as unknown as Record<string, unknown>;
    for (const method of [
      'addEventListener',
      'removeEventListener',
      'getMetrics',
      'getActiveQueries',
    ]) {
      expect(typeof loader[method]).toBe('function');
    }
  });

  it('reports zero loads and no bytes before anything is fetched', () => {
    const loader = makeLoader(buildStore(meshAttrs(), tetArrays()), meshAttrs());
    const before = loader.getMetrics();
    expect(before.type).toBe('mesh-whole-node');
    expect(before.path).toBe('/mesh');
    expect(before.loads).toBe(0);
    expect(before.bytesLoaded).toBe(0);
    expect(before.memoryUsed).toBe(0);
    // No spatial index here, so nothing to be in flight either.
    expect(loader.getActiveQueries()).toEqual([]);
  });

  it('counts the ONE fetch, its triangles and its decoded bytes', async () => {
    const attrs = meshAttrs({
      has_normals: true,
      normal_dims: [0, 1, 2],
      has_scalars: true,
      has_uvs: true,
      has_texture: true,
      texture_encoding: 'raw',
      texture_width: 2,
      texture_height: 1,
      texture_channels: 4,
      texture_color_space: 'srgb',
    });
    const store = buildStore(
      attrs,
      tetArrays({
        normals: { shape: [4, 3], dtype: '<f4', data: new Array(12).fill(0.5) },
        scalars: { shape: [4], dtype: '<f4', data: [1, 2, 3, 4] },
        uvs: { shape: [4, 2], dtype: '<f4', data: [0, 0, 1, 0, 0, 1, 1, 1] },
        texture: { shape: [1, 2, 4], dtype: '|u1', data: [1, 2, 3, 4, 5, 6, 7, 8] },
      })
    );
    const loader = makeLoader(store, attrs);
    await loader.loadMesh(VIEW);

    const m = loader.getMetrics();
    expect(m.loads).toBe(1);
    // `elementsLoaded` counts the DRAWN PRIMITIVE: 4 triangles, not 4 vertices
    // (which would be the same number here — hence a 4x3-face tetrahedron read
    // as 4 faces / 4 vertices is a poor witness; the byte assertion below is
    // what actually distinguishes the arrays).
    expect(m.elementsLoaded).toBe(4);
    // Decoded bytes: 12 float32 vertices + 12 uint32 faces + 12 float32
    // normals + 4 float32 scalars + 8 float32 UVs + 8 uint8 texels
    // = 48 + 48 + 48 + 16 + 32 + 8.
    expect(m.bytesLoaded).toBe(200);
    // Resident memory adds the projection scratch the loader keeps for the
    // node's life: position (4*3 float32 = 48) + mask (4 uint8) + faceScratch
    // (4*3 uint32 = 48).
    expect(m.memoryUsed).toBe(200 + 48 + 4 + 48);
    // No spatial query ran, so the panel's QUERY SPEED average gets no sample.
    expect(m.queries).toBe(0);
    expect(m.avgQueryTime).toBe(0);
    expect(m.spatialIndex).toBeUndefined();
  });

  it('does not re-count a cached re-serve', async () => {
    // The whole-node design's central claim, seen from the telemetry side: a
    // slice scrub re-serves the resident mesh, so `loads` must stay at 1 rather
    // than climbing once per view change.
    const loader = makeLoader(buildStore(meshAttrs(), tetArrays()), meshAttrs());
    await loader.loadMesh(VIEW);
    await loader.updateView(VIEW);
    await loader.updateView(VIEW);
    expect(loader.getMetrics().loads).toBe(1);
  });

  it('emits one load event with the same numbers it recorded', async () => {
    const loader = makeLoader(buildStore(meshAttrs(), tetArrays()), meshAttrs());
    const events: { type: string; data: { elements?: number; memory?: number } }[] = [];
    loader.addEventListener((e) => events.push(e));
    await loader.loadMesh(VIEW);

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe('load');
    expect(events[0].data.elements).toBe(4);
    // 48 vertex bytes + 48 face bytes; the event's `memory` field is the load's
    // bytes (what the monitor's bandwidth window integrates), NOT the resident
    // footprint — which is why it excludes the projection scratch above.
    expect(events[0].data.memory).toBe(96);
    expect(loader.getMetrics().bytesLoaded).toBe(96);
  });

  it('records a real failure as an error, and reports it', async () => {
    const store = buildStore(meshAttrs(), tetArrays());
    store.rejectKeyContaining = { needle: 'faces/0.0', error: new Error('boom') };
    const loader = makeLoader(store, meshAttrs());
    const events: { type: string }[] = [];
    loader.addEventListener((e) => events.push(e));

    await expect(loader.loadMesh(VIEW)).rejects.toThrow();
    expect(loader.getMetrics().errors).toBe(1);
    expect(events.map((e) => e.type)).toEqual(['error']);
  });

  it('does NOT record a deliberate abort as an error', async () => {
    // A dataset switch aborts every in-flight read. Counting those would fire
    // the advisor's "High Error Rate" recommendation every time the user
    // switches scenes — the same exclusion the sibling loaders make.
    const loader = makeLoader(buildStore(meshAttrs(), tetArrays()), meshAttrs());
    const controller = new AbortController();
    controller.abort();

    await expect(loader.updateView(VIEW, undefined, controller.signal)).rejects.toMatchObject({
      name: 'AbortError',
    });
    expect(loader.getMetrics().errors).toBe(0);
  });

  it('takes its visible-element count from the commit', async () => {
    // Pushed in rather than read out: a mesh is resident in full, so the
    // loader cannot know which faces the current slab indexes.
    const loader = makeLoader(buildStore(meshAttrs(), tetArrays()), meshAttrs());
    await loader.loadMesh(VIEW);
    expect(loader.getMetrics().visibleElements).toBe(0);
    loader.recordVisibleElements(3);
    expect(loader.getMetrics().visibleElements).toBe(3);
  });

  it('drops its LIVE figures on dispose but keeps the session counters', async () => {
    const loader = makeLoader(buildStore(meshAttrs(), tetArrays()), meshAttrs());
    await loader.loadMesh(VIEW);
    loader.recordVisibleElements(4);
    loader.dispose();

    const m = loader.getMetrics();
    // The payload is gone, so claiming it as resident memory would leave the
    // panel reporting memory for a torn-down node.
    expect(m.memoryUsed).toBe(0);
    expect(m.visibleElements).toBe(0);
    // History stays: those bytes really were fetched this session.
    expect(m.loads).toBe(1);
    expect(m.bytesLoaded).toBe(96);
  });

  it('hands out snapshots, not its live record', async () => {
    const loader = makeLoader(buildStore(meshAttrs(), tetArrays()), meshAttrs());
    await loader.loadMesh(VIEW);
    const snapshot = loader.getMetrics();
    loader.recordVisibleElements(4);
    expect(snapshot.visibleElements).toBe(0);
  });
});
