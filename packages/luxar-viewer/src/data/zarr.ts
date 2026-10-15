/**
 * Luxar Zarr facade.
 *
 * This module is the only production boundary that should import zarrita
 * directly. The rest of the viewer depends on Luxar-owned concepts (stores,
 * locations, arrays, chunks, selections) so future backend changes, zarrita
 * API movement, or local patches stay isolated here.
 */

import * as zarrita from 'zarrita';

import { boundedConcurrencyStore, noteFetchUrl } from '../utils/fetch-concurrency';
import { LuxarDeltaCodec } from './codecs/luxar-delta';
import {
  WorkerBloscCodec,
  setNativeBloscLoader,
  type NativeBloscCtor,
} from './codecs/worker-blosc';
import { isZippedStoreUrl } from './zip/entries';
import { LuxarZipStore } from './zip/store';
import type { AbsolutePath, AsyncReadable, GetOptions, Readable } from '@zarrita/storage';

/** `@zarrita/storage` primitives (path, readable-store, and get-option types) re-exported so callers depend only on this facade. */
export type { AbsolutePath, AsyncReadable, GetOptions, Readable };

/** zarrita's array element data-type tag (e.g. `float32`, `uint8`). */
export type DataType = zarrita.DataType;
/** The concrete typed-array type backing a given zarrita {@link DataType}. */
export type TypedArray<D extends DataType> = zarrita.TypedArray<D>;
/** A zarrita slice selection (start/stop/step) for indexing into an array. */
export type Slice = zarrita.Slice;
/** A resolved location within a store, used to address groups and arrays. */
export type Location<Store = Readable> = zarrita.Location<Store>;
/** A zarr group (a node with child arrays/groups and attributes). */
export type Group<Store extends Readable = Readable> = zarrita.Group<Store>;
/** A zarr array node parameterized by element {@link DataType} and backing store. */
export type Array<D extends DataType = DataType, Store extends Readable = Readable> = zarrita.Array<
  D,
  Store
>;
/** zarrita's HTTP fetch-backed store implementation. */
export type FetchStore = zarrita.FetchStore;

/** One entry from a listable store's contents: a node path and whether it is an array or group. */
export interface StoreContentsEntry {
  path: string;
  kind: 'array' | 'group';
}

/** A store that may optionally expose a synchronous `contents()` listing (present after consolidated-metadata wrapping). */
export type MaybeListableStore<Store extends Readable = Readable> = Store & {
  contents?: () => StoreContentsEntry[];
};

/** Options for {@link open}: node `kind`, whether to read attributes, and an optional abort `signal`. */
export type OpenOptions = {
  kind?: 'array' | 'group';
  attrs?: boolean;
  signal?: AbortSignal;
};

/**
 * Codec registry exposed through the facade for bootstrap-time codec setup.
 *
 * The annotation is intentionally loose (`Promise<unknown>` instead of the
 * inferred `Promise<CodecEntry>`) to keep the lib-build's emitted `.d.ts`
 * portable — zarrita's `CodecEntry` is defined in the transitive
 * `numcodecs` package, which isn't a public dep of @luxar/viewer. The only
 * consumer (`bootstrap.ts`) just chains `.catch()` on the returned
 * promise, so the loosened element type is sufficient.
 */
export const codecRegistry: Map<string, () => Promise<unknown>> = zarrita.registry;

// Register the Luxar-owned `luxar_delta_v1` zarr filter (columnar per-chunk
// delta+zigzag on quantized codes — see `./codecs/luxar-delta.ts`). Module
// scope, not bootstrap: any context that opens zarr arrays imports this
// facade, so main thread AND workers get the codec before any array open.
//
// BOTH names are required, because zarrita's registry has two namespaces and a
// store's format decides which one is consulted:
//   • format 2 — a `.zarray` filter `{id: "luxar_delta_v1"}` is looked up as
//     `numcodecs.luxar_delta_v1` (zarrita prefixes v2 filter ids, the same way
//     it registers `numcodecs.blosc`, `numcodecs.zstd`, …);
//   • format 3 — a codec-chain entry `{name: "luxar_delta_v1"}` is looked up
//     VERBATIM, alongside zarrita's own bare `blosc` / `zstd` / `bytes`.
// Registering only the prefixed name — which is all that was needed while Luxar
// wrote format 2 — makes every delta-filtered format-3 array throw
// `UnknownCodecError` at first chunk decode. Luxar now writes format 3 and
// existing stores stay format 2, so both are live simultaneously and neither
// can be dropped.
//
// The typeof guard only matters under unit tests that vi.mock('zarrita') with a
// registry stub lacking `.set` — in every real context the registry is
// zarrita's live Map.
if (typeof codecRegistry?.set === 'function') {
  const luxarDelta = () => Promise.resolve(LuxarDeltaCodec);
  codecRegistry.set('numcodecs.luxar_delta_v1', luxarDelta); // format 2
  codecRegistry.set('luxar_delta_v1', luxarDelta); // format 3
}

// zarrita's own numcodecs Blosc thunk, captured BEFORE the override below so
// the native codec stays reachable: it is the main-thread fallback, and what a
// data worker runs. Memoized — the registry thunk re-imports on every call.
const nativeBloscThunk = codecRegistry?.get?.('blosc') as
  (() => Promise<NativeBloscCtor>) | undefined;
let nativeBlosc: Promise<NativeBloscCtor> | null = null;

/**
 * Load the native numcodecs Blosc codec class (blosc WASM, instantiated on
 * first encode/decode). The registry's `blosc` entry is the worker-offloaded
 * {@link WorkerBloscCodec}; this is the codec it falls back to, and the one a
 * data worker decodes with.
 */
export function loadNativeBlosc(): Promise<NativeBloscCtor> {
  if (!nativeBloscThunk) return Promise.reject(new Error('zarr: no native blosc codec'));
  if (!nativeBlosc) {
    const loading = nativeBloscThunk();
    // A failed chunk import (flaky network) must not stick: the next call retries.
    loading.catch(() => {
      if (nativeBlosc === loading) nativeBlosc = null;
    });
    nativeBlosc = loading;
  }
  return nativeBlosc;
}

// Replace zarrita's `blosc` (format 3) and `numcodecs.blosc` (format 2) with
// the worker-offloaded codec (see `./codecs/worker-blosc.ts`). Codecs resolve
// per array on first decode, so module scope is early enough. The thunk also
// starts loading the native codec, so bootstrap's `get('blosc')()` warm-up
// still parses the main-thread fallback's WASM ahead of the first decode.
if (nativeBloscThunk && typeof codecRegistry?.set === 'function') {
  setNativeBloscLoader(loadNativeBlosc);
  const workerBlosc = (): Promise<unknown> => {
    loadNativeBlosc().catch(() => {
      /* Surfaced by the first decode that needs the fallback. */
    });
    return Promise.resolve(WorkerBloscCodec);
  };
  codecRegistry.set('blosc', workerBlosc);
  codecRegistry.set('numcodecs.blosc', workerBlosc);
}

/** Create the default HTTP-backed store for browser/network datasets.
 *
 * Wrapped in {@link boundedConcurrencyStore} so the no-cache path's chunk
 * fetches are throttled (the default caching path is throttled at its network
 * tier in `cache/multi-level-caching-store/fetch-retry.ts`). See
 * {@link withFetchGate} for why. */
export function createFetchStore(url: string): FetchStore {
  noteFetchUrl(url);
  return boundedConcurrencyStore(new zarrita.FetchStore(url));
}

/**
 * Create a store that reads a zipped Zarr archive (`.zarr.zip`) in place, over
 * HTTP range requests — one ranged GET per chunk, no unpacking.
 *
 * `ZipFileStore` comes from `@zarrita/storage` rather than `zarrita`, which
 * re-exports only `FetchStore`. Note it is flagged `@experimental` upstream;
 * we lean on that lightly, because only `get()` is ever called here — its
 * `getRange()` is the method that reaches into `unzipit` internals, and nothing
 * in the viewer calls `getRange` on a store. Browser builds may warn that
 * `unzipit` imports Node's `worker_threads`; that import is guarded by its Node
 * branch, and workers remain disabled in the browser path used here.
 *
 * NOT wrapped in {@link boundedConcurrencyStore}: the ranged GETs are gated one
 * level down, inside `fetchWithRetry`, which also gives them the retry budget.
 * Gating at both levels would let a gated `get` await a gated `read` and
 * deadlock the pool.
 *
 * KNOWN COST: reading an archive pays a fixed preamble before the first chunk —
 * `unzipit` reads a 65,557-byte tail to find the end-of-central-directory
 * record and then the central directory itself, so a store whose directory sits
 * just before that tail transfers part of it twice (~141 kB measured at 1081
 * members). {@link LuxarZipStore} at least defers that to the first read rather
 * than paying it at construction.
 */
export function createZipStore(url: string): AsyncReadable {
  return new LuxarZipStore(url);
}

/**
 * Pick the store implementation for a dataset URL: zipped archives get
 * {@link createZipStore}, everything else the directory-backed
 * {@link createFetchStore}.
 */
export function createStoreForUrl(url: string): AsyncReadable {
  return isZippedStoreUrl(url) ? createZipStore(url) : createFetchStore(url);
}

/** A zarr on-disk format, as the store's consolidated metadata identified it. */
export type StoreFormat = 'v2' | 'v3';

/** Format of each store {@link openStore} returned with a known format. */
const STORE_FORMATS = new WeakMap<object, StoreFormat>();

/**
 * The on-disk format of a store returned by {@link openStore}, or `null` when it
 * is unknown (no consolidated metadata, or a store that did not come from
 * `openStore`). Optional-sidecar probes use it to ask for ONE format's
 * documents rather than guessing both.
 */
export function storeFormatOf(store: unknown): StoreFormat | null {
  return typeof store === 'object' && store !== null ? (STORE_FORMATS.get(store) ?? null) : null;
}

/** Metadata documents of each format, by file name. */
const FORMAT_DOCS: Record<StoreFormat, readonly string[]> = {
  v2: ['.zattrs', '.zgroup', '.zarray', '.zmetadata'],
  v3: ['zarr.json'],
};

/** The format a metadata key belongs to, or `null` for any other key (a chunk). */
function formatOfMetaKey(key: string): StoreFormat | null {
  const name = key.slice(key.lastIndexOf('/') + 1);
  if (FORMAT_DOCS.v3.includes(name)) return 'v3';
  if (FORMAT_DOCS.v2.includes(name)) return 'v2';
  return null;
}

/** Parent node path of a metadata key (`/a/b/.zattrs` → `/a/b`, `/.zattrs` → `/`). */
function nodePathOf(key: string): string {
  const parent = key.slice(0, key.lastIndexOf('/'));
  return parent === '' ? '/' : parent;
}

/**
 * What {@link withMetadataGate} knows about the store above it. Empty while
 * consolidated metadata loads — those reads (`zarr.json`, `.zmetadata`) must
 * reach the store — and filled in once the consolidated wrapper is built.
 */
interface MetadataGateState {
  format: StoreFormat | null;
  /** Node paths the consolidated index lists (`contents()`). */
  listed: Set<string> | null;
  /** Consolidation reads the gate saw answered, which name the store's format. */
  answered: { v3Root: boolean; v2Index: boolean };
}

/** The gate's verdict for one key: drop it (`true`) or forward it. */
function gateDrops(state: MetadataGateState, key: string): boolean {
  const { format, listed } = state;
  if (format === null) return false;
  const docFormat = formatOfMetaKey(key);
  if (docFormat !== null && docFormat !== format) return true;
  return format === 'v2' && key.endsWith('/.zattrs') && listed?.has(nodePathOf(key)) === true;
}

/**
 * A pass-through store layer, placed UNDER zarrita's consolidated wrapper, that
 * drops metadata reads the consolidated index has already settled.
 *
 * zarrita's consolidated wrapper answers the documents its index holds and
 * forwards EVERY other key to the store beneath it — so each speculative read
 * became a real request and a console `GET … 404`:
 *
 * - the other format's documents. `zarrita.open` guesses the format from a
 *   counter keyed on the store OBJECT, so the first open on a fresh wrapper (and
 *   every open of an absent optional node) asks for `.zattrs` / `.zgroup` on a
 *   format-3 store, or `zarr.json` on a format-2 one. A store has one format;
 *   those documents cannot exist.
 * - format 2's `.zattrs` of a node the index lists without one. zarr writes
 *   `.zattrs` only for non-empty attributes and consolidation copies every one
 *   it finds, so a listed node whose `.zattrs` the index lacks has none — yet
 *   `open.v2` reads it for every array and group it opens.
 *
 * Deliberately NOT dropped: the documents of a node the index does not list.
 * A warm cache tier can serve a root index older than a sidecar (`luxar env
 * attach` keeps `content_hash`), and the sidecar loaders rely on such a read
 * still reaching the server; they consult a fresh listing first instead.
 *
 * Every other property (`getRange`, a caching store's own methods) is the
 * underlying store's, bound to it.
 */
function withMetadataGate<Store extends AsyncReadable>(
  store: Store,
  state: MetadataGateState
): Store {
  const get = async (key: AbsolutePath, options?: GetOptions): Promise<Uint8Array | undefined> => {
    if (gateDrops(state, key)) return undefined;
    const bytes = await store.get(key, options);
    if (bytes !== undefined && state.format === null) {
      if (key === '/zarr.json') state.answered.v3Root = true;
      if (key === '/.zmetadata') state.answered.v2Index = true;
    }
    return bytes;
  };
  return new Proxy(store, {
    get(target, prop) {
      if (prop === 'get') return get;
      const value: unknown = Reflect.get(target, prop, target);
      return typeof value === 'function'
        ? (value as (...a: unknown[]) => unknown).bind(target)
        : value;
    },
  });
}

/**
 * Open a raw store for Luxar reads, adding consolidated-metadata support when
 * the dataset provides it and falling back to the original store otherwise.
 */
export async function openStore<Store extends AsyncReadable>(
  rawStore: Store
): Promise<MaybeListableStore<Store>> {
  // Try format 3 FIRST. zarrita's `resolveFormats` defaults to `["v2", "v3"]`
  // for any store its version counter has not seen, so without this it probes
  // `.zmetadata` (a 404 on every format-3 store) before `zarr.json` — one
  // wasted round trip on the critical path of first paint, measured at 268 ms
  // against a CDN.
  //
  // The option takes an ORDER, not a pin, so format-2 stores keep loading —
  // they simply pay the extra probe instead, which is the right way round now
  // that Luxar writes format 3 and only legacy stores are format 2.
  //
  // The consolidated wrapper sits on a metadata gate that watches which
  // consolidation document answered: that names the store's format, after
  // which the gate stops the other format's documents (and format 2's
  // index-settled `.zattrs`) from being requested at all.
  const state: MetadataGateState = {
    format: null,
    listed: null,
    answered: { v3Root: false, v2Index: false },
  };
  const gated = withMetadataGate(rawStore, state);
  const opened = (await zarrita.withMaybeConsolidatedMetadata(gated, {
    format: ['v3', 'v2'],
  })) as MaybeListableStore<Store>;
  // No consolidated metadata: hand back the store as given — the gate has no
  // index to answer from.
  if (opened === gated) return rawStore as MaybeListableStore<Store>;
  // `.zmetadata` answering means format 2 (a format-3 `zarr.json` without an
  // index can answer first, then fail to consolidate).
  const format: StoreFormat | null = state.answered.v2Index
    ? 'v2'
    : state.answered.v3Root
      ? 'v3'
      : null;
  if (format !== null) {
    state.listed = new Set((opened.contents?.() ?? []).map((entry) => entry.path));
    state.format = format;
    STORE_FORMATS.set(opened, format);
  }
  return opened;
}

/** Return a root location for resolving dataset-relative paths. */
export function root<Store>(store: Store): Location<Store> {
  return zarrita.root(store);
}

type OpenFacade = {
  <Store extends Readable>(
    location: Location<Store> | Store,
    options: OpenOptions & { kind: 'group' }
  ): Promise<Group<Store>>;
  <Store extends Readable>(
    location: Location<Store> | Store,
    options: OpenOptions & { kind: 'array' }
  ): Promise<Array<DataType, Store>>;
  <Store extends Readable>(
    location: Location<Store> | Store,
    options?: OpenOptions
  ): Promise<Array<DataType, Store> | Group<Store>>;
};

/**
 * Open a zarr node (array or group) at a location, overloaded on
 * {@link OpenOptions.kind} so the return type narrows to the requested kind.
 * Thin wrapper over `zarrita.open` behind the facade.
 */
export const open: OpenFacade = ((location, options) => {
  return zarrita.open(location, options as Parameters<typeof zarrita.open>[1]);
}) as OpenFacade;

/**
 * Open a location as a zarr group.
 *
 * @param location - Location or store to open.
 * @param options - Open options minus `kind` (forced to `'group'`).
 * @returns The opened {@link Group}.
 */
export function openGroup<Store extends Readable>(
  location: Location<Store> | Store,
  options?: Omit<OpenOptions, 'kind'>
): Promise<Group<Store>> {
  return open(location, { ...options, kind: 'group' });
}

/**
 * Open a location as a zarr array.
 *
 * @param location - Location or store to open.
 * @param options - Open options minus `kind` (forced to `'array'`).
 * @returns The opened {@link Array}.
 */
export function openArray<Store extends Readable>(
  location: Location<Store> | Store,
  options?: Omit<OpenOptions, 'kind'>
): Promise<Array<DataType, Store>> {
  return open(location, { ...options, kind: 'array' });
}

/**
 * Read a whole array, or a slice selection of it, into a typed array.
 *
 * @param array - Opened zarr array.
 * @param selection - Per-dimension slice selection; omit to read the whole array.
 * @param options - Get options, e.g. an abort signal built by {@link abortOptions}.
 * @returns The decoded data with the selection's shape and stride.
 */
export async function readArray<D extends DataType, Store extends Readable>(
  array: Array<D, Store>,
  selection?: Slice[],
  options?: GetOptions
): Promise<{ data: TypedArray<D>; shape: number[]; stride: number[] }> {
  if (options) {
    return zarrita.get(array, selection ?? null, options);
  }
  return selection === undefined ? zarrita.get(array) : zarrita.get(array, selection);
}

// lifecycle-exempt: wraps a signal into read options; starts and owns no work
/**
 * Build a `GetOptions` carrying an `AbortSignal`, or `undefined` when there is
 * no signal. Lets read sites forward a per-update abort signal into `get()`
 * uniformly: `get(array, sel, abortOptions(signal))`. This makes zarrita honor
 * the signal (`throwIfAborted` between chunks + `store.get(key, { signal })`)
 * even for arrays NOT wrapped by the L0 `wrapWithCache` chokepoint — e.g. when
 * L0 caching is disabled or for `array_ref` target arrays opened directly.
 */
export function abortOptions(signal?: AbortSignal | null): GetOptions | undefined {
  return signal ? { signal } : undefined;
}

/** Build a slice selection. Pass `null` for an open start/end. */
export function slice(start?: number | null, end?: number | null): Slice {
  return zarrita.slice(start ?? null, end ?? null);
}

/**
 * Open a location as a zarr group, trying format 3 before format 2.
 *
 * The same v2-first default that {@link openStore} works around bites a second
 * time, independently: `zarrita.open` consults its version counter keyed on the
 * STORE OBJECT, and the consolidated wrapper `openStore` returns is a brand-new
 * object the counter has never seen. So the very first `open` on a format-3
 * store probes `.zattrs` and then `.zgroup` — two more 404s, measured at 279 ms
 * and 269 ms — before falling back to v3.
 *
 * Only the FIRST open pays it (a successful open increments the counter, so
 * later array opens already guess v3), which is why this helper is needed only
 * for the root group and not everywhere.
 *
 * Falls back to the format-guessing {@link open} when the v3 attempt reports a
 * missing node, so a genuine format-2 store still loads.
 */
export async function openGroupPreferV3<Store extends Readable>(
  location: Location<Store> | Store
): Promise<Group<Store>> {
  try {
    return await zarrita.open.v3(location, { kind: 'group' });
  } catch (error) {
    // A v3 miss means "not a v3 group here", not "nothing here" — fall through
    // to the guessing open, which will find the v2 documents. Any other error
    // (a real network fault, malformed JSON) must not be masked.
    if (!isNotFoundError(error) && !(error instanceof zarrita.InvalidMetadataError)) throw error;
    return openGroup(location);
  }
}

/**
 * Open a location as a zarr group of a KNOWN format, or by
 * {@link openGroupPreferV3} when `format` is `null`.
 *
 * For probing an optional node on a raw store (no consolidated index to answer
 * from): a format guess costs one 404 per wrong document — up to three for an
 * absent node — where a known format costs one. Format 2 checks `.zgroup`
 * before `open.v2` reads `.zattrs` (which a group without attributes lacks), so
 * an absent group is ONE request.
 *
 * @param location - Location to open.
 * @param format - The store's format, from {@link storeFormatOf} or the root
 *   document that answered.
 * @returns The opened {@link Group}; rejects with a not-found error when absent.
 */
export async function openGroupOfFormat<Store extends Readable>(
  location: Location<Store>,
  format: StoreFormat | null
): Promise<Group<Store>> {
  if (format === 'v3') return zarrita.open.v3(location, { kind: 'group' });
  if (format === 'v2') {
    const { path } = location.resolve('.zgroup');
    const zgroup = await location.store.get(path);
    if (zgroup === undefined) throw new zarrita.NotFoundError('v2 group', { path });
    // `open.v2` reads `.zgroup` again: hand it the bytes just fetched.
    const store = location.store;
    const replay = new Proxy(store, {
      get(target, prop) {
        if (prop === 'get') {
          return (key: AbsolutePath, options?: GetOptions) =>
            key === path ? Promise.resolve(zgroup) : target.get(key, options);
        }
        const value: unknown = Reflect.get(target, prop, target);
        return typeof value === 'function'
          ? (value as (...a: unknown[]) => unknown).bind(target)
          : value;
      },
    });
    return zarrita.open.v2(new zarrita.Location(replay, location.path), { kind: 'group' });
  }
  return openGroupPreferV3(location);
}

/** Normalize backend-specific missing-node errors behind a Luxar-owned helper. */
export function isNotFoundError(error: unknown): boolean {
  if (error instanceof zarrita.NotFoundError) return true;
  const message = error instanceof Error ? error.message : String(error);
  return (
    message.includes('404') ||
    message.includes('Not Found') ||
    message.includes('Node not found') ||
    message.includes('not found')
  );
}
