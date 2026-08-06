/**
 * Luxar Zarr facade.
 *
 * This module is the only production boundary that should import zarrita
 * directly. The rest of the viewer depends on Luxar-owned concepts (stores,
 * locations, arrays, chunks, selections) so future backend changes, zarrita
 * API movement, or local patches stay isolated here.
 */

import * as zarrita from 'zarrita';

import { boundedConcurrencyStore } from '../utils/fetch-concurrency';
import { LuxarDeltaCodec } from './codecs/luxar-delta';
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
 * `numcodecs` package, which isn't a public dep of luxar-viewer. The only
 * consumer (`bootstrap.ts`) just chains `.catch()` on the returned
 * promise, so the loosened element type is sufficient.
 */
export const codecRegistry: Map<string, () => Promise<unknown>> = zarrita.registry;

// Register the Luxar-owned `luxar_delta_v1` zarr filter (columnar per-chunk
// delta+zigzag on quantized codes — see `./codecs/luxar-delta.ts`). Module
// scope, not bootstrap: any context that opens zarr arrays imports this
// facade, so main thread AND workers get the codec before any array open.
// zarrita maps a v2 `.zarray` filter `{id: "luxar_delta_v1"}` to the codec
// name `numcodecs.luxar_delta_v1`. The typeof guard only matters under unit
// tests that vi.mock('zarrita') with a registry stub lacking `.set` — in
// every real context the registry is zarrita's live Map.
if (typeof codecRegistry?.set === 'function') {
  codecRegistry.set('numcodecs.luxar_delta_v1', () => Promise.resolve(LuxarDeltaCodec));
}

/** Create the default HTTP-backed store for browser/network datasets.
 *
 * Wrapped in {@link boundedConcurrencyStore} so the no-cache path's chunk
 * fetches are throttled (the default caching path is throttled at its network
 * tier in `cache/multi-level-caching-store/fetch-retry.ts`). See
 * {@link withFetchGate} for why. */
export function createFetchStore(url: string): FetchStore {
  return boundedConcurrencyStore(new zarrita.FetchStore(url));
}

/**
 * Open a raw store for Luxar reads, adding consolidated-metadata support when
 * the dataset provides it and falling back to the original store otherwise.
 */
export async function openStore<Store extends AsyncReadable>(
  rawStore: Store
): Promise<MaybeListableStore<Store>> {
  return (await zarrita.withMaybeConsolidatedMetadata(rawStore)) as MaybeListableStore<Store>;
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
