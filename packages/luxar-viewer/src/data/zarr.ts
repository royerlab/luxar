/**
 * Luxar Zarr facade.
 *
 * This module is the only production boundary that should import zarrita
 * directly. The rest of the viewer depends on Luxar-owned concepts (stores,
 * locations, arrays, chunks, selections) so future backend changes, zarrita
 * API movement, or local patches stay isolated here.
 */

import * as zarrita from 'zarrita';
import type {
  AbsolutePath,
  AsyncReadable,
  GetOptions,
  RangeQuery,
  Readable,
  SyncReadable,
} from '@zarrita/storage';

export type { AbsolutePath, AsyncReadable, GetOptions, RangeQuery, Readable, SyncReadable };

export type DataType = zarrita.DataType;
export type TypedArray<D extends DataType> = zarrita.TypedArray<D>;
export type Slice = zarrita.Slice;
export type Location<Store = Readable> = zarrita.Location<Store>;
export type Group<Store extends Readable = Readable> = zarrita.Group<Store>;
export type Array<D extends DataType = DataType, Store extends Readable = Readable> = zarrita.Array<
  D,
  Store
>;
export type FetchStore = zarrita.FetchStore;
export type ArrayChunk<D extends DataType = DataType> = Awaited<
  ReturnType<zarrita.Array<D, Readable>['getChunk']>
>;

export interface StoreContentsEntry {
  path: string;
  kind: 'array' | 'group';
}

export type ListableStore<Store extends Readable = Readable> = Store & {
  contents(): StoreContentsEntry[];
};

export type MaybeListableStore<Store extends Readable = Readable> = Store & {
  contents?: () => StoreContentsEntry[];
};

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

/** Create the default HTTP-backed store for browser/network datasets. */
export function createFetchStore(url: string): FetchStore {
  return new zarrita.FetchStore(url);
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

export const open: OpenFacade = ((location, options) => {
  return zarrita.open(location, options as Parameters<typeof zarrita.open>[1]);
}) as OpenFacade;

export function openGroup<Store extends Readable>(
  location: Location<Store> | Store,
  options?: Omit<OpenOptions, 'kind'>
): Promise<Group<Store>> {
  return open(location, { ...options, kind: 'group' });
}

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

export const get = readArray;

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
