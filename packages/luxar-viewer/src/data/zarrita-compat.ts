import * as zarr from 'zarrita';

/**
 * Minimal shape of stores augmented by zarrita's consolidated-metadata helpers.
 * `contents()` is present only when consolidated metadata exists; callers must
 * still feature-detect it (for example with `hasContentsMethod`).
 */
export type MaybeConsolidatedReadable<Store extends zarr.Readable> = Store & {
  contents?: () => Array<{ path: string; kind: 'array' | 'group' }>;
};

type ZarritaConsolidationApi = typeof zarr & {
  /** zarrita <= 0.5 helper. Removed/renamed in 0.7. */
  tryWithConsolidated?: (store: zarr.Readable, options?: unknown) => Promise<zarr.Readable>;
  /** zarrita >= 0.7 helper. */
  withMaybeConsolidatedMetadata?: (
    store: zarr.Readable,
    options?: unknown
  ) => Promise<zarr.Readable>;
};

/**
 * Wrap a store with consolidated metadata when available, while supporting both
 * zarrita 0.5 (`tryWithConsolidated`) and 0.7
 * (`withMaybeConsolidatedMetadata`).
 *
 * Keeping this indirection in one place lets the rest of the viewer depend on
 * behavior (fast metadata when present, no-op fallback when absent) instead of a
 * version-specific helper name.
 */
export async function withMaybeConsolidatedMetadata<Store extends zarr.Readable>(
  store: Store,
  options?: unknown
): Promise<MaybeConsolidatedReadable<Store>> {
  const api = zarr as ZarritaConsolidationApi;

  const withMaybe =
    'withMaybeConsolidatedMetadata' in api ? api.withMaybeConsolidatedMetadata : undefined;
  if (typeof withMaybe === 'function') {
    return (await withMaybe(store, options)) as MaybeConsolidatedReadable<Store>;
  }

  const tryWith = 'tryWithConsolidated' in api ? api.tryWithConsolidated : undefined;
  if (typeof tryWith === 'function') {
    return (await tryWith(store, options)) as MaybeConsolidatedReadable<Store>;
  }

  throw new Error(
    'Unsupported zarrita version: expected withMaybeConsolidatedMetadata() ' +
      'or tryWithConsolidated().'
  );
}
