/**
 * Which optional ROOT-LEVEL sidecar groups (`chunk_packs/`, `environment/`) a
 * store has, answered from an index known to be current — so a store without
 * them costs no request, and no console `GET … 404`.
 *
 * Both sidecars are excluded from the scene digest, so a root index that a cache
 * tier served (revalidated by `content_hash` alone) can predate them and its
 * silence proves nothing. Two sources ARE current:
 *
 * 1. The opened store's consolidated index, when it came from the network this
 *    load (`rootIndexFromNetwork`).
 * 2. The load-time root-document fetch (`cache/root-document-prefetch.ts`), which
 *    goes to the server on EVERY load to validate the cache. For a format-3
 *    store that document IS the consolidated index, so even a warm load whose
 *    store opened from the cache has this load's network copy in hand.
 *
 * Neither available (a format-2 store, whose index is `.zmetadata` rather than
 * the root `.zattrs`; a presigned or zipped source; a failed fetch): the answer
 * is "unknown" and the caller probes — but for ONE format, the one the store
 * turned out to be, instead of zarrita's guess at both.
 *
 * @module data/scene-loader/cache/root-listing
 */

import * as zarr from '../../zarr';
import { hasContentsMethod } from '../../../types/zarr';
import type { RootDocumentFetch } from '../../../cache/root-document-prefetch';

/**
 * A current answer about root-level groups, and the store's format.
 *
 * `lists` is `true` / `false` when a current index says, `undefined` when no
 * current index is available (probe).
 */
export interface RootListing {
  lists(group: string): boolean | undefined;
  /** The store's on-disk format when known, else `null`. */
  readonly format: zarr.StoreFormat | null;
  /** Whether {@link lists}' answer came from the opened store's own index. */
  readonly fromOpenedIndex: boolean;
}

/**
 * How long to wait for the load-time root-document fetch to settle before
 * falling back to a probe. It has been in flight since bootstrap and is almost
 * always settled by now; a slow one must not hold the scene load.
 */
const ROOT_DOCUMENT_WAIT_MS = 2000;

/** No current index: the caller probes (one format when `format` is known). */
function unknownListing(format: zarr.StoreFormat | null): RootListing {
  return { lists: () => undefined, format, fromOpenedIndex: false };
}

/** A listing over a set of member paths (no leading slash). */
function listingOf(
  members: ReadonlySet<string>,
  format: zarr.StoreFormat | null,
  fromOpenedIndex: boolean
): RootListing {
  return { lists: (group) => members.has(group), format, fromOpenedIndex };
}

/**
 * Group member paths of a format-3 root `zarr.json`'s consolidated index (no
 * leading slash), or `null` when the document carries no index.
 *
 * Exported for tests.
 */
export function consolidatedGroupsOf(bytes: Uint8Array): Set<string> | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return null;
  }
  const consolidated = (parsed as { consolidated_metadata?: { metadata?: unknown } } | null)
    ?.consolidated_metadata?.metadata;
  if (consolidated === null || typeof consolidated !== 'object') return null;
  const groups = new Set<string>();
  for (const [path, meta] of Object.entries(consolidated as Record<string, unknown>)) {
    if ((meta as { node_type?: unknown } | null)?.node_type === 'group') {
      groups.add(path.replace(/^\/+/, ''));
    }
  }
  return groups;
}

/** Resolve `promise`, or `undefined` after `ms` — without cancelling it. */
function within<T>(promise: Promise<T>, ms: number): Promise<T | undefined> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(undefined), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      () => {
        clearTimeout(timer);
        resolve(undefined);
      }
    );
  });
}

/** Group paths the opened store's index lists (no leading slash), or `null`. */
async function openedIndexGroups(store: unknown): Promise<Set<string> | null> {
  if (!hasContentsMethod(store)) return null;
  try {
    const contents = await store.contents();
    return new Set(
      contents.filter((entry) => entry.kind === 'group').map((e) => e.path.replace(/^\/+/, ''))
    );
  } catch {
    return null;
  }
}

/**
 * Build the {@link RootListing} for a just-opened store.
 *
 * @param store - The store `openStore` returned (its index and format).
 * @param indexFresh - Whether that index came from the network this load.
 * @param rootDocument - The load-time root-document fetch, or `null` when the
 *   URL does not use one.
 */
export async function resolveRootListing(
  store: unknown,
  indexFresh: boolean,
  rootDocument: Promise<RootDocumentFetch> | null
): Promise<RootListing> {
  const openedFormat = zarr.storeFormatOf(store);
  if (indexFresh) {
    const groups = await openedIndexGroups(store);
    if (groups !== null) return listingOf(groups, openedFormat, true);
  }
  const fetched = rootDocument ? await within(rootDocument, ROOT_DOCUMENT_WAIT_MS) : undefined;
  return networkListing(fetched?.served ?? null, openedFormat);
}

/**
 * The listing the load-time root document supports: its consolidated index for
 * a format-3 `zarr.json`, else "unknown" with the best-known format.
 */
function networkListing(
  served: RootDocumentFetch['served'],
  openedFormat: zarr.StoreFormat | null
): RootListing {
  if (served === null) return unknownListing(openedFormat);
  const format = openedFormat ?? (served.doc === 'zarr.json' ? 'v3' : 'v2');
  const groups = served.doc === 'zarr.json' ? consolidatedGroupsOf(served.bytes) : null;
  return groups === null ? unknownListing(format) : listingOf(groups, format, false);
}
