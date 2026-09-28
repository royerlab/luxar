/**
 * One load-time fetch of a dataset's root metadata document, shared by every
 * consumer that needs it.
 *
 * A cold load used to fetch the root document twice, back to back and strictly
 * after renderer init: once for cache validation (bypassing the cache tiers) and
 * again for the store open (through them). A CDN serving `max-age` answers the
 * second from the HTTP cache; a `no-store` host pays twice — 6.5 MB for the
 * h2afva scene. And nothing left the tab until the renderer had linked its
 * shaders.
 *
 * This module fixes both. {@link prefetchRootDocument} starts the fetch the
 * moment the `?src=` URL is known (bootstrap, before `app.init`), in parallel with
 * renderer init; the scene load then {@link claimRootDocument claims} that
 * in-flight response and hands it to BOTH readers through
 * {@link SharedRootDocumentSource}:
 *
 * - **Validation** reads its token from it (`probeIdentityToken`). The fetch
 *   bypasses every Luxar cache tier exactly as validation always has, and the
 *   token is derived by the same `validationTokenFromDocument`, so content-hash
 *   semantics are unchanged. Validation keeps its fail-fast budget: it waits for
 *   the shared response at most `timeoutMsOverride`, then takes the TTL path as
 *   before while the fetch carries on for the store open.
 * - **The store open** reads the same bytes as the source's answer for the root
 *   key, so they flow through the caching store's L1/L2 tiers like any fetched
 *   document (a later offline visit still finds them in OPFS).
 *
 * The response's `ETag` is kept too: the scene-identity watchdog seeds its first
 * poll's `If-None-Match` with it (see `SceneIdentityWatchdog.seedFromLoad`).
 *
 * Sharing only applies to a plain directory store over http(s): a `.zarr.zip`
 * keeps its root inside the archive, and a URL with a query string or fragment
 * (a presigned source) needs the query-preserving URL building the stores do
 * themselves — both fall through to the normal read path untouched.
 *
 * @module cache/root-document-prefetch
 */

import type { ChunkFetchOutcome, ChunkSource, ChunkSourceGetOptions } from './chunk-source';
import type { AbsolutePath, AsyncReadable, GetOptions } from '../data/zarr';
import { ROOT_ATTR_DOCS } from '../types/zarr-documents';
import { buildUrl, fetchWithRetry } from './multi-level-caching-store/fetch-retry';
import {
  validationTokenFromDocument,
  type RemoteValidationToken,
} from './multi-level-caching-store/validation-queue';

/** One of the root metadata documents, newest format first. */
export type RootDocName = (typeof ROOT_ATTR_DOCS)[number];

/** What the server said about one root document. */
export type RootDocOutcome =
  | { kind: 'ok'; bytes: Uint8Array<ArrayBuffer>; etag: string | null }
  /** A definitive "not here" (404, or the 403/410 a bucket answers for a missing key). */
  | { kind: 'missing' }
  /** Anything else — a 5xx, or retries exhausted. Says nothing about the document. */
  | { kind: 'error' };

/** The settled load-time fetch. */
export interface RootDocumentFetch {
  /** Per-document outcome, in request order. A document never requested is absent. */
  readonly outcomes: ReadonlyMap<RootDocName, RootDocOutcome>;
  /** The document that answered 2xx, or `null` when none did. */
  readonly served: { doc: RootDocName; bytes: Uint8Array<ArrayBuffer>; etag: string | null } | null;
  /**
   * The validation token for {@link served}, derived once and memoised; `null`
   * when nothing was served or the body is not JSON.
   */
  token(): Promise<RemoteValidationToken | null>;
  /** The token if {@link token} has already resolved, else `undefined`. */
  peekToken(): RemoteValidationToken | null | undefined;
}

/**
 * How long an UNCLAIMED bootstrap prefetch stays eligible. The load claims it
 * within milliseconds of bootstrap; past this, the page is doing something else
 * (dataset browser, a failed init) and a load starting now fetches afresh.
 */
const MAX_UNCLAIMED_AGE_MS = 30_000;

interface Entry {
  promise: Promise<RootDocumentFetch>;
  abort: AbortController;
  startedAt: number;
  claimed: boolean;
}

const entries = new Map<string, Entry>();

/** Map key: the base URL without trailing slashes (`normalizeURL` adds one). */
function keyOf(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, '');
}

/**
 * Whether a dataset URL's root document can be fetched and shared here: an
 * absolute http(s) directory store with no query string or fragment.
 */
export function isShareableRootUrl(url: string): boolean {
  if (!/^https?:\/\//i.test(url)) return false;
  if (url.includes('?') || url.includes('#')) return false;
  // A zipped store (`isZippedStoreUrl`, which lives in the data layer this one
  // may not import): its root document is inside the archive.
  return !url.toLowerCase().endsWith('.zip');
}

function makeFetch(
  outcomes: Map<RootDocName, RootDocOutcome>,
  served: RootDocumentFetch['served']
): RootDocumentFetch {
  let token: Promise<RemoteValidationToken | null> | null = null;
  let settledToken: RemoteValidationToken | null | undefined;
  return {
    outcomes,
    served,
    token() {
      if (!token) {
        token = (
          served
            ? validationTokenFromDocument(served.bytes, served.doc).catch(() => null)
            : Promise.resolve(null)
        ).then((value) => (settledToken = value));
      }
      return token;
    },
    peekToken: () => settledToken,
  };
}

async function fetchOne(url: string, signal: AbortSignal): Promise<RootDocOutcome> {
  try {
    const outcome = await fetchWithRetry(
      url,
      { signal, lane: 'metadata' },
      async ({ response, readBody }): Promise<RootDocOutcome> => {
        if (response.ok) {
          return { kind: 'ok', bytes: await readBody(), etag: response.headers.get('etag') };
        }
        // The same statuses the HTTP chunk source reads as "missing" for a
        // metadata key, so serving this verdict to the store changes nothing.
        if (response.status === 404 || response.status === 403 || response.status === 410) {
          return { kind: 'missing' };
        }
        return { kind: 'error' };
      }
    );
    return outcome ?? { kind: 'error' };
  } catch {
    return { kind: 'error' };
  }
}

/** Fetch `zarr.json`, then `.zattrs` if it was not served — the order validation uses. */
async function fetchRootDocument(baseUrl: string, signal: AbortSignal): Promise<RootDocumentFetch> {
  const outcomes = new Map<RootDocName, RootDocOutcome>();
  for (const doc of ROOT_ATTR_DOCS) {
    const outcome = await fetchOne(buildUrl(baseUrl, doc), signal);
    outcomes.set(doc, outcome);
    if (outcome.kind === 'ok') {
      return makeFetch(outcomes, { doc, bytes: outcome.bytes, etag: outcome.etag });
    }
    if (signal.aborted) break;
  }
  return makeFetch(outcomes, null);
}

function start(baseUrl: string, claimed: boolean): Entry {
  const abort = new AbortController();
  const entry: Entry = {
    promise: fetchRootDocument(baseUrl, abort.signal),
    abort,
    startedAt: Date.now(),
    claimed,
  };
  const key = keyOf(baseUrl);
  entries.set(key, entry);
  if (!claimed) {
    // Nobody claimed it (init failed, the dataset browser opened): let the bytes go.
    setTimeout(() => {
      if (entries.get(key) === entry && !entry.claimed) entries.delete(key);
    }, MAX_UNCLAIMED_AGE_MS);
  }
  return entry;
}

/**
 * Start fetching the root document for `baseUrl` now, so a load that follows
 * finds it in flight. Idempotent while an unclaimed fetch is pending; a no-op
 * for a URL {@link isShareableRootUrl} rejects.
 */
export function prefetchRootDocument(baseUrl: string): void {
  if (!isShareableRootUrl(baseUrl)) return;
  const existing = entries.get(keyOf(baseUrl));
  if (existing && !existing.claimed && Date.now() - existing.startedAt < MAX_UNCLAIMED_AGE_MS) {
    return;
  }
  start(baseUrl, false);
}

/**
 * Take the root-document fetch for a load of `baseUrl`: the unclaimed prefetch
 * when one is fresh, else a new fetch started now (a previous load's response is
 * never reused — each load validates against the server afresh). `null` for a URL
 * {@link isShareableRootUrl} rejects.
 */
export function claimRootDocument(baseUrl: string): Promise<RootDocumentFetch> | null {
  if (!isShareableRootUrl(baseUrl)) return null;
  const existing = entries.get(keyOf(baseUrl));
  if (existing && !existing.claimed && Date.now() - existing.startedAt < MAX_UNCLAIMED_AGE_MS) {
    existing.claimed = true;
    return existing.promise;
  }
  return start(baseUrl, true).promise;
}

/**
 * Drop the registry's reference to `promise` (the load is done with it). Other
 * holders keep theirs; only the registry entry goes, and only if it is still
 * this fetch.
 */
export function releaseRootDocument(baseUrl: string, promise: Promise<RootDocumentFetch>): void {
  const key = keyOf(baseUrl);
  if (entries.get(key)?.promise === promise) entries.delete(key);
}

/** Forget every entry, aborting any fetch still in flight (tests only). */
export function resetRootDocumentPrefetchForTests(): void {
  for (const entry of entries.values()) entry.abort.abort();
  entries.clear();
}

/**
 * Resolve `promise`, or `undefined` once `signal` aborts or `timeoutMs` passes —
 * without cancelling the shared work behind it.
 */
function settleWithin<T>(
  promise: Promise<T>,
  signal?: AbortSignal,
  timeoutMs?: number
): Promise<T | undefined> {
  if (signal?.aborted) return Promise.resolve(undefined);
  return new Promise<T | undefined>((resolve) => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const finish = (value: T | undefined): void => {
      if (timer !== undefined) clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
      resolve(value);
    };
    const onAbort = (): void => finish(undefined);
    signal?.addEventListener('abort', onAbort, { once: true });
    if (timeoutMs !== undefined)
      timer = setTimeout(() => finish(undefined), Math.max(1, timeoutMs));
    promise.then(finish, () => finish(undefined));
  });
}

/** The root document a store key names, or `undefined` for any other key. */
function rootDocOfKey(key: string): RootDocName | undefined {
  const bare = key.replace(/^\/+/, '');
  return ROOT_ATTR_DOCS.find((doc) => doc === bare);
}

/**
 * The shared fetch's verdict for a root key: bytes (a private copy — the store
 * may hand them on), a definitive miss, or `undefined` to read normally.
 */
async function sharedVerdictFor(
  shared: Promise<RootDocumentFetch>,
  doc: RootDocName,
  signal?: AbortSignal
): Promise<{ kind: 'ok'; data: Uint8Array } | { kind: 'missing' } | undefined> {
  const result = await settleWithin(shared, signal);
  const outcome = result?.outcomes.get(doc);
  if (outcome?.kind === 'ok') return { kind: 'ok', data: outcome.bytes.slice() };
  if (outcome?.kind === 'missing') return { kind: 'missing' };
  return undefined;
}

/**
 * A {@link ChunkSource} that answers the root document from the shared
 * load-time fetch and delegates everything else to `inner`.
 *
 * Serves each root document ONCE: after that the caching store holds it in its
 * tiers, and a re-read (a later invalidation) must reach the server. Validation
 * is answered from the same fetch until {@link release}.
 */
export class SharedRootDocumentSource implements ChunkSource {
  private shared: Promise<RootDocumentFetch> | null;
  private readonly served = new Set<RootDocName>();

  constructor(
    private readonly inner: ChunkSource,
    shared: Promise<RootDocumentFetch>
  ) {
    this.shared = shared;
  }

  /** The inner source's identity verbatim — the OPFS bucket must not move. */
  get identity(): string {
    return this.inner.identity;
  }

  get describe(): string {
    return this.inner.describe;
  }

  async get(
    key: string,
    signal?: AbortSignal,
    options?: ChunkSourceGetOptions
  ): Promise<ChunkFetchOutcome> {
    const doc = rootDocOfKey(key);
    const shared = this.shared;
    if (doc !== undefined && shared !== null && !this.served.has(doc)) {
      const verdict = await sharedVerdictFor(shared, doc, signal);
      if (signal?.aborted) return { kind: 'aborted' };
      if (verdict) {
        this.served.add(doc);
        return verdict.kind === 'ok'
          ? { kind: 'ok', data: verdict.data, bytesOverWire: verdict.data.byteLength }
          : { kind: 'missing' };
      }
    }
    // `options` carries the fetch-gate priority; dropping it would demote every
    // prefetch/refinement read to `demand`.
    return this.inner.get(key, signal, options);
  }

  async probeIdentityToken(options: {
    signal?: AbortSignal;
    timeoutMsOverride?: number;
  }): Promise<RemoteValidationToken | null> {
    const shared = this.shared;
    if (shared === null) return this.inner.probeIdentityToken(options);
    const result = await settleWithin(shared, options.signal, options.timeoutMsOverride);
    return result ? result.token() : null;
  }

  /** Stop answering from the shared fetch (the load is past it); frees the bytes. */
  release(): void {
    this.shared = null;
  }

  dispose(): void {
    this.shared = null;
    this.inner.dispose();
  }
}

/**
 * The same sharing for the cache-less path (`?noCache`), over a plain zarr
 * store: the first read of each root document is answered from the shared fetch.
 */
export function withSharedRootDocument(
  store: AsyncReadable,
  shared: Promise<RootDocumentFetch>
): AsyncReadable & { release(): void } {
  let pending: Promise<RootDocumentFetch> | null = shared;
  const served = new Set<RootDocName>();
  const wrapper: AsyncReadable & { release(): void } = {
    async get(key: AbsolutePath, options?: GetOptions): Promise<Uint8Array | undefined> {
      const doc = rootDocOfKey(key);
      const current = pending;
      if (doc !== undefined && current !== null && !served.has(doc)) {
        const signal = (options as { signal?: AbortSignal } | undefined)?.signal;
        const verdict = await sharedVerdictFor(current, doc, signal);
        if (verdict) {
          served.add(doc);
          return verdict.kind === 'ok' ? verdict.data : undefined;
        }
      }
      return store.get(key, options);
    },
    release() {
      pending = null;
    },
  };
  // Keep a ranged reader reachable (zarrita probes for `getRange`).
  const ranged = store as AsyncReadable & { getRange?: (...args: never[]) => unknown };
  if (typeof ranged.getRange === 'function') {
    (wrapper as typeof ranged).getRange = ranged.getRange.bind(store);
  }
  return wrapper;
}
