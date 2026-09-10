/**
 * Global bounded-concurrency gate for chunk network fetches.
 *
 * zarrita's `get` fans out one `fetch()` per chunk of a selection via an
 * internal `Promise.all`. A single visible range over a multi-million-element
 * LOD level spans thousands of chunks, so an unthrottled load fires thousands
 * of simultaneous requests and exhausts the browser
 * (`net::ERR_INSUFFICIENT_RESOURCES`) — the ceiling that capped large
 * substitutive-LOD scenes (~10M+ finest points).
 *
 * All data-fetch paths funnel through {@link withFetchGate}: the multi-level
 * caching store's network tier (`fetch-retry.ts`, the default), zipped-store
 * range reads (`range-reader.ts`), and the no-cache `FetchStore`
 * (`data/zarr.ts`). Shared lane counters cap the total in flight, keeping
 * throughput high while staying within the browser's socket/memory budget.
 *
 * Data and metadata use separate lanes. Body-sized chunk reads stay bounded,
 * and on multiplexed transports a root `zarr.json` / `.zattrs` probe does not
 * wait behind every active chunk body. HTTP/1.1 can still queue both lanes on
 * the browser's smaller per-origin socket pool.
 */

/**
 * Maximum chunk responses in flight across every data path at once.
 *
 * 24 keeps enough HTTP/2 streams ready to fill ordinary broadband while
 * bounding a representative 500 KiB chunk wave to about 12 MiB. Shared
 * globally — the cap is on total concurrency through response-body
 * consumption, not per store or per node. Metadata has its own four-slot lane.
 */
export const MAX_CONCURRENT_CHUNK_FETCHES = 24;
export const MAX_CONCURRENT_METADATA_FETCHES = 4;

export type FetchLane = 'data' | 'metadata';

const ZARR_METADATA_KEYS = new Set(['zarr.json', '.zarray', '.zattrs', '.zgroup', '.zmetadata']);

interface FetchGateState {
  active: number;
  readonly limit: number;
  readonly queue: Array<() => void>;
}

const fetchGates: Record<FetchLane, FetchGateState> = {
  data: { active: 0, limit: MAX_CONCURRENT_CHUNK_FETCHES, queue: [] },
  metadata: { active: 0, limit: MAX_CONCURRENT_METADATA_FETCHES, queue: [] },
};

let fetchProgressEpoch = 0;

export function fetchLaneForKey(key: string): FetchLane {
  const basename = key.slice(key.lastIndexOf('/') + 1);
  return ZARR_METADATA_KEYS.has(basename) ? 'metadata' : 'data';
}

/** Record response-body progress shared by all live fetch leases. */
export function noteFetchProgress(): void {
  fetchProgressEpoch += 1;
}

/** Monotonic aggregate-progress snapshot used by body-stall watchdogs. */
export function getFetchProgressEpoch(): number {
  return fetchProgressEpoch;
}

/** Reset aggregate progress state between isolated tests. */
export function resetFetchProgressEpoch(): void {
  fetchProgressEpoch = 0;
}

/** Current leases in one lane, including the caller while its body is read. */
export function getActiveFetchCount(lane: FetchLane): number {
  return fetchGates[lane].active;
}

/** p-limit-style gate: never lets more than the cap run concurrently. */
export function withFetchGate<T>(fn: () => Promise<T>, lane: FetchLane = 'data'): Promise<T> {
  const gate = fetchGates[lane];
  const acquire =
    gate.active < gate.limit
      ? ((gate.active += 1), Promise.resolve())
      : new Promise<void>((resolve) =>
          gate.queue.push(() => {
            gate.active += 1;
            resolve();
          })
        );
  return acquire.then(async () => {
    try {
      return await fn();
    } finally {
      gate.active -= 1;
      const next = gate.queue.shift(); // hand the freed slot to the next waiter
      if (next) next();
    }
  });
}

/**
 * Wrap a zarr store so its `get` / `getRange` go through {@link withFetchGate}.
 * A `Proxy` keeps the store's full surface intact (consolidated-metadata and
 * caching wrappers, `contents()`, etc.) while throttling only the two
 * data-fetching methods.
 */
export function boundedConcurrencyStore<S extends object>(store: S): S {
  return new Proxy(store, {
    get(target, prop, receiver) {
      if (prop === 'get' || prop === 'getRange') {
        const orig = Reflect.get(target, prop, receiver);
        if (typeof orig === 'function') {
          return (...args: unknown[]) =>
            withFetchGate(
              () => (orig as (...a: unknown[]) => Promise<unknown>).apply(target, args),
              typeof args[0] === 'string' ? fetchLaneForKey(args[0]) : 'data'
            );
        }
      }
      return Reflect.get(target, prop, receiver);
    },
  }) as S;
}
