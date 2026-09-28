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
 * wait behind every active chunk body.
 *
 * The caps must never exceed what the browser will actually put on the wire,
 * because `fetch-retry.ts` starts each attempt's header timer when the gate
 * admits it. Over HTTP/1.1 a browser opens only six sockets per origin; a gate
 * wider than that parks the surplus in the browser's own invisible queue with
 * the timer already running. Under a steady stream (a large backdrop still
 * loading while the view changes) those parked requests time out, their
 * retries rejoin the same queue, and every load fails together. So once a
 * plain `http:` URL is seen, both lanes shrink to fit the socket pool (see
 * {@link noteFetchUrl}).
 *
 * Perf counters per lane (`fetch.<lane>.requests`, `.highWater`,
 * `.queueWaitMs`) are tallied here; see `profiling/perf-counters.ts`.
 */

import { perfCounters, type PerfCounterSlot } from '../profiling/perf-counters';

/**
 * Maximum chunk responses in flight across every data path at once.
 *
 * 24 keeps enough HTTP/2 streams ready to fill ordinary broadband while
 * bounding a representative 500 KiB chunk wave to about 12 MiB. Shared
 * globally — the cap is on total concurrency through response-body
 * consumption, not per store or per node. Metadata has its own lane (four
 * slots normally, two after a plain `http:` URL is seen).
 */
export const MAX_CONCURRENT_CHUNK_FETCHES = 24;
export const MAX_CONCURRENT_METADATA_FETCHES = 4;

/**
 * Lane caps on an HTTP/1.1 origin: together they equal the six sockets every
 * current browser opens per origin, so an admitted request is a dispatched
 * request. Metadata keeps two sockets of its own so a scene-graph probe is not
 * stuck behind four chunk bodies.
 */
export const HTTP1_MAX_CONCURRENT_CHUNK_FETCHES = 4;
export const HTTP1_MAX_CONCURRENT_METADATA_FETCHES = 2;

export type FetchLane = 'data' | 'metadata';

const ZARR_METADATA_KEYS = new Set(['zarr.json', '.zarray', '.zattrs', '.zgroup', '.zmetadata']);

interface FetchGateState {
  active: number;
  limit: number;
  readonly queue: Array<() => void>;
  /** `fetch.<lane>.requests`: every gated call, queued or not. */
  readonly sRequests: PerfCounterSlot;
  /** `fetch.<lane>.highWater`: max leases in flight at once. */
  readonly sHighWater: PerfCounterSlot;
  /** `fetch.<lane>.queueWaitMs`: summed enqueue→start time of QUEUED calls. */
  readonly sQueueWaitMs: PerfCounterSlot;
}

function createGate(lane: FetchLane, limit: number): FetchGateState {
  return {
    active: 0,
    limit,
    queue: [],
    sRequests: perfCounters.slot(`fetch.${lane}.requests`),
    sHighWater: perfCounters.slot(`fetch.${lane}.highWater`),
    sQueueWaitMs: perfCounters.slot(`fetch.${lane}.queueWaitMs`),
  };
}

const fetchGates: Record<FetchLane, FetchGateState> = {
  data: createGate('data', MAX_CONCURRENT_CHUNK_FETCHES),
  metadata: createGate('metadata', MAX_CONCURRENT_METADATA_FETCHES),
};

let fetchProgressEpoch = 0;
let http1Origin = false;

/**
 * Record a URL about to be fetched. Browsers speak HTTP/2 and HTTP/3 only over
 * TLS, so a plain `http:` URL is HTTP/1.1 for certain: the local kiosk and
 * `luxar serve`, on localhost or a LAN. The first one shrinks both lanes to the
 * HTTP/1.1 caps for the rest of the session; the gate is global, and a mixed
 * scene only pays a narrower gate on its TLS origins. An `https:` origin that
 * happens to be HTTP/1.1 is not detected and keeps the wide caps.
 */
export function noteFetchUrl(url: string): void {
  if (http1Origin) return;
  let protocol: string;
  try {
    protocol = new URL(url, globalThis.location?.href).protocol;
  } catch {
    return;
  }
  if (protocol !== 'http:') return;
  http1Origin = true;
  fetchGates.data.limit = HTTP1_MAX_CONCURRENT_CHUNK_FETCHES;
  fetchGates.metadata.limit = HTTP1_MAX_CONCURRENT_METADATA_FETCHES;
}

/** Current cap of one lane. */
export function getFetchLaneLimit(lane: FetchLane): number {
  return fetchGates[lane].limit;
}

/** Restore the wide caps between isolated tests. */
export function resetFetchTransport(): void {
  http1Origin = false;
  fetchGates.data.limit = MAX_CONCURRENT_CHUNK_FETCHES;
  fetchGates.metadata.limit = MAX_CONCURRENT_METADATA_FETCHES;
}

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
  perfCounters.add(gate.sRequests);
  let acquire: Promise<void>;
  if (gate.active < gate.limit) {
    gate.active += 1;
    perfCounters.max(gate.sHighWater, gate.active);
    acquire = Promise.resolve();
  } else {
    const enqueuedAt = performance.now();
    acquire = new Promise<void>((resolve) =>
      gate.queue.push(() => {
        gate.active += 1;
        perfCounters.max(gate.sHighWater, gate.active);
        perfCounters.add(gate.sQueueWaitMs, performance.now() - enqueuedAt);
        resolve();
      })
    );
  }
  return acquire.then(async () => {
    try {
      return await fn();
    } finally {
      gate.active -= 1;
      // Hand freed slots to waiters, but only below the cap: after the cap
      // shrinks, leases already in flight must drain before new ones start.
      while (gate.active < gate.limit && gate.queue.length > 0) gate.queue.shift()!();
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
