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
 * Within a lane, waiters are served by PRIORITY CLASS — `demand` (a frame is
 * waiting on it), then `refinement` (a finer level of something already shown),
 * then `speculative` (the prefetcher guessing) — FIFO within a class, and
 * speculative requests never hold more than {@link MAX_SPECULATIVE_FETCH_SHARE}
 * of the lane. A queued request's class can be RAISED after it queued (a demand
 * caller joining a prefetch) through a {@link FetchPriorityCell}.
 *
 * The data lane is {@link MAX_CONCURRENT_CHUNK_FETCHES} wide by default and
 * {@link MAX_CONCURRENT_MULTIPLEXED_CHUNK_FETCHES} for an origin that resource
 * timing shows negotiated HTTP/2 or HTTP/3 (`nextHopProtocol`): there the
 * browser multiplexes streams on one connection and the 24-wide lane, not the
 * network, was the measured bottleneck before first frame.
 *
 * Perf counters per lane (`fetch.<lane>.requests`, `.highWater`,
 * `.queueWaitMs`) are tallied here; see `profiling/perf-counters.ts`.
 */

import { perfCounters, type PerfCounterSlot } from '../profiling/perf-counters';

/**
 * Maximum chunk responses in flight across every data path at once, for an
 * HTTP/1.1 or not-yet-identified origin.
 *
 * 24 keeps enough HTTP/2 streams ready to fill ordinary broadband while
 * bounding a representative 500 KiB chunk wave to about 12 MiB. Shared
 * globally — the cap is on total concurrency through response-body
 * consumption, not per store or per node. Metadata has its own lane (four
 * slots normally, two after a plain `http:` URL is seen).
 */
export const MAX_CONCURRENT_CHUNK_FETCHES = 24;

/**
 * Data-lane width for an origin known to multiplex (h2/h3).
 *
 * Measured on a hosted HTTP/2 simulation (100 ms RTT, 25 Mbps): at 24 the lane
 * sat pinned full before first frame; at 96 first frame improved 14% and
 * settle 15-21% (h2afva). This is also the GLOBAL ceiling of the data lane.
 */
export const MAX_CONCURRENT_MULTIPLEXED_CHUNK_FETCHES = 96;
export const MAX_CONCURRENT_METADATA_FETCHES = 4;

/**
 * Lane caps on an HTTP/1.1 origin: together they equal the six sockets every
 * current browser opens per origin, so an admitted request is a dispatched
 * request. Metadata keeps two sockets of its own so a scene-graph probe is not
 * stuck behind four chunk bodies.
 */
export const HTTP1_MAX_CONCURRENT_CHUNK_FETCHES = 4;
export const HTTP1_MAX_CONCURRENT_METADATA_FETCHES = 2;

/** Largest fraction of a lane that `speculative` requests may occupy. */
export const MAX_SPECULATIVE_FETCH_SHARE = 0.25;

export type FetchLane = 'data' | 'metadata';

/** Urgency class of a gated request, most urgent first. */
export type FetchPriority = 'demand' | 'refinement' | 'speculative';

const PRIORITY_RANK: Record<FetchPriority, number> = { demand: 0, refinement: 1, speculative: 2 };
const PRIORITY_CLASSES = 3;
const SPECULATIVE_RANK = PRIORITY_RANK.speculative;

/**
 * A request's priority that may be RAISED while it waits.
 *
 * The caching store coalesces same-key reads: a prefetch that queued as
 * `speculative` must jump ahead once a demand caller joins it, or a first-frame
 * chunk waits behind every other demand request at the speculative share.
 * Priorities only ever go up — a cell never demotes.
 */
export class FetchPriorityCell {
  #value: FetchPriority;
  #listeners: Array<() => void> = [];

  constructor(initial: FetchPriority) {
    this.#value = initial;
  }

  get value(): FetchPriority {
    return this.#value;
  }

  /** Raise to `to` if it is more urgent than the current class; no-op otherwise. */
  raise(to: FetchPriority): void {
    if (PRIORITY_RANK[to] >= PRIORITY_RANK[this.#value]) return;
    this.#value = to;
    for (const listener of this.#listeners.slice()) listener();
  }

  /** @internal Subscribe the gate to raises; returns the unsubscribe. */
  onRaise(listener: () => void): () => void {
    this.#listeners.push(listener);
    return () => {
      const at = this.#listeners.indexOf(listener);
      if (at >= 0) this.#listeners.splice(at, 1);
    };
  }
}

/** Priority classes carried on abort signals — see {@link tagSignalPriority}. */
const SIGNAL_PRIORITIES = new WeakMap<AbortSignal, FetchPriorityCell>();

/**
 * Give every gated read made under `signal` the priority `priority` (B9c).
 *
 * The abort signal is the one per-request value that already travels from a
 * loader's `updateView` through zarrita and the L0 chunk proxy down to the
 * store, so a whole class of reads (a refinement run's) can be classed without
 * threading a priority argument through every loader. The store, the L0
 * proxy's shared decode, and {@link boundedConcurrencyStore} consult
 * {@link signalPriority}; an untagged signal keeps its old default (`demand`).
 * The same pattern `tagSignalOrigin` uses for decode attribution.
 *
 * @returns The cell now attached to `signal` (an existing one is reused and
 *   raised, never lowered).
 */
export function tagSignalPriority(
  signal: AbortSignal,
  priority: FetchPriority | FetchPriorityCell
): FetchPriorityCell {
  const existing = SIGNAL_PRIORITIES.get(signal);
  if (priority instanceof FetchPriorityCell) {
    if (existing === undefined) SIGNAL_PRIORITIES.set(signal, priority);
    else existing.raise(priority.value);
    return existing ?? priority;
  }
  if (existing !== undefined) {
    existing.raise(priority);
    return existing;
  }
  const cell = new FetchPriorityCell(priority);
  SIGNAL_PRIORITIES.set(signal, cell);
  return cell;
}

/** The priority cell {@link tagSignalPriority} attached to `signal`, if any. */
export function signalPriority(
  signal: AbortSignal | null | undefined
): FetchPriorityCell | undefined {
  return signal ? SIGNAL_PRIORITIES.get(signal) : undefined;
}

/**
 * Growable ring-buffer FIFO: O(1) push and shift.
 *
 * `Array.prototype.shift` is O(n) — it re-indexes the whole backing store — and
 * a huge selection fan-out queues thousands of waiters, which measured as
 * 20-600 ms main-thread stalls in the release path alone.
 */
class Deque<T> {
  #items: (T | undefined)[] = new Array<T | undefined>(16);
  #head = 0;
  #size = 0;

  get length(): number {
    return this.#size;
  }

  push(item: T): void {
    if (this.#size === this.#items.length) this.#grow();
    this.#items[(this.#head + this.#size) % this.#items.length] = item;
    this.#size += 1;
  }

  peek(): T | undefined {
    return this.#size === 0 ? undefined : this.#items[this.#head];
  }

  shift(): T | undefined {
    if (this.#size === 0) return undefined;
    const item = this.#items[this.#head];
    this.#items[this.#head] = undefined;
    this.#head = (this.#head + 1) % this.#items.length;
    this.#size -= 1;
    return item;
  }

  #grow(): void {
    const next = new Array<T | undefined>(this.#items.length * 2);
    for (let i = 0; i < this.#size; i++) {
      next[i] = this.#items[(this.#head + i) % this.#items.length];
    }
    this.#items = next;
    this.#head = 0;
  }
}

/** One queued request. A promoted waiter leaves a dead entry behind. */
interface QueueEntry {
  readonly waiter: Waiter;
  live: boolean;
}

interface Waiter {
  readonly origin: string | undefined;
  readonly priority: FetchPriorityCell | FetchPriority;
  readonly start: () => void;
  entry: QueueEntry | undefined;
}

const ZARR_METADATA_KEYS = new Set(['zarr.json', '.zarray', '.zattrs', '.zgroup', '.zmetadata']);

interface FetchGateState {
  readonly lane: FetchLane;
  active: number;
  /**
   * The lane's cap before per-origin widening: the default width, or the
   * HTTP/1.1 socket cap once a plain `http:` URL has been seen.
   */
  limit: number;
  speculativeActive: number;
  readonly queues: Deque<QueueEntry>[];
  /** `fetch.<lane>.requests`: every gated call, queued or not. */
  readonly sRequests: PerfCounterSlot;
  /** `fetch.<lane>.highWater`: max leases in flight at once. */
  readonly sHighWater: PerfCounterSlot;
  /** `fetch.<lane>.queueWaitMs`: summed enqueue→start time of QUEUED calls. */
  readonly sQueueWaitMs: PerfCounterSlot;
}

function createGate(lane: FetchLane): FetchGateState {
  return {
    lane,
    active: 0,
    limit: lane === 'data' ? MAX_CONCURRENT_CHUNK_FETCHES : MAX_CONCURRENT_METADATA_FETCHES,
    speculativeActive: 0,
    queues: Array.from({ length: PRIORITY_CLASSES }, () => new Deque<QueueEntry>()),
    sRequests: perfCounters.slot(`fetch.${lane}.requests`),
    sHighWater: perfCounters.slot(`fetch.${lane}.highWater`),
    sQueueWaitMs: perfCounters.slot(`fetch.${lane}.queueWaitMs`),
  };
}

const fetchGates: Record<FetchLane, FetchGateState> = {
  data: createGate('data'),
  metadata: createGate('metadata'),
};

// ── Adaptive lane width: which origins multiplex ────────────────────────────

const multiplexedOrigins = new Set<string>();
let resourceObserverInstalled = false;

/**
 * Record the protocol an origin's responses arrived over.
 *
 * `h2` / `h3` (and their drafts) widen that origin's data lane; anything else
 * reported narrows it back. An EMPTY protocol — what resource timing reports
 * for a cross-origin response without `Timing-Allow-Origin` — is "cannot tell"
 * and changes nothing, so such a host stays on the safe default.
 */
export function noteOriginProtocol(origin: string, protocol: string | undefined): void {
  if (!protocol) return;
  if (/^h[23]\b|^h[23]-/.test(protocol)) multiplexedOrigins.add(origin);
  else multiplexedOrigins.delete(origin);
}

/** Forget every learned protocol (tests; a fresh session). */
export function resetOriginProtocols(): void {
  multiplexedOrigins.clear();
}

/** Parse a URL's origin, `undefined` when it has none we can key on. */
export function originOfUrl(url: string): string | undefined {
  try {
    const base = (globalThis as { location?: { href?: string } }).location?.href;
    const origin = new URL(url, base).origin;
    return origin === 'null' ? undefined : origin;
  } catch {
    return undefined;
  }
}

interface ResourceEntryLike {
  readonly name: string;
  readonly nextHopProtocol?: string;
}

function noteResourceEntries(entries: readonly unknown[]): void {
  for (const raw of entries) {
    const entry = raw as ResourceEntryLike;
    if (!entry.nextHopProtocol) continue;
    const origin = originOfUrl(entry.name);
    if (origin) noteOriginProtocol(origin, entry.nextHopProtocol);
  }
}

/**
 * Watch resource timing for negotiated protocols, once, lazily — on the first
 * gated request that names an origin. An observer rather than
 * `getEntriesByType` because the resource-timing BUFFER fills (250 entries by
 * default) long before a chunk-heavy load ends; observers still see each entry.
 */
function ensureResourceObserver(): void {
  if (resourceObserverInstalled) return;
  resourceObserverInstalled = true;
  const Observer = (globalThis as { PerformanceObserver?: typeof PerformanceObserver })
    .PerformanceObserver;
  if (typeof Observer !== 'function') return;
  try {
    const observer = new Observer((list) => noteResourceEntries(list.getEntries()));
    observer.observe({ type: 'resource', buffered: true });
  } catch {
    // No resource-timing support: every origin keeps the default lane.
  }
}

/** Set by {@link noteFetchUrl} once a plain `http:` (HTTP/1.1) URL is seen. */
let http1Origin = false;

function laneLimit(gate: FetchGateState, origin: string | undefined): number {
  // Once a plain `http:` URL is seen the whole gate is HTTP/1.1-capped (see
  // noteFetchUrl): widening any origin would park requests in the browser's
  // own socket queue with their header timers running.
  if (http1Origin || gate.lane === 'metadata') return gate.limit;
  return origin !== undefined && multiplexedOrigins.has(origin)
    ? MAX_CONCURRENT_MULTIPLEXED_CHUNK_FETCHES
    : gate.limit;
}

// ── Admission ───────────────────────────────────────────────────────────────

let fetchProgressEpoch = 0;

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

function rankOf(priority: FetchPriorityCell | FetchPriority): number {
  return PRIORITY_RANK[typeof priority === 'string' ? priority : priority.value];
}

function admissible(gate: FetchGateState, origin: string | undefined, rank: number): boolean {
  const limit = laneLimit(gate, origin);
  if (gate.active >= limit) return false;
  if (rank !== SPECULATIVE_RANK) return true;
  return gate.speculativeActive < Math.max(1, Math.floor(limit * MAX_SPECULATIVE_FETCH_SHARE));
}

/** Does any LIVE waiter of `rank` or more urgent sit in the queue? */
function hasQueuedAtOrAbove(gate: FetchGateState, rank: number): boolean {
  for (let r = 0; r <= rank; r++) {
    if (liveHead(gate, r)) return true;
  }
  return false;
}

/** Head of one class's queue, discarding entries left dead by a promotion. */
function liveHead(gate: FetchGateState, rank: number): QueueEntry | undefined {
  const queue = gate.queues[rank];
  let head = queue.peek();
  while (head && !head.live) {
    queue.shift();
    head = queue.peek();
  }
  return head;
}

function enqueue(gate: FetchGateState, waiter: Waiter): void {
  const entry: QueueEntry = { waiter, live: true };
  waiter.entry = entry;
  gate.queues[rankOf(waiter.priority)].push(entry);
}

/**
 * Start every queued waiter that may start now, most urgent class first.
 *
 * Each class is inspected at its HEAD only, so this is O(1) per start: a
 * speculative head blocked by its share does not stop a demand head, and a head
 * blocked by the lane width (mixed origins) stops only its own class.
 */
function dispatch(gate: FetchGateState): void {
  for (;;) {
    let started = false;
    for (let rank = 0; rank < PRIORITY_CLASSES && !started; rank++) {
      const head = liveHead(gate, rank);
      if (!head || !admissible(gate, head.waiter.origin, rank)) continue;
      gate.queues[rank].shift();
      head.live = false;
      head.waiter.entry = undefined;
      head.waiter.start();
      started = true;
    }
    if (!started) return;
  }
}

/**
 * p-limit-style gate: never lets more than the lane's width run concurrently.
 *
 * @param fn - The request; its lease is held until the returned promise settles.
 * @param lane - Data or metadata pool.
 * @param priority - Urgency class, or a {@link FetchPriorityCell} whose class
 *   may be raised while the request waits. Defaults to `demand`.
 * @param origin - The request's origin, which picks the data-lane width
 *   (multiplexed origins get the wide lane). Omitted = the default width.
 */
export function withFetchGate<T>(
  fn: () => Promise<T>,
  lane: FetchLane = 'data',
  priority: FetchPriority | FetchPriorityCell = 'demand',
  origin?: string
): Promise<T> {
  const gate = fetchGates[lane];
  perfCounters.add(gate.sRequests);
  if (origin !== undefined) ensureResourceObserver();

  // The class a lease is COUNTED under is fixed at start: a raise after that
  // changes nothing about the running request.
  let speculativeLease = false;
  const take = (): void => {
    gate.active += 1;
    speculativeLease = rankOf(priority) === SPECULATIVE_RANK;
    if (speculativeLease) gate.speculativeActive += 1;
    perfCounters.max(gate.sHighWater, gate.active);
  };

  let acquire: Promise<void>;
  const rank = rankOf(priority);
  if (!hasQueuedAtOrAbove(gate, rank) && admissible(gate, origin, rank)) {
    take();
    acquire = Promise.resolve();
  } else {
    const enqueuedAt = performance.now();
    acquire = new Promise<void>((resolve) => {
      let unsubscribe: (() => void) | undefined;
      const waiter: Waiter = {
        origin,
        priority,
        entry: undefined,
        start: () => {
          unsubscribe?.();
          take();
          perfCounters.add(gate.sQueueWaitMs, performance.now() - enqueuedAt);
          resolve();
        },
      };
      enqueue(gate, waiter);
      if (typeof priority !== 'string') {
        unsubscribe = priority.onRaise(() => {
          // Still queued: move to the tail of its new class and let it start
          // at once if that class has room (a speculative cap no longer binds).
          if (!waiter.entry?.live) return;
          waiter.entry.live = false;
          enqueue(gate, waiter);
          dispatch(gate);
        });
      }
    });
  }
  return acquire.then(async () => {
    try {
      return await fn();
    } finally {
      gate.active -= 1;
      if (speculativeLease) gate.speculativeActive -= 1;
      // Hand the freed slot to the most urgent waiter. dispatch() admits only
      // below laneLimit(), so after the cap shrinks, leases already in flight
      // drain before new ones start.
      dispatch(gate);
    }
  });
}

/**
 * The abort signal of a zarr store `get(key, opts)` / `getRange(key, range,
 * opts)` call: the last object argument carrying one.
 */
function signalOfStoreArgs(args: readonly unknown[]): AbortSignal | undefined {
  for (let i = args.length - 1; i >= 1; i--) {
    const signal = (args[i] as { signal?: unknown } | null | undefined)?.signal;
    if (signal instanceof AbortSignal) return signal;
  }
  return undefined;
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
              typeof args[0] === 'string' ? fetchLaneForKey(args[0]) : 'data',
              signalPriority(signalOfStoreArgs(args)) ?? 'demand'
            );
        }
      }
      return Reflect.get(target, prop, receiver);
    },
  }) as S;
}
