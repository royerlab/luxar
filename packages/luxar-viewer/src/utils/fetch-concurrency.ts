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
 * plain `http:` URL is seen, THAT ORIGIN's share of both lanes shrinks to fit
 * its socket pool (see {@link noteFetchUrl}); other origins keep their width.
 *
 * Within a lane, waiters are served by PRIORITY CLASS — `demand` (a frame is
 * waiting on it), then `refinement` (a finer level of something already shown),
 * then `speculative` (the prefetcher guessing) — FIFO within a class, and
 * speculative requests never hold more than `speculativeShare` of the lane. A queued request's class can be RAISED after it queued (a demand
 * caller joining a prefetch) through a {@link FetchPriorityCell}, and a request
 * whose caller aborts while it waits leaves the queue at once.
 *
 * Each class is queued per lane WIDTH, with a separate FIFO for each
 * HTTP/1.1 origin. A saturated origin cannot hold back another origin with
 * free sockets; FIFO holds within each origin and across admissible heads by
 * enqueue order.
 *
 * The data lane is `maxChunkFetches` (24) wide by default and
 * `maxMultiplexedChunkFetches` (96) for an origin that resource timing shows
 * negotiated HTTP/2 or HTTP/3 (`nextHopProtocol`): there the browser
 * multiplexes streams on one connection and the 24-wide lane, not the network,
 * was the measured bottleneck before first frame.
 *
 * Every width lives in `config.dataLoading.network.fetchGate` (validated, read
 * at each admission).
 *
 * Perf counters per lane (`fetch.<lane>.requests`, `.highWater`,
 * `.queueWaitMs`) are tallied here; see `profiling/perf-counters.ts`.
 */

import { config } from '../config';
import type { FetchGateConfig } from '../config/sections/data-loading/network/types';
import { perfCounters, type PerfCounterSlot } from '../profiling/perf-counters';

/** The gate's lane widths, read live so a test or tuning can change them. */
function gateConfig(): FetchGateConfig {
  return config.dataLoading.network.fetchGate;
}

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

/** One queued request. A promoted or aborted waiter leaves a dead entry behind. */
interface QueueEntry {
  readonly waiter: Waiter;
  /** Enqueue order, for FIFO across a class's width tiers. */
  readonly seq: number;
  live: boolean;
}

interface Waiter {
  readonly origin: string | undefined;
  readonly priority: FetchPriorityCell | FetchPriority;
  readonly start: () => void;
  entry: QueueEntry | undefined;
}

/**
 * Width tiers per class: 0 = an HTTP/1.1 origin (its socket cap), 1 = the
 * lane's default width, 2 = an origin known to multiplex (the wide data lane).
 * Decided at enqueue; admission re-checks the origin's live width, so a tier
 * only decides who may be passed, never a cap. A request never needs to wait
 * behind another tier's head: a head blocked by a NARROWER width is no reason
 * to hold a wider request, and one blocked by a WIDER width means the lane is
 * too full for the narrower request too.
 */
const WIDTH_TIERS = 3;

const ZARR_METADATA_KEYS = new Set(['zarr.json', '.zarray', '.zattrs', '.zgroup', '.zmetadata']);

/** Leases one origin holds in a lane (kept only while it holds any). */
interface OriginLeases {
  active: number;
  speculative: number;
}

interface FetchGateState {
  readonly lane: FetchLane;
  active: number;
  /** The lane's default width: its global cap for every origin not known to multiplex. */
  limit(): number;
  /** The per-origin cap of an HTTP/1.1 origin (see {@link noteFetchUrl}). */
  http1Limit(): number;
  speculativeActive: number;
  /** Live leases per origin, so an HTTP/1.1 origin is held to its own sockets. */
  readonly originLeases: Map<string, OriginLeases>;
  /** One FIFO per (priority class, width tier): index `rank * WIDTH_TIERS + tier`. */
  readonly queues: Deque<QueueEntry>[];
  /** HTTP/1.1 FIFO per origin and priority, so one saturated origin cannot block another. */
  readonly http1Queues: Map<string, Deque<QueueEntry>[]>;
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
    limit: () => (lane === 'data' ? gateConfig().maxChunkFetches : gateConfig().maxMetadataFetches),
    http1Limit: () =>
      lane === 'data' ? gateConfig().http1MaxChunkFetches : gateConfig().http1MaxMetadataFetches,
    speculativeActive: 0,
    originLeases: new Map(),
    queues: Array.from({ length: PRIORITY_CLASSES * WIDTH_TIERS }, () => new Deque<QueueEntry>()),
    http1Queues: new Map(),
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

/** Origins {@link noteFetchUrl} has seen over plain `http:` (HTTP/1.1 for certain). */
const http1Origins = new Set<string>();

function isHttp1(origin: string | undefined): origin is string {
  return origin !== undefined && http1Origins.has(origin);
}

/**
 * The lane's GLOBAL in-flight ceiling for a request from `origin`: the wide
 * lane for a multiplexed origin's data, the default width otherwise. An
 * HTTP/1.1 origin is additionally held to its own socket cap
 * ({@link FetchGateState.http1Limit}).
 */
function laneLimit(gate: FetchGateState, origin: string | undefined): number {
  if (gate.lane === 'metadata' || isHttp1(origin)) return gate.limit();
  return origin !== undefined && multiplexedOrigins.has(origin)
    ? gateConfig().maxMultiplexedChunkFetches
    : gate.limit();
}

/** How many of this lane's requests `origin` may have in flight at once. */
function originWidth(gate: FetchGateState, origin: string | undefined): number {
  return isHttp1(origin) ? gate.http1Limit() : laneLimit(gate, origin);
}

/** Speculative leases allowed out of a width of `limit`. */
function speculativeCap(limit: number): number {
  return Math.max(1, Math.floor(limit * gateConfig().speculativeShare));
}

// ── Admission ───────────────────────────────────────────────────────────────

let fetchProgressEpoch = 0;

/**
 * Record a URL about to be fetched. Browsers speak HTTP/2 and HTTP/3 only over
 * TLS, so a plain `http:` URL is HTTP/1.1 for certain: the local kiosk and
 * `luxar serve`, on localhost or a LAN. That ORIGIN is then held to the
 * HTTP/1.1 caps — its data and metadata leases together fit its six sockets —
 * for the rest of the session, while every other origin keeps its own width
 * (a mixed scene's TLS CDN is not narrowed by a LAN backdrop). An `https:`
 * origin that happens to be HTTP/1.1 is not detected and keeps the wide caps.
 */
export function noteFetchUrl(url: string): void {
  let parsed: URL;
  try {
    parsed = new URL(url, globalThis.location?.href);
  } catch {
    return;
  }
  if (parsed.protocol !== 'http:') return;
  http1Origins.add(parsed.origin);
}

/**
 * Current cap of one lane for requests from `origin`: the HTTP/1.1 socket cap
 * for an origin seen over plain `http:`, the wide lane for a multiplexed
 * origin's data, else the default width. Omitted = an origin-less request.
 */
export function getFetchLaneLimit(lane: FetchLane, origin?: string): number {
  return originWidth(fetchGates[lane], origin);
}

/** Forget every HTTP/1.1 origin between isolated tests. */
export function resetFetchTransport(): void {
  http1Origins.clear();
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

/** Does an HTTP/1.1 origin still have a socket (and, if speculative, its share) free? */
function originAdmits(gate: FetchGateState, origin: string, rank: number): boolean {
  const leases = gate.originLeases.get(origin);
  if (leases === undefined) return true;
  const cap = gate.http1Limit();
  if (leases.active >= cap) return false;
  return rank !== SPECULATIVE_RANK || leases.speculative < speculativeCap(cap);
}

function admissible(gate: FetchGateState, origin: string | undefined, rank: number): boolean {
  const limit = laneLimit(gate, origin);
  if (gate.active >= limit) return false;
  if (isHttp1(origin) && !originAdmits(gate, origin, rank)) return false;
  return rank !== SPECULATIVE_RANK || gate.speculativeActive < speculativeCap(limit);
}

/** Width tier of a request from `origin` (see {@link WIDTH_TIERS}). */
function tierOf(gate: FetchGateState, origin: string | undefined): number {
  if (isHttp1(origin)) return 0;
  return laneLimit(gate, origin) > gate.limit() ? 2 : 1;
}

/** Does any LIVE waiter of `rank` or more urgent sit in `tier`'s queues? */
function hasQueuedAtOrAbove(
  gate: FetchGateState,
  rank: number,
  tier: number,
  origin?: string
): boolean {
  const queues = tier === 0 && origin ? gate.http1Queues.get(origin) : undefined;
  for (let r = 0; r <= rank; r++) {
    if (liveQueueHead(queues?.[r] ?? gate.queues[r * WIDTH_TIERS + tier])) return true;
  }
  return false;
}

/** Count a lease against its origin (`delta` +1 on start, -1 on release). */
function tallyOriginLease(
  gate: FetchGateState,
  origin: string | undefined,
  speculative: boolean,
  delta: 1 | -1
): void {
  if (origin === undefined) return;
  let leases = gate.originLeases.get(origin);
  if (leases === undefined) {
    if (delta < 0) return;
    leases = { active: 0, speculative: 0 };
    gate.originLeases.set(origin, leases);
  }
  leases.active += delta;
  if (speculative) leases.speculative += delta;
  if (leases.active <= 0) gate.originLeases.delete(origin);
}

/** Head of one queue, discarding entries left dead by a promotion or an abort. */
function liveQueueHead(queue: Deque<QueueEntry>): QueueEntry | undefined {
  let head = queue.peek();
  while (head && !head.live) {
    queue.shift();
    head = queue.peek();
  }
  return head;
}

let nextSeq = 0;

function enqueue(gate: FetchGateState, waiter: Waiter): void {
  const entry: QueueEntry = { waiter, seq: nextSeq++, live: true };
  waiter.entry = entry;
  const index = rankOf(waiter.priority) * WIDTH_TIERS + tierOf(gate, waiter.origin);
  if (isHttp1(waiter.origin)) {
    let queues = gate.http1Queues.get(waiter.origin);
    if (!queues) {
      queues = Array.from({ length: PRIORITY_CLASSES }, () => new Deque<QueueEntry>());
      gate.http1Queues.set(waiter.origin, queues);
    }
    queues[rankOf(waiter.priority)].push(entry);
  } else {
    gate.queues[index].push(entry);
  }
}

/** The earliest admissible queue head in this priority class. */
function admissibleHead(gate: FetchGateState, rank: number): Deque<QueueEntry> | undefined {
  let best: Deque<QueueEntry> | undefined;
  let bestSeq = Infinity;
  for (let tier = 1; tier < WIDTH_TIERS; tier++) {
    const index = rank * WIDTH_TIERS + tier;
    const queue = gate.queues[index];
    const head = liveQueueHead(queue);
    if (head && head.seq < bestSeq && admissible(gate, head.waiter.origin, rank)) {
      best = queue;
      bestSeq = head.seq;
    }
  }
  for (const [origin, queues] of gate.http1Queues) {
    const queue = queues[rank];
    const head = liveQueueHead(queue);
    if (head && head.seq < bestSeq && admissible(gate, origin, rank)) {
      best = queue;
      bestSeq = head.seq;
    }
    if (queues.every((candidate) => !liveQueueHead(candidate))) gate.http1Queues.delete(origin);
  }
  return best;
}

/**
 * Start every queued waiter that may start now, most urgent class first.
 *
 * Each queue is inspected at its HEAD only, scanning the active HTTP/1.1
 * origins once per start: a
 * speculative head blocked by its share does not stop a demand head, and a head
 * blocked by its origin's lane width (mixed origins) stops only its own width
 * tier — a multiplexed origin's waiter of the same class still starts.
 */
function dispatch(gate: FetchGateState): void {
  for (;;) {
    let started = false;
    for (let rank = 0; rank < PRIORITY_CLASSES && !started; rank++) {
      const head = admissibleHead(gate, rank)?.shift();
      if (!head) continue;
      head.live = false;
      head.waiter.entry = undefined;
      head.waiter.start();
      started = true;
    }
    if (!started) return;
  }
}

/** Why a waiter left the queue unstarted: the signal's reason, or an AbortError. */
function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException('Fetch gate wait aborted', 'AbortError');
}

/** Still queued: move to the tail of its (raised) class and start it if there is room. */
function requeue(gate: FetchGateState, waiter: Waiter): void {
  if (!waiter.entry?.live) return;
  waiter.entry.live = false;
  enqueue(gate, waiter);
  dispatch(gate);
}

/**
 * Queue a request until {@link dispatch} starts it (`onStart` takes the lease),
 * or reject once `signal` aborts first — leaving a dead entry, not a queue place.
 */
function waitForSlot(
  gate: FetchGateState,
  origin: string | undefined,
  priority: FetchPriority | FetchPriorityCell,
  signal: AbortSignal | undefined,
  onStart: () => void
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const cleanups: Array<() => void> = [];
    const settle = (): void => cleanups.forEach((cleanup) => cleanup());
    const waiter: Waiter = {
      origin,
      priority,
      entry: undefined,
      start: () => {
        settle();
        onStart();
        resolve();
      },
    };
    enqueue(gate, waiter);
    // A speculative cap may no longer bind once raised: see requeue().
    if (typeof priority !== 'string') cleanups.push(priority.onRaise(() => requeue(gate, waiter)));
    if (!signal) return;
    const onAbort = (): void => {
      if (!waiter.entry?.live) return;
      waiter.entry.live = false;
      waiter.entry = undefined;
      settle();
      reject(abortReason(signal));
    };
    signal.addEventListener('abort', onAbort, { once: true });
    cleanups.push(() => signal.removeEventListener('abort', onAbort));
  });
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
 * @param signal - The caller's abort signal: aborting while the request still
 *   waits for a slot rejects it with the signal's reason and frees its queue
 *   place. Once started, `fn` owns the signal.
 */
export function withFetchGate<T>(
  fn: () => Promise<T>,
  lane: FetchLane = 'data',
  priority: FetchPriority | FetchPriorityCell = 'demand',
  origin?: string,
  signal?: AbortSignal
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
    tallyOriginLease(gate, origin, speculativeLease, 1);
    perfCounters.max(gate.sHighWater, gate.active);
  };

  let acquire: Promise<void>;
  const rank = rankOf(priority);
  const tier = tierOf(gate, origin);
  if (signal?.aborted) {
    acquire = Promise.reject(abortReason(signal));
  } else if (!hasQueuedAtOrAbove(gate, rank, tier, origin) && admissible(gate, origin, rank)) {
    take();
    acquire = Promise.resolve();
  } else {
    const enqueuedAt = performance.now();
    acquire = waitForSlot(gate, origin, priority, signal, () => {
      take();
      perfCounters.add(gate.sQueueWaitMs, performance.now() - enqueuedAt);
    });
  }
  return acquire.then(async () => {
    try {
      return await fn();
    } finally {
      gate.active -= 1;
      if (speculativeLease) gate.speculativeActive -= 1;
      tallyOriginLease(gate, origin, speculativeLease, -1);
      // Hand the freed slot to the most urgent waiter. dispatch() admits only
      // below the origin's width, so after an origin turns out to be HTTP/1.1,
      // its leases already in flight drain before new ones start.
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

/** The origin of a store that exposes its base `url` (zarrita's `FetchStore`). */
function originOfStore(store: object): string | undefined {
  const url = (store as { url?: unknown }).url;
  if (url instanceof URL) return originOfUrl(url.href);
  return typeof url === 'string' ? originOfUrl(url) : undefined;
}

/**
 * Wrap a zarr store so its `get` / `getRange` go through {@link withFetchGate}.
 * A `Proxy` keeps the store's full surface intact (consolidated-metadata and
 * caching wrappers, `contents()`, etc.) while throttling only the two
 * data-fetching methods. The gate is told the store's origin (so an h2/h3 host
 * gets its wide lane and an `http:` host its socket cap) and each call's abort
 * signal (so a superseded read leaves the queue).
 */
export function boundedConcurrencyStore<S extends object>(store: S): S {
  const origin = originOfStore(store);
  return new Proxy(store, {
    get(target, prop, receiver) {
      if (prop === 'get' || prop === 'getRange') {
        const orig = Reflect.get(target, prop, receiver);
        if (typeof orig === 'function') {
          return (...args: unknown[]) => {
            const signal = signalOfStoreArgs(args);
            return withFetchGate(
              () => (orig as (...a: unknown[]) => Promise<unknown>).apply(target, args),
              typeof args[0] === 'string' ? fetchLaneForKey(args[0]) : 'data',
              signalPriority(signal) ?? 'demand',
              origin,
              signal
            );
          };
        }
      }
      return Reflect.get(target, prop, receiver);
    },
  }) as S;
}
