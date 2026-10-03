/**
 * A small node's chunks from ONE request: the `luxar optimize --pack` reader.
 *
 * A packed store keeps every plain chunk object and adds, per small geometry
 * node, one pack file holding copies of that node's chunks (`luxar.io.chunk_pack`):
 * a 4-byte little-endian header length, a JSON header `{"members": {chunk key
 * relative to the node: [offset, length]}}`, then the bytes. The root
 * `chunk_packs` sidecar's attrs list each pack's key, SHA-256 and node prefix.
 * Thousands of ~1 KB rungs cost round trips, not bytes, so a laddered
 * timelapse part read as one pack instead of ~16 chunks plays measurably faster.
 *
 * This source sits UNDER the caching store, in the zip store's seam: it answers
 * each chunk key with that chunk's bytes, so L0/L1/L2 cache exactly the keys
 * and bytes a plain read would have produced, and no loader or decoder knows
 * packs exist. Every doubt lands on the plain read, never on wrong bytes: an
 * index built for another `content_hash` is not adopted, a pack that fails its
 * SHA-256 or cannot be fetched is dropped, and a key the pack does not hold
 * (an all-fill chunk zarr never wrote, a metadata document) is read plainly.
 *
 * A fetched pack's members are handed out once each (the caches hold them
 * from then on) and its unserved ones are held, least recently used first,
 * within {@link PackedChunkSource.maxHeldBytes}: a rung the view never loads
 * must not pin bytes outside every cache budget. A member read again after
 * the caches evicted it, or a member of a pack let go, fetches the pack again —
 * one request, as a plain read would cost.
 *
 * @module cache/chunk-source/packed-chunk-source
 */

import type { ChunkFetchOutcome, ChunkSource, ChunkSourceGetOptions } from '../chunk-source';
import type { RemoteValidationToken } from '../multi-level-caching-store/validation-queue';
import { sha256Hex } from '../multi-level-caching-store/sha256';
import { FetchPriorityCell } from '../../utils/fetch-concurrency';
import { log, Modules } from '../../utils/log';

interface PackEntry {
  key: string;
  sha256: string;
  prefix: string;
}

/** A fetched pack's members not yet served. */
interface Held {
  members: Map<string, Uint8Array>;
  /** Bytes of `members`. */
  bytes: number;
  /** The pack's bytes over the wire, reported once, by the first member served. */
  wireBytes: number;
}

interface Pack extends PackEntry {
  /** The fetch, in flight or holding unserved members; `null` = unusable. */
  held?: Promise<Held | null>;
  /** The in-flight fetch's class, raised by every caller that joins it. */
  priority?: FetchPriorityCell;
  /** Every member key (relative to the prefix), once the pack was read. */
  memberKeys?: ReadonlySet<string>;
}

const ABORTED = Symbol('aborted');

/** Unserved members held across all packs by default. */
const DEFAULT_MAX_HELD_BYTES = 4 * 1024 * 1024;

const normalized = (key: string): string => key.replace(/^\/+/, '');

function parsePack(bytes: Uint8Array, wireBytes: number): Held {
  const headerLength = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + headerLength))) as {
    members: Record<string, [number, number]>;
  };
  const data = 4 + headerLength;
  const held: Held = { members: new Map(), bytes: 0, wireBytes };
  for (const [rel, [offset, length]] of Object.entries(header.members)) {
    held.members.set(rel, bytes.slice(data + offset, data + offset + length));
    held.bytes += length;
  }
  return held;
}

/** `promise`, or {@link ABORTED} as soon as `signal` fires (the work goes on). */
function raceAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T | typeof ABORTED> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.resolve(ABORTED);
  return new Promise((resolve, reject) => {
    const onAbort = (): void => resolve(ABORTED);
    signal.addEventListener('abort', onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener('abort', onAbort);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener('abort', onAbort);
        reject(error as Error);
      }
    );
  });
}

/**
 * A {@link ChunkSource} that serves every chunk key of a packed node from one
 * fetch of its pack (`chunk_packs/<n>.pack`), and every other key, or any key
 * whose pack is missing, stale or fails its digest, from the wrapped source.
 */
export class PackedChunkSource implements ChunkSource {
  /** Packed node prefix (with its trailing `/`) → pack. */
  private readonly packs = new Map<string, Pack>();
  /** Packs whose fetch holds unserved members, least recently used first. */
  private readonly holding = new Map<Pack, Held>();
  private held = 0;
  /** Cap on {@link heldBytes}; the least recently used pack is let go past it. */
  readonly maxHeldBytes: number;

  /**
   * @param inner - The store's own byte source; every key no pack answers goes to it.
   * @param options.maxHeldBytes - Cap on unserved members held (default 4 MB).
   */
  constructor(
    readonly inner: ChunkSource,
    options: { maxHeldBytes?: number } = {}
  ) {
    this.maxHeldBytes = options.maxHeldBytes ?? DEFAULT_MAX_HELD_BYTES;
  }

  /** The inner source's identity verbatim — the OPFS bucket must not move. */
  get identity(): string {
    return this.inner.identity;
  }

  /** The inner source's description verbatim (a pack changes how bytes arrive, not what the store is). */
  get describe(): string {
    return this.inner.describe;
  }

  /** Bytes of fetched pack members not yet served (bounded by {@link maxHeldBytes}). */
  get heldBytes(): number {
    return this.held;
  }

  /**
   * Adopt the `chunk_packs` sidecar's attrs; returns the number of packs used.
   * An index built for another `content_hash` is stale and adopts nothing.
   */
  usePacks(index: unknown, contentHash: unknown): number {
    const attrs = index as { scene_content_hash?: unknown; packs?: PackEntry[] } | undefined;
    if (!attrs || attrs.scene_content_hash !== contentHash || !Array.isArray(attrs.packs)) {
      return 0;
    }
    for (const entry of attrs.packs) this.packs.set(entry.prefix, { ...entry });
    return attrs.packs.length;
  }

  /** Serve `key` from its node's pack when one is adopted and valid; otherwise read it plainly. */
  async get(
    key: string,
    signal?: AbortSignal,
    options?: ChunkSourceGetOptions
  ): Promise<ChunkFetchOutcome> {
    const name = normalized(key);
    const pack = this.packFor(name);
    const rel = pack ? name.slice(pack.prefix.length) : '';
    if (!pack || pack.memberKeys?.has(rel) === false) return this.inner.get(key, signal, options);
    const held = await this.heldWith(pack, rel, signal, options);
    if (held === ABORTED) return { kind: 'aborted' };
    const data = held?.members.get(rel);
    if (!held || !data) return this.inner.get(key, signal, options);
    return { kind: 'ok', data, bytesOverWire: this.take(pack, held, rel, data) };
  }

  /** The pack whose node holds `name`: packs are disjoint subtrees, so at most one. */
  private packFor(name: string): Pack | undefined {
    if (this.packs.size === 0) return undefined;
    for (let i = name.indexOf('/'); i >= 0; i = name.indexOf('/', i + 1)) {
      const pack = this.packs.get(name.slice(0, i + 1));
      if (pack) return pack;
    }
    return undefined;
  }

  /** The pack's unserved members, fetched again when `rel` was already served. */
  private async heldWith(
    pack: Pack,
    rel: string,
    signal: AbortSignal | undefined,
    options: ChunkSourceGetOptions | undefined
  ): Promise<Held | null | typeof ABORTED> {
    const held = await raceAbort(this.fetchOnce(pack, options), signal);
    if (held === ABORTED || !held || held.members.has(rel) || !pack.memberKeys?.has(rel)) {
      return held;
    }
    // Served before, and its cached copy evicted since: read the pack again.
    this.release(pack, held);
    return raceAbort(this.fetchOnce(pack, options), signal);
  }

  /**
   * Join the pack's fetch or start one. The shared fetch's class is raised to
   * this caller's, and follows any later raise of it while the caller waits.
   */
  private async fetchOnce(
    pack: Pack,
    options: ChunkSourceGetOptions | undefined
  ): Promise<Held | null> {
    const caller = options?.priority;
    const initial = caller?.value ?? 'demand';
    if (!pack.held) {
      pack.priority = new FetchPriorityCell(initial);
      pack.held = this.fetchPack(pack, pack.priority);
    }
    const shared = pack.priority;
    shared?.raise(initial);
    const unlink = caller?.onRaise(() => shared?.raise(caller.value));
    try {
      return await pack.held;
    } finally {
      unlink?.();
    }
  }

  private async fetchPack(pack: Pack, priority: FetchPriorityCell): Promise<Held | null> {
    let held: Held | null = null;
    try {
      const outcome = await this.inner.get(pack.key, undefined, { priority });
      if (outcome.kind === 'ok') {
        if ((await sha256Hex(outcome.data)) === pack.sha256) {
          held = parsePack(outcome.data, outcome.bytesOverWire);
        } else {
          log.warning(Modules.CACHE, `chunk pack ${pack.key} fails its digest; reading plainly`);
        }
      }
    } catch (error) {
      log.warning(Modules.CACHE, `chunk pack ${pack.key} unreadable; reading plainly`, error);
    }
    if (!held) {
      if (this.packs.get(pack.prefix) === pack) this.packs.delete(pack.prefix);
      return null;
    }
    pack.memberKeys ??= new Set(held.members.keys());
    this.hold(pack, held);
    return held;
  }

  /** Account a fetched pack's members, letting the least recently used packs go past the cap. */
  private hold(pack: Pack, held: Held): void {
    this.holding.set(pack, held);
    this.held += held.bytes;
    for (const [oldest, oldestHeld] of this.holding) {
      if (this.held <= this.maxHeldBytes || oldest === pack) break;
      this.release(oldest, oldestHeld);
    }
  }

  /** Hand out member `rel` once; returns the bytes over the wire it reports. */
  private take(pack: Pack, held: Held, rel: string, data: Uint8Array): number {
    held.members.delete(rel);
    held.bytes -= data.byteLength;
    if (this.holding.get(pack) === held) {
      this.held -= data.byteLength;
      this.holding.delete(pack); // re-insert as the most recently used
      this.holding.set(pack, held);
      if (held.members.size === 0) this.release(pack, held);
    }
    const wire = held.wireBytes;
    held.wireBytes = 0;
    return wire;
  }

  /** Let a pack's held members go; the next read of one of them fetches it again. */
  private release(pack: Pack, held: Held): void {
    if (this.holding.get(pack) !== held) return; // already let go
    this.holding.delete(pack);
    this.held -= held.bytes;
    pack.held = undefined;
  }

  probeIdentityToken(options: {
    signal?: AbortSignal;
    timeoutMsOverride?: number;
  }): Promise<RemoteValidationToken | null> {
    return this.inner.probeIdentityToken(options);
  }

  dispose(): void {
    this.packs.clear();
    this.holding.clear();
    this.held = 0;
    this.inner.dispose();
  }
}
