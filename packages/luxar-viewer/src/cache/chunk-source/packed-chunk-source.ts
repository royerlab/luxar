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
 * (an all-fill chunk zarr never wrote) is read plainly.
 *
 * @module cache/chunk-source/packed-chunk-source
 */

import type { ChunkFetchOutcome, ChunkSource, ChunkSourceGetOptions } from '../chunk-source';
import type { RemoteValidationToken } from '../multi-level-caching-store/validation-queue';
import { sha256Hex } from '../multi-level-caching-store/sha256';
import { log, Modules } from '../../utils/log';

interface PackEntry {
  key: string;
  sha256: string;
  prefix: string;
}

/** A fetched pack: its members' bytes, dropped once every one has been served. */
type Members = Map<string, Uint8Array>;

interface Pack extends PackEntry {
  /** In flight or held while some member is unserved; `null` = unusable. */
  members?: Promise<Members | null>;
}

const normalized = (key: string): string => key.replace(/^\/+/, '');

function parsePack(bytes: Uint8Array): Members {
  const headerLength = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + headerLength))) as {
    members: Record<string, [number, number]>;
  };
  const data = 4 + headerLength;
  const members: Members = new Map();
  for (const [rel, [offset, length]] of Object.entries(header.members)) {
    members.set(rel, bytes.slice(data + offset, data + offset + length));
  }
  return members;
}

export class PackedChunkSource implements ChunkSource {
  /** Packed node prefix (with its trailing `/`) → pack. */
  private readonly packs = new Map<string, Pack>();

  /** @param inner - The store's own byte source; every key no pack answers goes to it. */
  constructor(readonly inner: ChunkSource) {}

  /** The inner source's identity verbatim — the OPFS bucket must not move. */
  get identity(): string {
    return this.inner.identity;
  }

  get describe(): string {
    return this.inner.describe;
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

  async get(
    key: string,
    signal?: AbortSignal,
    options?: ChunkSourceGetOptions
  ): Promise<ChunkFetchOutcome> {
    const name = normalized(key);
    const pack = this.packFor(name);
    if (!pack) return this.inner.get(key, signal, options);
    // Shared by every member's read, so it is not tied to any one caller's signal.
    pack.members ??= this.fetchPack(pack, options);
    const members = await pack.members;
    if (signal?.aborted) return { kind: 'aborted' };
    const rel = name.slice(pack.prefix.length);
    const data = members?.get(rel);
    if (!members) this.packs.delete(pack.prefix);
    if (!members || !data) return this.inner.get(key, signal, options);
    members.delete(rel);
    // Every member is in the caches now; a later re-read refetches the pack.
    if (members.size === 0) pack.members = undefined;
    return { kind: 'ok', data, bytesOverWire: data.byteLength };
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

  private async fetchPack(pack: Pack, options?: ChunkSourceGetOptions): Promise<Members | null> {
    const outcome = await this.inner.get(pack.key, undefined, options);
    if (outcome.kind !== 'ok') return null;
    if ((await sha256Hex(outcome.data)) !== pack.sha256) {
      log.warning(Modules.CACHE, `chunk pack ${pack.key} fails its digest; reading plainly`);
      return null;
    }
    return parsePack(outcome.data);
  }

  probeIdentityToken(options: {
    signal?: AbortSignal;
    timeoutMsOverride?: number;
  }): Promise<RemoteValidationToken | null> {
    return this.inner.probeIdentityToken(options);
  }

  dispose(): void {
    this.packs.clear();
    this.inner.dispose();
  }
}
