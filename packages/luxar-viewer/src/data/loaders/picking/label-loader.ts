/**
 * Label Loader — Lazy per-element CSR text fetching from zarr.
 *
 * Text channels are stored per node as two zarr arrays:
 *   - labels: label_offsets + label_bytes
 *   - keys: key_offsets + key_bytes
 *
 * Label i = label_bytes[offsets[i] : offsets[i+1]], decoded as UTF-8.
 * Empty labels (offsets[i] === offsets[i+1]) return null.
 */

import { LRUCache } from '../../../cache/lru-cache';
import { log, Modules } from '../../../utils/log';
import * as zarr from '../../zarr';
import { readArray, slice } from '../../zarr';

/** Open zarr arrays backing one CSR text channel. */
export interface LabelArrays {
  offsets: zarr.Array<zarr.DataType>;
  bytes: zarr.Array<zarr.DataType>;
}

const DEFAULT_MAX_CACHE_BYTES = 1024 * 1024;
// Roughly 16k short labels per 1 MB budget; cache keys add a small uncounted overhead.
const MIN_CACHE_ENTRY_BYTES = 64;

function cachedLabelSize(label: string | null): number {
  return Math.max(MIN_CACHE_ENTRY_BYTES, (label?.length ?? 0) * 2);
}

function decodeByteRange(offsets: BigUint64Array): { start: number; end: number } | null {
  if (offsets.length !== 2) throw new Error(`expected 2 offsets, got ${offsets.length}`);

  const start = Number(offsets[0]);
  const end = Number(offsets[1]);
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start) {
    throw new Error(`invalid byte range [${start}, ${end})`);
  }
  return start === end ? null : { start, end };
}

function requireByteCount(bytes: Uint8Array, expected: number): void {
  if (bytes.length !== expected) {
    throw new Error(`expected ${expected} label bytes, got ${bytes.length}`);
  }
}

export class LabelLoader {
  /** Opened array handles, or null for nodes known not to carry this channel. */
  private arraysCache = new Map<string, LabelArrays | null>();

  /** In-flight array opens, coalesced per node. */
  private arraysInflight = new Map<string, Promise<LabelArrays | null>>();

  /** Bounded decoded-label cache shared across nodes. */
  private labelCache: LRUCache<string | null>;

  /** In-flight label reads, coalesced per node and element. */
  private labelInflight = new Map<string, Promise<string | null | undefined>>();

  /** UTF-8 text decoder (reused). */
  private decoder = new TextDecoder('utf-8');

  /**
   * @param channel Which per-element string channel to read (issue #1917).
   *   `'labels'` (default) reads `label_offsets` / `label_bytes`; `'keys'`
   *   reads `key_offsets` / `key_bytes`. Both use the same CSR encoding.
   * @param maxCacheBytes Byte budget for decoded strings across all nodes.
   */
  constructor(
    private rootLoc: zarr.Location<zarr.Readable>,
    private channel: 'labels' | 'keys' = 'labels',
    maxCacheBytes: number = DEFAULT_MAX_CACHE_BYTES
  ) {
    this.labelCache = new LRUCache<string | null>(maxCacheBytes, cachedLabelSize);
  }

  /** Get one label, or null when absent, empty, invalid, or unreadable. */
  async getLabel(nodePath: string, elementIndex: number): Promise<string | null> {
    if (!Number.isInteger(elementIndex) || elementIndex < 0) return null;

    const cacheKey = `${nodePath}\0${elementIndex}`;
    const cached = this.labelCache.get(cacheKey);
    if (cached !== undefined) return cached;

    const existing = this.labelInflight.get(cacheKey);
    if (existing) return (await existing) ?? null;

    const promise = this.fetchLabel(nodePath, elementIndex);
    this.labelInflight.set(cacheKey, promise);
    try {
      const label = await promise;
      if (label !== undefined) this.labelCache.set(cacheKey, label);
      return label ?? null;
    } finally {
      this.labelInflight.delete(cacheKey);
    }
  }

  /** Check if a node has this string channel in its cached metadata. */
  hasLabels(nodeAttrs: Record<string, unknown>): boolean {
    return nodeAttrs?.[this.channel === 'keys' ? 'has_keys' : 'has_labels'] === true;
  }

  /** Clean up caches. */
  dispose(): void {
    this.arraysCache.clear();
    this.arraysInflight.clear();
    this.labelCache.clear();
    this.labelInflight.clear();
  }

  private async loadArrays(nodePath: string): Promise<LabelArrays | null> {
    const cached = this.arraysCache.get(nodePath);
    if (cached !== undefined) return cached;

    const existing = this.arraysInflight.get(nodePath);
    if (existing) return existing;

    const promise = this.openArrays(nodePath);
    this.arraysInflight.set(nodePath, promise);
    try {
      const arrays = await promise;
      this.arraysCache.set(nodePath, arrays);
      return arrays;
    } catch (error) {
      log.warning(
        Modules.SCENE_LOADER,
        `Failed to open ${this.channel} for ${nodePath}: ${error instanceof Error ? error.message : error}`
      );
      if (zarr.isNotFoundError(error)) this.arraysCache.set(nodePath, null);
      return null;
    } finally {
      this.arraysInflight.delete(nodePath);
    }
  }

  private async openArrays(nodePath: string): Promise<LabelArrays | null> {
    const cleanPath = nodePath.startsWith('/') ? nodePath.slice(1) : nodePath;
    const prefix = this.channel === 'keys' ? 'key' : 'label';
    const offsetsLoc = this.rootLoc.resolve(`${cleanPath}/${prefix}_offsets`);
    const bytesLoc = this.rootLoc.resolve(`${cleanPath}/${prefix}_bytes`);

    let offsets: zarr.Array<zarr.DataType>;
    try {
      offsets = await zarr.open(offsetsLoc, { kind: 'array' });
    } catch (error) {
      if (!zarr.isNotFoundError(error)) throw error;
      log.info(Modules.SCENE_LOADER, `Node carries no ${this.channel}: ${nodePath}`);
      return null;
    }

    const bytes = await zarr.open(bytesLoc, { kind: 'array' });
    return { offsets, bytes };
  }

  private async fetchLabel(
    nodePath: string,
    elementIndex: number
  ): Promise<string | null | undefined> {
    try {
      const arrays = await this.loadArrays(nodePath);
      if (!arrays) return undefined;
      if (elementIndex >= arrays.offsets.shape[0] - 1) return null;

      const offsetResult = await readArray(arrays.offsets, [slice(elementIndex, elementIndex + 2)]);
      const range = decodeByteRange(offsetResult.data as BigUint64Array);
      if (!range) return null;

      const byteResult = await readArray(arrays.bytes, [slice(range.start, range.end)]);
      const bytes = byteResult.data as Uint8Array;
      requireByteCount(bytes, range.end - range.start);
      return this.decoder.decode(bytes);
    } catch (error) {
      log.warning(
        Modules.SCENE_LOADER,
        `Failed to load ${this.channel} for ${nodePath}[${elementIndex}]: ${
          error instanceof Error ? error.message : error
        }`
      );
      return undefined;
    }
  }
}
