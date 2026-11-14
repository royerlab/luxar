/**
 * Byte-tracking containers for the GPU buffer pool's two kinds of residency.
 *
 * `GPUBufferPool.getResidentBytes()` is the single source of truth for the
 * VRAM budget and is queried by the LOD registry every frame and by the
 * byte-eviction pass on every acquire/release. It used to walk every active
 * and pooled buffer on each call; these containers keep the totals
 * INCREMENTALLY instead, so the query is O(1) (B9b, #2944).
 *
 * They are drop-in `Map` subclasses so every existing reader (adapters,
 * evictor, stats, tests) keeps working unchanged. The contract that keeps the
 * totals exact:
 *
 *  - `ActiveBufferMap`: every `set` / `delete` / `clear` adjusts the total.
 *    Replacing a key subtracts the bytes charged to the previous entry.
 *  - `FreeBucketMap`: the bytes charged to each bucket are remembered, and
 *    `set(bucket, array)` / `delete(bucket)` / `clear()` RESYNC that bucket
 *    (subtract what it was charged, charge what it now holds). So an in-place
 *    array mutation (`push` / `splice`) is exact as soon as it is followed by
 *    a `set` or `delete` of that bucket. `pushBuffer` / `takeAt` /
 *    `removeBuffer` do both in one call and are what the pool uses.
 *
 * Bytes are charged with `estimateGeometryBytes`, cached per geometry, and the
 * exact amount charged is what is later subtracted — so the totals cannot
 * drift even if a geometry's cached size were invalidated while resident (no
 * production code does that: pooled geometries have a fixed capacity).
 *
 * @module rendering/gpu-buffer-pool/byte-tracked-maps
 */

import { estimateGeometryBytes } from './geometry-bytes';
import type { PooledBuffer } from './pool-stats';

/** nodeId → active (in-use) buffer, with a running byte total. */
export class ActiveBufferMap extends Map<string, PooledBuffer> {
  /** Bytes charged per key — subtracted exactly on removal. */
  private readonly charged = new Map<string, number>();
  private total = 0;

  /** Sum of `estimateGeometryBytes` over every active buffer. O(1). */
  get bytes(): number {
    return this.total;
  }

  override set(nodeId: string, buffer: PooledBuffer): this {
    // `super()` has no iterable, so `set` never runs before the fields exist.
    this.uncharge(nodeId);
    const bytes = estimateGeometryBytes(buffer.geometry);
    this.charged.set(nodeId, bytes);
    this.total += bytes;
    return super.set(nodeId, buffer);
  }

  override delete(nodeId: string): boolean {
    this.uncharge(nodeId);
    return super.delete(nodeId);
  }

  override clear(): void {
    this.charged.clear();
    this.total = 0;
    super.clear();
  }

  private uncharge(nodeId: string): void {
    const bytes = this.charged.get(nodeId);
    if (bytes === undefined) return;
    this.total -= bytes;
    this.charged.delete(nodeId);
  }
}

/** Sum of cached byte estimates over a bucket's buffers. */
function bucketBytes(buffers: readonly PooledBuffer[]): number {
  let total = 0;
  for (const buffer of buffers) total += estimateGeometryBytes(buffer.geometry);
  return total;
}

/** capacity bucket → released (pooled) buffers, with a running byte total. */
export class FreeBucketMap extends Map<number, PooledBuffer[]> {
  /** Bytes charged per bucket at its last resync. */
  private readonly charged = new Map<number, number>();
  private total = 0;

  /** Sum of `estimateGeometryBytes` over every pooled buffer. O(1). */
  get bytes(): number {
    return this.total;
  }

  /** Store `buffers` under `bucket` and resync that bucket's charge. */
  override set(bucket: number, buffers: PooledBuffer[]): this {
    this.uncharge(bucket);
    const bytes = bucketBytes(buffers);
    this.charged.set(bucket, bytes);
    this.total += bytes;
    return super.set(bucket, buffers);
  }

  override delete(bucket: number): boolean {
    this.uncharge(bucket);
    return super.delete(bucket);
  }

  override clear(): void {
    this.charged.clear();
    this.total = 0;
    super.clear();
  }

  /** Append `buffer` to `bucket` (creating it), charging its bytes. */
  pushBuffer(bucket: number, buffer: PooledBuffer): void {
    const buffers = this.get(bucket);
    if (!buffers) {
      this.set(bucket, [buffer]);
      return;
    }
    buffers.push(buffer);
    this.charge(bucket, estimateGeometryBytes(buffer.geometry));
  }

  /**
   * Splice the buffer at `index` out of `bucket`, uncharging it. The bucket
   * stays in the map even when emptied (the adopt path never deleted it).
   */
  takeAt(bucket: number, index: number): PooledBuffer {
    const buffers = this.get(bucket);
    const buffer = buffers?.[index];
    if (!buffers || !buffer) {
      throw new RangeError(`FreeBucketMap.takeAt: no buffer at ${bucket}[${index}]`);
    }
    buffers.splice(index, 1);
    this.charge(bucket, -estimateGeometryBytes(buffer.geometry));
    return buffer;
  }

  /**
   * Remove the first buffer matching `match` (buckets in map order), uncharging
   * it, and return it. `deleteEmptyBucket` drops a bucket the removal empties —
   * callers keep their historical bucket policy, since bucket order decides
   * which buffers the batch-capped LRU sweep reaches first.
   */
  removeFirst(
    match: (buffer: PooledBuffer) => boolean,
    deleteEmptyBucket: boolean
  ): PooledBuffer | undefined {
    for (const [bucket, buffers] of this) {
      const index = buffers.findIndex(match);
      if (index === -1) continue;
      const [buffer] = buffers.splice(index, 1);
      if (deleteEmptyBucket && buffers.length === 0) this.delete(bucket);
      else this.charge(bucket, -estimateGeometryBytes(buffer.geometry));
      return buffer;
    }
    return undefined;
  }

  private charge(bucket: number, delta: number): void {
    this.charged.set(bucket, (this.charged.get(bucket) ?? 0) + delta);
    this.total += delta;
  }

  private uncharge(bucket: number): void {
    const bytes = this.charged.get(bucket);
    if (bytes === undefined) return;
    this.total -= bytes;
    this.charged.delete(bucket);
  }
}
