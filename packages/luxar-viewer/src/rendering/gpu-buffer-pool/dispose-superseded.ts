/**
 * Post-grow disposal for released geometries that cannot be reused.
 *
 * @module rendering/gpu-buffer-pool/dispose-superseded
 */

import type { PooledBuffer } from './pool-stats';

/**
 * Remove and dispose a just-released geometry after its replacement has been
 * acquired successfully. A false result means the release-time eviction sweep
 * already removed the buffer, so callers must not increment eviction counts.
 *
 * Disposal errors are intentionally contained: the replacement is active and
 * the old entry is no longer adoptable, so propagating here would make grow
 * recovery reinstate an already-disposed, already-unbucketed geometry.
 */
export function disposeSupersededBuffer(
  buffersByCapacity: Map<number, PooledBuffer[]>,
  released: PooledBuffer
): boolean {
  for (const [capacity, buffers] of buffersByCapacity) {
    const index = buffers.indexOf(released);
    if (index === -1) continue;
    buffers.splice(index, 1);
    if (buffers.length === 0) buffersByCapacity.delete(capacity);
    try {
      released.geometry.dispose();
    } catch {
      // The replacement is already active and the old buffer has been
      // removed from the pool, so cleanup cannot safely roll the grow back.
    }
    return true;
  }
  return false;
}
