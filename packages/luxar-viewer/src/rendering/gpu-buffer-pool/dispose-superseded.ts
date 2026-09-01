import type { PooledBuffer } from './pool-stats';

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
