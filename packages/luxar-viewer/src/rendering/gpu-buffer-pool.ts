/**
 * Re-export stub. The orchestrator class and per-type adapters live in
 * the `./gpu-buffer-pool/` folder. Consumers import `GPUBufferPool` (and
 * the public types/utilities) from this path unchanged.
 */

export {
  GPUBufferPool,
  estimateGeometryBytes,
  invalidateCachedByteSize,
  selectBuffersToEvict,
} from './gpu-buffer-pool/index';
export type {
  PackedGSplatsData,
  PointsAttributeTypes,
  PooledBuffer,
  TypePoolStats,
  PoolStats,
  PooledBufferRef,
} from './gpu-buffer-pool/index';
