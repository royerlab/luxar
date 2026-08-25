import { POOLED_GEOMETRY_TYPES } from '../../../types/data-monitor-types';
import type { AccumulatorProvider, MemoryMetrics } from '../../../types/data-monitor-types';

export interface MemoryMetricsParams {
  gpuBufferPoolProvider: { getStats: () => MemoryMetrics['gpuPool'] } | null;
  accumulatorProviders: Record<(typeof POOLED_GEOMETRY_TYPES)[number], AccumulatorProvider | null>;
}

export function aggregateMemoryMetrics({
  gpuBufferPoolProvider,
  accumulatorProviders,
}: MemoryMetricsParams): MemoryMetrics {
  return {
    gpuPool: gpuBufferPoolProvider?.getStats() ?? null,
    accumulators: Object.fromEntries(
      POOLED_GEOMETRY_TYPES.map((type) => [type, accumulatorProviders[type]?.getStats() ?? null])
    ) as MemoryMetrics['accumulators'],
  };
}
