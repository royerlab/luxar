import { describe, expect, it, vi } from 'vitest';
import { aggregateMemoryMetrics } from '../../../../../ui/data-loading-monitor/metrics/memory';
import { POOLED_GEOMETRY_TYPES } from '../../../../../types/data-monitor-types';

describe('aggregateMemoryMetrics', () => {
  it('reads every registered provider and preserves empty slots', () => {
    const accumulatorProviders = Object.fromEntries(
      POOLED_GEOMETRY_TYPES.map((type, index) => [
        type,
        index === 1
          ? null
          : {
              getStats: vi.fn(() => ({
                capacity: index + 1,
                allocations: 2,
                growthEvents: 3,
                memoryMB: 4,
              })),
            },
      ])
    ) as Parameters<typeof aggregateMemoryMetrics>[0]['accumulatorProviders'];
    const gpuPool = { allocations: 1, reuses: 2, evictions: 3, byType: {} } as never;

    const result = aggregateMemoryMetrics({
      gpuBufferPoolProvider: { getStats: () => gpuPool },
      accumulatorProviders,
    });

    expect(result.gpuPool).toBe(gpuPool);
    expect(result.accumulators.points?.capacity).toBe(1);
    expect(result.accumulators.lines).toBeNull();
    expect(result.accumulators.gsplats?.capacity).toBe(3);
  });
});
