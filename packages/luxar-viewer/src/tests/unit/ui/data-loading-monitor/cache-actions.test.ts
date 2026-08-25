// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest';
import { createCacheActions } from '../../../../ui/data-loading-monitor/cache-actions';
import { MonitorProviderRegistry } from '../../../../ui/data-loading-monitor/providers';

describe('createCacheActions', () => {
  it('does nothing when a provider is absent', () => {
    const changed = vi.fn();
    createCacheActions(new MonitorProviderRegistry(vi.fn()), changed).clearL0Cache();
    expect(changed).not.toHaveBeenCalled();
  });

  it('honors destructive confirmation before clearing L2', async () => {
    const providers = new MonitorProviderRegistry(vi.fn());
    const clearL2 = vi.fn().mockResolvedValue(undefined);
    providers.setCacheStatsProvider({
      getStats: () => ({
        l1: {
          metadataSize: 0,
          chunksSize: 0,
          metadataCount: 0,
          chunksCount: 0,
          hits: 0,
          misses: 0,
          evictions: 0,
        },
        l2: { size: 1, count: 0, reads: 0, writes: 0, misses: 0 },
        network: { bytesTransferred: 0, requestCount: 0, bandwidth: 0 },
      }),
      clearL1: vi.fn(),
      clearL2,
      clearAll: vi.fn(),
      isEnabled: () => true,
    });
    vi.spyOn(window, 'confirm').mockReturnValue(false);

    await createCacheActions(providers, vi.fn()).clearL2Cache();
    expect(clearL2).not.toHaveBeenCalled();
  });
});
