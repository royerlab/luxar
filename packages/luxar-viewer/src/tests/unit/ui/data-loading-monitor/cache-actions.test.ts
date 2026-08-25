// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { notifier } from '../../../../utils/cross-layer/notifier';
import { createCacheActions } from '../../../../ui/data-loading-monitor/cache-actions';
import { MonitorProviderRegistry } from '../../../../ui/data-loading-monitor/providers';

describe('createCacheActions', () => {
  afterEach(() => vi.restoreAllMocks());

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

  it('reports the freed L2 size after a successful clear', async () => {
    const providers = new MonitorProviderRegistry(vi.fn());
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
        l2: { size: 3 * 1024 * 1024, count: 2, reads: 0, writes: 0, misses: 0 },
        network: { bytesTransferred: 0, requestCount: 0, bandwidth: 0 },
      }),
      clearL1: vi.fn(),
      clearL2: vi.fn().mockResolvedValue(undefined),
      clearAll: vi.fn(),
      isEnabled: () => true,
    });
    const toast = vi.spyOn(notifier, 'toast').mockImplementation(() => undefined);

    await createCacheActions(providers, vi.fn()).clearL2Cache({ skipConfirm: true });

    expect(toast).toHaveBeenCalledWith('L2 cache cleared (3.0 MB freed)');
  });

  it('clears L0 and SliceCache before awaiting the provider-wide clear', async () => {
    const providers = new MonitorProviderRegistry(vi.fn());
    const order: string[] = [];
    let finishClearAll!: () => void;
    const clearAllFinished = new Promise<void>((resolve) => {
      finishClearAll = resolve;
    });
    providers.setL0CacheProvider({
      getStats: () => undefined,
      clear: () => order.push('l0'),
    });
    providers.setSliceCacheProvider({
      getStats: () => undefined,
      clear: () => order.push('slice'),
    });
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
        l2: { size: 0, count: 0, reads: 0, writes: 0, misses: 0 },
        network: { bytesTransferred: 0, requestCount: 0, bandwidth: 0 },
      }),
      clearL1: vi.fn(),
      clearL2: vi.fn(),
      clearAll: () => {
        order.push('all-start');
        return clearAllFinished.then(() => {
          order.push('all-finish');
        });
      },
      isEnabled: () => true,
    });
    vi.spyOn(notifier, 'toast').mockImplementation(() => undefined);

    const clearing = createCacheActions(providers, vi.fn()).clearAllCaches({ skipConfirm: true });
    expect(order).toEqual(['l0', 'slice', 'all-start']);

    finishClearAll();
    await clearing;
    expect(order).toEqual(['l0', 'slice', 'all-start', 'all-finish']);
  });
});
