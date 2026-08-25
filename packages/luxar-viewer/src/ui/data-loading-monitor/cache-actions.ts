/**
 * Destructive cache workflows for DataLoadingMonitor. The orchestrator's
 * public clear methods delegate here for confirmation, tier ordering, toasts,
 * and the final UI refresh callback.
 */

import { notifier } from '../../utils/cross-layer/notifier';
import { log, Modules } from '../../utils/log';
import type { MonitorProviderRegistry } from './providers';

export interface CacheActions {
  clearL0Cache(): void;
  clearSliceCache(): void;
  clearL1Cache(): void;
  clearL2Cache(opts?: { skipConfirm?: boolean }): Promise<void>;
  clearAllCaches(opts?: { skipConfirm?: boolean }): Promise<void>;
}

/**
 * Show a confirmation dialog for destructive cache actions. Falls
 * back to `true` if `window.confirm` is unavailable (jsdom test env).
 */
function confirmDestructiveCacheAction(message: string): boolean {
  if (typeof window === 'undefined' || typeof window.confirm !== 'function') return true;
  return window.confirm(message);
}

export function createCacheActions(
  providers: MonitorProviderRegistry,
  onChanged: () => void
): CacheActions {
  return {
    clearL0Cache(): void {
      if (!providers.l0CacheProvider) return;
      providers.l0CacheProvider.clear();
      log.info(Modules.DATA_MONITOR, 'L0 cache cleared');
      onChanged();
    },
    clearSliceCache(): void {
      if (!providers.sliceCacheProvider) return;
      providers.sliceCacheProvider.clear();
      log.info(Modules.DATA_MONITOR, 'SliceCache cleared');
      onChanged();
    },
    clearL1Cache(): void {
      if (!providers.cacheStatsProvider) return;
      providers.cacheStatsProvider.clearL1();
      log.info(Modules.DATA_MONITOR, 'L1 cache cleared');
      onChanged();
    },
    async clearL2Cache(opts?: { skipConfirm?: boolean }): Promise<void> {
      if (!providers.cacheStatsProvider) return;
      if (!opts?.skipConfirm && !confirmDestructiveCacheAction('Clear L2 (persistent) cache?')) {
        return;
      }
      const sizeBefore = providers.cacheStatsProvider.getStats().l2.size;
      await providers.cacheStatsProvider.clearL2();
      log.info(Modules.DATA_MONITOR, 'L2 cache cleared');
      notifier.toast(
        sizeBefore > 0
          ? `L2 cache cleared (${(sizeBefore / 1024 / 1024).toFixed(1)} MB freed)`
          : 'L2 cache cleared'
      );
      onChanged();
    },
    async clearAllCaches(opts?: { skipConfirm?: boolean }): Promise<void> {
      if (
        !opts?.skipConfirm &&
        !confirmDestructiveCacheAction('Clear ALL caches (L0 + L1 + L2)?')
      ) {
        return;
      }
      // Clear L0 + SliceCache first (synchronous)
      providers.l0CacheProvider?.clear();
      providers.sliceCacheProvider?.clear();
      // Clear L1 + L2 (L2 is async)
      await providers.cacheStatsProvider?.clearAll();
      log.info(Modules.DATA_MONITOR, 'All caches cleared (S-cache + L0 + L1 + L2)');
      notifier.toast('All caches cleared');
      onChanged();
    },
  };
}
