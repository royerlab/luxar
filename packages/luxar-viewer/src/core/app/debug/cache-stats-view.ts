import { DataMonitorManager } from '../../../ui/data-monitor-manager';

/**
 * Open the data-loading monitor in expanded mode on the Cache tab.
 * Best-effort: silently skips when no monitor was created (e.g.
 * embedded contexts that disable the monitor). Called by LuxarApp
 * during init when the `openCacheStats` option is set, or when the
 * `?cacheStats` URL flag is present.
 */
export function openCacheStatsView(): void {
  const monitor = DataMonitorManager.getInstance().getDefaultMonitor();
  if (!monitor) return;
  monitor.show();
  monitor.expand();
  monitor.setActiveTab('cache');
}
