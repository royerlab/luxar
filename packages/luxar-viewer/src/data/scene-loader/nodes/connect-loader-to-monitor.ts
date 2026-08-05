/**
 * Duck-type-guarded loader → monitor wiring.
 *
 * Points loaders always implement the full `LoaderMonitor` surface;
 * Lines / GSplats expose those methods as optional. This helper checks
 * for the four-method shape (`addEventListener` / `removeEventListener`
 * / `getMetrics` / `getActiveQueries`) before wiring, so a loader that
 * doesn't carry the surface is silently skipped.
 *
 * Takes the monitor by parameter — no ctx needed. A null monitor short-
 * circuits (the data-monitor UI is opt-in per scene).
 */

import type { DataLoader } from '../../data-loader-types';
import type { LinesDataLoader } from '../../../types/lines';
import type { GSplatsDataLoader } from '../../../types/gsplats';
import type { MeshDataLoader } from '../../../types/mesh';
import type { LoaderMonitor } from '../../../types/data-monitor-types';
import type { SceneLoaderMonitorPort } from '../../scene-loader-monitor-port';

/**
 * Wire `loader` to `monitor` keyed by `path` when the monitor is active
 * and the loader implements the {@link LoaderMonitor} surface.
 */
export function connectLoaderToMonitor(
  path: string,
  loader: DataLoader | LinesDataLoader | GSplatsDataLoader | MeshDataLoader,
  monitor: SceneLoaderMonitorPort | null
): void {
  if (!monitor) return;

  const candidate = loader as Partial<LoaderMonitor>;
  if (
    typeof candidate.addEventListener === 'function' &&
    typeof candidate.removeEventListener === 'function' &&
    typeof candidate.getMetrics === 'function' &&
    typeof candidate.getActiveQueries === 'function'
  ) {
    monitor.connectLoader(path, candidate as LoaderMonitor);
  }
}
