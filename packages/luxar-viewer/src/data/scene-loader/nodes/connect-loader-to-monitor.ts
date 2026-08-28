/**
 * Duck-type-guarded loader → monitor wiring.
 *
 * Points loaders always implement the full `LoaderMonitor` surface;
 * Lines / GSplats / Mesh expose those methods as optional (all four
 * geometries' shipped loaders do implement them). This helper checks for
 * the four-method shape (`addEventListener` / `removeEventListener` /
 * `getMetrics` / `getActiveQueries`) before wiring, so a loader that
 * doesn't carry the surface is silently skipped.
 *
 * "Silently" is load-bearing to get right, and it bit mesh: `MeshWholeNodeLoader`
 * had none of the four, so every mesh node reached this helper, failed the shape
 * check, and was dropped without a log line — leaving the monitor's panel with
 * mesh nodes in its scene-graph tree but no mesh loader row, no mesh bytes in
 * its loader-memory total and no mesh loads in its rate windows.
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
