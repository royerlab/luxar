/**
 * Aggregate per-mesh `visibleSegmentCount` / `visibleSplatCount` userData
 * across the scene graph and report the totals to the data-loading
 * monitor. Called once per update cycle after lines/gsplats commits so
 * the monitor's HUD shows the post-clipping visible counts (not the
 * raw loaded counts).
 */

import * as THREE from 'three';
import { isLinesUserData } from '../../../types/lines';
import type { GSplatsUserData } from '../../../types/gsplats';
import type { SceneLoaderMonitorPort } from '../../scene-loader-monitor-port';

/**
 * Traverse `rootGroup`, sum the per-mesh visible-counts userData, and
 * push the totals to `monitor`. No-op when either argument is null.
 */
export function updateVisibleCountsInMonitor(
  rootGroup: THREE.Group | null,
  monitor: SceneLoaderMonitorPort | null
): void {
  if (!rootGroup || !monitor) return;

  let totalVisibleSegments = 0;
  let totalVisibleSplats = 0;

  rootGroup.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      if (isLinesUserData(object.userData)) {
        totalVisibleSegments += object.userData.visibleSegmentCount ?? 0;
      } else if (object.userData?.nodeType === 'gsplats') {
        totalVisibleSplats += (object.userData as GSplatsUserData).visibleSplatCount ?? 0;
      }
    }
  });

  monitor.updateVisibleSegments(totalVisibleSegments);
  monitor.updateVisibleSplats(totalVisibleSplats);
}
