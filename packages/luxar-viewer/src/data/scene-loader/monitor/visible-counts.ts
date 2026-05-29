/**
 * Aggregate per-mesh `visiblePointCount` / `visibleSegmentCount` /
 * `visibleSplatCount` userData across the scene graph and report the
 * totals to the data-loading monitor. Called once per update cycle after
 * points/lines/gsplats commits so the monitor's HUD shows the
 * post-clipping (and post-progressive-refinement) visible counts rather
 * than the raw loaded counts.
 */

import * as THREE from 'three';
import { isPointsUserData } from '../../../types/points';
import { isLinesUserData } from '../../../types/lines';
import { isGSplatsUserData } from '../../../types/gsplats';
import type { SceneLoaderMonitorPort } from '../../scene-loader-monitor-port';

/**
 * Traverse `rootGroup`, sum the per-mesh visible-counts userData for all
 * three geometry types symmetrically, and push the totals to `monitor`.
 * No-op when either argument is null.
 */
export function updateVisibleCountsInMonitor(
  rootGroup: THREE.Group | null,
  monitor: SceneLoaderMonitorPort | null
): void {
  if (!rootGroup || !monitor) return;

  let totalVisiblePoints = 0;
  let totalVisibleSegments = 0;
  let totalVisibleSplats = 0;

  rootGroup.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      if (isPointsUserData(object.userData)) {
        totalVisiblePoints += object.userData.visiblePointCount ?? 0;
      } else if (isLinesUserData(object.userData)) {
        totalVisibleSegments += object.userData.visibleSegmentCount ?? 0;
      } else if (isGSplatsUserData(object.userData)) {
        totalVisibleSplats += object.userData.visibleSplatCount ?? 0;
      }
    }
  });

  monitor.updateVisiblePoints(totalVisiblePoints);
  monitor.updateVisibleSegments(totalVisibleSegments);
  monitor.updateVisibleSplats(totalVisibleSplats);
}
