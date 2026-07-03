/**
 * Aggregate per-mesh `visiblePointCount` / `visibleSegmentCount` /
 * `visibleSplatCount` userData (points + lines + gsplats, symmetrically)
 * across the scene graph and report the totals to the data-loading
 * monitor. Called once per update cycle after the points/lines/gsplats
 * commits so the monitor's HUD shows the post-clipping (and
 * post-progressive-refinement) visible counts rather than the raw loaded
 * counts.
 *
 * Alongside the totals, a per-path map (mesh `name` is the scene-graph
 * path) is pushed via `updateVisibleCountsByPath` so the monitor's
 * scene-graph tree can show per-node visible counts in badge tooltips.
 *
 * Only meshes that are actually rendered are counted: the walk skips any
 * subtree whose root is `visible === false`. This excludes the inactive
 * levels of a substitutive `kind=lod` group (the registry hides all but
 * the active child) — without the skip every loaded level's userData would
 * be summed, inflating the visible count ~K×. It also excludes layers the
 * user has toggled off.
 */

import * as THREE from 'three';
import { isPointsUserData } from '../../../types/points';
import { isLinesUserData } from '../../../types/lines';
import { isGSplatsUserData } from '../../../types/gsplats';
import type { SceneLoaderMonitorPort } from '../../scene-loader-monitor-port';

/**
 * Traverse `rootGroup`, sum the per-mesh visible-counts userData for all
 * three geometry types symmetrically, and push the totals (plus a
 * per-path breakdown) to `monitor`. No-op when either argument is null.
 */
export function updateVisibleCountsInMonitor(
  rootGroup: THREE.Group | null,
  monitor: SceneLoaderMonitorPort | null
): void {
  if (!rootGroup || !monitor) return;

  let totalVisiblePoints = 0;
  let totalVisibleSegments = 0;
  let totalVisibleSplats = 0;
  const byPath = new Map<string, number>();

  // Manual recursion rather than THREE's `traverse`, which visits every
  // descendant regardless of visibility. Pruning at `visible === false`
  // boundaries means hidden subtrees (inactive LOD levels, toggled-off
  // layers) contribute nothing to the visible tally.
  const visit = (object: THREE.Object3D): void => {
    if (!object.visible) return;
    if (object instanceof THREE.Mesh) {
      let visible: number | undefined;
      if (isPointsUserData(object.userData)) {
        visible = object.userData.visiblePointCount ?? 0;
        totalVisiblePoints += visible;
      } else if (isLinesUserData(object.userData)) {
        visible = object.userData.visibleSegmentCount ?? 0;
        totalVisibleSegments += visible;
      } else if (isGSplatsUserData(object.userData)) {
        visible = object.userData.visibleSplatCount ?? 0;
        totalVisibleSplats += visible;
      }
      if (visible !== undefined && object.name) {
        // Mesh `name` is the scene-graph path (set by node-factory).
        byPath.set(object.name, (byPath.get(object.name) ?? 0) + visible);
      }
    }
    for (const child of object.children) visit(child);
  };
  // The root group's own visibility shouldn't gate the whole scene
  // (callers pass the scene root); descend straight into its children.
  for (const child of rootGroup.children) visit(child);

  monitor.updateVisiblePoints(totalVisiblePoints);
  monitor.updateVisibleSegments(totalVisibleSegments);
  monitor.updateVisibleSplats(totalVisibleSplats);
  monitor.updateVisibleCountsByPath(byPath);
}
