/**
 * Reads the per-node `committedLODCount` stamp off the live THREE scene, so
 * the monitor's Scene-Graph tab can report ladder rungs that are actually ON
 * SCREEN rather than the progressive loader's cursor.
 *
 * The two are not the same number, and they diverge precisely when someone is
 * trying to diagnose a problem. A loader advances its cursor as each rung
 * ARRIVES; the stamp is written by the commit that puts geometry on screen. A
 * pass whose commit fails or is superseded therefore leaves the loader
 * claiming rungs the viewer never drew.
 *
 * On the hosted Cosmicflows/Laniakea demo this was not a cosmetic gap: basins
 * that had run out of memory, stopped refining and frozen at a coarse prefix
 * reported "LOD 7/7 ~100%". The monitor was the one surface that could have
 * shown the stall, and it showed the opposite (#2426).
 *
 * Lives beside `draw-order-provider` (the other reader built over the root
 * group) so `monitor-wiring` stays free of THREE imports.
 *
 * @module data/scene-loader/monitor/committed-lod-reader
 */

import * as THREE from 'three';

/**
 * Build a reader over `rootGroup`. Cheap enough to call each monitor tick — it
 * walks the graph reading already-written userData and never triggers work.
 *
 * Unlike the visible-counts walk this does NOT prune hidden subtrees: a node
 * toggled off in the Layers panel still has a committed ladder, and reporting
 * it as 0 rungs would misrepresent a node the user has merely hidden.
 */
export function createCommittedLODCountReader(
  rootGroup: THREE.Group | null
): () => ReadonlyMap<string, number> {
  return () => {
    const out = new Map<string, number>();
    if (!rootGroup) return out;
    rootGroup.traverse((object) => {
      const count = (object.userData as { committedLODCount?: unknown } | undefined)
        ?.committedLODCount;
      if (object.name && typeof count === 'number') out.set(object.name, count);
    });
    return out;
  };
}
