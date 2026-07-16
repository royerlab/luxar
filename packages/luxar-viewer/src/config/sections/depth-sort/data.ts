import type { DepthSortConfig } from './types';

/**
 * Depth-sort scheduling defaults (depth-sorting Phase 3, spec §6).
 *
 * The thresholds trade ordering freshness against sort traffic: at 3° /
 * 5% of the bounding radius a slow orbit re-sorts a few times per
 * second while an idle camera never dispatches. Frames between dispatch
 * and resolve render the previous order — bounded staleness, standard
 * 3DGS behavior.
 */
export const depthSortConfig: DepthSortConfig = {
  enabled: true, // URL escape hatch: ?depthSort=0 (pins identity ordering)
  angleThresholdDeg: 3,
  translationFraction: 0.05,
};
