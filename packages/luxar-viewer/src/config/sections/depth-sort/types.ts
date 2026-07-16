/**
 * Depth-sort scheduling configuration (depth-sorting Phase 3).
 *
 * Gaussian splats in the order-dependent `normal` blending mode are
 * depth-sorted by an async worker (spec §5); this section tunes WHEN the
 * per-frame scheduler dispatches a re-sort as the camera moves (spec §6).
 * The sort kernel orders by view-space z, so a re-sort is only needed
 * when the view axis rotates (relative to the node) or the camera
 * translates along it far enough to change the behind-camera set —
 * translation orthogonal to the view axis cannot change the ordering.
 */
export interface DepthSortConfig {
  /** Master switch for depth sorting (default: true). When false, gsplat
   *  `normal`-mode nodes keep the identity (storage) order — Phase-1
   *  behavior. URL escape hatch: `?depthSort=0`. */
  enabled: boolean;
  /** View-axis rotation (relative to the node) that triggers a re-sort,
   *  in degrees (default: 3). Lower = fresher ordering, more sorts. */
  angleThresholdDeg: number;
  /** View-axis translation that triggers a re-sort, as a fraction of the
   *  node's bounding-sphere radius (default: 0.05). Only translation
   *  ALONG the view axis counts (see module doc above). */
  translationFraction: number;
}
