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
  // Generous relative to the data pool's 10 s: this one deadline covers a
  // cold WASM fetch + instantiate, and the reply lands on whatever the main
  // thread is doing. Missing it is transient (retried when the loader goes
  // idle), so erring long costs nothing but a later first sort.
  workerInitTimeoutMs: 30_000,
  // Measured: 2.5 ms at this count, 8 ms at 500k. Shared by every commit
  // between frame evaluations so a multi-node slice cannot multiply the cost.
  syncSortMaxElements: 250_000,
};
