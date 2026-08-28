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
  /** Deadline for ONE attempt at the SortWorker's one-time `initialize()`
   *  (WASM load + instantiate), in ms. Peer of
   *  `dataLoading.performance.workerInitTimeoutMs`. It bounds the attempt,
   *  not the subsystem: a miss TERMINATES that worker, and the bounded
   *  per-frame retry then constructs a FRESH one after a backoff and forces
   *  a re-registration sweep once one lands (`reregisterAfterLateWorkerInit`).
   *  So a short value buys faster failure detection at the price of extra
   *  worker spawns + WASM instantiations, plus that forced re-commit of every
   *  sorted node.
   *  `0` installs NO timer at all (withTimeout convention), which is a
   *  debugging escape hatch rather than a tuning option: a worker that
   *  neither answers nor errors then leaves the init promise pending
   *  forever, and every order-dependent commit parks another continuation on
   *  it — precisely the accumulation the deadline exists to prevent. */
  workerInitTimeoutMs: number;
  /** Largest element count for which the FIRST ordering after a commit is
   *  computed synchronously, on the main thread, inside the commit itself
   *  (default: 250,000). `0` disables the synchronous path entirely.
   *
   *  Without it, every commit of an order-dependent node writes a storage-order
   *  fallback and waits ~5 ms for the worker's answer, so at least one frame
   *  renders unsorted. That is invisible on a one-off load and continuous
   *  during nD playback, where a commit lands at EVERY timepoint: measured on
   *  the `cloud` demo, the fallback composited only 61.7% of sampled element
   *  pairs in correct back-to-front order against 100% for a real sort, once
   *  per timepoint, which reads as a flash.
   *
   *  The kernel is a counting sort — two O(n) passes plus a 65,536-bucket
   *  histogram — so the cost is bounded and measurable. Timed in-browser:
   *
   *  ```
   *  34k    100k   250k   500k    1M      1.65M
   *  0.8ms  1.1ms  2.5ms  8.0ms  16.2ms  31.7ms
   *  ```
   *
   *  250k stays inside a 60 Hz frame with room to spare on a slow machine and
   *  covers every animated demo node in the repo. Above it the async path is
   *  the only sane answer and the storage-order frame is accepted — but see
   *  `repairSortedIndexForCount`, which keeps that frame much closer to sorted
   *  than storage order was. */
  syncSortMaxElements: number;
}
