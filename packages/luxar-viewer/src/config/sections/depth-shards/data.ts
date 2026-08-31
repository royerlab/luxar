import type { DepthShardsConfig } from './types';

/**
 * Depth-shard defaults (cross-node depth ordering, spec §4).
 *
 * OFF by default. Turning it on changes the composited image of any scene with
 * two overlapping order-dependent nodes — for the better, but a change — and it
 * costs measured GPU time on exactly those scenes. Enable with
 * `?depthShards=<N>` or by flipping this once it has been run against real
 * datasets rather than the synthetic ones §7 measured.
 */
export const depthShardsConfig: DepthShardsConfig = {
  enabled: false, // URL: ?depthShards=N enables + pins, ?depthShards=0 disables
  // Generous because the measured cost saturates: 128 interleaved draws cost
  // +5.5 ms at 2M splats and 512 cost +6.4 ms. The lever is which nodes
  // interleave, not how finely they split.
  shardsPerNode: 16,
  // Bounds the term that IS linear in draw count, which dominates only at low
  // element counts where it is affordable anyway (~2 ms at 20k elements).
  maxInterleavedDraws: 512,
  // Below this a whole node is already a thin enough depth interval that the
  // error it can contribute is not worth an extra draw.
  minElements: 4_096,
};
