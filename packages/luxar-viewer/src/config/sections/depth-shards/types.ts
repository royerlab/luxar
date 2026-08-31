/**
 * Depth-shard configuration — cross-node depth ordering
 * (`docs/guides/specs/CROSS_NODE_DEPTH_ORDERING_SPEC.md`).
 */
export interface DepthShardsConfig {
  /**
   * Master switch. When false no node is ever split and draw order is exactly
   * what the group-major assignment always produced.
   *
   * URL escape hatch: `?depthShards=0`.
   *
   * **Default false**: the feature reduces a real ordering error, but it also
   * costs measured GPU time on scenes that overlap (see `maxInterleavedDraws`),
   * and it changes the composited image of any scene with two overlapping
   * order-dependent nodes. Both make it an opt-in until it has run against real
   * datasets rather than the synthetic ones §7 measured.
   */
  enabled: boolean;

  /**
   * Shards a qualifying node is split into.
   *
   * Generous on purpose. §7 measured the cost as sublinear in draw count and
   * saturating — at 2M splats, 128 interleaved draws cost +5.5 ms and 512 cost
   * +6.4 ms, while interleaving at all cost the first +5.5 ms — so the expensive
   * decision is WHICH nodes interleave, not how finely they are split. Tuning
   * this down forfeits ordering accuracy without recovering the cost.
   *
   * A node is never split into more shards than it has elements.
   */
  shardsPerNode: number;

  /**
   * Soft ceiling on total interleaved draws across the scene; exceeding it
   * scales every qualifying node's count down proportionally (never below 2).
   *
   * Bounds the small term that IS linear in draw count. That term dominates at
   * low element counts — at 20k elements, 512 draws made a 0.3-0.5 ms frame
   * 5-6× more expensive — though in absolute terms it stayed ~2 ms, which such a
   * scene can afford. 512 is where §7 stopped measuring, not a knee.
   */
  maxInterleavedDraws: number;

  /**
   * Nodes with fewer elements than this are never split.
   *
   * Below a few thousand elements the whole node is a thin enough depth interval
   * that the ordering error it can contribute is not worth an extra draw.
   */
  minElements: number;
}
