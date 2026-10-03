/**
 * Data loading memory configuration
 */
export interface DataLoadingMemoryConfig {
  /**
   * Fraction of the usable JS heap the viewer plans against; read by
   * `cache/heap-budget.ts`, which splits it between the cache pool and the
   * non-cache remainder (its own floors and shares are constants there).
   */
  targetHeapUsage: number;
}
