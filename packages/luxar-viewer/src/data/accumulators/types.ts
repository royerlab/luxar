/**
 * Shared interfaces for the per-geometry accumulator classes.
 *
 * Each geometry-type accumulator (Points/Lines/GSplats) implements the
 * `DataAccumulator<T>` contract with its own `TGetArgs` / `TFillArgs`
 * shape and reports stats via `AccumulatorStats`. Splitting this out
 * lets the three per-type files import from a common 80-line header
 * without re-declaring the contract.
 */

/**
 * Generic accumulator interface
 */
export interface DataAccumulator<
  TData,
  TGetArgs extends unknown[] = unknown[],
  TFillArgs extends unknown[] = unknown[],
> {
  /**
   * Ensure capacity (grow if needed)
   * @returns true if capacity was grown
   */
  ensureCapacity(needed: number): boolean;

  /**
   * Get data view (subarray of internal buffers)
   *
   * Signature varies by type:
   * - Points: getData(count: number): LoadedPointsData
   * - Lines: getData(segmentCount: number, vertexCount: number): LoadedLinesData
   * - GSplats: getData(count: number): LoadedGSplatsData
   */
  getData(...args: TGetArgs): TData;

  /**
   * Fill accumulator at offset(s)
   *
   * Signature varies by type:
   * - Points: fill(offset: number, data: Partial<TData>)
   * - Lines: fill(segmentOffset: number, vertexOffset: number, data: Partial<TData>)
   * - GSplats: fill(offset: number, data: Partial<TData>)
   */
  fill(...args: TFillArgs): void;

  /**
   * Get statistics
   */
  getStats(): AccumulatorStats;

  /**
   * Dispose (release memory)
   */
  dispose(): void;
}

export interface AccumulatorStats {
  capacity: number;
  allocations: number;
  growthEvents: number;
  memoryMB: number;
}
