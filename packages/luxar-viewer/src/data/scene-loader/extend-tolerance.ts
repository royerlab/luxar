/**
 * Pure helpers for extend_to_all tolerance derivation.
 *
 * Extracted from scene-loader.ts so the validation rules and the
 * cached "extend tolerance to infinity for these dim names" math can
 * be unit-tested without instantiating SceneLoader (which needs zarr,
 * ViewStateManager, LoaderRegistry, ...).
 *
 * @module data/scene-loader/extend-tolerance
 */

import type { SceneDimensions } from '../view-state-manager';

/**
 * The "infinite" tolerance value used to flag a dimension as
 * "extended across every value". Picked at 1e10 to keep numerical
 * routines stable while being effectively unreachable in real data.
 */
export const EXTEND_TO_ALL_TOLERANCE = 1e10;

/** True iff `obj` has at least one own enumerable property. */
export function hasOwnProperties(obj: Record<string, unknown>): boolean {
  for (const k in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, k)) return true;
  }
  return false;
}

/**
 * Validate that every dimension name in `extendDims` matches a
 * dimension's `.name` field in `dimensionMetadata`. Throws with a
 * comma-joined list of invalid names + the list of valid ones —
 * actionable error so the dataset author knows what to fix.
 */
export function validateExtendDims(
  extendDims: readonly string[],
  dimensionMetadata: ReadonlyArray<{ name?: string }>
): void {
  const validNames = new Set(
    dimensionMetadata.map((dim) => dim.name).filter((name): name is string => !!name)
  );
  const invalid = extendDims.filter((dimName) => !validNames.has(dimName));
  if (invalid.length > 0) {
    throw new Error(
      `Invalid extend_to_all dimension(s): ${invalid.join(', ')}. ` +
        `Valid dimensions: ${Array.from(validNames).join(', ')}`
    );
  }
}

/**
 * Return (or compute + cache) an extended tolerance array. Every
 * `extendDims` entry's matching dimension index in the tolerance
 * array is replaced with {@link EXTEND_TO_ALL_TOLERANCE}; everything
 * else is copied through from `baseTolerance` unchanged.
 *
 * The cache key is the sorted, comma-joined `extendDims` set so
 * order-permuted callers share the same cached array. Subsequent
 * calls with the same set return the same array reference (callers
 * must not mutate the result).
 *
 * @param baseTolerance - Per-dim tolerance array to extend.
 * @param extendDims - Dimension names that should be set to "all".
 * @param dimensionMetadata - The scene's dimensions, used to resolve
 *   names → indices.
 * @param cache - Mutable cache shared across calls.
 */
export function getOrComputeExtendedTolerance(
  baseTolerance: readonly number[],
  extendDims: readonly string[],
  dimensionMetadata: ReadonlyArray<{ name?: string }>,
  cache: Map<string, number[]>
): number[] {
  validateExtendDims(extendDims, dimensionMetadata);
  const key = extendDims.slice().sort().join(',');
  const cached = cache.get(key);
  if (cached) return cached;

  const computed = [...baseTolerance];
  for (const dimName of extendDims) {
    const dimIndex = dimensionMetadata.findIndex((d) => d.name === dimName);
    if (dimIndex >= 0 && dimIndex < computed.length) {
      computed[dimIndex] = EXTEND_TO_ALL_TOLERANCE;
    }
  }
  cache.set(key, computed);
  return computed;
}

/**
 * Type guard: narrow an arbitrary value to {@link SceneDimensions}.
 *
 * Only checks structural shape (`.dimensions` is an array). The
 * downstream code does the per-dimension validation; this guard
 * only rules out the obvious "not even an object with a dimensions
 * field" cases.
 */
export function isSceneDimensions(value: unknown): value is SceneDimensions {
  if (!value || typeof value !== 'object') return false;
  return Array.isArray((value as { dimensions?: unknown }).dimensions);
}
