/**
 * Generic numeric clamp.
 *
 * Lives in `utils/` (the cross-cutting foundation layer) so any layer can
 * import it without crossing layer boundaries. The original copy lived in a
 * GUI formatting helper that `rendering` could not reach under the
 * dependency-cruiser layer order.
 *
 * @module utils/clamp
 */

/**
 * Clamp a value to `[min, max]`. Either bound may be omitted; with both
 * omitted the value passes through unchanged.
 */
export function clamp(value: number, min?: number, max?: number): number {
  if (min !== undefined && value < min) return min;
  if (max !== undefined && value > max) return max;
  return value;
}
