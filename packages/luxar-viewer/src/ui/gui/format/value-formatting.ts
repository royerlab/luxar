/**
 * Value formatting utilities for controllers.
 *
 * `clamp` lives in `src/utils/clamp.ts` (cross-cutting foundation) so
 * `rendering` can import it without crossing layers. Re-exported here
 * for back-compat with existing `ui/` callers.
 */

export { clamp } from '../../../utils/clamp';

/**
 * Format a number for display based on step value
 *
 * @param value - Number to format
 * @param step - Step value (determines decimal places)
 * @returns Formatted string
 */
export function formatNumber(value: number, step?: number): string {
  if (step === undefined) {
    return String(value);
  }

  // Count decimal places in step
  const stepStr = String(step);
  const decimalIndex = stepStr.indexOf('.');
  const decimals = decimalIndex === -1 ? 0 : stepStr.length - decimalIndex - 1;

  return value.toFixed(decimals);
}

/**
 * Parse a value from string, with fallback
 *
 * @param str - String to parse
 * @param fallback - Fallback value if parsing fails
 * @returns Parsed number or fallback
 */
export function parseNumber(str: string, fallback: number): number {
  const parsed = parseFloat(str);
  return isNaN(parsed) ? fallback : parsed;
}
