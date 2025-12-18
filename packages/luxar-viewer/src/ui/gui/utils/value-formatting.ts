/**
 * Value formatting utilities for controllers
 */

/**
 * Clamp a value between min and max
 *
 * @param value - Value to clamp
 * @param min - Minimum value (optional)
 * @param max - Maximum value (optional)
 * @returns Clamped value
 */
export function clamp(value: number, min?: number, max?: number): number {
  if (min !== undefined && value < min) {
    return min;
  }
  if (max !== undefined && value > max) {
    return max;
  }
  return value;
}

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
