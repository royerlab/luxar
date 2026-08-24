/**
 * Number, byte, and memory-pressure formatting helpers.
 */

import { getColorClass } from './primitives';

// Helper functions (exported for use by value update functions in the monitor)

export function formatNumber(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + 'GB';
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + 'MB';
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + 'KB';
  return bytes.toFixed(0) + 'B';
}

/**
 * Get CSS color class for cache memory usage percentage
 */
export function getCacheMemoryColorClass(percent: number): string {
  if (percent <= 60) return getColorClass('success');
  if (percent <= 80) return getColorClass('warning');
  return getColorClass('error');
}
