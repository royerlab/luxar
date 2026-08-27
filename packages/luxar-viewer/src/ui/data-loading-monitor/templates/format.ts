/**
 * Number, byte, and memory-pressure formatting helpers.
 */

import type { CacheMetrics } from '../../../types/data-monitor-types';
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
 * Format the network summary shared by the initial Overview render and live updates.
 * Cumulative bytes span all cache tiers so the card remains informative on warm reloads;
 * providers predating that counter fall back to network bytes.
 */
export function networkSummary(network: CacheMetrics['network']): {
  dataLoaded: string;
  detail: string;
} {
  if (!network) return { dataLoaded: '0B', detail: '0B net' };

  const dataLoaded = network.totalBytesServed ?? network.bytesTransferred;
  // Demand reads served across all tiers; falls back to the network request
  // count for providers predating the field (mirrors the bytes fallback).
  const requestsServed = network.totalRequestsServed ?? network.requestCount;
  return {
    dataLoaded: formatBytes(dataLoaded),
    detail: `${formatBytes(network.bytesTransferred)} net · ${formatBytes(network.bandwidth)}/s · ${requestsServed.toLocaleString()} reqs`,
  };
}

/**
 * Get CSS color class for cache memory usage percentage
 */
export function getCacheMemoryColorClass(percent: number): string {
  if (percent <= 60) return getColorClass('success');
  if (percent <= 80) return getColorClass('warning');
  return getColorClass('error');
}
