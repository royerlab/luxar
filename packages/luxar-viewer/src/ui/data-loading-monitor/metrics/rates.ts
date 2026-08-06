/**
 * Rate calculation for the data-loading monitor. The Monitor walks its
 * event ring buffer once per `calculateRates` call to compute per-second
 * rolling rates (queries, loads) plus a bandwidth measure over a tighter
 * window. The computation is pure (events + window sizes → rates); the
 * cache-and-timeout dance stays monitor-internal so this module can be
 * unit-tested directly.
 */

import type { MonitorEvent } from '../../../types/data-monitor-types';

/** The rates the Monitor caches and exposes to its tabs. */
export interface RatesSnapshot {
  queriesPerSec: number;
  loadsPerSec: number;
  bandwidth: number;
  lastCalculated: number;
}

export interface CalculateRatesParams {
  /** Now-timestamp (ms). Pass `Date.now()` from the caller. */
  now: number;
  /** Event ring buffer to walk; iterated newest-first for early exit. */
  events: readonly MonitorEvent[];
  /** Rolling window over which to count queries/loads. */
  rateWindowMs: number;
  /** Tighter window for bandwidth (`load` event memory bytes). */
  bandwidthWindowMs: number;
  /** Cache timeout — re-use cached values if last calc was recent. */
  cacheTimeoutMs: number;
  /** Mutable target — rates write into this object in place. */
  rates: RatesSnapshot;
}

/**
 * Update `rates` in place based on the events newer than the
 * rolling window. No-ops when the cache is still fresh
 * (`now - rates.lastCalculated < cacheTimeoutMs`).
 */
export function calculateRates(params: CalculateRatesParams): void {
  const { now, events, rateWindowMs, bandwidthWindowMs, cacheTimeoutMs, rates } = params;

  if (now - rates.lastCalculated < cacheTimeoutMs) {
    return;
  }

  let queries = 0;
  let loads = 0;
  let bandwidth = 0;

  const rateCutoff = now - rateWindowMs;
  const bandwidthCutoff = now - bandwidthWindowMs;

  // Iterate backwards: events are append-only newest-last, so we
  // can early-exit when we cross the rate window cutoff.
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
    if (event.timestamp < rateCutoff) break;

    switch (event.type) {
      case 'query':
        queries++;
        break;
      case 'load':
        loads++;
        if (event.timestamp > bandwidthCutoff) {
          bandwidth += event.data.memory || 0;
        }
        break;
    }
  }

  const windowSeconds = rateWindowMs / 1000;
  rates.queriesPerSec = queries / windowSeconds;
  rates.loadsPerSec = loads / windowSeconds;
  // bandwidth is bytes/sec, normalised by the bandwidth window so the
  // value is comparable across configurations (the window is not
  // always 1000 ms).
  const bandwidthSeconds = bandwidthWindowMs / 1000;
  rates.bandwidth = bandwidthSeconds > 0 ? bandwidth / bandwidthSeconds : bandwidth;
  rates.lastCalculated = now;
}
