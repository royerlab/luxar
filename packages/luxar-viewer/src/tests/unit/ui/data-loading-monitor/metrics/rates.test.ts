/**
 * Unit tests for `calculateRates`. The pure-function shape lets us
 * drive each branch (cache short-circuit, event-type counts, rolling
 * windows) without spinning up the full DataLoadingMonitor.
 */

import { describe, it, expect } from 'vitest';
import {
  calculateRates,
  type RatesSnapshot,
} from '../../../../../ui/data-loading-monitor/metrics/rates';
import type { MonitorEvent } from '../../../../../types/data-monitor-types';

function freshRates(): RatesSnapshot {
  return {
    queriesPerSec: 0,
    loadsPerSec: 0,
    bandwidth: 0,
    lastCalculated: 0,
  };
}

function makeEvent(
  partial: Partial<MonitorEvent> & Pick<MonitorEvent, 'type' | 'timestamp'>
): MonitorEvent {
  return {
    loaderType: 'point-spatial-index',
    path: '/p',
    data: {},
    ...partial,
  } as MonitorEvent;
}

describe('calculateRates', () => {
  it('cache short-circuit: returns without recomputing if last calc was recent', () => {
    const rates = freshRates();
    rates.lastCalculated = 1000;
    rates.queriesPerSec = 42; // sentinel — must not change

    calculateRates({
      now: 1100, // only 100ms elapsed
      events: [],
      rateWindowMs: 5000,
      bandwidthWindowMs: 1000,
      cacheTimeoutMs: 500, // 100ms < 500ms → short-circuit
      rates,
    });

    expect(rates.queriesPerSec).toBe(42);
    expect(rates.lastCalculated).toBe(1000);
  });

  it('counts events of each type in the rate window', () => {
    const rates = freshRates();
    const now = 10_000;
    const events: MonitorEvent[] = [
      makeEvent({ type: 'query', timestamp: now - 4000 }),
      makeEvent({ type: 'query', timestamp: now - 3000 }),
      makeEvent({ type: 'load', timestamp: now - 2000, data: { memory: 100 } }),
    ];

    calculateRates({
      now,
      events,
      rateWindowMs: 5000,
      bandwidthWindowMs: 1000,
      cacheTimeoutMs: 100,
      rates,
    });

    // 5s window: 2 queries, 1 load
    // queriesPerSec = 2 / 5s = 0.4
    expect(rates.queriesPerSec).toBeCloseTo(0.4, 5);
    expect(rates.loadsPerSec).toBeCloseTo(0.2, 5);
  });

  it('bandwidth uses the tighter 1s window, not the full rate window', () => {
    const rates = freshRates();
    const now = 10_000;
    const events: MonitorEvent[] = [
      // Outside 1s window — counted for loadsPerSec, NOT bandwidth.
      makeEvent({ type: 'load', timestamp: now - 3000, data: { memory: 1000 } }),
      // Inside 1s window — counted for both.
      makeEvent({ type: 'load', timestamp: now - 500, data: { memory: 50 } }),
    ];

    calculateRates({
      now,
      events,
      rateWindowMs: 5000,
      bandwidthWindowMs: 1000,
      cacheTimeoutMs: 100,
      rates,
    });

    // 2 loads / 5s = 0.4 loads/sec.
    expect(rates.loadsPerSec).toBeCloseTo(0.4, 5);
    // Bandwidth: only the load inside the 1s window contributes.
    expect(rates.bandwidth).toBe(50);
  });

  it('events outside the rate window are excluded (early-exit on backwards walk)', () => {
    const rates = freshRates();
    const now = 10_000;
    // Older first (insertion order matches event ring buffer); the
    // backwards walk hits the newer events first.
    const events: MonitorEvent[] = [
      makeEvent({ type: 'query', timestamp: now - 10_000 }), // out of window
      makeEvent({ type: 'query', timestamp: now - 6_000 }), // out of window
      makeEvent({ type: 'query', timestamp: now - 1_000 }), // in window
    ];

    calculateRates({
      now,
      events,
      rateWindowMs: 5000,
      bandwidthWindowMs: 1000,
      cacheTimeoutMs: 100,
      rates,
    });

    // Only 1 query counted (1 / 5s = 0.2).
    expect(rates.queriesPerSec).toBeCloseTo(0.2, 5);
  });

  it('updates lastCalculated to now on completion', () => {
    const rates = freshRates();
    calculateRates({
      now: 12345,
      events: [],
      rateWindowMs: 5000,
      bandwidthWindowMs: 1000,
      cacheTimeoutMs: 100,
      rates,
    });
    expect(rates.lastCalculated).toBe(12345);
  });

  it('empty events produce zero rates', () => {
    const rates = freshRates();
    calculateRates({
      now: 10_000,
      events: [],
      rateWindowMs: 5000,
      bandwidthWindowMs: 1000,
      cacheTimeoutMs: 100,
      rates,
    });
    expect(rates.queriesPerSec).toBe(0);
    expect(rates.loadsPerSec).toBe(0);
    expect(rates.bandwidth).toBe(0);
  });

  it('load events without data.memory contribute 0 to bandwidth', () => {
    const rates = freshRates();
    const now = 10_000;
    const events: MonitorEvent[] = [
      makeEvent({ type: 'load', timestamp: now - 200, data: {} }),
      makeEvent({ type: 'load', timestamp: now - 100, data: { memory: 200 } }),
    ];

    calculateRates({
      now,
      events,
      rateWindowMs: 5000,
      bandwidthWindowMs: 1000,
      cacheTimeoutMs: 100,
      rates,
    });

    expect(rates.bandwidth).toBe(200); // only the second event has memory
  });
});
