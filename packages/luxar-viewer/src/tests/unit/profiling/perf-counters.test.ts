import { describe, expect, it } from 'vitest';

import { computePerfSnapshot } from '../../../core/app/debug/perf-snapshot';
import {
  PERF_RECORD_RING_SIZE,
  PerfCounters,
  perfCounters,
} from '../../../profiling/perf-counters';

describe('PerfCounters', () => {
  it('returns a stable slot per name and accumulates with add', () => {
    const c = new PerfCounters();
    const a = c.slot('a');
    expect(c.slot('a')).toBe(a);
    c.add(a);
    c.add(a, 4);
    expect(c.get('a')).toBe(5);
    expect(c.get('never')).toBe(0);
  });

  it('keeps a high-water mark with max and overwrites with gauge', () => {
    const c = new PerfCounters();
    const hw = c.slot('hw');
    c.max(hw, 3);
    c.max(hw, 2);
    c.max(hw, 7);
    expect(c.get('hw')).toBe(7);
    const g = c.slot('g');
    c.gauge(g, 9);
    c.gauge(g, 1);
    expect(c.get('g')).toBe(1);
  });

  it('grows past its initial capacity without losing values', () => {
    const c = new PerfCounters();
    for (let i = 0; i < 300; i++) c.inc(`k${i}`, i);
    expect(c.get('k0')).toBe(0);
    expect(c.get('k299')).toBe(299);
    expect(Object.keys(c.snapshot())).toHaveLength(300);
  });

  it('bounds each record ring, dropping the oldest', () => {
    const c = new PerfCounters();
    for (let i = 0; i < PERF_RECORD_RING_SIZE + 10; i++) c.record('tick', i);
    const recs = c.records('tick');
    expect(recs).toHaveLength(PERF_RECORD_RING_SIZE);
    expect(recs[0]).toBe(10);
    expect(c.records('other')).toEqual([]);
  });

  it('reset zeroes counters and drops records but keeps slots valid', () => {
    const c = new PerfCounters();
    const s = c.slot('s');
    c.add(s, 2);
    c.record('r', 1);
    c.reset();
    expect(c.get('s')).toBe(0);
    expect(c.records('r')).toEqual([]);
    c.add(s);
    expect(c.get('s')).toBe(1);
  });

  it('reset keeps gauges: a gauge describes current state, not a window', () => {
    // Owners republish a gauge only when it CHANGES (the slice cache's
    // pinned-bytes gauge does), so zeroing it would make a gated reading
    // report 0 until the next change.
    const c = new PerfCounters();
    const pinned = c.slot('scache.pinnedBytes');
    const hw = c.slot('fetch.highWater');
    c.gauge(pinned, 4096);
    c.max(hw, 7);
    c.inc('render.count', 3);
    c.reset();
    expect(c.get('scache.pinnedBytes')).toBe(4096);
    expect(c.get('fetch.highWater')).toBe(0);
    expect(c.get('render.count')).toBe(0);
    c.gauge(pinned, 1024);
    expect(c.get('scache.pinnedBytes')).toBe(1024);
  });

  it('is exposed through getPerf().counters, even before the runtime is wired', () => {
    perfCounters.inc('test.exposed', 3);
    const snap = computePerfSnapshot();
    expect(snap.runtimeReady).toBe(false);
    expect(snap.counters['test.exposed']).toBeGreaterThanOrEqual(3);
  });
});
