/**
 * Unit tests for the debug-only per-stage LOD load timing accumulator.
 *
 * The module holds module-level mutable state (`enabled` flag + a Map of
 * per-stage stats), so each test resets via `resetLodLoadStats()` and
 * `setLodLoadStatsEnabled(false)` in beforeEach/afterEach to stay isolated.
 *
 * Timing comes from `performance.now()`; for the timed-wrapper tests we
 * spy on it to make the recorded durations deterministic. The pure
 * aggregation tests drive `recordLodLoadStage` directly with fixed ms.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  setLodLoadStatsEnabled,
  lodLoadStatsEnabled,
  recordLodLoadStage,
  timeLodStage,
  timeLodStageSync,
  snapshotLodLoadStats,
  resetLodLoadStats,
} from '../../../../data/scene-loader/lod-load-stats';

beforeEach(() => {
  resetLodLoadStats();
  setLodLoadStatsEnabled(false);
});

afterEach(() => {
  vi.restoreAllMocks();
  resetLodLoadStats();
  setLodLoadStatsEnabled(false);
});

describe('setLodLoadStatsEnabled / lodLoadStatsEnabled', () => {
  it('reflects the enabled flag and defaults to disabled', () => {
    expect(lodLoadStatsEnabled()).toBe(false);
    setLodLoadStatsEnabled(true);
    expect(lodLoadStatsEnabled()).toBe(true);
    setLodLoadStatsEnabled(false);
    expect(lodLoadStatsEnabled()).toBe(false);
  });
});

describe('recordLodLoadStage — enabled guard', () => {
  it('is a no-op when disabled (snapshot stays empty)', () => {
    expect(lodLoadStatsEnabled()).toBe(false);
    recordLodLoadStage('fetch', 12.5);
    recordLodLoadStage('decode', 99);
    expect(snapshotLodLoadStats()).toEqual({});
  });

  it('records when enabled', () => {
    setLodLoadStatsEnabled(true);
    recordLodLoadStage('fetch', 10);
    const snap = snapshotLodLoadStats();
    expect(Object.keys(snap)).toEqual(['fetch']);
    expect(snap.fetch.count).toBe(1);
  });
});

describe('snapshotLodLoadStats — aggregation + rounding', () => {
  beforeEach(() => setLodLoadStatsEnabled(true));

  it('aggregates count/totalMs/avgMs/maxMs across multiple samples', () => {
    recordLodLoadStage('fetch', 10);
    recordLodLoadStage('fetch', 20);
    recordLodLoadStage('fetch', 30);
    const snap = snapshotLodLoadStats();
    expect(snap.fetch.count).toBe(3);
    expect(snap.fetch.totalMs).toBe(60);
    expect(snap.fetch.avgMs).toBe(20); // 60 / 3
    expect(snap.fetch.maxMs).toBe(30);
  });

  it('tracks the running maximum (not just the last sample)', () => {
    recordLodLoadStage('decode', 5);
    recordLodLoadStage('decode', 50);
    recordLodLoadStage('decode', 7);
    expect(snapshotLodLoadStats().decode.maxMs).toBe(50);
  });

  it('keeps distinct stage names separate', () => {
    recordLodLoadStage('fetch', 10);
    recordLodLoadStage('gpu-commit', 4);
    const snap = snapshotLodLoadStats();
    expect(Object.keys(snap).sort()).toEqual(['fetch', 'gpu-commit']);
    expect(snap.fetch.count).toBe(1);
    expect(snap['gpu-commit'].count).toBe(1);
  });

  it('rounds totalMs/maxMs to 2 decimals and avgMs to 3 decimals', () => {
    // total = 1.111 + 2.222 = 3.333 -> totalMs 3.33, avg = 1.6665 -> 1.667 (toFixed(3))
    recordLodLoadStage('round', 1.111);
    recordLodLoadStage('round', 2.222);
    const snap = snapshotLodLoadStats();
    expect(snap.round.totalMs).toBe(3.33); // +(3.333).toFixed(2)
    expect(snap.round.avgMs).toBe(1.667); // +(1.6665).toFixed(3)
    expect(snap.round.maxMs).toBe(2.22); // +(2.222).toFixed(2)
  });

  it('avgMs guards against division by an absent count (Math.max(1, count))', () => {
    // count is always >= 1 once recorded; this verifies a single sample
    // yields avg == total without producing NaN/Infinity.
    recordLodLoadStage('single', 8.5);
    const snap = snapshotLodLoadStats();
    expect(snap.single.count).toBe(1);
    expect(snap.single.avgMs).toBe(8.5);
    expect(Number.isFinite(snap.single.avgMs)).toBe(true);
  });

  it('returns a fresh plain object snapshot (not the internal Map)', () => {
    recordLodLoadStage('fetch', 1);
    const a = snapshotLodLoadStats();
    const b = snapshotLodLoadStats();
    expect(a).not.toBe(b);
    expect(a).toEqual(b);
  });
});

describe('timeLodStage (async)', () => {
  it('returns the wrapped value and records timing when enabled', async () => {
    setLodLoadStatsEnabled(true);
    let t = 100;
    vi.spyOn(performance, 'now').mockImplementation(() => {
      const v = t;
      t += 25; // first call t0=100, second (in finally) =125 -> 25ms
      return v;
    });
    const result = await timeLodStage('async-stage', async () => 'value');
    expect(result).toBe('value');
    const snap = snapshotLodLoadStats();
    expect(snap['async-stage'].count).toBe(1);
    expect(snap['async-stage'].totalMs).toBe(25);
  });

  it('records timing even when the wrapped fn rejects, and rethrows', async () => {
    setLodLoadStatsEnabled(true);
    let t = 0;
    vi.spyOn(performance, 'now').mockImplementation(() => {
      const v = t;
      t += 10;
      return v;
    });
    await expect(
      timeLodStage('async-throw', async () => {
        throw new Error('boom');
      })
    ).rejects.toThrow('boom');
    const snap = snapshotLodLoadStats();
    expect(snap['async-throw'].count).toBe(1);
    expect(snap['async-throw'].totalMs).toBe(10);
  });

  it('passes through (no recording) when disabled', async () => {
    expect(lodLoadStatsEnabled()).toBe(false);
    const result = await timeLodStage('disabled', async () => 42);
    expect(result).toBe(42);
    expect(snapshotLodLoadStats()).toEqual({});
  });
});

describe('timeLodStageSync', () => {
  it('returns the wrapped value and records timing when enabled', () => {
    setLodLoadStatsEnabled(true);
    let t = 1000;
    vi.spyOn(performance, 'now').mockImplementation(() => {
      const v = t;
      t += 3;
      return v;
    });
    const result = timeLodStageSync('sync-stage', () => 7);
    expect(result).toBe(7);
    const snap = snapshotLodLoadStats();
    expect(snap['sync-stage'].count).toBe(1);
    expect(snap['sync-stage'].totalMs).toBe(3);
  });

  it('records timing even when the wrapped fn throws, and rethrows', () => {
    setLodLoadStatsEnabled(true);
    let t = 0;
    vi.spyOn(performance, 'now').mockImplementation(() => {
      const v = t;
      t += 5;
      return v;
    });
    expect(() =>
      timeLodStageSync('sync-throw', () => {
        throw new Error('kaboom');
      })
    ).toThrow('kaboom');
    expect(snapshotLodLoadStats()['sync-throw'].totalMs).toBe(5);
  });

  it('passes through (no recording) when disabled', () => {
    expect(lodLoadStatsEnabled()).toBe(false);
    const result = timeLodStageSync('disabled-sync', () => 'ok');
    expect(result).toBe('ok');
    expect(snapshotLodLoadStats()).toEqual({});
  });
});

describe('resetLodLoadStats', () => {
  it('clears all recorded entries', () => {
    setLodLoadStatsEnabled(true);
    recordLodLoadStage('a', 1);
    recordLodLoadStage('b', 2);
    expect(Object.keys(snapshotLodLoadStats())).toHaveLength(2);
    resetLodLoadStats();
    expect(snapshotLodLoadStats()).toEqual({});
  });

  it('does NOT change the enabled flag (only clears the stats map)', () => {
    setLodLoadStatsEnabled(true);
    recordLodLoadStage('a', 1);
    resetLodLoadStats();
    expect(lodLoadStatsEnabled()).toBe(true);
    // Recording still works after reset.
    recordLodLoadStage('a', 9);
    expect(snapshotLodLoadStats().a.count).toBe(1);
  });
});
