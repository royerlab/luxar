import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { BandwidthWindow } from '../../../cache/multi-level-caching-store/bandwidth-window';

/**
 * Helper: reach into private fields to assert the R5 amortized-compaction
 * invariant. This is legitimate inside the unit's own test file — the
 * invariant is implementation-defined and lives here, not in callers.
 */
function internals(bw: BandwidthWindow) {
  return bw as unknown as {
    window: Array<{ timestamp: number; bytes: number }>;
    start: number;
  };
}

describe('BandwidthWindow', () => {
  const WINDOW_MS = 10_000;
  let bw: BandwidthWindow;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));
    bw = new BandwidthWindow(WINDOW_MS);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('rate()', () => {
    it('returns 0 on a fresh instance', () => {
      expect(bw.rate()).toBe(0);
    });

    it('returns bytes/sec across two entries inside the window', () => {
      // Two entries 5 seconds apart, 4000 + 2000 = 6000 bytes over 5s span.
      bw.record(4000);
      vi.advanceTimersByTime(3000);
      bw.record(2000);
      vi.advanceTimersByTime(2000);
      // now - first.timestamp = 5000ms => span = 5s, rate = 1200 B/s.
      const rate = bw.rate();
      expect(rate).toBeGreaterThan(1100);
      expect(rate).toBeLessThan(1300);
    });

    it('returns 0 when every entry has aged out of the window', () => {
      bw.record(1000);
      vi.advanceTimersByTime(WINDOW_MS + 1);
      expect(bw.rate()).toBe(0);
    });

    it('advances start past stale entries without losing fresh ones', () => {
      // 5 stale + 3 fresh.
      for (let i = 0; i < 5; i++) bw.record(1000);
      vi.advanceTimersByTime(WINDOW_MS + 1);
      for (let i = 0; i < 3; i++) bw.record(2000);

      const rate = bw.rate();
      expect(rate).toBeGreaterThan(0);
      // After the walk, start should have advanced past the 5 stale ones.
      expect(internals(bw).start).toBe(5);
    });
  });

  describe('R5 amortized compaction (load-bearing)', () => {
    it('compacts the array when the dead prefix exceeds half', () => {
      // Seed 21 stale entries.
      for (let i = 0; i < 21; i++) bw.record(100);
      // Age them out so the next rate() call walks start past all 21.
      vi.advanceTimersByTime(WINDOW_MS + 1);
      // One fresh entry.
      bw.record(1000);

      // Walk start past the 21 stale entries.
      bw.rate();
      expect(internals(bw).start).toBe(21);
      expect(internals(bw).window.length).toBe(22);

      // The next record() should observe start (21) > length/2 (≈11) and
      // compact: array sliced down to the live tail, start reset to 0.
      bw.record(500);
      expect(internals(bw).start).toBe(0);
      expect(internals(bw).window.length).toBeLessThan(22);
      // At least the two fresh entries remain.
      expect(internals(bw).window.length).toBeGreaterThanOrEqual(2);
    });

    it('keeps the buffer bounded under sustained record-and-prune cycles', () => {
      // Simulate 1000 cycles: record, then age out, then record again.
      // Without compaction, window.length would grow to 2000+. With R5 it
      // stays bounded by ~2× the live tail.
      for (let cycle = 0; cycle < 1000; cycle++) {
        bw.record(100);
        vi.advanceTimersByTime(WINDOW_MS + 1);
      }
      bw.rate(); // walk start to the end.
      bw.record(100); // triggers compaction (start ≫ length/2).

      // After the burst, the array should be aggressively compacted, not
      // unbounded.
      expect(internals(bw).window.length).toBeLessThan(50);
    });
  });

  describe('record()', () => {
    it('does not compact when dead prefix is small', () => {
      bw.record(1);
      bw.record(2);
      bw.record(3);
      // start is still 0; nothing aged out yet.
      bw.record(4);
      expect(internals(bw).start).toBe(0);
      expect(internals(bw).window.length).toBe(4);
    });
  });
});
