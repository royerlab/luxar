/**
 * Property tests for BandwidthWindow (fast-check).
 *
 * Companion to bandwidth-window.test.ts. Pins these algebraic invariants
 * across arbitrary sequences of record()/rate()/advance() operations:
 *
 *   [cache.md/H2][P12]
 *   - rate() >= 0 always (no negative-bandwidth artefact under any sequence).
 *   - rate() <= totalLiveBytes (the rate is bounded by the bytes still in
 *     the window divided by the windowSpan).
 *   - rate() === 0 when no record() has occurred OR when all entries have
 *     aged out beyond windowMs.
 *   - Compaction invariant: internal `window.length` stays bounded (no
 *     unbounded growth across record-and-prune cycles).
 *
 * The deterministic-trace tests in bandwidth-window.test.ts pin specific
 * boundary values (record(0), same-instant rate, R5 compaction). This
 * file exercises arbitrary traces to catch arithmetic mutations the
 * example-based suite would miss.
 */
import { describe, test, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fc from 'fast-check';
import { BandwidthWindow } from '../../../cache/multi-level-caching-store/bandwidth-window';

type Op = { kind: 'record'; bytes: number } | { kind: 'advance'; ms: number } | { kind: 'rate' };

const opArb: fc.Arbitrary<Op> = fc.oneof(
  {
    weight: 4,
    arbitrary: fc.record({
      kind: fc.constant<'record'>('record'),
      // Non-negative byte counts (record() is documented for transferred bytes).
      bytes: fc.integer({ min: 0, max: 100_000 }),
    }),
  },
  {
    weight: 3,
    arbitrary: fc.record({
      kind: fc.constant<'advance'>('advance'),
      // Bounded time advances so traces don't blow up.
      ms: fc.integer({ min: 0, max: 5_000 }),
    }),
  },
  {
    weight: 1,
    arbitrary: fc.record({ kind: fc.constant<'rate'>('rate') }),
  }
);

const WINDOW_MS = 10_000;

function internals(bw: BandwidthWindow) {
  return bw as unknown as {
    window: Array<{ timestamp: number; bytes: number }>;
    start: number;
  };
}

describe('BandwidthWindow [cache.md/H2][P12] property tests', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T00:00:00Z'));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test('rate() is always finite and non-negative under any trace', () => {
    fc.assert(
      fc.property(fc.array(opArb, { minLength: 0, maxLength: 200 }), (ops) => {
        const bw = new BandwidthWindow(WINDOW_MS);
        for (const op of ops) {
          if (op.kind === 'record') bw.record(op.bytes);
          else if (op.kind === 'advance') vi.advanceTimersByTime(op.ms);
          else {
            const r = bw.rate();
            // The load-bearing invariants.
            expect(Number.isFinite(r)).toBe(true);
            expect(r).toBeGreaterThanOrEqual(0);
          }
        }
        // Final rate post-trace is also well-defined.
        const final = bw.rate();
        expect(Number.isFinite(final)).toBe(true);
        expect(final).toBeGreaterThanOrEqual(0);
      }),
      { numRuns: 100 }
    );
  });

  test('rate() returns 0 when every entry has aged beyond the window', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 10_000 }), { minLength: 0, maxLength: 50 }),
        (byteRecords) => {
          const bw = new BandwidthWindow(WINDOW_MS);
          for (const b of byteRecords) bw.record(b);
          // Advance well past windowMs so EVERY entry is stale.
          vi.advanceTimersByTime(WINDOW_MS + 1);
          expect(bw.rate()).toBe(0);
        }
      ),
      { numRuns: 50 }
    );
  });

  test('rate() is bounded above by totalLiveBytes (rough bound: liveBytes/ms_per_window scaled to /s)', () => {
    // The strict algebraic bound is: rate <= sum(liveBytes) / max(1ms, span)
    // * 1000 (per source) — but with fake timers it's simpler to assert
    // the weaker integrative bound: rate <= totalRecordedBytes * 1000 / 1
    // (assuming the windowSpan is clamped to >=1ms). This catches any
    // mutation that multiplies by 10× or returns sum-only-without-span.
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: 1_000 }), { minLength: 1, maxLength: 20 }),
        (byteRecords) => {
          const bw = new BandwidthWindow(WINDOW_MS);
          let totalBytes = 0;
          for (const b of byteRecords) {
            bw.record(b);
            totalBytes += b;
          }
          const r = bw.rate();
          // Loose upper bound: rate cannot exceed totalBytes * 1000 B/s
          // (i.e. all bytes transferred in 1ms).
          expect(r).toBeLessThanOrEqual(totalBytes * 1000 + 1);
        }
      ),
      { numRuns: 50 }
    );
  });

  test('window.length stays bounded (R5 compaction prevents unbounded growth)', () => {
    fc.assert(
      fc.property(fc.integer({ min: 50, max: 500 }), (cycles) => {
        const bw = new BandwidthWindow(WINDOW_MS);
        // Sustained record-and-prune cycles: record one entry, age it
        // beyond the window, record another. Without compaction the
        // internal array would grow to 2*cycles.
        for (let i = 0; i < cycles; i++) {
          bw.record(100);
          vi.advanceTimersByTime(WINDOW_MS + 1);
        }
        bw.rate(); // walk start past stale entries
        bw.record(100); // trigger compaction
        // Post-compaction the array must hold a small constant number of
        // entries — definitely not O(cycles). The R5 invariant lets a
        // few entries linger before the next trim.
        const len = internals(bw).window.length;
        expect(len).toBeLessThan(cycles); // strict win over no-compaction
        expect(len).toBeLessThanOrEqual(100); // tight bound
      }),
      { numRuns: 20 }
    );
  });
});
