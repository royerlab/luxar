/**
 * Unit tests for UpdateProfiler.
 *
 * The profiler depends only on `performance.now()` (available in jsdom) and
 * the project log utility, so tests run with no DOM, no THREE, and no zarr.
 *
 * For tests that need exact ms values (EMA convergence, budget), we drive
 * `performance.now` via a controlled clock helper. For tests that only care
 * about ordering or non-negativity, we let the real clock run.
 *
 * Contract under test (post-fix):
 * - `rootEntry.children[*].count` increments exactly once per measurement:
 *   each child session merges itself into the persistent tree at end(), and
 *   the root session merges only its own counters (no re-merging of
 *   children, which would double-count).
 * - `time()` saves/restores `currentSessionContext`, so nested `time()` calls
 *   build a parent/child hierarchy as the class docstring advertises.
 * - Child `avgMs` therefore converges at the EMA rate (alpha=0.1), not at
 *   half that rate.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  UpdateProfiler,
  formatMs,
  hasOverBudget,
  type TimingEntry,
} from '../../../profiling/update-profiler';

const FRAME_BUDGET_MS = 16.67;

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/**
 * Controlled clock: each call to performance.now() returns the current value
 * of `now`. Tests drive it explicitly via clock.advance() / clock.set().
 */
function controlledClock() {
  let now = 0;
  vi.spyOn(performance, 'now').mockImplementation(() => now);
  return {
    advance(ms: number) {
      now += ms;
    },
    set(ms: number) {
      now = ms;
    },
    restore() {
      (performance.now as unknown as { mockRestore?: () => void }).mockRestore?.();
    },
  };
}

function findChild(entry: TimingEntry, name: string): TimingEntry | undefined {
  if (entry.name === name) return entry;
  for (const c of entry.children) {
    const found = findChild(c, name);
    if (found) return found;
  }
  return undefined;
}

// ---------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------

describe('formatMs', () => {
  it('formats sub-0.1ms values as "<0.1ms"', () => {
    expect(formatMs(0)).toBe('<0.1ms');
    expect(formatMs(0.05)).toBe('<0.1ms');
    expect(formatMs(0.099)).toBe('<0.1ms');
  });

  it('formats single-digit-ms values with one decimal', () => {
    expect(formatMs(1.0)).toBe('1.0ms');
    expect(formatMs(5.234)).toBe('5.2ms');
    expect(formatMs(9.99)).toBe('10.0ms');
  });

  it('rounds 10ms+ values to integers', () => {
    expect(formatMs(10)).toBe('10ms');
    expect(formatMs(12.6)).toBe('13ms');
    expect(formatMs(120)).toBe('120ms');
  });
});

describe('hasOverBudget', () => {
  const make = (over: boolean | undefined, children: TimingEntry[] = []): TimingEntry => ({
    name: 't',
    lastMs: 0,
    avgMs: 0,
    count: 0,
    overBudget: over,
    children,
  });

  it('returns false for an entry with no overBudget anywhere', () => {
    const root = make(false, [make(false), make(undefined)]);
    expect(hasOverBudget(root)).toBe(false);
  });

  it('returns true if the root itself is over budget', () => {
    const root = make(true);
    expect(hasOverBudget(root)).toBe(true);
  });

  it('returns true if any descendant is over budget', () => {
    const root = make(false, [make(false, [make(true)])]);
    expect(hasOverBudget(root)).toBe(true);
  });
});

// ---------------------------------------------------------------------
// UpdateProfiler — session lifecycle
// ---------------------------------------------------------------------

describe('UpdateProfiler — session lifecycle', () => {
  let profiler: UpdateProfiler;

  beforeEach(() => {
    profiler = new UpdateProfiler();
  });

  // [profiling.md/O3][P9] Consolidate the parallel "inactive before" /
  // "active between" pair into a single it.each over (action → expected
  // isActive) — the two cases differ only by the action taken.
  const setupNoop = (_p: UpdateProfiler): void => undefined;
  const setupBegin = (p: UpdateProfiler): void => {
    p.beginUpdate();
  };
  const setupBeginEnd = (p: UpdateProfiler): void => {
    p.beginUpdate();
    p.endUpdate();
  };
  it.each([
    ['before any update', setupNoop, false],
    ['inside an open update', setupBegin, true],
    ['after the update ends', setupBeginEnd, false],
  ] as const)('isActive() reports %s as %s', (_label, setup, expected) => {
    setup(profiler);
    expect(profiler.isActive()).toBe(expected);
  });

  it('current() returns a NoOp session when no update is active', () => {
    const s = profiler.current();
    expect(() => s.setMetadata({ chunks: 1 })).not.toThrow();
    expect(() => s.markSkipped('x')).not.toThrow();
    expect(() => s.end()).not.toThrow();
  });

  it('current() returns the active root session inside an update', () => {
    profiler.beginUpdate();
    const s = profiler.current();
    s.setMetadata({ points: 100 });
    profiler.endUpdate();

    expect(profiler.getTimings().metadata?.points).toBe(100);
  });

  it('begin() returns NoOp when no update is active', () => {
    const s = profiler.begin('Foo');
    s.setMetadata({ chunks: 5 });
    s.end();
    expect(profiler.getTimings().children).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------
// UpdateProfiler — time() helpers
// ---------------------------------------------------------------------

describe('UpdateProfiler — time() helpers', () => {
  let profiler: UpdateProfiler;

  beforeEach(() => {
    profiler = new UpdateProfiler();
  });

  it('time() runs a sync function and records its entry', () => {
    profiler.beginUpdate();
    const out = profiler.time('SyncOp', () => 42);
    profiler.endUpdate();
    expect(out).toBe(42);

    const op = findChild(profiler.getTimings(), 'SyncOp');
    expect(op).toBeDefined();
    expect(op!.count).toBe(1);
  });

  it('time() with an async function awaits and records its entry with non-zero duration', async () => {
    profiler.beginUpdate();
    const out = await profiler.time('AsyncOp', async () => {
      await Promise.resolve();
      return 'done';
    });
    profiler.endUpdate();
    expect(out).toBe('done');

    const entry = findChild(profiler.getTimings(), 'AsyncOp');
    expect(entry).toBeDefined();
    expect(entry!.count).toBe(1);
    expect(Number.isFinite(entry!.lastMs)).toBe(true);
  });

  it('time() ends the session and records the entry even when the function throws synchronously', () => {
    profiler.beginUpdate();
    expect(() =>
      profiler.time('Boom', () => {
        throw new Error('nope');
      })
    ).toThrow('nope');
    profiler.endUpdate();

    const entry = findChild(profiler.getTimings(), 'Boom');
    expect(entry).toBeDefined();
    expect(entry!.count).toBe(1);
  });

  it('time() ends the session and records the entry even when an async function rejects', async () => {
    profiler.beginUpdate();
    await expect(
      profiler.time('AsyncBoom', async () => {
        throw new Error('async-nope');
      })
    ).rejects.toThrow('async-nope');
    profiler.endUpdate();

    const entry = findChild(profiler.getTimings(), 'AsyncBoom');
    expect(entry).toBeDefined();
    expect(entry!.count).toBe(1);
  });

  it('timeWithMeta() exposes the session for metadata before completion', async () => {
    profiler.beginUpdate();
    await profiler.timeWithMeta('Meta', async (session) => {
      session.setMetadata({ chunks: 11, points: 2200 });
      return 'ok';
    });
    profiler.endUpdate();

    const op = findChild(profiler.getTimings(), 'Meta');
    expect(op?.metadata?.chunks).toBe(11);
    expect(op?.metadata?.points).toBe(2200);
  });

  it('timeTopLevel() supports concurrent operations: each becomes a direct child of root with non-zero count', async () => {
    // profiling.md W4 fix: previously asserted only the set of names.
    // Strengthen by checking each child has count >= 1 (proving each
    // session.end() actually ran) and is a DIRECT child of root, not
    // a grandchild.
    profiler.beginUpdate();
    await Promise.all([
      profiler.timeTopLevel('A', async () => Promise.resolve()),
      profiler.timeTopLevel('B', async () => Promise.resolve()),
      profiler.timeTopLevel('C', async () => Promise.resolve()),
    ]);
    profiler.endUpdate();

    const root = profiler.getTimings();
    const byName: Record<string, { count: number } | undefined> = {};
    for (const c of root.children) byName[c.name] = c;
    expect(Object.keys(byName).sort()).toEqual(['A', 'B', 'C']);
    expect(byName['A']!.count).toBe(1);
    expect(byName['B']!.count).toBe(1);
    expect(byName['C']!.count).toBe(1);
  });

  it('timeTopLevel() returns function result on happy path, propagates errors, AND still records the entry on the error path', async () => {
    // profiling.md W5 fix: previous version asserted only existence of
    // 'Pass' and 'Fail' entries. Strengthen by also pinning the count
    // for both — proves the error path's end() actually ran.
    profiler.beginUpdate();
    const out = await profiler.timeTopLevel('Pass', async () => 'x');
    expect(out).toBe('x');

    await expect(
      profiler.timeTopLevel('Fail', async () => {
        throw new Error('top-fail');
      })
    ).rejects.toThrow('top-fail');
    profiler.endUpdate();

    const pass = findChild(profiler.getTimings(), 'Pass');
    const fail = findChild(profiler.getTimings(), 'Fail');
    expect(pass).toBeDefined();
    expect(fail).toBeDefined();
    expect(pass!.count).toBeGreaterThanOrEqual(1);
    expect(fail!.count).toBeGreaterThanOrEqual(1);
  });

  it('beginTopLevel() returns a NoOp session whose methods are all idempotent no-ops when no update is active', () => {
    // profiling.md W6 fix: previous version only asserted .not.toThrow() on
    // end() + children.length === 0. The contract is broader: every
    // method on the returned NoOp must be safe to call.
    const s = profiler.beginTopLevel('NoOp');
    expect(() => s.end()).not.toThrow();
    expect(() => s.setMetadata({ chunks: 1 })).not.toThrow();
    expect(() => s.markSkipped('x')).not.toThrow();
    // Re-end should also be safe (NoOp is idempotent).
    expect(() => s.end()).not.toThrow();
    expect(profiler.getTimings().children).toHaveLength(0);
  });

  it('skip() records a 0-duration entry with skip metadata', () => {
    profiler.beginUpdate();
    profiler.skip('Skipped', 'extend_to_all');
    profiler.endUpdate();

    const op = findChild(profiler.getTimings(), 'Skipped');
    expect(op?.lastMs).toBe(0);
    expect(op?.metadata?.skipped).toBe(true);
    expect(op?.metadata?.skipReason).toBe('extend_to_all');
  });

  it('skipped entries report lastMs===0 and avgMs===0 regardless of begin→markSkipped delay', () => {
    // CRIT-1e: previously end() computed duration + applied the EMA BEFORE
    // markSkipped() zeroed lastMs/avgMs — so a `skip()` call that had any
    // real wall-clock delay between begin and markSkipped would leak that
    // overhead into the persistent state. Now end() must bypass the EMA
    // when markSkipped set the skip flag.
    //
    // We drive performance.now() via a controlled clock so we can guarantee
    // a 100ms gap between begin and markSkipped, then assert lastMs===0
    // and avgMs===0 on the persistent entry.
    const clock = controlledClock();
    try {
      const p = new UpdateProfiler();
      p.beginUpdate();

      // Manual sequence so we can inject a delay between begin and
      // markSkipped (skip() does them back-to-back so we can't use it here).
      const session = p.begin('SlowSkip');
      clock.advance(100); // 100ms of "work" before deciding to skip
      session.markSkipped('decided_to_skip_late');
      session.end();

      p.endUpdate();

      const op = findChild(p.getTimings(), 'SlowSkip');
      expect(op).toBeDefined();
      expect(op!.lastMs).toBe(0);
      expect(op!.avgMs).toBe(0);
      expect(op!.overBudget).toBe(false);
      expect(op!.metadata?.skipped).toBe(true);
      expect(op!.metadata?.skipReason).toBe('decided_to_skip_late');
    } finally {
      clock.restore();
    }
  });
});

// ---------------------------------------------------------------------
// UpdateProfiler — timing math (with controlled clock)
// ---------------------------------------------------------------------

describe('UpdateProfiler — timing math', () => {
  let clock: ReturnType<typeof controlledClock>;

  beforeEach(() => {
    clock = controlledClock();
  });

  afterEach(() => {
    clock.restore();
  });

  it('records duration on the persistent root after one update', () => {
    const profiler = new UpdateProfiler();

    // beginUpdate reads now=0 for root.startTime
    profiler.beginUpdate();
    clock.advance(20);
    // endUpdate reads now=20 → root duration = 20
    profiler.endUpdate();

    const root = profiler.getTimings();
    expect(root.lastMs).toBeCloseTo(20, 5);
    expect(root.avgMs).toBeCloseTo(20, 5);
    expect(root.overBudget).toBe(true); // 20 > 16.67
  });

  it('flags entries exceeding the 16.67ms budget', () => {
    const profiler = new UpdateProfiler();

    profiler.beginUpdate();
    profiler.time('Quick', () => clock.advance(5));
    profiler.time('Slow', () => clock.advance(50));
    profiler.endUpdate();

    const quick = findChild(profiler.getTimings(), 'Quick');
    const slow = findChild(profiler.getTimings(), 'Slow');
    expect(quick?.lastMs).toBeCloseTo(5, 5);
    expect(quick?.overBudget).toBe(false);
    expect(slow?.lastMs).toBeCloseTo(50, 5);
    expect(slow?.overBudget).toBe(true);
    expect(slow!.lastMs).toBeGreaterThan(FRAME_BUDGET_MS);
  });

  it('uses EMA (alpha=0.1) for the persistent root after multiple updates', () => {
    const profiler = new UpdateProfiler();

    // Update 1: 100ms
    profiler.beginUpdate();
    clock.advance(100);
    profiler.endUpdate();
    const after1 = profiler.getTimings().avgMs;

    // Update 2: 50ms → EMA = 0.1 * 50 + 0.9 * 100 = 95
    profiler.beginUpdate();
    clock.advance(50);
    profiler.endUpdate();
    const after2 = profiler.getTimings().avgMs;

    // Update 3: 50ms → EMA = 0.1 * 50 + 0.9 * 95 = 90.5
    profiler.beginUpdate();
    clock.advance(50);
    profiler.endUpdate();
    const after3 = profiler.getTimings().avgMs;

    expect(after1).toBeCloseTo(100, 5);
    expect(after2).toBeCloseTo(95, 5);
    expect(after3).toBeCloseTo(90.5, 5);
  });
});

// ---------------------------------------------------------------------
// UpdateProfiler — hierarchy
// ---------------------------------------------------------------------

describe('UpdateProfiler — hierarchy', () => {
  it('root child count equals number of updates (no double-increment in mergeChildEntry)', () => {
    const profiler = new UpdateProfiler();

    profiler.beginUpdate();
    profiler.time('Step', () => undefined);
    profiler.endUpdate();

    profiler.beginUpdate();
    profiler.time('Step', () => undefined);
    profiler.endUpdate();

    const root = profiler.getTimings();
    expect(root.children).toHaveLength(1);
    expect(root.children[0].name).toBe('Step');
    expect(root.children[0].count).toBe(2);
  });

  it('nested time() builds a parent/child hierarchy via currentSessionContext push/pop', () => {
    const profiler = new UpdateProfiler();

    profiler.beginUpdate();
    profiler.time('Outer', () => {
      profiler.time('Inner', () => undefined);
    });
    profiler.endUpdate();

    const root = profiler.getTimings();
    expect(root.children).toHaveLength(1);
    const outer = root.children[0];
    expect(outer.name).toBe('Outer');
    expect(outer.count).toBe(1);
    expect(outer.children).toHaveLength(1);
    const inner = outer.children[0];
    expect(inner.name).toBe('Inner');
    expect(inner.count).toBe(1);
  });

  it('child avgMs converges at EMA rate (alpha=0.1), not double-EMA', () => {
    // After 1st update: avgMs = 100 (seeded on count=1)
    // After 2nd update with duration 50:
    //   correct EMA: 0.1 * 50 + 0.9 * 100 = 95
    //   buggy double-EMA would land near 0.1 * (0.1*50+0.9*100) + 0.9*100
    //     i.e. mid-90s but biased; the strict assertion is that the value
    //     equals 95 exactly under a controlled clock.
    const clock = controlledClock();
    try {
      const profiler = new UpdateProfiler();

      profiler.beginUpdate();
      profiler.time('Step', () => clock.advance(100));
      profiler.endUpdate();
      const after1 = findChild(profiler.getTimings(), 'Step')!.avgMs;

      profiler.beginUpdate();
      profiler.time('Step', () => clock.advance(50));
      profiler.endUpdate();
      const after2 = findChild(profiler.getTimings(), 'Step')!.avgMs;

      expect(after1).toBeCloseTo(100, 5);
      expect(after2).toBeCloseTo(95, 5);
    } finally {
      clock.restore();
    }
  });
});

// ---------------------------------------------------------------------
// Listeners
// ---------------------------------------------------------------------

describe('UpdateProfiler — listeners', () => {
  it('notifies listeners after each completed update', () => {
    const profiler = new UpdateProfiler();
    const listener = vi.fn();
    profiler.addListener(listener);

    profiler.beginUpdate();
    profiler.endUpdate();
    expect(listener).toHaveBeenCalledTimes(1);

    profiler.beginUpdate();
    profiler.endUpdate();
    expect(listener).toHaveBeenCalledTimes(2);
  });

  it('notifies listeners on reset()', () => {
    const profiler = new UpdateProfiler();
    const listener = vi.fn();
    profiler.addListener(listener);

    profiler.reset();
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it('removeListener() stops further notifications', () => {
    const profiler = new UpdateProfiler();
    const listener = vi.fn();
    profiler.addListener(listener);
    profiler.removeListener(listener);

    profiler.beginUpdate();
    profiler.endUpdate();
    expect(listener).not.toHaveBeenCalled();
  });

  it('a throwing listener does not break the profiler', () => {
    const profiler = new UpdateProfiler();
    const ok = vi.fn();
    const bad = vi.fn(() => {
      throw new Error('listener-explode');
    });
    profiler.addListener(bad);
    profiler.addListener(ok);

    expect(() => {
      profiler.beginUpdate();
      profiler.endUpdate();
    }).not.toThrow();
    expect(ok).toHaveBeenCalledTimes(1);
    expect(bad).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------
// Reset & re-entry
// ---------------------------------------------------------------------

describe('UpdateProfiler — reset', () => {
  it('clears accumulated children and counts', () => {
    const profiler = new UpdateProfiler();
    profiler.beginUpdate();
    profiler.time('A', () => undefined);
    profiler.endUpdate();

    profiler.reset();

    const root = profiler.getTimings();
    expect(root.count).toBe(0);
    expect(root.children).toHaveLength(0);
    expect(root.lastMs).toBe(0);
  });

  it('reset() mid-update clears activeSession; subsequent endUpdate() is a no-op', () => {
    // CRIT-1d: reset() must clear activeSession / currentSessionContext /
    // activeSessions in addition to rebuilding rootEntry. Otherwise the
    // dangling RootSession would later try to merge into a tree it no
    // longer owns, polluting fresh root counters.
    const profiler = new UpdateProfiler();

    profiler.beginUpdate();
    expect(profiler.isActive()).toBe(true);

    // reset() mid-update — must wipe active-session state.
    profiler.reset();
    expect(profiler.isActive()).toBe(false);
    // current() must return a NoOp session whose methods are safe to call
    // (proves currentSessionContext was cleared).
    expect(() => profiler.current().setMetadata({ chunks: 7 })).not.toThrow();

    // Calling endUpdate() after reset() must not crash and must not
    // increment the freshly-zeroed root counters.
    expect(() => profiler.endUpdate()).not.toThrow();

    const root = profiler.getTimings();
    expect(root.count).toBe(0);
    expect(root.children).toHaveLength(0);
    expect(root.lastMs).toBe(0);
  });

  // [R11/A-G4 — OOS fixed in this commit] When reset() lands while a
  // timeTopLevel('parallel-load', fn) session is in flight, fn's eventual
  // session.end() must NOT pollute the freshly-rebuilt root tree. The
  // generation-counter gate in SessionImpl.end() drops the merge silently
  // when the profiler's generation has advanced past the session's
  // captured generation.
  it('reset() during an in-flight timeTopLevel() session clears activeSessions and isolates the new root', async () => {
    const profiler = new UpdateProfiler();
    profiler.beginUpdate();

    let resolveInner!: () => void;
    const innerWait = new Promise<void>((r) => {
      resolveInner = r;
    });

    const inFlight = profiler.timeTopLevel('parallel-load', async () => {
      await innerWait;
      return 42;
    });

    profiler.reset();
    expect(profiler.isActive()).toBe(false);

    // Let the abandoned session resolve. Its session.end() must run
    // (no exception) but its merge must NOT touch the new root tree.
    resolveInner();
    await expect(inFlight).resolves.toBe(42);

    const root = profiler.getTimings();
    expect(root.count).toBe(0);
    expect(root.children).toHaveLength(0);
  });
});

describe('UpdateProfiler — beginUpdate when previous unfinished', () => {
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('logs a warning if a previous session was not ended', () => {
    const profiler = new UpdateProfiler();
    profiler.beginUpdate();
    profiler.beginUpdate();
    expect(warnSpy).toHaveBeenCalled();
    profiler.endUpdate();
  });
});

describe('UpdateProfiler — same-update summing (per-update sequence)', () => {
  let clock: ReturnType<typeof controlledClock>;

  beforeEach(() => {
    clock = controlledClock();
  });

  afterEach(() => {
    clock.restore();
  });

  it('sums lastMs across same-name sessions within one update, count stays 1', () => {
    const profiler = new UpdateProfiler();
    profiler.beginUpdate();

    // Two 'Load Arrays' sessions in the same update (progressive loader
    // loading two LOD levels).
    const s1 = profiler.beginTopLevel('Load Arrays');
    clock.advance(30);
    s1.end();
    const s2 = profiler.beginTopLevel('Load Arrays');
    clock.advance(50);
    s2.end();

    profiler.endUpdate();

    const child = findChild(profiler.getTimings(), 'Load Arrays')!;
    expect(child.lastMs).toBe(80); // 30 + 50, NOT overwritten to 50
    expect(child.count).toBe(1); // count = #updates the op ran in
    expect(child.avgMs).toBe(80); // first update seeds the EMA with the sum
  });

  it('sums numeric metadata on same-update merges', () => {
    const profiler = new UpdateProfiler();
    profiler.beginUpdate();

    const s1 = profiler.beginTopLevel('Load Arrays');
    s1.setMetadata({ chunks: 3, cacheHits: 2, cacheMisses: 1 });
    s1.end();
    const s2 = profiler.beginTopLevel('Load Arrays');
    s2.setMetadata({ chunks: 5, cacheHits: 0, cacheMisses: 5 });
    s2.end();

    profiler.endUpdate();

    const child = findChild(profiler.getTimings(), 'Load Arrays')!;
    expect(child.metadata?.chunks).toBe(8);
    expect(child.metadata?.cacheHits).toBe(2);
    expect(child.metadata?.cacheMisses).toBe(6);
  });

  it('applies ONE EMA sample per update against the pre-update base', () => {
    const profiler = new UpdateProfiler();

    // Update 1: two sessions summing to 100 → avg seeds at 100.
    profiler.beginUpdate();
    const a1 = profiler.beginTopLevel('Op');
    clock.advance(60);
    a1.end();
    const a2 = profiler.beginTopLevel('Op');
    clock.advance(40);
    a2.end();
    profiler.endUpdate();

    // Update 2: two sessions summing to 50 → avg = 0.1*50 + 0.9*100 = 95,
    // NOT alpha applied twice (which would give 0.1*30 + 0.9*(0.1*20+0.9*100)).
    profiler.beginUpdate();
    const b1 = profiler.beginTopLevel('Op');
    clock.advance(20);
    b1.end();
    const b2 = profiler.beginTopLevel('Op');
    clock.advance(30);
    b2.end();
    profiler.endUpdate();

    const child = findChild(profiler.getTimings(), 'Op')!;
    expect(child.lastMs).toBe(50);
    expect(child.count).toBe(2);
    expect(child.avgMs).toBeCloseTo(0.1 * 50 + 0.9 * 100, 6);
  });

  it('recomputes overBudget from the summed lastMs', () => {
    const profiler = new UpdateProfiler();
    profiler.beginUpdate();

    // Two 10ms sessions: each under budget, sum is over.
    const s1 = profiler.beginTopLevel('Op');
    clock.advance(10);
    s1.end();
    const s2 = profiler.beginTopLevel('Op');
    clock.advance(10);
    s2.end();

    profiler.endUpdate();

    const child = findChild(profiler.getTimings(), 'Op')!;
    expect(child.lastMs).toBe(20);
    expect(child.overBudget).toBe(true);
  });
});

describe('UpdateProfiler — staleness', () => {
  let clock: ReturnType<typeof controlledClock>;

  beforeEach(() => {
    clock = controlledClock();
  });

  afterEach(() => {
    clock.restore();
  });

  it('marks entries that did not run in the latest update as stale', () => {
    const profiler = new UpdateProfiler();

    // Update 1: 'Load Arrays' runs (94ms).
    profiler.beginUpdate();
    const load = profiler.beginTopLevel('Load Arrays');
    clock.advance(94);
    load.end();
    profiler.endUpdate();

    // Update 2: only 'Project to 3D' runs — Load Arrays did NOT run.
    profiler.beginUpdate();
    const proj = profiler.beginTopLevel('Project to 3D');
    clock.advance(3);
    proj.end();
    profiler.endUpdate();

    const root = profiler.getTimings();
    const loadEntry = findChild(root, 'Load Arrays')!;
    const projEntry = findChild(root, 'Project to 3D')!;
    expect(loadEntry.stale).toBe(true);
    expect(loadEntry.lastMs).toBe(94); // last-known value preserved for display
    expect(projEntry.stale).toBeFalsy();
    expect(root.stale).toBeFalsy();
  });

  it('clears the stale flag when the op runs again', () => {
    const profiler = new UpdateProfiler();

    profiler.beginUpdate();
    profiler.beginTopLevel('Op').end();
    profiler.endUpdate();

    profiler.beginUpdate();
    profiler.beginTopLevel('Other').end();
    profiler.endUpdate();
    expect(findChild(profiler.getTimings(), 'Op')!.stale).toBe(true);

    profiler.beginUpdate();
    profiler.beginTopLevel('Op').end();
    profiler.endUpdate();
    expect(findChild(profiler.getTimings(), 'Op')!.stale).toBe(false);
  });

  it('hasOverBudget ignores stale children', () => {
    const profiler = new UpdateProfiler();

    profiler.beginUpdate();
    const slow = profiler.beginTopLevel('Slow');
    clock.advance(100);
    slow.end();
    profiler.endUpdate();

    profiler.beginUpdate();
    const fast = profiler.beginTopLevel('Fast');
    clock.advance(1);
    fast.end();
    profiler.endUpdate();

    // The stale 100ms 'Slow' child must not paint the fresh tree red;
    // root itself was 1ms in the latest update.
    const root = profiler.getTimings();
    expect(findChild(root, 'Slow')!.overBudget).toBe(true);
    expect(findChild(root, 'Slow')!.stale).toBe(true);
    expect(hasOverBudget(findChild(root, 'Slow')!)).toBe(false);
  });
});

describe('UpdateProfiler — refinement passes (beginPass)', () => {
  let clock: ReturnType<typeof controlledClock>;

  beforeEach(() => {
    clock = controlledClock();
  });

  afterEach(() => {
    clock.restore();
  });

  it('records pass sessions under the LOD Refinement root, not Total Update', () => {
    const profiler = new UpdateProfiler();

    const pass = profiler.beginPass();
    const node = pass.begin('GSplats (/g)');
    clock.advance(40);
    node.end();
    pass.end();

    const refinement = profiler.getRefinementTimings();
    expect(refinement.count).toBe(1);
    expect(findChild(refinement, 'GSplats (/g)')!.lastMs).toBe(40);
    // Total Update tree untouched.
    expect(profiler.getTimings().count).toBe(0);
    expect(findChild(profiler.getTimings(), 'GSplats (/g)')).toBeUndefined();
  });

  it('merges pass children into the refinement tree even when a same-named parent exists under Total Update', () => {
    const profiler = new UpdateProfiler();

    // Total Update tree gets a 'GSplats (/g)' → 'Load Arrays' chain.
    profiler.beginUpdate();
    const top = profiler.beginTopLevel('GSplats (/g)');
    const topLoad = top.begin('Load Arrays');
    clock.advance(10);
    topLoad.end();
    top.end();
    profiler.endUpdate();

    // Refinement pass with the SAME names.
    const pass = profiler.beginPass();
    const node = pass.begin('GSplats (/g)');
    const load = node.begin('Load Arrays');
    clock.advance(70);
    load.end();
    node.end();
    pass.end();

    // Each tree holds its own 'Load Arrays' value.
    expect(findChild(profiler.getTimings(), 'Load Arrays')!.lastMs).toBe(10);
    expect(findChild(profiler.getRefinementTimings(), 'Load Arrays')!.lastMs).toBe(70);
  });

  it('a pass ending mid-update does NOT disable the active update session', () => {
    const profiler = new UpdateProfiler();

    profiler.beginUpdate();
    const pass = profiler.beginPass();
    pass.end(); // must NOT null the active update session

    expect(profiler.isActive()).toBe(true);
    const top = profiler.beginTopLevel('Points (/p)');
    clock.advance(5);
    top.end();
    profiler.endUpdate();

    expect(findChild(profiler.getTimings(), 'Points (/p)')!.lastMs).toBe(5);
  });

  it('reset clears the refinement tree and pass counter', () => {
    const profiler = new UpdateProfiler();
    const pass = profiler.beginPass();
    pass.begin('GSplats (/g)').end();
    pass.end();
    expect(profiler.getRefinementTimings().count).toBe(1);

    profiler.reset();
    expect(profiler.getRefinementTimings().count).toBe(0);
    expect(profiler.getRefinementTimings().children).toEqual([]);
  });
});

describe('UpdateProfiler — depth-sort passes (beginDepthSortPass)', () => {
  let clock: ReturnType<typeof controlledClock>;

  beforeEach(() => {
    clock = controlledClock();
  });

  afterEach(() => {
    clock.restore();
  });

  it('records sort round-trips under the Depth Sort root with metadata, touching no other tree', () => {
    const profiler = new UpdateProfiler();

    const sort = profiler.beginDepthSortPass();
    clock.advance(12);
    sort.setMetadata({ splats: 1_000_000, info: '4.0 MB up' });
    sort.end();

    const depthSort = profiler.getDepthSortTimings();
    expect(depthSort.count).toBe(1);
    expect(depthSort.lastMs).toBe(12);
    expect(depthSort.metadata).toMatchObject({ splats: 1_000_000, info: '4.0 MB up' });
    expect(profiler.getTimings().count).toBe(0);
    expect(profiler.getRefinementTimings().count).toBe(0);
  });

  it('a sort pass ending mid-update does NOT disable the active update session', () => {
    const profiler = new UpdateProfiler();

    profiler.beginUpdate();
    const sort = profiler.beginDepthSortPass();
    sort.end(); // must NOT null the active update session

    expect(profiler.isActive()).toBe(true);
    const top = profiler.beginTopLevel('GSplats (/g)');
    clock.advance(5);
    top.end();
    profiler.endUpdate();

    expect(findChild(profiler.getTimings(), 'GSplats (/g)')!.lastMs).toBe(5);
  });

  it('successive sorts roll the count and lastMs forward', () => {
    const profiler = new UpdateProfiler();

    const first = profiler.beginDepthSortPass();
    clock.advance(20);
    first.end();
    const second = profiler.beginDepthSortPass();
    clock.advance(8);
    second.end();

    const depthSort = profiler.getDepthSortTimings();
    expect(depthSort.count).toBe(2);
    expect(depthSort.lastMs).toBe(8);
  });

  it('reset clears the depth-sort tree and sort counter', () => {
    const profiler = new UpdateProfiler();
    profiler.beginDepthSortPass().end();
    expect(profiler.getDepthSortTimings().count).toBe(1);

    profiler.reset();
    expect(profiler.getDepthSortTimings().count).toBe(0);
  });
});

// ---------------------------------------------------------------------
// Depth-sort completion stream (issue #711)
// ---------------------------------------------------------------------

describe('UpdateProfiler — depth-sort completion stream', () => {
  const DEPTH_SORT_COMPLETION_CAP = 512;

  type CompletionArg = Parameters<UpdateProfiler['recordDepthSortCompletion']>[0];
  const completion = (lastMs: number): CompletionArg => ({
    lastMs,
    kernelMs: null,
    boundaryMs: null,
    queueMs: null,
    splats: null,
  });

  it('N in, N out: total === N and events carry monotonic seq 1..N', () => {
    const profiler = new UpdateProfiler();
    const N = 20;
    for (let i = 0; i < N; i++) profiler.recordDepthSortCompletion(completion(i + 1));

    const { total, events } = profiler.getDepthSortCompletions();
    expect(total).toBe(N);
    expect(events).toHaveLength(N);
    expect(events.map((e) => e.seq)).toEqual(Array.from({ length: N }, (_, i) => i + 1));
    // seq strictly increases and starts at 1.
    for (let i = 1; i < events.length; i++) {
      expect(events[i].seq).toBe(events[i - 1].seq + 1);
    }
  });

  it('is a plain counter: no completion is ever dropped (unlike the seq-merged root)', () => {
    // The 'Depth Sort' profiler root drops a merge whose seq < the stored
    // lastSeq (a late/out-of-order resolve). This stream is independent of
    // that policy: every recorded completion counts, in any interleaving.
    const profiler = new UpdateProfiler();
    // Interleave depth-sort passes (which bump sortSeq) with completions to
    // simulate out-of-order resolves — the completion total must keep climbing.
    profiler.beginDepthSortPass().end();
    profiler.recordDepthSortCompletion(completion(5));
    profiler.beginDepthSortPass().end();
    profiler.recordDepthSortCompletion(completion(3));
    profiler.recordDepthSortCompletion(completion(9));

    const { total, events } = profiler.getDepthSortCompletions();
    expect(total).toBe(3);
    expect(events.map((e) => e.lastMs)).toEqual([5, 3, 9]);
    expect(events.map((e) => e.seq)).toEqual([1, 2, 3]);
  });

  it('bounded ring caps at the cap while total keeps counting beyond it', () => {
    const profiler = new UpdateProfiler();
    const K = 7;
    const n = DEPTH_SORT_COMPLETION_CAP + K;
    for (let i = 0; i < n; i++) profiler.recordDepthSortCompletion(completion(i + 1));

    const { total, events } = profiler.getDepthSortCompletions();
    expect(total).toBe(n);
    expect(events).toHaveLength(DEPTH_SORT_COMPLETION_CAP);
    // The retained events are the most recent ones by seq: seq K+1 .. n.
    expect(events[0].seq).toBe(K + 1);
    expect(events[events.length - 1].seq).toBe(n);
  });

  it('getDepthSortCompletions returns a COPY (mutating it does not corrupt state)', () => {
    const profiler = new UpdateProfiler();
    profiler.recordDepthSortCompletion(completion(1));
    const first = profiler.getDepthSortCompletions();
    first.events.push({
      seq: 999,
      lastMs: 0,
      kernelMs: null,
      boundaryMs: null,
      queueMs: null,
      splats: null,
    });

    const second = profiler.getDepthSortCompletions();
    expect(second.events).toHaveLength(1);
    expect(second.total).toBe(1);
  });

  it('carries the stage fields through unchanged', () => {
    const profiler = new UpdateProfiler();
    profiler.recordDepthSortCompletion({
      lastMs: 12,
      kernelMs: 4,
      boundaryMs: 2,
      queueMs: 6,
      splats: 1_000_000,
    });
    const { events } = profiler.getDepthSortCompletions();
    expect(events[0]).toMatchObject({
      seq: 1,
      lastMs: 12,
      kernelMs: 4,
      boundaryMs: 2,
      queueMs: 6,
      splats: 1_000_000,
    });
  });

  it('reset clears the completion total and events', () => {
    const profiler = new UpdateProfiler();
    profiler.recordDepthSortCompletion(completion(1));
    profiler.recordDepthSortCompletion(completion(2));
    expect(profiler.getDepthSortCompletions().total).toBe(2);

    profiler.reset();
    const { total, events } = profiler.getDepthSortCompletions();
    expect(total).toBe(0);
    expect(events).toHaveLength(0);
  });

  it('is independent of the seq-merged depth-sort root (both advance separately)', () => {
    const profiler = new UpdateProfiler();
    profiler.beginDepthSortPass().end();
    profiler.beginDepthSortPass().end();
    // Two passes recorded on the root, but no completions on the stream yet.
    expect(profiler.getDepthSortTimings().count).toBe(2);
    expect(profiler.getDepthSortCompletions().total).toBe(0);

    profiler.recordDepthSortCompletion(completion(1));
    expect(profiler.getDepthSortTimings().count).toBe(2);
    expect(profiler.getDepthSortCompletions().total).toBe(1);
  });
});
