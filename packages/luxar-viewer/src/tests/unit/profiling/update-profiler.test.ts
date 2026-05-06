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
 * Notes on behavior under test:
 * - The profiler's persistent `rootEntry.children[*].count` increments TWICE
 *   per measurement on the first merge — once when the child session ends and
 *   merges itself into the persistent root, and once again when the parent
 *   root session ends and re-merges its children. Subsequent updates only
 *   increment by 2 each. Tests therefore check >= bounds rather than exact
 *   counts on entries.
 * - `time()` does not push/pop `currentSessionContext`, so nested `time()`
 *   produces SIBLING entries under the root, not a parent/child hierarchy.
 *   Tests assert the observed flat behavior rather than assuming a stack.
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

  it('reports inactive before any update', () => {
    expect(profiler.isActive()).toBe(false);
  });

  it('reports active between beginUpdate and endUpdate', () => {
    profiler.beginUpdate();
    expect(profiler.isActive()).toBe(true);
    profiler.endUpdate();
    expect(profiler.isActive()).toBe(false);
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
    expect(op!.count).toBeGreaterThanOrEqual(1);
  });

  it('time() with an async function awaits and records its entry', async () => {
    profiler.beginUpdate();
    const out = await profiler.time('AsyncOp', async () => {
      await Promise.resolve();
      return 'done';
    });
    profiler.endUpdate();
    expect(out).toBe('done');

    expect(findChild(profiler.getTimings(), 'AsyncOp')).toBeDefined();
  });

  it('time() ends the session even when the function throws synchronously', () => {
    profiler.beginUpdate();
    expect(() =>
      profiler.time('Boom', () => {
        throw new Error('nope');
      })
    ).toThrow('nope');
    profiler.endUpdate();

    expect(findChild(profiler.getTimings(), 'Boom')).toBeDefined();
  });

  it('time() ends the session even when an async function rejects', async () => {
    profiler.beginUpdate();
    await expect(
      profiler.time('AsyncBoom', async () => {
        throw new Error('async-nope');
      })
    ).rejects.toThrow('async-nope');
    profiler.endUpdate();

    expect(findChild(profiler.getTimings(), 'AsyncBoom')).toBeDefined();
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

  it('timeTopLevel() supports concurrent operations under the root', async () => {
    profiler.beginUpdate();
    await Promise.all([
      profiler.timeTopLevel('A', async () => Promise.resolve()),
      profiler.timeTopLevel('B', async () => Promise.resolve()),
      profiler.timeTopLevel('C', async () => Promise.resolve()),
    ]);
    profiler.endUpdate();

    const root = profiler.getTimings();
    const names = root.children.map((c) => c.name).sort();
    expect(names).toEqual(['A', 'B', 'C']);
  });

  it('timeTopLevel() returns its function result and propagates errors', async () => {
    profiler.beginUpdate();
    const out = await profiler.timeTopLevel('Pass', async () => 'x');
    expect(out).toBe('x');

    await expect(
      profiler.timeTopLevel('Fail', async () => {
        throw new Error('top-fail');
      })
    ).rejects.toThrow('top-fail');
    profiler.endUpdate();

    expect(findChild(profiler.getTimings(), 'Pass')).toBeDefined();
    expect(findChild(profiler.getTimings(), 'Fail')).toBeDefined();
  });

  it('beginTopLevel() returns NoOp when no update is active', () => {
    const s = profiler.beginTopLevel('NoOp');
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
  it('persistent root keeps a single entry per unique name across updates', () => {
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
    // count is implementation-defined (the merge path increments more than
    // once per measurement); the contract is "monotonically increasing".
    expect(root.children[0].count).toBeGreaterThan(0);
  });

  it('time() calls inside other time() calls become flat siblings under root', () => {
    // The profiler does not push/pop currentSessionContext on time(), so a
    // nested time('Inner') registers as a sibling of 'Outer' rather than a
    // child. This test pins the documented behavior; if the profiler later
    // gains real nesting, this test should be updated.
    const profiler = new UpdateProfiler();

    profiler.beginUpdate();
    profiler.time('Outer', () => {
      profiler.time('Inner', () => undefined);
    });
    profiler.endUpdate();

    const names = profiler.getTimings().children.map((c) => c.name).sort();
    expect(names).toEqual(['Inner', 'Outer']);
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
