import { describe, expect, it, vi } from 'vitest';
import {
  MAX_LOCAL_WORKERS,
  chooseLocalWorkers,
  e2eWorkerPlan,
  formatE2EParallelismStamp,
  resolveE2EWorkers,
} from '../../../../tools/e2e-workers';

// `vi.spyOn(os, …)` cannot redefine an ESM namespace export, so the machine reads are stubbed by
// mocking the module itself. Each field defaults to `null` = pass through to the real `os`, so a
// test opts into exactly the one reading it needs. `loadavg` also accepts a FUNCTION, for the one
// test that needs the reading to move between two calls.
type LoadAverage = [number, number, number];
const osStub = vi.hoisted(() => ({
  loadavg: null as LoadAverage | (() => LoadAverage) | null,
  platform: null as NodeJS.Platform | null,
  parallelism: null as number | null,
}));

vi.mock('node:os', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:os')>();
  return {
    ...actual,
    loadavg: () => {
      const stub = osStub.loadavg;
      if (typeof stub === 'function') return stub();
      return stub ?? actual.loadavg();
    },
    platform: () => osStub.platform ?? actual.platform(),
    availableParallelism: () => osStub.parallelism ?? actual.availableParallelism(),
  };
});

function resetOsStub(): void {
  osStub.loadavg = null;
  osStub.platform = null;
  osStub.parallelism = null;
}

// The CPU count is pinned in every test whose expectation depends on it: the override clamp is
// `[1, cpus]`, and `os.availableParallelism()` honours the CPU-affinity mask, so a real reading
// would make these assertions fail under `taskset`, in a cpuset container, or on a 2-vCPU runner.
const PINNED_CPUS = 16;

describe('E2E local worker sizing', () => {
  it('scales the ceiling by the free fraction of the box', () => {
    // An idle 16-core box is fully free and runs at the dataset-server ceiling.
    expect(chooseLocalWorkers({ cpus: 16, load1: 0 })).toEqual({
      workers: MAX_LOCAL_WORKERS,
      reason: 'capacity',
    });
    // 75% free -> 3 of 4; 50% -> 2; 25% -> 1 (the measured-green configuration).
    expect(chooseLocalWorkers({ cpus: 16, load1: 4 }).workers).toBe(3);
    expect(chooseLocalWorkers({ cpus: 16, load1: 8 }).workers).toBe(2);
    expect(chooseLocalWorkers({ cpus: 16, load1: 12 })).toEqual({
      workers: 1,
      reason: 'capacity',
    });
  });

  it('puts the band edges at 7/8, 5/8 and 3/8 of the box free', () => {
    // Rounding to nearest makes each band a fraction of the box, so these edges hold at any
    // size; on 16 cores they land at loads of 2, 6 and 10.
    expect(chooseLocalWorkers({ cpus: 16, load1: 2 }).workers).toBe(4);
    expect(chooseLocalWorkers({ cpus: 16, load1: 2.01 }).workers).toBe(3);
    expect(chooseLocalWorkers({ cpus: 16, load1: 6 }).workers).toBe(3);
    expect(chooseLocalWorkers({ cpus: 16, load1: 6.01 }).workers).toBe(2);
    expect(chooseLocalWorkers({ cpus: 16, load1: 10 }).workers).toBe(2);
    expect(chooseLocalWorkers({ cpus: 16, load1: 10.01 }).workers).toBe(1);
    // The same fractions on a 4-core box: 0.5 free is the 2-worker band.
    expect(chooseLocalWorkers({ cpus: 4, load1: 0.5 }).workers).toBe(4);
    expect(chooseLocalWorkers({ cpus: 4, load1: 1 }).workers).toBe(3);
    expect(chooseLocalWorkers({ cpus: 4, load1: 2 }).workers).toBe(2);
    expect(chooseLocalWorkers({ cpus: 4, load1: 3 }).workers).toBe(1);
  });

  it('never drops below one worker, however oversubscribed the box is', () => {
    // Load above the core count makes the free fraction 0; the clamp, not the floor, decides.
    expect(chooseLocalWorkers({ cpus: 16, load1: 22.2 })).toEqual({
      workers: 1,
      reason: 'capacity',
    });
    expect(chooseLocalWorkers({ cpus: 16, load1: 1e6 }).workers).toBe(1);
  });

  it('never lowers the count on an idle box, however small', () => {
    // The scaling is scale-FREE: a small idle box keeps the historical ceiling, because nothing
    // has measured a 2- or 4-core machine as needing less.
    expect(chooseLocalWorkers({ cpus: 4, load1: 0 })).toEqual({
      workers: MAX_LOCAL_WORKERS,
      reason: 'capacity',
    });
    expect(chooseLocalWorkers({ cpus: 2, load1: 0 }).workers).toBe(MAX_LOCAL_WORKERS);
    expect(chooseLocalWorkers({ cpus: 1, load1: 0 }).workers).toBe(MAX_LOCAL_WORKERS);
    // …and a small box under load still backs off proportionally.
    expect(chooseLocalWorkers({ cpus: 4, load1: 2 }).workers).toBe(2);
  });

  it('rounds the scaled count to NEAREST, not down', () => {
    // 4 * 14.4/16 = 3.6 — nearest 4, floor 3. Flooring made the top count reachable only at a
    // load average of exactly 0, so an ordinary developer box at load 1.6 lost a worker.
    expect(chooseLocalWorkers({ cpus: 16, load1: 1.6 }).workers).toBe(4);
    // 4 * 9.6/16 = 2.4 — nearest 2, and a half rounds UP: 4 * 10/16 = 2.5 -> 3.
    expect(chooseLocalWorkers({ cpus: 16, load1: 6.4 }).workers).toBe(2);
    expect(chooseLocalWorkers({ cpus: 16, load1: 10 }).workers).toBe(2);
  });

  it('falls back to the ceiling when capacity is unknown rather than guessing low', () => {
    // Missing, NaN, infinite, or negative load — and an unusable CPU count — all mean
    // "no capacity signal", which is the historical hardcoded-4 behaviour.
    for (const load1 of [undefined, Number.NaN, Number.POSITIVE_INFINITY, -1]) {
      expect(chooseLocalWorkers({ cpus: 16, load1 })).toEqual({
        workers: MAX_LOCAL_WORKERS,
        reason: 'unknown-capacity',
      });
    }
    for (const cpus of [undefined, 0, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(chooseLocalWorkers({ cpus, load1: 12 })).toEqual({
        workers: MAX_LOCAL_WORKERS,
        reason: 'unknown-capacity',
      });
    }
    // A non-numeric load arriving from an untyped caller must not poison the arithmetic.
    const untyped = chooseLocalWorkers({ cpus: 16, load1: 'busy' as unknown as number });
    expect(untyped).toEqual({ workers: MAX_LOCAL_WORKERS, reason: 'unknown-capacity' });
  });

  it('honours an explicit integer override over any measurement', () => {
    // An override beats a loaded box in both directions, and may exceed the ceiling: it encodes
    // a measurement someone took on their own machine.
    expect(chooseLocalWorkers({ cpus: 16, load1: 22.2, override: 4 })).toEqual({
      workers: 4,
      reason: 'override',
    });
    expect(chooseLocalWorkers({ cpus: 16, load1: 0, override: 1 }).workers).toBe(1);
    expect(chooseLocalWorkers({ cpus: 16, load1: 0, override: 8 }).workers).toBe(8);
    // Playwright rejects 0 workers, so a non-positive pin clamps up to a serial run.
    expect(chooseLocalWorkers({ cpus: 16, load1: 0, override: 0 })).toEqual({
      workers: 1,
      reason: 'override',
    });
    expect(chooseLocalWorkers({ cpus: 16, load1: 0, override: -3 }).workers).toBe(1);
  });

  it('clamps an override to the logical CPU count at the top', () => {
    // `LUXAR_E2E_WORKERS=40` is a plausible typo for 4 and must not spawn 40 Chromiums; `1e21`
    // is an integer that would have Playwright pre-allocate 1e21 worker slots.
    expect(chooseLocalWorkers({ cpus: 16, load1: 0, override: 40 })).toEqual({
      workers: 16,
      reason: 'override',
    });
    expect(chooseLocalWorkers({ cpus: 16, load1: 0, override: 1e21 }).workers).toBe(16);
    expect(chooseLocalWorkers({ cpus: 2, load1: 0, override: 8 }).workers).toBe(2);
    // With no usable CPU count there is no defensible ceiling, so only the lower clamp applies.
    expect(chooseLocalWorkers({ cpus: undefined, load1: 0, override: 40 }).workers).toBe(40);
  });

  it('ignores a fractional or unparseable override instead of rounding it', () => {
    for (const override of [2.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(chooseLocalWorkers({ cpus: 16, load1: 8, override })).toEqual({
        workers: 2,
        reason: 'capacity',
      });
    }
  });

  it('reads CI, the env override, and this machine through the resolver', () => {
    resetOsStub();
    // Pinned, not read: the override clamp is `[1, cpus]`, so a real reading would make the
    // `' 3 '` expectation below depend on the affinity mask of whatever machine runs the suite.
    osStub.parallelism = PINNED_CPUS;
    const cpus = PINNED_CPUS;

    // CI keeps the historical single worker regardless of the box or the env override.
    expect(resolveE2EWorkers({ CI: '1' })).toEqual({
      workers: 1,
      reason: 'ci',
      cpus,
      load1: expect.any(Number),
    });
    expect(resolveE2EWorkers({ CI: 'true', LUXAR_E2E_WORKERS: '4' }).workers).toBe(1);

    // LUXAR_E2E_WORKERS is honoured locally; whitespace is tolerated, junk is ignored.
    expect(resolveE2EWorkers({ LUXAR_E2E_WORKERS: ' 3 ' })).toMatchObject({
      workers: 3,
      reason: 'override',
    });
    for (const raw of ['', '   ', 'two', '2.5']) {
      expect(resolveE2EWorkers({ LUXAR_E2E_WORKERS: raw }).reason).not.toBe('override');
    }

    // Without an override the decision is whatever this machine's own load implies.
    const measured = resolveE2EWorkers({});
    expect(measured.cpus).toBe(cpus);
    expect(measured.workers).toBe(
      chooseLocalWorkers({ cpus: measured.cpus, load1: measured.load1 }).workers
    );
    expect(measured.workers).toBeGreaterThanOrEqual(1);
    expect(measured.workers).toBeLessThanOrEqual(MAX_LOCAL_WORKERS);
    resetOsStub();
  });

  it('reads the ONE-minute load average, not the 5- or 15-minute one', () => {
    resetOsStub();
    osStub.platform = 'linux';
    osStub.parallelism = 16;
    // Three deliberately different figures: only index 0 gives 1 worker here.
    osStub.loadavg = [12, 0, 0];
    expect(resolveE2EWorkers({}).workers).toBe(1);
    // Reversed, only index 0 gives the ceiling.
    osStub.loadavg = [0, 12, 24];
    expect(resolveE2EWorkers({}).workers).toBe(MAX_LOCAL_WORKERS);
    resetOsStub();
  });

  it('treats a platform with no load average as no signal, not as idle', () => {
    resetOsStub();
    osStub.parallelism = 16;
    // Windows' `os.loadavg()` is a hardcoded [0, 0, 0]. On the capacity path that would read as
    // a fully idle box by accident; it must reach the explicit no-signal ceiling instead.
    osStub.platform = 'win32';
    osStub.loadavg = [0, 0, 0];
    const windows = resolveE2EWorkers({});
    expect(windows).toMatchObject({ workers: MAX_LOCAL_WORKERS, reason: 'unknown-capacity' });
    expect(Number.isNaN(windows.load1)).toBe(true);

    // The same numbers on Linux are a genuinely idle box, and take the capacity path.
    osStub.platform = 'linux';
    expect(resolveE2EWorkers({})).toMatchObject({
      workers: MAX_LOCAL_WORKERS,
      reason: 'capacity',
      load1: 0,
    });
    resetOsStub();
  });

  it('defaults to process.env, which is how the Playwright config calls it', () => {
    resetOsStub();
    // Pinned so the `[1, cpus]` override clamp cannot lower the expectation on a small or
    // affinity-restricted machine.
    osStub.parallelism = PINNED_CPUS;
    // `e2eWorkerPlan()` — what `playwright.config.ts` and the E2E global setup use — calls
    // `resolveE2EWorkers()` with no argument, so the default parameter is production code.
    // Exercise it rather than always injecting an env.
    const had = Object.prototype.hasOwnProperty.call(process.env, 'LUXAR_E2E_WORKERS');
    const previous = process.env.LUXAR_E2E_WORKERS;
    const hadCI = Object.prototype.hasOwnProperty.call(process.env, 'CI');
    const previousCI = process.env.CI;
    try {
      delete process.env.CI;
      process.env.LUXAR_E2E_WORKERS = '2';
      expect(resolveE2EWorkers()).toMatchObject({ workers: 2, reason: 'override' });
    } finally {
      if (had) process.env.LUXAR_E2E_WORKERS = previous;
      else delete process.env.LUXAR_E2E_WORKERS;
      if (hadCI) process.env.CI = previousCI;
      else delete process.env.CI;
      resetOsStub();
    }
  });

  it('decides once per process, so the config and the stamp cannot disagree', () => {
    resetOsStub();
    osStub.platform = 'linux';
    osStub.parallelism = PINNED_CPUS;
    // A load average that MOVES between reads — exactly what happens between the config module's
    // call and the global setup's, minutes later. Without the memo the two would disagree.
    const readings: LoadAverage[] = [
      [0, 0, 0],
      [22.2, 0, 0],
    ];
    osStub.loadavg = () => readings.shift() ?? [22.2, 0, 0];

    const first = e2eWorkerPlan();
    expect(e2eWorkerPlan()).toEqual(first);
    expect(first.load1).toBe(0);
    // Negative control: an unmemoized read really would have seen a different box by now.
    expect(resolveE2EWorkers({}).load1).toBe(22.2);
    resetOsStub();
  });

  it('stamps the run CEILING and says what was sized when the two differ', () => {
    const plan = { workers: 1, reason: 'capacity' as const, cpus: 16, load1: 22.2 };
    expect(formatE2EParallelismStamp(1, plan)).toBe(
      '[🧵] [E2E] parallelism: max 1 worker — 16 cpus, load1 22.2 (capacity)'
    );
    // `--workers=4` on a box this loaded: the line must not claim the heuristic chose it.
    expect(formatE2EParallelismStamp(4, plan)).toBe(
      '[🧵] [E2E] parallelism: max 4 workers — 16 cpus, load1 22.2 (capacity sized 1, run with 4)'
    );
    // A divergence DOWNWARDS need not come from the command line at all — `--ui`, a connected
    // watch session, and `playwright.perf.config.ts`'s own `workers: 1` all produce it — so the
    // line states both numbers and blames nothing.
    expect(
      formatE2EParallelismStamp(1, { workers: 3, reason: 'capacity', cpus: 16, load1: 1.9 })
    ).toBe(
      '[🧵] [E2E] parallelism: max 1 worker — 16 cpus, load1 1.9 (capacity sized 3, run with 1)'
    );
    // A platform without a load average has nothing to print for it.
    expect(
      formatE2EParallelismStamp(4, {
        workers: 4,
        reason: 'unknown-capacity',
        cpus: 8,
        load1: Number.NaN,
      })
    ).toBe('[🧵] [E2E] parallelism: max 4 workers — 8 cpus, load1 n/a (unknown-capacity)');
  });
});
