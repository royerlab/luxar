import * as os from 'node:os';
import { describe, expect, it } from 'vitest';
import {
  CORES_PER_WORKER,
  MAX_LOCAL_WORKERS,
  chooseLocalWorkers,
  resolveE2EWorkers,
} from '../../../../tools/e2e-workers';

describe('E2E local worker sizing', () => {
  it('runs at the ceiling only on a box with the spare cores for it', () => {
    // An idle 16-core box budgets 8 workers and is held at the dataset-server ceiling.
    expect(chooseLocalWorkers({ cpus: 16, load1: 0 })).toEqual({
      workers: MAX_LOCAL_WORKERS,
      reason: 'capacity',
    });
    // 8 spare cores is exactly the ceiling's budget — the last load at which nothing is lost.
    expect(chooseLocalWorkers({ cpus: 16, load1: 8 }).workers).toBe(MAX_LOCAL_WORKERS);
    // The measured failing regime: 4 spare cores buys 2 workers, not 4.
    expect(chooseLocalWorkers({ cpus: 16, load1: 12 })).toEqual({
      workers: 2,
      reason: 'capacity',
    });
  });

  it('never drops below one worker, however oversubscribed the box is', () => {
    // Load above the core count makes `spare` negative; the clamp, not the floor, decides.
    expect(chooseLocalWorkers({ cpus: 16, load1: 24 })).toEqual({
      workers: 1,
      reason: 'capacity',
    });
    expect(chooseLocalWorkers({ cpus: 16, load1: 1e6 }).workers).toBe(1);
  });

  it('sizes a small box by the same core budget', () => {
    // A 2-core laptop gets one worker even when idle: two cores is one worker's budget.
    expect(chooseLocalWorkers({ cpus: 2, load1: 0 })).toEqual({ workers: 1, reason: 'capacity' });
    expect(chooseLocalWorkers({ cpus: 2, load1: 1 }).workers).toBe(1);
    expect(chooseLocalWorkers({ cpus: CORES_PER_WORKER * 3, load1: 0 }).workers).toBe(3);
  });

  it('falls back to the ceiling when capacity is unknown rather than guessing low', () => {
    // Missing, NaN, infinite, or negative load — and an unusable CPU count — all mean
    // "no capacity signal", which is the historical hardcoded-4 behaviour.
    for (const load1 of [undefined, Number.NaN, Number.POSITIVE_INFINITY, -1]) {
      expect(chooseLocalWorkers({ cpus: 16, load1 })).toEqual({
        workers: MAX_LOCAL_WORKERS,
        reason: 'unknown-load',
      });
    }
    for (const cpus of [undefined, 0, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(chooseLocalWorkers({ cpus, load1: 12 })).toEqual({
        workers: MAX_LOCAL_WORKERS,
        reason: 'unknown-load',
      });
    }
    // A non-numeric load arriving from an untyped caller must not poison the arithmetic.
    const untyped = chooseLocalWorkers({ cpus: 16, load1: 'busy' as unknown as number });
    expect(untyped).toEqual({ workers: MAX_LOCAL_WORKERS, reason: 'unknown-load' });
  });

  it('honours an explicit integer override over any measurement', () => {
    // An override beats a loaded box in both directions, and is not capped at the ceiling:
    // it encodes a measurement someone took on their own machine.
    expect(chooseLocalWorkers({ cpus: 16, load1: 24, override: 4 })).toEqual({
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

  it('ignores a fractional or unparseable override instead of rounding it', () => {
    for (const override of [2.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(chooseLocalWorkers({ cpus: 16, load1: 12, override })).toEqual({
        workers: 2,
        reason: 'capacity',
      });
    }
  });

  it('reads CI, the env override, and this machine through the resolver', () => {
    const cpus = os.cpus().length;

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

    // Without an override the decision is whatever this machine's own numbers imply.
    const measured = resolveE2EWorkers({});
    expect(measured.cpus).toBe(cpus);
    expect(measured.workers).toBe(
      chooseLocalWorkers({ cpus: measured.cpus, load1: measured.load1 }).workers
    );
    expect(measured.workers).toBeGreaterThanOrEqual(1);
    expect(measured.workers).toBeLessThanOrEqual(MAX_LOCAL_WORKERS);
  });
});
