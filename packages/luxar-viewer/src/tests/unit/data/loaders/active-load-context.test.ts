/**
 * The per-call demand-load context the leaf loaders publish to their L0
 * proxies (`data/loaders/active-load-context.ts`): publication and cleanup,
 * including on throw, and — the reason it exists — reentrancy when two loads
 * overlap on one loader.
 */

import { describe, expect, it } from 'vitest';

import { ActiveLoadContext } from '../../../../data/loaders/active-load-context';

/** A load that stays in flight until `release()` is called. */
function parked(): { promise: Promise<void>; release: () => void } {
  let release!: () => void;
  const promise = new Promise<void>((resolve) => (release = resolve));
  return { promise, release };
}

describe('ActiveLoadContext — one call at a time', () => {
  it('publishes the signal for the load and withdraws it after', async () => {
    const calls = new ActiveLoadContext();
    const controller = new AbortController();
    const result = await calls.runWithSignal(controller.signal, async () => {
      expect(calls.signal).toBe(controller.signal);
      return 42;
    });
    expect(result).toBe(42);
    expect(calls.signal).toBeNull();
  });

  it('publishes null when no signal is supplied', async () => {
    const calls = new ActiveLoadContext();
    await calls.runWithSignal(undefined, async () => {
      expect(calls.signal).toBeNull();
    });
  });

  it('withdraws the signal even when the load throws', async () => {
    const calls = new ActiveLoadContext();
    await expect(
      calls.runWithSignal(new AbortController().signal, async () => {
        throw new Error('boom');
      })
    ).rejects.toThrow('boom');
    expect(calls.signal).toBeNull();
  });

  it('attaches a probe, reports allResident, and detaches it', async () => {
    const calls = new ActiveLoadContext();
    const { data, allResident } = await calls.runWithProbe(async () => {
      expect(calls.probe).not.toBeNull();
      return 'payload';
    });
    expect(data).toBe('payload');
    // A load that touches no chunks counts as resident.
    expect(allResident).toBe(true);
    expect(calls.probe).toBeNull();
  });

  it('reports allResident=false when the probe records a miss', async () => {
    const calls = new ActiveLoadContext();
    const { allResident } = await calls.runWithProbe(async () => {
      calls.probe!.record(false);
    });
    expect(allResident).toBe(false);
  });

  it('detaches the probe even when the load throws', async () => {
    const calls = new ActiveLoadContext();
    await expect(
      calls.runWithProbe(async () => {
        throw new Error('boom');
      })
    ).rejects.toThrow('boom');
    expect(calls.probe).toBeNull();
  });
});

describe('ActiveLoadContext — overlapping calls (reentrancy)', () => {
  it('the call that finishes first does not withdraw the other call’s signal', async () => {
    const calls = new ActiveLoadContext();
    const a = new AbortController();
    const slow = parked();
    const pending = calls.runWithSignal(a.signal, () => slow.promise);
    await calls.runWithSignal(new AbortController().signal, async () => undefined);

    expect(calls.signal).toBe(a.signal);
    a.abort();
    expect(calls.signal?.aborted).toBe(true);
    slow.release();
    await pending;
    expect(calls.signal).toBeNull();
  });

  it('while calls overlap, one aborted call does not cancel reads the other shares', async () => {
    const calls = new ActiveLoadContext();
    const a = new AbortController();
    const b = new AbortController();
    const pa = parked();
    const pb = parked();
    const runA = calls.runWithSignal(a.signal, () => pa.promise);
    const runB = calls.runWithSignal(b.signal, () => pb.promise);

    a.abort();
    expect(calls.signal?.aborted ?? false).toBe(false);
    b.abort();
    expect(calls.signal?.aborted).toBe(true);

    pa.release();
    pb.release();
    await Promise.all([runA, runB]);
  });

  it('the call that finishes first does not detach the other call’s probe', async () => {
    const calls = new ActiveLoadContext();
    const slow = parked();
    const runA = calls.runWithProbe(async () => {
      await slow.promise;
      calls.probe!.record(false); // A's own read, after B has finished
    });
    await calls.runWithProbe(async () => undefined);

    slow.release();
    const { allResident } = await runA;
    expect(allResident).toBe(false);
  });

  it('a read while calls overlap is recorded into every active probe', async () => {
    const calls = new ActiveLoadContext();
    const pa = parked();
    const pb = parked();
    const runA = calls.runWithProbe(() => pa.promise);
    const runB = calls.runWithProbe(() => pb.promise);
    calls.probe!.record(false);
    pa.release();
    pb.release();
    const [ra, rb] = await Promise.all([runA, runB]);
    expect(ra.allResident).toBe(false);
    expect(rb.allResident).toBe(false);
  });
});
