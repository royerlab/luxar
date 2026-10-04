/**
 * The per-call demand-load context the leaf loaders publish to their L0
 * proxies (`data/loaders/active-load-context.ts`): publication and cleanup,
 * including on throw, and — the reason it exists — reentrancy when two loads
 * overlap on one loader.
 */

import { describe, expect, it } from 'vitest';

import { ActiveLoadContext } from '../../../../data/loaders/active-load-context';
import {
  signalOrigin,
  tagSignalOrigin,
} from '../../../../cache/decompressed-chunk-cache/decode-origin';
import { signalPriority, tagSignalPriority } from '../../../../utils/fetch-concurrency';

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
      expect(calls.signal?.aborted).toBe(false);
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

  it('preserves a call signal’s origin and live priority raises', async () => {
    const calls = new ActiveLoadContext();
    const controller = new AbortController();
    const priority = tagSignalPriority(controller.signal, 'speculative');
    tagSignalOrigin(controller.signal, 'shadow');
    await calls.runWithSignal(controller.signal, async () => {
      expect(signalOrigin(calls.signal)).toBe('shadow');
      expect(signalPriority(calls.signal)?.value).toBe('speculative');
      priority.raise('demand');
      expect(signalPriority(calls.signal)?.value).toBe('demand');
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

  it('preserves the caller’s abort reason', async () => {
    const calls = new ActiveLoadContext();
    const controller = new AbortController();
    const wait = parked();
    const pending = calls.runWithSignal(controller.signal, () => wait.promise);
    const readSignal = calls.signal;
    const reason = new Error('superseded');
    controller.abort(reason);
    expect(readSignal?.reason).toBe(reason);
    wait.release();
    await pending;
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
  it('keeps the first origin and raised priority after a demand caller leaves', async () => {
    const calls = new ActiveLoadContext();
    const shadow = new AbortController();
    tagSignalPriority(shadow.signal, 'speculative');
    tagSignalOrigin(shadow.signal, 'shadow');
    const shadowWait = parked();
    const shadowLoad = calls.runWithSignal(shadow.signal, () => shadowWait.promise);
    const readSignal = calls.signal;

    const demand = new AbortController();
    tagSignalPriority(demand.signal, 'demand');
    tagSignalOrigin(demand.signal, 'demand');
    const demandWait = parked();
    const demandLoad = calls.runWithSignal(demand.signal, () => demandWait.promise);
    expect(calls.signal).toBe(readSignal);
    expect(signalOrigin(readSignal)).toBe('shadow');
    expect(signalPriority(readSignal)?.value).toBe('demand');

    demandWait.release();
    await demandLoad;
    expect(signalOrigin(calls.signal)).toBe('shadow');
    expect(signalPriority(calls.signal)?.value).toBe('demand');
    expect(readSignal?.aborted).toBe(false);
    shadowWait.release();
    await shadowLoad;
    expect(readSignal?.aborted).toBe(true);
    expect((readSignal?.reason as Error).name).toBe('AbortError');
  });

  it('the call that finishes first does not withdraw the other call’s signal', async () => {
    const calls = new ActiveLoadContext();
    const a = new AbortController();
    const slow = parked();
    const pending = calls.runWithSignal(a.signal, () => slow.promise);
    await calls.runWithSignal(new AbortController().signal, async () => undefined);

    expect(calls.signal?.aborted).toBe(false);
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

    const readSignal = calls.signal;
    expect(readSignal).not.toBeNull();
    a.abort();
    expect(readSignal?.aborted).toBe(false);
    b.abort();
    expect(readSignal?.aborted).toBe(true);

    pa.release();
    pb.release();
    await Promise.all([runA, runB]);
  });

  it('keeps an already-started read alive when a second caller joins', async () => {
    const calls = new ActiveLoadContext();
    const a = new AbortController();
    const b = new AbortController();
    const pa = parked();
    const pb = parked();
    const runA = calls.runWithSignal(a.signal, () => pa.promise);
    const readSignal = calls.signal;
    const runB = calls.runWithSignal(b.signal, () => pb.promise);

    a.abort();
    expect(readSignal?.aborted).toBe(false);
    b.abort();
    expect(readSignal?.aborted).toBe(true);

    pa.release();
    pb.release();
    await Promise.all([runA, runB]);
  });

  it('cancels a shared read when its last live caller finishes', async () => {
    const calls = new ActiveLoadContext();
    const a = new AbortController();
    const pa = parked();
    const runA = calls.runWithSignal(a.signal, () => pa.promise);
    const readSignal = calls.signal;

    await calls.runWithSignal(new AbortController().signal, async () => undefined);
    a.abort();
    expect(readSignal?.aborted).toBe(true);
    pa.release();
    await runA;
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
