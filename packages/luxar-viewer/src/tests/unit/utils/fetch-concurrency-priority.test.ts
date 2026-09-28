/**
 * Priority classes and the adaptive lane size of the global fetch gate.
 *
 * Measured on hosted HTTP/2 (100 ms RTT, 25 Mbps): the data lane sat pinned at
 * 24 in flight before first frame, with first-frame demand chunks FIFO-queued
 * behind prefetcher speculation. These pin the three properties that fix it:
 * a freed slot goes to the most urgent waiter, speculation can never hold more
 * than a quarter of the lane, and a multiplexed origin gets a wider lane.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { perfCounters } from '../../../profiling/perf-counters';
import {
  FetchPriorityCell,
  HTTP1_MAX_CONCURRENT_CHUNK_FETCHES,
  MAX_CONCURRENT_CHUNK_FETCHES,
  MAX_CONCURRENT_MULTIPLEXED_CHUNK_FETCHES,
  MAX_SPECULATIVE_FETCH_SHARE,
  noteFetchUrl,
  noteOriginProtocol,
  resetFetchTransport,
  resetOriginProtocols,
  withFetchGate,
  type FetchPriority,
} from '../../../utils/fetch-concurrency';

/** A gated call whose body stays open until released. */
function holder(
  priority: FetchPriority | FetchPriorityCell = 'demand',
  origin?: string,
  onStart?: () => void
) {
  let release!: () => void;
  let started = false;
  const released = new Promise<void>((resolve) => (release = resolve));
  const done = withFetchGate(
    () => {
      started = true;
      onStart?.();
      return released;
    },
    'data',
    priority,
    origin
  );
  return {
    done,
    release: () => release(),
    get started() {
      return started;
    },
  };
}

async function flush(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

afterEach(() => {
  resetOriginProtocols();
  resetFetchTransport();
  vi.unstubAllGlobals();
});

describe('fetch gate priority classes', () => {
  it('hands a freed slot to a queued demand request before an earlier speculative one', async () => {
    const full = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES }, () => holder('demand'));
    await flush();
    const speculative = holder('speculative');
    const demand = holder('demand');
    await flush();
    expect(speculative.started).toBe(false);
    expect(demand.started).toBe(false);

    full[0].release();
    await flush();

    expect(demand.started).toBe(true);
    expect(speculative.started).toBe(false);

    for (const h of [...full, speculative, demand]) h.release();
    await Promise.all([...full, speculative, demand].map((h) => h.done));
  });

  it('orders refinement between demand and speculative', async () => {
    const full = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES }, () => holder('demand'));
    await flush();
    const order: string[] = [];
    const speculative = holder('speculative', undefined, () => order.push('speculative'));
    const refinement = holder('refinement', undefined, () => order.push('refinement'));
    const demand = holder('demand', undefined, () => order.push('demand'));
    await flush();

    for (let i = 0; i < 3; i++) {
      full[i].release();
      await flush();
    }

    expect(order).toEqual(['demand', 'refinement', 'speculative']);
    for (const h of [...full, speculative, refinement, demand]) h.release();
    await Promise.all([...full, speculative, refinement, demand].map((h) => h.done));
  });

  it('never lets speculative requests hold more than their share of the lane', async () => {
    const cap = Math.floor(MAX_CONCURRENT_CHUNK_FETCHES * MAX_SPECULATIVE_FETCH_SHARE);
    let speculativeActive = 0;
    let speculativePeak = 0;
    const speculative = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES }, () =>
      holder('speculative', undefined, () => {
        speculativeActive += 1;
        speculativePeak = Math.max(speculativePeak, speculativeActive);
      })
    );
    await flush();
    expect(speculative.filter((h) => h.started)).toHaveLength(cap);

    // The rest of the lane stays free for demand, which starts immediately.
    const demand = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES - cap }, () => holder());
    await flush();
    expect(demand.every((h) => h.started)).toBe(true);

    // A demand request now WAITS (lane full). Freeing a speculative slot must
    // go to it, not to the next speculative waiter.
    const waiting = holder('demand');
    await flush();
    expect(waiting.started).toBe(false);
    const running = speculative.find((h) => h.started)!;
    speculativeActive -= 1;
    running.release();
    await flush();
    expect(waiting.started).toBe(true);
    expect(speculativePeak).toBe(cap);

    // Draining never lets the speculative backlog exceed its share either.
    const all = [...speculative, ...demand, waiting];
    while (all.some((h) => !h.started)) {
      const next = all.find((h) => h.started);
      if (!next) break;
      all.splice(all.indexOf(next), 1);
      if (speculative.includes(next) && next !== running) speculativeActive -= 1;
      next.release();
      await flush();
    }
    for (const h of all) h.release();
    await Promise.all([...speculative, ...demand, waiting].map((h) => h.done));
    expect(speculativePeak).toBe(cap);
  });

  it('promotes a queued speculative request when a demand caller joins it', async () => {
    const full = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES }, () => holder('demand'));
    await flush();
    const earlierSpeculative = holder('speculative');
    const cell = new FetchPriorityCell('speculative');
    const promoted = holder(cell);
    await flush();

    cell.raise('demand');
    full[0].release();
    await flush();

    // It now competes as demand (FIFO from the moment demand joined), so it
    // overtakes the speculative request that queued before it.
    expect(promoted.started).toBe(true);
    expect(earlierSpeculative.started).toBe(false);

    for (const h of [...full, promoted, earlierSpeculative]) h.release();
    await Promise.all([...full, promoted, earlierSpeculative].map((h) => h.done));
  });

  it('starts a speculative request held only by its share as soon as it is promoted', async () => {
    const cap = Math.floor(MAX_CONCURRENT_CHUNK_FETCHES * MAX_SPECULATIVE_FETCH_SHARE);
    const running = Array.from({ length: cap }, () => holder('speculative'));
    const cell = new FetchPriorityCell('speculative');
    const blocked = holder(cell);
    await flush();
    expect(blocked.started).toBe(false); // lane has room; the share binds

    cell.raise('demand');
    await flush();
    expect(blocked.started).toBe(true);

    for (const h of [...running, blocked]) h.release();
    await Promise.all([...running, blocked].map((h) => h.done));
  });

  it('keeps FIFO order within a class across a large fan-out', async () => {
    const full = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES }, () => holder('demand'));
    await flush();
    const order: number[] = [];
    const queued = Array.from({ length: 5_000 }, (_, i) =>
      withFetchGate(async () => {
        order.push(i);
      })
    );
    for (const h of full) h.release();
    await Promise.all([...full.map((h) => h.done), ...queued]);
    expect(order).toEqual(Array.from({ length: 5_000 }, (_, i) => i));
  });

  it('still tallies requests, highWater and queue wait per lane', async () => {
    perfCounters.reset();
    const full = Array.from({ length: MAX_CONCURRENT_CHUNK_FETCHES }, () => holder('demand'));
    const extra = holder('speculative');
    await flush();
    expect(perfCounters.get('fetch.data.requests')).toBe(MAX_CONCURRENT_CHUNK_FETCHES + 1);
    expect(perfCounters.get('fetch.data.highWater')).toBe(MAX_CONCURRENT_CHUNK_FETCHES);
    for (const h of [...full, extra]) h.release();
    await Promise.all([...full, extra].map((h) => h.done));
  });
});

describe('fetch gate adaptive lane size', () => {
  it('keeps an HTTP/1.1 (or unknown) origin at the default lane size', async () => {
    noteOriginProtocol('http://h1.example', 'http/1.1');
    const calls = Array.from({ length: MAX_CONCURRENT_MULTIPLEXED_CHUNK_FETCHES }, () =>
      holder('demand', 'http://h1.example')
    );
    await flush();
    expect(calls.filter((h) => h.started)).toHaveLength(MAX_CONCURRENT_CHUNK_FETCHES);
    for (const h of calls) h.release();
    await Promise.all(calls.map((h) => h.done));
  });

  it('widens the lane for an origin that negotiated HTTP/2 or HTTP/3', async () => {
    for (const protocol of ['h2', 'h3']) {
      const origin = `https://${protocol}.example`;
      noteOriginProtocol(origin, protocol);
      const calls = Array.from({ length: MAX_CONCURRENT_MULTIPLEXED_CHUNK_FETCHES + 4 }, () =>
        holder('demand', origin)
      );
      await flush();
      expect(calls.filter((h) => h.started)).toHaveLength(MAX_CONCURRENT_MULTIPLEXED_CHUNK_FETCHES);
      for (const h of calls) h.release();
      await Promise.all(calls.map((h) => h.done));
    }
  });

  it('does not widen a multiplexed origin once an http: URL has capped the gate', async () => {
    // A plain http: URL means HTTP/1.1's six sockets; widening any origin past
    // the cap would park admitted requests in the browser's own queue with
    // their header timers already running.
    const origin = 'https://h2.example';
    noteOriginProtocol(origin, 'h2');
    noteFetchUrl('http://localhost:8000/scene.luxar.zarr/zarr.json');
    const calls = Array.from({ length: HTTP1_MAX_CONCURRENT_CHUNK_FETCHES + 4 }, () =>
      holder('demand', origin)
    );
    await flush();
    expect(calls.filter((h) => h.started)).toHaveLength(HTTP1_MAX_CONCURRENT_CHUNK_FETCHES);
    for (const h of calls) h.release();
    await Promise.all(calls.map((h) => h.done));
  });

  it('learns the protocol from resource timing (nextHopProtocol)', async () => {
    vi.resetModules();
    let callback: ((list: { getEntries: () => unknown[] }) => void) | undefined;
    class FakeObserver {
      constructor(cb: (list: { getEntries: () => unknown[] }) => void) {
        callback = cb;
      }
      observe(): void {}
      disconnect(): void {}
    }
    vi.stubGlobal('PerformanceObserver', FakeObserver);
    const gate = await import('../../../utils/fetch-concurrency');

    const origin = 'https://cdn.example';
    // First call installs the observer; the lane is still the default.
    const first = Array.from({ length: gate.MAX_CONCURRENT_CHUNK_FETCHES + 1 }, () =>
      gate.withFetchGate(() => Promise.resolve(), 'data', 'demand', origin)
    );
    await Promise.all(first);
    expect(callback).toBeDefined();

    callback!({
      getEntries: () => [
        { name: `${origin}/scene.zarr.zip`, nextHopProtocol: 'h2' },
        { name: 'https://other.example/x', nextHopProtocol: 'http/1.1' },
      ],
    });

    const releases: Array<() => void> = [];
    let active = 0;
    const calls = Array.from({ length: gate.MAX_CONCURRENT_MULTIPLEXED_CHUNK_FETCHES }, () =>
      gate.withFetchGate(
        () =>
          new Promise<void>((resolve) => {
            active += 1;
            releases.push(resolve);
          }),
        'data',
        'demand',
        origin
      )
    );
    await flush();
    expect(active).toBe(gate.MAX_CONCURRENT_MULTIPLEXED_CHUNK_FETCHES);
    releases.forEach((r) => r());
    await Promise.all(calls);
  });
});
