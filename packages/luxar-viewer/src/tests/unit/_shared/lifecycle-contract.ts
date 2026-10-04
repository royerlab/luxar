/**
 * Shared lifecycle contract for every cancellable / disposable async
 * operation (the rows of `_conformance/async-operations.ts`).
 *
 * Abort, dispose, failure and retry paths run rarely, so they rot: the third
 * viewer review found a load interrupted by dispose, late depth-sort commits,
 * stranded worker calls, HDRI races and retry escalation. This file states the
 * standard cases once; each operation's test file calls
 * {@link defineLifecycleContract} with a thin adapter that wires the real code
 * to controllable inner work (a fetch, a decode, a worker reply — each one a
 * `deferred()` the harness settles in an order of its choosing). Mirrors the
 * `data/_shared/refinement-loop-contract.ts` precedent.
 *
 * The cases (a case the operation cannot express is declared `{ na: reason }`
 * and shows up as such in the run — it is never skipped silently):
 *
 * 1. `abort` — abort mid-flight: the operation settles promptly (before its
 *    inner work does) and nothing is committed after the abort.
 * 2. `dispose` — dispose mid-flight: a late settlement is ignored (no commit,
 *    no throw, nothing left in flight), and a call after dispose is a no-op.
 *    With `releasesWaitersOnDispose`, a caller awaiting the operation is
 *    released without its inner work ever settling (no stranded call).
 * 3. `failure` — the failure surfaces exactly once and leaves no stuck state:
 *    nothing in flight, and the next call starts fresh work and succeeds.
 * 4. `retry` — repeated failures followed by a success: every attempt starts
 *    the same amount of inner work (no escalation), the in-flight bookkeeping
 *    stays empty between attempts, and the final success lands.
 * 5. `supersede` — two overlapping operations, the older settling last: only
 *    the newest result lands.
 * 6. `doubleDispose` — dispose twice: no throw, nothing in flight.
 *
 * An adapter with a `reset` (a dataset switch that keeps the owner alive) also
 * gets "reset mid-flight": the late settlement is ignored and the owner stays
 * usable.
 *
 * Side effects are compared through `observe()`, which must describe only what
 * the operation COMMITS (an applied texture, a cached chunk, a loaded scene's
 * tail) — never request counts — so a superseded run and a fresh run of the
 * winner compare equal.
 */

import { describe, expect, it } from 'vitest';

/** The standard cases, in the order they are emitted. */
export const LIFECYCLE_CASES = [
  'abort',
  'dispose',
  'failure',
  'retry',
  'supersede',
  'doubleDispose',
] as const;

export type LifecycleCase = (typeof LIFECYCLE_CASES)[number];

/** A case the operation cannot express, with the reason (cite the code or spec). */
export interface NotApplicable {
  na: string;
}

/** One live instance of the operation's owner, wired to controllable inner work. */
export interface LifecycleSubject {
  /**
   * Start operation `n` (0, 1, … — distinct inputs, so a superseded run and
   * its successor are distinguishable). Returns the operation's promise, or
   * nothing for a fire-and-forget operation.
   */
  start(n: number, signal?: AbortSignal): Promise<unknown> | void;
  /** How many inner work items (fetches, decodes, replies) have been requested so far. */
  pending(): number;
  /** Settle the inner work item at `index` (request order). Called at most once per index. */
  settle(index: number, outcome: 'ok' | 'fail'): void;
  /** JSON-comparable snapshot of what the operation committed. */
  observe(): unknown;
  /** Bookkeeping that must drain (in-flight map entries, busy slots, locks). Default 0. */
  inFlight?(): number;
  /** Failures surfaced other than by rejecting `start`'s promise (warnings, error events). */
  failures?(): number;
  dispose(): void | Promise<void>;
  /** A mid-flight reset that keeps the owner usable (a dataset switch). */
  reset?(): void;
}

export interface LifecycleAdapter {
  create(): LifecycleSubject | Promise<LifecycleSubject>;
  /** For each case: `true` to run it, or why it does not apply. */
  cases: Record<LifecycleCase, true | NotApplicable>;
  /** Dispose releases a caller awaiting `start` even though its inner work never settles. */
  releasesWaitersOnDispose?: boolean;
  /** The subject has a {@link LifecycleSubject.reset}: emit the "reset mid-flight" case. */
  resets?: boolean;
  /** Why a call after dispose is not a no-op (the owner deliberately stays usable). */
  usableAfterDispose?: NotApplicable;
  /** Let pending callbacks run. Default: a burst of microtasks (safe under fake timers). */
  flush?(): Promise<void>;
}

interface TrackedOperation {
  settled: boolean;
  rejected: boolean;
}

/** The operation's promise, observed without letting a rejection go unhandled. */
function track(op: Promise<unknown> | void): TrackedOperation | undefined {
  if (!op) return undefined;
  const tracked: TrackedOperation = { settled: false, rejected: false };
  op.then(
    () => {
      tracked.settled = true;
    },
    () => {
      tracked.settled = true;
      tracked.rejected = true;
    }
  );
  return tracked;
}

async function microtasks(): Promise<void> {
  for (let i = 0; i < 25; i++) await Promise.resolve();
}

/** A subject plus the harness's view of which inner work it has settled. */
class Driver {
  private readonly settledIndices = new Set<number>();

  constructor(
    readonly subject: LifecycleSubject,
    private readonly adapter: LifecycleAdapter
  ) {}

  flush(): Promise<void> {
    return this.adapter.flush ? this.adapter.flush() : microtasks();
  }

  async start(n: number, signal?: AbortSignal): Promise<TrackedOperation | undefined> {
    const op = track(this.subject.start(n, signal));
    await this.flush();
    return op;
  }

  /** Settle every unsettled inner item at or after `from`, and whatever settling starts. */
  async drain(outcome: 'ok' | 'fail', from = 0): Promise<void> {
    for (let guard = 0; guard < 200; guard++) {
      let next = -1;
      for (let i = from; i < this.subject.pending(); i++) {
        if (!this.settledIndices.has(i)) {
          next = i;
          break;
        }
      }
      if (next < 0) return;
      this.settledIndices.add(next);
      this.subject.settle(next, outcome);
      await this.flush();
    }
    throw new Error('lifecycle contract: inner work never quiesced');
  }

  inFlight(): number {
    return this.subject.inFlight?.() ?? 0;
  }

  async dispose(): Promise<void> {
    await this.subject.dispose();
    await this.flush();
  }
}

/** Register the standard lifecycle cases for one operation. */
export function defineLifecycleContract(name: string, adapter: LifecycleAdapter): void {
  const driver = async (): Promise<Driver> => new Driver(await adapter.create(), adapter);

  /** What a fresh owner commits after running only operation `n` to success. */
  const reference = async (n: number): Promise<unknown> => {
    const d = await driver();
    await d.start(n);
    await d.drain('ok');
    const committed = d.subject.observe();
    await d.dispose();
    return committed;
  };

  const declare = (
    lifecycleCase: LifecycleCase,
    title: string,
    body: () => Promise<void>
  ): void => {
    const mode = adapter.cases[lifecycleCase];
    if (mode === true) {
      it(`${lifecycleCase}: ${title}`, body);
    } else {
      it(`${lifecycleCase}: n/a — ${mode.na}`, () => {
        expect(mode.na.trim().length, 'an n/a case must say why').toBeGreaterThan(0);
      });
    }
  };

  describe(`${name} — lifecycle contract`, () => {
    declare('abort', 'settles promptly and commits nothing after the abort', async () => {
      const d = await driver();
      const controller = new AbortController();
      const op = await d.start(0, controller.signal);
      expect(d.subject.pending(), 'the operation must be in flight').toBeGreaterThan(0);
      const before = d.subject.observe();
      controller.abort();
      await d.flush();
      if (op) expect(op.settled, 'aborting must settle the operation promptly').toBe(true);
      await d.drain('ok');
      expect(d.subject.observe()).toEqual(before);
      expect(d.inFlight()).toBe(0);
      await d.dispose();
    });

    declare('dispose', 'a late settlement after dispose is ignored', async () => {
      const d = await driver();
      const op = await d.start(0);
      expect(d.subject.pending(), 'the operation must be in flight').toBeGreaterThan(0);
      await d.dispose();
      const afterDispose = d.subject.observe();
      if (adapter.releasesWaitersOnDispose && op) {
        expect(op.settled, 'dispose must release a waiting caller').toBe(true);
      }
      await d.drain('ok');
      expect(d.subject.observe()).toEqual(afterDispose);
      expect(d.inFlight()).toBe(0);
    });

    if (adapter.cases.dispose === true) {
      const afterDispose = adapter.usableAfterDispose;
      if (afterDispose) {
        it(`dispose: a call after dispose — n/a, ${afterDispose.na}`, () => {
          expect(afterDispose.na.trim().length).toBeGreaterThan(0);
        });
      } else {
        it('dispose: a call after dispose is a no-op', async () => {
          const d = await driver();
          await d.dispose();
          const requested = d.subject.pending();
          const before = d.subject.observe();
          await d.start(1);
          await d.drain('ok');
          expect(d.subject.pending(), 'no inner work after dispose').toBe(requested);
          expect(d.subject.observe()).toEqual(before);
        });
      }
    }

    declare('failure', 'surfaces once and leaves nothing stuck', async () => {
      const d = await driver();
      const op = await d.start(0);
      await d.drain('fail');
      if (d.subject.failures) expect(d.subject.failures()).toBe(1);
      else expect(op?.rejected, 'the failure must surface').toBe(true);
      expect(d.inFlight()).toBe(0);
      const requested = d.subject.pending();
      await d.start(1);
      expect(d.subject.pending(), 'the next call must start fresh work').toBeGreaterThan(requested);
      await d.drain('ok');
      expect(d.subject.observe()).toEqual(await reference(1));
      await d.dispose();
    });

    declare('retry', 'every attempt costs the same and the last success lands', async () => {
      const d = await driver();
      const costs: number[] = [];
      for (let attempt = 0; attempt < 4; attempt++) {
        const requested = d.subject.pending();
        await d.start(0);
        costs.push(d.subject.pending() - requested);
        await d.drain('fail');
        expect(d.inFlight(), `in flight after failed attempt ${attempt}`).toBe(0);
      }
      expect(new Set(costs).size, `inner work per attempt: ${costs.join(', ')}`).toBe(1);
      await d.start(0);
      await d.drain('ok');
      expect(d.subject.observe()).toEqual(await reference(0));
      await d.dispose();
    });

    declare('supersede', 'only the newest result lands', async () => {
      const d = await driver();
      await d.start(0);
      const olderWork = d.subject.pending();
      await d.start(1);
      await d.drain('ok', olderWork); // the newer operation finishes first
      await d.drain('ok'); // then the older one settles, late
      expect(d.subject.observe()).toEqual(await reference(1));
      expect(d.inFlight()).toBe(0);
      await d.dispose();
    });

    declare('doubleDispose', 'is idempotent', async () => {
      const d = await driver();
      await d.start(0);
      await d.dispose();
      const afterFirst = d.subject.observe();
      await d.dispose();
      expect(d.subject.observe()).toEqual(afterFirst);
      await d.drain('ok');
      expect(d.inFlight()).toBe(0);
    });

    if (adapter.resets) {
      it('reset mid-flight: ignores the late settlement and stays usable', async () => {
        const d = await driver();
        const reset = d.subject.reset;
        expect(
          reset,
          'an adapter declaring `resets` must give its subject a reset()'
        ).toBeDefined();
        await d.start(0);
        reset!.call(d.subject);
        await d.flush();
        const afterReset = d.subject.observe();
        await d.drain('ok');
        expect(d.subject.observe()).toEqual(afterReset);
        await d.start(1);
        await d.drain('ok');
        expect(d.subject.observe()).toEqual(await reference(1));
        await d.dispose();
      });
    }
  });
}
