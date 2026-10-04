/**
 * `WorkerPool.runWithTimeout` under the shared lifecycle contract: an aborted
 * caller is released at once while the worker's slot stays busy until the
 * worker answers, a failed call frees everything, and disposing the pool
 * releases every caller whose worker will now never answer (the stranded-call
 * bug `settleOnEviction` fixed).
 */

import { describe } from 'vitest';
import { deferred, type Deferred } from '../../../../helpers/deferred';
import { fakeWorkerInstance, poolWithWorkers } from '../../../../helpers/fake-worker';
import { defineLifecycleContract } from '../../../_shared/lifecycle-contract';

describe('WorkerPool dispatch — lifecycle', () => {
  defineLifecycleContract('WorkerPool dispatch', {
    create: () => {
      const replies: Array<Deferred<number>> = [];
      const handle = (n: number): Promise<number> => {
        const reply = deferred<number>();
        replies.push(reply);
        return reply.promise.then(() => n);
      };
      const workers = [fakeWorkerInstance({ handle }), fakeWorkerInstance({ handle })];
      const pool = poolWithWorkers(workers);
      return {
        start: (n, signal) =>
          pool.runWithTimeout(
            `op${n}`,
            'projection',
            (api) => (api as unknown as { handle: typeof handle }).handle(n),
            signal
          ),
        pending: () => replies.length,
        settle: (index, outcome) =>
          outcome === 'ok'
            ? replies[index].resolve(index)
            : replies[index].reject(new Error('trap')),
        // The pool commits nothing: its caller owns the result.
        observe: () => null,
        // Busy slots must drain — an aborted caller's slot stays busy until
        // its worker answers, then frees.
        inFlight: () => workers.reduce((sum, w) => sum + w.activeQueries, 0),
        dispose: () => pool.dispose(),
      };
    },
    cases: {
      abort: true,
      dispose: true,
      failure: true,
      retry: true,
      supersede: {
        na: 'the pool does not order results: each caller supersedes its own calls through its signal',
      },
      doubleDispose: true,
    },
    releasesWaitersOnDispose: true,
    usableAfterDispose: {
      na: 'the pool is a page singleton that re-initialises on its next call (WorkerPool.dispose)',
    },
  });
});
